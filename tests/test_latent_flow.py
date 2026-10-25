import unittest
import torch
from latent_ddp_experiment import LatentDiT, decode_codes, draw
from diffusion_demo.core.models.vae import VQVAE
from diffusion_demo.core.models.pixel_jit import flow_loss
from diffusion_demo.utils.torch_env import MixedPrecision

class TestLatentFlow(unittest.TestCase):
    def test_codes_flow_and_batch_invariant_sampling(self):
        torch.set_num_threads(1);torch.manual_seed(8)
        vae=VQVAE(embedding_dim=4,num_embeddings=8,hidden_dims_enc=[8,8],hidden_dims_dec=[8,8],latent_spatial_size=12).eval()
        model=LatentDiT(img_size=12,in_channels=4,patch_size=2,hidden_size=24,depth=1,num_heads=3)
        weight=vae.vq.embeddings.weight
        ids=torch.randint(8,(2,144));clean=decode_codes(ids,weight,12)
        self.assertEqual(clean.shape,(2,4,12,12))
        loss=flow_loss(model,clean,torch.tensor([.1,.9]),torch.randn_like(clean));loss.backward()
        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()))
        model.eval();center=torch.zeros(1,4,1,1);scale=torch.ones_like(center);amp=MixedPrecision(torch.device('cpu'),'fp32')
        together=draw(model,vae,center,scale,amp,[0,1],123,3)
        separate=torch.cat([draw(model,vae,center,scale,amp,[i],123,3) for i in [0,1]])
        torch.testing.assert_close(together,separate,atol=1e-5,rtol=1e-5)
        snapped=draw(model,vae,center,scale,amp,[0],123,3,True)
        self.assertEqual(snapped.shape,(1,3,48,48));self.assertTrue(snapped.isfinite().all())

    def test_resume_matches_uninterrupted_training(self):
        import json,sys,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from latent_ddp_experiment import main
        torch.manual_seed(12)
        mc=dict(embedding_dim=4,num_embeddings=8,hidden_dims_enc=[8,8],hidden_dims_dec=[8,8],latent_spatial_size=4)
        vae=VQVAE(**mc)
        codes=torch.randint(8,(30000,16)).numpy()
        for components in [None, 2]:
            with tempfile.TemporaryDirectory() as root:
                root=Path(root);source=root/'vq.pth'
                torch.save(dict(model_type='vqvae',model_config=mc,model_state_dict=vae.state_dict(),metrics={}),source)
                cfg=dict(autoencoder=str(source),data='mocked-cache',images=8,save_every_images=8,batch_size=4,lr=.001,width=24,depth=1,heads=3,patch_size=2,seed=4,pca_components=components)
                config=root/'config.json'
                def run(output,images,resume=None):
                    config.write_text(json.dumps(dict(cfg,images=images)))
                    args=['latent_ddp_experiment.py','--cpu','--config',str(config),'--output',str(output)]
                    if resume:args+=['--resume',str(resume)]
                    with patch.object(sys,'argv',args),patch('latent_ddp_experiment.cache_codes',return_value=codes):main()
                run(root/'split',8)
                run(root/'split',16,root/'split/last.pt')
                run(root/'full',16)
                resumed=torch.load(root/'split/last.pt',weights_only=True);full=torch.load(root/'full/last.pt',weights_only=True)
                for key in ['model','ema']:
                    for name in full[key]:torch.testing.assert_close(resumed[key][name],full[key][name],rtol=0,atol=0)
                self.assertEqual(resumed['step'],full['step'])


class TestLatentPCA(unittest.TestCase):
    def test_weighted_pca_roundtrip_and_truncation(self):
        from latent_ddp_experiment import fit_pca,project_latent,restore_latent
        torch.manual_seed(7)
        w=torch.randn(16,4);p=torch.rand(16);p/=p.sum()
        full,scale=fit_pca(w,p,4);z=w[:,:,None,None]
        torch.testing.assert_close(restore_latent(project_latent(z,full),full),z,atol=1e-6,rtol=1e-6)
        projected=project_latent(z,full).flatten(1)/scale.flatten()
        torch.testing.assert_close((projected*p[:,None]).sum(0),torch.zeros(4),atol=1e-6,rtol=0)
        torch.testing.assert_close(projected.T@(projected*p[:,None]),torch.eye(4),atol=1e-6,rtol=1e-6)
        small,_=fit_pca(w,p,2)
        mse=((restore_latent(project_latent(z,small),small)-z).flatten(1).square()*p[:,None]).sum()
        total=((z-full['offset']).flatten(1).square()*p[:,None]).sum()
        self.assertAlmostEqual(float(1-mse/total),small['retained_variance'],places=6)

    def test_projected_sampling_has_decoder_channels(self):
        from latent_ddp_experiment import fit_pca
        torch.manual_seed(9)
        vae=VQVAE(embedding_dim=4,num_embeddings=8,hidden_dims_enc=[8,8],hidden_dims_dec=[8,8],latent_spatial_size=12).eval()
        projection,scale=fit_pca(vae.vq.embeddings.weight,torch.ones(8)/8,2)
        model=LatentDiT(img_size=12,in_channels=2,patch_size=2,hidden_size=24,depth=1,num_heads=3).eval()
        image=draw(model,vae,torch.zeros_like(scale),scale,MixedPrecision(torch.device('cpu'),'fp32'),[0],123,3,projection=projection)
        self.assertEqual(image.shape,(1,3,48,48));self.assertTrue(image.isfinite().all())

    def test_guide_rejects_different_projection(self):
        import copy
        from latent_ddp_experiment import validate_guide
        a=dict(model_config={'in_channels':2},center=torch.zeros(1,2,1,1),scale=torch.ones(1,2,1,1),projection=dict(basis=torch.eye(4)[:,:2],offset=torch.zeros(1,4,1,1)))
        b=copy.deepcopy(a);validate_guide(a,b)
        b['projection']['basis'][0,0]=0
        with self.assertRaisesRegex(ValueError,'projection'):validate_guide(a,b)
