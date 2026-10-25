import json,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from diffusion_demo.core.models.vae import VAE
from diffusion_demo.core.models.continuous_latents import posterior_normalization,sample_posterior
from latent_ddp_experiment import main

def test_train_only_normalization_includes_posterior_variance():
    a=np.zeros((1004,2,2,2,2),np.float32);a[:1000,0]=10000
    a[1000:,0]=3;a[1000:,1]=np.log(4)
    center,scale=posterior_normalization(a,'sample')
    torch.testing.assert_close(center,torch.full_like(center,3));torch.testing.assert_close(scale,torch.full_like(scale,2))
    torch.testing.assert_close(sample_posterior(torch.from_numpy(a[1000:]),'mu'),torch.full((4,2,2,2),3.))

def test_continuous_prior_resume_preserves_rng_and_weights(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(3)
    cfg_model=dict(latent_dim=2,hidden_dims=[4,8],image_size=48,latent_spatial_size=4)
    vae=VAE(**cfg_model);checkpoint=tmp_path/'vae.pth'
    torch.save(dict(model_type='vae',model_config=cfg_model,model_state_dict=vae.state_dict(),metrics={}),checkpoint)
    rng=np.random.default_rng(7);cache=rng.normal(size=(1008,2,2,4,4)).astype(np.float16);cache[:,1]-=2
    cfg=dict(autoencoder=str(checkpoint),data='mocked',batch_size=4,lr=1e-3,width=24,depth=1,heads=3,patch_size=2,posterior_mode='sample',seed=1)
    def run(folder,images,resume=None):
        path=tmp_path/'config.json';path.write_text(json.dumps(dict(cfg,images=images,save_every_images=8)))
        argv=['train','--cpu','--config',str(path),'--output',str(folder)]
        if resume:argv+=['--resume',str(resume)]
        with patch.object(sys,'argv',argv),patch('latent_ddp_experiment.cache_posterior',return_value=cache),patch('latent_ddp_experiment.draw',return_value=torch.zeros(1,3,48,48)):
            main()
    run(tmp_path/'split',8);run(tmp_path/'split',16,tmp_path/'split/last.pt');run(tmp_path/'full',16)
    split=torch.load(tmp_path/'split/last.pt',weights_only=True);full=torch.load(tmp_path/'full/last.pt',weights_only=True)
    for key in ['model','ema']:
        for name in full[key]:torch.testing.assert_close(full[key][name],split[key][name],rtol=0,atol=0)
    assert split['step']==4 and split['projection'] is None
    assert torch.equal(split['rng'][0]['cpu'],full['rng'][0]['cpu'])
