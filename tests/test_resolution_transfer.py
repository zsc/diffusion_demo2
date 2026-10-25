import copy,json,sys
import pytest,torch
from unittest.mock import patch
from diffusion_demo.core.models.vae import VQVAE
from diffusion_demo.core.models.resolution_transfer import resize_dit_state,validate_resolution_codec
from latent_ddp_experiment import LatentDiT,main


def test_native_96_codec_and_exact_frozen_weights():
    torch.set_num_threads(1)
    kw=dict(embedding_dim=4,num_embeddings=8,hidden_dims_enc=[8,8],hidden_dims_dec=[8,8])
    a=VQVAE(**kw,image_size=48,latent_spatial_size=12).eval()
    b=VQVAE(**kw,image_size=96,latent_spatial_size=24).eval();b.load_state_dict(a.state_dict())
    validate_resolution_codec(a,b)
    x=torch.randn(2,3,96,96)
    with torch.no_grad():z=b.encode(x);y=b.decode(z)
    assert z.shape==(2,4,24,24) and y.shape==x.shape
    with torch.no_grad():b.vq.embeddings.weight[0,0]+=1
    with pytest.raises(ValueError,match='identical'):validate_resolution_codec(a,b)


def test_position_interpolation_preserves_all_other_weights():
    old=dict(img_size=12,in_channels=2,patch_size=1,hidden_size=24,depth=1,num_heads=3)
    new=dict(old,img_size=24)
    a=LatentDiT(**old);b=LatentDiT(**new);state=resize_dit_state(a.state_dict(),old,new)
    b.load_state_dict(state,strict=True)
    assert state['pos_embed'].shape==(1,576,24)
    for key,value in a.state_dict().items():
        if key!='pos_embed':assert torch.equal(value,state[key])
    with pytest.raises(ValueError,match='Only'):resize_dit_state(a.state_dict(),old,dict(new,in_channels=4))


def test_warm_start_preserves_coordinates_and_resets_counter(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(3)
    kw=dict(embedding_dim=4,num_embeddings=8,hidden_dims_enc=[8,8],hidden_dims_dec=[8,8])
    a=VQVAE(**kw,image_size=48,latent_spatial_size=12).eval()
    old=tmp_path/'old';old.mkdir()
    source=dict(model_type='vqvae',model_config=dict(kw,image_size=48,latent_spatial_size=12),model_state_dict=a.state_dict(),metrics={})
    torch.save(source,old/'autoencoder.pth')
    source=copy.deepcopy(source);source['model_config'].update(image_size=96,latent_spatial_size=24);torch.save(source,tmp_path/'new.pth')
    mc=dict(img_size=12,in_channels=2,patch_size=1,hidden_size=24,depth=1,num_heads=3)
    m=LatentDiT(**mc);center=torch.ones(1,2,1,1)*.2;scale=torch.ones_like(center)*2
    proj=dict(basis=torch.eye(4)[:,:2],offset=torch.zeros(1,4,1,1),retained_variance=.9)
    torch.save(dict(model_config=mc,ema=m.state_dict(),center=center,scale=scale,projection=proj,step=999),old/'ema.pt')
    cfg=dict(autoencoder=str(tmp_path/'new.pth'),data='mocked',images=4,batch_size=4,save_every_images=4,lr=1e-4,width=24,depth=1,heads=3,patch_size=1,pca_components=2)
    config=tmp_path/'cfg.json';config.write_text(json.dumps(cfg));codes=torch.randint(8,(30000,576)).numpy()
    args=['train','--cpu','--config',str(config),'--output',str(tmp_path/'out'),'--init-checkpoint',str(old/'ema.pt')]
    with patch.object(sys,'argv',args),patch('latent_ddp_experiment.cache_codes',return_value=codes),patch('latent_ddp_experiment.draw',return_value=torch.zeros(1,3,96,96)):
        main()
    result=torch.load(tmp_path/'out/last.pt',weights_only=True)
    assert result['step']==1 and result['model_config']['img_size']==24
    assert torch.equal(center,result['center']) and torch.equal(scale,result['scale'])
    assert torch.equal(proj['basis'],result['projection']['basis'])
    assert all(v['step']==1 for v in result['optimizer']['state'].values())
    assert torch.isfinite(result['model']['pos_embed']).all()
    # Continue inherited coordinates without fitting a new PCA on the 96px codes.
    cfg['images']=8;config.write_text(json.dumps(cfg))
    resumed_args=['train','--cpu','--config',str(config),'--output',str(tmp_path/'out'),
                  '--resume',str(tmp_path/'out/last.pt')]
    full_args=['train','--cpu','--config',str(config),'--output',str(tmp_path/'full'),
               '--init-checkpoint',str(old/'ema.pt')]
    for argv in [resumed_args,full_args]:
        with patch.object(sys,'argv',argv),patch('latent_ddp_experiment.cache_codes',return_value=codes),patch('latent_ddp_experiment.draw',return_value=torch.zeros(1,3,96,96)):
            main()
    resumed=torch.load(tmp_path/'out/last.pt',weights_only=True)
    full=torch.load(tmp_path/'full/last.pt',weights_only=True)
    assert resumed['step']==full['step']==2
    for field in ['model','ema']:
        for name in full[field]:torch.testing.assert_close(resumed[field][name],full[field][name],atol=0,rtol=0)
    assert torch.equal(resumed['projection']['basis'],proj['basis'])
    assert torch.equal(resumed['center'],center) and torch.equal(resumed['scale'],scale)
