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


def test_data_expansion_preserves_ids_pixels_and_disjoint_holdout(tmp_path):
    import pytest
    from diffusion_demo.core.models.continuous_latents import validate_data_expansion,file_sha256
    import hashlib
    rng=np.random.default_rng(2);old=rng.integers(0,256,(5,4,4,3),dtype=np.uint8)
    new=np.concatenate([old,rng.integers(0,256,(2,4,4,3),dtype=np.uint8)])
    a,b=tmp_path/'old.npy',tmp_path/'new.npy';np.save(a,old);np.save(b,new)
    ma=dict(split_mode='validation_first',images=dict(validation=['v0','v1'],train=['t0','t1','t2']))
    mb=dict(split_mode='validation_first',images=dict(validation=['v0','v1'],train=['t0','t1','t2','t3','t4']))
    a.with_suffix('.json').write_text(json.dumps(ma));b.with_suffix('.json').write_text(json.dumps(mb))
    assert validate_data_expansion(a,b,val_count=2)['training_count']==5
    assert file_sha256(b)==hashlib.sha256(b.read_bytes()).hexdigest()
    for index in [0,3]:
        bad=new.copy();bad[index,0,0,0]^=1;np.save(b,bad)
        with pytest.raises(ValueError,match='pixels'):validate_data_expansion(a,b,val_count=2)
    np.save(b,new);mb['images']['train'][-1]='v0';b.with_suffix('.json').write_text(json.dumps(mb))
    with pytest.raises(ValueError,match='overlap'):validate_data_expansion(a,b,val_count=2)


def test_expanded_resume_keeps_latent_coordinates_and_requires_opt_in(tmp_path):
    import pytest
    torch.manual_seed(8)
    model_cfg=dict(latent_dim=2,hidden_dims=[4,8],image_size=48,latent_spatial_size=4)
    vae=VAE(**model_cfg);ae=tmp_path/'vae.pth'
    torch.save(dict(model_type='vae',model_config=model_cfg,model_state_dict=vae.state_dict(),metrics={}),ae)
    old=np.zeros((1004,2,2,4,4),np.float16);old[:,1]=-2
    new=np.concatenate([old,np.full((4,2,2,4,4),2,np.float16)])
    cfg=dict(autoencoder=str(ae),data='old',batch_size=4,lr=1e-3,width=24,depth=1,heads=3,patch_size=2,seed=1,posterior_mode='sample',save_every_images=8)
    def run(folder,images,data,cache,resume=None,allow=False):
        cp=tmp_path/'config.json';cp.write_text(json.dumps(dict(cfg,images=images,data=data)))
        argv=['train','--cpu','--config',str(cp),'--output',str(folder)]
        if resume:argv+=['--resume',str(resume)]
        if allow:argv+=['--allow-data-expansion']
        with patch.object(sys,'argv',argv),patch('latent_ddp_experiment.cache_posterior',return_value=cache),patch('latent_ddp_experiment.draw',return_value=torch.zeros(1,3,48,48)),patch('latent_ddp_experiment.validate_data_expansion',return_value={'training_count':8}):main()
    run(tmp_path/'old',8,'old',old)
    parent=tmp_path/'old/last.pt'
    with pytest.raises(ValueError,match='allow-data-expansion'):run(tmp_path/'rejected',16,'new',new,parent)
    run(tmp_path/'expanded',16,'new',new,parent,True)
    a=torch.load(parent,weights_only=True);b=torch.load(tmp_path/'expanded/last.pt',weights_only=True)
    assert b['step']==4 and b['config']['data']=='new'
    assert torch.equal(a['center'],b['center']) and torch.equal(a['scale'],b['scale'])
    assert all(v['step']==4 for v in b['optimizer']['state'].values())
    assert json.loads((tmp_path/'expanded/data_expansion.json').read_text())['exact_training_trajectory'] is False
