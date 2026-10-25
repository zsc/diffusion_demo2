import copy
import torch
from diffusion_demo.core.models.vae import VQVAE
from diffusion_demo.core.models.perceptual import PatchDiscriminator,assert_same_latent_mapping,load_perceptual


def test_mapping_check_allows_decoder_rejects_encoder():
    a=VQVAE();b=copy.deepcopy(a)
    with torch.no_grad():next(b.decoder.parameters()).add_(1)
    assert_same_latent_mapping(a,b)
    with torch.no_grad():next(b.encoder.parameters()).add_(1)
    import pytest
    with pytest.raises(ValueError,match='Latent mapping'):assert_same_latent_mapping(a,b)


def test_frozen_discriminator_still_differentiates_image():
    d=PatchDiscriminator(16).eval().requires_grad_(False)
    x=torch.randn(2,3,48,48,requires_grad=True);original=copy.deepcopy(d.state_dict())
    d(x).mean().backward()
    assert x.grad.abs().sum()>0
    assert all(p.grad is None for p in d.parameters())
    assert all(torch.equal(v,d.state_dict()[k]) for k,v in original.items())


def test_offline_perceptual_frozen_but_input_gradient():
    from pathlib import Path
    import pytest
    weights=Path('runs/vq_perceptual_20261025/assets/lpips_squeeze_v01.pth')
    if not weights.exists():pytest.skip('Explicit pretrained metric asset not installed')
    pytest.importorskip('lpips')
    model=load_perceptual(weights)
    x=torch.randn(2,3,48,48,requires_grad=True).tanh();x.retain_grad()
    y=torch.randn_like(x).tanh()
    model(x,y).mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert all(p.grad is None for p in model.parameters())
    assert model(x.detach(),x.detach()).abs().max()<1e-6


def test_joint_perceptual_training_and_selection(tmp_path):
    import numpy as np
    import pytest
    from pathlib import Path
    from diffusion_demo.core.training.train_vae import train_vae_model
    weights=Path('runs/vq_perceptual_20261025/assets/lpips_squeeze_v01.pth')
    if not weights.exists():pytest.skip('Explicit pretrained metric asset not installed')
    pytest.importorskip('lpips')
    data=tmp_path/'images.npy';np.save(data,np.random.default_rng(4).integers(0,256,(12,48,48,3),dtype=np.uint8))
    result=train_vae_model(model_type='vqvae',epochs=2,batch_size=4,data_limit=12,
        data_path=str(data),image_size=48,latent_spatial_size=12,embedding_dim=4,num_embeddings=16,
        vq_hidden_dims_enc=[8,8],vq_hidden_dims_dec=[8,8],codebook_decay=.99,
        model_ema_halflife_images=8,perceptual_weights=str(weights),perceptual_loss_weight=.1,
        selection_metric='lpips',split_mode='validation_first',val_count=4,
        device='cpu',precision='fp32',output_dir=str(tmp_path/'run'))
    payload=torch.load(result['best_model_path'],weights_only=True)
    assert 0<=payload['metrics']['val_lpips']<5
    assert payload['config']['selection_metric']=='lpips'
    assert payload['metrics']['perceptual_loss']>0
    assert payload['model_state_dict']['vq.ema_updates']>0
