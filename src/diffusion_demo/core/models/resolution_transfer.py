"""Explicit spatial transfer with a fixed VQ coordinate system."""
import copy
import torch
import torch.nn.functional as F


def validate_resolution_codec(old, new):
    """Changing grid size is allowed; changing encoder or codebook is not."""
    if old.image_size * new.latent_spatial_size != new.image_size * old.latent_spatial_size:
        raise ValueError('Image/latent downsampling ratio changed')
    a={k:v for k,v in old.state_dict().items() if k.startswith(('encoder.','vq.'))}
    b={k:v for k,v in new.state_dict().items() if k.startswith(('encoder.','vq.'))}
    if a.keys()!=b.keys() or any(not torch.equal(v.cpu(),b[k].cpu()) for k,v in a.items()):
        raise ValueError('Resolution transfer requires identical encoder and codebook')


def resize_dit_state(state, old_config, new_config):
    old=dict(old_config);new=dict(new_config)
    old_size=old.pop('img_size');new_size=new.pop('img_size')
    if old!=new:raise ValueError('Only the spatial size may change during resolution transfer')
    patch=old['patch_size']
    if old_size%patch or new_size%patch:raise ValueError('Invalid patch grid')
    result=copy.deepcopy(state)
    if old_size!=new_size:
        pos=state['pos_embed'];g=old_size//patch
        if pos.shape[1]!=g*g:raise ValueError('Position embedding shape mismatch')
        result['pos_embed']=F.interpolate(pos.float().reshape(1,g,g,-1).permute(0,3,1,2),
            size=(new_size//patch,new_size//patch),mode='bicubic',align_corners=False).flatten(2).transpose(1,2).to(pos.dtype)
    return result
