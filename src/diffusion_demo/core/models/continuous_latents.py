"""Frozen continuous VAE posterior caches and train-only channel normalization."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def validate_data_expansion(previous, expanded, val_count=1000):
    """Allow added training images only; preserve every old ID and pixel in order."""
    old,new=np.load(previous,mmap_mode='r'),np.load(expanded,mmap_mode='r')
    if old.dtype!=np.uint8 or new.dtype!=old.dtype or old.shape[1:]!=new.shape[1:] or not val_count<len(old)<len(new):
        raise ValueError('Data expansion requires a larger compatible uint8 cache')
    for start in range(0,len(old),256):
        if not np.array_equal(old[start:start+256],new[start:min(start+256,len(old))]):
            raise ValueError('Data expansion changed existing pixels or their order')
    a=json.loads(Path(previous).with_suffix('.json').read_text())
    b=json.loads(Path(expanded).with_suffix('.json').read_text())
    if a.get('split_mode')!='validation_first' or b.get('split_mode')!='validation_first':
        raise ValueError('Data expansion requires traced validation-first datasets')
    av,at=a['images']['validation'],a['images']['train'];bv,bt=b['images']['validation'],b['images']['train']
    if len(av)!=val_count or av!=bv or at!=bt[:len(at)] or len(av)+len(at)!=len(old) or len(bv)+len(bt)!=len(new):
        raise ValueError('Data expansion changed existing image IDs or split')
    if len(set(bv+bt))!=len(bv+bt):raise ValueError('Duplicate IDs or train/validation overlap')
    return dict(previous_data=str(previous),expanded_data=str(expanded),previous_train_count=len(at),
                training_count=len(bt),validation_count=len(bv),original_prefix_byte_identical=True)


def posterior_normalization(cache, mode='sample', val_count=1000):
    if mode not in ['mu','sample']:raise ValueError('Unknown posterior mode')
    if len(cache)<=val_count:raise ValueError('No training latents')
    c=cache.shape[2];total=torch.zeros(c,dtype=torch.float64);second=total.clone();n=0
    for start in range(val_count,len(cache),512):
        a=torch.from_numpy(np.array(cache[start:start+512],copy=True)).double()
        mu,lv=a[:,0],a[:,1];total+=mu.sum((0,2,3))
        second+=(mu.square()+(lv.exp() if mode=='sample' else 0)).sum((0,2,3))
        n+=mu.shape[0]*mu.shape[2]*mu.shape[3]
    mean=total/n;variance=(second/n-mean.square()).clamp_min(1e-8)
    return mean.float()[None,:,None,None],variance.sqrt().float()[None,:,None,None]


def sample_posterior(batch,mode='sample'):
    mu,lv=batch[:,0].float(),batch[:,1].float()
    if mode=='mu':return mu
    if mode!='sample':raise ValueError('Unknown posterior mode')
    return mu+lv.mul(.5).exp()*torch.randn_like(mu)


@torch.no_grad()
def cache_posterior(vae,data,output,device,amp,rank,world):
    output=Path(output);cache=output/'posterior.npy';meta=output/'posterior.json'
    expected=dict(data_sha256=file_sha256(data),autoencoder_sha256=file_sha256(output/'autoencoder.pth'),val_count=1000,format='N,mu_logvar,C,H,W float16')
    if cache.exists():
        if json.loads(meta.read_text())!=expected:raise ValueError('Stale posterior cache')
        return np.load(cache,mmap_mode='r')
    images=np.load(data,mmap_mode='r')
    if images.dtype!=np.uint8 or images.shape[1:]!=(vae.image_size,vae.image_size,3) or len(images)<=1000:raise ValueError('Invalid native image cache')
    ids=np.arange(rank,len(images),world);parts=[];vae.eval()
    for start in range(0,len(ids),128):
        x=torch.from_numpy(images[ids[start:start+128]].copy()).permute(0,3,1,2).to(device).float()/127.5-1
        with amp.autocast():mu,lv=vae.encoder(x)
        a=torch.stack([mu.float(),lv.float()],1)
        if not torch.isfinite(a).all():raise FloatingPointError('Nonfinite posterior')
        parts.append(a.cpu().numpy().astype(np.float16))
    np.save(output/f'posterior_rank{rank}.npy',np.concatenate(parts))
    if world>1:dist.barrier()
    if rank==0:
        merged=np.lib.format.open_memmap(output/'posterior_tmp.npy',mode='w+',dtype=np.float16,shape=(len(images),2,vae.latent_dim,vae.latent_spatial_size,vae.latent_spatial_size))
        for r in range(world):merged[r::world]=np.load(output/f'posterior_rank{r}.npy')
        merged.flush();del merged
        meta.write_text(json.dumps(expected,indent=2));(output/'posterior_tmp.npy').replace(cache)
    if world>1:dist.barrier()
    return np.load(cache,mmap_mode='r')
