"""Distributed latent DiT: VQ codes or continuous VAE posteriors, JiT flow.

Each task uses one reserved node. Checkpoints include the autoencoder, optimizer,
EMA and per-rank RNG state. Sampling seeds are independent of batch/rank layout.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image

from diffusion_demo.core.models.dit import DiT
from diffusion_demo.core.models.vae import VAE
from diffusion_demo.core.models.continuous_latents import cache_posterior, posterior_normalization, sample_posterior
from diffusion_demo.core.models.pixel_jit import flow_loss, heun_sample
from diffusion_demo.core.training.train_dit import load_vae
from diffusion_demo.utils.torch_env import MixedPrecision


def write_json(path, value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def grid(images, path):
    arrays=((images.float().cpu().clamp(-1,1)+1)*127.5).round().byte().permute(0,2,3,1).numpy()
    n,h,w,_=arrays.shape;canvas=np.full((math.ceil(n/8)*(h+2),8*(w+2),3),32,np.uint8)
    for i,im in enumerate(arrays):
        y,x=divmod(i,8);canvas[y*(h+2):y*(h+2)+h,x*(w+2):x*(w+2)+w]=im
    Image.fromarray(canvas).resize((canvas.shape[1]*3,canvas.shape[0]*3),Image.Resampling.NEAREST).save(path)


class LatentDiT(DiT):
    def forward(self,x,t):
        return super().forward(x,t*1000)


def decode_codes(codes, weight, size):
    return torch.nn.functional.embedding(codes.long(),weight).reshape(len(codes),size,size,-1).permute(0,3,1,2).contiguous()


def fit_pca(weight, probabilities, components):
    """Train-code-weighted PCA in FP64 on CPU. Never uses heldout assignments."""
    w=weight.detach().cpu().double();p=probabilities.detach().cpu().double()
    if not 1 <= components <= w.shape[1]:raise ValueError('Invalid PCA component count')
    offset=(w*p[:,None]).sum(0);centered=w-offset
    covariance=centered.T@(centered*p[:,None])
    values,vectors=torch.linalg.eigh(covariance)
    order=torch.argsort(values,descending=True);values=values[order];basis=vectors[:,order[:components]]
    # Stable signs make saved transforms easy to compare across processes.
    pivots=basis.abs().argmax(0);sign=basis[pivots,torch.arange(components)].sign();basis*=sign
    return dict(basis=basis.float(),offset=offset.float().view(1,-1,1,1),
                retained_variance=float(values[:components].sum()/values.sum())), values[:components].clamp_min(1e-8).sqrt().float().view(1,-1,1,1)


def project_latent(z, projection):
    if projection is None:return z
    return torch.einsum('bchw,ck->bkhw',z-projection['offset'],projection['basis'])


def restore_latent(z, projection):
    if projection is None:return z
    return torch.einsum('bkhw,ck->bchw',z,projection['basis'])+projection['offset']


def projection_to(projection, device):
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in projection.items()} if projection is not None else None


def validate_guide(main, guide):
    if main['model_config'] != guide['model_config']:
        raise ValueError('Guide architecture differs from the main model')
    if any(not torch.equal(main[k].cpu(),guide[k].cpu()) for k in ['center','scale']):
        raise ValueError('Guide latent normalization differs from the main model')
    a,b=main.get('projection'),guide.get('projection')
    if (a is None)!=(b is None) or (a is not None and any(not torch.equal(a[k].cpu(),b[k].cpu()) for k in ['basis','offset'])):
        raise ValueError('Guide latent projection differs from the main model')


@torch.no_grad()
def cache_codes(vae,data,output,device,amp,rank,world):
    # No augmentation: same explicit heldout split as pixel JiT.
    sha=hashlib.sha256(Path(data).read_bytes()).hexdigest()
    vq_sha=hashlib.sha256(Path(output/'autoencoder.pth').read_bytes()).hexdigest()
    cache=output/'codes.npy';meta=output/'codes.json'
    expected=dict(data_sha256=sha,autoencoder_sha256=vq_sha,val_count=1000)
    if cache.exists():
        if json.loads(meta.read_text()) != expected:raise ValueError('Stale latent cache')
        return np.load(cache)
    images=np.load(data,mmap_mode='r')
    if images.dtype!=np.uint8 or images.shape[1:]!=(vae.image_size,vae.image_size,3) or len(images)<=1000:
        raise ValueError(f'Expected uint8 N,{vae.image_size},{vae.image_size},3 data with >1000 images')
    ids=np.arange(rank,len(images),world);parts=[]
    for start in range(0,len(ids),128):
        x=torch.from_numpy(images[ids[start:start+128]].copy()).permute(0,3,1,2).to(device).float()/127.5-1
        with amp.autocast():
            _,_,indices=vae.vq(vae.encoder(x))
        parts.append(indices.reshape(len(x),-1).cpu().numpy().astype(np.int16))
    np.save(output/f'codes_rank{rank}.npy',np.concatenate(parts))
    if world>1:dist.barrier()
    if rank==0:
        allcodes=np.empty((len(images),vae.latent_spatial_size**2),np.int16)
        for r in range(world):allcodes[r::world]=np.load(output/f'codes_rank{r}.npy')
        np.save(cache,allcodes);write_json(meta,expected)
    if world>1:dist.barrier()
    return np.load(cache)


@torch.no_grad()
def draw(model,vae,center,scale,amp,ids,seed,steps,requantize=False,guide=None,guidance=1,projection=None):
    device=center.device;shape=(model.in_channels,*vae.get_latent_shape()[2:])
    noise=torch.stack([torch.randn(shape,device=device,generator=torch.Generator(device=device).manual_seed(seed+i)) for i in ids])
    def predictor(x,t):
        p=model(x,t)
        if guide is not None:
            q=guide(x,t);mask=((t>=.1)&(t<=.8)).view(-1,1,1,1)
            p=torch.where(mask,q+guidance*(p-q),p)
        return p
    with amp.autocast():
        z=restore_latent(heun_sample(predictor,noise,steps)*scale+center,projection)
        if requantize:
            if isinstance(vae,VAE):raise ValueError('Continuous VAE has no quantizer')
            z=vae.vq(z)[0]
        images=vae.decode(z)
    if not images.isfinite().all():raise FloatingPointError('Nonfinite samples')
    return images.float()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['train','eval'],default='train')
    ap.add_argument('--config',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--checkpoint');ap.add_argument('--resume');ap.add_argument('--guide')
    ap.add_argument('--init-checkpoint', help='Spatial warm start from EMA; preserve latent coordinates, reset optimizer and 96px image counter')
    ap.add_argument('--autoencoder', help='Evaluation decoder override; encoder and quantizer must match the trained prior')
    ap.add_argument('--guidance',type=float,default=1);ap.add_argument('--requantize',action='store_true')
    ap.add_argument('--seed',type=int,default=3333);ap.add_argument('--count',type=int,default=1024)
    ap.add_argument('--weights-dir');ap.add_argument('--cpu',action='store_true')
    args=ap.parse_args();cfg=json.loads(Path(args.config).read_text())
    if args.init_checkpoint and (args.resume or args.mode!='train'):raise ValueError('Warm start is exclusive with resume/eval')
    rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    torch.set_num_threads(2)
    device=torch.device('cpu' if args.cpu else f'cuda:{local}')
    if device.type=='cuda':torch.cuda.set_device(device)
    if world>1:dist.init_process_group('gloo' if args.cpu else 'nccl')
    amp=MixedPrecision(device,'fp32' if args.cpu else 'bf16')
    torch.manual_seed(cfg.get('seed',0))
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if args.mode=='eval':
        payload=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        model=LatentDiT(**payload['model_config']).to(device).eval()
        model.load_state_dict(payload['ema'])
        vae_path=Path(args.checkpoint).parent/'autoencoder.pth'
        vae=load_vae(str(vae_path),device=device)
        if args.autoencoder:
            from diffusion_demo.core.models.perceptual import assert_same_latent_mapping
            replacement=load_vae(args.autoencoder,device=device)
            assert_same_latent_mapping(vae,replacement)
            vae,vae_path=replacement,Path(args.autoencoder)
        center,scale=payload['center'].to(device),payload['scale'].to(device)
        projection=projection_to(payload.get('projection'),device)
        guide=None
        if args.guide:
            guide=copy.deepcopy(model)
            guide_payload=torch.load(args.guide,map_location='cpu',weights_only=True)
            validate_guide(payload,guide_payload);guide.load_state_dict(guide_payload['ema'])
        from cleanfid.inception_torchscript import InceptionV3W
        from cleanfid.resize import build_resizer
        inception=InceptionV3W(args.weights_dir,download=False,resize_inside=False).to(device).eval()
        resize=build_resizer('clean')
        @torch.no_grad()
        def features(arrays):
            x=torch.from_numpy(np.stack([resize(im) for im in arrays])).permute(0,3,1,2).to(device)
            return inception(x).float().cpu().numpy()
        ids=list(range(rank,args.count,world));fake=[];previews=[]
        for start in range(0,len(ids),16):
            selected=ids[start:start+16]
            im=draw(model,vae,center,scale,amp,selected,args.seed,cfg.get('sampling_steps',50),args.requantize,guide,args.guidance,projection)
            arr=((im.clamp(-1,1)+1)*127.5).round().byte().permute(0,2,3,1).cpu().numpy()
            fake.append(features(arr));previews.extend((i,a) for i,a in zip(selected,arr) if i<64)
        data=np.load(cfg['data'],mmap_mode='r');realids=list(range(rank,1000,world))
        real=[features(data[realids[i:i+16]]) for i in range(0,len(realids),16)]
        np.savez(output/f'features_rank{rank}.npz',fake=np.concatenate(fake),real=np.concatenate(real),fake_ids=ids,real_ids=realids)
        torch.save(previews,output/f'preview_rank{rank}.pt')
        if world>1:dist.barrier()
        if rank==0:
            shards=[np.load(output/f'features_rank{r}.npz') for r in range(world)]
            for key in ['real','fake']:
                values=np.concatenate([a[key] for a in shards]);order=np.concatenate([a[key+'_ids'] for a in shards])
                np.save(output/f'{key}_features.npy',values[np.argsort(order)])
            previews=sorted([item for r in range(world) for item in torch.load(output/f'preview_rank{r}.pt',weights_only=False)],key=lambda a:a[0])
            grid(torch.from_numpy(np.stack([a for _,a in previews])).permute(0,3,1,2).float()/127.5-1,output/'samples64.png')
            write_json(output/'evaluation.json',dict(checkpoint=args.checkpoint,autoencoder_sha256=hashlib.sha256(vae_path.read_bytes()).hexdigest(),seed=args.seed,requantize=args.requantize,guide=args.guide,guidance=args.guidance,count=args.count,steps=cfg.get('sampling_steps',50),world_size=world))
    else:
        if args.autoencoder:raise ValueError('Training autoencoder belongs in the config')
        if (output/'last.pt').exists() and not args.resume:raise ValueError('Use resume or a new output')
        if rank==0:
            source=torch.load(cfg['autoencoder'],map_location='cpu',weights_only=True)
            source={k:source[k] for k in ['model_type','model_config','model_state_dict','metrics']}
            if (output/'autoencoder.pth').exists():
                existing=torch.load(output/'autoencoder.pth',map_location='cpu',weights_only=True)
                if existing['model_config'] != source['model_config'] or any(
                    key not in existing['model_state_dict'] or not torch.equal(value,existing['model_state_dict'][key])
                    for key,value in source['model_state_dict'].items()):
                    raise ValueError('Output autoencoder differs from configured source')
            else:
                torch.save(source,output/'autoencoder.pth')
            write_json(output/'config.json',cfg)
        if world>1:dist.barrier()
        vae=load_vae(str(output/'autoencoder.pth'),device=device)
        continuous=isinstance(vae,VAE);size=vae.latent_spatial_size
        if continuous:
            if cfg.get('pca_components') or args.init_checkpoint:raise ValueError('Continuous VAE starts in its own channel coordinates without PCA')
            cached=cache_posterior(vae,cfg['data'],output,device,amp,rank,world)
            posterior_mode=cfg.get('posterior_mode','sample')
            center,scale=posterior_normalization(cached,posterior_mode)
            center,scale=center.to(device),scale.to(device)
            train=torch.from_numpy(np.array(cached[1000:],copy=True)).to(device)
            channels=vae.latent_dim
            if rank==0:write_json(output/'normalization.json',dict(mode=posterior_mode,fitted_indices=f'1000:{len(cached)}',center=center.cpu().flatten().tolist(),scale=scale.cpu().flatten().tolist()))
        else:
            cached=cache_codes(vae,cfg['data'],output,device,amp,rank,world)
            train=torch.from_numpy(cached[1000:].astype(np.int64)).to(device)
            weight=vae.vq.embeddings.weight.detach()
            counts=torch.bincount(train.flatten(),minlength=len(weight)).float();probs=counts/counts.sum()
            center=(probs[:,None]*weight).sum(0).view(1,-1,1,1)
            variance=(probs[:,None]*(weight-center.flatten()).square()).sum(0)
            scale=variance.sqrt().clamp_min(1e-4).view(1,-1,1,1)
            channels=vae.embedding_dim
        projection=None;initial=None
        if args.init_checkpoint:
            initial=torch.load(args.init_checkpoint,map_location='cpu',weights_only=True)
            from diffusion_demo.core.models.resolution_transfer import validate_resolution_codec
            old_vae=load_vae(str(Path(args.init_checkpoint).parent/'autoencoder.pth'),device='cpu')
            validate_resolution_codec(old_vae,vae);del old_vae
            channels=initial['model_config']['in_channels']
            if cfg.get('pca_components',vae.embedding_dim)!=channels:raise ValueError('Warm-start channel mismatch')
            projection=projection_to(initial.get('projection'),device)
            center,scale=initial['center'].to(device),initial['scale'].to(device)
            if rank==0:write_json(output/'transfer.json',dict(checkpoint=args.init_checkpoint,
                source_sha256=hashlib.sha256(Path(args.init_checkpoint).read_bytes()).hexdigest(),
                source_model_config=initial['model_config'],target_latent_grid=size,
                inherited_latent_coordinates=True,optimizer_reset=True,new_resolution_image_counter=0))
        elif args.resume:
            # A resolution warm start can retain a PCA basis fitted at 48px.
            # Resume must keep those checkpoint coordinates, not refit on 96px.
            payload=torch.load(args.resume,map_location='cpu',weights_only=True)
            if payload['config']['data']!=cfg['data'] or payload['config'].get('batch_size',512)!=cfg.get('batch_size',512):
                raise ValueError('Exact resume requires the same data and global batch')
            if continuous and payload['config'].get('posterior_mode','sample')!=posterior_mode:raise ValueError('Resume posterior mode mismatch')
            channels=payload['model_config']['in_channels']
            if cfg.get('pca_components') not in [None,channels]:raise ValueError('Resume channel mismatch')
            projection=projection_to(payload.get('projection'),device)
            center,scale=payload['center'].to(device),payload['scale'].to(device)
        elif cfg.get('pca_components'):
            channels=cfg['pca_components'];projection,scale=fit_pca(weight,probs,channels)
            projection=projection_to(projection,device);scale=scale.to(device);center=torch.zeros_like(scale)
            if rank==0:write_json(output/'projection.json',dict(components=channels,retained_variance=projection['retained_variance'],fitted_indices='1000:30000'))
        mc=dict(img_size=size,in_channels=channels,patch_size=cfg.get('patch_size',2),hidden_size=cfg.get('width',384),depth=cfg.get('depth',8),num_heads=cfg.get('heads',6))
        model=LatentDiT(**mc).to(device)
        # Zero output improves early flow regression stability.
        torch.nn.init.zeros_(model.final_layer[-1].weight);torch.nn.init.zeros_(model.final_layer[-1].bias)
        if initial is not None:
            from diffusion_demo.core.models.resolution_transfer import resize_dit_state
            model.load_state_dict(resize_dit_state(initial['ema'],initial['model_config'],mc),strict=True)
        ema=copy.deepcopy(model).eval().requires_grad_(False)
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],betas=(.9,.95),weight_decay=.01)
        step=0;history=[]
        if args.resume:
            if (payload['model_config']!=mc or not torch.equal(payload['center'],center.cpu())
                    or not torch.equal(payload['scale'],scale.cpu())):raise ValueError('Resume model/data mismatch')
            from diffusion_demo.core.models.perceptual import assert_same_latent_mapping
            previous_vae=load_vae(str(Path(args.resume).parent/'autoencoder.pth'),device='cpu')
            assert_same_latent_mapping(vae,previous_vae);del previous_vae
            old_projection=payload.get('projection')
            if (old_projection is None)!=(projection is None):raise ValueError('Resume projection mismatch')
            if projection is not None and any(not torch.equal(old_projection[k],projection[k].cpu()) for k in ['basis','offset']):raise ValueError('Resume PCA basis mismatch')
            model.load_state_dict(payload['model']);ema.load_state_dict(payload['ema']);optimizer.load_state_dict(payload['optimizer']);step=payload['step']
            history=payload['history']
        ddp=DDP(model,device_ids=[local] if device.type=='cuda' else None) if world>1 else model
        torch.manual_seed(cfg.get('seed',0)+100+rank)
        if args.resume:
            if len(payload['rng'])!=world:raise ValueError('Exact resume requires same world size')
            torch.set_rng_state(payload['rng'][rank]['cpu'])
            if device.type=='cuda':torch.cuda.set_rng_state(payload['rng'][rank]['cuda'],device)
        batch=cfg.get('batch_size',512)
        if batch%world:raise ValueError('Global batch must be divisible by world size')
        target=math.ceil(cfg['images']/batch);save_every=max(1,cfg.get('save_every_images',524288)//batch)
        start=time.perf_counter();start_step=step
        while step<target:
            model.train();ids=torch.randint(len(train),(batch//world,),device=device)
            latent=sample_posterior(train[ids],posterior_mode) if continuous else decode_codes(train[ids],weight,size)
            clean=(project_latent(latent,projection)-center)/scale
            t=(torch.randn(len(clean),device=device)*.8-.8).sigmoid();noise=torch.randn_like(clean)
            warmup=min(1,(step+1)/max(1,cfg.get('warmup_images',65536)/batch))
            for group in optimizer.param_groups:group['lr']=cfg['lr']*warmup
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast():loss=flow_loss(ddp,clean,t,noise)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
            optimizer.step();step+=1
            decay=min(.5**(batch/cfg.get('ema_halflife_images',131072)),(1+step)/(10+step))
            with torch.no_grad():
                torch._foreach_mul_(list(ema.parameters()),decay)
                torch._foreach_add_(list(ema.parameters()),list(model.parameters()),alpha=1-decay)
            if step%100==0 or step==target:
                value=loss.detach().clone()
                if world>1:dist.all_reduce(value);value/=world
                if rank==0:
                    record=dict(step=step,images_seen=step*batch,loss=value.item(),seconds=time.perf_counter()-start,images_per_second=(step-start_step)*batch/(time.perf_counter()-start))
                    history.append(record);write_json(output/'progress.json',record);print(json.dumps(record),flush=True)
            if step%save_every==0 or step==target or step*batch in cfg.get('snapshot_images',[]):
                rng=dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device).cpu() if device.type=='cuda' else None)
                rngs=[None]*world
                if world>1:dist.all_gather_object(rngs,rng)
                else:rngs=[rng]
                if rank==0:
                    payload=dict(format_version=1,model_config=mc,config=cfg,model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),step=step,center=center.cpu(),scale=scale.cpu(),projection=projection_to(projection,'cpu'),rng=rngs,history=history)
                    torch.save(payload,output/'last.tmp');(output/'last.tmp').replace(output/'last.pt')
                    # Inference-only snapshots preserve weak guide and allow matched-budget comparison.
                    slim={k:v for k,v in payload.items() if k not in ['optimizer','rng','model','history']}
                    torch.save(slim,output/f'ema_{step*batch}.pt')
                    ema.eval()
                    grid(draw(ema,vae,center,scale,amp,list(range(64)),2027,50,projection=projection),output/f'samples_{step*batch}.png')
                if world>1:dist.barrier()
        if rank==0:write_json(output/'result.json',dict(images_seen=step*batch,world_size=world,parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-start))
    if world>1:dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
