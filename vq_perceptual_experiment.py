"""Decoder-only VQ-VAE ablations on 16 reserved GPUs; fixed codebook and data split."""
import argparse,copy,hashlib,json,math,os,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusion_demo.core.models.perceptual import load_perceptual,PatchDiscriminator,assert_same_latent_mapping
from diffusion_demo.core.training.train_dit import load_vae
from diffusion_demo.utils.torch_env import MixedPrecision
from latent_ddp_experiment import decode_codes,grid,write_json


@torch.no_grad()
def validate(model, images, codes, perceptual, amp, rank, world, output=None):
    model.eval(); device=next(model.parameters()).device
    totals=torch.zeros(3,device=device); counts=torch.zeros(model.vq.num_embeddings,device=device)
    ids=torch.arange(rank,1000,world,device=device); previews=[]
    for ix in ids.split(32):
        x=images[ix].permute(0,3,1,2).float()/127.5-1
        with amp.autocast():
            z=decode_codes(codes[ix],model.vq.embeddings.weight,model.latent_spatial_size)
            y=model.decode(z)
        # Report the same full precision perceptual metric across all arms.
        p=perceptual(y.float(),x).flatten()
        totals+=torch.stack(((y.float()-x).square().mean((1,2,3)).sum()/4,p.sum(),torch.tensor(len(ix),device=device)))
        counts+=torch.bincount(codes[ix].flatten().long(),minlength=len(counts))
        if output:
            previews.extend((int(i),a.cpu(),b.cpu()) for i,a,b in zip(ix,x,y) if int(i)<32)
    if world>1:dist.all_reduce(totals);dist.all_reduce(counts)
    probs=counts/counts.sum();mse=(totals[0]/totals[2]).item()
    metrics=dict(psnr=-10*math.log10(mse),mse=mse,lpips=(totals[1]/totals[2]).item(),codes_used=int((counts>0).sum()),perplexity=float((-(probs*probs.clamp_min(1e-12).log()).sum()).exp()))
    if output:
        torch.save(previews,output/f'preview{rank}.pt')
        if world>1:dist.barrier()
        if rank==0:
            allp=sorted([i for r in range(world) for i in torch.load(output/f'preview{r}.pt',weights_only=True)],key=lambda a:a[0])
            grid(torch.stack([v for _,a,b in allp for v in (a,b)]),output/'reconstructions.png')
    return metrics


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--output',required=True);ap.add_argument('--cpu',action='store_true')
    args=ap.parse_args();cfg=json.loads(Path(args.config).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    torch.set_num_threads(2);torch.manual_seed(cfg.get('seed',0))
    device=torch.device('cpu' if args.cpu else f'cuda:{local}')
    if device.type=='cuda':torch.cuda.set_device(device)
    if world>1:dist.init_process_group('gloo' if args.cpu else 'nccl')
    amp=MixedPrecision(device,'fp32' if args.cpu else 'bf16')
    source=torch.load(cfg['autoencoder'],map_location='cpu',weights_only=True)
    model=load_vae(cfg['autoencoder'],device=device)
    frozen={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith(('encoder.','vq.'))}
    model.decoder.requires_grad_(True);ema=copy.deepcopy(model).eval().requires_grad_(False)
    decoder=DDP(model.decoder,device_ids=[local] if device.type=='cuda' else None) if world>1 else model.decoder
    optimizer=torch.optim.Adam(model.decoder.parameters(),lr=cfg.get('lr',1e-4),betas=(.9,.99))
    perceptual=load_perceptual(cfg['perceptual_weights'],device)
    discriminator=PatchDiscriminator().to(device)
    gan_weight=cfg.get('gan_weight',0);perceptual_weight=cfg.get('perceptual_weight',0)
    dd=DDP(discriminator,device_ids=[local] if device.type=='cuda' else None) if world>1 and gan_weight else discriminator
    dopt=torch.optim.Adam(discriminator.parameters(),lr=cfg.get('discriminator_lr',1e-4),betas=(.5,.9))
    arrays=np.load(cfg['data']);code_arrays=np.load(cfg['codes'])
    if arrays.ndim!=4 or arrays.shape[1:]!=(model.image_size,model.image_size,3) or len(arrays)<=1000 or code_arrays.shape!=(len(arrays),model.latent_spatial_size**2):raise ValueError('Expected native images and matching latent codes, first 1000 held out')
    # Cache identity must match both the original autoencoder and data.
    meta=json.loads(Path(cfg['codes']).with_suffix('.json').read_text())
    if meta['data_sha256']!=hashlib.sha256(Path(cfg['data']).read_bytes()).hexdigest():raise ValueError('Wrong code cache dataset')
    cache_model=load_vae(str(Path(cfg['codes']).parent/'autoencoder.pth'),device='cpu')
    assert_same_latent_mapping(model,cache_model);del cache_model
    images=torch.from_numpy(arrays).to(device);codes=torch.from_numpy(code_arrays.astype(np.int64)).to(device)
    batch=cfg.get('batch_size',512);steps=cfg.get('steps',2048);every=cfg.get('evaluate_every',512)
    if batch%world:raise ValueError('Batch not divisible by world')
    if (out/'result.json').exists():raise ValueError('Refusing to overwrite completed experiment')
    history=[];best=float('inf');start=time.perf_counter()
    initial=validate(ema,images,codes,perceptual,amp,rank,world)
    if rank==0:write_json(out/'config.json',cfg);write_json(out/'initial.json',initial)
    torch.manual_seed(cfg.get('seed',0)+rank+100)
    for step in range(1,steps+1):
        model.eval();model.decoder.train();ix=torch.randint(1000,len(images),(batch//world,),device=device)
        x=images[ix].permute(0,3,1,2).float()/127.5-1
        z=decode_codes(codes[ix],model.vq.embeddings.weight,model.latent_spatial_size)
        active=bool(gan_weight and step>cfg.get('gan_start',256))
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast():
            y=decoder(z)
            mse=(y.float()-x).square().mean()
            pl=perceptual(y,x).mean() if perceptual_weight else mse.new_zeros(())
        dl=mse.new_zeros(());gl=mse.new_zeros(())
        if active:
            discriminator.requires_grad_(True);dopt.zero_grad(set_to_none=True)
            with amp.autocast():
                real,fake=dd(torch.cat((x,y.detach()))).chunk(2)
                dl=(nn.functional.relu(1-real.float()).mean()+nn.functional.relu(1+fake.float()).mean())*.5
            dl.backward();dn=nn.utils.clip_grad_norm_(discriminator.parameters(),1)
            if not torch.isfinite(dn):raise FloatingPointError('Nonfinite discriminator gradient')
            dopt.step();discriminator.requires_grad_(False)
            # Unwrapped D is deliberate: no D gradient or reducer state in the G phase.
            with amp.autocast():gl=-discriminator(y).float().mean()
        loss=mse+perceptual_weight*pl+gan_weight*gl
        loss.backward();norm=nn.utils.clip_grad_norm_(model.decoder.parameters(),1)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite decoder gradient')
        optimizer.step()
        decay=min(.5**(batch/cfg.get('ema_halflife_images',8192)),(1+step)/(10+step))
        with torch.no_grad():
            for a,b in zip(ema.decoder.parameters(),model.decoder.parameters()):a.lerp_(b,1-decay)
            for a,b in zip(ema.decoder.buffers(),model.decoder.buffers()):a.copy_(b)
        if step%100==0 and rank==0:
            write_json(out/'progress.json',dict(step=step,steps=steps,images=step*batch,seconds=time.perf_counter()-start,mse=mse.item(),lpips=pl.item(),d_loss=dl.item(),g_loss=gl.item()))
        if step%every==0 or step==steps:
            # Use common BN buffers for reproducible global heldout evaluation.
            if world>1:
                for b in ema.decoder.buffers():dist.broadcast(b,0)
            metrics=validate(ema,images,codes,perceptual,amp,rank,world,out)
            metrics.update(step=step,images=step*batch,seconds=time.perf_counter()-start)
            score=metrics['lpips'] if perceptual_weight or gan_weight else metrics['mse']
            if rank==0:
                payload={k:source[k] for k in ['model_type','model_config']}
                payload.update(model_state_dict={k:v.cpu().clone() for k,v in ema.state_dict().items()},metrics=metrics,decoder_finetune=cfg)
                torch.save(payload,out/f'ae_{step}.pth')
                if score<best:
                    best=score;torch.save(payload,out/'best.pth');write_json(out/'best.json',metrics)
                history.append(metrics);write_json(out/'history.json',history);print(json.dumps(metrics),flush=True)
            if world>1:dist.barrier()
    if any(not torch.equal(value,model.state_dict()[key].cpu()) for key,value in frozen.items()):raise AssertionError('Frozen latent mapping changed')
    if rank==0:
        write_json(out/'result.json',dict(initial=initial,history=history,world_size=world,seconds=time.perf_counter()-start,frozen_latent_mapping_verified=True))
        torch.save(dict(decoder=model.decoder.state_dict(),ema_decoder=ema.decoder.state_dict(),optimizer=optimizer.state_dict(),discriminator=discriminator.state_dict(),discriminator_optimizer=dopt.state_dict(),config=cfg,step=steps),out/'training_state.pt')
    if world>1:dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
