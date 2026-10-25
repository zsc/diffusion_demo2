"""Distributed ordinary VAE reconstruction metrics, posterior mean versus fixed draws."""
import argparse,json,os,hashlib,math
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from diffusion_demo.core.models.vae import VAE
from diffusion_demo.core.models.perceptual import load_perceptual
from vae_pca_probe import gather,grid

@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--weights',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
 rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));device=torch.device('cuda:'+os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(device);torch.set_num_threads(2)
 if world>1:dist.init_process_group('nccl')
 d=torch.load(args.checkpoint,map_location='cpu',weights_only=True);assert d['model_type']=='vae';model=VAE(**d['model_config']).to(device).eval().requires_grad_(False);model.load_state_dict(d['model_state_dict'],strict=True)
 perceptual=load_perceptual(args.weights,device);data=np.load(args.data,mmap_mode='r');out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 ids=list(range(rank,1000,world));sums={m:np.zeros(3,np.float64) for m in ['mu','sample']};previews={m:[] for m in ['original','mu','sample']};kl_sum=0.;var_sum=torch.zeros(model.latent_dim,dtype=torch.float64);mean_sum=torch.zeros_like(var_sum);mean_squares=torch.zeros_like(var_sum);positions=0
 for offset in range(0,len(ids),16):
  batch=ids[offset:offset+16];x=torch.from_numpy(data[batch].copy()).permute(0,3,1,2).to(device).float()/127.5-1
  with torch.autocast('cuda',dtype=torch.bfloat16):mu,lv=model.encoder(x)
  mu,lv=mu.float(),lv.float();variance=lv.exp();kl_sum+=(-.5*(1+lv-mu.square()-variance)).sum().item()
  var_sum+=variance.sum((0,2,3)).cpu().double();mean_sum+=mu.sum((0,2,3)).cpu().double();mean_squares+=mu.square().sum((0,2,3)).cpu().double();positions+=len(x)*mu.shape[-1]*mu.shape[-2]
  eps=torch.stack([torch.randn(mu.shape[1:],generator=torch.Generator().manual_seed(2027+i)) for i in batch]).to(device)
  for mode,z in [('mu',mu),('sample',mu+variance.sqrt()*eps)]:
   with torch.autocast('cuda',dtype=torch.bfloat16):y=model.decode(z)
   error=((y.float()-x)/2).square().flatten(1).mean(1);p=perceptual(y.float(),x).flatten();sums[mode]+=np.array([error.sum().item(),p.sum().item(),len(x)])
   pixels=((y.float().clamp(-1,1)+1)*127.5).round().byte().permute(0,2,3,1).cpu().numpy();previews[mode].extend((i,a) for i,a in zip(batch,pixels) if i<64)
  previews['original'].extend((i,a) for i,a in zip(batch,data[batch]) if i<64)
 all_values=gather((sums,previews,kl_sum,var_sum,mean_sum,mean_squares,positions),world)
 if rank==0:
  result=dict(model_config=d['model_config'],checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),checkpoint_metrics=d['metrics'],world_size=world,heldout_count=1000,posterior_noise_seed=2027,mode='reconstruction, not unconditional generation')
  for mode in sums:
   total=sum(v[0][mode] for v in all_values);mse=total[0]/total[2];result[mode]=dict(mse=mse,psnr=-10*math.log10(mse),lpips=total[1]/total[2])
  n=sum(v[6] for v in all_values);mean=sum(v[4] for v in all_values)/n;signal=sum(v[5] for v in all_values)/n-mean.square();noise=sum(v[3] for v in all_values)/n
  result.update(kl_per_image=sum(v[2] for v in all_values)/1000,signal_variance_per_channel=signal.tolist(),posterior_variance_per_channel=noise.tolist())
  for mode in previews:
   items=dict(item for v in all_values for item in v[1][mode]);grid([items[i] for i in range(64)],out/(mode+'.png'))
  (out/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 if world>1:dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
