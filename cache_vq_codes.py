"""Cache native-resolution VQ codes inside a reserved distributed task."""
import argparse,os,shutil
from pathlib import Path
import torch
import torch.distributed as dist
from latent_ddp_experiment import cache_codes
from diffusion_demo.core.training.train_dit import load_vae
from diffusion_demo.utils.torch_env import MixedPrecision

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--autoencoder',required=True);ap.add_argument('--data',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
 rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0));torch.set_num_threads(2)
 device=torch.device('cuda',local);torch.cuda.set_device(device)
 if world>1:dist.init_process_group('nccl')
 out=Path(args.output)
 if rank==0:
  out.mkdir(parents=True,exist_ok=True)
  target=out/'autoencoder.pth'
  if target.exists():assert target.read_bytes()==Path(args.autoencoder).read_bytes()
  else:shutil.copy2(args.autoencoder,target)
 if world>1:dist.barrier()
 vae=load_vae(str(out/'autoencoder.pth'),device=device).eval();codes=cache_codes(vae,args.data,out,device,MixedPrecision(device,'bf16'),rank,world)
 if rank==0:print({'code_shape':codes.shape,'image_size':vae.image_size},flush=True)
 if world>1:dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
