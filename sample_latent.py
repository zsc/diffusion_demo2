"""Sample a trained VQ latent DiT without an Inception dependency."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from PIL import Image
import torch

from latent_ddp_experiment import LatentDiT, draw, grid, projection_to, validate_guide
from diffusion_demo.core.training.train_dit import load_vae
from diffusion_demo.utils.torch_env import MixedPrecision, get_device


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--autoencoder')
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--precision',choices=['auto','fp32','fp16','bf16'],default='auto')
    parser.add_argument('--count',type=int,default=64)
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--seed',type=int,default=2027)
    parser.add_argument('--steps',type=int,default=50)
    parser.add_argument('--requantize',action='store_true')
    parser.add_argument('--guide')
    parser.add_argument('--guidance',type=float,default=1)
    args=parser.parse_args()
    if min(args.count,args.batch,args.steps)<=0:raise ValueError('Positive count, batch and steps required')
    torch.set_num_threads(4)
    device=get_device(None if args.device=='auto' else args.device)
    checkpoint=Path(args.checkpoint)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=LatentDiT(**payload['model_config']).to(device).eval()
    model.load_state_dict(payload['ema'])
    vae_path=Path(args.autoencoder) if args.autoencoder else checkpoint.parent/'autoencoder.pth'
    vae=load_vae(vae_path,device=device)
    if args.autoencoder:
        from diffusion_demo.core.models.perceptual import assert_same_latent_mapping
        original=load_vae(checkpoint.parent/'autoencoder.pth',device='cpu')
        assert_same_latent_mapping(original,vae);del original
    center,scale=payload['center'].to(device),payload['scale'].to(device)
    projection=projection_to(payload.get('projection'),device)
    amp=MixedPrecision(device,args.precision)
    guide=None
    if args.guide:
        guide=LatentDiT(**payload['model_config']).to(device).eval()
        guide_payload=torch.load(args.guide,map_location='cpu',weights_only=True)
        validate_guide(payload,guide_payload);guide.load_state_dict(guide_payload['ema'])
    output=Path(args.output);(output/'images').mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();all_images=[]
    for offset in range(0,args.count,args.batch):
        ids=list(range(offset,min(args.count,offset+args.batch)))
        images=draw(model,vae,center,scale,amp,ids,args.seed,args.steps,args.requantize,guide,args.guidance,projection).cpu()
        arrays=((images.clamp(-1,1)+1)*127.5).round().byte().permute(0,2,3,1).numpy()
        for index,image in zip(ids,arrays):Image.fromarray(image).save(output/f'images/{index:05d}.png')
        all_images.append(images)
    grid(torch.cat(all_images),output/'grid.png')
    metadata=dict(vars(args),device=str(device),actual_precision=amp.precision,seconds=time.perf_counter()-start,
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  autoencoder_sha256=hashlib.sha256(vae_path.read_bytes()).hexdigest(),
                  trained_images=payload['step']*payload['config']['batch_size'])
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata))


if __name__=='__main__':main()
