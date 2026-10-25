"""Render the unselected first 32 holdout images using actual selected checkpoints."""
import json
from pathlib import Path
import numpy as np
import torch
from diffusion_demo.core.training.train_dit import load_vae
from diffusion_demo.core.models.perceptual import assert_same_latent_mapping
from latent_ddp_experiment import grid

@torch.inference_mode()
def main():
    torch.set_num_threads(4);root=Path(__file__).resolve().parents[1];run=root/'runs/vq_perceptual_20261025'
    x=torch.from_numpy(np.load(root/'cache/celeba_48x48_sprites_v2.npy')[:32].copy()).permute(0,3,1,2).float()/127.5-1
    original=load_vae(root/'runs/vq_latent_20261025/release/autoencoder.pth',device='cpu');z=original.encode(x)
    for p in sorted((run/'decoder').glob('*/best.pth')):
        model=load_vae(p,device='cpu');assert_same_latent_mapping(original,model);y=model.decode(z)
        grid(torch.stack([v for pair in zip(x,y) for v in pair]),p.parent/'recon_best.png')
        (p.parent/'recon_best_metadata.json').write_text(json.dumps(dict(checkpoint=str(p),device='cpu',precision='fp32',validation_indices='0:32',selected_step=torch.load(p,weights_only=True)['metrics']['step']),indent=2))

if __name__=='__main__':main()
