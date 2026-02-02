"""VAE/VQVAE training."""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..data_loader import denormalize, get_dataloader, tensor_to_pil
from ..models.vae import VAE, VQVAE, Discriminator
from ...utils.torch_env import get_device, get_dtype


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kld_beta: float = 1.0
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """VAE loss: reconstruction + KL divergence."""
    recon_loss = nn.functional.mse_loss(recon, x, reduction='sum') / x.shape[0]
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
    loss = recon_loss + kld_beta * kld
    
    return loss, {
        'loss': loss.item(),
        'recon_loss': recon_loss.item(),
        'kld': kld.item(),
    }


def vqvae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    vq_loss: torch.Tensor,
    vq_beta: float = 1.0
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """VQVAE loss: reconstruction + VQ loss."""
    recon_loss = nn.functional.mse_loss(recon, x, reduction='sum') / x.shape[0]
    loss = recon_loss + vq_beta * vq_loss
    
    return loss, {
        'loss': loss.item(),
        'recon_loss': recon_loss.item(),
        'vq_loss': vq_loss.item(),
    }


def train_vae_model(
    model_type: str = "vae",
    lr: float = 0.0002,
    batch_size: int = 128,
    epochs: int = 5,
    data_limit: int = 2048,
    use_bfloat16: bool = False,
    latent_dim: int = 256,
    embedding_dim: int = 64,
    num_embeddings: int = 512,
    commitment_cost: float = 0.25,
    kld_beta: float = 1.0,
    vq_beta: float = 1.0,
    vq_hidden_dims_enc: List[int] = None,
    vq_hidden_dims_dec: List[int] = None,
    use_gan: bool = False,
    progress_callback = None,
    log_callback = None,
) -> Dict:
    """Train VAE or VQVAE model.
    
    Returns:
        Dictionary with paths to checkpoints and artifacts.
    """
    device = get_device()
    dtype = get_dtype(use_bfloat16)
    
    # Default VQVAE hidden dims
    if vq_hidden_dims_enc is None:
        vq_hidden_dims_enc = [128, 64]
    if vq_hidden_dims_dec is None:
        vq_hidden_dims_dec = [64, 128]
    
    # Create model
    if model_type == "vae":
        model = VAE(latent_dim=latent_dim).to(device)
    else:  # vqvae
        model = VQVAE(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost,
            hidden_dims_enc=vq_hidden_dims_enc,
            hidden_dims_dec=vq_hidden_dims_dec,
        ).to(device)
    
    discriminator = None
    optimizer_D = None
    
    if use_gan:
        discriminator = Discriminator().to(device)
        optimizer_D = optim.Adam(discriminator.parameters(), lr=lr * 0.5, betas=(0.5, 0.999))
    
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    
    # Data loader
    dataloader = get_dataloader(batch_size=batch_size, data_limit=data_limit, shuffle=True)
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_type}_{timestamp}"
    checkpoint_dir = Path("checkpoints") / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = Path("results") / "vae_samples" / run_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("runs") / run_name
    
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # Training loop
    model.train()
    if discriminator:
        discriminator.train()
    
    global_step = 0
    artifacts = {
        'checkpoints': [],
        'images': [],
        'tensorboard_logdir': str(log_dir),
    }
    
    for epoch in range(epochs):
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(device)
            
            if use_bfloat16:
                batch = batch.to(dtype)
            
            optimizer.zero_grad()
            
            # Forward pass
            if model_type == "vae":
                recon, mu, logvar = model(batch)
                loss, metrics = vae_loss(recon, batch, mu, logvar, kld_beta)
            else:  # vqvae
                recon, vq_loss_val = model(batch)
                loss, metrics = vqvae_loss(recon, batch, vq_loss_val, vq_beta)
            
            # GAN loss
            if use_gan and discriminator:
                # Train discriminator
                optimizer_D.zero_grad()
                real_pred = discriminator(batch)
                fake_pred = discriminator(recon.detach())
                d_loss = -torch.mean(torch.log(real_pred + 1e-8) + torch.log(1 - fake_pred + 1e-8))
                d_loss.backward()
                optimizer_D.step()
                
                # Generator adversarial loss
                fake_pred = discriminator(recon)
                g_adv_loss = -torch.mean(torch.log(fake_pred + 1e-8))
                loss = loss + 0.01 * g_adv_loss
                metrics['d_loss'] = d_loss.item()
                metrics['g_adv_loss'] = g_adv_loss.item()
            
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(metrics['loss'])
            
            # Log to tensorboard
            for k, v in metrics.items():
                writer.add_scalar(f'train/{k}', v, global_step)
            
            pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})
            
            if log_callback and batch_idx % 10 == 0:
                log_callback(f"Epoch {epoch+1}, Batch {batch_idx}: loss={metrics['loss']:.4f}")
            
            global_step += 1
        
        # Save samples
        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(dataloader))[:16].to(device)
            if model_type == "vae":
                recon, _, _ = model(sample_batch)
            else:
                recon, _ = model(sample_batch)
            
            # Save comparison
            comparison = torch.cat([sample_batch[:8], recon[:8]], dim=0)
            img = tensor_to_pil(comparison)
            img_path = sample_dir / f"epoch_{epoch+1}.png"
            img.save(img_path)
            artifacts['images'].append(str(img_path))
            writer.add_image('reconstruction', denormalize(comparison[:8]), epoch)
        
        model.train()
        
        # Save checkpoint
        ckpt_path = checkpoint_dir / f"epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': sum(epoch_losses) / len(epoch_losses),
        }, ckpt_path)
        artifacts['checkpoints'].append(str(ckpt_path))
        
        if progress_callback:
            progress_callback(epoch + 1, epochs)
    
    # Save final model
    final_path = checkpoint_dir / "final_model.pth"
    torch.save(model.state_dict(), final_path)
    
    # Save discriminator if used
    disc_path = None
    if use_gan and discriminator:
        disc_path = checkpoint_dir / "discriminator.pth"
        torch.save(discriminator.state_dict(), disc_path)
    
    writer.close()
    
    return {
        'model_path': str(final_path),
        'discriminator_path': str(disc_path) if disc_path else None,
        'checkpoint_dir': str(checkpoint_dir),
        'run_name': run_name,
        'artifacts': artifacts,
    }
