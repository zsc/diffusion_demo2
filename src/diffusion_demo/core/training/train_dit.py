"""DiT training and sampling."""

import math
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..data_loader import denormalize, get_dataloader, tensor_to_pil
from ..models.dit import DiT, create_dit_for_latent
from ..models.vae import VAE, VQVAE
from ...utils.torch_env import get_device, get_dtype


def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """Linear beta schedule."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DDPM:
    """Denoising Diffusion Probabilistic Model."""
    
    def __init__(
        self,
        timesteps: int = 1000,
        schedule: str = "linear",
        device: str = "cuda"
    ):
        self.timesteps = timesteps
        self.device = device
        
        # Define beta schedule
        if schedule == "linear":
            self.betas = linear_beta_schedule(timesteps).to(device)
        else:  # cosine
            self.betas = cosine_beta_schedule(timesteps).to(device)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion process: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Training loss."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = model(x_noisy, t)
        
        loss = nn.functional.mse_loss(predicted_noise, noise)
        return loss
    
    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, t: int) -> torch.Tensor:
        """Single denoising step."""
        betas_t = self.betas[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t]
        
        # Predict noise
        t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        predicted_noise = model(x, t_batch)
        
        # Compute mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, shape: Tuple[int, ...], progress_callback=None) -> torch.Tensor:
        """Generate samples by iterative denoising."""
        device = next(model.parameters()).device
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        for i in tqdm(reversed(range(self.timesteps)), desc="Sampling", total=self.timesteps):
            img = self.p_sample(model, img, i)
            if progress_callback:
                progress_callback(self.timesteps - i, self.timesteps)
        
        return img


def load_vae(model_path: str, model_type: str = "vae", device: str = "cuda", **kwargs) -> nn.Module:
    """Load VAE/VQVAE from checkpoint."""
    if model_type == "vae":
        model = VAE(latent_dim=kwargs.get('latent_dim', 256))
    else:
        model = VQVAE(
            embedding_dim=kwargs.get('embedding_dim', 64),
            num_embeddings=kwargs.get('num_embeddings', 512),
        )
    
    state_dict = torch.load(model_path, map_location=device)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def encode_to_latent(
    vae: nn.Module,
    images: torch.Tensor,
    latent_is_flat: bool = False,
    flat_latent_dim: int = 256,
    latent_channels: int = 16,
    latent_spatial_dim: int = 4,
    latent_clamp_range: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """Encode images to latent space using VAE."""
    with torch.no_grad():
        if isinstance(vae, VAE):
            latent = vae.encode(images)
        else:  # VQVAE
            latent = vae.encode(images)
        
        if latent_is_flat:
            # Flatten latent
            latent = latent.view(latent.shape[0], -1)
            if latent.shape[1] != flat_latent_dim:
                # Project to correct dimension
                if not hasattr(vae, '_projection'):
                    vae._projection = nn.Linear(latent.shape[1], flat_latent_dim).to(latent.device)
                latent = vae._projection(latent)
            # Reshape back to spatial for DiT
            latent = latent.view(latent.shape[0], latent_channels, latent_spatial_dim, latent_spatial_dim)
        
        if latent_clamp_range:
            latent = torch.clamp(latent, latent_clamp_range[0], latent_clamp_range[1])
        
        return latent


def decode_from_latent(
    vae: nn.Module,
    latent: torch.Tensor,
    latent_is_flat: bool = False,
    flat_latent_dim: int = 256,
    vq_embedding_dim: int = 64,
) -> torch.Tensor:
    """Decode latent to images using VAE."""
    with torch.no_grad():
        if latent_is_flat:
            # Flatten if needed for VAE
            latent = latent.view(latent.shape[0], -1)
        
        images = vae.decode(latent)
        return images


def train_dit_model(
    vae_model_path: str,
    discriminator_model_path: Optional[str] = None,
    use_gan: bool = False,
    gan_loss_weight: float = 0.01,
    lr_d: float = 0.0001,
    ddpm_schedule: str = "linear",
    ddpm_timesteps: int = 1000,
    lr: float = 0.00001,
    batch_size: int = 64,
    epochs: int = 5,
    data_limit: int = 2048,
    use_bfloat16: bool = False,
    dit_patch_size: int = 2,
    dit_hidden_size: int = 384,
    dit_depth: int = 6,
    dit_num_heads: int = 6,
    vae_latent_is_flat: bool = False,
    vae_flat_latent_dim: int = 256,
    vae_latent_channels: int = 16,
    vae_latent_spatial_dim: int = 4,
    vae_latent_clamp_range: Optional[List[float]] = None,
    vae_model_type: str = "vae",
    progress_callback = None,
    log_callback = None,
) -> Dict:
    """Train DiT model.
    
    Returns:
        Dictionary with paths to checkpoints and artifacts.
    """
    device = get_device()
    dtype = get_dtype(use_bfloat16)
    
    # Load VAE
    vae_kwargs = {}
    if vae_model_type == "vae":
        vae_kwargs['latent_dim'] = vae_flat_latent_dim // (vae_latent_spatial_dim ** 2)
    else:
        vae_kwargs['embedding_dim'] = vae_flat_latent_dim // (vae_latent_spatial_dim ** 2)
        vae_kwargs['num_embeddings'] = 512
    
    vae = load_vae(vae_model_path, model_type=vae_model_type, device=device, **vae_kwargs)
    
    # Load discriminator if using GAN
    discriminator = None
    optimizer_D = None
    if use_gan and discriminator_model_path:
        from ..models.vae import Discriminator
        discriminator = Discriminator().to(device)
        disc_state = torch.load(discriminator_model_path, map_location=device)
        if 'model_state_dict' in disc_state:
            disc_state = disc_state['model_state_dict']
        discriminator.load_state_dict(disc_state)
        discriminator.eval()
        optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    
    # Create DiT model
    model = create_dit_for_latent(
        latent_channels=vae_latent_channels,
        latent_spatial_dim=vae_latent_spatial_dim,
        patch_size=dit_patch_size,
        hidden_size=dit_hidden_size,
        depth=dit_depth,
        num_heads=dit_num_heads,
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Data loader
    dataloader = get_dataloader(batch_size=batch_size, data_limit=data_limit, shuffle=True)
    
    # DDPM
    ddpm = DDPM(timesteps=ddpm_timesteps, schedule=ddpm_schedule, device=device)
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"dit_{timestamp}"
    checkpoint_dir = Path("checkpoints") / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = Path("results") / "dit_samples" / run_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("runs") / run_name
    
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # Training loop
    model.train()
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
            
            # Encode to latent
            latent = encode_to_latent(
                vae, batch,
                latent_is_flat=vae_latent_is_flat,
                flat_latent_dim=vae_flat_latent_dim,
                latent_channels=vae_latent_channels,
                latent_spatial_dim=vae_latent_spatial_dim,
                latent_clamp_range=tuple(vae_latent_clamp_range) if vae_latent_clamp_range else None,
            )
            
            if use_bfloat16:
                latent = latent.to(dtype)
            
            # Sample timestep
            t = torch.randint(0, ddpm.timesteps, (latent.shape[0],), device=device).long()
            
            optimizer.zero_grad()
            
            # Compute loss
            loss = ddpm.p_losses(model, latent, t)
            
            # GAN loss
            if use_gan and discriminator:
                # Predict noise and denoise one step
                predicted_noise = model(ddpm.q_sample(latent, t), t)
                alpha_t = ddpm.alphas_cumprod[t].view(-1, 1, 1, 1)
                sqrt_alpha_t = torch.sqrt(alpha_t)
                sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
                denoised = (ddpm.q_sample(latent, t) - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
                
                # Decode and discriminate
                fake_images = decode_from_latent(vae, denoised)
                fake_pred = discriminator(fake_images)
                g_loss = -torch.mean(torch.log(fake_pred + 1e-8))
                loss = loss + gan_loss_weight * g_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            writer.add_scalar('train/loss', loss.item(), global_step)
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            if log_callback and batch_idx % 10 == 0:
                log_callback(f"Epoch {epoch+1}, Batch {batch_idx}: loss={loss.item():.4f}")
            
            global_step += 1
        
        # Save samples
        model.eval()
        with torch.no_grad():
            samples = sample_dit(
                model, vae, ddpm,
                num_samples=16,
                latent_channels=vae_latent_channels,
                latent_spatial_dim=vae_latent_spatial_dim,
                vae_latent_is_flat=vae_latent_is_flat,
                vae_flat_latent_dim=vae_flat_latent_dim,
            )
            img = tensor_to_pil(samples)
            img_path = sample_dir / f"epoch_{epoch+1}.png"
            img.save(img_path)
            artifacts['images'].append(str(img_path))
            writer.add_image('samples', denormalize(samples[:8]), epoch)
        
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
    
    writer.close()
    
    return {
        'model_path': str(final_path),
        'checkpoint_dir': str(checkpoint_dir),
        'run_name': run_name,
        'artifacts': artifacts,
    }


def sample_dit(
    dit_model: nn.Module,
    vae_model: nn.Module,
    ddpm: DDPM,
    num_samples: int = 16,
    latent_channels: int = 16,
    latent_spatial_dim: int = 4,
    vae_latent_is_flat: bool = False,
    vae_flat_latent_dim: int = 256,
    progress_callback = None,
) -> torch.Tensor:
    """Generate samples using DiT and VAE."""
    device = next(dit_model.parameters()).device
    
    # Generate latent samples
    latent_shape = (num_samples, latent_channels, latent_spatial_dim, latent_spatial_dim)
    latent_samples = ddpm.p_sample_loop(dit_model, latent_shape, progress_callback)
    
    # Decode to images
    images = decode_from_latent(
        vae_model, latent_samples,
        latent_is_flat=vae_latent_is_flat,
        flat_latent_dim=vae_flat_latent_dim,
    )
    
    return images


def sample_images(
    vae_model_path: str,
    dit_model_path: str,
    num_samples: int = 16,
    ddpm_timesteps: int = 1000,
    use_bfloat16: bool = False,
    dit_patch_size: int = 2,
    dit_hidden_size: int = 384,
    dit_depth: int = 6,
    dit_num_heads: int = 6,
    vae_latent_channels: int = 16,
    vae_latent_spatial_dim: int = 4,
    vae_flat_latent_dim: int = 256,
    vq_embedding_dim_for_dit: int = 64,
    vae_model_type: str = "vae",
    progress_callback = None,
) -> torch.Tensor:
    """Generate images from trained VAE and DiT.
    
    Returns:
        Generated images tensor (B, C, H, W)
    """
    device = get_device()
    dtype = get_dtype(use_bfloat16)
    
    # Load VAE
    vae_kwargs = {}
    if vae_model_type == "vae":
        vae_kwargs['latent_dim'] = vae_flat_latent_dim // (vae_latent_spatial_dim ** 2)
    else:
        vae_kwargs['embedding_dim'] = vq_embedding_dim_for_dit
        vae_kwargs['num_embeddings'] = 512
    
    vae = load_vae(vae_model_path, model_type=vae_model_type, device=device, **vae_kwargs)
    
    # Load DiT
    dit = create_dit_for_latent(
        latent_channels=vae_latent_channels,
        latent_spatial_dim=vae_latent_spatial_dim,
        patch_size=dit_patch_size,
        hidden_size=dit_hidden_size,
        depth=dit_depth,
        num_heads=dit_num_heads,
    ).to(device)
    
    dit_state = torch.load(dit_model_path, map_location=device)
    if 'model_state_dict' in dit_state:
        dit_state = dit_state['model_state_dict']
    dit.load_state_dict(dit_state)
    dit.eval()
    
    # Create DDPM
    ddpm = DDPM(timesteps=ddpm_timesteps, schedule="linear", device=device)
    
    # Generate samples
    with torch.no_grad():
        samples = sample_dit(
            dit, vae, ddpm,
            num_samples=num_samples,
            latent_channels=vae_latent_channels,
            latent_spatial_dim=vae_latent_spatial_dim,
            vae_latent_is_flat=False,
            vae_flat_latent_dim=vae_flat_latent_dim,
            progress_callback=progress_callback,
        )
    
    return samples
