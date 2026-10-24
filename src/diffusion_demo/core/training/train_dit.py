"""DiT training and sampling."""

import math
import copy
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..data_loader import denormalize, get_dataloader, tensor_to_pil
from ..models.dit import DiT, create_dit_for_latent
from ..models.vae import VAE, VQVAE
from ...utils.torch_env import get_device, MixedPrecision, setup_seed
from .train_vae import make_loaders, split_manifest


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
        if timesteps < 1 or schedule not in {"linear", "cosine"}:
            raise ValueError("Positive timesteps and linear/cosine schedule required")
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
        
        loss = nn.functional.mse_loss(predicted_noise.float(), noise.float())
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

    @torch.no_grad()
    def ddim_sample_loop(self, model, shape, sampling_steps=50, progress_callback=None):
        """Deterministic DDIM (eta=0), retaining the training noise schedule."""
        if not 1 <= sampling_steps <= self.timesteps:
            raise ValueError('sampling_steps must be between 1 and training timesteps')
        device = next(model.parameters()).device
        x = torch.randn(shape, device=device)
        times = torch.linspace(self.timesteps - 1, 0, sampling_steps).long().tolist()
        for index, t in enumerate(times):
            prev = times[index + 1] if index + 1 < len(times) else -1
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            eps = model(x, t_batch).float()
            alpha = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[prev] if prev >= 0 else x.new_tensor(1.)
            x0 = (x - (1 - alpha).sqrt() * eps) / alpha.sqrt()
            x = alpha_prev.sqrt() * x0 + (1 - alpha_prev).sqrt() * eps
            if progress_callback:
                progress_callback(index + 1, len(times))
        return x


def load_vae(model_path: str, model_type='vae', device='cuda', **kwargs):
    """Load self-describing checkpoints, or infer architecture from legacy weights."""
    payload = torch.load(model_path, map_location='cpu', weights_only=True)
    state = payload.get('model_state_dict', payload)
    model_type = payload.get('model_type', 'vqvae' if 'vq.embeddings.weight' in state else 'vae')
    config = payload.get('model_config')
    if config is None:
        def conv_dims(prefix, transpose=False):
            keys = [k for k, v in state.items() if k.startswith(prefix) and k.endswith('.weight')
                    and v.ndim == 4 and v.shape[-1] == 4]
            keys.sort(key=lambda k: int(k.split('.')[2]))
            return [state[k].shape[1 if transpose else 0] for k in keys]
        if model_type == 'vae':
            config = dict(latent_dim=state['encoder.to_mu.weight'].shape[0],
                          hidden_dims=conv_dims('encoder.encoder.'))
        else:
            config = dict(embedding_dim=state['vq.embeddings.weight'].shape[1],
                          num_embeddings=state['vq.embeddings.weight'].shape[0],
                          hidden_dims_enc=conv_dims('encoder.encoder.'),
                          hidden_dims_dec=conv_dims('decoder.decoder.', True))
    model = VAE(**config) if model_type == 'vae' else VQVAE(**config)
    model.load_state_dict(state)
    return model.to(device).eval().requires_grad_(False)


def encode_to_latent(vae, images, latent_is_flat=False, flat_latent_dim=256,
                     latent_channels=16, latent_spatial_dim=4, latent_clamp_range=None):
    # Both supported autoencoders use spatial latents. Never insert an untrained,
    # irreversible random projection just to satisfy manually entered dimensions.
    with torch.no_grad():
        latent = vae.encode(images)
    if latent.ndim != 4:
        raise ValueError('Expected spatial VAE latents (B,C,H,W)')
    if latent_clamp_range is not None:
        latent = latent.clamp(*latent_clamp_range)
    return latent


def decode_from_latent(vae, latent, latent_is_flat=False, flat_latent_dim=256, vq_embedding_dim=64):
    return vae.decode(latent)


@torch.no_grad()
def cache_posterior(vae, loader, device, amp):
    """Cache posterior parameters, not a single stochastic VAE draw."""
    centers, logvars = [], []
    for images in tqdm(loader, desc='Cache VAE posterior'):
        with amp.autocast():
            images = images.to(device, non_blocking=True)
            if isinstance(vae, VAE):
                mean, logvar = vae.encoder(images)
            else:
                mean = vae.encode(images)
                logvar = torch.zeros_like(mean)
        centers.append(mean.float().cpu())
        logvars.append(logvar.float().cpu())
    return TensorDataset(torch.cat(centers), torch.cat(logvars))


def latent_stats(dataset, is_vae):
    mean, logvar = dataset.tensors
    center = mean.mean(dim=(0, 2, 3), keepdim=True)
    variance = (mean - center).square().mean(dim=(0, 2, 3), keepdim=True)
    if is_vae:
        variance += logvar.exp().mean(dim=(0, 2, 3), keepdim=True)
    return center, variance.sqrt().clamp_min(1e-6)


def latent_batch(batch, vae, device, amp, cached, center, scale, generator=None):
    with torch.no_grad(), amp.autocast():
        if cached:
            mean, logvar = (x.to(device, non_blocking=True) for x in batch)
        elif isinstance(vae, VAE):
            mean, logvar = vae.encoder(batch.to(device, non_blocking=True))
        else:
            mean = vae.encode(batch.to(device, non_blocking=True))
            logvar = None
        latent = mean.float()
        if isinstance(vae, VAE):
            noise = torch.randn(mean.shape, device=device, generator=generator)
            latent = latent + (0.5 * logvar.float()).exp() * noise
    return (latent - center) / scale


def train_dit_model(
    vae_model_path: str, discriminator_model_path=None, use_gan=False,
    gan_loss_weight=0.01, lr_d=0.0001, ddpm_schedule='linear', ddpm_timesteps=1000,
    lr=0.0001, batch_size=64, epochs=5, data_limit=2048, use_bfloat16=False,
    dit_patch_size=2, dit_hidden_size=384, dit_depth=6, dit_num_heads=6,
    vae_latent_is_flat=False, vae_flat_latent_dim=256, vae_latent_channels=16,
    vae_latent_spatial_dim=4, vae_latent_clamp_range=None, vae_model_type='vae',
    progress_callback=None, log_callback=None, seed=0, validation_fraction=0.1,
    num_workers=0, precision=None, device=None, output_dir=None,
    cache_latents=True, normalize_latents=True, ema_decay=0.999,
    sampling_steps=50, sample_every=1, num_preview_samples=16,
):
    if use_gan:
        raise ValueError('DiT adversarial training is unsupported; use the noise-prediction objective')
    if vae_latent_clamp_range is not None:
        raise ValueError('Latent clipping changes the VAE distribution; set vae_latent_clamp_range=null')
    if min(epochs, batch_size, data_limit, sample_every, num_preview_samples) <= 0 or not 0 <= ema_decay < 1:
        raise ValueError('Invalid training size, sample interval or EMA decay')
    if sampling_steps is not None and not 1 <= sampling_steps <= ddpm_timesteps:
        raise ValueError('sampling_steps must not exceed ddpm_timesteps')
    config = {k: v for k, v in locals().copy().items()
              if k not in {'progress_callback', 'log_callback'}}
    setup_seed(seed)
    device = get_device(device)
    amp = MixedPrecision(device, precision, use_bfloat16)
    vae = load_vae(vae_model_path, device=device)
    channels = vae.latent_dim if isinstance(vae, VAE) else vae.embedding_dim
    model_config = dict(latent_channels=channels, latent_spatial_dim=4, patch_size=dit_patch_size,
                        hidden_size=dit_hidden_size, depth=dit_depth, num_heads=dit_num_heads)
    model = create_dit_for_latent(**model_config).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    ddpm = DDPM(ddpm_timesteps, ddpm_schedule, device)
    train_loader, val_loader = make_loaders(batch_size, data_limit, seed, validation_fraction,
                                          num_workers, device.type == 'cuda')
    split = split_manifest(train_loader, val_loader)
    center, scale = torch.zeros(1, channels, 1, 1), torch.ones(1, channels, 1, 1)
    cache_start = time.perf_counter()
    if cache_latents or normalize_latents:
        posterior = cache_posterior(vae, DataLoader(train_loader.dataset, batch_size=batch_size), device, amp)
        if normalize_latents:
            center, scale = latent_stats(posterior, isinstance(vae, VAE))
        if cache_latents:
            train_loader = DataLoader(posterior, batch_size=batch_size, shuffle=True,
                                      pin_memory=device.type == 'cuda',
                                      generator=torch.Generator().manual_seed(seed + 1))
            if val_loader:
                val_loader = DataLoader(cache_posterior(vae, val_loader, device, amp),
                                        batch_size=batch_size, pin_memory=device.type == 'cuda')
    cache_seconds = time.perf_counter() - cache_start
    center, scale = center.to(device), scale.to(device)
    run_name = f'dit_{datetime.now():%Y%m%d_%H%M%S_%f}'
    root = Path(output_dir) if output_dir else Path('runs') / run_name
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = root / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    (root / 'config.json').write_text(json.dumps(config, indent=2))
    (root / 'split.json').write_text(json.dumps(split, indent=2))
    artifacts = dict(checkpoints=[], images=[], tensorboard_logdir=str(root / 'tensorboard'))
    history, global_step, updates = [], 0, 0
    writer = SummaryWriter(str(root / 'tensorboard'))
    start = time.perf_counter()
    try:
        for epoch in range(epochs):
            model.train()
            epoch_start, total_loss, seen = time.perf_counter(), 0.0, 0
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'DiT {epoch+1}/{epochs}')):
                latent = latent_batch(batch, vae, device, amp, cache_latents, center, scale)
                t = torch.randint(0, ddpm_timesteps, (len(latent),), device=device)
                optimizer.zero_grad(set_to_none=True)
                with amp.autocast():
                    loss = ddpm.p_losses(model, latent, t)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite DiT loss at step {global_step}')
                if amp.step(loss, optimizer, model.parameters()):
                    updates += 1
                    decay = min(ema_decay, (1 + updates) / (10 + updates))
                    with torch.no_grad():
                        for averaged, live in zip(ema.parameters(), model.parameters()):
                            averaged.lerp_(live, 1 - decay)
                loss_value = loss.item()
                total_loss += loss_value * len(latent)
                seen += len(latent)
                if global_step % 10 == 0:
                    writer.add_scalar('train/loss', loss_value, global_step)
                    if log_callback:
                        log_callback(f'Epoch {epoch+1}, batch {batch_idx}: loss={loss_value:.4f}')
                global_step += 1
            train_seconds = time.perf_counter() - epoch_start
            record = dict(epoch=epoch + 1, loss=total_loss / seen, train_seconds=train_seconds,
                          images_per_second=seen / train_seconds, optimizer_updates=updates)
            if val_loader:
                # Identical posterior noise, timesteps and diffusion noise each epoch.
                generator = torch.Generator(device=device).manual_seed(seed + 10_000)
                val_loss, val_seen = 0.0, 0
                with torch.no_grad(), amp.autocast():
                    for batch in val_loader:
                        latent = latent_batch(batch, vae, device, amp, cache_latents, center, scale, generator)
                        t = torch.randint(0, ddpm_timesteps, (len(latent),), device=device, generator=generator)
                        noise = torch.randn(latent.shape, device=device, generator=generator)
                        val_loss += ddpm.p_losses(ema, latent, t, noise).item() * len(latent)
                        val_seen += len(latent)
                record['val_ema_noise_mse'] = val_loss / val_seen
                writer.add_scalar('val/ema_noise_mse', record['val_ema_noise_mse'], epoch + 1)
            if (epoch + 1) % sample_every == 0 or epoch + 1 == epochs:
                sample_start = time.perf_counter()
                # Previewing does not perturb training RNG.
                devices = [device.index or 0] if device.type == 'cuda' else []
                with torch.random.fork_rng(devices=devices), torch.no_grad(), amp.autocast():
                    torch.manual_seed(seed + 20_000)
                    samples = sample_dit(ema, vae, ddpm, num_preview_samples, channels, 4,
                                         sampling_steps=sampling_steps, latent_mean=center, latent_std=scale)
                image_path = root / f'samples_{epoch+1:03d}.png'
                tensor_to_pil(samples.float()).save(image_path)
                artifacts['images'].append(str(image_path))
                writer.add_images('samples', denormalize(samples.float()), epoch + 1)
                record['sample_seconds'] = time.perf_counter() - sample_start
            history.append(record)
            final_path = checkpoint_dir / 'last.pth'
            torch.save(dict(format_version=2, config=config, model_config=model_config,
                            model_state_dict=model.state_dict(), ema_state_dict=ema.state_dict(),
                            optimizer_state_dict=optimizer.state_dict(), scaler_state_dict=amp.scaler.state_dict(),
                            epoch=epoch + 1, global_step=global_step, metrics=record,
                            diffusion_config=dict(timesteps=ddpm_timesteps, schedule=ddpm_schedule),
                            latent_mean=center.cpu(), latent_std=scale.cpu()), final_path)
            (root / 'metrics.json').write_text(json.dumps(dict(
                config=config, precision=amp.precision, model_config=model_config, epochs=history,
                cache_seconds=cache_seconds, total_seconds=cache_seconds + time.perf_counter() - start,
            ), indent=2))
            print(json.dumps(record), flush=True)
            if progress_callback:
                progress_callback(epoch + 1, epochs)
        artifacts['checkpoints'] = [str(final_path)]
        return dict(model_path=str(final_path), checkpoint_dir=str(checkpoint_dir), run_name=run_name,
                    artifacts=artifacts, metrics_path=str(root / 'metrics.json'))
    finally:
        writer.close()


@torch.no_grad()
def sample_dit(dit_model, vae_model, ddpm, num_samples=16, latent_channels=16,
               latent_spatial_dim=4, vae_latent_is_flat=False, vae_flat_latent_dim=256,
               progress_callback=None, sampling_steps=None, latent_mean=None, latent_std=None):
    shape = (num_samples, latent_channels, latent_spatial_dim, latent_spatial_dim)
    if sampling_steps is None:
        latent = ddpm.p_sample_loop(dit_model, shape, progress_callback)
    else:
        latent = ddpm.ddim_sample_loop(dit_model, shape, sampling_steps, progress_callback)
    if latent_std is not None:
        latent = latent * latent_std + latent_mean
    return decode_from_latent(vae_model, latent)


@torch.no_grad()
def sample_images(vae_model_path, dit_model_path, num_samples=16, ddpm_timesteps=1000,
                  use_bfloat16=False, dit_patch_size=2, dit_hidden_size=384, dit_depth=6,
                  dit_num_heads=6, vae_latent_channels=16, vae_latent_spatial_dim=4,
                  vae_flat_latent_dim=256, vq_embedding_dim_for_dit=64, vae_model_type='vae',
                  progress_callback=None, ddpm_schedule='linear', sampling_steps=None,
                  seed=0, precision=None, device=None):
    device = get_device(device)
    amp = MixedPrecision(device, precision, use_bfloat16)
    vae = load_vae(vae_model_path, device=device)
    payload = torch.load(dit_model_path, map_location='cpu', weights_only=True)
    config = payload.get('model_config', dict(latent_channels=vae_latent_channels,
        latent_spatial_dim=vae_latent_spatial_dim, patch_size=dit_patch_size,
        hidden_size=dit_hidden_size, depth=dit_depth, num_heads=dit_num_heads))
    channels = vae.latent_dim if isinstance(vae, VAE) else vae.embedding_dim
    if config['latent_channels'] != channels:
        raise ValueError('DiT latent channels do not match this VAE checkpoint')
    dit = create_dit_for_latent(**config).to(device).eval()
    dit.load_state_dict(payload.get('ema_state_dict', payload.get('model_state_dict', payload)))
    diffusion_config = payload.get('diffusion_config', dict(timesteps=ddpm_timesteps, schedule=ddpm_schedule))
    ddpm = DDPM(device=device, **diffusion_config)
    center, scale = payload.get('latent_mean'), payload.get('latent_std')
    setup_seed(seed)  # Initialization during checkpoint loading must not consume the sampling seed.
    with amp.autocast():
        return sample_dit(dit, vae, ddpm, num_samples, config['latent_channels'], config['latent_spatial_dim'],
                          progress_callback=progress_callback, sampling_steps=sampling_steps,
                          latent_mean=center.to(device) if center is not None else None,
                          latent_std=scale.to(device) if scale is not None else None).float()
