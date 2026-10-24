"""VAE/VQVAE training with reproducible holdout evaluation."""

import json
import hashlib
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..data_loader import CACHE_FILE, CelebASpritesDataset, denormalize, tensor_to_pil
from ..models.vae import VAE, VQVAE, Discriminator
from ...utils.torch_env import MixedPrecision, get_device, setup_seed


def vae_loss(recon, x, mu, logvar, kld_beta=1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    # Reductions and exp in FP32 even under autocast. Keep the original per-image
    # summed ELBO convention so existing beta settings and reports remain comparable.
    recon_loss = nn.functional.mse_loss(recon.float(), x.float(), reduction='sum') / x.shape[0]
    mu, logvar = mu.float(), logvar.float()
    kld = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum() / x.shape[0]
    loss = recon_loss + kld_beta * kld
    return loss, {'loss': loss.item(), 'recon_loss': recon_loss.item(), 'kld': kld.item()}


def vqvae_loss(recon, x, vq_loss, vq_beta=1.0):
    recon_loss = nn.functional.mse_loss(recon.float(), x.float(), reduction='sum') / x.shape[0]
    loss = recon_loss + vq_beta * vq_loss.float()
    return loss, {'loss': loss.item(), 'recon_loss': recon_loss.item(), 'vq_loss': vq_loss.item()}


def make_loaders(batch_size, data_limit, seed, validation_fraction, num_workers, pin_memory):
    if not 0 <= validation_fraction < 1:
        raise ValueError('validation_fraction must be in [0, 1)')
    dataset = CelebASpritesDataset(data_limit=data_limit)
    n_val = max(1, int(len(dataset) * validation_fraction)) if validation_fraction else 0
    if n_val >= len(dataset):
        raise ValueError('Need at least two images for a train/validation split')
    train, val = random_split(dataset, [len(dataset) - n_val, n_val],
                              generator=torch.Generator().manual_seed(seed))
    kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=pin_memory, persistent_workers=num_workers > 0)
    train_loader = DataLoader(train, shuffle=True,
                             generator=torch.Generator().manual_seed(seed + 1), **kwargs)
    val_loader = DataLoader(val, shuffle=False, **kwargs) if n_val else None
    return train_loader, val_loader


def split_manifest(train_loader, val_loader):
    images = train_loader.dataset.dataset.images
    return dict(cache_file=str(CACHE_FILE), selected_shape=list(images.shape),
                selected_sha256=hashlib.sha256(images.tobytes()).hexdigest(),
                train_indices=list(train_loader.dataset.indices),
                validation_indices=list(val_loader.dataset.indices) if val_loader else [])


@torch.no_grad()
def evaluate_reconstruction(model, loader, device, amp):
    """MSE in [0,1], using the VAE posterior mean, on unseen images."""
    model.eval()
    squared_error, pixels = 0.0, 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with amp.autocast():
            z = model.encoder(batch)[0] if isinstance(model, VAE) else model.encode(batch)
            recon = model.decode(z)
        squared_error += (recon.float() - batch).square().sum().item() / 4
        pixels += batch.numel()
    mse = squared_error / pixels
    return {'val_recon_mse': mse, 'val_psnr_db': -10 * math.log10(max(mse, 1e-12))}


def train_vae_model(
    model_type: str = 'vae', lr: float = 0.0002, batch_size: int = 128,
    epochs: int = 5, data_limit: int = 2048, use_bfloat16: bool = False,
    latent_dim: int = 256, embedding_dim: int = 64, num_embeddings: int = 512,
    commitment_cost: float = 0.25, kld_beta: float = 1.0, vq_beta: float = 1.0,
    vq_hidden_dims_enc: List[int] = None, vq_hidden_dims_dec: List[int] = None,
    use_gan: bool = False, progress_callback=None, log_callback=None,
    seed: int = 0, validation_fraction: float = 0.1, kl_warmup_epochs: int = 0,
    num_workers: int = 0, precision: str = None, device: str = None,
    output_dir: str = None,
) -> Dict:
    if model_type not in {'vae', 'vqvae'} or min(epochs, batch_size, data_limit) <= 0:
        raise ValueError('Invalid model type or nonpositive training size')
    if kl_warmup_epochs < 0:
        raise ValueError('kl_warmup_epochs must be nonnegative')
    config = {k: v for k, v in locals().copy().items()
              if k not in {'progress_callback', 'log_callback'}}
    setup_seed(seed)
    device = get_device(device)
    amp = MixedPrecision(device, precision, use_bfloat16)
    model_config = ({'latent_dim': latent_dim} if model_type == 'vae' else {
        'embedding_dim': embedding_dim, 'num_embeddings': num_embeddings,
        'commitment_cost': commitment_cost, 'hidden_dims_enc': vq_hidden_dims_enc or [128, 64],
        'hidden_dims_dec': vq_hidden_dims_dec or [64, 128]})
    model = (VAE(**model_config) if model_type == 'vae' else VQVAE(**model_config)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    discriminator = Discriminator().to(device) if use_gan else None
    optimizer_d = (torch.optim.Adam(discriminator.parameters(), lr=lr * 0.5, betas=(0.5, 0.999))
                   if use_gan else None)
    amp_d = MixedPrecision(device, precision, use_bfloat16) if use_gan else None
    train_loader, val_loader = make_loaders(batch_size, data_limit, seed, validation_fraction,
                                          num_workers, device.type == 'cuda')
    preview_loader = val_loader or DataLoader(train_loader.dataset, batch_size=min(8, batch_size))
    fixed_images = next(iter(preview_loader))[:8].to(device)
    run_name = f'{model_type}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    root = Path(output_dir) if output_dir else Path('runs') / run_name
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = root / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    (root / 'config.json').write_text(json.dumps(config, indent=2))
    (root / 'split.json').write_text(json.dumps(split_manifest(train_loader, val_loader), indent=2))
    artifacts = {'checkpoints': [], 'images': [], 'tensorboard_logdir': str(root / 'tensorboard')}
    history, global_step, best_mse = [], 0, float('inf')
    start = time.perf_counter()
    writer = SummaryWriter(str(root / 'tensorboard'))
    try:
        for epoch in range(epochs):
            model.train()
            epoch_start = time.perf_counter()
            totals, seen = {}, 0
            beta = kld_beta * min(1.0, (epoch + 1) / max(1, kl_warmup_epochs))
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'VAE {epoch+1}/{epochs}')):
                batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with amp.autocast():
                    if model_type == 'vae':
                        recon, mu, logvar = model(batch)
                        loss, metrics = vae_loss(recon, batch, mu, logvar, beta)
                    else:
                        recon, vq_loss = model(batch)
                        loss, metrics = vqvae_loss(recon, batch, vq_loss, vq_beta)
                if discriminator is not None:
                    discriminator.requires_grad_(True)
                    optimizer_d.zero_grad(set_to_none=True)
                    with amp_d.autocast():
                        real = discriminator(batch).float().clamp(1e-6, 1 - 1e-6)
                        fake = discriminator(recon.detach()).float().clamp(1e-6, 1 - 1e-6)
                        d_loss = -(real.log() + (1 - fake).log()).mean()
                    amp_d.step(d_loss, optimizer_d, discriminator.parameters())
                    discriminator.requires_grad_(False)
                    with amp.autocast():
                        adv = -discriminator(recon).float().clamp_min(1e-6).log().mean()
                        loss = loss + 0.01 * adv
                    metrics.update(d_loss=d_loss.item(), g_adv_loss=adv.item())
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite VAE loss at step {global_step}')
                amp.step(loss, optimizer, model.parameters())
                metrics['loss'] = loss.item()
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + value * len(batch)
                    if global_step % 10 == 0:
                        writer.add_scalar(f'train/{key}', value, global_step)
                seen += len(batch)
                if log_callback and batch_idx % 10 == 0:
                    log_callback(f'Epoch {epoch+1}, batch {batch_idx}: loss={metrics["loss"]:.4f}')
                global_step += 1
            train_seconds = time.perf_counter() - epoch_start
            record = dict(epoch=epoch + 1, **{k: v / seen for k, v in totals.items()},
                          kld_beta=beta, train_seconds=train_seconds, images_per_second=seen / train_seconds)
            if val_loader:
                record.update(evaluate_reconstruction(model, val_loader, device, amp))
                for key in ('val_recon_mse', 'val_psnr_db'):
                    writer.add_scalar(key, record[key], epoch + 1)
            model.eval()
            with torch.no_grad(), amp.autocast():
                z = model.encoder(fixed_images)[0] if model_type == 'vae' else model.encode(fixed_images)
                recon = model.decode(z)
            comparison = torch.cat([fixed_images, recon.float()])
            image_path = root / f'reconstruction_{epoch+1:03d}.png'
            tensor_to_pil(comparison).save(image_path)
            writer.add_images('reconstruction', denormalize(comparison), epoch + 1)
            artifacts['images'].append(str(image_path))
            history.append(record)
            payload = dict(format_version=2, model_type=model_type, model_config=model_config,
                           config=config, epoch=epoch + 1, global_step=global_step,
                           model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                           scaler_state_dict=amp.scaler.state_dict(), metrics=record)
            final_path = checkpoint_dir / 'last.pth'
            torch.save(payload, final_path)
            if val_loader and record['val_recon_mse'] < best_mse:
                best_mse = record['val_recon_mse']
                torch.save(payload, checkpoint_dir / 'best.pth')
            (root / 'metrics.json').write_text(json.dumps({
                'config': config, 'precision': amp.precision, 'train_images': len(train_loader.dataset),
                'validation_images': len(val_loader.dataset) if val_loader else 0,
                'epochs': history, 'total_seconds': time.perf_counter() - start,
            }, indent=2))
            print(json.dumps(record), flush=True)
            if progress_callback:
                progress_callback(epoch + 1, epochs)
        artifacts['checkpoints'] = [str(p) for p in checkpoint_dir.glob('*.pth')]
        disc_path = None
        if discriminator is not None:
            disc_path = checkpoint_dir / 'discriminator.pth'
            torch.save(discriminator.state_dict(), disc_path)
        return dict(model_path=str(final_path), best_model_path=str(checkpoint_dir / 'best.pth') if val_loader else None,
                    discriminator_path=str(disc_path) if disc_path else None,
                    checkpoint_dir=str(checkpoint_dir), run_name=run_name, artifacts=artifacts,
                    metrics_path=str(root / 'metrics.json'))
    finally:
        writer.close()
