"""VAE/VQVAE training with reproducible holdout evaluation."""

import copy
import json
import hashlib
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..data_loader import CelebASpritesDataset, denormalize, tensor_to_pil
from ..models.vae import VAE, VQVAE, Discriminator
from ...utils.torch_env import MixedPrecision, get_device, setup_seed


def vae_loss(recon, x, mu, logvar, kld_beta=1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    # Reductions and exp in FP32 even under autocast. Keep the original per-image
    # summed ELBO convention so existing beta settings and reports remain comparable.
    if recon.shape != x.shape:
        raise ValueError('Reconstruction shape differs from input; do not resize targets to hide a model mismatch')
    recon_loss = nn.functional.mse_loss(recon.float(), x.float(), reduction='sum') / x.shape[0]
    mu, logvar = mu.float(), logvar.float()
    kld = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum() / x.shape[0]
    loss = recon_loss + kld_beta * kld
    return loss, {'loss': loss.item(), 'recon_loss': recon_loss.item(), 'kld': kld.item()}


def vqvae_loss(recon, x, vq_loss, vq_beta=1.0, recon_reduction='mean'):
    if recon_reduction not in {'mean', 'sum'}:
        raise ValueError('VQ reconstruction reduction must be mean or sum')
    if recon.shape != x.shape:
        raise ValueError('Reconstruction shape differs from input')
    # VQ/codebook and commitment losses are means. Keep their relative scale
    # independent of image resolution; sum is an explicit legacy compatibility mode.
    recon_loss = nn.functional.mse_loss(recon.float(), x.float(), reduction=recon_reduction)
    if recon_reduction == 'sum':
        recon_loss = recon_loss / x.shape[0]
    loss = recon_loss + vq_beta * vq_loss.float()
    return loss, {'loss': loss.item(), 'recon_loss': recon_loss.item(), 'vq_loss': vq_loss.item()}


def make_loaders(batch_size, data_limit, seed, validation_fraction, num_workers, pin_memory,
                 data_path=None, image_size=48, split_mode='random', val_count=None, rank=0, world_size=1):
    if not 0 <= validation_fraction < 1:
        raise ValueError('validation_fraction must be in [0, 1)')
    if split_mode not in {'random', 'validation_first'}:
        raise ValueError('split_mode must be random or validation_first')
    dataset = CelebASpritesDataset(data_limit=data_limit, cache_path=data_path, image_size=image_size)
    n_val = val_count if val_count is not None else (max(1, int(len(dataset) * validation_fraction)) if validation_fraction else 0)
    if not isinstance(n_val, int) or n_val < 0:
        raise ValueError('val_count must be a nonnegative integer')
    if split_mode == 'validation_first' and (val_count is None or n_val == 0):
        raise ValueError('validation_first requires an explicit positive val_count')
    if n_val >= len(dataset):
        raise ValueError('Need at least two images for a train/validation split')
    if split_mode == 'validation_first':
        train, val = Subset(dataset, list(range(n_val, len(dataset)))), Subset(dataset, list(range(n_val)))
    else:
        train, val = random_split(dataset, [len(dataset) - n_val, n_val],
                                  generator=torch.Generator().manual_seed(seed))
    kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=pin_memory, persistent_workers=num_workers > 0)
    sampler = DistributedSampler(train, world_size, rank, shuffle=True, seed=seed, drop_last=True) if world_size > 1 else None
    train_loader = DataLoader(train, shuffle=sampler is None, sampler=sampler, drop_last=world_size > 1,
                             generator=torch.Generator().manual_seed(seed + rank + 1), **kwargs)
    if not len(train_loader):
        raise ValueError('Training split is too small for a complete distributed global batch')
    val_loader = DataLoader(val, shuffle=False, generator=torch.Generator().manual_seed(seed + 2), **kwargs) if n_val else None
    return train_loader, val_loader


def split_manifest(train_loader, val_loader):
    images = train_loader.dataset.dataset.images
    return dict(cache_file=train_loader.dataset.dataset.cache_file, selected_shape=list(images.shape),
                selected_sha256=hashlib.sha256(images.tobytes()).hexdigest(),
                train_indices=list(train_loader.dataset.indices),
                validation_indices=list(val_loader.dataset.indices) if val_loader else [])


@torch.no_grad()
def evaluate_reconstruction(model, loader, device, amp, perceptual=None):
    """MSE in [0,1], using the VAE posterior mean, on held-out images."""
    model.eval()
    squared_error, pixels, perceptual_sum, image_count = 0.0, 0, 0.0, 0
    counts = torch.zeros(model.vq.num_embeddings, device=device, dtype=torch.long) if isinstance(model, VQVAE) else None
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with amp.autocast():
            if isinstance(model, VAE):
                recon = model.decode(model.encoder(batch)[0])
            else:
                recon, _, indices = model(batch, return_indices=True)
                counts += torch.bincount(indices, minlength=len(counts))
        squared_error += (recon.float() - batch).square().sum().item() / 4
        pixels += batch.numel()
        if perceptual is not None:
            perceptual_sum += perceptual(recon.float(), batch.float()).sum().item()
            image_count += len(batch)
    mse = squared_error / pixels
    result = {'val_recon_mse': mse, 'val_psnr_db': -10 * math.log10(max(mse, 1e-12))}
    if perceptual is not None:
        result['val_lpips'] = perceptual_sum / image_count
    if counts is not None:
        result.update(codebook_metrics(counts, 'val'))
    return result


def codebook_metrics(counts, prefix):
    probabilities = counts.float() / counts.sum().clamp_min(1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return {f'{prefix}_codes_used': int((counts > 0).sum()),
            f'{prefix}_codebook_perplexity': entropy.exp().item()}


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
    vq_recon_reduction: str = 'mean', image_size: int = 48, latent_spatial_size: int = 4,
    distance_chunk_size: int = 4096, data_path: str = None,
    split_mode: str = 'random', val_count: int = None,
    hidden_dims: List[int] = None, gan_loss_weight: float = 0.01,
    init_checkpoint: str = None, distributed: bool = False,
    codebook_decay: float = 0.0, dead_code_threshold: float = 0.0,
    codebook_data_init: bool = False, model_ema_halflife_images: int = 0,
    perceptual_weights: str = None, perceptual_loss_weight: float = 0.0,
    selection_metric: str = 'mse',
) -> Dict:
    if model_type not in {'vae', 'vqvae'} or min(epochs, batch_size, data_limit) <= 0:
        raise ValueError('Invalid model type or nonpositive training size')
    if model_ema_halflife_images < 0:
        raise ValueError('Model EMA half-life must be nonnegative')
    if kl_warmup_epochs < 0:
        raise ValueError('kl_warmup_epochs must be nonnegative')
    if vq_recon_reduction not in {'mean', 'sum'}:
        raise ValueError('vq_recon_reduction must be mean or sum')
    if any(not math.isfinite(v) or v < 0 for v in [kld_beta, vq_beta, commitment_cost, gan_loss_weight, perceptual_loss_weight]) or not math.isfinite(lr) or lr <= 0:
        raise ValueError('Learning rate must be positive and loss weights finite and nonnegative')
    if selection_metric not in {'mse', 'lpips'}:
        raise ValueError('selection_metric must be mse or lpips')
    if (perceptual_loss_weight or selection_metric == 'lpips') and not perceptual_weights:
        raise ValueError('Perceptual loss/selection requires explicit offline LPIPS weights')
    if perceptual_loss_weight and (model_type != 'vqvae' or vq_recon_reduction != 'mean'):
        raise ValueError('Perceptual loss currently requires VQVAE with mean reconstruction')
    config = {k: v for k, v in locals().copy().items()
              if k not in {'progress_callback', 'log_callback'}}
    if distributed and not dist.is_initialized():
        raise ValueError('Initialize the process group before distributed training')
    rank, world_size = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    if batch_size % world_size:
        raise ValueError('Global batch_size must be divisible by world_size')
    if distributed and use_gan:
        raise ValueError('Distributed GAN training is not implemented; use reconstruction training')
    if distributed and output_dir is None:
        raise ValueError('Distributed training requires a shared output_dir')
    setup_seed(seed)
    device = get_device(device)
    amp = MixedPrecision(device, precision, use_bfloat16)
    perceptual = None
    if perceptual_weights:
        from ..models.perceptual import load_perceptual
        perceptual = load_perceptual(perceptual_weights, device)
    selection_key = 'val_lpips' if selection_metric == 'lpips' else 'val_recon_mse'
    model_config = ({'latent_dim': latent_dim, 'hidden_dims': hidden_dims or [128, 256]} if model_type == 'vae' else {
        'embedding_dim': embedding_dim, 'num_embeddings': num_embeddings,
        'commitment_cost': commitment_cost, 'hidden_dims_enc': vq_hidden_dims_enc or [128, 64],
        'hidden_dims_dec': vq_hidden_dims_dec or [64, 128], 'distance_chunk_size': distance_chunk_size,
        'codebook_decay': codebook_decay, 'dead_code_threshold': dead_code_threshold})
    model_config.update(image_size=image_size, latent_spatial_size=latent_spatial_size)
    model = (VAE(**model_config) if model_type == 'vae' else VQVAE(**model_config)).to(device)
    if init_checkpoint:
        payload = torch.load(init_checkpoint, map_location='cpu', weights_only=True)
        if payload.get('model_type', model_type) != model_type:
            raise ValueError('Initialization checkpoint has a different model type')
        # Fully convolutional weights transfer across spatial sizes. Strict loading
        # still rejects changes to channel widths, codebook size or architecture.
        state = dict(payload.get('model_state_dict', payload))
        # Explicit migration from gradient-updated to EMA-updated codebook.
        if model_type == 'vqvae' and codebook_decay and 'vq.ema_count' not in state:
            state.update({'vq.ema_count': torch.ones(num_embeddings),
                          'vq.ema_sum': state['vq.embeddings.weight'].clone(),
                          'vq.ema_updates': torch.zeros((), dtype=torch.long)})
        model.load_state_dict(state, strict=True)
    training_model = (DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None)
                      if distributed else model)
    # Different posterior draws across ranks, while weights and sampler seed agree.
    if distributed:
        setup_seed(seed + rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    discriminator = Discriminator().to(device) if use_gan else None
    optimizer_d = (torch.optim.Adam(discriminator.parameters(), lr=lr * 0.5, betas=(0.5, 0.999))
                   if use_gan else None)
    amp_d = MixedPrecision(device, precision, use_bfloat16) if use_gan else None
    train_loader, val_loader = make_loaders(batch_size // world_size, data_limit, seed, validation_fraction,
                                          num_workers, device.type == 'cuda', data_path, image_size,
                                          split_mode, val_count, rank, world_size)
    if codebook_data_init:
        if model_type != 'vqvae':
            raise ValueError('Codebook initialization requires VQVAE')
        model.eval()
        with torch.no_grad(), amp.autocast():
            model.vq.initialize_from_data(model.encoder(next(iter(train_loader)).to(device)))
    ema_model = copy.deepcopy(model).eval().requires_grad_(False) if model_ema_halflife_images else None
    preview_loader = val_loader or DataLoader(train_loader.dataset, batch_size=min(8, batch_size),
                                             generator=torch.Generator().manual_seed(seed + 3))
    fixed_images = next(iter(preview_loader))[:8].to(device)
    run_name = f'{model_type}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    root = Path(output_dir) if output_dir else Path('runs') / run_name
    if (root / 'config.json').exists():
        raise ValueError('Output already contains a run; choose a new output_dir')
    if distributed:
        dist.barrier()
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = root / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    if rank == 0:
        (root / 'config.json').write_text(json.dumps(config, indent=2))
        (root / 'split.json').write_text(json.dumps(split_manifest(train_loader, val_loader), indent=2))
    artifacts = {'checkpoints': [], 'images': [], 'tensorboard_logdir': str(root / 'tensorboard')}
    history, global_step, images_seen, best_score = [], 0, 0, float('inf')
    start = time.perf_counter()
    writer = SummaryWriter(str(root / 'tensorboard')) if rank == 0 else None
    try:
        for epoch in range(epochs):
            training_model.train()
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            epoch_start = time.perf_counter()
            totals, seen = {}, 0
            updates, skipped = 0, 0
            counts = torch.zeros(num_embeddings, device=device, dtype=torch.long) if model_type == 'vqvae' else None
            beta = kld_beta * min(1.0, (epoch + 1) / max(1, kl_warmup_epochs))
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'VAE {epoch+1}/{epochs}', disable=rank != 0)):
                batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with amp.autocast():
                    if model_type == 'vae':
                        recon, mu, logvar = training_model(batch)
                        loss, metrics = vae_loss(recon, batch, mu, logvar, beta)
                    else:
                        recon, vq_loss, indices = training_model(batch, return_indices=True)
                        counts += torch.bincount(indices, minlength=num_embeddings)
                        loss, metrics = vqvae_loss(recon, batch, vq_loss, vq_beta, vq_recon_reduction)
                    if perceptual_loss_weight:
                        pl = perceptual(recon, batch).mean()
                        loss = loss + perceptual_loss_weight * pl
                        metrics['perceptual_loss'] = pl.item()
                if discriminator is not None:
                    discriminator.train()
                    discriminator.requires_grad_(True)
                    optimizer_d.zero_grad(set_to_none=True)
                    with amp_d.autocast():
                        real = discriminator.logits(batch).float()
                        fake = discriminator.logits(recon.detach()).float()
                        d_loss = (nn.functional.softplus(-real) + nn.functional.softplus(fake)).mean()
                    amp_d.step(d_loss, optimizer_d, discriminator.parameters())
                    # Freezing parameters alone does not freeze BatchNorm buffers.
                    discriminator.eval()
                    discriminator.requires_grad_(False)
                    with amp.autocast():
                        adv = nn.functional.softplus(-discriminator.logits(recon).float()).mean()
                        loss = loss + gan_loss_weight * adv
                    metrics.update(d_loss=d_loss.item(), g_adv_loss=adv.item())
                finite = torch.isfinite(loss).to(torch.int32)
                if distributed:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite:
                    raise FloatingPointError(f'Nonfinite VAE loss at step {global_step}')
                updated = amp.step(loss, optimizer, model.parameters())
                if updated:
                    if model_type == 'vqvae':
                        model.vq.update_codebook()
                    if ema_model is not None:
                        decay = 0.5 ** (len(batch) * world_size / model_ema_halflife_images)
                        with torch.no_grad():
                            for (name, target), source in zip(ema_model.named_parameters(), model.parameters()):
                                if name.startswith('vq.'):
                                    target.copy_(source)
                                else:
                                    target.lerp_(source, 1 - decay)
                            for target, source in zip(ema_model.buffers(), model.buffers()):
                                target.copy_(source)
                updates += updated
                skipped += not updated
                metrics['loss'] = loss.item()
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + value * len(batch)
                    if writer is not None and global_step % 10 == 0:
                        writer.add_scalar(f'train/{key}', value, global_step)
                seen += len(batch)
                if rank == 0 and log_callback and batch_idx % 10 == 0:
                    log_callback(f'Epoch {epoch+1}, batch {batch_idx}: loss={metrics["loss"]:.4f}')
                global_step += 1
                images_seen += len(batch) * world_size
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            train_seconds = time.perf_counter() - epoch_start
            if distributed:
                values = torch.tensor([seen, *totals.values()], device=device, dtype=torch.float64)
                dist.all_reduce(values)
                seen = int(values[0].item())
                totals = dict(zip(totals, values[1:].tolist()))
                elapsed = torch.tensor(train_seconds, device=device)
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                train_seconds = elapsed.item()
                if counts is not None:
                    dist.all_reduce(counts)
                # DDP checkpoints/evaluation use rank-0 running BN statistics.
                for buffer in model.buffers():
                    dist.broadcast(buffer, src=0)
            record = dict(epoch=epoch + 1, **{k: v / seen for k, v in totals.items()},
                          kld_beta=beta, train_seconds=train_seconds, images_per_second=seen / train_seconds,
                          images_seen=images_seen, optimizer_updates=updates, skipped_updates=skipped,
                          world_size=world_size, global_batch_size=batch_size)
            if counts is not None:
                record.update(codebook_metrics(counts, 'train'))
            if rank == 0:
                if val_loader:
                    record.update(evaluate_reconstruction(model, val_loader, device, amp, perceptual))
                    for key in ('val_recon_mse', 'val_psnr_db'):
                        writer.add_scalar(key, record[key], epoch + 1)
                selected_model = model
                record['selected_weights'] = 'raw'
                if val_loader and ema_model is not None:
                    ema_metrics = evaluate_reconstruction(ema_model, val_loader, device, amp, perceptual)
                    record.update({'raw_' + k: v for k, v in record.copy().items() if k.startswith('val_')})
                    record.update({'ema_' + k: v for k, v in ema_metrics.items()})
                    if ema_metrics[selection_key] < record[selection_key]:
                        selected_model = ema_model
                        record.update(ema_metrics)
                        record['selected_weights'] = 'ema'
                selected_model.eval()
                with torch.no_grad(), amp.autocast():
                    z = selected_model.encoder(fixed_images)[0] if model_type == 'vae' else selected_model.encode(fixed_images)
                    recon = selected_model.decode(z)
                comparison = torch.cat([fixed_images, recon.float()])
                image_path = root / f'reconstruction_{epoch+1:03d}.png'
                tensor_to_pil(comparison).save(image_path)
                writer.add_images('reconstruction', denormalize(comparison), epoch + 1)
                artifacts['images'].append(str(image_path))
                history.append(record)
                payload = dict(format_version=3, model_type=model_type, model_config=model_config,
                               config=config, epoch=epoch + 1, global_step=global_step,
                               images_seen=images_seen,
                               model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                               scaler_state_dict=amp.scaler.state_dict(), metrics=record)
                if ema_model is not None:
                    payload['ema_state_dict'] = ema_model.state_dict()
                final_path = checkpoint_dir / 'last.pth'
                torch.save(payload, final_path)
                if val_loader and record[selection_key] < best_score:
                    best_score = record[selection_key]
                    selected_payload = dict(payload, model_state_dict=selected_model.state_dict())
                    # Selected inference weights need not correspond to the live optimizer.
                    selected_payload.pop('optimizer_state_dict')
                    torch.save(selected_payload, checkpoint_dir / 'best.pth')
                (root / 'metrics.json').write_text(json.dumps({
                    'config': config, 'precision': amp.precision, 'train_images': len(train_loader.dataset),
                    'validation_images': len(val_loader.dataset) if val_loader else 0,
                    'epochs': history, 'total_seconds': time.perf_counter() - start,
                }, indent=2))
                print(json.dumps(record), flush=True)
                if progress_callback:
                    progress_callback(epoch + 1, epochs)
            if distributed:
                dist.barrier()
        if rank != 0:
            return {}
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
        if writer is not None:
            writer.close()
