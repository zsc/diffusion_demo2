"""VAE and VQVAE models."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class ResBlock(nn.Module):
    """Residual block for VAE."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class VAEEncoder(nn.Module):
    """Convolutional posterior with a configurable spatial bottleneck."""
    
    def __init__(self, latent_dim: int = 256, hidden_dims: List[int] = [128, 256], latent_spatial_size: int = 4):
        super().__init__()
        self.latent_spatial_size = latent_spatial_size
        layers = []
        in_channels = 3
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(in_channels, h_dim, 4, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.ReLU(),
                ResBlock(h_dim),
            ])
            in_channels = h_dim
        
        # Final conv to latent space
        # After 2 strided convs: 48 -> 24 -> 12
        layers.extend([
            nn.Conv2d(in_channels, latent_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(),
        ])
        
        self.encoder = nn.Sequential(*layers)
        
        # Posterior heads downsample once more before optional spatial pooling
        self.to_mu = nn.Conv2d(latent_dim, latent_dim, 4, stride=2, padding=1)  # 12 -> 6
        self.to_logvar = nn.Conv2d(latent_dim, latent_dim, 4, stride=2, padding=1)  # 12 -> 6
        
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # These layers existed in legacy checkpoints but were never used in forward.
        # Discard only those known unused tensors; keep strict checks for everything else.
        for layer in ("down_mu", "down_logvar"):
            for parameter in ("weight", "bias"):
                state_dict.pop(prefix + layer + "." + parameter, None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)  # (B, latent_dim, 12, 12)
        mu = self.to_mu(x)  # (B, latent_dim, 6, 6)
        logvar = self.to_logvar(x)
        
        if self.latent_spatial_size > min(mu.shape[-2:]):
            raise ValueError("latent_spatial_size exceeds the encoder grid; pooling cannot add detail")
        mu = F.adaptive_avg_pool2d(mu, self.latent_spatial_size)
        logvar = F.adaptive_avg_pool2d(logvar, self.latent_spatial_size)
        
        return mu, logvar


class VAEDecoder(nn.Module):
    """Convolutional decoder with a configurable output resolution."""
    
    def __init__(self, latent_dim: int = 256, hidden_dims: List[int] = [256, 128], image_size: int = 48):
        super().__init__()
        self.image_size = image_size
        
        self.input_conv = nn.Conv2d(latent_dim, latent_dim, 3, padding=1)
        
        layers = []
        in_channels = latent_dim
        
        # Each transpose convolution doubles the spatial size
        for h_dim in hidden_dims:
            layers.extend([
                nn.ConvTranspose2d(in_channels, h_dim, 4, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.ReLU(),
                ResBlock(h_dim),
            ])
            in_channels = h_dim
        
        # Final projection to RGB after resizing to the requested output resolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, 3, 3, padding=1),
            nn.Tanh(),
        )
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.input_conv(z)
        x = self.decoder(z)
        # Legacy grid=4 needs interpolation; native grids retain more spatial detail
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        return self.final_conv(x)


class VAE(nn.Module):
    """Standard VAE."""
    
    def __init__(self, latent_dim: int = 256, hidden_dims: List[int] = [128, 256],
                 image_size: int = 48, latent_spatial_size: int = 4):
        super().__init__()
        if min(image_size, latent_spatial_size) <= 0:
            raise ValueError("Image and latent sizes must be positive")
        self.image_size, self.latent_spatial_size = image_size, latent_spatial_size
        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(latent_dim, hidden_dims, latent_spatial_size)
        self.decoder = VAEDecoder(latent_dim, list(reversed(hidden_dims)), image_size)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        mu, logvar = mu.float(), logvar.float()
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def get_latent_shape(self, batch_size: int = 1) -> Tuple[int, ...]:
        """Get the shape of latent representation."""
        return (batch_size, self.latent_dim, self.latent_spatial_size, self.latent_spatial_size)


class VectorQuantizer(nn.Module):
    """Vector Quantizer for VQVAE."""
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        distance_chunk_size: int = 4096,
        codebook_decay: float = 0.0,
        dead_code_threshold: float = 0.0,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        if min(num_embeddings, embedding_dim, distance_chunk_size) <= 0 or commitment_cost < 0:
            raise ValueError("Invalid codebook size, distance chunk size or commitment cost")
        if not 0 <= codebook_decay < 1 or dead_code_threshold < 0:
            raise ValueError("Invalid EMA decay or dead-code threshold")
        self.codebook_decay = codebook_decay
        self.dead_code_threshold = dead_code_threshold
        self.distance_chunk_size = distance_chunk_size
        
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        if codebook_decay:
            self.embeddings.weight.requires_grad_(False)
            self.register_buffer('ema_count', torch.ones(num_embeddings))
            self.register_buffer('ema_sum', self.embeddings.weight.detach().clone())
            self.register_buffer('ema_updates', torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def initialize_from_data(self, z):
        """Training-only rank-0 feature sampling, shared identically across ranks."""
        values = z.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim).float()
        ids = torch.randperm(len(values), device=z.device)[:self.num_embeddings]
        if len(ids) < self.num_embeddings:
            ids = torch.arange(self.num_embeddings, device=z.device) % len(values)
        weight = values[ids].contiguous()
        if dist.is_initialized():
            dist.broadcast(weight, 0)
        self.embeddings.weight.copy_(weight)
        if self.codebook_decay:
            self.ema_count.fill_(1)
            self.ema_sum.copy_(weight)
            self.ema_updates.zero_()

    @torch.no_grad()
    def update_codebook(self):
        """Called only after a successful optimizer update, never by evaluation."""
        if not self.codebook_decay or not hasattr(self, '_pending'):
            return
        values, indices = self._pending
        del self._pending
        counts = torch.bincount(indices, minlength=self.num_embeddings).float()
        sums = torch.zeros_like(self.ema_sum)
        sums.index_add_(0, indices, values)
        if dist.is_initialized():
            dist.all_reduce(counts)
            dist.all_reduce(sums)
        # Counts are tokens per global batch; threshold is relative to uniform usage.
        decay = self.codebook_decay
        self.ema_count.lerp_(counts, 1 - decay)
        self.ema_sum.lerp_(sums, 1 - decay)
        self.ema_updates.add_(1)
        self.embeddings.weight.copy_(self.ema_sum / self.ema_count.clamp_min(1e-5)[:, None])
        if self.dead_code_threshold and self.ema_updates.item() >= 100 and self.ema_updates.item() % 100 == 0:
            threshold = self.dead_code_threshold * self.ema_count.mean()
            dead = self.ema_count < threshold
            replacements = values[torch.randint(len(values), (self.num_embeddings,), device=values.device)].contiguous()
            if dist.is_initialized():
                dist.broadcast(replacements, 0)
            reset_count = self.ema_count.mean().clamp_min(1)
            self.embeddings.weight[dead] = replacements[dead]
            self.ema_count[dead] = reset_count
            self.ema_sum[dead] = replacements[dead] * reset_count
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, C, H, W) continuous latent
        Returns:
            quantized: (B, C, H, W)
            vq_loss: scalar
            encoding_indices: (B*H*W,)
        """
        B, C, H, W = z.shape
        assert C == self.embedding_dim
        
        # Flatten
        flat_z = z.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        
        # Calculate distances
        with torch.no_grad(), torch.autocast(device_type=z.device.type, enabled=False):
            embeddings_fp32 = self.embeddings.weight.float()
            embedding_norm = embeddings_fp32.square().sum(dim=1)
            indices = []
            for chunk in flat_z.split(self.distance_chunk_size):
                chunk = chunk.float()
                distances = chunk.square().sum(dim=1, keepdim=True) + embedding_norm - 2 * chunk @ embeddings_fp32.t()
                indices.append(distances.argmin(dim=1))
            encoding_indices = torch.cat(indices)
        quantized_flat = self.embeddings(encoding_indices)
        quantized = quantized_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        # VQ loss
        e_latent_loss = F.mse_loss(quantized.detach().float(), z.float())
        q_latent_loss = F.mse_loss(quantized.float(), z.detach().float())
        vq_loss = (0 if self.codebook_decay else q_latent_loss) + self.commitment_cost * e_latent_loss
        if self.training and self.codebook_decay:
            self._pending = (flat_z.detach().float(), encoding_indices.detach())
        
        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        
        return quantized, vq_loss, encoding_indices


class VQVAEEncoder(nn.Module):
    """VQVAE Encoder."""
    
    def __init__(self, embedding_dim: int, hidden_dims: List[int] = [128, 64], latent_spatial_size: int = 4):
        super().__init__()
        self.latent_spatial_size = latent_spatial_size
        layers = []
        in_channels = 3
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(in_channels, h_dim, 4, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.ReLU(),
            ])
            in_channels = h_dim
        
        # Preserve the spatial size after the strided encoder blocks
        layers.append(nn.Conv2d(in_channels, embedding_dim, 3, stride=1, padding=1))
        
        self.encoder = nn.Sequential(*layers)
        self.to_latent = nn.Conv2d(embedding_dim, embedding_dim, 3, stride=1, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        if self.latent_spatial_size > min(x.shape[-2:]):
            raise ValueError("latent_spatial_size exceeds the encoder grid; pooling cannot add detail")
        x = F.adaptive_avg_pool2d(x, self.latent_spatial_size)
        return self.to_latent(x)


class VQVAEDecoder(nn.Module):
    """VQVAE Decoder."""
    
    def __init__(self, embedding_dim: int, hidden_dims: List[int] = [64, 128], image_size: int = 48):
        super().__init__()
        self.image_size = image_size
        self.input_conv = nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1)
        
        layers = []
        in_channels = embedding_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.ConvTranspose2d(in_channels, h_dim, 4, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.ReLU(),
            ])
            in_channels = h_dim
        
        # Final RGB projection at the configured output resolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, 3, 3, padding=1),
            nn.Tanh(),
        )
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.input_conv(z)
        x = self.decoder(z)
        # Legacy grid=4 needs interpolation; native grids retain more spatial detail
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        return self.final_conv(x)


class VQVAE(nn.Module):
    """Vector Quantized VAE."""
    
    def __init__(
        self,
        embedding_dim: int = 64,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
        hidden_dims_enc: List[int] = [128, 64],
        hidden_dims_dec: List[int] = [64, 128],
        image_size: int = 48,
        latent_spatial_size: int = 4,
        distance_chunk_size: int = 4096,
        codebook_decay: float = 0.0,
        dead_code_threshold: float = 0.0,
    ):
        super().__init__()
        if min(image_size, latent_spatial_size) <= 0:
            raise ValueError("Image and latent sizes must be positive")
        self.image_size, self.latent_spatial_size = image_size, latent_spatial_size
        self.embedding_dim = embedding_dim
        self.encoder = VQVAEEncoder(embedding_dim, hidden_dims_enc, latent_spatial_size)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost, distance_chunk_size, codebook_decay, dead_code_threshold)
        self.decoder = VQVAEDecoder(embedding_dim, hidden_dims_dec, image_size)
    
    def forward(self, x: torch.Tensor, return_indices: bool = False):
        z = self.encoder(x)
        quantized, vq_loss, indices = self.vq(z)
        recon = self.decoder(quantized)
        if return_indices:
            return recon, vq_loss, indices
        return recon, vq_loss
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        quantized, _, indices = self.vq(z)
        return quantized
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def get_latent_shape(self, batch_size: int = 1) -> Tuple[int, ...]:
        return (batch_size, self.embedding_dim, self.latent_spatial_size, self.latent_spatial_size)


class Discriminator(nn.Module):
    """Discriminator for GAN training."""
    
    def __init__(self, image_size: int = 48):
        super().__init__()
        
        def block(in_channels: int, out_channels: int, normalize: bool = True):
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(3, 64, normalize=False),   # 48 -> 24
            *block(64, 128),                   # 24 -> 12
            *block(128, 256),                  # 12 -> 6
            *block(256, 512),                  # 6 -> 3
            nn.Conv2d(512, 1, 3, padding=1),   # 3 -> 3
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Training uses logits; keep probability-valued forward for legacy callers."""
        for layer in list(self.model.children())[:-1]:
            x = layer(x)
        return x
