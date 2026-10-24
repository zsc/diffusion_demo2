"""VAE and VQVAE models."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """VAE Encoder: 48x48 -> 4x4 (with latent_dim channels)."""
    
    def __init__(self, latent_dim: int = 256, hidden_dims: List[int] = [128, 256]):
        super().__init__()
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
        
        # Additional downsampling to get to 4x4
        self.to_mu = nn.Conv2d(latent_dim, latent_dim, 4, stride=2, padding=1)  # 12 -> 6
        self.to_logvar = nn.Conv2d(latent_dim, latent_dim, 4, stride=2, padding=1)  # 12 -> 6
        
        # Another downsample to 4x4 (approx)
        self.down_mu = nn.Conv2d(latent_dim, latent_dim, 3, stride=1, padding=1)
        self.down_logvar = nn.Conv2d(latent_dim, latent_dim, 3, stride=1, padding=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)  # (B, latent_dim, 12, 12)
        mu = self.to_mu(x)  # (B, latent_dim, 6, 6)
        logvar = self.to_logvar(x)
        
        # Downsample to 4x4 using adaptive pooling
        mu = F.adaptive_avg_pool2d(mu, (4, 4))
        logvar = F.adaptive_avg_pool2d(logvar, (4, 4))
        
        return mu, logvar


class VAEDecoder(nn.Module):
    """VAE Decoder: 4x4 -> 48x48."""
    
    def __init__(self, latent_dim: int = 256, hidden_dims: List[int] = [256, 128]):
        super().__init__()
        
        self.input_conv = nn.Conv2d(latent_dim, latent_dim, 3, padding=1)
        
        layers = []
        in_channels = latent_dim
        
        # 4 -> 8 -> 16 -> 32
        for h_dim in hidden_dims:
            layers.extend([
                nn.ConvTranspose2d(in_channels, h_dim, 4, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.ReLU(),
                ResBlock(h_dim),
            ])
            in_channels = h_dim
        
        # Final conv to 3 channels (will upsample to 48x48)
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
        # Upsample from 32x32 to 48x48 using interpolate
        x = nn.functional.interpolate(x, size=(48, 48), mode='bilinear', align_corners=False)
        return self.final_conv(x)


class VAE(nn.Module):
    """Standard VAE."""
    
    def __init__(self, latent_dim: int = 256, hidden_dims: List[int] = [128, 256]):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(latent_dim, hidden_dims)
        self.decoder = VAEDecoder(latent_dim, list(reversed(hidden_dims)))
    
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
        return (batch_size, self.latent_dim, 4, 4)


class VectorQuantizer(nn.Module):
    """Vector Quantizer for VQVAE."""
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
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
        with torch.autocast(device_type=z.device.type, enabled=False):
            flat_z_fp32 = flat_z.float()
            embeddings_fp32 = self.embeddings.weight.float()
            distances = (
                torch.sum(flat_z_fp32 ** 2, dim=1, keepdim=True)
                + torch.sum(embeddings_fp32 ** 2, dim=1)
                - 2 * torch.matmul(flat_z_fp32, embeddings_fp32.t())
            )
        
        # Get closest embeddings
        encoding_indices = torch.argmin(distances, dim=1)
        quantized_flat = self.embeddings(encoding_indices)
        quantized = quantized_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        # VQ loss
        e_latent_loss = F.mse_loss(quantized.detach().float(), z.float())
        q_latent_loss = F.mse_loss(quantized.float(), z.detach().float())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        
        return quantized, vq_loss, encoding_indices


class VQVAEEncoder(nn.Module):
    """VQVAE Encoder."""
    
    def __init__(self, embedding_dim: int, hidden_dims: List[int] = [128, 64]):
        super().__init__()
        layers = []
        in_channels = 3
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(in_channels, h_dim, 4, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.ReLU(),
            ])
            in_channels = h_dim
        
        # 48 -> 24 -> 12 -> 6
        layers.append(nn.Conv2d(in_channels, embedding_dim, 3, stride=1, padding=1))
        
        self.encoder = nn.Sequential(*layers)
        self.to_latent = nn.Conv2d(embedding_dim, embedding_dim, 3, stride=1, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        # Downsample to 4x4
        x = F.adaptive_avg_pool2d(x, (4, 4))
        return self.to_latent(x)


class VQVAEDecoder(nn.Module):
    """VQVAE Decoder."""
    
    def __init__(self, embedding_dim: int, hidden_dims: List[int] = [64, 128]):
        super().__init__()
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
        
        # Final conv layers (will upsample to 48x48)
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
        # Upsample from 32x32 to 48x48 using interpolate
        x = nn.functional.interpolate(x, size=(48, 48), mode='bilinear', align_corners=False)
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
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encoder = VQVAEEncoder(embedding_dim, hidden_dims_enc)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = VQVAEDecoder(embedding_dim, hidden_dims_dec)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        quantized, vq_loss, _ = self.vq(z)
        recon = self.decoder(quantized)
        return recon, vq_loss
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        quantized, _, indices = self.vq(z)
        return quantized
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


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
