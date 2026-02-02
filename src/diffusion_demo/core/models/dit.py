"""Diffusion Transformer (DiT) model."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply modulation (scale and shift)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """Embeds class labels or unconditional tokens."""
    
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(num_classes + 1, hidden_size)
        self.dropout_prob = dropout_prob
        self.num_classes = num_classes
    
    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding(labels)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class DiTBlock(nn.Module):
    """DiT block with adaptive layer norm."""
    
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) input tokens
            c: (B, C) condition (timestep + label embedding)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Attention with residual
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        
        # MLP with residual
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


class DiT(nn.Module):
    """Diffusion Transformer."""
    
    def __init__(
        self,
        img_size: int = 4,  # Latent spatial size (e.g., 4x4)
        patch_size: int = 2,
        in_channels: int = 16,  # Latent channels (VAE latent dim)
        hidden_size: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_classes: int = 10,  # Not used but kept for compatibility
        learn_sigma: bool = False,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_patches = (img_size // patch_size) ** 2
        self.learn_sigma = learn_sigma
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, hidden_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        
        # Time and label embedders
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])
        
        # Final layer
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, patch_size * patch_size * self.out_channels),
        )
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Initialize transformer blocks
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        self.apply(_basic_init)
        
        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
    
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patches back to image.
        
        Args:
            x: (B, N, patch_size^2 * C)
        Returns:
            img: (B, C, H, W)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = self.img_size // p
        
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = rearrange(x, 'b h w p1 p2 c -> b c (h p1) (w p2)')
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) noised latent
            t: (B,) diffusion timesteps
            y: (B,) class labels (optional)
        Returns:
            (B, C, H, W) predicted noise
        """
        # Patchify
        x = self.patch_embed(x)  # (B, N, D)
        x = x + self.pos_embed
        
        # Get condition embedding
        t_emb = self.t_embedder(t)
        if y is None:
            y = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, c)
        
        # Final layer
        x = self.final_layer(x)
        x = self.unpatchify(x)
        
        return x


def create_dit_for_latent(
    latent_channels: int = 16,
    latent_spatial_dim: int = 4,
    patch_size: int = 2,
    hidden_size: int = 384,
    depth: int = 6,
    num_heads: int = 6,
) -> DiT:
    """Create DiT model for latent space diffusion.
    
    Args:
        latent_channels: Number of channels in latent space
        latent_spatial_dim: Spatial dimension of latent (e.g., 4 for 4x4)
        patch_size: Patch size for DiT
        hidden_size: Hidden dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
    """
    return DiT(
        img_size=latent_spatial_dim,
        patch_size=patch_size,
        in_channels=latent_channels,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        num_classes=10,
        learn_sigma=False,
    )
