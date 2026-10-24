"""Small pixel-space clean-image predictor, inspired by JiT and cstft_jit.

References: https://github.com/LTH14/JiT and the local cstft_jit experiments.
This implementation has no pretrained weights or VAE. Time runs noise -> data.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        if hasattr(self, "_rms_impl"):
            return self._rms_impl(x, self.weight)
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight.to(x.dtype)


def sincos(values, width):
    freq = torch.exp(-math.log(10000) * torch.arange(width // 2).float() / (width // 2))
    phase = values.float()[..., None] * freq.to(values.device)
    return torch.cat([phase.cos(), phase.sin()], -1)


class Attention(nn.Module):
    def __init__(self, width, heads, grid, position_scale=1.0):
        super().__init__()
        self.heads = heads
        dim = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.qnorm, self.knorm = RMSNorm(dim), RMSNorm(dim)
        yy, xx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
        freq = torch.exp(-math.log(10000) * torch.arange(dim // 4).float() / (dim // 4))
        phase = torch.cat([yy.flatten()[:, None] * freq, xx.flatten()[:, None] * freq], -1) * position_scale
        self.register_buffer("rope_cos", phase.cos()[None, None], persistent=False)
        self.register_buffer("rope_sin", phase.sin()[None, None], persistent=False)

    def rope(self, x):
        if hasattr(self, "_rope_impl"):
            return self._rope_impl(x, self.rope_cos, self.rope_sin)
        a, b = x[..., 0::2], x[..., 1::2]
        c, s = self.rope_cos.to(x.dtype), self.rope_sin.to(x.dtype)
        return torch.stack([a * c - b * s, a * s + b * c], -1).flatten(-2)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        if getattr(self, "_contiguous_qk", False):
            q, k = q.contiguous(), k.contiguous()
        q, k = self.rope(self.qnorm(q)), self.rope(self.knorm(k))
        y = F.scaled_dot_product_attention(q, k, v)
        return self.proj(y.transpose(1, 2).reshape(b, n, d))


class SwiGLU(nn.Module):
    def __init__(self, width):
        super().__init__()
        hidden = math.ceil((8 * width / 3) / 64) * 64
        self.gate = nn.Linear(width, hidden)
        self.up = nn.Linear(width, hidden)
        self.down = nn.Linear(hidden, width)

    def forward(self, x):
        if hasattr(self, "_activation_impl"):
            return self.down(self._activation_impl(self.gate(x), self.up(x)))
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, width, heads, grid, position_scale=1.0):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(width), RMSNorm(width)
        self.attn = Attention(width, heads, grid, position_scale)
        self.mlp = SwiGLU(width)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))

    def forward(self, x, cond):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(cond).unsqueeze(1).chunk(6, -1)
        if hasattr(self, "_modulate_impl"):
            x = self._residual_impl(x, gate1, self.attn(self._modulate_impl(self.norm1(x), scale1, shift1)))
            return self._residual_impl(x, gate2, self.mlp(self._modulate_impl(self.norm2(x), scale2, shift2)))
        x = x + gate1 * self.attn(self.norm1(x) * (1 + scale1) + shift1)
        return x + gate2 * self.mlp(self.norm2(x) * (1 + scale2) + shift2)


class PixelJiT(nn.Module):
    def __init__(self, image_size=48, patch_size=4, width=384, depth=8, heads=6,
                 bottleneck=64, conv_channels=32, conv_layers=3, position_scale=1.0):
        super().__init__()
        if image_size % patch_size or width % 4 or width % heads or (width // heads) % 4:
            raise ValueError("Invalid patch, embedding or RoPE dimensions")
        if not math.isfinite(position_scale) or position_scale <= 0:
            raise ValueError("position_scale must be finite and positive")
        self.config = dict(image_size=image_size, patch_size=patch_size, width=width,
                           depth=depth, heads=heads, bottleneck=bottleneck,
                           conv_channels=conv_channels, conv_layers=conv_layers,
                           position_scale=position_scale)
        self.image_size, self.patch_size = image_size, patch_size
        grid = image_size // patch_size
        self.patch = nn.Sequential(nn.Conv2d(3, bottleneck, patch_size, patch_size, bias=False),
                                   nn.Conv2d(bottleneck, width, 1))
        yy, xx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
        pos = torch.cat([sincos(yy.flatten() * position_scale, width // 2),
                         sincos(xx.flatten() * position_scale, width // 2)], -1)
        self.register_buffer("pos", pos[None], persistent=False)
        self.time = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([Block(width, heads, grid, position_scale) for _ in range(depth)])
        self.norm = RMSNorm(width)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.out = nn.Linear(width, 3 * patch_size * patch_size)
        conv = []
        for i in range(conv_layers - 1):
            conv.extend([nn.Conv2d(3 if i == 0 else conv_channels, conv_channels, 3, padding=1),
                         nn.GroupNorm(8, conv_channels), nn.SiLU()])
        if conv_layers:
            conv.append(nn.Conv2d(conv_channels if conv_layers > 1 else 3, 3, 3, padding=1))
        self.refine = nn.Sequential(*conv) if conv else None
        self.apply(self.init_linear)
        for block in self.blocks:
            nn.init.zeros_(block.ada[-1].weight)
            nn.init.zeros_(block.ada[-1].bias)
        for layer in [self.ada[-1], self.out] + ([self.refine[-1]] if self.refine else []):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def init_linear(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, images, t):
        if images.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError(f"Expected N,3,{self.image_size},{self.image_size} images")
        x = self.patch(images).flatten(2).transpose(1, 2)
        if getattr(self, "_contiguous_tokens", False):
            x = x.contiguous()
        x = x + self.pos.to(x.dtype)
        cond = self.time(sincos(t * 1000, 256).to(x.dtype))
        for block in self.blocks:
            x = block(x, cond)
        shift, scale = self.ada(cond).unsqueeze(1).chunk(2, -1)
        normalized = self.norm(x)
        x = self.out(self._modulate_impl(normalized, scale, shift) if hasattr(self, "_modulate_impl")
                     else normalized * (1 + scale) + shift)
        b = x.shape[0]
        p, g = self.patch_size, self.image_size // self.patch_size
        x = x.reshape(b, g, g, p, p, 3).permute(0, 5, 1, 3, 2, 4).reshape(b, 3, self.image_size, self.image_size)
        return x + self.refine(x) if self.refine is not None else x


def build_pixel_model(config, device):
    """Construct CPU buffers serially before restoring normal execution threads.

    The PPU PyTorch build can SIGSEGV in parallel CPU cos/sin while constructing
    the 96px positional grid. This affects initialization, not GPU attention.
    Keep exactly the same tensor math and learned weights, with a serial init.
    """
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        model = PixelJiT(**config)
    finally:
        torch.set_num_threads(threads)
    return model.to(device)


def flow_loss(model, clean, t, noise, t_eps=0.05):
    t4 = t[:, None, None, None]
    noisy = t4 * clean + (1 - t4) * noise
    predicted = model(noisy, t).float()
    # Algebraically identical to velocity MSE, with subtraction in FP32.
    return ((predicted - clean.float()) / (1 - t4).clamp_min(t_eps)).square().mean()


@torch.no_grad()
def heun_sample(model, noise, steps=50, t_eps=0.05):
    x = noise.float().clone()
    times = torch.linspace(0, 1, steps + 1, device=x.device)
    for i in range(steps):
        t, next_t = times[i], times[i + 1]
        dt = next_t - t
        v = (model(x, t.expand(len(x))).float() - x) / (1 - t).clamp_min(t_eps)
        trial = x + dt * v
        if i < steps - 1:
            next_v = (model(trial, next_t.expand(len(x))).float() - trial) / (1 - next_t).clamp_min(t_eps)
            x = x + dt * (v + next_v) * 0.5
        else:
            x = trial
    return x
