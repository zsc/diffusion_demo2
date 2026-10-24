"""Sample full or EMA-only PixelJiT checkpoints without training dependencies."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from pixel_experiment import Runtime, save_grid, write_json
from diffusion_demo.core.models.pixel_jit import build_pixel_model, heun_sample


class AutoGuidedPredictor(torch.nn.Module):
    """Extrapolate good and weaker clean-image predictions during sampling.

    Reference: https://arxiv.org/abs/2406.02507 (unconditional autoguidance).
    Scale 1 is exactly the original model, with no extra forward call.
    """
    def __init__(self, model, guide, scale=1.0, start=0.0, end=1.0):
        super().__init__()
        self.model, self.guide, self.scale = model, guide, scale
        self.start, self.end = start, end

    def forward(self, x, t):
        good = self.model(x, t)
        if self.scale == 1:
            return good
        active = ((t >= self.start) & (t <= self.end))[:, None, None, None]
        if not bool(active.any()):
            return good
        bad = self.guide(x, t)
        return good.float() + active * (self.scale - 1) * (good.float() - bad.float())


def apply_autoguidance(model, cfg, guide_checkpoint, scale, device, start=0.0, end=1.0):
    if not np.isfinite(scale) or scale < 1:
        raise ValueError("guidance must be finite and >= 1")
    if not 0 <= start < end <= 1:
        raise ValueError("guidance time interval must satisfy 0 <= start < end <= 1")
    if scale == 1:
        return model
    if not guide_checkpoint:
        raise ValueError("guidance > 1 requires a weaker model checkpoint")
    guide, guide_cfg = load_ema(guide_checkpoint, device)
    if guide_cfg["model"]["image_size"] != cfg["model"]["image_size"]:
        raise ValueError("Guidance image resolutions differ")
    return AutoGuidedPredictor(model, guide, scale, start, end).eval()


def load_ema(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_pixel_model(state["config"]["model"], device).eval().requires_grad_(False)
    model.load_state_dict(state["ema"])
    return model, state["config"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--count", type=int, default=64)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--guide-checkpoint")
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--guidance-start", type=float, default=0.0)
    ap.add_argument("--guidance-end", type=float, default=1.0)
    args = ap.parse_args()
    if min(args.count, args.steps, args.batch) < 1:
        raise ValueError("count, steps and batch must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    rt = Runtime(args.precision, args.device)
    model, cfg = load_ema(args.checkpoint, rt.device)
    model = apply_autoguidance(model, cfg, args.guide_checkpoint, args.guidance, rt.device,
                              args.guidance_start, args.guidance_end)
    generator = torch.Generator(device=rt.device).manual_seed(args.seed)
    size = cfg["model"].get("image_size", 48)
    noise = torch.randn(args.count, 3, size, size, device=rt.device, generator=generator)
    with rt.amp():
        images = torch.cat([heun_sample(model, n, args.steps).cpu() for n in noise.split(args.batch)])
    if not images.isfinite().all():
        raise FloatingPointError("Non-finite generated image")
    np.save(output / "samples.npy", images.numpy())
    save_grid(images, output / "grid.png")
    save_grid(images[:16], output / "preview16.png", columns=4)
    u8 = ((images.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).numpy()
    (output / "images").mkdir()
    for index, array in enumerate(u8):
        Image.fromarray(array).save(output / "images" / f"{index:05d}.png")
    write_json(output / "metadata.json", dict(checkpoint=args.checkpoint,
               checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
               guide_checkpoint=args.guide_checkpoint, guidance=args.guidance,
               guidance_interval=[args.guidance_start, args.guidance_end],
               guide_sha256=hashlib.sha256(Path(args.guide_checkpoint).read_bytes()).hexdigest() if args.guide_checkpoint else None,
               seed=args.seed, count=args.count, steps=args.steps, precision=rt.precision,
               resolution=size, config=cfg))
    print(json.dumps(dict(output=str(output.resolve()), count=args.count, seed=args.seed)))


if __name__ == "__main__":
    main()
