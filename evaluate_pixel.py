"""Parallel held-out clean-fid/KID evaluation with fixed per-image noise seeds.

Run with torchrun on the same allocated node after training. Inception is only an
evaluation network; the generative model is trained from scratch. Small-sample
FID is useful for paired comparisons here, not comparison to published FID50k.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from cleanfid.inception_torchscript import InceptionV3W
from cleanfid.resize import build_resizer
from scripts.score_pixel_features import score_features

from pixel_experiment import Runtime, save_grid, write_json
from sample_pixel import load_ema, apply_autoguidance
from diffusion_demo.core.models.pixel_jit import heun_sample
from pixel_data import split_indices


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--count", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=3333)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--guide-checkpoint")
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--guidance-start", type=float, default=0.0)
    ap.add_argument("--guidance-end", type=float, default=1.0)
    args = ap.parse_args()
    rank, local_rank, world = [int(os.environ.get(k, d)) for k, d in
                               [("RANK", 0), ("LOCAL_RANK", 0), ("WORLD_SIZE", 1)]]
    torch.set_num_threads(4)
    device = torch.device("cuda", local_rank) if args.device == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model, cfg = load_ema(args.checkpoint, device)
    size = cfg["model"].get("image_size", 48)
    model = apply_autoguidance(model, cfg, args.guide_checkpoint, args.guidance, device,
                              args.guidance_start, args.guidance_end)
    rt = Runtime(cfg.get("precision", "bf16"), device)
    inception = InceptionV3W(args.weights_dir, download=False, resize_inside=False).to(device).eval()
    resize = build_resizer("clean")
    started = time.perf_counter()
    gen = torch.Generator(device=device)
    indices = list(range(rank, args.count, world))
    def features(arrays):
        resized = np.stack([resize(im) for im in arrays])
        x = torch.from_numpy(resized).permute(0, 3, 1, 2).to(device)
        return inception(x).float().cpu().numpy()
    fake_features, previews = [], []
    for start in range(0, len(indices), args.batch):
        batch_ids = indices[start:start + args.batch]
        noise = torch.stack([torch.randn(3, size, size, device=device, generator=gen.manual_seed(args.seed + i))
                             for i in batch_ids])
        with rt.amp():
            images = heun_sample(model, noise, args.steps)
        if not images.isfinite().all():
            raise FloatingPointError("Invalid evaluation samples")
        arrays = ((images.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        fake_features.append(features(arrays))
        previews.extend([(i, image) for i, image in zip(batch_ids, arrays) if i < 64])
    data = np.load(args.data, mmap_mode="r")
    if data.dtype != np.uint8 or data.shape[1:] != (size, size, 3):
        raise ValueError(f"Expected uint8 N,{size},{size},3 evaluation data")
    _, split = split_indices(len(data), cfg)
    local_real = split[rank::world]
    real_features = [features(data[local_real[start:start + args.batch]]) for start in range(0, len(local_real), args.batch)]
    result = dict(fake=np.concatenate(fake_features), real=np.concatenate(real_features), previews=previews,
                  fake_ids=indices, real_ids=local_real)
    gathered = [None] * world if rank == 0 else None
    if world > 1:
        dist.gather_object(result, gathered, dst=0)
    else:
        gathered = [result]
    if rank == 0:
        fake = np.concatenate([r["fake"] for r in gathered])
        real = np.concatenate([r["real"] for r in gathered])
        fake = fake[np.argsort(np.concatenate([r["fake_ids"] for r in gathered]))]
        real = real[np.argsort(np.concatenate([r["real_ids"] for r in gathered]))]
        if not (np.isfinite(fake).all() and np.isfinite(real).all()):
            raise FloatingPointError("Invalid Inception features")
        np.save(output / "fake_features.npy", fake)
        np.save(output / "real_features.npy", real)
        previews = sorted([p for r in gathered for p in r["previews"]], key=lambda p:p[0])
        images = torch.from_numpy(np.stack([p[1] for p in previews])).permute(0, 3, 1, 2).float() / 127.5 - 1
        save_grid(images, output / "independent_seed_grid.png")
        report = dict(**score_features(real, fake),
                      checkpoint=args.checkpoint, seed=args.seed, heun_steps=args.steps,
                      guide_checkpoint=args.guide_checkpoint, guidance=args.guidance,
                      guidance_interval=[args.guidance_start, args.guidance_end],
                      elapsed_seconds=time.perf_counter() - started, world_size=world,
                      inception_sha256=hashlib.sha256((Path(args.weights_dir)/"inception-2015-12-05.pt").read_bytes()).hexdigest(),
                      note="Small-sample paired comparison, not FID50k. KID subset SD is not a confidence interval.")
        write_json(output / "quality.json", report)
        print(json.dumps(report), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
