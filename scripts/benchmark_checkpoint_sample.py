"""Benchmark the actual saved sampling recipe after one full warm-up."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pixel_experiment import Runtime, environment, write_json
from sample_pixel import load_ema, apply_autoguidance
from diffusion_demo.core.models.pixel_jit import heun_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--guide-checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--precision", default="fp16")
    args = ap.parse_args()
    torch.set_num_threads(4)
    rt = Runtime(args.precision, "cuda")
    model, cfg = load_ema(args.checkpoint, rt.device)
    info = environment(model, rt)
    model = apply_autoguidance(model, cfg, args.guide_checkpoint, 2, rt.device, .1, .8)
    gen = torch.Generator(device=rt.device).manual_seed(2027)
    size = cfg["model"].get("image_size", 48)
    noise = torch.randn(16, 3, size, size, device=rt.device, generator=gen)
    timings = []
    with rt.amp():
        heun_sample(model, noise, 50)
        rt.sync()
        for _ in range(3):
            start = time.perf_counter()
            samples = heun_sample(model, noise, 50)
            rt.sync()
            timings.append(time.perf_counter() - start)
    result = dict(environment=info, count=16, heun_steps=50, guidance=2,
                  guidance_interval=[.1, .8], warmup_calls=1, repeats_seconds=timings,
                  median_seconds=statistics.median(timings), finite=bool(samples.isfinite().all()),
                  checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                  guide_sha256=hashlib.sha256(Path(args.guide_checkpoint).read_bytes()).hexdigest())
    write_json(Path(args.output), result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
