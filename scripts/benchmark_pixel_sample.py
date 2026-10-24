"""Measure steady-state 50-step Heun inference, excluding first-use compilation."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pixel_experiment import Runtime, environment, write_json
from diffusion_demo.core.models.pixel_jit import build_pixel_model, heun_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    torch.set_num_threads(4)
    torch.manual_seed(42)
    rt = Runtime(config["precision"], config["device"])
    model = build_pixel_model(config["model"], rt.device).eval().requires_grad_(False)
    noise = torch.randn(16, 3, model.image_size, model.image_size, device=rt.device)
    samples = []
    with rt.amp():
        heun_sample(model, noise, 50)
        rt.sync()
        for _ in range(3):
            start = time.perf_counter()
            result = heun_sample(model, noise, 50)
            rt.sync()
            samples.append(time.perf_counter() - start)
    report = dict(environment=environment(model, rt), warmup_calls=1, samples=16, heun_steps=50,
                  repeats_seconds=samples, median_seconds=statistics.median(samples),
                  finite=bool(result.isfinite().all()))
    write_json(Path(args.output), report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
