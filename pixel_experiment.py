"""Train/benchmark/sample a resumable pixel JiT on the complete 48px dataset."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np
from PIL import Image
import torch

from diffusion_demo.core.models.pixel_jit import build_pixel_model, flow_loss, heun_sample
from pixel_transfer import inherit_pixel_weights
from pixel_data import split_indices


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def save_grid(x, path, columns=8, scale=3):
    x = ((x.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().byte()
    n, _, h, w = x.shape
    grid = np.full((math.ceil(n / columns) * (h + 2), columns * (w + 2), 3), 32, np.uint8)
    for i, image in enumerate(x.permute(0, 2, 3, 1).numpy()):
        r, c = divmod(i, columns)
        grid[r * (h + 2):r * (h + 2) + h, c * (w + 2):c * (w + 2) + w] = image
    img = Image.fromarray(grid)
    img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST).save(path)


class Runtime:
    def __init__(self, precision, device):
        self.device = torch.device(device)
        self.precision = precision if self.device.type == "cuda" else "fp32"
        self.dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[self.precision]
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.precision == "fp16")

    def amp(self):
        return torch.autocast(self.device.type, dtype=self.dtype, enabled=self.precision != "fp32")

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()


@torch.no_grad()
def update_ema(ema, model, decay):
    torch._foreach_mul_(list(ema.parameters()), decay)
    torch._foreach_add_(list(ema.parameters()), list(model.parameters()), alpha=1 - decay)


def train_step(model, ema, optimizer, rt, clean, config, step):
    t = (torch.randn(len(clean), device=clean.device) * config.get("p_std", 0.8) + config.get("p_mean", -0.8)).sigmoid()
    noise = torch.randn_like(clean)
    optimizer.zero_grad(set_to_none=True)
    with rt.amp():
        loss = flow_loss(model, clean, t, noise)
    scale = rt.scaler.get_scale()
    rt.scaler.scale(loss).backward()
    rt.scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    rt.scaler.step(optimizer)
    rt.scaler.update()
    updated = rt.scaler.get_scale() >= scale
    if updated:
        update_ema(ema, model, min(config.get("ema_decay", 0.9996), (1 + step) / (10 + step)))
    return loss.detach(), norm.detach(), updated


def environment(model, rt):
    return dict(torch=torch.__version__, python=platform.python_version(), precision=rt.precision,
                device=torch.cuda.get_device_name() if rt.device.type == "cuda" else "cpu",
                cuda=torch.version.cuda, parameters=sum(p.numel() for p in model.parameters()),
                model=model.config, visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [Path(__file__), Path("src/diffusion_demo/core/models/pixel_jit.py")]})


def benchmark(model, ema, optimizer, rt, config, output, repeats=3, steps=40):
    batch = config["batch_size"]
    clean = torch.randn(batch, 3, model.image_size, model.image_size, device=rt.device).clamp(-1, 1)
    for i in range(10):
        train_step(model, ema, optimizer, rt, clean, config, i + 1)
    measurements = []
    for r in range(repeats):
        rt.sync()
        start = time.perf_counter()
        losses = []
        for i in range(steps):
            loss, _, _ = train_step(model, ema, optimizer, rt, clean, config, 20 + i)
            losses.append(loss)
        rt.sync()
        duration = time.perf_counter() - start
        value = dict(seconds=duration, milliseconds_per_step=1000 * duration / steps,
                     images_per_second=batch * steps / duration,
                     finite=bool(torch.stack(losses).isfinite().all()))
        measurements.append(value)
        print(json.dumps(value), flush=True)
    noise = torch.randn(16, 3, model.image_size, model.image_size, device=rt.device)
    rt.sync()
    start = time.perf_counter()
    with rt.amp():
        samples = heun_sample(ema, noise, 50)
    rt.sync()
    result = dict(environment=environment(model, rt), batch_size=batch, warmup_steps=10,
                  steps_per_repeat=steps, repeats=measurements,
                  median_images_per_second=statistics.median(v["images_per_second"] for v in measurements),
                  heun_50_16_images_seconds=time.perf_counter() - start,
                  samples_finite=bool(samples.isfinite().all()),
                  max_memory_mb=torch.cuda.max_memory_allocated() / 2**20 if rt.device.type == "cuda" else 0)
    write_json(output / "benchmark.json", result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--data", default="cache/celeba_48x48_sprites_v2.npy")
    start = ap.add_mutually_exclusive_group()
    start.add_argument("--resume")
    start.add_argument("--init-from")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--benchmark-steps", type=int, default=40)
    ap.add_argument("--sample-only", action="store_true")
    ap.add_argument("--sample-seed", type=int, default=2026)
    args = ap.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists() and not (args.resume or args.sample_only):
        raise ValueError("Output already used; explicitly resume or use a new directory")
    torch.set_num_threads(config.get("cpu_threads", 4))
    torch.manual_seed(config.get("seed", 42))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    rt = Runtime(config.get("precision", "fp32"), config.get("device", "cuda"))
    model = build_pixel_model(config["model"], rt.device)
    size = model.image_size
    initialization = inherit_pixel_weights(model, args.init_from) if args.init_from else None
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.95), weight_decay=0)
    start_step = 0
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ckpt["config"]["model"] != config["model"]:
            raise ValueError("Resume architecture mismatch")
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        rt.scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"]
        initialization = ckpt.get("initialization")
    if initialization:
        write_json(output / "initialization.json", initialization)
    if not args.sample_only:
        write_json(output / "config.json", config)
    write_json(output / "environment.json", environment(model, rt))
    print(json.dumps(environment(model, rt)), flush=True)
    if args.benchmark:
        benchmark(model, ema, optimizer, rt, config, output, steps=args.benchmark_steps)
        return
    if args.sample_only:
        if not ckpt:
            raise ValueError("Sampling requires --resume checkpoint")
        torch.manual_seed(args.sample_seed)
        with rt.amp():
            images = heun_sample(ema, torch.randn(64, 3, size, size, device=rt.device), config.get("sample_steps", 50))
        save_grid(images, output / f"samples_seed{args.sample_seed}.png")
        np.save(output / f"samples_seed{args.sample_seed}.npy", images.cpu().numpy())
        return
    array = np.load(args.data)
    if array.dtype != np.uint8 or array.shape[1:] != (size, size, 3):
        raise ValueError(f"Expected uint8 N,{size},{size},3 image data")
    data_hash = hashlib.sha256(array.tobytes()).hexdigest()
    train_indices, val_indices = split_indices(len(array), config)
    write_json(output / "split.json", dict(data_sha256=data_hash, train_indices=train_indices.tolist(),
                                           val_indices=val_indices.tolist()))
    if ckpt and ckpt["data_sha256"] != data_hash:
        raise ValueError("Resume data mismatch")
    # uint8 residency is only 198 MiB for all 30k images; no slow image decoding in the loop.
    train_data = torch.from_numpy(array[train_indices].copy()).permute(0, 3, 1, 2).contiguous().to(rt.device)
    val_data = torch.from_numpy(array[val_indices].copy()).permute(0, 3, 1, 2).contiguous().to(rt.device)
    save_grid(val_data[:64].float() / 127.5 - 1, output / "real_validation.png")
    sample_gen = torch.Generator(device=rt.device).manual_seed(2026)
    fixed_noise = torch.randn(64, 3, size, size, device=rt.device, generator=sample_gen)
    if ckpt:
        torch.set_rng_state(ckpt["rng_cpu"])
        if rt.device.type == "cuda":
            torch.cuda.set_rng_state(ckpt["rng_cuda"], rt.device)
    total_steps = config["steps"]
    batch = config["batch_size"]
    losses, norms, skipped = [], [], 0
    rt.sync()
    timer = time.perf_counter()
    for step in range(start_step + 1, total_steps + 1):
        idx = torch.randint(len(train_data), (batch,), device=rt.device)
        clean = train_data[idx].float() / 127.5 - 1
        flips = torch.rand(batch, 1, 1, 1, device=rt.device) < 0.5
        clean = torch.where(flips, clean.flip(-1), clean)
        for group in optimizer.param_groups:
            group["lr"] = config["lr"] * min(1, step / max(1, config.get("warmup_steps", 500)))
        loss, norm, updated = train_step(model, ema, optimizer, rt, clean, config, step)
        losses.append(loss)
        norms.append(norm)
        skipped += not updated
        if step % config.get("log_every", 50) == 0:
            rt.sync()
            elapsed = time.perf_counter() - timer
            value = dict(step=step, loss=torch.stack(losses).mean().item(),
                         grad_norm=torch.stack(norms).mean().item(), skipped=skipped,
                         images_per_second=len(losses) * batch / elapsed, lr=optimizer.param_groups[0]["lr"])
            if not math.isfinite(value["loss"]):
                raise FloatingPointError(value)
            print(json.dumps(value), flush=True)
            with (output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(value) + "\n")
            losses, norms, skipped = [], [], 0
            timer = time.perf_counter()
        if step % config.get("sample_every", 1000) == 0 or step == total_steps:
            # fork_rng keeps preview/validation from modifying the training random stream.
            with torch.random.fork_rng(devices=[rt.device.index or 0] if rt.device.type == "cuda" else []):
                torch.manual_seed(1729)
                vloss = []
                with torch.no_grad(), rt.amp():
                    for start in range(0, min(len(val_data), 256), batch):
                        vclean = val_data[start:start + batch].float() / 127.5 - 1
                        vt = (torch.randn(len(vclean), device=rt.device) * 0.8 - 0.8).sigmoid()
                        vloss.append(flow_loss(ema, vclean, vt, torch.randn_like(vclean)))
                    images = torch.cat([heun_sample(ema, noise, config.get("sample_steps", 50))
                                        for noise in fixed_noise.split(16)])
                if not images.isfinite().all():
                    raise FloatingPointError("Non-finite samples")
                save_grid(images, output / f"samples_{step:06d}.png")
                np.save(output / f"samples_{step:06d}.npy", images.cpu().numpy())
                val = dict(step=step, ema_val_loss=torch.stack(vloss).mean().item())
                print(json.dumps(val), flush=True)
                with (output / "validation.jsonl").open("a") as f:
                    f.write(json.dumps(val) + "\n")
            timer = time.perf_counter()
        if step % config.get("checkpoint_every", 1000) == 0 or step == total_steps:
            state = dict(step=step, model=model.state_dict(), ema=ema.state_dict(),
                         optimizer=optimizer.state_dict(), scaler=rt.scaler.state_dict(), config=config,
                         initialization=initialization,
                         data_sha256=data_hash, rng_cpu=torch.get_rng_state(),
                         rng_cuda=torch.cuda.get_rng_state(rt.device) if rt.device.type == "cuda" else None)
            torch.save(state, output / "last.pt.tmp")
            (output / "last.pt.tmp").replace(output / "last.pt")
            if step % 5000 == 0:
                torch.save(dict(step=step, ema=ema.state_dict(), config=config), output / f"ema_{step:06d}.pt")
            timer = time.perf_counter()
    write_json(output / "completed.json", dict(step=total_steps, finished_at=time.time()))


if __name__ == "__main__":
    main()
