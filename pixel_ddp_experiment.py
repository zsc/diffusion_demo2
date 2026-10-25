"""One-node DDP experiments with image-budget-aligned warmup, EMA and evaluation."""
import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from pixel_experiment import Runtime, environment, save_grid, write_json, update_ema
from diffusion_demo.core.models.pixel_jit import build_pixel_model, flow_loss, heun_sample
from pixel_transfer import inherit_pixel_weights
from pixel_data import split_indices


def training_time_parameters(cfg):
    location = float(cfg.get("time_logit_mean", -.8))
    scale = float(cfg.get("time_logit_std", .8))
    if not math.isfinite(location) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid training time distribution")
    return location, scale


def training_time_sampling(cfg):
    mode = cfg.get("time_sampling", "iid")
    if mode not in ("iid", "stratified"):
        raise ValueError("Invalid training time sampling")
    return mode


def sample_training_times(batch, device, location, scale, mode="iid"):
    if mode == "iid":
        normal = torch.randn(batch, device=device)
    elif mode == "stratified":
        # One independent jitter per equal-probability normal-CDF stratum.
        # The batch-average objective is unchanged; examples/noise are IID.
        uniform = (torch.arange(batch, device=device) + torch.rand(batch, device=device)) / batch
        eps = torch.finfo(uniform.dtype).eps
        normal = math.sqrt(2) * torch.erfinv(2 * uniform.clamp(eps, 1 - eps) - 1)
    else:
        raise ValueError("Invalid training time sampling")
    return (normal * scale + location).sigmoid()


def validate_resume_config(previous, cfg, allow_lr_change=False, allow_time_change=False):
    for key in ["model", "global_batch", "seed", "lr", "warmup_images", "ema_reference_batch",
                "ema_reference_decay", "split_mode", "val_count", "precision"]:
        if previous.get(key) != cfg.get(key):
            if key == "lr" and allow_lr_change and math.isfinite(cfg["lr"]) and cfg["lr"] > 0:
                continue
            raise ValueError(f"Resume changed {key}")
    time_changed = (training_time_parameters(previous) != training_time_parameters(cfg) or
                    training_time_sampling(previous) != training_time_sampling(cfg))
    if time_changed and not allow_time_change:
        raise ValueError("Resume changed training time distribution")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--data", required=True)
    start = ap.add_mutually_exclusive_group()
    start.add_argument("--resume")
    start.add_argument("--init-from", help="Initialize all learned weights from a lower-resolution EMA")
    ap.add_argument("--allow-resume-lr-change", action="store_true",
                    help="Explicitly change LR while retaining optimizer, EMA, counters and RNG")
    ap.add_argument("--allow-resume-time-change", action="store_true",
                    help="Explicitly change training time distribution; validation distribution stays fixed")
    args = ap.parse_args()
    if args.allow_resume_lr_change and not args.resume:
        ap.error("--allow-resume-lr-change requires --resume")
    if args.allow_resume_time_change and not args.resume:
        ap.error("--allow-resume-time-change requires --resume")
    cfg = json.loads(Path(args.config).read_text())
    time_mean, time_std = training_time_parameters(cfg)
    time_sampling = training_time_sampling(cfg)
    rank, local_rank, world = [int(os.environ.get(k, d)) for k, d in
                               [("RANK", 0), ("LOCAL_RANK", 0), ("WORLD_SIZE", 1)]]
    compiler_cache = None
    if cfg.get("execution_backend", "eager").startswith("compile"):
        # Shared-filesystem caches produced stale handles when 16 ranks compiled at once.
        # Keep each rank's generated code on the node-local filesystem.
        run_key = hashlib.sha256(str(Path(args.output).resolve()).encode()).hexdigest()[:16]
        compiler_cache = Path(tempfile.gettempdir()) / "pixel_jit_compile" / run_key / f"rank{rank}"
        os.environ["TRITON_CACHE_DIR"] = str(compiler_cache / "triton")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(compiler_cache / "inductor")
        os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    device = torch.device("cuda", local_rank) if cfg.get("device", "cuda") == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    def barrier():
        if world > 1:
            dist.barrier()
    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if (output / "config.json").exists() and not args.resume:
            raise ValueError("Output exists; use explicit resume")
        (output / "completed.json").unlink(missing_ok=True)
        write_json(output / "config.json", cfg)
    barrier()
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    rt = Runtime(cfg.get("precision", "bf16"), device)
    model = build_pixel_model(cfg["model"], device)
    size = model.image_size
    initialization = inherit_pixel_weights(model, args.init_from) if args.init_from else None
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(.9, .95), weight_decay=0)
    batch = cfg["global_batch"] // world
    if cfg["global_batch"] % world:
        raise ValueError("Global batch must divide world size")
    step, images_seen, restored = 0, 0, None
    continuation = None
    if args.resume:
        restored = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume_config(restored["config"], cfg, args.allow_resume_lr_change, args.allow_resume_time_change)
        if restored["world_size"] != world:
            raise ValueError("Exact resume requires the same world size")
        model.load_state_dict(restored["model"])
        ema.load_state_dict(restored["ema"])
        optimizer.load_state_dict(restored["optimizer"])
        rt.scaler.load_state_dict(restored["scaler"])
        step, images_seen = restored["step"], restored["images_seen"]
        initialization = restored.get("initialization")
        continuation = dict(checkpoint=str(Path(args.resume).resolve()), step=step,
                            images_seen=images_seen, previous_lr=restored["config"]["lr"],
                            lr=cfg["lr"], explicit_lr_change=args.allow_resume_lr_change,
                            previous_time_distribution=training_time_parameters(restored["config"]),
                            time_distribution=(time_mean, time_std), explicit_time_change=args.allow_resume_time_change,
                            previous_time_sampling=training_time_sampling(restored["config"]),
                            time_sampling=time_sampling,
                            previous_continuation=restored.get("continuation"))
        if rank == 0:
            write_json(output / "continuation.json", continuation)
    execution_backend = cfg.get("execution_backend", "eager")
    from pixel_accelerate import prepare_training_model
    training_model = prepare_training_model(model, execution_backend)
    if rank == 0 and initialization:
        write_json(output / "initialization.json", initialization)
    wrapped = DDP(training_model, device_ids=[local_rank] if device.type == "cuda" else None,
                  broadcast_buffers=False, gradient_as_bucket_view=True, static_graph=True) if world > 1 else training_model
    array = np.load(args.data)
    if array.dtype != np.uint8 or array.shape[1:] != (size, size, 3):
        raise ValueError(f"Expected uint8 N,{size},{size},3")
    data_hash = hashlib.sha256(array.tobytes()).hexdigest()
    if restored and restored["data_sha256"] != data_hash:
        raise ValueError("Resume data mismatch")
    train_idx, val_idx = split_indices(len(array), cfg)
    train_data = torch.from_numpy(array[train_idx].copy()).permute(0, 3, 1, 2).contiguous().to(device)
    val_data = torch.from_numpy(array[val_idx].copy()).permute(0, 3, 1, 2).contiguous().to(device) if rank == 0 else None
    del array
    if rank == 0:
        env = environment(model, rt)
        env.update(world_size=world, per_rank_batch=batch, global_batch=cfg["global_batch"],
                   ddp_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        env["compiler_cache"] = str(compiler_cache) if compiler_cache else None
        env["execution_backend"] = execution_backend
        env["ddp_graph_partitioning"] = execution_backend != "compile_full"
        env["previous_execution_backend"] = restored["config"].get("execution_backend", "eager") if restored else None
        if execution_backend != "eager":
            env["acceleration_source_sha256"] = hashlib.sha256(Path("pixel_accelerate.py").read_bytes()).hexdigest()
        write_json(output / "environment.json", env)
        write_json(output / "split.json", dict(data_sha256=data_hash, train_indices=train_idx.tolist(), val_indices=val_idx.tolist()))
        save_grid(val_data[:64].float() / 127.5 - 1, output / "real_validation.png")
        print(json.dumps(env), flush=True)
    # Common initialization, independent per-rank data and diffusion randomness.
    torch.manual_seed(seed + 100003 * rank)
    if restored:
        torch.set_rng_state(restored["rng"][rank]["cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(restored["rng"][rank]["cuda"], device)
    barrier()
    rt.sync()
    started = time.perf_counter()
    interval_start = started
    since_eval, interval_losses, interval_norms = 0, [], []
    interval_steps, skipped = 0, 0
    stop = False
    max_images = cfg["max_images"]
    eval_images = cfg["eval_every_images"]
    while images_seen < max_images and not stop:
        step += 1
        idx = torch.randint(len(train_data), (batch,), device=device)
        clean = train_data[idx].float() / 127.5 - 1
        clean = torch.where(torch.rand(batch, 1, 1, 1, device=device) < .5, clean.flip(-1), clean)
        t = sample_training_times(batch, device, time_mean, time_std, time_sampling)
        noise = torch.randn_like(clean)
        lr = cfg["lr"] * min(1, (images_seen + cfg["global_batch"]) / cfg.get("warmup_images", 128000))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with rt.amp():
            loss = flow_loss(wrapped, clean, t, noise)
        previous_scale = rt.scaler.get_scale()
        rt.scaler.scale(loss).backward()
        rt.scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        rt.scaler.step(optimizer)
        rt.scaler.update()
        updated = rt.scaler.get_scale() >= previous_scale
        images_seen += cfg["global_batch"]
        ref_updates = images_seen / cfg.get("ema_reference_batch", 256)
        decay = min(cfg.get("ema_reference_decay", .9996), (1 + ref_updates) / (10 + ref_updates))
        decay **= cfg["global_batch"] / cfg.get("ema_reference_batch", 256)
        if updated:
            update_ema(ema, model, decay)
        skipped += not updated
        since_eval += cfg["global_batch"]
        interval_losses.append(loss.detach())
        interval_norms.append(norm.detach())
        interval_steps += 1
        if step % cfg.get("log_every", 25) == 0:
            values = torch.stack([torch.stack(interval_losses).mean(), torch.stack(interval_norms).mean()])
            if world > 1:
                dist.all_reduce(values)
                values /= world
            rt.sync()
            elapsed = time.perf_counter() - interval_start
            if rank == 0:
                record = dict(step=step, images_seen=images_seen, loss=values[0].item(), grad_norm=values[1].item(),
                              images_per_second=cfg["global_batch"] * interval_steps / elapsed,
                              lr=lr, ema_decay=decay, skipped=skipped, elapsed_seconds=time.perf_counter() - started)
                if not math.isfinite(record["loss"]):
                    raise FloatingPointError(record)
                print(json.dumps(record), flush=True)
                with (output / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
            flag = torch.tensor(int((output / "STOP").exists()) if rank == 0 else 0, device=device)
            if world > 1:
                dist.broadcast(flag, 0)
            stop = bool(flag.item())
            interval_losses, interval_norms, interval_steps, skipped = [], [], 0, 0
            interval_start = time.perf_counter()
        if since_eval >= eval_images or images_seen >= max_images or stop:
            barrier()
            if rank == 0:
                with torch.random.fork_rng(devices=[local_rank] if device.type == "cuda" else []):
                    torch.manual_seed(1729)
                    val_losses = []
                    with torch.no_grad(), rt.amp():
                        for start in range(0, min(256, len(val_data)), 64):
                            clean = val_data[start:start + 64].float() / 127.5 - 1
                            t = (torch.randn(len(clean), device=device) * .8 - .8).sigmoid()
                            val_losses.append(flow_loss(ema, clean, t, torch.randn_like(clean)))
                        torch.manual_seed(2026)
                        noise = torch.randn(64, 3, size, size, device=device)
                        samples = torch.cat([heun_sample(ema, n, cfg.get("sample_steps", 50)) for n in noise.split(16)])
                    if not samples.isfinite().all():
                        raise FloatingPointError("Samples are not finite")
                    label = f"{images_seen:09d}"
                    save_grid(samples, output / f"samples_{label}.png")
                    np.save(output / f"samples_{label}.npy", samples.cpu().numpy())
                    record = dict(step=step, images_seen=images_seen, ema_val_loss=torch.stack(val_losses).mean().item(),
                                  elapsed_seconds=time.perf_counter() - started)
                    with (output / "validation.jsonl").open("a") as f:
                        f.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)
            rng = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
            all_rng = [None] * world if rank == 0 else None
            if world > 1:
                dist.gather_object(rng, all_rng, dst=0)
            else:
                all_rng = [rng]
            if rank == 0:
                state = dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                             scaler=rt.scaler.state_dict(), config=cfg, data_sha256=data_hash,
                             step=step, images_seen=images_seen, world_size=world, rng=all_rng,
                             initialization=initialization, continuation=continuation)
                torch.save(state, output / "last.pt.tmp")
                (output / "last.pt.tmp").replace(output / "last.pt")
                torch.save(dict(ema=ema.state_dict(), config=cfg, images_seen=images_seen,
                                initialization=initialization), output / f"ema_{images_seen:09d}.pt")
            barrier()
            since_eval = 0
            interval_losses, interval_norms, interval_steps, skipped = [], [], 0, 0
            interval_start = time.perf_counter()
    if rank == 0:
        write_json(output / ("stopped.json" if stop else "completed.json"),
                   dict(step=step, images_seen=images_seen, elapsed_seconds=time.perf_counter() - started, finished_at=time.time()))
    barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
