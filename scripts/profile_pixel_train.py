"""Profile the real PixelJiT update; timing runs never enable a profiler.

Includes input augmentation, flow loss, backward, clipping, AdamW and EMA.
Uses a fixed 1024-image training bank; excludes DDP and checkpoint/preview work.
"""
import argparse
import copy
from contextlib import nullcontext
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pixel_experiment import Runtime, environment, update_ema, write_json
from sample_pixel import load_ema
from diffusion_demo.core.models.pixel_jit import flow_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--optimizer", choices=["default", "foreach", "fused"], default="default")
    ap.add_argument("--capture", choices=["none", "torch", "external"], default="none")
    ap.add_argument("--relaxed-fusion", action="store_true", help="Experimental: permit eliminated intermediate precision casts")
    ap.add_argument("--execution", default="eager", choices=["eager", "contiguous", "compile_norm", "compile_pointwise"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    rt = Runtime("bf16", "cuda")
    model, cfg = load_ema(args.checkpoint, rt.device)
    model.train().requires_grad_(True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    from pixel_accelerate import configure_execution
    configure_execution(model, args.execution, emulate_precision_casts=not args.relaxed_fusion)
    kwargs = {} if args.optimizer == "default" else {args.optimizer: True}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .95), weight_decay=0, **kwargs)
    array = np.load(args.data, mmap_mode="r")
    bank = torch.from_numpy(array[1000:2024].copy()).permute(0, 3, 1, 2).contiguous().to(rt.device)
    def scope(name):
        return torch.profiler.record_function(name) if args.capture == "torch" else nullcontext()
    def step():
        with scope("phase.input"):
            idx = torch.randint(len(bank), (args.batch,), device=rt.device)
            clean = bank[idx].float() / 127.5 - 1
            clean = torch.where(torch.rand(args.batch, 1, 1, 1, device=rt.device) < .5, clean.flip(-1), clean)
            t = (torch.randn(args.batch, device=rt.device) * .8 - .8).sigmoid()
            noise = torch.randn_like(clean)
            optimizer.zero_grad(set_to_none=True)
        with scope("phase.forward_loss"), rt.amp():
            loss = flow_loss(model, clean, t, noise)
        with scope("phase.backward"):
            loss.backward()
        with scope("phase.clip"):
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        with scope("phase.optimizer"):
            optimizer.step()
        with scope("phase.ema"):
            update_ema(ema, model, .9996 ** 4)
        return loss.detach(), norm.detach()
    losses = []
    warmup_started = time.perf_counter()
    for _ in range(args.warmup):
        loss, norm = step()
        losses.append(loss)
    rt.sync()
    warmup_seconds = time.perf_counter() - warmup_started
    result = dict(execution=args.execution, emulate_precision_casts=not args.relaxed_fusion, warmup_seconds=warmup_seconds, environment=environment(model, rt), checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                  batch=args.batch, tokens=(model.image_size // model.patch_size) ** 2,
                  optimizer=args.optimizer, capture=args.capture, warmup=args.warmup,
                  note="Single GPU; real 1024-image bank; includes augmentation, backward, clipping, AdamW and EMA; excludes DDP, preview and saving.")
    if args.capture == "external":
        torch.cuda.profiler.start()
        loss, norm = step()
        rt.sync()
        torch.cuda.profiler.stop()
        losses.append(loss)
    elif args.capture == "torch":
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, with_stack=True) as prof:
            loss, norm = step()
            rt.sync()
        losses.append(loss)
        trace = output / "torch_trace.json"
        prof.export_chrome_trace(str(trace))
        with gzip.open(trace.with_suffix(".json.gz"), "wb") as f:
            f.write(trace.read_bytes())
        trace.unlink()
        (output / "operators.txt").write_text(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=100))
    else:
        times = []
        for _ in range(args.repeats):
            rt.sync()
            start = time.perf_counter()
            for _ in range(args.steps):
                loss, norm = step()
                losses.append(loss)
            rt.sync()
            times.append((time.perf_counter() - start) / args.steps)
        result.update(repeat_step_seconds=times, median_step_seconds=statistics.median(times),
                      images_per_second=args.batch / statistics.median(times), steps_per_repeat=args.steps)
    result.update(loss_finite=bool(torch.stack(losses).isfinite().all()), last_loss=float(loss), last_grad_norm=float(norm),
                  optimizer_state_dtypes=sorted({str(v.dtype) for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)}))
    write_json(output / "result.json", result)
    if not result["loss_finite"]:
        raise FloatingPointError("Non-finite profile workload loss")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
