"""Paired real-model update check before using fused AdamW in training."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pixel_experiment import Runtime, write_json
from sample_pixel import load_ema
from diffusion_demo.core.models.pixel_jit import flow_loss


@torch.no_grad()
def relative_error(left, right):
    diff2 = torch.zeros((), device="cuda", dtype=torch.float64)
    ref2 = torch.zeros_like(diff2)
    maximum = torch.zeros((), device="cuda")
    for a, b in zip(left, right):
        delta = a.float() - b.float()
        diff2 += delta.double().square().sum()
        ref2 += a.double().square().sum()
        maximum = torch.maximum(maximum, delta.abs().max())
    return dict(relative_l2=float((diff2 / ref2.clamp_min(1e-30)).sqrt()), max_abs=float(maximum))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--relaxed-fusion", action="store_true", help="Experimental: permit eliminated intermediate precision casts")
    ap.add_argument("--execution", default="eager", choices=["eager", "contiguous", "compile_norm", "compile_pointwise"])
    ap.add_argument("--optimizer", default="fused", choices=["default", "fused"])
    args = ap.parse_args()
    torch.set_num_threads(4)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    rt = Runtime("bf16", "cuda")
    reference, _ = load_ema(args.checkpoint, rt.device)
    candidate = copy.deepcopy(reference)
    from pixel_accelerate import configure_execution
    configure_execution(candidate, args.execution, emulate_precision_casts=not args.relaxed_fusion)
    models = [reference.train().requires_grad_(True), candidate.train().requires_grad_(True)]
    opts = [torch.optim.AdamW(m.parameters(), lr=1e-4, betas=(.9, .95), eps=1e-8, weight_decay=0, **kw)
            for m, kw in zip(models, [{}, {} if args.optimizer == "default" else {"fused": True}])]
    array = np.load(args.data, mmap_mode="r")
    clean = torch.from_numpy(array[1000:1064].copy()).permute(0, 3, 1, 2).to(rt.device).float() / 127.5 - 1
    generator = torch.Generator(device=rt.device).manual_seed(8642)
    losses, gradients = [], []
    for _ in range(args.steps):
        t = (torch.randn(64, device=rt.device, generator=generator) * .8 - .8).sigmoid()
        noise = torch.randn(clean.shape, device=rt.device, generator=generator)
        pair = []
        for model, opt in zip(models, opts):
            opt.zero_grad(set_to_none=True)
            with rt.amp():
                loss = flow_loss(model, clean, t, noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            pair.append(float(loss.detach()))
        losses.append(pair)
        gradients.append(relative_error([p.grad for p in reference.parameters()], [p.grad for p in candidate.parameters()]))
    parameters = relative_error(list(reference.parameters()), list(candidate.parameters()))
    states = [sorted({str(v.dtype) for s in opt.state.values() for v in s.values() if torch.is_tensor(v)}) for opt in opts]
    loss_error = max(abs(a - b) / max(abs(a), 1e-12) for a, b in losses)
    gates = dict(parameter_relative_l2=1e-5, gradient_relative_l2=.005, loss_relative=.002)
    passed = (parameters["relative_l2"] < gates["parameter_relative_l2"] and
              max(g["relative_l2"] for g in gradients) < gates["gradient_relative_l2"] and
              loss_error < gates["loss_relative"] and all(s == ["torch.float32"] for s in states))
    result = dict(execution=args.execution, deterministic=args.deterministic, steps=args.steps, emulate_precision_casts=not args.relaxed_fusion, optimizer=args.optimizer, passed=passed, gates=gates, parameters=parameters, gradients=gradients,
                  losses=losses, max_loss_relative=loss_error, state_dtypes=states,
                  note="Five matched full-model BF16 updates with FP32 parameters/gradients/states; not a long-run convergence proof.")
    write_json(Path(args.output), result)
    print(json.dumps(result), flush=True)
    if not passed:
        raise RuntimeError("Candidate did not pass the predeclared numerical gates")


if __name__ == "__main__":
    main()
