"""Parallel candidate generation; each candidate has its own recorded seed.

Run under torchrun on an allocated CLUSTER node. API judging runs locally later;
this script does not read or transmit any API credential.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from PIL import Image
import torch
import torch.distributed as dist

from pixel_experiment import Runtime, write_json
from sample_pixel import load_ema, apply_autoguidance
from diffusion_demo.core.models.pixel_jit import heun_sample


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=120000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--guide-checkpoint")
    parser.add_argument("--guidance", type=float, default=3)
    parser.add_argument("--guidance-start", type=float, default=.1)
    parser.add_argument("--guidance-end", type=float, default=.8)
    args = parser.parse_args()
    if min(args.count, args.steps, args.batch) < 1:
        parser.error("count, steps and batch must be positive")
    rank, local_rank, world = [int(os.environ.get(k, d)) for k, d in [("RANK", 0), ("LOCAL_RANK", 0), ("WORLD_SIZE", 1)]]
    torch.set_num_threads(4)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        (output / "images").mkdir()
    if world > 1:
        dist.barrier()
    model, cfg = load_ema(args.checkpoint, device)
    model = apply_autoguidance(model, cfg, args.guide_checkpoint, args.guidance, device,
                              args.guidance_start, args.guidance_end)
    rt = Runtime("bf16", device)
    generator = torch.Generator(device=device)
    ids = list(range(rank, args.count, world))
    started = time.perf_counter()
    for offset in range(0, len(ids), args.batch):
        indices = ids[offset:offset + args.batch]
        noise = torch.stack([torch.randn(3, cfg["model"]["image_size"], cfg["model"]["image_size"],
                             device=device, generator=generator.manual_seed(args.seed + i)) for i in indices])
        with rt.amp():
            samples = heun_sample(model, noise, args.steps)
        if not samples.isfinite().all():
            raise FloatingPointError("Non-finite generated samples")
        arrays = ((samples.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        for i, array in zip(indices, arrays):
            Image.fromarray(array).save(output / "images" / f"{i:05d}.png")
    if world > 1:
        dist.barrier()
    if rank == 0:
        paths = sorted((output / "images").glob("*.png"))
        assert len(paths) == args.count
        metadata = dict(**vars(args), world_size=world, precision="bf16", seconds=time.perf_counter() - started,
                        config=cfg,
                        checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                        guide_sha256=hashlib.sha256(Path(args.guide_checkpoint).read_bytes()).hexdigest() if args.guide_checkpoint else None,
                        seed_rule="candidate i uses seed+i, independent of rank assignment",
                        candidates=[dict(id=i, seed=args.seed+i, file=f"images/{i:05d}.png",
                                         sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for i,path in enumerate(paths)])
        write_json(output / "metadata.json", metadata)
        print(json.dumps(dict(output=str(output), count=len(paths), seconds=metadata["seconds"])), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
