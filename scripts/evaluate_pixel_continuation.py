"""Evaluate a fixed manifest of continuation checkpoints on one 16-PPU node.

Manifest rows specify checkpoint, label, seed and guidance. It is deliberately
separate from training so intermediate EMA files can be compared without
pausing training or reading the mutable last.pt checkpoint.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output)
    rows = json.loads(Path(args.manifest).read_text())
    output.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": f"src:{root}/eval_assets/packages",
           "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
    for key in ["WORLD_SIZE", "RANK", "LOCAL_RANK"]:
        env.pop(key, None)
    results = []
    for row in rows:
        checkpoint = root / row["checkpoint"]
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        destination = output / row["label"]
        destination.mkdir(parents=True, exist_ok=True)
        quality = destination / "quality.json"
        scale, seed = row.get("guidance", 1), row.get("seed", 3333)
        if quality.exists():
            q = json.loads(quality.read_text())
            if (q["checkpoint"], q["seed"], q["guidance"]) != (str(checkpoint), seed, scale):
                raise ValueError(f"Existing evaluation has a different recipe: {quality}")
        else:
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                       "--nnodes=1", "--nproc_per_node=16", "evaluate_pixel.py",
                       "--checkpoint", str(checkpoint), "--data", str(root / "data96/celeba96.npy"),
                       "--weights-dir", str(root / "eval_assets"), "--output", str(destination),
                       "--seed", str(seed)]
            if scale > 1:
                command += ["--guide-checkpoint", str(root / "round96/scaled_lr2e4/ema_000256000.pt"),
                            "--guidance", str(scale), "--guidance-start", "0.1", "--guidance-end", "0.8"]
            with (destination / "eval.log").open("w") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=300)
            q = json.loads(quality.read_text())
        results.append({"label": row["label"], **q})
        print(json.dumps({k: results[-1][k] for k in ["label", "seed", "guidance", "clean_fid", "kid_mean"]}), flush=True)
        (output / (Path(args.manifest).stem + "_results.json")).write_text(json.dumps(results, indent=2) + "\n")
    (output / (Path(args.manifest).stem + "_completed.json")).write_text(json.dumps({"evaluations": len(results)}))


if __name__ == "__main__":
    main()
