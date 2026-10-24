"""Bounded single-PPU benchmarks and separate ASight/PyTorch captures."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "PYTHONFAULTHANDLER": "1", "TRITON_CACHE_DIR": str(out / "triton_cache")}
    versions = {}
    for package in ["torch", "triton", "flash_attn"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    (out / "versions.json").write_text(json.dumps(versions, indent=2))
    results = []
    def run(label, command, timeout=240):
        with (out / (label + ".log")).open("w") as log:
            p = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        results.append(dict(label=label, returncode=p.returncode, command=command))
        (out / "commands.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(results[-1]), flush=True)
        if p.returncode:
            raise RuntimeError(f"{label} failed; see its preserved log")
    def workload(label, optimizer, capture="none"):
        return [sys.executable, "-u", "scripts/profile_pixel_train.py", "--checkpoint", args.checkpoint,
                "--data", args.data, "--output", str(out / label), "--optimizer", optimizer, "--capture", capture]
    # Interleaved independent processes; profiled timings are never used here.
    for repeat in range(3):
        order = ["default", "fused"] if repeat % 2 == 0 else ["fused", "default"]
        for opt in order:
            label = f"bench_{repeat}_{opt}"
            run(label, workload(label, opt))
    asys = "/usr/local/PPU_SDK/asight/bin/asys"
    for opt in ["default", "fused"]:
        label = "asys_" + opt
        prefix = str(out / label)
        run(label, [asys, "profile", "--trace", "hggc,hgtx,acblas", "--sample", "none",
                    "--capture-range", "hggcProfilerApi", "--capture-range-end", "stop",
                    "--max-report-size", "128", "--output", prefix] + workload(label + "_workload", opt, "asys"))
        run(label + "_export", [asys, "export", "--output", prefix + ".sqlite", prefix + ".asysrep"])
    run("torch_default", workload("torch_default", "default", "torch"))
    (out / "completed.json").write_text(json.dumps(dict(success=True, commands=len(results))))


if __name__ == "__main__":
    main()
