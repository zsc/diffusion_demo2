"""Generate count*N candidates, then select one per group with a vision model."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=64, help="Number of final selected images")
    parser.add_argument("--best-of-n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--guide-checkpoint")
    parser.add_argument("--guidance", type=float, default=1)
    parser.add_argument("--guidance-start", type=float, default=.1)
    parser.add_argument("--guidance-end", type=float, default=.8)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--judge-concurrency", type=int, default=4)
    args = parser.parse_args()
    if min(args.count, args.best_of_n) < 1:
        parser.error("count and best-of-n must be positive")
    root = Path(__file__).resolve().parent
    output = Path(args.output).resolve()
    candidates = output / "candidates"
    command = [sys.executable, str(root / "sample_pixel.py"), "--checkpoint", str(Path(args.checkpoint).resolve()),
               "--output", str(candidates), "--count", str(args.count * args.best_of_n),
               "--seed", str(args.seed), "--steps", str(args.steps), "--batch", str(args.batch),
               "--device", args.device, "--precision", args.precision, "--guidance", str(args.guidance),
               "--guidance-start", str(args.guidance_start), "--guidance-end", str(args.guidance_end)]
    if args.guide_checkpoint:
        command += ["--guide-checkpoint", str(Path(args.guide_checkpoint).resolve())]
    subprocess.run(command, check=True, cwd=root)
    ns = sorted({1, args.best_of_n, *[n for n in [2, 4, 8, 16] if n <= args.best_of_n]})
    subprocess.run([sys.executable, str(root / "scripts/rank_face_candidates.py"),
                    "--input", str(candidates / "images"), "--output", str(output / "ranking"),
                    "--group-size", str(args.best_of_n), "--ns", ",".join(map(str, ns)),
                    "--concurrency", str(args.judge_concurrency), "--env-file", args.env_file], check=True, cwd=root)


if __name__ == "__main__":
    main()
