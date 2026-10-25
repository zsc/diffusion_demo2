"""Package one explicitly selected long-training checkpoint and native samples."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--training", required=True, help="Completed training directory relative to root")
    parser.add_argument("--output", default="pixel96_further")
    parser.add_argument("--guidance", type=float, default=3)
    parser.add_argument("--ema-images", type=int, help="Select an earlier immutable EMA if it outperforms the final step")
    args = parser.parse_args()
    root = Path(args.root)
    training, target = root / args.training, root / args.output
    completed = json.loads((training / "completed.json").read_text())
    images = args.ema_images or completed["images_seen"]
    checkpoint = training / f"ema_{images:09d}.pt"
    guide = root / "round96/scaled_lr2e4/ema_000256000.pt"
    target.mkdir(parents=True, exist_ok=True)
    if (target / "completed.json").exists():
        raise ValueError("Selection is already packaged; choose a new output directory")
    env = {**os.environ, "PYTHONPATH": f"src:{root}/eval_assets/packages"}
    command = [sys.executable, "sample_pixel.py", "--checkpoint", str(checkpoint),
               "--output", str(target / "sample_gpu_seed2027"), "--seed", "2027",
               "--count", "64", "--steps", "50", "--batch", "16"]
    if args.guidance > 1:
        command += ["--guide-checkpoint", str(guide), "--guidance", str(args.guidance),
                    "--guidance-start", "0.1", "--guidance-end", "0.8"]
    subprocess.run(command, env=env, check=True, timeout=180)
    sources = {"model_ema.pt": checkpoint, "last.pt": training / "last.pt"}
    if args.guidance > 1:
        sources["guide_ema.pt"] = guide
    files = {}
    for name, source in sources.items():
        shutil.copy2(source, target / name)
        with (target / name).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        files[name] = dict(source=str(source), sha256=digest, bytes=(target / name).stat().st_size)
    recipe = dict(images_seen=images, latest_resume_images=completed["images_seen"], training=str(training), guidance=args.guidance,
                  guidance_interval=[.1, .8], steps=50, sample_seed=2027, sample_count=64,
                  resolution=96, files=files, training_config=json.loads((training / "config.json").read_text()),
                  continuation=json.loads((training / "continuation.json").read_text()),
                  note="Native 96px samples, no face repair, super-resolution or sample filtering. Old 8.192M baseline remains separate. model_ema.pt is the selected EMA; last.pt is the latest full resume state and may be later than the selected EMA.")
    selection_file = root / "round96_continuation_eval/selection.json"
    if selection_file.exists():
        recipe["selection"] = json.loads(selection_file.read_text())
        shutil.copy2(selection_file, target / "selection.json")
    recipe["confirmation"] = [json.loads(p.read_text()) for p in sorted(
        (root / "round96_continuation_eval").glob("*seed7777/quality.json"))]
    recipe["baseline_confirmation"] = json.loads(
        (root / "round96_final_sampling/selected_seed7777/quality.json").read_text())
    (target / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    (target / "completed.json").write_text(json.dumps(dict(success=True, images_seen=images)))
    print(json.dumps(dict(output=str(target), images_seen=images, guidance=args.guidance)), flush=True)


if __name__ == "__main__":
    main()
