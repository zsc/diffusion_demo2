"""Build a traceable 96px cache from original aligned CelebA JPEGs, not sprites."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--train-count", type=int, default=30000)
    ap.add_argument("--val-count", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".json").exists():
        raise ValueError("Refusing to overwrite an existing dataset")
    started = time.perf_counter()
    with zipfile.ZipFile(args.zip) as z:
        members = {Path(n).name: n for n in z.namelist() if n.lower().endswith(".jpg")}
        rows = list(csv.DictReader(io.StringIO(z.read("list_eval_partition.csv").decode())))
        rng = np.random.default_rng(96)
        selected = {}
        for label, partition, count in [("validation", "1", args.val_count), ("train", "0", args.train_count)]:
            pool = sorted(r["image_id"] for r in rows if r["partition"] == partition)
            if count > len(pool):
                raise ValueError("Requested split exceeds available images")
            selected[label] = sorted(rng.choice(pool, count, replace=False).tolist())
        names = selected["validation"] + selected["train"]
        array = np.lib.format.open_memmap(output.with_suffix(".tmp.npy"), mode="w+", dtype=np.uint8,
                                          shape=(len(names), 96, 96, 3))
        source_hash = hashlib.sha256()
        def decode(raw):
            with Image.open(io.BytesIO(raw)) as image:
                if image.size != (178, 218):
                    raise ValueError(f"Unexpected source image size {image.size}")
                return np.array(image.convert("RGB").crop((0, 20, 178, 198)).resize((96, 96), Image.Resampling.LANCZOS))
        with ThreadPoolExecutor(args.workers) as pool:
            for start in range(0, len(names), 512):
                chunk = names[start:start + 512]
                raw = [z.read(members[n]) for n in chunk]
                for name, data in zip(chunk, raw):
                    source_hash.update(name.encode() + b"\0" + data)
                array[start:start + len(chunk)] = np.stack(list(pool.map(decode, raw)))
                if start % 5120 == 0:
                    print(json.dumps(dict(images=start + len(chunk), total=len(names))), flush=True)
        array.flush()
        data_sha = hashlib.sha256(array.tobytes()).hexdigest()
        # First 16 true high-resolution targets for visual QA.
        grid = Image.new("RGB", (4 * 98, 4 * 98), (32, 32, 32))
        for i, im in enumerate(array[:16]):
            grid.paste(Image.fromarray(im), ((i % 4) * 98, (i // 4) * 98))
        grid.resize((784, 784), Image.Resampling.NEAREST).save(output.with_suffix(".png"))
        del array
        output.with_suffix(".tmp.npy").replace(output)
    metadata = dict(source_zip=str(args.zip), original_size=[178, 218], crop=[0, 20, 178, 198],
                    resolution=96, resize="Pillow LANCZOS", upsampled=False,
                    source_selected_sha256=source_hash.hexdigest(), array_sha256=data_sha,
                    split_mode="validation_first", selection_seed=96, images=selected,
                    elapsed_seconds=time.perf_counter() - started,
                    note="Official CelebA train/val partitions. Old 48px sprites lack source IDs; overlap with their pretraining set is unknown.")
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metadata.items() if k != "images"}), flush=True)


if __name__ == "__main__":
    main()
