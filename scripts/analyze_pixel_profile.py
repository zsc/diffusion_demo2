"""Read raw ASight kernel names, preserve unknown costs, and summarize timing."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import statistics


def category(name):
    low = name.lower()
    if low.startswith("triton_"):
        return "fused Triton"
    if "gemm" in low or "acblas" in low:
        return "GEMM"
    if "flash" in low or "attention" in low:
        return "attention"
    if "conv" in low or "im2col" in low:
        return "convolution"
    if "adam" in low:
        return "AdamW"
    if "multi_tensor" in low:
        return "foreach (optimizer/EMA/clip; unresolved)"
    if "copy_kernel" in low or "cast_kernel" in low:
        return "copy/cast"
    if "elementwise" in low or "reduce_kernel" in low or "catarray" in low:
        return "pointwise/reduction/layout"
    if "norm" in low:
        return "normalization"
    return "other (inspect names/torch attribution)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    args = ap.parse_args()
    root = Path(args.directory)
    result = {"benchmarks": {}, "profiles": []}
    benchmark_groups = defaultdict(list)
    for p in sorted(root.glob("bench_*/result.json")):
        row = json.loads(p.read_text())
        label = row.get("execution", row["optimizer"])
        if "execution" in row and row["optimizer"] != "default":
            label += "_" + row["optimizer"]
        benchmark_groups[label].append(row)
    for label, rows in benchmark_groups.items():
        result["benchmarks"][label] = dict(processes=len(rows),
                                          median_ms=statistics.median(r["median_step_seconds"] for r in rows) * 1000,
                                          process_ms=[r["median_step_seconds"] * 1000 for r in rows],
                                          all_finite=all(r["loss_finite"] for r in rows))
    for path in sorted(root.glob("*.sqlite")):
        with sqlite3.connect(path) as con:
            con.row_factory = sqlite3.Row
            rows = [dict(r) for r in con.execute("SELECT k.*,s.value AS name FROM HGPTI_ACTIVITY_KIND_KERNEL k "
                                                "JOIN StringIds s ON k.demangledName=s.id ORDER BY k.start")]
        if not rows:
            raise ValueError(f"No kernels captured in {path}")
        groups, names, intervals = defaultdict(float), defaultdict(lambda: [0, 0.]), []
        for row in rows:
            ms = (row["end"] - row["start"]) / 1e6
            groups[category(row["name"])] += ms
            names[row["name"]][0] += 1
            names[row["name"]][1] += ms
            if not intervals or row["start"] > intervals[-1][1]:
                intervals.append([row["start"], row["end"]])
            else:
                intervals[-1][1] = max(intervals[-1][1], row["end"])
        span = (max(r["end"] for r in rows) - min(r["start"] for r in rows)) / 1e6
        union = sum(b - a for a, b in intervals) / 1e6
        result["profiles"].append(dict(name=path.stem, kernel_count=len(rows), span_ms=span, kernel_union_ms=union,
                                       kernel_sum_ms=sum(groups.values()), coverage=union / span,
                                       categories_ms=dict(sorted(groups.items(), key=lambda kv: -kv[1])),
                                       kernels=[dict(name=n, calls=v[0], total_ms=v[1]) for n, v in sorted(names.items(), key=lambda kv: -kv[1][1])]))
    result["note"] = "Name buckets are descriptive; foreach is not exclusively AdamW here. Coverage is timeline coverage, not occupancy or compute utilization. Formal timings exclude profilers."
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"benchmarks": result["benchmarks"], "profiles": [{k: v for k, v in p.items() if k != "kernels"} for p in result["profiles"]]}, indent=2))


if __name__ == "__main__":
    main()
