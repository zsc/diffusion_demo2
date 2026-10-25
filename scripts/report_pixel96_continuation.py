"""A reproducible report of longer 96px training and fixed-recipe comparisons."""
import argparse
import html
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from report_pixel_experiments import collect, embed, read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/gpu96")
    parser.add_argument("--best", default="runs/pixel96_further")
    parser.add_argument("--output", default="reports/pixel96_continuation_2026-10-24.html")
    args = parser.parse_args()
    root, best, output = Path(args.root), Path(args.best), Path(args.output)
    directories = [root / "round96_8m/scaled_lr2e4"]
    for folder in ["round96_16m", "round96_24m", "round96_32m"]:
        directories.extend(sorted((root / folder).glob("*")))
    runs = [collect(p, str(p.relative_to(root))) for p in directories if (p / "config.json").exists()]
    quality = []
    for folder in ["round96_final_sampling", "round96_continuation_eval"]:
        for p in sorted((root / folder).glob("*/quality.json")):
            quality.append({"label": folder + "/" + p.parent.name, **read_json(p)})
    recipe = read_json(best / "recipe.json", {})
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    table = []
    for run in runs:
        v = run["validation"]
        speed = statistics.median([m["images_per_second"] for m in run["metrics"][1:]]) if len(run["metrics"]) > 1 else 0
        continuation = read_json(Path(run["directory"]) / "continuation.json", {})
        run["continuation"] = continuation
        trajectory = v
        if continuation:
            parent_label = "/".join(Path(continuation["checkpoint"]).parts[-3:-1])
            parents = [x for x in runs if x["label"] == parent_label and x["validation"]]
            if parents:
                trajectory = [parents[0]["validation"][-1], *v]
        if trajectory:
            ax.plot([x["images_seen"] / 1e6 for x in trajectory], [x["ema_val_loss"] for x in trajectory], ".-", label=run["label"])
        initial = continuation.get("images_seen", run["initial_images"])
        fid = run["quality"].get("clean_fid")
        fields = [run["label"], f'{run["config"]["lr"]:g}',
                  f'{initial/1e6:.3f} → {run["final_images"]/1e6:.3f}',
                  f'{run["train_seconds"]/60:.2f}' if run["train_seconds"] else "—",
                  f'{speed:.0f}', f'{v[-1]["ema_val_loss"]:.6f}' if v else "—",
                  f'{fid:.3f}' if fid is not None else "—"]
        table.append("<tr>" + "".join(f"<td>{html.escape(s)}</td>" for s in fields) + "</tr>")
    ax.set(xlabel="Cumulative 96px training images (millions)", ylabel="Fixed-noise validation EMA MSE")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot = output.with_suffix(".png")
    fig.savefig(plot, dpi=150)
    plt.close(fig)
    qrows = []
    for q in quality:
        fields = [q["label"], str(q["seed"]), str(q["guidance"]), f'{q["clean_fid"]:.3f}', f'{q["kid_mean"]:.5f}']
        qrows.append("<tr>" + "".join(f"<td>{html.escape(s)}</td>" for s in fields) + "</tr>")
    gallery = []
    for title, path in [("8.192M baseline", Path("runs/pixel96_best/sample_gpu_seed2027/preview16.png")),
                        ("Longer training", best / "sample_gpu_seed2027/preview16.png")]:
        if path.exists():
            gallery.append(f'<section><h2>{title}</h2><img src="{embed(path)}"></section>')
    document = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>96px 充分续训</title>
<style>body{max-width:1300px;margin:36px auto;padding:0 24px;font:16px/1.6 system-ui;color:#182b34}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;text-align:right;border-bottom:1px solid #ddd}td:first-child,th:first-child{text-align:left}img{max-width:100%;height:auto}.gallery{display:grid;grid-template-columns:1fr 1fr;gap:20px}.scroll{overflow:auto}</style>
<h1>96×96：从 819.2 万张次继续训练</h1><p>''' + html.escape(recipe.get("summary", "续训进行中")) + '''</p>
<p>两组从同一个 8.192M 完整 checkpoint 出发，每组 16 GPU、global batch 1024。继承模型、EMA、AdamW 动量、随机数状态和累计步数；比较保持 2e-4 与降低到 1e-4。两组继续到 24.576M；另从 16.384M 的 1e-4 分支降到 5e-5，比较低学习率收尾。融合仅作用于训练中的逐点运算，采样与 EMA 使用原计算路径。阶段路径和学习率变化保存在 continuation.json。</p>
<p>所有训练图均来自固定 30,000 张官方训练分区图片；图像张次会重复计数，不代表独立图片数。验证为固定 1,000 张官方验证分区。旧 48px 预训练图片缺少原始 ID，无法排除与此验证集重叠。</p>
<div class="scroll"><table><tr><th>实验</th><th>学习率</th><th>96px 张次 M</th><th>阶段分钟</th><th>图/秒中位数</th><th>验证 MSE</th><th>raw FID</th></tr>''' + "".join(table) + '</table></div><img src="' + embed(plot) + '''">
<h2>固定采样参数的质量对照</h2><p>Heun 50 步，1,024 张生成图对 1,000 张验证图，小样本 FID/KID 越低越好。guidance 使用同一早期 256k guide，区间 [0.1,0.8]；seed3333 用于比较和选择，seed7777 用于确认。不同训练长度有不同算力预算。</p>
<div class="scroll"><table><tr><th>评估</th><th>seed</th><th>guidance</th><th>FID</th><th>KID</th></tr>''' + "".join(qrows) + '''</table></div>
<p>下图均为 seed2027 连续生成的前 16 张，原生 96×96，无筛选、修复或超分。</p><div class="gallery">''' + "".join(gallery) + '</div></html>'
    output.write_text(document)
    output.with_suffix(".json").write_text(json.dumps(dict(training=runs, quality=quality, recipe=recipe), indent=2, ensure_ascii=False) + "\n")
    print(output.resolve())


if __name__ == "__main__":
    main()
