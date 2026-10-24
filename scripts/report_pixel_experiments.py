"""Create a self-contained report from actual pixel experiment artifacts."""
import argparse
import base64
import html
import io
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def embed(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def collect(directory, label, quality_directory=None):
    cfg = read_json(directory / "config.json")
    if not cfg:
        return None
    batch = cfg.get("global_batch", cfg.get("batch_size"))
    metrics = read_lines(directory / "metrics.jsonl")
    validation = read_lines(directory / "validation.jsonl")
    for row in validation:
        row.setdefault("images_seen", row["step"] * batch)
    quality_dir = quality_directory or directory / "quality"
    quality = read_json(quality_dir / "quality.json") or read_json(quality_dir / "feature_scores.json", {})
    completed = read_json(directory / "completed.json") or read_json(directory / "stopped.json", {})
    initial = max(0, metrics[0].get("images_seen", metrics[0]["step"] * batch) - cfg.get("log_every", 50) * batch) if metrics else 0
    return dict(label=label, directory=str(directory), config=cfg, metrics=metrics,
                validation=validation, quality=quality, initial_images=initial,
                final_images=completed.get("images_seen", validation[-1]["images_seen"] if validation else 0),
                environment=read_json(directory / "environment.json", {}),
                train_seconds=completed.get("elapsed_seconds", validation[-1].get("elapsed_seconds") if validation else None),
                preview=str(quality_dir / "independent_seed_grid.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", nargs="+", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--baseline-quality", required=True)
    ap.add_argument("--guidance", nargs="*", default=[])
    ap.add_argument("--recipe", default="runs/pixel_best/recipe.json")
    ap.add_argument("--benchmark", default="runs/pixel_best/speed_comparison.json")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    runs = [collect(Path(args.baseline), "single PPU / b256 / 10k", Path(args.baseline_quality))]
    for folder in args.rounds:
        p = Path(folder)
        runs += [collect(d, p.name + "/" + d.name) for d in sorted(p.iterdir()) if d.is_dir()]
    runs = [r for r in runs if r is not None]
    guided = []
    for folder in args.guidance:
        for p in sorted(Path(folder).glob("*/quality.json")):
            q = read_json(p)
            q.update(label=Path(folder).name + "/" + p.parent.name,
                     preview=str(p.parent / "independent_seed_grid.png"))
            guided.append(q)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    recipe = read_json(Path(args.recipe), {})
    benchmark = read_json(Path(args.benchmark), {})
    output.with_suffix(".json").write_text(json.dumps(dict(training=runs, guidance=guided,
                                                          recipe=recipe, benchmark=benchmark), indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for r in runs:
        val = r["validation"]
        if not val:
            continue
        for ax, key, scale in [(axes[0], "images_seen", 1e6), (axes[1], "elapsed_seconds", 60)]:
            if not all(key in v for v in val):
                continue
            ax.plot([v.get(key, 0) / scale for v in val], [v["ema_val_loss"] for v in val], ".-", label=r["label"])
    for ax in axes:
        ax.grid(alpha=.2)
        ax.set_ylabel("Fixed-noise held-out EMA velocity MSE")
    axes[0].set_xlabel("Cumulative training images (millions)")
    axes[1].set_xlabel("Training-loop minutes in this phase")
    axes[0].legend(fontsize=6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    rows, galleries = [], []
    def fmt(x, digits=3):
        return "—" if x is None else f"{x:,.{digits}f}"
    for r in runs:
        cfg, q, val = r["config"], r["quality"], r["validation"]
        last = val[-1] if val else {}
        speeds = [m["images_per_second"] for m in r["metrics"][1:] if "images_per_second" in m]
        fields = [html.escape(r["label"]), fmt(r["environment"].get("parameters", 0) / 1e6, 1),
                  str(cfg.get("global_batch", cfg.get("batch_size"))), f'{cfg["lr"]:g}',
                  fmt(r["initial_images"] / 1e6, 2) + " → " + fmt(r["final_images"] / 1e6, 2),
                  fmt(r["train_seconds"] / 60 if r["train_seconds"] is not None else None, 2), fmt(statistics.median(speeds) if speeds else None, 0),
                  fmt(last.get("ema_val_loss"), 5), fmt(q.get("clean_fid"), 2), fmt(q.get("kid_mean"), 5)]
        rows.append("<tr>" + "".join("<td>" + x + "</td>" for x in fields) + "</tr>")
        preview = Path(r["preview"])
        if not preview.exists():
            previews = sorted(Path(r["directory"]).glob("samples_*.png"))
            preview = previews[-1] if previews else preview
            note = "固定训练预览种子 2026；独立评估尚未返回。"
        else:
            note = "独立评估种子 3333 + 图像编号，前 64 张完整展示。"
        if preview.exists():
            galleries.append(f'<section><h2>{html.escape(r["label"])}</h2><p>{note}</p><img src="{embed(preview)}"></section>')
    guidance_rows = []
    for q in guided:
        interval = q.get("guidance_interval", [0, 1])
        fields = [html.escape(q["label"]), str(q.get("guidance", 1)), str(interval), str(q["seed"]),
                  fmt(q["clean_fid"], 2), fmt(q["kid_mean"], 5)]
        guidance_rows.append("<tr>" + "".join("<td>" + v + "</td>" for v in fields) + "</tr>")
        preview = Path(q["preview"])
        if preview.exists():
            galleries.append(f'<section><h2>{html.escape(q["label"])}</h2><p>固定独立噪声种子 {q["seed"]}；完整前 64 张。</p><img src="{embed(preview)}"></section>')
    guidance_table = ""
    if guided:
        guidance_table = '<h2>采样消融</h2><p><a href="https://arxiv.org/abs/2406.02507">Autoguidance</a> 使用本项目较弱的 checkpoint 引导预测；<a href="https://arxiv.org/abs/2404.07724">限制引导区间</a>用于检查强引导的条纹伪影。t=0 为噪声，t=1 为数据。强度 1 是原模型；启用引导会增加一次弱模型前向。指标与整张网格共同用于选择。</p><table><tr>' + "".join('<th>'+s+'</th>' for s in ["采样实验", "强度", "t 区间", "种子", "FID ↓", "KID ↓"]) + "</tr>" + "".join(guidance_rows) + "</table>"
    recommendation = ""
    if recipe:
        recommendation = '<section><h2>最终选用的方案</h2><p>' + html.escape(recipe["summary"]) + '</p>'
        for comparison in recipe.get("quality_comparisons", []):
            recommendation += (f'<p>种子 {comparison["seed"]}：未引导 FID {comparison["unguided_fid"]:.2f} → '
                               f'区间引导 {comparison["guided_fid"]:.2f}；{html.escape(comparison["purpose"])}。</p>')
        preview = Path(recipe["preview"])
        if preview.exists():
            recommendation += f'<img src="{embed(preview)}"><p>RTX 2080 SUPER，FP16，seed=2027；连续生成的前 16 张，未筛选。原生 48×48，仅最近邻放大展示。</p>'
        recommendation += '</section>'
    speed_section = '<h2>速度基准</h2>'
    if benchmark:
        speed_section += '<table><tr><th>相同工作负载</th><th>单 PPU</th><th>2080 SUPER</th><th>PPU 倍率</th></tr>'
        for row in benchmark["comparisons"]:
            speed_section += ('<tr><td>' + html.escape(row["label"]) + '</td>'
                              f'<td>{row["ppu"]:.3f} {html.escape(row["unit"])}</td>'
                              f'<td>{row["desk"]:.3f} {html.escape(row["unit"])}</td>'
                              f'<td>{row["speedup"]:.2f}×</td></tr>')
        speed_section += '</table><p>' + html.escape(benchmark["note"]) + '</p>'
    document = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Pixel JiT 人脸实验</title>
<style>body{max-width:1400px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#182b34;background:#fafafa}h1{font-size:30px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;border-bottom:1px solid #d5dfe2;text-align:right}td:first-child,th:first-child{text-align:left}img{max-width:100%;height:auto}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:24px}section{background:white;padding:18px;border:1px solid #e1e6e8;border-radius:10px}h2{font-size:17px}p{max-width:1100px}.scroll{overflow:auto}</style>
<h1>Pixel JiT：人脸生成与 16 PPU 实验</h1>
''' + recommendation + '''<p>48×48 RGB，从零训练；不使用预训练生成权重、VAE、超分或人脸修复。数据为修复后的 30,000 张 CelebA sprite 图片，固定随机划分 29,000 张训练 / 1,000 张验证；并非按人物身份划分。</p>
<p>每个分布式实验分配一台 16 PPU 节点。warmup、EMA 与检查点按看图量对齐。相同架构的首轮用于 batch / 学习率比较，后续按结果调整训练长度、学习率与容量。最终样本来自 EMA、50 步 Heun。</p>
<p>FID / KID 使用 clean Inception 特征，1,024 张独立噪声生成图与 1,000 张留出图。FID 采用等价的 float64 对称 PSD 求解。这些是小样本配对比较，不是 FID50k；不同训练预算的结果不能当作严格等算力排名。</p>
<div class="scroll"><table><thead><tr>''' + "".join(f"<th>{s}</th>" for s in ["实验", "参数 M", "batch", "lr", "看图量 M", "本阶段分钟", "稳态图/秒", "验证 MSE", "FID ↓", "KID ↓"]) + "</tr></thead><tbody>" + "".join(rows) + '''</tbody></table></div>
<p>分钟数为训练循环时间，包含循环内采样/保存，不包含进程启动和独立评估。续跑的看图量是累计值，分钟数只包含本阶段。吞吐是日志区间中位数，已排除首个区间。</p>
<img src="''' + plot + '">' + speed_section + guidance_table + '<div class="gallery">' + "".join(galleries) + "</div></html>"
    output.write_text(document)
    print(output.resolve())


if __name__ == "__main__":
    main()
