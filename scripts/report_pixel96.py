"""Summarize the 48-to-96 transfer experiment from saved measurements and grids."""
import argparse
import html
import json
from pathlib import Path

from report_pixel_experiments import collect, embed, read_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/ppu96")
    ap.add_argument("--output", default="reports/pixel96_transfer_2026-10-24.html")
    ap.add_argument("--best", default="runs/pixel96_best")
    args = ap.parse_args()
    root, best, output = Path(args.root), Path(args.best), Path(args.output)
    runs = [collect(p, str(p.relative_to(root))) for p in sorted(root.glob("round96*/*")) if p.is_dir()]
    runs = [r for r in runs if r]
    data = read_json(root / "data96/celeba96.json", {})
    data = {k: v for k, v in data.items() if k != "images"}
    recipe = read_json(best / "recipe.json", {})
    rows, gallery = [], []
    def fmt(x, digits=3):
        return "—" if x is None else f"{x:,.{digits}f}"
    for run in runs:
        directory, cfg, q = Path(run["directory"]), run["config"], run["quality"]
        init = read_json(directory / "initialization.json")
        run["initialization"] = init
        fields = [html.escape(run["label"]), "48px EMA" if init else "random",
                  str(cfg["model"].get("position_scale", 1)), str(cfg["lr"]),
                  fmt(run["final_images"] / 1e6, 3), fmt(run["train_seconds"] / 60 if run["train_seconds"] else None, 2),
                  fmt(run["validation"][-1]["ema_val_loss"] if run["validation"] else None, 5),
                  fmt(q.get("clean_fid"), 2), fmt(q.get("kid_mean"), 5)]
        rows.append("<tr>" + "".join("<td>" + x + "</td>" for x in fields) + "</tr>")
        preview = Path(run["preview"])
        if not preview.exists():
            previews = sorted(directory.glob("samples_*.png"))
            preview = previews[-1] if previews else preview
        if preview.exists():
            gallery.append(f'<section><h2>{html.escape(run["label"])}</h2><img src="{embed(preview)}"></section>')
    sampling = []
    for folder in ["round96_guidance", "round96_final_sampling"]:
        for p in sorted((root/folder).glob("*/quality.json")):
            q = read_json(p, {})
            q.update(label=folder+"/"+p.parent.name)
            sampling.append(q)
    sampling_rows = "".join("<tr>"+"".join("<td>"+x+"</td>" for x in [html.escape(q["label"]),str(q["seed"]),str(q["guidance"]),fmt(q["clean_fid"],3),fmt(q["kid_mean"],5)])+"</tr>" for q in sampling)
    sampling_html = '<section><h2>采样对照与独立 seed 确认</h2><p>guidance=3 是根据 seed3333 筛选，再用 seed7777 独立确认；最终展示 seed2027 的连续前16张。256k guide 是原生96px早期 EMA。小样本 FID 仅用于本轮比较。</p><div class="scroll"><table><tr><th>方案</th><th>seed</th><th>guidance</th><th>FID</th><th>KID</th></tr>'+sampling_rows+'</table></div></section>' if sampling else ""
    recommendation = ""
    if recipe:
        recommendation += '<section><h2>选用方案</h2><p>' + html.escape(recipe["summary"]) + '</p>'
        preview = Path(recipe["preview"])
        if preview.exists():
            recommendation += '<img src="' + embed(preview) + '">'
        recommendation += '<p>原生 96×96，连续生成的前 16 张；无超分、人脸修复或筛选。</p></section>'
    source = root / "data96/celeba96.png"
    source_html = f'<section><h2>真实 96px 目标数据</h2><img src="{embed(source)}"></section>' if source.exists() else ""
    document = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>48 → 96 JiT 权重迁移</title>
<style>body{max-width:1400px;margin:40px auto;padding:0 24px;background:#fafafa;color:#182b34;font:16px/1.6 system-ui}h1{font-size:30px}h2{font-size:18px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:10px;border-bottom:1px solid #ccd5db;text-align:right}td:first-child,th:first-child{text-align:left}img{max-width:100%;height:auto}section{background:white;padding:20px;border:1px solid #ddd;border-radius:10px;margin:18px 0}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:24px}.scroll{overflow:auto}</style>
<h1>48×48 → 96×96：继承全部学习参数</h1>''' + recommendation + '''
<p>保持 patch=4、width=384、depth=8、heads=6，21.89M 参数全部从已训练的 48px 主模型 EMA 严格加载。输入投影、Transformer、输出投影、卷积细化头均不随机重置；AdamW、学习率 warmup 和 EMA 年龄重新开始。精确续跑仍使用 --resume，跨分辨率初始化使用 --init-from。</p>
<p>网格从 12×12 变为 24×24，token 从 144 增到 576。绝对 sin/cos 位置编码和 RoPE 是非持久化解析 buffer，重新生成而非从 checkpoint 加载。position_scale=1 使用新网格原生坐标；0.5 使用原来一半的坐标步长。全局注意力矩阵元素数变为 16 倍，整体训练耗时并非必然 16 倍。</p>
<p>数据从原始 178×218 CelebA JPEG 中心裁切 178×178，再用 Lanczos 缩至 96×96，没有放大旧 48px 图片。固定抽取官方训练分区 30,000 张、验证分区 1,000 张。旧 sprites 缺少原图 ID，无法确认它们与新验证图的重叠；此处是 96px 微调留出集评估，并非保证未用于 48px 预训练的独立测试。</p>
<p>每组一台 16 PPU。首轮相同的新增 96px 看图预算，比较位置坐标、微调学习率及随机初始化；继承组额外使用已完成的 819.2 万张次 48px 预训练，因此不是同总训练算力比较。FID/KID 是 1,024 张生成图对 1,000 张 96px 验证图的小样本比较，不与旧 48px FID 横向比较；时间包含训练循环内保存和预览，续跑只计本阶段。</p>
<div class="scroll"><table><tr>''' + "".join('<th>' + s + '</th>' for s in ["实验", "初始化", "位置缩放", "lr", "累计看图 M", "阶段分钟", "验证 MSE", "FID ↓", "KID ↓"]) + '</tr>' + ''.join(rows) + '</table></div>' + sampling_html + source_html + '<div class="gallery">' + ''.join(gallery) + '</div></html>'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)
    output.with_suffix(".json").write_text(json.dumps(dict(data=data, training=runs, sampling=sampling, recipe=recipe), ensure_ascii=False, indent=2) + "\n")
    print(output.resolve())


if __name__ == "__main__":
    main()
