"""Report generated candidates, selections and judge limitations without a server."""
import argparse
import base64
import html
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/pixel96_bestofn_20261025")
    parser.add_argument("--output", default="reports/pixel96_bestofn_2026-10-25.html")
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output)
    sections = []
    records = []
    for path in sorted(root.glob("*/selection.json")):
        value = json.loads(path.read_text()); records.append(dict(run=path.parent.name, **value))
        scores = json.loads((path.parent / "scores.json").read_text())
        selections = value["selections"]
        rows = ''.join(f'<tr><td>{n}</td><td>{q["outputs"]}</td><td>{q["mean_judge_score"]:.2f}</td></tr>'
                       for n,q in selections.items())
        pictures = []
        for n in [min(map(int, selections)), max(map(int, selections))]:
            preview = path.parent / f'best_of_{n}/preview.png'
            encoded = base64.b64encode(preview.read_bytes()).decode()
            pictures.append(f'<div><h3>N={n}</h3><img src="data:image/png;base64,{encoded}"></div>')
        chosen = selections[str(max(map(int, selections)))]["candidate_ids"]
        issues = ''.join(f'<tr><td>{i}</td><td>{scores[i]["score"]:.1f}</td><td>{html.escape(scores[i]["issues"])}</td></tr>' for i in chosen)
        sections.append(f'<section><h2>{html.escape(path.parent.name)}</h2><p>{len(scores)} 张候选；总 token {value["usage_all_candidates"]["total_tokens"]:,}。原始 PNG 全部保留。</p><table><tr><th>N</th><th>最终张数</th><th>视觉模型评分均值 /100</th></tr>{rows}</table><div class="pair">'+''.join(pictures)+f'</div><details><summary>入选图片与评分理由</summary><table>{issues}</table></details></section>')
    status_path = root / "cluster_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    document = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>96px Best-of-N</title>
<style>body{max-width:1280px;margin:32px auto;padding:0 24px;font:16px/1.6 system-ui;color:#182b34;background:#fafafa}section{background:white;padding:20px;margin:20px 0;border:1px solid #ddd;border-radius:10px}table{border-collapse:collapse}td,th{padding:6px 16px;border-bottom:1px solid #ddd;text-align:left}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}img{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}h1{font-size:28px}</style>
<h1>原生 96×96 人脸：视觉模型 Best-of-N</h1>
<p>随机性来自起始高斯噪声；Heun 积分中没有再注入噪声。固定模型、seed、采样设置可基本复现，但不同硬件的浮点计算可能有差别。当前是无条件模型，因此不同候选可能是完全不同的人脸。</p>
<p>固定每 8 张为一组，N=1/2/4/8 使用每组前 N 张中的最高分。评分权重：五官结构 40%、自然程度 30%、细节 20%、整体一致性 10%。不评价颜值、身份或人口属性。相同分数取最早候选，保留全量图片、SHA256、提示词、原始响应和 token 用量。</p>
<p><strong>筛选后同一评分器的均分上涨是选择机制的结果，不是独立质量证明。</strong>筛选会改变分布与多样性，旧模型未筛选的 FID 不能当作 best-of-N 的 FID。图中是筛选结果，仍可能有五官伪影。评分器通过兼容视觉 API 的端点调用；密钥只在本机读取。</p>
''' + ''.join(sections) + '<section><h2>CLUSTER 采样变体状态</h2><pre>' + html.escape(json.dumps(status, ensure_ascii=False, indent=2)) + '</pre></section></html>'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)
    output.with_suffix('.json').write_text(json.dumps(dict(runs=records, cluster=status),ensure_ascii=False,indent=2)+'\n')
    print(output.resolve())


if __name__ == '__main__':
    main()
