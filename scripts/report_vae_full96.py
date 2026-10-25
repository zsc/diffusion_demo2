"""Ranked, reproducible report for matched CelebA latent DiT experiments."""
import html
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'runs/vae_full96_20261025'
OUTPUT = ROOT / 'reports/vae_full96_2026-10-25.html'
LABELS = {'full162k': '完整训练集 · 162,770 张', 'control30k': '等量对照 · 30,000 张'}


def esc(value):
    return html.escape(str(value), quote=True)


def relative(path):
    return '../' + path.relative_to(ROOT).as_posix()


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def collect(status):
    """Discover completed evaluations, including future arms in the same run."""
    rows = []
    groups = {}
    for tag, arm in status.items():
        if not isinstance(arm, dict) or 'task' not in arm:
            continue
        if arm['task']['state'] in {'FAILED', 'CANCELED', 'CANCELLED'}:
            continue
        for name, metrics in arm.get('scores', {}).items():
            if not all(math.isfinite(metrics[k]) for k in ['clean_fid', 'kid_mean']):
                continue
            path = R / tag / 'eval' / name
            meta = read(path / 'evaluation.json', {})
            seed = meta.get('seed', name.rsplit('_seed', 1)[-1])
            recipe = name.rsplit('_seed', 1)[0]
            guide = meta.get('guide')
            if guide:
                exposure = int(Path(guide).stem.rsplit('_', 1)[-1])
                method = f'{exposure / 1e6:.2f}M 弱模型 · 引导强度 {meta["guidance"]:g}'
            else:
                method = '无引导'
            # Prevent different sampling settings from silently sharing a mean.
            signature = (tag, recipe, meta.get('checkpoint'), guide,
                         meta.get('guidance'), meta.get('steps'), meta.get('count'),
                         metrics.get('real_count'), metrics.get('fid_solver'))
            row = dict(tag=tag, name=name, seed=seed, method=method, path=path,
                       metrics=metrics, meta=meta, arm=arm)
            rows.append(row)
            groups.setdefault(signature, []).append(row)
    protocols = {(x['metrics'].get('real_count'), x['metrics'].get('fake_count'),
                  x['metrics'].get('fid_solver')) for x in rows}
    if len(protocols) > 1:
        raise ValueError('Different evaluation protocols must have separate ranking tables')
    summaries = []
    for group in groups.values():
        group.sort(key=lambda x: int(x['seed']))
        summaries.append(dict(rows=group, fid=statistics.mean(x['metrics']['clean_fid'] for x in group),
                              kid=statistics.mean(x['metrics']['kid_mean'] for x in group)))
    summaries.sort(key=lambda x: (x['fid'], x['kid'], x['rows'][0]['tag']))
    rows.sort(key=lambda x: (x['metrics']['clean_fid'], x['metrics']['kid_mean'], x['tag'], str(x['seed'])))
    for rank, row in enumerate(rows, 1):
        row['anchor'] = f'eval-{rank}'
    return summaries, rows


def thumbnails(row):
    """Display fixed first four cells via CSS; leave source pixels untouched."""
    path = row['path'] / 'samples64.png'
    if not path.exists():
        return '<span class="muted">样例待回收</span>'
    cells = ''.join(f'<span class="face"><img src="{esc(relative(path))}" '
                    f'style="left:-{i * 100}%" alt="seed {row["seed"]}，固定样例 {i + 1}"></span>'
                    for i in range(4))
    return (f'<a class="faces" href="#{row["anchor"]}" aria-label="查看完整64张样本">{cells}</a>'
            f'<small>seed {row["seed"]} · 前 4 张 · <a href="#{row["anchor"]}">完整 64 张 ↗</a></small>')


STYLE = '''
:root{color-scheme:light;--ink:#172b35;--muted:#61727c;--line:#dce5e9;--accent:#12685e}
*{box-sizing:border-box}body{font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--ink);background:#f5f7f8;margin:0}
main{max-width:1440px;margin:auto;padding:32px 28px 70px}h1{font-size:30px;line-height:1.3;letter-spacing:-.6px;margin:8px 0 12px}h2{font-size:22px;margin:30px 0 12px}h3{font-size:17px;margin:0 0 6px}
p{margin:10px 0}a{color:var(--accent);text-underline-offset:3px}.eyebrow{font-size:12px;font-weight:700;color:var(--accent);letter-spacing:1.5px}.muted,small{color:var(--muted)}small{display:block;font-size:12px;line-height:1.7}.intro{max-width:1000px}.notice{border-left:3px solid var(--accent);padding:10px 16px;background:#e9f2ef;margin:18px 0}
nav{display:flex;gap:20px;font-size:13px;margin:16px 0 24px}.table-wrap{background:white;border:1px solid var(--line);border-radius:12px;overflow:auto}table{border-collapse:collapse;width:100%}th,td{padding:16px 14px;text-align:left;vertical-align:middle;border-bottom:1px solid var(--line)}th{background:#edf2f4;color:#50636e;font-size:12px;white-space:nowrap}tbody tr:last-child td{border-bottom:0}.summary{min-width:1120px}.rank{font-size:18px;font-weight:700;color:#778a93;width:58px}.first .rank{color:var(--accent)}.metric{font-size:23px;font-weight:700;font-variant-numeric:tabular-nums;line-height:1.3}.method{min-width:245px}.pill{display:inline-block;font-size:11px;font-weight:600;border-radius:4px;padding:1px 6px;background:#e8f1ed;color:var(--accent);margin-left:6px}.faces{display:flex;gap:4px;width:max-content;margin-bottom:6px}.face{position:relative;display:block;width:62px;height:62px;overflow:hidden;border-radius:5px;background:#ddd}.face img{position:absolute;top:0;width:800%;max-width:none;height:auto}.numeric{font-variant-numeric:tabular-nums;white-space:nowrap}.protocol{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card,.sample{padding:20px;background:white;border:1px solid var(--line);border-radius:12px}.sample{margin:18px 0;scroll-margin-top:18px}.sample img,.training img{display:block;width:100%;max-width:1000px;height:auto;margin:14px auto 0}.sample .caption{display:flex;justify-content:space-between;gap:20px}.sample .metric{font-size:20px}.training{margin:18px 0}.training summary{cursor:pointer;font-weight:600;padding:12px;background:#edf2f4;border-radius:8px}code{font-size:12px;overflow-wrap:anywhere}footer{margin-top:30px;font-size:12px;color:var(--muted)}
@media(max-width:700px){main{padding:22px 14px}h1{font-size:25px}.protocol{grid-template-columns:1fr}nav{gap:13px;flex-wrap:wrap}.sample .caption{display:block}}
'''


def main():
    if not (R / 'status.json').exists():
        return
    status = read(R / 'status.json')
    audit = read(R / 'data_audit.json')
    summaries, rows = collect(status)
    parts = ['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>完整训练集 · VAE latent DiT · 指标排名</title>', f'<style>{STYLE}</style></head><body><main>',
             '<div class="eyebrow">实验结果 / 96 × 96 / 2026-10-25</div>',
             '<h1>完整训练集 · VAE latent DiT</h1>',
             '<p class="intro">固定 16 通道 VAE，比较 30k 与官方完整 162,770 张训练集。两组从相同 16.78M 状态续训至 33.55M 次看图；仅改变 DiT 训练数据规模。</p>',
             '<nav><a href="#ranking">方法排名</a><a href="#details">单次评估排名</a><a href="#samples">完整样本</a><a href="#protocol">方法与数据审计</a></nav>',
             '<h2 id="ranking">方法排名</h2>',
             '<p class="muted">按同一方法的 FID 算术均值升序；并列时按 KID 升序。逐项列出种子和评估次数，单种子结果不等同于双种子验证。缩略图固定取 seed 3333 的前 4 张，无挑选、无增强。</p>',
             '<div class="table-wrap"><table class="summary"><thead><tr><th>排名</th><th>方法与训练数据</th><th>FID ↓ / 各种子</th><th>平均 KID ↓</th><th>训练速度</th><th>固定样例</th></tr></thead><tbody>']
    for rank, summary in enumerate(summaries, 1):
        group = summary['rows'];row = group[0];tag = row['tag']
        representative = next((x for x in group if x['seed'] == 3333), row)
        progress = row['arm'].get('train/progress.json', {})
        seeds = '<br>'.join(f'seed {x["seed"]}: {x["metrics"]["clean_fid"]:.3f}' for x in group)
        label = esc(LABELS.get(tag, tag))
        mainline = '<span class="pill">完整集主线</span>' if tag == 'full162k' else ''
        parts.append(f'<tr class="{"first" if rank == 1 else ""}"><td class="rank">{rank:02}</td>'
                     f'<td class="method"><strong>{label}</strong>{mainline}<small>{esc(row["method"])} · Heun {row["meta"].get("steps", "—")}</small>'
                     f'<small>冻结 VAE16 · 任务 {row["arm"]["task"]["task_id"]}</small></td>'
                     f'<td><div class="metric">{summary["fid"]:.3f}</div><small>{len(group)} 次评估<br>{seeds}</small></td>'
                     f'<td class="numeric">{summary["kid"]:.5f}</td>'
                     f'<td class="numeric">{progress.get("images_per_second", 0) / 1000:.2f}k<small>图像 / 秒 · 1×16 GPU</small></td>'
                     f'<td>{thumbnails(representative)}</td></tr>')
    parts.append('</tbody></table></div>')
    paired = {x['rows'][0]['tag']: x for x in summaries if len(x['rows']) == 2 and x['rows'][0]['name'].startswith('guide4m_g2_')}
    if {'full162k', 'control30k'} <= paired.keys():
        full, control = paired['full162k']['fid'], paired['control30k']['fid']
        parts.append(f'<p class="notice"><strong>结论：两组当前基本持平。</strong> 引导采样平均 FID 为完整集 {full:.3f}、30k 对照 {control:.3f}，差值仅 {full-control:+.3f}。数字排名不代表显著质量差异；本轮尚不能证明扩大数据集改善 FID。完整集保留为后续主线，VAE 本身仍是原 30k 训练版本。</p>')
    pending = [f'{LABELS.get(tag, tag)}（{arm["task"]["state"]}）' for tag, arm in status.items()
               if isinstance(arm, dict) and 'task' in arm and
               (not arm.get('scores') or arm['task']['state'] in {'FAILED', 'CANCELED', 'CANCELLED'})]
    if pending:
        parts.append('<p class="notice">未完成评估，不参与排名：' + esc('、'.join(pending)) + '</p>')
    parts.extend(['<h2 id="details">单次评估排名</h2>', '<p class="muted">下表及后续完整样本网格均按单次 FID 升序。方法总体结论以上方跨种子均值为准。</p>',
                  '<div class="table-wrap"><table><thead><tr><th>排名</th><th>训练数据</th><th>采样方法</th><th>种子</th><th>FID ↓</th><th>KID ↓</th><th>样本</th></tr></thead><tbody>'])
    for rank, row in enumerate(rows, 1):
        m = row['metrics']
        parts.append(f'<tr><td>{rank:02}</td><td>{esc(LABELS.get(row["tag"],row["tag"]))}</td><td>{esc(row["method"])}</td><td>{row["seed"]}</td><td class="numeric">{m["clean_fid"]:.3f}</td><td class="numeric">{m["kid_mean"]:.5f}</td><td><a href="#{row["anchor"]}">64 张 ↗</a></td></tr>')
    parts.extend(['</tbody></table></div>', '<h2 id="samples">完整样本 · 按 FID 排名</h2>', '<p class="muted">每次评估均展示连续前 64 张。整体已能辨认人脸，仍有局部五官变形、过曝与牙齿伪影。</p>'])
    for rank, row in enumerate(rows, 1):
        path = row['path'] / 'samples64.png'
        if path.exists():
            parts.append(f'<section class="sample" id="{row["anchor"]}"><div class="caption"><div><h3>#{rank} · {esc(LABELS.get(row["tag"],row["tag"]))}</h3><small>{esc(row["method"])} · seed {row["seed"]} · <a href="{esc(relative(path))}">打开原图 ↗</a></small></div><div class="metric">FID {row["metrics"]["clean_fid"]:.3f}<small>KID {row["metrics"]["kid_mean"]:.5f}</small></div></div><img loading="lazy" src="{esc(relative(path))}" alt="{esc(row["tag"])} {esc(row["name"])} 连续64张生成样本"></section>')
    parts.extend(['<h2 id="protocol">方法与数据审计</h2><div class="protocol"><section class="card"><h3>相同训练与评估条件</h3>',
                  '<p>两组均从 16,777,216 次曝光状态继承 DiT、EMA、优化器和随机状态，追加 16,777,216 次曝光。batch 2048、lr 4e-4、每组 1×16 GPU。冻结 12×12×16 VAE，不使用 PCA，逐步重新采样后验与扩散噪声。</p>',
                  '<p>每次评估 1,024 张生成图 / 同一组 1,000 张验证图，Heun50。引导使用最初 4,194,304 次曝光 EMA，强度 2，作用于 t∈[0.1,0.8]。保留 latent 归一化，确保主模型、引导和 decoder 坐标一致。</p>',
                  '<p class="muted">FID 是小样本筛选指标；验证集也用于 VAE 与采样方法选择。跨种子平均 KID 为各次 kid_mean 的均值，并非合并样本后重新计算。不同分辨率、验证集或指标实现不混排。</p></section><section class="card"><h3>官方划分与可复现记录</h3>',
                  f'<p>全部官方训练 ID 覆盖：{audit["training_exact_official_split"]}。训练/验证重叠 {audit["train_val_overlap"]}；训练/测试重叠 {audit["train_test_overlap"]}。原 1,000 张验证图与 30k 训练图的 ID、顺序、像素均保留。</p>',
                  '<p>数据切换改变采样分布，并非原训练轨迹的精确续接。显式扩容路径验证旧缓存前缀；新数据 SHA 采用流式计算。两组最终权重与冻结 VAE 已取回并校验。</p>',
                  f'<p><a href="{relative(R / "data_audit.json")}">数据审计</a> · <a href="{relative(R / "selected.json")}">主线采样配置</a> · <a href="{relative(R / "model_hashes.json")}">模型哈希</a> · <a href="{relative(R / "status.json")}">完整指标记录</a></p>',
                  '<p class="muted">历史参考：扩容前 16M 模型，4M 弱模型 / g2 / seed3333 FID 57.065；96px pixel JiT 曾达约 53.65。训练历史不同，不纳入本轮等量训练排名。</p></section></div>',
                  '<h2>训练过程预览</h2><p class="muted">以下中间检查点没有独立 FID，保留为过程记录，不参与质量排名。</p>'])
    tag_order = list(dict.fromkeys(x['rows'][0]['tag'] for x in summaries))
    for tag in tag_order:
        for path in sorted((R / tag / 'train').glob('samples*.png'), key=lambda p: int(p.stem.rsplit('_', 1)[-1])):
            parts.append(f'<details class="training"><summary>{esc(LABELS.get(tag,tag))} · {int(path.stem.rsplit("_",1)[-1]):,} 次曝光</summary><img loading="lazy" src="{esc(relative(path))}" alt="训练预览"></details>')
    parts.append('<footer>由 scripts/report_vae_full96.py 从原始指标自动生成；排名、均值和样本链接随新评估更新。</footer></main></body></html>')
    OUTPUT.write_text(''.join(parts))


if __name__ == '__main__':
    main()
