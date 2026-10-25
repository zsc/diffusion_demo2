"""Compare fixed, unfiltered samples and matched-budget full-data continuations."""
import json
from pathlib import Path

from PIL import Image, ImageDraw


root = Path(__file__).resolve().parents[1]
run = root / 'runs/pixel96_full2_20261025'
previews = run / 'previews'
status = json.loads((run / 'status.json').read_text())
quality = status['quality']
tags = ['control30k', 'full', 'full_lr1e4', 'full_balanced']
names = ['3 万张对照', '全量标准 LR 5e-5', '全量 LR 1e-4', '全量高噪声比例略增']
assert len(quality) == 28 and all(t['pipeline_complete'] for t in status['tasks'])
rows = []
for tag, name in zip(tags, names):
    means = [sum(quality[f'{tag}_{n}_g2_seed{seed}']['clean_fid'] for seed in [3333, 7777]) / 2
             for n in [20480000, 24576000, 32768000]]
    rows.append(dict(tag=tag, name=name, mean_fid_by_exposure=means,
                     final_fid_seeds=[quality[f'{tag}_32768000_g2_seed{s}']['clean_fid'] for s in [3333, 7777]],
                     mean_final_kid=sum(quality[f'{tag}_32768000_g2_seed{s}']['kid_mean'] for s in [3333, 7777]) / 2,
                     unguided_fid=quality[f'{tag}_32768000_g1_seed3333']['clean_fid']))


def contact_sheet(seed, guidance=2):
    canvas = Image.new('RGB', (1720, 40 + 212 * len(tags)), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f'Seed {seed} | guidance {guidance} | first 8 fixed samples | Heun50 | no selection | native96px, nearest-neighbor display', fill='black')
    for i, tag in enumerate(tags):
        y = 40 + i * 212
        draw.text((8, y + 80), tag, fill='black')
        path = previews / f'{tag}_32768000_g{guidance}_seed{seed}'
        q = json.loads((path / 'quality.json').read_text())
        assert q['seed'] == seed and q['guidance'] == guidance
        with Image.open(path / 'independent_seed_grid.png') as grid:
            assert grid.size == (2352, 2352)
            for c in range(8):
                canvas.paste(grid.crop((294*c, 0, 294*c+288, 288)).resize((192, 192), Image.Resampling.NEAREST),
                             (160+c*194, y))
    filename = f'pixel96_full2_comparison_g{guidance}_seed{seed}_20261025.png'
    canvas.save(root / 'reports' / filename)
    return filename


images = [contact_sheet(3333), contact_sheet(7777), contact_sheet(3333, 1)]
review = dict(completed_tasks=4, evaluations=28, latest_round_exposures_per_arm=16384000,
              exposures_since_common_ema=32768000, validation_images=1000, fake_images_per_eval=1024,
              best_of_n=False, winner='full_lr1e4', rows=rows,
              sample_policy='First eight sample indices, fixed seeds, no selection or image enhancement.',
              limitations='Sampling seeds are not independent training replicates; small-sample FID/KID screening only.')
(run / 'review.json').write_text(json.dumps(review, indent=2, ensure_ascii=False))
table = '<tr><th>配置</th><th>20.48M 平均 FID</th><th>24.576M 平均 FID</th><th>32.768M 平均 FID</th><th>最终无引导 FID</th><th>最终平均 KID</th></tr>'
paired = '<tr><th>配置</th><th>seed3333 FID</th><th>seed7777 FID</th></tr>'
for row in rows:
    table += '<tr><td>' + row['name'] + '</td>' + ''.join(f'<td>{v:.4f}</td>' for v in [*row['mean_fid_by_exposure'], row['unguided_fid'], row['mean_final_kid']]) + '</tr>'
    paired += '<tr><td>' + row['name'] + '</td>' + ''.join(f'<td>{v:.4f}</td>' for v in row['final_fid_seeds']) + '</tr>'
body = '''<!doctype html><meta charset="utf-8"><title>全量 CelebA 第二轮对比</title>
<style>body{max-width:1200px;margin:40px auto;font:16px/1.7 system-ui;color:#183047}table{border-collapse:collapse}td,th{padding:8px 12px;border:1px solid #ddd}img{width:100%}</style>
<h1>全量 CelebA 第二轮 · 2026-10-25</h1>
<p>四组训练及 28 次评估均成功完成，每组本轮新增 16.384M 曝光，从共同 EMA 初始化阶段起累计 32.768M 曝光。验证集仍为原来的 1000 张。3 万张对照与全量标准组各续接自身完整状态；两个变体从上一轮全量标准组完整状态出发，各只改一个参数。本轮没有重置优化器、EMA 或随机状态。</p>
<p><strong>学习率 1e-4 的全量组最好：</strong>guidance=2 的平均 FID 51.125，优于全量标准 51.604 与 3 万张对照 53.347；两个 seed 都胜出，平均 KID 和无引导 FID 也更好。上一轮全量最佳平均 FID 为 52.666。本轮标准组继续改善；高噪声比例组在中途 24.576M 的平均 FID 51.559 好于最终的 51.708，未显示继续延长就会改善。</p>
<table>''' + table + '</table><h2>最终两个固定 seed</h2><table>' + paired + '''</table>
<p>FID/KID 越低越好。每个评估生成 1024 张图，对照固定 1000 张验证图；guidance=2、Heun50、相同早期 guide。两个 seed 是采样复核，不是独立重复训练；结果用于筛选，不能据此宣称统计显著性。</p>
<h2>固定样本</h2><p>以下每行固定前 8 张，没有筛选或增强。行顺序：3 万张对照、全量标准、全量 LR 1e-4、全量高噪声比例略增。显示为原始 96px 图的最近邻放大。</p>
'''
for filename, caption in zip(images, ['guidance=2，seed3333', 'guidance=2，seed7777', '无引导，seed3333']):
    body += f'<h3>{caption}</h3><img src="{filename}">'
body += '<p><a href="../runs/pixel96_full2_20261025/previews/full_lr1e4_32768000_g2_seed3333/independent_seed_grid.png">优胜组固定前 64 张完整网格</a></p>'
(root / 'reports/pixel96_full2_20261025.html').write_text(body)
print(root / 'reports/pixel96_full2_20261025.html')
