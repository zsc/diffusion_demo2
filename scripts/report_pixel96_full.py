"""Render the full-training-set experiment plan and latest observed status."""
import html
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
directory = root / 'runs/pixel96_full_20261025'
plan = json.loads((directory / 'plan.json').read_text())
submission = json.loads((directory / 'submission.json').read_text())
status_path = directory / 'status.json'
status = json.loads(status_path.read_text()) if status_path.exists() else submission
quality = status.get('quality', {})
tasks = {t['tag']: t for t in status['tasks']}
table = '<tr><th>实验</th><th>训练图数</th><th>Batch</th><th>学习率</th><th>任务</th><th>状态</th><th>新增曝光</th><th>images/s</th></tr>'
for tag, arm in plan['arms'].items():
    task, config = tasks[tag], arm['config']
    metrics = task.get('metrics', [])
    latest = metrics[-1] if metrics else {}
    values = [tag, f"{arm['training_count']:,}", config['global_batch'], config['lr'],
              task['task_id'], task['state'], f"{latest.get('images_seen', 0):,}",
              round(latest['images_per_second']) if latest else '尚无指标']
    table += '<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in values) + '</tr>'
body = '''<!doctype html><meta charset="utf-8"><title>全量 CelebA 续训对照</title>
<style>body{max-width:1150px;margin:40px auto;font:16px/1.7 system-ui;color:#183047}td,th{border:1px solid #ddd;padding:8px 12px}table{border-collapse:collapse}code{overflow-wrap:anywhere}</style>
<h1>全量 CelebA 续训对照 · 2026-10-25</h1>
<p>从当前最佳之一的 stratified EMA（累计计数 32.768M）初始化四组，保留全部学习参数。该权重在相同 guidance=2、两个固定 seed 下平均 FID 53.646，与先前 cleaner 的 53.648 基本持平。最新 batch/LR/EMA 轮没有超过它。四组统一重置 AdamW、EMA 年龄及本轮曝光计数；不是从随机权重训练。</p>
<h2>已提交的四组实验</h2><table>''' + table + '''</table>
<p>每组申请单节点 16 GPU，均使用已验证的 compile_full。每组追加 16,384,000 次图片曝光，3 万张对照约等效 546.13 遍，全量训练集约 100.66 遍。control30k 与 full 配置文件完全相同，仅数据路径不同；其余两组各只改变一个超参数。状态为最近一次检查快照。</p>
<h2>数据一致性</h2>
<p>从官方 train 分区扩到全部 162,770 张，新增 132,770 张。原来的 30,000 张图片和 1,000 张验证图保持 ID、顺序与像素完全不变。新缓存以前 1,000 张验证图和原 30,000 张训练图为原样前缀，追加其余官方训练图片；没有纳入官方验证或测试图片。预处理仍为原图 178×218、裁剪 (0,20,178,198)、Lanczos 缩放至 96×96。</p>
<p>全量数组 SHA256：<code>''' + submission['data']['array_sha256'] + '''</code><br>验证像素 SHA256：<code>''' + submission['data']['validation_sha256'] + '''</code><br>共同初始权重 SHA256：<code>''' + submission['checkpoint_sha256'] + '''</code></p>
<h2>固定曝光量评估</h2>
<p>保存 4.096M、8.192M、16.384M 曝光处的 checkpoint；训练结束后自动依次评估这些 checkpoint，均用 seed 3333/7777、Heun50、guidance=2，并补充最终模型的无引导结果。FID/KID 仍使用原缓存中的同一批 1,000 张验证图，每次生成 1,024 张；不使用 best-of-N 或外部评分 API。</p>
<h2>验证</h2><p>数据扩展夹具测试已通过：完整训练分区覆盖、验证集和旧训练像素保留、排除验证/测试泄漏、数组哈希匹配、拒绝覆写。既有权重继承与 96px 训练采样测试两项通过，主对照配置完全一致，两个变体均只改一个超参数，所有评估曝光点均整除 batch。</p>
'''
if len(quality) == 28:
    body += '<h2>已完成：四组训练与 28 次评估</h2><table><tr><th>实验</th><th>4.096M 平均 FID</th><th>8.192M 平均 FID</th><th>16.384M 平均 FID</th><th>最终无引导 FID</th></tr>'
    for tag in plan['arms']:
        means = [sum(quality[f'{tag}_{n}_g2_seed{seed}']['clean_fid'] for seed in [3333, 7777]) / 2
                 for n in [4096000, 8192000, 16384000]]
        raw = quality[f'{tag}_16384000_g1_seed3333']['clean_fid']
        body += '<tr><td>' + tag + '</td>' + ''.join(f'<td>{v:.4f}</td>' for v in [*means, raw]) + '</tr>'
    body += '''</table><p>标准全量组最佳：最终 guidance=2 的两个 seed 分别为 52.4293 / 52.9032，对照为 53.6350 / 53.2943；平均降低约 0.80。无引导 FID 从 69.2377 降到 65.9567。全量组随曝光量持续改善，大 batch 和低学习率未超过它。每组训练约 23.5 分钟（包含首次编译），最后记录均无跳过更新。</p>
<p>固定样本显示改善是局部的，并非每张都更好；仍有眼睛、嘴部等失真。两次采样 seed 不是两次独立训练；小样本 FID/KID 结果用于实验筛选，尚不能作统计显著性结论。</p>
<h2>固定样本对比</h2><p>每行固定前 8 张，无筛选、无增强；行顺序为初始模型、3 万张对照、全量标准、大 batch、低学习率。</p>
<img style="width:100%" src="pixel96_full_comparison_seed3333_20261025.png"><img style="width:100%" src="pixel96_full_comparison_seed7777_20261025.png">
<h2>下一轮</h2><p>继续全量标准和 3 万张对照，各继承自身的完整训练状态，再追加相同的 16.384M 曝光。另两组从全量标准的完整状态出发，分别将学习率改为 1e-4、或将 time logit mean 从 0 调为 -0.2。各只改变一个参数，沿用相同验证集与固定评估规则。新轮不重新初始化优化器。</p>'''
(root / 'reports/pixel96_full_20261025.html').write_text(body)
print(root / 'reports/pixel96_full_20261025.html')
