"""Build a standalone HTML report from two measured experiment runs.

Requires matplotlib (report generation only; not a training dependency).
"""
import argparse
import base64
import html
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def embedded(path):
    return 'data:image/png;base64,' + base64.b64encode(Path(path).read_bytes()).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--benchmark', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    baseline, candidate, benchmark_dir = map(Path, (args.baseline, args.candidate, args.benchmark))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    b = json.loads((baseline / 'vae/metrics.json').read_text())
    c = json.loads((candidate / 'vae/metrics.json').read_text())
    d = json.loads((candidate / 'dit/metrics.json').read_text())
    bench = json.loads((benchmark_dir / 'benchmark.json').read_text())
    env = json.loads((candidate / 'environment.json').read_text())
    bs = json.loads((baseline / 'vae/split.json').read_text())
    cs = json.loads((candidate / 'vae/split.json').read_text())
    assert bs == cs, 'The two runs must use identical data and split indices'
    br, cr = b['epochs'][-1], c['epochs'][-1]
    improvement = (1 - cr['val_recon_mse'] / br['val_recon_mse']) * 100
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for metrics, label, color in ((b, 'beta=1', '#64748b'), (c, 'beta=0.1 + warmup', '#0d9488')):
        xs = [r['epoch'] for r in metrics['epochs']]
        axes[0].plot(xs, [r['val_psnr_db'] for r in metrics['epochs']], 'o-', label=label, color=color)
        axes[1].plot(xs, [r['recon_loss'] / 6912 / 4 for r in metrics['epochs']], 'o-', label=label, color=color)
    axes[0].set(title='Held-out reconstruction', xlabel='VAE epoch', ylabel='PSNR (dB), higher is better')
    axes[1].set(title='Stochastic training reconstruction', xlabel='VAE epoch', ylabel='MSE in [0, 1]')
    axes[2].plot([r['epoch'] for r in d['epochs']], [r['val_ema_noise_mse'] for r in d['epochs']], color='#6366f1')
    axes[2].set(title='Held-out diffusion loss (EMA)', xlabel='DiT epoch', ylabel='Noise MSE', yscale='log')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False, fontsize=8)
    plot = output.with_suffix('.png')
    fig.savefig(plot, dpi=180)
    plt.close(fig)
    benchmark_rows = []
    for key in ((False, 'fp32', 'sdpa'), (True, 'fp32', 'sdpa'), (True, 'fp16', 'sdpa'), (True, 'fp16', 'manual')):
        rows = [r for r in bench['trials'] if (r['cached'], r['precision'], r['attention']) == key]
        values = [r['images_per_second'] for r in rows]
        benchmark_rows.append(f'<tr><td>{"缓存 posterior" if key[0] else "每批在线编码"}</td><td>{key[1]}</td>'
                              f'<td>{key[2]}</td><td>{statistics.median(values):,.0f}</td>'
                              f'<td>{min(values):,.0f}–{max(values):,.0f}</td></tr>')
    sampling = bench['sampling']
    ratio = sampling[0]['seconds'] / sampling[1]['seconds']
    vae_images = ''.join(f'<figure><img src="{embedded(root / "vae" / ("reconstruction_%03d.png" % row["epoch"]))}">'
                         f'<figcaption>{label} · PSNR {row["val_psnr_db"]:.2f} dB<br>上两行：原图；下两行：重建</figcaption></figure>'
                         for root, row, label in ((baseline, br, '基线'), (candidate, cr, '候选')))
    sample_paths = sorted((candidate / 'dit').glob('samples_*.png'))
    samples = ''.join(f'<figure><img src="{embedded(p)}"><figcaption>{html.escape(p.stem)}</figcaption></figure>'
                      for p in (sample_paths[0], sample_paths[-1]))
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Diffusion 实验优化 · 2026-10-24</title><style>
body{{font:16px/1.7 system-ui,sans-serif;background:#f4f6f8;color:#1b2738;margin:0}}main{{max-width:1120px;margin:auto;padding:40px 24px}}
h1{{font-size:32px;margin-bottom:6px}}h2{{margin-top:32px}}.muted,figcaption{{color:#566477}}section{{background:white;border:1px solid #dfe5eb;border-radius:12px;padding:24px;margin:20px 0}}
.stats,.images{{display:flex;gap:24px;flex-wrap:wrap}}.stat{{flex:1;min-width:210px}}.stat strong{{display:block;font-size:30px;color:#0f766e}}
figure{{margin:8px 0;flex:1;min-width:260px;text-align:center}}figure img{{width:288px;max-width:100%;image-rendering:auto}}.plot{{width:100%}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #dfe5eb;text-align:left;padding:9px}}pre{{overflow:auto;background:#f5f7f9;padding:16px;font-size:12px}}a{{color:#0f766e}}code{{font-size:.9em}}
</style><main><p class="muted">实测记录 · 2026-10-24 · RTX 2080 SUPER / 8GB</p><h1>先修训练链路，再比较质量与速度</h1>
<p>使用本机 GPU 实际训练；所有数值来自保存的 metrics / benchmark JSON。生成质量仍有限，重建指标改善不等于生成质量已解决。</p>
<section class="stats"><div class="stat">验证重建 PSNR<strong>{br['val_psnr_db']:.2f} → {cr['val_psnr_db']:.2f} dB</strong></div>
<div class="stat">验证 MSE 降幅<strong>{improvement:.1f}%</strong></div><div class="stat">50 步 DDIM 采样加速<strong>{ratio:.1f}×</strong></div></section>
<section><h2>对照条件</h2><p>同一前 2,048 张图，同一 seed=42，训练 1,792 张、验证 256 张；VAE 均为 16×4×4 latent，batch=16、Adam lr=2e-4、8 轮、FP16。基线 β=1；候选 β=0.1，前三轮从 β/3 线性升到 β。这个对照同时改变 β 和 warmup，不能区分两者的独立贡献。</p>
<p>已修复 sprite 每行只读 100 列的问题：按实际宽度读取全部 200 列、30,000 张，新缓存为 v2。两个质量对照使用相同 v2 数据指纹和相同划分。验证 PSNR 根据 [0,1] 上的全像素 MSE 计算，使用 posterior mean；没有按身份划分，不能作为身份泛化测试。</p>
<img class="plot" src="{embedded(plot)}"><table><tr><th>VAE</th><th>验证 MSE</th><th>PSNR</th><th>训练秒数（各轮合计）</th></tr>
<tr><td>β=1</td><td>{br['val_recon_mse']:.6f}</td><td>{br['val_psnr_db']:.2f} dB</td><td>{sum(r['train_seconds'] for r in b['epochs']):.1f}</td></tr>
<tr><td>β=0.1 + warmup</td><td>{cr['val_recon_mse']:.6f}</td><td>{cr['val_psnr_db']:.2f} dB</td><td>{sum(r['train_seconds'] for r in c['epochs']):.1f}</td></tr></table></section>
<section><h2>同一验证样本的重建</h2><div class="images">{vae_images}</div></section>
<section><h2>DiT 生成结果</h2><p>4 层、hidden=128、patch=1（16 tokens），lr=1e-4，batch=64，{len(d['epochs'])} 轮；posterior 缓存、按训练集统计量标准化、EMA、50 步 DDIM。下图固定同一随机种子。末轮验证噪声 MSE={d['epochs'][-1]['val_ema_noise_mse']:.4f}。这不是 FID，也不能量化脸部真实性或多样性。</p><div class="images">{samples}</div>
<p>下一步优先扩大训练数据，再比较更大的 latent 空间分辨率；当前 4×4 bottleneck 的重建仍模糊。只降低 KL 会削弱标准正态先验约束，所以需要训练 DiT 来拟合实际 latent 分布。</p></section>
<section><h2>性能实测</h2><p>同一小型 DiT，batch=32，每项预热 10 步、计时 60 步、重复 3 次并交替顺序；报告中位数。固定 GPU 上的真实图像 batch 驻留在显存，训练计时不含数据读取与一次性建缓存。该性能测试使用本次候选 checkpoint，只用于相同结构的算子开销比较。GPU 同时有其他进程，绝对速度会波动。</p>
<table><tr><th>输入</th><th>精度</th><th>Attention</th><th>中位数 images/s</th><th>三次范围</th></tr>{''.join(benchmark_rows)}</table>
<p>小型 DiT 上 FP32 比 FP16 更快，因此推荐 VAE 用 auto（此卡为 FP16）、DiT 用 FP32。SDPA 不等于这张卡一定运行 FlashAttention。</p>
<p>同一 checkpoint 生成 16 张：1000 步 DDPM {sampling[0]['seconds']:.3f}s；50 步 DDIM {sampling[1]['seconds']:.3f}s。两种采样器轨迹不同，速度比不代表相同图像质量。</p></section>
<section><h2>修复与验证</h2><ul><li>autocast + FP16 GradScaler；KL、VQ 距离和 loss 归约保留 FP32。</li><li>checkpoint 自动恢复 VAE/DiT 结构、diffusion schedule、latent normalization 和 EMA。</li><li>移除随机 latent 投影；缓存 VAE 均值和方差，每批重新采样 posterior。</li><li>TensorBoard 批量图像日志、固定验证集、固定预览种子和数据指纹。</li><li>8 项回归测试通过；另测 legacy CLI 小 batch、VQVAE/GAN、Flask 路由及 GPU 完整链路。</li></ul>
<p>算法实现参考：<a href="https://docs.pytorch.org/docs/stable/amp.html">PyTorch AMP</a>、<a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html">PyTorch SDPA</a>、<a href="https://arxiv.org/abs/2010.02502">DDIM 原论文</a>。</p></section>
<details><summary>配置与环境</summary><pre>{html.escape(json.dumps(dict(baseline=b['config'], candidate=c['config'], dit=d['config'], environment=env), indent=2, ensure_ascii=False))}</pre></details>
<p class="muted">数据 SHA256：{cs['selected_sha256']}<br>本次仅单个 seed；需多 seed 和更长训练验证结论的稳定性。</p></main></html>'''
    output.write_text(document)
    output.with_suffix('.json').write_text(json.dumps(dict(baseline=b, candidate=c, dit=d, benchmark=bench,
                                                         split=cs, environment=env), indent=2))
    print(output.resolve())


if __name__ == '__main__':
    main()
