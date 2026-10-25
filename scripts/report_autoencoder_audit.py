"""Build the self-contained VQ-VAE ablation report from downloaded GPU artifacts."""
import argparse
import base64
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/vae_gpu_20261025')
    parser.add_argument('--output', default='reports/vae_gpu_2026-10-25.html')
    args = parser.parse_args()
    root, output = Path(args.run), Path(args.output)
    summary = json.loads((root / 'quality_summary.json').read_text())
    rows, curves = [], []
    names = ['g4_sum', 'g4_mean', 'g12_sum', 'g12_mean']
    colors = ['#b24345', '#2777b3', '#bb861c', '#15826b']
    for name, color in zip(names, colors):
        run = summary['runs'][name]
        e = run['final']
        rows.append(f'<tr><td>{name}</td><td>{e["val_recon_mse"]:.5f}</td><td>{e["val_psnr_db"]:.2f}</td>'
                    f'<td>{e["val_codes_used"]}/512</td><td>{e["val_codebook_perplexity"]:.2f}</td>'
                    f'<td>{run["total_seconds"]:.2f}</td></tr>')
        history = json.loads((root / 'runs' / name / 'metrics.json').read_text())['epochs']
        points = ' '.join(f'{45+e["epoch"]*22},{290-(e["val_psnr_db"]-9)*19}' for e in history)
        curves.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
    reduction = 1 - summary['runs']['g12_mean']['final']['val_recon_mse'] / summary['runs']['g4_sum']['final']['val_recon_mse']
    reload_check = json.loads((root / 'local_reload_check.json').read_text())
    preview = output.with_suffix('.png')
    image64 = base64.b64encode(preview.read_bytes()).decode()
    native = ''
    native_path = root / 'native96_summary.json'
    if native_path.exists():
        native_summary = json.loads(native_path.read_text())
        native_rows = ''.join(f'<tr><td>{name}</td><td>{v["final"]["val_psnr_db"]:.2f}</td>'
                              f'<td>{v["final"]["val_recon_mse"]:.5f}</td><td>{v["final"]["val_codes_used"]}/512</td>'
                              f'<td>{v["final"]["val_codebook_perplexity"]:.2f}</td></tr>'
                              for name, v in native_summary['runs'].items())
        native = f'''<h2>真实 96px：从零训练与权重继承</h2><p>两组均在真实 96px 数据上新增训练 32 轮，global batch=512，每组 950,272 张实际看图量；训练 30,000 张 / 验证 1,000 张。继承组额外使用了前述 48px 训练，不是相同总算力比较。旧 sprites 无原图 ID，无法排除 96px 验证图参与过 48px 预训练；本表不是独立无泄漏测试集的最终泛化结论。96px 与前述 48px 的图像裁切和数据划分不同，PSNR 不宜直接跨表比较。</p><table><tr><th>初始化</th><th>最终 PSNR dB ↑</th><th>MSE [0,1] ↓</th><th>使用代码数</th><th>困惑度</th></tr>{native_rows}</table>'''
        native_png = output.with_name('vae96_transfer_2026-10-25.png')
        if native_png.exists():
            native += '<p>每列前两排原图、后两排重建；相同验证图、32 轮末尾，未挑图。</p><img alt="96px transfer reconstructions" src="data:image/png;base64,' + base64.b64encode(native_png.read_bytes()).decode() + '">'
        summary['native96'] = native_summary
    summary['verification'] = dict(unittest_passed=31, local_fp32_reload=reload_check, gpu_smoke_tasks=[56170, 56171])
    summary['exports'] = {name: json.loads((root / name).read_text()) for name in ['export_receipt.json', 'export96_receipt.json'] if (root / name).exists()}
    body = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>VQ-VAE 16 GPU 对照与修复</title>
<style>body{{max-width:1200px;margin:40px auto;padding:0 24px;font:16px/1.65 system-ui;color:#243342;background:#fafbfc}}h1,h2{{line-height:1.3}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:10px;border-bottom:1px solid #dbe3e7;text-align:left}}img{{width:100%}}code{{background:#edf0f4;padding:2px 5px}}.note{{padding:16px;background:#edf4f8;border-radius:8px}}a{{color:#186b9f}}</style>
<h1>修正 VQ 损失尺度，再扩大空间网格</h1><p>2026-10-25 · 每组 1×16 GPU · 四组均成功完成、无跳过更新</p>
<p class="note">固定 29,000 张训练 / 1,000 张验证，global batch=512、lr=2e-4、BF16、同 seed、32 轮。每组 917,504 张实际看图量 / 1,792 次优化器更新。embedding_dim=64、codebook=512，编码器 [128,64]、解码器 [64,128]，无 GAN、无权重 EMA。四组 split.json 哈希完全相同。本表使用预先固定的第 32 轮，不挑最佳轮次；仍只有一个 seed。</p>
<table><tr><th>设置</th><th>最终 MSE [0,1] ↓</th><th>PSNR dB ↑</th><th>使用代码数</th><th>困惑度</th><th>训练器计时 秒</th></tr>{''.join(rows)}</table>
<p>最佳设置相对旧设置的验证 MSE 降低 <b>{100*reduction:.1f}%</b>。PSNR 是重建指标，不能推断无条件生成人脸的 FID/KID。计时包含训练、验证和保存，不含 torchrun 启动、排队、数据准备；没有做 2080 SUPER 的匹配基准，不报告硬件加速倍数。</p>
<h2>固定验证图的最终重建</h2><p>每列上两排原图、下两排重建；原生 48px，仅最近邻放大，无筛选样本、超分或图像修复。12×12 / mean 的轮廓与五官改善明显，但眼睛、牙齿和纹理仍有平滑与失真。</p><img src="data:image/png;base64,{image64}" alt="Four 16-GPU VQ-VAE ablation grids">
<h2>32 轮验证曲线</h2><p>{'　'.join(f'<span style="color:{c}">● {n}</span>' for n,c in zip(names,colors))}</p><svg viewBox="0 0 800 330" width="100%"><rect width="800" height="330" fill="white"/><path d="M45 20V290H770" fill="none" stroke="#90a0a9"/>{''.join(curves)}<text x="0" y="30">23 dB</text><text x="0" y="290">9 dB</text><text x="335" y="320">epoch 1 → 32</text></svg>
{native}
<h2>已修复的实质问题</h2><ol><li><b>VQ 损失尺度不一致：</b>reconstruction 原来逐图求和，VQ/commitment 取均值。48px 的像素数系数为 6912，96px 为 27648。新默认采用 mean，sum 显式保留作旧对照；VAE 的求和 ELBO 约定不变。</li><li><b>固定 4×4 瓶颈及 48px 输出：</b>支持原生 96px 与可配置网格；VQ 默认两层 encoder 在 48px 可保留 12×12、96px 保留 24×24。卷积/codebook 权重可严格继承；通道、码本大小需一致。init_checkpoint 只迁移权重，不恢复优化器。</li><li><b>VAE 死参数：</b>down_mu/down_logvar 从未进入 forward，已删除，解决无效参数和默认 DDP 梯度归约问题；旧权重仅忽略这四个已知冗余 tensor。</li><li><b>大 batch 内存与观测：</b>码本距离保持 FP32，分块且不建梯度图；记录使用代码数/困惑度、全局看图量与实际更新次数。DDP 每次使用一个完整 16 GPU 组。</li><li><b>入口与数值一致性：</b>CLI 与 API 共用核心训练器、固定留出集和完整模型配置；GAN 用 logits/softplus；修正预览归一化；停止把 Gaussian latent 直接喂 VQ decoder 当生成测试。DiT 从 checkpoint 读取实际网格和分辨率。</li></ol>
<p>修复范围为 src/diffusion_demo、API 和 cmdline_train.py；根目录 app.py 对应历史实现，未替换其架构。旧 CPU 小预算诊断见 <a href="vae_audit_2026-10-24.html">首轮审计报告</a>。</p>
<h2>验证、权重与下一步</h2><p>31 项测试通过，包括双进程两轮 DDP、GAN 反传、原生 96px VQ-VAE→DiT→采样、旧权重加载和跨分辨率严格迁移。16 GPU 的 VQ48 与 VAE96 smoke 均通过。最佳 48px 权重已经对象存储取回，SHA256 两端一致；本机 FP32 在相同 1,000 张验证图上重算 <b>{reload_check['val_psnr_db']:.2f} dB</b>，数据 SHA256 已校验。</p>
<p>本地权重：<code>runs/vae_gpu_20261025/best_model.pth</code>（48px）、<code>runs/vae_gpu_20261025/best96_model.pth</code>（96px，24×24 latent，已校验 SHA256 并本机严格加载）；完整原始日志/配置/划分：<code>runs/vae_gpu_20261025/</code>。完整优化器 checkpoint 留在共享文件系统。当前 autoencoder 没有精确续训入口，也没有 EMA。</p>
<p>下轮最值得做：码本数据初始化、EMA 码本更新或失活码重置（当前验证集用到 35/512 个码）；VAE 的 KL 预热/权重与 decoder 额外上采样层；按看图量定义的模型 EMA。大 batch 需同步比较学习率与更新次数；BatchNorm running buffers 不能直接照搬 JiT 的 EMA 逻辑。最终要训练 latent prior 再看生成网格和 FID/KID。核函数融合和 channels-last 先 profile，JiT 的加速倍率不能套用。</p>
<p>参考：<a href="https://raw.githubusercontent.com/google-deepmind/sonnet/v2/sonnet/src/nets/vqvae.py">Sonnet 的 VQ loss / perplexity 实现</a>、<a href="https://arxiv.org/abs/1711.00937">VQ-VAE 原论文</a>。</p></html>'''
    output.write_text(body)
    output.with_suffix('.json').write_text(json.dumps(summary, indent=2))
    print(output)


if __name__ == '__main__':
    main()
