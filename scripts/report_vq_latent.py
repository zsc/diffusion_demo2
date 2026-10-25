"""Build an evidence-linked report from actual VQ / latent DiT experiment artifacts."""
import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/vq_latent_20261025'
OUT=ROOT/'reports'


def embed(path):
    return '<img src="data:image/png;base64,'+base64.b64encode(Path(path).read_bytes()).decode()+'">'


def main():
    rows=json.loads((RUN/'vq_summary.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,3.6))
    for row in rows:
        name=row['name'];epochs=json.loads((RUN/f'vq/{name}/metrics.json').read_text())['epochs']
        x=[e['images_seen']/1e6 for e in epochs]
        for axis,key in zip(axes,['val_psnr_db','val_codebook_perplexity']):
            axis.plot(x,[e[key] for e in epochs],label=name)
    for axis,label in zip(axes,['Validation PSNR (dB)','Validation codebook perplexity']):
        axis.set(xlabel='Additional training images (millions)',ylabel=label);axis.grid(alpha=.2);axis.legend()
    fig.tight_layout();fig.savefig(OUT/'vq_ema_curves_2026-10-25.png',dpi=160);plt.close(fig)
    # Every panel uses the same first eight validation images, without selection.
    old=ROOT/'runs/vae_gpu_20261025/runs/g12_mean/reconstruction_032.png'
    if not old.exists():
        old=next((ROOT/'runs/vae_gpu_20261025/runs/g12_mean').glob('reconstruction*.png'))
    imgs=[Image.open(old)]+[Image.open(RUN/f'vq/{r["name"]}/reconstruction_{r["best"]["epoch"]:03d}.png') for r in rows]
    labels=['Original','Previous VQ: 21.52 dB','Continue: 23.83 dB','Codebook EMA: 25.79 dB','EMA + reset: 25.80 dB']
    panels=[imgs[0].crop((0,0,192,96))]+[im.crop((0,96,192,192)) for im in imgs]
    fig,axes=plt.subplots(5,1,figsize=(8,10))
    for ax,im,label in zip(axes,panels,labels):ax.imshow(im,interpolation='nearest');ax.set_title(label);ax.axis('off')
    fig.tight_layout();fig.savefig(OUT/'vq_reconstruction_2026-10-25.png',dpi=140);plt.close(fig)
    table=''.join(f'<tr><td>{r["name"]}</td><td>{r["best"]["val_psnr_db"]:.3f}</td><td>{r["best"]["val_codes_used"]}/512</td><td>{r["best"]["val_codebook_perplexity"]:.1f}</td><td>{r["best"]["raw_val_psnr_db"]:.3f}</td><td>{r["best"]["ema_val_psnr_db"]:.3f}</td></tr>' for r in rows)
    evaluations=[]
    for path in sorted((RUN/'eval').glob('*/feature_scores.json')):
        scores=json.loads(path.read_text());metadata=json.loads((path.parent/'evaluation.json').read_text())
        evaluations.append(dict(name=path.parent.name,**scores,**metadata))
    recipe_path=RUN/'release/recipe.json'
    recipe=json.loads(recipe_path.read_text()) if recipe_path.exists() else None
    parts=['<!doctype html><meta charset="utf-8"><title>VQ EMA 与 latent DiT 实验</title><style>body{font:16px/1.65 system-ui;max-width:1120px;margin:40px auto;padding:0 20px;color:#172331}h1,h2{line-height:1.3}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ccc;text-align:left}.note{background:#f1f5f9;padding:16px}.gallery{display:grid;grid-template-columns:1fr 1fr;gap:20px}code{overflow-wrap:anywhere}</style>',
    '<h1>48×48 VQ-VAE → latent DiT</h1><p>2026-10-25 · 每个任务独占 1×16 GPU · 全部模型训练使用真实 CelebA 数据</p>',
    '<h2>码本与模型 EMA</h2><p>三组均从上一轮 21.52 dB 的 VQ-VAE 继承权重，重建 loss 使用 mean；共同续训 64 轮 / 1,835,008 张图像 / global batch 512。训练 29,000 张、验证固定前 1,000 张。模型 EMA 半衰期 8,192 张；码本 EMA decay 0.99。数据初始化只使用训练集编码特征。</p>',
    '<table><tr><th>实验</th><th>验证 PSNR ↑</th><th>使用码数</th><th>Perplexity ↑</th><th>Raw PSNR</th><th>模型 EMA PSNR</th></tr>'+table+'</table>',
    '<p>continue：原梯度码本继续训练；ema：训练数据初始化 + 全局同步码本 EMA；revive：再增加每 100 步死码重置，阈值为均匀使用量的 10%。只有成功 optimizer step 才更新码本；验证/采样不更新。模型 EMA 平滑 encoder/decoder，复制码本与 BN 状态，避免重复平均码本。</p>',
    '<p class="note">数据初始化与码本 EMA 是组合消融，不能单独归因。死码重置只增加约 0.01 dB，单次训练尚不足以证明其有实质增益。充分续训本身已带来约 2.31 dB 的改善。</p>',
    embed(OUT/'vq_ema_curves_2026-10-25.png'),embed(OUT/'vq_reconstruction_2026-10-25.png'),
    '<h2>Latent DiT 生成评估</h2><p>旧、新 VQ-VAE 分别冻结；DiT 使用 64×12×12 latent、patch 2、width 384、depth 8、6 heads，按训练码分布进行逐通道标准化。沿用 JiT 的干净样本预测 / velocity loss，EMA 半衰期 131,072 张，Heun 50 步。两组第一阶段都训练 4,194,304 张图像。归一化统计不使用验证集；缓存保存离散码以减少读取和编码开销。</p>']
    if recipe and 'confirm_best' in recipe:
        best=recipe['confirm_best'];old=recipe['confirm_old']
        selected=recipe['selection']['name']
        parts.insert(3,f'<p class="note"><b>结论：</b>码本占用从 35/512 增到 512/512，验证重建 PSNR 从 21.52 提升到 25.80 dB。选定配方 <code>{selected}</code> 在独立采样种子 7777 上的 FID 为 <b>{best["clean_fid"]:.2f}</b>，旧码本 DiT 为 {old["clean_fid"]:.2f}。脸部轮廓与局部细节更清楚，但仍有模糊、五官变形；尚未超过先前 pixel JiT 的 FID 44.27 / KID 0.03211。</p>')
        parts.append(f'<p>所选主模型训练 {recipe.get("trained_images", 4194304):,} 张图像，弱模型训练 524,288 张图像；引导强度 2.0，仅 t∈[0.1,0.8] 生效，Heun 50 步，不做末端重新量化。主模型从 4M 保留 optimizer/EMA 续训到 8M，学习率降到 1e-4。同一个弱引导模型下，FID 从 70.11 改善到 67.16；换成 2M 的较强引导模型反而退步，因此保留早期弱模型。</p>')
    if evaluations:
        parts+=['<table><tr><th>采样方案</th><th>种子</th><th>FID ↓</th><th>KID ↓</th><th>样本数</th></tr>'+''.join(f'<tr><td>{html.escape(e["name"])}</td><td>{e["seed"]}</td><td>{e["clean_fid"]:.2f}</td><td>{e["kid_mean"]:.5f}</td><td>{e["fake_count"]}</td></tr>' for e in evaluations)+'</table>']
    else:parts+=['<p>评估尚未完成，当前不作生成质量结论。</p>']
    parts+=['<div class="gallery">']
    for path in sorted((RUN/'eval').glob('*/samples64.png')):
        parts+=['<div><h3>'+html.escape(path.parent.name)+'</h3>'+embed(path)+'</div>']
    parts+=['</div><h2>历史方案的视觉参照</h2><div class="gallery"><div><h3>旧 VAE / latent DiT（较小数据与模型）</h3>'+embed(ROOT/'runs/20261024_181538_local_recommended/dit/samples_150.png')+'</div><div><h3>Pixel JiT 最佳方案（seed 2027）</h3>'+embed(ROOT/'runs/pixel_best/sample_seed2027/grid.png')+'</div></div>']
    parts+=['<p class="note">FID/KID 使用同一 clean Inception 与同一 1,000 张验证集，生成 1,024 张；这是小样本内部对照，不是 FID50k。独立种子确认指采样种子，训练仍只有一个种子。本轮采样图为连续前 64 张，无筛选（历史 VAE 图保留原存档的 16 张）；48px 原始像素以最近邻放大。PSNR 衡量重建，不保证生成自然度。旧 VAE/DiT、最强 pixel JiT 的架构、数据量和预算不同，不能作为严格因果消融。</p>',
    '<h2>复现与核验</h2><p>模型、配置、训练曲线、任务记录、传输 SHA256 与完整样本保存在 <code>runs/vq_latent_20261025/</code>。训练入口为 <code>vae_ddp_experiment.py</code> 与 <code>latent_ddp_experiment.py</code>；CLUSTER 通过相应 <code>scripts/run_*_gpu_group.sh</code> 启动。latent checkpoint 保存 optimizer、EMA、每 rank RNG 和归一化统计，可继续训练；CPU 中断恢复测试与连续训练逐位一致。</p>']
    (OUT/'vq_latent_2026-10-25.html').write_text(''.join(parts))
    (OUT/'vq_latent_2026-10-25.json').write_text(json.dumps(dict(vq=rows,evaluations=evaluations),indent=2))


if __name__=='__main__':main()
