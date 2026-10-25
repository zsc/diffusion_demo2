"""Evidence report for decoder LPIPS/GAN and latent DiT; renders only actual model samples."""
import base64,html,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/vq_perceptual_20261025';OUT=ROOT/'reports'

def embed(p):return '<img src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'">'
def read(p):return json.loads(p.read_text())

def main():
    OUT.mkdir(exist_ok=True);decoders=[];evaluations=[]
    for p in sorted((RUN/'decoder').glob('*/result.json')):
        decoders.append(dict(name=p.parent.name,**read(p),best=read(p.parent/'best.json'),config=read(p.parent/'config.json')))
    for p in sorted((RUN/'eval').glob('*/feature_scores.json')):
        evaluations.append(dict(name=p.parent.name,**read(p),**read(p.parent/'evaluation.json')))
    summary=dict(decoders=decoders,evaluations=evaluations)
    if (RUN/'release/recipe.json').exists():summary['recipe']=read(RUN/'release/recipe.json')
    (OUT/'vq_perceptual_2026-10-25.json').write_text(json.dumps(summary,indent=2))
    parts=['<!doctype html><meta charset="utf-8"><title>VQ 感知损失与 GAN 实验</title><style>body{font:16px/1.6 system-ui;max-width:1160px;margin:36px auto;padding:0 24px;color:#172331}h1,h2{line-height:1.3}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}th,td{padding:8px;text-align:left;border-bottom:1px solid #ddd}.note{background:#eef3f8;padding:16px}.gallery{display:grid;grid-template-columns:1fr 1fr;gap:20px}code{overflow-wrap:anywhere}</style>',
    '<h1>48×48 VQ-VAE：感知损失、GAN 与 latent DiT</h1><p>2026-10-25 · 每个任务独占 1×16 GPU · 原生 48px 样本以最近邻放大，没有后处理锐化或生成图替代</p>',
    '<p class="note">目标：改善真实训练模型的细节，并与之前最强 pixel JiT 比较。固定 29,000 张训练图像、前 1,000 张验证图像；所有 FID/KID 使用相同 clean Inception、1,024 张生成图和同一验证集。这是小样本内部对照，不是 FID50k。</p>',
    '<h2>解码器对照</h2><p>从码本 EMA VQ-VAE（512/512 条目使用、perplexity 420.5、PSNR 25.80 dB）出发，冻结整个编码器和量化器，包括 BN 与码本 EMA 缓冲区。每组训练 2,048 步 × global batch 512 = 1,048,576 张图像；decoder Adam 1e-4、EMA 半衰期 8,192 张。三组共同随机种子、采样序列和初始权重。</p>',
    '<p>MSE：像素均方误差；LPIPS：MSE + 0.1 × LPIPS-Squeeze v0.1；GAN：再加 0.005 × hinge generator loss，第 256 步后启用判别器。判别器为三层 stride-2 卷积 + GroupNorm 的 patch logits，Adam 1e-4。感知网络为冻结的 ImageNet 预训练 SqueezeNet 和官方校准权重，直接处理 [-1,1] 的 48×48 RGB。训练允许梯度穿过感知网络回到解码器，感知网络本身不更新。<a href="https://github.com/richzhang/PerceptualSimilarity">LPIPS 官方实现</a></p>',
    '<table><tr><th>方法</th><th>PSNR ↑</th><th>LPIPS ↓</th><th>码本</th><th>所选步数 / GAN 步数</th><th>训练循环秒数</th></tr>']
    confirmed=[e for e in evaluations if e['seed']==7777]
    if confirmed:
        best=min(confirmed,key=lambda e:e['clean_fid'])
        conclusion=(f'<h2>已完成的结果</h2><p>独立采样种子 7777：<strong>{html.escape(best["name"])}，FID {best["clean_fid"]:.3f}</strong>；此前 latent DiT 为 64.986，pixel JiT 为 44.267。'
                    '本轮感知损失明显改善重建 LPIPS，但冻结解码器替换和联合训练的初步生成对照均未优于充分训练的 MSE 基线。GAN 的已测配置未带来生成收益。清晰的人脸已经能生成，仍有模糊、眼睛不对称和局部伪影，尚未超过 pixel JiT。</p>')
        sample=RUN/'sample_cpu_seed2027/grid.png'
        if sample.exists():conclusion+='<h3>本机复核：seed2027 的连续 64 张</h3>'+embed(sample)
        parts.insert(3,conclusion)
    parallel=RUN/'parallel_status.json'
    if parallel.exists():
        data=read(parallel);summary['parallel']=data
        rows=''.join('<tr><td>'+html.escape(t['name'])+'</td><td>'+str(t['task_id'])+'</td><td>'+html.escape(t['state'])+'</td></tr>' for t in data['tasks'])
        parts.insert(4,'<h2>新增 4×16 GPU 集群</h2><p>每个实验独立申请 1 节点 × 16 GPU；四个训练对照为 PCA8 续训到 16M、感知 VQ 对应 DiT 续训到 16M、基线续训到 32M、PCA8 batch2048 从零训练到 16M（lr4e-4）。每组完成后自动提交固定种子 3333 的 FID/KID 与 64 张图评估。此表是生成报告时的状态快照，队列中的任务不计入已完成结论。</p><table><tr><th>任务</th><th>ID</th><th>状态</th></tr>'+rows+'</table>')
    for d in decoders:
        m=d['best'];parts.append(f'<tr><td>{html.escape(d["name"])}</td><td>{m["psnr"]:.3f}</td><td>{m["lpips"]:.5f}</td><td>{m["codes_used"]}/512</td><td>{m["step"]} / {max(0,m["step"]-d["config"].get("gan_start",0)) if d["config"].get("gan_weight",0) else 0}</td><td>{d["seconds"]:.1f}</td></tr>')
    parts+=['</table><p>检查点：MSE 组按验证 MSE、感知组按验证 LPIPS 选择；实际生成优劣另用 FID/KID 与图像判断。LPIPS 是优化目标，因此其改善不能单独证明更自然。循环时间不含任务排队、环境启动；第一次遇到新算子的编译会增加时间，不用于跨硬件速度结论。</p>']
    if decoders:
        # Remote previews are exact sampled/checkpoint images; display the first eight without selection.
        panel=Image.new('RGB',(8*150,50+len(decoders)*200),'white');draw=ImageDraw.Draw(panel)
        for row,d in enumerate(decoders):
            im=Image.open(RUN/'decoder'/d['name']/'recon_best.png')
            draw.text((8,20+row*200),d['name'],fill='black')
            for i in range(8):
                tile=2*i+1;x=(tile%8)*150;y=(tile//8)*150
                panel.paste(im.crop((x,y,x+144,y+144)),(i*150,50+row*200))
        panel.save(OUT/'vq_perceptual_reconstruction_2026-10-25.png');parts.append(embed(OUT/'vq_perceptual_reconstruction_2026-10-25.png'))
        parts.append('<p>上图为所选检查点在本机 CPU / FP32 下的连续前八张重建；指标来自 GPU / BF16，舍入略有差异。完整原图/重建交替图如下。</p><div class="gallery">')
        for d in decoders:parts+=['<div><h3>'+html.escape(d['name'])+'</h3>'+embed(RUN/'decoder'/d['name']/'recon_best.png')+'</div>']
        parts.append('</div>')
    joint=RUN/'joint/ae/metrics.json'
    if joint.exists():
        data=read(joint);best=min(data['epochs'],key=lambda m:m['val_lpips'])
        summary['joint_vq']=best
        parts += [f'<h2>联合 VQ 感知训练</h2><p>Encoder、decoder 与全局同步码本 EMA 一起训练，追加 32 轮 / 917,504 张图像，lr=1e-4，LPIPS 权重 0.1。所选第 {best["epoch"]} 轮：PSNR {best["val_psnr_db"]:.3f} dB，LPIPS {best["val_lpips"]:.5f}，码本 {best["val_codes_used"]}/512，perplexity {best["val_codebook_perplexity"]:.1f}。对应 latent DiT 从零训练，因为 latent 映射已改变。</p>']
        parts.append(embed(RUN/f'joint/ae/reconstruction_{best["epoch"]:03d}.png'))
    parts+=['<p>额外对照 lpips03 将 LPIPS 权重增至 0.3；ganlow 使用 LPIPS 0.1 + GAN 0.001，判别器 lr=5e-5，第 512 步后启用。ganlow 按 LPIPS 选中的 step512 尚未进行对抗更新，不能算 GAN 收益；因此另测 ganlow_last 和 gan_last 的末期检查点。</p><h2>生成质量</h2><p>第一阶段只替换解码器，沿用相同的 8,388,608-image latent DiT 与 524,288-image 弱模型。通道归一化、encoder/codebook 严格逐张量相等后才允许复用。另有 patch1 从零训练 4M 对照（144 tokens），以及联合 VQ 感知训练后重新训练的 DiT。PCA8 对照仅按训练集码频率拟合加权投影，在 8 维训练、恢复到 64 维解码。后续从原始 optimizer、EMA 和每 rank RNG 续训到 16,777,216 张，学习率 5e-5；冻结的 latent 分布不变。Heun 50 步，interval guidance 2.0，t∈[0.1,0.8]，不做末端重新量化。</p>',
    '<table><tr><th>实验</th><th>采样种子</th><th>训练图像数 / 引导</th><th>FID ↓</th><th>KID ↓</th><th>样本数</th></tr>']
    for e in evaluations:parts.append(f'<tr><td>{html.escape(e["name"])}</td><td>{e["seed"]}</td><td>{Path(e["checkpoint"]).stem.removeprefix("ema_")} / {e["guidance"]}</td><td>{e["clean_fid"]:.3f}</td><td>{e["kid_mean"]:.5f}</td><td>{e["fake_count"]}</td></tr>')
    parts+=['<tr><td>此前最佳 latent DiT</td><td>7777</td><td>8,388,608 / 2</td><td>64.986</td><td>0.04844</td><td>1024</td></tr><tr><td>此前最佳 pixel JiT</td><td>7777</td><td>8,192,000 / 2</td><td>44.267</td><td>0.03211</td><td>1024</td></tr></table>',
    '<p>种子 3333 用于方案选择，7777 用于固定方案后的独立采样确认；训练只有一个种子。pixel JiT 与 latent DiT 的架构及预算不同，此处比较最终质量，不作等算力结论。</p><div class="gallery">']
    for p in sorted((RUN/'eval').glob('*/samples64.png')):parts+=['<div><h3>'+html.escape(p.parent.name)+'</h3>'+embed(p)+'</div>']
    parts+=['</div><h2>复现</h2><p>入口：<code>vq_perceptual_experiment.py</code>、<code>latent_ddp_experiment.py</code>、<code>sample_latent.py</code>。配置与任务记录、验证指标、传输 SHA256、所选权重存于 <code>runs/vq_perceptual_20261025/</code>。decoder loss 支持独立配置；训练结束验证 encoder 和 quantizer 没有任何改变。</p>']
    (OUT/'vq_perceptual_2026-10-25.html').write_text(''.join(parts))
    (OUT/'vq_perceptual_2026-10-25.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__':main()
