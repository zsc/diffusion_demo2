"""Report measured channel PCA spectrum and frozen VAE reconstructions."""
import base64,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'runs/vae_pca_20261025'
def embed(path):return '<img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">'
def main():
 s=json.loads((R/'results.json').read_text());out=ROOT/'reports';out.mkdir(exist_ok=True)
 parts=['<!doctype html><meta charset="utf-8"><title>普通 VAE 通道 PCA 实测</title><style>body{font:16px/1.7 system-ui;max-width:1250px;margin:30px auto;padding:0 20px;color:#172331}table{width:100%;border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}img{width:100%}.note{background:#edf3f8;padding:15px}</style><h1>普通 VAE：16 通道 PCA 截断实验</h1><p>2026-10-25 · 1×16 GPU · 冻结编码器与解码器，无微调</p><p class="note">这里是早期普通 VAE，不是当前 VQVAE：48px、4×4空间网格、16通道、1792张图训练8轮。分别检查KL=0.1与KL=1。PCA仅在原训练集拟合，主表使用另外1000张未参与训练或检查点选择的图片。此结果不能代表训练充分的VAE，也不能直接推断latent DiT生成质量。</p><p>mu：对后验均值做PCA；sample：对后验采样分布做PCA，协方差精确使用 Cov(mu)+E[diag(exp(logvar))]。每张验证图使用固定的一份后验噪声，所有截断版本共享。投影只缩减通道，保持4×4网格；投影还原后送入原decoder。</p>']
 decisions={}
 for name,label in [('kl01','KL=0.1'),('kl1','KL=1')]:
  d=s[name];parts+=['<h2>'+label+'</h2><table><tr><th>latent</th><th>通道</th><th>训练方差保留</th><th>PSNR ↑</th><th>PSNR 损失</th><th>LPIPS ↓</th><th>LPIPS 增量</th></tr>']
  for mode in ['mu','sample']:
   metrics=d['evaluations']['unseen1000_'+mode];base=metrics['16'];eligible=[]
   for k in [16,12,8,4,2]:
    m=metrics[str(k)];loss=base['psnr']-m['psnr'];delta=m['lpips']-base['lpips']
    if loss<=.2 and delta<=.005:eligible.append(k)
    parts.append(f'<tr><td>{mode}</td><td>{k}</td><td>{m["retained_variance"]*100:.3f}%</td><td>{m["psnr"]:.3f}</td><td>{loss:.3f}</td><td>{m["lpips"]:.5f}</td><td>{delta:+.5f}</td></tr>')
   decisions[name+'_'+mode]=dict(min_channels_under_screen=min(eligible),max_psnr_loss=.2,max_lpips_increase=.005)
  parts.append('</table>')
  for mode in ['mu','sample']:
   parts+=['<h3>'+label+' / '+mode+'：固定连续16张，原图→16→12→8→4→2通道</h3>',embed(R/name/'unseen1000_original.png')]
   for k in [16,12,8,4,2]:parts+=['<p>'+str(k)+'通道</p>',embed(R/name/f'unseen1000_{mode}_k{k}.png')]
 parts.append('<p>轻损失筛查阈值：PSNR下降≤0.2 dB且LPIPS增加≤0.005；这只是后续实验的筛查条件，不是感知无损证明。完整原验证集指标、谱、PCA参数与固定图片见 runs/vae_pca_20261025/。</p>')
 (out/'vae_pca_2026-10-25.html').write_text(''.join(parts));(R/'screening.json').write_text(json.dumps(decisions,indent=2));print(json.dumps(decisions))
if __name__=='__main__':main()
