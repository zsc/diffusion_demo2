"""Actual ordinary VAE16 continuous latent DiT results."""
from pathlib import Path
import base64,json,html
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'runs/vae_latent96_20261025'
def main():
 if not (R/'status.json').exists():return
 s=json.loads((R/'status.json').read_text());t=s['task'];p=s.get('train/progress.json',{})
 parts=['<!doctype html><meta charset="utf-8"><title>普通VAE16 latent DiT</title><style>body{font:16px/1.7 system-ui;max-width:1200px;margin:30px auto;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}img{width:100%}</style><h1>普通 VAE16 → 连续 latent DiT</h1>',f'<p>任务 {t["task_id"]} · {html.escape(t["state"])} · 已训练 {p.get("images_seen",0):,} 张 · 1×16 GPU</p>', '<p>冻结所选原生96px、12×12×16 VAE（重建PSNR29.673 / LPIPS0.07729）；不使用PCA。缓存后验均值与logvar，每次训练重新抽取后验噪声。归一化仅使用训练图像1000:31000，并包含后验方差。DiT从零训练，patch1、144 tokens、width384/depth8/heads6，batch2048、lr4e-4、EMA半衰期131072张；目标16,777,216张。</p><p>当前VQVAE-DiT参考：同96px验证集，16M阶段guidance2 FID72.02（seed3333）、71.63（seed7777）。两条路线的预训练历史、latent网格和codec不同，不是匹配总算力的对照；VAE重建更好并不保证生成更好。</p><table><tr><th>训练图像</th><th>采样模式</th><th>FID</th><th>KID</th></tr>']
 for name,m in sorted(s['evaluations'].items()):
  n,mode=name.split('/');parts.append(f'<tr><td>{int(n):,}</td><td>{html.escape(mode)}</td><td>{m["clean_fid"]:.3f}</td><td>{m["kid_mean"]:.5f}</td></tr>')
 parts.append('</table><p>评估1024生成/1000真实，Heun50；guided=2，固定52万张弱模型作引导。全部连续样本，无挑选。验证图用于VAE检查点选择，因此不是完全独立的测试集。</p>')
 for path in sorted((R/'eval').glob('*/*/samples64.png')):
  parts+=['<h2>'+html.escape(str(path.parent.relative_to(R/'eval')))+'</h2>','<img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">']
 for path in sorted((R/'train').glob('samples*.png'),key=lambda p:int(p.stem.split('_')[-1])):
  parts+=['<h2>无引导训练预览 '+path.stem.split('_')[-1]+'</h2>','<img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">']
 (ROOT/'reports/vae_latent96_2026-10-25.html').write_text(''.join(parts))
if __name__=='__main__':main()
