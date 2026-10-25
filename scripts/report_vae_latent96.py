"""Actual ordinary VAE16 continuous latent DiT results and guidance ablations."""
from pathlib import Path
import base64,json,html
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'runs/vae_latent96_20261025'
def main():
 if not (R/'status.json').exists():return
 s=json.loads((R/'status.json').read_text());t=s['task'];p=s.get('train/progress.json',{})
 parts=['<!doctype html><meta charset="utf-8"><title>普通VAE16 latent DiT</title><style>body{font:16px/1.7 system-ui;max-width:1200px;margin:30px auto;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}img{width:100%}.good{background:#eef8ed}</style><h1>普通 VAE16 → 连续 latent DiT</h1>',f'<p>任务 {t["task_id"]} · {html.escape(t["state"])} · 已训练 {p.get("images_seen",0):,} 张 · 1×16 GPU</p>', '<p>冻结所选原生96px、12×12×16 VAE（重建PSNR29.673 / LPIPS0.07729）；不使用PCA。缓存后验均值与logvar，每次训练重新抽取后验噪声。归一化仅使用训练图像1000:31000，并包含后验方差。DiT从零训练，patch1、144 tokens、width384/depth8/heads6，batch2048、lr4e-4、EMA半衰期131072张。</p>', '<p>16M无引导FID 69.82，优于本次8M的77.56；52万张弱模型、强度2的引导会放大对比度和伪影，16M FID退化到93.24，不推荐沿用。4M弱模型、强度2将16M FID改善至57.07；8M弱模型、强度2为62.60，4M弱模型、强度1.5为61.82。因此当前选择4M弱模型、强度2。更晚的引导实验列在下方。人脸总体可辨，仍有嘴部、眼睛及侧脸变形。</p>',f'<p>后一段8M训练记录吞吐 {p.get("images_per_second",0):,.0f} 张/秒，约为此前VQVAE路线12.2k张/秒的3.5倍。latent tokens从576降至144，训练配方不同；这不是单卡硬件加速比的结论。</p>', '<p>VQVAE-DiT参考：同96px验证集、16M阶段，无引导FID89.02（seed3333）；guidance2为72.02（seed3333）、71.63（seed7777）。两条路线的预训练历史、latent网格和codec不同，不是匹配总算力的对照。同期仓库96px pixel JiT对照FID约53.65，当前连续VAE尚未超过它；该对照拥有更长训练历史。此处不能与48px pixel JiT FID直接比较。</p><table><tr><th>训练图像</th><th>采样模式</th><th>FID↓</th><th>KID↓</th></tr>']
 rows=[]
 for name,m in sorted(s['evaluations'].items()):
  n,mode=name.split('/');rows.append((int(n),mode.replace('guided_','guide0.5M_g2_'),m))
 for group,n in [('guidance8m',8388608),('guidance16m',16777216)]:
  for path in sorted((R/group).glob('*/feature_scores.json')):
   meta=json.loads((path.parent/'evaluation.json').read_text())
   rows.append((n,path.parent.name+f' (seed{meta["seed"]})',json.loads(path.read_text())))
 for n,mode,m in sorted(rows,key=lambda row:(row[0],row[2]['clean_fid'])):
  parts.append(f'<tr><td>{n:,}</td><td>{html.escape(mode)}</td><td>{m["clean_fid"]:.3f}</td><td>{m["kid_mean"]:.5f}</td></tr>')
 parts.append('</table><p>评估1024生成/1000真实，Heun50。引导只作用于t∈[0.1,0.8]；guide4m/8m分别指同一次训练的4,194,304/8,388,608张EMA检查点，g为强度。不同guide和强度属于采样消融，未修改训练。全部展示连续64张样本，无挑选。验证图用于VAE检查点选择与采样配方比较，因此不是完全独立的测试集；1024样本FID也有采样波动。</p>')
 paths=sorted((R/'eval').glob('*/*/samples64.png'))
 paths+=sorted((R/'guidance8m').glob('*/samples64.png'))+sorted((R/'guidance16m').glob('*/samples64.png'))
 for path in paths:
  parts+=['<h2>'+html.escape(str(path.parent.relative_to(R)))+'</h2>','<img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">']
 for path in sorted((R/'train').glob('samples*.png'),key=lambda p:int(p.stem.split('_')[-1])):
  parts+=['<h2>无引导训练预览 '+path.stem.split('_')[-1]+'</h2>','<img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">']
 (ROOT/'reports/vae_latent96_2026-10-25.html').write_text(''.join(parts))
if __name__=='__main__':main()
