"""Native96 directly trained narrow continuous VAE experiment report."""
from pathlib import Path
import json,base64
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'runs/vae_narrow96_20261025'
def main():
 if not (R/'status.json').exists():return
 s=json.loads((R/'status.json').read_text());parts=['<!doctype html><meta charset="utf-8"><title>原生96px窄latent VAE</title><style>body{font:16px/1.7 system-ui;max-width:1200px;margin:30px auto;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}img{width:100%}</style><h1>直接训练 8 / 12 / 16 通道普通 VAE</h1><p>2026-10-25 · 每组1×16 GPU · 原生96px · 不使用PCA</p><p>固定12×12空间网格、encoder宽度[64,128]；从零训练128轮，每轮58个global batch512，共3,801,088次看图。lr2e-4，求和重建ELBO的KL beta=0.01，8轮KL预热，模型EMA半衰期8192张。30,000训练图/1,000验证图，三组同划分与seed。每组先做独立16 GPU两步smoke，再从零正式训练。</p><p>训练按后验采样优化，检查点按验证集后验均值重建MSE选择。完成后对最佳与最后权重分别检查mu和固定后验噪声的PSNR/LPIPS。这里是重建，不是无条件生成；这个验证集同时用于选检查点，不是独立测试集。</p><table><tr><th>通道</th><th>任务</th><th>状态</th><th>smoke</th><th>轮数</th><th>最新PSNR</th><th>最佳PSNR</th></tr>']
 for name,d in s.items():
  latest=d.get('latest',{});best=d.get('best',{});fmt=lambda x:'—' if x is None else f'{x:.3f}'
  parts.append(f'<tr><td>{name}</td><td>{d["task"]["task_id"]}</td><td>{d["task"]["state"]}</td><td>{d["smoke_passed"]}</td><td>{latest.get("epoch",0)}/128</td><td>{fmt(latest.get("val_psnr_db"))}</td><td>{fmt(best.get("val_psnr_db"))}</td></tr>')
 parts.append('</table><h2>完成后的重建评估</h2><table><tr><th>模型</th><th>权重</th><th>latent</th><th>PSNR</th><th>LPIPS</th></tr>')
 for name,d in s.items():
  for which,e in d['evaluations'].items():
   for mode in ['mu','sample']:parts.append(f'<tr><td>{name}</td><td>{which}</td><td>{mode}</td><td>{e[mode]["psnr"]:.3f}</td><td>{e[mode]["lpips"]:.5f}</td></tr>')
 parts.append('</table>')
 for name in s:
  for path in sorted((R/name/'eval').glob('*/*.png')):
   parts.extend(['<h3>'+name+' / '+str(path.relative_to(R/name))+'</h3>','<img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">'])
  previews=sorted((R/name/'train').glob('reconstruction*.png'))
  if previews:
   path=previews[-1];parts.extend(['<h3>'+name+' '+path.stem+'：原图与后验均值重建</h3>','<img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">'])
 (ROOT/'reports/vae_narrow96_2026-10-25.html').write_text(''.join(parts))
if __name__=='__main__':main()
