"""Render four trained candidates with identical heldout/sample settings."""
import base64,html,json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/vq_perceptual_20261025'
NAMES=['pca8_b2048_16m','pca8_16m','base32m','joint16m']
LABELS={'pca8_b2048_16m':'PCA8 / batch 2048 / 16M','pca8_16m':'PCA8 / batch 512 / 16M','base32m':'64D baseline / batch 512 / 32M','joint16m':'Joint LPIPS VQ / batch 512 / 16M'}
def read(p):return json.loads(p.read_text())
def embed(p):return '<img src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'">'
def main():
 out=ROOT/'reports';out.mkdir(exist_ok=True)
 rows=[];panel=Image.new('RGB',(1200,4*350),'#edf2f6');draw=ImageDraw.Draw(panel)
 try:font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',23)
 except OSError:font=ImageFont.load_default()
 for row,name in enumerate(NAMES):
  score=RUN/'eval'/name/'feature_scores.json';metric=read(score) if score.exists() else None
  grid=RUN/'eval'/name/'samples64.png';kind='GPU / BF16'
  if not grid.exists():grid=RUN/'comparison_samples'/name/'grid.png';kind='CPU / FP32; GPU evaluation pending'
  label=LABELS[name]+(f" | FID {metric['clean_fid']:.2f}" if metric else ' | FID pending / CPU sample')
  draw.text((12,row*350+8),label,fill='#172331',font=font)
  if grid.exists():
   source=Image.open(grid);assert source.size==(1200,1200)
   # Always first 16 samples in their original order, with nearest-neighbor enlargement only.
   panel.paste(source.crop((0,0,1200,300)),(0,row*350+42))
  cfg=read(RUN/'parallel4'/name/'config.json');result=read(RUN/'parallel4'/name/'result.json');progress=read(RUN/'parallel4'/name/'progress.json')
  rows.append(dict(name=name,label=LABELS[name],metrics=metric,config=cfg,result=result,images_per_second=progress['images_per_second'],sample_backend=kind,grid=str(grid)))
 panelpath=out/'vq_parallel_comparison_2026-10-25.png';panel.save(panelpath)
 parts=['<!doctype html><meta charset="utf-8"><title>四组 latent DiT 对比</title><style>body{font:16px/1.65 system-ui;color:#172331;max-width:1200px;margin:32px auto;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;text-align:left;padding:9px}img{width:100%;height:auto}.note{background:#eef3f8;padding:15px}</style><h1>四组 latent DiT：样本与续训选择</h1><p>2026-10-25 · 每组独占 1×16 GPU · 四组训练均已完成</p><p class="note">统一 seed3333、Heun 50、guidance2（t 0.1–0.8），不重新量化。FID/KID 为相同 GPU/BF16 流程，1024 个生成样本与固定 1000 张验证图。PCA8 和 64D 使用不同维数的噪声，不能把跨架构同编号的人脸当作身份匹配对照。CPU 补充图使用不同后端的随机数实现，也不用于逐张配对；最终表格的 FID 统一取自 GPU。</p><table><tr><th>方法</th><th>累计训练图像</th><th>FID ↓</th><th>KID ↓</th><th>本阶段秒数</th><th>图像/秒</th></tr>']
 for x in rows:
  m=x['metrics'];parts.append('<tr><td>'+html.escape(x['label'])+'</td><td>'+f"{x['config']['images']:,}"+'</td><td>'+(f"{m['clean_fid']:.3f}" if m else '待评估')+'</td><td>'+(f"{m['kid_mean']:.5f}" if m else '—')+'</td><td>'+f"{x['result']['seconds']:.1f}"+'</td><td>'+f"{x['images_per_second']:,.0f}"+'</td></tr>')
 parts+=['</table><p>PCA8/b512 与感知分支本阶段由 4M 续训至 16M，基线由 16M 至 32M；b2048 是从零至 16M。时间不含排队和启动；大 batch 同时使用不同学习率与训练路径，不能把质量差异单独归因于 batch。</p><h2>连续前 16 张样本</h2>',embed(panelpath)]
 selection=RUN/'followup_selection.json'
 if selection.exists():
  q=read(selection);parts+=['<h2>续训选择</h2><p>'+html.escape(q['reason'])+'</p><p>'+html.escape(q['training_plan'])+'</p>']
  parts+=['<p>PCA8/b2048 与 32M 基线的 FID 仅相差约 0.48，不能据此断言统计显著优势。优先续训 PCA8/b2048 同时考虑了更少的已训练图像数、更高的实测吞吐和样本质量；仍有模糊与五官不对称，后续看两个采样种子的结果。</p>']
  status=RUN/'followup_status.json'
  if status.exists():
   state=read(status);parts+=['<p>最近记录状态：'+html.escape(state['task']['state'])+'；已训练图像：'+str(state.get('train/progress.json',{}).get('images_seen','等待启动'))+'</p>']
   for seed in [3333,7777]:
    key=f'eval_seed{seed}/feature_scores.json'
    if key in state:parts+=['<p>续训后 seed '+str(seed)+'：FID '+f"{state[key]['clean_fid']:.3f}"+'</p>']
 for x in rows:
  p=Path(x['grid'])
  if p.exists():parts+=['<h2>'+html.escape(x['label'])+'</h2><p>'+html.escape(x['sample_backend'])+'；连续 64 张，无筛选。</p>',embed(p)]
 (out/'vq_parallel_comparison_2026-10-25.html').write_text(''.join(parts))
 (out/'vq_parallel_comparison_2026-10-25.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
