"""Render measured 96px transfer recipe comparisons and unselected samples."""
import base64,html,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/vq96_variants_20261025';BASE=ROOT/'runs/vq96_transfer_20261025'
def load(p):return json.loads(p.read_text()) if p.exists() else {}
def embed(p):return '<img src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'">'
def main():
 state=load(RUN/'status.json');baseline=load(BASE/'status.json');out=ROOT/'reports';out.mkdir(exist_ok=True)
 parts=['<!doctype html><meta charset="utf-8"><title>96px latent DiT 变体对照</title><style>body{font:16px/1.7 system-ui;max-width:1280px;margin:30px auto;padding:0 20px;color:#172331}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}img{width:100%}.grids{display:grid;grid-template-columns:1fr 1fr;gap:20px}.note{padding:15px;background:#eef3f8}</style><h1>96×96 VQ latent DiT：学习率与大 batch</h1><p>每组1×16 GPU；2026-10-25</p><p class="note">所有组继承同一份48px/32M PCA8 EMA；编码器、码本与PCA坐标不变，使用同一个微调后的96px MSE解码器。先比较新增4,194,304张96px图像；主线另继续到8,388,608张。除batch和学习率外配置一致。两种batch的优化步数不同，因此同图像预算并不代表同优化步数。</p>', '<table><tr><th>组</th><th>Batch</th><th>LR</th><th>任务</th><th>状态</th><th>已训练图像</th></tr>']
 records={'baseline_b2048_lr1e4':dict(task=baseline.get('task',{}),config=dict(batch_size=2048,lr=1e-4),**{'train/progress.json':baseline.get('latent/progress.json',{}),'eval':state.get('baseline4m',{})})}
 records.update({k:v for k,v in state.items() if 'task' in v})
 for name,d in records.items():
  t=d['task'];c=d['config'];p=d.get('train/progress.json',{});parts.append(f'<tr><td>{html.escape(name)}</td><td>{c["batch_size"]}</td><td>{c["lr"]:g}</td><td>{t.get("task_id","—")}</td><td>{t.get("state","—")}</td><td>{p.get("images_seen",0):,}</td></tr>')
 parts+=['</table><h2>4M图像预算下的实际评估</h2><p>固定seed3333，1024生成/1000真实，Heun50；guided=2，使用各自52万张早期模型作为弱引导。因此先用raw列判断训练配方，再观察引导后的效果。数值越低越好；小样本FID差距需独立种子确认。</p><table><tr><th>组</th><th>Raw FID</th><th>Guided FID</th><th>Guided KID</th></tr>']
 for name,d in records.items():
  e=d.get('eval',{});a=e.get('raw_seed3333',{});b=e.get('guided_seed3333',{});fmt=lambda x:'—' if x is None else f'{x:.4f}'
  parts.append('<tr><td>'+html.escape(name)+'</td><td>'+fmt(a.get('clean_fid'))+'</td><td>'+fmt(b.get('clean_fid'))+'</td><td>'+fmt(b.get('kid_mean'))+'</td></tr>')
 parts+=['</table><h2>全部连续样本，无挑选</h2><p>训练预览固定seed2027、无引导。相同步数标签对应相同新增图像预算；所有模型维度相同，可按格对应观察。</p>']
 for count in [524288,2097152,4194304]:
  panels=[]
  for name,d in records.items():
   p=(BASE/'latent'/f'samples_{count}.png') if name.startswith('baseline_') else RUN/name/'train'/f'samples_{count}.png'
   if p.exists():panels.append('<section><h3>'+html.escape(name)+'</h3>'+embed(p)+'</section>')
  if panels:parts+=['<h2>'+f'{count:,}'+'张</h2><div class="grids">'+''.join(panels)+'</div>']
 for name in records:
  folder=RUN/'baseline4m' if name.startswith('baseline_') else RUN/name
  for mode in ['raw_seed3333','guided_seed3333']:
   p=folder/'eval'/mode/'samples64.png'
   if p.exists():parts+=['<h3>'+html.escape(name+' / '+mode)+'</h3>',embed(p)]
 parts.append('<p>96px为原始CelebA JPEG裁剪后的原生分辨率，非48px放大。旧48px预训练数据缺原图ID，与96px验证集的重叠未知。不同分辨率FID不可直接比较。</p>')
 (out/'vq96_variants_2026-10-25.html').write_text(''.join(parts));(out/'vq96_variants_2026-10-25.json').write_text(json.dumps(state,indent=2))
if __name__=='__main__':main()
