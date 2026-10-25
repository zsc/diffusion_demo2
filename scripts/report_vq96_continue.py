"""Actual 96px continuation checkpoints, evaluations and samples."""
import json,base64,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'runs/vq96_continue_20261025'
def main():
 if not (R/'status.json').exists():return
 s=json.loads((R/'status.json').read_text());t=s['task'];p=s.get('train/progress.json',{})
 parts=['<!doctype html><meta charset="utf-8"><title>96px 最佳分支续训</title><style>body{font:16px/1.7 system-ui;max-width:1160px;margin:30px auto;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}img{width:100%}</style><h1>96px 最佳分支：4M → 8M → 16M</h1>',f'<p>任务 {t["task_id"]} · {html.escape(t["state"])} · 已训练 {p.get("images_seen",4194304):,} 张96px图像 · 每组1×16 GPU</p>', '<p>batch2048 / lr2e-4。恢复完整模型、EMA、optimizer及每卡随机状态，保持48px继承的PCA basis/offset/center/scale。不加GAN。4M有引导FID85.17；另一个lr1e-4分支8M为82.78（seed3333），82.54（seed7777）。本次同分辨率、同数据、相同采样协议。</p><table><tr><th>新增96px训练图像</th><th>采样</th><th>FID</th><th>KID</th></tr>']
 for name,m in sorted(s['evaluations'].items()):
  count,mode=name.split('/');parts.append(f'<tr><td>{int(count):,}</td><td>{html.escape(mode)}</td><td>{m["clean_fid"]:.3f}</td><td>{m["kid_mean"]:.5f}</td></tr>')
 parts.append('</table><p>评估：1024生成/1000真实，Heun50；guided=2，固定使用52万张弱模型引导。所有样本按原顺序展示，无筛选。旧48px预训练数据缺少原图ID，与96px验证集的重叠未知。</p>')
 for path in sorted((R/'eval').glob('*/*/samples64.png')):
  parts+=['<h2>'+html.escape(str(path.parent.relative_to(R/'eval')))+'</h2>','<img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">']
 for path in sorted((R/'train').glob('samples*.png'),key=lambda p:int(p.stem.split('_')[-1])):
  parts+=['<h2>无引导训练预览 '+path.stem.split('_')[-1]+'</h2>','<img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">']
 out=ROOT/'reports/vq96_continue_2026-10-25.html';out.write_text(''.join(parts))
if __name__=='__main__':main()
