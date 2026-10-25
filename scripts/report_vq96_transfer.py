"""Report only actual native 96px VQ/DiT training artifacts."""
import base64,html,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/vq96_transfer_20261025'
def read(p):return json.loads(p.read_text())
def embed(p):return '<img src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'">'
def main():
 out=ROOT/'reports';out.mkdir(exist_ok=True);state=read(RUN/'status.json') if (RUN/'status.json').exists() else {}
 parts=['<!doctype html><meta charset="utf-8"><title>原生 96px VQ latent DiT 迁移</title><style>body{font:16px/1.7 system-ui;max-width:1176px;margin:32px auto;padding:0 22px;color:#172331}img{width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:8px;text-align:left}.note{background:#eef3f8;padding:16px}</style><h1>从最佳 48px PCA8 分支继承到原生 96×96</h1><p>2026-10-25 · 每个实验 1×16 GPU · 不加 GAN</p><p class="note">选择 PCA8 / batch2048：四组同口径 FID 为 56.75、57.23（64D/32M）、59.76（PCA8/b512）、63.44（感知 VQ）。所选分支随后续训至 33,554,432 张，seed3333 FID 54.95、seed7777 FID 56.01。96px 使用这份更优 EMA 权重。</p>',
 '<p>VQVAE：48→96 输出、12×12→24×24 latent，编码器与码本权重严格相同；先单独微调 MSE 解码器 4096×512 张。DiT：patch1 tokens 从144增至576，仅插值 learned position embedding，其余权重不变。PCA basis/offset/center/scale 全部继承；重置 optimizer 和 96px 图像计数。高分辨率首轮 batch2048、lr1e-4，新增 8,388,608 张训练图像。</p>',
 '<p>数据来自原始 CelebA JPEG 的真实 96px 缓存，共30,000训练 + 1,000验证，非48px放大。新分区原图ID不交叉，但旧48px sprites没有原图ID，无法排除旧预训练接触新验证图。48px与96px FID不直接比较。</p>']
 t=state.get('task',{});p=state.get('latent/progress.json',{});parts.append('<h2>运行状态</h2><p>任务 '+str(t.get('task_id','—'))+' · '+html.escape(t.get('state','尚未记录'))+' · 新增96px图像 '+str(p.get('images_seen',0))+'</p>')
 initial=state.get('decoder/initial.json');best=state.get('decoder/best.json')
 if initial and best:
  parts.append(f'<p>同一96px验证集，解码器微调前→最佳：PSNR {initial["psnr"]:.3f}→{best["psnr"]:.3f} dB；LPIPS {initial["lpips"]:.5f}→{best["lpips"]:.5f}；码本使用 {best["codes_used"]}/512。</p>')
 for label,path in [('96px原图/重建交替图',RUN/'decoder/reconstructions.png')]:
  if path.exists():parts+=['<h2>'+label+'</h2>',embed(path)]
 for path in sorted((RUN/'latent').glob('samples_*.png'),key=lambda p:int(p.stem.split('_')[-1])):
  count=int(path.stem.split('_')[-1]);parts+=['<h2>新增训练 '+f'{count:,}'+' 张后的连续64个样本</h2><p>seed2027，Heun50，无引导；原生96×96，显示网格仅使用最近邻放大。</p>',embed(path)]
 if state.get('evaluations'):
  parts.append('<h2>完成后的生成质量评估</h2><table><tr><th>实验</th><th>FID</th><th>KID</th></tr>')
  for name,m in state['evaluations'].items():parts.append(f'<tr><td>{html.escape(name)}</td><td>{m["clean_fid"]:.3f}</td><td>{m["kid_mean"]:.5f}</td></tr>')
  parts.append('</table>')
 parts.append('<p>代码与元数据：latent_ddp_experiment.py --init-checkpoint；configs/latent_vq96_pca8_transfer.json；runs/vq96_transfer_20261025/。完成后自动回收指标和样本。</p>')
 (out/'vq96_transfer_2026-10-25.html').write_text(''.join(parts));(out/'vq96_transfer_2026-10-25.json').write_text(json.dumps(state,indent=2))
if __name__=='__main__':main()
