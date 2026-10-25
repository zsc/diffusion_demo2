"""Report the recorded single-device short benchmarks without mixing precisions."""
import html,json,statistics
from pathlib import Path
root=Path('runs/compile_compare_20261025');rows=[]
for group in ['local','local_tuning','gpu','gpu_tuning','gpu_final']:
 for p in sorted((root/group).glob('*.json')):
  d=json.loads(p.read_text())
  if not isinstance(d,dict) or 'median_images_per_second' not in d:continue
  times=[r['images_per_second'] for r in d['measurements']]
  rows.append(dict(group=group,file=str(p.resolve()),resolution=d['resolution'],device=d['device'],precision=d['precision'],microbatch=d['microbatch'],global_batch=d['global_batch'],execution=d['execution'],optimizer=d['optimizer'],images_per_second=d['median_images_per_second'],min_images_per_second=min(times),max_images_per_second=max(times),peak_gib=d['peak_allocated_mib']/1024,warmup_seconds=d['warmup_seconds'],gradient_error=d['numerical_check']['gradient_relative_l2'],passed=d['numerical_check']['passed'],torch=d['torch'],cuda=d['cuda'],source_sha256=d['source_sha256']))
def row(group,res,execution,micro,precision='fp16',optimizer='default'):
 return next(r for r in rows if r['group']==group and r['resolution']==res and r['execution']==execution and r['microbatch']==micro and r['precision']==precision and r['optimizer']==optimizer)
comparisons=[]
for res,micro in [(48,64),(96,32)]:
 for ex in ['eager','compile_pointwise']:
  a=row('local',res,ex,micro);b=row('gpu',res,ex,micro)
  comparisons.append(dict(resolution=res,precision='fp16',microbatch=micro,execution=ex,local=a['images_per_second'],gpu=b['images_per_second'],gpu_over_local=b['images_per_second']/a['images_per_second']))
for res,micro,group in [(96,32,'gpu_tuning'),(96,64,'gpu_final'),(48,64,'gpu_final')]:
 try:
  a=row('local_tuning' if micro==64 and res==96 else 'local',res,'compile_full',micro)
  b=row(group,res,'compile_full',micro)
 except StopIteration:continue
 comparisons.append(dict(resolution=res,precision='fp16',microbatch=micro,execution='compile_full',local=a['images_per_second'],gpu=b['images_per_second'],gpu_over_local=b['images_per_second']/a['images_per_second']))
best_local={res:max((r for r in rows if r['group'].startswith('local') and r['resolution']==res and r['passed']),key=lambda x:x['images_per_second']) for res in [48,96]}
estimates={label:dict(compute_hours=8192000/best_local[48]['images_per_second']/3600+images96/best_local[96]['images_per_second']/3600,images48=8192000,images96=images96) for label,images96 in [('old_best',20480000),('shown_refinement',28672000),('latest_noise_branch',36864000)]}
result=dict(best_local=best_local,estimated_reproduction=estimates,scope='Short single-device synthetic-data training-step benchmark; no saved training model or full reproduction.',protocol='Same 21.891M parameter PixelJiT source, global batch1024, FP32 parameters/AdamW states, resident uint8 bank, augmentation + forward/backward + gradient accumulation + clipping + AdamW + EMA. Three warmup updates and five timed updates per variant, synchronized. Compilation/warmup excluded from throughput. No DDP, checkpoint I/O, validation or sample generation in timing.',rows=rows,matched_fp16_comparisons=comparisons,limitations=['The desktop GPU and server GPU use different PyTorch/backend versions. Same RNG seed does not guarantee identical synthetic tensors across devices.','Numerical gates compare identical warmed weights/inputs within each device; reference batches above32 are accumulated in chunks to fit memory. This is not long-run convergence validation.','Warmup seconds include three updates and cold-start costs, not pure compiler time.','CLUSTER reserved16GPU per task, but each benchmark process used only visible device0. Reported GPU throughput is directly measured single-device performance.'])
(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
headers=['设备','分辨率','精度','microbatch','模式','优化器','张/秒','显存 GiB','梯度误差 %','检查']
body=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [r['device'],r['resolution'],r['precision'],r['microbatch'],r['execution'],r['optimizer'],f"{r['images_per_second']:.1f}",f"{r['peak_gib']:.2f}",f"{r['gradient_error']*100:.4f}",'PASS' if r['passed'] else 'FAIL'])+'</tr>' for r in rows)
page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>2080 SUPER / GPU 编译短测</title><style>body{font:16px system-ui;max-width:1250px;margin:40px auto;padding:0 20px;background:#f7f9fc;color:#203044}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{padding:10px;border-bottom:1px solid #ddd;text-align:left}th{background:#e5edf7}p{line-height:1.7}</style><h1>2080 SUPER / 单 GPU 编译与融合短测</h1><p>2026-10-25 · 有效 batch1024 · 含数据增强、前向、反向、梯度累积、裁剪、AdamW 和 EMA。仅短测，没有启动正式训练。</p><table><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr>'+body+'</table><p>'+html.escape(result['protocol'])+'</p>'+''.join('<p>'+html.escape(x)+'</p>' for x in result['limitations'])+'</html>'
Path('reports/pixel_compile_comparison_20261025.html').write_text(page)
print(json.dumps(comparisons,indent=2))
