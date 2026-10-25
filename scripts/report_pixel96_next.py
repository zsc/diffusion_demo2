"""Summarize prior unfiltered evidence, compiler adoption, and follow-up status."""
import html,json
from pathlib import Path
root=Path('runs/pixel96_next_20261025');plan=json.loads((root/'plan.json').read_text());gate=json.loads((root/'compile_gate.json').read_text());prior=json.loads(Path('runs/pixel96_noise_20261025/evaluation_status.json').read_text())
status=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else dict(tasks=json.loads((root/'tasks.json').read_text()),quality={})
labels={'control':'当前配方继续','cleaner':'更偏低噪声','wider':'拓宽噪声分布','lower_lr':'降低学习率'}
evidence=[]
for tag in ['continue_lr5e5','finetune_lr2e5','more_low_noise','more_high_noise']:
 vals=[prior['quality'][f'{tag}_16384000_g2_seed{seed}']['clean_fid'] for seed in [3333,7777]];evidence.append(dict(tag=tag,fid_seed3333=vals[0],fid_seed7777=vals[1]))
result=dict(plan=plan,compile_gate=gate,previous_unfiltered_evidence=evidence,status=status)
(root/'review.json').write_text(json.dumps(result,indent=2)+'\n')
table1=''.join(f"<tr><td>{x['tag']}</td><td>{x['fid_seed3333']:.4f}</td><td>{x['fid_seed7777']:.4f}</td></tr>" for x in evidence)
table2=''
for t in status['tasks']:
 c=plan['arms'][t['tag']];table2+=f"<tr><td>{labels[t['tag']]}</td><td>{c['time_logit_mean']}</td><td>{c['time_logit_std']}</td><td>{c['lr']}</td><td>{t['task_id']}</td><td>{html.escape(t['state'])}</td></tr>"
page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>96px低噪声模型：编译加速与下一轮实验</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#203044;background:#f7f9fc}p{line-height:1.7}table{border-collapse:collapse;background:white;width:100%;margin:24px 0}td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd}th{background:#e4edf8}img{width:100%;max-width:784px}</style><h1>96px 低噪声分支：加速与后续实验</h1><p>2026-10-25。best-of-N和API评分保持暂停。下列比较均为未筛选样本。</p><h2>上一轮结果</h2><p>同一模型家族、guidance2、两个固定噪声种子；每次1024生成图和1000验证图，属于小样本筛选指标。</p><table><tr><th>路线</th><th>FID seed3333</th><th>FID seed7777</th></tr>'''+table1+'''</table><h2>真实16卡加速验证</h2>'''+f"<p>局部融合 {gate['benchmarks']['compile_pointwise']['median_images_per_second']:.0f} → 整模型编译 {gate['benchmarks']['compile_full']['median_images_per_second']:.0f} 张/秒（+{(gate['speedup']-1)*100:.1f}%）。同一真实checkpoint续训128步，权重/EMA/损失误差均通过预设阈值，16rank随机状态完全一致，没有跳步。额外编译启动开销未计入稳态吞吐。</p>"+'''<h2>下一轮四组</h2><p>全部继承最佳低噪声分支的完整状态，不重置优化器或EMA。有效batch1024，每组追加8.192M图像，采用整模型编译；每个任务使用1×16GPU。只改变表中单个因素，中途4.096M和最终checkpoint均有无筛选评估。</p><table><tr><th>实验</th><th>时间logit均值</th><th>标准差</th><th>学习率</th><th>任务</th><th>状态</th></tr>'''+table2+'''</table><p>时间t=0表示噪声，t=1表示干净图像，因此更高均值增加较低噪声训练样本；更大标准差同时增加两端覆盖。新实验的效果须等评估完成后判断。</p><h2>当前最佳分支固定种子无筛选样张</h2><img src="../runs/pixel96_noise_20261025/previews/more_low_noise_16384000_g2_seed3333/independent_seed_grid.png"><p>原始生成预览，未做best-of-N或图像增强。</p></html>'''
Path('reports/pixel96_next_20261025.html').write_text(page)
print('report updated')
