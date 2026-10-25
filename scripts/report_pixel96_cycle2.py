# coding: utf-8
"""Review four completed arms and their evidence-based continuation/replacements."""
import html,json
from pathlib import Path
root=Path('runs/pixel96_cycle2_20261025');plan=json.loads((root/'plan.json').read_text())
prior=json.loads(Path('runs/pixel96_next_20261025/status.json').read_text());q=prior['quality']
parent=json.loads(Path('runs/pixel96_noise_20261025/evaluation_status.json').read_text())['quality']
rows=[dict(tag='parent',name='上一轮最佳',g1=63.5009,g2seed3333=parent['more_low_noise_16384000_g2_seed3333']['clean_fid'],g2seed7777=parent['more_low_noise_16384000_g2_seed7777']['clean_fid'],decision='保留比较基准')]
labels={'control':'原配方继续','cleaner':'更低噪声','wider':'更宽噪声分布','lower_lr':'降低学习率'}
decisions={'control':'基本持平，替换','cleaner':'两个种子均改善，继续','wider':'g2退步；g3仍不如当前最佳，替换','lower_lr':'没有稳定提升，替换'}
for tag,name in labels.items():
 rows.append(dict(tag=tag,name=name,g1=q[f'{tag}_24576000_g1_seed3333']['clean_fid'],g2seed3333=q[f'{tag}_24576000_g2_seed3333']['clean_fid'],g2seed7777=q[f'{tag}_24576000_g2_seed7777']['clean_fid'],decision=decisions[tag]))
status=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else dict(tasks=json.loads((root/'tasks.json').read_text()))
result=dict(completed_comparison=rows,wider_guidance3={str(seed):q[f'wider_24576000_g3_seed{seed}']['clean_fid'] for seed in [3333,7777]},plan=plan,status=status,figures=['reports/pixel96_next_comparison_seed3333_20261025.png','reports/pixel96_next_comparison_seed7777_20261025.png'],visual_observation='Matched fixed first8 samples per seed: modest differences, some clearer facial details in cleaner, artifacts remain. Wider with g3 makes edges/highlights harsher. No image selection, enhancement or external judge.')
(root/'review.json').write_text(json.dumps(result,indent=2)+'\n')
body=''.join(f"<tr><td>{r['name']}</td><td>{r['g1']:.2f}</td><td>{r['g2seed3333']:.2f}</td><td>{r['g2seed7777']:.2f}</td><td>{r['decision']}</td></tr>" for r in rows)
newlabels={'continue_cleaner':'继续最佳 cleaner','balanced':'折中均值','narrow':'更窄分布','stratified':'分层时间采样'}
newrows=''
for t in status['tasks']:
 c=plan['arms'][t['tag']];newrows+=f"<tr><td>{newlabels[t['tag']]}</td><td>{c['time_logit_mean']}</td><td>{c['time_logit_std']}</td><td>{c['time_sampling']}</td><td>{t['task_id']}</td><td>{html.escape(t['state'])}</td></tr>"
page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>96px四组实验复盘与继续训练</title><style>body{font:16px system-ui;max-width:1300px;padding:24px;margin:auto;background:#f7f9fc;color:#203044}p{line-height:1.7}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd}th{background:#e3ebf6}img{width:100%;margin-top:20px}</style><h1>96px 四组实验复盘与下一轮</h1><p>2026-10-25。四组训练和评估全部完成。统一固定种子、Heun50、guide区间[0.1,0.8]，没有best-of-N或API评分。FID越低越好。</p><table><tr><th>实验</th><th>无guidance FID</th><th>g2 / seed3333</th><th>g2 / seed7777</th><th>决定</th></tr>'''+body+'''</table><p>cleaner在两个g2种子都有改善，但无guidance质量回退，因此它是当前有guidance采样的最佳候选，不代表所有采样设置都改善。宽噪声补测g3得到54.79/54.45，仍逊于cleaner的g2。每次1024生成图与1000验证图，属于筛选评估，不是FID50k或统计显著性证明。</p><h2>固定样张比较</h2><p>每行固定前8张，不挑图。英文行名对应上表；seed3333最后一行为宽噪声g3。原生96px，仅最近邻放大显示。</p><img src="pixel96_next_comparison_seed3333_20261025.png"><img src="pixel96_next_comparison_seed7777_20261025.png"><h2>已提交的继续与替换实验</h2><table><tr><th>实验</th><th>均值</th><th>标准差</th><th>采样方式</th><th>任务</th><th>状态</th></tr>'''+newrows+'''</table><p>四组都从cleaner完整checkpoint（计数24.576M）继续，保留优化器和EMA，各追加8.192M图像，每组1×16GPU。保留中途4.096M checkpoint并自动评估；没有延长已持平的独立旧分支。</p><p>全部采用已验证整模型编译，16卡稳态约1.31万张/秒，比旧局部融合提高18.3%。新分层时间采样每个batch覆盖不同CDF区间，保持批均值目标不变；通过覆盖、随机状态和双进程精确断点续训测试，GPU运行前再做有限值检查。</p></html>'''
Path('reports/pixel96_cycle2_review_20261025.html').write_text(page)
print(json.dumps(rows,ensure_ascii=False,indent=2))
