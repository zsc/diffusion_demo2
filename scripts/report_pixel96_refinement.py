"""Render measured refinement, judge selection and follow-up task evidence."""
import base64
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    old = ROOT / 'runs/pixel96_bestofn_20261025'
    new = ROOT / 'runs/pixel96_refine_bestofn_20261025'
    status = json.loads((ROOT / 'runs/pixel96_refine_20261025/review_status.json').read_text())
    quality = status['eval']
    rows = []
    for tag in ['baseline', 'b1024_lr5e5', 'b1024_lr1e4', 'b2048_lr5e5', 'b2048_lr1e4']:
        values = []
        for g in [2, 3]:
            label = f'baseline_g{g}' if tag == 'baseline' else f'{tag}_8192000_g{g}'
            values.append(f"{quality[label]['clean_fid']:.3f}" if label in quality else 'pending')
        rows.append(f'<tr><td>{tag}</td>'+''.join(f'<td>{v}</td>' for v in values)+'</tr>')
    sections = ['<h2>96px 训练：未筛选 FID</h2><p>相同 seed3333、Heun50、早期 guide、区间[0.1,0.8]，1024 fake /1000 real。追加训练8.192M；同一验证集上的小样本选择比较，不能当作 FID50k 或独立测试结果。</p><table><tr><th>模型</th><th>g2</th><th>g3</th></tr>'+''.join(rows)+'</table>']
    scoring = {}
    for title, root in [('旧模型采样变体',old),('新模型 guidance2 候选',new)]:
        rows=[]
        for p in sorted(root.glob('*/selection.json')):
            if p.parent.name=='pilot64':continue
            s=json.loads(p.read_text());scoring[str(p.relative_to(ROOT))]=s['selections']
            scores=json.loads((p.parent/'scores.json').read_text())
            mean=sum(x['score'] for x in scores)/len(scores)
            rows.append(f'<tr><td>{html.escape(p.parent.name)}</td><td>{len(scores)}</td><td>{mean:.2f}</td>'+''.join(f"<td>{s['selections'][str(n)]['mean_judge_score']:.2f}</td>" for n in [1,2,4,8])+'</tr>')
        sections.append(f'<h2>{title}</h2><table><tr><th>配置</th><th>候选数</th><th>全池均分</th><th>N1</th><th>N2</th><th>N4</th><th>N8</th></tr>'+''.join(rows)+'</table>')
    sections.append('<p>视觉模型分数满分100，按技术视觉缺陷评分。N1/2/4/8在固定每8张一组中选择，全部候选和缓存保留。筛选提高同一评分器分数是选择效应，不能等同于独立质量证明或 FID 改善。</p>')
    for filename,title in [('pixel96_refine_guided_comparison_20261025.png','新旧模型：相同 seed、g2，未筛选前8张'),('pixel96_refine_raw_comparison_20261025.png','四组训练：相同 seed2026，无 guidance，未筛选前8张'),('pixel96_bestofn_comparison_20261025.png','旧模型采样：固定前8个候选组，N1 与 N8'),('pixel96_refine_bestofn_comparison_20261025.png','新旧模型：g2，前8个候选组 N1 与 N8')]:
        p=ROOT/'reports'/filename
        if p.exists():sections.append(f'<h2>{title}</h2><img src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}">')
    tasks=[t for t in status['tasks'] if t['group']=='round96_noise_20261025']
    sections.append('<h2>后续实验</h2><p>前四组从 b1024_lr5e5 的完整8.192M checkpoint继承模型、EMA、优化器和RNG；继续到16.384M。对照lr5e-5、lr2e-5、logit-time均值-0.4（偏低噪声）、-1.2（偏高噪声）。标准差均0.8，验证分布固定为原来的均值-0.8、标准差0.8。不同训练分布的train loss不可直接横向比较。另将 best-of-N 最高分的 b2048_lr5e5 从其完整 checkpoint 续训到16.384M，排队接续空闲节点。</p><pre>'+html.escape(json.dumps(tasks,ensure_ascii=False,indent=2))+'</pre>')
    if (ROOT/'runs/best_of_n_paused_20261025.json').exists():
        sections[-1] = sections[-1].replace(
            '另将 best-of-N 最高分的 b2048_lr5e5 从其完整 checkpoint 续训到16.384M，排队接续空闲节点。',
            '用户已暂停 best-of-N；专为该方向追加的任务59190已取消，已完成产物保留。')
        sections.insert(0, '<p><strong>当前状态：best-of-N 与视觉模型评分已暂停。资源用于普通训练和噪声分布变体；下方评分是保留的历史结果。</strong></p>')
    p=ROOT/'reports/pixel96_refinement_review_20261025.html'
    p.write_text('<!doctype html><meta charset="utf-8"><title>96px refinement review</title><style>body{max-width:1350px;margin:32px auto;padding:0 24px;font:16px/1.6 system-ui;color:#203040}table{border-collapse:collapse}td,th{padding:7px 16px;border-bottom:1px solid #ddd;text-align:left}img{max-width:100%}pre{white-space:pre-wrap}h2{margin-top:36px}</style><h1>96×96 训练与 best-of-N 对比</h1>'+''.join(sections))
    p.with_suffix('.json').write_text(json.dumps(dict(status=status,judge=scoring),ensure_ascii=False,indent=2)+'\n')
    print(p)

if __name__=='__main__':main()
