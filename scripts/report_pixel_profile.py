"""Self-contained report of the actual 96px PPU training-step profile."""
import argparse
import html
import json
import statistics
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='runs/profile96_v1')
    ap.add_argument('--output',default='reports/pixel96_profile_2026-10-24.html')
    args=ap.parse_args();root=Path(args.root);out=Path(args.output)
    q=json.loads((root/'summary.json').read_text())
    versions=json.loads((root/'versions.json').read_text())
    p=next(x for x in q['profiles'] if x['name']=='asys_default')
    baseline=q['benchmarks']['default']['median_ms'];fused=q['benchmarks']['fused']['median_ms']
    labels={'pointwise/reduction/layout':'逐元素 / 归约 / 布局','copy/cast':'复制 / 类型转换','GEMM':'GEMM','attention':'Attention','convolution':'卷积','foreach (optimizer/EMA/clip; unresolved)':'foreach（优化器、EMA、clip 未细分）','other (inspect names/torch attribution)':'其他','normalization':'独立 normalization kernel'}
    rows=[]
    for name,ms in p['categories_ms'].items():
        rows.append(f'<tr><td>{html.escape(labels.get(name,name))}</td><td>{ms:.3f}</td><td>{100*ms/p["kernel_sum_ms"]:.1f}%</td><td><i style="width:{100*ms/p["kernel_sum_ms"]:.1f}%"></i></td></tr>')
    kernels=''.join(f'<tr><td>{k["calls"]}</td><td>{k["total_ms"]:.3f}</td><td><code>{html.escape(k["name"])}</code></td></tr>' for k in p['kernels'])
    fusion_html = ""
    fusion_path = Path("runs/profile96_fusion_v2/summary.json")
    if fusion_path.exists():
        fusion = json.loads(fusion_path.read_text())
        b = fusion["benchmarks"]["eager"]["median_ms"]
        f = fusion["benchmarks"]["compile_pointwise"]["median_ms"]
        fp = fusion["profiles"][0]
        fusion_html = f"<section><h2>融合实测：{b/f:.2f}× 单 PPU 吞吐</h2><p>三个独立进程交错比较，保留中间精度转换（emulate_precision_casts=True）：{b:.3f} → {f:.3f} ms/step，耗时下降 {100*(1-f/b):.1f}%，吞吐增加 {100*(b/f-1):.1f}%。计时不启用 profiler；包含数据增强、前后向、clip、AdamW、EMA。编译与预热在计时外，第一次冷启动仍需额外时间。</p>"
        fusion_html += f"<p>独立 ASight 捕获：2321 → {fp['kernel_count']} 个 kernel，累计设备时间 148.881 → {fp['kernel_sum_ms']:.3f} ms；GEMM 和 attention 时长基本不变。使用 PyTorch Inductor 调用已安装的厂商 Triton，融合 RMSNorm、RoPE、SwiGLU 激活、调制与残差；未重写 GEMM/attention。</p>"
        fusion_html += "<p>数值边界：原版的非确定性 A/A 对照也未通过初始严格门槛；确定性 A/A 可逐位复现。融合版本在确定性测试中仍有约 1% 梯度相对误差、5 步后约 0.024% 参数相对误差，初始严格门槛失败记录完整保留。另行预设 BF16 算子门槛（输出相对 L2 &lt;0.6%、梯度 &lt;2%）后，6 类算子的 eager/compiled 前后向均通过独立 FP64 autograd 检查；实际最大误差约 0.37%。这支持开展独立收敛实验，不证明长期训练完全等价。</p>"
        compare = Path("runs/ppu96/round96_fusion_compare")
        pairs = {}
        for backend in ["eager", "compile_pointwise"]:
            d = compare/backend
            if not (d/"completed.json").exists() or not (d/"quality/quality.json").exists():continue
            metrics = [json.loads(line) for line in (d/"metrics.jsonl").read_text().splitlines()]
            quality = json.loads((d/"quality/quality.json").read_text())
            val = json.loads((d/"validation.jsonl").read_text().splitlines()[-1])
            completed = json.loads((d/"completed.json").read_text())
            pairs[backend] = dict(steady_images_per_second=statistics.median(m['images_per_second'] for m in metrics[2:]),elapsed_seconds=completed['elapsed_seconds'],fid=quality['clean_fid'],validation_loss=val['ema_val_loss'],skipped=sum(m.get('skipped',0) for m in metrics))
        if len(pairs) == 2:
            a,c=pairs['eager'],pairs['compile_pointwise']
            passed=c['validation_loss']/a['validation_loss']<=1.01 and c['fid']-a['fid']<=2.0 and c['skipped']==0
            fusion_html += f"<h2>16 PPU 真实训练对照</h2><p>从同一个 4.096M checkpoint 恢复模型、优化器与 RNG，各训练 1000 步 / 1.024M 张次。稳态吞吐 {a['steady_images_per_second']:.0f} → {c['steady_images_per_second']:.0f} 张/秒（{c['steady_images_per_second']/a['steady_images_per_second']:.2f}×）；含首次编译、训练、保存和预览的循环时间 {a['elapsed_seconds']:.1f} → {c['elapsed_seconds']:.1f} 秒。</p><p>验证 MSE {a['validation_loss']:.6f} → {c['validation_loss']:.6f}；FID {a['fid']:.3f} → {c['fid']:.3f}；融合组跳过更新数 {c['skipped']}。预先登记的短程质量标准（MSE 不超过基线 1.01 倍、FID 增加不超过 2、零跳过更新）：{'通过' if passed else '未通过'}。单 seed、小样本对照不能替代长程多 seed 验证。</p>"
            (compare/'comparison_summary.json').write_text(json.dumps(dict(runs=pairs,quality_gate_passed=passed),indent=2)+'\n')
        fusion_html += "<p>16 rank 共用 CPFS 编译缓存曾触发 Stale file handle。已改用节点本地 /tmp、按实验及 rank 隔离的编译缓存，失败尝试保留。训练默认仍为 eager，可通过配置 execution_backend=compile_pointwise 显式启用实验路径。</p></section>"
    text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>96px PixelJiT PPU profile</title><style>
body{{max-width:1150px;margin:36px auto;padding:0 24px;color:#20313b;background:#f8fafb;font:16px/1.6 system-ui}}section{{background:white;padding:22px;border:1px solid #d4dde2;border-radius:10px;margin:20px 0}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;border-bottom:1px solid #ddd;padding:8px}}i{{display:block;background:#2982a3;height:16px}}td:last-child{{min-width:180px}}code{{font-size:12px;overflow-wrap:anywhere}}h1{{font-size:28px}}h2{{font-size:20px}}</style>
<h1>96×96 PixelJiT：ASight / Triton 优化实验</h1>
<p>真实配置：单 PPU-ZW810E，batch 64，576 tokens，width 384，depth 8，heads 6；21.89M 参数。BF16 计算，FP32 参数、梯度及 AdamW 状态。输入为真实 96px CelebA 训练图，计时包含增强、前向、反向、clip、AdamW 和 EMA；不含 DDP、保存和采样。</p>
<section><h2>已测：换 fused AdamW 收益很小</h2><p>三个独立进程交错配对，每进程预热 10 步，3 × 30 步计时，无 profiler。默认 AdamW <b>{baseline:.3f} ms/step</b>，fused AdamW <b>{fused:.3f} ms/step</b>：耗时降低 <b>{100*(1-fused/baseline):.2f}%</b>，吞吐增加 {100*(baseline/fused-1):.2f}%。此处仅为性能候选，未据此切换正式训练。</p><p>三轮默认：{q['benchmarks']['default']['process_ms']}；fused：{q['benchmarks']['fused']['process_ms']} ms。全部训练 loss 有限，优化器状态均为 FP32；有限值本身不证明数值等价。</p></section>
<section><h2>瓶颈：分解的小算子与数据搬运</h2><table><tr><th>按原始 kernel 名称分组</th><th>累计 ms</th><th>设备 kernel 时间占比</th><th></th></tr>{''.join(rows)}</table>
<p>共 {p['kernel_count']} 个 kernel，时间跨度 {p['span_ms']:.3f} ms，kernel 时长总和 {p['kernel_sum_ms']:.3f} ms。RMSNorm 被分解为乘法、归约及转换，不能由独立 normalization kernel 的小占比推断其成本很低。分类器明确排除了 <code>nocast</code> 名称对 copy/cast 的误匹配。</p><p>ASight 仅用于定位；带 profiler 的时间不作为吞吐结论。时间线覆盖率 {100*p['coverage']:.1f}% 不等于 occupancy、计算利用率或带宽利用率；没有硬件计数器证据。</p></section>
{fusion_html}
<section><h2>复现与证据</h2><p>版本：{html.escape(json.dumps(versions))}。入口：<code>scripts/profile_pixel_suite.py</code>；逐元素候选：<code>scripts/profile_pixel_fusion_suite.py</code>；原始结果：<code>runs/profile96_v1/</code>。</p><p>ASight 使用 <code>asys profile --trace hggc,hgtx,acblas --sample none --capture-range hggcProfilerApi --capture-range-end stop</code>，预热后捕获一个完整更新；导出 SQLite。PyTorch profiler 另进程采集形状、调用栈和阶段范围，两个 profiler 的关联 ID 不混用。</p></section>
<details><summary>全部原始 kernel 名称与耗时</summary><table><tr><th>调用数</th><th>ms</th><th>kernel</th></tr>{kernels}</table></details></html>'''
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(text)
    print(out.resolve())

if __name__=='__main__':main()
