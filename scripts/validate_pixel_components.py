"""Independent FP64 autograd references for PPU fusion outputs and gradients.

Tolerances account for BF16 rounding (0.6% output, 2% gradient relative L2).
This gate does not erase the stricter full-model gate failures or establish
long-run convergence; an isolated paired training experiment is still required.
"""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pixel_accelerate import rms,rope,swiglu,modulate,residual


@torch.no_grad()
def error(actual, ref):
    delta=actual.double()-ref
    return dict(relative_l2=float((delta.square().sum()/ref.square().sum().clamp_min(1e-30)).sqrt()), max_abs=float(delta.abs().max()),finite=bool(actual.isfinite().all()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(97213);torch.use_deterministic_algorithms(True)
    device='cuda';dtype=torch.bfloat16;rows=[]
    def rand(*shape):return torch.randn(*shape,device=device,dtype=dtype)
    cases=[]
    cases.append(('rms_tokens',rms,[rand(8,576,384),torch.empty(384,device=device).uniform_(.8,1.2)],lambda x,w:x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)*w))
    packed=rand(8,576,3,6,64);q=packed[:,:,0].transpose(1,2)
    cases.append(('rms_qk_strided',rms,[q,torch.empty(64,device=device).uniform_(.8,1.2)],lambda x,w:x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)*w))
    phase=torch.randn(1,1,576,32,device=device);c=phase.cos().to(dtype);s=phase.sin().to(dtype)
    def rope1(x):return rope(x,c,s)
    def rope64(x):return torch.stack([x[...,0::2]*c.double()-x[...,1::2]*s.double(),x[...,0::2]*s.double()+x[...,1::2]*c.double()],-1).flatten(-2)
    cases.append(('rope',rope1,[rand(8,6,576,64)],rope64))
    cases.append(('swiglu',swiglu,[rand(8,576,1024),rand(8,576,1024)],swiglu))
    cases.append(('modulate',modulate,[rand(8,576,384),rand(8,1,384)*.1,rand(8,1,384)*.1],modulate))
    cases.append(('residual',residual,[rand(8,576,384),rand(8,1,384)*.1,rand(8,576,384)],residual))
    for name,fn,inputs,reference in cases:
        ref_inputs=[x.detach().to(dtype).double().requires_grad_() for x in inputs]
        target=reference(*ref_inputs);dy=torch.randn_like(target).to(dtype)
        grads=torch.autograd.grad(target,ref_inputs,dy.double())
        compiled=torch.compile(fn,fullgraph=True,dynamic=False,options={'emulate_precision_casts':True})
        for label,impl in [('eager',fn),('compiled',compiled)]:
            xs=[x.detach().clone().requires_grad_() for x in inputs]
            y=impl(*xs);gs=torch.autograd.grad(y,xs,dy)
            out=error(y,target);grad=[error(a,b) for a,b in zip(gs,grads)]
            passed=out['finite'] and out['relative_l2']<.006 and all(g['finite'] and g['relative_l2']<.02 for g in grad)
            rows.append(dict(case=name,provider=label,output=out,gradients=grad,passed=passed))
            print(json.dumps(rows[-1]),flush=True)
    result=dict(passed=all(r['passed'] for r in rows),gates=dict(output_relative_l2=.006,gradient_relative_l2=.02),cases=rows,reference='FP64 autograd on quantized BF16 inputs; RMS weights quantized as in the actual forward')
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise RuntimeError('Component validation failed')

if __name__=='__main__':main()
