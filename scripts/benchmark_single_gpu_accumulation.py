"""Bounded FP16 timing of true global-batch1024 updates, including gradient accumulation.

Synthetic uint8 data is kept on GPU with the same dataset residency as training.
Does not load, overwrite or emit trained checkpoints.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch
from diffusion_demo.core.models.pixel_jit import build_pixel_model, flow_loss


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--resolution',type=int,required=True)
    p.add_argument('--microbatch',type=int,required=True)
    p.add_argument('--global-batch',type=int,default=1024)
    p.add_argument('--output',required=True)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--execution',choices=['eager','compile_norm','compile_pointwise','compile_full'],default='eager')
    p.add_argument('--optimizer',choices=['default','fused'],default='default')
    p.add_argument('--precision',choices=['fp16','bf16'],default='fp16')
    p.add_argument('--validate',action='store_true')
    a=p.parse_args();assert a.global_batch%a.microbatch==0
    torch.set_num_threads(4);torch.manual_seed(42);device=torch.device('cuda')
    cfg=dict(image_size=a.resolution,patch_size=4,width=384,depth=8,heads=6,bottleneck=64,conv_channels=32,conv_layers=3,position_scale=.5 if a.resolution==96 else 1.)
    model=build_pixel_model(cfg,device);ema=copy.deepcopy(model).eval().requires_grad_(False)
    from pixel_accelerate import configure_execution
    configure_execution(model, 'eager' if a.execution=='compile_full' else a.execution)
    forward_model=torch.compile(model,fullgraph=True,dynamic=False,options={'emulate_precision_casts':True}) if a.execution=='compile_full' else model
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=0,**({'fused':True} if a.optimizer=='fused' else {}))
    dtype=torch.float16 if a.precision=='fp16' else torch.bfloat16
    scaler=torch.amp.GradScaler('cuda',init_scale=1024,enabled=a.precision=='fp16')
    count=31000 if a.resolution==96 else 30000
    data=torch.randint(0,256,(count,3,a.resolution,a.resolution),device=device,dtype=torch.uint8)
    accumulation=a.global_batch//a.microbatch
    def step():
        opt.zero_grad(set_to_none=True);loss_sum=torch.zeros((),device=device)
        for _ in range(accumulation):
            clean=data[torch.randint(29000,(a.microbatch,),device=device)].float()/127.5-1
            clean=torch.where(torch.rand(a.microbatch,1,1,1,device=device)<.5,clean.flip(-1),clean)
            t=(torch.randn(a.microbatch,device=device)*.8-.8).sigmoid();noise=torch.randn_like(clean)
            with torch.autocast('cuda',dtype=dtype):loss=flow_loss(forward_model,clean,t,noise)/accumulation
            scaler.scale(loss).backward();loss_sum+=loss.detach()
        scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        scale=scaler.get_scale();scaler.step(opt);scaler.update()
        with torch.no_grad():
            torch._foreach_mul_(list(ema.parameters()),.9984)
            torch._foreach_add_(list(ema.parameters()),list(model.parameters()),alpha=.0016)
        return loss_sum,norm,scaler.get_scale()<scale
    measurements=[];started=time.perf_counter()
    for _ in range(a.warmup):step()
    torch.cuda.synchronize();warmup_seconds=time.perf_counter()-started
    torch.cuda.reset_peak_memory_stats()
    for i in range(a.repeats):
        torch.cuda.synchronize();started=time.perf_counter();loss,norm,skipped=step();torch.cuda.synchronize();elapsed=time.perf_counter()-started
        row=dict(seconds=elapsed,images_per_second=a.global_batch/elapsed,loss=loss.item(),grad_norm=norm.item(),skipped=skipped)
        assert torch.isfinite(loss).item() and torch.isfinite(norm).item() and not skipped,row
        measurements.append(row);print(json.dumps(dict(repeat=i,**row)),flush=True)
    result=dict(resolution=a.resolution,microbatch=a.microbatch,global_batch=a.global_batch,accumulation=accumulation,device=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,precision=a.precision,execution=a.execution,optimizer=a.optimizer,model=cfg,parameters=sum(p.numel() for p in model.parameters()),data_resident_bytes=data.numel(),warmup_updates=a.warmup,warmup_seconds=warmup_seconds,measurements=measurements,median_images_per_second=statistics.median(x['images_per_second'] for x in measurements),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,source_sha256=hashlib.sha256(Path('src/diffusion_demo/core/models/pixel_jit.py').read_bytes()).hexdigest(),note='Synthetic data; full accumulated optimizer and EMA updates. Excludes checkpoint I/O, validation and sampling. No full training performed.')
    if a.validate:
        # Compare warmed, nonzero weights using identical fixed inputs. No optimizer update.
        opt.zero_grad(set_to_none=True)
        clean=data[:a.microbatch].float()/127.5-1
        t=torch.linspace(.1,.9,a.microbatch,device=device);noise=torch.randn_like(clean)
        with torch.autocast('cuda',dtype=dtype):candidate_loss=flow_loss(forward_model,clean,t,noise)
        (candidate_loss*1024).backward()
        candidate_grads=[p.grad.detach().float().cpu()/1024 for p in model.parameters()]
        opt.zero_grad(set_to_none=True)
        reference=build_pixel_model(cfg,device);reference.load_state_dict(model.state_dict())
        configure_execution(reference,'eager')
        reference_loss=torch.zeros((),device=device)
        for start in range(0,a.microbatch,32):
            stop=min(start+32,a.microbatch)
            with torch.autocast('cuda',dtype=dtype):part=flow_loss(reference,clean[start:stop],t[start:stop],noise[start:stop])*(stop-start)/a.microbatch
            (part*1024).backward();reference_loss+=part.detach()
        diff2=ref2=maximum=0.
        for cg,rp in zip(candidate_grads,reference.parameters()):
            rg=rp.grad.detach().float().cpu()/1024;delta=cg-rg
            diff2+=float(delta.double().square().sum());ref2+=float(rg.double().square().sum());maximum=max(maximum,float(delta.abs().max()))
        relative=(diff2/max(ref2,1e-30))**.5
        loss_relative=abs(float(candidate_loss.detach())-float(reference_loss))/max(abs(float(reference_loss)),1e-12)
        result['numerical_check']=dict(gradient_relative_l2=relative,gradient_max_abs=maximum,loss_relative=loss_relative,passed=relative<.005 and loss_relative<.002,gradient_limit=.005,loss_limit=.002,note='Same warmed weights; reference accumulates chunks up to32 to fit memory; not a convergence test.')
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print('RESULT '+json.dumps(result),flush=True)

if __name__=='__main__':main()
