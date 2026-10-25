"""Bounded matched 16-GPU continuations before adopting whole-model compilation."""
import argparse,json,os,statistics,subprocess,sys
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
r=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
parent=r/'round96_noise_20261025/more_low_noise/last.pt'
previous=json.loads((parent.parent/'config.json').read_text());start=16384000
rows={};env=dict(os.environ,PYTHONPATH=str(Path.cwd()/'src')+':'+str(Path.cwd()),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
for k in ['RANK','WORLD_SIZE','LOCAL_RANK']:env.pop(k,None)
for backend in ['compile_pointwise','compile_full']:
 cfg=dict(previous,execution_backend=backend,max_images=start+131072,eval_every_images=131072,sample_steps=2,log_every=16)
 cp=out/(backend+'.json');cp.write_text(json.dumps(cfg,indent=2))
 command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=16','pixel_ddp_experiment.py','--config',str(cp),'--output',str(out/backend),'--data',str(r/'data96/celeba96.npy'),'--resume',str(parent)]
 with (out/(backend+'.log')).open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600,check=True)
 metrics=[json.loads(l) for l in (out/backend/'metrics.jsonl').read_text().splitlines()]
 rows[backend]=dict(median_images_per_second=statistics.median(x['images_per_second'] for x in metrics[2:]),measurements=metrics)
 print(json.dumps({backend:rows[backend]['median_images_per_second']}),flush=True)
torch.set_num_threads(4)
x=torch.load(out/'compile_pointwise/last.pt',map_location='cpu',weights_only=False);y=torch.load(out/'compile_full/last.pt',map_location='cpu',weights_only=False)
assert x['step']==y['step'] and x['images_seen']==y['images_seen']==start+131072
errors={}
for group in ['model','ema']:
 assert x[group].keys()==y[group].keys()
 diff2=ref2=0.;maximum=0.
 for key in x[group]:
  left=x[group][key].double();delta=left-y[group][key].double();diff2+=float(delta.square().sum());ref2+=float(left.square().sum());maximum=max(maximum,float(delta.abs().max()))
 errors[group]=dict(relative_l2=(diff2/max(ref2,1e-30))**.5,max_abs=maximum)
assert all(torch.equal(u['cpu'],v['cpu']) and torch.equal(u['cuda'],v['cuda']) for u,v in zip(x['rng'],y['rng']))
loss_error=max(abs(u['loss']-v['loss'])/abs(u['loss']) for u,v in zip(rows['compile_pointwise']['measurements'],rows['compile_full']['measurements']))
limits=dict(model_relative_l2=.001,ema_relative_l2=.0001,loss_relative=.005)
passed=errors['model']['relative_l2']<limits['model_relative_l2'] and errors['ema']['relative_l2']<limits['ema_relative_l2'] and loss_error<limits['loss_relative'] and all(m['skipped']==0 for row in rows.values() for m in row['measurements'])
result=dict(passed=passed,limits=limits,errors=errors,loss_relative=loss_error,rng_identical=True,benchmarks=rows,speedup=rows['compile_full']['median_images_per_second']/rows['compile_pointwise']['median_images_per_second'],note='Same full checkpoint, data and16rank RNG,128 full optimizer/EMA updates. Original unwrapped checkpoint keys retained. Excludes first two timed log intervals. Not long-run convergence proof.')
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('RESULT '+json.dumps(result),flush=True)
if not passed:raise RuntimeError('DDP numerical gate failed')
