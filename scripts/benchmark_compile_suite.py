"""Bounded, isolated single-device compiler ablation. Never saves model weights."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--gpu',action='store_true');a=p.parse_args()
out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,PYTHONPATH=str(Path.cwd()/'src')+':'+str(Path.cwd()),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='4',TORCHINDUCTOR_CACHE_DIR=str(out/'inductor'),TRITON_CACHE_DIR=str(out/'triton'))
variants=[('eager','default'),('compile_pointwise','default'),('compile_full','default'),('compile_pointwise','fused')]
if a.gpu:variants=[('eager','default'),('compile_pointwise','default')]
rows=[]
for resolution,micro in [(96,32),(48,64)]:
 for execution,optimizer in variants:
  label=f'{resolution}_{execution}_{optimizer}';cmd=[sys.executable,'-u','scripts/benchmark_single_gpu_accumulation.py','--resolution',str(resolution),'--microbatch',str(micro),'--execution',execution,'--optimizer',optimizer,'--precision','fp16','--warmup','3','--repeats','5','--validate','--output',str(out/(label+'.json'))]
  started=time.time()
  with (out/(label+'.log')).open('w') as log:
   try:code=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600).returncode
   except subprocess.TimeoutExpired:code=124
  row=dict(label=label,returncode=code,wall_seconds=time.time()-started);rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
(out/'completed.json').write_text(json.dumps(rows,indent=2))
