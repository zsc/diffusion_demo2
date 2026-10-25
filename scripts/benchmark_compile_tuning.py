"""Short microbatch tuning and reverse-order confirmation of the compile benchmark."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--gpu',action='store_true');a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,PYTHONPATH=str(Path.cwd()/'src')+':'+str(Path.cwd()),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='4',TORCHINDUCTOR_CACHE_DIR=str(out.parent/'results/inductor'),TRITON_CACHE_DIR=str(out.parent/'results/triton'))
variants=[(96,64,'compile_full','default','fp16'),(96,64,'compile_pointwise','default','fp16'),(48,128,'compile_full','default','fp16'),(48,128,'compile_pointwise','default','fp16'),(96,32,'compile_full','fused','fp16'),(96,32,'eager','default','fp16')]
if a.gpu:variants=[(96,64,'compile_pointwise','default','bf16'),(96,64,'eager','default','bf16'),(96,32,'compile_full','default','fp16')]
rows=[]
for resolution,micro,execution,optimizer,precision in variants:
 label=f'{resolution}_b{micro}_{execution}_{optimizer}_{precision}'
 cmd=[sys.executable,'-u','scripts/benchmark_single_gpu_accumulation.py','--resolution',str(resolution),'--microbatch',str(micro),'--execution',execution,'--optimizer',optimizer,'--precision',precision,'--warmup','3','--repeats','5','--validate','--output',str(out/(label+'.json'))]
 started=time.time()
 with (out/(label+'.log')).open('w') as log:
  try:code=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600).returncode
  except subprocess.TimeoutExpired:code=124
 rows.append(dict(label=label,returncode=code,wall_seconds=time.time()-started));(out/'commands.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows[-1]),flush=True)
(out/'completed.json').write_text(json.dumps(rows,indent=2))
