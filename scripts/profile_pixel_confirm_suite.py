"""Numerics, three independent timing pairs, then an external profiler capture."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,TRITON_CACHE_DIR=str(out/'triton_cache'),TORCHINDUCTOR_CACHE_DIR=str(out/'inductor_cache'),TORCHINDUCTOR_COMPILE_THREADS='4',PYTHONFAULTHANDLER='1')
    results=[]
    def run(label,cmd):
        with (out/(label+'.log')).open('w') as log:
            try:code=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=480).returncode
            except subprocess.TimeoutExpired:code=124
        results.append(dict(label=label,returncode=code));(out/'commands.json').write_text(json.dumps(results,indent=2));print(json.dumps(results[-1]),flush=True)
        return code
    base=['--checkpoint',args.checkpoint,'--data',args.data]
    # Keep failed gates as evidence; no training adoption is performed here.
    run('gate_norm',[sys.executable,'scripts/validate_pixel_optimizer.py',*base,'--optimizer','default','--execution','compile_norm','--output',str(out/'gate_norm.json')])
    run('gate_pointwise',[sys.executable,'scripts/validate_pixel_optimizer.py',*base,'--optimizer','default','--execution','compile_pointwise','--output',str(out/'gate_pointwise.json')])
    for repeat in range(3):
        order=['eager','compile_pointwise'] if repeat%2==0 else ['compile_pointwise','eager']
        for execution in order:
            label=f'bench_{repeat}_{execution}'
            run(label,[sys.executable,'scripts/profile_pixel_train.py',*base,'--execution',execution,'--output',str(out/label)])
    profiler=os.environ.get('PROFILER_BIN')
    if profiler:
        prefix=str(out/'profiler_pointwise')
        code=run('profiler_pointwise',[profiler,'profile','--output',prefix,sys.executable,'scripts/profile_pixel_train.py',*base,'--execution','compile_pointwise','--capture','external','--output',str(out/'profiler_pointwise_workload')])
        if not code:run('profiler_export',[profiler,'export','--output',prefix+'.sqlite',prefix+'.report'])
    (out/'completed.json').write_text(json.dumps(results))

if __name__=='__main__':main()
