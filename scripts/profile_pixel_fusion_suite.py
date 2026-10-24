"""Independent eager/fusion benchmarks, with bounded compile probes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, TRITON_CACHE_DIR=str(out/'triton_cache'), TORCHINDUCTOR_CACHE_DIR=str(out/'inductor_cache'), TORCHINDUCTOR_COMPILE_THREADS='4', PYTHONFAULTHANDLER='1')
    results=[]
    for execution in ['eager', 'compile_norm', 'compile_pointwise']:
        cmd=[sys.executable,'-u','scripts/profile_pixel_train.py','--checkpoint',args.checkpoint,'--data',args.data,'--output',str(out/execution),'--execution',execution,'--warmup','10','--steps','20','--repeats','3']
        with (out/(execution+'.log')).open('w') as f:
            try:
                p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=480)
                code=p.returncode
            except subprocess.TimeoutExpired:
                code=124
        results.append(dict(execution=execution,returncode=code))
        (out/'commands.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(results[-1]),flush=True)
    (out/'completed.json').write_text(json.dumps(results))

if __name__=='__main__':main()
