"""Direct CPU control through the Fuyao SSH gateway, reached via air4.

Gateway/pod were resolved with air4's Fuyao SDK. No credentials are copied.
GPU workloads must still be submitted to the SOFY scheduler, not run here.
"""
import argparse
import base64
import json
from pathlib import Path
import shlex
import subprocess


def run_remote(code, timeout=90, job='bifrost-2026101312015311-zhousc6', host='8.130.87.145', port=32222):
    remote_cmd=shlex.join(['/usr/local/bin/python3','-c',"exec(__import__('base64').b64decode("+repr(base64.b64encode(code.encode()).decode())+"))"])
    inner='\n'.join([
        'import os,subprocess,sys',
        "env=dict(os.environ,BIFROST_NAMESPACE='default',BIFROST_USER='zhousc6@xiaopeng.com',BIFROST_CLUSTER='fuyao_ppu_c3',BIFROST_CONTAINER='pytorch')",
        'cmd='+repr(['ssh','-n','-p',str(port),'-o','BatchMode=yes','-o','ConnectTimeout=15',
                    # Fuyao gateway rotates keys across backends; user explicitly approved this on 2026-10-24.
                    '-o','UserKnownHostsFile=/dev/null','-o','StrictHostKeyChecking=no',
                    '-o','SendEnv=BIFROST_NAMESPACE','-o','SendEnv=BIFROST_USER','-o','SendEnv=BIFROST_CLUSTER','-o','SendEnv=BIFROST_CONTAINER',
                    f'{job}-master-0@{host}',remote_cmd]),
        'p=subprocess.run(cmd,env=env,timeout='+str(timeout)+')',
        'sys.exit(p.returncode)',
    ])
    air_cmd=shlex.join(['/usr/bin/python3','-c',inner])
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','air4',air_cmd],capture_output=True,text=True,timeout=timeout+30)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--python-file',required=True)
    ap.add_argument('--timeout',type=int,default=90)
    args=ap.parse_args()
    p=run_remote(Path(args.python_file).read_text(),args.timeout)
    if p.returncode:raise RuntimeError(f'Gateway command failed ({p.returncode}): {p.stderr[-1200:]}')
    print(p.stdout)

if __name__=='__main__':main()
