"""Transfer experiment files through OSS using narrowly scoped signed URLs.

Signing uses air4's existing oss2/config; long-lived credentials never leave air4.
Run this under tmux on this Mac so SSH inherits the working network context.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import textwrap
import urllib.request

PREFIX = "Datasets_Pretrain/personal_backup/zhousc6/diffusion_demo2/20261024/"
JOB = "bifrost-2026102409500201-zhousc6"


def signed_url(method, key, internal=False):
    endpoint = "https://oss-cn-wulanchabu-internal.aliyuncs.com" if internal else "https://oss-cn-wulanchabu.aliyuncs.com"
    code = f"""
import configparser, json
from pathlib import Path
import oss2
cfg = configparser.ConfigParser()
cfg.read(Path.home() / '.ossutilconfig')
c = cfg['Credentials']
auth = oss2.StsAuth(c['accessKeyID'], c['accessKeySecret'], c['stsToken']) if c.get('stsToken') else oss2.Auth(c['accessKeyID'], c['accessKeySecret'])
bucket = oss2.Bucket(auth, {endpoint!r}, 'ips-prediction')
print(json.dumps(bucket.sign_url({method!r}, {key!r}, 3600, slash_safe=True)))
"""
    command = shlex.join(["/Users/zhousc6/miniforge3/envs/stepaudioedit/bin/python", "-c", code])
    result = subprocess.run(["ssh", "air4", command], capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError("OSS signing failed; credentials were not copied")
    return json.loads(result.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["local-put", "local-get", "remote-put", "remote-get"])
    ap.add_argument("file", help="Local or cluster path, depending on mode")
    ap.add_argument("name", help="Object name within this experiment's fixed prefix")
    ap.add_argument("--gateway", action="store_true", help="Use the verified 4x16 Fuyao SSH gateway via air4")
    ap.add_argument("--job", default=os.environ.get("SOFY_JOB_NAME", JOB))
    args = ap.parse_args()
    if args.name.startswith("/") or ".." in Path(args.name).parts:
        raise ValueError("Object must stay in experiment prefix")
    method = "PUT" if args.mode.endswith("put") else "GET"
    key = PREFIX + args.name
    url = signed_url(method, key, internal=args.mode.startswith("remote"))
    # The same stdlib transfer implementation runs on either end. Never print URL.
    code = f"""
import urllib.request, hashlib, json, os
from pathlib import Path
p = Path({args.file!r})
method = {method!r}
url = {url!r}
if method == 'PUT':
 with p.open('rb') as f:
  req = urllib.request.Request(url, data=f, method='PUT', headers={{'Content-Length':str(p.stat().st_size),'Content-Type':''}})
  with urllib.request.urlopen(req, timeout=600) as response:
   assert response.status == 200
else:
 p.parent.mkdir(parents=True, exist_ok=True)
 with urllib.request.urlopen(url, timeout=600) as response, p.with_suffix(p.suffix+'.download').open('wb') as f:
  while True:
   chunk = response.read(1024*1024)
   if not chunk: break
   f.write(chunk)
 p.with_suffix(p.suffix+'.download').replace(p)
h = hashlib.sha256()
with p.open('rb') as f:
 for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
print(json.dumps({{'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest(),'method':method}}))
"""
    if args.mode.startswith("local"):
        try:
            exec(compile(code, "oss_transfer", "exec"), {})
        except Exception as exc:
            # URL-bearing exception details can contain the scoped signature.
            raise RuntimeError(f"OSS {method} failed: {type(exc).__name__}") from None
    else:
        guarded = "try:\n" + textwrap.indent(code, " ") + "\nexcept Exception as exc:\n print('TRANSFER_ERROR', type(exc).__name__, getattr(exc, 'code', None))\n"
        payload = base64.b64encode(guarded.encode()).decode()
        command = shlex.join(["/usr/local/bin/python3", "-c", "exec(__import__('base64').b64decode(" + repr(payload) + "))"])
        remote = shlex.join(["/Users/zhousc6/d/sofy/.venv-codex/bin/sofy", "shell", "--job-name", args.job,
                            "--target", "master", "--timeout", "600", "--cmd", command])
        if args.gateway:
            from fuyao_bridge import run_remote
            result = run_remote(guarded, timeout=650)
        else:
            result = subprocess.run(["ssh", "air4", remote], capture_output=True, text=True, timeout=700)
        # Print only the transfer receipt; never echo a failed command with its URL.
        receipts = [line for line in result.stdout.splitlines() if line.startswith('{"path":')]
        if result.returncode or not receipts:
            details = [line for line in result.stdout.splitlines() if line.startswith("TRANSFER_ERROR")]
            raise RuntimeError(f"Remote OSS {method} failed; no transfer receipt; {details}")
        print("\n".join(receipts))


if __name__ == "__main__":
    main()
