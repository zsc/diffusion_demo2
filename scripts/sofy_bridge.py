"""Run a Python probe, or fetch small artifacts, through air4's SOFY CLI.

On this Mac run under a task-owned tmux session to inherit its working SSH route.
Large training checkpoints remain on CPFS; this helper caps each fetch at 8 MiB.
"""
import argparse
import base64
import json
from pathlib import Path
import re
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", default="bifrost-2026102409500201-zhousc6")
    parser.add_argument("--python-file")
    parser.add_argument("--get", nargs="+", help="Remote file paths")
    parser.add_argument("--output", default="runs/ppu_artifacts")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    if bool(args.python_file) == bool(args.get):
        parser.error("Choose --python-file or --get")
    if args.python_file:
        code = Path(args.python_file).read_text()
    else:
        code = "\n".join([
            "import base64, json", "from pathlib import Path",
            "paths = " + repr(args.get),
            "for name in paths:",
            " p = Path(name)",
            " if p.stat().st_size > 8 * 1024**2: raise ValueError('Fetch limit: 8 MiB')",
            " print('DIFFUSION_FILE ' + json.dumps({'name':p.name,'data':base64.b64encode(p.read_bytes()).decode()}))",
        ])
    payload = base64.b64encode(code.encode()).decode()
    command = shlex.join(["/usr/local/bin/python3", "-c", "exec(__import__('base64').b64decode(" + repr(payload) + "))"])
    remote = shlex.join(["/Users/zhousc6/d/sofy/.venv-codex/bin/sofy", "shell", "--job-name", args.job,
                        "--target", "master", "--timeout", str(args.timeout), "--cmd", command])
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "air4", remote],
                            capture_output=True, text=True, timeout=args.timeout + 90)
    # SOFY shell itself may exit 0 even when the command on the node failed.
    remote_codes = re.findall(r"^=== .+ \(exit=(-?\d+)\) ===$", result.stdout, re.MULTILINE)
    if result.returncode or any(int(code) != 0 for code in remote_codes):
        raise RuntimeError(f"Remote command failed: {result.stdout[-4000:]}\n{result.stderr[-4000:]}")
    if args.python_file:
        print(result.stdout)
    else:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        count = 0
        for line in result.stdout.splitlines():
            if line.startswith("DIFFUSION_FILE "):
                file = json.loads(line.removeprefix("DIFFUSION_FILE "))
                p = output / Path(file["name"]).name
                p.write_bytes(base64.b64decode(file["data"]))
                count += 1
                print(p.resolve())
        if count != len(args.get):
            raise RuntimeError(f"Fetched {count}/{len(args.get)} files: {result.stdout[-3000:]} {result.stderr}")
    if result.returncode:
        raise RuntimeError(result.stderr)


if __name__ == "__main__":
    main()
