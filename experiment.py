"""Run a reproducible VAE -> DiT experiment from a JSON configuration."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from diffusion_demo.core.training.train_dit import train_dit_model
from diffusion_demo.core.training.train_vae import train_vae_model
from diffusion_demo.utils.torch_env import get_torch_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    unknown = set(config) - {'common', 'vae', 'dit', 'vae_model_path'}
    if unknown:
        parser.error(f'Unknown sections: {sorted(unknown)}')
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'config.json').write_text(json.dumps(config, indent=2))
    environment = get_torch_info()
    environment.update(python=sys.version, executable=sys.executable)
    source_root = Path(__file__).resolve().parent
    files = sorted((source_root / 'src').rglob('*.py')) + [Path(__file__).resolve()]
    environment['source_sha256'] = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in files}
    try:
        environment['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        environment['git_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    (root / 'environment.json').write_text(json.dumps(environment, indent=2))
    common, result = config.get('common', {}), {}
    vae_path = config.get('vae_model_path')
    if 'vae' in config:
        result['vae'] = train_vae_model(**(common | config['vae'] | {'output_dir': str(root / 'vae')}))
        vae_path = result['vae']['best_model_path'] or result['vae']['model_path']
        (root / 'result.json').write_text(json.dumps(result, indent=2))
    if 'dit' in config:
        if not vae_path:
            parser.error('DiT needs either a vae section or vae_model_path')
        result['dit'] = train_dit_model(**(common | config['dit'] | {
            'vae_model_path': vae_path, 'output_dir': str(root / 'dit')}))
    (root / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
