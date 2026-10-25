"""Download official SqueezeNet + LPIPS v0.1 weights once; package an offline checkpoint."""
import argparse,hashlib,json
from pathlib import Path
import torch
import lpips


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    model=lpips.LPIPS(net='squeeze',version='0.1').eval().requires_grad_(False)
    torch.save(model.state_dict(),out)
    provenance=dict(package='lpips==0.1.4',network='squeezenet1_1 ImageNet1K V1',calibration='LPIPS v0.1 squeeze',
                    input='RGB [-1,1], native 48x48, no resizing',
                    sources=['https://github.com/richzhang/PerceptualSimilarity',
                             'https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth'],
                    sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    out.with_suffix('.json').write_text(json.dumps(provenance,indent=2));print(json.dumps(provenance))

if __name__=='__main__':main()
