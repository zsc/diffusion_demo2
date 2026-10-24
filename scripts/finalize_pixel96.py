"""Bounded 16-PPU sampling screen, independent seed confirmation, native96 samples."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);args=ap.parse_args();r=Path(args.root)
    ckpt=r/'round96_8m/scaled_lr2e4/ema_008192000.pt';guide=r/'round96/scaled_lr2e4/ema_000256000.pt'
    out=r/'round96_final_sampling';out.mkdir(exist_ok=True)
    env={**os.environ,'PYTHONPATH':f'src:{r}/eval_assets/packages','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
    for key in ['WORLD_SIZE','RANK','LOCAL_RANK']:env.pop(key,None)
    def evaluate(scale,seed,label):
        d=out/label;d.mkdir(exist_ok=True)
        cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=16','evaluate_pixel.py','--checkpoint',str(ckpt),'--data',str(r/'data96/celeba96.npy'),'--weights-dir',str(r/'eval_assets'),'--output',str(d),'--seed',str(seed)]
        if scale>1:cmd+=['--guide-checkpoint',str(guide),'--guidance',str(scale),'--guidance-start','0.1','--guidance-end','0.8']
        with (d/'eval.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=240)
        q=json.loads((d/'quality.json').read_text());print(label,q['clean_fid'],flush=True);return q
    candidates={1.0:json.loads((r/'round96_8m/scaled_lr2e4/quality/quality.json').read_text())}
    for scale in [2.0,3.0,4.0]:candidates[scale]=evaluate(scale,3333,f'g{scale}_seed3333')
    best=min(candidates,key=lambda s:candidates[s]['clean_fid'])
    (out/'selection.json').write_text(json.dumps(dict(guidance=best,selection_seed=3333,candidates=candidates),indent=2))
    evaluate(1.0,7777,'raw_seed7777')
    if best>1:evaluate(best,7777,'selected_seed7777')
    target=r/'pixel96_best';target.mkdir(exist_ok=True)
    sample=target/'sample_ppu_seed2027'
    cmd=[sys.executable,'sample_pixel.py','--checkpoint',str(ckpt),'--output',str(sample),'--seed','2027','--count','64','--steps','50','--batch','16']
    if best>1:cmd+=['--guide-checkpoint',str(guide),'--guidance',str(best),'--guidance-start','0.1','--guidance-end','0.8']
    subprocess.run(cmd,env=env,check=True,timeout=180)
    shutil.copy2(ckpt,target/'model_ema.pt');shutil.copy2(r/'round96_8m/scaled_lr2e4/last.pt',target/'last.pt')
    if best>1:shutil.copy2(guide,target/'guide_ema.pt')
    (target/'selection.json').write_text((out/'selection.json').read_text())
    (target/'completed.json').write_text(json.dumps(dict(success=True,guidance=best)))
    print('FINAL96_READY',best,flush=True)

if __name__=='__main__':main()
