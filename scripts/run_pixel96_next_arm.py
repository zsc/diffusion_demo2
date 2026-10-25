"""One full16GPU continuation arm, followed by unfiltered matched evaluations."""
import argparse,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--tag',choices=['control','cleaner','wider','lower_lr'],required=True);a=p.parse_args()
r=Path(a.root);tag=a.tag;out=r/'round96_next_20261025'/tag
parent=r/'round96_noise_20261025/more_low_noise/last.pt';cfg=Path('configs')/f'pixel96_next_{tag}.json'
gate=json.loads((r/'pixel96_full_compile_gate_20261025/result.json').read_text())
assert gate['passed'],'Full compiler DDP gate must pass before training'
assert not out.exists(),'Use a new arm directory, never silently restart/overwrite an arm'
new=json.loads(cfg.read_text());previous=json.loads((parent.parent/'config.json').read_text())
assert new['execution_backend']=='compile_full'
env=dict(os.environ,QUALITY_EVAL='0',ALLOW_RESUME_LR_CHANGE='1' if new['lr']!=previous['lr'] else '0',ALLOW_RESUME_TIME_CHANGE='1' if (new['time_logit_mean'],new['time_logit_std'])!=(previous['time_logit_mean'],previous['time_logit_std']) else '0')
subprocess.run(['bash','scripts/run_gpu_group.sh',str(cfg),str(out),str(r/'data96/celeba96.npy'),str(r/'eval_assets'),str(parent)],env=env,check=True,timeout=2400)
assert json.loads((out/'completed.json').read_text())['images_seen']==24576000
rows=[]
for images,scale,seed in [(20480000,2,3333),(24576000,1,3333),(24576000,2,3333),(24576000,2,7777),(24576000,3,3333)]:
 rows.append(dict(checkpoint=f'round96_next_20261025/{tag}/ema_{images:09d}.pt',label=f'{tag}_{images}_g{scale}_seed{seed}',guidance=scale,seed=seed))
manifest=out/(tag+'_evaluation.json');manifest.write_text(json.dumps(rows,indent=2)+'\n')
subprocess.run([sys.executable,'-u','scripts/evaluate_pixel_continuation.py','--root',str(r),'--manifest',str(manifest),'--output',str(r/'round96_next_eval_20261025')],check=True,timeout=1500)
(out/'pipeline_completed.json').write_text(json.dumps(dict(training_images=8192000,unfiltered_evaluations=len(rows),best_of_n=False)))
