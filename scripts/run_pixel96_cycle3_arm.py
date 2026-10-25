"""Short matched sample-budget study: larger batches, LR annealing, and shared EMAs."""
import argparse,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--tag',choices=['control_ema','batch2048','batch4096','cosine'],required=True);a=p.parse_args()
r=Path(a.root);tag=a.tag;out=r/'round96_cycle3_20261025'/tag;parent=r/'round96_cycle2_20261025/stratified/last.pt'
cfg=Path('configs')/f'pixel96_cycle3_{tag}.json';new=json.loads(cfg.read_text());previous=json.loads((parent.parent/'config.json').read_text())
assert not out.exists(),'Never overwrite a previous arm'
assert new['execution_backend']=='compile_full'
env=dict(os.environ,QUALITY_EVAL='0',ALLOW_RESUME_LR_CHANGE='1' if tag=='cosine' else '0',ALLOW_RESUME_TIME_CHANGE='0',ALLOW_RESUME_BATCH_CHANGE='1' if new['global_batch']!=previous['global_batch'] else '0')
subprocess.run(['bash','scripts/run_gpu_group.sh',str(cfg),str(out),str(r/'data96/celeba96.npy'),str(r/'eval_assets'),str(parent)],env=env,check=True,timeout=2400)
assert json.loads((out/'completed.json').read_text())['images_seen']==36864000
rows=[]
for images,g,seed in [(34816000,2,3333),(36864000,1,3333),(36864000,2,3333),(36864000,2,7777),(36864000,3,3333)]:
 rows.append(dict(checkpoint=f'round96_cycle3_20261025/{tag}/ema_{images:09d}.pt',label=f'{tag}_{images}_g{g}_seed{seed}',guidance=g,seed=seed))
for decay in new.get('extra_ema_decays',[]):
 for seed in [3333,7777]:
  rows.append(dict(checkpoint=f'round96_cycle3_20261025/{tag}/ema_decay{decay}_036864000.pt',label=f'{tag}_decay{decay}_36864000_g2_seed{seed}',guidance=2,seed=seed))
manifest=out/(tag+'_evaluation.json');manifest.write_text(json.dumps(rows,indent=2)+'\n')
subprocess.run([sys.executable,'-u','scripts/evaluate_pixel_continuation.py','--root',str(r),'--manifest',str(manifest),'--output',str(r/'round96_cycle3_eval_20261025')],check=True,timeout=1800)
(out/'pipeline_completed.json').write_text(json.dumps(dict(training_images=4096000,unfiltered_evaluations=len(rows),best_of_n=False)))
