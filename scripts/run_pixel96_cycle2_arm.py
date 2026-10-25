"""Continue the selected96px model or run one controlled replacement variant."""
import argparse,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--tag',choices=['continue_cleaner','balanced','narrow','stratified'],required=True);a=p.parse_args()
r=Path(a.root);tag=a.tag;out=r/'round96_cycle2_20261025'/tag
parent=r/'round96_next_20261025/cleaner/last.pt';cfg=Path('configs')/f'pixel96_cycle2_{tag}.json'
assert json.loads((r/'pixel96_full_compile_gate_20261025/result.json').read_text())['passed']
assert not out.exists(),'Never overwrite an arm'
new=json.loads(cfg.read_text());previous=json.loads((parent.parent/'config.json').read_text())
assert new['execution_backend']=='compile_full'
env=dict(os.environ,QUALITY_EVAL='0',ALLOW_RESUME_LR_CHANGE='0',ALLOW_RESUME_TIME_CHANGE='1' if any(new.get(k,d)!=previous.get(k,d) for k,d in [('time_logit_mean',-.8),('time_logit_std',.8),('time_sampling','iid')]) else '0')
# Device preflight is especially relevant to the new inverse-CDF sampler.
preflight="import torch; from pixel_ddp_experiment import sample_training_times; torch.manual_seed(19); t=sample_training_times(64,'cuda',0.,.8,"+repr(new.get('time_sampling','iid'))+"); assert t.isfinite().all() and ((t>0)&(t<1)).all(); print('TIME_SAMPLER_GPU_PREFLIGHT_OK',flush=True)"
subprocess.run([sys.executable,'-c',preflight],env={**env,'PYTHONPATH':str(Path.cwd()/'src')+':'+str(Path.cwd())},check=True,timeout=60)
subprocess.run(['bash','scripts/run_gpu_group.sh',str(cfg),str(out),str(r/'data96/celeba96.npy'),str(r/'eval_assets'),str(parent)],env=env,check=True,timeout=2400)
assert json.loads((out/'completed.json').read_text())['images_seen']==32768000
rows=[]
for images,scale,seed in [(28672000,2,3333),(32768000,1,3333),(32768000,2,3333),(32768000,2,7777),(32768000,3,3333)]:
 rows.append(dict(checkpoint=f'round96_cycle2_20261025/{tag}/ema_{images:09d}.pt',label=f'{tag}_{images}_g{scale}_seed{seed}',guidance=scale,seed=seed))
manifest=out/(tag+'_evaluation.json');manifest.write_text(json.dumps(rows,indent=2)+'\n')
subprocess.run([sys.executable,'-u','scripts/evaluate_pixel_continuation.py','--root',str(r),'--manifest',str(manifest),'--output',str(r/'round96_cycle2_eval_20261025')],check=True,timeout=1500)
(out/'pipeline_completed.json').write_text(json.dumps(dict(training_images=8192000,unfiltered_evaluations=len(rows),best_of_n=False)))
