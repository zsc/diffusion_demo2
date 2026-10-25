"""Frozen VAE channel-PCA diagnostic; fit on training IDs only."""
import argparse,hashlib,json,math,os,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from diffusion_demo.core.models.vae import VAE
from diffusion_demo.core.models.perceptual import load_perceptual

def pca_from_moments(count,total,outer,noise):
    center=total/count
    covariance=outer/count-torch.outer(center,center)
    result={}
    for name,cov in [('mu',covariance),('sample',covariance+torch.diag(noise/count))]:
        eig,basis=torch.linalg.eigh((cov+cov.T)/2)
        order=torch.argsort(eig,descending=True);eig=eig[order].clamp_min(0)
        result[name]=dict(center=center.float(),basis=basis[:,order].float(),eigenvalues=eig,
                          retained=(eig.cumsum(0)/eig.sum().clamp_min(1e-20)))
    return result

def project_restore(z,center,basis,k):
    c=center[None,:,None,None];p=basis[:,:k]
    return torch.einsum('ck,bkhw->bchw',p,torch.einsum('ck,bchw->bkhw',p,z-c))+c

def gather(value,world):
    if world==1:return [value]
    values=[None]*world;dist.all_gather_object(values,value);return values

def grid(items,path):
    a=np.stack(items);n,h,w,_=a.shape;out=np.full((math.ceil(n/8)*(h+2),8*(w+2),3),32,np.uint8)
    for i,x in enumerate(a):y,c=divmod(i,8);out[y*(h+2):y*(h+2)+h,c*(w+2):c*(w+2)+w]=x
    Image.fromarray(out).resize((out.shape[1]*3,out.shape[0]*3),Image.Resampling.NEAREST).save(path)

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--cpu',action='store_true');args=ap.parse_args()
    root=Path(args.root);rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));device=torch.device('cpu' if args.cpu else 'cuda:'+os.environ.get('LOCAL_RANK','0'))
    torch.set_num_threads(2)
    if not args.cpu:torch.cuda.set_device(device)
    if world>1:dist.init_process_group('gloo' if args.cpu else 'nccl')
    data=np.load(root/'data.npy',mmap_mode='r');lpips=load_perceptual(root/'lpips.pth',device)
    start=time.perf_counter();summary={}
    def amp():return torch.autocast(device.type,dtype=torch.bfloat16,enabled=not args.cpu)
    def images(ids):return torch.from_numpy(data[ids].copy()).permute(0,3,1,2).to(device).float()/127.5-1
    def pixels(x):return ((x.clamp(-1,1)+1)*127.5).round().byte().permute(0,2,3,1).cpu().numpy()
    for name in ['kl01','kl1']:
        folder=root/name;ckpt=folder/'model.pth';d=torch.load(ckpt,map_location='cpu',weights_only=True);assert d['model_type']=='vae'
        model=VAE(**d['model_config']).to(device).eval().requires_grad_(False);model.load_state_dict(d['model_state_dict'],strict=True)
        split=json.loads((folder/'split.json').read_text());train=split['train_indices'];assert hashlib.sha256(data[:2048].tobytes()).hexdigest()==split['selected_sha256']
        assert not set(train)&set(split['validation_indices']);c=model.latent_dim
        total=torch.zeros(c,dtype=torch.float64);outer=torch.zeros(c,c,dtype=torch.float64);noise=torch.zeros(c,dtype=torch.float64);count=0
        ids=train[rank::world]
        for offset in range(0,len(ids),32):
            with amp():mu,lv=model.encoder(images(ids[offset:offset+32]))
            a=mu.float().permute(0,2,3,1).reshape(-1,c).cpu().double();v=lv.float().exp().permute(0,2,3,1).reshape(-1,c).cpu().double()
            total+=a.sum(0);outer+=a.T@a;noise+=v.sum(0);count+=len(a)
        stats=gather((count,total,outer,noise),world)
        projections=pca_from_moments(sum(x[0] for x in stats),sum(x[1] for x in stats),sum(x[2] for x in stats),sum(x[3] for x in stats))
        result=dict(checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),model_config=d['model_config'],training_images=len(train),training_epochs=8,train_only_pca=True,spectrum={k:{f:v.tolist() for f,v in p.items() if f in ['eigenvalues','retained']} for k,p in projections.items()},evaluations={})
        if rank==0:torch.save(projections,folder/'projections.pt')
        for splitname,evalids in [('validation',split['validation_indices']),('unseen1000',list(range(2048,3048)))]:
            chosen=evalids[:16];shard=evalids[rank::world]
            for mode in ['mu','sample']:
                projection=projections[mode];center=projection['center'].to(device);basis=projection['basis'].to(device)
                sums={k:np.zeros(3,np.float64) for k in [2,4,8,12,16]};previews={k:[] for k in sums};original=[]
                for offset in range(0,len(shard),16):
                    batch=shard[offset:offset+16];x=images(batch)
                    with amp():mu,lv=model.encoder(x)
                    z=mu.float()
                    if mode=='sample':
                        # Same per-image posterior noise for every truncation, rank and batch.
                        eps=torch.stack([torch.randn(z.shape[1:],generator=torch.Generator().manual_seed(2027+i)) for i in batch]).to(device)
                        z=z+lv.float().mul(.5).exp()*eps
                    with amp():full=model.decode(z)
                    for k in sums:
                        zk=z if k==c else project_restore(z,center,basis,k)
                        with amp():y=full if k==c else model.decode(zk)
                        mse=((y.float()-x)/2).square().flatten(1).mean(1)
                        perceptual=lpips(y.float(),x).flatten()
                        delta=((y.float()-full.float())/2).square().flatten(1).mean(1)
                        sums[k]+=np.array([mse.sum().item(),perceptual.sum().item(),delta.sum().item()])
                        py=pixels(y)
                        previews[k].extend((i,a) for i,a in zip(batch,py) if i in chosen)
                    original.extend((i,a) for i,a in zip(batch,pixels(x)) if i in chosen)
                all_sums=gather(sums,world);all_previews=gather(previews,world);all_original=gather(original,world)
                if rank==0:
                    metrics={}
                    for k in sums:
                        m=sum(a[k] for a in all_sums)/len(evalids)
                        metrics[str(k)]=dict(mse=float(m[0]),psnr=-10*math.log10(m[0]),lpips=float(m[1]),mse_to_uncompressed=float(m[2]),retained_variance=float(projection['retained'][k-1]),count=len(evalids))
                        by_id=dict(item for g in all_previews for item in g[k]);grid([by_id[i] for i in chosen],folder/f'{splitname}_{mode}_k{k}.png')
                    by_id=dict(item for g in all_original for item in g);grid([by_id[i] for i in chosen],folder/f'{splitname}_original.png')
                    result['evaluations'][splitname+'_'+mode]=metrics
                    (folder/'result.json').write_text(json.dumps(result,indent=2))
        if rank==0:summary[name]=result
    if rank==0:
        summary.update(metadata=dict(world_size=world,precision='fp32' if args.cpu else 'bf16',seconds=time.perf_counter()-start,data_sha256=hashlib.sha256((root/'data.npy').read_bytes()).hexdigest(),notes='Early 16-channel grid4 VAE models trained on 1792 images for 8 epochs; not a mature VAE or a VQVAE. PCA fitted only on original training IDs. Primary unseen1000 IDs 2048:3048 were not used for model training/checkpoint selection. Sample PCA covariance analytically includes E[diag(exp(logvar))]; fixed one posterior draw per heldout image for reconstruction.'))
        (root/'results.json').write_text(json.dumps(summary,indent=2));print(json.dumps(dict(done=True,seconds=summary['metadata']['seconds'])))
    if world>1:dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
