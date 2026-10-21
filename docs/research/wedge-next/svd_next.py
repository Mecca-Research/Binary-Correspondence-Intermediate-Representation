"""Post-training output-mode factorization. Approximate ranks are not lossless.
Only weights transfer; old AdamW state is deliberately not converted.
"""
from candidates import *
from run_experiments import recall_batches,evaluate
from build_next import compile_source,lowrank,grouped_gate,reuse
rows=[]
if __name__=='__main__':
    fast=grouped_gate(reuse((ROOT/'build/global_norope.c').read_text()))
    for q in (16,32,64,128):compile_source(lowrank(fast,q),f'svd{q}')
    for seed in SEEDS:
        source=ROOT/'baseline-checkpoints'/f'global_norope-{seed}.npz';s=dict(np.load(source,allow_pickle=False))
        r0=Recipe(**json.loads(str(s['recipe'])));old={name:s['weights'][off:off+n].reshape(shape) for name,shape,off,n in layout(r0)}
        originals={}
        for t in (128,512,1024):
            xy=recall_batches(16,8,t,1971029+t)
            with Native(replace(r0,batch=8,context=t),threads=4,library_name='fast',weights=s['weights']) as n:
                originals[t]=evaluate(n,xy)
            rows.append({'seed':seed,'rank':'dense','context':t,'test':originals[t],'parameters':len(s['weights'])})
        svd={}
        for layer in range(r0.layers):
            svd[layer]=np.linalg.svd(old[f'l{layer}.project'].astype(np.float64),full_matrices=False)
        for q in (16,32,64,128):
            r=replace(r0,bottleneck=q);w=np.empty(sum(n for _,_,_,n in layout(r)),np.float32);errs=[]
            for name,shape,off,n in layout(r):
                if name.endswith(('.project_in','.project_out')):
                    layer=int(name.split('.')[0][1:]);u,a,v=svd[layer];root=np.sqrt(a[:q]);left=(u[:,:q]*root[None,:]).astype(np.float32);right=(root[:,None]*v[:q]).astype(np.float32)
                    value=right if name.endswith('.project_in') else left
                    if name.endswith('.project_in'):
                        orig=old[f'l{layer}.project'].astype(np.float64);recon=left.astype(np.float64)@right.astype(np.float64)
                        errs.append(float(np.linalg.norm(orig-recon)/np.linalg.norm(orig)))
                else:value=old[name]
                w[off:off+n]=value.ravel()
            for t in (128,512,1024):
                xy=recall_batches(16,8,t,1971029+t)
                with Native(replace(r,batch=8,context=t),threads=4,library_name=f'svd{q}',weights=w) as n:metric=evaluate(n,xy)
                row={'seed':seed,'rank':q,'context':t,'test':metric,'parameters':len(w),
                    'relative_projector_weight_errors':errs,'weights_sha256':hashlib.sha256(w.tobytes()).hexdigest(),
                    'scope':'Inference-only factorization of pretrained dense recall weights; no fine tuning; optimizer state reset, not migrated.'}
                rows.append(row);print(json.dumps(row),flush=True)
            path=ROOT/'svd-weights'/f'q{q}-{seed}.npz';path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,weights=w,recipe=json.dumps(asdict(r)),source_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (ROOT/'results/svd.json').write_text(json.dumps(rows,indent=2))
