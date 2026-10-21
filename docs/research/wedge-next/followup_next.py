"""Held-out paired hybrid tests and full-model guarded-precision validation."""
from candidates import *
from run_experiments import bytes_data,lm_batches,recall_batches,evaluate
from build_next import once
import torch,importlib
from torch.nn import functional as F
torch.set_num_threads(1)

if __name__=='__main__':
    source=(ROOT/'oracle.py').read_text()
    needle='                    feature=(unit.float()/math.sqrt(r.subspaces)).flatten(-2)'
    replacement='''                    a=u.double();c=v.double();fp=a[...,i]*c[...,j]-a[...,j]*c[...,i]
                    np64=torch.linalg.vector_norm(fp,dim=-1,keepdim=True)
                    precise=torch.where(np64>1e-6,fp/np64.clamp_min(1e-6),torch.zeros_like(fp))
                    prod=a.square().sum(-1,keepdim=True)*c.square().sum(-1,keepdim=True)
                    det=prod-(a*c).sum(-1,keepdim=True).square()
                    unit=torch.where((prod==0)|(det<=1e-4*prod),precise,unit)
'''+needle
    (ROOT/'oracle_precise.py').write_text(once(source,needle,replacement));oracle=importlib.import_module('oracle_precise')
    r=Recipe(kind='global',width=32,layers=2,heads=2,ff=48,batch=2,context=11)
    rng=np.random.default_rng(52218);x=rng.integers(0,260,(2,11),dtype=np.uint32);x[:,1]=x[:,0];y=np.roll(x,-1,1)
    with Native(r,threads=1,library_name='precise') as n:
        p=n.forward(x).copy();loss=n.backward(x,y);gr=n.buffers['gradients'].copy()
        m=oracle.torch_model_next(r,n.buffers['weights']);pp=m(torch.tensor(x.astype(np.int64)))
        ll=F.cross_entropy(pp.flatten(0,1),torch.tensor(y.astype(np.int64)).flatten());ll.backward()
        gg=np.concatenate([a.grad.numpy().ravel() for a in m.params])
        checks={'max_logit_error':float(np.max(np.abs(p-pp.detach().numpy()))),'loss_error':abs(loss-ll.item()),
            'gradient_max_error':float(np.max(np.abs(gr-gg))),'gradient_relative_l2':float(np.linalg.norm(gr-gg)/np.linalg.norm(gg))}
        assert checks['max_logit_error']<1e-5 and checks['gradient_max_error']<1e-4 and checks['gradient_relative_l2']<2e-4,checks
    data,_=bytes_data();rows=[]
    # Hybrid/global checkpoints were selected before these fresh, paired samples.
    for seed in SEEDS:
        # Close the comparison using attention on exactly the hybrid LM test bytes.
        d=json.loads((ROOT/'results'/f'byte_lm-attention-{seed}.json').read_text())
        state=dict(np.load(ROOT/d['checkpoint'],allow_pickle=False));r=Recipe(**json.loads(str(state['recipe'])))
        xy=lm_batches(data['test'],64,r.batch,r.context,20261102)
        with Native(r,threads=4,library_name='attention') as n:n.restore(state);test=evaluate(n,xy)
        rows.append({'task':'byte_lm_hybrid_paired','name':'attention','seed':seed,'test':test,
            'input_sha256':hashlib.sha256(xy[0].tobytes()+xy[1].tobytes()).hexdigest()})
        d=json.loads((ROOT/'hybrid/results'/f'recall_fixed-hybrid-{seed}.json').read_text())
        state=dict(np.load(ROOT/'hybrid'/d['checkpoint'],allow_pickle=False));r0=Recipe(**json.loads(str(state['recipe'])))
        for t in (64,128,256,512,1024):
            r=replace(r0,batch=8,context=t);xy=recall_batches(16,8,t,931029+t)
            with Native(r,threads=4,library_name='hybrid') as n:n.restore(state);test=evaluate(n,xy);assert n.digest()==d['state_sha256']
            row={'task':'hybrid_length','name':'hybrid','seed':seed,'context':t,'test':test,
                 'input_sha256':hashlib.sha256(xy[0].tobytes()+xy[1].tobytes()).hexdigest()}
            rows.append(row);print(json.dumps(row),flush=True)
    # Guard is a new numeric contract, so report inference changes, not identity.
    for seed in SEEDS:
        d=json.loads((ROOT/'results'/f'recall_fixed-global-{seed}.json').read_text());state=dict(np.load(ROOT/d['checkpoint'],allow_pickle=False))
        r=replace(Recipe(**json.loads(str(state['recipe']))),batch=8,context=1024);xy=recall_batches(16,8,1024,931029+1024)
        with Native(r,threads=4,library_name='precise') as n:n.restore(state);test=evaluate(n,xy)
        rows.append({'task':'guarded_global_length','name':'precise','seed':seed,'context':1024,'test':test})
    (ROOT/'results/followup.json').write_text(json.dumps({'precision_full_model':checks,'rows':rows},indent=2))
    print('followup pass',checks)
