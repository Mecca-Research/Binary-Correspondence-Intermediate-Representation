"""Independent gradient check for microbatch accumulation conventions.

BCIR leaves the SUM of microbatch-mean gradients in its gradient array. The
optimizer applies 1/accumulation internally. Compare normalized gradients.
"""
from policy import *
from calibrate import single_policy
from tasks import ordered_records,trim
from oracle import torch_model_next
import torch
from torch.nn import functional as F
torch.set_num_threads(1)

if __name__=='__main__':
    out=[]
    for family in ('global','attention'):
        r=Recipe(kind='transformer' if family=='attention' else 'global',width=32,layers=2,ff=48,heads=4)
        v='attention' if family=='attention' else 'compact';rng=np.random.default_rng(14627)
        x=rng.integers(0,260,(4,11),dtype=np.uint32);y=np.full_like(x,IGNORE);y[:,-1]=x[:,1]
        model=torch_model_next(r,initial(r,123));p=model(torch.tensor(x.astype(np.int64)));mask=torch.tensor(y!=IGNORE)
        loss=F.cross_entropy(p[mask],torch.tensor(y.astype(np.int64))[mask]);loss.backward()
        gg=np.concatenate([a.grad.numpy().ravel() for a in model.params])
        for micro in (4,2,1):
            policy=single_policy(r,family,4,11,micro,v,4)
            with Runner(policy,seed=123) as n:
                e=n.update(x,y,.0001);g=n.buffers['gradients'].copy()/(4//micro)
                rel=float(np.linalg.norm(g-gg)/np.linalg.norm(gg))
                row={'family':family,'microbatch':micro,'normalized_gradient_relative_l2_vs_torch':rel,
                    'loss_error':abs(e['loss']-loss.item())}
                assert rel<2e-4,row;out.append(row)
    r=Recipe(kind='transformer',ff=256,rope=True);record=ordered_records(1,4,20261418,fixed=256)[0]
    x,y=trim(record);weights=[];grads=[]
    for micro in (4,2,1):
        p=single_policy(r,'attention',4,256,micro,'attention',4)
        with Runner(p,seed=20261018) as n:
            e=n.update(x,y,float(.001/30));weights.append(n.buffers['weights'].copy());grads.append(n.buffers['gradients'].copy()/(4//micro))
            if micro<4:
                row={'family':'attention-full-shape','microbatch':micro,
                    'normalized_gradient_relative_l2_vs_full':float(np.linalg.norm(grads[-1]-grads[0])/np.linalg.norm(grads[0])),
                    'weight_relative_l2_after_one_update':float(np.linalg.norm(weights[-1]-weights[0])/np.linalg.norm(weights[0]))}
                assert row['normalized_gradient_relative_l2_vs_full']<2e-5,row;out.append(row)
    result={'checks':out,'gradient_buffer_contract':'raw sum of microbatch means; optimizer applies1/accumulation internally',
        'scope':'Checks normalization and independent gradients, not equality of long nonconvex training trajectories.'}
    (ROOT/'results/accumulation-check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
