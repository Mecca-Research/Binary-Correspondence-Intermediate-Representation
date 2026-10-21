"""Reproduce inherited absent-plane sensitivity and qualify the common guard."""
from validate_query import *
out=[]
for query in (False,True):
    for duplicate in (False,True):
        rng=np.random.default_rng(754)
        x=rng.integers(0,260,(2,11),dtype=np.uint32);y=rng.integers(0,260,(2,11),dtype=np.uint32)
        if duplicate:x[:,1]=x[:,0]
        for guard in (False,True):
            r=Recipe(kind='global',width=32,layers=2,ff=48,batch=2,context=11,query_aware=query,precision_guard=guard)
            variant=('query_' if query else '')+('guard_' if guard else '')+'compact'
            with Native(r,seed=812,threads=1,library_name=variant) as n:
                logits=n.forward(x).copy();ce=n.backward(x,y);g=n.buffers['gradients'].copy()
                model=torch_model_next(r,n.buffers['weights']);p=model(torch.tensor(x.astype(np.int64)))
                F.cross_entropy(p.flatten(0,1),torch.tensor(y.astype(np.int64)).flatten()).backward()
                gg=np.concatenate([v.grad.numpy().ravel() for v in model.params])
                row={'query':query,'duplicate_first_two':duplicate,'guard':guard,'variant':variant,
                  'gradient_relative':relative(g,gg),'c_gradient_norm':float(np.linalg.norm(g.astype(np.float64))),
                  'torch_gradient_norm':float(np.linalg.norm(gg.astype(np.float64))),
                  'logit_max':float(np.max(abs(logits-p.detach().numpy())))}
                if guard or not duplicate:assert row['gradient_relative']<1e-5,row
                out.append(row)
(ROOT/'results/conditioning-check.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
