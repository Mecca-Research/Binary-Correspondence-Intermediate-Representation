"""Qualify BF16 workspace and compact rematerialization before learning runs."""
from study_bf16 import *


if __name__=='__main__':
    bf_freeze();rng=np.random.default_rng(818026);out=[]
    for model in ('global','query_full','query_matched'):
        for t in (17,128):
            r=replace(recipe(model),context=t,batch=2,contract=26)
            x=rng.integers(0,260,(2,t),dtype=np.uint32);y=np.full_like(x,IGNORE);y[:,-1]=x[:,1]
            ref=None
            for variant in FAMILIES[family(model)]:
                with Native(r,seed=282,threads=4,library_name=variant) as n:
                    logits=n.forward(x).copy();loss=n.backward(x,y);grads=n.buffers['gradients'].copy()
                    n.train(x[None],y[None],[.0005]);state=n.digest()
                    if ref is None:ref=(logits,loss,grads,state)
                    else:
                        assert np.array_equal(logits,ref[0]),(model,t,variant,'logits')
                        assert loss==ref[1] and np.array_equal(grads,ref[2]),(model,t,variant,'VJP')
                        assert state==ref[3],(model,t,variant,'update')
                    out.append({'model':model,'context':t,'variant':variant,'packing_bytes':int(n.state.workspace_bytes),
                                'masked_loss':loss,'state_sha256':state,'cached_vs_recomputed':'bitwise equal'})
    (OUT/'validation.json').write_text(json.dumps({'rows':out,'checks_passed':len(out)},indent=2))
    print(json.dumps({'checks_passed':len(out)},indent=2))
