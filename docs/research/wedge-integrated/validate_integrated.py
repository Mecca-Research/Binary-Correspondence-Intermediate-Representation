"""Focused native-policy, dynamic-shape, gradient, resume and refusal checks."""
from policy import *
from tasks import ordered_records,trim
from calibrate import single_policy
import torch,importlib
from torch.nn import functional as F
torch.set_num_threads(1)

def force_policy(r,family,specs,threads):
    shapes=[(b,t) for b,t,m,v in specs];p=BudgetPolicy(r,family,shapes,128*2**20,threads)
    p.choices={(b,t):candidate(r,b,t,m,v,threads,p.staging) for b,t,m,v in specs}
    p.workspace_floats=max(c['activations']+c['scratch'] for c in p.choices.values())
    p.resident_bound_bytes=next(iter(p.choices.values()))['fixed_bytes']+p.workspace_floats*4
    p.policy_id=ident(p.to_dict(include_id=False));return p

if __name__=='__main__':
    results={'transitions':[],'trimming':[],'independent_gradients':[]};rng=np.random.default_rng(27108)
    r=Recipe(kind='global',width=32,layers=3,ff=48)
    specs=[(4,64,4,'compact'),(4,128,4,'frame_ff_compact'),(4,256,4,'layer_frame_ff')]
    # One persistent optimizer/pool while both physical shape and C module change.
    for threads in (1,4):
        p=force_policy(r,'global',specs,threads);reference=None
        with Runner(p,seed=100) as n:
            for step,(b,t,m,v) in enumerate(specs*2):
                x=rng.integers(0,260,(b,t),dtype=np.uint32);y=np.full_like(x,IGNORE);y[:,-1]=x[:,1]
                before=n.snapshot();event=n.update(x,y,.0005)
                with Native(replace(r,batch=b,context=t),seed=100,threads=threads,library_name='fast') as ref:
                    ref.restore(before);ref.train(x[None],y[None],[.0005]);assert n.digest()==ref.digest(),('transition',threads,step)
                if step==2:saved=n.snapshot()
            final=n.digest();n.restore(saved)
            # Restore is verified by replaying retained effective updates separately below.
            results['transitions'].append({'threads':threads,'bind_count':n.bind_count,'final_state_sha256':final,'per_update_native_reference':'bitwise identical'})
    # Resume across a different physical shape and reconstruction library.
    p=force_policy(r,'global',specs,4);sequence=[]
    for b,t,m,v in specs*2:
        x=rng.integers(0,260,(b,t),dtype=np.uint32);y=np.roll(x,-1,1);sequence.append((x,y))
    with Runner(p,seed=100) as n:
        for x,y in sequence[:3]:n.update(x,y,.0005)
        state=n.snapshot()
        for x,y in sequence[3:]:n.update(x,y,.0005)
        expected=n.digest()
    with Runner(p,seed=999) as n:
        n.restore(state)
        for x,y in sequence[3:]:n.update(x,y,.0005)
        assert n.digest()==expected
    results['resume_sha256']=expected
    cross_thread=[]
    for threads in (1,4):
        pp=force_policy(r,'global',specs,threads)
        with Runner(pp,seed=100) as n:
            for x,y in sequence:n.update(x,y,.0005)
            cross_thread.append(n.digest())
    assert cross_thread[0]==cross_thread[1]
    results['cross_thread_dynamic_shapes_sha256']=cross_thread[0]
    full=Recipe(kind='global',ff=257)
    pref=single_policy(full,'global',4,1024,1,'fast',4)
    resize=[]
    with Runner(pref,seed=717) as ref,Runner(pref,seed=717) as n:
        weight_pointer=n.buffers['weights'].ctypes.data
        for budget in (96,32,20,64):
            pnew=BudgetPolicy(full,'global',[(4,1024)],budget*2**20,4,micro_mode='one')
            before=n.digest();n.reconfigure(pnew);assert n.digest()==before and n.buffers['weights'].ctypes.data==weight_pointer
            x=rng.integers(0,260,(4,1024),dtype=np.uint32);y=np.roll(x,-1,1)
            event=n.update(x,y,.0001);ref.update(x,y,.0001);assert n.digest()==ref.digest()
            resize.append({'budget_mib':budget,'variant':event['variant'],'resident_bound_bytes':pnew.resident_bound_bytes,'state_sha256':n.digest()})
    results['budget_resize']=resize
    # Removing masked suffixes preserves logits, and is compared numerically for
    # gradients because the blocked weight-reduction partition changes with N.
    for t in (64,128):
        rec=ordered_records(1,4,114+t,fixed=t)[0];xx,yy=trim(rec)
        with Native(replace(r,batch=4,context=256),seed=100,threads=1,library_name='compact') as padded:
            full=padded.forward(rec['x']).copy();loss=padded.backward(rec['x'],rec['y']);g=padded.buffers['gradients'].copy()
        with Native(replace(r,batch=4,context=t),seed=100,threads=1,library_name='compact') as cropped:
            pred=cropped.forward(xx).copy();ce=cropped.backward(xx,yy);gg=cropped.buffers['gradients'].copy()
        row={'length':t,'max_logit_error':float(np.max(np.abs(full[:,:t]-pred))),
             'loss_error':abs(loss-ce),'gradient_max_error':float(np.max(np.abs(g-gg))),
             'gradient_relative_l2':float(np.linalg.norm(g-gg)/np.linalg.norm(g)),
             'gradient_bitwise_equal':bool(np.array_equal(g,gg))}
        assert row['max_logit_error']<1e-6 and row['loss_error']<1e-6 and row['gradient_relative_l2']<2e-5,row
        results['trimming'].append(row)
    for family,variant,module in (('global','compact','oracle'),('guard','guard_compact','oracle_precise'),('hybrid','hybrid_compact','oracle_hybrid')):
        rr=replace(r,width=32,layers=2,ff=48,batch=2,context=11)
        x=rng.integers(0,260,(2,11),dtype=np.uint32)
        if family=='guard':x[:,1]=x[:,0]
        y=np.full_like(x,IGNORE);y[:,-1]=x[:,1]
        with Native(rr,seed=100,threads=1,library_name=variant) as n:
            logits=n.forward(x).copy();loss=n.backward(x,y);g=n.buffers['gradients'].copy()
            model=importlib.import_module(module).torch_model_next(rr,n.buffers['weights']);p=model(torch.tensor(x.astype(np.int64)))
            mask=torch.tensor(y!=IGNORE);ty=torch.tensor(y.astype(np.int64));ce=F.cross_entropy(p[mask],ty[mask]);ce.backward()
            gg=np.concatenate([v.grad.numpy().ravel() for v in model.params])
            row={'family':family,'logit_max_error':float(np.max(np.abs(logits-p.detach().numpy()))),
                'gradient_max_error':float(np.max(np.abs(g-gg))),'gradient_relative_l2':float(np.linalg.norm(g-gg)/np.linalg.norm(gg))}
            assert row['logit_max_error']<1e-5 and row['gradient_max_error']<1e-4 and row['gradient_relative_l2']<2e-4,row
            results['independent_gradients'].append(row)
    p=single_policy(r,'global',2,64,1,'compact',2)
    with Runner(p) as n:
        x=np.zeros((2,64),np.uint32);y=np.full_like(x,IGNORE);before=n.digest()
        for kind in ('all-masked','unequal-microbatch-counts'):
            if kind.startswith('unequal'):y[0,-1]=0
            try:n.update(x,y,.0005)
            except ValueError:pass
            else:raise AssertionError(kind+' should be refused')
            assert n.digest()==before
    results['loss_weight_refusals']='pass: all masked and unequal supervised counts'
    try:BudgetPolicy(Recipe(kind='global',ff=257),'global',[(4,1024)],16*2**20,4,micro_mode='all')
    except MemoryError:pass
    else:raise AssertionError('impossible budget admitted')
    results['impossible_budget']='refused before array allocation'
    for variant in ('compact_ubsan','layer_frame_ff_ubsan','guard_compact_ubsan','hybrid_compact_ubsan'):
        p=force_policy(r,'global',[(2,64,2,variant)],2)
        with Runner(p) as n:
            for x,y in sequence[:1]:n.update(x[:2,:64],y[:2,:64],.0005)
    results['ubsan']='pass: compact, layer/frame/FF, guarded compact, hybrid compact'
    rr=replace(r,batch=2,context=64)
    with Native(rr,threads=4,library_name='guard_audit') as n:
        n.lib.wedge_audit_get.restype=ct.c_ulonglong;n.lib.wedge_audit_reset()
        x=rng.integers(0,260,(2,64),dtype=np.uint32);a=n.forward(x).copy()
        counts=[int(n.lib.wedge_audit_get(i)) for i in range(9)]
        assert counts[0]==2*64*r.layers and sum(counts[3:])==counts[0] and counts[2]<=counts[1]<=counts[0],counts
        with Native(rr,threads=4,library_name='guard_compact',weights=n.buffers['weights']) as ref:assert np.array_equal(a,ref.forward(x))
        results['precision_audit_counters']=counts
    (ROOT/'results/validation.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
