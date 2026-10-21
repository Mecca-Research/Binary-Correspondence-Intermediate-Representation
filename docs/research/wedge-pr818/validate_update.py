"""Independent AD and legacy/merged state parity for the actual research overlays."""
from validate_query import *
from probe_query import guarded_geometry,control_models,budgets
from study import recipe

def legacy_parity(rng):
    out=[]
    for model,new,old in (('global','guard_compact','old_guard_compact'),('query_full','query_guard_compact','old_query_guard_compact'),('query_matched','query_guard_compact','old_query_guard_compact'),('attention','attention','old_attention')):
        for t in (17,64,128):
            r=replace(recipe(model),context=t,batch=2)
            x=rng.integers(0,260,(2,t),dtype=np.uint32);y=np.full_like(x,IGNORE);y[:,-1]=x[:,1]
            with Native(r,seed=83,threads=4,library_name=old) as before,Native(r,seed=83,threads=4,library_name=new) as after:
                a=before.forward(x).copy();b=after.forward(x).copy();assert np.array_equal(a,b)
                l=before.backward(x,y);ll=after.backward(x,y)
                assert l==ll and np.array_equal(before.buffers['gradients'],after.buffers['gradients'])
                for _ in range(3):
                    before.train(x[None],y[None],[.0005]);after.train(x[None],y[None],[.0005]);assert before.digest()==after.digest(),(model,t)
                out.append({'model':model,'context':t,'forward':'bitwise equal','gradients':'bitwise equal','state_sha256':after.digest()})
    return out

def bf16_checks(rng):
    out=[]
    for model in ('query_full','attention'):
        r=replace(recipe(model),ff=256,context=32,batch=2,contract=26)
        variant='query_guard_compact' if model=='query_full' else 'attention'
        x=rng.integers(0,260,(2,32),dtype=np.uint32);y=np.roll(x,-1,1)
        vals=[]
        for name in (variant+'_probe',variant+'_nopanels'):
            with Native(r,seed=33,threads=4,library_name=name) as n:
                if name.endswith('_probe'):
                    n.lib.bcir_decoder_workspace_probe.restype=SZ;n.lib.bcir_decoder_panels_probe.restype=ct.c_uint
                    n.lib.bcir_decoder_workspace_probe();n.lib.bcir_decoder_panels_probe()
                n.train(x[None],y[None],[.0005]);vals.append(n.digest())
                if name.endswith('_probe'):
                    observed=int(n.lib.bcir_decoder_workspace_probe());bits=int(n.lib.bcir_decoder_panels_probe())
                    assert observed<=n.state.workspace_bytes and bits==7,(observed,n.state.workspace_bytes,bits)
                    out.append({'model':model,'contract':26,'ff':256,'panel_paths_bitmask':bits,'packing_bytes':int(n.state.workspace_bytes),'observed_max_packing_bytes':observed,'state_sha256':n.digest()})
        assert vals[0]==vals[1],('panel parity',model)
    return out

if __name__=='__main__':
    rng=np.random.default_rng(202610219)
    result={'legacy_parity':legacy_parity(rng),'models':model_checks(rng),'guarded_geometry':guarded_geometry(rng),
        'controls':control_models(rng),'state':state_checks(rng),'budget_transitions':budgets(rng),'bf16_panels':bf16_checks(rng)}
    (ROOT/'results/validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
