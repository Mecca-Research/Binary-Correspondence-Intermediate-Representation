"""Use memory savings for microbatch size at fixed effective batch, real C loop."""
from candidates import *
from experiment import _Optimizer,_Event
from run_experiments import bytes_data,lm_batches
data,_=bytes_data();steps=8;effective=4;t=1024
x,y=lm_batches(data['validation'],steps,effective,t,123);rows=[];budget=64*2**20
models=('baseline','fast','compact','frame_ff_compact','layer_compact')
for rep in range(4):
    for lib in models[rep%len(models):]+models[:rep%len(models)]:
        chosen=None
        for b in (4,2,1):
            r=Recipe(kind='global',ff=257,batch=b,context=t)
            with Native(r,threads=4,library_name=lib) as n:
                needed=int((4*n.plan.parameters+n.plan.activations+n.plan.scratch)*4)+8*b*t
            if needed<=budget:chosen=(b,needed);break
        assert chosen is not None
        b,needed=chosen;accum=effective//b;r=replace(r,batch=b)
        with Native(r,seed=100,threads=4,library_name=lib) as n:
            xx=np.ascontiguousarray(x.reshape(steps,accum,b,t));yy=np.ascontiguousarray(y.reshape(steps,accum,b,t));lr=np.full(steps,.0001,np.float64)
            opt=_Optimizer(.0001,.9,.95,1e-8,.1,1.);ev=(_Event*steps)();done=SZ()
            # Two warmups consume the same first two effective batches.
            for start,count in ((0,2),(2,6)):
                xxx=np.ascontiguousarray(xx[start:start+count]);yyy=np.ascontiguousarray(yy[start:start+count]);rates=np.ascontiguousarray(lr[start:start+count])
                before=time.perf_counter();rc=n.lib.bcir_decoder_run_accum(ct.byref(n.state),ct.byref(opt),ptr(xxx,UP),ptr(yyy,UP),xxx.size,
                    rates.ctypes.data_as(ct.POINTER(ct.c_double)),count,accum,ev,steps,ct.byref(done));seconds=time.perf_counter()-before
                n.check(rc);assert done.value==count
            rows.append({'library':lib,'rep':rep,'effective_batch':effective,'microbatch':b,'accumulation':accum,
                'budget_bytes':budget,'native_buffers_plus_one_input_pair_bytes':needed,'context':t,'threads':4,
                'ms_per_effective_update':seconds*1000/6,'tokens_per_second':6*effective*t/seconds,
                'state_sha256':n.digest(),'note':'Same examples/effective update. Different microbatch shapes have different rounded gradient accumulation orders.'})
            print(json.dumps(rows[-1]),flush=True)
(ROOT/'results/budget.json').write_text(json.dumps(rows,indent=2))
