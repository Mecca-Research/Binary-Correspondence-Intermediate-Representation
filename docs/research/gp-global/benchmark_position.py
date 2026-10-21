"""Alternating T1024 updates for rotary versus identity global paths."""
from run_experiments import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--threads',type=int,default=1);a=p.parse_args()
data,_=bytes_data();x,y=lm_batches(data['validation'],13,1,1024,123);rows=[]
models=[('transformer','transformer'),('global','global'),('global','global_norope'),('global_symmetric','global_symmetric_norope')]
for rep in range(6):
    order=models[rep%4:]+models[:rep%4]
    for kind,lib in order:
        r=Recipe(kind=kind,ff=256 if kind=='transformer' else 257,batch=1,context=1024)
        with Native(r,seed=100,threads=a.threads,library_name=lib) as n:
            n.train(x[:3],y[:3],[.0001]*3);seconds,_=n.train(x[3:],y[3:],[.0001]*10)
            row={'kind':kind,'library_name':lib,'rep':rep,'threads':a.threads,'context':1024,'batch':1,
                'ms_per_update':seconds*100,'state_sha256':n.digest(),
                'transient_bytes':int((n.plan.activations+n.plan.scratch)*4)}
            rows.append(row);print(json.dumps(row),flush=True)
(ROOT/'position-ablation'/f'timing-{a.threads}thread.json').write_text(json.dumps(rows,indent=2))
