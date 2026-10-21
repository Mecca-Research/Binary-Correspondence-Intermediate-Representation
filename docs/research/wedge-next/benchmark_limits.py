"""Fresh alternating timing block for additional endpoints, no stacked speedups."""
from candidates import *
from run_experiments import bytes_data,lm_batches
import argparse
p=argparse.ArgumentParser();p.add_argument('--threads',type=int,required=True);a=p.parse_args()
data,_=bytes_data();rows=[];models=('baseline','fast','compact','frame_ff_compact','layer_compact','layer_frame_ff','hybrid','precise')
t=1024;x,y=lm_batches(data['validation'],13,1,t,123);r=Recipe(kind='global',ff=257,batch=1,context=t)
for rep in range(6):
    for name in models[rep%len(models):]+models[:rep%len(models)]:
        with Native(r,seed=100,threads=a.threads,library_name=name) as n:
            n.train(x[:3],y[:3],[.0001]*3);sec,ev=n.train(x[3:],y[3:],[.0001]*10)
            row={'name':name,'context':t,'threads':a.threads,'rep':rep,'ms_per_update':sec*100,
                'parameters':int(n.plan.parameters),'arena_bytes':int(n.plan.activations+n.plan.scratch)*4,
                'state_sha256':n.digest(),'last_loss':ev[-1]['loss']}
            rows.append(row);print(json.dumps(row),flush=True)
(ROOT/'results'/f'benchmark-limits-{a.threads}thread.json').write_text(json.dumps(rows,indent=2))
