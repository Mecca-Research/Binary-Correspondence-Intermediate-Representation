"""Alternating, same-session native latency/memory comparison at matched shapes."""
from experiment import *
from run_experiments import bytes_data,lm_batches
import resource, argparse

parser=argparse.ArgumentParser();parser.add_argument('--threads',type=int,default=4);parser.add_argument('--reps',type=int,default=7)
args=parser.parse_args()
data,_=bytes_data();rows=[]
for context in [64,128,256,512,1024]:
    x,y=lm_batches(data['validation'],13,1,context,123)
    for rep in range(args.reps):
        kinds=('transformer','grassmann','symmetric','global','global_symmetric','projector')
        kinds=kinds[rep%len(kinds):]+kinds[:rep%len(kinds)]
        for kind in kinds:
            r=Recipe(kind=kind,ff=256 if kind=='transformer' else 250 if kind=='projector' else 257,batch=1,context=context)
            with Native(r,seed=100,threads=args.threads) as m:
                m.train(x[:3],y[:3],[.0001]*3)
                seconds,events=m.train(x[3:],y[3:],[.0001]*10)
                row={'kind':kind,'context':context,'rep':rep,'batch':1,'threads':args.threads,
                    'ms_per_update':seconds*100,'transient_bytes':int((m.plan.activations+m.plan.scratch)*4),
                    'total_arena_bytes':int((4*m.plan.parameters+m.plan.activations+m.plan.scratch)*4),
                    'state_sha256':m.digest(),'last_loss':events[-1]['loss']}
                rows.append(row);print(json.dumps(row),flush=True)
(ROOT/'results'/('scaling.json' if args.threads==4 else f'scaling-{args.threads}thread.json')).write_text(json.dumps(rows,indent=2))
