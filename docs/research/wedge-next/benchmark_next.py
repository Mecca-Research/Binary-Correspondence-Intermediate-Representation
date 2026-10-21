"""Serialized alternating whole-update benchmarks, with exact state evidence."""
from candidates import *
from run_experiments import bytes_data,lm_batches
import argparse
pa=argparse.ArgumentParser();pa.add_argument('--threads',type=int,default=1);pa.add_argument('--mode',choices=['execution','architecture'],default='execution');args=pa.parse_args()
data,_=bytes_data();rows=[]
models=[(lib,Recipe(kind='global',ff=257),lib) for lib in EXACT]+[('attention',ARCH['attention'][0],'attention')] if args.mode=='execution' else [(name,*pair) for name,pair in ARCH.items()]
contexts=(128,1024) if args.mode=='execution' else (128,512,1024)
for t in contexts:
    x,y=lm_batches(data['validation'],13,1,t,123)
    for rep in range(6):
        order=models[rep%len(models):]+models[:rep%len(models)]
        for name,r0,lib in order:
            r=replace(r0,batch=1,context=t)
            with Native(r,seed=100,threads=args.threads,library_name=lib) as n:
                n.train(x[:3],y[:3],[.0001]*3);seconds,events=n.train(x[3:],y[3:],[.0001]*10)
                row={'name':name,'library':lib,'threads':args.threads,'batch':1,'context':t,'rep':rep,
                    'ms_per_update':seconds*100,'parameters':int(n.plan.parameters),
                    'activation_bytes':int(n.plan.activations)*4,'scratch_bytes':int(n.plan.scratch)*4,
                    'arena_bytes':int(n.plan.activations+n.plan.scratch)*4,'persistent_bytes':int(n.plan.parameters)*16,
                    'state_sha256':n.digest(),'last_loss':events[-1]['loss']}
                rows.append(row);print(json.dumps(row),flush=True)
(ROOT/'results'/f'benchmark-{args.mode}-{args.threads}thread.json').write_text(json.dumps(rows,indent=2))
