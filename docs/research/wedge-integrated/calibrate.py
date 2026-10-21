"""Native-policy calibration on train bytes; quality test data never select plans."""
from policy import *
from tasks import bytes_data,lm_batches
import argparse

def single_policy(r,family,eff,t,m,variant,threads):
    p=BudgetPolicy(r,family,[(eff,t)],512*2**20,threads,micro_mode='all')
    c=candidate(r,eff,t,m,variant,threads,eff*t);c.update({'budget_bytes':p.budget,'calibrated_ms':None,'calibration_kind':'calibration-trial'})
    p.choices={(eff,t):c};p.workspace_floats=c['activations']+c['scratch'];p.resident_bound_bytes=c['resident_bound_bytes']
    p.policy_id=ident(p.to_dict(include_id=False));return p

if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--threads',type=int,default=4);a=pa.parse_args()
    data,_=bytes_data();rows=[];started=time.perf_counter()
    r=Recipe(kind='global',ff=257)
    shapes=[(4,64),(4,128),(4,256),(4,1024),(8,32),(2,128)]
    cases=[]
    for eff,t in shapes:
        for family in ('global','guard','hybrid') if t!=1024 else ('global',):
            for m in ((1,2,4) if t==1024 else (eff,)):
                for variant in FAMILIES[family]:cases.append((family,eff,t,m,variant))
    # Three repetitions with rotating trial order, one warmup + three timed updates.
    for rep in range(3):
        order=cases[rep%len(cases):]+cases[:rep%len(cases)]
        for family,eff,t,m,variant in order:
            x,y=lm_batches(data['train'],4,eff,t,41300+t)
            p=single_policy(r,family,eff,t,m,variant,a.threads)
            with Runner(p,seed=711) as n:
                n.update(x[0],y[0],.0001);events=[n.update(xx,yy,.0001) for xx,yy in zip(x[1:],y[1:])]
                c=p.choices[(eff,t)]
                row={**c,'rep':rep,'wall_ms_per_update':1000*sum(e['wall_seconds'] for e in events)/3,
                    'native_ms_per_update':1000*sum(e['native_seconds'] for e in events)/3,
                    'state_sha256':n.digest(),'last_loss':events[-1]['loss']}
                rows.append(row);print(json.dumps(row),flush=True)
    out={'rows':rows,'lookup_ms':calibration_map(rows),'total_wall_seconds':time.perf_counter()-started,
        'selection':'lowest median measured update wall time among feasible plans; no quality test data used',
        'host_cpu':next(l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),
        'threads':a.threads,'scope':'Three repetitions, one warmup and three timed effective updates. Calibration code/input/allocation overhead is recorded separately.'}
    (ROOT/'results/calibration.json').write_text(json.dumps(out,indent=2));print('calibration complete',out['total_wall_seconds'])
