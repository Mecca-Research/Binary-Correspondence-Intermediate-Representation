"""Execute calibrated and simple policies on new inputs under declared budgets."""
from policy import *
from tasks import bytes_data,lm_batches
import resource

def rss_bytes():
    line=next(x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:'))
    return int(line.split()[1])*1024

if __name__=='__main__':
    calibration=json.loads((ROOT/'results/calibration.json').read_text());lookup=calibration['lookup_ms'];data,_=bytes_data()
    r=Recipe(kind='global',ff=257);rows=[];refusals=[];policies={}
    # Six held-out repetitions, two warmups plus four timed effective updates.
    x,y=lm_batches(data['train'],6,4,1024,20261711)
    for rep in range(6):
        cases=[(b,kind) for b in (20,24,32,48,64,96,160) for kind in ('calibrated','heuristic')]
        cases=cases[rep%len(cases):]+cases[:rep%len(cases)]
        for budget,kind in cases:
            p=BudgetPolicy(r,'global',[(4,1024)],budget*2**20,4,lookup if kind=='calibrated' else None,micro_mode='all')
            policies[p.policy_id]=p.to_dict();rss_before=rss_bytes();begin=time.perf_counter()
            with Runner(p,seed=711) as n:
                startup=time.perf_counter()-begin
                for xx,yy in zip(x[:2],y[:2]):n.update(xx,yy,.0001)
                events=[n.update(xx,yy,.0001) for xx,yy in zip(x[2:],y[2:])]
                c=p.choices[(4,1024)]
                row={'budget_mib':budget,'kind':kind,'rep':rep,'policy_id':p.policy_id,
                    'variant':c['variant'],'microbatch':c['microbatch'],'accumulation':c['accumulation'],
                    'resident_bound_bytes':p.resident_bound_bytes,'arena_bytes':p.workspace_floats*4,
                    'wall_ms_per_update':1000*sum(e['wall_seconds'] for e in events)/4,
                    'native_ms_per_update':1000*sum(e['native_seconds'] for e in events)/4,
                    'startup_seconds':startup,'state_sha256':n.digest(),'last_loss':events[-1]['loss'],
                    'rss_before':rss_before,'rss_after_updates':rss_bytes(),'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
                assert p.resident_bound_bytes<=budget*2**20
                rows.append(row);print(json.dumps(row),flush=True)
    for budget in (8,16,18):
        try:BudgetPolicy(r,'global',[(4,1024)],budget*2**20,4,lookup,micro_mode='all')
        except MemoryError as e:refusals.append({'budget_mib':budget,'reason':str(e)})
        else:raise AssertionError('under-minimum storage budget admitted')
    (ROOT/'results/budget-probe.json').write_text(json.dumps({'rows':rows,'policies':policies,'refusals':refusals,
        'input_sha256':hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest(),
        'scope':'Controlled live native arrays, pool-control buffer, staged effective update and fixed C records. Excludes thread stacks, code, Python/NumPy metadata, datasets, checkpoint copies and allocator-retained pages; not an RSS cap.',
        'calibration_seconds':calibration['total_wall_seconds'],'calibration_repetitions':3,'heldout_repetitions':6},indent=2))
    print('budget probe complete')
