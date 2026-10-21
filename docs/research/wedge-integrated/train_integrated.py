"""Frozen 30-run integration study. Validation selects; final tests are separate."""
from policy import *
from tasks import *
from run_experiments import rates
import datetime,subprocess,argparse,platform

SEEDS=(20261018,20261019,20261020)
CASES={
 'ordered_fixed_global':{'task':'ordered','family':'global','mode':'fixed','budget_mib':32},
 'ordered_padded_global':{'task':'ordered','family':'global','mode':'padded','budget_mib':32},
 'ordered_bucket_global':{'task':'ordered','family':'global','mode':'bucket','budget_mib':32},
 'ordered_bucket_guard':{'task':'ordered','family':'guard','mode':'bucket','budget_mib':32},
 'ordered_bucket_attention':{'task':'ordered','family':'attention','mode':'bucket','budget_mib':64},
 'kv_bucket_global':{'task':'kv','family':'global','mode':'bucket','budget_mib':32},
 'kv_bucket_attention':{'task':'kv','family':'attention','mode':'bucket','budget_mib':64},
 'lm_bucket_global':{'task':'lm','family':'global','mode':'bucket','budget_mib':32},
 'lm_bucket_hybrid':{'task':'lm','family':'hybrid','mode':'bucket','budget_mib':32},
 'lm_bucket_attention':{'task':'lm','family':'attention','mode':'bucket','budget_mib':32},
}

def base_recipe(family):return Recipe(kind='transformer' if family=='attention' else 'global',ff=256 if family=='attention' else 257,rope=family=='attention')
def shapes(task):return [(8,32),(4,64),(2,128)] if task=='lm' else [(4,t) for t in LENGTHS]
def training_records(case,seed,data):
    task=case['task'];steps=600 if task=='lm' else 900
    if task=='lm':return lm_records(steps,seed+400,data)
    if task=='kv':return kv_records(steps,4,seed+400)
    return ordered_records(steps,4,seed+400,fixed=128 if case['mode']=='fixed' else None)

def validation_records(task,data):
    out=[]
    for t in (32,64,128) if task=='lm' else LENGTHS:
        b=256//t if task=='lm' else 4
        out+=eval_records(task,t,32,b,30300+t,data=data,split='validation')
    return out

def freeze():
    _,data=bytes_data()
    files=[ROOT/n for n in ('policy.py','tasks.py','train_integrated.py','experiment.py','native_abi.py','build_integrated.py','oracle.py','oracle_hybrid.py','oracle_precise.py','validate_integrated.py','calibrate.py')]
    files+=sorted((ROOT/'build').glob('*.c'))+sorted(p for p in (ROOT/'vendor').iterdir() if p.is_file())
    protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seeds':list(SEEDS),'cases':CASES,
        'steps':{'ordered':900,'kv':900,'lm':600},'optimizer':{'peak_lr':.001,'warmup':30,'floor_fraction':.1,'beta1':.9,'beta2':.95,'epsilon':1e-8,'decay':.1,'clip':1},
        'selection':'Lowest aggregate validation CE at 100-update intervals, balanced evaluation lengths; no final-test selection.',
        'ordered':'All values and distractors tokens0..15; BOS256 and query257. Fixed128 or balanced batch-homogeneous64/128/256. Padded and bucketed cases have the same canonical prefixes, targets, update order and effective batch4.',
        'kv':'Random unique keys32..47 and values0..15, marker258, filler64..95. Last query specifies one key. Two pairs for first300 updates, four thereafter. Balanced lengths64/128/256; four examples/update.',
        'language':'Balanced lengths32/64/128; effective batch8/4/2 gives exactly256 supervised bytes/update, 153600 bytes/run. Raw WikiText2 train, validation and test splits remain disjoint.',
        'controller':'Quality-controlled runs retain full effective microbatch; budget chooses recomputation. Separate constrained-runtime experiment also chooses microbatch/accumulation.',
        'budgets':'Native array/pool/staging/C-record bounds, not process RSS. Recall attention reference gets64MiB so full effective microbatch4 fits; global/guard use32MiB. All language cases use32MiB.',
        'validation_seed':30300,'final_test_seed_base':20261110,'no_test_driven_candidates':True,'hyperparameter_search':'None; fixed learning rates/budgets/recipes shared by task.',
        'data':data,'foundation':json.loads((ROOT/'FOUNDATION.json').read_text()),
        'calibration_sha256':hashlib.sha256((ROOT/'results/calibration.json').read_bytes()).hexdigest(),
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'host':{'cpu':next(l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),
            'affinity':sorted(__import__('os').sched_getaffinity(0)),'cpu_quota':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
            'memory_limit':Path('/sys/fs/cgroup/memory.max').read_text().strip(),'python':sys.version,'numpy':np.__version__,'gcc':subprocess.check_output(['gcc','--version'],text=True).splitlines()[0]}}
    path=ROOT/'protocol.json'
    if path.exists():
        existing=json.loads(path.read_text())
        for name,h in existing['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
        return existing
    path.write_text(json.dumps(protocol,indent=2));return protocol

def run(name,seed,data,calibration):
    path=ROOT/'results'/f'train-{name}-{seed}.json'
    if path.exists():return
    case=CASES[name];task=case['task'];family=case['family'];r=base_recipe(family)
    records=training_records(case,seed,data);val=validation_records(task,data)
    p=BudgetPolicy(r,family,shapes(task),case['budget_mib']*2**20,4,calibration,micro_mode='full')
    learning_rates=rates(len(records));history=[];segments=[];events=[];bestce=float('inf');best=None
    logical_hash=input_digest(records);logical_positions=sum(row['length']*len(row['x']) for row in records)
    physical_positions=sum((row['x'].shape[1] if case['mode']=='padded' else row['length'])*len(row['x']) for row in records)
    started=time.perf_counter()
    with Runner(p,seed=seed) as n:
        for off in range(0,len(records),100):
            begin=time.perf_counter()
            for row,lr in zip(records[off:off+100],learning_rates[off:off+100]):
                x,y=trim(row,padded=case['mode']=='padded');events.append(n.update(x,y,float(lr)))
            elapsed=time.perf_counter()-begin;segments.append({'updates':100,'seconds':elapsed,'native_seconds':sum(e['native_seconds'] for e in events[-100:])})
            digest=n.digest();metric,_,_=evaluate_records(n,val,task);assert n.digest()==digest
            history.append({'step':off+100,'validation':metric})
            if metric['ce_nats']<bestce:bestce=metric['ce_nats'];best=n.snapshot();beststep=off+100
            print(json.dumps({'case':name,'seed':seed,'step':off+100,'validation':metric,'training_ms_per_update':elapsed*10,
                'resident_mib':p.resident_bound_bytes/2**20,'bind_count':n.bind_count}),flush=True)
        final_digest=n.digest();n.restore(best);selected_digest=n.digest()
        ck=ROOT/'checkpoints'/f'{name}-{seed}.npz'
        np.savez_compressed(ck,**best,recipe=json.dumps(asdict(r)),seed=seed,family=family,case=name)
        result={'case':name,'seed':seed,**case,'steps':len(records),'recipe':asdict(r),'policy':p.to_dict(),
            'parameters':int(n.plan.parameters),'history':history,'segments':segments,'events':events,
            'selected_step':beststep,'selected_validation_ce':bestce,'state_sha256':selected_digest,'final_training_state_sha256':final_digest,
            'checkpoint':str(ck.relative_to(ROOT)),'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),
            'logical_input_sha256':logical_hash,'logical_positions':logical_positions,'physical_positions':physical_positions,
            'supervised_tokens':sum(int((row['y']!=IGNORE).sum()) for row in records),
            'training_seconds':sum(s['seconds'] for s in segments),'native_training_seconds':sum(s['native_seconds'] for s in segments),
            'wall_seconds_including_validation':time.perf_counter()-started,'bind_count_including_validation':n.bind_count,
            'choice_counts_training':n.choice_counts}
        path.write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--freeze-only',action='store_true');args=pa.parse_args()
    protocol=freeze()
    if args.freeze_only:raise SystemExit
    data,_=bytes_data();cal=json.loads((ROOT/'results/calibration.json').read_text())['lookup_ms']
    names=list(CASES)
    for i,seed in enumerate(SEEDS):
        for name in names[i:]+names[:i]:run(name,seed,data,cal)
