"""Predeclared matched native-C study of the full query-aware wedge head."""
from policy import *
from tasks import *
from run_experiments import rates
import datetime,subprocess,os,argparse

SEEDS=(20261018,20261019,20261020)
MODELS=('global','query_full','query_matched','attention')
TASKS=('kv_curriculum','kv_single','ordered','lm')

def recipe(model):
    if model=='attention':return Recipe(kind='transformer',ff=256,rope=True)
    return Recipe(kind='hybrid' if model=='hybrid' else 'global',ff=250 if model=='query_matched' else 257,
                  query_aware=model.startswith('query'),precision_guard=True)

def family(model):return 'query' if model.startswith('query') else model
def shape_set(task):return [(8,32),(4,64),(2,128)] if task=='lm' else [(4,t) for t in LENGTHS]
def records(task,seed,data):
    if task=='lm':return lm_records(600,seed+400,data)
    if task=='ordered':return ordered_records(900,4,seed+400)
    return kv_records(900,4,seed+400,pairs=1 if task=='kv_single' else None)
def val_records(task,data):
    out=[]
    for t in (32,64,128) if task=='lm' else LENGTHS:
        out+=eval_records('lm' if task=='lm' else 'ordered' if task=='ordered' else 'kv',t,32,256//t if task=='lm' else 4,
                          30300+t,data=data,split='validation',pairs=1 if task=='kv_single' else 4)
    return out
def cases():return [(task,model) for task in TASKS for model in (*MODELS,'hybrid') if model!='hybrid' or task=='lm']

def freeze():
    _,meta=bytes_data();files=[ROOT/n for n in ('experiment.py','native_abi.py','policy.py','tasks.py','oracle.py','query_core.h','build_upgrade.py','study.py','validate_update.py','validate_query.py','probe_query.py','precision_core.h','run_experiments.py')]
    files+=sorted((ROOT/'build').glob('*.c'))+sorted((ROOT/'foundation').glob('*.c'))+sorted(p for folder in ('vendor-new','vendor-old') for p in (ROOT/folder).iterdir() if p.is_file())
    protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seeds':SEEDS,'cases':cases(),
      'parameters':{m:sum(n for _,_,_,n in layout(recipe(m))) for m in (*MODELS,'hybrid')},
      'numerical_contract':'All global-family candidates use the inherited FP64 ill-conditioning guard, with zero feature/VJP for absent planes <=1e-6. Query norm is separately clamped at1e-6; contraction rows reduce in FP64, round to FP32. Full231-coordinate Wp readout retained.',
      'initialization':'Stable per-parameter name/seed; unchanged tensors identical across candidates. New Wc uses std1.0, as Wp, for unit-normalized feature scale. No zero-initialized start or per-candidate tuning.',
      'training':{'kv_curriculum':'900 updates B4; unique random keys/values, 2pairs for300 then4; balanced64/128/256.',
        'kv_single':'900 updates B4; same task with1pair throughout; balanced64/128/256.',
        'ordered':'900 updates B4; first-value recall with shared target/distractor alphabet; balanced64/128/256.',
        'lm':'600 updates, exactly256 supervised bytes/update, balanced32/64/128; batches8/4/2; independent rawWikiText2 splits.'},
      'optimizer':{'algorithm':'native C AdamW','peak_lr':.001,'warmup_updates':30,'floor_lr':.0001,'beta1':.9,'beta2':.95,'epsilon':1e-8,'weight_decay':.1,'clip_norm':1.},
      'memory':'32MiB controlled arrays/pool/staging/C-record budget for every model; global/query/hybrid full effective microbatch; attention largest fitting microbatch and accumulation. Not RSS. No timing calibration; choose cheaper fitting reconstruction tier.',
      'selection':'Lowest aggregate validation CE every100 updates. All lengths aggregated; same validation per task/candidate/seed. Initial step not eligible.',
      'hypotheses':['Merged FP32 kernels/updater preserve the previous learned state under the same contract14.','Measure any remaining runtime gain after the earlier aaa2e6c source pin.','Keep geometry, data, initialization, masked loss, memory budget and update count unchanged. BF16 and attention-FMA are separate numerical experiments.'],
      'final_tests':{'seed_base':20262100,'association_pairs':[1,2,4,8],'association_lengths':[64,128,256,512,1024],
        'primary':'kv_curriculum trained models on4pairs T128 and256; kv_single trained models on1pair T128 and256.',
        'examples_per_length_pair':256,'long_examples':128,'counterfactuals':'4pair value swaps preserving token multiset; query-key-only changes at fixed records; report both-correct rates and predicted-in-set rate.',
        'ordered_lengths':[128,256,512,1024,2048],'lm_lengths':[32,64,128,512]},
      'interpretation':'Three seeds, small prototype; no universal capability or throughput claims. No final-test tuning. Any later exploratory extension gets a separate protocol before new data.',
      'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'data':meta,
      'foundation':{'archive_sha256':'51aab5b06fcc17492d3860af931587897e723b09032f40a0b9e4e47b30a9ed32','native_before':'aaa2e6c0ee0ee15b83fe4d0373b48bd893e80d6d','native_after':'237651126684569e5b756dcc8d1c89ed2e86d0e3','PR_head':'f3e7f0be5f467ac154da8d0e096921acbc177ba4'},
      'host':{'cpu':next(l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),
        'affinity':sorted(os.sched_getaffinity(0)),'cpu_quota':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),'memory_limit':Path('/sys/fs/cgroup/memory.max').read_text().strip(),
        'python':sys.version,'numpy':np.__version__,'gcc':subprocess.check_output(['gcc','--version'],text=True).splitlines()[0]}}
    p=ROOT/'protocol.json'
    if p.exists():
        saved=json.loads(p.read_text())
        for name,h in saved['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
        return saved
    assert (ROOT/'results/validation.json').exists(),'run independent validation before freezing'
    p.write_text(json.dumps(protocol,indent=2));return protocol

def run(task,model,seed,data):
    name=f'{task}-{model}-{seed}';path=ROOT/'results'/f'train-{name}.json'
    if path.exists():return
    r=recipe(model);rs=records(task,seed,data);validation=val_records(task,data)
    policy=BudgetPolicy(r,family(model),shape_set(task),32*2**20,4,micro_mode='all' if model=='attention' else 'full')
    learning_rates=rates(len(rs));events=[];history=[];segments=[];bestce=float('inf');best=None
    labeltask='lm' if task=='lm' else 'ordered' if task=='ordered' else 'kv'
    with Runner(policy,seed=seed) as n:
        for off in range(0,len(rs),100):
            start=time.perf_counter()
            for row,lr in zip(rs[off:off+100],learning_rates[off:off+100]):
                x,y=trim(row);events.append(n.update(x,y,float(lr)))
            elapsed=time.perf_counter()-start;segments.append({'updates':100,'seconds':elapsed,'native_seconds':sum(e['native_seconds'] for e in events[-100:])})
            digest=n.digest();metric,_,_=evaluate_records(n,validation,labeltask);assert n.digest()==digest
            history.append({'step':off+100,'validation':metric})
            if metric['ce_nats']<bestce:bestce=metric['ce_nats'];best=n.snapshot();beststep=off+100
            print(json.dumps({'case':name,'step':off+100,'validation':metric,'ms_per_update':elapsed*10,'buffers_mib':policy.resident_bound_bytes/2**20}),flush=True)
        finaldigest=n.digest();n.restore(best);selected=n.digest()
        ck=ROOT/'checkpoints'/f'{name}.npz';np.savez_compressed(ck,**best,recipe=json.dumps(asdict(r)),seed=seed,family=family(model),case=name)
        result={'task':task,'model':model,'seed':seed,'steps':len(rs),'recipe':asdict(r),'parameters':int(n.plan.parameters),'policy':policy.to_dict(),
          'history':history,'segments':segments,'events':events,'selected_step':beststep,'selected_validation_ce':bestce,
          'state_sha256':selected,'final_training_state_sha256':finaldigest,'checkpoint':str(ck.relative_to(ROOT)),
          'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),'input_sha256':input_digest(rs),
          'physical_positions':sum(len(row['x'])*row['length'] for row in rs),'supervised_tokens':sum(int((row['y']!=IGNORE).sum()) for row in rs),
          'training_seconds':sum(s['seconds'] for s in segments),'native_training_seconds':sum(s['native_seconds'] for s in segments)}
        path.write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--freeze-only',action='store_true');pa.add_argument('--task');pa.add_argument('--model');pa.add_argument('--seed',type=int);args=pa.parse_args()
    freeze()
    if args.freeze_only:raise SystemExit
    data,_=bytes_data();cs=cases()
    for i,seed in enumerate(SEEDS):
        for task,model in cs[i:]+cs[:i]:
            if (not args.task or task==args.task) and (not args.model or model==args.model) and (not args.seed or seed==args.seed):run(task,model,seed,data)
