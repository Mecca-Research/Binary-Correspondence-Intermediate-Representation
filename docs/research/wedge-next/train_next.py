"""Frozen native-C comparison. Validation selects; held-out samples never tune."""
from candidates import *
from run_experiments import bytes_data,lm_batches,recall_batches,evaluate,rates
import datetime,platform,subprocess,argparse

def variable_recall(steps,b,t,seed):
    rng=np.random.default_rng(seed);x=rng.integers(16,32,(steps,b,t),dtype=np.uint32)
    lengths=rng.choice([64,128,256],(steps,b));x[:,:,0]=256;x[:,:,1]=rng.integers(0,16,(steps,b),dtype=np.uint32)
    y=np.full_like(x,IGNORE)
    for a in range(steps):
        for c in range(b):x[a,c,lengths[a,c]-1]=257;y[a,c,lengths[a,c]-1]=x[a,c,1]
    return x,y

def ordered_recall(steps,b,t,seed):
    rng=np.random.default_rng(seed);x=rng.integers(0,16,(steps,b,t),dtype=np.uint32)
    x[:,:,0]=256;x[:,:,-1]=257;y=np.full_like(x,IGNORE);y[:,:,-1]=x[:,:,1]
    return x,y

TASKS={'byte_lm':{'steps':600,'batch':4,'context':64},
       'recall_fixed':{'steps':400,'batch':8,'context':128},
       'recall_variable':{'steps':400,'batch':4,'context':256},
       'recall_ordered':{'steps':600,'batch':8,'context':128}}
TEST_SEEDS={'byte_lm':20261029,'recall_fixed':20261030,'recall_variable':20261031,'recall_ordered':20261101}

def data_for(task,n,b,t,seed,data,split):
    if task=='byte_lm':return lm_batches(data[split],n,b,t,seed)
    if task=='recall_variable':return variable_recall(n,b,t,seed)
    if task=='recall_ordered':return ordered_recall(n,b,t,seed)
    return recall_batches(n,b,t,seed)

def run(task,name,seed,data):
    path=ROOT/'results'/f'{task}-{name}-{seed}.json'
    if path.exists():return
    d=TASKS[task];r0,lib=ARCH[name];r=replace(r0,batch=d['batch'],context=d['context']);steps=d['steps']
    train=data_for(task,steps,r.batch,r.context,seed+200,data,'train')
    val=data_for(task,32 if task=='byte_lm' else 64,r.batch,r.context,103,data,'validation')
    test=data_for(task,64 if task=='byte_lm' else 256,r.batch,r.context,TEST_SEEDS[task],data,'test')
    history=[];events=[];segments=[];best=None;bestce=float('inf')
    with Native(r,seed=seed,threads=4,library_name=lib) as n:
        for off in range(0,steps,100):
            seconds,ev=n.train(train[0][off:off+100],train[1][off:off+100],rates(steps)[off:off+100]);events+=ev
            result=evaluate(n,val);history.append({'step':off+100,'validation':result});segments.append({'steps':100,'seconds':seconds})
            if result['ce_nats']<bestce:bestce=result['ce_nats'];best=n.snapshot();beststep=off+100
            print(json.dumps({'task':task,'name':name,'seed':seed,'step':off+100,'validation':result,'ms_per_update':seconds*10}),flush=True)
        n.restore(best);test_result=evaluate(n,test);digest=n.digest()
        ck=ROOT/'checkpoints'/f'{task}-{name}-{seed}.npz';ck.parent.mkdir(exist_ok=True)
        np.savez_compressed(ck,**best,recipe=json.dumps(asdict(r)),seed=seed,library_name=lib)
        result={'task':task,'name':name,'seed':seed,'recipe':asdict(r),'library_name':lib,'selected_step':beststep,
            'parameters':int(n.plan.parameters),'transient_bytes':int((n.plan.activations+n.plan.scratch)*4),
            'history':history,'events':events,'segments':segments,'test':test_result,'state_sha256':digest,
            'checkpoint':str(ck.relative_to(ROOT)),'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),
            'input_sha256':hashlib.sha256(train[0].tobytes()+train[1].tobytes()).hexdigest(),
            'supervised_tokens':int((train[1]!=IGNORE).sum()),'physical_input_tokens':int(train[0].size)}
    path.write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze-only',action='store_true');p.add_argument('--task');args=p.parse_args()
    data,meta=bytes_data()
    protocol={'stage':'predeclared architecture and memory follow-up','bcir_commit':'aaa2e6c0ee0ee15b83fe4d0373b48bd893e80d6d',
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seeds':list(SEEDS),'tasks':TASKS,
        'task_models':{'byte_lm':list(ARCH),'recall_fixed':list(ARCH),
            'recall_variable':['global','lowrank32_budget','multi12_budget'],
            'recall_ordered':['global','attention']},
        'selection':'lowest validation CE at 100-step intervals; no test tuning; no test-driven candidate changes',
        'optimizer':{'lr_peak':.001,'warmup':30,'floor':.0001,'betas':[.9,.95],'epsilon':1e-8,'decay':.1,'clip':1},
        'test_seeds':TEST_SEEDS,'data':meta,
        'variable_length':'single physical context256, per-example query placed at sampled length64/128/256, zero loss elsewhere; future padding is causally unreachable',
        'ordered_recall':'first value and all distractors share tokens0..15; query257 at end; loss only at query. This challenges the disjoint-vocabulary pooled-content shortcut.',
        'multilinear':'output-mode low-rank factorization of skew readout tensor; product of two rank12 subspaces; restrictions change model class and need training, no lossless-compression claim',
        'host':{'cpu':next(l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),
                'cpu_quota':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),'memory_limit':Path('/sys/fs/cgroup/memory.max').read_text().strip(),
                'python':sys.version,'numpy':np.__version__,'gcc':subprocess.check_output(['gcc','--version'],text=True).splitlines()[0]},
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'train_next.py',ROOT/'experiment.py',ROOT/'candidates.py',ROOT/'build_next.py',ROOT/'multi_core.h']}}
    pf=ROOT/'protocol.json'
    if pf.exists():protocol=json.loads(pf.read_text())
    else:pf.write_text(json.dumps(protocol,indent=2))
    if args.freeze_only:raise SystemExit
    for task,names in protocol['task_models'].items():
        if args.task and task!=args.task:continue
        for si,seed in enumerate(SEEDS):
            order=names[si%len(names):]+names[:si%len(names)]
            for name in order:run(task,name,seed,data)
