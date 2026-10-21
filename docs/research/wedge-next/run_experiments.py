"""Frozen-protocol, serial native training; no test-based tuning or selection."""
from experiment import *
from dataclasses import replace
import argparse, datetime, platform, resource, subprocess
import pyarrow.parquet as pq

KINDS=('global','global_symmetric','projector')
SEEDS=(20261017,20261018,20261019)

def bytes_data():
    out={};meta={}
    for split in ['train','validation','test']:
        path=ROOT/'data'/f'wikitext2-{split}.parquet'
        text=''.join(pq.read_table(path,columns=['text'])['text'].to_pylist()).encode('utf-8')
        out[split]=np.frombuffer(text,np.uint8)
        meta[split]={'parquet_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'utf8_sha256':hashlib.sha256(text).hexdigest(),'bytes':len(text),
                'source_url':f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/{split}-00000-of-00001.parquet'}
    return out,meta

def lm_batches(data,n,b,t,seed):
    rng=np.random.default_rng(seed);starts=rng.integers(0,len(data)-t-1,(n,b))
    a=data[starts[...,None]+np.arange(t+1)]
    return np.ascontiguousarray(a[...,:-1],np.uint32),np.ascontiguousarray(a[...,1:],np.uint32)

def recall_batches(n,b,t,seed):
    rng=np.random.default_rng(seed)
    x=rng.integers(16,32,(n,b,t),dtype=np.uint32)
    x[:,:,0]=256;x[:,:,1]=rng.integers(0,16,(n,b),dtype=np.uint32);x[:,:,-1]=257
    y=np.full_like(x,IGNORE);y[:,:,-1]=x[:,:,1]
    return x,y

def evaluate(m,xy):
    ce=correct=count=0
    for x,y in zip(*xy):
        a,b,c=metrics(m.forward(x),y);ce+=a;correct+=b;count+=c
    return {'ce_nats':ce/count,'bits_per_byte':ce/count/math.log(2),'accuracy':correct/count,
            'scored_tokens':count}

def rates(steps):
    a=np.arange(1,steps+1);return .001*np.minimum(a/30,1)*(.1+.9*.5*(1+np.cos(np.pi*np.maximum(a-30,0)/max(steps-30,1))))

def run(task,kind,seed,data,protocol):
    r=Recipe(kind=kind,ff=250 if kind=='projector' else 257,
             batch=4 if task=='wikitext2_byte' else 8,context=64 if task=='wikitext2_byte' else 128)
    steps=protocol['steps'][task];nr=ROOT/'results'/f'{task}-{kind}-{seed}.json'
    if nr.exists():return json.loads(nr.read_text())
    if task=='wikitext2_byte':
        train=lm_batches(data['train'],steps,r.batch,r.context,seed+200)
        val=lm_batches(data['validation'],32,r.batch,r.context,101)
        test=lm_batches(data['test'],64,r.batch,r.context,20261020)
    else:
        train=recall_batches(steps,r.batch,r.context,seed+200)
        val=recall_batches(64,r.batch,r.context,103)
        test=recall_batches(256,r.batch,r.context,20261021)
    history=[];best=None;best_ce=float('inf');segments=[];all_events=[]
    with Native(r,seed=seed,threads=4) as m:
        start=time.perf_counter();history.append({'step':0,'validation':evaluate(m,val)})
        for off in range(0,steps,100):
            end=min(off+100,steps)
            seconds,events=m.train(train[0][off:end],train[1][off:end],rates(steps)[off:end])
            ev=evaluate(m,val);history.append({'step':end,'validation':ev})
            segments.append({'first_step':off+1,'steps':end-off,'seconds':seconds,'ms_per_update':seconds*1000/(end-off)})
            all_events+=events
            if ev['ce_nats']<best_ce:best_ce=ev['ce_nats'];best=m.snapshot();best_step=end
            print(json.dumps({'task':task,'kind':kind,'seed':seed,'step':end,'validation_ce':ev['ce_nats'],
                'validation_accuracy':ev['accuracy'],'ms_per_step':seconds*1000/(end-off)}),flush=True)
        m.restore(best);final=evaluate(m,test)
        ck=ROOT/'checkpoints'/f'{task}-{kind}-{seed}.npz';ck.parent.mkdir(exist_ok=True)
        np.savez_compressed(ck,**best,recipe=json.dumps(asdict(r)),seed=seed)
        d={'task':task,'kind':kind,'seed':seed,'recipe':asdict(r),'parameters':int(m.plan.parameters),
           'transient_bytes':int((m.plan.activations+m.plan.scratch)*4),
           'total_arena_bytes':int((4*m.plan.parameters+m.plan.activations+m.plan.scratch)*4),
           'input_sha256':hashlib.sha256(train[0].tobytes()+train[1].tobytes()).hexdigest(),
           'train_updates':steps,'train_input_tokens':int(train[0].size),
           'supervised_tokens':int((train[1]!=IGNORE).sum()),'train_seconds':sum(v['seconds'] for v in segments),
           'median_segment_ms':float(np.median([v['ms_per_update'] for v in segments[1:]])),
           'wall_seconds':time.perf_counter()-start,'selected_step':best_step,
           'test':final,'state_sha256':m.digest(),'checkpoint':str(ck.relative_to(ROOT)),
           'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),
           'history':history,'segments':segments,'events':all_events}
    nr.write_text(json.dumps(d,indent=2));return d

def main():
    p=argparse.ArgumentParser();p.add_argument('--only-task');p.add_argument('--only-seed',type=int);args=p.parse_args()
    (ROOT/'results').mkdir(exist_ok=True)
    data,meta=bytes_data()
    protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'bcir_commit':'aaa2e6c0ee0ee15b83fe4d0373b48bd893e80d6d','seeds':list(SEEDS),
       'tasks':['wikitext2_byte','first_token_recall'],'kinds':list(KINDS),
       'steps':{'wikitext2_byte':600,'first_token_recall':400},'threads':4,
       'optimizer':{'name':'native_BCIR_AdamW','peak_lr':.001,'warmup':30,'floor_lr':.0001,
            'beta1':.9,'beta2':.95,'epsilon':1e-8,'decay':.1,'clip':1.0},
       'selection':'lowest validation CE at 100-update intervals; final test once after selection',
       'context':{'wikitext2_byte':64,'first_token_recall':128},
       'matching':'global wedge and symmetric control 525696 parameters; projector FF250 gives 526080 (+0.073%); same width/depth/vocab/pool/optimizer/data per seed',
       'gp_rework':'rank22; current rotary vector and strictly earlier prefix sum divided by sqrt(t); normalized decomposable wedge, matched symmetric bilinear control, or ridge projector U(U^TU+1e-4I)^-1U^T/sqrt(2); pre-RMSNorm and SwiGLU; no local-offset averaging',
       'test_seeds':{'wikitext2_byte':20261020,'first_token_recall':20261021},
       'baseline':'Prior pilot transformer/local wedge/local symmetric selected checkpoints evaluated on these same newly sampled test windows; no additional baseline training. All training and validation batches unchanged.',
       'hypotheses':['Global reach removes the previous hard receptive-field limit but compression may still prevent retrieval.',
          'Prefix scans improve execution independently of wedge versus symmetric geometry.',
          'Projector removes orientation dependence, which may help invariance but discard sequence information.'],
       'frozen_choices':'Three new recipes, no per-model tuning, fixed updates/selection before reading new test outcomes.',
       'not_reproduced':'paper 13-18M WordPiece LM and frozen DistilBERT/SNLI classification result',
       'data':meta,'environment':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
           'cpu_model':next(l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),
           'affinity':sorted(__import__('os').sched_getaffinity(0)),
           'cpu_quota':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
           'memory_limit':Path('/sys/fs/cgroup/memory.max').read_text().strip(),
           'gcc':subprocess.check_output(['gcc','--version'],text=True).splitlines()[0]}}
    pf=ROOT/'protocol.json'
    if pf.exists():protocol=json.loads(pf.read_text())
    else:pf.write_text(json.dumps(protocol,indent=2))
    for task in protocol['tasks']:
        if args.only_task and task!=args.only_task:continue
        for si,seed in enumerate(SEEDS):
            if args.only_seed is not None and seed!=args.only_seed:continue
            order=KINDS[si:]+KINDS[:si]  # balanced ordering across seeds
            for kind in order:run(task,kind,seed,data,protocol)

if __name__=='__main__':main()
