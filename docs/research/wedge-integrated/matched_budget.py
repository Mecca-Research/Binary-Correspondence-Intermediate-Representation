"""Separate matched-storage follow-up: attention also adapts to 32 MiB.

The main comparison used full effective microbatches for gradient-convention
control. This stage gives attention accumulation, with no optimizer/data tuning.
Fresh paired test samples compare all three selected recipes after selection.
"""
from train_integrated import *
OUT=ROOT/'matched-budget'

if __name__=='__main__':
    (OUT/'results').mkdir(parents=True,exist_ok=True);(OUT/'checkpoints').mkdir(exist_ok=True)
    protocol={'stage':'matched32MiB attention follow-up after main quality evaluation',
        'models':['retained global32','retained attention64','new attention32 with accumulation'],
        'seeds':list(SEEDS),'updates':900,'effective_batch':4,'lengths':[64,128,256],
        'data':'identical ordered-variable prefixes, labels, update order, optimizer and validation as main study',
        'microbatch':'largest feasible from1/2/4; attention has one layout, no timing tuning',
        'fresh_test_seed_base':20261810,'selection':'lowest validation CE at100-update intervals; no final-test checkpoint selection',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));data,_=bytes_data();train_rows=[];test_rows=[]
    for seed in SEEDS:
        result=OUT/'results'/f'train-attention32-{seed}.json'
        if not result.exists():
            records=ordered_records(900,4,seed+400);val=validation_records('ordered',data);r=base_recipe('attention')
            p=BudgetPolicy(r,'attention',shapes('ordered'),32*2**20,4,micro_mode='all')
            events=[];segments=[];history=[];best=None;bestce=float('inf')
            with Runner(p,seed=seed) as n:
                lr=rates(900)
                for off in range(0,900,100):
                    begin=time.perf_counter()
                    for row,rate in zip(records[off:off+100],lr[off:off+100]):
                        x,y=trim(row);events.append(n.update(x,y,float(rate)))
                    seconds=time.perf_counter()-begin;segments.append({'steps':100,'seconds':seconds})
                    metric,_,_=evaluate_records(n,val,'ordered');history.append({'step':off+100,'validation':metric})
                    if metric['ce_nats']<bestce:best=n.snapshot();bestce=metric['ce_nats'];beststep=off+100
                    print(json.dumps({'seed':seed,'step':off+100,'validation':metric,'seconds':seconds}),flush=True)
                n.restore(best);ck=OUT/'checkpoints'/f'attention32-{seed}.npz'
                np.savez_compressed(ck,**best,recipe=json.dumps(asdict(r)),seed=seed,family='attention')
                d={'case':'ordered_bucket_attention32','seed':seed,'family':'attention','steps':900,'parameters':int(n.plan.parameters),
                    'policy':p.to_dict(),'selected_step':beststep,'selected_validation_ce':bestce,'state_sha256':n.digest(),
                    'checkpoint':str(ck.relative_to(ROOT)),'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),
                    'logical_input_sha256':input_digest(records),'history':history,'segments':segments,'events':events,
                    'training_seconds':sum(a['seconds'] for a in segments)}
                result.write_text(json.dumps(d,indent=2))
        new=json.loads(result.read_text());train_rows.append(new)
        for d in (json.loads((ROOT/'results'/f'train-ordered_bucket_global-{seed}.json').read_text()),
                  json.loads((ROOT/'results'/f'train-ordered_bucket_attention-{seed}.json').read_text()),new):
            ck=ROOT/d['checkpoint'];assert hashlib.sha256(ck.read_bytes()).hexdigest()==d['checkpoint_sha256']
            state=dict(np.load(ck,allow_pickle=False));r=Recipe(**json.loads(str(state['recipe'])))
            ptest=BudgetPolicy(r,d['family'],[(4,t) for t in (256,1024,2048)],512*2**20,4,micro_mode='all')
            with Runner(ptest) as n:
                n.restore(state);assert n.digest()==d['state_sha256']
                for t in (256,1024,2048):
                    rec=eval_records('ordered',t,32 if t==2048 else 128,4,20261810+t)
                    metric,_,_=evaluate_records(n,rec,'ordered')
                    row={'case':d['case'],'seed':seed,'context':t,'test':metric,'logical_input_sha256':input_digest(rec)}
                    test_rows.append(row);print(json.dumps(row),flush=True)
    (OUT/'results/training.json').write_text(json.dumps(train_rows,indent=2))
    (OUT/'results/evaluation.json').write_text(json.dumps(test_rows,indent=2))
