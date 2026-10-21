"""Separate numerical-family learning follow-up; no tuning of FP32 results.

All 15 cases are declared before their first update. Model/data/selection stay
fixed; only GEMM contract14 becomes26. FF257/250 are deliberately preserved.
"""
from study import *
from dataclasses import replace

OUT=ROOT/'bf16'
BF_CASES=(('kv_curriculum','global'),('kv_curriculum','query_full'),('kv_curriculum','attention'),
          ('ordered','query_full'),('lm','query_full'))


def bf_freeze():
    freeze();OUT.mkdir(exist_ok=True);(OUT/'results').mkdir(exist_ok=True);(OUT/'checkpoints').mkdir(exist_ok=True)
    p=OUT/'protocol.json'
    protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'primary_protocol_sha256':hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),
      'seeds':SEEDS,'cases':BF_CASES,'contract':26,'training_updates':12600,
      'controlled':'Same parameter shapes, initialization, data streams, optimizer, rates, supervised counts, memory envelope, validation frequency and selection as the FP32 primary study.',
      'difference':'Only BF16 GEMM arithmetic plus its declared packing buffers. Attention-FMA bit32 stays off. FP32 optimizer and stored parameters retained.',
      'panel_admission':'Original query/global FF257 and FF250 recipes are not changed to fit FF256 panels. Dedicated FF256 panel timing/parity is a separate synthetic probe.',
      'evaluation':'Rerun fixed regression samples: association4pairs T128/T256 and value-swap/query-change T128; orderedT1024/T2048; LM32/64/128. Samples previously used by the FP32 study, so no claim of fresh holdout.',
      'claim_limit':'This AMD EPYC host lacks AMX. Results cannot reproduce or reject upstream Intel-AMX throughput numbers.'}
    if p.exists():
        old=json.loads(p.read_text());assert old['source_sha256']==protocol['source_sha256'];return old
    p.write_text(json.dumps(protocol,indent=2));return protocol


def run_bf(task,model,seed,data):
    name=f'{task}-{model}-{seed}';path=OUT/'results'/f'train-{name}.json'
    if path.exists():return
    r=replace(recipe(model),contract=26);rs=records(task,seed,data);validation=val_records(task,data)
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
            print(json.dumps({'case':name,'contract':26,'step':off+100,'validation':metric,'ms_per_update':elapsed*10,'buffers_mib':policy.resident_bound_bytes/2**20}),flush=True)
        finaldigest=n.digest();n.restore(best);selected=n.digest()
        ck=OUT/'checkpoints'/f'{name}.npz';np.savez_compressed(ck,**best,recipe=json.dumps(asdict(r)),seed=seed,family=family(model),case=name)
        result={'task':task,'model':model,'seed':seed,'steps':len(rs),'recipe':asdict(r),'parameters':int(n.plan.parameters),'policy':policy.to_dict(),
          'history':history,'segments':segments,'events':events,'selected_step':beststep,'selected_validation_ce':bestce,
          'state_sha256':selected,'final_training_state_sha256':finaldigest,'checkpoint':str(ck.relative_to(ROOT)),
          'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),'input_sha256':input_digest(rs),
          'physical_positions':sum(len(row['x'])*row['length'] for row in rs),'supervised_tokens':sum(int((row['y']!=IGNORE).sum()) for row in rs),
          'training_seconds':sum(s['seconds'] for s in segments),'native_training_seconds':sum(s['native_seconds'] for s in segments)}
        path.write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--freeze-only',action='store_true');pa.add_argument('--task');pa.add_argument('--model');pa.add_argument('--seed',type=int);args=pa.parse_args()
    bf_freeze()
    if args.freeze_only:raise SystemExit
    data,_=bytes_data()
    for i,seed in enumerate(SEEDS):
        for task,model in BF_CASES[i:]+BF_CASES[:i]:
            if (not args.task or task==args.task) and (not args.model or model==args.model) and (not args.seed or seed==args.seed):run_bf(task,model,seed,data)
