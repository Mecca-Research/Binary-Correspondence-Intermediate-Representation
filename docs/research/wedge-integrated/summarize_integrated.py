"""Audited aggregates; all per-seed/per-repetition evidence remains available."""
from train_integrated import *
import statistics as st,collections

def stats(v):
    v=list(v);return {'mean':st.mean(v),'median':st.median(v),'min':min(v),'max':max(v),'n':len(v)}
def group(rows,keys):
    out=collections.defaultdict(list)
    for r in rows:out[tuple(r[k] for k in keys)].append(r)
    return out
def read(p):return json.loads((ROOT/p).read_text())

if __name__=='__main__':
    protocol=read('protocol.json');audit={'frozen_source_hashes':'pass','checkpoints':[],'paired_inputs':'pass'}
    assert hashlib.sha256((ROOT/'results/calibration.json').read_bytes()).hexdigest()==protocol['calibration_sha256']
    audit['frozen_calibration_hash']='pass'
    for name,h in protocol['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    training=[json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('train-*.json'))];assert len(training)==30
    for (task,seed),rr in group([r for r in training if r['mode']!='fixed'],['task','seed']).items():
        assert len({r['logical_input_sha256'] for r in rr})==1,(task,seed)
    # Restore all selected from-scratch and archived continuation states.
    records=training+read('matched-budget/results/training.json')
    extension=read('results/continuation-extension.json')
    for r in extension:
        for role,s in r['states'].items():records.append({**s,'family':'global','case':role,'seed':r['seed']})
    for d in records:
        path=ROOT/d['checkpoint'];assert hashlib.sha256(path.read_bytes()).hexdigest()==d['checkpoint_sha256']
        snap=dict(np.load(path,allow_pickle=False));recipe=Recipe(**json.loads(str(snap['recipe'])))
        p=BudgetPolicy(recipe,d['family'],[(1,64)],64*2**20,1)
        with Runner(p) as n:n.restore(snap);assert n.digest()==d['state_sha256']
        audit['checkpoints'].append({'checkpoint':d['checkpoint'],'sha256':d['checkpoint_sha256'],'state_sha256':d['state_sha256']})
    assert len(audit['checkpoints'])==42
    summaries=[]
    for (case,),rr in group(training,['case']).items():
        threshold=[]
        for r in rr:
            assert r['policy']['resident_bound_bytes']<=r['budget_mib']*2**20
            assert all(e['microbatch']==e['effective_batch'] for e in r['events'])
            step=next((h['step'] for h in r['history'] if h['validation']['accuracy']>=.99),None)
            if step:threshold.append({'seed':r['seed'],'step':step,'seconds':sum(s['seconds'] for s in r['segments'][:step//100])})
        summaries.append({'case':case,'task':rr[0]['task'],'family':rr[0]['family'],'parameters':rr[0]['parameters'],
            'training_seconds':stats(r['training_seconds'] for r in rr),'native_seconds':stats(r['native_training_seconds'] for r in rr),
            'resident_mib':stats(r['policy']['resident_bound_bytes']/2**20 for r in rr),
            'physical_positions':rr[0]['physical_positions'],'logical_positions':rr[0]['logical_positions'],
            'selected_steps':[r['selected_step'] for r in rr],'time_to_99_validation':threshold,
            'clip_fraction':stats(sum(e['grad_norm']>1 for e in r['events'])/r['steps'] for r in rr)})
    ev=read('results/evaluation.json');quality=[]
    for (case,scenario,t),rr in group(ev,['case','scenario','context']).items():
        assert len({r['logical_input_sha256'] for r in rr})==1
        quality.append({'case':case,'scenario':scenario,'context':t,'accuracy':stats(r['test']['accuracy'] for r in rr),
            'ce_nats':stats(r['test']['ce_nats'] for r in rr),'sample_count_per_seed':rr[0]['test']['scored_tokens']})
    for key,rr in group(ev,['task','scenario','context']).items():
        assert len({r['logical_input_sha256'] for r in rr})==1,key
    language=[]
    for case in ('lm_bucket_global','lm_bucket_hybrid','lm_bucket_attention'):
        vals=[]
        for seed in SEEDS:
            rr=[r for r in ev if r['case']==case and r['seed']==seed and r['scenario']=='main' and r['context']<=128]
            count=sum(r['test']['scored_tokens'] for r in rr)
            vals.append(sum(r['test']['ce_nats']*r['test']['scored_tokens'] for r in rr)/count)
        language.append({'case':case,'ce_nats':stats(vals),'scored_bytes_per_seed':count})
    cf=[]
    for (case,t),rr in group(read('results/counterfactuals.json'),['case','context']).items():
        cf.append({'case':case,'context':t,**{k:stats(r[k] for r in rr) for k in ('original_accuracy','swapped_accuracy','both_correct_rate','prediction_change_rate')}})
    budget_data=read('results/budget-probe.json');budget=[]
    for (b,kind),rr in group(budget_data['rows'],['budget_mib','kind']).items():
        first=rr[0];assert len(rr)==6 and len({r['state_sha256'] for r in rr})==1
        budget.append({'budget_mib':b,'kind':kind,'variant':first['variant'],'microbatch':first['microbatch'],
            'resident_mib':first['resident_bound_bytes']/2**20,'wall_ms':stats(r['wall_ms_per_update'] for r in rr)})
    for micro,rr in group(budget_data['rows'],['microbatch']).items():
        assert len({r['state_sha256'] for r in rr})==1,('same-micro state mismatch',micro)
    diffs=[]
    for seed in SEEDS:
        def state(case):
            d=next(r for r in training if r['case']==case and r['seed']==seed)
            return d,dict(np.load(ROOT/d['checkpoint'],allow_pickle=False))
        a,pa=state('ordered_padded_global');b,pb=state('ordered_bucket_global');c,pc=state('ordered_bucket_guard')
        diffs.append({'seed':seed,'padded_bucket_weight_relative_l2':float(np.linalg.norm(pa['weights']-pb['weights'])/np.linalg.norm(pa['weights'])),
            'guard_bucket_bitwise_state_equal':b['state_sha256']==c['state_sha256']})
    matched_training=read('matched-budget/results/training.json');matched_quality=[]
    for (case,t),rr in group(read('matched-budget/results/evaluation.json'),['case','context']).items():
        matched_quality.append({'case':case,'context':t,'accuracy':stats(r['test']['accuracy'] for r in rr),'ce_nats':stats(r['test']['ce_nats'] for r in rr)})
    matched={'training_seconds':stats(r['training_seconds'] for r in matched_training),
        'resident_mib':matched_training[0]['policy']['resident_bound_bytes']/2**20,
        'quality':matched_quality,'choices':matched_training[0]['policy']['choices']}
    extended=[]
    for role in ('before','short_control','long_control','long_adaptive'):
        for t in (1024,2048,4096):
            rr=[next(a for a in r['evaluation'][role] if a['context']==t) for r in extension]
            extended.append({'role':role,'context':t,'accuracy':stats(a['test']['accuracy'] for a in rr),'ce_nats':stats(a['test']['ce_nats'] for a in rr)})
    matched_diffs=[]
    for seed in SEEDS:
        old=next(r for r in training if r['case']=='ordered_bucket_attention' and r['seed']==seed)
        new=next(r for r in matched_training if r['seed']==seed)
        a=dict(np.load(ROOT/old['checkpoint'],allow_pickle=False));b=dict(np.load(ROOT/new['checkpoint'],allow_pickle=False))
        matched_diffs.append({'seed':seed,'attention_full_vs_accum_weight_relative_l2':float(np.linalg.norm(a['weights']-b['weights'])/np.linalg.norm(a['weights']))})
    summary={'from_scratch_runs':33,'from_scratch_updates':27000,'continuation_arms':9,'continuation_updates':360,'deterministic_replay_updates':240,
        'training':summaries,'quality':quality,'language':language,'counterfactuals':cf,'budget':budget,
        'calibration_seconds':budget_data['calibration_seconds'],'precision_audit':read('results/precision-audit.json'),
        'execution_state_comparisons':diffs,'matched_attention32':matched,'matched_attention_parameter_differences':matched_diffs,
        'continuation':read('results/continuation.json'),'continuation_extension':extended,'registry_reference':read('results/registry-reference.json'),
        'accumulation_check':read('results/accumulation-check.json'),
        'scope':'Small native-C CPU models; means/ranges across3 seeds or6 timing repetitions. No confidence intervals or PyTorch training-throughput comparison.'}
    (ROOT/'results/summary.json').write_text(json.dumps(summary,indent=2));(ROOT/'results/final-audit.json').write_text(json.dumps(audit,indent=2))
    print('PASS: frozen sources, paired inputs,42 checkpoint hashes/restores, memory bounds and fixed-microbatch state parity')
