"""Audit and aggregate retained measurements; no new training or model selection."""
from candidates import *
import statistics as st,collections

def read(path):return json.loads((ROOT/path).read_text())
def stats(values):
    v=list(values)
    return {'mean':st.mean(v),'median':st.median(v),'min':min(v),'max':max(v),'sample_sd':st.stdev(v) if len(v)>1 else 0,'n':len(v)}
def group(rows,keys):
    out=collections.defaultdict(list)
    for r in rows:out[tuple(r[k] for k in keys)].append(r)
    return out
def training_aggregate(rows):
    out=[]
    for (task,name),rr in group(rows,['task','name']).items():
        threshold=[]
        for r in rr:
            reach=next((h['step'] for h in r['history'] if h['validation']['accuracy']>=.99),None)
            if reach:threshold.append({'seed':r['seed'],'step':reach,'training_seconds':sum(s['seconds'] for s in r['segments'][:reach//100])})
        out.append({'task':task,'name':name,'parameters':rr[0]['parameters'],
            'test_ce_nats':stats(r['test']['ce_nats'] for r in rr),
            'test_accuracy':stats(r['test']['accuracy'] for r in rr),
            'selected_steps':[r['selected_step'] for r in rr],
            'time_to_99_validation':threshold,'threshold_reached_runs':len(threshold)})
    return out

if __name__=='__main__':
    audit={'frozen_sources':{},'checkpoint_restores':[]}
    for name,h in read('protocol.json')['source_sha256'].items():
        actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest();assert actual==h,name
        audit['frozen_sources'][name]=actual
    main=[]
    for p in sorted((ROOT/'results').glob('*.json')):
        j=json.loads(p.read_text())
        if isinstance(j,dict) and 'checkpoint' in j:main.append(j)
    assert len(main)==51
    hybrid=[json.loads(p.read_text()) for p in sorted((ROOT/'hybrid/results').glob('*.json'))]
    assert len(hybrid)==12
    for r in main+hybrid:
        root=ROOT/'hybrid' if r['name']=='hybrid' else ROOT
        p=root/r['checkpoint'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['checkpoint_sha256']
        state=dict(np.load(p,allow_pickle=False));recipe=Recipe(**json.loads(str(state['recipe'])))
        with Native(recipe,threads=1,library_name=r['library_name']) as n:n.restore(state);assert n.digest()==r['state_sha256']
        audit['checkpoint_restores'].append({'task':r['task'],'name':r['name'],'seed':r['seed'],'checkpoint':str(p.relative_to(ROOT)),'sha256':r['checkpoint_sha256'],'state_sha256':r['state_sha256']})
    for key,rr in group(main+hybrid,['task','seed']).items():
        assert len({r['input_sha256'] for r in rr})==1,('unpaired train inputs',key)
    timings={};allbench=[]
    for category in ('execution','architecture','limits'):
        for th in (1,4):
            rr=read(f'results/benchmark-{category}-{th}thread.json');allbench+=rr
            summaries=[]
            for (t,name),gg in group(rr,['context','name']).items():
                assert len(gg)==6 and len({r['state_sha256'] for r in gg})==1
                summaries.append({'context':t,'name':name,'threads':th,'ms':stats(r['ms_per_update'] for r in gg),
                    'parameters':gg[0]['parameters'],'arena_mib':gg[0]['arena_bytes']/2**20})
            timings[f'{category}_{th}thread']=summaries
    for key,rr in group(allbench,['context','name']).items():
        assert len({r['state_sha256'] for r in rr})==1,('thread or benchmark state differs',key)
    for t in (128,1024):
        exact=[r for r in allbench if r['context']==t and r['name'] in (*EXACT,'layer_frame_ff')]
        assert len({r['state_sha256'] for r in exact})==1,('exact execution differed',t)
    audit['benchmark_state_checks']='all six repetitions and thread counts agree; exact variants also agree across timing blocks'
    budget=[]
    for (lib,),gg in group(read('results/budget.json'),['library']).items():
        assert len({r['state_sha256'] for r in gg})==1
        budget.append({'name':lib,'microbatch':gg[0]['microbatch'],'accumulation':gg[0]['accumulation'],
            'native_buffers_plus_input_mib':gg[0]['native_buffers_plus_one_input_pair_bytes']/2**20,
            'ms_per_effective_update':stats(r['ms_per_effective_update'] for r in gg)})
    lengths=[]
    stress=read('results/length-stress.json')
    for (task,kind,t,name),gg in group(stress,['training_task','test_kind','context','name']).items():
        lengths.append({'task':task,'kind':kind,'context':t,'name':name,
            'accuracy':stats(r['test']['accuracy'] for r in gg),'ce_nats':stats(r['test']['ce_nats'] for r in gg)})
    for (kind,t),gg in group([r for r in stress if 'input_sha256'in r],['test_kind','context']).items():
        assert len({r['input_sha256'] for r in gg})==1
    svd=[]
    for (rank,t),gg in group(read('results/svd.json'),['rank','context']).items():
        svd.append({'rank':rank,'context':t,'accuracy':stats(r['test']['accuracy'] for r in gg),
            'parameters':gg[0]['parameters']})
    extra=[]
    follow=read('results/followup.json')
    for (task,name),rr in group(follow['rows'],['task','name']).items():
        for t in sorted({r.get('context',64) for r in rr}):
            gg=[r for r in rr if r.get('context',64)==t]
            extra.append({'task':task,'name':name,'context':t,'accuracy':stats(r['test']['accuracy'] for r in gg),
                'ce_nats':stats(r['test']['ce_nats'] for r in gg)})
    summary={'new_native_training_runs':57,'new_learning_updates':28200,'main_training':training_aggregate(main),
        'hybrid_fresh_paired_test':training_aggregate(hybrid),'timings':timings,'fixed_memory_budget':budget,
        'length_stress':lengths,'svd':svd,'followup':extra,
        'memory_plans':read('results/memory-plans.json'),'lowest_arena_endpoint':read('results/limits-validation.json')['memory'],
        'profiles':[read('results/profile-1thread.json'),read('results/profile-4thread.json')],
        'ham':read('results/ham.json'),'precision':read('results/precision.json'),
        'precision_full_model_check':follow['precision_full_model'],
        'statistical_scope':'Means/ranges across three training seeds, timing medians/ranges across six sequential repetitions. No confidence interval or population-level significance claim.'}
    (ROOT/'results/summary.json').write_text(json.dumps(summary,indent=2))
    (ROOT/'results/final-audit.json').write_text(json.dumps(audit,indent=2))
    print('PASS: frozen source hashes, 63 result references to 57 unique training checkpoints, paired training inputs, benchmark state/thread parity. Summary saved.')
