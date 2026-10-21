"""Audit-friendly aggregation: per-seed values and means/ranges, no pooled-seed CI."""
from study import *
import collections

def aggregate(rows,keys):
    result={}
    for k in keys:
        a=[float(row[k]) for row in rows if k in row]
        if a:result[k]={'mean':float(np.mean(a)),'min':min(a),'max':max(a),'values':a}
    return result

if __name__=='__main__':
    train=[json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('train-*.json'))]
    ev=[json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('eval-*.json'))]
    assert len(train)==len(ev)==51,(len(train),len(ev))
    groups=collections.defaultdict(list);paired=collections.defaultdict(list);primary=collections.defaultdict(list);language=collections.defaultdict(list)
    for d in ev:
        task,model,seed=d['task'],d['model'],d['seed']
        for row in d['rows']:
            groups[(task,model,row['pairs'],row['context'])].append({'seed':seed,**row['test']})
        for row in d['paired']:paired[(task,model,row['context'],row['mode'])].append({'seed':seed,'both_correct_rate':row['both_correct_rate'],'prediction_change_rate':row['prediction_change_rate'],
            'original_accuracy':row['original']['accuracy'],'changed_accuracy':row['changed']['accuracy']})
        if task.startswith('kv'):
            rs=[row for row in d['rows'] if row['pairs']==(1 if task=='kv_single' else 4) and row['context'] in (128,256)]
            count=sum(row['test']['scored_tokens'] for row in rs)
            primary[(task,model)].append({'seed':seed,'scored_tokens':count,**{k:sum(row['test'][k]*row['test']['scored_tokens'] for row in rs)/count
                for k in ('accuracy','ce_nats','prediction_in_candidate_set_rate')}})
        if task=='lm':
            rs=[row for row in d['rows'] if row['context'] in (32,64,128)];count=sum(row['test']['scored_tokens'] for row in rs)
            language[model].append({'seed':seed,'scored_tokens':count,**{k:sum(row['test'][k]*row['test']['scored_tokens'] for row in rs)/count for k in ('accuracy','ce_nats')}})
    result={'scope':'Three-seed means/ranges. Shared paired test samples across models/seeds; seeds are independently trained. No independence/CI claim for pooled repeated examples.',
      'training':{},'ordered_time_to_quality':{},'primary_association':{},'primary_differences':{},'language':{},'curves':[],'counterfactuals':[]}
    for task,model in cases():
        ds=[d for d in train if d['task']==task and d['model']==model]
        rr=[{'training_seconds':d['training_seconds'],'native_training_seconds':d['native_training_seconds'],
            'buffers_mib':d['policy']['resident_bound_bytes']/2**20,'parameters':d['parameters'],'selected_step':d['selected_step'],
            'ms_per_update':1000*d['training_seconds']/d['steps']} for d in ds]
        result['training'][f'{task}/{model}']={'seeds':[d['seed'] for d in ds],**aggregate(rr,rr[0].keys()),
            'physical_positions_per_run':ds[0]['physical_positions'],'supervised_tokens_per_run':ds[0]['supervised_tokens']}
        if task=='ordered':
            crosses=[]
            for d in ds:
                hit=next(((i,h) for i,h in enumerate(d['history']) if h['validation']['accuracy']>=.99),None)
                if hit:crosses.append({'seed':d['seed'],'first_checked_step':hit[1]['step'],'training_seconds':sum(seg['seconds'] for seg in d['segments'][:hit[0]+1])})
            result['ordered_time_to_quality'][model]={'threshold':'first observed aggregate validation accuracy >=99%, checked every100 updates','per_seed':crosses,
                **aggregate(crosses,('first_checked_step','training_seconds'))}
    for (task,model),rs in primary.items():
        result['primary_association'][f'{task}/{model}']={'seeds':[r['seed'] for r in rs],**aggregate(rs,('accuracy','ce_nats','prediction_in_candidate_set_rate'))}
    for task in ('kv_single','kv_curriculum'):
        baseline={r['seed']:r for r in primary[(task,'global')]}
        for model in ('query_full','query_matched','attention'):
            rr=[{'seed':r['seed'],'accuracy_delta':r['accuracy']-baseline[r['seed']]['accuracy'],
                 'ce_delta':r['ce_nats']-baseline[r['seed']]['ce_nats']} for r in primary[(task,model)]]
            result['primary_differences'][f'{task}/{model}-global']={'seeds':[r['seed'] for r in rr],**aggregate(rr,('accuracy_delta','ce_delta'))}
    for model,rs in language.items():result['language'][model]={'seeds':[r['seed'] for r in rs],**aggregate(rs,('accuracy','ce_nats'))}
    for (task,model,pairs,t),rs in sorted(groups.items()):
        result['curves'].append({'task':task,'model':model,'pairs':pairs,'context':t,'seeds':[r['seed'] for r in rs],
            **aggregate(rs,('accuracy','ce_nats','prediction_in_candidate_set_rate','scored_tokens'))})
    for (task,model,t,mode),rs in sorted(paired.items()):
        result['counterfactuals'].append({'task':task,'model':model,'context':t,'mode':mode,'seeds':[r['seed'] for r in rs],
            **aggregate(rs,('both_correct_rate','prediction_change_rate','original_accuracy','changed_accuracy'))})
    result['count']={'training_runs':len(train),'selected_checkpoints':len(train),'quality_training_updates':sum(d['steps'] for d in train),
        'evaluation_rows':sum(len(d['rows']) for d in ev),'counterfactual_pairs_of_batches':sum(len(d['paired']) for d in ev),
        'supervised_training_tokens':sum(d['supervised_tokens'] for d in train),'physical_training_positions':sum(d['physical_positions'] for d in train)}
    (ROOT/'results/summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ('count','primary_association','primary_differences','language')},indent=2))
