"""Held-out evaluation after all checkpoints have been selected on validation."""
from train_integrated import *
import collections

def recall_policy(r,family,lengths):
    # A common evaluation storage envelope, separate from the training budgets.
    return BudgetPolicy(r,family,[(4,t) for t in lengths],512*2**20,4,micro_mode='all')

if __name__=='__main__':
    data,_=bytes_data();rows=[];paired=[];restores=[];audits=[]
    paths=sorted((ROOT/'results').glob('train-*.json'));assert len(paths)==30,len(paths)
    for path in paths:
        d=json.loads(path.read_text());ck=ROOT/d['checkpoint'];assert hashlib.sha256(ck.read_bytes()).hexdigest()==d['checkpoint_sha256']
        state=dict(np.load(ck,allow_pickle=False));r=Recipe(**json.loads(str(state['recipe'])));family=d['family'];task=d['task']
        lengths=(32,64,128,256,512) if task=='lm' else (64,128,256,512,1024,2048)
        p=recall_policy(r,family,lengths)
        with Runner(p) as n:
            n.restore(state);assert n.digest()==d['state_sha256']
            restores.append({'case':d['case'],'seed':d['seed'],'checkpoint_sha256':d['checkpoint_sha256'],'state_sha256':n.digest()})
            for t in lengths:
                count=32 if t==2048 else 64 if task=='lm' else 128
                rec=eval_records(task,t,count,4,20261110+t,data=data,pairs=4)
                metric,pred,label=evaluate_records(n,rec,task);assert n.digest()==d['state_sha256']
                row={'case':d['case'],'seed':d['seed'],'family':family,'task':task,'scenario':'main','context':t,
                    'test':metric,'logical_input_sha256':input_digest(rec),'checkpoint_sha256':d['checkpoint_sha256']}
                rows.append(row);print(json.dumps(row),flush=True)
            if task!='lm':
                for t in (128,512):
                    rec=eval_records(task,t,128,4,20261210+t,pairs=4);swaps=[swapped(a,task) for a in rec]
                    original,pp,yy=evaluate_records(n,rec,task);counter,cp,cy=evaluate_records(n,swaps,task)
                    pp=np.array(pp);yy=np.array(yy);cp=np.array(cp);cy=np.array(cy)
                    row={'case':d['case'],'seed':d['seed'],'task':task,'context':t,
                        'original_accuracy':original['accuracy'],'swapped_accuracy':counter['accuracy'],
                        'both_correct_rate':float(np.mean((pp==yy)&(cp==cy))),
                        'prediction_change_rate':float(np.mean(pp!=cp)),
                        'original_input_sha256':input_digest(rec),'swapped_input_sha256':input_digest(swaps)}
                    paired.append(row);print(json.dumps({'counterfactual':row}),flush=True)
                if task=='kv':
                    for num in (1,2,8):
                        rec=eval_records(task,256,128,4,20261310+num,pairs=num)
                        metric,_,_=evaluate_records(n,rec,task)
                        rows.append({'case':d['case'],'seed':d['seed'],'family':family,'task':task,
                            'scenario':f'{num}_pairs','pairs':num,'context':256,'test':metric,'logical_input_sha256':input_digest(rec)})
                if d['case'] in ('ordered_bucket_global','ordered_bucket_guard'):
                    rec=eval_records('ordered',1024,128,4,20261410,repeat_probability=.9)
                    metric,_,_=evaluate_records(n,rec,task)
                    rows.append({'case':d['case'],'seed':d['seed'],'family':family,'task':task,'scenario':'repeat_heavy',
                        'context':1024,'test':metric,'logical_input_sha256':input_digest(rec),
                        'scope':'90% target-token repetition makes this a numerical stress distribution, not a harder retrieval task.'})
        if d['case']=='ordered_bucket_guard':
            for scenario,prob,t in (('ordinary',0.,128),('ordinary',0.,1024),('repeat_heavy',.9,1024)):
                rec=eval_records('ordered',t,32,4,20261510+t,repeat_probability=prob)
                counts=np.zeros(9,dtype=np.uint64)
                with Native(replace(r,batch=4,context=t),threads=4,library_name='guard_audit') as audit:
                    audit.restore(state);audit.lib.wedge_audit_get.restype=ct.c_ulonglong;audit.lib.wedge_audit_reset()
                    for rr in rec:audit.forward(trim(rr)[0])
                    counts=np.array([int(audit.lib.wedge_audit_get(i)) for i in range(9)],np.uint64)
                    assert sum(counts[3:])==counts[0] and counts[2]<=counts[1]<=counts[0]
                    assert audit.digest()==d['state_sha256']
                audits.append({'case':d['case'],'seed':d['seed'],'scenario':scenario,'context':t,'pair_checks':int(counts[0]),
                    'guard_selected':int(counts[1]),'absent_planes':int(counts[2]),
                    'guard_fraction':float(counts[1]/counts[0]),'absent_fraction':float(counts[2]/counts[0]),
                    'gram_ratio_histogram':counts[3:].astype(int).tolist(),
                    'bins':['<=1e-8','(1e-8,1e-6]','(1e-6,1e-4]','(1e-4,1e-2]','(1e-2,0.5]','>0.5'],
                    'scope':'Audit-only atomic counters, excluded from training and timed variants.'})
    (ROOT/'results/evaluation.json').write_text(json.dumps(rows,indent=2))
    (ROOT/'results/counterfactuals.json').write_text(json.dumps(paired,indent=2))
    (ROOT/'results/precision-audit.json').write_text(json.dumps(audits,indent=2))
    (ROOT/'results/checkpoint-audit.json').write_text(json.dumps(restores,indent=2))
    print('held-out evaluation complete',len(rows),'rows',len(paired),'paired tests')
