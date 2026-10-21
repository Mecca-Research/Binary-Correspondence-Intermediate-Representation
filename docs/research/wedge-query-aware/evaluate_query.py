"""Fresh, paired held-out and binding-counterfactual tests, after validation selection."""
from study import *
import argparse

BASE=20262100
def check_records(task,t,pairs=4):
    n=64 if task=='ordered' and t==2048 else 128 if t>=1024 else 256
    return eval_records(task,t,n,4,BASE+100*pairs+t,data=DATA,pairs=pairs)

def query_changed(record):
    out={**record,'x':record['x'].copy(),'y':record['y'].copy()};t=record['length']
    for b in range(len(out['x'])):
        old=record['query_pair'][b];new=(old+1)%record['pairs'];position=record['value_positions'][b][new]
        out['x'][b,t-1]=out['x'][b,position-1];out['y'][b,t-1]=out['x'][b,position]
        assert out['y'][b,t-1]!=record['y'][b,t-1]
    return out

def score(n,rs,task):
    metric,pred,label=evaluate_records(n,rs,task)
    if task=='kv':
        sets=[row['x'][b,row['value_positions'][b]] for row in rs for b in range(len(row['x']))]
        metric['prediction_in_candidate_set_rate']=float(np.mean([v in s for v,s in zip(pred,sets)]))
    return metric,pred,label

def run(path):
    d=json.loads(path.read_text());task=d['task'];model=d['model'];seed=d['seed'];name=f'{task}-{model}-{seed}'
    output=ROOT/'results'/f'eval-{name}.json'
    if output.exists():return
    ck=ROOT/d['checkpoint'];assert hashlib.sha256(ck.read_bytes()).hexdigest()==d['checkpoint_sha256']
    state=dict(np.load(ck,allow_pickle=False));r=Recipe(**json.loads(str(state['recipe'])))
    lengths=(32,64,128,512) if task=='lm' else (128,256,512,1024,2048) if task=='ordered' else (64,128,256,512,1024)
    policy=BudgetPolicy(r,family(model),[(4,t) for t in lengths],512*2**20,4,micro_mode='all')
    rows=[];paired=[]
    with Runner(policy) as n:
        n.restore(state);assert n.digest()==d['state_sha256']
        for paircount in (1,2,4,8) if task.startswith('kv') else (0,):
            for t in lengths:
                if task=='lm':rs=eval_records('lm',t,64,4,BASE+t,data=DATA)
                else:rs=check_records('ordered' if task=='ordered' else 'kv',t,paircount or 4)
                metric,pred,label=score(n,rs,'lm' if task=='lm' else 'ordered' if task=='ordered' else 'kv')
                rows.append({'context':t,'pairs':paircount,'test':metric,'input_sha256':input_digest(rs),'predictions':pred,'labels':label})
                print(json.dumps({'case':name,'context':t,'pairs':paircount,'test':metric}),flush=True)
        if task!='lm':
            taskkind='ordered' if task=='ordered' else 'kv'
            for t in (128,512):
                rs=check_records(taskkind,t);original,pp,yy=score(n,rs,taskkind)
                for mode in ('value_swap','query_change') if taskkind=='kv' else ('value_swap',):
                    changed=[swapped(row,taskkind) if mode=='value_swap' else query_changed(row) for row in rs]
                    altered,cp,cy=score(n,changed,taskkind)
                    pp=np.array(pp);yy=np.array(yy);cp=np.array(cp);cy=np.array(cy)
                    paired.append({'context':t,'mode':mode,'original':original,'changed':altered,
                        'both_correct_rate':float(np.mean((pp==yy)&(cp==cy))),
                        'prediction_change_rate':float(np.mean(pp!=cp)),
                        'original_sha256':input_digest(rs),'changed_sha256':input_digest(changed),
                        'original_predictions':pp.tolist(),'changed_predictions':cp.tolist(),
                        'original_labels':yy.tolist(),'changed_labels':cy.tolist()})
        assert n.digest()==d['state_sha256']
    output.write_text(json.dumps({'task':task,'model':model,'seed':seed,'checkpoint_sha256':d['checkpoint_sha256'],
        'state_sha256':d['state_sha256'],'selected_step':d['selected_step'],'evaluation_policy':policy.to_dict(),
        'rows':rows,'paired':paired},indent=2))

if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--case');args=pa.parse_args()
    freeze();DATA,_=bytes_data();sources=('evaluate_query.py',)
    ep={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'study_protocol_sha256':hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),
        'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},'seed_base':BASE,
        'samples':'association256/length/pairs/seed exceptT1024=128; ordered256 exceptT1024=128,T2048=64; LM64 windows/length/seed',
        'inference_budget':'common512MiB controlled-buffer envelope, independent of32MiB training; no model/optimizer updates or test-based selection'}
    epath=ROOT/'evaluation-protocol.json'
    if not epath.exists():epath.write_text(json.dumps(ep,indent=2))
    else:
        old=json.loads(epath.read_text())
        for name,h in old['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
    paths=sorted((ROOT/'results').glob('train-*.json'));assert len(paths)==len(cases())*len(SEEDS),len(paths)
    for path in paths:
        if not args.case or args.case in path.name:run(path)
