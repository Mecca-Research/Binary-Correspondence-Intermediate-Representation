"""Fixed, focused quality regression for the separately declared BF16 family."""
from study_bf16 import *
import evaluate_query as eq


def evaluate(path,data):
    d=json.loads(path.read_text());task=d['task'];model=d['model'];seed=d['seed']
    out=OUT/'results'/path.name.replace('train-','eval-')
    if out.exists():return
    ck=ROOT/d['checkpoint'];assert hashlib.sha256(ck.read_bytes()).hexdigest()==d['checkpoint_sha256']
    state=dict(np.load(ck,allow_pickle=False));r=Recipe(**json.loads(str(state['recipe'])))
    lengths=(32,64,128) if task=='lm' else (1024,2048) if task=='ordered' else (128,256)
    policy=BudgetPolicy(r,family(model),[(4,t) for t in lengths],512*2**20,4,micro_mode='all')
    rows=[];paired=[]
    with Runner(policy) as n:
        n.restore(state);assert n.digest()==d['state_sha256']
        for t in lengths:
            rs=eval_records('lm',t,64,4,eq.BASE+t,data=data) if task=='lm' else eq.check_records('ordered' if task=='ordered' else 'kv',t,4)
            metric,pred,label=eq.score(n,rs,'lm' if task=='lm' else 'ordered' if task=='ordered' else 'kv')
            rows.append({'context':t,'pairs':4 if task.startswith('kv') else 0,'test':metric,'input_sha256':input_digest(rs),'predictions':pred,'labels':label})
            print(json.dumps({'case':path.stem,'context':t,'test':metric}),flush=True)
        if task.startswith('kv'):
            rs=eq.check_records('kv',128,4);original,pp,yy=eq.score(n,rs,'kv');pp=np.asarray(pp);yy=np.asarray(yy)
            for mode in ('value_swap','query_change'):
                changed=[swapped(row,'kv') if mode=='value_swap' else eq.query_changed(row) for row in rs]
                altered,cp,cy=eq.score(n,changed,'kv');cp=np.asarray(cp);cy=np.asarray(cy)
                paired.append({'context':128,'mode':mode,'original':original,'changed':altered,
                    'both_correct_rate':float(np.mean((pp==yy)&(cp==cy))),
                    'prediction_change_rate':float(np.mean(pp!=cp)),
                    'original_sha256':input_digest(rs),'changed_sha256':input_digest(changed),
                    'original_predictions':pp.tolist(),'changed_predictions':cp.tolist(),
                    'original_labels':yy.tolist(),'changed_labels':cy.tolist()})
        assert n.digest()==d['state_sha256']
    out.write_text(json.dumps({'task':task,'model':model,'seed':seed,'checkpoint_sha256':d['checkpoint_sha256'],
        'state_sha256':d['state_sha256'],'selected_step':d['selected_step'],'evaluation_policy':policy.to_dict(),
        'rows':rows,'paired':paired},indent=2))


if __name__=='__main__':
    bf_freeze();data,_=bytes_data();eq.DATA=data
    ep={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('evaluate_bf16.py','evaluate_query.py')},
        'BF16_protocol_sha256':hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
        'samples':'Fixed previous FP32 regression samples, not a fresh unseen test; no test-based selection.'}
    p=OUT/'evaluation-protocol.json'
    if p.exists():assert json.loads(p.read_text())['source_sha256']==ep['source_sha256']
    else:p.write_text(json.dumps(ep,indent=2))
    paths=sorted((OUT/'results').glob('train-*.json'));assert len(paths)==15,len(paths)
    for p in paths:evaluate(p,data)
