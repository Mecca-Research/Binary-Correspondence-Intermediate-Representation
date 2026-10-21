from candidates import *
from run_experiments import recall_batches,evaluate
from train_next import ordered_recall
rows=[];restores=[]
for path in sorted((ROOT/'results').glob('*.json')):
    d=json.loads(path.read_text())
    if not isinstance(d,dict) or 'checkpoint' not in d:continue
    ck=ROOT/d['checkpoint'];assert hashlib.sha256(ck.read_bytes()).hexdigest()==d['checkpoint_sha256']
    state=dict(np.load(ck,allow_pickle=False));r0=Recipe(**json.loads(str(state['recipe'])));lib=d['library_name']
    with Native(r0,threads=1,library_name=lib) as n:n.restore(state);assert n.digest()==d['state_sha256']
    restores.append({'checkpoint':d['checkpoint'],'checkpoint_sha256':d['checkpoint_sha256'],'state_sha256':d['state_sha256']})
    if d['task']=='byte_lm':continue
    for t in (64,128,256,512,1024):
        rr=replace(r0,batch=8,context=t)
        task='ordered' if d['task']=='recall_ordered' else 'disjoint_vocab'
        make=ordered_recall if task=='ordered' else recall_batches
        xy=make(16,8,t,931029+t)
        with Native(rr,threads=4,library_name=lib) as n:
            n.restore(state);metric=evaluate(n,xy);assert n.digest()==d['state_sha256']
        row={'training_task':d['task'],'name':d['name'],'seed':d['seed'],'context':t,'test_kind':task,
            'test':metric,'input_sha256':hashlib.sha256(xy[0].tobytes()+xy[1].tobytes()).hexdigest()}
        rows.append(row);print(json.dumps(row),flush=True)
    if d['task']=='recall_fixed':
        rr=replace(r0,batch=8,context=128);xy=ordered_recall(16,8,128,1951029)
        with Native(rr,threads=4,library_name=lib) as n:n.restore(state);metric=evaluate(n,xy)
        rows.append({'training_task':d['task'],'name':d['name'],'seed':d['seed'],'context':128,'test_kind':'ordered_OOD','test':metric})
assert len(restores)==51,len(restores)
(ROOT/'results/length-stress.json').write_text(json.dumps(rows,indent=2))
(ROOT/'results/checkpoint-verification.json').write_text(json.dumps(restores,indent=2))
