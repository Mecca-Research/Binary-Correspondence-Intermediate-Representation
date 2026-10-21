"""Post-training length stress test; no fitting or checkpoint reselection."""
from run_experiments import *
from dataclasses import replace
rows=[]
for context in (64,128,256,512,1024):
    test=recall_batches(16,8,context,881020+context) # 128 fresh examples, same across recipes/seeds
    for kind in ('transformer','global','global_symmetric','projector'):
        for seed in SEEDS:
            d=json.loads((ROOT/'results'/f'first_token_recall-{kind}-{seed}.json').read_text())
            ck=ROOT/d['checkpoint'];assert hashlib.sha256(ck.read_bytes()).hexdigest()==d['checkpoint_sha256']
            s=dict(np.load(ck,allow_pickle=False));r=replace(Recipe(**json.loads(str(s['recipe']))),context=context,batch=8)
            with Native(r,threads=4) as m:
                m.restore(s);assert m.digest()==d['state_sha256'];result=evaluate(m,test)
                assert m.digest()==d['state_sha256']
            row={'kind':kind,'seed':seed,'context':context,'training_context':128,'test':result,
                'input_sha256':hashlib.sha256(test[0].tobytes()+test[1].tobytes()).hexdigest(),
                'protocol':'16 batches of 8, post-training length stress test; no training or model selection'}
            rows.append(row);print(json.dumps(row),flush=True)
(ROOT/'results/recall-length.json').write_text(json.dumps(rows,indent=2))
