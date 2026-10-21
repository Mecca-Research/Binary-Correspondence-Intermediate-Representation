"""Reevaluate the pilot's fixed, validation-selected checkpoints on new test samples."""
from run_experiments import *
import shutil
data,_=bytes_data();old=ROOT.parent/'gp-experiment'
(ROOT/'baseline-checkpoints').mkdir(exist_ok=True)
for task in ('wikitext2_byte','first_token_recall'):
    for seed in SEEDS:
        for kind in ('transformer','grassmann','symmetric'):
            src=(old/'results' if old.exists() else ROOT/'pilot-results')/f'{task}-{kind}-{seed}.json'
            d=json.loads(src.read_text())
            ck=old/d['checkpoint'] if old.exists() else ROOT/'baseline-checkpoints'/Path(d['checkpoint']).name
            assert hashlib.sha256(ck.read_bytes()).hexdigest()==d['checkpoint_sha256']
            s=dict(np.load(ck,allow_pickle=False));r=Recipe(**json.loads(str(s['recipe'])))
            test=lm_batches(data['test'],64,r.batch,r.context,20261020) if task=='wikitext2_byte' else recall_batches(256,r.batch,r.context,20261021)
            with Native(r,seed=seed,threads=4) as m:
                m.restore(s);assert m.digest()==d['state_sha256']
                result=evaluate(m,test)
            target=ROOT/'baseline-checkpoints'/ck.name
            if ck.resolve()!=target.resolve():shutil.copy2(ck,target)
            d['pilot_test']=d['test'];d['test']=result
            d['baseline_reused_training']=True;d['checkpoint']=str(target.relative_to(ROOT))
            d['source_result_sha256']=hashlib.sha256(src.read_bytes()).hexdigest()
            (ROOT/'results'/f'{task}-{kind}-{seed}.json').write_text(json.dumps(d,indent=2))
            print(json.dumps({'task':task,'kind':kind,'seed':seed,'test':result}),flush=True)
