"""Verify retained source/experiment/checkpoint evidence before packaging."""
from experiment import *
rows=[];changes=[]
frozen=json.loads((ROOT/'frozen-implementation.json').read_text())
for name in ('build_native.py','experiment.py','run_experiments.py','global_core.h','protocol.json',
             'build/global.c','build/global_symmetric.c','build/projector.c'):
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==frozen[name],name
for name,h in frozen.items():
    if (ROOT/name).exists() and hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:
        changes.append({'path':name,'reason':'Measurement harness expanded to six recipes, or baseline ingestion made self-contained for archive replay, after learning protocol/model freeze; no training implementation changed.'})
assert set(r['path'] for r in changes)=={'benchmark.py','evaluate_baselines.py'},changes
for path in sorted((ROOT/'results').glob('*-202610*.json')):
    d=json.loads(path.read_text());ck=ROOT/d['checkpoint']
    assert hashlib.sha256(ck.read_bytes()).hexdigest()==d['checkpoint_sha256']
    state=dict(np.load(ck,allow_pickle=False));r=Recipe(**json.loads(str(state['recipe'])))
    for name in ('weights','moment1','moment2'):assert np.isfinite(state[name]).all()
    with Native(r,threads=1) as n:
        n.restore(state);assert n.digest()==d['state_sha256']
    assert all(math.isfinite(e['loss']) and math.isfinite(e['grad_norm']) for e in d['events'])
    rows.append({'result':path.name,'checkpoint':d['checkpoint'],'checkpoint_sha256':d['checkpoint_sha256'],
                 'restored_state_sha256':d['state_sha256'],'parameters':d['parameters']})
assert len(rows)==36
ab=[]
if (ROOT/'position-ablation/recall-length.json').exists():
    ap=json.loads((ROOT/'position-ablation/protocol.json').read_text())
    assert hashlib.sha256((ROOT/'position_ablation.py').read_bytes()).hexdigest()==ap['source_sha256']
    for path in sorted((ROOT/'position-ablation').glob('*_norope-202610*.json')):
        d=json.loads(path.read_text());ck=ROOT/d['checkpoint']
        assert hashlib.sha256(ck.read_bytes()).hexdigest()==d['checkpoint_sha256']
        s=dict(np.load(ck,allow_pickle=False));r=Recipe(**json.loads(str(s['recipe'])))
        with Native(r,threads=1,library_name=d['library_name']) as n:
            n.restore(s);assert n.digest()==d['state_sha256']
        ab.append({'checkpoint':d['checkpoint'],'checkpoint_sha256':d['checkpoint_sha256'],'state_sha256':d['state_sha256']})
    assert len(ab)==6
    for k,sha in ap['native_source_sha256'].items():assert hashlib.sha256((ROOT/'build'/f'{k}_norope.c').read_bytes()).hexdigest()==sha
out={'checkpoints':rows,'position_ablation_checkpoints':ab,'training_sources_unchanged_since_freeze':True,'post_freeze_harness_changes':changes}
(ROOT/'results/checkpoint-verification.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'checkpoints':len(rows),'position_ablation_checkpoints':len(ab),'training_sources_unchanged_since_freeze':True,'post_freeze_harness_changes':changes}))
