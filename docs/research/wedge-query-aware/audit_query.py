"""Final source, checkpoint, pairing and native-plan audit without new training."""
from study import *

if __name__=='__main__':
    protocol=freeze();ep=json.loads((ROOT/'evaluation-protocol.json').read_text())
    for name,h in ep['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    train=[json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('train-*.json'))]
    ev=[json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('eval-*.json'))]
    assert len(train)==len(ev)==len(cases())*len(SEEDS)==51
    inputsets={};testsets={};restores=[]
    for d in train:
        k=(d['task'],d['seed']);inputsets.setdefault(k,set()).add(d['input_sha256'])
        assert d['policy']['resident_bound_bytes']<=32*2**20
        for c in d['policy']['choices']:
            assert c['resident_bound_bytes']<=d['policy']['resident_bound_bytes']
            assert c['parameters']==d['parameters']
            assert c['microbatch']*c['accumulation']==c['effective_batch']
            assert hashlib.sha256((ROOT/'build'/f'{c["variant"]}.c').read_bytes()).hexdigest()==c['source_sha256']
        ck=ROOT/d['checkpoint'];assert hashlib.sha256(ck.read_bytes()).hexdigest()==d['checkpoint_sha256']
        state=dict(np.load(ck,allow_pickle=False));r=Recipe(**json.loads(str(state['recipe'])))
        assert asdict(r)==d['recipe'] and sum(n for _,_,_,n in layout(r))==d['parameters']
        p=BudgetPolicy(r,family(d['model']),[(4,16)],32*2**20,4,micro_mode='full')
        with Runner(p) as n:
            n.restore(state);assert n.digest()==d['state_sha256'];assert n.state.step==d['selected_step']
            assert all(np.isfinite(n.buffers[k]).all() for k in ('weights','moment1','moment2'))
            restores.append({'task':d['task'],'model':d['model'],'seed':d['seed'],'step':int(n.state.step),
                'checkpoint_sha256':d['checkpoint_sha256'],'state_sha256':n.digest()})
    assert all(len(v)==1 for v in inputsets.values())
    for d in ev:
        source=next(x for x in train if (x['task'],x['model'],x['seed'])==(d['task'],d['model'],d['seed']))
        assert d['checkpoint_sha256']==source['checkpoint_sha256'] and d['state_sha256']==source['state_sha256']
        for row in d['rows']:
            k=(d['task'],row['pairs'],row['context']);testsets.setdefault(k,set()).add(row['input_sha256'])
            assert len(row['predictions'])==len(row['labels'])==row['test']['scored_tokens']
            assert np.isclose(np.mean(np.array(row['predictions'])==row['labels']),row['test']['accuracy'])
            assert np.isfinite(row['test']['ce_nats'])
        for row in d['paired']:
            p=np.array(row['original_predictions']);c=np.array(row['changed_predictions']);y=np.array(row['original_labels']);cy=np.array(row['changed_labels'])
            assert np.all(y!=cy)
            assert np.isclose(np.mean((p==y)&(c==cy)),row['both_correct_rate'])
            assert np.isclose(np.mean(p!=c),row['prediction_change_rate'])
    assert all(len(v)==1 for v in testsets.values())
    validation=json.loads((ROOT/'results/validation.json').read_text());probes=json.loads((ROOT/'results/precision-budget-probe.json').read_text())
    out={'verdict':'pass','source_hashes_verified':len(protocol['source_sha256'])+len(ep['source_sha256']),
        'checkpoint_restores':restores,'paired_training_input_groups':len(inputsets),'paired_test_input_groups':len(testsets),
        'training_buffer_bound':'all51 runs and admitted shapes <=32MiB','evaluation_state':'unchanged digests recorded for all51 checkpoints',
        'max_independent_model_gradient_relative':max(r['gradient_relative'] for r in validation['models']),
        'guarded_geometry_max_gradient_relative':max(r['gradient_relative'] for r in probes['guarded_geometry']),
        'qualifications':'GCC13 focused research gates only. Full repository matrix, LLVM/Clang/MLIR23, Python3.15, ARM/Windows and accelerators unqualified.'}
    (ROOT/'results/final-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='checkpoint_restores'},indent=2))
