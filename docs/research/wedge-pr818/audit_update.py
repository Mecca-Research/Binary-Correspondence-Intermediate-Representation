"""Before/after identities, with source and data provenance, not rounded curves."""
from study import *


if __name__=='__main__':
    freeze();before=ROOT/'baseline/results';after=ROOT/'results'
    rows=[]
    keys=('parameters','recipe','steps','selected_step','selected_validation_ce','state_sha256',
          'final_training_state_sha256','input_sha256','physical_positions','supervised_tokens','history')
    paths=sorted(after.glob('train-*.json'));assert len(paths)==51,len(paths)
    for path in paths:
        a=json.loads((before/path.name).read_text());b=json.loads(path.read_text())
        fields={k:a[k]==b[k] for k in keys}
        fields['all_per_update_loss_gradient_norm_step']=all(
            all(x[k]==y[k] for k in ('loss','grad_norm','step')) for x,y in zip(a['events'],b['events'])) and len(a['events'])==len(b['events'])
        epath=after/path.name.replace('train-','eval-');erow=None
        if epath.exists():
            ea=json.loads((before/epath.name).read_text());eb=json.loads(epath.read_text())
            erow={'test_rows_equal':ea['rows']==eb['rows'],'counterfactuals_equal':ea['paired']==eb['paired'],
                  'test_row_count':len(eb['rows']),'counterfactual_count':len(eb['paired'])}
        rows.append({'case':path.stem.removeprefix('train-'),'checks':fields,'evaluation':erow,
                     'before_state_sha256':a['state_sha256'],'after_state_sha256':b['state_sha256'],
                     'input_sha256':b['input_sha256']})
    out={'training_runs':len(rows),'training_updates':sum(json.loads(p.read_text())['steps'] for p in paths),
        'all_training_checks_pass':all(all(r['checks'].values()) for r in rows),
        'evaluated_runs':sum(r['evaluation'] is not None for r in rows),
        'all_regression_rows_equal':all(r['evaluation'] and r['evaluation']['test_rows_equal'] and r['evaluation']['counterfactuals_equal'] for r in rows),
        'scope':'Exact stored FP32 state/history/loss/gradient-norm and fixed held-out prediction comparison. Wall times, source-derived policy IDs and checkpoint ZIP container hashes intentionally excluded.',
        'rows':rows}
    (after/'before-after-audit.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
