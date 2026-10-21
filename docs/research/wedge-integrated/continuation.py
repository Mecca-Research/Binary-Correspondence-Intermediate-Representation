"""Short, separately declared continuation through changing memory budgets.

Compare 40 identical extra updates from each selected global checkpoint:
ample fixed storage/full microbatch versus calibrated shrinking storage.
No continuation checkpoint is selected on test results.
"""
from train_integrated import *

if __name__=='__main__':
    protocol={'stage':'40-update memory-transition continuation after main training',
        'case':'ordered_bucket_global','seeds':list(SEEDS),'updates':40,'effective_batch':4,'context':1024,
        'lr':.0001,'budgets_mib':[160,64,32,20],'updates_per_budget':10,
        'control':'160MiB with full microbatch4 and cached fast plan',
        'adaptive':'calibrated feasible recomputation/microbatch choice, reconfigure only at update boundaries',
        'data_seed_offset':700,'fresh_test_seed':20261693,'test_examples':128,
        'selection':'no best-test checkpoint selection; evaluate before and after all40 updates',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT/'continuation-protocol.json').write_text(json.dumps(protocol,indent=2))
    lookup=json.loads((ROOT/'results/calibration.json').read_text())['lookup_ms'];rows=[]
    for seed in SEEDS:
        d=json.loads((ROOT/'results'/f'train-ordered_bucket_global-{seed}.json').read_text());state=dict(np.load(ROOT/d['checkpoint'],allow_pickle=False))
        r=Recipe(**json.loads(str(state['recipe'])))
        records=ordered_records(40,4,seed+700,fixed=1024);test=eval_records('ordered',1024,128,4,20261693)
        pcontrol=BudgetPolicy(r,'global',[(4,1024)],160*2**20,4,micro_mode='full')
        # Force the cached fast layout for the ample-storage control, before allocation.
        c=candidate(r,4,1024,4,'fast',4,4096);pcontrol.choices={(4,1024):c};pcontrol.workspace_floats=c['activations']+c['scratch']
        pcontrol.resident_bound_bytes=c['resident_bound_bytes'];pcontrol.policy_id=ident(pcontrol.to_dict(include_id=False))
        padaptive=BudgetPolicy(r,'global',[(4,1024)],160*2**20,4,lookup,micro_mode='all')
        with Runner(pcontrol) as control,Runner(padaptive) as adaptive:
            control.restore(state);adaptive.restore(state);assert control.digest()==adaptive.digest()==d['state_sha256']
            before,_,_=evaluate_records(control,test,'ordered');events={'control':[],'adaptive':[]};decisions=[]
            for i,row in enumerate(records):
                if i%10==0:
                    budget=protocol['budgets_mib'][i//10];pnew=BudgetPolicy(r,'global',[(4,1024)],budget*2**20,4,lookup,micro_mode='all')
                    digest=adaptive.digest();adaptive.reconfigure(pnew);assert adaptive.digest()==digest
                    decisions.append({'first_update':i+1,'policy':pnew.to_dict()})
                x,y=trim(row);events['control'].append(control.update(x,y,.0001));events['adaptive'].append(adaptive.update(x,y,.0001))
            ctest,cp,cy=evaluate_records(control,test,'ordered');atest,ap,ay=evaluate_records(adaptive,test,'ordered')
            w=control.buffers['weights'];v=adaptive.buffers['weights']
            row={'seed':seed,'before_test':before,'control_test':ctest,'adaptive_test':atest,
                'max_weight_difference':float(np.max(np.abs(w-v))),'relative_weight_l2':float(np.linalg.norm(w-v)/np.linalg.norm(w)),
                'prediction_agreement':float(np.mean(np.array(cp)==np.array(ap))),
                'control_state_sha256':control.digest(),'adaptive_state_sha256':adaptive.digest(),
                'control_last_step':int(control.state.step),'adaptive_last_step':int(adaptive.state.step),
                'events':events,'decisions':decisions,'logical_input_sha256':input_digest(records),
                'fresh_test_input_sha256':input_digest(test)}
            assert row['control_last_step']==row['adaptive_last_step']==d['selected_step']+40
            assert np.isfinite(row['relative_weight_l2']) and row['relative_weight_l2']<1e-3,row['relative_weight_l2']
            rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('events','decisions')}),flush=True)
    (ROOT/'results/continuation.json').write_text(json.dumps(rows,indent=2))
