"""Archive deterministic continuation states and add a shorter-context control.

Replays the existing40-update long-context experiment and checks both final
digests against it. A new40-update T128 control uses the same first values and
learning rate. New T2048/T4096 tests are evaluation only, with no selection.
"""
from train_integrated import *

def ample(r,t,budget=160):
    p=BudgetPolicy(r,'global',[(4,t)],budget*2**20,4,micro_mode='full')
    c=candidate(r,4,t,4,'fast',4,4*t);p.choices={(4,t):c};p.workspace_floats=c['activations']+c['scratch']
    p.resident_bound_bytes=c['resident_bound_bytes'];p.policy_id=ident(p.to_dict(include_id=False));return p

if __name__=='__main__':
    previous=json.loads((ROOT/'results/continuation.json').read_text());lookup=json.loads((ROOT/'results/calibration.json').read_text())['lookup_ms']
    protocol={'stage':'continuation archival, deterministic replay and short-tail control',
        'updates':40,'lr':.0001,'long_context':1024,'short_context':128,'effective_batch':4,
        'long_arms':'identical data and budget sequence to continuation-protocol.json; require matching final state digests',
        'short_control':'same first values/labels as long data, crop to128 and move query to final position; same40 updates but1/8 as many physical tokens',
        'evaluation_lengths':[1024,2048,4096],'examples':[128,64,32],
        'fresh_test_seed_base':20261910,'selection':'none; all arms evaluated only after all40 updates',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT/'continuation-extension-protocol.json').write_text(json.dumps(protocol,indent=2));rows=[]
    for seed in SEEDS:
        d=json.loads((ROOT/'results'/f'train-ordered_bucket_global-{seed}.json').read_text());state=dict(np.load(ROOT/d['checkpoint'],allow_pickle=False))
        r=Recipe(**json.loads(str(state['recipe'])));records=ordered_records(40,4,seed+700,fixed=1024)
        policies={'long_control':ample(r,1024),'long_adaptive':BudgetPolicy(r,'global',[(4,1024)],160*2**20,4,lookup,micro_mode='all'),
            'short_control':ample(r,128,32)}
        runners={k:Runner(p) for k,p in policies.items()};events={k:[] for k in runners}
        try:
            for n in runners.values():n.restore(state)
            for i,row in enumerate(records):
                if i%10==0:
                    budget=(160,64,32,20)[i//10];p=BudgetPolicy(r,'global',[(4,1024)],budget*2**20,4,lookup,micro_mode='all')
                    runners['long_adaptive'].reconfigure(p)
                x,y=trim(row)
                for role in ('long_control','long_adaptive'):events[role].append(runners[role].update(x,y,.0001))
                xx=np.ascontiguousarray(x[:,:128]);xx[:,-1]=257;yy=np.full_like(xx,IGNORE);yy[:,-1]=x[:,1]
                events['short_control'].append(runners['short_control'].update(xx,yy,.0001))
            old=next(a for a in previous if a['seed']==seed)
            assert runners['long_control'].digest()==old['control_state_sha256']
            assert runners['long_adaptive'].digest()==old['adaptive_state_sha256']
            snapshots={k:n.snapshot() for k,n in runners.items()}
            metadata={}
            for role,n in runners.items():
                ck=ROOT/'checkpoints'/f'continuation-{role}-{seed}.npz'
                np.savez_compressed(ck,**snapshots[role],recipe=json.dumps(asdict(r)),seed=seed,family='global',role=role)
                metadata[role]={'checkpoint':str(ck.relative_to(ROOT)),'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),
                    'state_sha256':n.digest(),'training_seconds':sum(e['wall_seconds'] for e in events[role]),
                    'physical_positions':40*4*(128 if role=='short_control' else 1024),
                    'logical_examples':160,'events':events[role]}
        finally:
            for n in runners.values():n.close()
        peval=BudgetPolicy(r,'global',[(4,t) for t in protocol['evaluation_lengths']],512*2**20,4,micro_mode='all')
        metrics={}
        with Runner(peval) as n:
            for role,snap in [('before',state),*snapshots.items()]:
                n.restore(snap);before=n.digest();metrics[role]=[]
                for t,count in zip(protocol['evaluation_lengths'],protocol['examples']):
                    rec=eval_records('ordered',t,count,4,20261910+t);metric,_,_=evaluate_records(n,rec,'ordered')
                    assert n.digest()==before
                    result={'role':role,'context':t,'test':metric,'logical_input_sha256':input_digest(rec)}
                    metrics[role].append(result);print(json.dumps({'seed':seed,**result}),flush=True)
        rows.append({'seed':seed,'replay_matches_initial_digests':True,'states':metadata,'evaluation':metrics})
    (ROOT/'results/continuation-extension.json').write_text(json.dumps(rows,indent=2))
