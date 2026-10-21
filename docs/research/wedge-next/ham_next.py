"""Actual BCIR planner/verification on declared memory envelopes; no hardware DMA."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'BCIR-wedge-next'))
from bcir.frontends.models.assessment import BankEnvelope,HardwareEnvelope,LinkEnvelope
from bcir.kbcir.ham import HAMResource,HAMAccess,HAMWorkload,lower_ham_workload,verify_ham_plan
from bcir.kbcir.static_memory import LayoutItem,first_fit_layout,exact_layout,layout_lower_bound
H=lambda x:hashlib.sha256(x.encode()).hexdigest()

if __name__=='__main__':
    plans=json.loads((ROOT/'results/memory-plans.json').read_text());rows=[]
    # Conservatively keep each job's entire native allocation resident as one unit.
    # This is a read-residency scenario, not a physical training execution trace.
    for p in plans:
        if p['variant'] not in ('baseline','compact','frame_ff_compact','layer_compact','attention'):continue
        size=p['arena_bytes']+p['persistent_bytes']
        for capacity_mib in (32,48,64,96,128,192):
            capacity=capacity_mib*2**20
            hardware=HardwareEnvelope('declared-residency-example','x86_64',
                (BankEnvelope('ram','RAM','host',2**30,2**30,20_000_000_000,20_000_000_000),
                 BankEnvelope('fast','VRAM','gpu',capacity,capacity,100_000_000_000,100_000_000_000)),
                (LinkEnvelope('ram','fast',16_000_000_000,1000),LinkEnvelope('fast','ram',16_000_000_000,1000)),())
            resources=tuple(HAMResource(i,f'{p["variant"]}-job-{i}','tensor',size,'ram',H(f'{p["variant"]}:{i}')) for i in (1,2))
            access=tuple(HAMAccess(i+1,1+i%2,'fast','read',f'working-set-read-{i}') for i in range(6))
            work=HAMWorkload(hardware.digest,resources,access,0,2)
            try:
                lowered=lower_ham_workload(work,hardware);a=lowered.artifact
                assert verify_ham_plan(a,work,hardware)==()
                path=ROOT/'ham-artifacts'/f'{p["variant"]}-{capacity_mib}MiB.json';path.parent.mkdir(exist_ok=True)
                path.write_text(a.to_json())
                rows.append({'variant':p['variant'],'capacity_mib':capacity_mib,'one_job_bytes':size,'status':'verified',
                    'logical_transfer_bytes':a.logical_transfer_bytes,'wire_transfer_bytes':a.wire_transfer_bytes,
                    'predicted_transfer_ns':a.predicted_transfer_ns,'evictions':sum(x.kind=='evict' for x in a.actions),
                    'capacity_evictions':sum(x.kind=='evict' and x.reason.startswith('capacity:') for x in a.actions),
                    'action_count':len(a.actions),'plan_digest':a.digest,'artifact':str(path.relative_to(ROOT))})
            except ValueError as e:
                rows.append({'variant':p['variant'],'capacity_mib':capacity_mib,'one_job_bytes':size,'status':'refused','reason':str(e)})
    # Conservative scratch lifetime fixture. Existing C compact offsets are safe under
    # these declared serialized boundaries. The solver's bound concerns this fixture.
    n,d,f,p,r=1024,128,257,231,22
    specs=[('cur',d,3,16,0),('head_dn',d,2,4,d),('head_inv',1,3,4,3*d+p),
        ('ff_dh',f,4,6,d),('ff_dg',f,5,7,d+f),('ff_du',f,5,7,d+2*f),
        ('ff_dn',d,6,8,d),('ff_inv',1,7,8,3*d+p),('gate_dn',d,8,15,d),
        ('gate_dv',d,8,12,2*d),('gate_dc',d,8,11,3*d),('feature_dp',p,11,13,3*d),
        ('frame_dz',2*r,12,14,3*d+p),('input_inv',1,14,15,3*d+p)]
    items=[LayoutItem(i+1,count*n*4,64,first,last) for i,(_,count,first,last,_) in enumerate(specs)]
    # Check C's chosen offsets separately from the solver's potentially different plan.
    for i,a in enumerate(items):
        for j,b in enumerate(items[:i]):
            if a.conflicts(b):
                ao=specs[i][4]*n*4;bo=specs[j][4]*n*4
                assert ao+a.size<=bo or bo+b.size<=ao,(specs[i][0],specs[j][0])
    ff=first_fit_layout(items);exact=exact_layout(items,budget=200000)
    fixture={'items':[{'name':s[0],'size_bytes':it.size,'first':it.first_tick,'last':it.last_tick,'c_offset':s[4]*n*4} for s,it in zip(specs,items)],
        'c_extent_bytes':max((s[4]+s[1])*n*4 for s in specs),'c_alias_check':'pass',
        'first_fit_extent':ff.extent,'lower_bound':exact.lower_bound,'exact_extent':exact.extent,
        'stop_reason':exact.stop_reason,'work_units':exact.expansions,'offsets':exact.offsets,
        'scope':'These serialized lifetimes and disjoint VJP buffers only; changing operators/alias contracts changes the problem. Not a global training-memory optimum.'}
    (ROOT/'results/ham.json').write_text(json.dumps({'scenarios':rows,'scratch_fixture':fixture,
       'scope':'Planner-model residency test, two independent read-only whole native working sets, six alternating accesses, declared 16GB/s links. Does not execute GPU transfers or training.'},indent=2))
    print(json.dumps({'scenarios':rows,'scratch_fixture':fixture},indent=2))
