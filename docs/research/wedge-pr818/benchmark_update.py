"""Same-process, paired legacy/merged update timings and packed-buffer probes.

Run only after the main training process has ended. Each timed observation starts
from the same optimizer state; order alternates AB/BA. Timing includes the native
training call and excludes state restore, digest and result serialization.
"""
from study import *
from dataclasses import replace
import statistics
import mmap,re


def huge_arena(n):
    """Production-style common 2MiB-aligned mapping; retain actual page evidence."""
    huge=2**21;names=('weights','gradients','moment1','moment2','activations','scratch')
    initial=n.snapshot();lengths=[len(n.buffers[name]) for name in names];offsets=[];end=0
    for length in lengths:
        end=(end+63)&~63;offsets.append(end);end+=4*length
    end=(end+63)&~63;po=end;pb=0 if n.pack is None else n.pack.nbytes;end+=pb
    mapping=mmap.mmap(-1,end+huge,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS)
    try:mapping.madvise(mmap.MADV_HUGEPAGE);advised=True
    except (OSError,AttributeError):advised=False
    lead=-ct.addressof(ct.c_char.from_buffer(mapping))%huge
    whole=np.frombuffer(mapping,dtype=np.uint8,offset=lead,count=end);whole.fill(0)
    n.buffers={name:np.frombuffer(mapping,dtype=np.float32,offset=lead+off,count=length)
               for name,off,length in zip(names,offsets,lengths)}
    n.pack=np.frombuffer(mapping,dtype=np.uint8,offset=lead+po,count=pb) if pb else None
    state_type=type(n.state);extra=[] if state_type is _State3 else [None if n.pack is None else ct.c_void_p(n.pack.ctypes.data),pb]
    n.state=state_type(n.plan,*[ptr(n.buffers[name]) for name in names],*lengths,0,1.,1.,_Provider(),n.pool,n.r.contract,*extra)
    n.restore(initial);n.huge_mapping=mapping
    return {'advice_accepted':advised,'alignment_bytes':huge,'payload_bytes':end,'address':whole.ctypes.data}


def page_evidence(info):
    if info is None:return None
    lo=info['address'];hi=lo+info['payload_bytes'];entries=[];current=None
    for line in Path('/proc/self/smaps').read_text().splitlines():
        m=re.match(r'^([0-9a-f]+)-([0-9a-f]+) ',line)
        if m:
            a,b=(int(z,16) for z in m.groups());current={} if a<hi and b>lo else None
            if current is not None:entries.append(current)
        elif current is not None:
            for key in ('Size','Rss','AnonHugePages','THPeligible'):
                if line.startswith(key+':'):current[key]=int(line.split()[1])
    return {k:v for k,v in info.items() if k!='address'}|{'overlapping_VMA_records':entries,
           'observed_AnonHugePages_kib':sum(e.get('AnonHugePages',0) for e in entries)}


def interval(ratios):
    a=np.asarray(ratios,np.float64)
    rng=np.random.default_rng(81820261018)
    bs=np.median(rng.choice(a,(10000,len(a)),replace=True),axis=1)
    return {'median':float(np.median(a)),'mean':float(np.mean(a)),
            'min':float(a.min()),'max':float(a.max()),
            'paired_bootstrap_95_percent_interval':np.quantile(bs,[.025,.975]).tolist()}


def timed_pair(label,r,left,right,rounds=24,steps=3,equal=True,left_huge=False,right_huge=False,right_contract=None):
    rng=np.random.default_rng(81820261018+r.context+r.ff)
    x=rng.integers(0,r.vocab,(steps,r.batch,r.context),dtype=np.uint32)
    y=np.roll(x,-1,axis=2).copy()
    rates=np.full(steps,.0001,np.float64)
    rows=[]
    rb=r if right_contract is None else replace(r,contract=right_contract)
    with Native(r,seed=881,threads=4,library_name=left) as a, Native(rb,seed=881,threads=4,library_name=right) as b:
        pa=huge_arena(a) if left_huge else None;pb=huge_arena(b) if right_huge else None
        init=a.snapshot()
        if not equal:
            aa=a.forward(x[0]).copy();bb=b.forward(x[0]).copy()
            la=a.backward(x[0],y[0]);lb=b.backward(x[0],y[0])
            ga=a.buffers['gradients'].astype(np.float64);gb=b.buffers['gradients'].astype(np.float64)
            numerical={'a_loss':la,'b_loss':lb,'logits_max_abs_delta':float(np.max(np.abs(aa-bb))),
                       'gradient_relative_l2_delta':float(np.linalg.norm(ga-gb)/np.linalg.norm(ga))}
        else:numerical=None
        for n in (a,b):n.restore(init);n.train(x,y,rates)
        a.restore(init);b.restore(init)
        if equal:
            a.train(x,y,rates);b.train(x,y,rates)
            assert a.digest()==b.digest(),(label,'pre-timing parity')
        for q in range(rounds):
            vals={};digests={};losses={}
            for side,n in ((('a',a),('b',b)) if q%2==0 else (('b',b),('a',a))):
                n.restore(init)
                seconds,events=n.train(x,y,rates)
                vals[side]=seconds/steps;digests[side]=n.digest();losses[side]=[e['loss'] for e in events]
            if equal:
                assert digests['a']==digests['b'] and losses['a']==losses['b'],(label,q)
            rows.append({'round':q,'order':'AB' if q%2==0 else 'BA',
                         'a_seconds_per_update':vals['a'],'b_seconds_per_update':vals['b'],
                         'a_over_b':vals['a']/vals['b'],'a_state_sha256':digests['a'],'b_state_sha256':digests['b'],
                         'losses_bitwise_equal':losses['a']==losses['b']})
        def buffers(n):return sum(v.nbytes for v in n.buffers.values())+(0 if n.pack is None else n.pack.nbytes)
        out={'label':label,'recipe':asdict(r),'b_recipe':asdict(rb),'parameters':int(a.plan.parameters),'threads':4,
             'a':left,'b':right,'timed_rounds':rounds,'updates_per_observation':steps,
             'bitwise_equality_required':equal,
             'state_and_loss_bitwise_equal':all(z['a_state_sha256']==z['b_state_sha256'] and z['losses_bitwise_equal'] for z in rows),
             'initial_numerical_comparison':numerical,'a_buffer_bytes':buffers(a),'b_buffer_bytes':buffers(b),
             'a_packing_bytes':0 if a.pack is None else a.pack.nbytes,
             'b_packing_bytes':0 if b.pack is None else b.pack.nbytes,
             'a_huge_pages':page_evidence(pa),'b_huge_pages':page_evidence(pb),
             'a_median_ms_per_update':1000*statistics.median(z['a_seconds_per_update'] for z in rows),
             'b_median_ms_per_update':1000*statistics.median(z['b_seconds_per_update'] for z in rows),
             'a_over_b':interval([z['a_over_b'] for z in rows]),'observations':rows}
        if right.endswith('_probe'):
            b.lib.bcir_decoder_workspace_probe.restype=SZ;b.lib.bcir_decoder_panels_probe.restype=ct.c_uint
            out['b_observed_packing_bytes']=int(b.lib.bcir_decoder_workspace_probe())
            out['b_panel_paths_bitmask']=int(b.lib.bcir_decoder_panels_probe())
    print(json.dumps({k:out[k] for k in ('label','a_median_ms_per_update','b_median_ms_per_update','a_over_b')}),flush=True)
    return out


def updater_pair(rounds=24,updates=32):
    r=replace(recipe('query_full'),context=32,batch=1)
    rng=np.random.default_rng(818032);rows=[]
    with Native(r,seed=881,threads=4,library_name='old_query_guard_compact') as a,Native(r,seed=881,threads=4,library_name='query_guard_compact') as b:
        init=a.snapshot();g=rng.normal(0,.01,int(a.plan.parameters)).astype(np.float32)
        opt=_Optimizer(.0001,.9,.95,1e-8,.1,1.)
        for n in (a,b):n.buffers['gradients'][:]=g
        def block(n):
            gn=ct.c_double();start=time.perf_counter()
            for _ in range(updates):n.check(n.lib.bcir_decoder_update(ct.byref(n.state),ct.byref(opt),1.,ct.byref(gn)))
            return (time.perf_counter()-start)/updates
        for n in (a,b):n.restore(init);block(n)
        for q in range(rounds):
            vals={}
            for side,n in ((('a',a),('b',b)) if q%2==0 else (('b',b),('a',a))):
                n.restore(init);vals[side]=block(n)
            assert a.digest()==b.digest()
            rows.append({'round':q,'order':'AB' if q%2==0 else 'BA','a_seconds_per_update':vals['a'],
                         'b_seconds_per_update':vals['b'],'a_over_b':vals['a']/vals['b']})
        out={'label':'FP32/AdamW-only/534144parameters','parameters':int(a.plan.parameters),'threads':4,
             'a':'old_query_guard_compact','b':'query_guard_compact','timed_rounds':rounds,'updates_per_observation':updates,
             'state_and_loss_bitwise_equal':True,'scope':'Fixed finite gradient; whole native update API including clipping/norm and state validation, no backward.',
             'a_median_ms_per_update':1000*statistics.median(z['a_seconds_per_update'] for z in rows),
             'b_median_ms_per_update':1000*statistics.median(z['b_seconds_per_update'] for z in rows),
             'a_over_b':interval([z['a_over_b'] for z in rows]),'observations':rows}
    print(json.dumps({k:out[k] for k in ('label','a_median_ms_per_update','b_median_ms_per_update','a_over_b')}),flush=True)
    return out


if __name__=='__main__':
    freeze()
    available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,available[:4])
    protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'study_protocol_sha256':hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),
      'cpu_affinity':sorted(os.sched_getaffinity(0)),
      'method':'24 paired AB/BA rounds, three native updates each, warm pools, fixed synthetic dense targets and identical initial state restored outside each timed call. No other experiment job runs concurrently.',
      'scope':'Same four logical CPUs and four native threads. Inference/selection/data loading excluded. Compact cached arena for both sides. Old/new and panel comparisons hold allocation placement fixed; Placement rows separately compare merged default allocation against a merged 2MiB-aligned madvise arena.',
      'interpretation':'Ratios above1 favor B. Bootstrap intervals describe repeated timing observations on this host, not population/hardware uncertainty. BF16 panel tests compare one arithmetic contract with panels disabled/enabled.'}
    (ROOT/'timing-protocol.json').write_text(json.dumps(protocol,indent=2))
    # _Optimizer is an underscored ABI name and is not exported by import-star.
    from native_abi import _Optimizer,_State3,_Provider
    rows=[updater_pair()]
    for t in (128,256,512):
        for m,new in (('global','guard_compact'),('query_full','query_guard_compact'),('query_matched','query_guard_compact'),('attention','attention')):
            r=replace(recipe(m),context=t,batch=4)
            rows.append(timed_pair(f'FP32/{m}/B4T{t}',r,'old_'+new,new))
            (ROOT/'results/paired-timing.json').write_text(json.dumps({'protocol':protocol,'rows':rows},indent=2))
    for t,b in ((32,2),(256,4)):
        for m,new in (('query_full','query_guard_compact'),('attention','attention')):
            r=replace(recipe(m),context=t,batch=b,ff=256,contract=26)
            rows.append(timed_pair(f'BF16-panels/{m}/B{b}T{t}/FF256',r,new+'_nopanels',new+'_probe'))
            (ROOT/'results/paired-timing.json').write_text(json.dumps({'protocol':protocol,'rows':rows},indent=2))
    for t in (128,256,512):
        r=replace(recipe('query_full'),context=t,batch=4)
        rows.append(timed_pair(f'Placement/query_full/B4T{t}',r,'query_guard_compact','query_guard_compact',right_huge=True))
        (ROOT/'results/paired-timing.json').write_text(json.dumps({'protocol':protocol,'rows':rows},indent=2))
    for t in (128,256,512):
        r=replace(recipe('attention'),context=t,batch=4)
        rows.append(timed_pair(f'Numerics/attention-FMA/B4T{t}',r,'attention','attention',right_contract=46,equal=False))
        (ROOT/'results/paired-timing.json').write_text(json.dumps({'protocol':protocol,'rows':rows},indent=2))
    for m,new in (('query_full','query_guard_compact'),('attention','attention')):
        r=replace(recipe(m),context=256,batch=4)
        rows.append(timed_pair(f'Numerics/BF16/{m}/B4T256',r,new,new,right_contract=26,equal=False))
        (ROOT/'results/paired-timing.json').write_text(json.dumps({'protocol':protocol,'rows':rows},indent=2))
    lib=ct.CDLL(str(ROOT/'build/attention.so'))
    mask=int(lib.bcir_tensor_mm_bf16_variants())
    metadata={'bf16_available_variants':mask,'bf16_instances':{str(v):name for v,name in ((1,'portable C'),(4,'AVX-512F'),(8,'Intel AMX')) if mask&v},
              'AMX_available':bool(mask&8),'floating_point_contracts':{'primary':14,'BF16':26}}
    (ROOT/'results/paired-timing.json').write_text(json.dumps({'protocol':protocol,'metadata':metadata,'rows':rows},indent=2))
