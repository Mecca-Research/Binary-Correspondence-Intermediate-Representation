"""Budgeted host controller over BCIR C plans, with reusable bounded workspace.

The declared budget covers native-array allocations, pool/staging storage and
fixed C call records. It does not claim to bound process RSS or Python objects.
No optimizer or tensor arithmetic falls back to Python.
"""
from experiment import *
from native_abi import _Spec,_Plan,_State,_Optimizer,_Event,_Provider
from dataclasses import replace
import statistics

FAMILIES={
 'query':('query_guard_compact','query_guard_frame_ff_compact','query_guard_layer_frame_ff'),
 'global':('guard_compact','guard_frame_ff_compact','guard_layer_frame_ff'),
 'guard':('guard_compact','guard_frame_ff_compact','guard_layer_frame_ff'),
 'hybrid':('guard_hybrid_compact',),
 'attention':('attention',),
}

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
def ident(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def aligned(n,dtype=np.float32):
    dt=np.dtype(dtype);base=np.zeros(int(n)+64//dt.itemsize,dtype=dt)
    shift=(-base.ctypes.data)%64//dt.itemsize
    return base[shift:shift+int(n)]

_LIBS={}
def library(name):
    if name not in _LIBS:
        lib=ct.CDLL(str(ROOT/'build'/f'{name}.so'))
        assert lib.bcir_decoder_abi_version()==4,'budget controller requires updated ABI4'
        lib.bcir_decoder_workspace_bytes.argtypes=[ct.POINTER(_Plan),ct.c_uint32];lib.bcir_decoder_workspace_bytes.restype=SZ
        lib.bcir_decoder_make_plan.argtypes=[ct.POINTER(_Spec),ct.POINTER(_Plan)]
        lib.bcir_tensor_pool_size.argtypes=[ct.c_uint];lib.bcir_tensor_pool_size.restype=SZ
        lib.bcir_tensor_pool_init.argtypes=[ct.c_void_p,SZ,ct.c_uint];lib.bcir_tensor_pool_init.restype=ct.c_void_p
        lib.bcir_tensor_pool_stop.argtypes=[ct.c_void_p]
        lib.bcir_decoder_forward.argtypes=[ct.POINTER(_State),UP,SZ]
        lib.bcir_decoder_loss_backward.argtypes=[ct.POINTER(_State),UP,UP,SZ,ct.c_int,ct.POINTER(ct.c_double)]
        lib.bcir_decoder_run_accum.argtypes=[ct.POINTER(_State),ct.POINTER(_Optimizer),UP,UP,SZ,
            ct.POINTER(ct.c_double),SZ,SZ,ct.POINTER(_Event),SZ,ct.POINTER(SZ)]
        _LIBS[name]=lib
    return _LIBS[name]

def plan(r,variant):
    spec=_Spec(r.vocab,r.width,r.heads,r.heads,r.layers,r.ff,r.batch,r.context,1,10000.,r.eps)
    p=_Plan();Native.check(library(variant).bcir_decoder_make_plan(ct.byref(spec),ct.byref(p)))
    assert p.parameters==sum(n for _,_,_,n in layout(r))
    return p

def candidate(r,effective,context,micro,variant,threads,staging_tokens):
    if effective%micro:raise ValueError('microbatch must divide effective batch')
    rr=replace(r,batch=micro,context=context);p=plan(rr,variant)
    pool=int(library(variant).bcir_tensor_pool_size(threads))
    pack_bytes=int(library(variant).bcir_decoder_workspace_bytes(ct.byref(p),r.contract))
    if pack_bytes>=2**63:raise MemoryError('native packing-workspace overflow')
    arena_end=int(p.activations+p.scratch)
    pack_offset=(arena_end+15)//16*16 if pack_bytes else arena_end
    workspace_floats=pack_offset+(pack_bytes+3)//4
    # Four persistent arrays + one unified workspace, each with 64 bytes alignment slack.
    # Two staging arrays have their own 64 bytes slack; the pool allocation adds 64.
    # C records are explicit; CPython/NumPy metadata, code and datasets are excluded.
    records=ct.sizeof(_State)+ct.sizeof(_Optimizer)+ct.sizeof(_Event)+ct.sizeof(ct.c_double)+ct.sizeof(SZ)
    fixed=int(p.parameters)*16+pool+8*staging_tokens+8*64+records
    out={'variant':variant,'effective_batch':effective,'microbatch':micro,'accumulation':effective//micro,
        'context':context,'parameters':int(p.parameters),'activations':int(p.activations),'scratch':int(p.scratch),
        'packing_bytes':pack_bytes,'packing_offset_floats':pack_offset,'workspace_floats':workspace_floats,
        'arena_bytes':4*workspace_floats,'fixed_bytes':fixed,'resident_bound_bytes':fixed+4*workspace_floats,
        'threads':threads,'staging_tokens':staging_tokens,
        'source_sha256':hashlib.sha256((ROOT/'build'/f'{variant}.c').read_bytes()).hexdigest()}
    out['candidate_id']=ident(out);return out

class BudgetPolicy:
    def __init__(self,r,family,shapes,budget_bytes,threads=4,calibration=None,micro_mode='full'):
        if family in ('global','query','hybrid') and not r.precision_guard:
            raise ValueError('this study portfolio requires the explicit precision_guard recipe contract')
        self.recipe=r;self.family=family;self.threads=threads;self.budget=int(budget_bytes)
        self.shapes=tuple(sorted(set(shapes)));self.staging=max(b*t for b,t in self.shapes)
        self.calibration=calibration or {};self.choices={};self.refusals=[]
        for b,t in self.shapes:
            micros=(b,) if micro_mode=='full' else (1,) if micro_mode=='one' else tuple(q for q in (1,2,4,8,16) if q<=b and b%q==0)
            allc=[candidate(r,b,t,m,v,threads,self.staging) for m in micros for v in FAMILIES[family]]
            feasible=[c for c in allc if c['resident_bound_bytes']<=self.budget]
            if not feasible:
                raise MemoryError(f'No native plan fits B{b}/T{t}, budget {self.budget}; minimum {min(c["resident_bound_bytes"] for c in allc)}')
            def score(c):
                k=f'{c["variant"]}:{b}:{t}:{c["microbatch"]}:{threads}'
                # Full-microbatch quality-controlled runs can use global timing estimates
                # for guarded/hybrid layouts, whose arithmetic contracts are separate.
                fallback=k.replace('guard_','').replace('hybrid_','')
                ms=self.calibration.get(k,self.calibration.get(fallback))
                if ms is None:
                    # Declared uncalibrated fallback: prefer larger microbatches, then
                    # cheaper recomputation tier; calibration status remains visible.
                    tier=next(i for i,v in enumerate(FAMILIES[family]) if v==c['variant'])
                    return (1,-c['microbatch'],tier,c['arena_bytes'])
                return (0,ms,c['arena_bytes'],c['variant'])
            chosen=min(feasible,key=score).copy();chosen['budget_bytes']=self.budget
            key=f'{chosen["variant"]}:{b}:{t}:{chosen["microbatch"]}:{threads}'
            fallback=key.replace('guard_','').replace('hybrid_','')
            chosen['calibrated_ms']=self.calibration.get(key,self.calibration.get(fallback))
            chosen['calibration_kind']='direct' if key in self.calibration else 'global-layout-proxy' if fallback in self.calibration else 'uncalibrated-fallback'
            self.choices[(b,t)]=chosen
        self.workspace_floats=max(c['workspace_floats'] for c in self.choices.values())
        # Staging/fixed sizes are common to all candidates, so max arena still fits.
        self.resident_bound_bytes=next(iter(self.choices.values()))['fixed_bytes']+self.workspace_floats*4
        assert self.resident_bound_bytes<=self.budget
        self.policy_id=ident(self.to_dict(include_id=False))
    def to_dict(self,include_id=True):
        out={'family':self.family,'budget_bytes':self.budget,'threads':self.threads,'recipe':asdict(self.recipe),
            'staging_tokens':self.staging,'resident_bound_bytes':self.resident_bound_bytes,
            'choices':list(self.choices.values()),'scope':'native array/pool/staging/C-record bound; excludes code, Python objects, datasets and checkpoints'}
        if include_id:out['policy_id']=self.policy_id
        return out

class Runner(Native):
    """One optimizer state and workspace; C plans are rebound at update boundaries."""
    def __init__(self,policy,seed=1):
        self.policy=policy;self.threads=policy.threads;self.r=policy.recipe
        first=next(iter(policy.choices.values()));self.lib=library(first['variant']);self.pool_lib=self.lib
        self.plan=plan(replace(self.r,batch=first['microbatch'],context=first['context']),first['variant'])
        p=int(self.plan.parameters);self.buffers={n:aligned(p) for n in ('weights','gradients','moment1','moment2')}
        self.buffers['weights'][:]=initial(self.r,seed)
        self.workspace=aligned(policy.workspace_floats);self.stage_x=aligned(policy.staging,np.uint32);self.stage_y=aligned(policy.staging,np.uint32)
        self.pool_memory=ct.create_string_buffer(self.lib.bcir_tensor_pool_size(self.threads)+64)
        self.pool=self.lib.bcir_tensor_pool_init((ct.addressof(self.pool_memory)+63)&~63,self.lib.bcir_tensor_pool_size(self.threads),self.threads)
        if not self.pool:raise RuntimeError('native pool initialization failed')
        self.state=_State();self.state.step=0;self.state.beta1_power=1.;self.state.beta2_power=1.
        self.last_binding=None;self.bind_count=0;self.choice_counts={};self.total_native_seconds=0.;self.total_wall_seconds=0.
        self.bind(first['effective_batch'],first['context'])
    def close(self):
        if self.pool:self.pool_lib.bcir_tensor_pool_stop(self.pool);self.pool=None
    def reconfigure(self,new_policy):
        """Release old work/staging arrays before resizing; retain optimizer state.

        Only called at completed update boundaries. Public inference returns
        copies, so the supported controller API has no borrowed workspace views.
        Freed pages may remain in the allocator: this is a live-array, not RSS bound.
        """
        old_spec={k:v for k,v in asdict(self.policy.recipe).items() if k not in ('batch','context')}
        new_spec={k:v for k,v in asdict(new_policy.recipe).items() if k not in ('batch','context')}
        if old_spec!=new_spec or self.policy.family!=new_policy.family or self.threads!=new_policy.threads:
            raise ValueError('reconfiguration must preserve model, numerical family and pool ABI')
        step,b1,b2=self.state.step,self.state.beta1_power,self.state.beta2_power
        self.state=_State();self.state.step=step;self.state.beta1_power=b1;self.state.beta2_power=b2
        del self.buffers['activations'];del self.buffers['scratch']
        self.pack=None;self.workspace=None;self.stage_x=None;self.stage_y=None
        self.policy=new_policy
        self.workspace=aligned(new_policy.workspace_floats)
        self.stage_x=aligned(new_policy.staging,np.uint32);self.stage_y=aligned(new_policy.staging,np.uint32)
        self.last_binding=None
        first=next(iter(new_policy.choices.values()));self.bind(first['effective_batch'],first['context'])
    def bind(self,effective,context):
        c=self.policy.choices[(effective,context)];key=(c['variant'],c['microbatch'],context)
        if key==self.last_binding:return c
        if hashlib.sha256((ROOT/'build'/f'{c["variant"]}.c').read_bytes()).hexdigest()!=c['source_sha256']:
            raise ValueError('native plan source changed after admission')
        step,b1,b2=self.state.step,self.state.beta1_power,self.state.beta2_power
        self.r=replace(self.policy.recipe,batch=c['microbatch'],context=context);self.lib=library(c['variant']);self.plan=plan(self.r,c['variant'])
        a,s=int(self.plan.activations),int(self.plan.scratch)
        assert a+s<=len(self.workspace)
        self.buffers['activations']=self.workspace[:a];self.buffers['scratch']=self.workspace[a:a+s]
        po=c['packing_offset_floats'];pb=c['packing_bytes']
        self.pack=self.workspace[po:po+(pb+3)//4].view(np.uint8)[:pb] if pb else None
        lengths=[self.plan.parameters]*4+[a,s]
        self.state=_State(self.plan,*[ptr(self.buffers[n]) for n in ('weights','gradients','moment1','moment2','activations','scratch')],
            *lengths,step,b1,b2,_Provider(),self.pool,self.r.contract,
            None if self.pack is None else ct.c_void_p(self.pack.ctypes.data),pb)
        self.last_binding=key;self.bind_count+=1;return c
    def update(self,x,y,lr):
        before=time.perf_counter();x=np.asarray(x,dtype=np.uint32);y=np.asarray(y,dtype=np.uint32)
        if x.ndim!=2 or x.shape!=y.shape:raise ValueError('expected one effective update of [batch,time] inputs and targets')
        b,t=x.shape;c=self.bind(b,t);m=c['microbatch'];k=c['accumulation']
        counts=(y.reshape(k,m,t)!=IGNORE).sum(axis=(1,2))
        if not counts[0] or np.any(counts!=counts[0]):
            raise ValueError('equal supervised counts per microbatch required; unweighted unequal-loss accumulation is refused')
        size=x.size;xx=self.stage_x[:size].reshape(x.shape);yy=self.stage_y[:size].reshape(y.shape)
        np.copyto(xx,x);np.copyto(yy,y)
        opt=_Optimizer(lr,.9,.95,1e-8,.1,1.);rate=ct.c_double(lr);event=_Event();done=SZ()
        native_start=time.perf_counter()
        rc=self.lib.bcir_decoder_run_accum(ct.byref(self.state),ct.byref(opt),ptr(xx,UP),ptr(yy,UP),size,
            ct.byref(rate),1,k,ct.byref(event),1,ct.byref(done))
        native_seconds=time.perf_counter()-native_start;self.check(rc);assert done.value==1
        wall_seconds=time.perf_counter()-before;self.total_native_seconds+=native_seconds;self.total_wall_seconds+=wall_seconds
        self.choice_counts[c['candidate_id']]=self.choice_counts.get(c['candidate_id'],0)+1
        return {'step':int(event.step),'loss':event.loss,'grad_norm':event.grad_norm,'native_seconds':native_seconds,
            'wall_seconds':wall_seconds,'variant':c['variant'],'microbatch':m,'context':t,'effective_batch':b,
            'candidate_id':c['candidate_id'],'resident_bound_bytes':self.policy.resident_bound_bytes}
    def infer(self,x):
        x=np.asarray(x,dtype=np.uint32);b,t=x.shape;c=self.bind(b,t);m=c['microbatch'];out=[]
        for start in range(0,b,m):out.append(self.forward(np.ascontiguousarray(x[start:start+m])).copy())
        return np.concatenate(out,axis=0)

def calibration_map(rows):
    out={}
    for r in rows:
        key=f'{r["variant"]}:{r["effective_batch"]}:{r["context"]}:{r["microbatch"]}:{r["threads"]}'
        out.setdefault(key,[]).append(r['wall_ms_per_update'])
    return {k:statistics.median(v) for k,v in out.items()}
