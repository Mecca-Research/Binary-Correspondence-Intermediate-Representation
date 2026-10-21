"""Independent algebra/AD, native state, causal and bounded-buffer regression gates."""
from policy import *
from oracle import torch_model_next
import torch
from torch.nn import functional as F
torch.set_num_threads(1)

def relative(a,b):
    return float(np.linalg.norm(a.astype(np.float64)-b.astype(np.float64))/max(np.linalg.norm(b.astype(np.float64)),1e-30))

def force_policy(r,variant,b,t,m=None,threads=4):
    family='query' if r.query_aware else 'attention' if r.kind=='transformer' else 'global'
    p=BudgetPolicy(r,family,[(b,t)],256*2**20,threads)
    c=candidate(r,b,t,m or b,variant,threads,p.staging)
    p.choices={(b,t):c};p.workspace_floats=c['workspace_floats']
    p.resident_bound_bytes=c['resident_bound_bytes'];p.policy_id=ident(p.to_dict(include_id=False))
    return p

def contraction(u,p):
    q=(u.double()/torch.linalg.vector_norm(u.double()).clamp_min(1e-6)).float()
    matrix=torch.zeros((22,22),dtype=p.dtype);i,j=torch.triu_indices(22,22,1)
    matrix[i,j]=p;matrix[j,i]=-p
    return (q.double()@matrix.double()).float()

def primitive_checks(rng):
    lib=library('query_compact');lib.gp_query_contract_pair.argtypes=[FP,FP,FP]
    lib.gp_query_contract_vjp.argtypes=[FP,FP,FP,FP,FP]
    lib.gp_wedge_forward.argtypes=[ct.c_void_p,SZ,SZ,FP,FP]
    lib.gp_query_contract_forward.argtypes=[ct.c_void_p,SZ,FP,FP,FP]
    lib.gp_query_wedge_backward.argtypes=[ct.c_void_p,SZ,SZ,FP,FP,FP,FP]
    out=[]
    for scale in (1.,1e-8,1e3,0.):
        u=(rng.normal(size=22)*scale).astype(np.float32);p=rng.normal(size=231).astype(np.float32)
        dc=rng.normal(size=22).astype(np.float32);c=np.zeros(22,np.float32);dp=np.zeros(231,np.float32);du=np.zeros(22,np.float32)
        lib.gp_query_contract_pair(ptr(u),ptr(p),ptr(c));lib.gp_query_contract_vjp(ptr(u),ptr(p),ptr(dc),ptr(dp),ptr(du))
        tu=torch.tensor(u,requires_grad=True);tp=torch.tensor(p,requires_grad=True);tc=contraction(tu,tp)
        (tc*torch.tensor(dc)).sum().backward()
        row={'query_scale':scale,'forward_max':float(np.max(abs(c-tc.detach().numpy()))),
             'du_relative':relative(du,tu.grad.numpy()),'dp_relative':relative(dp,tp.grad.numpy())}
        assert row['forward_max']<1e-6 and row['du_relative']<2e-6 and row['dp_relative']<2e-6,row
        out.append(row)
    # Independent directional finite differences for the complete native prefix geometry.
    from experiment import torch_wedge
    for mode in ('ordinary','zero','collinear','nearly_collinear','tiny_query'):
        b,t=2,7;z=rng.normal(size=(b,t,22)).astype(np.float32)
        if mode=='zero':z[:]=0
        if mode in ('collinear','nearly_collinear'):
            z[:]=np.linspace(.5,1.5,22,dtype=np.float32)
            if mode=='nearly_collinear':z+=rng.normal(size=z.shape).astype(np.float32)*1e-4
        if mode=='tiny_query':z[:,3]*=1e-8
        frames=np.zeros(z.size*2,np.float32);frames[:z.size]=z.ravel();p=np.zeros(b*t*231,np.float32);c=np.zeros(b*t*22,np.float32)
        dp=rng.normal(size=p.shape).astype(np.float32);dc=rng.normal(size=c.shape).astype(np.float32);dz=np.zeros_like(frames)
        lib.gp_wedge_forward(None,b,t,ptr(frames),ptr(p))
        lib.gp_query_contract_forward(None,b*t,ptr(frames),ptr(p),ptr(c))
        lib.gp_query_wedge_backward(None,b,t,ptr(frames),ptr(dp),ptr(dc),ptr(dz))
        tz=torch.tensor(z,requires_grad=True);tp=torch_wedge(tz,'global')
        tq=(tz.double()/torch.linalg.vector_norm(tz.double(),dim=-1,keepdim=True).clamp_min(1e-6)).float()
        i,j=torch.triu_indices(22,22,1);mat=torch.zeros((b,t,22,22));mat[...,i,j]=tp;mat[...,j,i]=-tp
        tc=torch.einsum('bti,btij->btj',tq.double(),mat.double()).float()
        loss=(tp*torch.tensor(dp.reshape(b,t,231))).sum()+(tc*torch.tensor(dc.reshape(b,t,22))).sum();loss.backward()
        row={'geometry_case':mode,'forward_max':float(np.max(abs(c-tc.detach().numpy().ravel()))),
             'dz_relative':relative(dz[:z.size],tz.grad.numpy().ravel()),'finite':bool(np.isfinite(dz).all())}
        assert row['forward_max']<2e-5 and row['dz_relative']<1e-4 and row['finite'],row
        if mode=='ordinary':
            direction=rng.normal(size=z.shape).astype(np.float32);direction/=np.linalg.norm(direction)
            def objective(zz):
                f=np.zeros_like(frames);f[:z.size]=zz.ravel();pp=np.zeros_like(p);cc=np.zeros_like(c)
                lib.gp_wedge_forward(None,b,t,ptr(f),ptr(pp));lib.gp_query_contract_forward(None,b*t,ptr(f),ptr(pp),ptr(cc))
                return float(pp.astype(np.float64)@dp+cc.astype(np.float64)@dc)
            step=.003;fd=(objective(z+step*direction)-objective(z-step*direction))/(2*step)
            ad=float(dz[:z.size].astype(np.float64)@direction.ravel())
            row['directional_fd']=fd;row['directional_ad']=ad;row['fd_relative']=abs(fd-ad)/max(abs(fd),abs(ad),1e-8)
            assert row['fd_relative']<.01,row
        out.append(row)
    return out

def model_checks(rng):
    out=[]
    for width,layers,ff,b,t in ((32,2,48,2,11),(128,3,257,2,17),(128,3,250,2,13)):
        r=Recipe(kind='global',width=width,layers=layers,ff=ff,batch=b,context=t,query_aware=True,precision_guard=True)
        x=rng.integers(0,260,(b,t),dtype=np.uint32);y=rng.integers(0,260,(b,t),dtype=np.uint32)
        x[:,1]=x[:,0]
        for variant in FAMILIES['query']:
            with Native(r,seed=812,threads=1,library_name=variant) as n:
                logits=n.forward(x).copy();ce=n.backward(x,y);g=n.buffers['gradients'].copy()
                model=torch_model_next(r,n.buffers['weights']);pp=model(torch.tensor(x.astype(np.int64)))
                loss=F.cross_entropy(pp.flatten(0,1),torch.tensor(y.astype(np.int64)).flatten());loss.backward()
                gg=np.concatenate([v.grad.numpy().ravel() for v in model.params])
                row={'width':width,'layers':layers,'ff':ff,'variant':variant,'logit_max':float(np.max(abs(logits-pp.detach().numpy()))),
                     'loss_error':abs(ce-loss.item()),'gradient_relative':relative(g,gg),'gradient_max':float(np.max(abs(g-gg)))}
                assert row['logit_max']<1e-5 and row['loss_error']<2e-6 and row['gradient_relative']<1e-5,row
                out.append(row)
        # Future suffixes must not change earlier logits in this causal head.
        with Native(r,seed=812,library_name='query_guard_compact') as n:
            a=n.forward(x).copy();xx=x.copy();xx[:,t//2:]=(xx[:,t//2:]+17)%260;aa=n.forward(xx).copy()
            assert np.array_equal(a[:,:t//2],aa[:,:t//2])
    return out

def state_checks(rng):
    r=Recipe(kind='global',width=32,layers=2,ff=48,query_aware=True,precision_guard=True)
    sequence=[]
    for t in (11,23,37,11,23,37):
        x=rng.integers(0,260,(2,t),dtype=np.uint32);y=np.roll(x,-1,1);sequence.append((x,y))
    states=[];resume=None
    for threads in (1,4):
        with Runner(force_policy(r,'query_guard_compact',2,11,threads=threads),seed=83) as n:
            for step,(x,y) in enumerate(sequence):
                variant=FAMILIES['query'][step%3];n.reconfigure(force_policy(r,variant,2,x.shape[1],threads=threads))
                before=n.snapshot();n.update(x,y,.0003)
                with Native(replace(r,batch=2,context=x.shape[1]),seed=83,threads=threads,library_name='query_guard_compact') as ref:
                    ref.restore(before);ref.train(x[None],y[None],[.0003]);assert ref.digest()==n.digest(),(step,variant)
                if step==2:resume=n.snapshot()
            states.append(n.digest())
    assert states[0]==states[1]
    with Runner(force_policy(r,'query_guard_compact',2,11),seed=999) as n:
        n.restore(resume)
        for x,y in sequence[3:]:
            n.reconfigure(force_policy(r,'query_guard_compact',2,x.shape[1]));n.update(x,y,.0003)
        assert n.digest()==states[0]
    # Check exact ABI-sized allocations with live red zones, including UBSan modules.
    redzones=[]
    for variant in (*FAMILIES['query'],'query_guard_compact_ubsan','query_guard_layer_frame_ff_ubsan'):
        rr=replace(r,batch=2,context=29)
        with Native(rr,seed=83,threads=4,library_name=variant) as n:
            zones=[]
            names=('weights','gradients','moment1','moment2','activations','scratch')
            for name in names:
                old=n.buffers[name];base=np.full(len(old)+128,123456.75,np.float32);view=base[64:-64];view[:]=old
                n.buffers[name]=view;setattr(n.state,name,ptr(view));zones.append(base)
            x=rng.integers(0,260,(2,29),dtype=np.uint32);y=np.roll(x,-1,1)
            n.train(x[None],y[None],[.0003])
            assert all(np.all(a[:64]==123456.75) and np.all(a[-64:]==123456.75) for a in zones)
            redzones.append(variant)
    # Accumulation: normalize the stored SUM of microbatch means, as BCIR requires.
    rr=replace(r,width=32,ff=48);x=rng.integers(0,260,(4,23),dtype=np.uint32);y=np.roll(x,-1,1)
    vals=[]
    for m in (4,1):
        with Runner(force_policy(rr,'query_guard_compact',4,23,m),seed=83) as n:
            n.update(x,y,.0003);vals.append((n.buffers['gradients'].copy()/(4//m),n.buffers['weights'].copy()))
    grad=relative(vals[1][0],vals[0][0]);weights=relative(vals[1][1],vals[0][1]);assert grad<1e-5 and weights<1e-7
    return {'dynamic_shape_variant_thread_digest':states[0],'resume':'exact','redzones_ubsan':redzones,
            'normalized_accumulation_gradient_relative':grad,'first_updated_weights_relative':weights,'causality':'exact prefix equality'}

if __name__=='__main__':
    rng=np.random.default_rng(202610210)
    result={'primitive':primitive_checks(rng),'models':model_checks(rng),'state':state_checks(rng)}
    (ROOT/'results/validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
