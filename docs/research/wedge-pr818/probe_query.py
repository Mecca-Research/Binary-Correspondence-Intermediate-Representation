"""Additional guarded geometry and budget-transition checks after training."""
from validate_query import *
from study import recipe,family

def guarded_geometry(rng):
    out=[];lib=library('query_guard_compact')
    lib.gp_wedge_forward.argtypes=[ct.c_void_p,SZ,SZ,FP,FP]
    lib.gp_query_contract_forward.argtypes=[ct.c_void_p,SZ,FP,FP,FP]
    lib.gp_query_wedge_backward.argtypes=[ct.c_void_p,SZ,SZ,FP,FP,FP,FP]
    for mode in ('ordinary','zero','collinear','nearly_collinear','tiny_query'):
        b,t=2,7;z=rng.normal(size=(b,t,22)).astype(np.float32)
        if mode=='zero':z[:]=0
        if mode in ('collinear','nearly_collinear'):
            z[:]=np.linspace(.5,1.5,22,dtype=np.float32)
            if mode=='nearly_collinear':z+=rng.normal(size=z.shape).astype(np.float32)*1e-4
        if mode=='tiny_query':z[:,3]*=1e-8
        frames=np.zeros(z.size*2,np.float32);frames[:z.size]=z.ravel();p=np.zeros(b*t*231,np.float32);c=np.zeros(b*t*22,np.float32)
        dp=rng.normal(size=p.shape).astype(np.float32);dc=rng.normal(size=c.shape).astype(np.float32);dz=np.zeros_like(frames)
        lib.gp_wedge_forward(None,b,t,ptr(frames),ptr(p));lib.gp_query_contract_forward(None,b*t,ptr(frames),ptr(p),ptr(c))
        lib.gp_query_wedge_backward(None,b,t,ptr(frames),ptr(dp),ptr(dc),ptr(dz))
        u=torch.tensor(z,requires_grad=True);prefix=torch.cat([torch.zeros_like(u[:,:1]).double(),u[:,:-1].double().cumsum(1)],1)
        v=(prefix*torch.arange(t,dtype=torch.float64).clamp_min(1).rsqrt()[None,:,None]).float()
        i,j=torch.triu_indices(22,22,1);raw=u[...,i]*v[...,j]-u[...,j]*v[...,i]
        ordinary=raw.double()/torch.linalg.vector_norm(raw.double(),dim=-1,keepdim=True).clamp_min(1e-6)
        a=u.double();vv=v.double();f64=a[...,i]*vv[...,j]-a[...,j]*vv[...,i]
        norm64=torch.linalg.vector_norm(f64,dim=-1,keepdim=True)
        precise=torch.where(norm64>1e-6,f64/norm64.clamp_min(1e-6),torch.zeros_like(f64))
        prod=a.square().sum(-1,keepdim=True)*vv.square().sum(-1,keepdim=True);det=prod-(a*vv).sum(-1,keepdim=True).square()
        feature=torch.where((prod==0)|(det<=1e-4*prod),precise,ordinary).float()
        q=(a/torch.linalg.vector_norm(a,dim=-1,keepdim=True).clamp_min(1e-6)).float()
        matrix=torch.zeros((b,t,22,22));matrix[...,i,j]=feature;matrix[...,j,i]=-feature
        ctorch=torch.einsum('bti,btij->btj',q.double(),matrix.double()).float()
        loss=(feature*torch.tensor(dp.reshape(b,t,231))).sum()+(ctorch*torch.tensor(dc.reshape(b,t,22))).sum();loss.backward()
        row={'case':mode,'p_max':float(np.max(abs(p-feature.detach().numpy().ravel()))),
            'c_max':float(np.max(abs(c-ctorch.detach().numpy().ravel()))),'gradient_relative':relative(dz[:z.size],u.grad.numpy().ravel())}
        assert row['p_max']<1e-6 and row['c_max']<1e-6 and row['gradient_relative']<1e-5,row
        out.append(row)
    return out

def budgets(rng):
    out=[];r=recipe('query_full')
    # Keep microbatch1 fixed to isolate actual arena/recomputation transitions.
    reference=force_policy(r,'query_guard_compact',4,1024,1)
    with Runner(reference,seed=97) as ref,Runner(reference,seed=97) as n:
        pointer=n.buffers['weights'].ctypes.data
        for mib in (96,32,20,64):
            p=BudgetPolicy(r,'query',[(4,1024)],mib*2**20,4,micro_mode='one')
            before=n.digest();n.reconfigure(p);assert before==n.digest() and n.buffers['weights'].ctypes.data==pointer
            x=rng.integers(0,260,(4,1024),dtype=np.uint32);y=np.roll(x,-1,1)
            event=n.update(x,y,.0001);ref.update(x,y,.0001);assert n.digest()==ref.digest()
            out.append({'budget_mib':mib,'variant':event['variant'],'resident_bound_bytes':p.resident_bound_bytes,'state_sha256':n.digest()})
        before=n.digest()
        try:BudgetPolicy(r,'query',[(4,1024)],8*2**20,4,micro_mode='all')
        except MemoryError:pass
        else:raise AssertionError('unfit query plan admitted')
        assert n.digest()==before
    return out

def control_models(rng):
    out=[]
    for model,variant in (('global','guard_compact'),('hybrid','guard_hybrid_compact'),('attention','attention')):
        r=replace(recipe(model),width=32,layers=2,ff=48,batch=2,context=11)
        x=rng.integers(0,260,(2,11),dtype=np.uint32);x[:,1]=x[:,0];y=np.roll(x,-1,1)
        with Native(r,seed=83,threads=1,library_name=variant) as n:
            logits=n.forward(x).copy();ce=n.backward(x,y);g=n.buffers['gradients'].copy()
            reference=torch_model_next(r,n.buffers['weights']);p=reference(torch.tensor(x.astype(np.int64)))
            loss=F.cross_entropy(p.flatten(0,1),torch.tensor(y.astype(np.int64)).flatten());loss.backward()
            gg=np.concatenate([v.grad.numpy().ravel() for v in reference.params])
            row={'model':model,'variant':variant,'logit_max':float(np.max(abs(logits-p.detach().numpy()))),'gradient_relative':relative(g,gg),'loss_error':abs(ce-loss.item())}
            assert row['logit_max']<1e-5 and row['gradient_relative']<1e-5 and row['loss_error']<2e-6,row
            out.append(row)
    return out

if __name__=='__main__':
    rng=np.random.default_rng(765177)
    result={'guarded_geometry':guarded_geometry(rng),'control_models':control_models(rng),'budget_transitions':budgets(rng)}
    (ROOT/'results/precision-budget-probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
