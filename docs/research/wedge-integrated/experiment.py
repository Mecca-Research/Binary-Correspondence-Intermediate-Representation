"""Matched native-C sequence models and independent Torch reference.

Run validate.py before run_experiments.py. The optional Torch module is a
correctness reference, not the engine used for native training measurements.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import ctypes as ct
import hashlib, json, math, sys, time
import numpy as np

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent/'BCIR-gp-head'
sys.path.insert(0,str(REPO))
from native_abi import _Spec,_Plan,_State,_Optimizer,_Event,_Provider

FP=ct.POINTER(ct.c_float);UP=ct.POINTER(ct.c_uint32);SZ=ct.c_size_t
OFFSETS=(1,2,4,8,16,32)
IGNORE=np.uint32(0xffffffff)

@dataclass(frozen=True)
class Recipe:
    kind:str='transformer'
    width:int=128
    layers:int=3
    heads:int=4
    ff:int=256
    rank:int=22
    vocab:int=260
    batch:int=4
    context:int=64
    contract:int=14  # kblock + exp + fixed-shape reduce
    eps:float=1e-5
    subspaces:int=1
    bottleneck:int=0
    rope:bool=False

def layout(r):
    d,f,v=r.width,r.ff,r.vocab
    headrank=r.rank//r.subspaces
    p=r.subspaces*headrank*(headrank-1)//2
    rows=[('embedding',(v,d))]
    for l in range(r.layers):
        if r.kind=='transformer':
            mix=[('q',(d,d)),('k',(d,d)),('v',(d,d)),('o',(d,d))]
        else:
            project=[('project_in',(r.bottleneck,p)),('project_out',(d,r.bottleneck))] if r.bottleneck else [('project',(d,p))]
            mix=[('reduce',(r.rank,d)),*project,('gate_h',(d,d)),('gate_g',(d,d))]
        rows += [(f'l{l}.{n}',s) for n,s in [('norm1',(d,)),*mix,('norm2',(d,)),
               ('ff_gate',(f,d)),('ff_up',(f,d)),('ff_down',(d,f))]]
    rows += [('final_norm',(d,))]
    out=[];off=0
    for name,shape in rows:
        n=math.prod(shape);out.append((name,shape,off,n));off+=n
    return out

def initial(r,seed):
    rows=layout(r);w=np.empty(sum(n for _,_,_,n in rows),np.float32)
    for name,shape,off,n in rows:
        sr=int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:8],'little')
        rng=np.random.default_rng(sr)
        if 'norm' in name:a=np.ones(shape,np.float32)
        else:
            # Unit-norm wedge features need unit-scale projection weights.
            scale=1.0 if name.endswith(('.project','.project_in')) else .02 if name=='embedding' else 1/math.sqrt(shape[-1])
            if name.endswith(('.o','.ff_down')):scale/=math.sqrt(2*r.layers)
            a=(rng.standard_normal(shape)*scale).astype(np.float32)
        w[off:off+n]=a.ravel()
    return w

def ptr(a,typ=FP):return a.ctypes.data_as(typ)

class Native:
    def __init__(self,r,seed=1,threads=4,weights=None,library_name=None):
        self.r=r;self.threads=threads
        self.lib=ct.CDLL(str(ROOT/'build'/f'{library_name or r.kind}.so'))
        lib=self.lib
        lib.bcir_decoder_make_plan.argtypes=[ct.POINTER(_Spec),ct.POINTER(_Plan)]
        lib.bcir_tensor_pool_size.argtypes=[ct.c_uint];lib.bcir_tensor_pool_size.restype=SZ
        lib.bcir_tensor_pool_init.argtypes=[ct.c_void_p,SZ,ct.c_uint];lib.bcir_tensor_pool_init.restype=ct.c_void_p
        lib.bcir_tensor_pool_stop.argtypes=[ct.c_void_p]
        lib.bcir_decoder_forward.argtypes=[ct.POINTER(_State),UP,SZ]
        lib.bcir_decoder_loss_backward.argtypes=[ct.POINTER(_State),UP,UP,SZ,ct.c_int,ct.POINTER(ct.c_double)]
        lib.bcir_decoder_update.argtypes=[ct.POINTER(_State),ct.POINTER(_Optimizer),ct.c_double,ct.POINTER(ct.c_double)]
        lib.bcir_decoder_run_accum.argtypes=[ct.POINTER(_State),ct.POINTER(_Optimizer),UP,UP,SZ,
                ct.POINTER(ct.c_double),SZ,SZ,ct.POINTER(_Event),SZ,ct.POINTER(SZ)]
        spec=_Spec(r.vocab,r.width,r.heads,r.heads,r.layers,r.ff,r.batch,r.context,1,10000.,r.eps)
        plan=_Plan();self.check(lib.bcir_decoder_make_plan(ct.byref(spec),ct.byref(plan)))
        self.plan=plan
        names=['weights','gradients','moment1','moment2','activations','scratch']
        lengths=[plan.parameters]*4+[plan.activations,plan.scratch]
        self.buffers={}
        for name,n in zip(names,lengths):
            base=np.zeros(int(n)+16,np.float32);shift=(-base.ctypes.data//4)%16
            self.buffers[name]=base[shift:shift+int(n)]
        assert sum(n for _,_,_,n in layout(r))==plan.parameters
        self.buffers['weights'][:]=initial(r,seed) if weights is None else weights
        self.pool_memory=ct.create_string_buffer(lib.bcir_tensor_pool_size(threads)+64)
        self.pool=lib.bcir_tensor_pool_init((ct.addressof(self.pool_memory)+63)&~63,
                    lib.bcir_tensor_pool_size(threads),threads)
        if not self.pool:raise RuntimeError('pool init')
        self.state=_State(plan,*[ptr(self.buffers[n]) for n in names],*lengths,0,1.,1.,_Provider(),self.pool,r.contract)
    @staticmethod
    def check(rc):
        if rc:raise RuntimeError(f'native return code {rc}')
    def close(self):
        if self.pool:self.lib.bcir_tensor_pool_stop(self.pool);self.pool=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def forward(self,x):
        x=np.ascontiguousarray(x,dtype=np.uint32)
        self.check(self.lib.bcir_decoder_forward(ct.byref(self.state),ptr(x,UP),x.size))
        return self.buffers['activations'][-x.size*self.r.vocab:].reshape(*x.shape,self.r.vocab)
    def backward(self,x,y):
        x=np.ascontiguousarray(x,dtype=np.uint32);y=np.ascontiguousarray(y,dtype=np.uint32)
        loss=ct.c_double()
        self.check(self.lib.bcir_decoder_loss_backward(ct.byref(self.state),ptr(x,UP),ptr(y,UP),x.size,0,ct.byref(loss)))
        return loss.value
    def train(self,x,y,rates):
        x=np.ascontiguousarray(x,dtype=np.uint32);y=np.ascontiguousarray(y,dtype=np.uint32)
        rates=np.ascontiguousarray(rates,dtype=np.float64);steps=len(rates)
        events=(_Event*steps)();done=SZ()
        opt=_Optimizer(float(rates[0]),.9,.95,1e-8,.1,1.)
        start=time.perf_counter()
        rc=self.lib.bcir_decoder_run_accum(ct.byref(self.state),ct.byref(opt),ptr(x,UP),ptr(y,UP),x.size,
                rates.ctypes.data_as(ct.POINTER(ct.c_double)),steps,1,events,steps,ct.byref(done))
        seconds=time.perf_counter()-start;self.check(rc);assert done.value==steps
        return seconds,[{'step':e.step,'loss':e.loss,'grad_norm':e.grad_norm} for e in events]
    def digest(self):
        h=hashlib.sha256()
        for n in ['weights','moment1','moment2']:h.update(self.buffers[n].tobytes())
        h.update(json.dumps([self.state.step,self.state.beta1_power,self.state.beta2_power]).encode())
        return h.hexdigest()
    def snapshot(self):
        return {n:self.buffers[n].copy() for n in ['weights','moment1','moment2']}|{'step':int(self.state.step),'b1':self.state.beta1_power,'b2':self.state.beta2_power}
    def restore(self,s):
        for n in ['weights','moment1','moment2']:self.buffers[n][:]=s[n]
        self.state.step=int(s['step']);self.state.beta1_power=float(s['b1']);self.state.beta2_power=float(s['b2'])

def torch_wedge(z,kind,offsets=OFFSETS):
    import torch
    b,t,r=z.shape
    if kind in ('global','global_symmetric','projector'):
        # Casts match native prefix-cache storage; accumulation is FP64.
        prefix=torch.cat([torch.zeros_like(z[:,:1]),z[:,:-1].double().cumsum(1).to(z.dtype)],1)
        scale=torch.arange(t,dtype=torch.float64,device=z.device).clamp_min(1).rsqrt()
        # Native rounds after scaling, so compute sum and scaling before cast.
        prefix=torch.cat([torch.zeros_like(z[:,:1]).double(),z[:,:-1].double().cumsum(1)],1)
        v=(prefix*scale[None,:,None]).to(z.dtype);u=z
        if kind=='projector':
            u=u.double();v=v.double()
            aa=u.square().sum(-1,keepdim=True)+1e-4
            dd=v.square().sum(-1,keepdim=True)+1e-4
            bb=(u*v).sum(-1,keepdim=True);det=aa*dd-bb*bb
            a=dd/det;b0=-bb/det;d=aa/det
            matrix=u.unsqueeze(-1)*(a*u+b0*v).unsqueeze(-2)+v.unsqueeze(-1)*(b0*u+d*v).unsqueeze(-2)
            i,j=torch.triu_indices(r,r,0,device=z.device)
            return (matrix[...,i,j]/math.sqrt(2)).to(z.dtype)
        i,j=torch.triu_indices(r,r,1,device=z.device)
        p=u[...,i]*v[...,j]+u[...,j]*v[...,i] if kind=='global_symmetric' else u[...,i]*v[...,j]-u[...,j]*v[...,i]
        norm=torch.linalg.vector_norm(p.double(),dim=-1,keepdim=True).clamp_min(1e-6)
        return (p.double()/norm).to(z.dtype)
    i,j=torch.triu_indices(r,r,1,device=z.device)
    total=torch.zeros(b,t,len(i),dtype=z.dtype,device=z.device)
    count=torch.zeros(t,dtype=z.dtype,device=z.device)
    for dt in offsets:
        if dt>=t:continue
        if kind=='future':u,v=z[:,:-dt],z[:,dt:];sl=slice(None,-dt)
        else:u,v=z[:,dt:],z[:,:-dt];sl=slice(dt,None)
        p=u[...,i]*v[...,j]+u[...,j]*v[...,i] if kind=='symmetric' else u[...,i]*v[...,j]-u[...,j]*v[...,i]
        norm=torch.linalg.vector_norm(p.double(),dim=-1,keepdim=True).clamp_min(1e-6)
        total[:,sl]=total[:,sl]+(p.double()/norm).to(z.dtype)
        count[sl]+=1
    return total/count.clamp_min(1)[None,:,None]

def torch_model(r,flat,dtype=None):
    import torch
    from torch import nn
    from torch.nn import functional as F
    if dtype is None:dtype=torch.float32
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.params=nn.ParameterList([nn.Parameter(torch.tensor(flat[o:o+n].reshape(sh),dtype=dtype)) for _,sh,o,n in layout(r)])
        def forward(self,ids):
            q=iter(self.params);emb=next(q);h=F.embedding(ids,emb)
            def rms(x,w):return (x*torch.rsqrt(x.square().mean(-1,keepdim=True)+r.eps))*w
            def rope(x):
                b,t,heads,dim=x.shape
                freq=torch.pow(torch.tensor(10000.,dtype=torch.float64),-2*torch.arange(dim//2,dtype=torch.float64)/dim)
                angles=torch.arange(t,dtype=torch.float64)[:,None]*freq[None,:]
                c=angles.cos().to(dtype)[None,:,None,:];s=angles.sin().to(dtype)[None,:,None,:]
                a,bb=x[...,:dim//2],x[...,dim//2:]
                return torch.cat([a*c-bb*s,bb*c+a*s],-1)
            for _ in range(r.layers):
                n1,wq,wk,wv,wo,n2,wg,wu,wd=[next(q) for _ in range(9)]
                h1=rms(h,n1)
                if r.kind=='transformer':
                    b,t,d=h.shape
                    qq=rope(F.linear(h1,wq).reshape(b,t,r.heads,d//r.heads)).transpose(1,2)
                    kk=rope(F.linear(h1,wk).reshape(b,t,r.heads,d//r.heads)).transpose(1,2)
                    vv=F.linear(h1,wv).reshape(b,t,r.heads,d//r.heads).transpose(1,2)
                    ctx=F.scaled_dot_product_attention(qq,kk,vv,is_causal=True).transpose(1,2).reshape(b,t,d)
                    mixed=F.linear(ctx,wo)
                else:
                    z=rope(F.linear(h1,wq).unsqueeze(2)).squeeze(2);p=torch_wedge(z,r.kind)
                    g=F.linear(p,wk);alpha=torch.sigmoid(F.linear(h1,wv)+F.linear(g,wo))
                    mixed=alpha*h1+(1-alpha)*g
                h=h+mixed;h2=rms(h,n2)
                h=h+F.linear(F.silu(F.linear(h2,wg))*F.linear(h2,wu),wd)
            return F.linear(rms(h,next(q)),emb)
    return Model()

def metrics(logits,targets):
    mask=targets!=IGNORE
    l=logits[mask].astype(np.float64);y=targets[mask].astype(np.int64)
    m=l.max(1);ce=np.log(np.exp(l-m[:,None]).sum(1))+m-l[np.arange(len(y)),y]
    return float(ce.sum()),int((l.argmax(1)==y).sum()),len(y)
