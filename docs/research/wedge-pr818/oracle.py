"""Independent Torch equations for the experimental C architectures."""
from experiment import *
import torch
from torch import nn
from torch.nn import functional as F
def torch_model_next(r,flat):
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.names=[n for n,_,_,_ in layout(r)]
            self.params=nn.ParameterList([nn.Parameter(torch.tensor(flat[o:o+n].reshape(sh))) for _,sh,o,n in layout(r)])
        def forward(self,ids):
            p=dict(zip(self.names,self.params));emb=p['embedding'];h=F.embedding(ids,emb)
            def rms(x,w):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+r.eps)*w
            def rope(x):
                b,t,heads,dim=x.shape
                angle=torch.arange(t,dtype=torch.float64)[:,None]*10000.**(-2*torch.arange(dim//2,dtype=torch.float64)[None,:]/dim)
                c=angle.cos().float()[None,:,None];s=angle.sin().float()[None,:,None]
                a,b=x[...,:dim//2],x[...,dim//2:];return torch.cat([a*c-b*s,b*c+a*s],-1)
            for layer in range(r.layers):
                w=lambda n:p[f'l{layer}.{n}']
                n1=rms(h,w('norm1'));b,t,d=h.shape
                if r.kind=='transformer':
                    q=rope(F.linear(n1,w('q')).reshape(b,t,r.heads,d//r.heads)).transpose(1,2)
                    k=rope(F.linear(n1,w('k')).reshape(b,t,r.heads,d//r.heads)).transpose(1,2)
                    v=F.linear(n1,w('v')).reshape(b,t,r.heads,d//r.heads).transpose(1,2)
                    mixed=F.linear(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(b,t,d),w('o'))
                else:
                    z=F.linear(n1,w('reduce'))
                    if r.rope:z=rope(z.unsqueeze(2)).squeeze(2)
                    prefix=torch.cat([torch.zeros_like(z[:,:1]).double(),z[:,:-1].double().cumsum(1)],1)
                    scale=torch.arange(t,dtype=torch.float64).clamp_min(1).rsqrt()
                    v=(prefix*scale[None,:,None]).float()
                    if r.kind=='hybrid':
                        previous=torch.cat([torch.zeros_like(z[:,:1]),z[:,:-1]],1)
                        v=(0.5*prefix*scale[None,:,None]+0.5*previous.double()).float()
                    rh=r.rank//r.subspaces;u=z.reshape(b,t,r.subspaces,rh);v=v.reshape(b,t,r.subspaces,rh)
                    i,j=torch.triu_indices(rh,rh,1);f=u[...,i]*v[...,j]-u[...,j]*v[...,i]
                    norm=torch.linalg.vector_norm(f.double(),dim=-1,keepdim=True)
                    unit=f.double()/norm.clamp_min(1e-6)
                    if r.subspaces>1:unit=torch.where(norm>1e-6,unit,torch.zeros_like(unit))
                    if r.precision_guard:
                        a=u.double();vv=v.double();fp=a[...,i]*vv[...,j]-a[...,j]*vv[...,i]
                        norm64=torch.linalg.vector_norm(fp,dim=-1,keepdim=True)
                        precise=torch.where(norm64>1e-6,fp/norm64.clamp_min(1e-6),torch.zeros_like(fp))
                        prod=a.square().sum(-1,keepdim=True)*vv.square().sum(-1,keepdim=True)
                        det=prod-(a*vv).sum(-1,keepdim=True).square()
                        unit=torch.where((prod==0)|(det<=1e-4*prod),precise,unit)
                    feature=(unit.float()/math.sqrt(r.subspaces)).flatten(-2)
                    g=F.linear(F.linear(feature,w('project_in')),w('project_out')) if r.bottleneck else F.linear(feature,w('project'))
                    if r.query_aware:
                        query=z.double()/torch.linalg.vector_norm(z.double(),dim=-1,keepdim=True).clamp_min(1e-6)
                        query=query.float()
                        # Reconstruct the antisymmetric matrix independently of C's packed loop.
                        matrix=torch.zeros((*feature.shape[:-1],r.rank,r.rank),dtype=feature.dtype)
                        matrix[...,i,j]=feature
                        matrix[...,j,i]=-feature
                        c=torch.einsum('...i,...ij->...j',query.double(),matrix.double()).float()
                        g=g+F.linear(c,w('contract'))
                    alpha=torch.sigmoid(F.linear(n1,w('gate_h'))+F.linear(g,w('gate_g')))
                    mixed=alpha*n1+(1-alpha)*g
                h=h+mixed;n2=rms(h,w('norm2'))
                h=h+F.linear(F.silu(F.linear(n2,w('ff_gate')))*F.linear(n2,w('ff_up')),w('ff_down'))
            return F.linear(rms(h,p['final_norm']),emb)
    return Model()
