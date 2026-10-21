from candidates import *
from build_next import compile_source,once
import torch
torch.set_num_threads(1)

def build_precision():
    source=(ROOT/'build/fast.c').read_text();marker='static void gp_pair_forward('
    source=once(source,marker,(ROOT/'precision_core.h').read_text()+'\n'+marker)
    source=once(source,'static void gp_pair_forward(const float *u,const float *v,float *out) {',
        'static void gp_pair_forward(const float *u,const float *v,float *out) {\n  if(wedge_ill_conditioned(u,v)){wedge_precise_forward(u,v,out);return;}')
    source=once(source,'static void gp_pair_backward(const float *u,const float *v,const float *dp,float *du,float *dv) {',
        'static void gp_pair_backward(const float *u,const float *v,const float *dp,float *du,float *dv) {\n  if(wedge_ill_conditioned(u,v)){wedge_precise_backward(u,v,dp,du,dv);return;}')
    compile_source(source,'precise')
    return source

if __name__=='__main__':
    build_precision();rng=np.random.default_rng(7108);u=rng.normal(size=22).astype(np.float32);v=rng.normal(size=22).astype(np.float32)
    i,j=np.triu_indices(22,1);rows=[]
    import itertools
    a,b,c,d=np.array(list(itertools.combinations(range(22),4))).T
    def residual(p):
        m=np.zeros((22,22));m[i,j]=p;m-=m.T
        return float(np.max(np.abs(m[a,b]*m[c,d]-m[a,c]*m[b,d]+m[a,d]*m[b,c])))
    for coefficient in (0.,1.,10.,100.,1000.,10000.,1000000.):
        vv=(v.astype(np.float64)+coefficient*u.astype(np.float64)).astype(np.float32);z=np.stack([vv,u])
        dp=rng.normal(size=(1,2,231)).astype(np.float32);dp[:,0]=0
        for lib in ('fast','precise'):
            with Native(Recipe(kind='global',ff=257),threads=1,library_name=lib) as n:
                f=n.lib.gp_wedge_forward;f.argtypes=[ct.c_void_p,SZ,SZ,FP,FP]
                g=n.lib.gp_wedge_backward;g.argtypes=[ct.c_void_p,SZ,SZ,FP,FP,FP]
                cache=np.zeros(z.size*2,np.float32);cache[:z.size]=z.ravel();out=np.empty((1,2,231),np.float32);dz=np.empty_like(cache)
                f(None,1,2,ptr(cache),ptr(out));g(None,1,2,ptr(cache),ptr(dp),ptr(dz))
                tz=torch.tensor(z,requires_grad=True);uu=tz[1].double();ww=tz[0].double()
                pp=uu[i]*ww[j]-uu[j]*ww[i];norm=torch.linalg.vector_norm(pp);tp=torch.where(norm>1e-6,pp/norm.clamp_min(1e-6),torch.zeros_like(pp)).float()
                (tp*torch.tensor(dp[0,1])).sum().backward()
                rel=float(np.linalg.norm((dz[:z.size].reshape(z.shape)-tz.grad.numpy()).astype(np.float64))/max(np.linalg.norm(tz.grad.numpy().astype(np.float64)),1e-30))
                row={'library':lib,'coefficient':coefficient,'max_plucker_residual':residual(out[0,1]),
                    'feature_error_vs_same_input_fp64':float(np.linalg.norm(out[0,1]-tp.detach().numpy())),
                    'gradient_relative_l2_vs_fp64':rel}
                aa=float(np.dot(u.astype(np.float64),u.astype(np.float64)));bb=float(np.dot(vv.astype(np.float64),vv.astype(np.float64)));ab=float(np.dot(u.astype(np.float64),vv.astype(np.float64)))
                selected=(aa*bb-ab*ab)<=1e-4*aa*bb
                row['guard_selected_fp64']=selected
                if lib=='precise':
                    assert rel<1e-4,row
                    if selected:assert row['max_plucker_residual']<1e-7,row
                    else:assert row['max_plucker_residual']<1e-6,row
                rows.append(row)
    # Exact collinearity is an explicitly absent plane under the new contract.
    with Native(Recipe(kind='global',ff=257),threads=1,library_name='precise') as n:
        f=n.lib.gp_wedge_forward;f.argtypes=[ct.c_void_p,SZ,SZ,FP,FP]
        g=n.lib.gp_wedge_backward;g.argtypes=[ct.c_void_p,SZ,SZ,FP,FP,FP]
        z=np.stack([u,u]);cache=np.zeros(z.size*2,np.float32);cache[:z.size]=z.ravel();out=np.empty((1,2,231),np.float32);dz=np.empty_like(cache)
        f(None,1,2,ptr(cache),ptr(out));g(None,1,2,ptr(cache),ptr(dp),ptr(dz));assert not np.any(out) and not np.any(dz[:z.size])
    (ROOT/'results/precision.json').write_text(json.dumps({'primitive':rows,'zero_plane_policy':'zero feature and VJP','input_quantization_reversed':False},indent=2))
    print(json.dumps(rows,indent=2))
