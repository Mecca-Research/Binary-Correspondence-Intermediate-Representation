"""Native behavior under v -> v + c*u: exact wedge invariance vs finite precision.
Also compares FP64 pair products on identical FP32 input to isolate arithmetic
cancellation from already-quantized input. Diagnostic only, no model fitting.
"""
from experiment import *
import itertools
rng=np.random.default_rng(7108);u=rng.normal(size=22).astype(np.float32);v=rng.normal(size=22).astype(np.float32)
i,j=np.triu_indices(22,1);rows=[]
a,b,c,d=np.array(list(itertools.combinations(range(22),4))).T
def residual(p):
    m=np.zeros((22,22));m[i,j]=p;m-=m.T
    return float(np.max(np.abs(m[a,b]*m[c,d]-m[a,c]*m[b,d]+m[a,d]*m[b,c])))
for kind in ('global','global_symmetric','projector'):
    pc=253 if kind=='projector' else 231
    with Native(Recipe(kind=kind,ff=250 if kind=='projector' else 257),threads=1) as n:
        f=n.lib.gp_wedge_forward;f.argtypes=[ct.c_void_p,SZ,SZ,FP,FP]
        ref=None
        for coefficient in (0.,1.,10.,100.,1000.,10000.,1000000.):
            vv=(v.astype(np.float64)+coefficient*u.astype(np.float64)).astype(np.float32)
            z=np.stack([vv,u]);cache=np.zeros(z.size*2,np.float32);cache[:z.size]=z.ravel();out=np.empty((1,2,pc),np.float32)
            f(None,1,2,ptr(cache),ptr(out));feature=out[0,1].astype(np.float64)
            if ref is None:ref=feature.copy()
            row={'kind':kind,'coefficient':coefficient,'native_feature_change_l2':float(np.linalg.norm(feature-ref))}
            if kind=='global':
                uu=u.astype(np.float64);ww=vv.astype(np.float64)
                p=uu[i]*ww[j]-uu[j]*ww[i];p/=max(np.linalg.norm(p),1e-6)
                row['native_vs_same_input_fp64_products_l2']=float(np.linalg.norm(feature-p))
                row['same_input_fp64_feature_change_l2']=float(np.linalg.norm(p-ref))
                row['native_max_plucker_residual']=residual(feature)
                row['same_input_fp64_max_plucker_residual']=residual(p)
            rows.append(row)
(ROOT/'results/common-mode.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
