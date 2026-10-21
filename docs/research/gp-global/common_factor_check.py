"""Erratum witness: local normalized wedges share u, so their sum is decomposable."""
from experiment import *
import itertools
rng=np.random.default_rng(710);z=rng.normal(size=(2,64,22)).astype(np.float32)
p=np.empty((2,64,231),np.float32)
with Native(Recipe(kind='grassmann'),threads=1) as n:
    f=n.lib.gp_wedge_forward;f.argtypes=[ct.c_void_p,SZ,SZ,FP,FP]
    f(None,2,64,ptr(z),ptr(p))
i,j=np.triu_indices(22,1);effective=np.zeros_like(z,dtype=np.float64)
for t in range(64):
    valid=[dt for dt in OFFSETS if dt<=t]
    for dt in valid:
        u=z[:,t];v=z[:,t-dt]
        pair=u[:,i]*v[:,j]-u[:,j]*v[:,i]
        norm=np.maximum(np.linalg.norm(pair.astype(np.float64),axis=-1),1e-6)
        effective[:,t]+=v.astype(np.float64)/norm[:,None]/len(valid)
expected=z[:,:,i].astype(np.float64)*effective[:,:,j]-z[:,:,j].astype(np.float64)*effective[:,:,i]
mat=np.zeros((2,64,22,22),np.float64);mat[:,:,i,j]=p;mat=mat-mat.swapaxes(-1,-2)
a,b,c,d=np.array(list(itertools.combinations(range(22),4))).T
res=mat[:,:,a,b]*mat[:,:,c,d]-mat[:,:,a,c]*mat[:,:,b,d]+mat[:,:,a,d]*mat[:,:,b,c]
out={'native_vs_common_factor_max_error':float(np.max(np.abs(p-expected))),
    'native_local_max_plucker_residual':float(np.max(np.abs(res))),
    'tested_frames':128,'relations_per_frame':len(a),
    'identity':'sum_s (u wedge v_s)/max(norm(u wedge v_s),eps)/m = u wedge [sum_s v_s/max(norm(u wedge v_s),eps)/m]',
    'correction':'A generic average of wedges can be nondecomposable, but the prior local mixer shares u within a token. Its nonzero features remain decomposable in exact arithmetic. The earlier generic counterexample did not establish failure for that mixer.'}
assert out['native_vs_common_factor_max_error']<1e-6 and out['native_local_max_plucker_residual']<1e-7
(ROOT/'results/common-factor-erratum.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
