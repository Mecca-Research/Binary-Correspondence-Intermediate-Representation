"""Chunked feature mixer equals full-prefix C operation, retaining O(batch*rank) state."""
from experiment import *
import subprocess
out=[];rng=np.random.default_rng(891)
for kind in ('global','global_symmetric','projector'):
    so=ROOT/'build'/f'{kind}-stream.so'
    cmd=['gcc','-std=c11','-O3','-fPIC','-shared','-ffp-contract=off','-fno-math-errno','-pthread',
        '-I',str(ROOT/'vendor'),'-I',str(ROOT),'-DGP_RANK=22',
        f'-DGP_SYMMETRIC={int(kind=="global_symmetric")}',f'-DGP_PROJECTOR={int(kind=="projector")}',
        str(ROOT/'vendor/bcir_tensor.c'),str(ROOT/'streaming_probe.c'),'-lm','-o',str(so)]
    subprocess.run(cmd,check=True,capture_output=True,timeout=120)
    lib=ct.CDLL(str(so));fn=lib.gp_stream_forward
    fn.argtypes=[SZ,SZ,FP,FP,ct.POINTER(ct.c_double),ct.POINTER(ct.c_uint64)]
    z=rng.normal(size=(2,513,22)).astype(np.float32);pc=253 if kind=='projector' else 231
    cache=np.zeros(z.size*2,np.float32);cache[:z.size]=z.ravel();full=np.zeros((2,513,pc),np.float32)
    with Native(Recipe(kind=kind,ff=250 if kind=='projector' else 257),threads=1) as n:
        f=n.lib.gp_wedge_forward;f.argtypes=[ct.c_void_p,SZ,SZ,FP,FP];f(None,2,513,ptr(cache),ptr(full))
    for chunk in (1,7,64,513):
        sums=np.zeros((2,22),np.float64);steps=np.zeros(2,np.uint64);parts=[]
        for start in range(0,513,chunk):
            zz=np.ascontiguousarray(z[:,start:start+chunk]);p=np.empty((*zz.shape[:2],pc),np.float32)
            fn(2,zz.shape[1],ptr(zz),ptr(p),sums.ctypes.data_as(ct.POINTER(ct.c_double)),steps.ctypes.data_as(ct.POINTER(ct.c_uint64)))
            parts.append(p)
        streamed=np.concatenate(parts,1);assert np.array_equal(streamed,full)
        out.append({'kind':kind,'chunk':chunk,'steps':steps.tolist(),'max_feature_error':0.,
            'persistent_bytes_per_stream':22*8+8,'scope':'feature mixer only; excludes model weights, full decoder and training arenas'})
(ROOT/'results/streaming.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
