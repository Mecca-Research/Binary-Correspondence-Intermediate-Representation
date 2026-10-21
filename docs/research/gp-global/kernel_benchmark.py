"""Direct-prefix versus scan math, same native pair algebra; primitive only."""
from experiment import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--threads',type=int,default=1);args=p.parse_args()
rows=[];rng=np.random.default_rng(701)
for kind in ('global','global_symmetric','projector'):
    with Native(Recipe(kind=kind,ff=250 if kind=='projector' else 257),threads=args.threads) as n:
        f=n.lib.gp_wedge_forward;f.argtypes=[ct.c_void_p,SZ,SZ,FP,FP]
        g=n.lib.gp_wedge_backward;g.argtypes=[ct.c_void_p,SZ,SZ,FP,FP,FP]
        ref=n.lib.gp_prefix_reference;ref.argtypes=[SZ,SZ,FP,FP,FP,FP]
        for t in (128,1024,4096):
            pc=253 if kind=='projector' else 231
            z=np.zeros(t*22*2,np.float32);z[:t*22]=rng.normal(size=t*22)
            dp=rng.normal(size=(t,pc)).astype(np.float32);q=np.empty_like(dp);dz=np.empty_like(z)
            rq=np.empty_like(dp);rdz=np.empty(t*22,np.float32)
            def scan():f(n.pool,1,t,ptr(z),ptr(q));g(n.pool,1,t,ptr(z),ptr(dp),ptr(dz))
            def direct():ref(1,t,ptr(z),ptr(dp),ptr(rq),ptr(rdz))
            scan();direct();assert np.array_equal(q,rq)
            rel=float(np.linalg.norm(dz[:t*22]-rdz)/np.linalg.norm(rdz));assert rel<1e-5
            for rep in range(7):
                for method in (('scan','direct') if rep%2==0 else ('direct','scan')):
                    fn=scan if method=='scan' else direct
                    # Repeat short kernels enough to amortize ctypes/timer overhead.
                    count=10 if t<4096 else 3
                    start=time.perf_counter()
                    for _ in range(count):fn()
                    ms=(time.perf_counter()-start)*1000/count
                    rows.append({'kind':kind,'context':t,'method':method,'threads':args.threads,
                        'rep':rep,'forward_backward_ms':ms,'gradient_relative_l2_vs_direct':rel})
            print(json.dumps({'kind':kind,'context':t,'scan_ms':float(np.median([r['forward_backward_ms'] for r in rows if r['kind']==kind and r['context']==t and r['method']=='scan'])),
                'direct_ms':float(np.median([r['forward_backward_ms'] for r in rows if r['kind']==kind and r['context']==t and r['method']=='direct']))}),flush=True)
(ROOT/'results'/f'kernel-{args.threads}thread.json').write_text(json.dumps(rows,indent=2))
