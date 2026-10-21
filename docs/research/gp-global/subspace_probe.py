"""Diagnostic retrieval under nuisance basis changes, not an LM benchmark.

Features are computed by native C. NumPy only evaluates retrieval distances.
An orientation-invariant Plucker similarity is an essential cheap control.
"""
from experiment import *
rng=np.random.default_rng(551);r=22;classes=32;queries=3200
planes=np.stack([np.linalg.qr(rng.normal(size=(r,2)))[0] for _ in range(classes)]).astype(np.float32)
labels=np.arange(queries)%classes;rng.shuffle(labels)
rows=[]
for noise in (0.,.02,.1):
    bases=[]
    for label in labels:
        # Random rotation/reflection and bounded singular values [0.5,2].
        left=np.linalg.qr(rng.normal(size=(2,2)))[0];right=np.linalg.qr(rng.normal(size=(2,2)))[0]
        # Explicit fair reflection coin: standard QR's determinant convention is biased.
        left[:,0]*=rng.choice([-1,1])
        transform=left@np.diag(rng.uniform(.5,2,2))@right.T
        bases.append(planes[label]@transform+rng.normal(size=(r,2))*noise)
    bases=np.asarray(bases,np.float32)
    allbases=np.concatenate([planes,bases])
    z=np.stack([allbases[:,:,1],allbases[:,:,0]],1).copy()
    for kind in ('global','global_symmetric','projector'):
        pc=253 if kind=='projector' else 231
        cache=np.zeros(z.size*2,np.float32);cache[:z.size]=z.ravel()
        p=np.empty((len(z),2,pc),np.float32)
        with Native(Recipe(kind=kind,ff=250 if kind=='projector' else 257),threads=1) as n:
            f=n.lib.gp_wedge_forward;f.argtypes=[ct.c_void_p,SZ,SZ,FP,FP]
            f(None,len(z),2,ptr(cache),ptr(p))
        features=p[:,1].astype(np.float64)
        if kind=='projector':
            i,j=np.triu_indices(r,0);features[:,i!=j]*=math.sqrt(2)
        features/=np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-30)
        scores=features[classes:]@features[:classes].T
        rows.append({'kind':kind,'similarity':'signed inner product','noise_std':noise,
            'accuracy':float((scores.argmax(1)==labels).mean()),'queries':queries,'classes':classes})
        if kind=='global':
            rows.append({'kind':kind,'similarity':'absolute inner product (orientation-invariant)',
                'noise_std':noise,'accuracy':float((np.abs(scores).argmax(1)==labels).mean()),'queries':queries,'classes':classes})
(ROOT/'results/subspace.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
