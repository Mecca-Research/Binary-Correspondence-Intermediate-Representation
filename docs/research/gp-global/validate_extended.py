"""Independent full-model and primitive validation for the prefix geometry experiment."""
from experiment import *
from build_native import build
import torch
from torch.nn import functional as F

torch.set_num_threads(1)
KINDS=('global','global_symmetric','projector')
rng=np.random.default_rng(9041);out={}
(ROOT/'results').mkdir(exist_ok=True)

for kind in KINDS:
    rows=[]
    for masked_loss in (False,True):
        r=Recipe(kind=kind,width=16,layers=2,heads=2,ff=25,batch=2,context=9)
        x=rng.integers(0,260,(2,9),dtype=np.uint32);y=np.roll(x,-1,1)
        if masked_loss:y[:,:-2]=IGNORE
        with Native(r,threads=1) as n:
            c=n.forward(x).copy();loss=n.backward(x,y)
            m=torch_model(r,n.buffers['weights']);p=m(torch.from_numpy(x.astype(np.int64)))
            ty=torch.from_numpy(y.astype(np.int64));mask=ty!=int(IGNORE)
            lt=F.cross_entropy(p[mask],ty[mask]);lt.backward()
            gr=np.concatenate([p.grad.numpy().ravel() for p in m.params]);ng=n.buffers['gradients']
            row={'masked':masked_loss,'max_logit_error':float(np.max(np.abs(c-p.detach().numpy()))),
                 'loss_error':abs(loss-lt.item()),'max_gradient_error':float(np.max(np.abs(ng-gr))),
                 'gradient_relative_l2':float(np.linalg.norm(ng-gr)/np.linalg.norm(gr))}
            assert row['max_logit_error']<1e-5 and row['loss_error']<1e-5,row
            assert row['max_gradient_error']<1e-4 and row['gradient_relative_l2']<2e-4,row
            before=n.digest();empty=np.full_like(y,IGNORE);vv=ct.c_double()
            rc=n.lib.bcir_decoder_loss_backward(ct.byref(n.state),ptr(x,UP),ptr(empty,UP),x.size,0,ct.byref(vv))
            assert rc==1 and before==n.digest()
            rows.append(row)
    out[kind]=rows

ident=[];causal=[];primitives=[]
for kind in KINDS:
    r=Recipe(kind=kind,width=32,layers=2,heads=2,ff=48,batch=2,context=17)
    x=rng.integers(0,260,(5,2,17),dtype=np.uint32);y=rng.integers(0,260,x.shape,dtype=np.uint32)
    states=[]
    for threads in (1,4):
        with Native(r,threads=threads) as n:
            n.train(x[:3],y[:3],[.0005]*3);s=n.snapshot()
            n.train(x[3:],y[3:],[.0005]*2);end=n.digest()
            n.restore(s);n.train(x[3:],y[3:],[.0005]*2);assert end==n.digest()
            states.append(end)
    assert states[0]==states[1]
    ident.append({'kind':kind,'threads':[1,4],'full_state_sha256':states[0],'resume':'exact'})
    r=Recipe(kind=kind,width=32,layers=3,heads=2,ff=48,batch=2,context=128)
    x=rng.integers(0,260,(2,128),dtype=np.uint32)
    with Native(r,threads=1) as n:
        a=n.forward(x).copy();xx=x.copy();xx[:,8:]=(xx[:,8:]+61)%260
        prefix=float(np.max(np.abs(a[:,:8]-n.forward(xx)[:,:8])))
        xx=x.copy();xx[:,1]=(xx[:,1]+97)%260
        far=float(np.max(np.abs(a[:,-1]-n.forward(xx)[:,-1])))
        assert prefix==0 and far>1e-8
        causal.append({'kind':kind,'future_perturbation_prefix_change':prefix,'position1_to127_change':far})
        f=n.lib.gp_wedge_forward;f.argtypes=[ct.c_void_p,SZ,SZ,FP,FP]
        g=n.lib.gp_wedge_backward;g.argtypes=[ct.c_void_p,SZ,SZ,FP,FP,FP]
        ref=n.lib.gp_prefix_reference;ref.argtypes=[SZ,SZ,FP,FP,FP,FP]
        for case in ('random','zero','collinear','near_collinear'):
            z=rng.normal(size=(2,9,22)).astype(np.float32)
            if case=='zero':z[:]=0
            if case in ('collinear','near_collinear'):
                z[:]=z[:,:1]
                if case=='near_collinear':z+=rng.normal(size=z.shape).astype(np.float32)*1e-3
            pc=253 if kind=='projector' else 231
            dp=rng.normal(size=(2,9,pc)).astype(np.float32)
            cache=np.zeros(z.size*2,np.float32);cache[:z.size]=z.ravel()
            p=np.zeros_like(dp);dz=np.zeros_like(cache)
            f(None,2,9,ptr(cache),ptr(p));g(None,2,9,ptr(cache),ptr(dp),ptr(dz))
            tz=torch.tensor(z,requires_grad=True);tp=torch_wedge(tz,kind)
            (tp*torch.tensor(dp)).sum().backward();tg=tz.grad.numpy()
            ng=dz[:z.size].reshape(z.shape)
            rel=float(np.linalg.norm((ng-tg).astype(np.float64))/max(np.linalg.norm(tg.astype(np.float64)),1e-30))
            ferr=float(np.max(np.abs(p-tp.detach().numpy())))
            assert np.isfinite(ng).all() and np.isfinite(p).all()
            # Clamp degeneracies can magnify FP32 primitive ordering; record absolute too.
            assert rel<.002 or np.max(np.abs(ng-tg))<1e-5,(kind,case,rel)
            assert ferr<1e-4,(kind,case,ferr)
            rp=np.empty_like(p);rd=np.empty_like(z)
            ref(2,9,ptr(z),ptr(dp),ptr(rp),ptr(rd))
            assert np.array_equal(p,rp)
            refrel=float(np.linalg.norm((ng-rd).astype(np.float64))/max(np.linalg.norm(rd.astype(np.float64)),1e-30))
            assert refrel<1e-5 or np.max(np.abs(ng-rd))<1e-5
            fd=[]
            if case=='random':
                for ix in rng.choice(z.size,24,replace=False):
                    zz=cache.copy();eps=.002;zz[ix]+=eps;pa=np.empty_like(p);f(None,2,9,ptr(zz),ptr(pa))
                    zz[ix]-=2*eps;pb=np.empty_like(p);f(None,2,9,ptr(zz),ptr(pb))
                    num=float(((pa-pb).astype(np.float64)*dp).sum()/(2*eps))
                    fd.append(abs(num-float(ng.ravel()[ix])))
                assert max(fd)<.005
            primitives.append({'kind':kind,'case':case,'forward_max_error':ferr,'gradient_relative_l2':rel,
                'gradient_max_error':float(np.max(np.abs(ng-tg))),
                'scan_vs_direct_gradient_relative_l2':refrel,'finite_difference_samples':len(fd),
                'max_finite_difference_error':max(fd) if fd else None})

# Invariance and decomposability checked independently in FP64; ridge error is explicit.
u=rng.normal(size=22);v=rng.normal(size=22);basis=np.stack([u,v],1)
i,j=np.triu_indices(22,1)
def wedge(a):
    p=a[:,0][i]*a[:,1][j]-a[:,0][j]*a[:,1][i];return p/np.linalg.norm(p)
def projector(a):return a@np.linalg.inv(a.T@a+np.eye(2)*1e-4)@a.T
rotation=np.array([[.6,-.8],[.8,.6]]);general=np.array([[2.,.3],[-.4,.7]])
p=wedge(basis);pp=projector(basis);quads=[]
import itertools
for a,b,c,d in itertools.combinations(range(22),4):
    mat=np.zeros((22,22));mat[i,j]=p;mat-=mat.T
    quads.append(abs(mat[a,b]*mat[c,d]-mat[a,c]*mat[b,d]+mat[a,d]*mat[b,c]))
out['geometry']={'wedge_max_plucker_residual':max(quads),
    'wedge_positive_basis_change_max_error':float(np.max(np.abs(p-wedge(basis@general)))),
    'wedge_reflection_distance':float(np.linalg.norm(p-wedge(basis[:,::-1]))),
    'ridge_projector_orthogonal_basis_change_max_error':float(np.max(np.abs(pp-projector(basis@rotation)))),
    'ridge_projector_general_basis_change_max_error':float(np.max(np.abs(pp-projector(basis@general)))),
    'ridge_projector_idempotence_max_error':float(np.max(np.abs(pp@pp-pp))),
    'ridge':1e-4,'note':'Ridge projector is smooth and approximately a rank-two orthogonal projector; it is not exactly idempotent or GL(2)-invariant.'}
out['identity']=ident;out['causality']=causal;out['primitives']=primitives
for kind in ('global','projector'):
    build(REPO,kind+'_ubsan',sanitize=True)
    with Native(Recipe(kind=kind,width=32,layers=2,heads=2,ff=48,batch=2,context=17),threads=2,library_name=kind+'_ubsan') as n:
        n.train(x[:2,None] if False else rng.integers(0,260,(4,2,17),dtype=np.uint32),
                rng.integers(0,260,(4,2,17),dtype=np.uint32),[.0005]*4)
out['ubsan']='pass: four complete training steps each for global wedge and projector, two threads'
(ROOT/'results'/'validation.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2),flush=True)
