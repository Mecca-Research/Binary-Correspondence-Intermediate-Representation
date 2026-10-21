from candidates import *
from oracle import torch_model_next
from build_next import compile_source
import torch
from torch.nn import functional as F
torch.set_num_threads(1);rng=np.random.default_rng(1904);out={'exact':[],'architectures':[]}
for context in (9,64):
    r=Recipe(kind='global',width=32,layers=3,ff=48,batch=2,context=context)
    x=rng.integers(0,260,(5,2,context),dtype=np.uint32);y=rng.integers(0,260,x.shape,dtype=np.uint32);y[:,:,:-2]=IGNORE
    reference=None
    for lib in EXACT:
        for threads in (1,4):
            with Native(r,threads=threads,library_name=lib) as n:
                logits=n.forward(x[0]).copy();loss=n.backward(x[0],y[0]);grads=n.buffers['gradients'].copy()
                n.train(x[:3],y[:3],[.0005]*3);state=n.snapshot();n.train(x[3:],y[3:],[.0005]*2);digest=n.digest()
                n.restore(state);n.train(x[3:],y[3:],[.0005]*2);assert n.digest()==digest
                if reference is None:reference=(logits,loss,grads,digest)
                assert np.array_equal(logits,reference[0]),('logits',lib,threads,context,float(np.max(np.abs(logits-reference[0]))))
                assert loss==reference[1] and np.array_equal(grads,reference[2]),('grads',lib,threads,context)
                assert digest==reference[3],('state',lib,threads,context)
                out['exact'].append({'library':lib,'context':context,'threads':threads,'state_sha256':digest,
                    'transient_bytes':int((n.plan.activations+n.plan.scratch)*4),'logits_gradients_state':'bitwise identical'})
for name,(r0,lib) in ARCH.items():
    r=replace(r0,width=32,layers=2,heads=2,ff=48,batch=2,context=11)
    x=rng.integers(0,260,(2,11),dtype=np.uint32);y=rng.integers(0,260,x.shape,dtype=np.uint32);y[:,:-2]=IGNORE
    with Native(r,threads=1,library_name=lib) as n:
        c=n.forward(x).copy();loss=n.backward(x,y)
        m=torch_model_next(r,n.buffers['weights']);p=m(torch.tensor(x.astype(np.int64)))
        ty=torch.tensor(y.astype(np.int64));mask=ty!=int(IGNORE);lt=F.cross_entropy(p[mask],ty[mask]);lt.backward()
        gr=np.concatenate([pp.grad.numpy().ravel() for pp in m.params]);ng=n.buffers['gradients']
        row={'name':name,'logit_max_error':float(np.max(np.abs(c-p.detach().numpy()))),'loss_error':abs(loss-lt.item()),
             'gradient_max_error':float(np.max(np.abs(ng-gr))),'gradient_relative_l2':float(np.linalg.norm(ng-gr)/np.linalg.norm(gr))}
        assert row['logit_max_error']<1e-5 and row['loss_error']<1e-5,row
        assert row['gradient_max_error']<1e-4 and row['gradient_relative_l2']<2e-4,row
        changed=x.copy();changed[:,5:]=(changed[:,5:]+49)%260
        assert np.array_equal(c[:,:5],n.forward(changed)[:,:5])
        before=n.digest();empty=np.full_like(y,IGNORE);value=ct.c_double()
        rc=n.lib.bcir_decoder_loss_backward(ct.byref(n.state),ptr(x,UP),ptr(empty,UP),x.size,0,ct.byref(value))
        assert rc==1 and n.digest()==before
        out['architectures'].append(row)
for lib in ('frame_ff_remat','layer_remat','lowrank32','multi12'):
    source=(ROOT/'build'/f'{lib}.c').read_text()
    compile_source(source,lib+'_ubsan',rank=24 if lib=='multi12' else 22,heads=2 if lib=='multi12' else 1,sanitize=True)
    r=replace(ARCH['multi12' if lib=='multi12' else 'lowrank32' if lib=='lowrank32' else 'global'][0],
        width=32,layers=2,ff=48,batch=2,context=11)
    with Native(r,threads=2,library_name=lib+'_ubsan') as n:
        xx=rng.integers(0,260,(4,2,11),dtype=np.uint32);yy=rng.integers(0,260,xx.shape,dtype=np.uint32)
        n.train(xx,yy,[.0005]*4)
out['ubsan']='pass: frame/FF rematerialization, layer rematerialization, low-rank and multi-subspace models; 4 complete steps each'
(ROOT/'results/validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
