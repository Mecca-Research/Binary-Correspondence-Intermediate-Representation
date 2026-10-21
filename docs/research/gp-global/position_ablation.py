"""Exploratory follow-up prompted by length failure. Remove only rotary maps.

Separate frozen protocol; do not overwrite the main experiment or its hashes.
Both wedge and symmetric controls get identical training/selection budgets.
"""
from run_experiments import *
import torch
from torch import nn
from torch.nn import functional as F

OUT=ROOT/'position-ablation';OUT.mkdir(exist_ok=True)
torch.set_num_threads(1)
def reference(r,flat):
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.params=nn.ParameterList([nn.Parameter(torch.tensor(flat[o:o+n].reshape(sh))) for _,sh,o,n in layout(r)])
        def forward(self,ids):
            q=iter(self.params);emb=next(q);h=F.embedding(ids,emb)
            def rms(x,w):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+r.eps)*w
            for _ in range(r.layers):
                n1,wr,wp,wh,wg,n2,wfg,wfu,wfd=[next(q) for _ in range(9)]
                h1=rms(h,n1);z=F.linear(h1,wr) # deliberate absence of positional rotation
                p=torch_wedge(z,r.kind);g=F.linear(p,wp)
                alpha=torch.sigmoid(F.linear(h1,wh)+F.linear(g,wg))
                h=h+alpha*h1+(1-alpha)*g;h2=rms(h,n2)
                h=h+F.linear(F.silu(F.linear(h2,wfg))*F.linear(h2,wfu),wfd)
            return F.linear(rms(h,next(q)),emb)
    return Model()

def build(kind):
    source=(ROOT/'build'/f'{kind}.c').read_text()
    for call in ['bcir_tensor_rope_pool(s->pool,g.b,g.t,1,GP_RANK,sp->rope_base,a.q,0);',
                 'bcir_tensor_rope_pool(s->pool,g.b,g.t,1,GP_RANK,s->plan.spec.rope_base,dq,1);']:
        assert source.count(call)==1;source=source.replace(call,'/* Position ablation: identity map. */')
    cf=ROOT/'build'/f'{kind}_norope.c';cf.write_text(source)
    metadata=json.loads((ROOT/'build'/f'{kind}-build.json').read_text());cmd=metadata['command'][:]
    cmd[cmd.index(str(ROOT/'build'/f'{kind}.c'))]=str(cf)
    cmd[-1]=str(ROOT/'build'/f'{kind}_norope.so')
    subprocess.run(cmd,check=True,capture_output=True,timeout=120)
    metadata.update(command=cmd,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        change='Remove forward and inverse rotary maps only; same prefix geometry, parameters, FF and optimizer.')
    (OUT/f'{kind}-build.json').write_text(json.dumps(metadata,indent=2))
    return kind+'_norope'

if __name__=='__main__':
    libs={k:build(k) for k in ('global','global_symmetric')}
    protocol={'status':'exploratory follow-up after observing main length stress test',
        'change':'remove rotary position encoding from reduced global input; no other architecture/hyperparameter change',
        'recipes':['global','global_symmetric'],'seeds':list(SEEDS),'steps':400,'batch':8,'context':128,
        'validation':'same fixed 512 examples; lowest CE at 100-step intervals',
        'test':'same primary 2048 examples and length-stress 128 examples per length; reused benchmark samples, not a fresh confirmatory holdout',
        'hypothesis':'Position-specific phase interactions may account for recall length failure and the wedge/symmetric optimization gap.',
        'optimizer':'identical to main experiment','additional_tuning':'none',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'native_source_sha256':{k:hashlib.sha256((ROOT/'build'/f'{lib}.c').read_bytes()).hexdigest() for k,lib in libs.items()}}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    validation=[];rng=np.random.default_rng(980)
    for kind,lib in libs.items():
        r=Recipe(kind=kind,width=32,layers=2,heads=2,ff=48,batch=2,context=9)
        x=rng.integers(0,260,(2,9),dtype=np.uint32);y=rng.integers(0,260,(2,9),dtype=np.uint32);y[:,:-2]=IGNORE
        with Native(r,threads=1,library_name=lib) as n:
            c=n.forward(x).copy();loss=n.backward(x,y)
            m=reference(r,n.buffers['weights']);p=m(torch.tensor(x.astype(np.int64)))
            ty=torch.tensor(y.astype(np.int64));mask=ty!=int(IGNORE)
            lt=F.cross_entropy(p[mask],ty[mask]);lt.backward()
            tg=np.concatenate([pp.grad.numpy().ravel() for pp in m.params])
            row={'kind':kind,'max_logit_error':float(np.max(np.abs(c-p.detach().numpy()))),
                'max_gradient_error':float(np.max(np.abs(n.buffers['gradients']-tg))),
                'loss_error':abs(loss-lt.item())}
            assert row['max_logit_error']<1e-5 and row['max_gradient_error']<1e-4 and row['loss_error']<1e-5
        xx=np.stack([x]*4);yy=np.stack([y]*4);digests=[]
        for threads in (1,4):
            with Native(r,threads=threads,library_name=lib) as n:
                n.train(xx[:2],yy[:2],[.0005]*2);state=n.snapshot();n.train(xx[2:],yy[2:],[.0005]*2)
                digest=n.digest();n.restore(state);n.train(xx[2:],yy[2:],[.0005]*2);assert n.digest()==digest;digests.append(digest)
        assert digests[0]==digests[1];row['thread_identity_resume']=digests[0];validation.append(row)
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2))
    length_rows=[]
    for si,seed in enumerate(SEEDS):
        kinds=('global','global_symmetric') if si%2==0 else ('global_symmetric','global')
        for kind in kinds:
            r=Recipe(kind=kind,ff=257,batch=8,context=128);lib=libs[kind]
            train=recall_batches(400,8,128,seed+200);val=recall_batches(64,8,128,103)
            history=[];best=None;bestce=float('inf');events=[]
            with Native(r,seed=seed,threads=4,library_name=lib) as n:
                for off in range(0,400,100):
                    _,ev=n.train(train[0][off:off+100],train[1][off:off+100],rates(400)[off:off+100]);events+=ev
                    result=evaluate(n,val);history.append({'step':off+100,'validation':result})
                    if result['ce_nats']<bestce:bestce=result['ce_nats'];best=n.snapshot();beststep=off+100
                    print(json.dumps({'kind':lib,'seed':seed,'step':off+100,'validation':result}),flush=True)
                n.restore(best);test=evaluate(n,recall_batches(256,8,128,20261021));digest=n.digest()
                ck=OUT/f'{lib}-{seed}.npz';np.savez_compressed(ck,**best,recipe=json.dumps(asdict(r)),seed=seed,library_name=lib)
            d={'kind':kind,'library_name':lib,'seed':seed,'recipe':asdict(r),'selected_step':beststep,
               'history':history,'events':events,'test':test,'state_sha256':digest,
               'checkpoint':str(ck.relative_to(ROOT)),'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),
               'input_sha256':hashlib.sha256(train[0].tobytes()+train[1].tobytes()).hexdigest()}
            main=json.loads((ROOT/'results'/f'first_token_recall-{kind}-{seed}.json').read_text());assert d['input_sha256']==main['input_sha256']
            (OUT/f'{lib}-{seed}.json').write_text(json.dumps(d,indent=2))
            for context in (64,128,256,512,1024):
                rr=replace(r,context=context);xy=recall_batches(16,8,context,881020+context)
                with Native(rr,threads=4,library_name=lib) as n:
                    n.restore(best);result=evaluate(n,xy);assert n.digest()==digest
                length_rows.append({'kind':kind,'library_name':lib,'seed':seed,'context':context,'test':result})
            print(json.dumps({'kind':lib,'seed':seed,'test':test,'length_accuracy':{v['context']:v['test']['accuracy'] for v in length_rows if v['seed']==seed and v['kind']==kind}}),flush=True)
    (OUT/'recall-length.json').write_text(json.dumps(length_rows,indent=2))
