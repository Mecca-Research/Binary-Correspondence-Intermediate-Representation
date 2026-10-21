"""Exploratory one-plane local/global blend after the main comparison.

No extra parameters: wedge(current, .5*global_prefix + .5*previous).
Common current factor preserves decomposability. New test samples are shared
with the already-selected global baseline, without retraining that baseline.
"""
from candidates import *
from build_next import compile_source,once
import train_next as train
from run_experiments import bytes_data,evaluate
import shutil,importlib,torch
from torch.nn import functional as F
OUT=ROOT/'hybrid';OUT.mkdir(exist_ok=True);(OUT/'results').mkdir(exist_ok=True)
torch.set_num_threads(1)

if __name__=='__main__':
    source=(ROOT/'build/fast.c').read_text()
    source=once(source,'prefix[t*GP_RANK+k]=(float)(sum[k]*scale);',
        'prefix[t*GP_RANK+k]=(float)(0.5*sum[k]*scale+(t?0.5*(double)z[(t-1)*GP_RANK+k]:0));')
    source=once(source,'dz[t*GP_RANK+k]+=(float)carry[k];',
        'dz[t*GP_RANK+k]+=(float)carry[k];\n      if(t+1<j->time)dz[t*GP_RANK+k]+=0.5f*dv[(t+1)*GP_RANK+k];')
    source=once(source,'carry[k]+=(double)dv[t*GP_RANK+k]*scale;',
        'carry[k]+=0.5*(double)dv[t*GP_RANK+k]*scale;')
    compile_source(source,'hybrid')
    ref=(ROOT/'oracle.py').read_text()
    ref=once(ref,'v=(prefix*scale[None,:,None]).float()',
        'local=torch.cat([torch.zeros_like(z[:,:1]),z[:,:-1]],1).double()\n                    v=(0.5*prefix*scale[None,:,None]+0.5*local).float()')
    (ROOT/'oracle_hybrid.py').write_text(ref);oracle=importlib.import_module('oracle_hybrid')
    r=Recipe(kind='global',width=32,layers=2,heads=2,ff=48,batch=2,context=11)
    x=(np.arange(22,dtype=np.uint32).reshape(2,11)*7)%260;y=np.roll(x,-1,1);y[:,:-2]=IGNORE
    with Native(r,threads=1,library_name='hybrid') as n:
        c=n.forward(x).copy();loss=n.backward(x,y);m=oracle.torch_model_next(r,n.buffers['weights']);p=m(torch.tensor(x.astype(np.int64)))
        ty=torch.tensor(y.astype(np.int64));mask=ty!=int(IGNORE);lt=F.cross_entropy(p[mask],ty[mask]);lt.backward()
        gr=np.concatenate([pp.grad.numpy().ravel() for pp in m.params]);ng=n.buffers['gradients']
        val={'logit_max_error':float(np.max(np.abs(c-p.detach().numpy()))),'gradient_max_error':float(np.max(np.abs(gr-ng))),
            'gradient_relative_l2':float(np.linalg.norm(gr-ng)/np.linalg.norm(gr)),'loss_error':abs(loss-lt.item())}
        assert val['logit_max_error']<1e-5 and val['gradient_max_error']<1e-4 and val['gradient_relative_l2']<2e-4,val
        xx=x.copy();xx[:,5:]=(xx[:,5:]+19)%260;assert np.array_equal(c[:,:5],n.forward(xx)[:,:5])
    states=[]
    for th in (1,4):
        with Native(r,threads=th,library_name='hybrid') as n:
            n.train(np.stack([x]*4),np.stack([y]*4),[.0005]*4);states.append(n.digest())
    assert states[0]==states[1];val['thread_identity']=states[0]
    (OUT/'validation.json').write_text(json.dumps(val,indent=2))
    protocol={'stage':'exploratory follow-up after main global-only language comparison','blend':.5,
        'equation':'v_t=0.5*sum_{j<t} z_j/sqrt(t)+0.5*z_(t-1), then a single normalized wedge with current z_t',
        'train':'same batches, seeds, optimizer, fixed budgets and validation selection as main byte_lm/recall_fixed',
        'test_seeds':{'byte_lm':20261102,'recall_fixed':20261103},'hyperparameter_search':'none; one fixed equal blend',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'native_source_sha256':hashlib.sha256(source.encode()).hexdigest()}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    data,_=bytes_data();train.ROOT=OUT;train.ARCH={'hybrid':(Recipe(kind='global',ff=257),'hybrid')}
    train.TEST_SEEDS.update(protocol['test_seeds'])
    for task in ('byte_lm','recall_fixed'):
        for seed in SEEDS:
            train.run(task,'hybrid',seed,data)
            src=ROOT/'results'/f'{task}-global-{seed}.json';d=json.loads(src.read_text());ck=ROOT/d['checkpoint']
            state=dict(np.load(ck,allow_pickle=False));rr=Recipe(**json.loads(str(state['recipe'])))
            xy=train.data_for(task,64 if task=='byte_lm' else 256,rr.batch,rr.context,protocol['test_seeds'][task],data,'test')
            with Native(rr,threads=4,library_name='fast') as n:n.restore(state);result=evaluate(n,xy)
            d['original_test']=d['test'];d['test']=result;d['reused_baseline_training']=True
            # Keep the checkpoint reference relative to the experiment root.
            (OUT/'results'/src.name).write_text(json.dumps(d,indent=2));print(json.dumps({'baseline':task,'seed':seed,'test':result}),flush=True)
