"""Paired ordered/associative recall and byte-language curricula.

One recall query per example makes equal-microbatch gradient averaging exact
for the declared loss weights. A batch has one physical length after trimming.
"""
from experiment import *
from run_experiments import bytes_data,lm_batches

LENGTHS=(64,128,256)
def balanced_lengths(steps,seed,lengths=LENGTHS):
    assert steps%len(lengths)==0
    a=np.tile(np.array(lengths,np.int64),steps//len(lengths));np.random.default_rng(seed).shuffle(a);return a

def ordered_records(steps,batch,seed,fixed=None,repeat_probability=0.):
    rng=np.random.default_rng(seed);lengths=np.full(steps,fixed) if fixed else balanced_lengths(steps,seed+1)
    rows=[]
    for t in lengths:
        t=int(t);x=rng.integers(0,16,(batch,256 if t<=256 else t),dtype=np.uint32)
        if repeat_probability:
            mask=rng.random(x.shape)<repeat_probability;x[mask]=np.broadcast_to(x[:,1,None],x.shape)[mask]
        x[:,0]=256;x[:,t-1]=257;y=np.full_like(x,IGNORE);y[:,t-1]=x[:,1]
        rows.append({'x':x,'y':y,'length':t})
    return rows

def kv_records(steps,batch,seed,fixed=None,pairs=None):
    rng=np.random.default_rng(seed);lengths=np.full(steps,fixed) if fixed else balanced_lengths(steps,seed+1)
    rows=[]
    for step,t in enumerate(lengths):
        t=int(t);n=pairs if pairs is not None else 2 if step<300 else 4
        x=rng.integers(64,96,(batch,t),dtype=np.uint32);x[:,0]=256;y=np.full_like(x,IGNORE)
        loc=[];query=[]
        slots=np.arange(3,t-5,4);assert n<=len(slots)
        for b in range(batch):
            pos=rng.choice(slots,n,replace=False);keys=rng.choice(np.arange(32,48),n,replace=False).astype(np.uint32)
            values=rng.choice(np.arange(16),n,replace=False).astype(np.uint32);q=int(rng.integers(n))
            x[b,pos]=258;x[b,pos+1]=keys;x[b,pos+2]=values
            x[b,t-2]=257;x[b,t-1]=keys[q];y[b,t-1]=values[q]
            loc.append((pos+2).tolist());query.append(q)
        rows.append({'x':x,'y':y,'length':t,'value_positions':loc,'query_pair':query,'pairs':n})
    return rows

def lm_records(steps,seed,data,split='train'):
    lengths=balanced_lengths(steps,seed+1,(32,64,128));rng=np.random.default_rng(seed);rows=[]
    for t in lengths:
        t=int(t);b=256//t;starts=rng.integers(0,len(data[split])-t-1,b)
        a=data[split][starts[:,None]+np.arange(t+1)]
        rows.append({'x':np.ascontiguousarray(a[:,:-1],np.uint32),'y':np.ascontiguousarray(a[:,1:],np.uint32),'length':t})
    return rows

def trim(record,padded=False):
    t=record['x'].shape[1] if padded else record['length']
    return np.ascontiguousarray(record['x'][:,:t]),np.ascontiguousarray(record['y'][:,:t])

def input_digest(records):
    h=hashlib.sha256()
    for r in records:
        x,y=trim(r);h.update(json.dumps(list(x.shape)).encode());h.update(x.tobytes());h.update(y.tobytes())
    return h.hexdigest()

def swapped(record,task):
    out={**record,'x':record['x'].copy(),'y':record['y'].copy()};t=record['length'];x=out['x'];y=out['y']
    for b in range(len(x)):
        if task=='ordered':
            choices=np.flatnonzero((x[b,2:t-1]!=x[b,1]) & (x[b,2:t-1]<16))+2
            assert len(choices);a,c=1,int(choices[0])
            x[b,a],x[b,c]=x[b,c],x[b,a];y[b,t-1]=x[b,1]
        else:
            q=record['query_pair'][b];other=(q+1)%record['pairs']
            a=record['value_positions'][b][q];c=record['value_positions'][b][other]
            x[b,a],x[b,c]=x[b,c],x[b,a];y[b,t-1]=x[b,a]
        assert y[b,t-1]!=record['y'][b,t-1]
        assert np.array_equal(np.sort(x[b,:t]),np.sort(record['x'][b,:t]))
    return out

def eval_records(task,length,n,batch,seed,data=None,pairs=4,repeat_probability=0.,split='test'):
    assert n%batch==0
    if task=='ordered':return ordered_records(n//batch,batch,seed,fixed=length,repeat_probability=repeat_probability)
    if task=='kv':return kv_records(n//batch,batch,seed,fixed=length,pairs=pairs)
    x,y=lm_batches(data[split],n//batch,batch,length,seed)
    return [{'x':a,'y':b,'length':length} for a,b in zip(x,y)]

def evaluate_records(runner,records,task):
    ce=correct=count=invalid=0;predictions=[];labels=[]
    for row in records:
        x,y=trim(row);p=runner.infer(x);mask=y!=IGNORE;l=p[mask].astype(np.float64);yy=y[mask].astype(np.int64)
        m=l.max(1);ce+=float((np.log(np.exp(l-m[:,None]).sum(1))+m-l[np.arange(len(yy)),yy]).sum())
        pp=l.argmax(1);correct+=int((pp==yy).sum());count+=len(yy)
        if task!='lm':invalid+=int((pp>=16).sum())
        predictions.extend(pp.tolist());labels.extend(yy.tolist())
    return {'ce_nats':ce/count,'accuracy':correct/count,'scored_tokens':count,'invalid_output_rate':invalid/count},predictions,labels
