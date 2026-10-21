"""Grammar-known C lookup control; no neural weights or training claim."""
from tasks import *
import subprocess

if __name__=='__main__':
    cmd=['gcc','-std=c11','-O3','-fPIC','-shared',str(ROOT/'registry_reference.c'),'-o',str(ROOT/'build/registry_reference.so')]
    subprocess.run(cmd,check=True,capture_output=True)
    lib=ct.CDLL(str(ROOT/'build/registry_reference.so'));lib.registry_predict.argtypes=[UP,SZ,SZ,UP]
    lib.registry_state_bytes.restype=SZ;rows=[]
    def check(records,name):
        total=correct=0
        for r in records:
            x,y=trim(r);out=np.empty(len(x),np.uint32)
            assert lib.registry_predict(ptr(x,UP),len(x),x.shape[1],ptr(out,UP))==0
            target=y[:,-1];total+=len(x);correct+=int((target==out).sum());assert np.array_equal(target,out)
        rows.append({'scenario':name,'examples':total,'correct':correct,'accuracy':correct/total,'logical_input_sha256':input_digest(records)})
    for t in (64,128,256,512,1024,2048):
        check(eval_records('kv',t,32 if t==2048 else 128,4,20261110+t,pairs=4),f'4-pairs-T{t}')
    for t in (128,512):
        r=eval_records('kv',t,128,4,20261210+t,pairs=4)
        check(r,f'counterfactual-original-T{t}');check([swapped(x,'kv') for x in r],f'counterfactual-swapped-T{t}')
    for pairs in (1,2,8):check(eval_records('kv',256,128,4,20261310+pairs,pairs=pairs),f'{pairs}-pairs-T256')
    # Missing-query refusal is part of the explicitly declared parser contract.
    bad=np.zeros((1,64),np.uint32);out=np.empty(1,np.uint32)
    assert lib.registry_predict(ptr(bad,UP),1,64,ptr(out,UP))!=0 and out[0]==IGNORE
    cmd2=['gcc','-std=c11','-O1','-fsanitize=undefined','-fno-sanitize-recover=all','-fPIC','-shared',str(ROOT/'registry_reference.c'),'-o',str(ROOT/'build/registry_reference_ubsan.so')]
    subprocess.run(cmd2,check=True,capture_output=True)
    san=ct.CDLL(str(ROOT/'build/registry_reference_ubsan.so'));san.registry_predict.argtypes=[UP,SZ,SZ,UP]
    rec=eval_records('kv',256,128,4,20261314,pairs=4)
    for r in rec:
        x,y=trim(r);out=np.empty(len(x),np.uint32);assert san.registry_predict(ptr(x,UP),len(x),256,ptr(out,UP))==0
        assert np.array_equal(out,y[:,-1])
    result={'rows':rows,'state_bytes_per_stream':int(lib.registry_state_bytes()),'ubsan':'pass','missing-query-refusal':'pass',
        'source_sha256':hashlib.sha256((ROOT/'registry_reference.c').read_bytes()).hexdigest(),
        'scope':'Exact grammar-known parser/lookup reference with no learned parameters. Confirms dataset labels and demonstrates structured-memory feasibility; it does not learn the grammar or constitute a general neural-model comparison.'}
    (ROOT/'results/registry-reference.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
