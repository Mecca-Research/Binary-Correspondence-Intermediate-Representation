"""Port the saved full-head research overlays to merged PR818, without changing equations.

Tensor unit, GEMM dispatch, panel paths, validity checks and AdamW come from the
merged source. Geometry, compact plans, causal prefix/VJP and masked loss stay
the saved research implementation. Exact anchors refuse silent source drift.
"""
from pathlib import Path
import subprocess,json,hashlib,argparse
ROOT=Path(__file__).resolve().parent
PIN='237651126684569e5b756dcc8d1c89ed2e86d0e3'
NATIVE=(ROOT/'vendor-new/bcir_decoder_train.c').read_text()

def once(s,a,b):
    assert s.count(a)==1,(a[:100],s.count(a));return s.replace(a,b)
def block(s,a,b):return s[s.index(a):s.index(b,s.index(a))]

def workspace(global_head):
    if not global_head:return block(NATIVE,'static size_t ws_sum(', 'typedef struct layer ')
    return '''static size_t ws_sum(size_t a,size_t b){return a>SIZE_MAX-b?SIZE_MAX:a+b;}
size_t bcir_decoder_workspace_bytes(const bcir_decoder_plan *plan,uint32_t contract){
  bcir_decoder_plan p;geometry g;size_t q,c,most=0;
  if(!(contract&BCIR_DT_CONTRACT_BF16))return 0;
  if(!plan||bcir_decoder_make_plan(&plan->spec,&p))return SIZE_MAX;g=geom(&p.spec);
  {const size_t n=g.n,d=g.d,f=g.f,v=g.v,r=GP_RANK,a=GP_P;
   /* Upper bound of each actual research-head group, including the query side channel. */
   const size_t groups[][4][3]={
    {{n,r,d},{n,d,d}},{{n,d,a}},{{n,d,r}},{{n,d,d}},
    {{n,f,d},{n,f,d}},{{n,d,f}},{{n,v,d}},
    {{n,d,v},{v,d,n}},{{n,f,d},{d,f,n}},
    {{n,d,f},{n,d,f},{f,d,n},{f,d,n}},
    {{n,d,d},{d,d,n}},{{n,a,d},{d,a,n}},
    {{n,r,d},{d,r,n}},{{n,d,r},{r,d,n}}
   };
   for(q=0;q<sizeof(groups)/sizeof(groups[0]);q++){
    size_t sum=0;for(c=0;c<4&&groups[q][c][0];c++)sum=ws_sum(sum,bcir_tensor_mm_bf16_workspace(groups[q][c][0],groups[q][c][1],groups[q][c][2]));
    if(sum>most)most=sum;
   }
  }return most;
}
'''

def upgraded(s,name):
    s=once(s,'uint32_t bcir_decoder_abi_version(void) { return 3; }','uint32_t bcir_decoder_abi_version(void) { return 4; }')
    s=once(s,block(s,'static int overlap(', 'static int ids('),block(NATIVE,'static int overlap(', 'static int ids('))
    s=once(s,'typedef struct layer ',workspace(name!='attention')+'\ntypedef struct layer ')
    s=once(s,block(s,'static int gemm_contract(', '/* y = silu('),block(NATIVE,'static int gemm_contract(', '/* y = silu('))
    s=once(s,'static void swiglu_backward(', 'static int swiglu_v2(const bcir_decoder_state *s){return (s->contract&BCIR_DT_CONTRACT_EXP)!=0;}\nstatic void swiglu_backward(')
    if name=='attention':
        s=once(s,block(s,'static void attention(', '/* The forward,'),block(NATIVE,'static unsigned attention_contract(', '/* The forward,'))
        s=once(s,'bcir_tensor_attention_backward_inplace_pool(s->pool,g.b,g.t,g.h,g.kh,g.hd,a.q,a.k,a.v,a.p,\n        dc,dq,dk,dv);',
            'bcir_tensor_attention_backward_contract_pool(s->pool,attention_contract(s),g.b,g.t,g.h,g.kh,g.hd,a.q,a.k,a.v,a.p,dc,dq,dk,dv);')
    oldff='    silu_mul(s,g.nf,a.gate,a.up,a.hidden,&bad);\n    linear(s,g.n,g.f,g.d,a.hidden,p.down,y);'
    newff=block(NATIVE,'    if (panels(s,g,a.hidden,a.hidden)) {','    bcir_tensor_add_pool(s->pool,g.nd,a.x2,y);')
    s=once(s,oldff,newff.rstrip())
    # Reconstruct FF gates before the block; its down VJP writes only dh, apart.
    start=s.index('    linear_backward(s,g.n,g.f,g.d,a.hidden,p.down,cur,dh,grad.down,0.0f,wb);')
    end=s.index('    /* NDT-RMS-ACC:',start)
    old=s[start:end];remat=''
    anchor='    swiglu_backward(s,g.nf,a.gate,a.up,dh,dg,du);'
    if 'frame_ff' in name or 'layer_frame' in name:
        remat=old[old.index('\n')+1:old.index(anchor)]
        assert 'linears(s,g.n,g.d,a.n2,2,out,w,yy);' in remat or 'forward_layer' in remat
        # Panel writers use nonlocal tiled stores; compact FF's cross aliases are only
        # qualified for binary32 elementwise writers. Keep the packed path off here.
        s='#define BCIR_DT_NO_PANELS 1\n'+s
        start=s.index('    linear_backward(s,g.n,g.f,g.d,a.hidden,p.down,cur,dh,grad.down,0.0f,wb);');end=s.index('    /* NDT-RMS-ACC:',start)
    newback=block(NATIVE,'    /* the hidden activation is a panel exactly when', '    /* NDT-RMS-ACC:')
    s=s[:start]+remat+newback+s[end:]
    tail=NATIVE[NATIVE.index('typedef struct adamw_terms '):]
    # Preserve research ignored-target support; production dense next-token loss has none.
    tail=once(tail,'!ids(s,targets,total)','!target_ids(s,targets,total)')
    s=s[:s.index('typedef struct adamw_terms ')]+tail
    return s

def compile_one(source,name,old=False,sanitize=False,no_panels=False,probe=False):
    path=ROOT/'build'/f'{name}.c';path.write_text(source)
    tensor=ROOT/('vendor-old' if old else 'vendor-new')/'bcir_tensor.c'
    header=ROOT/('vendor-old' if old else 'vendor-new')
    cmd=['gcc','-std=c11','-O3','-fPIC','-shared','-ffp-contract=off','-fno-math-errno','-pthread',
         '-I',str(header),'-DGP_RANK=22','-DGP_PROJECTOR=0','-DGP_SYMMETRIC=0','-DGP_HEADS=1']
    if sanitize:cmd+=['-fsanitize=undefined','-fno-sanitize-recover=all']
    if no_panels:cmd+=['-DBCIR_DT_NO_PANELS=1']
    if probe:cmd+=['-DBCIR_DT_WORKSPACE_PROBE=1','-DBCIR_DT_PANELS_PROBE=1']
    cmd +=[str(tensor),str(path),'-lm','-o',str(path.with_suffix('.so'))]
    record=ROOT/'build'/f'{name}-build.json'
    if record.exists() and path.with_suffix('.so').exists():
        prior=json.loads(record.read_text())
        if prior['command']==cmd and prior['source_sha256']==hashlib.sha256(source.encode()).hexdigest():
            print(name,'unchanged build reused',flush=True);return
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
    if r.returncode:raise RuntimeError(r.stderr)
    (ROOT/'build'/f'{name}-build.json').write_text(json.dumps({'command':cmd,'pin':PIN if not old else 'aaa2e6c0ee0ee15b83fe4d0373b48bd893e80d6d',
        'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'stderr':r.stderr},indent=2));print(name,'built',flush=True)

if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--extras',action='store_true');args=pa.parse_args()
    names=('guard_compact','guard_frame_ff_compact','guard_layer_frame_ff','query_guard_compact','query_guard_frame_ff_compact','query_guard_layer_frame_ff','guard_hybrid_compact','attention')
    for name in names:
        source=(ROOT/'foundation'/f'{name}.c').read_text();compile_one(upgraded(source,name),name)
        if name in ('guard_compact','query_guard_compact','attention'):compile_one(source,'old_'+name,old=True)
    for name in ('query_guard_compact','query_guard_layer_frame_ff'):
        source=(ROOT/'build'/f'{name}.c').read_text();compile_one(source,name+'_ubsan',sanitize=True)
    for name in ('query_guard_compact','attention'):
        source=(ROOT/'build'/f'{name}.c').read_text();compile_one(source,name+'_probe',probe=True)
        compile_one(source,name+'_nopanels',no_panels=True)
