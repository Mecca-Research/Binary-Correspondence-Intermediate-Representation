"""Generate isolated native C variants from the retained, pinned experiment.

No repository source is modified. Exact execution variants preserve all model
parameters. Architectural restrictions are separately named and trained.
"""
from pathlib import Path
import json,hashlib,subprocess
ROOT=Path(__file__).resolve().parent
PIN='aaa2e6c0ee0ee15b83fe4d0373b48bd893e80d6d'
def once(s,a,b):
    assert s.count(a)==1,(a[:100],s.count(a));return s.replace(a,b)

GATE=r'''
typedef struct wedge_gate_job {size_t n;const float *h,*g,*cur;float *a,*x,*dc,*dv,*dn;} wedge_gate_job;
static void wedge_gate_fwd_task(void *ctx,size_t task) {
  wedge_gate_job *j=ctx;size_t i,end=(task+1)*512;if(end>j->n)end=j->n;
  for(i=task*512;i<end;i++) {
    float e=bcir_tensor_expf(-(j->a[i]+j->x[i]));
    j->a[i]=1.0f/(1.0f+e);j->x[i]=j->a[i]*j->h[i]+(1.0f-j->a[i])*j->g[i];
  }
}
static void wedge_gate_bwd_task(void *ctx,size_t task) {
  wedge_gate_job *j=ctx;size_t i,end=(task+1)*512;if(end>j->n)end=j->n;
  for(i=task*512;i<end;i++) {float alpha=j->a[i];
    j->dc[i]=j->cur[i]*(j->h[i]-j->g[i])*alpha*(1.0f-alpha);
    j->dv[i]=j->cur[i]*(1.0f-alpha);j->dn[i]=j->cur[i]*alpha;
  }
}
'''

def reuse(s):
    s=once(s,'!term(&p.scratch,5,nd)','!term(&p.scratch,4,nd)')
    s=once(s,'!add(&p.scratch,nv)||','/* logits consumed in-place by loss */')
    s=once(s,'cur=s->scratch; tmp=cur+g.nd; dn=tmp+g.nd;', 'cur=s->scratch; tmp=cur; dn=cur+g.nd;')
    s=once(s,'dh=du+g.nf; dl=dh+g.nf;', 'dh=du+g.nf; dl=logits;')
    return s

def grouped_gate(s):
    marker='/* The forward, and whether every activation'
    s=once(s,marker,GATE+'\n'+marker)
    s=once(s,'    linear(s,g.n,g.d,GP_RANK,a.n1,p.q,a.q);', '''    {
      size_t out[2]={GP_RANK,g.d};const float *w[2]={p.q,p.v};float *yy[2]={a.q,a.p};
      linears(s,g.n,g.d,a.n1,2,out,w,yy);
    }''')
    s=once(s,'    linear(s,g.n,g.d,g.d,a.n1,p.v,a.p);','    /* gate_h grouped with reduce */')
    start=s.index('    for(i=0;i<g.nd;i++) {',s.index('static int forward('));end=s.index('    if(!finite_pool',start)
    s=s[:start]+'''    {wedge_gate_job j={g.nd,a.n1,a.v,NULL,a.p,a.x2,NULL,NULL,NULL};
      bcir_tensor_pool_run(s->pool,wedge_gate_fwd_task,&j,(g.nd+511)/512);}
'''+s[end:]
    start=s.index('    for(i=0;i<g.nd;i++) {',s.index('static int loss_backward_with('));end=s.index('    linear_backward',start)
    s=s[:start]+'''    {wedge_gate_job j={g.nd,a.n1,a.v,cur,a.p,NULL,dc,dv,dn};
      bcir_tensor_pool_run(s->pool,wedge_gate_bwd_task,&j,(g.nd+511)/512);}
'''+s[end:]
    return s

def ff_remat(s):
    s=once(s,'!term(&la,3,nf)', '!add(&la,nf)')
    s=once(s,'+3*g.nf; return g;', '+g.nf; return g;')
    if 'static cache activations(float *x,geometry g)' in s:
        s=once(s,'static cache activations(float *x,geometry g)', 'static cache activations(bcir_decoder_state *s,float *x,geometry g)')
    s=s.replace('activations(cs+l*g.la,g)', 'activations(s,cs+l*g.la,g)')
    s=once(s,'  a.gate=x; x+=g.nf; a.up=x; x+=g.nf; a.hidden=x; return a;', '''  /* Cross alias: dg overwrites up only after its read; du writes gate last. */
  a.up=s->scratch+4*g.nd+2*g.n*GP_RANK+g.n*GP_P;
  a.gate=a.up+g.nf;a.hidden=x;return a;''')
    call='    swiglu_backward(s,g.nf,a.gate,a.up,dh,dg,du);'
    s=once(s,call,'''    {
      size_t out[2]={g.f,g.f};const float *w[2]={p.gate,p.up};float *yy[2]={a.gate,a.up};
      linears(s,g.n,g.d,a.n2,2,out,w,yy);
    }
'''+call)
    return s

def frame_remat(s):
    # Retain the two vectors (already needed for the VJP), rematerialize packed p.
    s=once(s,'!add(&la,np)||','')
    s=once(s,'+g.n*GP_P+3*g.nf; return g;','+3*g.nf; return g;')
    if 'static cache activations(float *x,geometry g)' in s:
        s=once(s,'static cache activations(float *x,geometry g)', 'static cache activations(bcir_decoder_state *s,float *x,geometry g)')
    s=s.replace('activations(cs+l*g.la,g)','activations(s,cs+l*g.la,g)')
    s=once(s,'a.k=x; x+=g.n*GP_P;', 'a.k=s->scratch+2*g.nd+2*g.n*GP_RANK;')
    s=once(s,'    linear_backward(s,g.n,GP_P,g.d,a.k,p.k,dv,dk,grad.k,0.0f,wb);', '''    gp_wedge_forward(s->pool,g.b,g.t,a.q,a.k);
    /* Weight VJP consumes p before the input VJP reuses its address. */
    gemm(s,1,0,g.d,GP_P,g.n,1.0f,dv,a.k,wb,grad.k);
    gemm(s,0,0,g.n,GP_P,g.d,1.0f,dv,p.k,0.0f,dk);''')
    return s

def layer_remat(s):
    s=once(s,'!term(&p.activations,s->layers,la)', '!add(&p.activations,la)')
    # Norm/logits offsets and cache offset must agree in forward and backward.
    s=s.replace('norm=cs+g.l*g.la','norm=cs+g.la').replace('activations(cs+l*g.la,g)','activations(cs,g)')
    start=s.index('    bcir_tensor_rms_scan_pool',s.index('static int forward('))
    end=s.index('\n  }\n  bcir_tensor_rms_scan_pool',start)
    body=s[start:end]
    # Factor the exact same layer body; recomputing writes an identical residual y.
    helper='''static void forward_layer(bcir_decoder_state *s,geometry g,layer p,cache a,
    float *x,float *y,int *badp) {
  const bcir_decoder_spec *sp=&s->plan.spec;size_t i;int bad=0;
'''+body+'''\n  *badp|=bad;\n}\n'''
    s=s[:start]+'    forward_layer(s,g,p,a,x,y,&bad);'+s[end:]
    marker='static int forward(bcir_decoder_state *s'
    s=once(s,marker,helper+'\n'+marker)
    anchor='    cache a=activations(cs,g); float *x=xs+l*g.nd;'
    s=once(s,anchor,anchor+'''
    {int bad=0;forward_layer(s,g,p,a,x,x+g.nd,&bad);if(bad)return BCIR_DT_NUMERIC;}''')
    return s

def lowrank(s,q):
    # W_project = C[D,Q] B[Q,P]; packed parameter blocks, saved N*Q intermediate.
    s=once(s,'!term(&lp,(size_t)GP_RANK+GP_P,s->width)',
        f'!term(&lp,GP_RANK,s->width)||!term(&lp,{q},GP_P+(size_t)s->width)')
    s=once(s,'(GP_RANK+GP_P)*g.d', f'GP_RANK*g.d+{q}*(GP_P+g.d)')
    s=once(s,'p.k=x; x+=g.d*GP_P;', f'p.k=x; x+={q}*(GP_P+g.d);')
    s=once(s,'!add(&la,np)', f'!add(&la,np)||!term(&la,{q},n)')
    s=once(s,'g.n*GP_P+3*g.nf; return g;', f'g.n*GP_P+{q}*g.n+3*g.nf; return g;')
    s=once(s,'a.p=x; x+=g.nd; a.ctx=NULL;', f'a.p=x; x+=g.nd; a.ctx=x;x+={q}*g.n;')
    # Scratch has Q*N extra, pointed to by the otherwise unused tmp name.
    s=once(s,'!add(&p.scratch,np)',f'!add(&p.scratch,np)||!term(&p.scratch,{q},n)')
    s=once(s,'dh=du+g.nf; dl=logits;', f'dh=du+g.nf; tmp=dh+g.nf; dl=logits;')
    s=once(s,'    linear(s,g.n,GP_P,g.d,a.k,p.k,a.v);',f'''    linear(s,g.n,GP_P,{q},a.k,p.k,a.ctx);
    linear(s,g.n,{q},g.d,a.ctx,p.k+{q}*GP_P,a.v);''')
    s=once(s,'    linear_backward(s,g.n,GP_P,g.d,a.k,p.k,dv,dk,grad.k,0.0f,wb);',f'''    linear_backward(s,g.n,{q},g.d,a.ctx,p.k+{q}*GP_P,dv,tmp,grad.k+{q}*GP_P,0.0f,wb);
    linear_backward(s,g.n,GP_P,{q},a.k,p.k,tmp,dk,grad.k,0.0f,wb);''')
    return s

def compact(s,ff_cached=True):
    # Life ranges are serialized kernel boundaries. Reuse dead FF VJPs in the mixer.
    a=s.index('!term(&p.scratch,4,nd)');b=s.index('p.parameters>SIZE_MAX/4',a)
    s=s[:a]+'''!term(&p.scratch,1,nd)||!term(&p.scratch,3,nf)||
      !product(n,3*(size_t)s->width+GP_P+2*GP_RANK,&scratch_alt)||scratch_alt>SIZE_MAX/4||'''+s[b:]
    s=once(s,'size_t n,nd,nf,nv,nr,np,lp=0,la=0;', 'size_t n,nd,nf,nv,nr,np,lp=0,la=0,scratch_alt=0;')
    s=once(s,'  *out=p;return BCIR_DT_OK;', '  if(scratch_alt>p.scratch)p.scratch=scratch_alt;\n  *out=p;return BCIR_DT_OK;')
    start=s.index('  cur=s->scratch;');end=s.index('\n  final=',start)
    s=s[:start]+'''  cur=s->scratch;tmp=cur;dn=cur+g.nd;dq=cur+3*g.nd+g.n*GP_P;dk=cur+3*g.nd;
  dv=cur+2*g.nd;dc=cur+3*g.nd;dh=cur+g.nd;dg=dh+g.nf;du=dg+g.nf;dl=logits;'''+s[end:]
    if 'a.k=s->scratch+2*g.nd+2*g.n*GP_RANK;' in s:
        s=s.replace('a.k=s->scratch+2*g.nd+2*g.n*GP_RANK;', 'a.k=s->scratch+3*g.nd;')
    if 'a.up=s->scratch+4*g.nd+2*g.n*GP_RANK+g.n*GP_P;' in s:
        s=s.replace('a.up=s->scratch+4*g.nd+2*g.n*GP_RANK+g.n*GP_P;', 'a.up=s->scratch+g.nd+g.nf;')
    return s

def compile_source(source,name,rank=22,heads=1,sanitize=False):
    cf=ROOT/'build'/f'{name}.c';cf.write_text(source)
    cmd=['gcc','-std=c11','-O3','-fPIC','-shared','-ffp-contract=off','-fno-math-errno','-pthread',
         '-I',str(ROOT/'vendor'),f'-DGP_RANK={rank}','-DGP_PROJECTOR=0','-DGP_SYMMETRIC=0',
         f'-DGP_HEADS={heads}',str(ROOT/'vendor/bcir_tensor.c'),str(cf),'-lm','-o',str(cf.with_suffix('.so'))]
    if sanitize:cmd[2:2]=['-fsanitize=undefined','-fno-sanitize-recover=all']
    r=subprocess.run(cmd,text=True,capture_output=True,timeout=120)
    if r.returncode:raise RuntimeError(r.stderr)
    (ROOT/'build'/f'{name}-build.json').write_text(json.dumps({'command':cmd,'bcir_commit':PIN,
        'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'stderr':r.stderr},indent=2))
    print(name,'built',flush=True)

def multi(s):
    start=s.index('static void gp_pair_forward(');end=s.index('static void gp_forward_task(',start)
    # The override is before make_plan/geometry, so every arena uses the new feature count.
    return s[:start]+(ROOT/'multi_core.h').read_text()+'\n'+s[end:]

if __name__=='__main__':
    base=(ROOT/'build/global_norope.c').read_text()
    fast=grouped_gate(reuse(base))
    variants={'baseline':base,'reuse':reuse(base),'fast':fast,'compact':compact(fast),
              'ff_remat':ff_remat(fast),'frame_remat':frame_remat(fast),
              'frame_ff_remat':ff_remat(frame_remat(fast)),
              'frame_ff_compact':compact(ff_remat(frame_remat(fast))),
              'layer_remat':layer_remat(fast),'layer_compact':compact(layer_remat(fast)),
              'lowrank32':lowrank(fast,32)}
    for name,s in variants.items():compile_source(s,name)
    compile_source(multi(fast),'multi12',rank=24,heads=2)
    compile_source((ROOT/'build/transformer.c').read_text(),'attention')
