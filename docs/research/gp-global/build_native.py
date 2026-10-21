"""Generate an isolated, reviewable native experiment from pinned BCIR sources.

No BCIR checkout is modified. Both models share the same C tensor kernels,
optimizer and optional UINT32_MAX loss-mask extension. GP adds no heap allocation.
"""
from pathlib import Path
import argparse, hashlib, json, shutil, subprocess

ROOT = Path(__file__).resolve().parent
PIN = 'aaa2e6c0ee0ee15b83fe4d0373b48bd893e80d6d'

GP_CORE = r'''
#define GP_P ((GP_RANK*(GP_RANK-1))/2)
static const size_t gp_offsets[] = {1,2,4,8,16,32};
#define GP_M (sizeof(gp_offsets)/sizeof(gp_offsets[0]))
static size_t gp_other(size_t t,size_t dt,size_t time) {
#if GP_FUTURE
  return t+dt<time ? t+dt : time;
#else
  return t>=dt ? t-dt : time;
#endif
}
typedef struct gp_job { size_t batch,time; const float *z,*dp; float *p,*dz; } gp_job;
static void gp_forward_task(void *ctx,size_t task) {
  gp_job *j=(gp_job *)ctx; size_t t=task%j->time,b=task/j->time,i,k,c,m,count=0,other;
  float *out=j->p+task*GP_P; const float *u=j->z+task*GP_RANK;
  for(c=0;c<GP_P;c++) out[c]=0;
  for(m=0;m<GP_M;m++) if(gp_other(t,gp_offsets[m],j->time)<j->time) count++;
  if(!count) return;
  for(m=0;m<GP_M;m++) {
    float p[GP_P]; double norm=0; const float *v;
    other=gp_other(t,gp_offsets[m],j->time); if(other==j->time) continue;
    v=j->z+(b*j->time+other)*GP_RANK;
    for(i=0,c=0;i<GP_RANK;i++) for(k=i+1;k<GP_RANK;k++,c++) {
#if GP_SYMMETRIC
      p[c]=u[i]*v[k]+u[k]*v[i];
#else
      p[c]=u[i]*v[k]-u[k]*v[i];
#endif
      norm+=(double)p[c]*p[c];
    }
    norm=sqrt(norm); if(norm<1e-6) norm=1e-6;
    for(c=0;c<GP_P;c++) out[c]+=(float)((double)p[c]/norm)/(float)count;
  }
}
/* A batch is owned by one task: deterministic scatter into both endpoints,
 * no atomics, no races, no per-offset saved feature tensor. */
static void gp_backward_task(void *ctx,size_t task) {
  gp_job *j=(gp_job *)ctx; size_t t,m,i,k,c,other,count;
  float *dz=j->dz+task*j->time*GP_RANK;
  const float *z=j->z+task*j->time*GP_RANK,*dp=j->dp+task*j->time*GP_P;
  memset(dz,0,j->time*GP_RANK*sizeof(float));
  for(t=0;t<j->time;t++) {
    const float *u=z+t*GP_RANK; count=0;
    for(m=0;m<GP_M;m++) if(gp_other(t,gp_offsets[m],j->time)<j->time) count++;
    if(!count) continue;
    for(m=0;m<GP_M;m++) {
      float p[GP_P]; double norm=0,dot=0; const float *v; int clamped;
      other=gp_other(t,gp_offsets[m],j->time); if(other==j->time) continue;
      v=z+other*GP_RANK;
      for(i=0,c=0;i<GP_RANK;i++) for(k=i+1;k<GP_RANK;k++,c++) {
#if GP_SYMMETRIC
        p[c]=u[i]*v[k]+u[k]*v[i];
#else
        p[c]=u[i]*v[k]-u[k]*v[i];
#endif
        norm+=(double)p[c]*p[c];
        dot+=(double)dp[t*GP_P+c]*(double)p[c]/(double)count;
      }
      norm=sqrt(norm); clamped=norm<=1e-6; if(clamped) norm=1e-6;
      for(i=0,c=0;i<GP_RANK;i++) for(k=i+1;k<GP_RANK;k++,c++) {
        float q=(float)(((double)dp[t*GP_P+c]/count-
            (clamped ? 0.0 : (double)p[c]*dot/(norm*norm)))/norm);
        dz[t*GP_RANK+i]+=q*v[k];
#if GP_SYMMETRIC
        dz[t*GP_RANK+k]+=q*v[i]; dz[other*GP_RANK+i]+=q*u[k];
#else
        dz[t*GP_RANK+k]-=q*v[i]; dz[other*GP_RANK+i]-=q*u[k];
#endif
        dz[other*GP_RANK+k]+=q*u[i];
      }
    }
  }
}
void gp_wedge_forward(bcir_tensor_pool *pool,size_t batch,size_t time,const float *z,float *p) {
  gp_job j={batch,time,z,NULL,p,NULL};
  bcir_tensor_pool_run(pool,gp_forward_task,&j,batch*time);
}
void gp_wedge_backward(bcir_tensor_pool *pool,size_t batch,size_t time,const float *z,
    const float *dp,float *dz) {
  gp_job j={batch,time,z,dp,NULL,dz};
  bcir_tensor_pool_run(pool,gp_backward_task,&j,batch);
}
'''

def replace(s, old, new):
    assert s.count(old)==1, (old[:100],s.count(old))
    return s.replace(old,new)

def masked(s):
    marker='typedef struct loss_job {'
    s=s.replace(marker,'''static int target_ids(bcir_decoder_state *s,const uint32_t *x,size_t n) {
  size_t i;
  if(n>SIZE_MAX/sizeof(uint32_t) || !external(s,x,n*sizeof(uint32_t)) ||
      (uintptr_t)x%_Alignof(uint32_t)) return 0;
  for(i=0;i<n;i++) if(x[i]!=UINT32_MAX && x[i]>=s->plan.spec.vocab) return 0;
  return 1;
}
'''+marker,1)
    s=replace(s,'size_t n,v; const float *logits;', 'size_t n,v,denom; const float *logits;')
    s=replace(s,'    for (j=1;j<v;j++) if (row[j]>maxv) maxv=row[j];', '''    if(jb->targets[i]==UINT32_MAX) {
      term=0; memset(d,0,v*sizeof(float));
      memcpy(jb->terms+i*sizeof(double),&term,sizeof(double)); continue;
    }
    for (j=1;j<v;j++) if (row[j]>maxv) maxv=row[j];''')
    s=replace(s,'/(double)jb->n);','/(double)jb->denom);')
    s=replace(s,'geometry g; size_t i,l; double ce=0.0; loss_job lj;',
              'geometry g; size_t i,l,supervised=0; double ce=0.0; loss_job lj;')
    s=replace(s,'if (n!=g.n || !ids(s,tokens,n) || !ids(s,targets,n) ||',
              'if (n!=g.n || !ids(s,tokens,n) || !target_ids(s,targets,n) ||')
    s=replace(s,'  if (!(how&DT_KNOWN) &&', '''  for(i=0;i<n;i++) supervised+=(targets[i]!=UINT32_MAX);
  if(!supervised) return BCIR_DT_INVALID;
  if (!(how&DT_KNOWN) &&''')
    s=replace(s,'lj.n=g.n; lj.v=g.v; lj.logits=', 'lj.n=g.n; lj.v=g.v; lj.denom=supervised; lj.logits=')
    s=replace(s,'*loss=ce/(double)g.n;','*loss=ce/(double)supervised;')
    s=replace(s,'!ids(s,targets,total)', '!target_ids(s,targets,total)')
    return s

def grassmann(s):
    start=s.index('int bcir_decoder_make_plan(');end=s.index('static int overlap(',start)
    s=s[:start]+'''int bcir_decoder_make_plan(const bcir_decoder_spec *s,bcir_decoder_plan *out) {
  bcir_decoder_plan p; size_t n,nd,nf,nv,nr,np,lp=0,la=0;
  if(!s||!out||!s->vocab||!s->width||!s->heads||!s->kvheads||!s->layers||!s->ff||
      !s->batch||!s->time||s->tied>1||s->vocab>(1u<<20)||s->width>65536||
      s->layers>4096||s->batch>65536||s->time>65536||s->ff>(1u<<20)||
      !isfinite(s->rms_eps)||s->rms_eps<=0||s->rms_eps>FLT_MAX) return BCIR_DT_INVALID;
  p.spec=*s;p.parameters=p.activations=p.scratch=0;
  if(!product(s->batch,s->time,&n)||!product(n,s->width,&nd)||!product(n,s->ff,&nf)||
      !product(n,s->vocab,&nv)||!product(n,GP_RANK,&nr)||!product(n,GP_P,&np)||
      !term(&lp,2,s->width)||!term(&lp,(size_t)GP_RANK+GP_P,s->width)||
      !term(&lp,2*(size_t)s->width,s->width)||!term(&lp,3*(size_t)s->width,s->ff)||
      !term(&p.parameters,s->vocab,s->width)||!term(&p.parameters,s->layers,lp)||
      !add(&p.parameters,s->width)||(!s->tied&&!term(&p.parameters,s->vocab,s->width))||
      !term(&la,5,nd)||!add(&la,nr)||!add(&la,np)||!term(&la,3,nf)||
      !term(&p.activations,(size_t)s->layers+1,nd)||!term(&p.activations,s->layers,la)||
      !add(&p.activations,nd)||!add(&p.activations,nv)||!term(&p.scratch,5,nd)||
      !add(&p.scratch,nr)||!add(&p.scratch,np)||!term(&p.scratch,3,nf)||
      !add(&p.scratch,nv)||p.parameters>SIZE_MAX/4||p.activations>SIZE_MAX/4||
      p.scratch>SIZE_MAX/4) return BCIR_DT_CAPACITY;
  *out=p;return BCIR_DT_OK;
}
'''+s[end:]
    s=replace(s,'g.a=g.b*g.h*g.t*g.t; g.lp=2*g.d+2*g.d*g.d+2*g.d*g.k+3*g.d*g.f;\n  g.la=5*g.nd+2*g.nk+g.a+3*g.nf; return g;',
        'g.a=g.n*GP_P; g.lp=2*g.d+(GP_RANK+GP_P)*g.d+2*g.d*g.d+3*g.d*g.f;\n  g.la=5*g.nd+g.n*GP_RANK+g.n*GP_P+3*g.nf; return g;')
    s=replace(s,'p.norm1=x; x+=g.d; p.q=x; x+=g.d*g.d; p.k=x; x+=g.k*g.d;\n  p.v=x; x+=g.k*g.d;',
        'p.norm1=x; x+=g.d; p.q=x; x+=GP_RANK*g.d; p.k=x; x+=g.d*GP_P;\n  p.v=x; x+=g.d*g.d;')
    s=replace(s,'a.n1=x; x+=g.nd; a.q=x; x+=g.nd; a.k=x; x+=g.nk; a.v=x; x+=g.nk;\n  a.p=x; x+=g.a; a.ctx=x; x+=g.nd; a.x2=x; x+=g.nd; a.n2=x; x+=g.nd;',
        'a.n1=x; x+=g.nd; a.q=x; x+=g.n*GP_RANK; a.k=x; x+=g.n*GP_P; a.v=x; x+=g.nd;\n  a.p=x; x+=g.nd; a.ctx=NULL; a.x2=x; x+=g.nd; a.n2=x; x+=g.nd;')
    start=s.index('    {\n      size_t out[3];',s.index('static int forward('))
    end=s.index('    bcir_tensor_add_pool(s->pool,g.nd,x,a.x2);',start)
    s=s[:start]+'''    linear(s,g.n,g.d,GP_RANK,a.n1,p.q,a.q);
    bcir_tensor_rope_pool(s->pool,g.b,g.t,1,GP_RANK,sp->rope_base,a.q,0);
    gp_wedge_forward(s->pool,g.b,g.t,a.q,a.k);
    linear(s,g.n,GP_P,g.d,a.k,p.k,a.v);
    linear(s,g.n,g.d,g.d,a.n1,p.v,a.p);
    linear(s,g.n,g.d,g.d,a.v,p.o,a.x2);
    for(i=0;i<g.nd;i++) {
      float e=bcir_tensor_expf(-(a.p[i]+a.x2[i]));
      a.p[i]=1.0f/(1.0f+e);
      a.x2[i]=a.p[i]*a.n1[i]+(1.0f-a.p[i])*a.v[i];
    }
    if(!finite_pool(s,a.q,g.n*GP_RANK)||!finite_pool(s,a.k,g.n*GP_P)||
        !finite_pool(s,a.v,g.nd)||!finite_pool(s,a.p,g.nd)) bad=1;
'''+s[end:]
    s=replace(s,'cur=s->scratch; tmp=cur+g.nd; dn=tmp+g.nd; dq=dn+g.nd; dk=dq+g.nd;\n  dv=dk+g.nk; dc=dv+g.nk;',
        'cur=s->scratch; tmp=cur+g.nd; dn=tmp+g.nd; dq=dn+g.nd; dk=dq+g.n*GP_RANK;\n  dv=dk+g.n*GP_P; dc=dv+g.nd;')
    start=s.index('    linear_backward(s,g.n,g.d,g.d,a.ctx,p.o,cur,dc,grad.o,0.0f,wb);')
    end=s.index('    rms_backward(s,g.n,g.d,x,p.norm1,dn,cur,grad.norm1,dq,BCIR_TENSOR_RMSB_ADD|gf);',start)
    s=s[:start]+'''    for(i=0;i<g.nd;i++) {
      float alpha=a.p[i];
      dc[i]=cur[i]*(a.n1[i]-a.v[i])*alpha*(1.0f-alpha);
      dv[i]=cur[i]*(1.0f-alpha); dn[i]=cur[i]*alpha;
    }
    linear_backward(s,g.n,g.d,g.d,a.n1,p.v,dc,dn,grad.v,1.0f,wb);
    linear_backward(s,g.n,g.d,g.d,a.v,p.o,dc,dv,grad.o,1.0f,wb);
    linear_backward(s,g.n,GP_P,g.d,a.k,p.k,dv,dk,grad.k,0.0f,wb);
    gp_wedge_backward(s->pool,g.b,g.t,a.q,dk,dq);
    bcir_tensor_rope_pool(s->pool,g.b,g.t,1,GP_RANK,s->plan.spec.rope_base,dq,1);
    linear_backward(s,g.n,g.d,GP_RANK,a.n1,p.q,dq,dn,grad.q,1.0f,wb);
'''+s[end:]
    return s

def build(repo, name, rank=22, symmetric=False, future=False, sanitize=False):
    vendor=ROOT/'vendor';vendor.mkdir(exist_ok=True)
    for p in ['bcir_decoder_train.c','bcir_decoder_train.h','bcir_tensor.c','bcir_tensor.h']:
        shutil.copy2(repo/'runtime/c'/p,vendor/p)
    for p in ['LICENSE','LICENSE.md']:
        if (repo/p).exists():shutil.copy2(repo/p,vendor/p)
    source=(vendor/'bcir_decoder_train.c').read_text()
    if name!='stock':source=masked(source)
    is_global=name.startswith(('global','projector'))
    if name not in ('transformer','stock'):
        source=grassmann(source)
        if is_global:
            # Reserve cached prefix and its adjoint in caller-owned arenas.
            source=source.replace('!add(&la,nr)', '!term(&la,2,nr)')
            source=source.replace('!add(&p.scratch,nr)', '!term(&p.scratch,2,nr)')
            source=source.replace('g.la=5*g.nd+g.n*GP_RANK+', 'g.la=5*g.nd+2*g.n*GP_RANK+')
            source=source.replace('a.q=x; x+=g.n*GP_RANK;', 'a.q=x; x+=2*g.n*GP_RANK;')
            source=source.replace('dk=dq+g.n*GP_RANK;', 'dk=dq+2*g.n*GP_RANK;')
            source=(ROOT/'global_core.h').read_text()+'\n'+source
        else:source=GP_CORE+'\n'+source
        # The new operator needs the tensor and math declarations before its definition.
        source='#include "bcir_decoder_train.h"\n#include <math.h>\n#include <string.h>\n'+source
    out=ROOT/'build';out.mkdir(exist_ok=True)
    cfile=out/(name+'.c');cfile.write_text(source)
    cmd=['gcc','-std=c11','-O3','-fPIC','-shared','-ffp-contract=off','-fno-math-errno',
         '-pthread','-I',str(vendor),f'-DGP_RANK={rank}',f'-DGP_SYMMETRIC={int(symmetric)}',
         f'-DGP_FUTURE={int(future)}',f'-DGP_PROJECTOR={int(name.startswith("projector"))}',
         str(vendor/'bcir_tensor.c'),str(cfile),'-lm','-o',str(out/(name+'.so'))]
    if sanitize:cmd[2:2]=['-fsanitize=undefined','-fno-sanitize-recover=all']
    r=subprocess.run(cmd,text=True,capture_output=True,timeout=120)
    if r.returncode:raise RuntimeError(r.stderr)
    (out/(name+'-build.json')).write_text(json.dumps({'command':cmd,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'bcir_commit':PIN,'stderr':r.stderr},indent=2))
    return out/(name+'.so')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);args=p.parse_args()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.repo,text=True).strip()==PIN
    for name,sym,future in [('transformer',False,False),('grassmann',False,False),('symmetric',True,False),
                          ('global',False,False),('global_symmetric',True,False),('projector',False,False)]:
        print(build(args.repo,name,symmetric=sym,future=future),flush=True)
