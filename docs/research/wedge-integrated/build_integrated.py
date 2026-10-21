"""Build pinned native foundations and compact precision/hybrid combinations."""
from build_next import ROOT,once,compile_source
import json,hashlib

GLOBAL=('fast','compact','frame_ff_compact','layer_compact','layer_frame_ff')

def hybrid(source):
    source=once(source,'prefix[t*GP_RANK+k]=(float)(sum[k]*scale);',
        'prefix[t*GP_RANK+k]=(float)(0.5*sum[k]*scale+(t?0.5*(double)z[(t-1)*GP_RANK+k]:0));')
    source=once(source,'dz[t*GP_RANK+k]+=(float)carry[k];',
        'dz[t*GP_RANK+k]+=(float)carry[k];\n      if(t+1<j->time)dz[t*GP_RANK+k]+=0.5f*dv[(t+1)*GP_RANK+k];')
    return once(source,'carry[k]+=(double)dv[t*GP_RANK+k]*scale;',
        'carry[k]+=0.5*(double)dv[t*GP_RANK+k]*scale;')

def guarded(source):
    source=once(source,'static void gp_pair_forward(',(ROOT/'precision_core.h').read_text()+'\nstatic void gp_pair_forward(')
    source=once(source,'static void gp_pair_forward(const float *u,const float *v,float *out) {',
        'static void gp_pair_forward(const float *u,const float *v,float *out) {\n  if(wedge_ill_conditioned(u,v)){wedge_precise_forward(u,v,out);return;}')
    return once(source,'static void gp_pair_backward(const float *u,const float *v,const float *dp,float *du,float *dv) {',
        'static void gp_pair_backward(const float *u,const float *v,const float *dp,float *du,float *dv) {\n  if(wedge_ill_conditioned(u,v)){wedge_precise_backward(u,v,dp,du,dv);return;}')

def audit(source):
    # Audit-only: counters are atomic on the pool, never used in timed/training builds.
    header='''#include <stdatomic.h>
static _Atomic unsigned long long wedge_audit_counts[9];
void wedge_audit_reset(void){unsigned i;for(i=0;i<9;i++)atomic_store(&wedge_audit_counts[i],0);}
unsigned long long wedge_audit_get(unsigned i){return i<9?atomic_load(&wedge_audit_counts[i]):0;}
static void wedge_audit_frame(const float *u,const float *v){
  unsigned i,bin=0;double uu=0,vv=0,uv=0,prod,ratio;
  for(i=0;i<GP_RANK;i++){uu+=(double)u[i]*u[i];vv+=(double)v[i]*v[i];uv+=(double)u[i]*v[i];}
  prod=uu*vv;ratio=prod>0?(prod-uv*uv)/prod:0;
  if(ratio>1e-8)bin=1;if(ratio>1e-6)bin=2;if(ratio>1e-4)bin=3;if(ratio>1e-2)bin=4;if(ratio>0.5)bin=5;
  atomic_fetch_add(&wedge_audit_counts[0],1);
  atomic_fetch_add(&wedge_audit_counts[3+bin],1);
  if(prod==0||ratio<=1e-4)atomic_fetch_add(&wedge_audit_counts[1],1);
}
'''
    # GP_RANK has already been defined before the pair kernel in the generated C.
    source=once(source,'static void gp_pair_forward(',header+'\nstatic void gp_pair_forward(')
    source=once(source,'static void gp_pair_forward(const float *u,const float *v,float *out) {',
        'static void gp_pair_forward(const float *u,const float *v,float *out) {\n  wedge_audit_frame(u,v);')
    source=once(source,'if(n<=1e-6){for(c=0;c<GP_P;c++)out[c]=0;return;}',
        'if(n<=1e-6){atomic_fetch_add(&wedge_audit_counts[2],1);for(c=0;c<GP_P;c++)out[c]=0;return;}')
    # Counters must be declared before precise_forward uses them.
    start=source.index(header);source=source[:start]+source[start+len(header):]
    marker='static int wedge_ill_conditioned('
    return once(source,marker,header+'\n'+marker)

if __name__=='__main__':
    for name in (*GLOBAL,'hybrid','attention'):
        compile_source((ROOT/'build'/f'{name}.c').read_text(),name)
    for name in ('compact','frame_ff_compact','layer_frame_ff'):
        source=(ROOT/'build'/f'{name}.c').read_text()
        compile_source(hybrid(source),'hybrid_'+name)
        compile_source(guarded(source),'guard_'+name)
    compile_source(audit((ROOT/'build/guard_compact.c').read_text()),'guard_audit')
    for name in ('compact','layer_frame_ff','guard_compact','hybrid_compact'):
        compile_source((ROOT/'build'/f'{name}.c').read_text(),name+'_ubsan',sanitize=True)
    print('integrated native portfolio built')
