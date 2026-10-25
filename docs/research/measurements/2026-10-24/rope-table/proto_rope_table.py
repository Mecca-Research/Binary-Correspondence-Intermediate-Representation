"""ROPE-TABLE-1: the trainer's rotations reading one table of the step's cosines and sines, filled once a
call with the unit's own expressions (a prototype: the table in a static buffer, the direct path only).
usage: proto_rope_table.py SRC_DIR DST_DIR  (bcir_tensor.c and bcir_decoder_train.c)"""

import sys
from pathlib import Path

src_dir, dst_dir = Path(sys.argv[1]), Path(sys.argv[2])


def once(text, old, new):
    if text.count(old) != 1:
        raise SystemExit(f"anchor found {text.count(old)} times: {old[:70]!r}")
    return text.replace(old, new)


t = (src_dir / "bcir_tensor.c").read_text(encoding="utf-8")
add = r"""
/* ROPE-TABLE-1 (prototype): position pos's half cosines then its half sines, each the expression
 * rope_rows evaluates -- (float)cos(pos*f[j]) and (float)sin(pos*f[j]), f[j]=base^(-2j/dim) */
typedef struct ropet_job { size_t t,dim; double base; float *table; const float *ctab; size_t b,heads; float *x; int inverse; } ropet_job;
static void ropet_fill_task(void *ctx, size_t task) {
  ropet_job *j=(ropet_job *)ctx; size_t half=j->dim/2,p0=task*BCIR_ROW_CHUNK,p1=p0+BCIR_ROW_CHUNK<j->t ? p0+BCIR_ROW_CHUNK : j->t,pos,k,k0;
  for (k0=0;k0<half;k0+=BCIR_ROPE_J) {
    size_t nk=half-k0<BCIR_ROPE_J ? half-k0 : BCIR_ROPE_J; double f[BCIR_ROPE_J];
    for (k=0;k<nk;k++) f[k]=pow(j->base,-2.0*(double)(k0+k)/(double)j->dim);
    for (pos=p0;pos<p1;pos++) for (k=0;k<nk;k++) {
      double angle=(double)pos*f[k];
      j->table[pos*j->dim+k0+k]=(float)cos(angle); j->table[pos*j->dim+half+k0+k]=(float)sin(angle);
    }
  }
}
void bcir_tensor_rope_table_pool(bcir_tensor_pool *pool, size_t t, size_t dim, double base, float *table);
void bcir_tensor_rope_table_pool(bcir_tensor_pool *pool, size_t t, size_t dim, double base, float *table) {
  ropet_job j; memset(&j,0,sizeof j); j.t=t; j.dim=dim; j.base=base; j.table=table;
  pool_run(pool,ropet_fill_task,&j,(t+BCIR_ROW_CHUNK-1)/BCIR_ROW_CHUNK);
}
static void ropet_task(void *ctx, size_t task) {
  ropet_job *j=(ropet_job *)ctx; size_t half=j->dim/2,p0=task*BCIR_ROW_CHUNK,p1=p0+BCIR_ROW_CHUNK<j->t ? p0+BCIR_ROW_CHUNK : j->t,pos,r,h,k;
  for (pos=p0;pos<p1;pos++) {
    const float *c=j->ctab+pos*j->dim,*s0=c+half; float s[BCIR_ROPE_J*8];
    for (k=0;k<half;k++) s[k]=s0[k]*(j->inverse ? -1.0f : 1.0f);
    for (r=0;r<j->b;r++) for (h=0;h<j->heads;h++) {
      float *lo=j->x+((r*j->t+pos)*j->heads+h)*j->dim,*hi=lo+half;
      for (k=0;k<half;k++) { float a=lo[k],z=hi[k]; lo[k]=a*c[k]-z*s[k]; hi[k]=z*c[k]+a*s[k]; }
    }
  }
}
void bcir_tensor_rope_lent_pool(bcir_tensor_pool *pool, size_t b, size_t t, size_t heads, size_t dim,
    const float *table, float *x, int inverse);
void bcir_tensor_rope_lent_pool(bcir_tensor_pool *pool, size_t b, size_t t, size_t heads, size_t dim,
    const float *table, float *x, int inverse) {
  ropet_job j; memset(&j,0,sizeof j); j.t=t; j.dim=dim; j.ctab=table; j.b=b; j.heads=heads; j.x=x; j.inverse=inverse;
  if (dim/2>BCIR_ROPE_J*8) { bcir_tensor_rope_pool(pool,b,t,heads,dim,10000.0,x,inverse); return; }
  pool_run(pool,ropet_task,&j,(t+BCIR_ROW_CHUNK-1)/BCIR_ROW_CHUNK);
}
static bcir_tensor_mm_fn mm(const bcir_tensor_provider *p) {"""
t = once(t, "\nstatic bcir_tensor_mm_fn mm(const bcir_tensor_provider *p) {", add)
(dst_dir / "bcir_tensor.c").write_text(t, encoding="utf-8")

d = (src_dir / "bcir_decoder_train.c").read_text(encoding="utf-8")
proto = r"""
void bcir_tensor_rope_table_pool(bcir_tensor_pool *pool, size_t t, size_t dim, double base, float *table);
void bcir_tensor_rope_lent_pool(bcir_tensor_pool *pool, size_t b, size_t t, size_t heads, size_t dim,
    const float *table, float *x, int inverse);
#include <stdlib.h>
static float *proto_tab; static size_t proto_tab_n;
static void proto_fill(bcir_decoder_state *s,size_t t,size_t hd) {
  if (proto_tab_n<t*hd) { free(proto_tab); proto_tab=(float *)malloc(t*hd*sizeof(float)); proto_tab_n=t*hd; }
  bcir_tensor_rope_table_pool(s->pool,t,hd,s->plan.spec.rope_base,proto_tab);
}
static int layer_forward(bcir_decoder_state *s,geometry g,size_t l,float *slot,float *x,int full,"""
d = once(
    d,
    "\nstatic int layer_forward(bcir_decoder_state *s,geometry g,size_t l,float *slot,float *x,int full,",
    proto,
)
d = once(
    d,
    """  bcir_tensor_rope_pool(s->pool,g.b,g.t,g.h,g.hd,sp->rope_base,a.q,0);
  bcir_tensor_rope_pool(s->pool,g.b,g.t,g.kh,g.hd,sp->rope_base,a.k,0);""",
    """  if (!l) proto_fill(s,g.t,g.hd);
  (void)sp;
  bcir_tensor_rope_lent_pool(s->pool,g.b,g.t,g.h,g.hd,proto_tab,a.q,0);
  bcir_tensor_rope_lent_pool(s->pool,g.b,g.t,g.kh,g.hd,proto_tab,a.k,0);""",
)
d = once(
    d,
    """    bcir_tensor_rope_pool(s->pool,g.b,g.t,g.h,g.hd,s->plan.spec.rope_base,dq,1);
    bcir_tensor_rope_pool(s->pool,g.b,g.t,g.kh,g.hd,s->plan.spec.rope_base,dk,1);""",
    """    bcir_tensor_rope_lent_pool(s->pool,g.b,g.t,g.h,g.hd,proto_tab,dq,1);
    bcir_tensor_rope_lent_pool(s->pool,g.b,g.t,g.kh,g.hd,proto_tab,dk,1);""",
)
(dst_dir / "bcir_decoder_train.c").write_text(d, encoding="utf-8")
