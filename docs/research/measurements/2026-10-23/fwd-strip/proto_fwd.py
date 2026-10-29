"""FWD-STRIP prototypes on a copy of bcir_tensor.c (ATT-CONTIG's tree), chosen by -DPROTO_FWD:
 1 strip   -- the stored forward a strip of rows at a time in place (att_strip on the stored P's own rows,
              att_strip_rows rows), each strip's rows' future columns zeroed by memset just before it
 2 stripnt -- the same, the future columns zeroed by streaming (non-temporal) stores
 3 fillnt  -- the forward as it is, its fill's future columns by streaming stores
Every value is att_head's: att_strip computes a strip's rows bit for bit as att_head does, and every
future column ends +0.0 either way. usage: proto_fwd.py SRC OUT"""

import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()
out = Path(sys.argv[2])
helper = r"""
#ifndef PROTO_FWD
#define PROTO_FWD 0
#endif
#ifndef PROTO_FROM
#define PROTO_FROM 1024
#endif
/* zero floats [from, to) of a row, the 64-byte-aligned middle by streaming stores where nt */
static void proto_zero(float *row, size_t from, size_t to, int nt) {
#ifdef BCIR_MM_DISPATCH
  if (nt && to>from) {
    float *p=row+from,*e=row+to;
    while (p<e && ((uintptr_t)p&63)) *p++=0.0f;
    proto_stream(&p,e);
    while (p<e) *p++=0.0f;
    return;
  }
#endif
  (void)nt;
  if (to>from) memset(row+from,0,(to-from)*sizeof(float));
}
"""
stream = r"""
#ifdef BCIR_MM_DISPATCH
__attribute__((target("avx512f"))) static void proto_stream(float **pp, float *e) {
  float *p=*pp; __m512 z=_mm512_setzero_ps();
  for (;p+16<=e;p+=16) _mm512_stream_ps(p,z);
  *pp=p;
}
__attribute__((target("avx512f"))) static void proto_fence(void) { _mm_sfence(); }
#endif
"""
anchor = "/* One (batch, query head): its scores, softmax rows and context; nothing shared with another. */\n"
assert src.count(anchor) == 1
src = src.replace(anchor, stream + helper + anchor, 1)
old_head = "static void att_head(const att_job *a, size_t r, size_t h) {\n  size_t t=a->t,nh=a->nh,nk=a->nk,d=a->d,kh=h/(nh/nk),i;\n  float *ph=a->p+(r*nh+h)*t*t; mm_args x;\n"
assert src.count(old_head) == 1
# att_strip is defined after att_head: declare it first
decl = "static void att_strip(const att_job *a, size_t r, size_t h, size_t i0, size_t i1, float *s);\nstatic size_t att_strip_rows(size_t t);\n"
new_head = (
    decl
    + old_head
    + r"""  if ((PROTO_FWD==1 || PROTO_FWD==2) && t>=PROTO_FROM) {
    size_t rows=att_strip_rows(t),i0;
    for (i0=0;i0<t;i0+=rows) {
      size_t i1=t-i0<rows ? t : i0+rows;
      for (i=i0;i<i1;i++) proto_zero(ph+i*t,i+1,t,PROTO_FWD==2);
      att_strip(a,r,h,i0,i1,ph+i0*t);
    }
#ifdef BCIR_MM_DISPATCH
    if (PROTO_FWD==2) proto_fence();
#endif
    if (a->bad) att_verdict(a,r,h);
    return;
  }
"""
)
src = src.replace(old_head, new_head, 1)
old_fill = "    for (i=0;i<v1;i++) { size_t end=(i/32+1)*32<v1 ? (i/32+1)*32 : v1; memset(ph+i*t+end,0,(t-end)*sizeof(float)); }\n"
assert src.count(old_fill) == 1
new_fill = (
    "    for (i=0;i<v1;i++) { size_t end=(i/32+1)*32<v1 ? (i/32+1)*32 : v1; proto_zero(ph+i*t,end,t,PROTO_FWD==3); }\n"
    "#ifdef BCIR_MM_DISPATCH\n    if (PROTO_FWD==3) proto_fence();\n#endif\n"
)
src = src.replace(old_fill, new_fill, 1)
out.write_text(src)
print("prototype written", out)
