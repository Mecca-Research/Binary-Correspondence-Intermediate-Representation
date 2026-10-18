"""PANELS-WRITER experiment: insert the strip writer (v4) into a copy of bcir_tensor.c.
python3 patch_v4.py SRC_TENSOR_C OUT_TENSOR_C"""

import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()
anchor = "/* Rows in ranges of BCIR_ROW_CHUNK: each row's statistics are its own. */\n"
assert src.count(anchor) == 1
field_old = "  unsigned char *bad;\n} panel_job;\n"
assert src.count(field_old) == 1
src = src.replace(
    field_old,
    "  unsigned char *bad;\n  size_t strip; int nt;     /* PANELS-WRITER experiment */\n} panel_job;\n",
)
V4 = r"""/* ---- PANELS-WRITER experiment: the strip writer (v4) -------------------------------------------
 * 32 rows x strip columns of each output's binary32 values in an L1 buffer; the A panel written a
 * tile at a time (16 rows of 32 columns, 1 KB contiguous), the second panel converted from the same
 * binary32 values by the same conversion (so the same bits as the A panel's), never read back. */
#define PANEL4_MAX 256
static void panel4_emit_generic(const panel_job *j, int k, size_t r0, size_t c0, size_t w,
    const float *v, size_t ld) {
  size_t rr,l;
  for (rr=0;rr<32;rr++) for (l=0;l<w;l++) {
    uint16_t h=bf16_of(v[rr*ld+l]);
    j->pa[k][panel_a_at(j->c,r0+rr,c0+l)]=h;
    if (j->at) j->pb[k][panel_a_at(j->r,c0+l,r0+rr)]=h; else j->pb[k][panel_b_at(j->r,r0+rr,c0+l)]=h;
  }
}
#ifdef BCIR_MM_DISPATCH
#define PANEL4_BODY(NAME, TARGET, CVT) \
TARGET static void NAME##_panel4_emit(const panel_job *j, int k, size_t r0, size_t c0, size_t w, \
    const float *v, size_t ld) { \
  size_t g,q,r,jb,p; uint16_t *pa=j->pa[k],*pb=j->pb[k]; _Alignas(64) float wd[256],tr[256]; \
  for (g=0;g<2;g++) for (q=0;q<w;q+=32) { \
    uint16_t *dst=pa+panel_a_at(j->c,r0+16*g,c0+q); \
    for (r=0;r<16;r++) { \
      const float *s=v+(16*g+r)*ld+q; \
      __m512i h=CVT(_mm512_loadu_ps(s),_mm512_loadu_ps(s+16)); \
      if (j->nt&1) _mm512_stream_si512((void *)(dst+r*32),h); else _mm512_storeu_si512(dst+r*32,h); \
    } \
  } \
  for (jb=0;jb<w;jb+=16) { \
    for (p=0;p<16;p++) { \
      __m512i h=BF512_PAIRS_BW(CVT(_mm512_loadu_ps(v+(2*p)*ld+jb),_mm512_loadu_ps(v+(2*p+1)*ld+jb))); \
      if (j->at) _mm512_store_si512(wd+16*p,h); \
      else { \
        uint16_t *dst=pb+panel_b_at(j->r,r0,c0+jb)+p*32; \
        if (j->nt&2) _mm512_stream_si512((void *)dst,h); else _mm512_storeu_si512(dst,h); \
      } \
    } \
    if (j->at) { \
      uint16_t *dst=pb+panel_a_at(j->r,c0+jb,r0); \
      if (j->nt&2) { \
        kb512_transpose16(wd,16,tr,16); \
        for (p=0;p<16;p++) _mm512_stream_si512((void *)(dst+p*32),_mm512_load_si512(tr+16*p)); \
      } \
      else kb512_transpose16(wd,16,(float *)(void *)dst,16); \
    } \
  } \
}
PANEL4_BODY(bf512, BF16_AVX512BW, bf512_cvt)
#ifdef BCIR_BF16_AMX
PANEL4_BODY(bfne, BF16_AMXHOST, bfne_cvt)
#endif
#endif
static void panel4_emit(const panel_job *j, int k, size_t r0, size_t c0, size_t w, const float *v,
    size_t ld) {
#ifdef BCIR_MM_DISPATCH
#ifdef BCIR_BF16_AMX
  if (j->conv==8) { bfne_panel4_emit(j,k,r0,c0,w,v,ld); return; }
#endif
  if (j->conv>=4) { bf512_panel4_emit(j,k,r0,c0,w,v,ld); return; }
#endif
  panel4_emit_generic(j,k,r0,c0,w,v,ld);
}
static void panel4_task(void *ctx, size_t t) {
  const panel_job *j=(const panel_job *)ctx; size_t b0=t*j->per,nb=j->r/32,b,rr,c0,ld=j->strip;
  size_t b1=b0+j->per<nb ? b0+j->per : nb; int two=j->op>=PANEL_SWIGLU1,bad=0;
  _Alignas(64) float v0[32*PANEL4_MAX],v1[32*PANEL4_MAX];
  for (b=b0;b<b1;b++) for (c0=0;c0<j->c;c0+=ld) {
    size_t w=j->c-c0<ld ? j->c-c0 : ld;
    for (rr=0;rr<32;rr++) {
      panel_values(j,b*32+rr,c0,w,v0+rr*ld,v1+rr*ld);
      if (j->bad) bad|=!span_finite_run(j->variant,v0+rr*ld,w);
    }
    panel4_emit(j,0,b*32,c0,w,v0,ld);
    if (two) panel4_emit(j,1,b*32,c0,w,v1,ld);
  }
#ifdef BCIR_MM_DISPATCH
  if (j->nt) _mm_sfence();
#endif
  if (j->bad) j->bad[t]=(unsigned char)bad;
}
static int panel4_run(bcir_tensor_pool *pool, panel_job *j, int *bad) {
  unsigned char bytes[BCIR_TENSOR_SCAN_TASKS]; size_t blocks,tasks;
  if (j->r%32 || j->c%32 || !j->strip || j->strip%32 || j->strip>PANEL4_MAX) return 0;
  if (!j->r || !j->c) return 1;
  blocks=j->r/32; tasks=blocks<BCIR_TENSOR_SCAN_TASKS ? blocks : BCIR_TENSOR_SCAN_TASKS;
  j->per=(blocks+tasks-1)/tasks; tasks=(blocks+j->per-1)/j->per;
  j->variant=mm_widest(); j->conv=bf16_widest(); j->bad=bad ? bytes : NULL;
  pool_run(pool,panel4_task,j,tasks);
  if (bad) scan_fold(bad,bytes,tasks);
  return 1;
}
int bcir_tensor_silu_mul_panels4_pool(bcir_tensor_pool *pool, int v2, size_t r, size_t c,
    const float *x, const float *u, unsigned short *a, unsigned short *b, int *bad, size_t strip, int nt);
int bcir_tensor_silu_mul_panels4_pool(bcir_tensor_pool *pool, int v2, size_t r, size_t c,
    const float *x, const float *u, unsigned short *a, unsigned short *b, int *bad, size_t strip, int nt) {
  panel_job j;
  memset(&j,0,sizeof j); j.r=r; j.c=c; j.op=v2 ? PANEL_SILU2 : PANEL_SILU1; j.x=x; j.u=u;
  j.pa[0]=a; j.pb[0]=b; j.at=0; j.strip=strip; j.nt=nt;
  return panel4_run(pool,&j,bad);
}
int bcir_tensor_swiglu_backward_panels4_pool(bcir_tensor_pool *pool, int v2, size_t r, size_t c,
    const float *g, const float *u, const float *dy, unsigned short *dga, unsigned short *dgt,
    unsigned short *dua, unsigned short *dut, size_t strip, int nt);
int bcir_tensor_swiglu_backward_panels4_pool(bcir_tensor_pool *pool, int v2, size_t r, size_t c,
    const float *g, const float *u, const float *dy, unsigned short *dga, unsigned short *dgt,
    unsigned short *dua, unsigned short *dut, size_t strip, int nt) {
  panel_job j;
  memset(&j,0,sizeof j); j.r=r; j.c=c; j.op=v2 ? PANEL_SWIGLU2 : PANEL_SWIGLU1; j.x=g; j.u=u;
  j.dy=dy; j.pa[0]=dga; j.pb[0]=dgt; j.pa[1]=dua; j.pb[1]=dut; j.at=1; j.strip=strip; j.nt=nt;
  return panel4_run(pool,&j,NULL);
}
"""
src = src.replace(anchor, V4 + anchor)

# the shipped two-pass writer with the same store kinds (j->nt; 0 in the shipped entries): bit moves only
reps = [
    (
        "    _mm512_storeu_si512(j->pa[k]+panel_a_at(j->c,i,c0+q),CVT(_mm512_loadu_ps(v+q),_mm512_loadu_ps(v+q+16))); \\\n",
        "    { __m512i h_=CVT(_mm512_loadu_ps(v+q),_mm512_loadu_ps(v+q+16)); \\\n"
        "      if (j->nt&1) _mm512_stream_si512((void *)(j->pa[k]+panel_a_at(j->c,i,c0+q)),h_); \\\n"
        "      else _mm512_storeu_si512(j->pa[k]+panel_a_at(j->c,i,c0+q),h_); } \\\n",
    ),
    (
        "      else _mm512_storeu_si512(j->pb[k]+panel_b_at(j->r,r0+2*p,16*jb),pr);\n",
        "      else if (j->nt&2) _mm512_stream_si512((void *)(j->pb[k]+panel_b_at(j->r,r0+2*p,16*jb)),pr);\n"
        "      else _mm512_storeu_si512(j->pb[k]+panel_b_at(j->r,r0+2*p,16*jb),pr);\n",
    ),
    (
        "    if (j->at) kb512_transpose16(wd,16,(float *)(void *)(j->pb[k]+panel_a_at(j->r,16*jb,r0)),16);\n",
        "    if (j->at && (j->nt&2)) { _Alignas(64) float tr_[256]; size_t p_; kb512_transpose16(wd,16,tr_,16);\n"
        "      for (p_=0;p_<16;p_++) _mm512_stream_si512((void *)(j->pb[k]+panel_a_at(j->r,16*jb,r0)+p_*32),_mm512_load_si512(tr_+16*p_)); }\n"
        "    else if (j->at) kb512_transpose16(wd,16,(float *)(void *)(j->pb[k]+panel_a_at(j->r,16*jb,r0)),16);\n",
    ),
    (
        "  if (j->bad) j->bad[t]=(unsigned char)bad;\n}\n#ifdef BCIR_DT_PANELS_PROBE\n",
        "#ifdef BCIR_MM_DISPATCH\n  if (j->nt) _mm_sfence();\n#endif\n  if (j->bad) j->bad[t]=(unsigned char)bad;\n}\n#ifdef BCIR_DT_PANELS_PROBE\n",
    ),
]
for old, new in reps:
    assert src.count(old) == 1, old[:70]
    src = src.replace(old, new)
src = src.replace(
    anchor,
    r"""int bcir_tensor_silu_mul_panels_nt_pool(bcir_tensor_pool *pool, size_t r, size_t c, const float *x,
    const float *u, unsigned short *a, unsigned short *b, int *bad, int nt);
int bcir_tensor_silu_mul_panels_nt_pool(bcir_tensor_pool *pool, size_t r, size_t c, const float *x,
    const float *u, unsigned short *a, unsigned short *b, int *bad, int nt) {
  panel_job j;
  memset(&j,0,sizeof j); j.r=r; j.c=c; j.op=PANEL_SILU2; j.x=x; j.u=u; j.pa[0]=a; j.pb[0]=b; j.nt=nt;
  return panel_run(pool,&j,bad);
}
int bcir_tensor_swiglu_backward_panels_nt_pool(bcir_tensor_pool *pool, size_t r, size_t c,
    const float *g, const float *u, const float *dy, unsigned short *dga, unsigned short *dgt,
    unsigned short *dua, unsigned short *dut, int nt);
int bcir_tensor_swiglu_backward_panels_nt_pool(bcir_tensor_pool *pool, size_t r, size_t c,
    const float *g, const float *u, const float *dy, unsigned short *dga, unsigned short *dgt,
    unsigned short *dua, unsigned short *dut, int nt) {
  panel_job j;
  memset(&j,0,sizeof j); j.r=r; j.c=c; j.op=PANEL_SWIGLU2; j.x=g; j.u=u; j.dy=dy;
  j.pa[0]=dga; j.pb[0]=dgt; j.pa[1]=dua; j.pb[1]=dut; j.at=1; j.nt=nt;
  return panel_run(pool,&j,NULL);
}
"""
    + anchor,
)
Path(sys.argv[2]).write_text(src)
print("v4 inserted")
