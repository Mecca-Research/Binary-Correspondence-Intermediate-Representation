"""ATT-BOUND's padded-strip prototype (measured, not taken): usage proto_stream_pad.py SRC_TENSOR_C OUT_TENSOR_C.
FWD-STREAM's strip computed at a padded stride and copied out with ordinary stores, no sfence.
As variant A (mk_proto.py), but the strip is computed at a padded stride -- lds = the context rounded up to
16 floats, plus 16 (64 bytes) -- after the forward's four copies (Q, K, V, the context), so a 16-row
transpose's rows fall in 16 different L1 sets where a power-of-two context put them all in one; then
streamed out to p at its own stride t."""

import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()


def sub(old, new, text=None):
    global src
    t = src if text is None else text
    n = t.count(old)
    if n != 1:
        raise SystemExit(f"anchor found {n} times: {old[:80]!r}")
    t = t.replace(old, new)
    if text is None:
        src = t
    return t


# 1 att_strip_ld: att_strip with the strip's stride a parameter
start = src.index(
    "static void att_strip(const att_job *a, size_t r, size_t h, size_t i0, size_t i1, float *s) {"
)
end = src.index("/* INFER-ATTN: one (batch, query head) of a scratch forward")
body = src[start:end]
ld = body.replace(
    "static void att_strip(const att_job *a, size_t r, size_t h, size_t i0, size_t i1, float *s) {",
    "static void att_strip_ld(const att_job *a, size_t r, size_t h, size_t i0, size_t i1, float *s, size_t lds) {",
)
for old, new in (
    ("(i0,v0,d,q,ldq,k,ldk,s,t);", "(i0,v0,d,q,ldq,k,ldk,s,lds);"),
    ("memset(s+(i-i0)*t,0,i1*sizeof(float));", "memset(s+(i-i0)*lds,0,i1*sizeof(float));"),
    ("x.ldb=ldk; x.ldc=t;", "x.ldb=ldk; x.ldc=lds;"),
    ("x.c=s+(v0-i0)*t; x.mode=MM_FULL;", "x.c=s+(v0-i0)*lds; x.mode=MM_FULL;"),
    ("x.c=s+(v0-i0)*t+v0; x.mode=MM_LOWER;", "x.c=s+(v0-i0)*lds+v0; x.mode=MM_LOWER;"),
    ("att_softmax_block(a,s,t,i0,i1-i0);", "att_softmax_block(a,s,lds,i0,i1-i0);"),
    ("(i0,v0,d,s,t,v,ldk,y,ldq);", "(i0,v0,d,s,lds,v,ldk,y,ldq);"),
    ("x.alpha=1.0f; x.lda=t; x.ldb=ldk;", "x.alpha=1.0f; x.lda=lds; x.ldb=ldk;"),
    ("x.k=v0; x.a=s+(v0-i0)*t; x.b=v;", "x.k=v0; x.a=s+(v0-i0)*lds; x.b=v;"),
    ("x.k=i1-v0; x.a=s+(v0-i0)*t+v0;", "x.k=i1-v0; x.a=s+(v0-i0)*lds+v0;"),
):
    ld = sub(old, new, ld)
assert "*t" not in ld.split("{", 1)[1].replace("r*t*", "").replace("(r*t", ""), (
    "a stride t left in att_strip_ld"
)
src = src[:end] + ld + src[end:]
# 2 the streaming copy and the strip forward's streaming branch
sub(
    """static void att_head_strips(const att_job *a, size_t r, size_t h) {
  size_t t=a->t,rows=att_strip_rows(t),i0,i;
  float *ph=a->p+(r*a->nh+h)*t*t;
#ifdef BCIR_DT_ATTENTION_PROBE
  (void)__atomic_fetch_add(&attention_strips_seen,(size_t)1,__ATOMIC_RELAXED);
#endif
""",
    """#ifdef BCIR_MM_DISPATCH
__attribute__((target("avx512f"))) static void att512_stream_rows(const float *s, size_t lds, float *ph, size_t t,
    size_t i0, size_t i1) {
  size_t i;
  for (i=i0;i<i1;i++) {
    const float *src=s+(i-i0)*lds; float *dst=ph+i*t; size_t j=0,end=(i+16)/16*16;
    if (end>t) end=t;
    while (j<end && ((uintptr_t)(dst+j)&63)) { dst[j]=src[j]; j++; }
    for (;j+16<=end;j+=16) _mm512_storeu_ps(dst+j,_mm512_loadu_ps(src+j));
    for (;j<end;j++) dst[j]=src[j];
  }
}
#endif
static void att_head_strips(const att_job *a, size_t r, size_t h) {
  size_t t=a->t,rows=att_strip_rows(t),i0,i;
  float *ph=a->p+(r*a->nh+h)*t*t;
#ifdef BCIR_DT_ATTENTION_PROBE
  (void)__atomic_fetch_add(&attention_strips_seen,(size_t)1,__ATOMIC_RELAXED);
#endif
#ifdef BCIR_MM_DISPATCH
  if (a->keep && a->lower && a->variant==4) {  /* FWD-STREAM prototype C */
    size_t lds=(t+15)/16*16+16;
    for (i0=0;i0<t;i0+=32) {
      size_t i1=t-i0<32 ? t : i0+32;
      att_strip_ld(a,r,h,i0,i1,a->keep,lds);
      att512_stream_rows(a->keep,lds,ph,t,i0,i1);
    }
    if (a->bad) att_verdict(a,r,h);
    return;
  }
#endif
""",
)
# 3 the forward's sub-job: the strip after the four copies (3 copies and the kept tiles' room free)
sub(
    """  s.b=1; s.nh=1; s.nk=1; s.group=1; s.q=qc; s.k=kc; s.v=vc; s.y=yc; s.bad=NULL; s.ws=NULL;
""",
    """  s.b=1; s.nh=1; s.nk=1; s.group=1; s.q=qc; s.k=kc; s.v=vc; s.y=yc; s.bad=NULL; s.ws=NULL;
  s.keep=yc+f;  /* FWD-STREAM prototype C: a padded strip after the four copies */
""",
)
Path(sys.argv[2]).write_text(src)
print("prototype written")
