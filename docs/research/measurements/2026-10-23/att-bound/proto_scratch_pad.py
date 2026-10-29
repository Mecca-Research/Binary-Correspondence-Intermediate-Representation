"""ATT-BOUND's SCRATCH-PAD prototype (measured, not taken): usage proto_scratch_pad.py SRC_TENSOR_C OUT_TENSOR_C.
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
# 2 the scratch forward's strips at a padded stride in the region, where one is lent
sub(
    """static void att_head_scratch(const att_job *a, size_t r, size_t h) {
  size_t t=a->t,rows=att_strip_rows(t),i0;
  float *s=a->p+(r*a->nh+h)*att_scratch_rows(t)*t;
""",
    """static void att_head_scratch(const att_job *a, size_t r, size_t h) {
  size_t t=a->t,rows=att_strip_rows(t),i0;
  float *s=a->p+(r*a->nh+h)*att_scratch_rows(t)*t;
#ifdef BCIR_MM_DISPATCH
  if (a->keep && a->variant==4) {  /* SCRATCH-PAD prototype */
    size_t lds=(t+15)/16*16+16,room=3*((t*a->d+15)/16*16)+(t/64+(t%64!=0))*32*64;
    if (rows*lds>room) rows=32;
    for (i0=0;i0<t;i0+=rows) att_strip_ld(a,r,h,i0,t-i0<rows ? t : i0+rows,a->keep,lds);
    if (a->bad) att_verdict(a,r,h);
    return;
  }
#endif
""",
)
# 3 the scratch forward's sub-job on the copies: its strip after the four copies
sub(
    """  s.b=1; s.nh=1; s.nk=1; s.group=1; s.q=qc; s.k=kc; s.v=vc; s.y=yc; s.bad=NULL; s.ws=NULL;
""",
    """  s.b=1; s.nh=1; s.nk=1; s.group=1; s.q=qc; s.k=kc; s.v=vc; s.y=yc; s.bad=NULL; s.ws=NULL;
  s.keep=a->scratch ? yc+f : NULL;  /* SCRATCH-PAD prototype: a padded strip after the four copies */
""",
)
Path(sys.argv[2]).write_text(src)
print("SCRATCH-PAD prototype written")
