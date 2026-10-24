"""ATT-BOUND's FWD-STREAM prototype (measured, not taken): usage proto_stream.py SRC_TENSOR_C OUT_TENSOR_C.
Each stored forward under LOWER on ATT-CONTIG's copies (AVX-512) computes a 32-row strip in its region's
kept-tiles room (unused by the forward), then streams the strip's rows j < roundup16(i+1) to p with
non-temporal stores (no read-for-ownership of p's lines), an sfence after the head's last strip."""

import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()


def sub(old, new, count=1):
    global src
    n = src.count(old)
    if n != count:
        raise SystemExit(f"anchor found {n} times, want {count}: {old[:80]!r}")
    src = src.replace(old, new)


# 1 the streaming copy, AVX-512, beside the strip forward
sub(
    """static void att_head_strips(const att_job *a, size_t r, size_t h) {
  size_t t=a->t,rows=att_strip_rows(t),i0,i;
  float *ph=a->p+(r*a->nh+h)*t*t;
#ifdef BCIR_DT_ATTENTION_PROBE
  (void)__atomic_fetch_add(&attention_strips_seen,(size_t)1,__ATOMIC_RELAXED);
#endif
""",
    """#ifdef BCIR_MM_DISPATCH
__attribute__((target("avx512f"))) static void att512_stream_rows(const float *s, size_t t, float *ph,
    size_t i0, size_t i1) {
  size_t i;
  for (i=i0;i<i1;i++) {
    const float *src=s+(i-i0)*t; float *dst=ph+i*t; size_t j=0,end=(i+16)/16*16;
    if (end>t) end=t;
    while (j<end && ((uintptr_t)(dst+j)&63)) { dst[j]=src[j]; j++; }
    for (;j+16<=end;j+=16) _mm512_stream_ps(dst+j,_mm512_loadu_ps(src+j));
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
  if (a->keep && a->lower && a->variant==4) {  /* FWD-STREAM prototype */
    for (i0=0;i0<t;i0+=32) {
      size_t i1=t-i0<32 ? t : i0+32;
      att_strip(a,r,h,i0,i1,a->keep);
      att512_stream_rows(a->keep,t,ph,i0,i1);
    }
    _mm_sfence();
    if (a->bad) att_verdict(a,r,h);
    return;
  }
#endif
""",
)
# 2 the forward's sub-job on the copies: the strip's room is the region's kept-tiles room
sub(
    """  s.b=1; s.nh=1; s.nk=1; s.group=1; s.q=qc; s.k=kc; s.v=vc; s.y=yc; s.bad=NULL; s.ws=NULL;
""",
    """  s.b=1; s.nh=1; s.nk=1; s.group=1; s.q=qc; s.k=kc; s.v=vc; s.y=yc; s.bad=NULL; s.ws=NULL;
  s.keep=qc+7*f;  /* FWD-STREAM prototype: a strip's room, the kept tiles' (unused by the forward) */
""",
)
Path(sys.argv[2]).write_text(src)
print("prototype written")
