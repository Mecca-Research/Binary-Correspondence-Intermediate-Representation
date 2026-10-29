"""ATT-BOUND's SOFTMAX-DIV prototype (measured, not taken): usage proto_softmax_div.py SRC_TENSOR_C OUT_TENSOR_C.
The AVX-512 softmax's normalising pass -- each row's probabilities divided by its sum -- in 16-lane
vdivps instead of the baseline-ISA loop the compilers vectorise to 4-lane divps; IEEE division is
correctly rounded at every width, so every quotient is the same."""

import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()


def sub(old, new):
    global src
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"anchor found {n} times: {old[:80]!r}")
    src = src.replace(old, new)


sub(
    """static void att_softmax_block(const att_job *a, float *s, size_t ld, size_t i0, size_t n) {
""",
    """#ifdef BCIR_MM_DISPATCH
/* SOFTMAX-DIV prototype: rows i0 .. i0+15's probabilities j <= i divided by their sums, 16 lanes */
__attribute__((target("avx512f"))) static void att512_divide16(size_t ld, size_t i0, float *rows, const float *sum) {
  size_t u;
  for (u=0;u<16;u++) {
    float *pi=rows+u*ld; size_t n=i0+u+1,j=0; __m512 sv=_mm512_set1_ps(sum[u]);
    for (;j+16<=n;j+=16) _mm512_storeu_ps(pi+j,_mm512_div_ps(_mm512_loadu_ps(pi+j),sv));
    if (j<n) { __mmask16 k=att512_lanes(n-j); _mm512_mask_storeu_ps(pi+j,k,_mm512_maskz_div_ps(k,_mm512_maskz_loadu_ps(k,pi+j),sv)); }
  }
}
#endif
static void att_softmax_block(const att_job *a, float *s, size_t ld, size_t i0, size_t n) {
""",
)
sub(
    """    for (u=0;u<16;u++) { float *pi=rows+u*ld; for (j=0;j<=i+u;j++) pi[j]/=sum[u]; }
""",
    """    att512_divide16(ld,i,rows,sum);  /* SOFTMAX-DIV prototype */
""",
)
Path(sys.argv[2]).write_text(src)
print("prototype written")
