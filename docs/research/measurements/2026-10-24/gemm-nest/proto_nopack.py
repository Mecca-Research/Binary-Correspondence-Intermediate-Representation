"""NOPACK-0's prototype: the k-blocked AVX-512 kernel reads an untransposed B's rows in place -- 48 columns a depth
through masked loads, the lanes past the block's columns +0.0 as the packed tile's padding is -- where B's rows are
whole cache lines apart and B starts on one, instead of packing a tile per (chunk, column block, K block); every
fused chain the same terms in the same order. usage: proto_nopack.py SRC OUT"""

import sys
from pathlib import Path

t = Path(sys.argv[1]).read_text()


def once(old, new):
    global t
    assert t.count(old) == 1, (old[:70], t.count(old))
    t = t.replace(old, new)


# the block: a tile stride and the lanes its columns reach
once(
    """__attribute__((target("avx512f"))) BCIR_MM_INLINE void kb512_block(size_t mr, int nv, const mm_args *x,
    const float *bt, size_t i, size_t jb, size_t nj, size_t lb, size_t nl) {
  __m512 acc0[8],acc1[8],acc2[8],alpha,beta; size_t r,l; __mmask16 keep,scale;""",
    """__attribute__((target("avx512f"))) BCIR_MM_INLINE void kb512_block(size_t mr, int nv, const mm_args *x,
    const float *bt, size_t ldt, size_t i, size_t jb, size_t nj, size_t lb, size_t nl) {
  __m512 acc0[8],acc1[8],acc2[8],alpha,beta; size_t r,l; __mmask16 keep,scale;
  __mmask16 t0=(__mmask16)(nj>=16 ? 0xffffu : (1u<<nj)-1u);
  __mmask16 t1=(__mmask16)(nj>=32 ? 0xffffu : nj>16 ? (1u<<(nj-16))-1u : 0u);
  __mmask16 t2=(__mmask16)(nj>=48 ? 0xffffu : nj>32 ? (1u<<(nj-32))-1u : 0u);""",
)
once(
    """    for (l=0;l<nl;l++,ap+=lda) {
      __m512 b0=_mm512_load_ps(bt+l*48),b1=_mm512_setzero_ps(),b2=_mm512_setzero_ps();
      if (nv>1) b1=_mm512_load_ps(bt+l*48+16);
      if (nv>2) b2=_mm512_load_ps(bt+l*48+32);
      _mm_prefetch((const char *)(ap+BCIR_MM_KB_PREFETCH*lda),_MM_HINT_T0);""",
    """    for (l=0;l<nl;l++,ap+=lda) {
      __m512 b0=_mm512_maskz_loadu_ps(t0,bt+l*ldt),b1=_mm512_setzero_ps(),b2=_mm512_setzero_ps();
      if (nv>1) b1=_mm512_maskz_loadu_ps(t1,bt+l*ldt+16);
      if (nv>2) b2=_mm512_maskz_loadu_ps(t2,bt+l*ldt+32);
      _mm_prefetch((const char *)(ap+BCIR_MM_KB_PREFETCH*lda),_MM_HINT_T0);""",
)
once(
    """    for (l=0;l<nl;l++) {
      __m512 b0=_mm512_load_ps(bt+l*48),b1=_mm512_setzero_ps(),b2=_mm512_setzero_ps();
      if (nv>1) b1=_mm512_load_ps(bt+l*48+16);
      if (nv>2) b2=_mm512_load_ps(bt+l*48+32);
      for (r=0;r<mr;r++) {
        __m512 av=_mm512_set1_ps(ap[r*lda+l]);""",
    """    for (l=0;l<nl;l++) {
      __m512 b0=_mm512_maskz_loadu_ps(t0,bt+l*ldt),b1=_mm512_setzero_ps(),b2=_mm512_setzero_ps();
      if (nv>1) b1=_mm512_maskz_loadu_ps(t1,bt+l*ldt+16);
      if (nv>2) b2=_mm512_maskz_loadu_ps(t2,bt+l*ldt+32);
      for (r=0;r<mr;r++) {
        __m512 av=_mm512_set1_ps(ap[r*lda+l]);""",
)
once(
    """__attribute__((target("avx512f"))) BCIR_MM_INLINE void kb512_rows(int nv, const mm_args *x, const float *bt,
    size_t r0, size_t r1, size_t jb, size_t nj, size_t lb, size_t nl) {
  size_t i;
  for (i=r0;i+8<=r1;i+=8) kb512_block(8,nv,x,bt,i,jb,nj,lb,nl);
  for (;i+4<=r1;i+=4) kb512_block(4,nv,x,bt,i,jb,nj,lb,nl);
  for (;i<r1;i++) kb512_block(1,nv,x,bt,i,jb,nj,lb,nl);
}""",
    """__attribute__((target("avx512f"))) BCIR_MM_INLINE void kb512_rows(int nv, const mm_args *x, const float *bt,
    size_t ldt, size_t r0, size_t r1, size_t jb, size_t nj, size_t lb, size_t nl) {
  size_t i;
  for (i=r0;i+8<=r1;i+=8) kb512_block(8,nv,x,bt,ldt,i,jb,nj,lb,nl);
  for (;i+4<=r1;i+=4) kb512_block(4,nv,x,bt,ldt,i,jb,nj,lb,nl);
  for (;i<r1;i++) kb512_block(1,nv,x,bt,ldt,i,jb,nj,lb,nl);
}""",
)
once(
    """      nl=kblock_next(x->k-lb);
      kb512_pack(x,bt,jb,nj,lb,nl);
      if (nj>32) kb512_rows(3,x,bt,r->r0,r->r1,jb,nj,lb,nl);  /* MM-ROWS: the vectors the block reaches */
      else if (nj>16) kb512_rows(2,x,bt,r->r0,r->r1,jb,nj,lb,nl);
      else kb512_rows(1,x,bt,r->r0,r->r1,jb,nj,lb,nl);""",
    """      const float *tile=bt; size_t ldt=48;
      nl=kblock_next(x->k-lb);
      if (!x->tb && x->ldb%16==0 && ((uintptr_t)x->b&63)==0) { tile=x->b+lb*x->ldb+jb; ldt=x->ldb; }  /* NOPACK */
      else kb512_pack(x,bt,jb,nj,lb,nl);
      if (nj>32) kb512_rows(3,x,tile,ldt,r->r0,r->r1,jb,nj,lb,nl);  /* MM-ROWS: the vectors the block reaches */
      else if (nj>16) kb512_rows(2,x,tile,ldt,r->r0,r->r1,jb,nj,lb,nl);
      else kb512_rows(1,x,tile,ldt,r->r0,r->r1,jb,nj,lb,nl);""",
)
Path(sys.argv[2]).write_text(t)
print("NOPACK prototype:", sys.argv[2])
