"""BWD-SWEEP-0: a copy of the tree's unit whose banded backward (AVX-512) takes dV and dK inside its 16-row
groups' sweep instead of two transposed walks over the whole head: each group adds its rows' terms to dV from
its band before the score adjoints overwrite it, and to dK from the score adjoints after -- att512_btr_block over
the group's 16 terms, every block the group reaches, its diagonal terms as the walk's, from +0.0 in the group
holding the block (the walk's first band). Each accumulator takes the same terms in the same (ascending) order,
stored and loaded exactly between groups. usage: proto_bwd.py SRC OUT"""

import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()
anchor = '__attribute__((target("avx512f"))) BCIR_MM_INLINE void att512_bgroup_any(int f, size_t R, size_t d, float scale,\n'
assert src.count(anchor) == 1
bband = """/* BWD-SWEEP: one group's terms (rows R..R+15) of a transposed product over the bands -- every 8-row block of out
 * the group reaches (j0 < R+16), the walk's register block over the band [R, R+16), from +0.0 where the group
 * holds the block's diagonal and `first` */
__attribute__((target("avx512f"))) BCIR_MM_INLINE void att512_bband(int f, size_t R, size_t t, size_t d,
    const float *pb, const float *x, size_t ldx, float *out, size_t ldo, int first) {
  size_t c0,j0;
  for (c0=0;c0<d;c0+=48) {
    size_t nc=d-c0<48 ? d-c0 : 48;
    __mmask16 m0=att512_lanes(nc),m1=att512_lanes(nc>16 ? nc-16 : 0),m2=att512_lanes(nc>32 ? nc-32 : 0);
    for (j0=0;j0<R+16;j0+=8) {
      int zero=first && j0>=R;
      if (nc>32) att512_btr_block(f,3,j0,R,R+16,c0,t,pb,x,ldx,out,ldo,zero,m0,m1,m2);
      else if (nc>16) att512_btr_block(f,2,j0,R,R+16,c0,t,pb,x,ldx,out,ldo,zero,m0,m1,m2);
      else att512_btr_block(f,1,j0,R,R+16,c0,t,pb,x,ldx,out,ldo,zero,m0,m1,m2);
    }
  }
}
"""
src = src.replace(anchor, bband + anchor)
old_sig = """    const float *dy, size_t ldy, const float *v, const float *k, size_t ldk, float *band, float *kb, size_t kept,
    float *dq, size_t ldq) {
  _Alignas(64) float dyt[BCIR_ATT_TQ*16];"""
assert src.count(old_sig) == 1
src = src.replace(
    old_sig,
    """    const float *dy, size_t ldy, const float *v, const float *k, size_t ldk, float *band, float *kb, size_t kept,
    float *dq, size_t ldq, size_t t, const float *ph, const float *q, float *dv, float *dk, int first) {
  _Alignas(64) float dyt[BCIR_ATT_TQ*16];""",
)
old_p1 = """  for (jb=0;jb<=R;jb+=16) {
    __m512 acc[16];
    if (jb<kept) for (kk=0;kk<16;kk++) acc[kk]=_mm512_loadu_ps(kb+(jb+kk)*16);"""
assert src.count(old_p1) == 1
src = src.replace(
    old_p1,
    "  att512_bband(f,R,t,d,ph,dy,ldy,dv,ldk,first);  /* dV: the group's terms, from its probabilities */\n"
    + old_p1,
)
old_dq = """  for (c0=0;c0<d;c0+=48) {
    size_t nc=d-c0<48 ? d-c0 : 48;
    __mmask16 m0=att512_lanes(nc),m1=att512_lanes(nc>16 ? nc-16 : 0),m2=att512_lanes(nc>32 ? nc-32 : 0);
    for (rb=0;rb<16;rb+=8) {
      if (nc>32) att512_tctx_block(f,3,R,rb,c0,band,(size_t)-1,sc,k,ldk,dq,ldq,m0,m1,m2);"""
assert src.count(old_dq) == 1
src = src.replace(
    old_dq,
    "  att512_bband(f,R,t,d,ph,q,ldy,dk,ldk,first);  /* dK: the group's terms, from its score adjoints */\n"
    + old_dq,
)
for nm, fv in (("att512_bgroup", "0"), ("att512_bgroup_fma", "1")):
    old = """__attribute__((target("avx512f"))) static void %s(size_t R, size_t d, float scale, const float *dy,
    size_t ldy, const float *v, const float *k, size_t ldk, float *band, float *kb, size_t kept, float *dq,
    size_t ldq) {
  att512_bgroup_any(%s,R,d,scale,dy,ldy,v,k,ldk,band,kb,kept,dq,ldq);""" % (nm, fv)
    assert src.count(old) == 1, nm
    src = src.replace(
        old,
        """__attribute__((target("avx512f"))) static void %s(size_t R, size_t d, float scale, const float *dy,
    size_t ldy, const float *v, const float *k, size_t ldk, float *band, float *kb, size_t kept, float *dq,
    size_t ldq, size_t t, const float *ph, const float *q, float *dv, float *dk, int first) {
  att512_bgroup_any(%s,R,d,scale,dy,ldy,v,k,ldk,band,kb,kept,dq,ldq,t,ph,q,dv,dk,first);"""
        % (nm, fv),
    )
old_head = """  (a->fused ? att512_btransposed_fma : att512_btransposed)(t,d,ph,dyh,ldh,dvh,ldk,first);
  for (R=0;R<t;R+=16)
    (a->fused ? att512_bgroup_fma : att512_bgroup)(R,d,scale,dyh,ldh,vk,kk,ldk,ph+R*t,kb,kept,dqh,ldh);
  (a->fused ? att512_btransposed_fma : att512_btransposed)(t,d,ph,qh,ldh,dkh,ldk,first);
"""
assert src.count(old_head) == 1
src = src.replace(
    old_head,
    """  for (R=0;R<t;R+=16)
    (a->fused ? att512_bgroup_fma : att512_bgroup)(R,d,scale,dyh,ldh,vk,kk,ldk,ph+R*t,kb,kept,dqh,ldh,t,ph,qh,dvh,dkh,first);
""",
)
Path(sys.argv[2]).write_text(src)
print("wrote", sys.argv[2])
