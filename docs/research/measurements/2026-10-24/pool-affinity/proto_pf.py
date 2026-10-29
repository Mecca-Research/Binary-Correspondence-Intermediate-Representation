"""BWD-PF-0: a copy of the tree's unit whose banded backward prefetches P's lines ahead of its streams (the L2
streamer stops at each 4 KiB page): in the banded walks' blocks (att512_btr_block over a band ib.., keys j0..j0+7)
the lines PF_D blocks of eight keys ahead in each of the band's two 16-row runs; in a group's pass 0 the band's
line PF_G keys ahead of each 16-key block. Prefetches change no value. usage: proto_pf.py SRC OUT MODE
(MODE walk | group | both)"""

import sys
from pathlib import Path

s = Path(sys.argv[1]).read_text()
mode = sys.argv[3]
hdr = "#ifndef PF_D\n#define PF_D 4\n#endif\n#ifndef PF_G\n#define PF_G 64\n#endif\n#ifndef PF_HINT\n#define PF_HINT _MM_HINT_T0\n#endif\n"
anchor = '__attribute__((target("avx512f"))) BCIR_MM_INLINE void att512_btr_block(int f, int nv, size_t j0, size_t ib,'
assert s.count(anchor) == 1
s = s.replace(anchor, hdr + anchor)
if mode in ("walk", "both"):
    old = """  for (i=diag ? j0+7 : ib;i<ie;i++) {  /* all eight rows take term i */
    const float *xi=x+i*ldx+c0,*mi=pb+(i/16)*16*t+i%16+j0*16;"""
    assert s.count(old) == 1
    new = (
        """  if (c0==0) {  /* BWD-PF: this band's lines PF_D blocks ahead, both of its 16-row runs */
    size_t jp=j0+8*PF_D,b0=ib/16,bl;
    for (bl=b0;bl*16<ie && bl<b0+2;bl++) if (jp+7<(bl+1)*16 || jp<ie)
      for (r=0;r<8;r++) if (jp+r<t) _mm_prefetch((const char *)(pb+bl*16*t+(jp+r)*16),PF_HINT);
  }
"""
        + old
    )
    s = s.replace(old, new)
if mode in ("group", "both"):
    old = """  att512_tq(d,dy+R*ldy,ldy,dyt);
  for (jb=0;jb<=R;jb+=16) {
    __m512 acc[16];
    att512_bdp_block(f,d,dyt,v+jb*ldk,ldk,acc);"""
    assert s.count(old) == 1
    new = """  att512_tq(d,dy+R*ldy,ldy,dyt);
  for (jb=0;jb<=R;jb+=16) {
    __m512 acc[16];
    { size_t jp; for (jp=jb+PF_G;jp<jb+PF_G+16 && jp<=R+15;jp++) _mm_prefetch((const char *)(band+jp*16),PF_HINT); }
    att512_bdp_block(f,d,dyt,v+jb*ldk,ldk,acc);"""
    s = s.replace(old, new)
Path(sys.argv[2]).write_text(s)
print("wrote", sys.argv[2], mode)
