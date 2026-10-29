"""Phase counters (TSC, thread-local, flushed a head) in a unit with the banded backward: the tree's
(MODE tree: [0] the dV walk, [1] the groups, [5] the dK walk) or BWD-SWEEP-0's (MODE sweep: in each group
[2] pass 0, [0] dV's terms, [3] pass 1, [5] dK's terms, [4] dQ). usage: instrument_sweep.py MODE SRC OUT"""

import sys
from pathlib import Path

mode, src, out = sys.argv[1], Path(sys.argv[2]).read_text(), Path(sys.argv[3])
hdr = (
    "#include <x86intrin.h>\nunsigned long long bcir_bw_cycles[16];\n"
    "static __thread unsigned long long bw_local[16];\n"
    "#define BW_ADD(k,c) (bw_local[k]+=__rdtsc()-(c))\n"
    "static void bw_flush(void) {\n  int k;\n  for (k=0;k<16;k++) if (bw_local[k]) {\n"
    "    __atomic_fetch_add(&bcir_bw_cycles[k],bw_local[k],__ATOMIC_RELAXED); bw_local[k]=0;\n  }\n}\n"
)
first_fn = '__attribute__((target("avx512f"))) BCIR_MM_INLINE void exp1_512n('
assert src.count(first_fn) == 1
src = src.replace(first_fn, hdr + first_fn, 1)
if mode == "tree":
    old = (
        "  (a->fused ? att512_btransposed_fma : att512_btransposed)(t,d,ph,dyh,ldh,dvh,ldk,first);\n"
        "  for (R=0;R<t;R+=16)\n"
        "    (a->fused ? att512_bgroup_fma : att512_bgroup)(R,d,scale,dyh,ldh,vk,kk,ldk,ph+R*t,kb,kept,dqh,ldh);\n"
        "  (a->fused ? att512_btransposed_fma : att512_btransposed)(t,d,ph,qh,ldh,dkh,ldk,first);\n"
    )
    new = (
        "  { unsigned long long c=__rdtsc(); (a->fused ? att512_btransposed_fma : att512_btransposed)(t,d,ph,dyh,ldh,dvh,ldk,first); BW_ADD(0,c); }\n"
        "  { unsigned long long c=__rdtsc(); for (R=0;R<t;R+=16)\n"
        "    (a->fused ? att512_bgroup_fma : att512_bgroup)(R,d,scale,dyh,ldh,vk,kk,ldk,ph+R*t,kb,kept,dqh,ldh); BW_ADD(1,c); }\n"
        "  { unsigned long long c=__rdtsc(); (a->fused ? att512_btransposed_fma : att512_btransposed)(t,d,ph,qh,ldh,dkh,ldk,first); BW_ADD(5,c); }\n"
        "  bw_flush();\n"
    )
    assert src.count(old) == 1
    src = src.replace(old, new)
else:
    reps = [
        (
            "  att512_tq(d,dy+R*ldy,ldy,dyt);\n  for (jb=0;jb<=R;jb+=16) {\n",
            "  unsigned long long cc=__rdtsc();\n  att512_tq(d,dy+R*ldy,ldy,dyt);\n  for (jb=0;jb<=R;jb+=16) {\n",
        ),
        (
            "  att512_bband(f,R,t,d,ph,dy,ldy,dv,ldk,first);",
            "  BW_ADD(2,cc); cc=__rdtsc();\n  att512_bband(f,R,t,d,ph,dy,ldy,dv,ldk,first); BW_ADD(0,cc); cc=__rdtsc();",
        ),
        (
            "  att512_bband(f,R,t,d,ph,q,ldy,dk,ldk,first);",
            "  BW_ADD(3,cc); cc=__rdtsc();\n  att512_bband(f,R,t,d,ph,q,ldy,dk,ldk,first); BW_ADD(5,cc); cc=__rdtsc();",
        ),
    ]
    for o, n in reps:
        assert src.count(o) == 1, o
        src = src.replace(o, n)
    # dQ: the tail of bgroup_any, timed to its end
    o = "      else att512_tctx_block(f,1,R,rb,c0,band,(size_t)-1,sc,k,ldk,dq,ldq,m0,m1,m2);\n    }\n  }\n}\n"
    assert src.count(o) == 1
    src = src.replace(o, o[:-2] + "  BW_ADD(4,cc);\n}\n")
    o = "    (a->fused ? att512_bgroup_fma : att512_bgroup)(R,d,scale,dyh,ldh,vk,kk,ldk,ph+R*t,kb,kept,dqh,ldh,t,ph,qh,dvh,dkh,first);\n"
    assert src.count(o) == 1
    src = src.replace(o, o + "  bw_flush();\n")
out.write_text(src)
print("instrumented", mode, out)
