"""instrument_bwd4.py's phases over a unit with P-BANDS: the row path's phases as there, and the banded backward's --
[0] dV = P^T dY (the banded walk), [1] the 16-row groups (their adjoint blocks and dots, score adjoints and dQ),
[5] dK = dS^T Q (the banded walk) -- with [6] a contiguous task whole and [10]..[13] its copies as there.
usage: instrument_bands.py SRC OUT  (instrument_bwd4.py beside it)"""

import runpy
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
src, out = sys.argv[1], Path(sys.argv[2])
sys.argv = [str(here / "instrument_bwd4.py"), src, str(out)]
runpy.run_path(str(here / "instrument_bwd4.py"), run_name="__main__")
s = out.read_text()
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
)
assert s.count(old) == 1, "the banded backward's calls"
s = s.replace(old, new)
hdr = "static void att_head_backward_bands512(const att_job *a, size_t r, size_t h, float *stack, size_t room) {\n"
assert s.count(hdr) == 1
# the counters' header must precede the banded head: move instrument_bwd4's block up to it
block_start = s.index("#include <x86intrin.h>\nunsigned long long bcir_bw_cycles[16];\n")
block_end = s.index(
    "static void att_head_backward(const att_job *a, size_t r, size_t h, const float *vt, float *tp) {\n"
)
block = s[block_start:block_end]
s = s[:block_start] + s[block_end:]
s = s.replace(hdr, block + hdr)
out.write_text(s)
print("instrumented", out)
