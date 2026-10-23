"""ATT-BOUND's phase instrumentation: ../../2026-10-22/attn-rate/instrument.py's scratch copy of the tensor unit
(each scratch strip's scores, softmax and context timed in TSC cycles), and inside the AVX-512 softmax its max
pass, its exponential-and-sum pass and its division timed apart -- bcir_ar_cycles [0] scores, [1] softmax,
[2] context, [3] max, [4] exp+sum, [5] division. usage: instrument_phases.py SRC_BCIR_TENSOR_C OUT_C"""

import runpy
import sys
from pathlib import Path

src_path, out_path = sys.argv[1], sys.argv[2]
inst = Path(__file__).resolve().parent.parent.parent / "2026-10-22" / "attn-rate" / "instrument.py"
sys.argv = [str(inst), src_path, out_path]
runpy.run_path(str(inst), run_name="__main__")
src = (
    Path(out_path)
    .read_text()
    .replace("unsigned long long bcir_ar_cycles[4];", "unsigned long long bcir_ar_cycles[8];")
)
old = """    att512_rowchain(ld,i,rows,maxv,0);
    for (u=0;u<16;u++) maxv[u]*=scale;
    if (a->exp2) att512_expsum(ld,i,rows,scale,maxv,sum);
    else {
"""
new = """    { unsigned long long c3=__rdtsc(); att512_rowchain(ld,i,rows,maxv,0); __atomic_fetch_add(&bcir_ar_cycles[3],__rdtsc()-c3,__ATOMIC_RELAXED); }
    for (u=0;u<16;u++) maxv[u]*=scale;
    { unsigned long long c4=__rdtsc(); if (a->exp2) att512_expsum(ld,i,rows,scale,maxv,sum); __atomic_fetch_add(&bcir_ar_cycles[4],__rdtsc()-c4,__ATOMIC_RELAXED); }
    if (!a->exp2) {
"""
assert src.count(old) == 1
src = src.replace(old, new)
old2 = "    for (u=0;u<16;u++) { float *pi=rows+u*ld; for (j=0;j<=i+u;j++) pi[j]/=sum[u]; }\n"
new2 = (
    "    { unsigned long long c5=__rdtsc(); for (u=0;u<16;u++) { float *pi=rows+u*ld; for (j=0;j<=i+u;j++) pi[j]/=sum[u]; } "
    "__atomic_fetch_add(&bcir_ar_cycles[5],__rdtsc()-c5,__ATOMIC_RELAXED); }\n"
)
assert src.count(old2) == 1
src = src.replace(old2, new2)
Path(out_path).write_text(
    "#include <x86intrin.h>\nextern unsigned long long bcir_ar_cycles[8];\n" + src
)
print("instrumented (8 counters)", out_path)
