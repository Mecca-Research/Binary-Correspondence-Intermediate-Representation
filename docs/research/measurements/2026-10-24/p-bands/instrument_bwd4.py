"""A copy of bcir_tensor.c with each head's backward timed a phase at a time (TSC cycles, summed per phase over
the call and its threads), thread-local counters flushed once a task (instrument_bwd3.py's shared atomics cost
a contended line every tile): [0] dV = P^T dY, [1] pass 0's tiles' register blocks, [2] pass 0's row dots,
[3] pass 1's score adjoints, [4] dQ = dS K, [5] dK = dS^T Q, [6] a contiguous task whole, [7] pass 1's tiles
evaluated again, [8] the tiles' zero fill, [9] the tiles' value panels, [10] a task's K and V copies in (and
its dK and dV v1 rows zeroed), [11] a head's Q and dY copies in (and its dQ v1 rows zeroed), [12] a head's dQ
copy out, [13] a task's dK and dV copies out. usage: instrument_bwd4.py SRC OUT"""

import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()
out = Path(sys.argv[2])
hdr = (
    "#include <x86intrin.h>\nunsigned long long bcir_bw_cycles[16];\n"
    "static __thread unsigned long long bw_local[16];\n"
    "#define BW_ADD(k,c) (bw_local[k]+=__rdtsc()-(c))\n"
    "static void bw_flush(void) {\n  int k;\n  for (k=0;k<16;k++) if (bw_local[k]) {\n"
    "    __atomic_fetch_add(&bcir_bw_cycles[k],bw_local[k],__ATOMIC_RELAXED); bw_local[k]=0;\n  }\n}\n"
)
anchor = "static void att_head_backward(const att_job *a, size_t r, size_t h, const float *vt, float *tp) {\n"
assert src.count(anchor) == 1
src = src.replace(anchor, hdr + anchor, 1)
edits = [
    (  # dV
        "    x.c=dv+(r*t*nk+kh)*d; x.ldc=ldk; x.mode=MM_KGE;\n"
        "    att_rows(variant,a->fused,&x,t,ATT_TRANSPOSED,first);\n",
        "    x.c=dv+(r*t*nk+kh)*d; x.ldc=ldk; x.mode=MM_KGE;\n"
        "    { unsigned long long c=__rdtsc(); att_rows(variant,a->fused,&x,t,ATT_TRANSPOSED,first); BW_ADD(0,c); }\n",
    ),
    (
        "        if (!pass || (!a->keep && jt>=BCIR_ATT_KEEP)) {\n",
        "        unsigned long long ct=__rdtsc();\n        if (!pass || (!a->keep && jt>=BCIR_ATT_KEEP)) {\n",
    ),
    (
        "          memset(tile,0,rows*BCIR_ATT_COLS*sizeof(float));\n",
        "          memset(tile,0,rows*BCIR_ATT_COLS*sizeof(float));\n          BW_ADD(8,ct); ct=__rdtsc();\n",
    ),
    (
        "            att512_panel(cols,d,vk+jt*ldk,ldk,tp);\n",
        "            att512_panel(cols,d,vk+jt*ldk,ldk,tp);\n            BW_ADD(9,ct); ct=__rdtsc();\n",
    ),
    (
        "        i=0;\n#ifdef BCIR_MM_DISPATCH\n        if (!pass && variant==4) for (;i+16<=rows;i+=16)\n",
        "        BW_ADD(pass ? 7 : 1,ct); ct=__rdtsc();\n"
        "        i=0;\n#ifdef BCIR_MM_DISPATCH\n        if (!pass && variant==4) for (;i+16<=rows;i+=16)\n",
    ),
    (
        "            else pi[j]=pi[j]*(dp-dot[i])*scale;\n          }\n        }\n",
        "            else pi[j]=pi[j]*(dp-dot[i])*scale;\n          }\n        }\n        BW_ADD(pass ? 3 : 2,ct);\n",
    ),
    (  # dQ and dK
        "    x.c=dq+(r*t*nh+h)*d; x.ldc=ldh; x.mode=MM_KLE;\n"
        "    att_rows(variant,a->fused,&x,t,ATT_CONTEXT,1);\n"
        "    x.ta=1; x.b=qh; x.ldb=ldh; x.c=dk+(r*t*nk+kh)*d; x.ldc=ldk; x.mode=MM_KGE;\n"
        "    att_rows(variant,a->fused,&x,t,ATT_TRANSPOSED,first);\n",
        "    x.c=dq+(r*t*nh+h)*d; x.ldc=ldh; x.mode=MM_KLE;\n"
        "    { unsigned long long c=__rdtsc(); att_rows(variant,a->fused,&x,t,ATT_CONTEXT,1); BW_ADD(4,c); }\n"
        "    x.ta=1; x.b=qh; x.ldb=ldh; x.c=dk+(r*t*nk+kh)*d; x.ldc=ldk; x.mode=MM_KGE;\n"
        "    { unsigned long long c=__rdtsc(); att_rows(variant,a->fused,&x,t,ATT_TRANSPOSED,first); BW_ADD(5,c); }\n",
    ),
    (  # a task's K and V in
        "  att_rows_in(kc,a->k+(r*t*nk+kh)*d,t,d,nk*d);\n  att_rows_in(vc,a->v+(r*t*nk+kh)*d,t,d,nk*d);\n"
        "  for (i=v1;i<t;i++) { memset(dkc+i*d,0,d*sizeof(float)); memset(dvc+i*d,0,d*sizeof(float)); }\n"
        "  s.b=1;",
        "  { unsigned long long c=__rdtsc();\n"
        "  att_rows_in(kc,a->k+(r*t*nk+kh)*d,t,d,nk*d);\n  att_rows_in(vc,a->v+(r*t*nk+kh)*d,t,d,nk*d);\n"
        "  for (i=v1;i<t;i++) { memset(dkc+i*d,0,d*sizeof(float)); memset(dvc+i*d,0,d*sizeof(float)); }\n"
        "  BW_ADD(10,c); }\n  s.b=1;",
    ),
    (  # a head's Q and dY in
        "      att_rows_in(qc,a->q+(r*t*nh+h)*d,t,d,nh*d);\n      att_rows_in(dyc,a->dy+(r*t*nh+h)*d,t,d,nh*d);\n"
        "      for (i=v1;i<t;i++) memset(dqc+i*d,0,d*sizeof(float));\n",
        "      { unsigned long long c=__rdtsc();\n"
        "      att_rows_in(qc,a->q+(r*t*nh+h)*d,t,d,nh*d);\n      att_rows_in(dyc,a->dy+(r*t*nh+h)*d,t,d,nh*d);\n"
        "      for (i=v1;i<t;i++) memset(dqc+i*d,0,d*sizeof(float));\n      BW_ADD(11,c); }\n",
    ),
    (  # a head's dQ out
        "      att_rows_out(a->dq+(r*t*nh+h)*d,dqc,t,d,nh*d);\n",
        "      { unsigned long long c=__rdtsc(); att_rows_out(a->dq+(r*t*nh+h)*d,dqc,t,d,nh*d); BW_ADD(12,c); }\n",
    ),
    (  # a task's dK and dV out
        "  att_rows_out(a->dk+(r*t*nk+kh)*d,dkc,t,d,nk*d);\n  att_rows_out(a->dv+(r*t*nk+kh)*d,dvc,t,d,nk*d);\n}\n",
        "  { unsigned long long c=__rdtsc();\n"
        "  att_rows_out(a->dk+(r*t*nk+kh)*d,dkc,t,d,nk*d);\n  att_rows_out(a->dv+(r*t*nk+kh)*d,dvc,t,d,nk*d);\n"
        "  BW_ADD(13,c); }\n}\n",
    ),
    (  # the in-place task's flush
        "    for (h=first;h<first+a->group;h++) att_head_backward(a,r,h,vt,tp);\n  }\n}\n",
        "    for (h=first;h<first+a->group;h++) att_head_backward(a,r,h,vt,tp);\n  }\n  bw_flush();\n}\n",
    ),
    (  # a contiguous task whole, and the flush
        "static void att_task_backward_w(void *ctx, size_t task, unsigned worker) {\n"
        "  att_job *a=(att_job *)ctx;\n"
        "  if (!a->ws) { att_task_backward(ctx,task); return; }\n"
        "  INSTANCE_SAW(INSTANCE_ATTN,a->variant);\n"
        "  att_task_backward_contig(a,task,worker);\n",
        "static void att_task_backward_w(void *ctx, size_t task, unsigned worker) {\n"
        "  att_job *a=(att_job *)ctx;\n"
        "  if (!a->ws) { att_task_backward(ctx,task); return; }\n"
        "  INSTANCE_SAW(INSTANCE_ATTN,a->variant);\n"
        "  { unsigned long long c=__rdtsc(); att_task_backward_contig(a,task,worker); BW_ADD(6,c); }\n"
        "  bw_flush();\n",
    ),
]
for old, new in edits:
    assert src.count(old) == 1, old[:70]
    src = src.replace(old, new, 1)
out.write_text(src)
print("instrumented", out)
