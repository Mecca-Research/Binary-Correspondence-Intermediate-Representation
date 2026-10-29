"""A copy of bcir_tensor.c (ATT-CONTIG's tree) with the binary32 attention timed a phase at a time (TSC cycles
summed over a call's tasks). Forward, bcir_fw_cycles: [0] the fill, [1] scores, [2] softmax, [3] context,
[4] the copies in, [5] the context's copy out. Backward, bcir_bw_cycles: [0] dV = P^T dY, [1] pass 0's dP
tiles, [2] pass 0's row dots, [3] pass 1's recomputed dP tiles, [4] pass 1's score adjoints, [5] dQ = dS K,
[6] dK = dS^T Q, [7] the copies in and out. usage: instrument_att3.py SRC OUT"""

import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()
out = Path(sys.argv[2])
hdr = (
    "#include <x86intrin.h>\nunsigned long long bcir_bw_cycles[8],bcir_fw_cycles[8];\n"
    "#define BW_ADD(k,c) __atomic_fetch_add(&bcir_bw_cycles[k],__rdtsc()-(c),__ATOMIC_RELAXED)\n"
    "#define FW_ADD(k,c) __atomic_fetch_add(&bcir_fw_cycles[k],__rdtsc()-(c),__ATOMIC_RELAXED)\n"
)
anchor = "static void att_head(const att_job *a, size_t r, size_t h) {"
assert src.count(anchor) == 1
src = src.replace(anchor, hdr + anchor, 1)
edits = [
    # forward: the fill, scores, softmax, context
    (
        "  {\n    size_t v1=att_v1_rows(a->variant,t);\n    for (i=0;i<v1;i++) {",
        "  unsigned long long fc=__rdtsc();\n  {\n    size_t v1=att_v1_rows(a->variant,t);\n    for (i=0;i<v1;i++) {",
    ),
    (
        "  x.b=a->k+(r*t*nk+kh)*d; x.ldb=nk*d; x.c=ph; x.ldc=t; x.mode=MM_LOWER;\n"
        "  att_rows(a->variant,a->fused,&x,t,ATT_SCORES,1);\n  att_softmax(a,ph);\n",
        "  x.b=a->k+(r*t*nk+kh)*d; x.ldb=nk*d; x.c=ph; x.ldc=t; x.mode=MM_LOWER;\n"
        "  FW_ADD(0,fc); fc=__rdtsc();\n  att_rows(a->variant,a->fused,&x,t,ATT_SCORES,1);\n"
        "  FW_ADD(1,fc); fc=__rdtsc();\n  att_softmax(a,ph);\n  FW_ADD(2,fc); fc=__rdtsc();\n",
    ),
    (
        "  x.c=a->y+(r*t*nh+h)*d; x.ldc=nh*d; x.mode=MM_KLE;\n  att_rows(a->variant,a->fused,&x,t,ATT_CONTEXT,1);\n"
        "  if (a->bad) att_verdict(a,r,h);\n",
        "  x.c=a->y+(r*t*nh+h)*d; x.ldc=nh*d; x.mode=MM_KLE;\n  att_rows(a->variant,a->fused,&x,t,ATT_CONTEXT,1);\n"
        "  FW_ADD(3,fc);\n  if (a->bad) att_verdict(a,r,h);\n",
    ),
    # forward copies
    (
        "  float *qc=att_ctg_region(a,task,worker),*kc=qc+f,*vc=kc+f,*yc=vc+f; att_job s=*a;\n"
        "  att_rows_in(qc,a->q+(r*t*nh+h)*d,t,d,nh*d);\n  att_rows_in(kc,a->k+(r*t*nk+kh)*d,t,d,nk*d);\n"
        "  att_rows_in(vc,a->v+(r*t*nk+kh)*d,t,d,nk*d);\n",
        "  float *qc=att_ctg_region(a,task,worker),*kc=qc+f,*vc=kc+f,*yc=vc+f; att_job s=*a;\n"
        "  unsigned long long cc=__rdtsc();\n"
        "  att_rows_in(qc,a->q+(r*t*nh+h)*d,t,d,nh*d);\n  att_rows_in(kc,a->k+(r*t*nk+kh)*d,t,d,nk*d);\n"
        "  att_rows_in(vc,a->v+(r*t*nk+kh)*d,t,d,nk*d);\n  FW_ADD(4,cc);\n",
    ),
    (
        "  att_rows_out(a->y+(r*t*nh+h)*d,yc,t,d,nh*d);\n",
        "  { unsigned long long co=__rdtsc(); att_rows_out(a->y+(r*t*nh+h)*d,yc,t,d,nh*d); FW_ADD(5,co); }\n",
    ),
    # backward: as instrument_bwd2.py, with ATT-CONTIG's `first`
    (
        "    x.c=dv+(r*t*nk+kh)*d; x.ldc=ldk; x.mode=MM_KGE;\n    att_rows(variant,a->fused,&x,t,ATT_TRANSPOSED,first);\n"
        "    for (i0=0;",
        "    x.c=dv+(r*t*nk+kh)*d; x.ldc=ldk; x.mode=MM_KGE;\n"
        "    { unsigned long long c=__rdtsc(); att_rows(variant,a->fused,&x,t,ATT_TRANSPOSED,first); BW_ADD(0,c); }\n"
        "    for (i0=0;",
    ),
    (
        "        if (!pass || jt>=BCIR_ATT_KEEP) {\n          memset(tile,0,rows*BCIR_ATT_COLS*sizeof(float));\n",
        "        unsigned long long ct=__rdtsc();\n"
        "        if (!pass || jt>=BCIR_ATT_KEEP) {\n          memset(tile,0,rows*BCIR_ATT_COLS*sizeof(float));\n",
    ),
    (
        "        i=0;\n#ifdef BCIR_MM_DISPATCH\n        if (!pass && variant==4) for (;i+16<=rows;i+=16)\n",
        "        BW_ADD(pass ? 3 : 1,ct); ct=__rdtsc();\n"
        "        i=0;\n#ifdef BCIR_MM_DISPATCH\n        if (!pass && variant==4) for (;i+16<=rows;i+=16)\n",
    ),
    (
        "            else pi[j]=pi[j]*(dp-dot[i])*scale;\n          }\n        }\n",
        "            else pi[j]=pi[j]*(dp-dot[i])*scale;\n          }\n        }\n        BW_ADD(pass ? 4 : 2,ct);\n",
    ),
    (
        "    x.c=dq+(r*t*nh+h)*d; x.ldc=ldh; x.mode=MM_KLE;\n    att_rows(variant,a->fused,&x,t,ATT_CONTEXT,1);\n"
        "    x.ta=1; x.b=qh; x.ldb=ldh; x.c=dk+(r*t*nk+kh)*d; x.ldc=ldk; x.mode=MM_KGE;\n"
        "    att_rows(variant,a->fused,&x,t,ATT_TRANSPOSED,first);\n",
        "    x.c=dq+(r*t*nh+h)*d; x.ldc=ldh; x.mode=MM_KLE;\n"
        "    { unsigned long long c=__rdtsc(); att_rows(variant,a->fused,&x,t,ATT_CONTEXT,1); BW_ADD(5,c); }\n"
        "    x.ta=1; x.b=qh; x.ldb=ldh; x.c=dk+(r*t*nk+kh)*d; x.ldc=ldk; x.mode=MM_KGE;\n"
        "    { unsigned long long c=__rdtsc(); att_rows(variant,a->fused,&x,t,ATT_TRANSPOSED,first); BW_ADD(6,c); }\n",
    ),
    # backward copies: K, V in; each head's Q, dY in and dQ out; dK, dV out
    (
        "  att_rows_in(kc,a->k+(r*t*nk+kh)*d,t,d,nk*d);\n  att_rows_in(vc,a->v+(r*t*nk+kh)*d,t,d,nk*d);\n"
        "  for (i=v1;i<t;i++) { memset(dkc+i*d,0,d*sizeof(float)); memset(dvc+i*d,0,d*sizeof(float)); }\n",
        "  unsigned long long cb=__rdtsc();\n"
        "  att_rows_in(kc,a->k+(r*t*nk+kh)*d,t,d,nk*d);\n  att_rows_in(vc,a->v+(r*t*nk+kh)*d,t,d,nk*d);\n"
        "  for (i=v1;i<t;i++) { memset(dkc+i*d,0,d*sizeof(float)); memset(dvc+i*d,0,d*sizeof(float)); }\n"
        "  BW_ADD(7,cb);\n",
    ),
    (
        "      att_rows_in(qc,a->q+(r*t*nh+h)*d,t,d,nh*d);\n      att_rows_in(dyc,a->dy+(r*t*nh+h)*d,t,d,nh*d);\n"
        "      for (i=v1;i<t;i++) memset(dqc+i*d,0,d*sizeof(float));\n",
        "      cb=__rdtsc();\n"
        "      att_rows_in(qc,a->q+(r*t*nh+h)*d,t,d,nh*d);\n      att_rows_in(dyc,a->dy+(r*t*nh+h)*d,t,d,nh*d);\n"
        "      for (i=v1;i<t;i++) memset(dqc+i*d,0,d*sizeof(float));\n      BW_ADD(7,cb);\n",
    ),
    (
        "      att_rows_out(a->dq+(r*t*nh+h)*d,dqc,t,d,nh*d);\n",
        "      cb=__rdtsc(); att_rows_out(a->dq+(r*t*nh+h)*d,dqc,t,d,nh*d); BW_ADD(7,cb);\n",
    ),
    (
        "  att_rows_out(a->dk+(r*t*nk+kh)*d,dkc,t,d,nk*d);\n  att_rows_out(a->dv+(r*t*nk+kh)*d,dvc,t,d,nk*d);\n}\n",
        "  cb=__rdtsc();\n  att_rows_out(a->dk+(r*t*nk+kh)*d,dkc,t,d,nk*d);\n  att_rows_out(a->dv+(r*t*nk+kh)*d,dvc,t,d,nk*d);\n"
        "  BW_ADD(7,cb);\n}\n",
    ),
]
for old, new in edits:
    assert src.count(old) == 1, (src.count(old), old[:90])
    src = src.replace(old, new, 1)
out.write_text(src)
print("instrumented", out, len(edits), "edits")
