"""KSPLIT-ROWS-0: a copy of the tensor unit whose split calls run a task a (K block, row range) -- each task its block's
partial over BCIR_MM_KSPLIT_ROWS rows of C (rounded to even ranges of a multiple of 16 rows), every value the
unit's: each element's partial is the same fmaf chain whichever task computes it, and the fold is unchanged.
BCIR_MM_KSPLIT_ROWS 0 keeps the unit's whole-C tasks. usage: proto_ksrows.py SRC_bcir_tensor.c DST_bcir_tensor.c"""

import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text(encoding="utf-8")


def once(text, old, new):
    if text.count(old) != 1:
        raise SystemExit(f"anchor found {text.count(old)} times: {old[:70]!r}")
    return text.replace(old, new)


src = once(
    src,
    "typedef struct mm_job { mm_args x; float beta; int variant,fused; size_t nr,cols,rowsets,ks; float *ws; } mm_job;",
    "#ifndef BCIR_MM_KSPLIT_ROWS\n#define BCIR_MM_KSPLIT_ROWS 128\n#endif\n"
    "typedef struct mm_job { mm_args x; float beta; int variant,fused; size_t nr,cols,rowsets,ks,ksr,ksrows; float *ws; }"
    " mm_job;",
)
src = once(
    src,
    """static void mm_ksplit_task(const mm_job *j, size_t b) {
  mm_args xb=j->x; mm_range r; size_t l0,nl;
  kblock_at(j->x.k,b,&l0,&nl);""",
    """static void mm_ksplit_task(const mm_job *j, size_t t) {
  mm_args xb=j->x; mm_range r; size_t l0,nl,b=t/j->ksr,q=t%j->ksr;
  kblock_at(j->x.k,b,&l0,&nl);""",
)
src = once(
    src,
    "  r.jb0=0; r.jb1=j->x.n; r.r0=0; r.r1=j->x.m;\n  mm_run_one(j->variant,j->fused,&xb,&r);\n}",
    "  r.jb0=0; r.jb1=j->x.n; r.r0=q*j->ksrows; r.r1=r.r0+j->ksrows<j->x.m ? r.r0+j->ksrows : j->x.m;\n"
    "  mm_run_one(j->variant,j->fused,&xb,&r);\n}",
)
src = once(src, "  j->ks=0; j->ws=NULL;\n", "  j->ks=0; j->ws=NULL; j->ksr=1; j->ksrows=m;\n")
src = once(
    src,
    "  j->ks=blocks; j->ws=(float *)(void *)*room;\n",
    "  j->ks=blocks; j->ws=(float *)(void *)*room;\n"
    "  if (BCIR_MM_KSPLIT_ROWS && j->x.m>BCIR_MM_KSPLIT_ROWS) {\n"
    "    size_t ranges=(j->x.m+BCIR_MM_KSPLIT_ROWS-1)/BCIR_MM_KSPLIT_ROWS,rows=(j->x.m+ranges-1)/ranges;\n"
    "    rows=(rows+15)/16*16; j->ksrows=rows; j->ksr=(j->x.m+rows-1)/rows;\n"
    "  }\n",
)
src = once(
    src,
    "  *room+=step; *left-=step;\n  return blocks;\n}",
    "  *room+=step; *left-=step;\n  return blocks*j->ksr;\n}",
)
src = once(
    src,
    "  if (j->ks) return (double)j->x.m*(double)j->x.n*(double)BCIR_MM_KBLOCK;  /* KSPLIT: a block of all of C */",
    "  if (j->ks) return (double)j->ksrows*(double)j->x.n*(double)BCIR_MM_KBLOCK;  /* KSPLIT: a block of its rows */",
)
Path(sys.argv[2]).write_text(src, encoding="utf-8")
