"""WADJ-TILE-0: a copy of the tree's unit whose k-blocked jobs over few rows (m <= WADJ_M: the step's weight
adjoints, 128 to 260 rows over the batch's depth) take their rows in ranges of WADJ_RR instead of
BCIR_MM_ROWS's 128 -- 3 tasks become 12 at 128 x 128 -- every other rule as it was. A function of m, n and the
contract alone, so a chain's calls still share one tiling; a row partition is bit-identical to the whole
(mm_kernel's note). usage: proto_wadj.py SRC OUT"""

import sys
from pathlib import Path

s = Path(sys.argv[1]).read_text()
reps = [
    (
        "typedef struct mm_job { mm_args x; float beta; int variant,fused; size_t nr,cols,rowsets; } mm_job;",
        "#ifndef WADJ_M\n#define WADJ_M 512\n#endif\n#ifndef WADJ_RR\n#define WADJ_RR 32\n#endif\n"
        "typedef struct mm_job { mm_args x; float beta; int variant,fused; size_t nr,cols,rowsets,rr; } mm_job;",
    ),
    (
        "  r.r0=row*BCIR_MM_ROWS; r.r1=r.r0+BCIR_MM_ROWS<j->x.m ? r.r0+BCIR_MM_ROWS : j->x.m;\n  INSTANCE_SAW(INSTANCE_GEMM,j->variant);",
        "  r.r0=row*j->rr; r.r1=r.r0+j->rr<j->x.m ? r.r0+j->rr : j->x.m;\n  INSTANCE_SAW(INSTANCE_GEMM,j->variant);",
    ),
    (
        "  if (fused==MM_KBLOCK) {  /* MM-ROWS: whole blocks a task, BCIR_MM_TASKS tasks where the rows allow */\n"
        "    size_t blocks=(n+j->nr-1)/j->nr,ranges=(m+BCIR_MM_ROWS-1)/BCIR_MM_ROWS;",
        "  j->rr=fused==MM_KBLOCK && m<=WADJ_M ? WADJ_RR : BCIR_MM_ROWS;\n"
        "  if (fused==MM_KBLOCK) {  /* MM-ROWS: whole blocks a task, BCIR_MM_TASKS tasks where the rows allow */\n"
        "    size_t blocks=(n+j->nr-1)/j->nr,ranges=(m+j->rr-1)/j->rr;",
    ),
    (
        "  j->cols=(n+j->nr-1)/j->nr; j->rowsets=(m+BCIR_MM_ROWS-1)/BCIR_MM_ROWS;",
        "  j->cols=(n+j->nr-1)/j->nr; j->rowsets=(m+j->rr-1)/j->rr;",
    ),
    (
        "  double rows=(double)(j->x.m<BCIR_MM_ROWS ? j->x.m : BCIR_MM_ROWS),cols=(double)(j->x.n<j->nr ? j->x.n : j->nr);",
        "  double rows=(double)(j->x.m<j->rr ? j->x.m : j->rr),cols=(double)(j->x.n<j->nr ? j->x.n : j->nr);",
    ),
]
for o, n in reps:
    assert s.count(o) == 1, o[:80]
    s = s.replace(o, n)
Path(sys.argv[2]).write_text(s)
print("wrote", sys.argv[2])
