"""WADJ-ROWS-0's prototype: a deep k-blocked product of few rows -- the step's weight adjoints, 128 to 260 rows over the
batch's depth -- in row ranges small enough that its job has BCIR_MM_TASKS tasks, where one range of 128 rows left it
three to nine; a chain's calls take its first call's ranges. usage: proto_wadj.py SRC OUT"""

import sys
from pathlib import Path

t = Path(sys.argv[1]).read_text()


def once(old, new):
    global t
    assert t.count(old) == 1, (old[:70], t.count(old))
    t = t.replace(old, new)


once(
    "typedef struct mm_job { mm_args x; float beta; int variant,fused; size_t nr,cols,rowsets; } mm_job;",
    "typedef struct mm_job { mm_args x; float beta; int variant,fused; size_t nr,cols,rowsets,rows; } mm_job;",
)
once(
    """  r.r0=row*BCIR_MM_ROWS; r.r1=r.r0+BCIR_MM_ROWS<j->x.m ? r.r0+BCIR_MM_ROWS : j->x.m;
  INSTANCE_SAW(INSTANCE_GEMM,j->variant);""",
    """  r.r0=row*j->rows; r.r1=r.r0+j->rows<j->x.m ? r.r0+j->rows : j->x.m;
  INSTANCE_SAW(INSTANCE_GEMM,j->variant);""",
)
once(
    """  j->cols=(n+j->nr-1)/j->nr; j->rowsets=(m+BCIR_MM_ROWS-1)/BCIR_MM_ROWS;
  return j->cols*j->rowsets;""",
    """  j->rows=BCIR_MM_ROWS;
  if (fused==MM_KBLOCK && m<=4*BCIR_MM_ROWS && k>=2*BCIR_MM_KBLOCK) {  /* WADJ-ROWS: a deep product of few rows */
    size_t cols=(n+j->nr-1)/j->nr,want=(BCIR_MM_TASKS+cols-1)/cols,rows=(m+want-1)/want;
    rows=(rows+7)/8*8;
    if (rows<16) rows=16;
    if (rows<j->rows) j->rows=rows;
  }
  j->cols=(n+j->nr-1)/j->nr; j->rowsets=(m+j->rows-1)/j->rows;
  return j->cols*j->rowsets;""",
)
once(
    """  double rows=(double)(j->x.m<BCIR_MM_ROWS ? j->x.m : BCIR_MM_ROWS),cols=(double)(j->x.n<j->nr ? j->x.n : j->nr);""",
    """  double rows=(double)(j->x.m<j->rows ? j->x.m : j->rows),cols=(double)(j->x.n<j->nr ? j->x.n : j->nr);""",
)
once(
    """  for (q=0;q<nx;q++) { size_t tasks=mm_call_init(&j.x[q],fused,variant,x+q); if (!q) j.split=tasks; }""",
    """  for (q=0;q<nx;q++) {
    size_t tasks=mm_call_init(&j.x[q],fused,variant,x+q);
    if (!q) j.split=tasks;
    else { j.x[q].rows=j.x[0].rows; j.x[q].rowsets=j.x[0].rowsets; }  /* one tiling for the chain */
  }""",
)
Path(sys.argv[2]).write_text(t)
print("WADJ-ROWS prototype:", sys.argv[2])
