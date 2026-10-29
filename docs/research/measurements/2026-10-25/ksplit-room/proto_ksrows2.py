"""KSPLIT-ROWS-1: a copy of the tensor unit whose split calls run a task a (K block, row range) where the block
tasks alone would leave the pool's workers short (fewer than two tasks a worker) or a K block's panel of A
(BCIR_MM_KBLOCK x m floats) would outgrow BCIR_MM_KSPLIT_PANEL floats (1 MiB, half a core's L2): ranges of
BCIR_MM_KSPLIT_ROWS rows for the panel, as many as two tasks a worker need for the balance, even ranges of a multiple
of 16 rows. Every value the unit's: each element's partial is the same fmaf chain whichever task computes it, and
the fold is unchanged. usage: proto_ksrows2.py SRC_bcir_tensor.c DST_bcir_tensor.c"""

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
    "#ifndef BCIR_MM_KSPLIT_PANEL\n#define BCIR_MM_KSPLIT_PANEL ((size_t)1<<18)\n#endif\n"
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
    "static size_t mm_ksplit(mm_job *j, size_t tasks, unsigned char **room, size_t *left) {",
    "static size_t mm_ksplit(mm_job *j, size_t tasks, unsigned char **room, size_t *left, unsigned threads) {",
)
src = once(
    src,
    "  j->ks=blocks; j->ws=(float *)(void *)*room;\n",
    "  j->ks=blocks; j->ws=(float *)(void *)*room;\n"
    "  {\n"
    "    size_t m=j->x.m,ranges=1,rows;\n"
    "    if ((size_t)BCIR_MM_KBLOCK*m>BCIR_MM_KSPLIT_PANEL) ranges=(m+BCIR_MM_KSPLIT_ROWS-1)/BCIR_MM_KSPLIT_ROWS;\n"
    "    if (threads>1 && blocks*ranges<2*(size_t)threads) {\n"
    "      size_t want=(2*(size_t)threads+blocks-1)/blocks;\n"
    "      if (want>ranges) ranges=want;\n"
    "    }\n"
    "    if (ranges>(m+15)/16) ranges=(m+15)/16;\n"
    "    rows=(m+ranges-1)/ranges; rows=(rows+15)/16*16;\n"
    "    j->ksrows=rows; j->ksr=(m+rows-1)/rows;\n"
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
src = once(
    src,
    "  tasks=mm_ksplit(&j,tasks,&room,&left);\n",
    "  tasks=mm_ksplit(&j,tasks,&room,&left,pool ? pool->threads : 1);\n",
)
src = once(
    src,
    "  j.split=mm_ksplit(&j.a,j.split,&room,&left);\n",
    "  j.split=mm_ksplit(&j.a,j.split,&room,&left,pool ? pool->threads : 1);\n",
)
src = once(
    src,
    "  second=mm_ksplit(&j.b,second,&room,&left);\n",
    "  second=mm_ksplit(&j.b,second,&room,&left,pool ? pool->threads : 1);\n",
)
src = once(
    src,
    "      j.start[q+1]=j.start[q]+mm_ksplit(&j.y[q],mm_call_init(&j.y[q],fused,variant,y+q),&room,&left);\n",
    "      j.start[q+1]=j.start[q]+mm_ksplit(&j.y[q],mm_call_init(&j.y[q],fused,variant,y+q),&room,&left,\n"
    "          pool ? pool->threads : 1);\n",
)
Path(sys.argv[2]).write_text(src, encoding="utf-8")
