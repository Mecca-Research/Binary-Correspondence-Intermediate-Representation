"""NDT-PACK (2026-10-17), measured and not kept: three orders and caches of the pooled blocked
GEMM's tasks, each as exact-match edits to a copy of runtime/c/bcir_tensor.c (d68e3d3's, whose
GEMM is aaa2e6c's). Every variant keeps the partition -- the same tiles of C, each one task's, its
K blocks in order -- so every product is the same floats; only which tasks run together and what a
thread keeps between them change.

  colmajor  a job's task index walks one column block's row sets before the next column block
            (col = task / rowsets, row = task % rowsets; the kept order is row-major)
  cache     colmajor, and a thread keeps the AVX-512 kernel's packed tiles of one column block --
            every K block of it, up to K = 1024 -- between the tasks of one job, keyed by a
            per-job sequence number, in a thread-local buffer (the blueprint's lent workspace
            would hold the same floats; this is the measuring form)
  grp       tasks numbered in groups of `threads` column blocks, a group's row sets in turn: the
            tasks that start together are distinct columns of one row set and a thread's next
            task is most often its last one's column (the group width taken from the pool, a
            static the prototype sets before each job)

  python3 variants.py SRC_DIR OUT_DIR   ->  OUT_DIR/{colmajor,cache,grp}/bcir_tensor.c (+ .h)
"""

import shutil
import sys
from pathlib import Path


def sub(text, old, new):
    assert text.count(old) == 1, (old[:80], text.count(old))
    return text.replace(old, new)


ROWMAJOR = "  mm_job *j=(mm_job *)ctx; mm_range r; size_t col=task%j->cols,row=task/j->cols,i,jj;"
COLMAJOR = (
    "  mm_job *j=(mm_job *)ctx; mm_range r; size_t col=task/j->rowsets,row=task%j->rowsets,i,jj;"
)


def colmajor(c):
    return sub(c, ROWMAJOR, COLMAJOR)


def cache(c):
    c = colmajor(c)
    c = sub(
        c,
        "  float beta; int fold;  /* fold: the blocked kernels scale c by beta in their first K block */\n} mm_args;",
        "  float beta; int fold;  /* fold: the blocked kernels scale c by beta in their first K block */\n"
        "  size_t seq;  /* NDT-PACK: the pool job's sequence number (0: no tile cache) */\n} mm_args;",
    )
    c = sub(
        c,
        '__attribute__((target("avx512f"))) static void mm_avx512_kb(const mm_args *x, const mm_range *r) {\n'
        "  _Alignas(64) float bt[BCIR_MM_KBLOCK*48]; size_t i,jb,lb,nl;\n"
        "  for (jb=r->jb0;jb<r->jb1;jb+=48) {\n"
        "    size_t nj=x->n-jb<48 ? x->n-jb : 48;\n"
        "    for (lb=0;lb<x->k;lb+=nl) {\n"
        "      nl=kblock_next(x->k-lb);\n"
        "      kb512_pack(x,bt,jb,nj,lb,nl);\n"
        "      for (i=r->r0;i+8<=r->r1;i+=8) kb512_block(8,x,bt,i,jb,nj,lb,nl);\n"
        "      for (;i+4<=r->r1;i+=4) kb512_block(4,x,bt,i,jb,nj,lb,nl);\n"
        "      for (;i<r->r1;i++) kb512_block(1,x,bt,i,jb,nj,lb,nl);\n"
        "    }\n"
        "  }\n"
        "}",
        "/* NDT-PACK: a thread's packed tiles of one column block, every K block of it, kept between the\n"
        " * tasks of one pool job (x->seq names the job; the tiles are the pack's floats, so a task that\n"
        " * finds them computes what it would have packed) */\n"
        "#ifndef BCIR_MM_TILE_CACHE_K\n#define BCIR_MM_TILE_CACHE_K 1024\n#endif\n"
        "static __thread size_t kb_cache_seq,kb_cache_jb;\n"
        "static __thread _Alignas(64) float kb_cache_tile[BCIR_MM_TILE_CACHE_K*48];\n"
        '__attribute__((target("avx512f"))) static void mm_avx512_kb(const mm_args *x, const mm_range *r) {\n'
        "  _Alignas(64) float bt[BCIR_MM_KBLOCK*48]; size_t i,jb,lb,nl;\n"
        "  for (jb=r->jb0;jb<r->jb1;jb+=48) {\n"
        "    size_t nj=x->n-jb<48 ? x->n-jb : 48;\n"
        "    int cache=x->seq!=0 && x->k<=BCIR_MM_TILE_CACHE_K;\n"
        "    int hit=cache && kb_cache_seq==x->seq && kb_cache_jb==jb;\n"
        "    if (cache && !hit) kb_cache_seq=0;\n"
        "    for (lb=0;lb<x->k;lb+=nl) {\n"
        "      float *t=cache ? kb_cache_tile+lb*48 : bt;\n"
        "      nl=kblock_next(x->k-lb);\n"
        "      if (!hit) kb512_pack(x,t,jb,nj,lb,nl);\n"
        "      for (i=r->r0;i+8<=r->r1;i+=8) kb512_block(8,x,t,i,jb,nj,lb,nl);\n"
        "      for (;i+4<=r->r1;i+=4) kb512_block(4,x,t,i,jb,nj,lb,nl);\n"
        "      for (;i<r->r1;i++) kb512_block(1,x,t,i,jb,nj,lb,nl);\n"
        "    }\n"
        "    if (cache) { kb_cache_seq=x->seq; kb_cache_jb=jb; }\n"
        "  }\n"
        "}",
    )
    c = sub(
        c,
        "  x.b=b; x.ldb=tb ? k : n; x.c=c; x.ldc=n; x.mode=MM_FULL; x.beta=beta; x.fold=fold;\n  r.jb0=0;",
        "  x.b=b; x.ldb=tb ? k : n; x.c=c; x.ldc=n; x.mode=MM_FULL; x.beta=beta; x.fold=fold; x.seq=0;\n  r.jb0=0;",
    )
    c = sub(
        c,
        "typedef struct mm_job { mm_args x; float beta; int variant,fused; size_t nr,cols,rowsets; } mm_job;\n",
        "typedef struct mm_job { mm_args x; float beta; int variant,fused; size_t nr,cols,rowsets; } mm_job;\n"
        "#ifdef BCIR_MM_DISPATCH\nstatic size_t mm_job_seq;\n#endif\n",
    )
    c = sub(
        c,
        "  j->x.beta=beta; j->x.fold=fused==MM_KBLOCK && k;\n",
        "  j->x.beta=beta; j->x.fold=fused==MM_KBLOCK && k;\n"
        "#ifdef BCIR_MM_DISPATCH\n  j->x.seq=__atomic_add_fetch(&mm_job_seq,(size_t)1,__ATOMIC_RELAXED);\n#else\n  j->x.seq=0;\n#endif\n",
    )
    return c


def grp(c):
    c = sub(
        c,
        "typedef struct mm_job { mm_args x; float beta; int variant,fused; size_t nr,cols,rowsets; } mm_job;",
        "typedef struct mm_job { mm_args x; float beta; int variant,fused; size_t nr,cols,rowsets,g; } mm_job;\n"
        "/* grp: tasks numbered in groups of g column blocks, a group's row sets in turn: the g tasks that start\n"
        " * together are g columns of one row set, and a thread's next task is most often its last one's column */\n"
        "static void mm_tile(const mm_job *j, size_t task, size_t *col, size_t *row) {\n"
        "  size_t g=j->g ? (j->g<j->cols ? j->g : j->cols) : 1,grp=task/(g*j->rowsets),u=task%(g*j->rowsets);\n"
        "  size_t w=(grp+1)*g<=j->cols ? g : j->cols-grp*g;\n"
        "  *row=u/w; *col=grp*g+u%w;\n"
        "}",
    )
    c = sub(
        c,
        ROWMAJOR,
        "  mm_job *j=(mm_job *)ctx; mm_range r; size_t col,row,i,jj;\n  mm_tile(j,task,&col,&row);",
    )
    c = sub(
        c,
        "  j->cols=(n+j->nr-1)/j->nr; j->rowsets=(m+BCIR_MM_ROWS-1)/BCIR_MM_ROWS;\n",
        "  j->cols=(n+j->nr-1)/j->nr; j->rowsets=(m+BCIR_MM_ROWS-1)/BCIR_MM_ROWS; j->g=mm_group_g;\n",
    )
    c = sub(
        c,
        "static size_t mm_job_init(mm_job *j,",
        "static size_t mm_group_g=1;\nstatic size_t mm_job_init(mm_job *j,",
    )
    c = sub(
        c,
        "  tasks=mm_job_init(&j,fused,ta,tb,m,n,k,alpha,a,b,beta,c);\n  pool_run(pool,mm_task,&j,tasks);",
        "  mm_group_g=pool->threads;\n  tasks=mm_job_init(&j,fused,ta,tb,m,n,k,alpha,a,b,beta,c);\n  pool_run(pool,mm_task,&j,tasks);",
    )
    c = sub(
        c,
        "  j.split=mm_job_init(&j.a,fused,x->ta,x->tb,x->m,x->n,x->k,x->alpha,x->a,x->b,x->beta,x->c);",
        "  mm_group_g=pool->threads;\n  j.split=mm_job_init(&j.a,fused,x->ta,x->tb,x->m,x->n,x->k,x->alpha,x->a,x->b,x->beta,x->c);",
    )
    c = sub(
        c,
        "  for (q=0;q<nx;q++) { size_t tasks=mm_call_init(&j.x[q],fused,x+q); if (!q) j.split=tasks; }",
        "  mm_group_g=pool->threads;\n  for (q=0;q<nx;q++) { size_t tasks=mm_call_init(&j.x[q],fused,x+q); if (!q) j.split=tasks; }",
    )
    return c


def main(argv):
    src, out = Path(argv[1]), Path(argv[2])
    base = (src / "bcir_tensor.c").read_text(encoding="utf-8")
    for name, fn in (("colmajor", colmajor), ("cache", cache), ("grp", grp)):
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "bcir_tensor.c").write_text(fn(base), encoding="utf-8")
        for extra in ("bcir_tensor.h", "bcir_decoder_train.c", "bcir_decoder_train.h"):
            if (src / extra).exists():
                shutil.copyfile(src / extra, d / extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
