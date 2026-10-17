"""NDT-PREPACK's prototype, measured and not kept (2026-10-16): the pooled blocked GEMM packs each
column block's tiles once, in a pack phase of its own (one task per call and column block, the
variant's own pack) into a workspace attached to the pool, and the compute phase's tasks read them
there -- the calls of one job that read one B (the same matrix, stride, transpose, columns and
depth) sharing one copy -- instead of every 128-row task packing its own. Applied with exact-match
edits to a copy of runtime/c at 64bfd62 (bcir_tensor.c, bcir_tensor.h); with PREPACK_STATIC=1 in
the environment every pool also attaches a static workspace of BCIR_PREPACK_STATIC floats, the
measurement's form (build with -DBCIR_PREPACK_STATIC=4194304, 16 MB, more than the step's largest
job takes: the down projection's pair, 1,486,848 floats). Every product is the same floats either
way, so every result is too.

usage: [PREPACK_STATIC=1] prototype.py DIR   (DIR holds bcir_tensor.c and bcir_tensor.h)
"""

import sys
from pathlib import Path

d = Path(sys.argv[1])


def edit(name, pairs):
    p = d / name
    t = p.read_text(encoding="utf-8")
    for old, new in pairs:
        n = t.count(old)
        if n != 1:
            raise SystemExit(f"{name}: anchor found {n} times: {old[:90]!r}")
        t = t.replace(old, new, 1)
    p.write_text(t, encoding="utf-8")


edit(
    "bcir_tensor.h",
    [
        (
            "BCIR_DT_API unsigned bcir_tensor_pool_threads(const bcir_tensor_pool *pool); /* NULL: 1 */\n",
            "BCIR_DT_API unsigned bcir_tensor_pool_threads(const bcir_tensor_pool *pool); /* NULL: 1 */\n"
            "/* A workspace for the blocked GEMM's shared tiles (NDT-PREPACK): with one attached, a pooled\n"
            " * blocked product packs each column block's tiles once, in a phase of its own, and every row\n"
            " * task reads them there -- without it each task packs its own copy, as before; the tiles are\n"
            " * the same floats either way, so every result is too. The memory stays the caller's, used by\n"
            " * one job at a time; a call whose tiles do not fit in what is left packs its own. Returns the\n"
            " * floats usable (from the first 64-byte boundary); NULL memory or 0 bytes detaches. */\n"
            "BCIR_DT_API size_t bcir_tensor_pool_attach(bcir_tensor_pool *pool, void *memory, size_t bytes);\n",
        ),
    ],
)

edit(
    "bcir_tensor.c",
    [
        # the generic kernel's pack as a function, and the packed tiles as an input
        (
            "BCIR_MM_INLINE void mm_kernel_kb(size_t mr, size_t nr, const mm_args *x, size_t jb0, size_t jb1,\n"
            "    size_t r0, size_t r1) {\n"
            "  _Alignas(64) float bt[BCIR_MM_KBLOCK*BCIR_MM_NRMAX];\n"
            "  size_t i,j,l,jb,lb,nl;\n"
            "  for (jb=jb0;jb<jb1;jb+=nr) {\n"
            "    size_t nj=x->n-jb<nr ? x->n-jb : nr;\n"
            "    for (lb=0;lb<x->k;lb+=nl) {\n"
            "      nl=kblock_next(x->k-lb);\n"
            "      if (x->tb) {\n"
            "        for (j=0;j<nj;j++) {\n"
            "          const float *column=x->b+(jb+j)*x->ldb+lb;\n"
            "          for (l=0;l<nl;l++) bt[l*nr+j]=column[l];\n"
            "        }\n"
            "      } else {\n"
            "        for (l=0;l<nl;l++) {\n"
            "          const float *row=x->b+(lb+l)*x->ldb+jb;\n"
            "          for (j=0;j<nj;j++) bt[l*nr+j]=row[j];\n"
            "        }\n"
            "      }\n"
            "      for (j=nj;j<nr;j++) for (l=0;l<nl;l++) bt[l*nr+j]=0.0f;  /* the padding columns */\n"
            "      for (i=r0;i+mr<=r1;i+=mr) mm_block_kb(mr,nr,x,bt,i,jb,nj,lb,nl);\n"
            "      for (;i<r1;i++) mm_block_kb(1,nr,x,bt,i,jb,nj,lb,nl);\n"
            "    }\n"
            "  }\n"
            "}\n",
            "static void kb_pack(const mm_args *x, float *bt, size_t nr, size_t jb, size_t nj, size_t lb,\n"
            "    size_t nl) {\n"
            "  size_t j,l;\n"
            "  if (x->tb) {\n"
            "    for (j=0;j<nj;j++) {\n"
            "      const float *column=x->b+(jb+j)*x->ldb+lb;\n"
            "      for (l=0;l<nl;l++) bt[l*nr+j]=column[l];\n"
            "    }\n"
            "  } else {\n"
            "    for (l=0;l<nl;l++) {\n"
            "      const float *row=x->b+(lb+l)*x->ldb+jb;\n"
            "      for (j=0;j<nj;j++) bt[l*nr+j]=row[j];\n"
            "    }\n"
            "  }\n"
            "  for (j=nj;j<nr;j++) for (l=0;l<nl;l++) bt[l*nr+j]=0.0f;  /* the padding columns */\n"
            "}\n"
            "/* `packed`, when set, holds the range's column blocks' tiles as the pack would write them, block\n"
            " * after block, each block's depths in order (NDT-PREPACK): read there, not packed again. */\n"
            "BCIR_MM_INLINE void mm_kernel_kb(size_t mr, size_t nr, const mm_args *x, size_t jb0, size_t jb1,\n"
            "    size_t r0, size_t r1, const float *packed) {\n"
            "  _Alignas(64) float bt[BCIR_MM_KBLOCK*BCIR_MM_NRMAX];\n"
            "  size_t i,jb,lb,nl;\n"
            "  for (jb=jb0;jb<jb1;jb+=nr) {\n"
            "    size_t nj=x->n-jb<nr ? x->n-jb : nr;\n"
            "    const float *col=packed ? packed+(jb-jb0)/nr*x->k*nr : NULL;\n"
            "    for (lb=0;lb<x->k;lb+=nl) {\n"
            "      const float *tile=bt;\n"
            "      nl=kblock_next(x->k-lb);\n"
            "      if (col) tile=col+lb*nr; else kb_pack(x,bt,nr,jb,nj,lb,nl);\n"
            "      for (i=r0;i+mr<=r1;i+=mr) mm_block_kb(mr,nr,x,tile,i,jb,nj,lb,nl);\n"
            "      for (;i<r1;i++) mm_block_kb(1,nr,x,tile,i,jb,nj,lb,nl);\n"
            "    }\n"
            "  }\n"
            "}\n",
        ),
        (
            "typedef struct mm_range { size_t jb0,jb1,r0,r1; } mm_range;\n",
            "typedef struct mm_range { size_t jb0,jb1,r0,r1; const float *packed; } mm_range;\n",
        ),
        (
            "static void mm_generic_kb(const mm_args *x, const mm_range *r) { mm_kernel_kb(2,32,x,r->jb0,r->jb1,r->r0,r->r1); }\n",
            "static void mm_generic_kb(const mm_args *x, const mm_range *r) { mm_kernel_kb(2,32,x,r->jb0,r->jb1,r->r0,r->r1,r->packed); }\n",
        ),
        # AVX-512
        (
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
            "}\n",
            '__attribute__((target("avx512f"))) static void mm_avx512_kb(const mm_args *x, const mm_range *r) {\n'
            "  _Alignas(64) float bt[BCIR_MM_KBLOCK*48]; size_t i,jb,lb,nl;\n"
            "  for (jb=r->jb0;jb<r->jb1;jb+=48) {\n"
            "    size_t nj=x->n-jb<48 ? x->n-jb : 48;\n"
            "    const float *col=r->packed ? r->packed+(jb-r->jb0)/48*x->k*48 : NULL;\n"
            "    for (lb=0;lb<x->k;lb+=nl) {\n"
            "      const float *tile=bt;\n"
            "      nl=kblock_next(x->k-lb);\n"
            "      if (col) tile=col+lb*48; else kb512_pack(x,bt,jb,nj,lb,nl);\n"
            "      for (i=r->r0;i+8<=r->r1;i+=8) kb512_block(8,x,tile,i,jb,nj,lb,nl);\n"
            "      for (;i+4<=r->r1;i+=4) kb512_block(4,x,tile,i,jb,nj,lb,nl);\n"
            "      for (;i<r->r1;i++) kb512_block(1,x,tile,i,jb,nj,lb,nl);\n"
            "    }\n"
            "  }\n"
            "}\n",
        ),
        # AVX2
        (
            '__attribute__((target("avx2,fma"))) static void mm_avx2_kb(const mm_args *x, const mm_range *r) {\n'
            "  _Alignas(32) float bt[BCIR_MM_KBLOCK*16]; size_t i,jb,lb,nl;\n"
            "  for (jb=r->jb0;jb<r->jb1;jb+=16) {\n"
            "    size_t nj=x->n-jb<16 ? x->n-jb : 16;\n"
            "    for (lb=0;lb<x->k;lb+=nl) {\n"
            "      nl=kblock_next(x->k-lb);\n"
            "      kb256_pack(x,bt,jb,nj,lb,nl);\n"
            "      for (i=r->r0;i+6<=r->r1;i+=6) kb256_block(6,x,bt,i,jb,nj,lb,nl);\n"
            "      for (;i+2<=r->r1;i+=2) kb256_block(2,x,bt,i,jb,nj,lb,nl);\n"
            "      for (;i<r->r1;i++) kb256_block(1,x,bt,i,jb,nj,lb,nl);\n"
            "    }\n"
            "  }\n"
            "}\n",
            '__attribute__((target("avx2,fma"))) static void mm_avx2_kb(const mm_args *x, const mm_range *r) {\n'
            "  _Alignas(32) float bt[BCIR_MM_KBLOCK*16]; size_t i,jb,lb,nl;\n"
            "  for (jb=r->jb0;jb<r->jb1;jb+=16) {\n"
            "    size_t nj=x->n-jb<16 ? x->n-jb : 16;\n"
            "    const float *col=r->packed ? r->packed+(jb-r->jb0)/16*x->k*16 : NULL;\n"
            "    for (lb=0;lb<x->k;lb+=nl) {\n"
            "      const float *tile=bt;\n"
            "      nl=kblock_next(x->k-lb);\n"
            "      if (col) tile=col+lb*16; else kb256_pack(x,bt,jb,nj,lb,nl);\n"
            "      for (i=r->r0;i+6<=r->r1;i+=6) kb256_block(6,x,tile,i,jb,nj,lb,nl);\n"
            "      for (;i+2<=r->r1;i+=2) kb256_block(2,x,tile,i,jb,nj,lb,nl);\n"
            "      for (;i<r->r1;i++) kb256_block(1,x,tile,i,jb,nj,lb,nl);\n"
            "    }\n"
            "  }\n"
            "}\n",
        ),
        # every other range starts with no packed tiles
        (
            "  if (!x->m || !x->n || !x->k) return;\n  r.jb0=0; r.jb1=x->n; r.r0=0; r.r1=x->m;\n",
            "  if (!x->m || !x->n || !x->k) return;\n  r.jb0=0; r.jb1=x->n; r.r0=0; r.r1=x->m; r.packed=NULL;\n",
        ),
        (
            "  r.jb0=0; r.jb1=n; r.r0=0; r.r1=m;\n  mm_run_range(variant,fused,&x,&r);\n  return variant;\n",
            "  r.jb0=0; r.jb1=n; r.r0=0; r.r1=m; r.packed=NULL;\n  mm_run_range(variant,fused,&x,&r);\n  return variant;\n",
        ),
        # the pool's workspace
        (
            "  pool_worker *records;\n};\n",
            "  pool_worker *records;\n  float *pack; size_t pack_floats;  /* the attached workspace (NDT-PREPACK) */\n};\n",
        ),
        (
            "unsigned bcir_tensor_pool_threads(const bcir_tensor_pool *pool) { return pool ? pool->threads : 1; }\n",
            "unsigned bcir_tensor_pool_threads(const bcir_tensor_pool *pool) { return pool ? pool->threads : 1; }\n"
            "size_t bcir_tensor_pool_attach(bcir_tensor_pool *pool, void *memory, size_t bytes) {\n"
            "  uintptr_t at,skip;\n"
            "  if (!pool) return 0;\n"
            "  pool->pack=NULL; pool->pack_floats=0;\n"
            "  if (!memory) return 0;\n"
            "  at=(uintptr_t)memory; skip=(64u-at%64u)%64u;\n"
            "  if (bytes<skip+64u) return 0;\n"
            "  pool->pack=(float *)(void *)((unsigned char *)memory+skip); pool->pack_floats=(bytes-skip)/sizeof(float)/16u*16u;\n"
            "  return pool->pack_floats;\n"
            "}\n",
        ),
        # the job's tiles, and the task reading them
        (
            "typedef struct mm_job { mm_args x; float beta; int variant,fused; size_t nr,cols,rowsets; } mm_job;\n"
            "static void mm_task(void *ctx, size_t task) {\n"
            "  mm_job *j=(mm_job *)ctx; mm_range r; size_t col=task%j->cols,row=task/j->cols,i,jj;\n"
            "  r.jb0=col*j->nr; r.jb1=r.jb0+j->nr<j->x.n ? r.jb0+j->nr : j->x.n;\n",
            "typedef struct mm_job {\n"
            "  mm_args x; float beta; int variant,fused; size_t nr,cols,rowsets;\n"
            "  float *packed;  /* NDT-PREPACK: the call's tiles, column block after column block, or NULL */\n"
            "} mm_job;\n"
            "static void mm_task(void *ctx, size_t task) {\n"
            "  mm_job *j=(mm_job *)ctx; mm_range r; size_t col=task%j->cols,row=task/j->cols,i,jj;\n"
            "  r.jb0=col*j->nr; r.jb1=r.jb0+j->nr<j->x.n ? r.jb0+j->nr : j->x.n;\n"
            "  r.packed=j->packed ? j->packed+col*j->x.k*j->nr : NULL;\n",
        ),
        (
            "  j->cols=(n+j->nr-1)/j->nr; j->rowsets=(m+BCIR_MM_ROWS-1)/BCIR_MM_ROWS;\n  return j->cols*j->rowsets;\n}\n",
            "  j->cols=(n+j->nr-1)/j->nr; j->rowsets=(m+BCIR_MM_ROWS-1)/BCIR_MM_ROWS; j->packed=NULL;\n"
            "  return j->cols*j->rowsets;\n}\n"
            "/* NDT-PREPACK: the pack phase. A blocked call of more than one row range whose tiles fit in what\n"
            " * is left of the pool's workspace is given its region, in call order; one task then packs one\n"
            " * column block's tiles there -- every K block of the rule, with the variant's own pack -- and\n"
            " * the compute phase's tasks read them. A call without a region packs per task, as before. */\n"
            "#define MM_PACK_JOBS (2*BCIR_TENSOR_MM_GROUP)\n"
            "typedef struct mm_pack_job { mm_job *jobs[MM_PACK_JOBS]; size_t count,start[MM_PACK_JOBS+1]; } mm_pack_job;\n"
            "static void mm_pack_tiles(const mm_job *j, size_t col) {\n"
            "  const mm_args *x=&j->x; size_t jb=col*j->nr,nj=x->n-jb<j->nr ? x->n-jb : j->nr,lb,nl;\n"
            "  float *dst=j->packed+col*x->k*j->nr;\n"
            "  for (lb=0;lb<x->k;lb+=nl) {\n"
            "    nl=kblock_next(x->k-lb);\n"
            "#ifdef BCIR_MM_DISPATCH\n"
            "    if (j->variant==4) { kb512_pack(x,dst+lb*48,jb,nj,lb,nl); continue; }\n"
            "    if (j->variant==2) { kb256_pack(x,dst+lb*16,jb,nj,lb,nl); continue; }\n"
            "#endif\n"
            "    kb_pack(x,dst+lb*j->nr,j->nr,jb,nj,lb,nl);\n"
            "  }\n"
            "}\n"
            "static void mm_pack_task(void *ctx, size_t task) {\n"
            "  mm_pack_job *p=(mm_pack_job *)ctx; size_t q;\n"
            "  for (q=0;task>=p->start[q+1];q++) {}\n"
            "  mm_pack_tiles(p->jobs[q],task-p->start[q]);\n"
            "}\n"
            "/* Give the calls their regions and run the pack phase: the compute phase follows. */\n"
            "static void mm_prepack(bcir_tensor_pool *pool, mm_job *const *jobs, size_t count) {\n"
            "  mm_pack_job p; size_t q,e,used=0;\n"
            "  if (!pool || !pool->pack) return;\n"
            "  p.count=0; p.start[0]=0;\n"
            "  for (q=0;q<count;q++) {\n"
            "    mm_job *j=jobs[q]; size_t need;\n"
            "    if (j->fused!=MM_KBLOCK || !j->x.k || j->rowsets<2) continue;\n"
            "    /* a B an earlier call of the job packed -- the weight adjoints' one input -- is read there */\n"
            "    for (e=0;e<p.count;e++) {\n"
            "      const mm_args *o=&p.jobs[e]->x;\n"
            "      if (o->b==j->x.b && o->ldb==j->x.ldb && o->tb==j->x.tb && o->k==j->x.k && o->n==j->x.n) break;\n"
            "    }\n"
            "    if (e<p.count) { j->packed=p.jobs[e]->packed; continue; }\n"
            "    need=j->cols*j->x.k*j->nr;\n"
            "    if (need>pool->pack_floats-used) continue;\n"
            "    j->packed=pool->pack+used; used+=need;\n"
            "    p.jobs[p.count]=j; p.start[p.count+1]=p.start[p.count]+j->cols; p.count++;\n"
            "  }\n"
            "  if (p.count) pool_run(pool,mm_pack_task,&p,p.start[p.count]);\n"
            "}\n",
        ),
        (
            "  tasks=mm_job_init(&j,fused,ta,tb,m,n,k,alpha,a,b,beta,c);\n  pool_run(pool,mm_task,&j,tasks);\n}\n",
            "  tasks=mm_job_init(&j,fused,ta,tb,m,n,k,alpha,a,b,beta,c);\n"
            "  { mm_job *one=&j; mm_prepack(pool,&one,1); }\n"
            "  pool_run(pool,mm_task,&j,tasks);\n}\n",
        ),
        (
            "  second=mm_job_init(&j.b,fused,y->ta,y->tb,y->m,y->n,y->k,y->alpha,y->a,y->b,y->beta,y->c);\n"
            "  pool_run(pool,mm_pair_task,&j,j.split+second);\n",
            "  second=mm_job_init(&j.b,fused,y->ta,y->tb,y->m,y->n,y->k,y->alpha,y->a,y->b,y->beta,y->c);\n"
            "  { mm_job *two[2]; two[0]=&j.a; two[1]=&j.b; mm_prepack(pool,two,2); }\n"
            "  pool_run(pool,mm_pair_task,&j,j.split+second);\n",
        ),
        (
            "  for (q=0;q<ny;q++) j.start[q+1]=j.start[q]+mm_call_init(&j.y[q],fused,y+q);\n"
            "  pool_run(pool,mm_group_task,&j,j.split+j.start[ny]);\n",
            "  for (q=0;q<ny;q++) j.start[q+1]=j.start[q]+mm_call_init(&j.y[q],fused,y+q);\n"
            "  {\n"
            "    mm_job *all[2*BCIR_TENSOR_MM_GROUP]; size_t count=0;\n"
            "    for (q=0;q<nx;q++) all[count++]=&j.x[q];\n"
            "    for (q=0;q<ny;q++) all[count++]=&j.y[q];\n"
            "    mm_prepack(pool,all,count);\n"
            "  }\n"
            "  pool_run(pool,mm_group_task,&j,j.split+j.start[ny]);\n",
        ),
    ],
)
print("patched", d)

edit(
    "bcir_tensor.c",
    [
        (
            "  rest.jb0=0; rest.jb1=x->n; rest.r0=whole; rest.r1=t;\n",
            "  rest.jb0=0; rest.jb1=x->n; rest.r0=whole; rest.r1=t; rest.packed=NULL;\n",
        ),
    ],
)
print("patched (2)", d)
if __import__("os").environ.get("PREPACK_STATIC"):
    edit(
        "bcir_tensor.c",
        [
            (
                "#if defined(BCIR_POOL_PTHREADS) || defined(BCIR_POOL_WIN32)\n  if (threads>1 && !pool_start(p)) return NULL;\n#endif\n  return p;\n}\n",
                "#if defined(BCIR_POOL_PTHREADS) || defined(BCIR_POOL_WIN32)\n  if (threads>1 && !pool_start(p)) return NULL;\n#endif\n"
                "#ifdef BCIR_PREPACK_STATIC  /* the prototype's measurement only: one static workspace */\n"
                "  { static float workspace[(size_t)BCIR_PREPACK_STATIC]; bcir_tensor_pool_attach(p,workspace,sizeof(workspace)); }\n"
                "#endif\n"
                "  return p;\n}\n",
            ),
        ],
    )
    print("static workspace hack applied")

edit(
    "bcir_tensor.c",
    [
        (
            "    float alpha, const float *a, const float *b, float beta, float *c) {\n  if (!m || !n) return 0;\n  j->beta=beta;",
            "    float alpha, const float *a, const float *b, float beta, float *c) {\n"
            "  j->fused=fused; j->rowsets=0; j->packed=NULL;  /* an empty product: no tasks, no tiles */\n"
            "  if (!m || !n) return 0;\n  j->beta=beta;",
        ),
        (
            "    if (j->fused!=MM_KBLOCK || !j->x.k || j->rowsets<2) continue;\n",
            "    if (j->rowsets<2 || j->fused!=MM_KBLOCK || !j->x.k) continue;\n",
        ),
    ],
)
print("patched (3)", d)
