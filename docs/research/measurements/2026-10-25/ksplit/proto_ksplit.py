"""KSPLIT-0: a copy of the tensor unit whose blocked-contract weight adjoints (ta, not tb, at least four K blocks
deep) run a pool task a K block: each task computes its block's product over all of C into its own slice of a
partials buffer -- the kernel itself, called on the block alone with alpha 1 and a slice pre-filled with -0.0
under beta 1, so the slice holds the block's accumulators exactly (-0.0 + p is p, -0.0 included) -- and a fold job
then gives each element of C its blocks in order with the kernel's epilogue arithmetic: the first block's beta
scaling (NDT-BETA), then c + alpha*P, a rounded product and a rounded sum, block after block. Every value is the
nest's. The buffer is malloc'd for the prototype (the trainer would lend it). usage: proto_ksplit.py SRC DST"""

import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text(encoding="utf-8")


def once(text, old, new):
    if text.count(old) != 1:
        raise SystemExit(f"anchor found {text.count(old)} times: {old[:70]!r}")
    return text.replace(old, new)


src = once(
    src,
    "typedef struct mm_job { mm_args x; float beta; int variant,fused; size_t nr,cols,rowsets; } mm_job;\n",
    """typedef struct mm_job { mm_args x; float beta; int variant,fused; size_t nr,cols,rowsets;
  size_t ks; float *ws; size_t kstart[64],klen[64]; } mm_job;  /* KSPLIT-0: ks K blocks, a task each */
#include <stdlib.h>
#define KS_ARENA ((size_t)96<<20)
static float *ks_arena; static size_t ks_used;
static float *ks_take(size_t floats) {
  size_t bytes=(floats*sizeof(float)+63)/64*64;
  if (!ks_arena) { size_t i; ks_arena=(float *)aligned_alloc(64,KS_ARENA); if (!ks_arena) return NULL;
    for (i=0;i<KS_ARENA/sizeof(float);i++) ks_arena[i]=0.0f; }
  if (ks_used+bytes>KS_ARENA) return NULL;
  { float *p=(float *)((char *)ks_arena+ks_used); ks_used+=bytes; return p; }
}
static void ks_task(mm_job *j, size_t b) {  /* KSPLIT-0: block b's product over all of C, raw */
  mm_args xb=j->x; mm_range r; size_t mn=j->x.m*j->x.n; float *p=j->ws+b*mn;
  xb.k=j->klen[b]; xb.a=j->x.a+j->kstart[b]*j->x.lda; xb.b=j->x.b+j->kstart[b]*j->x.ldb;
  xb.c=p; xb.ldc=j->x.n; xb.alpha=1.0f; xb.beta=0.0f; xb.fold=2;  /* 2: the accumulators stored as they are */
  r.jb0=0; r.jb1=j->x.n; r.r0=0; r.r1=j->x.m;
  mm_run_one(j->variant,j->fused,&xb,&r);
}
typedef struct ks_fold_job { mm_job *j; } ks_fold_job;
static void ks_fold_task(void *ctx, size_t task) {  /* rows [16 task, 16 task + 16) of C, every block in order */
  mm_job *j=((ks_fold_job *)ctx)->j; size_t i,jj,b,n=j->x.n,mn=j->x.m*n;
  size_t r0=task*16,r1=r0+16<j->x.m ? r0+16 : j->x.m;
  for (i=r0;i<r1;i++) {
    float *ci=j->x.c+i*j->x.ldc;
    for (b=0;b<j->ks;b++) {
      const float *p=j->ws+b*mn+i*n;
      for (jj=0;jj<n;jj++) {
        float c=b==0 && j->x.fold ? mm_scaled(j->x.beta,ci[jj]) : ci[jj];
        float t=j->x.alpha*p[jj];
        ci[jj]=c+t;
      }
    }
  }
}
static void ks_fold(bcir_tensor_pool *pool, mm_job *j);
""",
)
src = once(
    src,
    """  mm_job *j=(mm_job *)ctx; mm_range r; size_t col=task%j->cols,row=task/j->cols,i,jj;
  r.jb0=col*j->nr;""",
    """  mm_job *j=(mm_job *)ctx; mm_range r; size_t col=task%j->cols,row=task/j->cols,i,jj;
  if (j->ks) { ks_task(j,task); return; }
  r.jb0=col*j->nr;""",
)
src = once(
    src,
    """  j->cols=(n+j->nr-1)/j->nr; j->rowsets=(m+BCIR_MM_ROWS-1)/BCIR_MM_ROWS;
  return j->cols*j->rowsets;
}""",
    """  j->cols=(n+j->nr-1)/j->nr; j->rowsets=(m+BCIR_MM_ROWS-1)/BCIR_MM_ROWS; j->ks=0; j->ws=NULL;
  if (fused==MM_KBLOCK && ta && !tb && k>=4*BCIR_MM_KBLOCK && variant==4) {  /* KSPLIT-0 */
    size_t ls=0,cnt=0;
    while (ls<k && cnt<64) { size_t len=kblock_next(k-ls); j->kstart[cnt]=ls; j->klen[cnt]=len; cnt++; ls+=len; }
    if (ls==k) {
      j->ws=ks_take(cnt*m*n);
      if (j->ws) { j->ks=cnt; return cnt; }
    }
  }
  return j->cols*j->rowsets;
}""",
)
src = once(
    src,
    """  tasks=mm_job_init(&j,fused,variant,ta,tb,m,n,k,alpha,a,b,beta,c);
  pool_run(pool,mm_task,&j,tasks);
}""",
    """  ks_used=0;
  tasks=mm_job_init(&j,fused,variant,ta,tb,m,n,k,alpha,a,b,beta,c);
  pool_run(pool,mm_task,&j,tasks);
  if (j.ks) ks_fold(pool,&j);
}""",
)
src = once(
    src,
    """static double mm_task_work(const mm_job *j) {
  double rows""",
    """static double mm_task_work(const mm_job *j) {
  if (j->ks) return (double)j->x.m*(double)j->x.n*(double)BCIR_MM_KBLOCK;
  {
  double rows""",
)
src = once(
    src,
    """  return rows*cols*(double)(j->x.k ? j->x.k : 1);
}""",
    """  return rows*cols*(double)(j->x.k ? j->x.k : 1);
  }
}
static void ks_fold(bcir_tensor_pool *pool, mm_job *j) {
  ks_fold_job f; f.j=j;
  pool_run(pool,ks_fold_task,&f,(j->x.m+15)/16);
  j->ws=NULL;
}""",
)
src = once(
    src,
    """  pool_run_led(pool,mm_pair_task,&j,j.total,j.lead ? j.total-j.split : 0);
}""",
    """  pool_run_led(pool,mm_pair_task,&j,j.total,j.lead ? j.total-j.split : 0);
  if (j.a.ks) ks_fold(pool,&j.a);
  if (j.b.ks) ks_fold(pool,&j.b);
}""",
)
src = once(
    src,
    """  pool_run_led(pool,mm_group_task,&j,j.total,j.lead ? j.total-j.split : 0);
}""",
    """  pool_run_led(pool,mm_group_task,&j,j.total,j.lead ? j.total-j.split : 0);
  for (q=0;q<nx;q++) if (j.x[q].ks) ks_fold(pool,&j.x[q]);
  for (q=0;q<ny;q++) if (j.y[q].ks) ks_fold(pool,&j.y[q]);
}""",
)

src = once(
    src,
    """  j.split=mm_job_init(&j.a,fused,variant,x->ta,x->tb,x->m,x->n,x->k,x->alpha,x->a,x->b,x->beta,x->c);""",
    """  ks_used=0;
  j.split=mm_job_init(&j.a,fused,variant,x->ta,x->tb,x->m,x->n,x->k,x->alpha,x->a,x->b,x->beta,x->c);""",
)
src = once(
    src,
    """  j.nx=nx; j.split=0; j.start[0]=0;""",
    """  ks_used=0;
  j.nx=nx; j.split=0; j.start[0]=0;""",
)

src = once(
    src,
    """  alpha=_mm512_set1_ps(x->alpha); beta=_mm512_set1_ps(x->beta);
  keep=(__mmask16)(x->fold && lb==0 && x->beta==0.0f ? 0u : 0xffffu);""",
    """  if (x->fold==2) {  /* KSPLIT-0: the block's own product, raw */
    __mmask16 m0=(__mmask16)(nj>=16 ? 0xffffu : (1u<<nj)-1u);
    __mmask16 m1=(__mmask16)(nj>=32 ? 0xffffu : nj>16 ? (1u<<(nj-16))-1u : 0u);
    __mmask16 m2=(__mmask16)(nj>32 ? (1u<<(nj-32))-1u : 0u);
    for (r=0;r<mr;r++) {
      float *ci=x->c+(i+r)*x->ldc+jb;
      _mm512_mask_storeu_ps(ci,m0,acc0[r]);
      if (nv>1) _mm512_mask_storeu_ps(ci+16,m1,acc1[r]);
      if (nv>2) _mm512_mask_storeu_ps(ci+32,m2,acc2[r]);
    }
    return;
  }
  alpha=_mm512_set1_ps(x->alpha); beta=_mm512_set1_ps(x->beta);
  keep=(__mmask16)(x->fold && lb==0 && x->beta==0.0f ? 0u : 0xffffu);""",
)
Path(sys.argv[2]).write_text(src, encoding="utf-8")
print("KSPLIT-0 copy written")
