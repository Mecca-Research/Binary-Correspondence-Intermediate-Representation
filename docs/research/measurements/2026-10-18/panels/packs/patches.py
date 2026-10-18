"""PACK-ATTR and GEMM-OVERLAP (2026-10-18): four scratch copies of runtime/c/bcir_tensor.c, each beside
an unchanged bcir_decoder_train.c, for timing only -- the tree is not edited.

  probe  every pack task timed, by role (the forward x W^T, the input adjoint dy W, the weight adjoint
         dy^T x) and operand (A, B), and the wall time of each group's pack and compute phases
         (bcir_tensor_pack_probe reads and clears them)
  swap   the probe, with the untransposed B pack's two loops swapped (each tile written contiguously)
  fused  one job a group: the packs first, then the compute, a compute task waiting until every
         pack task of its call and the calls before it is done
  fine   one job a group, a compute region waiting for the strips it reads: its 32-row A strip (A
         untransposed) and its columns' B strips (B transposed), otherwise the operand's whole pack

    python3 patches.py RUNTIME_C OUT"""

import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
base = (src / "bcir_tensor.c").read_text()
unit = (src / "bcir_decoder_train.c").read_text()
HEAD = "#include <stdatomic.h>\n#include <immintrin.h>\n#include <time.h>\n"


def once(text, old, new):
    assert text.count(old) == 1, old[:60]
    return text.replace(old, new)


def write(name, text):
    (out / name).mkdir(parents=True, exist_ok=True)
    (out / name / "bcir_tensor.c").write_text(HEAD + text)
    (out / name / "bcir_decoder_train.c").write_text(unit)


TASK = """static void bf16_pack_task(void *ctx, size_t task) {
  bf16_job *j=(bf16_job *)ctx; size_t q=0; const bf16_call *c;
  while (task>=j->packs[q+1]) q++;
  c=&j->call[q]; task-=j->packs[q];
"""
RUN = """  bf16_prefix(&j);
  pool_run(pool,bf16_pack_task,&j,j.packs[calls]);
  pool_run(pool,bf16_compute_task,&j,j.split+j.tasks[calls-nchain]);
  return 1;
}"""
PROBE_TASK = """static _Atomic unsigned long long probe_ns[3][2],probe_wall[2],probe_groups;
static unsigned long long probe_now(void) {
  struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
  return (unsigned long long)t.tv_sec*1000000000ull+(unsigned long long)t.tv_nsec;
}
void bcir_tensor_pack_probe(unsigned long long *out);
void bcir_tensor_pack_probe(unsigned long long *out) {
  int r,o;
  for (r=0;r<3;r++) for (o=0;o<2;o++) out[2*r+o]=atomic_exchange(&probe_ns[r][o],0);
  out[6]=atomic_exchange(&probe_wall[0],0); out[7]=atomic_exchange(&probe_wall[1],0);
  out[8]=atomic_exchange(&probe_groups,0);
}
static void bf16_pack_task_real(void *ctx, size_t task);
static void bf16_pack_task(void *ctx, size_t task) {
  bf16_job *j=(bf16_job *)ctx; size_t q=0; const bf16_call *c; unsigned long long t0=probe_now(); int role,op;
  while (task>=j->packs[q+1]) q++;
  c=&j->call[q]; role=c->x.ta ? 2 : (c->x.tb ? 0 : 1); op=task-j->packs[q]<c->pa ? 0 : 1;
  bf16_pack_task_real(ctx,task);
  atomic_fetch_add(&probe_ns[role][op],probe_now()-t0);
}
static void bf16_pack_task_real(void *ctx, size_t task) {
  bf16_job *j=(bf16_job *)ctx; size_t q=0; const bf16_call *c;
  while (task>=j->packs[q+1]) q++;
  c=&j->call[q]; task-=j->packs[q];
"""
PROBE_RUN = """  bf16_prefix(&j);
  { unsigned long long t0=probe_now(),t1;
    pool_run(pool,bf16_pack_task,&j,j.packs[calls]); t1=probe_now();
    pool_run(pool,bf16_compute_task,&j,j.split+j.tasks[calls-nchain]);
    atomic_fetch_add(&probe_wall[0],t1-t0); atomic_fetch_add(&probe_wall[1],probe_now()-t1);
    atomic_fetch_add(&probe_groups,1); }
  return 1;
}"""
probe = once(once(base, TASK, PROBE_TASK), RUN, PROBE_RUN)
write("probe", probe)
SWAP_OLD = "    for (p=0;p<16;p++) for (lc=0;lc<c->np/16;lc++) { \\\n"
write("swap", once(probe, SWAP_OLD, "    for (lc=0;lc<c->np/16;lc++) for (p=0;p<16;p++) { \\\n"))

GROUP = "static int bf16_group_any("
FUSED = r"""typedef struct bf16_fused { bf16_job *j; size_t npack; _Atomic size_t done[BF16_CALLS]; } bf16_fused;
static void bf16_fused_task(void *ctx, size_t task) {
  bf16_fused *f=(bf16_fused *)ctx; bf16_job *j=f->j; size_t q,upto;
  if (task<f->npack) {
    bf16_pack_task(j,task);
    for (q=0;task>=j->packs[q+1];q++) {}
    atomic_fetch_add_explicit(&f->done[q],1,memory_order_release);
    return;
  }
  task-=f->npack;
  if (task<j->split) upto=j->nchain;
  else { size_t t=task-j->split; for (q=0;t>=j->tasks[q+1];q++) {} upto=j->nchain+q+1; }
  for (q=0;q<upto;q++) {
    size_t need=j->packs[q+1]-j->packs[q];
    while (atomic_load_explicit(&f->done[q],memory_order_acquire)<need) _mm_pause();
  }
  bf16_compute_task(j,task);
}
"""
FUSED_RUN = """  bf16_prefix(&j);
  { bf16_fused f; f.j=&j; f.npack=j.packs[calls]; for (q=0;q<calls;q++) atomic_init(&f.done[q],0);
    pool_run(pool,bf16_fused_task,&f,f.npack+j.split+j.tasks[calls-nchain]); }
  return 1;
}"""
write("fused", once(once(base, GROUP, FUSED + GROUP), RUN, FUSED_RUN))

LOOP = """    if (pre && pre[q].a) c->ap=pre[q].a;
    for (r=0;r<q && !c->ap;r++)
      if (!(pre && pre[r].a) && bf16_packs(bf16_nth(chain,nchain,y,r)) &&
          bf16_same_a(bf16_nth(chain,nchain,y,r),x)) c->ap=j.call[r].ap;
    if (!c->ap) { c->wa=base; c->ap=base; base+=c->mp*c->kt*32; c->pa=x->ta ? c->kt : c->mp/32; }
    if (pre && pre[q].b) c->bp=pre[q].b;
    for (r=0;r<q && !c->bp;r++)
      if (!(pre && pre[r].b) && bf16_packs(bf16_nth(chain,nchain,y,r)) &&
          bf16_same_b(bf16_nth(chain,nchain,y,r),x)) c->bp=j.call[r].bp;
    if (!c->bp) { c->wb=base; c->bp=base; base+=c->np*c->kt*32; c->pb=x->tb ? c->np/32 : c->kt; }"""
FINE_LOOP = """    if (pre && pre[q].a) c->ap=pre[q].a;
    for (r=0;r<q && !c->ap;r++)
      if (!(pre && pre[r].a) && bf16_packs(bf16_nth(chain,nchain,y,r)) &&
          bf16_same_a(bf16_nth(chain,nchain,y,r),x)) { c->ap=j.call[r].ap; f.oa[q]=f.oa[r]; }
    if (!c->ap) { c->wa=base; c->ap=base; base+=c->mp*c->kt*32; c->pa=x->ta ? c->kt : c->mp/32; f.oa[q]=(int)q; }
    if (pre && pre[q].b) c->bp=pre[q].b;
    for (r=0;r<q && !c->bp;r++)
      if (!(pre && pre[r].b) && bf16_packs(bf16_nth(chain,nchain,y,r)) &&
          bf16_same_b(bf16_nth(chain,nchain,y,r),x)) { c->bp=j.call[r].bp; f.ob[q]=f.ob[r]; }
    if (!c->bp) { c->wb=base; c->bp=base; base+=c->np*c->kt*32; c->pb=x->tb ? c->np/32 : c->kt; f.ob[q]=(int)q; }"""
DECL = "  size_t q,r,calls=nchain+ny,need,usable=bytes; int grouped,panels=0; bf16_job j; uint16_t *base;\n"
INIT = (
    "  base=bf16_base(workspace,&usable);\n  j.ncall=calls; j.nchain=nchain; j.variant=variant;\n"
)
FINE = r"""#define FINE_MAX 64
typedef struct bf16_fine {
  bf16_job *j; size_t npack; _Atomic size_t done[BF16_CALLS];
  _Atomic unsigned char fa[BF16_CALLS][FINE_MAX],fb[BF16_CALLS][FINE_MAX]; int oa[BF16_CALLS],ob[BF16_CALLS];
} bf16_fine;
static void fine_wait(_Atomic unsigned char *flag) {
  while (!atomic_load_explicit(flag,memory_order_acquire)) _mm_pause();
}
static void fine_needs(bf16_fine *f, size_t q, size_t t) {
  const bf16_call *c=&f->j->call[q]; int oa=f->oa[q],ob=f->ob[q]; size_t s;
  if (oa>=0) {
    const bf16_call *o=&f->j->call[oa];
    if (!c->x.ta) fine_wait(&f->fa[oa][t%c->tr]);
    else for (s=0;s<o->pa;s++) fine_wait(&f->fa[oa][s]);
  }
  if (ob>=0) {
    const bf16_call *o=&f->j->call[ob];
    if (c->x.tb) {
      size_t c0=(t/c->tr)*c->cols,c1=c0+c->cols;
      if (c1>c->np) c1=c->np;
      for (s=c0/32;s<(c1+31)/32;s++) fine_wait(&f->fb[ob][s]);
    }
    else for (s=0;s<o->pb;s++) fine_wait(&f->fb[ob][s]);
  }
}
static void bf16_fine_task(void *ctx, size_t task) {
  bf16_fine *f=(bf16_fine *)ctx; bf16_job *j=f->j; size_t q;
  if (task<f->npack) {
    size_t l; bf16_pack_task(j,task);
    for (q=0;task>=j->packs[q+1];q++) {}
    l=task-j->packs[q];
    if (l<j->call[q].pa) atomic_store_explicit(&f->fa[q][l],1,memory_order_release);
    else atomic_store_explicit(&f->fb[q][l-j->call[q].pa],1,memory_order_release);
    return;
  }
  task-=f->npack;
  if (task<j->split) { for (q=0;q<j->nchain;q++) fine_needs(f,q,task); }
  else { size_t t=task-j->split; for (q=0;t>=j->tasks[q+1];q++) {} fine_needs(f,j->nchain+q,t-j->tasks[q]); }
  bf16_compute_task(j,task);
}
"""
FINE_RUN = """  bf16_prefix(&j);
  { int ok=1; f.j=&j; f.npack=j.packs[calls];
    for (q=0;q<calls;q++) {
      ok&=j.call[q].pa<=FINE_MAX && j.call[q].pb<=FINE_MAX; atomic_init(&f.done[q],0);
      for (r=0;r<FINE_MAX;r++) { atomic_init(&f.fa[q][r],0); atomic_init(&f.fb[q][r],0); }
    }
    if (ok) pool_run(pool,bf16_fine_task,&f,f.npack+j.split+j.tasks[calls-nchain]);
    else { pool_run(pool,bf16_pack_task,&j,j.packs[calls]);
      pool_run(pool,bf16_compute_task,&j,j.split+j.tasks[calls-nchain]); } }
  return 1;
}"""
fine = once(base, GROUP, FINE + GROUP)
fine = once(fine, LOOP, FINE_LOOP)
fine = once(fine, DECL, DECL + "  static bf16_fine f;\n")
fine = once(fine, INIT, INIT + "  for (q=0;q<calls;q++) { f.oa[q]=-1; f.ob[q]=-1; }\n")
write("fine", once(fine, RUN, FINE_RUN))
write("base", base)
print("probe, swap, fused, fine, base written")
