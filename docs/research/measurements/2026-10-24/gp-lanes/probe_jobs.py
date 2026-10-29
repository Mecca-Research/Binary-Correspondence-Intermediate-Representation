"""Copies the tensor unit and the trainer with a log of every pool job a worker at a time: the job's
launch (the trainer's record index, set as each launch starts), its task count, its wall TSC cycles
at the caller, and for each of four workers when it entered the job after its dispatch, its cycles
inside tasks, when it left, and its task count. Read and reset by bcir_probe_jobs; the TSC's rate a
microsecond by bcir_probe_tsc_per_us. Every value the units compute is theirs: the probe only reads
the clock around the calls. usage: probe_jobs.py SRC_DIR DST_DIR"""

import shutil
import sys
from pathlib import Path

SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
DST.mkdir(parents=True, exist_ok=True)
for name in ("bcir_tensor.h", "bcir_decoder_train.h"):
    shutil.copyfile(SRC / name, DST / name)
unit = (SRC / "bcir_tensor.c").read_text(encoding="utf-8")
train = (SRC / "bcir_decoder_train.c").read_text(encoding="utf-8")


def once(text, old, new):
    if text.count(old) != 1:
        raise SystemExit(f"anchor found {text.count(old)} times: {old[:70]!r}")
    return text.replace(old, new)


HEAD = """#include <x86intrin.h>
#define PROBE_JOBS 8192
typedef struct probe_job {
  unsigned long long launch,tasks,wall,enter[4],busy[4],leave[4],ntask[4];
} probe_job;
static probe_job probe_log[PROBE_JOBS];
static size_t probe_seq;
static volatile size_t probe_j;
static volatile unsigned long long probe_d0;
size_t bcir_probe_launch;
size_t bcir_probe_jobs(unsigned long long *out, size_t capacity);
size_t bcir_probe_jobs(unsigned long long *out, size_t capacity) {  /* 19 words a job */
  size_t n=probe_seq<PROBE_JOBS ? probe_seq : PROBE_JOBS,j,q;
  for (j=0;j<n && (j+1)*19<=capacity;j++) {
    const probe_job *r=&probe_log[j]; unsigned long long *o=out+19*j;
    o[0]=r->launch; o[1]=r->tasks; o[2]=r->wall;
    for (q=0;q<4;q++) { o[3+q]=r->enter[q]; o[7+q]=r->busy[q]; o[11+q]=r->leave[q]; o[15+q]=r->ntask[q]; }
  }
  memset(probe_log,0,sizeof probe_log); probe_seq=0;
  return j;
}
double bcir_probe_tsc_per_us(void);
double bcir_probe_tsc_per_us(void) {  /* the TSC against the C11 clock over 50 ms */
  struct timespec a,b; unsigned long long t0,t1; double us;
  timespec_get(&a,TIME_UTC); t0=__rdtsc();
  do { timespec_get(&b,TIME_UTC); } while ((b.tv_sec-a.tv_sec)*1e9+(b.tv_nsec-a.tv_nsec)<5e7);
  t1=__rdtsc(); us=((b.tv_sec-a.tv_sec)*1e9+(b.tv_nsec-a.tv_nsec))/1e3;
  return (double)(t1-t0)/us;
}
#define PROBE_RUN(call) do { unsigned long long a_=__rdtsc(); call; \\
  probe_log[probe_j].busy[id&3u]+=__rdtsc()-a_; probe_log[probe_j].ntask[id&3u]++; } while (0)
"""
unit = once(unit, "#ifndef BCIR_POOL_SPINS\n", HEAD + "#ifndef BCIR_POOL_SPINS\n")
unit = once(
    unit,
    """  if (id>=workers) return;  /* past the job's cap: the worker reports without claiming a task */
#if defined(BCIR_POOL_ATOMIC)
  {""",
    """  if (id>=workers) return;  /* past the job's cap: the worker reports without claiming a task */
#if defined(BCIR_POOL_ATOMIC)
  probe_log[probe_j].enter[id&3u]=__rdtsc()-probe_d0;
  {""",
)
unit = once(
    unit,
    """    while ((t=__atomic_fetch_add(&p->next,(size_t)1,__ATOMIC_RELAXED))<head) {
      if (fnw) fnw(ctx,t,id); else fn(ctx,t);
    }""",
    """    while ((t=__atomic_fetch_add(&p->next,(size_t)1,__ATOMIC_RELAXED))<head) {
      PROBE_RUN(if (fnw) fnw(ctx,t,id); else fn(ctx,t));
    }""",
)
unit = once(
    unit,
    """      while ((t=__atomic_fetch_add(&b->next,(size_t)1,__ATOMIC_RELAXED))<b->end) {
        if (fnw) fnw(ctx,t,id); else fn(ctx,t);
      }
    }
  }
#else""",
    """      while ((t=__atomic_fetch_add(&b->next,(size_t)1,__ATOMIC_RELAXED))<b->end) {
        PROBE_RUN(if (fnw) fnw(ctx,t,id); else fn(ctx,t));
      }
    }
  }
  probe_log[probe_j].leave[id&3u]=__rdtsc()-probe_d0;
#else""",
)
unit = once(
    unit,
    """#if defined(BCIR_POOL_PTHREADS) || defined(BCIR_POOL_WIN32)
  POOL_LOCK(p);
  p->fn=fn; p->fnw=fnw; p->ctx=ctx; p->tasks=tasks;""",
    """#if defined(BCIR_POOL_PTHREADS) || defined(BCIR_POOL_WIN32)
  unsigned long long probe_w0=__rdtsc(); probe_d0=probe_w0;
  probe_j=probe_seq<PROBE_JOBS ? probe_seq : PROBE_JOBS-1;
  probe_log[probe_j].launch=bcir_probe_launch; probe_log[probe_j].tasks=tasks;
  POOL_LOCK(p);
  p->fn=fn; p->fnw=fnw; p->ctx=ctx; p->tasks=tasks;""",
)
unit = once(
    unit,
    """  if (POOL_LOAD(p->pending)) {
    POOL_LOCK(p);
    while (POOL_LOAD(p->pending)) POOL_WAIT(p,done);
    POOL_UNLOCK(p);
  }
#else""",
    """  if (POOL_LOAD(p->pending)) {
    POOL_LOCK(p);
    while (POOL_LOAD(p->pending)) POOL_WAIT(p,done);
    POOL_UNLOCK(p);
  }
  probe_log[probe_j].wall=__rdtsc()-probe_w0; probe_seq++;
#else""",
)
train = once(
    train,
    "    DT_TIME_START(t0);\n    job=px_job(w,records-i);",
    "    DT_TIME_START(t0);\n    bcir_probe_launch=i;\n    job=px_job(w,records-i);",
)
train = once(
    train,
    "uint32_t bcir_decoder_abi_version(void) { return 5; }",
    "extern size_t bcir_probe_launch;\nuint32_t bcir_decoder_abi_version(void) { return 5; }",
)
(DST / "bcir_tensor.c").write_text(unit, encoding="utf-8")
(DST / "bcir_decoder_train.c").write_text(train, encoding="utf-8")
print("instrumented copies written to", DST)
