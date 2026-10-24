"""Copies the tensor unit with TSC counters of each pool job a worker at a time: when the worker entered the job
after its dispatch, its cycles inside tasks, its tasks, and when it left the job; and each job's wall cycles at the
caller; and the CPU each worker ran each job on. Read and reset by bcir_probe_read and bcir_probe_cpus (the
counters summed over the jobs since the last read).
usage: probe_tasks.py SRC/bcir_tensor.c DST/bcir_tensor.c"""

import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text(encoding="utf-8")


def once(text, old, new):
    if text.count(old) != 1:
        raise SystemExit(f"anchor found {text.count(old)} times: {old[:70]!r}")
    return text.replace(old, new)


head = """#include <x86intrin.h>
#include <sched.h>
static volatile unsigned long long probe_d0;
static unsigned long long probe_cpu[8][8];
void bcir_probe_cpus(unsigned long long *out);
void bcir_probe_cpus(unsigned long long *out) {  /* 8 workers x the job count each ran on CPUs 0..7 */
  unsigned i,j;
  for (i=0;i<8;i++) for (j=0;j<8;j++) { out[8*i+j]=probe_cpu[i][j]; probe_cpu[i][j]=0; }
}
static unsigned long long probe_enter[8],probe_busy[8],probe_leave[8],probe_ntask[8],probe_wall,probe_jobs;
void bcir_probe_read(unsigned long long *out);
void bcir_probe_read(unsigned long long *out) {  /* 8 x {enter, busy, leave, tasks}, then wall, jobs */
  unsigned i;
  for (i=0;i<8;i++) { out[4*i]=probe_enter[i]; out[4*i+1]=probe_busy[i]; out[4*i+2]=probe_leave[i];
    out[4*i+3]=probe_ntask[i]; probe_enter[i]=probe_busy[i]=probe_leave[i]=probe_ntask[i]=0; }
  out[32]=probe_wall; out[33]=probe_jobs; probe_wall=probe_jobs=0;
}
#define PROBE_RUN(call) do { unsigned long long a_=__rdtsc(); call; \\
  probe_busy[id]+=__rdtsc()-a_; probe_ntask[id]++; } while (0)
"""
src = once(src, "#ifndef BCIR_POOL_SPINS\n", head + "#ifndef BCIR_POOL_SPINS\n")
src = once(
    src,
    """  if (id>=workers) return;  /* past the job's cap: the worker reports without claiming a task */
#if defined(BCIR_POOL_ATOMIC)
  {""",
    """  if (id>=workers) return;  /* past the job's cap: the worker reports without claiming a task */
#if defined(BCIR_POOL_ATOMIC)
  probe_enter[id]+=__rdtsc()-probe_d0;
  { int c_=sched_getcpu(); if (c_>=0 && c_<8) probe_cpu[id][c_]++; }
  {""",
)
old_a = """    while ((t=__atomic_fetch_add(&p->next,(size_t)1,__ATOMIC_RELAXED))<head) {
      if (fnw) fnw(ctx,t,id); else fn(ctx,t);
    }"""
src = once(
    src,
    old_a,
    """    while ((t=__atomic_fetch_add(&p->next,(size_t)1,__ATOMIC_RELAXED))<head) {
      PROBE_RUN(if (fnw) fnw(ctx,t,id); else fn(ctx,t));
    }""",
)
old_b = """      while ((t=__atomic_fetch_add(&b->next,(size_t)1,__ATOMIC_RELAXED))<b->end) {
        if (fnw) fnw(ctx,t,id); else fn(ctx,t);
      }
    }
  }
#else"""
src = once(
    src,
    old_b,
    """      while ((t=__atomic_fetch_add(&b->next,(size_t)1,__ATOMIC_RELAXED))<b->end) {
        PROBE_RUN(if (fnw) fnw(ctx,t,id); else fn(ctx,t));
      }
    }
  }
  probe_leave[id]+=__rdtsc()-probe_d0;
#else""",
)
src = once(
    src,
    """#if defined(BCIR_POOL_PTHREADS) || defined(BCIR_POOL_WIN32)
  POOL_LOCK(p);
  p->fn=fn; p->fnw=fnw; p->ctx=ctx; p->tasks=tasks;""",
    """#if defined(BCIR_POOL_PTHREADS) || defined(BCIR_POOL_WIN32)
  unsigned long long probe_w0=__rdtsc(); probe_d0=probe_w0;
  POOL_LOCK(p);
  p->fn=fn; p->fnw=fnw; p->ctx=ctx; p->tasks=tasks;""",
)
src = once(
    src,
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
  probe_wall+=__rdtsc()-probe_w0; probe_jobs++;
#else""",
)
Path(sys.argv[2]).write_text("#define _GNU_SOURCE 1\n" + src, encoding="utf-8")
print("instrumented copy written")
