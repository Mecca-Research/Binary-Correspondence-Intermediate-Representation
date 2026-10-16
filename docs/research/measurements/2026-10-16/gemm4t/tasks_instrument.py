"""Copy the trainer's four C units into OUT with every GEMM task timed (NDT-GEMM-4T's instrument).

    python3 tasks_instrument.py SRC OUT

SRC is a tree's runtime/c. The copy records, for each task `mm_task` runs, the job, the task, a
thread index, the CPU and the monotonic start and end (`bcir_prof_reset`, `bcir_prof_count`,
`bcir_prof_get`), and routes a one-thread pool through the same tasks run in order on the caller
(the trainer's own one-thread path is the serial kernel over the whole matrix), so a task's one-
and four-thread times compare like for like. Build it with the trainer's flags without -Werror
(the routing leaves the serial helper unused); the results stay the kernel's bit for bit.
"""

import shutil
import sys
from pathlib import Path

PROF = r"""
#include <time.h>
#include <sched.h>
#define PROF_MAX (1u<<20)
typedef struct { unsigned job, task, tid, cpu; double t0, t1; } prof_rec;
static prof_rec prof_recs[PROF_MAX]; static size_t prof_n; static unsigned prof_job, prof_tids;
static _Thread_local int prof_tid=-1;
static double prof_now(void) {
  struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec;
}
void bcir_prof_reset(void) { prof_n=0; prof_job=0; }
size_t bcir_prof_count(void) { return prof_n<PROF_MAX ? prof_n : PROF_MAX; }
void bcir_prof_get(size_t k, unsigned *v, double *t) {
  v[0]=prof_recs[k].job; v[1]=prof_recs[k].task; v[2]=prof_recs[k].tid; v[3]=prof_recs[k].cpu;
  t[0]=prof_recs[k].t0; t[1]=prof_recs[k].t1;
}
"""
EDITS = (
    (
        "typedef struct mm_job { mm_args x; float beta; int variant,fused; size_t nr,cols,rowsets; } mm_job;\n",
        PROF
        + "typedef struct mm_job { mm_args x; float beta; int variant,fused; size_t nr,cols,rowsets; } mm_job;\n",
    ),
    (
        "  mm_job *j=(mm_job *)ctx; mm_range r; size_t col=task%j->cols,row=task/j->cols,i,jj;\n",
        "  mm_job *j=(mm_job *)ctx; mm_range r; size_t col=task%j->cols,row=task/j->cols,i,jj;"
        " double prof_t0=prof_now();\n",
    ),
    (
        "  if (j->x.k) mm_run_range(j->variant,j->fused,&j->x,&r);\n}\n",
        "  if (j->x.k) mm_run_range(j->variant,j->fused,&j->x,&r);\n"
        "  { size_t k=__atomic_fetch_add(&prof_n,(size_t)1,__ATOMIC_RELAXED);\n"
        "    if (prof_tid<0) prof_tid=(int)__atomic_fetch_add(&prof_tids,1u,__ATOMIC_RELAXED);\n"
        "    if (k<PROF_MAX) { prof_recs[k].job=prof_job; prof_recs[k].task=(unsigned)task;\n"
        "      prof_recs[k].tid=(unsigned)prof_tid; prof_recs[k].cpu=(unsigned)sched_getcpu();\n"
        "      prof_recs[k].t0=prof_t0; prof_recs[k].t1=prof_now(); } }\n}\n",
    ),
    (
        "  if (!pool || pool->threads==1) { mm_serial_any(fused,ta,tb,m,n,k,alpha,a,b,beta,c); return; }\n"
        "  tasks=mm_job_init(&j,fused,ta,tb,m,n,k,alpha,a,b,beta,c);\n",
        "  tasks=mm_job_init(&j,fused,ta,tb,m,n,k,alpha,a,b,beta,c);  /* one thread: the same tasks */\n",
    ),
)


def main(argv) -> int:
    src, out = Path(argv[1]), Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for name in ("bcir_tensor.h", "bcir_tensor.c", "bcir_decoder_train.h", "bcir_decoder_train.c"):
        shutil.copy(src / name, out / name)
    path = out / "bcir_tensor.c"
    text = path.read_text()
    for old, new in EDITS:
        if text.count(old) != 1:
            raise SystemExit(f"anchor not found exactly once: {old[:60]!r}")
        text = text.replace(old, new)
    path.write_text("#define _GNU_SOURCE 1\n" + text)
    print("instrumented", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
