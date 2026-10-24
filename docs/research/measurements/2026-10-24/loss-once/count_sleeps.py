"""Copies the tensor unit with counters of the pool's sleeps: each time a worker leaves its spins and yields for the
condition variable (a sleep), each job dispatched, and each time the caller falls back to waiting on `done`; a
destructor prints them to stderr at exit. usage: count_sleeps.py SRC/bcir_tensor.c DST/bcir_tensor.c"""

import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text(encoding="utf-8")


def once(text, old, new):
    if text.count(old) != 1:
        raise SystemExit(f"anchor found {text.count(old)} times: {old[:60]!r}")
    return text.replace(old, new)


WAIT_OLD = (
    "      else {\n"
    "        POOL_LOCK(p);\n"
    "        while (POOL_LOAD(p->generation)==seen && !POOL_LOAD(p->stop)) POOL_WAIT(p,wake);"
)
WAIT_NEW = (
    "      else {\n"
    "        __atomic_add_fetch(&probe_sleeps,1ul,__ATOMIC_RELAXED);\n"
    "        POOL_LOCK(p);\n"
    "        while (POOL_LOAD(p->generation)==seen && !POOL_LOAD(p->stop)) POOL_WAIT(p,wake);"
)
DONE_OLD = "  if (POOL_LOAD(p->pending)) {\n    POOL_LOCK(p);\n    while (POOL_LOAD(p->pending)) POOL_WAIT(p,done);"
DONE_NEW = (
    "  if (POOL_LOAD(p->pending)) {\n"
    "    __atomic_add_fetch(&probe_main_waits,1ul,__ATOMIC_RELAXED);\n"
    "    POOL_LOCK(p);\n"
    "    while (POOL_LOAD(p->pending)) POOL_WAIT(p,done);"
)
JOB_OLD = "  __atomic_store_n(&p->pending,p->threads-1,__ATOMIC_RELAXED);"
JOB_NEW = "  __atomic_add_fetch(&probe_jobs,1ul,__ATOMIC_RELAXED);\n" + JOB_OLD
src = once(src, WAIT_OLD, WAIT_NEW)
src = once(src, DONE_OLD, DONE_NEW)
src = once(src, JOB_OLD, JOB_NEW)
head = (
    "#include <stdio.h>\nstatic unsigned long probe_sleeps,probe_jobs,probe_main_waits;\n"
    "__attribute__((destructor)) static void probe_report(void) {\n"
    '  fprintf(stderr,"pool probe: jobs %lu worker sleeps %lu caller waits %lu\\n",probe_jobs,probe_sleeps,'
    "probe_main_waits);\n}\n"
)
# after the unit's own includes: before its first function, at the pool section's spin constants
src = once(src, "#ifndef BCIR_POOL_SPINS\n", head + "#ifndef BCIR_POOL_SPINS\n")
Path(sys.argv[2]).write_text(src, encoding="utf-8")
print("instrumented copy written")
