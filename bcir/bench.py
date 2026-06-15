"""bcir.bench -- the measured-evidence rail (the honest counterpart to the modeled
K_BCIR score).

The optimizer's score is a *model*. This rail asks the silicon: does the
cost-model-selected lane geometry actually win? It compiles BCIR's selected
realization and a naive baseline (the scalar width-1 kernel) with the host
toolchain, times both, and reports the measured speedup.

The niche it targets is the **driver-as-compiler** case: a resident toolchain
compiles fast (a low opt level), so the explicit lane geometry BCIR emits -- the
width-W loop -- delivers SIMD/ILP that the naive scalar kernel does not get from
the compiler alone. At aggressive `-O3` the compiler's own vectorizer closes the
gap; at the `-O1`/`-O2` a JIT actually uses, BCIR's choice is measurably faster.
This is the first measured evidence that the cost model earns its keep, not a
claim of replacing LLVM's backend.

Offline tooling (it shells out to a compiler); never the hot path. Timings are
environment-dependent -- the rail reports the measured number, it does not pin it.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass

from .examples import PROGRAMS
from .kbcir import TARGETS, optimize
from .kbcir.cost import Theta
from .kbcir.weights import PERF, POLICIES
from .lower.c_kernel import emit_gather_kernel_c, emit_kernel_c, emit_reduce_c, find_reduce
from .lower.llvm import find_elementwise

_THETAS = {"cool": Theta.cool(), "hot": Theta.hot(), "mem_bound": Theta.mem_bound()}
_NS_RE = re.compile(r"^NS (\d+)$", re.MULTILINE)


@dataclass(frozen=True)
class Measurement:
    label: str
    width: int
    opt: str
    n: int
    reps: int
    ns_per_call: int           # best (minimum) wall-time per kernel call
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class Comparison:
    program: str
    target: str
    opt: str
    bcir: Measurement          # BCIR's cost-model-selected realization
    baseline: Measurement      # the naive scalar (width-1) baseline

    @property
    def speedup_milli(self) -> int:
        """baseline / bcir in milli (1000 == parity; >1000 == BCIR faster)."""
        if not (self.bcir.ok and self.baseline.ok and self.bcir.ns_per_call > 0):
            return 0
        return (self.baseline.ns_per_call * 1000) // self.bcir.ns_per_call

    @property
    def bcir_wins(self) -> bool:
        return (self.bcir.ok and self.baseline.ok
                and self.bcir.ns_per_call < self.baseline.ns_per_call)


def _emit_timed_c(module, result, fn_name: str, elem: str, n: int, reps: int,
                  width_override) -> str:
    kernel = emit_kernel_c(module, result, fn_name, elem, width_override=width_override)
    claim, _ = find_elementwise(module, result)
    op = "+" if elem != "i32" else "+"
    ctype = "int32_t" if elem == "i32" else "float"
    init = ("A[i] = (float)(i % 1000); B[i] = (float)(i % 7);"
            if elem != "i32" else "A[i] = (int32_t)(i % 1000); B[i] = (int32_t)(i % 7);")
    return (
        "#include <stddef.h>\n#include <stdio.h>\n#include <stdlib.h>\n"
        "#include <time.h>\n#include <limits.h>\n" + ("#include <stdint.h>\n" if elem == "i32" else "")
        + "\n" + kernel
        + f"""
static long now_ns(void) {{ struct timespec t; timespec_get(&t, TIME_UTC);
  return (long)t.tv_sec * 1000000000L + (long)t.tv_nsec; }}

int main(void) {{
  size_t n = {n}u;
  {ctype} *A = malloc(n*sizeof *A), *B = malloc(n*sizeof *B), *C = malloc(n*sizeof *C);
  if (!A || !B || !C) return 2;
  for (size_t i = 0; i < n; ++i) {{ {init} C[i] = 0; }}
  {fn_name}(A, B, C, n);                         /* warm up */
  long best = LONG_MAX;
  for (int r = 0; r < {reps}; ++r) {{
    long t0 = now_ns();
    {fn_name}(A, B, C, n);
    long dt = now_ns() - t0;
    if (dt > 0 && dt < best) best = dt;
  }}
  volatile {ctype} sink = C[n - 1]; (void)sink;  /* keep the kernel live */
  printf("NS %ld\\n", best);
  free(A); free(B); free(C);
  return 0;
}}
"""
    )


def _which(*names):
    from shutil import which
    for n in names:
        p = which(n)
        if p:
            return p
    return None


def measure(module, result, *, label: str, elem: str = "f32", opt: str = "-O1",
            n: int = 1 << 20, reps: int = 200, width_override=None) -> Measurement:
    """Compile + time one realization (the selected width, or `width_override`)."""
    _, cand = find_elementwise(module, result)
    w = int(width_override) if width_override else int(cand.width)
    cc = _which("clang", "cc", "gcc")
    if cc is None:
        return Measurement(label, w, opt, n, reps, 0, False, "no C compiler")
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "bench.c")
        exe = os.path.join(d, "bench")
        with open(src, "w") as f:
            f.write(_emit_timed_c(module, result, "bcir_kernel", elem, n, reps, width_override))
        build = None
        for std in ("-std=c23", "-std=c2x"):
            build = subprocess.run([cc, std, opt, src, "-o", exe], capture_output=True, text=True)
            if build.returncode == 0:
                break
        if build is None or build.returncode != 0:
            return Measurement(label, w, opt, n, reps, 0, False,
                               "build failed: " + (build.stderr if build else ""))
        run = subprocess.run([exe], capture_output=True, text=True)
        m = _NS_RE.search(run.stdout or "")
        if run.returncode != 0 or not m:
            return Measurement(label, w, opt, n, reps, 0, False, "run failed: " + run.stdout + run.stderr)
        return Measurement(label, w, opt, n, reps, int(m.group(1)), True)


def compare(program, *, target: str = "x86_avx512", theta: str = "cool",
            policy: str = "latency", elem: str = "f32", opt: str = "-O1",
            n: int = 1 << 20, reps: int = 200) -> Comparison:
    """Time BCIR's selected realization against the naive scalar (width-1) baseline
    at one optimization level -- the measured evidence for the lane-geometry choice."""
    module = PROGRAMS[program]() if isinstance(program, str) else program
    name = program if isinstance(program, str) else getattr(program, "name", "module")
    h = TARGETS[target]
    th = _THETAS.get(theta, Theta.cool())
    pol = POLICIES.get(policy, PERF)
    result = optimize(module, h, th, pol)
    bcir = measure(module, result, label="bcir-selected", elem=elem, opt=opt, n=n, reps=reps)
    base = measure(module, result, label="scalar-baseline", elem=elem, opt=opt, n=n,
                   reps=reps, width_override=1)
    return Comparison(program=name, target=h.name, opt=opt, bcir=bcir, baseline=base)


def bench_available() -> bool:
    return _which("clang", "cc", "gcc") is not None


# --- gather avoidance: BCIR's direct realization vs the gather it avoids ----------

@dataclass(frozen=True)
class GatherComparison:
    program: str
    opt: str
    n: int
    reps: int
    shuffle: bool
    direct: Measurement        # BCIR's cost-model choice (contiguous, vectorizable)
    gather: Measurement        # the avoided realization (indexed loads -> gather_penalty)

    @property
    def speedup_milli(self) -> int:
        """gather / direct in milli (>1000 == the avoided gather is that much dearer)."""
        if not (self.direct.ok and self.gather.ok and self.direct.ns_per_call > 0):
            return 0
        return (self.gather.ns_per_call * 1000) // self.direct.ns_per_call

    @property
    def avoidance_wins(self) -> bool:
        return (self.direct.ok and self.gather.ok
                and self.direct.ns_per_call < self.gather.ns_per_call)


_DIRECT_RE = re.compile(r"^DIRECT (\d+)$", re.MULTILINE)
_GATHER_RE = re.compile(r"^GATHER (\d+)$", re.MULTILINE)


def _emit_gather_bench_c(module, result, elem: str, n: int, reps: int, shuffle: bool) -> str:
    direct = emit_kernel_c(module, result, "bcir_direct", elem)        # BCIR's choice
    gather = emit_gather_kernel_c(module, result, "bcir_gather", elem)  # the avoided form
    op = "+" if find_elementwise(module, result)[0].opcode.name == "ADD" else (
        "-" if find_elementwise(module, result)[0].opcode.name == "SUB" else "*")
    ctype = "int32_t" if elem == "i32" else "float"
    init = ("A[i] = (float)(i % 1000); B[i] = (float)(i % 7);"
            if elem != "i32" else "A[i] = (int32_t)(i % 1000); B[i] = (int32_t)(i % 7);")
    shuf = ("for (size_t i = n - 1; i > 0; --i) { size_t j = (size_t)(lcg() % (i + 1));"
            " long t = idx[i]; idx[i] = idx[j]; idx[j] = t; }") if shuffle else ""
    return (
        "#include <stddef.h>\n#include <stdio.h>\n#include <stdlib.h>\n"
        "#include <time.h>\n#include <limits.h>\n#include <stdint.h>\n\n"
        + direct + "\n" + gather
        + f"""
static long now_ns(void) {{ struct timespec t; timespec_get(&t, TIME_UTC);
  return (long)t.tv_sec * 1000000000L + (long)t.tv_nsec; }}
static uint64_t g_st = 0xBC12u;
static uint64_t lcg(void) {{ g_st = g_st * 6364136223846793005ULL + 1442695040888963407ULL; return g_st; }}

int main(void) {{
  size_t n = {n}u;
  {ctype} *A = malloc(n*sizeof *A), *B = malloc(n*sizeof *B);
  {ctype} *Cd = malloc(n*sizeof *Cd), *Cg = malloc(n*sizeof *Cg);
  long *idx = malloc(n*sizeof *idx);
  if (!A || !B || !Cd || !Cg || !idx) return 2;
  for (size_t i = 0; i < n; ++i) {{ {init} Cd[i] = 0; Cg[i] = 0; idx[i] = (long)i; }}
  {shuf}
  bcir_direct(A, B, Cd, n);                       /* warm up + correctness baseline */
  bcir_gather(A, B, Cg, idx, n);
  for (size_t i = 0; i < n; ++i) {{
    {ctype} wd = A[i] {op} B[i];
    {ctype} wg = A[idx[i]] {op} B[i];
    if (Cd[i] != wd || Cg[i] != wg) {{ printf("FAIL i=%zu\\n", i); return 1; }}
  }}
  long bd = LONG_MAX, bg = LONG_MAX;
  for (int r = 0; r < {reps}; ++r) {{ long t0 = now_ns(); bcir_direct(A, B, Cd, n);
    long d = now_ns() - t0; if (d > 0 && d < bd) bd = d; }}
  for (int r = 0; r < {reps}; ++r) {{ long t0 = now_ns(); bcir_gather(A, B, Cg, idx, n);
    long d = now_ns() - t0; if (d > 0 && d < bg) bg = d; }}
  volatile {ctype} s = Cd[n - 1] + Cg[n - 1]; (void)s;
  printf("DIRECT %ld\\nGATHER %ld\\n", bd, bg);
  free(A); free(B); free(Cd); free(Cg); free(idx);
  return 0;
}}
"""
    )


def compare_gather(program, *, target: str = "x86_avx512", theta: str = "cool",
                   policy: str = "latency", elem: str = "f32", opt: str = "-O2",
                   n: int = 1 << 20, reps: int = 200, shuffle: bool = True) -> GatherComparison:
    """Time BCIR's direct (gather-avoiding) realization against the gather form of
    the same elementwise computation -- the measured value of avoiding GGG. With
    `shuffle`, the gather indices are a random permutation (real cache-miss
    gather, the gather_penalty realized); otherwise identity (the indexed-load
    overhead alone)."""
    module = PROGRAMS[program]() if isinstance(program, str) else program
    name = program if isinstance(program, str) else getattr(program, "name", "module")
    h = TARGETS[target]
    th = _THETAS.get(theta, Theta.cool())
    pol = POLICIES.get(policy, PERF)
    result = optimize(module, h, th, pol)
    _, cand = find_elementwise(module, result)
    bad = Measurement("", cand.width, opt, n, reps, 0, False, "build/run failed")

    cc = _which("clang", "cc", "gcc")
    if cc is None:
        m = Measurement("", cand.width, opt, n, reps, 0, False, "no C compiler")
        return GatherComparison(name, opt, n, reps, shuffle, m, m)
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "gbench.c")
        exe = os.path.join(d, "gbench")
        with open(src, "w") as f:
            f.write(_emit_gather_bench_c(module, result, elem, n, reps, shuffle))
        build = None
        for std in ("-std=c23", "-std=c2x"):
            build = subprocess.run([cc, std, opt, src, "-o", exe], capture_output=True, text=True)
            if build.returncode == 0:
                break
        if build is None or build.returncode != 0:
            return GatherComparison(name, opt, n, reps, shuffle, bad, bad)
        run = subprocess.run([exe], capture_output=True, text=True)
        dm = _DIRECT_RE.search(run.stdout or "")
        gm = _GATHER_RE.search(run.stdout or "")
        if run.returncode != 0 or not (dm and gm):
            return GatherComparison(name, opt, n, reps, shuffle, bad, bad)
        direct = Measurement("bcir-direct", cand.width, opt, n, reps, int(dm.group(1)), True)
        gather = Measurement("gather-avoided", 1, opt, n, reps, int(gm.group(1)), True)
        return GatherComparison(name, opt, n, reps, shuffle, direct, gather)


# --- gather/blocked reduction: a genuine gather program, lowered two ways --------

@dataclass(frozen=True)
class ReduceComparison:
    program: str
    opt: str
    n: int
    reps: int
    selected: str              # the realization BCIR's cost model selected ("blocked")
    correct: bool              # blocked and gather computed the identical sum
    blocked: Measurement
    gather: Measurement

    @property
    def speedup_milli(self) -> int:
        if not (self.blocked.ok and self.gather.ok and self.blocked.ns_per_call > 0):
            return 0
        return (self.gather.ns_per_call * 1000) // self.blocked.ns_per_call

    @property
    def avoidance_wins(self) -> bool:
        return (self.correct and self.blocked.ok and self.gather.ok
                and self.blocked.ns_per_call < self.gather.ns_per_call)


_BLK_RE = re.compile(r"^BLOCKED (\d+)$", re.MULTILINE)
_GRD_RE = re.compile(r"^GATHER (\d+)$", re.MULTILINE)


def _emit_reduce_bench_c(module, result, n: int, reps: int) -> str:
    blocked = emit_reduce_c(module, result, "bcir_blocked", "i32", gather=False)
    gather = emit_reduce_c(module, result, "bcir_gather_reduce", "i32", gather=True)
    return (
        "#include <stddef.h>\n#include <stdio.h>\n#include <stdlib.h>\n"
        "#include <time.h>\n#include <limits.h>\n#include <stdint.h>\n\n"
        + blocked + "\n" + gather
        + f"""
static long now_ns(void) {{ struct timespec t; timespec_get(&t, TIME_UTC);
  return (long)t.tv_sec * 1000000000L + (long)t.tv_nsec; }}
static uint64_t g_st = 0xBC12u;
static uint64_t lcg(void) {{ g_st = g_st * 6364136223846793005ULL + 1442695040888963407ULL; return g_st; }}

int main(void) {{
  size_t n = {n}u;
  int32_t *T = malloc(n*sizeof *T);
  long *idx = malloc(n*sizeof *idx);
  if (!T || !idx) return 2;
  for (size_t i = 0; i < n; ++i) {{ T[i] = (int32_t)(i % 1000); idx[i] = (long)i; }}
  for (size_t i = n - 1; i > 0; --i) {{ size_t j = (size_t)(lcg() % (i + 1));
    long t = idx[i]; idx[i] = idx[j]; idx[j] = t; }}             /* a permutation */
  int32_t sb = bcir_blocked(T, n);
  int32_t sg = bcir_gather_reduce(T, idx, n);
  if (sb != sg) {{ printf("FAIL %d != %d\\n", sb, sg); return 1; }}  /* identical sum */
  long bb = LONG_MAX, bg = LONG_MAX;
  for (int r = 0; r < {reps}; ++r) {{ long t0 = now_ns(); volatile int32_t s = bcir_blocked(T, n);
    (void)s; long d = now_ns() - t0; if (d > 0 && d < bb) bb = d; }}
  for (int r = 0; r < {reps}; ++r) {{ long t0 = now_ns(); volatile int32_t s = bcir_gather_reduce(T, idx, n);
    (void)s; long d = now_ns() - t0; if (d > 0 && d < bg) bg = d; }}
  printf("BLOCKED %ld\\nGATHER %ld\\n", bb, bg);
  free(T); free(idx);
  return 0;
}}
"""
    )


def compare_reduce(program="gather_reduce", *, target: str = "x86_avx512", theta: str = "cool",
                   policy: str = "latency", opt: str = "-O2", n: int = 1 << 20,
                   reps: int = 40) -> ReduceComparison:
    """Lower a genuine `reduce.gather` program two ways and measure: BCIR's
    selected blocked (sequential) realization vs the gather (random) form. The
    harness verifies both compute the identical integer sum (a correct
    transformation), then times them -- the gather_penalty realized end to end on a
    real gather workload BCIR actually lowered."""
    module = PROGRAMS[program]() if isinstance(program, str) else program
    name = program if isinstance(program, str) else getattr(program, "name", "module")
    h = TARGETS[target]
    th = _THETAS.get(theta, Theta.cool())
    pol = POLICIES.get(policy, PERF)
    result = optimize(module, h, th, pol)
    _, cand = find_reduce(module, result)
    bad = Measurement("", 1, opt, n, reps, 0, False, "build/run failed")
    cc = _which("clang", "cc", "gcc")
    if cc is None:
        return ReduceComparison(name, opt, n, reps, cand.name, False, bad, bad)
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "rbench.c")
        exe = os.path.join(d, "rbench")
        with open(src, "w") as f:
            f.write(_emit_reduce_bench_c(module, result, n, reps))
        build = None
        for std in ("-std=c23", "-std=c2x"):
            build = subprocess.run([cc, std, opt, src, "-o", exe], capture_output=True, text=True)
            if build.returncode == 0:
                break
        if build is None or build.returncode != 0:
            return ReduceComparison(name, opt, n, reps, cand.name, False, bad, bad)
        run = subprocess.run([exe], capture_output=True, text=True)
        bm, gm = _BLK_RE.search(run.stdout or ""), _GRD_RE.search(run.stdout or "")
        correct = run.returncode == 0 and "FAIL" not in run.stdout
        if not (bm and gm and correct):
            return ReduceComparison(name, opt, n, reps, cand.name, False, bad, bad)
        blocked = Measurement("blocked", 1, opt, n, reps, int(bm.group(1)), True)
        gather = Measurement("gather", 1, opt, n, reps, int(gm.group(1)), True)
        return ReduceComparison(name, opt, n, reps, cand.name, True, blocked, gather)
