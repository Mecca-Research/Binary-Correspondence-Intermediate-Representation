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
from .lower.c_kernel import emit_kernel_c
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
