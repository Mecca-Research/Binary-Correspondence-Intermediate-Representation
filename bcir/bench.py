"""bcir.bench -- the measured-evidence rail (the honest counterpart to the modeled
K_BCIR score).

The optimizer's score is a *model*. This rail asks the silicon: does the
cost-model-selected lane geometry actually win? It compiles BCIR's selected
realization and a naive baseline (the scalar width-1 kernel) with the host
toolchain, times both, and reports the measured speedup.

Honest finding (the measured Clang comparison, docs/research/CLANG_COMPARISON.md): for a
plain elementwise kernel the *loop form* is measured-neutral -- the kernel is
memory-bandwidth-bound and a modern `-O2`/`-O3` vectorizer realizes the selected
lane width from an idiomatic loop on its own. Hand-blocked vs idiomatic is within
measurement noise: under rigorous separate-process, alternated, median-of-N timing
neither reliably wins (the winner flips with size, opt level, and cache
alignment). The lane width is therefore the cost model's *recommendation to the
backend* -- a floor at the full hardware lane (let the compiler interleave), a
ceiling when narrower than the lane (a deliberate thermal/power throttle, honored
by a hard cap) -- not a hand-coded speedup. BCIR's *measured* wins come from
choices the compiler cannot make for you: **gather avoidance** (contiguous vs
random access -- `compare_gather` ~6.5x, `compare_reduce` ~16x, `compare_strided`
~1.4x) and budget feasibility, not loop blocking.

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
from .kbcir.cost import Theta, default_target_name
from .kbcir.weights import PERF, POLICIES
from .lower.c_kernel import (
    emit_gather_kernel_c,
    emit_kernel_c,
    emit_reduce_c,
    emit_strided_c,
    find_reduce,
    find_strided,
)
from .lower.llvm import find_elementwise

_THETAS = {"cool": Theta.cool(), "hot": Theta.hot(), "mem_bound": Theta.mem_bound()}
_NS_RE = re.compile(r"^NS (\d+)$", re.MULTILINE)
_PAIRED_RE = re.compile(r"^PAIRED (\d+\.\d+)$", re.MULTILINE)

#: Calls of one kernel that a round times back to back; the round's time for that kernel is the
#: block's best call, so a round never times a kernel's first call after the other kernel's.
_BLOCK = 3

#: The headers every emitted timing driver includes, here and in `bcir.clang_compare`: what
#: `NOW_NS_C` needs (<stdint.h>, <time.h>), and <inttypes.h> to print its readings with
#: `PRId64`, since an `int64_t` is a `long` on one host and a `long long` on another.
DRIVER_HEADERS_C = (
    "#include <inttypes.h>\n#include <limits.h>\n#include <stddef.h>\n#include <stdint.h>\n"
    "#include <stdio.h>\n#include <stdlib.h>\n#include <time.h>\n"
)

#: The one clock every emitted timing driver reads: nanoseconds since the epoch, in 64 bits
#: wherever the driver is compiled. Each driver used to carry a clock of its own, computed in
#: `long`: `(long)t.tv_sec * 1000000000L` needs 61 bits, and a `long` has 32 on LLP64 hosts
#: (64-bit Windows) and ILP32 ones, where it overflowed -- undefined behaviour, and a timing
#: that is garbage where it is not a trap. A reading is held in an `int64_t` and printed with
#: `PRId64`. It follows `DRIVER_HEADERS_C`.
NOW_NS_C = """static int64_t now_ns(void) {
  struct timespec t;
  timespec_get(&t, TIME_UTC);
  return (int64_t)t.tv_sec * INT64_C(1000000000) + (int64_t)t.tv_nsec;
}
"""


def _paired_rounds_c(call_a: str, call_b: str, label_a: str, label_b: str, reps: int) -> str:
    """The timing loop every driver here shares: at least `reps` calls of each kernel, in rounds
    of `_BLOCK` calls a side, back to back in one process over the same buffers, the order
    alternating round by round (an even count of rounds). It prints each kernel's best call over
    every round (`<label> <ns>`, nanoseconds of `NOW_NS_C`, held in `int64_t`) and the median
    over rounds of the per-round ratio of the two blocks' best calls, b over a
    (`PAIRED <ratio>`); a round in which a kernel measured no time is a failed run.

    WHY ROUNDS. These drivers timed every call of one kernel, then every call of the other, and
    the dense-parity pair ran in two processes. Ten harness runs of identical code read
    `native.dense-parity` from 0.65 to 1.60, though its two kernels compile from the same C, and
    `native.gather-avoidance` from 3.9 to 6.2; in rounds, 0.97 to 1.01 and 4.5 to 5.1. Calls a few
    milliseconds apart see the same host, the same pages and the same clock speed, and the median
    discards the round a change splits. What reaches one kernel and not the other is the kernels'
    own difference, not the timing's: `native.blocked-reduction`'s random gather follows the
    memory system's latency, which no pairing cancels."""
    rounds = max(4, -(-reps // _BLOCK))
    rounds += rounds % 2
    # Every name is prefixed and scoped to its own block: the calls are the driver's statements,
    # and a counter named like one of the driver's variables (the strided driver's `k`) would
    # time the kernel on the counter instead.
    return f"""
  {{
  int64_t pr_best_a = INT64_MAX, pr_best_b = INT64_MAX;
  double pr_ratio[{rounds}];
  for (int pr_r = 0; pr_r < {rounds}; ++pr_r) {{
    int64_t pr_a = INT64_MAX, pr_b = INT64_MAX;
    for (int pr_half = 0; pr_half < 2; ++pr_half) {{
      if ((pr_half == 0) == (pr_r % 2 == 0)) {{
        for (int pr_k = 0; pr_k < {_BLOCK}; ++pr_k) {{
          int64_t pr_t0 = now_ns(); {call_a} int64_t pr_d = now_ns() - pr_t0;
          if (pr_d > 0 && pr_d < pr_a) pr_a = pr_d;
        }}
      }} else {{
        for (int pr_k = 0; pr_k < {_BLOCK}; ++pr_k) {{
          int64_t pr_t0 = now_ns(); {call_b} int64_t pr_d = now_ns() - pr_t0;
          if (pr_d > 0 && pr_d < pr_b) pr_b = pr_d;
        }}
      }}
    }}
    if (pr_a == INT64_MAX || pr_b == INT64_MAX) {{
      printf("FAIL round %d measured no time\\n", pr_r);
      return 1;
    }}
    if (pr_a < pr_best_a) pr_best_a = pr_a;
    if (pr_b < pr_best_b) pr_best_b = pr_b;
    pr_ratio[pr_r] = (double)pr_b / (double)pr_a;
  }}
  for (int pr_i = 1; pr_i < {rounds}; ++pr_i) {{
    double pr_v = pr_ratio[pr_i];
    int pr_j = pr_i - 1;
    while (pr_j >= 0 && pr_ratio[pr_j] > pr_v) {{ pr_ratio[pr_j + 1] = pr_ratio[pr_j]; --pr_j; }}
    pr_ratio[pr_j + 1] = pr_v;
  }}
  printf("{label_a} %" PRId64 "\\n{label_b} %" PRId64 "\\nPAIRED %.6f\\n", pr_best_a, pr_best_b,
         (pr_ratio[{rounds} / 2 - 1] + pr_ratio[{rounds} / 2]) / 2.0);
  }}
"""


def _paired_milli(stdout: str) -> int:
    """The driver's `PAIRED` line in milli, or 0 when it printed none."""
    m = _PAIRED_RE.search(stdout or "")
    return round(float(m.group(1)) * 1000) if m else 0


@dataclass(frozen=True)
class Measurement:
    label: str
    width: int
    opt: str
    n: int
    reps: int
    ns_per_call: int  # best (minimum) wall-time per kernel call
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class Comparison:
    program: str
    target: str
    opt: str
    bcir: Measurement  # BCIR's cost-model-selected realization
    baseline: Measurement  # the naive scalar (width-1) baseline
    paired_milli: int = 0  # median over paired rounds of baseline / bcir, in milli (0: none)

    @property
    def speedup_milli(self) -> int:
        """baseline / bcir in milli (1000 == parity; >1000 == BCIR faster)."""
        if not (self.bcir.ok and self.baseline.ok and self.bcir.ns_per_call > 0):
            return 0
        return (self.baseline.ns_per_call * 1000) // self.bcir.ns_per_call

    @property
    def bcir_wins(self) -> bool:
        return (
            self.bcir.ok and self.baseline.ok and self.bcir.ns_per_call < self.baseline.ns_per_call
        )


def _emit_timed_c(
    module, result, fn_name: str, elem: str, n: int, reps: int, width_override, hw_width=None
) -> str:
    kernel = emit_kernel_c(
        module, result, fn_name, elem, width_override=width_override, hw_width=hw_width
    )
    claim, _ = find_elementwise(module, result)
    op = "+" if elem != "i32" else "+"
    ctype = "int32_t" if elem == "i32" else "float"
    init = (
        "A[i] = (float)(i % 1000); B[i] = (float)(i % 7);"
        if elem != "i32"
        else "A[i] = (int32_t)(i % 1000); B[i] = (int32_t)(i % 7);"
    )
    return (
        DRIVER_HEADERS_C
        + "\n"
        + kernel
        + "\n"
        + NOW_NS_C
        + f"""
int main(void) {{
  size_t n = {n}u;
  {ctype} *A = malloc(n*sizeof *A), *B = malloc(n*sizeof *B), *C = malloc(n*sizeof *C);
  if (!A || !B || !C) return 2;
  for (size_t i = 0; i < n; ++i) {{ {init} C[i] = 0; }}
  {fn_name}(A, B, C, n);                         /* warm up */
  int64_t best = INT64_MAX;
  for (int r = 0; r < {reps}; ++r) {{
    int64_t t0 = now_ns();
    {fn_name}(A, B, C, n);
    int64_t dt = now_ns() - t0;
    if (dt > 0 && dt < best) best = dt;
  }}
  volatile {ctype} sink = C[n - 1]; (void)sink;  /* keep the kernel live */
  printf("NS %" PRId64 "\\n", best);
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


def measure(
    module,
    result,
    *,
    label: str,
    elem: str = "f32",
    opt: str = "-O1",
    n: int = 1 << 20,
    reps: int = 200,
    width_override=None,
    hw_width=None,
) -> Measurement:
    """Compile + time one realization (the selected width, or `width_override`).
    `hw_width` (the target's widest lane) drives the width-aware lowering so the
    timed kernel is exactly the deployed one (go-fast at the full lane)."""
    _, cand = find_elementwise(module, result)
    w = int(width_override) if width_override else int(cand.width)
    cc = _which("clang", "cc", "gcc")
    if cc is None:
        return Measurement(label, w, opt, n, reps, 0, False, "no C compiler")
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "bench.c")
        exe = os.path.join(d, "bench")
        with open(src, "w", newline="\n") as f:
            f.write(
                _emit_timed_c(
                    module, result, "bcir_kernel", elem, n, reps, width_override, hw_width=hw_width
                )
            )
        build = None
        for std in ("-std=c23", "-std=c2x"):
            build = subprocess.run([cc, std, opt, src, "-o", exe], capture_output=True, text=True)
            if build.returncode == 0:
                break
        if build is None or build.returncode != 0:
            return Measurement(
                label, w, opt, n, reps, 0, False, "build failed: " + (build.stderr if build else "")
            )
        run = subprocess.run([exe], capture_output=True, text=True)
        m = _NS_RE.search(run.stdout or "")
        if run.returncode != 0 or not m:
            return Measurement(
                label, w, opt, n, reps, 0, False, "run failed: " + run.stdout + run.stderr
            )
        return Measurement(label, w, opt, n, reps, int(m.group(1)), True)


_SELECTED_RE = re.compile(r"^SELECTED (\d+)$", re.MULTILINE)
_SCALAR_RE = re.compile(r"^SCALAR (\d+)$", re.MULTILINE)


def _emit_parity_bench_c(module, result, elem: str, n: int, reps: int, hw_width=None) -> str:
    """BCIR's selected realization and the scalar (width-1) baseline in ONE program over the same
    buffers, timed in paired rounds (`_paired_rounds_c`). Each used to be its own program, timed
    in its own process, so their ratio carried the two processes' pages and the time between
    them as well as the kernels."""
    selected = emit_kernel_c(module, result, "bcir_selected", elem, hw_width=hw_width)
    scalar = emit_kernel_c(module, result, "bcir_scalar", elem, width_override=1, hw_width=hw_width)
    ctype = "int32_t" if elem == "i32" else "float"
    init = (
        "A[i] = (float)(i % 1000); B[i] = (float)(i % 7);"
        if elem != "i32"
        else "A[i] = (int32_t)(i % 1000); B[i] = (int32_t)(i % 7);"
    )
    timing = _paired_rounds_c(
        "bcir_selected(A, B, C, n);", "bcir_scalar(A, B, C, n);", "SELECTED", "SCALAR", reps
    )
    opcode = find_elementwise(module, result)[0].opcode.name
    op = "+" if opcode == "ADD" else ("-" if opcode == "SUB" else "*")
    # both kernels write C; checked before the timed calls and after them (`_emit_gather_bench_c`)
    check = f"""  for (size_t i = 0; i < n; ++i) {{
    {ctype} want = A[i] {op} B[i];
    if (C[i] != want) {{ printf("FAIL i=%zu\\n", i); return 1; }}
  }}"""
    return (
        DRIVER_HEADERS_C
        + "\n"
        + selected
        + "\n"
        + scalar
        + "\n"
        + NOW_NS_C
        + f"""
int main(void) {{
  size_t n = {n}u;
  {ctype} *A = malloc(n*sizeof *A), *B = malloc(n*sizeof *B), *C = malloc(n*sizeof *C);
  if (!A || !B || !C) return 2;
  for (size_t i = 0; i < n; ++i) {{ {init} C[i] = 0; }}
  bcir_selected(A, B, C, n);                      /* warm up */
  bcir_scalar(A, B, C, n);
{check}
{timing}
{check}
  volatile {ctype} sink = C[n - 1]; (void)sink;  /* keep the kernels live */
  free(A); free(B); free(C);
  return 0;
}}
"""
    )


def compare(
    program,
    *,
    target: str = default_target_name(),
    theta: str = "cool",
    policy: str = "latency",
    elem: str = "f32",
    opt: str = "-O1",
    n: int = 1 << 20,
    reps: int = 200,
) -> Comparison:
    """Time BCIR's selected realization against the naive scalar (width-1) baseline
    at one optimization level -- the measured evidence for the lane-geometry choice.
    Both kernels run in one program, over the same buffers, in paired rounds; `measure`
    still times one realization alone."""
    module = PROGRAMS[program]() if isinstance(program, str) else program
    name = program if isinstance(program, str) else getattr(program, "name", "module")
    h = TARGETS[target]
    th = _THETAS.get(theta, Theta.cool())
    pol = POLICIES.get(policy, PERF)
    result = optimize(module, h, th, pol)
    _, cand = find_elementwise(module, result)

    def failed(detail: str) -> Comparison:
        bcir = Measurement("bcir-selected", int(cand.width), opt, n, reps, 0, False, detail)
        base = Measurement("scalar-baseline", 1, opt, n, reps, 0, False, detail)
        return Comparison(program=name, target=h.name, opt=opt, bcir=bcir, baseline=base)

    cc = _which("clang", "cc", "gcc")
    if cc is None:
        return failed("no C compiler")
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "pbench.c")
        exe = os.path.join(d, "pbench")
        with open(src, "w", newline="\n") as f:
            f.write(_emit_parity_bench_c(module, result, elem, n, reps, hw_width=h.vector_width))
        build = None
        for std in ("-std=c23", "-std=c2x"):
            build = subprocess.run([cc, std, opt, src, "-o", exe], capture_output=True, text=True)
            if build.returncode == 0:
                break
        if build is None or build.returncode != 0:
            return failed("build failed: " + (build.stderr if build else ""))
        run = subprocess.run([exe], capture_output=True, text=True)
        sm, bm = _SELECTED_RE.search(run.stdout or ""), _SCALAR_RE.search(run.stdout or "")
        if run.returncode != 0 or "FAIL" in run.stdout or not (sm and bm):
            return failed("run failed: " + run.stdout + run.stderr)
        bcir = Measurement("bcir-selected", int(cand.width), opt, n, reps, int(sm.group(1)), True)
        base = Measurement("scalar-baseline", 1, opt, n, reps, int(bm.group(1)), True)
        return Comparison(
            program=name,
            target=h.name,
            opt=opt,
            bcir=bcir,
            baseline=base,
            paired_milli=_paired_milli(run.stdout),
        )


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
    direct: Measurement  # BCIR's cost-model choice (contiguous, vectorizable)
    gather: Measurement  # the avoided realization (indexed loads -> gather_penalty)
    paired_milli: int = 0  # median over paired rounds of gather / direct, in milli (0: none)

    @property
    def speedup_milli(self) -> int:
        """gather / direct in milli (>1000 == the avoided gather is that much dearer)."""
        if not (self.direct.ok and self.gather.ok and self.direct.ns_per_call > 0):
            return 0
        return (self.gather.ns_per_call * 1000) // self.direct.ns_per_call

    @property
    def avoidance_wins(self) -> bool:
        return (
            self.direct.ok and self.gather.ok and self.direct.ns_per_call < self.gather.ns_per_call
        )


_DIRECT_RE = re.compile(r"^DIRECT (\d+)$", re.MULTILINE)
_GATHER_RE = re.compile(r"^GATHER (\d+)$", re.MULTILINE)


def _emit_gather_bench_c(
    module, result, elem: str, n: int, reps: int, shuffle: bool, hw_width=None
) -> str:
    direct = emit_kernel_c(module, result, "bcir_direct", elem, hw_width=hw_width)  # BCIR's choice
    gather = emit_gather_kernel_c(module, result, "bcir_gather", elem)  # the avoided form
    op = (
        "+"
        if find_elementwise(module, result)[0].opcode.name == "ADD"
        else ("-" if find_elementwise(module, result)[0].opcode.name == "SUB" else "*")
    )
    ctype = "int32_t" if elem == "i32" else "float"
    init = (
        "A[i] = (float)(i % 1000); B[i] = (float)(i % 7);"
        if elem != "i32"
        else "A[i] = (int32_t)(i % 1000); B[i] = (int32_t)(i % 7);"
    )
    shuf = (
        (
            "for (size_t i = n - 1; i > 0; --i) { size_t j = (size_t)(lcg() % (i + 1));"
            " long t = idx[i]; idx[i] = idx[j]; idx[j] = t; }"
        )
        if shuffle
        else ""
    )
    timing = _paired_rounds_c(
        "bcir_direct(A, B, Cd, n);", "bcir_gather(A, B, Cg, idx, n);", "DIRECT", "GATHER", reps
    )
    # checked before the timed calls and after them: a timed call that computed something else
    # (the wrong buffer, the wrong stride) fails the run instead of timing the wrong work
    check = f"""  for (size_t i = 0; i < n; ++i) {{
    {ctype} wd = A[i] {op} B[i];
    {ctype} wg = A[idx[i]] {op} B[i];
    if (Cd[i] != wd || Cg[i] != wg) {{ printf("FAIL i=%zu\\n", i); return 1; }}
  }}"""
    return (
        DRIVER_HEADERS_C
        + "\n"
        + direct
        + "\n"
        + gather
        + "\n"
        + NOW_NS_C
        + f"""static uint64_t g_st = 0xBC12u;
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
{check}
{timing}
{check}
  volatile {ctype} s = Cd[n - 1] + Cg[n - 1]; (void)s;
  free(A); free(B); free(Cd); free(Cg); free(idx);
  return 0;
}}
"""
    )


def compare_gather(
    program,
    *,
    target: str = default_target_name(),
    theta: str = "cool",
    policy: str = "latency",
    elem: str = "f32",
    opt: str = "-O2",
    n: int = 1 << 20,
    reps: int = 200,
    shuffle: bool = True,
) -> GatherComparison:
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
        with open(src, "w", newline="\n") as f:
            f.write(
                _emit_gather_bench_c(
                    module, result, elem, n, reps, shuffle, hw_width=h.vector_width
                )
            )
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
        return GatherComparison(
            name, opt, n, reps, shuffle, direct, gather, _paired_milli(run.stdout)
        )


# --- gather/blocked reduction: a genuine gather program, lowered two ways --------


@dataclass(frozen=True)
class ReduceComparison:
    program: str
    opt: str
    n: int
    reps: int
    selected: str  # the realization BCIR's cost model selected ("blocked")
    correct: bool  # blocked and gather computed the identical sum
    blocked: Measurement
    gather: Measurement
    paired_milli: int = 0  # median over paired rounds of gather / blocked, in milli (0: none)

    @property
    def speedup_milli(self) -> int:
        if not (self.blocked.ok and self.gather.ok and self.blocked.ns_per_call > 0):
            return 0
        return (self.gather.ns_per_call * 1000) // self.blocked.ns_per_call

    @property
    def avoidance_wins(self) -> bool:
        return (
            self.correct
            and self.blocked.ok
            and self.gather.ok
            and self.blocked.ns_per_call < self.gather.ns_per_call
        )


_BLK_RE = re.compile(r"^BLOCKED (\d+)$", re.MULTILINE)
_GRD_RE = re.compile(r"^GATHER (\d+)$", re.MULTILINE)


def _emit_reduce_bench_c(module, result, n: int, reps: int) -> str:
    blocked = emit_reduce_c(module, result, "bcir_blocked", "i32", gather=False)
    gather = emit_reduce_c(module, result, "bcir_gather_reduce", "i32", gather=True)
    timing = _paired_rounds_c(
        "{ volatile int32_t s = bcir_blocked(T, n); (void)s; }",
        "{ volatile int32_t s = bcir_gather_reduce(T, idx, n); (void)s; }",
        "BLOCKED",
        "GATHER",
        reps,
    )
    return (
        DRIVER_HEADERS_C
        + "\n"
        + blocked
        + "\n"
        + gather
        + "\n"
        + NOW_NS_C
        + f"""static uint64_t g_st = 0xBC12u;
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
{timing}
  if (bcir_blocked(T, n) != sb || bcir_gather_reduce(T, idx, n) != sg) {{
    printf("FAIL the sums moved under timing\\n");
    return 1;
  }}
  free(T); free(idx);
  return 0;
}}
"""
    )


def compare_reduce(
    program="gather_reduce",
    *,
    target: str = default_target_name(),
    theta: str = "cool",
    policy: str = "latency",
    opt: str = "-O2",
    n: int = 1 << 20,
    reps: int = 40,
) -> ReduceComparison:
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
        with open(src, "w", newline="\n") as f:
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
        return ReduceComparison(
            name, opt, n, reps, cand.name, True, blocked, gather, _paired_milli(run.stdout)
        )


# --- strided gather avoidance: a non-reduction gather program, lowered two ways --


def _emit_strided_bench_c(module, result, n: int, reps: int, k: int) -> str:
    direct = emit_strided_c(module, result, "bcir_strided", "f32", gather=False)
    gather = emit_strided_c(module, result, "bcir_strided_gather", "f32", gather=True)
    timing = _paired_rounds_c(
        "bcir_strided(X, Yd, n, k);",
        "bcir_strided_gather(X, idx, Yg, n);",
        "BLOCKED",
        "GATHER",
        reps,
    )
    return (
        DRIVER_HEADERS_C
        + "\n"
        + direct
        + "\n"
        + gather
        + "\n"
        + NOW_NS_C
        + f"""
int main(void) {{
  size_t n = {n}u, k = {k}u;
  float *X = malloc(n*k*sizeof *X), *Yd = malloc(n*sizeof *Yd), *Yg = malloc(n*sizeof *Yg);
  long *idx = malloc(n*sizeof *idx);
  if (!X || !Yd || !Yg || !idx) return 2;
  for (size_t i = 0; i < n*k; ++i) X[i] = (float)(i % 1000);
  for (size_t i = 0; i < n; ++i) {{ idx[i] = (long)(i*k); Yd[i] = 0; Yg[i] = 0; }}
  bcir_strided(X, Yd, n, k);
  bcir_strided_gather(X, idx, Yg, n);
  for (size_t i = 0; i < n; ++i) if (Yd[i] != Yg[i]) {{ printf("FAIL i=%zu\\n", i); return 1; }}
{timing}
  for (size_t i = 0; i < n; ++i) if (Yd[i] != Yg[i]) {{ printf("FAIL i=%zu\\n", i); return 1; }}
  volatile float s = Yd[n - 1] + Yg[n - 1]; (void)s;
  free(X); free(Yd); free(Yg); free(idx);
  return 0;
}}
"""
    )


def compare_strided(
    program="saxpy_strided",
    *,
    target: str = default_target_name(),
    theta: str = "cool",
    policy: str = "latency",
    opt: str = "-O2",
    n: int = 1 << 22,
    reps: int = 40,
) -> ReduceComparison:
    """Lower a strided program two ways and measure: BCIR's selected direct strided
    realization vs the gather form of the same access (`X[i*k]` vs `X[idx[i]]`,
    idx[i]=i*k). The harness verifies identical Y, then times them -- a
    non-reduction gather avoidance (BCIR knows the stride, so it avoids the slower
    gather instruction)."""
    module = PROGRAMS[program]() if isinstance(program, str) else program
    name = program if isinstance(program, str) else getattr(program, "name", "module")
    h = TARGETS[target]
    th = _THETAS.get(theta, Theta.cool())
    pol = POLICIES.get(policy, PERF)
    result = optimize(module, h, th, pol)
    claim, cand = find_strided(module, result)
    k = max(1, claim.stride_k)
    bad = Measurement("", 1, opt, n, reps, 0, False, "build/run failed")
    cc = _which("clang", "cc", "gcc")
    if cc is None:
        return ReduceComparison(name, opt, n, reps, cand.name, False, bad, bad)
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "tbench.c")
        exe = os.path.join(d, "tbench")
        with open(src, "w", newline="\n") as f:
            f.write(_emit_strided_bench_c(module, result, n, reps, k))
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
        direct = Measurement("strided", 1, opt, n, reps, int(bm.group(1)), True)
        gather = Measurement("gather", 1, opt, n, reps, int(gm.group(1)), True)
        return ReduceComparison(
            name, opt, n, reps, cand.name, True, direct, gather, _paired_milli(run.stdout)
        )
