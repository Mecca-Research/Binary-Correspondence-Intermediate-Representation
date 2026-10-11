"""The measured-evidence rail: time BCIR's selected realization vs the scalar
baseline. Timings are environment-dependent, so these tests assert the rail
*runs* and yields valid measurements -- not a speedup threshold (which would be
flaky). The honest finding (elementwise kernels are bandwidth-bound, so width is
measured-neutral -- confirming the memory-dominated cost model) is documented in
the strategy roadmap, not pinned here.
"""

from bcir.bench import (
    Comparison,
    GatherComparison,
    Measurement,
    ReduceComparison,
    bench_available,
    compare,
    compare_gather,
    compare_reduce,
    compare_strided,
    measure,
)
from bcir.examples import gather_reduce, saxpy_strided, vector_add
from bcir.kbcir import TARGETS, optimize
import platform
from bcir.kbcir.cost import Theta
from bcir.kbcir.realize import candidates_for
from bcir.model import Lane


def test_rail_produces_valid_measurements():
    if not bench_available():
        return  # skip cleanly without a C compiler
    c = compare("vector_add", target="x86_avx512", theta="cool", opt="-O2", n=1 << 16, reps=50)
    assert isinstance(c, Comparison)
    assert c.bcir.ok and c.baseline.ok
    assert c.bcir.ns_per_call > 0 and c.baseline.ns_per_call > 0
    assert c.bcir.width == 16 and c.baseline.width == 1  # selected vs scalar
    assert c.speedup_milli > 0  # a finite measured ratio


def test_measure_reports_the_selected_width():
    if not bench_available():
        return
    m = vector_add(1024)
    r = optimize(m, TARGETS["x86_avx512"], Theta.cool())
    meas = measure(m, r, label="sel", opt="-O2", n=1 << 16, reps=20)
    assert isinstance(meas, Measurement) and meas.ok and meas.width == 16


def test_both_realizations_are_correct_under_timing():
    # the timed harness self-initializes and runs; a build/run failure surfaces as
    # ok=False with detail (the rail never silently reports a bogus speedup).
    if not bench_available():
        return
    c = compare("vector_add", opt="-O1", n=1 << 14, reps=20)
    assert c.bcir.ok, c.bcir.detail
    assert c.baseline.ok, c.baseline.detail


# --- gather avoidance: the cost-model decision + the measured penalty -------------


def test_cost_model_avoids_gather_on_a_strided_claim():
    # On a strided claim both a strided (U) and a gather (GGG) candidate exist; the
    # gather candidate must cost strictly more (gather_penalty), so it is never the
    # cheapest -- BCIR avoids GGG by construction.
    m = saxpy_strided(1024)
    claim = m.phases[0].claims[0]
    h = TARGETS["x86_avx512"]
    cands = candidates_for(claim, h, m.resource(claim.rd[0]))
    by_lane = {c.lane: c for c in cands}
    assert Lane.U in by_lane and Lane.GGG in by_lane  # both realizations offered
    from bcir.kbcir.weights import PERF, weights

    w = weights(h, Theta.cool(), 0, PERF)
    assert by_lane[Lane.GGG].base.dot(w) > by_lane[Lane.U].base.dot(w)  # gather is dearer


def test_gather_avoidance_is_measured_and_correct():
    if not bench_available():
        return
    # a working set that exceeds cache (4 MB) so the random gather pays real
    # misses -- the gather_penalty realized on silicon.
    c = compare_gather("vector_add", opt="-O2", n=1 << 20, reps=40, shuffle=True)
    assert isinstance(c, GatherComparison)
    assert c.direct.ok and c.gather.ok  # both built + self-checked
    assert c.direct.ns_per_call > 0 and c.gather.ns_per_call > 0
    # avoiding GGG is measurably faster. We observe ~6.5x; assert a conservative
    # 1.5x floor to stay CI-robust across cache sizes.
    assert c.avoidance_wins and c.speedup_milli >= 1500


def test_identity_gather_isolates_indexed_load_overhead():
    if not bench_available():
        return
    c = compare_gather("vector_add", opt="-O2", n=1 << 16, reps=30, shuffle=False)
    assert c.direct.ok and c.gather.ok and c.speedup_milli > 0


# --- gather/blocked reduction: a real gather program lowered two ways -------------


def test_cost_model_selects_blocked_for_a_reducible_gather():
    # `reduce.gather` offers blocked (sequential) + gather (random); BCIR picks
    # blocked -- the gather avoided, by construction.
    m = gather_reduce(1024)
    claim = m.phases[0].claims[0]
    cands = candidates_for(claim, TARGETS["x86_avx512"], m.resource(claim.rd[0]))
    assert {c.name for c in cands} == {"blocked", "gather"}
    r = optimize(m, TARGETS["x86_avx512"], Theta.cool())
    assert r.steps[0].candidate.name == "blocked"


def test_reduce_gather_avoidance_is_correct_and_measured():
    if not bench_available():
        return
    c = compare_reduce("gather_reduce", opt="-O2", n=1 << 20, reps=30)
    assert isinstance(c, ReduceComparison)
    assert c.selected == "blocked" and c.correct  # identical integer sum
    # observed ~17x; assert a conservative 2x floor (random-load reduction is
    # dominated by gather latency, so the win is large and robust).
    assert c.avoidance_wins and c.speedup_milli >= 2000


def test_strided_gather_avoidance_is_measured_non_reduction():
    # saxpy_strided: the cost model picks the direct strided realization over the
    # gather; a non-reduction gather avoidance (BCIR knows the stride).
    m = saxpy_strided(1024)
    r = optimize(m, TARGETS["x86_avx512"], Theta.cool())
    assert r.steps[0].candidate.name == "strided"  # gather avoided by selection
    if not bench_available():
        return
    c = compare_strided("saxpy_strided", opt="-O2", n=1 << 22, reps=30)
    assert c.selected == "strided" and c.correct  # identical Y (host-independent)
    # The measured win (~1.4x on x86, the gather-instruction overhead) is a *marginal* floor
    # that depends on the host's gather behaviour + calibration; per this file's own contract
    # (timings are environment-dependent), assert it only on the arch it was characterized on
    # (x86). On other hosts (an uncalibrated, shared ARM runner) the gather-avoidance
    # measurement is best-effort -- honest where realized, never faked.
    if platform.machine() in ("x86_64", "amd64"):
        assert c.avoidance_wins and c.speedup_milli >= 1100


# --- the paired rounds every driver times with, on a clock whose answers are known ----------

#: The compiler the paired-rounds witness builds with (`bcir.bench`'s own choice), or None: the
#: capability `tools/testing/faults/native-paired.json` requires.
_CC = __import__("bcir.bench", fromlist=["_which"])._which("clang", "cc", "gcc")


def _paired_program(tmp: str) -> str:
    """`bcir.bench._paired_rounds_c` around two fake kernels on a fake clock, built: a call costs
    its cost times (1000 + 5% per call made before it) / 1000, five times over from call `step`
    on. argv: step, drift (per mille a call), cost_b; kernel a costs 1000. The program declares
    the names a driver might (`k`, `r`, `a`, ...) and every call checks it still sees them."""
    import os
    import subprocess

    from bcir.bench import DRIVER_HEADERS_C, _paired_rounds_c

    names = "size_t k = 4; int r = 7, half = 1, i = 2, j = 3; long a = 5, b = 6, d = 8, t0 = 9;"
    same = (
        "k == 4 && r == 7 && half == 1 && i == 2 && j == 3 && a == 5 && b == 6 && d == 8 && t0 == 9"
    )
    timing = _paired_rounds_c(
        f"kernel(1000); bad |= !({same});", f"kernel(cost_b); bad |= !({same});", "A", "B", 30
    )
    src = f"""{DRIVER_HEADERS_C}
static int64_t fake_ns;
static long calls, step, drift, cost_b;
static int64_t now_ns(void) {{ return fake_ns; }}
static void kernel(long cost) {{
  fake_ns += cost * (1000 + calls * drift) / 1000 * (calls >= step ? 5 : 1);
  ++calls;
}}
int main(int argc, char **argv) {{
  if (argc != 4) return 2;
  step = atol(argv[1]);
  drift = atol(argv[2]);
  cost_b = atol(argv[3]);
  int bad = 0;
  {names}
{timing}
  if (bad) printf("BAD a call saw the timing loop's names, not the driver's\\n");
  printf("CALLS %ld\\n", calls);
  return 0;
}}
"""
    path = os.path.join(tmp, "paired.c")
    with open(path, "w", newline="\n") as f:
        f.write(src)
    exe = os.path.join(tmp, "paired")
    subprocess.run([_CC, "-std=c2x", "-O1", path, "-o", exe], check=True, capture_output=True)
    return exe


def test_the_paired_rounds_cancel_a_load_and_keep_the_drivers_names():
    """The timing loop every native driver shares, held to its properties on fake kernels whose
    costs are known: kernel a costs 1000 and kernel b 900, a true ratio of 0.9.

    A load that steps from 1x to 5x at a chosen call reads 4.5 through the windowed estimator it
    replaces -- every a call, then every b call, the best of each -- when it lands between the
    windows; through rounds the ratio reads exactly 0.9 wherever it lands, and a b that truly
    costs 2000 reads 2.0 under every load. A host that slows down 5% a call reads 0.958 in one
    fixed order and within 1% of 0.9 alternated. And every call still sees the driver's own `k`,
    `r`, `a`, `b`: the first version named its counter `k`, and the strided driver timed its
    kernel at strides 0, 1 and 2 instead of 4."""
    import subprocess
    import tempfile

    if _CC is None:
        return  # no C compiler: the witness is a skip, never a pass

    def read(exe, step, drift=0, cost_b=900):
        out = subprocess.run(
            [exe, str(step), str(drift), str(cost_b)], capture_output=True, text=True, timeout=60
        )
        assert out.returncode == 0, out
        assert "BAD" not in out.stdout, out.stdout
        fields = dict(line.split() for line in out.stdout.splitlines())
        return float(fields["PAIRED"]), int(fields["A"]), int(fields["B"]), int(fields["CALLS"])

    with tempfile.TemporaryDirectory() as tmp:
        exe = _paired_program(tmp)
        paired, a, b, calls = read(exe, 10**9)
        assert (paired, a, b, calls) == (0.9, 1000, 900, 60), (paired, a, b, calls)
        windowed = (900 * 5) / 1000  # a's 30 calls unloaded, b's 30 after the step
        assert windowed == 4.5
        for step in range(calls + 1):
            assert read(exe, step)[0] == 0.9, step
            assert read(exe, step, cost_b=2000)[0] == 2.0, step
        drifted = read(exe, 10**9, drift=50)[0]
        assert abs(drifted - 0.9) < 0.009, drifted


# --- the one clock every driver reads, held to 64 bits where a `long` has 32 ------------------


class _Toolchain:
    """Truthy where `bcir.toolchain` resolves `tools` here (and `node` is on PATH, when asked),
    resolved when first read: what `check_tests.py --require` asks of this module when
    `tools/testing/faults/now-ns64.json` runs it. The tests that need them return without."""

    def __init__(self, *tools: str, node: bool = False) -> None:
        self._tools, self._node, self._paths = tools, node, None

    def paths(self) -> dict[str, str] | None:
        if self._paths is None:
            from shutil import which

            from bcir.toolchain import resolve_llvm_tools

            found = resolve_llvm_tools(*self._tools, pipeline="the drivers' 32-bit long witness")
            paths = dict(found.paths) if found.ok else {}
            node = which("node") if self._node else None
            if self._node:
                paths = {**paths, "node": node} if paths and node else {}
            self._paths = paths
        return self._paths or None

    def __bool__(self) -> bool:
        return self.paths() is not None


#: clang (any target's front end serves): every driver type-checked where a `long` has 32 bits.
_CLANG = _Toolchain("clang")
#: clang, wasm-ld and node: the clock run on wasm32, where a `long` has 32 bits.
_WASM = _Toolchain("clang", "wasm-ld", node=True)

#: The targets where a `long` has 32 bits: 64-bit Windows (LLP64) and wasm32 (ILP32).
_LONG32_TARGETS = ("x86_64-pc-windows-msvc", "wasm32")

#: The C library a driver calls, declared for a front end that has none for the target at hand
#: (`-ffreestanding -nostdinc`): the shapes C11 gives them, `time_t` 64 bits as in the UCRT and
#: wasi-libc, and `PRId64` the compiler's own spelling for the target. <stdint.h>, <stddef.h>
#: and <limits.h> are the compiler's.
_STAND_INS = {
    "inttypes.h": "#include <stdint.h>\n#define PRId64 __INT64_FMTd__\n",
    "stdio.h": "int printf(const char *restrict format, ...) __attribute__((format(printf, 1, 2)));\n",
    "stdlib.h": (
        "#include <stddef.h>\nvoid *malloc(size_t size);\nvoid free(void *ptr);\n"
        "int atoi(const char *s);\n"
    ),
    "time.h": (
        "struct timespec { long long tv_sec; long tv_nsec; };\n#define TIME_UTC 1\n"
        "int timespec_get(struct timespec *ts, int base);\n"
    ),
}

#: The clock's <time.h> on wasm32: a `timespec_get` that answers what the probe was handed, its
#: `time_t` 64 bits as in wasi-libc and the UCRT, so only the clock's own arithmetic can lose
#: the seconds.
_CLOCK_PROBE_C = """#include <stdint.h>
struct timespec { long long tv_sec; long tv_nsec; };
#define TIME_UTC 1
static long long probe_sec;
static long probe_nsec;
static int timespec_get(struct timespec *t, int base) {
  t->tv_sec = probe_sec;
  t->tv_nsec = probe_nsec;
  return base;
}
"""

_CLOCK_PROBE_JS = r"""
const fs = require('fs');
const [path, ...pairs] = process.argv.slice(2);
WebAssembly.instantiate(fs.readFileSync(path), {}).then(({ instance }) => {
  for (const pair of pairs) {
    const [sec, nsec] = pair.split(':');
    console.log(`NS ${instance.exports.probe(BigInt(sec), Number(nsec))}`);
  }
}).catch((e) => { console.log('ERROR ' + e); process.exit(2); });
"""


def _driver_sources() -> dict[str, str]:
    """Every timing driver the oracle emits, each built small: `bcir.bench`'s `_emit_*_c` (the
    timed driver for both element types) and both drivers of each `bcir.clang_compare`
    `_*_srcs` pair. An emitter this does not build fails every test that reads it, so a new
    driver cannot skip the clock's law."""
    import re

    from bcir import bench, clang_compare
    from bcir.lower.c_kernel import find_strided

    target = TARGETS["x86_avx512"]

    def plan(module):
        return module, optimize(module, target, Theta.cool())

    m, r = plan(vector_add(1024))
    mg, rg = plan(gather_reduce(1024))
    ms, rs = plan(saxpy_strided(1024))
    k = max(1, find_strided(ms, rs)[0].stride_k)
    w = target.vector_width
    built = {
        "_emit_timed_c": [
            bench._emit_timed_c(m, r, "bcir_kernel", elem, 1024, 4, None, hw_width=w)
            for elem in ("f32", "i32")
        ],
        "_emit_parity_bench_c": [bench._emit_parity_bench_c(m, r, "f32", 1024, 4, hw_width=w)],
        "_emit_gather_bench_c": [
            bench._emit_gather_bench_c(m, r, "f32", 1024, 4, True, hw_width=w)
        ],
        "_emit_reduce_bench_c": [bench._emit_reduce_bench_c(mg, rg, 1024, 4)],
        "_emit_strided_bench_c": [bench._emit_strided_bench_c(ms, rs, 1024, 4, k)],
        "_elementwise_srcs": list(clang_compare._elementwise_srcs(1024)),
        "_gather_srcs": list(clang_compare._gather_srcs(1024)),
        "_reduce_srcs": list(clang_compare._reduce_srcs(1024)),
        "_strided_srcs": list(clang_compare._strided_srcs(1024)),
    }
    emitters = {name for name in vars(bench) if re.fullmatch(r"_emit_\w+_c", name)}
    emitters |= {name for name in vars(clang_compare) if re.fullmatch(r"_\w+_srcs", name)}
    assert emitters == set(built), sorted(emitters ^ set(built))  # an emitter not built here
    return {f"{name}[{i}]": src for name, srcs in built.items() for i, src in enumerate(srcs)}


def test_every_driver_reads_the_one_64_bit_clock():
    """Every timing driver the oracle emits -- `bcir.bench`'s and `bcir.clang_compare`'s -- reads
    `bcir.bench.NOW_NS_C`: it includes `DRIVER_HEADERS_C`, then defines `now_ns` once, as that
    clock. The clock used to be defined six times, computed in `long` each time; six copies of
    one clock are six places for its defect to come back."""
    from bcir.bench import DRIVER_HEADERS_C, NOW_NS_C

    sources = _driver_sources()
    assert len(sources) == 14, sorted(sources)
    for name, src in sources.items():
        assert src.count(NOW_NS_C) == 1, name
        assert src.count("now_ns(void)") == 1, name  # no clock of its own beside it
        assert src.count(DRIVER_HEADERS_C) == 1, name
        assert src.index(DRIVER_HEADERS_C) < src.index(NOW_NS_C), name


def test_every_driver_compiles_where_long_has_32_bits():
    """Every driver, type-checked for the targets where a `long` has 32 bits (64-bit Windows and
    wasm32), with a clock reading narrowed into 32 bits (`-Wshorten-64-to-32`) or printed with
    a `long`'s format (`-Wformat`) an error. Where a `long` has 64 bits, as on every host CI
    builds the drivers on, both compile silently. Needs clang (the thorough tier)."""
    import os
    import subprocess
    import tempfile

    if not _CLANG:
        return  # no clang: the witness is a skip, never a pass
    clang = _CLANG.paths()["clang"]
    resource = subprocess.run(
        [clang, "-print-resource-dir"], capture_output=True, text=True, timeout=60, check=True
    ).stdout.strip()
    with tempfile.TemporaryDirectory() as tmp:
        for name, text in _STAND_INS.items():
            with open(os.path.join(tmp, name), "w", newline="\n") as f:
                f.write(text)
        path = os.path.join(tmp, "driver.c")
        for name, src in _driver_sources().items():
            with open(path, "w", newline="\n") as f:
                f.write(src)
            for target in _LONG32_TARGETS:
                run = subprocess.run(
                    [
                        clang,
                        f"--target={target}",
                        "-fsyntax-only",
                        "-ffreestanding",
                        "-nostdinc",
                        "-isystem",
                        tmp,
                        "-isystem",
                        os.path.join(resource, "include"),
                        "-std=c2x",
                        "-Werror=shorten-64-to-32",
                        "-Werror=format",
                        "-Werror=implicit-function-declaration",
                        path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                assert run.returncode == 0, (name, target, run.stderr[-3000:])


def test_the_clock_reads_the_epoch_in_64_bits_where_long_has_32():
    """`NOW_NS_C` run on wasm32, where a `long` has 32 bits as on 64-bit Windows, against a clock
    that answers what it is handed: 2026-10-11, the first second past a 32-bit `time_t`, the
    last instant an `int64_t` of nanoseconds holds, and a nanosecond field alone. Each reads
    back exactly, seconds * 10^9 + nanoseconds. The drivers' former clock read 2026-10-11 as
    -696082667: `(long)t.tv_sec * 1000000000L` wrapped in 32 bits. Built with -fwrapv, so a
    computation in 32 bits answers wrongly instead of however undefined behaviour resolves.
    Needs clang, wasm-ld and node (the thorough tier)."""
    import os
    import subprocess
    import tempfile

    from bcir.bench import NOW_NS_C

    if not _WASM:
        return  # no wasm toolchain or node: the witness is a skip, never a pass
    tools = _WASM.paths()
    cases = [(1791737781, 123456789), (2**31, 0), (9223372036, 854775807), (0, 999999999)]
    probe = (
        "int64_t probe(long long sec, long nsec) {\n"
        "  probe_sec = sec;\n  probe_nsec = nsec;\n  return now_ns();\n}\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        src, wasm, js = (os.path.join(tmp, name) for name in ("clock.c", "clock.wasm", "clock.js"))
        with open(src, "w", newline="\n") as f:
            f.write(_CLOCK_PROBE_C + NOW_NS_C + probe)
        with open(js, "w", newline="\n") as f:
            f.write(_CLOCK_PROBE_JS)
        build = subprocess.run(
            [
                tools["clang"],
                "--target=wasm32",
                f"-fuse-ld={tools['wasm-ld']}",
                "-ffreestanding",
                "-nostdlib",
                "-O2",
                "-fwrapv",
                "-Wl,--no-entry",
                "-Wl,--export=probe",
                src,
                "-o",
                wasm,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert build.returncode == 0, build.stdout + build.stderr
        run = subprocess.run(
            [tools["node"], js, wasm, *(f"{sec}:{nsec}" for sec, nsec in cases)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        assert run.stdout.splitlines() == [f"NS {sec * 10**9 + nsec}" for sec, nsec in cases], (
            run.stdout
        )
