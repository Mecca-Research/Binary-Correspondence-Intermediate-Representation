"""Physics-anchored cost calibration: measure, quantize, freeze, generation-tag.

The L1 learning law (LangRef Sec. 13): learning and measurement may only enter
the planner as **frozen, generation-tagged, integer tables** -- never as
plan-time inference. This module is the measurement half of that law:

  1. **Measure** (`run_microbench`): host microbenchmarks time four access
     regimes over a buffer that defeats the cache -- streaming, strided,
     random (gather), and pure compute. Index orders are seeded/deterministic;
     only the timing varies. Medians-of-repeats reject noise.
  2. **Quantize** (`calibrate_from_raw`, a pure function): raw timings become
     Q8 ratios against the streaming baseline (stream == 256 by definition),
     then the cost-model constants the seeds guessed:
         gather_penalty = random_q8 / 256      (whole-multiple per-access penalty)
         base_overhead  = 4 * strided_q8 / 256 (unit-stride access-op cost scale)
         mem_unit       = 1                    (the bandwidth unit, definitional)
     A ratio-1 measurement reproduces the seeded constants exactly (the
     degenerate case) -- the checked-in reference table is exactly that.
  3. **Freeze** (`CalibratedProfile`): the table is an immutable dataclass with
     a JSON wire form (`tables/*.json`), applied to a `TargetProfile` by
     `apply()` -- a deterministic data substitution, no code change.
  4. **Generation-tag** (`cal_gen`): every table carries a generation and
     provenance (host fingerprint, sample count); profiles record the
     generation they were calibrated under (R11-style freshness for tables).

Fidelity note: the stdlib harness measures through the interpreter, so absolute
numbers are conservative; the *ratios* still order the regimes correctly, and
the table schema is the contract -- a native (C-runtime) backend can fill the
same tables with bare-metal numbers without touching the planner.
"""

from __future__ import annotations

import json
import os
import platform
import time
from array import array
from dataclasses import asdict, dataclass, replace

from .._artifact_json import read_bounded_text, strict_json_loads
from .cost import TargetProfile

Q8 = 256  # 1.0 in Q8 fixed point (the streaming baseline by definition)
TABLES_DIR = os.path.join(os.path.dirname(__file__), "tables")


@dataclass(frozen=True)
class MicrobenchRaw:
    """Raw median timings (ns) for the four access regimes. Inputs to the pure
    quantizer; never consumed by the planner directly."""

    stream_ns: int
    strided_ns: int
    random_ns: int
    compute_ns: int
    n: int = 0
    repeats: int = 0


@dataclass(frozen=True)
class CalibratedProfile:
    """A frozen, generation-tagged Q8 cost table for one target (the L1 artifact)."""

    name: str
    cal_gen: int
    samples: int
    provenance: str
    stream_q8: int = Q8       # == 256 by definition (verified, law R8)
    strided_q8: int = Q8
    random_q8: int = 32 * Q8
    compute_q8: int = Q8

    @property
    def gather_penalty(self) -> int:
        return max(1, self.random_q8 // Q8)

    @property
    def base_overhead(self) -> int:
        return max(1, (4 * self.strided_q8) // Q8)

    @property
    def mem_unit(self) -> int:
        return 1

    def apply(self, h: TargetProfile) -> TargetProfile:
        """Substitute the measured constants into a target profile (frozen data
        in, frozen data out -- the planner never sees the measurement)."""
        return replace(h, gather_penalty=self.gather_penalty,
                       base_overhead=self.base_overhead, mem_unit=self.mem_unit,
                       cal_gen=self.cal_gen)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @staticmethod
    def from_json(text: str) -> "CalibratedProfile":
        d = strict_json_loads(text, "calibrated profile")
        fields = {"name", "cal_gen", "samples", "provenance", "stream_q8",
                  "strided_q8", "random_q8", "compute_q8"}
        if not isinstance(d, dict) or set(d) != fields:
            raise ValueError(f"calibrated profile fields must be exactly {sorted(fields)}")
        for key in ("name", "provenance"):
            value = d[key]
            if (not isinstance(value, str) or not value or len(value) > 4096 or
                    any(ord(ch) < 0x20 for ch in value)):
                raise ValueError(f"calibrated profile {key} must be a bounded string")
        for key in ("cal_gen", "samples", "stream_q8", "strided_q8", "random_q8",
                    "compute_q8"):
            value = d[key]
            if (isinstance(value, bool) or not isinstance(value, int) or
                    value < 0 or value > (1 << 63) - 1):
                raise ValueError(f"calibrated profile {key} must be a non-negative i63")
        if d["stream_q8"] != Q8:
            raise ValueError(f"calibrated profile stream_q8 must be {Q8}")
        if any(d[key] < Q8 for key in ("strided_q8", "random_q8", "compute_q8")):
            raise ValueError("calibrated profile Q8 ratios must not be below the stream baseline")
        return CalibratedProfile(**d)


def load_table(path: str) -> CalibratedProfile:
    return CalibratedProfile.from_json(read_bounded_text(path, "calibrated profile"))


def reference_table() -> CalibratedProfile:
    """The checked-in ratio-1 reference table (reproduces the seeded constants)."""
    return load_table(os.path.join(TABLES_DIR, "x86_64_reference.json"))


# --- the measurement half (host microbenchmarks; deterministic access orders) ---

def _median(xs: list[int]) -> int:
    s = sorted(xs)
    return s[len(s) // 2]


def _shuffled(n: int) -> list[int]:
    """A seeded LCG permutation: deterministic across hosts (no `random` import
    state); only the *timing* of the run varies."""
    idx = list(range(n))
    state = 0xBC12
    for i in range(n - 1, 0, -1):
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        j = state % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    return idx


def _time_pass(buf, idx) -> int:
    t0 = time.perf_counter_ns()
    s = 0.0
    for j in idx:
        s += buf[j]
    t1 = time.perf_counter_ns()
    return max(1, t1 - t0) if s >= 0 else 1  # keep `s` live


def run_microbench(n: int = 1 << 18, repeats: int = 5, stride: int = 16) -> MicrobenchRaw:
    """Time the four regimes. All passes touch exactly n indices through the
    same indexed loop, so the indirection cost cancels and the ratio isolates
    the access order."""
    buf = array("d", (float(i % 97) for i in range(n)))
    seq = list(range(n))
    strided = [(i * stride) % n for i in range(n)]
    rand = _shuffled(n)

    t_seq = _median([_time_pass(buf, seq) for _ in range(repeats)])
    t_str = _median([_time_pass(buf, strided) for _ in range(repeats)])
    t_rnd = _median([_time_pass(buf, rand) for _ in range(repeats)])

    def _compute_pass() -> int:
        t0 = time.perf_counter_ns()
        s = 1.0
        for _ in range(n):
            s = s * 1.0000001 + 0.5
        t1 = time.perf_counter_ns()
        return max(1, t1 - t0) if s >= 0 else 1

    t_cmp = _median([_compute_pass() for _ in range(repeats)])
    return MicrobenchRaw(stream_ns=t_seq, strided_ns=t_str, random_ns=t_rnd,
                         compute_ns=t_cmp, n=n, repeats=repeats)


def calibrate_from_raw(raw: MicrobenchRaw, h: TargetProfile, cal_gen: int = 1) -> CalibratedProfile:
    """Quantize raw timings into a frozen Q8 table (pure, deterministic).

    Ratios are floored at 1.0 (Q8 256): an access regime is never cheaper than
    streaming, and a ratio-1 measurement reproduces the seeded constants.
    """
    base = max(1, raw.stream_ns)

    def q8(ns: int) -> int:
        return max(Q8, (ns * Q8) // base)

    return CalibratedProfile(
        name=h.name, cal_gen=max(1, cal_gen), samples=raw.repeats,
        provenance=f"microbench host={platform.machine()} n={raw.n} repeats={raw.repeats}",
        stream_q8=Q8, strided_q8=q8(raw.strided_ns), random_q8=q8(raw.random_ns),
        compute_q8=q8(raw.compute_ns),
    )


def calibrate_profile(h: TargetProfile, n: int = 1 << 18, repeats: int = 5,
                      cal_gen: int = 1) -> CalibratedProfile:
    """Measure this host and freeze the table (measure -> quantize -> freeze)."""
    return calibrate_from_raw(run_microbench(n=n, repeats=repeats), h, cal_gen)


# Path to the bare-metal C microbench (the measurement half on real silicon).
_NATIVE_SRC = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "runtime", "c", "bcir_microbench.c"))


def native_available() -> bool:
    """True iff a C compiler and the native microbench source are both present."""
    from shutil import which
    return os.path.exists(_NATIVE_SRC) and bool(which("clang") or which("cc") or which("gcc"))


def calibrate_native(h: TargetProfile, n: int = 1 << 22, repeats: int = 5,
                     cal_gen: int = 1, cc: str = None) -> CalibratedProfile:
    """Measure this host with the **bare-metal C microbench** (cache effects are
    real, unlike the interpreter harness) and freeze the Q8 table for `h`.

    Compiles `runtime/c/bcir_microbench.c`, runs it, and parses its JSON (the same
    `CalibratedProfile` schema). Raises `RuntimeError` if no compiler/source.
    Offline (L2/L3) only -- the frozen table is the artifact the planner consumes.
    """
    import subprocess
    import tempfile
    from shutil import which

    cc = cc or which("clang") or which("cc") or which("gcc")
    if cc is None or not os.path.exists(_NATIVE_SRC):
        raise RuntimeError("native microbench needs a C compiler + runtime/c/bcir_microbench.c")
    with tempfile.TemporaryDirectory() as d:
        exe = os.path.join(d, "mb")
        build = subprocess.run([cc, "-O2", "-std=c11", _NATIVE_SRC, "-o", exe],
                               capture_output=True, text=True)
        if build.returncode != 0:
            raise RuntimeError("native microbench build failed:\n" + build.stderr)
        run = subprocess.run([exe, str(n), str(repeats), str(cal_gen)],
                             capture_output=True, text=True)
        if run.returncode != 0:
            raise RuntimeError("native microbench run failed:\n" + run.stderr)
        raw = CalibratedProfile.from_json(run.stdout)
    # adopt the target's name; keep the measured ratios + generation.
    return replace(raw, name=h.name, cal_gen=max(1, cal_gen))


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .cost import TARGETS

    p = argparse.ArgumentParser(prog="bcir.kbcir.microbench",
                                description="Measure, quantize, and freeze a cost table")
    p.add_argument("--target", default="x86_avx512", choices=sorted(TARGETS))
    p.add_argument("--out", default=None, help="write the frozen table (JSON) here")
    p.add_argument("--n", type=int, default=1 << 18)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--cal-gen", type=int, default=1)
    p.add_argument("--bayes", action="store_true",
                   help="Bayesian table: conjugate posterior + conformal +/- delta "
                        "over --bayes-runs repeated measurements")
    p.add_argument("--bayes-runs", type=int, default=8)
    p.add_argument("--coverage", type=float, default=0.9)
    args = p.parse_args(argv)

    h = TARGETS[args.target]
    if args.bayes:
        from .bayescal import bayes_calibrate
        raws = [run_microbench(n=args.n, repeats=args.repeats) for _ in range(args.bayes_runs)]
        bt = bayes_calibrate(h, raws, coverage=args.coverage, cal_gen=args.cal_gen)
        text = bt.to_json()
        lo, hi = bt.gather_penalty_interval()
        summary = (f"[bayescal] cal_gen={bt.cal_gen} gather_penalty={bt.gather_penalty} "
                   f"in [{lo},{hi}] @ {args.coverage:.0%} (delta_q8={bt.random_delta_q8})")
    else:
        bt = calibrate_profile(h, n=args.n, repeats=args.repeats, cal_gen=args.cal_gen)
        text = bt.to_json()
        summary = f"[microbench] cal_gen={bt.cal_gen} gather_penalty={bt.gather_penalty}"

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"{summary} -> {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
