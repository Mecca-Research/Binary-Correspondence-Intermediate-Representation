#!/usr/bin/env python3
"""OT/BG (2026-10-21): run the backend-optimality experiments this host can run, write every one's
record under the measurement contract (`tools/perf/ot_contract.py`), and state the gates.

The sixteen experiments PR #821 proposes need comparators, devices and rigs a host may lack; a
record says which. This runner writes all sixteen: measured where the host and the repository hold
what the experiment needs, `unavailable` naming the capabilities the host lacks (never a simulated
pass), `blocked` naming the prerequisite the repository lacks. What runs here, on BCIR's own rails:

  OT-01  the first common block (U = X Wg, V = X Wu, Y = silu(U) * V, and its backward: dU, dV,
         dX = dU Wg^T + dV Wu^T, dWg = X^T dU, dWu = X^T dV) on BCIR's native kernels (B-DIRECT:
         `bcir_tensor_mm`, `bcir_tensor_silu_mul_pool`, `bcir_tensor_swiglu_backward`) against an
         independent float64 reference (NumPy), on the proposal's shapes with the K = 257 and
         N = 1025 tails: every element within the binary32 contract's bound -- a GEMM element
         within gamma_K (|A| |B|) of the exact product (any summation order), the elementwise
         SiLU-gate and its backward within a few units in the last place -- and the coverage ledger
         (the zero-size product a no-op; strided operands unsupported: the entries take dense rows)
  OT-03  the native step from the direct C loop (B-DIRECT: `bcir_decoder_run_accum`) and from its
         plan (B-PLAN: the step program, and its StreamPack and ExecutionPlanV1 lowered by the C),
         one state and one event sequence bit for bit, on every state case of the trace corpus
         (one per decision set the state could depend on; three for a smoke run)
  OT-04  one causal plan intervention on a common block: the step program with every layer's
         q/k/v and gate/up projections a call each (PX-3's `calls`) against the default's group
         jobs -- the launches it changes, the state it keeps, and the step's time over the
         default's in paired, rotated rounds
  OT-06  the arena frontier: the bench step's transient bytes as the C plan holds them, as GEM+'s
         static planner places them by lifetime (PX-4), and G26's lower bound
  OT-14  the trace observer's cost: the native step on a library with the launch probe against one
         without, in paired, rotated rounds, with the probe's coverage (records, none dropped)

OT-04 and OT-14 time the 9.84M comparison decoder of the PX records (`COMPARE`: 8 sequences of 128
tokens, binary32, one thread); `--smoke` times the bench step and runs OT-01 on one small shape.

    python3 tools/perf/ot_run.py OUT.json [--rows ROWS.json] [--rounds 12] [--smoke]

`--rows` names `tools/perf/gemplus_baseline.py --compare --group train --json` output: the GEM+
rows the gates read (BG1's train.plan.* rows, each met at 0). The gates are stated last; the exit
status is 0 when every record validates.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.perf import ot_contract as oc  # noqa: E402

#: The proposal's first CPU grid (section 5.1): M rows 16, 128 and 512, K depth 256 and 768, N width
#: 1024, with the K = 257 / N = 1025 tails at each M (the transposed operands are the backward's).
OT01_SHAPES = tuple((m, k, 1024) for m in (16, 128, 512) for k in (256, 768)) + tuple(
    (m, 257, 1025) for m in (16, 128, 512)
)
OT01_SMOKE = ((4, 33, 65),)
#: binary32's unit roundoff
U = 2.0**-24
#: The timed experiments' trainer (OT-04, OT-14): the 9.84M comparison decoder of the PX records
#: (vocabulary 260, width 384, 8 heads over 4 KV heads, 6 layers, FF 1024, tied; 8 sequences of
#: 128 tokens) under binary32 at one thread; `--smoke` times the bench step instead.
COMPARE = ("compare", 8, 128, "kblock+exp+reduce+attn")
#: The elementwise kernels' bound in units of their result's magnitude: libm's expf within 2 ulp,
#: then at most four more roundings (the SiLU's divide, the gate's product; the backward's four)
ELEMENTWISE_ULPS = 8


def _target(threads: int) -> dict:
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            cpu = next((ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name")),
                       cpu)  # fmt: skip
    except OSError:
        pass
    return {"cpu": cpu, "threads": threads}


def _versions() -> dict:
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip() or "unknown"  # fmt: skip
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "bcir",
                            "runtime/c", "tools/perf"], capture_output=True,
                           text=True).stdout.strip()  # fmt: skip
    out = {"bcir": commit + ("+dirty" if dirty else ""), "python": platform.python_version()}
    try:
        import numpy

        out["numpy"] = numpy.__version__
    except ImportError:
        pass
    from bcir.toolchain import host_c_compiler

    cc = host_c_compiler()
    if cc:
        first = subprocess.run([cc, "--version"], capture_output=True, text=True).stdout
        out["cc"] = first.splitlines()[0] if first else cc
    return out


def _interval(ratios: list[float]) -> tuple[float, list[float]]:
    """The geometric mean of paired ratios and its 95% Student-t interval."""
    t95 = {3: 3.182, 4: 2.776, 5: 2.571, 7: 2.365, 9: 2.262, 11: 2.201, 15: 2.131, 19: 2.093,
           23: 2.069, 31: 2.040}  # fmt: skip
    logs = [math.log(r) for r in ratios]
    mean = statistics.fmean(logs)
    t = t95[max(df for df in t95 if df <= len(logs) - 1)]
    half = t * statistics.stdev(logs) / math.sqrt(len(logs))
    return math.exp(mean), [math.exp(mean - half), math.exp(mean + half)]


# --- OT-01 ----------------------------------------------------------------------------------------


def gemm_check(name: str, got, a, b) -> dict:
    """A binary32 product `got` of float64 operands `a` (m x k) and `b` (k x n) held to the GEMM
    contract's bound in any summation order: |got - a b| <= gamma_k (|a| |b|) + u |a b| elementwise
    (Higham's bound for a k-term dot product, plus the output's rounding). Returns the worst
    error's ratio to its bound -- within the contract at most 1."""
    import numpy as np

    exact = a @ b
    depth = a.shape[1]
    gamma = depth * U / (1 - depth * U)
    bound = gamma * (np.abs(a) @ np.abs(b)) + np.abs(exact) * U
    err = np.abs(got.astype(np.float64) - exact)
    ratio = float(np.max(err / np.maximum(bound, np.finfo(np.float64).tiny)))
    return {"tensor": name, "kind": "gemm", "depth": depth, "max_abs_error": float(err.max()),
            "max_error_over_bound": ratio}  # fmt: skip


def elementwise_check(name: str, got, exact, scale) -> dict:
    """A binary32 elementwise result `got` held to `ELEMENTWISE_ULPS` units of binary32's roundoff
    of `scale` (the magnitude its terms carry) from the float64 `exact`. Returns the worst error's
    ratio to its bound -- within the contract at most 1."""
    import numpy as np

    err = np.abs(got.astype(np.float64) - exact)
    bound = ELEMENTWISE_ULPS * U * np.maximum(scale, np.finfo(np.float32).tiny)
    ratio = float(np.max(err / bound))
    return {"tensor": name, "kind": "elementwise", "max_abs_error": float(err.max()),
            "max_error_over_bound": ratio}  # fmt: skip


def ot01(lib, shapes) -> dict:
    """The first common block forward and backward on the native kernels against float64."""
    import numpy as np

    mm = lib.bcir_tensor_mm
    fp = ct.POINTER(ct.c_float)
    mm.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_size_t, ct.c_size_t, ct.c_size_t,
                   ct.c_float, fp, fp, ct.c_float, fp]  # fmt: skip
    mm.restype = None
    silu_mul = lib.bcir_tensor_silu_mul_pool
    silu_mul.argtypes = [ct.c_void_p, ct.c_size_t, fp, fp, fp]
    silu_mul.restype = None
    swiglu_bwd = lib.bcir_tensor_swiglu_backward
    swiglu_bwd.argtypes = [ct.c_size_t, fp, fp, fp, fp, fp]
    swiglu_bwd.restype = None

    def ptr(a):
        return a.ctypes.data_as(fp)

    def gemm(ta, tb, a, b, m, n, k, beta=0.0, c=None):
        out = np.zeros((m, n), np.float32) if c is None else c
        mm(None, ta, tb, m, n, k, 1.0, ptr(a), ptr(b), beta, ptr(out))
        return out

    results, raw, worst = [], [], 0.0
    rng = np.random.default_rng(821)
    for m, k, n in shapes:
        x = rng.standard_normal((m, k), dtype=np.float32)
        wg = (rng.standard_normal((k, n), dtype=np.float32) / np.sqrt(k)).astype(np.float32)
        wu = (rng.standard_normal((k, n), dtype=np.float32) / np.sqrt(k)).astype(np.float32)
        dy = rng.standard_normal((m, n), dtype=np.float32)
        # BCIR: forward
        u, v = gemm(0, 0, x, wg, m, n, k), gemm(0, 0, x, wu, m, n, k)
        y = np.empty((m, n), np.float32)
        silu_mul(None, m * n, ptr(u), ptr(v), ptr(y))
        # backward
        du, dv = np.empty_like(u), np.empty_like(v)
        swiglu_bwd(m * n, ptr(u), ptr(v), ptr(dy), ptr(du), ptr(dv))
        dx = gemm(0, 1, du, wg, m, k, n)  # dU Wg^T
        dx = gemm(0, 1, dv, wu, m, k, n, beta=1.0, c=dx)  # + dV Wu^T
        dwg, dwu = gemm(1, 0, x, du, k, n, m), gemm(1, 0, x, dv, k, n, m)  # X^T dU, X^T dV
        # the float64 reference, from BCIR's own intermediates where an elementwise stage feeds a
        # product (so each check holds one kernel to its contract, not the chain's accumulation)
        X, Wg, Wu = x.astype(np.float64), wg.astype(np.float64), wu.astype(np.float64)

        U64, V64, DY = u.astype(np.float64), v.astype(np.float64), dy.astype(np.float64)
        sig = 1.0 / (1.0 + np.exp(-U64))
        silu = U64 * sig
        y_exact = silu * V64
        dg_exact = DY * V64 * (sig * (1.0 + U64 * (1.0 - sig)))
        du_exact = DY * silu
        checks = [
            gemm_check("U", u, X, Wg), gemm_check("V", v, X, Wu),
            elementwise_check("Y", y, y_exact, np.abs(silu) * np.abs(V64) + np.abs(y_exact)),
            elementwise_check("dU", du, dg_exact, np.abs(DY) * np.abs(V64)
                              * (np.abs(sig) + np.abs(U64) * sig * (1 - sig)) + np.abs(dg_exact)),
            elementwise_check("dV", dv, du_exact, np.abs(DY) * np.abs(silu) + np.abs(du_exact)),
            gemm_check("dX", dx, np.concatenate([du, dv], 1).astype(np.float64),
                       np.concatenate([Wg.T, Wu.T], 0)),
            gemm_check("dWg", dwg, X.T, du.astype(np.float64)),
            gemm_check("dWu", dwu, X.T, dv.astype(np.float64)),
        ]  # fmt: skip
        case_worst = max(c["max_error_over_bound"] for c in checks)
        worst = max(worst, case_worst)
        results.append({"shape": {"m": m, "k": k, "n": n}, "within_bound": case_worst <= 1.0,
                        "max_error_over_bound": case_worst})  # fmt: skip
        raw.append({"shape": {"m": m, "k": k, "n": n}, "checks": checks})
    # the coverage ledger: what the entries admit beyond dense row-major operands -- the
    # zero-size product writes nothing (m = 0 over a sentinel output)
    sentinel = np.full((2, 2), 7.0, np.float32)
    operand = np.ones((2, 3), np.float32)
    mm(None, 0, 0, 0, 2, 3, 1.0, ptr(operand), ptr(operand), 0.0, ptr(sentinel))
    untouched = bool((sentinel == 7.0).all())
    ledger = [
        {"operation": "gemm nn / nt / tn (dense row-major, alpha, beta)", "state": "supported"},
        {"operation": "silu(U) * V", "state": "supported"},
        {"operation": "swiglu backward (dU, dV)", "state": "supported"},
        {"operation": "the zero-size product", "state": "supported",
         "detail": "m = 0 writes nothing" if untouched else "m = 0 WROTE its output"},
        {"operation": "strided or non-contiguous operands", "state": "unsupported",
         "detail": "the tensor entries take dense row-major operands and their transposes"},
    ]  # fmt: skip
    results.append(
        {
            "ledger": ledger,
            "worst_error_over_bound": worst,
            "within_bound": worst <= 1.0 and untouched,
        }
    )
    return {
        "experiment": "OT-01", "status": "measured", "routes": ["B-DIRECT"],
        "semantic": "swiglu.block.fwd+bwd.binary32", "reference": "numpy.float64",
        "results": results, "raw": raw,
        "memory_scopes": {"logical_bytes": int(sum(4 * (m * k + 2 * k * n + 3 * m * n)
                                                   for m, k, n in shapes))},
        "labels": {"unsupported": ["strided or non-contiguous operands", "P-EAGER (no PyTorch "
                                   "on this host)"] if "torch" not in oc.host_capabilities()
                   else ["strided or non-contiguous operands"], "fallback": []},
        "gaps": ["capability"],
    }  # fmt: skip


# --- OT-03, OT-04, OT-06, OT-14 ---------------------------------------------------------------------


def ot03(packed, traced, smoke: bool) -> dict:
    """The direct step, its program and its pack: one state, on the trace corpus's state cases --
    one per decision set the state could depend on (contract, recompute, workspace tier, threads,
    provider, a failing bound; multi-step and accumulated where the corpus has it), three for a
    smoke run -- each pair on one build, as the rows compare them (the pack and the direct step on
    the pack build, the program and the direct step on the trace build), and the two builds'
    states."""
    from bcir.tests import native_decoder_fixtures as nf

    corpus = nf.trace_corpus()
    picked = nf.state_cases(corpus)[: 3 if smoke else None]
    results, raw = [], []
    for i in picked:
        case = corpus[i]
        direct = nf.trace_case(packed, case, trace=False)
        pack = nf.pack_case(packed, case)
        direct_t = nf.trace_case(traced, case, trace=False)
        prog = nf.program_case(traced, case)
        pack_same = pack.status == 0 and (pack.state, pack.events) == (direct.state, direct.events)
        prog_same = prog.status == 0 and (prog.state, prog.events) == (direct_t.state,
                                                                       direct_t.events)  # fmt: skip
        builds_same = (direct.state, direct.events) == (direct_t.state, direct_t.events)
        results.append({"case": str(case), "pack_is_direct": pack_same,
                        "program_is_direct": prog_same, "builds_agree": builds_same,
                        "one_state": pack_same and prog_same and builds_same})  # fmt: skip
        raw.append({"case": str(case), "direct": direct.state, "pack": pack.state,
                    "pack_status": pack.status, "direct_trace_build": direct_t.state,
                    "program": prog.state, "program_status": prog.status})  # fmt: skip
    return {
        "experiment": "OT-03", "status": "measured", "routes": ["B-DIRECT", "B-PLAN"],
        "semantic": "native.decoder.step", "reference": "bcir.direct.step",
        "results": results, "raw": raw, "memory_scopes": {},
        "labels": {"unsupported": ["P-EAGER (no PyTorch on this host)"]
                   if "torch" not in oc.host_capabilities() else [], "fallback": []},
        "gaps": ["integration"],
    }  # fmt: skip


def _timed_rounds(arms: dict, rounds: int) -> dict:
    """Paired, rotated rounds of each arm's one-step call: {arm: [ms, ...]}."""
    names = tuple(arms)
    ms = {a: [] for a in names}
    for fn in arms.values():
        fn()  # warm
    for r in range(rounds):
        for j in range(len(names)):
            arm = names[(r + j) % len(names)]
            t0 = time.perf_counter()
            arms[arm]()
            ms[arm].append(1000 * (time.perf_counter() - t0))
    return ms


def case_name(case) -> dict:
    """A timed case as its record names it: the decoder's shape, the call's and the contract."""
    s = case.spec
    return {"vocab": s.vocab_size, "d_model": s.d_model, "n_heads": s.n_heads,
            "kv_heads": s.kv_heads, "n_layers": s.n_layers, "d_ff": s.d_ff, "batch": case.batch,
            "time": case.time, "contract": case.contract, "threads": case.threads}  # fmt: skip


def timed_case(smoke: bool):
    """The trainer OT-04 and OT-14 time: the bench step for a smoke run, else `COMPARE`."""
    from bcir.frontends.models.decode import DecoderSpec
    from bcir.tests import native_decoder_fixtures as nf

    if smoke:
        spec, batch, time_ = nf.BENCH
        return nf.TraceCase(spec, batch, time_, "v1")
    _, batch, time_, contract = COMPARE
    spec = DecoderSpec(260, 384, 8, 6, 1024, activation="silu_gate", n_kv_heads=4,
                       tied_embeddings=True)  # fmt: skip
    return nf.TraceCase(spec, batch, time_, contract)


def ot04(path, rounds: int, case) -> dict:
    """One causal intervention: every layer's projections a call each, against the group jobs."""
    from bcir.hosted.models.native_launch import choice_sites, program_words, step_program
    from bcir.tests import native_decoder_fixtures as nf

    spec, batch, time_ = case.spec, case.batch, case.time
    trainer, decisions = nf._case_trainer(path, case)
    other, _ = nf._case_trainer(path, case)
    try:
        sites = {s: "calls" for s in choice_sites(spec) if s.endswith(("qkv.fwd", "gateup.fwd"))}
        default = step_program(spec, batch, time_, **decisions)
        switched = step_program(spec, batch, time_, **decisions, sites=sites)
        tokens, targets, optimizer = nf._case_inputs(case)
        words = {"default": program_words(default), "switched": program_words(switched)}

        def run(tr, w):
            return lambda: tr.run_program(w, tokens, targets, steps=1, optimizer=optimizer)

        ms = _timed_rounds({"default": run(trainer, words["default"]),
                            "switched": run(other, words["switched"])}, rounds)  # fmt: skip
        mean, ci = _interval([s / d for s, d in zip(ms["switched"], ms["default"])])
        same = nf.state_digest(trainer) == nf.state_digest(other)
    finally:
        trainer.close()
        other.close()
    launches = {"default": len(default), "switched": len(switched)}
    return {
        "experiment": "OT-04", "status": "measured", "routes": ["B-DIRECT", "B-PLAN"],
        "semantic": "native.decoder.step.sites", "reference": "bcir.default.program",
        "results": [{"intervention": "every layer's q/k/v and gate/up projections a call each",
                     "case": case_name(case), "launches": launches, "one_state": same,
                     "time_ratio": {"geometric_mean": mean, "ci95": ci, "rounds": rounds}}],
        "raw": [{"ms_per_step": {k: [round(v, 4) for v in vs] for k, vs in ms.items()}}],
        "memory_scopes": {}, "labels": {"unsupported": [], "fallback": []},
        "gaps": ["integration", "selection"],
    }  # fmt: skip


def ot06() -> dict:
    """The bench step's transient bytes: the C plan's, the placed (PX-4) and G26's bound."""
    from bcir.tests import native_decoder_fixtures as nf

    ratio, held, bound, incumbent = nf.transient_ratio()
    placed_ratio, placed, placed_bound, optimal = nf.placed_transient()
    return {
        "experiment": "OT-06", "status": "measured", "routes": ["B-DIRECT", "B-PLAN"],
        "semantic": "native.decoder.transients", "reference": "g26.lower.bound",
        "results": [{"c_plan_over_bound": ratio, "placed_over_bound": placed_ratio,
                     "placed_optimal": optimal}],
        "raw": [{"c_plan_bytes": held, "bound_bytes": bound, "incumbent_bytes": incumbent,
                 "placed_bytes": placed, "placed_bound_bytes": placed_bound}],
        "memory_scopes": {"c_plan_transient_bytes": held, "placed_transient_bytes": placed,
                          "lower_bound_bytes": bound},
        "labels": {"unsupported": [], "fallback": []}, "gaps": ["bound-slack", "realization"],
    }  # fmt: skip


def ot14(plain, traced, rounds: int, case) -> dict:
    """The trace observer's cost on the step: the probe's library against the plain one."""
    from bcir.hosted.models.native_launch import RECORD_WORDS
    from bcir.tests import native_decoder_fixtures as nf

    a, _ = nf._case_trainer(plain, case)
    b, _ = nf._case_trainer(traced, case)
    try:
        lib = b.lib
        nf._probe(lib)
        capacity = 1 << 16  # room for every launch of the warm-up and the rounds
        words = (ct.c_uint64 * (capacity * RECORD_WORDS))()
        tokens, targets, optimizer = nf._case_inputs(case)
        lib.bcir_decoder_trace_probe_begin(ct.addressof(b.state), ct.addressof(words), capacity)
        try:
            ms = _timed_rounds({"off": lambda: a.run(tokens, targets, steps=1, optimizer=optimizer),
                                "on": lambda: b.run(tokens, targets, steps=1, optimizer=optimizer)},
                               rounds)  # fmt: skip
        finally:
            dropped = ct.c_size_t()
            count = lib.bcir_decoder_trace_probe_end(ct.byref(dropped))
        mean, ci = _interval([on / off for on, off in zip(ms["on"], ms["off"])])
        same = nf.state_digest(a) == nf.state_digest(b)
    finally:
        a.close()
        b.close()
    return {
        "experiment": "OT-14", "status": "measured", "routes": ["B-DIRECT"],
        "semantic": "native.decoder.step.trace", "reference": "bcir.untraced.step",
        "results": [{"case": case_name(case),
                     "observer_over_plain": {"geometric_mean": mean, "ci95": ci,
                                             "rounds": rounds},
                     "records": int(count), "dropped": int(dropped.value), "one_state": same}],
        "raw": [{"ms_per_step": {k: [round(v, 4) for v in vs] for k, vs in ms.items()}}],
        "memory_scopes": {"trace_buffer_bytes": capacity * RECORD_WORDS * 8},
        "labels": {"unsupported": ["device counters (no PMU readable here)"]
                   if "pmu" not in oc.host_capabilities() else [], "fallback": []},
        "gaps": ["system"],
    }  # fmt: skip


# --- the others -------------------------------------------------------------------------------------

#: The prerequisite each experiment's runner needs from the repository, where one is missing.
BLOCKED = {
    "OT-15": "RuntimeChannel has its direct adapter alone: no shared-memory queue, Linux process or "
             "native adapter exists to decompose against it (the driver ladder's M2, M3 and M8)",
}  # fmt: skip


def unrun(experiment: oc.Experiment, caps: set[str]) -> dict:
    lacking = [c for c in experiment.needs if c not in caps]
    if lacking:
        return {"experiment": experiment.id, "status": "unavailable", "missing": lacking,
                "reason": "this host lacks " + ", ".join(oc.CAPABILITIES[c] for c in lacking)}  # fmt: skip
    if experiment.id in BLOCKED:
        return {"experiment": experiment.id, "status": "blocked",
                "reason": BLOCKED[experiment.id]}  # fmt: skip
    raise RuntimeError(f"{experiment.id} has every capability and no runner")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", type=Path)
    p.add_argument("--rows", type=Path, help="gemplus_baseline.py --json output (the gates' rows)")
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--smoke", action="store_true", help="tiny shapes and four rounds")
    a = p.parse_args(argv)
    caps = oc.host_capabilities()
    from bcir.tests import native_decoder_fixtures as nf

    records, rounds = [], 4 if a.smoke else a.rounds
    with tempfile.TemporaryDirectory(prefix="bcir-ot-") as tmp:
        plain = nf.build_library(tmp)
        packed = nf.pack_library(Path(tmp) / "pack")
        traced = nf.trace_library(Path(tmp) / "trace")
        runners = {}
        if plain is not None and "numpy" in caps:
            runners["OT-01"] = lambda: ot01(ct.CDLL(str(plain)),
                                            OT01_SMOKE if a.smoke else OT01_SHAPES)  # fmt: skip
        if packed is not None and traced is not None:
            runners["OT-03"] = lambda: ot03(packed, traced, a.smoke)
        case = timed_case(a.smoke)
        if packed is not None:
            runners["OT-04"] = lambda: ot04(packed, rounds, case)
        runners["OT-06"] = ot06
        if plain is not None and traced is not None:
            runners["OT-14"] = lambda: ot14(plain, traced, rounds, case)
        versions = _versions()
        for experiment in oc.EXPERIMENTS:
            run = runners.get(experiment.id)
            record = run() if run else unrun(experiment, caps)
            if record["status"] == "measured":
                record = {**record, "target": _target(1), "versions": versions}
            records.append(oc.validate_record(record))
            print(f"{experiment.id} {record['status']}", flush=True)
    rows = {}
    if a.rows:  # gemplus_baseline.py --compare --json: each row's key and the value it measured
        doc = json.loads(a.rows.read_text())
        rows = {r["key"]: r["measured"] for r in doc["rows"]
                if type(r.get("key")) is str and isinstance(r.get("measured"), (int, float))}  # fmt: skip
    gates = oc.evaluate_gates(records, rows, capabilities=caps)
    out = {"proposal": oc.PROPOSAL, "capabilities": sorted(caps), "target": _target(1),
           "versions": versions, "records": records, "rows": rows,
           "gates": {g: {"state": s, "unmet": u} for g, (s, u) in gates.items()}}  # fmt: skip
    a.out.write_text(
        json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    for g, (state, unmet) in gates.items():
        print(f"{g} {state}" + (f" ({len(unmet)} unmet)" if unmet else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
