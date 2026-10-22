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
tokens, binary32, one thread), each beside an A/A control -- a second trainer running the reference
arm in the same rotation -- so an effect is read against this host's own noise; `--smoke` times the
bench step and runs OT-01 on one small shape.

    python3 tools/perf/ot_run.py OUT.json [--rows ROWS.json] [--rounds 24] [--smoke]
        [--triton SESSION.json] [--frontier DATA_DIR]

`--rows` names the report `tools/perf/gemplus_baseline.py --compare --group train --json ROWS.json`
writes: the GEM+ rows the gates read (BG1's train.plan.* rows, each met at 0). `--triton` names a
Triton-CPU pilot session (OT-05's T-DSL rows), `--frontier` train_comparison.py's prepared data
and its runs (OT-10). Every file named is read before any experiment runs, and one the run cannot
use is refused then (exit status 2). The gates are stated last; the exit status is 0 when every
record validates.
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


def _versions(caps=frozenset()) -> dict:
    """What the run ran: this tree's commit (+dirty where bcir/, runtime/c or tools/perf differ from
    it), Python, NumPy, the C compiler, and -- where PyTorch is a capability -- the PyTorch the P-*
    routes ran, its CPU capability, threads and BLAS, each a plain string (`torch.__version__` is a
    str subclass, which the contract's versions refuse)."""
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
    if "torch" in caps:
        from tools.perf import ot_torch

        out.update({k: str(v) for k, v in ot_torch.torch_versions().items()})
    return out


def _ran_at(threads: int, frontier) -> dict:
    """The threads each measured experiment ran at, for its record's target: the PyTorch routes'
    runners the host's (OT-05's widest rows), OT-10 its runs' own, every other experiment one."""
    out = {"OT-02": threads, "OT-05": threads, "OT-11": threads}
    if frontier is not None:
        ran = [r["threads"] for r in frontier["results"] if type(r.get("threads")) is int]
        if ran:
            out["OT-10"] = max(ran)
    return out


def _stamp(record: dict, ran_at: dict, versions: dict) -> dict:
    """A measured record with its target -- this host's CPU and the threads it ran at -- and the
    versions that ran; any other record as it is."""
    if record["status"] != "measured":
        return record
    return {**record, "target": _target(ran_at.get(record["experiment"], 1)), "versions": versions}


def _interval(ratios: list[float]) -> tuple[float, list[float]]:
    """The geometric mean of paired ratios and its 95% Student-t interval."""
    t95 = {3: 3.182, 4: 2.776, 5: 2.571, 7: 2.365, 9: 2.262, 11: 2.201, 15: 2.131, 19: 2.093,
           23: 2.069, 31: 2.040}  # fmt: skip
    logs = [math.log(r) for r in ratios]
    mean = statistics.fmean(logs)
    t = t95[max(df for df in t95 if df <= len(logs) - 1)]
    half = t * statistics.stdev(logs) / math.sqrt(len(logs))
    return math.exp(mean), [math.exp(mean - half), math.exp(mean + half)]


def _unrun_route(route: str) -> str:
    """The label of a comparator route this run did not compare: why, on this interpreter -- no
    PyTorch, or PyTorch present and the route's runner not yet in this tree."""
    if "torch" not in oc.host_capabilities():
        return f"{route} (no PyTorch in this interpreter)"
    return f"{route} (no runner for this route yet)"


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


def ot01(lib, shapes, with_torch: bool = False) -> dict:
    """The first common block forward and backward on the native kernels against float64; with
    `with_torch`, the same block on PyTorch eager (P-EAGER, tools/perf/ot_torch.py) from the same
    inputs, held to the same bounds."""
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
        results.append({"route": "B-DIRECT", "shape": {"m": m, "k": k, "n": n},
                        "within_bound": case_worst <= 1.0, "max_error_over_bound": case_worst})  # fmt: skip
        raw.append({"route": "B-DIRECT", "shape": {"m": m, "k": k, "n": n}, "checks": checks})
    routes, unsupported, torch_worst = ["B-DIRECT"], [_unrun_route("P-EAGER")], 0.0
    if with_torch:
        from tools.perf import ot_torch

        torch_results, torch_raw, torch_worst = ot_torch.p_eager_block(shapes)
        results += torch_results
        raw += torch_raw
        routes, unsupported = ["B-DIRECT", "P-EAGER"], []
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
            "p_eager_worst_error_over_bound": torch_worst if with_torch else None,
            "within_bound": worst <= 1.0 and torch_worst <= 1.0 and untouched,
        }
    )
    return {
        "experiment": "OT-01", "status": "measured", "routes": routes,
        "semantic": "swiglu.block.fwd+bwd.binary32", "reference": "numpy.float64",
        "results": results, "raw": raw,
        "memory_scopes": {"logical_bytes": int(sum(4 * (m * k + 2 * k * n + 3 * m * n)
                                                   for m, k, n in shapes))},
        "labels": {"unsupported": ["strided or non-contiguous operands (B-DIRECT)", *unsupported],
                   "fallback": []},
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
        "labels": {"unsupported": [_unrun_route("P-EAGER")], "fallback": []},
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
    control, _ = nf._case_trainer(path, case)
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
                            "default2": run(control, words["default"]),
                            "switched": run(other, words["switched"])}, rounds)  # fmt: skip
        mean, ci = _interval([s / d for s, d in zip(ms["switched"], ms["default"])])
        aa_mean, aa_ci = _interval([s / d for s, d in zip(ms["default2"], ms["default"])])
        same = nf.state_digest(trainer) == nf.state_digest(other) == nf.state_digest(control)
    finally:
        trainer.close()
        control.close()
        other.close()
    launches = {"default": len(default), "switched": len(switched)}
    return {
        "experiment": "OT-04", "status": "measured", "routes": ["B-DIRECT", "B-PLAN"],
        "semantic": "native.decoder.step.sites", "reference": "bcir.default.program",
        "results": [{"intervention": "every layer's q/k/v and gate/up projections a call each",
                     "case": case_name(case), "launches": launches, "one_state": same,
                     "time_ratio": {"geometric_mean": mean, "ci95": ci, "rounds": rounds},
                     "aa_control": {"geometric_mean": aa_mean, "ci95": aa_ci, "rounds": rounds}}],
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
    a2, _ = nf._case_trainer(plain, case)
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
                                "off2": lambda: a2.run(tokens, targets, steps=1, optimizer=optimizer),
                                "on": lambda: b.run(tokens, targets, steps=1, optimizer=optimizer)},
                               rounds)  # fmt: skip
        finally:
            dropped = ct.c_size_t()
            count = lib.bcir_decoder_trace_probe_end(ct.byref(dropped))
        mean, ci = _interval([on / off for on, off in zip(ms["on"], ms["off"])])
        aa_mean, aa_ci = _interval([o2 / off for o2, off in zip(ms["off2"], ms["off"])])
        same = nf.state_digest(a) == nf.state_digest(b) == nf.state_digest(a2)
    finally:
        a.close()
        a2.close()
        b.close()
    return {
        "experiment": "OT-14", "status": "measured", "routes": ["B-DIRECT"],
        "semantic": "native.decoder.step.trace", "reference": "bcir.untraced.step",
        "results": [{"case": case_name(case),
                     "observer_over_plain": {"geometric_mean": mean, "ci95": ci,
                                             "rounds": rounds},
                     "aa_control": {"geometric_mean": aa_mean, "ci95": aa_ci, "rounds": rounds},
                     "records": int(count), "dropped": int(dropped.value), "one_state": same}],
        "raw": [{"ms_per_step": {k: [round(v, 4) for v in vs] for k, vs in ms.items()}}],
        "memory_scopes": {"trace_buffer_bytes": capacity * RECORD_WORDS * 8},
        "labels": {"unsupported": ["device counters (" + ("not read by this runner yet"
                                   if "pmu" in oc.host_capabilities() else "no PMU readable here")
                                   + ")"], "fallback": []},
        "gaps": ["system"],
    }  # fmt: skip


# --- OT-02, OT-05, OT-11: the PyTorch routes (tools/perf/ot_torch.py) --------------------------------

#: The experiments whose runner needs PyTorch: measured where it imports, else unavailable.
TORCH_RUNNERS = ("OT-02", "OT-05", "OT-11")


def _first_line(reason: str) -> str:
    return reason.strip().splitlines()[0] if reason.strip() else reason


def ot02(threads: int, smoke: bool) -> dict:
    """Capture cost of the decoder's training step: P-EAGER (no capture), P-FX, P-AOT and P-IND --
    each cold in its own process, the module as written and its loss as one graph -- beside B-PLAN's
    planning and lowering of the same step and the C library's build."""
    from tools.perf import ot_torch

    out = ot_torch.capture_cost(threads, smoke)
    results, failed = [], []
    for name, row in out["routes"].items():
        if "failed" in row:
            failed.append(f"{name}: {row['failed']}")
            continue
        if name != "B-PLAN":
            row = {**row, "capture_s": row["first_s"] - row["warm_s"],
                   "break_reasons": sorted({_first_line(r) for r in row["break_reasons"]})}  # fmt: skip
        results.append({"route": name, **row})
    ran = sorted({r["route"].split("/")[0] for r in results}, key=list(oc.ROUTES).index)
    return {
        "experiment": "OT-02", "status": "measured", "routes": ran,
        "semantic": "native.decoder.step.capture", "reference": "p-eager.uncaptured.step",
        "results": [{k: v for k, v in out.items() if k != "routes"}, *results],
        "raw": [out["routes"]], "memory_scopes": {},
        "labels": {"unsupported": ["the compilers' peak memory (not read by this runner)",
                                   *failed], "fallback": []},
        "gaps": ["capability", "compilation"],
    }  # fmt: skip


#: The Triton-CPU pilot's protocol (PR #821's tools/perf/triton_cpu_compare.py, its own
#: interpreter: the experimental triton-cpu build needs Python 3.12 and runs without PyTorch).
TRITON_PROTOCOL = "bcir-triton-cpu-pilot-v2"


def _triton_evidence(path) -> dict:
    """A Triton-CPU pilot session as OT-05's T-DSL rows: its protocol, its PASS, every case's
    correctness on both sides and its paired ratio -- and its BCIR side this tree's tensor unit (the
    session's hash of bcir_tensor.c), or it is no evidence about this tree (ValueError). A case
    times one or more of BCIR's implementations against Triton's kernel (the GEMM's fused and
    k-blocked contracts, SwiGLU's libm and BCIR exponentials), each with its own correctness and
    ratio: a row each, every one correct on both sides."""
    import hashlib

    doc = oc.load_json(path)
    if not isinstance(doc, dict) or doc.get("protocol") != TRITON_PROTOCOL:
        raise ValueError(f"a Triton-CPU session names protocol {TRITON_PROTOCOL!r}")
    if doc.get("status") != "PASS" or not isinstance(doc.get("cases"), list) or not doc["cases"]:
        raise ValueError("a Triton-CPU session passed and carries its cases")
    here = hashlib.sha256((ROOT / "runtime" / "c" / "bcir_tensor.c").read_bytes()).hexdigest()
    if doc.get("tensor_c_sha256") != here:
        raise ValueError("the Triton-CPU session's BCIR side is not this tree's bcir_tensor.c")
    rows = []
    for c in doc["cases"]:
        name = f"{c.get('kind')} {c.get('shape')}" if isinstance(c, dict) else repr(c)
        ratios = c.get("ratios") if isinstance(c, dict) else None
        if not isinstance(ratios, dict) or not ratios:
            raise ValueError(f"a Triton-CPU case times one of BCIR's implementations: {name}")
        correct, median = c.get("correctness"), c.get("median_ns")
        for impl, ratio in sorted(ratios.items()):
            ok = (
                impl.startswith("bcir")
                and isinstance(correct, dict)
                and all(
                    isinstance(correct.get(side), dict) and correct[side].get("pass") is True
                    for side in (impl, "triton")
                )
            )
            timed = (
                isinstance(ratio, dict)
                and type(ratio.get("bcir_over_triton")) is float
                and isinstance(median, dict)
                and all(type(median.get(side)) is float for side in (impl, "triton"))
            )
            if not ok or not timed:
                raise ValueError(f"a Triton-CPU case is correct on both sides: {name} ({impl})")
            rows.append({"kind": c["kind"], "shape": c["shape"], "threads": c.get("threads"),
                         "implementation": impl,
                         "median_ns": {impl: median[impl], "triton": median["triton"]},
                         "bcir_over_triton": ratio["bcir_over_triton"],
                         "ci95": ratio.get("paired_bootstrap_95pct")})  # fmt: skip
    source = {
        k: doc.get(k)
        for k in (
            "protocol",
            "utc",
            "cpu",
            "rounds",
            "harness_sha256",
            "triton_package",
            "triton_cpu_source",
            "triton_cpu_wheel_sha256",
            "tensor_c_sha256",
            "numerics",
        )
    }
    return {"summary": {"cases": rows, "numerics": doc.get("numerics")}, "source": source}


def ot05(plain, threads: int, smoke: bool, rounds: int, triton=None) -> dict:
    """The block portfolio: B-DIRECT under three GEMM contracts and under the trainer's (kblock+exp),
    P-EAGER and P-IND, at one thread and at the host's (OT-01's shapes), each beside B-DIRECT's A/A
    control."""
    from tools.perf import ot_torch

    shapes = OT01_SMOKE if smoke else OT01_SHAPES
    out = ot_torch.portfolio(plain, shapes, sorted({1, threads}), max(rounds, 4))
    ran = sorted({a.split("/")[0] for row in out["rows"] for a in row["forward"]["ms"]},
                 key=list(oc.ROUTES).index)  # fmt: skip
    unsupported = [*out["failures"],
                   "B-MLIR (BCIR's MLIR route lowers one matmul buffer nest, not this block)",
                   "T-GLUON (Gluon's explicit layouts target a GPU)"]  # fmt: skip
    if triton is None:
        unsupported.append(
            "T-DSL ("
            + (
                "its Triton-CPU rows are a separate run: name them --triton"
                if "triton" not in oc.host_capabilities()
                else "no runner yet"
            )
            + ")"
        )
    results = [{k: v for k, v in row.items() if k != "raw_ms"} for row in out["rows"]]
    raw = [{"shape": row["shape"], "threads": row["threads"],
            "forward": row["forward"]["raw_ms"], "forward_backward": row["forward_backward"]["raw_ms"]}
           for row in out["rows"]]  # fmt: skip
    if triton is not None:
        ran.append("T-DSL")
        results.append({"route": "T-DSL", **triton["summary"]})
        raw.append({"route": "T-DSL", "source": triton["source"]})
    for row in results:
        for phase in ("forward", "forward_backward"):
            if phase in row:
                row[phase] = {k: v for k, v in row[phase].items() if k != "raw_ms"}
    return {
        "experiment": "OT-05", "status": "measured", "routes": ran,
        "semantic": "swiglu.block.fwd+bwd.binary32", "reference": "b-direct.kblock",
        "results": results, "raw": raw,
        "memory_scopes": {"block_logical_bytes_largest": max(
            4 * (m * k + 2 * k * n + 6 * m * n + m * k + 2 * k * n) for m, k, n in shapes)},
        "labels": {"unsupported": unsupported,
                   "fallback": [f"P-* products: {ot_torch.TORCH_CONTRACT}; B-DIRECT's are its "
                                "contracts' fixed orders (v1, fused v2, blocked v3), its "
                                "elementwise passes v1's libm SiLU but in kblock+exp (the BCIR "
                                "exponential's)"]},
        "gaps": ["realization", "competitor"],
    }  # fmt: skip


#: train_comparison.py's backends as OT-10's routes (its `bcir` backend runs the direct C loop)
FRONTIER_ROUTES = {"bcir": "B-DIRECT", "torch-eager": "P-EAGER", "torch-compiled": "P-IND",
                   "torch-tuned": "P-TUNED"}  # fmt: skip


def ot10(data, task: str = "lm") -> dict:
    """The complete training frontier from train_comparison.py's runs over one prepared recipe
    (`data`: its prepare.json and runs/): each route's learning against its training seconds (its
    evaluations), its held-out cross-entropy, its median step and tokens a second, its compile and
    first steps, its peak RSS -- every run on the recipe's weights, schedule, batch and steps (the
    runs refused otherwise: a frontier across two recipes is not one)."""
    data = Path(data)
    prepared = oc.load_json(data / "prepare.json")
    runs = {}
    for path in sorted((data / "runs").glob(f"{task}-*.json")):
        r = oc.load_json(path)
        route = FRONTIER_ROUTES.get(r.get("backend"))
        if route is None or r.get("task") != task:
            continue
        if route in runs:
            raise ValueError(f"two {route} runs of {task} in {data}")
        runs[route] = (path.name, r)
    if not runs:
        raise ValueError(f"no train_comparison.py runs of {task} in {data}")
    steps = {len(r["losses"]) for _, r in runs.values()}
    if len(steps) != 1:
        raise ValueError(f"the runs trained different step counts: {sorted(steps)}")
    results = []
    for route in sorted(runs, key=list(oc.ROUTES).index):
        name, r = runs[route]
        results.append({
            "route": route, "run": name, "contract": r.get("contract"), "threads": r.get("threads"),
            "steps": len(r["losses"]), "median_step_ms": r["median_step_ms"],
            "tokens_per_second": r["tokens_per_second"], "train_seconds": r["train_seconds"],
            "compile_and_first_steps_seconds": r.get("compile_and_first_steps_seconds"),
            "curve": [{"step": e["step"], "train_seconds": e["train_seconds"],
                       "cross_entropy": e["cross_entropy"]} for e in r["evals"]],
            "final": {split: {"cross_entropy": v["cross_entropy"], "accuracy": v["accuracy"]}
                      for split, v in r["final"].items()},
            "peak_rss_mb": r["rss_mb"]["peak"], "state_sha256": r["state_sha256"],
        })  # fmt: skip
    rss = {r["route"]: int(r["peak_rss_mb"] * 2**20) for r in results}
    return {
        "experiment": "OT-10", "status": "measured", "routes": [r["route"] for r in results],
        "semantic": f"native.decoder.train.{task}", "reference": "p-eager.training.curve",
        "results": [{"recipe": {"weights_sha256": prepared.get("weights_sha256"),
                                "batch": prepared.get("batch"), "context": prepared.get("context"),
                                "parameters": prepared.get("parameters"), "steps": steps.pop(),
                                "task": task}}, *results],
        "raw": [{"route": r["route"], "losses": runs[r["route"]][1]["losses"]} for r in results],
        "memory_scopes": {f"{k}_peak_rss_bytes": v for k, v in rss.items()},
        "labels": {"unsupported": [f"{rt} (no run)" for rt in ("P-TUNED", "P-IND", "P-EAGER")
                                   if rt not in runs] + ["B-PLAN's own learning run (OT-03: its "
                                   "state is B-DIRECT's bit for bit; OT-04 prices the step)"],
                   "fallback": []},
        "gaps": ["competitor"],
    }  # fmt: skip


def ot11(threads: int, smoke: bool) -> dict:
    """P-IND's cache lifecycle over the decoder's step -- cold, warm, persistent, a shape change
    static and dynamic -- beside B-PLAN's (one library for every shape, a plan a shape)."""
    from tools.perf import ot_torch

    out = ot_torch.lifecycle(threads, smoke)
    failed = [
        f"{k}: {v['failed']}" for k, v in out.items() if isinstance(v, dict) and "failed" in v
    ]
    rows = {k: v for k, v in out.items()
            if k in ("cold_static", "persistent_static", "cold_dynamic") and "failed" not in v}  # fmt: skip
    results = [{"case": k, **{f: v.get(f) for f in ("first_s", "warm_s", "frames",
                                                      "second_shape_first_s", "second_shape_warm_s",
                                                      "frames_after_second_shape")}}
               for k, v in rows.items()]  # fmt: skip
    results.append({"case": "B-PLAN", **out["B-PLAN"]})
    triton = "triton" in oc.host_capabilities()
    return {
        "experiment": "OT-11", "status": "measured",
        "routes": ["P-IND", "B-PLAN"] if rows else ["B-PLAN"],
        "semantic": "native.decoder.step.lifecycle", "reference": "p-ind.cold.compile",
        "results": [{k: v for k, v in out.items() if k in ("decoder", "batch", "time", "second_time",
                                                             "threads")}, *results],
        "raw": [out], "memory_scopes": {},
        "labels": {"unsupported": [*failed,
                                   "P-TUNED (a budgeted autotuning lifecycle: not run here)",
                                   "T-DSL (" + ("no runner yet" if triton else
                                                "no Triton with a usable backend here") + ")"],
                   "fallback": []},
        "gaps": ["compilation"],
    }  # fmt: skip


# --- the others -------------------------------------------------------------------------------------

#: The prerequisite each experiment's runner needs from the repository, where one is missing.
BLOCKED = {
    "OT-10": "the complete-training frontier is read from train_comparison.py's runs over one "
             "prepared recipe: name its data directory (prepare.json and runs/) with --frontier",
    "OT-15": "RuntimeChannel has its direct adapter alone: no shared-memory queue, Linux process or "
             "native adapter exists to decompose against it (the driver ladder's M2, M3 and M8)",
}  # fmt: skip


def unrun(experiment: oc.Experiment, caps: set[str]) -> dict:
    """The record of an experiment this run does not measure: unavailable where the host lacks a
    capability it needs (named), else blocked by what the repository lacks -- the prerequisite
    `BLOCKED` names, or the experiment's runner, not yet in this tree. A host with every capability
    an experiment needs records it blocked until its runner lands; it never fails the run."""
    lacking = [c for c in experiment.needs if c not in caps]
    if lacking:
        return {"experiment": experiment.id, "status": "unavailable", "missing": lacking,
                "reason": "this host lacks " + ", ".join(oc.CAPABILITIES[c] for c in lacking)}  # fmt: skip
    reason = BLOCKED.get(experiment.id) or (
        f"no runner for {experiment.id} in this tree yet: its {', '.join(experiment.routes)} "
        f"comparison is a later OT slice"
    )
    return {"experiment": experiment.id, "status": "blocked", "reason": reason}


def _rows(path) -> dict:
    """The GEM+ rows the gates read, from the report `gemplus_baseline.py --compare --json PATH`
    writes: each row's key and the value it measured (a row not measured is absent). A file that is
    not that report -- empty, malformed, a key twice, no rows -- is refused (ValueError)."""
    try:
        doc = oc.load_json(path)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: the rows file is not JSON ({exc.msg})") from None
    if not isinstance(doc, dict) or not isinstance(doc.get("rows"), list) or not doc["rows"]:
        raise ValueError(f"{path}: a rows file is gemplus_baseline.py's report, its rows a list")
    return {r["key"]: r["measured"] for r in doc["rows"]
            if isinstance(r, dict) and type(r.get("key")) is str
            and type(r.get("measured")) in (int, float)}  # fmt: skip


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", type=Path)
    p.add_argument("--rows", type=Path, help="gemplus_baseline.py --json output (the gates' rows)")
    p.add_argument("--rounds", type=int, default=24)
    p.add_argument("--smoke", action="store_true", help="tiny shapes and four rounds")
    p.add_argument("--triton", type=Path, help="a Triton-CPU pilot session (OT-05's T-DSL rows)")
    p.add_argument(
        "--frontier",
        type=Path,
        help="train_comparison.py's prepared data with its runs (OT-10's routes)",
    )
    a = p.parse_args(argv)
    # every file the run reads, read before any experiment runs: a malformed one is a refusal now,
    # not a traceback after the run's hour of measurement
    try:
        rows = _rows(a.rows) if a.rows else {}
        triton = _triton_evidence(a.triton) if a.triton else None
        frontier = ot10(a.frontier) if a.frontier else None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ot_run: {exc}", file=sys.stderr)
        return 2
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")  # before PyTorch loads: no spinning team
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
                                            OT01_SMOKE if a.smoke else OT01_SHAPES,
                                            with_torch="torch" in caps)  # fmt: skip
        if packed is not None and traced is not None:
            runners["OT-03"] = lambda: ot03(packed, traced, a.smoke)
        case = timed_case(a.smoke)
        if packed is not None:
            runners["OT-04"] = lambda: ot04(packed, rounds, case)
        runners["OT-06"] = ot06
        if plain is not None and traced is not None:
            runners["OT-14"] = lambda: ot14(plain, traced, rounds, case)
        threads = os.cpu_count() or 1
        if "torch" in caps:
            runners["OT-02"] = lambda: ot02(threads, a.smoke)
            runners["OT-11"] = lambda: ot11(threads, a.smoke)
            if plain is not None:
                runners["OT-05"] = lambda: ot05(plain, threads, a.smoke, rounds, triton)
            if frontier is not None:
                runners["OT-10"] = lambda: frontier
        versions, ran_at = _versions(caps), _ran_at(threads, frontier)
        for experiment in oc.EXPERIMENTS:
            run = runners.get(experiment.id)
            record = _stamp(run() if run else unrun(experiment, caps), ran_at, versions)
            records.append(oc.validate_record(record))
            print(f"{experiment.id} {record['status']}", flush=True)
    gates = oc.evaluate_gates(records, rows, capabilities=caps)
    out = {"proposal": oc.PROPOSAL, "capabilities": sorted(caps), "target": _target(threads),
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
