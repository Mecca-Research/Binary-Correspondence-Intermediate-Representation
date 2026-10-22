#!/usr/bin/env python3
"""OT/BG (2026-10-22): the PyTorch comparator routes of the backend-optimality experiments.

PR #821's comparator routes -- P-EAGER (eager tensors and autograd), P-FX (Dynamo's capture with
an eager backend), P-AOT (AOTAutograd's capture with an eager backend), P-IND (torch.compile with
Inductor) and P-TUNED (Inductor's budgeted autotuning) -- run beside BCIR's own routes on the
experiments a host with PyTorch can run (`tools/perf/ot_run.py` calls these where PyTorch imports;
nothing in BCIR's core imports this module):

  OT-01  the first common block's numerics on P-EAGER from the same inputs as B-DIRECT's, held to
         the same binary32 bounds (ot_run.gemm_check / elementwise_check)
  OT-02  capture cost: the 9.84M comparison decoder's training step (its loss and backward)
         captured by P-FX, P-AOT and P-IND, each in a fresh process over an empty compiler cache --
         the first call's time, the warm call's, the graphs and graph breaks -- the module as
         written (its input checks are data-dependent branches) and its loss as one graph; beside
         B-PLAN's planning of the same step (the step program, its words, the TrainingPack, the
         StreamPack and ExecutionPlanV1 lowering) and the C library's build, BCIR's codegen
  OT-05  the block portfolio: the common block's forward and forward + backward at OT-01's shapes
         on B-DIRECT under three GEMM contracts (v1, the fused v2, the blocked v3) and on P-EAGER and
         P-IND, at one thread and at the host's, in rotated rounds beside an A/A control
  OT-11  the cache lifecycle: P-IND cold (a fresh process, an empty cache), warm (the same process,
         the same shape), persistent (a fresh process over the first's cache) and across a shape
         change (static and dynamic), beside B-PLAN (one library for every shape, a plan a shape)

Every route names its arithmetic contract: BCIR's GEMM contracts are fixed summation orders, one
result on every host and thread count; PyTorch's products are its BLAS's -- an order the library
chooses, with FMA -- so same dtype is not the same contract, and the records say which ran.
Timings are this host's, each read against the A/A control's interval; nothing here is a
threshold. A route that fails is recorded with its failure, never dropped.
"""

from __future__ import annotations

import ctypes as ct
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The 9.84M comparison decoder of the PX and OT records (OT-02, OT-11): vocabulary 260, width 384,
#: 8 heads over 4 KV heads, 6 layers, FF 1024, tied embeddings; 8 sequences of 128 tokens.
DECODER = (260, 384, 8, 6, 1024, 4)
DECODER_SHAPE = (8, 128)
#: A smoke run's decoder: the trace corpus's bench decoder (native_decoder_fixtures.BENCH).
SMOKE_DECODER = (256, 64, 4, 2, 192, 2)
SMOKE_SHAPE = (2, 32)
#: The GEMM contracts B-DIRECT runs in OT-05, each its entry in the plain library.
CONTRACT_ENTRIES = {"v1": "bcir_tensor_mm_pool", "fused": "bcir_tensor_mm_fused_pool",
                    "kblock": "bcir_tensor_mm_kblock_pool"}  # fmt: skip
TORCH_CONTRACT = "the BLAS PyTorch dispatches to (its own summation order, FMA)"


def torch_versions() -> dict:
    import torch

    out = {"torch": str(torch.__version__),
           "torch_cpu_capability": str(torch.backends.cpu.get_cpu_capability()),
           "torch_threads": str(torch.get_num_threads())}  # fmt: skip
    # the BLAS its build names (BLAS_INFO), with the library's own line where the build shows one
    show = torch.__config__.show().splitlines()
    blas = next((part.split("=", 1)[1] for ln in show for part in ln.replace(",", " ").split()
                 if part.startswith("BLAS_INFO=")), None)  # fmt: skip
    library = next((ln.strip(" -") for ln in show if "Math Kernel Library" in ln
                    or "OpenBLAS" in ln), None)  # fmt: skip
    if blas:
        out["torch_blas"] = blas + (f" ({library})" if library else "")
    return out


def interval(ratios: list[float]) -> dict:
    """The geometric mean of paired ratios and its 95% Student-t interval (ot_run._interval's)."""
    from tools.perf.ot_run import _interval

    mean, ci = _interval(ratios)
    return {"geometric_mean": mean, "ci95": ci, "rounds": len(ratios)}


def rotated(arms: dict, rounds: int) -> dict:
    """Paired, rotated rounds of each arm's call: {arm: [ms, ...]} (each arm warmed once)."""
    names = tuple(arms)
    out = {a: [] for a in names}
    for fn in arms.values():
        fn()
    for r in range(rounds):
        for j in range(len(names)):
            arm = names[(r + j) % len(names)]
            t0 = time.perf_counter()
            arms[arm]()
            out[arm].append(1000 * (time.perf_counter() - t0))
    return out


# --- OT-01: P-EAGER's numerics --------------------------------------------------------------------


def p_eager_block(shapes) -> tuple[list, list, float]:
    """The first common block forward and backward on PyTorch eager (autograd), from the inputs
    ot_run.ot01 draws (the same generator, the same order), every tensor held to B-DIRECT's bounds:
    (results, raw, worst error over its bound)."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    from tools.perf.ot_run import elementwise_check, gemm_check

    results, raw, worst = [], [], 0.0
    rng = np.random.default_rng(821)
    for m, k, n in shapes:
        x = rng.standard_normal((m, k), dtype=np.float32)
        wg = (rng.standard_normal((k, n), dtype=np.float32) / np.sqrt(k)).astype(np.float32)
        wu = (rng.standard_normal((k, n), dtype=np.float32) / np.sqrt(k)).astype(np.float32)
        dy = rng.standard_normal((m, n), dtype=np.float32)
        X, Wg, Wu = (torch.tensor(a, requires_grad=True) for a in (x, wg, wu))
        U, V = X @ Wg, X @ Wu
        U.retain_grad()
        V.retain_grad()
        Y = F.silu(U) * V
        Y.backward(torch.from_numpy(dy))
        u, v, y = (t.detach().numpy() for t in (U, V, Y))
        du, dv = U.grad.numpy(), V.grad.numpy()
        dx, dwg, dwu = X.grad.numpy(), Wg.grad.numpy(), Wu.grad.numpy()
        X64, Wg64, Wu64 = x.astype(np.float64), wg.astype(np.float64), wu.astype(np.float64)
        U64, V64, DY = u.astype(np.float64), v.astype(np.float64), dy.astype(np.float64)
        sig = 1.0 / (1.0 + np.exp(-U64))
        silu = U64 * sig
        y_exact = silu * V64
        dg_exact = DY * V64 * (sig * (1.0 + U64 * (1.0 - sig)))
        du_exact = DY * silu
        checks = [
            gemm_check("U", u, X64, Wg64), gemm_check("V", v, X64, Wu64),
            elementwise_check("Y", y, y_exact, np.abs(silu) * np.abs(V64) + np.abs(y_exact)),
            elementwise_check("dU", du, dg_exact, np.abs(DY) * np.abs(V64)
                              * (np.abs(sig) + np.abs(U64) * sig * (1 - sig)) + np.abs(dg_exact)),
            elementwise_check("dV", dv, du_exact, np.abs(DY) * np.abs(silu) + np.abs(du_exact)),
            gemm_check("dX", dx, np.concatenate([du, dv], 1).astype(np.float64),
                       np.concatenate([Wg64.T, Wu64.T], 0)),
            gemm_check("dWg", dwg, X64.T, du.astype(np.float64)),
            gemm_check("dWu", dwu, X64.T, dv.astype(np.float64)),
        ]  # fmt: skip
        case_worst = max(c["max_error_over_bound"] for c in checks)
        worst = max(worst, case_worst)
        results.append({"route": "P-EAGER", "shape": {"m": m, "k": k, "n": n},
                        "within_bound": case_worst <= 1.0, "max_error_over_bound": case_worst})  # fmt: skip
        raw.append({"route": "P-EAGER", "shape": {"m": m, "k": k, "n": n}, "checks": checks})
    return results, raw, worst


# --- OT-05: the block portfolio -------------------------------------------------------------------


def _native(lib):
    fp = ct.POINTER(ct.c_float)
    sz = ct.c_size_t
    for name in CONTRACT_ENTRIES.values():
        f = getattr(lib, name)
        f.argtypes = [
            ct.c_void_p,
            ct.c_int,
            ct.c_int,
            sz,
            sz,
            sz,
            ct.c_float,
            fp,
            fp,
            ct.c_float,
            fp,
        ]
        f.restype = None
    lib.bcir_tensor_silu_mul_pool.argtypes = [ct.c_void_p, sz, fp, fp, fp]
    lib.bcir_tensor_silu_mul_pool.restype = None
    lib.bcir_tensor_swiglu_backward_pool.argtypes = [ct.c_void_p, sz, fp, fp, fp, fp, fp]
    lib.bcir_tensor_swiglu_backward_pool.restype = None
    lib.bcir_tensor_pool_size.argtypes = [ct.c_uint]
    lib.bcir_tensor_pool_size.restype = sz
    lib.bcir_tensor_pool_init.argtypes = [ct.c_void_p, sz, ct.c_uint]
    lib.bcir_tensor_pool_init.restype = ct.c_void_p
    lib.bcir_tensor_pool_stop.argtypes = [ct.c_void_p]
    lib.bcir_tensor_pool_stop.restype = None
    return lib


class Pool:
    """A caller-owned tensor pool of `threads` threads (None: the serial path, one thread)."""

    def __init__(self, lib, threads: int):
        self.lib, self.handle, self.memory = lib, None, None
        if threads > 1:
            size = lib.bcir_tensor_pool_size(threads)
            self.memory = ct.create_string_buffer(size + 64)
            self.handle = lib.bcir_tensor_pool_init((ct.addressof(self.memory) + 63) & ~63, size,
                                                    threads)  # fmt: skip
            if not self.handle:
                raise RuntimeError(f"the tensor pool refused {threads} threads")

    def close(self):
        if self.handle:
            self.lib.bcir_tensor_pool_stop(self.handle)
            self.handle = None


def _block_native(lib, pool, mm, x, wg, wu, dy, out):
    """B-DIRECT's block under one GEMM entry `mm`: (forward, forward + backward) closures."""
    fp = ct.POINTER(ct.c_float)
    m, k = x.shape
    n = wg.shape[1]
    p = {name: a.ctypes.data_as(fp) for name, a in out.items()}
    px, pg, pu, pdy = (a.ctypes.data_as(fp) for a in (x, wg, wu, dy))
    h = pool.handle

    def forward():
        mm(h, 0, 0, m, n, k, 1.0, px, pg, 0.0, p["u"])
        mm(h, 0, 0, m, n, k, 1.0, px, pu, 0.0, p["v"])
        lib.bcir_tensor_silu_mul_pool(h, m * n, p["u"], p["v"], p["y"])

    def both():
        forward()
        lib.bcir_tensor_swiglu_backward_pool(h, m * n, p["u"], p["v"], pdy, p["du"], p["dv"])
        mm(h, 0, 1, m, k, n, 1.0, p["du"], pg, 0.0, p["dx"])
        mm(h, 0, 1, m, k, n, 1.0, p["dv"], pu, 1.0, p["dx"])
        mm(h, 1, 0, k, n, m, 1.0, px, p["du"], 0.0, p["dwg"])
        mm(h, 1, 0, k, n, m, 1.0, px, p["dv"], 0.0, p["dwu"])

    return forward, both


def _block_torch(x, wg, wu, dy, compiled: bool):
    """P-EAGER's (or P-IND's) block: (forward, forward + backward, compile seconds)."""
    import torch
    import torch.nn.functional as F

    def block(a, g, u):
        return F.silu(a @ g) * (a @ u)

    X, Wg, Wu, DY = (torch.from_numpy(a.copy()) for a in (x, wg, wu, dy))
    Xg, Wgg, Wug = (t.clone().requires_grad_(True) for t in (X, Wg, Wu))
    if compiled:
        # a fresh compile per shape: one code object recompiled past Dynamo's cache limit would
        # run eager silently, which this route must never record as Inductor's
        torch._dynamo.reset()
    fn = torch.compile(block, dynamic=False) if compiled else block
    t0 = time.perf_counter()
    if compiled:
        with torch.no_grad():
            fn(X, Wg, Wu)
        fn(Xg, Wgg, Wug).backward(DY)
    seconds = time.perf_counter() - t0

    def forward():
        with torch.no_grad():
            fn(X, Wg, Wu)

    def both():
        for t in (Xg, Wgg, Wug):
            t.grad = None
        fn(Xg, Wgg, Wug).backward(DY)

    return forward, both, seconds


def portfolio(lib_path, shapes, threads_list, rounds: int) -> dict:
    """OT-05: the block on B-DIRECT under each GEMM contract (and an A/A control under the blocked
    v3, the trainer's fast contract) against P-EAGER and P-IND, per thread count and shape."""
    import numpy as np
    import torch

    if rounds < 4:
        raise ValueError("a portfolio needs at least four rounds for its intervals")
    lib = _native(ct.CDLL(str(lib_path)))
    rows, failures = [], []
    for threads in threads_list:
        torch.set_num_threads(threads)
        pools = [Pool(lib, threads)]  # one pool, the arms in turn (a pool has one caller at a time)
        try:
            for m, k, n in shapes:
                rng = np.random.default_rng(1005 + m + k + n)
                x = rng.standard_normal((m, k), dtype=np.float32)
                wg = (rng.standard_normal((k, n), dtype=np.float32) / np.sqrt(k)).astype(np.float32)
                wu = (rng.standard_normal((k, n), dtype=np.float32) / np.sqrt(k)).astype(np.float32)
                dy = rng.standard_normal((m, n), dtype=np.float32)
                arms_f, arms_b, compile_s = {}, {}, {}
                for name, entry in [
                    *CONTRACT_ENTRIES.items(),
                    ("kblock-AA", CONTRACT_ENTRIES["kblock"]),
                ]:
                    out = {a: np.zeros(s, np.float32) for a, s in
                           (("u", (m, n)), ("v", (m, n)), ("y", (m, n)), ("du", (m, n)),
                            ("dv", (m, n)), ("dx", (m, k)), ("dwg", (k, n)), ("dwu", (k, n)))}  # fmt: skip
                    f, b = _block_native(lib, pools[0], getattr(lib, entry), x, wg, wu, dy, out)
                    arms_f[f"B-DIRECT/{name}"], arms_b[f"B-DIRECT/{name}"] = f, b
                for route, compiled in (("P-EAGER", False), ("P-IND", True)):
                    try:
                        f, b, s = _block_torch(x, wg, wu, dy, compiled)
                    except Exception as exc:  # a route that fails is recorded, never dropped
                        failures.append(f"{route} at {m}x{k}x{n}, {threads} thread(s): "
                                        f"{type(exc).__name__}: {exc}")  # fmt: skip
                        continue
                    arms_f[route], arms_b[route] = f, b
                    if compiled:
                        compile_s[route] = s
                row = {"shape": {"m": m, "k": k, "n": n}, "threads": threads,
                       "gemm_flop": {"forward": 4 * m * k * n, "forward_backward": 12 * m * k * n},
                       "compile_seconds": compile_s}  # fmt: skip
                for phase, arms in (("forward", arms_f), ("forward_backward", arms_b)):
                    ms = rotated(arms, rounds)
                    ref = ms["B-DIRECT/kblock"]
                    row[phase] = {
                        "ms": {a: statistics.median(v) for a, v in ms.items()},
                        "kblock_over": {a: interval([r / o for r, o in zip(ref, v)])
                                        for a, v in ms.items() if a != "B-DIRECT/kblock"},
                        "raw_ms": {a: [round(t, 4) for t in v] for a, v in ms.items()},
                    }  # fmt: skip
                rows.append(row)
        finally:
            for p in pools:
                p.close()
    return {"rows": rows, "failures": failures}


# --- OT-02 and OT-11: capture and the compiler's cache, each measurement its own process ----------


def _decoder_spec(dims):
    from bcir.frontends.models.decode import DecoderSpec

    vocab, width, heads, layers, ff, kv = dims
    return DecoderSpec(vocab, width, heads, layers, ff, activation="silu_gate", n_kv_heads=kv,
                       tied_embeddings=True)  # fmt: skip


def child_capture(backend: str, form: str, dims, shape, threads: int, shapes2=None) -> dict:
    """In a fresh process: the decoder's training step through torch.compile(backend) -- `form`
    'module' compiles HostedLlama as written, 'loss' its loss as one graph (fullgraph) -- the first
    call's seconds, the warm call's, the graphs and graph breaks Dynamo counted; with `shapes2`, a
    second shape after the first (the recompile it costs)."""
    import torch
    import torch.nn.functional as F
    from torch._dynamo.utils import counters

    from bcir.hosted.models.model import HostedLlama, _rope_cache

    torch.set_num_threads(threads)
    torch.manual_seed(0)
    spec = _decoder_spec(dims)
    model = HostedLlama(spec)
    # 'inductor' compiles as torch.compile defaults (dynamic=None: static, then automatic dynamic
    # on a recompile); ':static' and ':dynamic' name dynamic=False and dynamic=True (OT-11)
    backend, _, mode = backend.partition(":")
    dynamic = {"": None, "static": False, "dynamic": True}[mode]

    def loss_of(x, y):
        values = model.model.embed_tokens(x)
        cos, sin = _rope_cache(x.shape[1], spec.d_k, spec.rope_base, values.device, values.dtype)
        for layer in model.model.layers:
            values = layer(values, cos, sin)
        logits = F.linear(model.model.norm(values), model.lm_head.weight)
        return F.cross_entropy(logits.reshape(-1, spec.vocab_size), y.reshape(-1))

    if form == "module":
        fn = torch.compile(lambda x, y: model(x, y)[1], backend=backend, dynamic=dynamic)
    else:
        fn = torch.compile(loss_of, backend=backend, fullgraph=True, dynamic=dynamic)

    def step(b, t):
        g = torch.Generator().manual_seed(b * 1000 + t)
        x = torch.randint(0, spec.vocab_size, (b, t), generator=g)
        y = torch.randint(0, spec.vocab_size, (b, t), generator=g)
        t0 = time.perf_counter()
        fn(x, y).backward()
        return time.perf_counter() - t0

    out = {"first_s": step(*shape)}
    warm = [step(*shape) for _ in range(3)]
    out["warm_s"] = statistics.median(warm)
    out["frames"] = int(counters["stats"].get("unique_graphs", 0))
    out["graph_breaks"] = int(sum(counters["graph_break"].values()))
    out["break_reasons"] = sorted(counters["graph_break"])[:8]
    if shapes2:
        out["second_shape_first_s"] = step(*shapes2)
        out["second_shape_warm_s"] = statistics.median([step(*shapes2) for _ in range(2)])
        out["frames_after_second_shape"] = int(counters["stats"].get("unique_graphs", 0))
    return out


def _spawn(args: list[str], env: dict, timeout: int) -> dict:
    """One child measurement (`python -m tools.perf.ot_torch --child ...`): its JSON, or the
    failure as a record (a timeout, a non-zero exit, an unreadable line)."""
    try:
        r = subprocess.run([sys.executable, "-m", "tools.perf.ot_torch", "--child", *args],
                           cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=timeout)  # fmt: skip
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"failed": f"{type(exc).__name__}: {exc}"}
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    if r.returncode or not lines:
        return {"failed": f"exit {r.returncode}: {(r.stderr or r.stdout)[-400:]}"}
    try:
        return json.loads(lines[-1])
    except ValueError as exc:
        return {"failed": f"unreadable child output: {exc}"}


def _cache_env(cache: str | None) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    if cache is not None:
        env["TORCHINDUCTOR_CACHE_DIR"] = cache
        env["TRITON_CACHE_DIR"] = os.path.join(cache, "triton")
    return env


def b_plan_cost(dims, shape, threads: int, build: bool) -> dict:
    """B-PLAN on the same step: the step program, its words, the TrainingPack and the K_BCIR ->
    StreamPack + ExecutionPlanV1 lowering, each timed (ms); and the C library's build (s, the
    decoder's one shape-agnostic library: BCIR's codegen) where `build`."""
    from bcir.hosted.models import native_launch as nl
    from bcir.hosted.models import native_pack as npk
    from bcir.hosted.models.native import _CONTRACTS

    spec = _decoder_spec(dims)
    b, t = shape
    contract = _CONTRACTS["kblock+exp+reduce+attn"]
    out = {}
    t0 = time.perf_counter()
    program = nl.step_program(spec, b, t, contract=contract, threads=threads)
    out["step_program_ms"] = 1000 * (time.perf_counter() - t0)
    t0 = time.perf_counter()
    words = nl.program_words(program)
    out["program_words_ms"] = 1000 * (time.perf_counter() - t0)
    out["program_launches"] = len(program)
    out["program_words"] = len(words)
    t0 = time.perf_counter()
    tp = npk.program_pack(spec, b, t, program)
    out["training_pack_ms"] = 1000 * (time.perf_counter() - t0)
    t0 = time.perf_counter()
    pack, plan, *_ = npk.pack_artifacts(tp)
    out["lower_streampack_plan_ms"] = 1000 * (time.perf_counter() - t0)
    out["streampack_bytes"], out["plan_bytes"] = len(pack), len(plan)
    if build:
        from bcir.tests import native_decoder_fixtures as nf

        with tempfile.TemporaryDirectory(prefix="bcir-ot-build-") as tmp:
            t0 = time.perf_counter()
            path = nf.build_library(tmp)
            out["library_build_s"] = time.perf_counter() - t0 if path is not None else None
    return out


def capture_cost(threads: int, smoke: bool) -> dict:
    """OT-02: P-FX / P-AOT / P-IND over the decoder's step, each cold in its own process, the
    module as written and its loss as one graph; B-PLAN's planning of the same step."""
    dims, shape = (SMOKE_DECODER, SMOKE_SHAPE) if smoke else (DECODER, DECODER_SHAPE)
    routes = {"P-FX": "eager", "P-AOT": "aot_eager", "P-IND": "inductor"}
    out = {"decoder": dict(zip(("vocab", "width", "heads", "layers", "ff", "kv_heads"), dims)),
           "batch": shape[0], "time": shape[1], "threads": threads, "routes": {}}  # fmt: skip
    for route, backend in routes.items():
        for form in ("module", "loss"):
            with tempfile.TemporaryDirectory(prefix="bcir-ot-cache-") as cache:
                out["routes"][f"{route}/{form}"] = _spawn(
                    ["capture", backend, form, json.dumps(dims), json.dumps(shape), str(threads)],
                    _cache_env(cache), 1800)  # fmt: skip
    out["routes"]["B-PLAN"] = b_plan_cost(dims, shape, threads, build=True)
    return out


def lifecycle(threads: int, smoke: bool) -> dict:
    """OT-11: P-IND's cache over the decoder's step -- cold, warm, persistent, shape change."""
    dims, shape = (SMOKE_DECODER, SMOKE_SHAPE) if smoke else (DECODER, DECODER_SHAPE)
    second = (shape[0], shape[1] // 2)
    out = {"decoder": dict(zip(("vocab", "width", "heads", "layers", "ff", "kv_heads"), dims)),
           "batch": shape[0], "time": shape[1], "second_time": second[1], "threads": threads}  # fmt: skip
    with tempfile.TemporaryDirectory(prefix="bcir-ot-cache-") as cache:
        args = ["capture", "inductor:static", "loss", json.dumps(dims), json.dumps(shape),
                str(threads), json.dumps(second)]  # fmt: skip
        out["cold_static"] = _spawn(args, _cache_env(cache), 1800)
        out["persistent_static"] = _spawn(args, _cache_env(cache), 1800)
    with tempfile.TemporaryDirectory(prefix="bcir-ot-cache-") as cache:
        args[1] = "inductor:dynamic"
        out["cold_dynamic"] = _spawn(args, _cache_env(cache), 1800)
    plan = [b_plan_cost(dims, s, threads, build=False) for s in (shape, second)]
    out["B-PLAN"] = {"plans_ms": [p["step_program_ms"] + p["program_words_ms"] for p in plan],
                     "library": "one C library for every shape: a new shape costs its plan alone"}  # fmt: skip
    return out


def _child(argv: list[str]) -> int:
    kind = argv[0]
    if kind == "capture":
        backend, form, dims, shape, threads = (
            argv[1],
            argv[2],
            json.loads(argv[3]),
            json.loads(argv[4]),
            int(argv[5]),
        )
        second = json.loads(argv[6]) if len(argv) > 6 else None
        print(json.dumps(child_capture(backend, form, dims, shape, threads, second)), flush=True)
        return 0
    raise SystemExit(f"no child measurement {kind!r}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        sys.exit(_child(sys.argv[2:]))
    raise SystemExit("ot_torch.py is called by tools/perf/ot_run.py")
