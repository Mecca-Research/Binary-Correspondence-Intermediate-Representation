#!/usr/bin/env python3
"""Optional compiled CPU pilot; not a GPU or model-training benchmark.

Requires NumPy and the official experimental triton-cpu v3.8.0 release wheel.
No PyTorch, interpreter, external BLAS, or runtime edits. See the comparative
research document for installation, numerical contracts, and interpretation.
Run on an otherwise idle host; samples deliberately include host launch cost.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

os.environ.setdefault("TRITON_DEFAULT_BACKEND", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
# Park the inactive OpenMP team while measuring the other implementation.
# Default spinning otherwise makes two independent pools contend on two CPUs.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("GOMP_SPINCOUNT", "0")

import numpy as np
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[2]
CPU_SOURCE = "5a4885bf9b0b349a525b490a1df25f449f12d4e9"
WHEEL_SHA256 = "693f657501e6d791e889d5fd880d3b84a9bf4d8f3087dbb685534bb8533361fa"


class Span:
    """Typed, zero-copy NumPy buffer for Triton's pointer interface."""

    dtype = tl.float32

    def __init__(self, array):
        self.array = array

    def data_ptr(self):
        return self.array.ctypes.data


@triton.jit
def residual(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0)
    y = tl.load(Y + i, i < N, 0)
    tl.store(Y + i, x + y, i < N)


@triton.jit
def swiglu(X, U, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0)
    u = tl.load(U + i, i < N, 0)
    s = x / (1.0 + tl.exp(-x))
    tl.store(Y + i, s * u, i < N)


@triton.jit
def rmsnorm(X, G, Y, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    x = tl.load(X + row * WIDTH + j, j < WIDTH, 0)
    g = tl.load(G + j, j < WIDTH, 0)
    mean = tl.sum(x * x, 0) / WIDTH
    inv = 1.0 / tl.sqrt(mean + 1.0e-5)
    tl.store(Y + row * WIDTH + j, (x * inv) * g, j < WIDTH)


@triton.jit
def gemm(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0.0, tl.float32)
    for tile in range(triton.cdiv(K, BK)):
        k = tile * BK + kk
        a = tl.load(A + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), 0)
        b = tl.load(B + k[:, None] * N + n[None, :], (k[:, None] < K) & (n[None, :] < N), 0)
        acc = tl.dot(a, b, acc, input_precision="ieee", out_dtype=tl.float32)
    tl.store(C + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def command(args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def build_native(out, cc):
    target = out / "libbcir_tensor.so"
    args = [
        cc,
        "-std=c11",
        "-O3",
        "-fPIC",
        "-shared",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-pthread",
        "runtime/c/bcir_tensor.c",
        "-lm",
        "-o",
        str(target),
    ]
    start = time.perf_counter()
    subprocess.run(args, cwd=ROOT, check=True)
    lib = ct.CDLL(str(target))
    void, size, integer, fp = ct.c_void_p, ct.c_size_t, ct.c_int, ct.c_float
    signatures = {
        "add_pool": [void, size, void, void],
        "silu_mul_pool": [void, size, void, void, void],
        "silu_mul_v2_pool": [void, size, void, void, void],
        "rms_pool": [void, size, size, fp, void, void, void],
        "mm_pool": [void, integer, integer, size, size, size, fp, void, void, fp, void],
        "mm_fused_pool": [void, integer, integer, size, size, size, fp, void, void, fp, void],
        "mm_kblock_pool": [void, integer, integer, size, size, size, fp, void, void, fp, void],
    }
    for name, signature in signatures.items():
        function = getattr(lib, "bcir_tensor_" + name)
        function.argtypes, function.restype = signature, None
    lib.bcir_tensor_pool_size.argtypes = [ct.c_uint]
    lib.bcir_tensor_pool_size.restype = size
    lib.bcir_tensor_pool_init.argtypes = [void, size, ct.c_uint]
    lib.bcir_tensor_pool_init.restype = void
    lib.bcir_tensor_pool_stop.argtypes = [void]
    lib.bcir_tensor_pool_stop.restype = None
    return lib, {
        "command": args[:-1] + ["<output>/libbcir_tensor.so"],
        "elapsed_s": time.perf_counter() - start,
        "binary_sha256": digest(target),
        "compiler": command([cc, "--version"]).splitlines()[0],
    }


def elapsed(function, repeats):
    start = time.perf_counter_ns()
    for _ in range(repeats):
        function()
    return (time.perf_counter_ns() - start) / repeats


def calibrate(function, target_ns):
    repeats = 1
    while repeats < 4096:
        per_call = elapsed(function, repeats)
        if per_call * repeats >= target_ns:
            return repeats
        repeats = min(4096, max(repeats + 1, math.ceil(target_ns / max(per_call, 1))))
    return repeats


def errors(array, reference, tolerance):
    delta = np.abs(array.astype(np.float64) - reference)
    max_abs = float(np.max(delta))
    scale = max(float(np.max(np.abs(reference))), 1.0e-30)
    bound = tolerance * (1 + np.abs(reference))
    return {
        "pass": bool(np.all(np.isfinite(array)) and np.all(delta <= bound)),
        "max_abs": max_abs,
        "max_abs_over_max_reference": max_abs / scale,
        "criterion": f"abs(error) <= {tolerance} * (1 + abs(float64_reference))",
    }


def make_case(kind, shape, lib, pool, rng):
    def array(dim, low=-1, high=1):
        return rng.uniform(low, high, size=dim).astype(np.float32)

    outputs, native = {}, {}
    if kind in ("add", "swiglu"):
        x, u = array(shape, -4, 4), array(shape)
        reference = (
            x.astype(np.float64) + u
            if kind == "add"
            else (x.astype(np.float64) / (1 + np.exp(-x.astype(np.float64)))) * u
        )
        names = ("bcir", "triton") if kind == "add" else ("bcir_libm", "bcir_v2", "triton")
        for name in names:
            outputs[name] = np.empty_like(x)
        for name in names[:-1]:
            symbol = (
                "add_pool"
                if kind == "add"
                else ("silu_mul_v2_pool" if name == "bcir_v2" else "silu_mul_pool")
            )
            args = [pool, x.size, x.ctypes.data]
            if kind != "add":
                args.append(u.ctypes.data)
            args.append(outputs[name].ctypes.data)
            native[name] = lambda fn=getattr(lib, "bcir_tensor_" + symbol), args=args: fn(*args)

        def reset():
            if kind == "add":
                for output in outputs.values():
                    np.copyto(output, u)

        factory, choices = residual if kind == "add" else swiglu, [(256,), (1024,), (4096,)]

        def triton_spec(choice):
            (block,) = choice
            args = [Span(x)]
            if kind != "add":
                args.append(Span(u))
            args += [Span(outputs["triton"]), x.size, block]
            return factory, args, (triton.cdiv(x.size, block), 1, 1)

        resident_bytes = x.nbytes + outputs["triton"].nbytes + (u.nbytes if kind != "add" else 0)
        tolerance = 1e-7 if kind == "add" else 3e-6
    elif kind == "rms":
        rows, width = shape
        x, g = array((rows, width), -4, 4), array(width, 0.5, 1.5)
        xd = x.astype(np.float64)
        reference = (
            xd / np.sqrt(np.mean(xd * xd, axis=1, keepdims=True) + float(np.float32(1e-5)))
        ) * g
        outputs = {name: np.empty_like(x) for name in ("bcir", "triton")}
        args = [
            pool,
            rows,
            width,
            np.float32(1e-5),
            x.ctypes.data,
            g.ctypes.data,
            outputs["bcir"].ctypes.data,
        ]
        native["bcir"] = lambda: lib.bcir_tensor_rms_pool(*args)

        def reset():
            pass

        choices = [(triton.next_power_of_2(width),)]

        def triton_spec(choice):
            return (
                rmsnorm,
                [Span(x), Span(g), Span(outputs["triton"]), width, choice[0]],
                (rows, 1, 1),
            )

        resident_bytes = x.nbytes + g.nbytes + outputs["triton"].nbytes
        tolerance = 1e-5
    else:
        m, n, k = shape
        a, b = array((m, k)), array((k, n))
        reference = a.astype(np.float64) @ b.astype(np.float64)
        outputs = {
            name: np.empty((m, n), dtype=np.float32)
            for name in ("bcir_fused", "bcir_kblock", "triton")
        }
        # Two existing FP32 contracts; no BF16, TF32, or external BLAS.
        for name, symbol in (("bcir_fused", "mm_fused_pool"), ("bcir_kblock", "mm_kblock_pool")):
            args = [
                pool,
                0,
                0,
                m,
                n,
                k,
                1.0,
                a.ctypes.data,
                b.ctypes.data,
                0.0,
                outputs[name].ctypes.data,
            ]
            native[name] = lambda fn=getattr(lib, "bcir_tensor_" + symbol), args=args: fn(*args)

        def reset():
            pass

        choices = [(16, 32, 32), (32, 32, 32), (32, 64, 32)]

        def triton_spec(choice):
            bm, bn, bk = choice
            return (
                gemm,
                [Span(a), Span(b), Span(outputs["triton"]), m, n, k, bm, bn, bk],
                (triton.cdiv(m, bm), triton.cdiv(n, bn), 1),
            )

        resident_bytes = a.nbytes + b.nbytes + outputs["triton"].nbytes
        tolerance = 1e-5
    return native, outputs, reference, reset, choices, triton_spec, resident_bytes, tolerance


def ratio_interval(bcir, triton_ns, rng):
    # Paired rounds; descriptive only on a shared virtual host.
    indices = rng.integers(0, len(bcir), (2000, len(bcir)))
    ratios = np.median(np.asarray(bcir)[indices], axis=1) / np.median(
        np.asarray(triton_ns)[indices], axis=1
    )
    return {
        "bcir_over_triton": statistics.median(bcir) / statistics.median(triton_ns),
        "paired_bootstrap_95pct": np.quantile(ratios, [0.025, 0.975]).tolist(),
    }


def run_case(kind, shape, threads, lib, rng, rounds, target_ns, artifact_dir, data_seed):
    storage = ct.create_string_buffer(lib.bcir_tensor_pool_size(threads))
    pool = lib.bcir_tensor_pool_init(storage, len(storage), threads)
    if not pool:
        raise RuntimeError("BCIR pool init refused")
    try:
        native, outputs, reference, reset, choices, spec, resident, tol = make_case(
            kind, shape, lib, pool, np.random.default_rng(data_seed)
        )
        candidates, runs = [], []
        for choice in choices:
            fn, args, grid = spec(choice)
            reset()
            start = time.perf_counter()
            kernel = fn[grid](
                *args,
                num_cpu_threads=threads,
                enable_fast_math=False,
                enable_fp_fusion=(kind == "gemm"),
                ukernels=None,
            )
            cold = time.perf_counter() - start
            # Bind the compiled launcher: no per-call Python JIT specialization.
            runner = kernel[grid]
            # Both ctypes and the compiled Triton launcher receive cached addresses;
            # NumPy's .ctypes.data property is never part of either timed call.
            launch_args = [arg.data_ptr() if isinstance(arg, Span) else arg for arg in args]
            call = lambda runner=runner, args=launch_args: runner(*args)
            reset()
            call()
            accuracy = errors(outputs["triton"], reference, tol)
            if not accuracy["pass"]:
                raise AssertionError((kind, shape, choice, accuracy))
            reset()
            repeats = calibrate(call, target_ns)
            tuning_ns = [elapsed(call, repeats) for _ in range(5)]
            candidates.append(
                {
                    "configuration": list(choice),
                    "first_call_s": cold,
                    "validation": accuracy,
                    "tuning_ns_per_call": tuning_ns,
                    "repeats": repeats,
                    "compiled_hash": kernel.hash,
                    "metadata": kernel.metadata._asdict(),
                }
            )
            runs.append((call, kernel))
        selected = min(
            range(len(choices)),
            key=lambda i: statistics.median(candidates[i]["tuning_ns_per_call"]),
        )
        call, kernel = runs[selected]
        native["triton"] = call
        accuracy, checked_outputs = {}, {}
        for name, function in native.items():
            reset()
            function()
            accuracy[name] = errors(outputs[name], reference, tol)
            checked_outputs[name] = outputs[name].copy()
            if not accuracy[name]["pass"]:
                raise AssertionError((kind, shape, name, accuracy[name]))
        pair_delta = {
            name: float(np.max(np.abs(output.astype(np.float64) - checked_outputs["triton"])))
            for name, output in checked_outputs.items()
            if name != "triton"
        }
        repeats = {}
        for name, function in native.items():
            reset()
            repeats[name] = calibrate(function, target_ns)
        observations = []
        for index in range(rounds):
            order = list(native)
            rng.shuffle(order)
            row = {"round": index, "order": order, "ns_per_call": {}}
            for name in order:
                reset()
                # Outside timing: allow BCIR's bounded spin/yield tail to park too.
                time.sleep(0.002)
                row["ns_per_call"][name] = elapsed(native[name], repeats[name])
            observations.append(row)
        summary = {
            name: statistics.median([r["ns_per_call"][name] for r in observations])
            for name in native
        }
        ratios = {
            name: ratio_interval(
                [r["ns_per_call"][name] for r in observations],
                [r["ns_per_call"]["triton"] for r in observations],
                rng,
            )
            for name in native
            if name != "triton"
        }
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {}
        for stage in ("ttir", "ttcir", "llir", "asm"):
            content = kernel.asm[stage].encode()
            (artifact_dir / (stage + ".txt")).write_bytes(content)
            artifacts[stage] = hashlib.sha256(content).hexdigest()
        return {
            "kind": kind,
            "shape": list(shape),
            "threads": threads,
            "data_seed": data_seed,
            "resident_input_output_bytes_per_implementation": resident,
            "bcir_pool_bytes": len(storage),
            "candidates": candidates,
            "selected": selected,
            "correctness": accuracy,
            "max_abs_bcir_minus_triton": pair_delta,
            "repeats": repeats,
            "median_ns": summary,
            "ratios": ratios,
            "observations": observations,
            "selected_ir_sha256": artifacts,
        }
    finally:
        lib.bcir_tensor_pool_stop(pool)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cc", default="gcc")
    parser.add_argument("--rounds", type=int, default=21)
    parser.add_argument("--target-ms", type=float, default=8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.rounds < 5 or not 1 <= args.target_ms <= 100:
        parser.error("require rounds >= 5 and 1 <= target-ms <= 100")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        parser.error("choose a new --out; results.json already exists")
    if (out / "triton-cache").exists():
        parser.error("choose a new --out with an empty study JIT cache")
    os.environ["TRITON_CACHE_DIR"] = str(out / "triton-cache")
    if hasattr(os, "sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        if len(allowed) < 2:
            parser.error("this protocol requires two available logical CPUs")
        os.sched_setaffinity(0, allowed[:2])
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "cpu" or importlib.metadata.version("triton") != "3.8.0+cpu":
        parser.error("requires the official experimental Triton-CPU 3.8.0+cpu wheel")
    lib, build = build_native(out, args.cc)
    cases = [
        ("add", (4096,)),
        ("add", (1048579,)),
        ("swiglu", (4096,)),
        ("swiglu", (1048579,)),
        ("rms", (64, 128)),
        ("rms", (256, 257)),
        ("rms", (128, 1024)),
        ("gemm", (128, 128, 128)),
        ("gemm", (256, 256, 256)),
        ("gemm", (129, 257, 255)),
    ]
    if args.smoke:
        cases = [("add", (4099,)), ("swiglu", (4099,)), ("rms", (17, 257)), ("gemm", (17, 33, 31))]
    cpuinfo = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    cpu = next(
        (s.split(":", 1)[1].strip() for s in cpuinfo.splitlines() if s.startswith("model name")),
        "unknown",
    )
    result = {
        "protocol": "bcir-triton-cpu-pilot-v2",
        "status": "RUNNING",
        "seed": 820,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bcir_commit": command(["git", "rev-parse", "HEAD"]),
        "tensor_c_sha256": digest(ROOT / "runtime/c/bcir_tensor.c"),
        "tensor_h_sha256": digest(ROOT / "runtime/c/bcir_tensor.h"),
        "harness_sha256": digest(__file__),
        "triton_cpu_source": CPU_SOURCE,
        "triton_cpu_wheel_sha256": WHEEL_SHA256,
        "triton_package": importlib.metadata.version("triton"),
        "numpy": np.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": cpu,
        "affinity": sorted(os.sched_getaffinity(0)),
        "target": str(target),
        "rounds": args.rounds,
        "target_batch_ms": args.target_ms,
        "launch_boundary": "ctypes C pool vs bound CompiledKernel CPU launcher; cached integer pointers; synchronous",
        "cache_policy": "warm repeated buffers; no cache flush; empty study JIT cache initially",
        "idle_policy": {
            name: os.environ.get(name)
            for name in (
                "OMP_WAIT_POLICY",
                "GOMP_SPINCOUNT",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
        "between_batch_park_ms": 2,
        "numerics": "FP32, no fast math; GEMM permits FMA, others disable fusion; tolerance not bit parity",
        "gpu": "UNAVAILABLE-SKIPPED: no GPU on host",
        "model_training": "DEFERRED: PR #820",
        "build": build,
        "cases": [],
    }
    result["bcir_variant_masks"] = {
        name: getattr(lib, "bcir_tensor_" + name)()
        for name in ("mm_variants", "mm_fused_variants", "mm_kblock_variants")
    }
    rng = np.random.default_rng(820)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for threads in (1, 2):
            for case_index, (kind, shape) in enumerate(cases):
                label = f"{kind}-{'x'.join(map(str, shape))}-t{threads}"
                print("RUN", label, flush=True)
                row = run_case(
                    kind,
                    shape,
                    threads,
                    lib,
                    rng,
                    args.rounds,
                    args.target_ms * 1e6,
                    out / "ir" / label,
                    820 + case_index,
                )
                result["cases"].append(row)
                (out / "results.json").write_text(
                    json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
                )
                print(
                    "PASS",
                    label,
                    {k: round(v / 1000, 2) for k, v in row["median_ns"].items()},
                    flush=True,
                )
        result["status"] = "PASS"
    except Exception as error:
        result["status"], result["error"] = "FAIL", repr(error)
        raise
    finally:
        if was_enabled:
            gc.enable()
        (out / "results.json").write_text(
            json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
        )
    print(
        f"PASS: {len(result['cases'])} compiled CPU cases; GPU skipped; training deferred",
        flush=True,
    )


if __name__ == "__main__":
    main()
