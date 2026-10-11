"""Reproduce the 2026-10-10 GEMplus / ASN.1 / C tensor measurements.

Run from any directory: python run_performance_tests.py --stage gem --output DIR.
GEM and ASN.1 need Python and GCC; kernels additionally need NumPy, SciPy and
CPU PyTorch. Generated C and build products go to OUTPUT, never runtime/c.
This is a measurement harness, not a production runtime or a CI timing gate.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import gc
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import tracemalloc
from dataclasses import asdict, replace
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "bcir").is_dir())
sys.path.insert(0, str(REPO))
FP = ct.POINTER(ct.c_float)
BP = ct.POINTER(ct.c_uint8)


def command(args):
    run = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=600)
    if run.returncode:
        raise RuntimeError(f"{args}: {run.stdout[-1000:]} {run.stderr[-3000:]}")
    return run.stdout


def sha(data):
    return hashlib.sha256(data).hexdigest()


def samples(fn, rounds=7):
    fn()
    raw = []
    for _ in range(rounds):
        gc.collect()
        start = time.perf_counter_ns()
        value = fn()
        raw.append((time.perf_counter_ns() - start) / 1e6)
        del value
    return {"samples_ms": raw, "median_ms": statistics.median(raw)}


def gem(out):
    from bcir.abi.streampack_abi import decode, encode
    from bcir.gem.delta_chain import DeltaChain
    from bcir.gem.streampack import hydrate, hydrate_pipelined
    from bcir.kbcir.delta import Delta
    from bcir.kbcir.realize import optimize
    from bcir.kbcir.weights import PERF
    from bcir.tests import delta_fixtures as df
    from bcir.tests import hydrate_fixtures as hf
    from bcir.tests import planner_fixtures as pf
    from bcir.performance_audit import scheduler_fixture

    exe = pf.build_harness(str(out))
    if exe is None:
        raise RuntimeError("native planner/hydrate unavailable")
    rows = []
    for scale in (1, 2, 4, 8):
        module, target, theta = pf.audit_fixture(scale)
        result = optimize(module, target, theta)
        pack = hydrate_pipelined(module, result, plan="plan0", depth=2)
        wire = encode(pack)
        assert encode(decode(wire)) == wire
        funcs = {
            "plan": lambda module=module, target=target, theta=theta: optimize(
                module, target, theta
            ),
            "hydrate_plain": lambda module=module, result=result: hydrate(
                module, result, plan="plan0"
            ),
            "hydrate_pipelined": lambda module=module, result=result: hydrate_pipelined(
                module, result, plan="plan0", depth=2
            ),
            "encode": lambda pack=pack: encode(pack),
            "decode": lambda wire=wire: decode(wire),
        }
        timed = {name: samples(fn, 5) for name, fn in funcs.items()}
        native = {}
        for name, fn in (
            ("planner", pf.native_ms),
            ("planner_write_floor", pf.native_floor),
            ("hydrate", hf.native_ms),
            ("hydrate_write_floor", hf.native_floor),
        ):
            values = [fn(exe, str(out), scale) for _ in range(3)]
            ms = [x[0] if isinstance(x, tuple) else x for x in values]
            native[name] = {"samples_ms": ms, "median_ms": statistics.median(ms)}
            if isinstance(values[0], tuple):
                native[name]["bytes"] = values[0][1]
        m, r, records = hf._scale_records(scale)
        wanted = encode(hydrate(m, r, plan="plan0"))
        status, actual = hf.c_hydrate_batch(exe, str(out), [records])[0]
        assert status == "BCIR_OK" and actual == wanted
        timed["hydrate_plain_plus_encode"] = samples(
            lambda m=m, r=r: encode(hydrate(m, r, plan="plan0")), 5
        )
        chain = DeltaChain.build(module, target, theta, PERF, "plan0", 2)
        raw_delta, parity = [], 0
        for k in range(7):
            claims, resources = df.audit_delta(chain.module, k)
            # The audit's count-increase witness deliberately produces R7 overruns.
            # Use an in-bounds count decrease for this application-performance run.
            claims = (replace(claims[0], count=claims[0].count - 2 * (k + 1)),)
            new = df.declared(chain.module, claims, resources)
            gc.collect()
            start = time.perf_counter_ns()
            link = chain.apply(Delta(claims, resources))
            raw_delta.append((time.perf_counter_ns() - start) / 1e6)
            ref = df.reference_chain(new, target, theta, PERF, 2)
            assert link.data == ref.data
            assert link.diagnostics == ref.diagnostics == []
            parity += 1
        full = samples(
            lambda chain=chain, target=target, theta=theta: df.reference_chain(
                chain.module, target, theta, PERF, 2
            ),
            5,
        )
        peak = {}
        for name in ("plan", "decode"):
            gc.collect()
            tracemalloc.start()
            value = funcs[name]()
            _, maximum = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            del value
            peak[name] = maximum
        row = {
            "scale": scale,
            "claims": len(result.steps),
            "wire_bytes": len(wire),
            "wire_sha256": sha(wire),
            "python": timed,
            "native": native,
            "native_hydrate_byte_parity": True,
            "python_peak_allocated_bytes": peak,
            "delta": {
                "samples_ms": raw_delta,
                "median_ms": statistics.median(raw_delta),
                "full_chain": full,
                "byte_parity_cases": parity,
                "rebuilds": chain.rebuilds,
                "updated_wire_bytes": len(link.data),
                "diagnostic_count": len(link.diagnostics),
            },
        }
        rows.append(row)
        print(json.dumps({"gem_claims": row["claims"], "done": True}), flush=True)
    schedulers = []
    for scale in (1, 2, 4, 8, 16):
        module, target, durations = scheduler_fixture(scale)
        from bcir.gem.async_tokens import async_plan
        from bcir.gem.concurrency import schedule_concurrent
        from bcir.gem.schedule import schedule_eft

        tokens = async_plan(module)
        eft = schedule_eft(module, durations, target)
        assert len(tokens.forks) == len(eft.slots) == 512 * scale
        schedulers.append(
            {
                "claims": 512 * scale,
                "await_edges": sum(map(len, tokens.awaits.values())),
                "makespan": eft.makespan,
                "timing": {
                    name: samples(fn, 5)
                    for name, fn in (
                        (
                            "waves",
                            lambda module=module, target=target: schedule_concurrent(
                                module, target
                            ),
                        ),
                        ("tokens", lambda module=module: async_plan(module)),
                        (
                            "eft",
                            lambda durations=durations, module=module, target=target: schedule_eft(
                                module, durations, target
                            ),
                        ),
                    )
                },
            }
        )
    return {
        "rows": rows,
        "schedulers": schedulers,
        "scope": "Python fixture construction excluded; result allocation included. Native records decoded once outside timer, caller-owned output/scratch reused. Native hydrate compares plain hydrate+encode, not pipelined hydrate. Warm working sets; VM timings indicative.",
    }


CODEC_C = r"""
#define _POSIX_C_SOURCE 200809L
#include "probe_codec.h"
#include <time.h>
#include <string.h>
#define N @N@
#define CAP (N*4+128)
static uint8_t frames[4][CAP];
static size_t lengths[4];
static uint16_t values[N];
static volatile uint64_t sink;
int set_frame(int mode,const uint8_t *p,size_t n) {
  if(mode<0||mode>3||n>CAP)return 1;
  memcpy(frames[mode],p,n);lengths[mode]=n;
  for(size_t i=0;i<N;i++)values[i]=(uint16_t)((i*97+i/7)%260);
  return 0;
}
int do_encode(int mode,uint8_t *out,size_t cap,size_t *len) {
  probe_Batch v={N,values};
  if(mode==1)return probe_coer_encode_Batch(&v,out,cap,len);
  if(mode==2)return probe_uper_encode_Batch(&v,out,cap,len);
  size_t size=mode==0?2*N:(9*N+7)/8;
  if(cap<size)return 1;
  memset(out,0,size);
  for(size_t i=0;i<N;i++) {
    uint16_t x=values[i];
    if(mode==0) {out[i*2]=(uint8_t)(x>>8);out[i*2+1]=(uint8_t)x;}
    else {
      size_t pos=i*9,byte=pos/8;unsigned shift=(unsigned)(pos%8);
      out[byte]|=(uint8_t)(x>>(1+shift));
      out[byte+1]|=(uint8_t)(x<<(7-shift));
    }
  }
  *len=size;return 0;
}
int do_decode(int mode,const uint8_t *in,size_t len,uint32_t *out) {
  uint16_t arena_data[N];
  if(mode==1||mode==2) {
    probe_Batch value;probe_arena arena={(uint8_t*)arena_data,sizeof arena_data,0};
    int s=mode==1?probe_coer_decode_Batch(in,len,&value,&arena):probe_uper_decode_Batch(in,len,&value,&arena);
    if(s)return s;
    for(size_t i=0;i<N;i++)out[i]=value.v[i];
    return 0;
  }
  if(len!=(mode==0?2*N:(9*N+7)/8))return 1;
  if(mode==3 && (9*N)%8 && (in[len-1]&((1u<<(8-(9*N)%8))-1u)))return 1;
  if(mode==3) {
    for(size_t base=0;base<N;base+=8,in+=9)for(unsigned j=0;j<8;j++) {
      unsigned x=((((unsigned)in[j]<<8)|in[j+1])>>(7-j))&511u;
      if(x>259)return 1;
      out[base+j]=x;
    }
    return 0;
  }
  for(size_t i=0;i<N;i++) {
    unsigned x;
    if(mode==0)x=((unsigned)in[i*2]<<8)|in[i*2+1];
    else {size_t pos=i*9,byte=pos/8;unsigned shift=(unsigned)(pos%8);x=((((unsigned)in[byte]<<8)|in[byte+1])>>(7-shift))&511u;}
    if(x>259)return 1;
    out[i]=x;
  }
  return 0;
}
double bench(int mode,int op,size_t reps) {
  uint32_t out[N];uint8_t wire[CAP];size_t len=0;struct timespec a,b;
  clock_gettime(CLOCK_MONOTONIC,&a);
  for(size_t i=0;i<reps;i++) {
    int status=op?do_encode(mode,wire,sizeof wire,&len):do_decode(mode,frames[mode],lengths[mode],out);
    if(status)return -1;
    sink+=op?wire[i%len]:out[i%N];
  }
  clock_gettime(CLOCK_MONOTONIC,&b);
  return ((double)(b.tv_sec-a.tv_sec)*1e9+(b.tv_nsec-a.tv_nsec))/reps;
}
"""


def asn1(out):
    from bcir.frontends.asn1 import compile_module
    from bcir.asn1.cgen import generate, COER, UPER
    from bcir.asn1.oer import encode_oer
    from bcir.asn1.per import encode_per

    rows = []
    for n in (64, 512, 4096, 32768):
        directory = out / f"codec-{n}"
        directory.mkdir(exist_ok=True)
        schema = (
            f"Probe DEFINITIONS ::= BEGIN Batch ::= SEQUENCE (SIZE({n})) OF INTEGER (0..259) END"
        )
        types = compile_module(schema, "Probe.asn1").module.types
        codec = generate(types, ["Batch"], prefix="probe_", rules=[COER, UPER])
        (directory / codec.header_name).write_text(codec.header)
        (directory / "codec.c").write_text(codec.source)
        (directory / "bench.c").write_text(CODEC_C.replace("@N@", str(n)))
        library = directory / "codec.so"
        build = [
            "gcc",
            "-std=c11",
            "-O3",
            "-march=native",
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            "-I",
            directory,
            directory / "codec.c",
            directory / "bench.c",
            "-o",
            library,
        ]
        command(build)
        lib = ct.CDLL(str(library))
        lib.set_frame.argtypes = [ct.c_int, BP, ct.c_size_t]
        lib.do_encode.argtypes = [ct.c_int, BP, ct.c_size_t, ct.POINTER(ct.c_size_t)]
        lib.do_decode.argtypes = [ct.c_int, BP, ct.c_size_t, ct.POINTER(ct.c_uint32)]
        lib.bench.argtypes = [ct.c_int, ct.c_int, ct.c_size_t]
        lib.bench.restype = ct.c_double
        values = [(i * 97 + i // 7) % 260 for i in range(n)]
        raw = b"".join(x.to_bytes(2, "big") for x in values)
        frames = [raw, encode_oer(types["Batch"], values), encode_per(types["Batch"], values)]
        frames.append(frames[2])
        parity = rejects = 0
        for mode, frame in enumerate(frames):
            data = (ct.c_uint8 * len(frame)).from_buffer_copy(frame)
            assert lib.set_frame(mode, data, len(frame)) == 0
            target = (ct.c_uint32 * n)()
            assert lib.do_decode(mode, data, len(frame), target) == 0
            assert list(target) == values
            wire = (ct.c_uint8 * (4 * n + 128))()
            length = ct.c_size_t()
            assert lib.do_encode(mode, wire, len(wire), ct.byref(length)) == 0
            assert bytes(wire[: length.value]) == frame
            parity += 2
            for cut in sorted({0, 1, len(frame) // 2, len(frame) - 1}):
                assert lib.do_decode(mode, data, cut, target) != 0
                rejects += 1
            extra = (ct.c_uint8 * (len(frame) + 1)).from_buffer_copy(frame + b"\0")
            assert lib.do_decode(mode, extra, len(extra), target) != 0
            rejects += 1
        timed = {
            name: {"encode_ns": [], "decode_ns": []}
            for name in ("raw16", "coer", "uper", "packed9")
        }
        reps = max(64, 1000000 // n)
        for rep in range(11):
            order = list(range(8))
            random.Random(713 + rep).shuffle(order)
            for job in order:
                mode, op = divmod(job, 2)
                ns = lib.bench(mode, op, reps)
                assert ns > 0
                timed[list(timed)[mode]]["encode_ns" if op else "decode_ns"].append(ns)
        for times in timed.values():
            times["median_encode_ns"] = statistics.median(times["encode_ns"])
            times["median_decode_ns"] = statistics.median(times["decode_ns"])
        rows.append(
            {
                "tokens": n,
                "bytes": dict(zip(timed, map(len, frames))),
                "timing": timed,
                "parity_checks": parity,
                "rejection_checks": rejects,
                "compiler_command": list(map(str, build)),
                "generated_source_sha256": sha(codec.source.encode()),
            }
        )
        print(json.dumps({"asn1_tokens": n, "done": True}), flush=True)
    return {
        "rows": rows,
        "scope": "Fixed schema SIZE(N), token range 0..259, no explicit metadata. Warm single frame per size. Native encode; native decode+range check+uint32 materialization. Generated decoders additionally construct uint16 arena arrays. Packed9 is a schema-specialized control with identical UPER bytes. No tensor arithmetic or disk/network I/O.",
    }


def kernels(out):
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    import numpy as np
    import scipy
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    library = out / "tensor.so"
    command(
        [
            "gcc",
            "-std=c11",
            "-O3",
            "-march=native",
            "-fno-math-errno",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            REPO / "runtime/c/bcir_tensor.c",
            "-o",
            library,
            "-lm",
        ]
    )
    lib = ct.CDLL(str(library))
    args = [
        ct.c_void_p,
        ct.c_int,
        ct.c_int,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_float,
        FP,
        FP,
        ct.c_float,
        FP,
    ]
    lib.bcir_tensor_mm.argtypes = args
    lib.bcir_tensor_cblas_mm.argtypes = args
    blaspath = next((Path(scipy.__file__).parent.parent / "scipy.libs").glob("*openblas*.so"))
    blas = ct.CDLL(str(blaspath))
    blas.scipy_openblas_set_num_threads(1)
    provider = ct.c_void_p(ct.cast(blas.scipy_cblas_sgemm, ct.c_void_p).value)
    rng = np.random.default_rng(713)
    rows = []
    for m, n, k, tb in (
        (64, 64, 48, 0),
        (256, 384, 384, 0),
        (256, 1024, 384, 1),
        (384, 384, 256, 0),
    ):
        a = rng.standard_normal((m, k), dtype=np.float32) / 8
        b = rng.standard_normal((n, k) if tb else (k, n), dtype=np.float32) / 8
        outputs = {
            name: np.zeros((m, n), np.float32) for name in ("portable_c", "c_openblas", "pytorch")
        }
        x, y = torch.from_numpy(a), torch.from_numpy(b.T if tb else b)
        z = torch.from_numpy(outputs["pytorch"])

        def call(name, a=a, b=b, k=k, m=m, n=n, outputs=outputs, tb=tb, x=x, y=y, z=z):
            if name == "pytorch":
                torch.mm(x, y, out=z)
            else:
                fn = lib.bcir_tensor_mm if name == "portable_c" else lib.bcir_tensor_cblas_mm
                ctx = None if name == "portable_c" else ct.byref(provider)
                fn(
                    ctx,
                    0,
                    tb,
                    m,
                    n,
                    k,
                    1,
                    a.ctypes.data_as(FP),
                    b.ctypes.data_as(FP),
                    0,
                    outputs[name].ctypes.data_as(FP),
                )

        for name in outputs:
            call(name)
        ref = a.astype(np.float64) @ (b.T if tb else b).astype(np.float64)
        errors = {name: float(np.max(np.abs(value - ref))) for name, value in outputs.items()}
        for value in outputs.values():
            assert np.allclose(value, ref, atol=2e-5, rtol=1e-4)
        raw = {name: [] for name in outputs}
        repeats = max(1, min(64, 20000000 // (m * n * k)))
        for round_ in range(11):
            names = list(outputs)
            random.Random(913 + round_).shuffle(names)
            for name in names:
                start = time.perf_counter_ns()
                for _ in range(repeats):
                    call(name)
                raw[name].append((time.perf_counter_ns() - start) / repeats / 1e6)
        rows.append(
            {
                "m": m,
                "n": n,
                "k": k,
                "transpose_b": tb,
                "samples_ms": raw,
                "median_ms": {name: statistics.median(v) for name, v in raw.items()},
                "max_abs_error_vs_fp64": errors,
            }
        )
        print(json.dumps({"kernel_shape": [m, n, k, tb], "done": True}), flush=True)
    return {
        "rows": rows,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scope": "Single-thread FP32 GEMM, preallocated outputs, same alpha=1 beta=0/transpose, data preparation excluded, C ctypes/PyTorch call overhead included. Torch eager mm delegates to optimized CPU BLAS; this is a kernel comparison, not full training.",
    }


def its(out):
    from bcir.asn1 import its_native as native

    exe = native.build(str(out), cc="gcc", flags=("-O2",))
    expected = native.oracle_octets()
    commands = []
    for codec, envelope in expected:
        commands += [f"enc {codec} {envelope}", f"dec {codec} {expected[codec, envelope].hex()}"]
    answers = native.run(exe, commands)
    for i, ((_, envelope), wire) in enumerate(expected.items()):
        assert answers[2 * i] == "OK " + wire.hex()
        fields = answers[2 * i + 1].split()
        assert fields[0] == "OK" and int(fields[1]) == envelope and fields[3] == wire.hex()
    runs = [[asdict(x) for x in native.bench(exe, rounds=24, iters=3000)] for _ in range(3)]
    summary = []
    for envelope in range(4):
        for codec, _, _ in native.CODECS:
            med = {
                op: statistics.median(
                    next(
                        x["median_ns"]
                        for x in r
                        if x["codec"] == codec and x["envelope"] == envelope and x["op"] == op
                    )
                    for r in runs
                )
                for op in ("encode", "decode")
            }
            summary.append(
                {
                    "envelope": envelope,
                    "codec": codec,
                    "bytes": len(expected[codec, envelope]),
                    **med,
                }
            )
    return {
        "runs": runs,
        "summary": summary,
        "parity_checks": len(expected) * 2,
        "scope": "Repository reconstructed 2015 ITS protocol versus raw/tuned ASN.1 schemas, same logical content and generator, GCC13 -O2; not a universal pure-binary baseline.",
    }


def crc(out):
    import zlib

    library = out / "runtime.so"
    command(
        [
            "gcc",
            "-std=c11",
            "-O2",
            "-shared",
            "-fPIC",
            REPO / "runtime/c/bcir_runtime.c",
            "-o",
            library,
        ]
    )
    lib = ct.CDLL(str(library))
    lib.bcir_crc32.argtypes = [BP, ct.c_size_t]
    lib.bcir_crc32.restype = ct.c_uint32
    rows = []
    for size in (16384, 131072, 1048576, 8388608):
        data = random.Random(913).randbytes(size)
        buf = (ct.c_uint8 * size).from_buffer_copy(data)
        expected = zlib.crc32(data)
        assert lib.bcir_crc32(buf, size) == expected
        raw = {"bcir_c": [], "zlib": []}
        for rep in range(11):
            order = list(raw) if rep % 2 else list(reversed(raw))
            for name in order:
                fn = (
                    (lambda buf=buf, size=size: lib.bcir_crc32(buf, size))
                    if name == "bcir_c"
                    else (lambda data=data: zlib.crc32(data))
                )
                start = time.perf_counter_ns()
                result = fn()
                raw[name].append((time.perf_counter_ns() - start) / 1e6)
                assert result == expected
        rows.append(
            {
                "bytes": size,
                "samples_ms": raw,
                "median_ms": {name: statistics.median(v) for name, v in raw.items()},
                "crc32": expected,
            }
        )
    return {
        "rows": rows,
        "scope": "Identical IEEE CRC32 input/output, warm buffers, GCC13 -O2 production runtime versus Python zlib's native implementation. C ctypes/zlib Python entry overhead included; no checksum validation removed.",
    }


def frontier(out):
    from bcir.performance_audit import scheduler_fixture
    from bcir.gem.concurrency import hazard_predecessors, _frontier_predecessors

    rows = []
    for scale in (1, 2, 4, 8, 16):
        module, _, _ = scheduler_fixture(scale)
        claims = module.phases[0].claims
        old, new = hazard_predecessors(claims), _frontier_predecessors(claims)

        def closure(preds, claims=claims):
            masks = {}
            for claim in claims:
                mask = 0
                for pred in preds[claim.id]:
                    mask |= (1 << (pred - 1)) | masks[pred]
                masks[claim.id] = mask
            return masks

        assert closure(old) == closure(new)
        rows.append(
            {
                "claims": len(claims),
                "all_edges": sum(map(len, old.values())),
                "frontier_edges": sum(map(len, new.values())),
                "all_hazards": samples(lambda claims=claims: hazard_predecessors(claims), 5),
                "frontier": samples(lambda claims=claims: _frontier_predecessors(claims), 5),
                "transitive_closure_equal": True,
            }
        )
    return {
        "rows": rows,
        "scope": "Current hazard_predecessors versus existing private _frontier_predecessors, same fence-free scheduler fixtures. Every node's reachability mask identical. Recommendation only; async plan/verifier/MLIR not changed. Fence, phase and event corpora still required before production adoption.",
    }


def attention(out):
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    import numpy as np
    import torch
    import torch.nn.functional as functional

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    library = out / "tensor.so"
    command(
        [
            "gcc",
            "-std=c11",
            "-O3",
            "-march=native",
            "-fno-math-errno",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            REPO / "runtime/c/bcir_tensor.c",
            "-o",
            library,
            "-lm",
        ]
    )
    lib = ct.CDLL(str(library))
    lib.bcir_tensor_attention.argtypes = [ct.c_size_t] * 5 + [FP] * 5
    rng = np.random.default_rng(713)
    rows = []
    for context in (64, 128, 256):
        b, h, kh, d = 4, 8, 4, 48
        q = rng.standard_normal((b, context, h, d), dtype=np.float32) / 8
        k = rng.standard_normal((b, context, kh, d), dtype=np.float32) / 8
        v = rng.standard_normal((b, context, kh, d), dtype=np.float32) / 8
        p = np.zeros((b, h, context, context), np.float32)
        y = np.zeros_like(q)
        tq = torch.from_numpy(q).permute(0, 2, 1, 3).contiguous()
        tk = torch.from_numpy(k).permute(0, 2, 1, 3).repeat_interleave(h // kh, dim=1).contiguous()
        tv = torch.from_numpy(v).permute(0, 2, 1, 3).repeat_interleave(h // kh, dim=1).contiguous()

        def c_call(b=b, context=context, d=d, h=h, k=k, kh=kh, p=p, q=q, v=v, y=y):
            lib.bcir_tensor_attention(
                b, context, h, kh, d, *(x.ctypes.data_as(FP) for x in (q, k, v, p, y))
            )

        def torch_call(tk=tk, tq=tq, tv=tv):
            return functional.scaled_dot_product_attention(tq, tk, tv, is_causal=True)

        c_call()
        result = torch_call().permute(0, 2, 1, 3).numpy()
        error = float(np.max(np.abs(y - result)))
        assert np.allclose(y, result, atol=2e-6, rtol=1e-4)
        assert (np.triu(p, k=1) == 0).all()
        raw = {"native_c": [], "torch_sdpa": []}
        for rep in range(11):
            for name in list(raw) if rep % 2 else list(reversed(raw)):
                fn = c_call if name == "native_c" else torch_call
                start = time.perf_counter_ns()
                value = fn()
                raw[name].append((time.perf_counter_ns() - start) / 1e6)
                del value
        rows.append(
            {
                "batch": b,
                "context": context,
                "query_heads": h,
                "kv_heads": kh,
                "head_dim": d,
                "samples_ms": raw,
                "median_ms": {name: statistics.median(v) for name, v in raw.items()},
                "max_abs_error": error,
                "native_probabilities_bytes": p.nbytes,
                "torch_repeated_kv_extra_bytes": (tk.numel() + tv.numel() - k.size - v.size) * 4,
            }
        )
    return {
        "rows": rows,
        "torch": torch.__version__,
        "scope": "C causal GQA forward versus CPU Torch SDPA, single thread FP32, equal outputs. C BT-head-channel layout with preallocated full probability matrix; Torch contiguous B-head-T-channel, K/V repetition/preparation excluded, output allocation included. No backward/full-training speedup inferred.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("gem", "asn1", "kernels", "its", "crc", "frontier", "attention"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    record = {
        "source_commit": command(["git", "-C", REPO, "rev-parse", "HEAD"]).strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "compiler": command(["gcc", "--version"]).splitlines()[0],
        "stage": args.stage,
        "results": globals()[args.stage](args.output),
    }
    path = args.output / f"{args.stage}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"saved": str(path)}), flush=True)


if __name__ == "__main__":
    main()
