"""SYCL as a backend CHANNEL + a toolchain-gated DIFFERENTIAL ORACLE (never on the legality path).

This is the SYCL/SPIR-V GPU realization of the agreed strategy: SYCL is (a) a modeled backend channel the
planner prices via the K_BCIR cost model and routes to like any other, and (b) a differential oracle that
MEASURES that a heterogeneous device reproduces BCIR's own deterministic SAXPY reference before committing
to a resident driver. It mirrors the FIVE Area-B library wraps (BLAS/FFTW/LAPACK/GSL/SLEEF) in SPIRIT, but
with ONE critical architectural difference: SYCL is NOT a ``c.call.libm:`` external-symbol edge and has NO
``-l<lib>`` link rule -- it is a single-source C++ *compiler mode* (``-fsycl``), and the kernel is emitted
SYCL C++ source (a ``parallel_for``), not a call to a linkable symbol. That distinction is asserted here
(test_sycl_is_not_a_link_flag_edge: the five existing library link rules still resolve; a SYCL symbol
resolves to None/unknown).

Covers: the channel manifest validates + registers + a data_parallel claim routes to it; the SAXPY oracle
reference matches an independent per-element recompute (stdlib only) incl. a closed-form anchor, rejects
empty/mismatched inputs, and is side-effect-free; the bridged SAXPY tracks the reference within the R17
input round-trip bound (``(|a|+1)``-scaled -- exact arithmetic, no transcendental amplification); the
emitted C++ kernel wraps a SYCL ``parallel_for`` with a portable scalar fallback (and carries NO
``c.call.libm:`` label); the fallback path compiles (as C++) + runs + is correct; only if a real SYCL
compiler is present (``icpx`` / ``acpp`` / ``clang++ -fsycl`` that actually compiles+links a SYCL probe)
does the device path run and agree; R17 certifies the bridge step on a quantized call; and SYCL adds NO
link rule. The point: SYCL is measured as a backend before any resident driver, with its dynamic scheduler
held off the deterministic rail."""

import math
import os
import shutil
import subprocess
import tempfile

from bcir.channel_plugin import load_manifest, register_from_manifest
from bcir.channels import CHANNELS, HardwareChannel, channel, channel_suits, route_claim
from bcir.frontends.cfront.linkflags import NO_FLAG, library_for_callee
from bcir.kbcir.precision import accuracy_bound, quantization_error_bound
from bcir.kbcir.sycl_saxpy import saxpy_reference, saxpy_via_bridge
from bcir.lower.c_kernel import emit_sycl_saxpy_c
from bcir.model import Claim, Domain, Lane, Opcode, StrideClass

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MANIFEST = os.path.join(_REPO, "channels", "sycl.channel.json")


def _sycl_channel():
    """Return the process-local fixture without weakening duplicate registration.

    ``run_all`` reuses worker processes, so an earlier SYCL test may already have
    installed this exact fixture. Production registration must still reject an
    attempted replacement of a live backend.
    """
    try:
        return channel("sycl_spirv")
    except KeyError:
        return register_from_manifest(load_manifest(_MANIFEST))


def _independent_saxpy(a, x, y):
    """An INDEPENDENT per-element recompute (no BCIR code, no numpy): out[i] = a*x[i] + y[i]. The oracle
    saxpy_reference must agree with this."""
    return [a * xi + yi for xi, yi in zip(x, y)]


# --- (1) the channel manifest: validates, registers, and a data_parallel claim routes to it ------------

def test_sycl_manifest_validates_and_registers():
    m = load_manifest(_MANIFEST)
    assert m.validate() == [], m.validate()
    # identity + codegen + provenance + capabilities are the modeled SPIR-V/SYCL GPU channel.
    assert m.name == "sycl_spirv" and m.kind == "gpu"
    assert m.llvm_triple == "spirv64-unknown-unknown"
    assert m.e_machine == 0                                 # off-host: SPIR-V is not a host ELF
    assert m.provenance == "modeled" and m.modeled is True  # no resident driver / calibration yet
    assert m.profile.triple == "spirv64"
    assert m.profile.lane_widths[0] == 1                    # scalar width first (validate() requires it)
    assert m.profile.warp == 32                             # a warp-ish subgroup machine
    assert m.capabilities == frozenset({"data_parallel", "matmul", "reduce"})
    ch = _sycl_channel()
    assert isinstance(ch, HardwareChannel) and ch.name == "sycl_spirv"
    assert ch.kind == "gpu" and not ch.is_host_elf          # off-host (no host ELF object)


def test_a_data_parallel_claim_routes_to_the_sycl_channel():
    # SAXPY is the canonical data_parallel op; the sycl channel DECLARES data_parallel, so an elementwise
    # (non-reduce, non-gather, non-matmul) claim maps to the data_parallel cap and is eligible for it, and
    # among a candidate set it is the specialized pick (mirrors how test_channel_plugin exercises capability
    # routing -- a plugin's declared tag wins over a legacy kind-routed CPU fallback).
    sycl = _sycl_channel()
    claim = Claim(id=7100, opcode=Opcode.MUL, lane=Lane.U, stride_class=StrideClass.STRIDED, count=8,
                  rd=(71,), wr=(72,), op="call.saxpy", domain=Domain.RAM)
    assert channel_suits(claim, sycl)                       # the declared data_parallel tag matches
    # build a candidate set: the host CPU channel (legacy, kind-routed universal) + the sycl plugin. The
    # plugin declares one of the claim's concrete required caps, so route_claim prefers it over the
    # unspecialized CPU fallback.
    cpu = CHANNELS["x86_avx512"]
    chosen = route_claim(claim, [cpu, sycl])
    assert chosen is sycl, chosen.name


# --- (2) the oracle reference: matches an independent recompute (+ a known closed-form) ----------------

def test_saxpy_reference_matches_an_independent_recompute():
    import random
    rng = random.Random(0x7100)
    a = rng.uniform(-3, 3)
    x = [rng.uniform(-5, 5) for _ in range(17)]
    y = [rng.uniform(-5, 5) for _ in range(17)]
    got = saxpy_reference(a, x, y)
    want = _independent_saxpy(a, x, y)                      # independent of BCIR's code
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert abs(g - w) <= 1e-9 * (1.0 + abs(w)), (g, w)


def test_saxpy_reference_known_closed_form():
    # a=2, x=[1,2,3], y=[10,20,30] -> [2*1+10, 2*2+20, 2*3+30] = [12, 24, 36]. Closed-form, no BCIR code.
    got = saxpy_reference(2.0, [1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
    assert got == [12.0, 24.0, 36.0]
    # a=0 -> out == y exactly; a=1 -> out == x+y.
    assert saxpy_reference(0.0, [1.0, 2.0], [7.0, 9.0]) == [7.0, 9.0]
    assert saxpy_reference(1.0, [1.0, 2.0], [7.0, 9.0]) == [8.0, 11.0]


def test_saxpy_reference_rejects_empty_and_mismatch():
    try:
        saxpy_reference(2.0, [], []); assert False, "empty saxpy should raise"
    except ValueError:
        pass
    try:
        saxpy_reference(2.0, [1.0, 2.0], [1.0]); assert False, "length mismatch should raise"
    except ValueError:
        pass


def test_saxpy_reference_is_side_effect_free():
    x = [3.0, 1.0, -4.0, 1.5, 0.0]
    y = [2.0, -1.0, 5.0, 0.5, -2.0]
    x0, y0 = list(x), list(y)
    saxpy_reference(2.5, x, y)
    assert x == x0 and y == y0


# --- (3) the oracle bridge: tracks the reference within the R17 quant-error bound ----------------------

def test_bridged_saxpy_tracks_the_reference_within_quant_error():
    # SAXPY is EXACT arithmetic (mul+add, NO transcendental), so the only error source is the R17 input
    # round-trip. A round-trip error e on x is scaled by |a| (the multiply); the round-trip error on y
    # passes through the add at unit gain -> per element the bridged result is perturbed by at most
    # (|a| + 1) * e. (Contrast the SLEEF exp wrap, whose transcendental amplifies by exp(x).)
    import random
    from bcir.kbcir.quantize import max_abs_error
    rng = random.Random(0x71F7)
    a = 1.75
    x = [rng.uniform(-4, 4) for _ in range(16)]
    y = [rng.uniform(-4, 4) for _ in range(16)]
    qx = max_abs_error(x, group_size=8, bits=8)            # the R17 input round-trip bound on x (real units)
    qy = max_abs_error(y, group_size=8, bits=8)            # ... and on y
    bound = abs(a) * qx + qy + 1e-6                         # exact-arithmetic bound: |a|*e_x + 1*e_y
    ref = saxpy_reference(a, x, y)
    got = saxpy_via_bridge(a, x, y, group_size=8, bits=8)
    for r, g in zip(ref, got):
        assert abs(r - g) <= bound, (r, g, qx, qy)


def test_bridge_is_a_clean_roundtrip_then_exact_arithmetic():
    # saxpy_via_bridge == reference SAXPY of the dequantized (round-tripped) inputs -- the multiply-add
    # sees exactly the bridged values, so the bridge is the only error source (the arithmetic is exact,
    # adds no certified error -- like the dense BLAS gemm path, not the exp wrap).
    import random
    from bcir.kbcir.quantize import dequantize, quantize_per_group
    rng = random.Random(710)
    a = 2.0
    x = [rng.uniform(-2, 2) for _ in range(12)]
    y = [rng.uniform(-2, 2) for _ in range(12)]
    xd = dequantize(quantize_per_group(x, 8, 8))
    yd = dequantize(quantize_per_group(y, 8, 8))
    assert saxpy_via_bridge(a, x, y, 8, 8) == saxpy_reference(a, xd, yd)


# --- (4) the emitted kernel: a SYCL parallel_for + a portable scalar fallback, NOT a c.call.libm: edge -

def test_emit_wraps_a_sycl_parallel_for_with_a_scalar_fallback():
    c = emit_sycl_saxpy_c(8, "bcir_saxpy")
    assert "parallel_for" in c                                            # the SYCL device kernel
    assert "sycl::" in c                                                  # SYCL namespace (queue/range/id)
    assert "#if defined(BCIR_USE_SYCL)" in c                             # toolchain-selected device path
    assert "#include <sycl/sycl.hpp>" in c                              # the SYCL header on the device path
    assert "a * x[i] + y[i]" in c                                        # the portable scalar fallback SAXPY
    # the signature bakes n in: (a, x, y, out), C++ (no extern "C").
    assert "void bcir_saxpy(float a, const float *x, const float *y, float *out)" in c
    # CRITICAL: SYCL is a compiler MODE, not a c.call.libm: external-symbol edge -- the emitter mints NO
    # such label (unlike the five Area-B library wraps).
    assert "c.call.libm:" not in c
    try:
        emit_sycl_saxpy_c(0, "bcir_saxpy"); assert False, "n=0 should raise"
    except ValueError:
        pass


def _cxx():
    """A host C++ compiler (g++/clang++/c++), or None (the fallback compile/run self-skips)."""
    return shutil.which("g++") or shutil.which("clang++") or shutil.which("c++")


def _run_saxpy_kernel(cxx, kernel, a, x, y, define=None, extra=None):
    """Compile the emitted SAXPY kernel (as C++) + a main that calls it on (a, x, y) and prints each result;
    return the parsed float list. Models test_sleef's _run_exp_kernel, adapted for the (a, x, y, out)
    signature and a C++ compiler."""
    n = len(x)
    xb = ", ".join(f"{v:.8f}f" for v in x)
    yb = ", ".join(f"{v:.8f}f" for v in y)
    main = (f"\n#include <cstdio>\nint main(void){{\n"
            f"  float a = {a:.8f}f;\n"
            f"  float x[{n}] = {{{xb}}};\n"
            f"  float y[{n}] = {{{yb}}};\n"
            f"  float out[{n}];\n  bcir_saxpy(a, x, y, out);\n"
            f"  for (int i = 0; i < {n}; ++i) printf(\"%.8f\\n\", (double)out[i]);\n"
            f"  return 0;\n}}\n")
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "s.cpp")
        with open(src, "w") as f:
            f.write(kernel + main)
        exe = os.path.join(d, "s")
        cmd = [cxx, "-std=c++17", "-O2"]
        if define:
            cmd.append(define)
        if extra:
            cmd += extra
        cmd += [src, "-o", exe]
        bld = subprocess.run(cmd, capture_output=True, text=True)
        assert bld.returncode == 0, bld.stderr
        out = subprocess.run([exe], capture_output=True, text=True)
        assert out.returncode == 0, out.stdout + out.stderr
        return [float(v) for v in out.stdout.split()]


def test_fallback_path_compiles_runs_and_is_correct():
    cxx = _cxx()
    if not cxx:
        return                                              # quick tier hides the toolchain -> self-skip
    a = 2.0
    x = [1.0, 2.0, 3.0, 0.5, -1.0, 4.0, -2.5, 0.0]
    y = [10.0, 20.0, 30.0, -0.5, 1.0, -4.0, 2.5, 7.0]
    ref = saxpy_reference(a, x, y)
    want = _independent_saxpy(a, x, y)                      # the genuine independent check (stdlib)
    got = _run_saxpy_kernel(cxx, emit_sycl_saxpy_c(len(x), "bcir_saxpy"), a, x, y)   # NO -fsycl
    assert len(got) == len(x)
    for g, r, w in zip(got, ref, want):
        # the C++ fallback IS the same SAXPY (float32 vs the oracle's float64) -> agree to float round-off.
        assert abs(g - r) < 1e-4 * (1.0 + abs(r)), (g, r)
        assert abs(g - w) < 1e-4 * (1.0 + abs(w)), (g, w)


def _sycl_cxx():
    """A SYCL compiler invocation that actually COMPILES + LINKS a SYCL probe TU, or None (the device path
    self-skips -- the expected case on CI). Tries, in order, ``icpx``, ``acpp``, then ``clang++ -fsycl``;
    accepts a candidate ONLY if a tiny ``#include <sycl/sycl.hpp>`` + ``sycl::queue`` program builds with
    ``-fsycl`` (so a plain clang++ without the SYCL runtime is correctly rejected). Returns the (compiler,
    extra-flags) pair or None."""
    candidates = []
    if shutil.which("icpx"):
        candidates.append((shutil.which("icpx"), ["-fsycl"]))
    if shutil.which("acpp"):
        candidates.append((shutil.which("acpp"), []))       # acpp/Open SYCL drives -fsycl itself
    if shutil.which("clang++"):
        candidates.append((shutil.which("clang++"), ["-fsycl"]))
    probe = ("#include <sycl/sycl.hpp>\n"
             "int main(){ sycl::queue q; (void)q; return 0; }\n")
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "probe.cpp")
        with open(src, "w") as f:
            f.write(probe)
        for cxx, extra in candidates:
            exe = os.path.join(d, "probe")
            r = subprocess.run([cxx, "-std=c++17", *extra, src, "-o", exe],
                               capture_output=True, text=True)
            if r.returncode == 0:
                return cxx, extra
    return None


def test_sycl_device_path_agrees_when_a_sycl_compiler_is_present():
    sy = _sycl_cxx()
    if not sy:
        return                                              # no SYCL toolchain here -> the device path self-skips
    cxx, extra = sy
    a = 2.0
    x = [1.0, 2.0, 3.0, 0.5, -1.0, 4.0, -2.5, 0.0]
    y = [10.0, 20.0, 30.0, -0.5, 1.0, -4.0, 2.5, 7.0]
    ref = saxpy_reference(a, x, y)
    got = _run_saxpy_kernel(cxx, emit_sycl_saxpy_c(len(x), "bcir_saxpy"), a, x, y,
                            define="-DBCIR_USE_SYCL", extra=extra)
    for g, r in zip(got, ref):
        # the SYCL device computes the identical SAXPY; agree within a loose float tolerance.
        assert abs(g - r) < 1e-2 * (1.0 + abs(r)), (g, r)


# --- (5) R17 boundary: the bridge step is certified on the wrapped call --------------------------------

def test_r17_certifies_the_bridge_on_a_quantized_saxpy_call():
    # a SAXPY call that consumes quantized lanes carries the bridge's round-trip step in its bound (the
    # multiply-add itself is exact); a dense call does not. Mirrors test_sleef/test_gsl/test_blas.
    def call(qbits):
        return Claim(id=7100, opcode=Opcode.MUL, lane=Lane.U, stride_class=StrideClass.UNIT, count=1,
                     rd=(71,), wr=(72,), op="call.saxpy", domain=Domain.RAM, quantized_bits=qbits)
    assert accuracy_bound(call(8)) == accuracy_bound(call(0)) + 1     # the +1 ULP bridge step
    assert accuracy_bound(call(0)) == 1                              # dense call: the op's own 1 ULP
    assert quantization_error_bound() == 1                          # the R17 grid bound the bridge is held to


# --- (6) SYCL is NOT a link-flag edge: the five library rules still map; a SYCL symbol is unknown ------

def test_sycl_is_not_a_link_flag_edge():
    # CRITICAL architectural assertion: SYCL is a compiler MODE (-fsycl), NOT a `c.call.libm:` -l<lib>
    # external-symbol edge. So NO link rule was added for it (linkflags.py is untouched): a SYCL-ish symbol
    # resolves to None (unknown), while the five existing library rules + libm + libc still resolve.
    assert library_for_callee("sycl_saxpy_kernel") is None          # a SYCL-ish symbol: unknown (no rule)
    assert library_for_callee("bcir_saxpy") is None                 # the emitted kernel name: unknown too
    # no regression on the five existing library classifications + libm + libc-implicit.
    assert library_for_callee("cblas_sgemm") == "-lcblas"           # B5 BLAS still maps
    assert library_for_callee("fftwf_execute") == "-lfftw3"         # B2 FFTW still maps
    assert library_for_callee("LAPACKE_sgesv") == "-llapack"        # #61 LAPACK still maps
    assert library_for_callee("gsl_stats_mean") == "-lgsl"          # #62 GSL still maps
    assert library_for_callee("Sleef_expf1_u10") == "-lsleef"       # #63 SLEEF still maps
    assert library_for_callee("expf") == "-lm"                      # libm still maps
    assert library_for_callee("free") == NO_FLAG                    # libc-implicit, known (not unknown)


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")
    sys.exit(0)
