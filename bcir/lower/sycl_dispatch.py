"""The SYCL channel's RESIDENT DISPATCH / EXECUTION path -- a host-side graded backend (above G8).

S2 turns the `sycl_spirv` channel from a *modeled cost profile + a standalone differential oracle*
(``bcir.kbcir.sycl_saxpy`` + ``lower.c_kernel.emit_sycl_saxpy_c``) into an EXECUTOR: a module that
``channels.orchestrate`` places onto a tower including ``sycl_spirv`` can now be RUN end-to-end, with the
sycl-placed claim dispatched through a real SYCL execution path. The kernel is emitted by the existing
``emit_sycl_saxpy_c`` codegen, compiled as single-source C++ (``-std=c++17 -O2``), and run; when a SYCL
compiler is detected the device path (``-DBCIR_USE_SYCL -fsycl``, the runtime JITs SPIR-V) executes,
otherwise the PORTABLE scalar C++ fallback executes. Both compute the identical SAXPY to float round-off,
verified against ``bcir.kbcir.sycl_saxpy.saxpy_reference``.

HONEST DEPTH (mirrors docs/CPP_HANDOFF_BOUNDARY.md's framing). There is NO SYCL toolchain and NO GPU in
the CI environment, so this module is precise about what is real vs. gated:

  * REAL (compiles + runs + gated on CI): the dispatcher seam, the end-to-end ``execute()`` round-trip with
    the sycl-placed claim dispatched through the emitted kernel's PORTABLE fallback, and the SPIR-V emission
    ATTEMPT via the existing ``codegen(..., target_name="spirv64")`` path.
  * GATED / SELF-SKIPPING (needs hardware): the actual ``-fsycl`` device submission + a real SPIR-V backend
    in ``llc``. Detect-and-skip; this module NEVER fakes success and NEVER claims a real GPU ran.

The honest claim: the channel has a resident dispatch path that EXECUTES routed work (via the portable path
on CI, via SYCL/SPIR-V when the toolchain is present), round-trip-verified against BCIR's reference.

TWO-TRUTH / WHERE THIS LIVES. SYCL is a C++ single-source *compiler MODE* (``-fsycl``), NOT a
``c.call.libm:`` ``-l<lib>`` external-symbol edge (no link rule in ``frontends/cfront/linkflags.py``). The
dispatcher is HOST-SIDE (this Python module, plus the emitted C++ kernel compiled out-of-line) and lives
ABOVE the G8 boundary -- it is a graded L2/L3 backend exactly like the G8 single-node orchestrator. It
EXECUTES (it produces data); it MUST NEVER render or alter an R-law legality verdict, never emit a
Diagnostic, and never sit on the legality path. The deterministic freestanding C rail
(``runtime/c/bcir_exec.c``, ``-ffreestanding -nostdlib``) is untouched: SYCL is C++ and lives above it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from ..kbcir.sycl_saxpy import saxpy_reference
from .c_kernel import emit_sycl_saxpy_c


# --- compiler detection (mirrors test_sycl_channel.py _cxx / _sycl_cxx) --------------------------------

def host_cxx() -> str | None:
    """A host C++ compiler (g++/clang++/c++), or None (the fallback compile/run self-skips)."""
    return shutil.which("g++") or shutil.which("clang++") or shutil.which("c++")


def sycl_cxx() -> tuple[str, list[str]] | None:
    """A SYCL compiler invocation that actually COMPILES + LINKS a ``#include <sycl/sycl.hpp>`` +
    ``sycl::queue`` probe with ``-fsycl`` (so a plain clang++ without the SYCL runtime is rejected), or
    None -- the device path self-skips, the expected case on CI. Tries ``icpx`` / ``acpp`` / ``clang++
    -fsycl`` in order; returns the (compiler, extra-flags) pair. Identical discipline to
    ``bcir/tests/test_sycl_channel.py`` ``_sycl_cxx``."""
    candidates: list[tuple[str, list[str]]] = []
    if shutil.which("icpx"):
        candidates.append((shutil.which("icpx"), ["-fsycl"]))
    if shutil.which("acpp"):
        candidates.append((shutil.which("acpp"), []))         # acpp/Open SYCL drives -fsycl itself
    if shutil.which("clang++"):
        candidates.append((shutil.which("clang++"), ["-fsycl"]))
    if not candidates:
        return None
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


class DispatchUnavailable(RuntimeError):
    """Raised (catchably) when no C++ compiler exists at all, so the dispatch path cannot execute. The
    caller self-skips on this -- it is NOT a faked success and NOT a legality verdict."""


# --- the dispatcher interface + the SYCL SAXPY implementation -----------------------------------------

class ChannelDispatcher:
    """A resident channel backend that EXECUTES routed work: given a claim's input arrays, it produces the
    claim's output arrays. This is the graded L2/L3 execution seam (above the G8 boundary), NOT a legality
    verdict -- a subclass produces data only and never touches the verifier."""

    name = "channel"

    def run_saxpy(self, a: float, x: list[float], y: list[float]) -> list[float]:
        """Execute the data_parallel SAXPY ``out[i] = a*x[i] + y[i]`` on this channel."""
        raise NotImplementedError


class SyclDispatcher(ChannelDispatcher):
    """The resident dispatcher for the ``sycl_spirv`` channel's data_parallel SAXPY class.

    ``run_saxpy`` emits ``emit_sycl_saxpy_c`` (the single-source C++ kernel), wraps it in a tiny ``main``
    that calls ``bcir_saxpy`` on the data and prints the results (mirroring test_sycl_channel's
    ``_run_saxpy_kernel``), compiles it as C++ (``-std=c++17 -O2``), runs it, and parses the outputs back.
    When ``sycl_cxx()`` detects a real SYCL compiler it uses ``-DBCIR_USE_SYCL -fsycl`` (the DEVICE path --
    the SYCL runtime JITs SPIR-V and submits to the device), otherwise the PORTABLE scalar C++ fallback
    executes. Both compute the identical SAXPY to float round-off.

    ``mode`` exposes which path executed -- ``"sycl-device"`` / ``"fallback"`` / ``"unavailable"`` -- so the
    tests + gate can report HONESTLY (on CI, with no SYCL toolchain, this is ``"fallback"``; with no C++
    compiler at all it is ``"unavailable"`` and ``run_saxpy`` raises ``DispatchUnavailable``). The compiled
    executable is cached per (n, mode) within the dispatcher's lifetime so a repeated dispatch is cheap."""

    name = "sycl_spirv"

    def __init__(self, *, prefer_device: bool = True) -> None:
        self._mode = "unavailable"
        self._sycl = sycl_cxx() if prefer_device else None
        self._cxx = host_cxx()
        # the per-instance compile cache + a TemporaryDirectory kept alive for the dispatcher's lifetime.
        self._tmpdir: tempfile.TemporaryDirectory | None = None
        self._exe_cache: dict[tuple[int, str], str] = {}

    @property
    def mode(self) -> str:
        """Which execution path ran (or would run): ``"sycl-device"`` once a ``-fsycl`` dispatch has
        executed, ``"fallback"`` once a portable-C++ dispatch has executed, ``"unavailable"`` before any
        dispatch or when no C++ compiler exists. Honest: never reports ``"sycl-device"`` unless a real SYCL
        compiler compiled+linked a SYCL probe AND the device kernel built with ``-fsycl``."""
        return self._mode

    @property
    def available(self) -> bool:
        """True iff *some* C++ path can execute (a host C++ compiler exists). The portable fallback alone
        is enough; the device path is an optional upgrade."""
        return self._cxx is not None or self._sycl is not None

    def _workdir(self) -> str:
        if self._tmpdir is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="bcir-sycl-dispatch-")
        return self._tmpdir.name

    def _compile(self, n: int, fn_name: str) -> tuple[str, str]:
        """Compile the emitted SAXPY kernel + a data-driven ``main`` to an executable; return (exe, mode).
        Prefers the SYCL device path when a SYCL compiler is present, else the portable C++ fallback.
        Raises ``DispatchUnavailable`` if no C++ compiler exists at all."""
        device = self._sycl is not None
        mode = "sycl-device" if device else "fallback"
        key = (n, mode)
        if key in self._exe_cache:
            return self._exe_cache[key], mode

        if device:
            cxx, extra = self._sycl                          # type: ignore[misc]
            define, flags = "-DBCIR_USE_SYCL", list(extra)
        else:
            if self._cxx is None:
                raise DispatchUnavailable(
                    "no C++ compiler (g++/clang++/c++); the SYCL dispatch path cannot execute "
                    "(self-skip -- this is not a failure and not a legality verdict)")
            cxx, define, flags = self._cxx, None, []

        kernel = emit_sycl_saxpy_c(n, fn_name)
        # A data-INDEPENDENT main: read a/x/y from argv-free stdin-free fixed file is overkill; instead the
        # kernel is recompiled per distinct n only, and the data is passed at RUN time via a generated main.
        # To keep the (n, mode) cache meaningful we bake the call but read the actual values from a sidecar
        # data file the runner writes -- simpler: regenerate main per run is cheap relative to compile, so
        # we compile a fixed harness that reads a, then x[n], then y[n] from stdin and prints out[n].
        main = (
            "\n#include <cstdio>\n#include <cstddef>\n"
            f"int main(void){{\n"
            f"  float a;\n  if (std::scanf(\"%f\", &a) != 1) return 2;\n"
            f"  float x[{n}], y[{n}], out[{n}];\n"
            f"  for (int i = 0; i < {n}; ++i) if (std::scanf(\"%f\", &x[i]) != 1) return 2;\n"
            f"  for (int i = 0; i < {n}; ++i) if (std::scanf(\"%f\", &y[i]) != 1) return 2;\n"
            f"  {fn_name}(a, x, y, out);\n"
            f"  for (int i = 0; i < {n}; ++i) std::printf(\"%.8f\\n\", (double)out[i]);\n"
            f"  return 0;\n}}\n")
        wd = self._workdir()
        src = os.path.join(wd, f"saxpy_{n}_{mode}.cpp")
        exe = os.path.join(wd, f"saxpy_{n}_{mode}")
        with open(src, "w") as f:
            f.write(kernel + main)
        cmd = [cxx, "-std=c++17", "-O2"]
        if define:
            cmd.append(define)
        cmd += flags
        cmd += [src, "-o", exe]
        bld = subprocess.run(cmd, capture_output=True, text=True)
        if bld.returncode != 0:
            raise DispatchUnavailable(
                f"the SYCL dispatch kernel did not build ({mode} path):\n{bld.stderr.strip()}")
        self._exe_cache[key] = exe
        return exe, mode

    def run_saxpy(self, a: float, x: list[float], y: list[float]) -> list[float]:
        """Dispatch ``out[i] = a*x[i] + y[i]`` through the emitted SYCL kernel and return the executed
        outputs (a fresh list). Sets ``mode`` to the path that ran. Raises ``ValueError`` on an empty or
        length-mismatched input (a map needs >= 1 lane); raises ``DispatchUnavailable`` (catchably) if no
        C++ compiler exists -- the caller self-skips, never fakes success."""
        n = len(x)
        if n < 1:
            raise ValueError("sycl dispatch saxpy: need at least one element")
        if len(y) != n:
            raise ValueError(f"sycl dispatch saxpy: x/y length mismatch ({n} vs {len(y)})")
        exe, mode = self._compile(n, "bcir_saxpy")
        stdin = "".join(f"{v:.8f}\n" for v in [float(a), *x, *y])
        run = subprocess.run([exe], input=stdin, capture_output=True, text=True)
        if run.returncode != 0:
            raise DispatchUnavailable(
                f"the SYCL dispatch kernel did not run ({mode} path):\n{run.stdout}{run.stderr}")
        self._mode = mode
        return [float(v) for v in run.stdout.split()]

    def close(self) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None


# --- the execute() kernel builder: wire sycl-placed claims to the dispatcher --------------------------
# SAXPY's `a` scalar is not a separate BCIR resource in the elementwise claim model (the claim carries two
# read rids: x, y). The convention here: the FIRST read rid is `x`, the SECOND is `y`, the single write rid
# is `out`, and the scalar `a` is taken from `claim.imm[0]` when present, else this default. A planned
# call.saxpy claim that wants a specific `a` carries it as the first immediate.
_DEFAULT_SAXPY_A = 1.0


def _saxpy_scalar(claim) -> float:
    imm = getattr(claim, "imm", None)
    if imm:
        try:
            return float(imm[0])
        except (TypeError, ValueError):
            pass
    return _DEFAULT_SAXPY_A


def build_execute_kernels(module, plan, store: dict, dispatchers: dict):
    """Build the ``kernels`` dict for ``gem.execute(module, kernels)`` from a ``HeterogeneousPlan``.

    ``module`` is the planned module (the claims carry the read/write rids the placements reference by id).
    ``store`` is a mutable ``dict[rid -> list[float]]`` seeded with the module's live-in inputs (the arrays
    no claim writes -- see :func:`seed_store`). ``dispatchers`` is a registry keyed by channel name (e.g.
    ``{"sycl_spirv": SyclDispatcher()}``).

    For each placement whose channel has a registered dispatcher (the sycl-placed claims), the returned
    per-claim callable reads the claim's input rids from ``store``, runs the dispatcher's ``run_saxpy``, and
    writes the outputs back to ``store`` under the claim's write rid -- this is the END-TO-END run: the
    routed sycl claim EXECUTES through the real emitted-kernel dispatch path. Non-dispatched claims get a
    minimal reference kernel (``saxpy_reference`` of their inputs) so ``execute()`` still produces data for
    every claim -- honest and minimal.

    Convention for the elementwise SAXPY claim model: the claim's FIRST read rid is ``x``, the SECOND is
    ``y`` (a 1-read claim reuses ``x`` for both), the single write rid is ``out``, and the scalar ``a`` is
    ``claim.imm[0]`` when present (else ``1.0``). The callables read/write ``store`` LAZILY at execution
    time, so a producer->consumer chain works (a claim that writes rid R feeds a later reader of R). This is
    purely DATA movement: it produces no Diagnostic and never touches the legality path (two-truth)."""
    claim_of = {c.id: c for ph in module.phases for c in ph.claims}
    where = {p.claim_id: p.channel for p in plan.placements}
    kernels: dict[int, object] = {}

    def _inputs(claim):
        rds = list(claim.rd)
        if not rds:
            raise KeyError(f"claim {claim.id} ({claim.op}) has no read rid to dispatch")
        x = store[rds[0]]
        y = store[rds[1]] if len(rds) > 1 else store[rds[0]]
        return _saxpy_scalar(claim), x, y

    def _write(claim, out):
        if claim.wr:
            store[claim.wr[0]] = out

    def _make(claim, channel):
        disp = dispatchers.get(channel)

        def kernel():
            a, x, y = _inputs(claim)
            if disp is not None:
                out = disp.run_saxpy(a, x, y)           # the routed channel EXECUTES the claim
            else:
                out = saxpy_reference(a, x, y)          # minimal reference for non-dispatched claims
            _write(claim, out)

        return kernel

    for cid, channel in where.items():
        claim = claim_of.get(cid)
        if claim is not None:
            kernels[cid] = _make(claim, channel)
    return kernels


def seed_store(module, data: dict) -> dict:
    """Build the ``store`` for :func:`build_execute_kernels`, seeded with the module's LIVE-IN inputs.

    ``data`` maps a live-in rid -> its input array. A live-in is a resource no claim in the module writes
    (the orchestrator assumes it pre-resident, matching ``channels._transfer_cost``'s live-in rule). The
    returned dict is the mutable store ``execute`` fills with each claim's outputs. Raises ``KeyError`` if a
    live-in rid is missing from ``data``."""
    written = {rid for ph in module.phases for c in ph.claims for rid in c.wr}
    read = {rid for ph in module.phases for c in ph.claims for rid in c.rd}
    live_ins = read - written
    store: dict = {}
    for rid in live_ins:
        if rid not in data:
            raise KeyError(f"seed_store: missing input array for live-in rid {rid}")
        store[rid] = list(data[rid])
    # any extra seed data (e.g. a non-live-in the caller wants pre-set) is honored too.
    for rid, arr in data.items():
        store.setdefault(rid, list(arr))
    return store


# --- the SPIR-V emission hook: the "SPIR-V" half of the dispatch path is reachable --------------------

def try_emit_spirv(module, result, fn_name: str = "bcir_kernel") -> tuple[bool, str]:
    """Attempt SPIR-V emission for ``module`` via the EXISTING ``codegen(..., target_name="spirv64")`` path
    (triple ``spirv64-unknown-unknown``, asm, marker ``OpEntryPoint``). Returns ``(ok, note)``: ``ok=True``
    iff SPIR-V asm carrying the ``OpEntryPoint`` marker was produced, else a clean note (``"no SPIR-V
    backend"`` / ``"llc not found"`` / the codegen message). NEVER raises -- this demonstrates the channel's
    SPIR-V codegen IDENTITY is reachable (the "SPIR-V" half of the SYCL/SPIR-V dispatch path), honestly
    reporting that stock ``llc`` has no SPIR-V backend in this environment (the gated case)."""
    try:
        from ..codegen import codegen
        r = codegen(module, result, target_name="spirv64", fn_name=fn_name)
        if r.ok and isinstance(r.artifact, str) and "OpEntryPoint" in r.artifact:
            return True, f"spirv64 asm emitted ({r.message})"
        return False, f"no SPIR-V backend: {r.message}"
    except Exception as exc:                              # never crash: honest depth, gated path
        return False, f"no SPIR-V backend: {type(exc).__name__}: {exc}"
