# SYCL interop — a backend channel + a differential oracle, never on the legality path

SYCL (a single-source C++ host runtime + device programming model, compiled to SPIR-V) joins BCIR in
exactly **two** roles, and is held out of a third on purpose:

1. **A backend CHANNEL.** SYCL/SPIR-V is a modeled GPU channel
   ([`channels/sycl.channel.json`](../channels/sycl.channel.json), `name: "sycl_spirv"`, `kind: "gpu"`,
   triple `spirv64-unknown-unknown`). The planner prices it with the **K_BCIR cost model** on a warp-ish
   subgroup profile (lane widths `[1, 32]`, `warp: 32`, a 2-tier L1/HBM hierarchy, the fabric/sync cost
   dims) and routes a `data_parallel` / `matmul` / `reduce` claim to it like any other channel — no core
   edit, a pure plugin (see [`channels/README.md`](../channels/README.md) and
   [`HETEROGENEOUS_CHANNELS.md`](HETEROGENEOUS_CHANNELS.md)).

2. **A differential ORACLE.** Before committing to a resident SYCL driver, we **measure** that a
   heterogeneous device reproduces BCIR's own deterministic reference. The canonical data-parallel op is
   **SAXPY** (`out[i] = a*x[i] + y[i]`): its reference is a trivially exact per-element multiply-add
   ([`bcir/kbcir/sycl_saxpy.py`](../bcir/kbcir/sycl_saxpy.py) `saxpy_reference`), and the emitted
   single-source C++ kernel ([`bcir/lower/c_kernel.py`](../bcir/lower/c_kernel.py) `emit_sycl_saxpy_c`)
   runs a SYCL `parallel_for` on the device path (`-fsycl`, `BCIR_USE_SYCL`) and a **portable scalar C++
   fallback** otherwise — both compute the identical SAXPY to float round-off. The toolchain-gated gate
   ([`tools/cpp/check_sycl.sh`](../tools/cpp/check_sycl.sh), wired into
   [`tools/c/check_runtime.sh`](../tools/c/check_runtime.sh)) compiles + runs the fallback as the real
   reference-verification work and the device path only when a real SYCL compiler is present (it
   self-skips on CI). The R17 Q8↔float32↔Q8 bridge at the boundary is the **only** certified error source:
   SAXPY is exact arithmetic (mul + add, no transcendental), so the bridged result is perturbed by at most
   `(|a| + 1) · e` per element (the input round-trip alone).

3. **NOT a legality verdict.** SYCL lives **above** the G8 C++ boundary
   ([`CPP_HANDOFF_BOUNDARY.md`](CPP_HANDOFF_BOUNDARY.md)) — it is a host runtime / device programming model
   on the C++ side of the deterministic C/IR rail. It MAY dispatch a frozen StreamPack, but its **dynamic
   runtime scheduler must never touch the legality path**. Legality remains the C/IR rail's verdict
   (the two-truth quarantine): a SYCL device's measured agreement *informs* (a differential oracle), it
   never *legislates*.

## The bright line: SYCL is a compiler MODE, not a `c.call.libm` link edge

This is the critical architectural difference from the five Area-B library wraps
(BLAS/FFTW/LAPACK/GSL/SLEEF), which mirror SYCL **in spirit** but are fundamentally different in mechanism:

| | The 5 library wraps | SYCL |
|---|---|---|
| Mechanism | call an external symbol through the `c.call.libm:` FFI edge | a single-source C++ **compiler mode** (`-fsycl`) |
| Kernel | `c.call.libm:cblas_sgemm` / `Sleef_expf1_u10` / … | emitted SYCL C++ source (a `parallel_for`) — no symbol |
| Link rule | one `-l<lib>` rule in `linkflags.py` (`-lcblas`, `-lsleef`, …) | **none** — `-fsycl` is a driver flag, not a `-l` |
| Compiler | any C compiler (`cc`) | a SYCL C++ compiler (`icpx -fsycl` / `acpp` / `clang++ -fsycl`) |

Consequences honored in this segment:

- **No `c.call.libm:` label** is minted for SYCL (`emit_sycl_saxpy_c` carries none — asserted in
  [`bcir/tests/test_sycl_channel.py`](../bcir/tests/test_sycl_channel.py)).
- **No rule was added to** [`bcir/frontends/cfront/linkflags.py`](../bcir/frontends/cfront/linkflags.py).
  The five library link rules still resolve (`cblas_sgemm → -lcblas`, …, `Sleef_expf1_u10 → -lsleef`),
  `expf → -lm`, `free → NO_FLAG`; a SYCL-ish symbol (`sycl_saxpy_kernel`, `bcir_saxpy`) resolves to
  `None` (unknown) — proving SYCL adds no link edge.
- The kernel is emitted as **C++** (no `extern "C"`), consistent with SYCL living on the C++ side of the
  G8 boundary; the fallback compiles with a plain `g++` / `clang++` (no SYCL header needed).

No `mlir/` changes and no new R-laws were introduced (exactly like the SLEEF wrap). SYCL is measured as a
heterogeneous backend — priced, routed, and differentially verified — with its dynamic runtime held off
the deterministic rail.

See also: [`HETEROGENEOUS_CHANNELS.md`](HETEROGENEOUS_CHANNELS.md),
[`CPP_HANDOFF_BOUNDARY.md`](CPP_HANDOFF_BOUNDARY.md), [`channels/README.md`](../channels/README.md).
