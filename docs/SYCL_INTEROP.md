# SYCL interop — a backend channel + a differential oracle, never on the legality path

SYCL (a single-source C++ host runtime + device programming model, compiled to SPIR-V) joins BCIR in
exactly **two** roles, and is held out of a third on purpose:

1. **A backend CHANNEL.** SYCL/SPIR-V is a modeled GPU channel
   ([`channels/sycl.channel.json`](../channels/sycl.channel.json), `name: "sycl_spirv"`, `kind: "gpu"`,
   triple `spirv64-unknown-unknown`). The planner prices it with the **K_BCIR cost model** on a warp-ish
   subgroup profile (lane widths `[1, 32]`, `warp: 32`, a 2-tier L1/HBM hierarchy, the fabric/sync cost
   dims) and routes a `data_parallel` / `matmul` / `reduce` claim to it like any other channel — no core
   edit, a pure plugin (see [`channels/README.md`](../channels/README.md) and
   [`HETEROGENEOUS_CHANNELS.md`](HETEROGENEOUS_CHANNELS.md)). Cross-device placement cost is now wired into
   `orchestrate`, so the `sycl_spirv` channel is chosen for a claim **only when its compute savings beat
   the host↔device transfer** (the `fabric`/`sync` cost of moving the claim's operands on/off the device) —
   a small operand offloads, a large one stays on the host. See the "Cross-device placement cost
   (fabric/sync)" section of [`HETEROGENEOUS_CHANNELS.md`](HETEROGENEOUS_CHANNELS.md).

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

## Resident dispatch (the channel executes)

Beyond pricing + routing + the standalone differential oracle, the `sycl_spirv` channel now has a
host-side **resident dispatcher** ([`bcir/lower/sycl_dispatch.py`](../bcir/lower/sycl_dispatch.py)
`SyclDispatcher`): a module [`orchestrate`](../bcir/channels.py) places onto a tower including
`sycl_spirv` can be **run end-to-end** through [`gem.execute`](../bcir/gem/execute.py), with the
sycl-placed claim dispatched through the emitted SYCL kernel.

`SyclDispatcher.run_saxpy(a, x, y)` emits `emit_sycl_saxpy_c`, compiles it as single-source C++
(`-std=c++17 -O2`), and runs it on fixed data: the **device path** (`-DBCIR_USE_SYCL -fsycl`, the SYCL
runtime JITs SPIR-V and submits to the device) when a SYCL compiler is detected (`icpx` / `acpp` /
`clang++ -fsycl` that compiles+links a SYCL probe), else the **portable scalar C++ fallback**. Its `mode`
property reports which path ran (`"sycl-device"` / `"fallback"` / `"unavailable"`). `build_execute_kernels`
wires each sycl-placed claim into the `kernels` dict `execute()` consumes — the callable reads the claim's
inputs from a `store`, runs the dispatcher, and writes the outputs back — so the routed claim genuinely
**executes**, round-trip-verified against `saxpy_reference` to float round-off. The SPIR-V codegen identity
is reachable via `try_emit_spirv`, which drives the existing `codegen(…, target_name="spirv64")` path
(triple `spirv64-unknown-unknown`, marker `OpEntryPoint`) — best-effort: stock `llc` has no SPIR-V backend,
so it returns a clean `"no SPIR-V backend"` note and never crashes.

**Honest depth** (the same framing as [`CPP_HANDOFF_BOUNDARY.md`](CPP_HANDOFF_BOUNDARY.md)): there is no
SYCL toolchain and no GPU on CI. What is **real + gated on CI**: the dispatcher seam, the end-to-end
`execute()` round-trip via the portable fallback ([`bcir/tests/test_sycl_dispatch.py`](../bcir/tests/test_sycl_dispatch.py),
gate marker `#sycl-dispatch` in [`tools/cpp/check_sycl.sh`](../tools/cpp/check_sycl.sh)), and the SPIR-V
emission attempt. What is **gated / self-skipping** (needs hardware): the `-fsycl` device submission and a
real SPIR-V backend in `llc` — detect-and-skip, never faked. The claim is precise: the channel has a
resident dispatch path that *executes routed work* (portable path on CI, SYCL/SPIR-V when present),
not that it runs on a real GPU. The dispatcher is host-side, lives **above** the G8 boundary, and is a
graded L2/L3 backend like the G8 orchestrator — it produces data and **never** renders or alters a
legality verdict (the two-truth quarantine holds: `verify()` is unchanged by a dispatch).

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
