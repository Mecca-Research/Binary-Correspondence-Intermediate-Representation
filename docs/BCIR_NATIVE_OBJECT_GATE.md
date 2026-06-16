# BCIR Native Object Emission — the Decision Gate

> **Status: DEFERRED (correct).** BCIR does **not** hand-roll an instruction selector /
> ELF+relocation emitter. The warranted path — emit the K_BCIR-planned kernel and let
> the **resident compiler** finish it to a real object — is implemented and tested for
> one target end-to-end (eBPF and x86-64 scalar). This document is the *gate*: the
> explicit criteria under which that decision would flip, and the stop criteria for a
> native-isel experiment if one is ever taken.

Pairs with `BCIR_LOWERING_PLAN.md` §5 #3 and `BCIR_MASTER_ROADMAP.md` row 6
("Native backend — only if warranted ⛔ DEFERRED").

## 1. The decision

BCIR's job is the **planning + verification brain** (the K_BCIR cost algebra, the
verifier, the StreamPack ABI), not a code generator. Two ways to reach machine code
from a plan:

1. **Resident-compiler path (chosen).** Emit portable **C23** (`lower.c_kernel`) or
   **LLVM IR** (`lower.llvm`) for the selected realization and hand it to the
   compiler that already lives next to the deployment — `clang`/`llc`/`lli`, a GPU
   shader compiler, an accelerator runtime. The lane width, bounds, precision, and
   non-aliasing are the K_BCIR plan; instruction selection is the resident backend's
   job. This is `codegen` (LLVM IR → `llc` → object/asm) and `codegen_object_c`
   (C → `clang --target=… -c` → object).
2. **BCIR-native isel (deferred).** A hand-rolled instruction selector + register
   allocator + ELF/relocation writer inside BCIR. Large, target-specific, and a
   permanent maintenance surface that duplicates what LLVM already does well.

**The gate:** pursue (2) only for a *specific target whose economics beat (1)*. Absent
that, (1) is strictly better — it is a real, shippable artifact today, it tracks the
host backend's isel improvements for free, and it keeps BCIR's surface the planning
brain rather than a second-rate codegen.

## 2. The warranted slice (done): one target end-to-end via the resident compiler

`codegen.codegen_object_c(module, result, target)` closes the loop to a **real native
object** without any BCIR-native isel:

| target | element | flags | result | machine |
|---|---|---|---|---|
| `bpf` (eBPF) | `int32_t` (eBPF has no FP) | `-ffreestanding` (libc-free) | relocatable ELF, `bcir_kernel` symbol | `EM_BPF` (247) |
| `x86_64` (scalar) | `int32_t` | host | relocatable ELF | `EM_X86_64` (62) |

The object's ELF header is verified (`\x7fELF` + `e_machine`), so the test asserts a
genuine native object for the expected ISA, not just a zero-exit compile. eBPF is the
canonical "integer-only scalar kernel" the roadmap names; x86-64 scalar is the second
gate target. Tested in `bcir/tests/test_native_object_gate.py` (skips cleanly when
`clang` lacks the `bpf`/cross target).

This is the "one target end-to-end" milestone **without** crossing the gate into
hand-rolled isel — exactly the roadmap's intent.

## 3. GO criteria — what would warrant BCIR-native isel

Build a native backend for a target **T** only when *all* of these hold (and record
the measurement that flips each):

- **G1 — No adequate resident backend.** T has no usable `llc`/`clang`/driver
  backend, or the resident one is unavailable at the deployment point (e.g. a bare
  accelerator with no on-device compiler), so path (1) cannot produce an artifact.
- **G2 — Measured economics.** A native emitter beats "emit C/LLVM + resident
  backend" on a metric that matters *for T* by a margin worth a permanent codegen:
  e.g. ≥ 2× lower emit-to-runnable latency in a driver-resident JIT **or** code the
  resident backend provably cannot express (a verified eBPF map-access pattern, a
  PIM/CIM controller microcode the plan already proves safe).
- **G3 — Bounded surface.** T's realization set is small and integer/scalar (eBPF,
  a fixed accelerator ISA) so the emitter is a finite, verifiable table — *not* a
  general SSA→isel→regalloc pipeline.
- **G4 — Conformance harness ready.** The differential net can cross-check the native
  object against the C/LLVM path (same inputs → same results), so a native backend is
  gated by parity the way every other rail is.

If any of G1–G4 fails, **stay on path (1).** As of today none of the seeded targets
(x86/ARM/RISC-V/NVPTX/eBPF) satisfies G1+G2 — every one has a resident LLVM backend
the codegen path already drives — so the gate holds.

## 4. STOP criteria — if a native-isel experiment is taken

A native-backend experiment for one target is **time-boxed** and abandoned (reverting
to path (1)) the moment any of these trips:

- **S1 — Parity gap.** The native object diverges from the C/LLVM path on the
  differential net and the fix is not a small, local table change.
- **S2 — Surface blow-up.** The emitter needs general register allocation, spilling,
  or multi-block scheduling to cover the target's realizations (it stopped being a
  finite table — G3 violated in practice).
- **S3 — Margin evaporates.** Re-measuring G2 on real silicon shows < 1.5× advantage
  over path (1), or the resident backend closed the gap.
- **S4 — Maintenance tax.** The native path needs per-toolchain-version upkeep that
  exceeds the planning core's churn.

The experiment ships **only** if it clears G1–G4 *and* never trips S1–S4 across the
measurement window; otherwise the deliverable is the negative result + this doc.

## 5. Current verdict

Path (1) is implemented, tested, and produces real objects for one target end-to-end
(eBPF + x86-64). No seeded target meets the GO bar (every one has a resident LLVM
backend). **Native isel stays deferred; the gate stands.** Revisit when a deployment
target with no resident backend (G1) and a measured ≥ 2× economics case (G2) appears —
most plausibly a bare PIM/CIM controller or a driver-resident eBPF JIT under a latency
SLA.
