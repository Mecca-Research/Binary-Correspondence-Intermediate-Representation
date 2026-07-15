# BCIR Native Backend — Feasibility, Cost & Development Roadmap

> **Verdict (unchanged, now quantified): a *general* BCIR-native instruction selector is NOT
> warranted — keep riding the resident backend (LLVM/clang).** The feasible, valuable slice — *if and
> only if* a deployment target ever clears the gate — is a **time-boxed, bounded, table-driven
> direct-encoder for ONE target** (a driver-resident eBPF JIT, or a no-`clang` WASM encoder extending
> the existing `stackify` path), gated by the existing dual-rail differential net. This document is the
> deeper feasibility/cost analysis and the phased plan that `BCIR_NATIVE_OBJECT_GATE.md` references; the
> gate doc owns the GO/STOP *criteria*, this doc owns the *cost model and the build roadmap*.

Pairs with [`BCIR_NATIVE_OBJECT_GATE.md`](../BCIR_NATIVE_OBJECT_GATE.md) (the decision
gate), [`BCIR_MASTER_ROADMAP.md`](../BCIR_MASTER_ROADMAP.md) §4.2, and the
[`BCIR_MACHINE_CODE_HAL_ISA_AUDIT.md`](../BCIR_MACHINE_CODE_HAL_ISA_AUDIT.md) MC10–MC14
register.

---

> **Cross-ref (wave 14):** the machine-code/HAL/ABI/ISA gap audit
> (`BCIR_MACHINE_CODE_HAL_ISA_AUDIT.md`) confirmed this document's boundary: the missing
> tools (disassembler/lister, hex dump, pack-level linker, peek/poke, ROP v2) are all
> NATIVE-side plan/artifact tools, not codegen — the gate below is unaffected.

## 1. What "native backend" means here

"Native backend" = **BCIR emits machine code (a real `.o` / executable) directly, without LLVM/clang** —
i.e. BCIR itself does instruction selection (ISel), register allocation (RA), scheduling, ABI lowering,
and ELF object emission (sections, symbols, relocations) for a target ISA.

This is distinct from everything BCIR does today, which is **"plan + verify, then hand the realization to
the resident compiler."** The question the user raised — *do the ASM1–3b edges and the cfront/emit/lower
machinery cycle back to Clang and lose the performance paths?* — is answered in
the master roadmap's resident-toolchain boundary and re-derived in §2 below: **no.** The native instructions are preserved at
each edge; only ISel/RA/scheduling of the *surrounding* code is delegated to LLVM, by design, because
that is exactly the part LLVM does world-class and a hand-rolled backend would only lose to. The "native
backend" is the *option to stop delegating that part* — a large, permanent investment — and this report
prices it.

---

## 2. Current state — the codegen spectrum BCIR already populates

BCIR is **not** "a thing that emits C and stops." It already spans most of a real codegen spectrum;
native ISel is the single missing point at the far end. Every existing rail keeps the K_BCIR plan
(lane width, tiling, bounds, precision, non-aliasing) and delegates only final instruction selection.

| # | Path | Module | Output | Final codegen by | Status |
|---|------|--------|--------|------------------|--------|
| 1a | C source (faithful + kernel) | `bcir/lower/c_kernel.py`, `frontends/cfront/emit.py` | C23 (incl. **verbatim inline asm**, native port-I/O / fence asm under `--target`) | `clang`/`gcc` | shipped |
| 1b | Textual LLVM IR | `bcir/lower/llvm.py` | standard SSA (`<W x float>` vector ld/op/st) | `llc` / `lli` | shipped |
| 1c | Object / asm via `llc` | `bcir/codegen/codegen.py` (+ `codegen/targets.py`) | relocatable **ELF** (eBPF, x86-64, aarch64) | `llc` | shipped, gate-tested |
| 1d | Real `.o` via C | `codegen.codegen_object_c` | relocatable ELF, `bcir_kernel` symbol, verified `e_machine` | `clang -c` | shipped (`test_native_object_gate.py`) |
| 1e | In-process JIT | `bcir/lower/jit.py` | LLVM IR linked + run | `lli` | shipped |
| 1f | WASM module | `bcir/lower/wasm.py` | `\0asm` binary, validated, run under node | `clang --target=wasm32` + `wasm-ld` | shipped |
| 1g | **Direct stack-bytecode encoder (NO LLVM)** | `bcir/lower/stackify.py` | postfix stack-op sequence → per-format encoders (`to_wasm`/`to_jvm`/`to_cil`) | **BCIR itself** | shipped (the existing direct-encode precedent) |
| 1h | MLIR law rail → LLVM IR | `mlir/lib/passes/BCIRConvertToLLVM.cpp` (`bcir.barrier`→`llvm.fence`, gem ops→loops) | LLVM dialect → LLVM IR | LLVM | shipped |
| **2** | **BCIR-native machine-code ISel** | — | `.o` with BCIR's own ISel/RA/reloc | **BCIR (none)** | **DEFERRED (this report)** |

Three things follow from the table:

1. **The asm edges are not "lowered back to C and re-optimized."** Inline asm is emitted verbatim
   (ASM1, opaque trusted edge); port-I/O emits the real `in`/`out` (ASM2); fences emit the real
   `mfence`/`dmb ish`/`fence rw,rw` under `--target` or the C11 builtin by default (ASM3/SEG6/SEG7).
   The instruction reaches the assembler intact — zero abstraction penalty. Measured: **match on dense
   (0.98–1.00×) vs clang**, with wins (1.3–14× on irregular memory) coming from the K_BCIR *plan*, not
   from bypassing LLVM (`CLANG_COMPARISON.md`).
2. **A direct (non-LLVM) encoder already exists** for stack machines (`stackify.py` → WASM/JVM/CIL
   bytecode). That is the cheap end of "native": a stack target needs **no register allocator and no
   scheduler**. It is the precedent and the template for any future direct-encode work.
3. **The expensive end** — register-machine ISel for x86-64/aarch64/riscv with RA + scheduling +
   relocations + DWARF — is the one box that is empty, and is what §3–§5 price.

---

## 3. What a *general* native backend requires (and why it is enormous)

A production register-machine backend is, irreducibly, the following components. For each: what it is,
and the realistic cost to do it *well enough to beat path (1)* rather than merely run.

| Component | What it must do | Cost reality |
|---|---|---|
| **Instruction selection** | claim/MLIR DAG → target instructions (pattern match, addressing modes, immediates, two-address fixups) | Per ISA. LLVM's x86 `.td` is ~tens of thousands of lines. A *bounded scalar-integer* table is small; a *general* one is not. |
| **Register allocation** | live-range analysis, interference, coloring/linear-scan, **spilling**, calling-convention constraints | The hard part. A real RA with spilling is a multi-thousand-line, bug-prone subsystem. (`stackify` avoids it entirely — stack machines have no registers.) |
| **Instruction scheduling** | latency/port modeling, reorder for ILP; matters most on in-order/VLIW | Optional for correctness, decisive for *performance* vs LLVM. Reproducing LLVM's schedulers is the point where "native" stops being cheaper. |
| **ABI / calling convention** | arg/return classification, stack frame layout, callee-saved, varargs, struct-by-value | Per ISA + per OS. Subtle, conformance-critical (must match the system ABI exactly to link with anything). |
| **Object emission** | ELF sections, symbol table, **relocations** (the part the gate names), alignment, `.text`/`.data`/`.bss`, `e_machine` | Mechanical but exacting; relocations are a long tail (PC-rel, GOT/PLT, TLS). |
| **Debug / unwind info** | DWARF line tables, `.eh_frame` / `.debug_*` | Required for a *usable* toolchain (`bcir-cc`-as-`cc`); a large, format-heavy effort on its own (`BCIR_MASTER_ROADMAP.md:1623`). |
| **Linking** | either emit relocatable `.o` for the system linker (cheaper) or a full linker (don't) | Emit `.o` + use `ld`; never write a linker. |
| **Per-ISA encoders** | the actual byte encodings (ModR/M/SIB for x86, fixed-width for aarch64/riscv) | One per ISA. Fixed-width ISAs (aarch64, riscv, eBPF) are *far* cheaper than x86's variable-length encoding. |

**Order-of-magnitude:** a *general, optimizing, multi-ISA* native backend is an LLVM-sized undertaking
(LLVM's backends are collectively millions of LOC and hundreds of person-years). **BCIR will not build
that, and should not** — it would duplicate LLVM, lose to it on schedule quality, and add a permanent
per-toolchain-version maintenance tax (gate STOP criterion **S4**). This is not a close call.

**The only feasible shape** is the gate's **G3 — a bounded, integer/scalar, finite-table emitter for ONE
target**, with **no general RA and no general scheduler** (either a stack target, or a target with so few
registers / so simple an ABI that allocation is a fixed assignment). That is a *different, much smaller*
artifact, priced in §5.

---

## 4. The gate, restated and assessed (status: all GO criteria unmet)

From `BCIR_NATIVE_OBJECT_GATE.md`. Build native ISel for a target **T** only when **all** hold:

- **G1 — No adequate resident backend** for T at the deployment point (e.g. a bare accelerator with no
  on-device compiler). *Status: unmet for every seeded target (x86/ARM/RISC-V/NVPTX/eBPF all have a
  resident LLVM backend).*
- **G2 — Measured ≥2× economics** on a metric that matters for T (e.g. emit-to-runnable latency in a
  driver-resident JIT), or code the resident backend provably can't express. *Status: unmeasured /
  unmet — no deployment case has produced the number.*
- **G3 — Bounded surface**: T's realization set is small + integer/scalar → a finite verifiable table,
  not an SSA→isel→regalloc pipeline. *Status: achievable only for eBPF / a fixed accelerator ISA / a
  stack target — NOT for general x86/aarch64.*
- **G4 — Conformance harness ready**: the dual-rail differential net cross-checks the native object
  against the C/LLVM path (same inputs → same results). *Status: **MET** — the differential net + the
  StreamPack ABI + `test_native_object_gate.py` already exist; this is the one criterion BCIR has
  already paid for.*

STOP criteria (abandon a time-boxed experiment if any trips): **S1** parity gap that isn't a small table
fix; **S2** surface blow-up (needs general RA/spilling/multi-block scheduling); **S3** margin evaporates
(<1.5× on real silicon, or the resident backend closes the gap); **S4** maintenance tax exceeds the
planning core's churn.

**Today: G4 holds, G1+G2 do not, G3 holds only for bounded targets. The gate stands; native ISel stays
deferred.**

---

## 5. The candidate bounded targets, priced and ranked

If/when a deployment forces the question, these are the only shapes worth considering. Each is a
*finite-table direct encoder*, gated by the differential net (G4), time-boxed by S1–S4.

### 5.1 Ranking

| Rank | Target | Why it could clear G1+G2 | RA needed? | Encoding cost | Feasibility | Net assessment |
|---|---|---|---|---|---|---|
| **A** | **Direct WASM encoder (no `clang`)** | G1: a deploy point with no clang/wasm-ld but a WASM runtime; G2: emit-to-runnable latency | **No** (stack machine) | Low (LEB128 + section framing) | **Highest** | Best first experiment: extends `stackify.py`, no RA, no scheduler, validator is the conformance oracle. |
| **B** | **Driver-resident eBPF JIT** | G1: in-kernel/agent with no compiler; G2: ≥2× emit-to-runnable under a latency SLA | Minimal (eBPF: 10 regs, fixed ABI, no spill for bounded kernels) | Low (fixed 64-bit insns) | **High** | The gate's named case. Integer-only, verifier-friendly, finite table. Real G2 case exists (driver JIT). |
| **C** | **Bare PIM/CIM accelerator microcode** | G1: **met by construction** (no resident backend exists) | N/A (fixed microcode) | Target-specific | **Conditional** | The gate's other named case. Only real when such silicon is in hand; G3 holds if the ISA is a fixed table. |
| D | General x86-64/aarch64 native `.o` | nothing — every one has LLVM | **Yes** (general RA) | High (x86 variable-length) | **Rejected** | Violates G3 in practice (S2); duplicates LLVM; permanent S4 tax. Do not build. |

### 5.2 Why A (WASM) is the right *first* experiment if the gate ever opens

- **No RA, no scheduler** — the two hardest, most bug-prone subsystems are absent for a stack machine.
- **The encoder is small** — WASM is LEB128 integers + a handful of typed sections; `stackify.py`
  already produces the postfix op sequence and a `to_wasm` renderer (mnemonics). The remaining work is
  *byte encoding + module framing + a function/type/export section*, not a compiler.
- **The conformance oracle is free** — a WASM **validator** (and `node`/`wasmtime` execution) already
  cross-checks every module against the path-1f output, so G4 is satisfied with the existing net.
- **It proves the thesis cheaply** — "BCIR can emit a runnable artifact with no resident compiler in the
  loop" — without ever touching register allocation. If even this trips S1–S4, the negative result is
  decisive and cheap.

---

## 6. Development roadmap (executed ONLY if the gate opens for a specific T)

Strictly dependency-ordered under the master roadmap's §4.2 gate. Each
phase has an explicit GO/NO-GO; the default at every gate is **revert to path (1)**.

- **N0 — Conformance substrate (do first; reusable regardless).** Formalize the native-object
  differential harness as a first-class rail: `(claims) → {path-1 artifact, native artifact} → run on
  the same seeded inputs → assert identical results + identical StreamPack provenance digest`. This is
  mostly **assembling existing pieces** (`test_native_object_gate.py` + the differential net + the
  executor `runtime/c/bcir_exec.c`). **This is the only phase worth doing pre-gate**, because it also
  hardens path (1) and is the G4 net every later phase needs. *GO to N1 only on a measured G1+G2 case.*
- **N1 — One bounded direct-encoder, time-boxed (target A or B).** Implement the finite-table encoder
  for the single chosen target (WASM byte-encoder extending `stackify`, or an eBPF insn table). **No
  general RA, no general scheduler** — if either becomes necessary, **S2 trips → STOP**. Gate every
  output through the N0 net (S1). *GO to N2 only if parity holds with table-only changes.*
- **N2 — Measure G2 on the real deployment.** Emit-to-runnable latency (JIT case) or
  expressiveness (a pattern the resident backend can't emit) on real silicon/runtime. **S3 trips if
  <1.5×.** *GO to N3 only if ≥2× sustained across the measurement window.*
- **N3 — Ship as a gated channel, not a default.** The native encoder registers as a **channel**
  (`bcir/channel_plugin.py`: codegen identity, calibration, provenance=`real`), selected only for T,
  never replacing path (1) for any target that has a resident backend. Permanent **S4** watch.

**Exit:** the experiment ships **only** if it clears G1–G4 and never trips S1–S4 across the window;
otherwise the deliverable is *the negative result + this document* (a legitimate, intended outcome).

---

## 7. What to do *now* (and how it de-risks any future native work)

The native backend is correctly deferred. The immediately valuable work is on the **path-(1) spectrum**,
which is exactly the user's stated sequencing and also the best possible *preparation* for a future
bounded encoder:

1. **Harden the asm-edge / per-ISA emit surface and carry it into the MLIR→LLVM path** (the user's
   step 2). Widening the native per-ISA fence/port-I/O coverage and threading the SEG6/7 MemOrdering
   through `BCIRConvertToLLVM` strengthens the *exact* surface a bounded native encoder would later
   consume — and ships value today on the warranted rail (real instructions, LLVM does ISel).
2. **Promote N0 (the native-object conformance net) to a standing CI rail** whenever convenient — it
   hardens path (1) now and is the prerequisite G4 net for any future N1.
3. **Keep native ISel deferred** until a concrete deployment presents G1 (no resident backend) **and** a
   measured G2 (≥2×) — most plausibly a bare PIM/CIM controller or a driver-resident eBPF JIT under a
   latency SLA.

---

## 8. Bottom line

- A **general** BCIR-native backend is an LLVM-scale, permanent-maintenance undertaking that would
  *lose* to LLVM on schedule quality and *duplicate* what it does well. **Do not build it.** This is the
  considered, measured position, not an absence of ambition.
- BCIR's value is the **planning + verification brain** (K_BCIR cost algebra, the R1–R23 verifier, the
  StreamPack ABI, the dual-rail conformance net) layered *above* a world-class backend it gets for free.
  The asm edges already preserve native instructions; nothing is lost to "cycling back to Clang."
- The **feasible, bounded** native slice — a time-boxed, table-driven, RA-free direct encoder for **one**
  target (WASM first, eBPF JIT second), gated by the existing differential net and the STOP criteria —
  is real, but only warranted once a deployment clears **G1+G2**. Until then, the encoder's negative
  result is the deliverable, and the live work is hardening the path-(1) emit spectrum.
- **Recommended next step (executing the user's plan):** harden the asm-edge emit and widen the native
  per-ISA surface into the MLIR→LLVM path; stand up the N0 conformance net opportunistically; revisit
  native ISel only at a measured gate opening.
