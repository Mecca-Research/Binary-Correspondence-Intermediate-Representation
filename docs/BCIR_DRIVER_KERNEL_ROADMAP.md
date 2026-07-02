# BCIR Driver / Kernel Roadmap — bring-up ordering, language placement, and the pre-driver hardening gate

> **Scope.** This document is the deep-dive that **Phase D** of
> [`BCIR_MASTER_ROADMAP.md`](BCIR_MASTER_ROADMAP.md) §5.7 points at: *how* BCIR grows a
> verified device-driver / bare-metal capability — what to harden **before** any driver code,
> which driver/firmware **layers** belong in direct assembly vs C vs C++, and the realistic
> **dependency ordering** of the kernel/firmware infrastructure around the first (UART) driver.
> It does **not** restate Phase C (the verifiable-C keystone) or the channel-plugin contract;
> it sequences the work that rides on top of them. Pairs with
> [`CFRONT_GUIDE.md`](CFRONT_GUIDE.md) (the ASM1/2/3 edges + the C subset),
> [`CPP_HANDOFF_BOUNDARY.md`](CPP_HANDOFF_BOUNDARY.md) (the G8 C++ boundary),
> [`BCIR_NATIVE_OBJECT_GATE.md`](BCIR_NATIVE_OBJECT_GATE.md) (native isel stays gated), and
> [`BCIR_ML_AI_INTEGRATION_ROADMAP.md`](BCIR_ML_AI_INTEGRATION_ROADMAP.md) (the parallel ML arc).

## 0. The one-paragraph orientation

BCIR is the **planning + verification brain that emits verified kernels** — it is *not* a
bootloader, an OS, or firmware. For drivers that means BCIR **emits a verified driver/parser
kernel** and the resident compiler finishes it to a freestanding object; BCIR never *implements*
UEFI/ACPI/SMBIOS/TPM. Read every section below through that lens. Two corrections fall straight
out of it and frame the whole roadmap:

1. **A *polled* UART needs essentially no platform infrastructure.** It is the canonical "first
   thing that works" precisely because it needs no interrupts, no timers, no ACPI, no PCI, no
   paging. Its *only* genuine prerequisites — an ordered register-access edge (MMIO **or** x86
   port-I/O) and a bounded poll loop — **already exist and are dual-rail-verified**. The first
   useful driver is, in BCIR terms, *already met*: `runtime/c/cfront_driver_uart.c` +
   `runtime/c/uart_regs.h` compile end-to-end through `bcir-cc`, R1–R18-clean and
   Clang-equivalent, with **zero inline asm**.
2. **Every firmware/security spec the user named is, at its core, a byte-level data structure** (a
   table, a log, or a command/response wire format). BCIR's relationship to all of them is the
   *same shape*: **emit a verified parser/marshaller kernel that consumes the structure** — never
   implement the firmware. This is exactly what the K_BCIR verifier + the StreamPack ABI are built
   for (bounded indexing, length-prefixed records, R1–R21 legality).

The work therefore sequences as: **(Part I) a cheap pre-driver hardening gate → (Part II) the
asm/C/C++ layering that says where each line of a driver lives → (Part III) the dependency ladder
that orders everything after the UART → (Part IV) the concrete slice plan.**

---

## Part I — The pre-driver hardening gate (Phase 0: do this first)

The user's instinct is correct: harden the machinery the drivers will stand on **before** building
drivers. An audit of the existing test / red-team / sanitizer / parity surface (1768 Python
conformance tests across 143 files, 81 MLIR FileCheck tests, the `check_runtime.sh` byte-identity
gate, the RT1–RT7 red-team series, the 500k-run StreamPack libFuzzer campaign) shows coverage is
**strong on the system / parser / IR / C-runtime / fence axis** and on ML *correctness* — but it
surfaces a small number of genuine, cheap, high-leverage gaps. Ranked:

| ID | Gate | Why it matters before drivers | Size | Priority |
|---|---|---|---|---|
| **H1** | **Wire `tools/c/sanitize_cfront.sh` into CI.** | This ASan+UBSan+LSan+Valgrind harness over the cfront twin is **already written and well-built** (clang+gcc builds, all `cfront_*.c` fixtures, a 300-valid/400-malformed seeded fuzz campaign, a leak pass, a UBSan-trap stage, a vacuous-gate guard) — but it is **referenced nowhere in `.github/`**, so it runs on no CI job. The new SEG6/SEG7 **fence** code and the `c.asm` lowering in `bcir_cfront.c` therefore get **no memory-safety gate anywhere**. Because the script globs `cfront_*.c`, it *already* sweeps the fence fixture (`cfront_atomic.c`) the moment it runs — the coverage is written but **dark**. Drivers are exactly "the C twin under malformed device input," so this is the single highest-ROI item. | ~1 hr | **MUST — do first** |
| **H2** | **Area-B numerical red-team (`test_area_b_redteam.py`).** | The 7 `c.call.libm:` wraps (LAPACK / BLAS / GSL / SLEEF / libcerf / FFTW-1D/2D) are **happy-path + reference-verified only** — no NaN/Inf, no ill-conditioned matrices, no dimension edges (n=0,1), no error-code paths. **libcerf is the sharpest risk:** its tests *intentionally* stay in `\|x\|≤3` to dodge the `exp(x²)` overflow, so the library's entire full-range robustness claim is **never validated**. Drivers that touch the numerical edges will hit exactly these regimes. | ~2–3 d | **MUST (numerical path)** |
| **H3** | **Fence/asm C-twin malformed-input fuzz.** | The new fence + `c.asm` C-twin paths are exercised by the dual-rail *parity* loop (which proves output correctness) but not by a *memory-safety-under-malformed-input* fuzzer of their own. Route `cfront_atomic.c` + asm fixtures through the existing `cfuzz` corruptor in `sanitize_cfront.sh`. Mostly already wired by H1. | ~1 d | **SHOULD (parallel)** |
| **H4** | **ML convergence demos for the predict-only models.** | ✅ **LANDED** (the `(H4)`-marked tests): E3 readout training (`test_transformer.py`), E4 Tier-A/Tier-B training (`test_recurrent.py`), the E5 closed-form Gaussian-NB **fit-quality** gate (`test_classical.py` — E5 stays predict-only *by design*, the E7 finding), and the E6 autoencoder full-weight reconstruction descent + k-means inertia monotonicity (`test_unsupervised.py`). Honest scope: E3 trains a linear readout over the *frozen* block (transcendentals sit outside the closed-set Tape); E4 Tier-A now trains the RNN weights THEMSELVES end-to-end (BPTT through `tape.var` weights) and the GRU has its own readout-convergence gate; E6 trains every weight to a near-zero absolute threshold. All four route through the shared convergence gate (`bcir/tests/_convergence.py`). Remaining deepening (E3 block-weight training via transcendental adjoints) is an ML-track item, not driver-critical. | — | **DONE** |
| **H5** | **Native-codegen-outside-LLVM honesty + the gated WASM byte-encoder.** | The honesty half is ✅ **LANDED**: `bcir/tests/test_stackify_exec.py` assembles the `to_jvm()` emit into a real `.class` and **executes it on a real JVM** against a StackOp-interpreter oracle, plus an always-on 60-expression cross-encoder semantic differential over the JVM/CIL/WASM mnemonics; the CIL emit is likewise **assembled by the real `ilasm` and executed under `mono`** against the same oracle (the oracle CI job installs `mono-devel`; a clean documented skip without a CLR). The **direct, RA-free WASM byte-encoder** that [`BCIR_NATIVE_BACKEND_FEASIBILITY.md`](BCIR_NATIVE_BACKEND_FEASIBILITY.md) names as the "best first experiment" **does not exist yet** and per that doc's own gate it **stays deferred** (no deployment clears G1+G2). | byte-encoder gated | **DONE (honesty) / DEFER (byte-encoder, per the gate)** |

**Things that are already well-covered — do not manufacture work:** the SEG6.1/SEG7
order-parameterized **fences** have full dual-rail parity (`test_c_cfront_barrier.py`,
`test_fence_order_edge_cases_dual_rail`, 29 barrier tests, the MLIR `memory_ordering.mlir`, and
per-ISA assemble-on-native gating); the **`bcir.asm` MLIR op (SEG8.1)** is FileCheck-gated
(round-trip + `-convert-bcir-to-llvm` lowering + verifier negatives) **and** now
**assemble-smoke-tested** — `tools/wsl/check_asm_lowering.sh` pipes the lowered IR through
`mlir-translate-20 --mlir-to-llvmir | llc-20 -filetype=obj` (a real `.o`, exit 0) and greps the
emitted asm for the expected instruction, covering `bcir.asm`/`bcir.portio`/`bcir.creg_*`/the
volatile MMIO ops/`bcir.msr_*` (`rdmsr`/`wrmsr`). (This closed a real masking bug: the GCC
`%w1`/`=a,Nd` portio templates
text-checked but `llc` rejected them — the lowering now emits the LLVM-IR `${N:mod}` /
`={ax},N{dx}` forms, GCC-template translation runs in `bcir.asm`, and a `+` read-write output is
rejected at lowering.)

**Phase-0 exit criterion:** H1 + H2 merged and green (H3 folded into H1). H4/H5 may run in parallel
or after; they do not gate drivers.

---

## Part II — The language-placement model (the "more important question")

**Which driver/firmware layers go in direct assembly, which in C, which in C++?** BCIR already
encodes most of the well-established systems answer as *concrete, verified machinery*. The model is
three tiers, and the dividing principle is **verifiability**: assembly is the *trust boundary*
(opaque, not internally verified), C is the *verified bulk*, C++ is *orchestration above the rail*.

### Tier 1 — Direct assembly: the irreducible trust edges

**Principle:** assembly only where there is *no C abstraction* — CPU state the C abstract machine
cannot name (mode, control registers, the stack pointer before C runs, exact register save/restore)
or device/ordering primitives with no portable C spelling. In BCIR these are **trusted opaque
effect edges**: off the legality value-path, carried verbatim, marked `barriered` so the verifier
never reorders/fuses/DCEs them — but their *internal* semantics are trusted, not verified. **Keep
this tier as small as possible; every line here is unverified.**

| Edge | Status | Notes |
|---|---|---|
| Inline asm (**ASM1**, `c.asm` / `c.asm.volatile`; MLIR `bcir.asm`) | ✅ **have** | The general escape hatch; `"memory"`/volatile ⇒ `barriered`. MLIR twin lowers to `llvm.inline_asm` (SEG8.1). |
| Port-I/O (**ASM2**, `c.portio.in/out.{b,w,l}`; MLIR `bcir.portio`) | ✅ **have** | x86 `in`/`out`, isolated `__ioport` I/O space, barriered. Non-x86 → honest diagnostic. MLIR twin `bcir.portio` lowers to `llvm.inline_asm` (SEG8.2, **landed**), in LLVM-IR `${N:mod}` / `={ax},N{dx}` form (the post-clang-frontend translation of the cfront GCC templates), **assemble-smoke-tested** through `llc`. |
| Memory fences (**ASM3**, `c.fence[.acquire/.release]`; MLIR `bcir.barrier`) | ✅ **have** | Per-ISA `mfence`/`dmb`/`fence`, `memory_order`-parameterized (SEG6/7). |
| **Entry / reset stub + stack setup before C** | ❌ **missing** | A `bcir.entry` naked-function edge (no prologue, sets `sp`, jumps to C `main`). The genuine floor of bring-up. |
| **Control-register access** (CR0/CR2/CR3/CR4/CR8) | ✅ **have (MLIR rail, D1.3)** | `bcir.creg_read`/`bcir.creg_write` lower to `llvm.inline_asm "mov %crN, $0"` / `"mov $0, %crN"`. The cfront-rail twin (`c.creg`/`c.msr`) is a later increment (today cfront expresses these as a raw ASM1 blob). |
| **Model-specific-register access** (`rdmsr`/`wrmsr`) | ✅ **have (MLIR rail, D1.4)** | `bcir.msr_read`/`bcir.msr_write` take a runtime `i32` MSR index (→`ECX`) and a clean `i64` value; the read is a multi-output `rdmsr` (`={ax},={dx}`) reassembled to i64, the write splits the i64 across `EDX:EAX` into `wrmsr`. Both `has_side_effects` + `~{memory}`, assemble-smoke-checked. |
| **Descriptor-table loads** (`lgdt`/`lidt`/`ltr`) + segment reloads | ❌ **missing** | Raw-asm only today. |
| **Interrupt entry trampoline** (full-frame save/restore + `iret`) | ❌ **missing — largest new surface** | Needs the currently-rejected `asm goto` *or* a dedicated trampoline op. The ISR *body* is verified C; only the entry/`iret` shim is asm. |

### Tier 2 — C: the verifiable bulk (where most of a driver lives)

**Principle:** everything expressible with C's value/memory semantics belongs in C, because that is
exactly what BCIR lowers to a claim graph and **verifies under R1–R21**. A driver's *entire logic
body* is here today: register protocols and read-modify-write, **MMIO accessors**, polled loops
with bounded spins, bitfield control words, device state machines, and table parsers
(ACPI/SMBIOS/PCI-config walks).

> **MMIO is NOT a gap — it is first-class.** A `volatile`-qualified register is *not* an asm
> escape: it is a `Domain.MMIO` resource with ordered, `barriered`, provably-RAM-disjoint volatile
> load/store (cfront L5, shipped + parity-gated). The UART fixture proves it: `u->SR`, `u->DR`,
> `u->BRR`, `u->CR` are plain `volatile uint32_t` struct members that lower to ordered MMIO access
> — no `c.mmio` edge needed; the volatile-qualified type *is* the edge. *The MLIR-rail MMIO gap is
> now closed (D1.2, landed):* `bcir.volatile_load` / `bcir.volatile_store` are first-class ops that
> lower to `llvm.inttoptr` + a **volatile** `llvm.load`/`llvm.store` (mirroring the cfront
> `*(volatile T*)(intaddr)` emit), so the MMIO accessor is first-class on the law rail as it already
> is in cfront — not just the `Domain.MMIO` enum value riding the barrier/hazard machinery.

### Tier 3 — C++: orchestration above the rail (the G8 boundary)

**Principle:** C++ buys *runtime-dynamic, allocating, OO/RAII orchestration* — the abstractions the
freestanding C rail deliberately lacks. It sits **above** the verified C kernels as a
dispatcher/manager, talking to them only across a stable `extern "C"` ABI over an immutable,
CRC-sealed frozen artifact (the **two-truth quarantine**: C++ may schedule/shard/retry/replicate;
it may **never** mutate artifact bytes or re-derive an R-law verdict — a C++ failure is a
`HandoffError` operational fault, never a legality verdict). See
[`CPP_HANDOFF_BOUNDARY.md`](CPP_HANDOFF_BOUNDARY.md).

**For a single device driver, C++ buys nothing** and *costs* the freestanding constraints
(no exceptions/RTTI, freestanding libstdc++); `_BitInt` + bitfields already express register maps
in verifiable C. C++ earns its place **only above the per-device layer**: a multi-device
**driver-model / device-manager** that enumerates and places devices at runtime, RAII resource
handles, retry/replication — the existing `Orchestrator` hierarchy (single-node real;
distributed dispatch stubbed).

### The one-line rule

> **Per-device MMIO/register/IRQ logic → verified C (Tier 2). The irreducible CPU-state and
> ordering primitives → trusted asm edges (Tier 1: ASM1/2/3 today, plus the boot/IRQ edges still to
> build). The cross-device lifetime + policy → C++ at the G8 boundary (Tier 3).**

The new Tier-1 edges to build, in rough bring-up priority: **(1)** `bcir.portio` MLIR op (SEG8.2,
the named next slice — reuses the SEG8.1 `bcir.asm` lowering); **(2)** entry/reset stub + stack
init; **(3)** typed control-register/MSR edges; **(4)** descriptor-table loads (`lgdt`/`lidt`);
**(5)** the interrupt trampoline; **(6)** `bcir.volatile_load/store` in the MLIR rail so MMIO is
first-class there as it already is in cfront.

---

## Part III — Kernel/firmware dependency ordering (what's before vs after UART)

### The premise correction

For a **polled** UART the answer to "what infrastructure do we need first?" is **almost nothing**.
Every item the user named is *independent of*, *after*, or *not BCIR's job* — **none is a genuine
prerequisite**. The named list describes an *interrupt-driven, enumerated, measured-boot* platform
(a much later phase), not the first useful driver.

### Item-by-item classification

| Item | vs. polled UART | BCIR's realistic role | Why |
|---|---|---|---|
| **ISA interface** (legacy x86 port-I/O bus, COM1 @ 0x3F8) | **BEFORE** (the *only* prereq class) | **Emit it — done** (ASM2 + ASM1 + ASM3) | A UART is reached via MMIO *or* x86 port-I/O; one ordered-access mechanism is the irreducible floor. Both ship. |
| **Interrupt controller** (8259 PIC / APIC / IOAPIC) + IDT *(implied gateway item)* | **AFTER** | Emit a verified PIC/APIC-program kernel + IDT-builder (the IDT is a data table) | The gateway from *polled* to *interrupt-driven* anything. |
| **HPET** (High Precision Event Timer) | **AFTER** (and after interrupts) | Emit a verified HPET-program kernel; *consume* the ACPI HPET table for its base | A timer is useful once interrupts deliver its events; PIT (8254) is the simpler predecessor. |
| **PCI Firmware** (config space via 0xCF8/0xCFC) | **AFTER UART; BEFORE any PCI device** | Emit a verified PCI-enumeration kernel (config walk via the *same* port-I/O edge) + BAR/capability parser | The enumeration substrate; nothing PCI can be driven until config space is walked. |
| **e1000 NIC** | **AFTER** (PCI enum + interrupts + DMA) | Emit a verified e1000 driver kernel (descriptor-ring builder + register programming) | Depends on PCI enum (find device/BARs), then MMIO + DMA ring management. |
| **USB HID driver** | **AFTER** (deepest stack) | Emit verified descriptor-parser + transfer-ring kernels; the HID report-descriptor parser is a textbook verifier target | PCI enum → host controller (xHCI/EHCI) → USB stack → HID class. |
| **UEFI / ACPI** | **out-of-scope to *implement*; AFTER to *consume*** | Emit a verified ACPI **static-table parser** (RSDP→XSDT→MADT/MCFG/HPET/FADT) | BCIR consumes ACPI data (roadmap already names "PCIe/ACPI data" ingestion); it will not be UEFI or ship an AML interpreter. |
| **SMBIOS** | **independent / AFTER; out-of-scope to *produce*** | Emit a verified SMBIOS table-walk **parser** | A firmware-populated platform-inventory blob; read-only structured traversal, ideal StreamPack target. Orthogonal to driving any device. |
| **TCG PC Client Platform Firmware Profile** | **out-of-scope (firmware *behavior* spec)** | At most a verified **event-log parser** + digest-marshalling helpers | Dictates what firmware measures into which PCRs and when; BCIR does not perform measured boot. |
| **TPM 2.0 / PC Client TPM PP** | **out-of-scope to *be* a TPM; strong fit to *talk to* one** | Emit a verified TPM2 **command marshaller/unmarshaller** kernel | TPM is external silicon; the PP is a Common Criteria cert of *the chip*. The TPM2 command/response structures are length-prefixed bounded byte formats — a StreamPack-ABI sweet spot. Needs a TIS transport edge (just above the ISA floor). |
| **Virtual memory** (paging / MMU) | **independent / AFTER** | Emit a verified page-table-builder / MMU-config kernel | A flat-address polled driver runs fine without paging; needed only for higher-half kernels, protection, user/kernel split. |
| **Virtual machines** (VT-x/AMD-V) | **AFTER (capstone)** | Mostly out-of-scope as a hypervisor; plausibly emit verified VMCS/VMCB-builder or virtio device-model parser kernels | Being a VMM is an OS/hypervisor concern; BCIR's role is verified data-structure kernels. Top of the ladder. |

### The smallest prereq set for the first driver

For the polled UART, the **complete** prerequisite set is exactly: **(1)** an ordered register-access
edge (MMIO **or** port-I/O — both shipped), **(2)** a bounded poll loop (cfront L6, shipped),
**(3)** transitively, the verifiable-C path that compiles+verifies+attests them (Phase C, done for
this driver). **There is no fourth item.**

### The "after UART" ladder

Each rung unlocks the next; nothing here is a UART prerequisite — it is all *upward* from the floor.

```
RUNG 0  (FLOOR — done)   ISA edge: ordered MMIO  -OR-  x86 port-I/O (0x3F8)
                          └─► POLLED UART  ◄── the first useful driver (already shipped)

RUNG 1  Interrupt substrate:  8259 PIC / APIC / IOAPIC  +  IDT (data table)
                          └─► unlocks every interrupt-driven device
                          └─► INTERRUPT-DRIVEN UART (the natural follow-on)

RUNG 2  Timers:           PIT (8254, simplest)  →  HPET  [HPET base via ACPI table]
                          └─► needs RUNG 1 to deliver tick interrupts

RUNG 3  Platform-table parsers (orthogonal; buildable in parallel from RUNG 1):
                          ACPI (MADT→IOAPIC, MCFG→ECAM, HPET→base), SMBIOS, TCG event-log
                          └─► feeds RUNGs 1 / 2 / 4

RUNG 4  PCI enumeration:  config-space walk via 0xCF8/0xCFC (same edge as UART)
                          └─► PREREQUISITE for every PCI device driver below

RUNG 5  PCI device drivers (each needs RUNG 1 + RUNG 4 + DMA):
                          ├─ e1000 NIC          (descriptor rings + MMIO regs)
                          ├─ USB host ctlr (xHCI/EHCI) ─► USB stack ─► USB HID
                          └─ TPM 2.0 driver     (command marshaller; TIS transport)

RUNG 6  Virtual memory:   paging / MMU page-table builder
                          └─► higher-half kernel, protection, user/kernel split

RUNG 7  Virtual machines: VMCS/VMCB builders, virtio device-model parsers (capstone)
```

Two ordering facts to pin: **PCI enumeration (RUNG 4) is strictly before any PCI device driver
(RUNG 5)** — e1000/USB can't be found until config space is walked, and the config accessor is the
*same* port-I/O edge the UART can use; and **interrupts (RUNG 1) gate the interrupt-driven version
of everything**, including a faster UART and HPET's usefulness.

---

## Part IV — The sequenced slice plan

Putting Parts I–III together as the dependency-ordered program (each slice keeps the dual-rail
discipline + a CI-green draft PR per the established cadence):

### Phase 0 — Pre-driver hardening gate
- **D0.1 (H1)** — wire `sanitize_cfront.sh` into the CI `c-runtime` job. *(MUST, ~1 hr — the single
  highest-ROI item; lights up the dark ASan/UBSan/Valgrind gate over the new fence + asm C-twin code.)*
- **D0.2 (H2)** — `test_area_b_redteam.py`: NaN/Inf, ill-conditioning, dimension edges across the 7
  `c.call.libm:` wraps; the `\|x\|>3` libcerf regime first. *(MUST for the numerical path.)*
- **D0.3 (H3)** — fold fence/asm fixtures into the malformed-input fuzz corruptor. *(SHOULD; mostly
  delivered by D0.1.)*
- *Parallel/deferred:* **H4** ML convergence demos, **H5** native-codegen honesty (gated).

### Phase 1 — Driver foundation (the asm-edge + MLIR surface drivers stand on)
- **D1.1 = SEG8.2 code slice** — ✅ **landed.** The **`bcir.portio` MLIR op**: x86 `in`/`out`
  emitted *as* inline asm, reusing the SEG8.1 `bcir.asm` → `llvm.inline_asm` lowering (same generic
  attribute-list builder, so it compiles on LLVM-20 *and* the CI's LLVM-22). A `#bcir.port_dir`
  direction enum (`in`/`out`) + a `{8,16,32}`-bit width (b/w/l); `in` = 1 operand (port) + 1 result,
  `out` = 2 operands (value, port) + 0 results. The lowering emits the six templates in **LLVM-IR
  operand syntax** (`inb ${1:w}, ${0:b}` … / `outb ${0:b}, ${1:w}` …) with constraints `"={ax},N{dx}"`
  (in) / `"{ax},N{dx}"` (out), always `has_side_effects`. The fully-qualified forms are what LLVM’s x86
  backend needs — cfront’s GCC `%w1`/`=a,Nd` spellings are correct on the C→clang rail but `llc` rejects
  them on this MLIR→LLVM path, so the MLIR rail emits clang’s post-frontend translation directly. Op +
  verifier + FileCheck round-trip/lowering/negatives **and an assemble-smoke-test** (the lowered IR is
  piped through `mlir-translate-20 | llc-20` to a real `.o`, asserting `inb %dx, %al` etc.;
  `mlir/test/passes/portio_roundtrip.mlir`, `portio.mlir`, `portio_verify_neg.mlir`,
  `asm_lowering_smoke.mlir`), dual-rail with the cfront `c.portio.*` claim. (Like `bcir.asm`, the
  oracle→MLIR emitter wiring is a later increment.)
- **D1.2** — ✅ **landed.** `bcir.volatile_load` / `bcir.volatile_store` so MMIO is first-class in the
  law rail as it already is in cfront emit: the ops take the resolved integer register address and lower
  to `llvm.inttoptr` + a **volatile** `llvm.load`/`llvm.store` (mirroring the cfront
  `*(volatile T*)(intaddr)` emit; `volatile` set via the generated setter so it is LLVM-20/22 stable).
- **D1.3** — the **boot/CPU-state asm edges**: the typed **control-register** edge
  (`bcir.creg_read`/`bcir.creg_write` for CR0/CR2/CR3/CR4/CR8) is ✅ **landed** (CR3 is the paging
  gateway, the most foundational of these).
- **D1.4** — the typed **model-specific-register** edge (`bcir.msr_read`/`bcir.msr_write` →
  `rdmsr`/`wrmsr`) is ✅ **landed**: a runtime `i32` MSR index (→`ECX`) and a clean `i64` value
  (the `EDX:EAX` split is handled in the lowering — multi-output `rdmsr` reassembled, `wrmsr`
  fed the split halves), `has_side_effects` + `~{memory}`, assemble-smoke-checked. The remaining
  boot edges land *as a driver demands them* (not speculatively): entry/reset stub + stack init;
  `lgdt`/`lidt`. The interrupt trampoline (the largest new surface) lands only when going
  interrupt-driven (RUNG 1).

### Phase 2 — The first driver, formalized, then the ladder
- **D2.1** — promote the existing polled UART (`cfront_driver_uart.c` + `uart_regs.h`) into a real
  **channel-backed driver** behind a `channel.json` plugin (the Phase-D "first real driver" loop):
  generate/JIT the channel's kernel from the imported register map, drain `TelemetryRing` → encode
  a telemetry frame → push it out the verified UART TX loop (wiring the *missing consumer* the
  frozen `bcir_telemetry_frame` codec already feeds).
- **D2.2 → D2.7** — climb the RUNG 1→7 ladder (interrupts+IDT → timers → table parsers → PCI enum →
  e1000/USB/TPM drivers → paging → VMs), each as a verified emit-a-kernel slice.

### What stays explicitly out of scope (by design, not omission)
Implementing UEFI / ACPI-AML / measured boot / being a TPM or a VMM. BCIR emits **verified
table/log/command parser-marshaller kernels that consume firmware data**; it is the planning +
verification brain that emits verified kernels, not the OS/firmware that hosts them. Native
instruction selection likewise **stays gated** (see [`BCIR_NATIVE_OBJECT_GATE.md`](BCIR_NATIVE_OBJECT_GATE.md));
the warranted path is the resident compiler finishing BCIR-emitted freestanding C/LLVM.

---

## 5. Decision summary

- **Before drivers:** turn on the hardening that already exists (H1) and red-team the numerical
  wraps (H2). Cheap, high-leverage, and the things a driver will actually stress.
- **Layering:** a driver is *mostly verified C* (Tier 2, including first-class MMIO), a *thin
  trusted-asm floor* (Tier 1: ASM1/2/3 today + the boot/IRQ edges to build), and — only above the
  per-device layer — *C++ orchestration at G8* (Tier 3).
- **Ordering:** a polled UART needs **nothing** new; it's already shipped. Everything the user named
  is independent / after / out-of-scope, and the firmware/security specs are *parser-kernel*
  opportunities, not implementation targets. The post-UART work is the RUNG 1→7 ladder.
- **Next code slice:** the Phase-0 hardening gate (D0.1 sanitizer-in-CI, D0.2 Area-B red-team), the
  `bcir.portio` op (D1.1), the `bcir.volatile_load/store` MMIO ops (D1.2), the control-register edge
  (D1.3), and the MSR edge (D1.4) are all **landed**; the remaining boot edges (entry/reset stub +
  stack init, `lgdt`/`lidt`) land *as a driver demands them*, then Phase 2's first channel-backed
  UART driver.

---

## Part V — BCIR kernel feasibility audit + the Linux compatibility plan (2026-07-02)

> A feasibility audit answering the second-part questions: can BCIR's tropical optimizer +
> components produce **adaptive drivers / intelligent microkernels** that bridge GCC/Clang/
> WASM/JVM/CLI(.NET)/IDL(CORBA)/SPIR/SGML-family/SDL/various IRs and optimize every layer
> above; does each driver need its own ISA/firmware-tailored ML layer; which ISO/industry
> standards bind the driver/kernel; and how to break down the Linux master kernel (export
> hardware tables, keep POSIX/ABI/syscall backward-compat, triage IPC) while leaving room for
> a Linux-independent AI microkernel. Draws on the Phase-4 toolchain/POSIX research and the
> Linux-kernel implementation-pattern analysis in the prior-project corpus. Conservative
> verdicts; a *plan*, not a claim of built work. The deep Linux breakdown is itself a
> multi-quarter research program — this is the initial, gateable plan.

### V.0 One-paragraph orientation

BCIR is already "a driver is a compiler" (Part II): the cfront rail compiles a register-map
header to a verified claim graph, plans it, and emits a freestanding kernel. The kernel
vision extends that in three moves: (1) make the driver **adaptive** — the same claim graph,
re-planned per live Θ and per channel, with a learned prior choosing realizations (the
"intelligent kernel"); (2) make BCIR a **multi-backend hub** — the claim graph is
backend-neutral, so one driver lowers to GCC/Clang-C, WASM, JVM/CLI stack bytecode, SPIR-V,
or a native object, each already a partial rail today; (3) keep **Linux backward-compatible**
through a POSIX/ABI compatibility layer while the AI microkernel grows independently. The
binding reality from the toolchain research: a full self-hosting toolchain + POSIX + libc +
compat layer is a 3–5 year, 15–25 engineer effort — so the plan is *incremental interop
first, independence later*, never a big-bang kernel.

### V.1 Adaptive drivers / intelligent kernels from the tropical optimizer

**The mechanism (what makes a driver "adaptive").** A driver's logic is a claim graph.
K_BCIR already selects realizations per target capability and per Θ; the adaptive driver adds
the calibration loop *resident* (Part V.6): the driver measures its own device (MMIO latency,
IRQ rate, queue depth, thermal) through the telemetry ring, folds it into Θ, and **re-plans**
— choosing poll-vs-interrupt, batch size, DMA descriptor count, prefetch distance, DVFS clock
— under a frozen L1 prior, verified by the same R-laws. This is the recursive-intelligence
seed (ML/AI roadmap §2 Phase E) applied to a device: *learning which realization gives the
best measured energy/latency on the current silicon*, with the two-truth quarantine keeping
the learned choice off the legality path (a mis-learned batch size is slower, never unsafe).

**Is it real?** The *substrate* is real (planner, Θ, RCSP budgets, the resident-calibloop
design, the barriered MMIO/port-I/O/fence edges). The *adaptive loop on a device* is the
measured-replan result, which stays **rig-gated** (Part I H5 / master roadmap §5.4) — feasible
and staged, not demonstrated. **The kernel-side precedent is sched-ext/BPF:** Linux itself now accepts pluggable,
verifier-checked scheduling policies as BPF programs, and BCIR's `codegen_object_c` already
emits **real eBPF ELF objects** — so a *learned-then-frozen* scheduling/IO policy can be
compiled by BCIR and installed into a stock Linux kernel today, long before any freestanding
kernel exists. **Verdict: feasible; the first slices are (a) a BCIR-compiled frozen policy
installed via sched-ext/eBPF on stock Linux (lowest-risk "intelligent kernel component"
demo), and (b) the D2.1 channel-backed UART re-planning its poll loop under measured Θ,
degrade-honest in CI.**

### V.2 The multi-backend hub (GCC / Clang / WASM / JVM / CLI / SPIR-V / IDL / markup / IRs)

The claim graph is the neutral center; each backend is a lowering. Honest current state and
the plan:

| Backend | Today | Plan | Fit |
|---|---|---|---|
| GCC / Clang C | ✅ portable C23 emitter + `--fallback` | the primary rail — keep | native |
| LLVM IR / llc (x86/arm/riscv/nvptx/bpf) | ✅ per-target codegen + one elementwise `.ll` shape | generalize `.ll` to arbitrary claim graphs (master §5.9) | native |
| WASM | ✅ exec-validated (`lower/wasm.py`) + a gated byte-encoder (H5) | keep; the portable-deployment channel | good |
| JVM bytecode | ◑ stackify emitter, **execution-validated** (`.class` run) | a real class-file backend is a larger slice | stack-machine map is clean |
| CLI / .NET (CIL) | ✅ stackify emitter, **execution-validated** (`.il` assembled by the real `ilasm`, run under `mono` in CI) | a real assembly backend is a larger slice | same stack-machine map as JVM |
| SPIR-V | ◑ SYCL channel (differential oracle; scalar C++ fallback in CI) | the GPU-compute lowering; real device path needs a `-fsycl` toolchain | modeled |
| IDL / CORBA, SDL, SGML/XML/PMML | ⬜ none | these are **interface/serialization description** layers, not compute backends — the right BCIR role is the ETL/parse rail (emit a *verified parser/marshaller kernel* that consumes the schema), exactly the firmware-table posture of Part III; PMML additionally maps to §7's `ModelManifest` as a model-exchange import | parser-kernel, not codegen |
| Foreign IR bridges (MLIR dialects, SPIR, other IRs) | ✅ the law rail lives inside MLIR | conversion passes, per-dialect on demand, behind the §5.14 Phase-2 filter (only law-bearing semantics get representation) | native mechanism |

**The technical enabler for "optimize all layers above":** because the driver is a compiler
and the communication with hardware is a *typed, costed claim edge* (MMIO/port-I/O/DMA), the
layers above (a libc call, a syscall node, a language runtime) can be modeled as claims in the
*same* graph and co-optimized — the toolchain research's "graph-native syscalls" idea
(syscall = a graph node with input/result/error edges, so independent syscalls parallelize) is
exactly a BCIR claim with rd/wr edges and a phase DAG. The unifying move: treat **each execution backend as a channel with its own calibration
artifact**, so the planner chooses the backend (JVM vs WASM vs native) the way it chooses a
lane width today — the six-target capability matrix already proves this retargeting at
kernel scale. **Verdict: the hub is architecturally sound and partially real; JVM/CIL/SPIR-V
real backends are separable slices, and the markup/IDL family belongs on the parser-kernel
rail, not the codegen rail.**

### V.3 Does each driver need its own ISA/firmware-tailored ML layer?

**No — and the architecture already says why.** The correct decomposition (from the channel
model + the ML/AI two-truth line) is:
- **One shared learning *mechanism*** — the calibrator/ranker/regret/replay organs are
  device-agnostic code (they operate on the 12-d cost vector and telemetry episodes).
- **Per-channel *frozen artifacts*** — each ISA/firmware gets its own **calibration artifact**
  (the `channel.json` `CalibrationArtifact` ref: measured cost table + a small frozen prior
  keyed by that device's op-shape classes), NOT its own ML code.

So a new device ships a *manifest + a frozen Q8 table*, not a bespoke neural net. ISA nuances
(SVE scalable width, NVPTX warp, PIM offload economics) are already `TargetProfile`/capability
data; firmware nuances (register maps, IRQ topology) are claim-graph inputs. A genuinely novel
accelerator *may* warrant a larger per-channel prior, but it rides the same freeze-and-gate
path. **Verdict: per-driver *tailored artifacts* yes; per-driver *ML layers* no — that would
multiply the quarantine surface with no benefit. This is the single most important
architectural answer in this audit.**

### V.4 ISO / industry standards the driver + kernel must follow

A conformance map (what binds, and BCIR's posture — consume/emit-verified, rarely implement):

| Domain | Standard | BCIR posture |
|---|---|---|
| Language | ISO/IEC 9899:2024 (C23), ISO/IEC 14882 (C++) | the cfront target; C23 is the substrate |
| OS interface | **POSIX IEEE Std 1003.1-2024**, Single UNIX Specification | compatibility layer (V.5); strict + BCIR-optimized modes |
| C library / ABI | System V psABI (x86-64, AArch64, RISC-V), Itanium C++ ABI, DWARF, ELF | emit-conformant objects; ABI matrix already models data layout |
| Firmware/boot | UEFI (2.x), ACPI (6.x), SMBIOS/DMI, Device Tree (DTB) | **consume** via verified parser kernels (Part III RUNG 3); never implement UEFI/AML |
| Buses/devices | PCIe base spec, NVMe, xHCI/USB, SATA/AHCI, MMIO conventions | verified enumeration/descriptor-parser kernels (RUNG 4–5) |
| Interrupts/timers | APIC/x2APIC, IOAPIC, HPET, PIT | program-kernel emit (RUNG 1–2) |
| Security/measured boot | TCG PC Client Platform Firmware Profile, TPM 2.0 (ISO/IEC 11889) | event-log parser + TPM2 command marshaller only (Part III) |
| Numerics | IEEE 754-2019, IEC 60559 (via C23 Annex F/H); ISO 80000 units for telemetry | R17 accuracy law governs the float boundary |
| Interchange | Unicode (tokenizer), ISO 8601 (time), IDL/CORBA (OMG), SPIR-V (Khronos), PMML/SGML/XML | parser/marshaller-kernel rail (V.2) |
| Safety (aspirational, if BCIR targets regulated domains) | MISRA C, ISO 26262 (automotive), DO-178C (avionics), IEC 61508 | the verified-C + provenance + replay story is a strong fit; formal certification is out of near-term scope |

**Verdict: the binding near-term set is C23 + POSIX-2024 + System V psABI/ELF/DWARF +
PCIe/NVMe/xHCI/APIC/ACPI/UEFI (consume) + IEEE 754. Everything else is parser-kernel or
aspirational.** The "consume/emit-verified, don't implement" posture is what keeps the
conformance surface tractable.

### V.5 The Linux breakdown + backward-compatibility plan (initial)

The toolchain research's complexity verdict frames this: POSIX + libc + compat layer alone is
~9/10 complexity, multi-year. So the plan is **layered interop, incremental, measurement-
driven** — Linux-compatible microkernel first, Linux-independent AI kernel in parallel behind
it, never a rewrite-the-world step.

**(a) Export the hardware tables (the cleanest, highest-value first move).** This is squarely
BCIR's strength (Part III / Phase D): parse Linux/UAPI/CMSIS headers, ACPI/SMBIOS/PCI tables,
and ISA/opcode/register maps into **verified BCIR claim-graph descriptors** via the cfront +
ETL rails. The Linux-pattern analysis names the exact targets (efi_system_table, memblock,
per-CPU vector tables, PCIe scan, NVMe queues, xHCI rings). These become `channel.json` +
register-map inputs — *no Linux code runs*, only its data structures are ingested and
re-verified. **First slice: an ACPI static-table (RSDP→XSDT→MADT/MCFG) parser kernel + a PCIe
config-space enumerator, both emit-verified.**

**(b) POSIX / ABI / syscall backward-compatibility (the interop contract).** Adopt the
research's recommended stack: **port musl libc** (clean, MIT, minimal Linux assumptions) as
the foundation; define a **graph-native syscall convention** (a syscall = a claim with
`syscall_number` + argument input edges + result/error output edges, so independent syscalls
parallelize and dependencies are explicit); provide a **Linux x86-64 (then aarch64) ABI
compatibility layer** — direct-map common syscalls to BCIR primitives, semantically emulate
the rare/complex ones (epoll/inotify/fork), ENOSYS-with-logging for the unsupported tail.
Expected overhead from the research: 5–15% for a well-optimized native compat layer, which is
acceptable. Two modes, per the research: **strict-POSIX** (sequential semantics, maximum
compatibility) and **BCIR-optimized** (parallel-when-provably-safe, guarded by the effect/
commutation analysis the cfront rail already computes). The R-law/effect footprint machinery
is precisely what proves "safe to parallelize."

**(c) IPC triage — keep vs re-derive vs drop.** The organizing question is *which Linux IPC
mechanisms map cleanly onto the claim-graph + StreamPack + channel model*:
- **Keep + map to native (good BCIR fit):** shared memory (≈ registry-first shared
  Resources with generations), pipes/FIFOs and Unix sockets (≈ `!bcir.token` streams / GEM
  StreamPack byte channels), eventfd/futex-style wakeups (≈ async fork/await tokens),
  memory-mapped I/O (already first-class). These are dataflow-shaped and the model expresses
  them directly.
- **Emulate at the compat boundary (keep for compatibility, don't privilege):** System V IPC
  (message queues, semaphores, shm), signals (asynchronous, conflict with deterministic graph
  execution — provide but isolate), netlink.
- **Candidate to drop / not re-derive in the AI kernel (legacy weight):** the heavier
  ambient-authority surface — arbitrary `ioctl` multiplexing, `/proc`-as-API, the broadest
  namespace/cgroup matrix — kept *only* behind the emulation layer for Linux compatibility,
  deliberately **not** part of the independent AI-microkernel core (which prefers typed,
  capability-scoped channels over ambient ioctls). This is the "what to remove from the legacy
  master kernel" answer: not remove from *compat*, but *exclude from the native core*.

**(d) The two-track structure (the key to "backward-compatible AND independent").**
- **Track L (Linux-compat microkernel):** musl + the ABI/syscall compat layer + emulated IPC
  + ingested hardware tables → runs POSIX/Linux applications with acceptable overhead. This is
  the adoption path.
- **Track N (native AI microkernel):** graph-native syscalls, typed channel IPC, adaptive
  drivers, the ML/AI ecosystem (ML/AI roadmap §8) — **no Linux dependency**. Track L is a
  compatibility *subsystem that runs on* Track N (the WSL-in-reverse model the research
  recommends over Wine-style emulation), so the independent kernel is primary and Linux compat
  is a guest contract, not a foundation.

**(e) Rebuild all drivers to BCIR format.** Feasible *incrementally, not wholesale*: each
driver is a verified claim graph emitted from its ingested register map (Part III RUNG ladder),
prioritized by the toolchain research's device order (serial/UART → interrupt controller →
timers → PCIe enum → NVMe/USB/NIC). The volume is large but the *method* is uniform and
gated, and the wins compound (each driver joins the adaptive-calibration loop). **Verdict:
feasible as a long RUNG-ordered program; the near-term deliverable is not "all drivers" but
the table-export tooling (a) + the first channel-backed adaptive driver + the strict-POSIX
compat-layer skeleton over musl.**

### V.6 Verdicts + first slices

| Question | Verdict | First gateable slice |
|---|---|---|
| Adaptive drivers from the tropical optimizer (V.1) | Feasible; loop rig-gated for the *measured* win | D2.1 UART re-planning its poll loop under measured Θ (degrade-honest) |
| Multi-backend hub (V.2) | Sound; C/LLVM/WASM real, JVM/CIL/SPIR-V separable, IDL/markup = parser-kernel | generalize the `.ll` emitter to arbitrary claim graphs |
| Per-driver ML layer? (V.3) | **No** — shared mechanism + per-channel frozen artifacts | a second `channel.json` `CalibrationArtifact` for a real device |
| Standards conformance (V.4) | Bounded near-term set; consume-don't-implement | ELF/DWARF-conformant object emit gate |
| Export Linux hardware tables (V.5a) | **Highest-value, cleanest** — pure BCIR strength | ACPI static-table + PCIe config-space parser kernels |
| POSIX/ABI/syscall compat (V.5b) | Feasible via musl + graph-native syscalls; multi-year at full scope | strict-POSIX skeleton over musl, x86-64 first |
| IPC triage (V.5c) | Clear keep/emulate/exclude split | map pipes/shm/futex to token-stream/shared-Resource natives |
| Two-track Linux-compat + independent (V.5d) | Feasible; Track N primary, Track L guest | the compat subsystem as a channel |
| Rebuild all drivers (V.5e) | Feasible incrementally, RUNG-ordered | the table-export tooling first, not the drivers |

**Scope honesty.** The full Linux breakdown (complete syscall coverage, container/namespace/
cgroup support, the whole device tree) is, by the toolchain research's own estimate, a 3–5
year / 15–25 engineer program and is explicitly **beyond this request** — this Part V is the
*initial plan and feasibility verdict*, sequenced so the highest-value, most-BCIR-shaped work
(hardware-table export, the first adaptive driver, the POSIX skeleton) lands first and each
step is independently useful even if the full independent kernel is never completed.
