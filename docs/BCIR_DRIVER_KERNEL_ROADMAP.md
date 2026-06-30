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
| **H4** | **ML convergence demos for the predict-only models.** | ML *correctness* is genuinely strong (finite-difference gradient gates, XOR-MLP end-to-end, OLS/PCA recovery to 1e-7, BPTT==FD). The gap is **training at scale**: E3 Transformer, E4 LSTM/GRU (Tier-B), E5 classical (KNN/tree/SVM/NB *fitting* is untested), and the E6 autoencoder have **no convergence demo**. Add a tiny LSTM sequence-prediction run, one classical-ML *fit* test each (vs sklearn to tolerance), and an autoencoder reconstruction-descent run. | ~2 d | **PARALLEL / defer** (not on the driver critical path) |
| **H5** | **Native-codegen-outside-LLVM honesty + the gated WASM byte-encoder.** | Today "native codegen outside LLVM" = `stackify.py` **golden-string** JVM/CIL renderers (emit-only, never assembled/executed) + LLVM-routed WASM. The **direct, RA-free WASM byte-encoder** that [`BCIR_NATIVE_BACKEND_FEASIBILITY.md`](BCIR_NATIVE_BACKEND_FEASIBILITY.md) names as the "best first experiment" **does not exist yet**. Per that doc's own gate it **stays deferred** (no deployment clears G1+G2). The cheap honesty fix, if pursued, is execution-validating the JVM/CIL emit; the bounded slice is a LEB128 + section-framing `to_wasm_bytes` extending stackify, validated by the existing WASM validator + node execution. | ~3–5 d, gated | **DEFER (per the gate)** |

**Things that are already well-covered — do not manufacture work:** the SEG6.1/SEG7
order-parameterized **fences** have full dual-rail parity (`test_c_cfront_barrier.py`,
`test_fence_order_edge_cases_dual_rail`, 29 barrier tests, the MLIR `memory_ordering.mlir`, and
per-ISA assemble-on-native gating); the **`bcir.asm` MLIR op (SEG8.1)** is FileCheck-gated
(round-trip + `-convert-bcir-to-llvm` lowering + verifier negatives) — though note it is checked as
*text*, never assembled+run, so it carries **moderate** residual risk if a driver leans hard on
inline-asm edges (a future "assemble the emitted `llvm.inline_asm` and smoke-run it" step would
close it).

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
| Port-I/O (**ASM2**, `c.portio.in/out.{b,w,l}`) | ✅ **have** (cfront); ⏳ **MLIR op `bcir.portio` is named-but-not-built (SEG8.2)** | x86 `in`/`out`, isolated `__ioport` I/O space, barriered. Non-x86 → honest diagnostic. |
| Memory fences (**ASM3**, `c.fence[.acquire/.release]`; MLIR `bcir.barrier`) | ✅ **have** | Per-ISA `mfence`/`dmb`/`fence`, `memory_order`-parameterized (SEG6/7). |
| **Entry / reset stub + stack setup before C** | ❌ **missing** | A `bcir.entry` naked-function edge (no prologue, sets `sp`, jumps to C `main`). The genuine floor of bring-up. |
| **Control-register / MSR access** (CR0/CR3/CR4, `rdmsr`/`wrmsr`) | ❌ **missing as a typed edge** | Expressible today only as a *raw* ASM1 blob; deserves a typed `c.creg`/`c.msr` edge (a small per-ISA table, like ASM2). |
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
> — no `c.mmio` edge needed; the volatile-qualified type *is* the edge. *The only MMIO gap is in
> the MLIR law rail:* there is no dedicated `bcir.volatile_load/store` op yet — device access in
> MLIR currently rides the barrier/hazard machinery + the `Domain.MMIO` enum value, rather than a
> first-class op.

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
- **D1.1 = SEG8.2 code slice** — the **`bcir.portio` MLIR op** (the named next op): per-ISA x86
  `in`/`out` emitted *as* inline asm, reusing the SEG8.1 `bcir.asm` → `llvm.inline_asm` lowering;
  op + verifier + FileCheck round-trip/lowering/negatives, dual-rail with the cfront `c.portio.*`
  claim.
- **D1.2** — `bcir.volatile_load/store` (or an MMIO attribute on `bcir.load/store`) so MMIO is
  first-class in the law rail as it already is in cfront emit.
- **D1.3** — the **boot/CPU-state asm edges** *as drivers demand them* (not speculatively): entry/
  reset stub + stack init; typed control-register/MSR edges; `lgdt`/`lidt`. The interrupt
  trampoline (the largest new surface) lands only when going interrupt-driven (RUNG 1).

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
- **Next code slice:** `bcir.portio` (SEG8.2), after the Phase-0 hardening gate.
