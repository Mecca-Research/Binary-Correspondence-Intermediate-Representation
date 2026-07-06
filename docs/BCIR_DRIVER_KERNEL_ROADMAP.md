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
> [`BCIR_NATIVE_OBJECT_GATE.md`](BCIR_NATIVE_OBJECT_GATE.md) (native isel stays gated),
> [`BCIR_ML_AI_INTEGRATION_ROADMAP.md`](BCIR_ML_AI_INTEGRATION_ROADMAP.md) (the parallel ML arc), and
> [`BCIR_UART_DRIVER_BLUEPRINT.md`](BCIR_UART_DRIVER_BLUEPRINT.md) — the **full execution-ready
> blueprint for Phase D slice 1** (the 16550/16750 driver program: normative device model from
> thirteen vendor documents, variant + capability matrices, field-errata research, and build
> slices U0–U9 with laws, tests, gates and ML placement already decided).
>
> **Structure:** Parts I–V are the original bring-up/placement/ordering analysis; Part VI is
> the hardened driver seam (D-R1..D-R6); Part VII the remaining-gaps audit (A/B tracks, now
> landed); Part VIII the machine-code/HAL/ABI/ISA audit (the MC-track); and **Part IX the
> comprehensive driver catalog** — the per-driver blueprint contract, the
> ML-seam-per-device-class mandate, the **BCIR-IPC track** (Linux IPC slimmed to a
> registry-first ring substrate for JIT microkernels + modular POSIX compat), and the full
> phased build order (waves D0–D15 + arch/firmware scoping).

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
| **H1** | **Wire `tools/c/sanitize_cfront.sh` into CI.** | This ASan+UBSan+LSan+Valgrind harness over the cfront twin is **already written and well-built** (clang+gcc builds, all `cfront_*.c` fixtures, a 300-valid/400-malformed seeded fuzz campaign, a leak pass, a UBSan-trap stage, a vacuous-gate guard) — and it is now ✅ **WIRED INTO CI** (the C-runtime job's "cfront twin sanitizer sweep" step runs `SANITIZE_SKIP_VALGRIND=1 tools/c/sanitize_cfront.sh` on every push: both ASan engines + clang's UBSan-trap over all 183 `cfront_*.c` fixtures + the seeded 300/400 fuzz campaign; Valgrind stays local-only for time). Because the script globs `cfront_*.c`, the SEG6/SEG7 **fence** code and the `c.asm` lowering are inside the sweep — the coverage is live. Drivers are exactly "the C twin under malformed device input." | — | **DONE** |
| **H2** | **Area-B numerical red-team (`test_area_b_redteam.py`).** | The 7 `c.call.libm:` wraps (LAPACK / BLAS / GSL / SLEEF / libcerf / FFTW-1D/2D) are **happy-path + reference-verified only** — no NaN/Inf, no ill-conditioned matrices, no dimension edges (n=0,1), no error-code paths. **libcerf's sharpest risk is now CLOSED:** the full-range reference (`erfcx_reference_full`: asymptotic series for x≥8, naive to −26, correctly-rounded `inf` where the value exceeds double range) is red-team-gated across the whole line — the naive form's genuine OverflowError at `\|x\|>26.6` is pinned as fact, the tails validated to x=1e6 (`test_cerf.py` H2 tests). Drivers that touch the numerical edges will hit exactly these regimes. | ~2–3 d | **MUST (numerical path)** |
| **H3** | **Fence/asm C-twin malformed-input fuzz.** | The new fence + `c.asm` C-twin paths are exercised by the dual-rail *parity* loop (which proves output correctness) but not by a *memory-safety-under-malformed-input* fuzzer of their own. ✅ **DONE via H1's CI wiring**: `sanitize_cfront.sh` globs every `cfront_*.c` — the fence (`cfront_atomic.c`) and asm fixtures ride the ASan/UBSan sweep and the seeded malformed-input campaign on every push. | — | **DONE** |
| **H4** | **ML convergence demos for the predict-only models.** | ✅ **LANDED** (the `(H4)`-marked tests): E3 readout training (`test_transformer.py`), E4 Tier-A/Tier-B training (`test_recurrent.py`), the E5 closed-form Gaussian-NB **fit-quality** gate (`test_classical.py` — E5 stays predict-only *by design*, the E7 finding), and the E6 autoencoder full-weight reconstruction descent + k-means inertia monotonicity (`test_unsupervised.py`). Honest scope: E3 trains a linear readout over the *frozen* block (transcendentals sit outside the closed-set Tape); E4 Tier-A now trains the RNN weights THEMSELVES end-to-end (BPTT through `tape.var` weights) and the GRU has its own readout-convergence gate; E6 trains every weight to a near-zero absolute threshold. All four route through the shared convergence gate (`bcir/tests/_convergence.py`). E3 block weights now also train through the transcendental tail (closed-form softmax/layernorm adjoints, FD-verified; the FF+LayerNorm tail recovers a teacher under the shared gate -- `test_transformer_grads.py`); the attention-projection backward has since LANDED too (`mha_projection_grads`: closed-form w_q/w_k/w_v/w_o + block-input gradients, FD-verified against the independent MHA forward, student projections recover a teacher's masked-MHA function under the shared gate) -- EVERY transformer block weight can now receive a gradient. | — | **DONE** |
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

## Part VI — The hardened driver seam (2026-07-04): six principles analyzed, four laws landed

Six proposed hardening principles for the driver/compiler boundary were audited against the
build. Two are **superseded by construction** (BCIR's core thesis already is the stronger
form); four had genuine gaps, now closed as the **D-R rules** with machinery
(`bcir/kbcir/device_manifest.py`, `bcir.device_manifest` on the law rail,
`test_device_manifest.py`, `verify_device_manifest.mlir`).

| # | Proposed principle | Verdict vs the build | Where it lives now |
|---|---|---|---|
| P1 | Static Hardware Schema / Device Manifest (no runtime discovery that changes execution logic; banks + interconnect + distance as immutable types) | **Gap → D-R1 LANDED.** `TargetProfile`/`CHANNELS`/`CalibratedProfile` were already static and frozen, and the UART blueprint's `UartPlacement` already said "capabilities are data, attested, never probed" — but no single exported artifact bundled banks + interconnect + distances as one digestable type object. | `DeviceManifest` (banks, Q8 distance matrix, `cal_gen`/target tie, sha256 digest, envelope with the house refusals incl. TAMPERED); `bcir.device_manifest` op + verifier. **The discovery law**: `probe_agree` — observed facts may **veto** (refuse to run), never **steer** (reroute/resize/substitute); the UART `caps_mismatch` rule, promoted repo-wide. |
| P2 | Memory-bank typing (SRAM/HBM/DRAM as distinct IR types; moves need an explicit Cast/DMA op) | **Gap → D-R2 LANDED.** `Domain` already typed resources (RAM/VRAM/NVM/MMIO/CXL/HBM) and the cost model priced tiers — but nothing FORBADE an implicit cross-tier operand. | `check_bank_moves`: a claim spanning two **memory tiers** must be an explicit `mem.move.*` with exactly one source + one destination. **MMIO is exempt by measurement**: the real GPIO fixture mixes MMIO/RAM in 8 plain load/store claims *by design* — register I/O is the R3 rail's ordering law, not a DMA move. The law is **vacuous over the entire existing corpus** (pinned: train step, streamed step, decode session, the GPIO driver). |
| P3 | Distance-aware op-codes (near vs far moves priced by the cost model) | **Gap → D-R3 LANDED.** `gather_penalty`/`mem_channels`/tier factors priced *classes* of access; no pairwise bank distance existed. | The manifest's Q8 matrix + `move_cost` (bytes × pairwise distance — near HBM-peer < far SRAM→HBM, staying put free, pinned) + `move_claim` minting `mem.move.near` / `mem.move.far` op-codes that land in the destination domain. |
| P4 | Strided buffer views only (no generic malloc; full dimensional stride vector; native-tile validation — 15×15 vs 16 refused; zero-copy DMA from stride info) | **Gap → D-R4 LANDED** (the view + tile law; DMA programming is U-track work). The R21 lifetime laws policed malloc/free and `plan_matmul`'s divisor tiles couldn't *produce* a 15×15 — but nothing REFUSED one submitted from outside, and no stride-vector-only allocation currency existed. | `StridedView` (a request without its full per-dimension stride vector cannot be spelled) + `check_strided_view` (coverage, positivity, bank capacity, offset alignment, **native-tile divisibility** — 15×15 vs 16-native refuses at plan time) + the **R22 seam on the law rail**: a `gem.matmul` adjacent to a `bcir.device_manifest` must submit native-tile-multiple tiles. Zero-copy DMA programming from the stride vector = the UART/dma U-track's job (the blueprint's §1.5 deferral holds until a DMA-bearing device lands). |
| P5 | Command buffers only (pre-compiled graphs, fence/semaphore barriers, fused dispatch) | **SUPERSEDED by construction.** This is BCIR's core thesis: the **StreamPack IS the command buffer** (a pre-compiled execution graph with R10 provenance + R11 generation staleness), **phases/deps ARE the fences** (the compiler states dependencies; `async_plan`'s awaits are the token fences), and `bcir_sp_execute(pack, …, kernel_fn, ctx)` **IS the fused dispatch packet** (graph + config + kernels in one call; there is no single-claim submission API to misuse — the D1 steps 3–7 and the 0.4b replay contract are the proof chain). | No change needed. Documented here as the standing law: *the driver seam accepts packs, never instructions*. |
| P6 | Raw firmware-ISA passthrough (ISA definition language; compiler-side register allocation; driver places the blob) | **SUPERSEDED in pattern; the accelerator half is deferred with a named unblock.** The ISA-definition-language idea is already the house method twice over: **TableGen/ODS defines the op law** (compiler and verifier consume one definition), and the UART blueprint's `uart_model.py → emit_header` generates the register ISA from ONE normative table both rails consume. The binary StreamPack is the "blob" for the BCIR executor. Deterministic register allocation against a *foreign* accelerator ISA is exactly the gated native-object work (`BCIR_NATIVE_OBJECT_GATE.md`) — it enters via a `HardwareChannel` whose `modeled=False` accelerator lands with a real ISA table, not before. | The channel registry + the native-object gate carry it; nothing to build until real accelerator silicon/an ISA doc arrives (the rig-gate pattern). |

**The D-R rule card (normative for every future BCIR driver, U-track included):**
- **D-R1** — one attested `DeviceManifest` per device build; discovery may veto, never steer.
- **D-R2** — memory tiers are types; crossing them is an explicit `mem.move.*` cast; MMIO
  register I/O stays under R3.
- **D-R3** — data movement is priced from the manifest's distance matrix, never guessed.
- **D-R4** — allocation currency is the `StridedView` (full stride vector, bank-fitting,
  aligned, native-tile-divisible); fragmentation is a plan-time refusal (R22 on the rail).
- **D-R5** *(standing, was already true)* — the driver seam accepts StreamPacks, never
  instructions; fences are phase deps; dispatch is fused by construction.
- **D-R6** *(standing, was already true)* — one normative table generates both rails' ISA
  view (ODS for ops; `uart_model.py`-style generators for device registers); foreign-ISA
  passthrough waits at the native-object gate.

**U-track integration note.** The UART blueprint's U0 `RegMapContract` is now understood as
the MMIO specialization of D-R1's manifest; U2's driver state gains nothing (no memory-tier
banks on a 16550), but any DMA-bearing device blueprint (the next driver research phase)
MUST author a `DeviceManifest` in U0 and route its buffer programming through
`StridedView`s — the blueprint template inherits D-R1..D-R6 as design axioms.

---

## Part VII — Remaining-gaps audit (2026-07-04): what is still open before the next driver/kernel analysis

> **Wave-13 status flips (same day):** A1+B1 **LANDED** (event phases on both rails —
> `Phase.event` / `bcir.phase`'s `event` attr, the EV1–EV3 laws in `kbcir/events.py` +
> the R3/EV seam in `-bcir-verify`, the U4 16550 RX fixture; the interrupt-context
> ordering seam is EV3). A2 **LANDED** (`kbcir/dma.py`: descriptor rings compiled from
> `StridedView` pairs — two-pointer byte walk, coalescing/merge/scatter; D-R3-priced
> with a fragmentation premium; `dma_transfer_module` composes D-R2+D-R3+D-R4+A1+B1
> into one law-clean module). A3 **LANDED** (`paged_kv.py`: page-claim wiring; eviction
> as a registry act + scheduled claim; admission as appending phases, hash-identical to
> upfront). A4 **LANDED** (`&x` address constants render as the linker's relocation;
> `sizeof`/`_Alignof` fold in global initializers against the CHOSEN ABI's layout
> oracle; forward refs still refuse). A5 **LANDED** (D1.8 `plan_stream_count` — the
> non-monotonic frontier measured, 2 streams beats 1/4/8 on the house fixture; D3.4
> `orchestrate_guided` + certificate — and the RECORDED uniformity finding: gemm class
> winners cannot diverge by size under the L2 linear cost model; the L3 tile/cache
> model is the named follow-on). B2–B4 remain standing rules / rig-gated, as designed.
> The table below is preserved as the audit of record.

Taken after wave 12 (rung-6 serving complete: streaming + `TokenDFA` schema constraint;
rung-7 opened: paged KV as registry resources, continuous batching measured as wave
scheduling). This is the ranked inventory of what remains for the IR and for driver/kernel
hardening, bucketed by what it **blocks** — the next research phase authors device
blueprints from collected datasheets, and each blueprint class has a named prerequisite
here. Items are ranked within each bucket.

### A. IR-side gaps (buildable now; no hardware required)

| # | Gap | What exists / what's missing | Blocks | Anchors |
|---|---|---|---|---|
| A1 | **IRQ/event phase model** — the largest missing IR concept for drivers. | The phase DAG is program-ordered; the interrupt trampoline is a named Tier-1 asm edge (Part II) and the PIC/APIC rung is sequenced (Part III rung 1) — but there is no first-class *asynchronous entry*: a phase whose dependency is an EVENT source rather than another phase, with the R-law that event phases touch resources only through the same hazard discipline (mask/unmask as explicit claims; no implicit state shared with the interrupted flow). | Every interrupt-driven blueprint — the 16550/16750 IIR/ISR dispatch (blueprint U4) is the first fixture waiting. | Part III rung 1; `BCIR_UART_DRIVER_BLUEPRINT.md` U4. |
| A2 | **DMA programming from `StridedView`s** — D-R4's deferred half. | `mem.move.*` op-codes exist and are priced (D-R3), and the stride vector already carries everything a scatter-gather descriptor needs — but nothing lowers a `mem.move` to a descriptor-ring program. The IR shape (a `dma.program` claim consuming a src/dst view pair + a completion tied to A1's event model) can be designed before hardware arrives. | Every DMA-bearing blueprint: 16750 DMA-mode signaling, e1000 descriptor rings, NVMe queues (Part III AFTER list). | `kbcir/device_manifest.py` (`move_claim`); blueprint §1.5 deferral; Part VI P4. |
| A3 | **Rung-7 page-claim wiring.** | `PagedKV` pages carry live generations (R11 speaks KV), but decode claims do not yet read/write their page RIDs directly. Wiring them in lets the token DAG see page-level hazards — which makes **eviction a scheduled claim** and **mid-flight session admission an append of phases to a live module**, not new machinery. | Rung-7 slices 2+ (serving scale-out); nothing driver-side. | `frontends/models/paged_kv.py` (docstring deferrals); `test_paged_kv.py`. |
| A4 | **Const-expr tail: `&x` address constants + `sizeof`.** | The §5.9 integer evaluator landed on both rails (wave 9). Address-constant global initializers still refuse LOUDLY in linkable emit (an address is not an integer — needs a relocation story), and `sizeof` inside constant expressions is deferred because ABI layout is chosen at **lower** time (the evaluator would need the lowering's layout oracle). | Linkable completeness edge cases; some real UAPI headers. | `BCIR_MASTER_ROADMAP.md` §5.9 + Phase-3 *Remaining (named)*. |
| A5 | **D1.8 + D3.4 (ML-side, not driver-blocking).** | D1.8: the optimizer should CHOOSE the stream count from the cost model (today the caller passes `streams`). D3.4: orchestrate wiring + calibrated per-channel profiles so class winners genuinely diverge. | Training-graph polish; cost-model fidelity. | `kbcir/train_graph.py`; `kbcir/channel_prior.py`. |

### B. Driver/kernel hardening gaps (law-side)

| # | Gap | Status | Anchors |
|---|---|---|---|
| B1 | **Interrupt-context ordering seam.** §5.14's volatile/atomic ops carry the single-claim MMIO law (R3 rail); what's missing is the *cross-claim* seam for interrupt context — the law that an event phase's MMIO claims order against the interrupted flow's (A1 is the prerequisite; the law lands with it). | Open, paired with A1. | §5.14; Part VI P2 (MMIO exemption). |
| B2 | **Re-measure the D-R2 exemption corpus when DMA lands.** The MMIO exemption is *measured* (8 by-design MMIO/RAM mixes in the GPIO fixture; corpus-vacuous pinned). A DMA-bearing driver adds genuinely new mixing patterns — the exemption set must be re-measured then, not assumed. | Standing rule; trips automatically (the pin fails loudly if the corpus shifts). | `test_device_manifest.py` (corpus-vacuousness pin). |
| B3 | **Manifest lifecycle beyond `cal_gen`.** The DeviceManifest refuses STALE/tampered at load; there is no story yet for hot-plug / suspend-resume (device-state generation bumps as R11-style staleness). Low urgency until a hot-pluggable fixture exists. | Deferred with a named unblock (a hot-plug-capable blueprint). | `kbcir/device_manifest.py` (envelope refusals). |
| B4 | **`probe_agree` distance coverage.** The veto covers capacity/tile/ghost-bank lies; probing the *distance matrix* (measured latency vs the pinned Q8 entries) is rig-gated — it needs real multi-bank silicon. | Rig-gated (CT4 pattern). | Part VI P1; `HARDWARE_VALIDATION.md`. |

### C. Rig-gated / maintainer one-liners (not build work)

- **v0.3b tag push**: `git tag -a v0.3b 37d0b6a -m "BCIR 0.3b" && git push origin v0.3b`
  (tag pushes 403 from the build sandbox).
- **Asset-gated real-file runs**: `BCIR_HF_MODEL_DIR=<dir> python -m bcir.tests.test_model_ingest`
  and `... test_model_spm` — now exercising real trained weights AND a real SentencePiece
  tokenizer through the full serving path (HF downloads are proxy-blocked in the sandbox).
- **Native-object gate** (`BCIR_NATIVE_OBJECT_GATE.md`): foreign-ISA register allocation
  waits for a real accelerator ISA table (Part VI P6).
- **Real-silicon calibration** (`HARDWARE_VALIDATION.md`): measured-then-pinned distance
  matrices, PMU/RAPL replan — lights up when a bare-metal rig runs the runbook.

### D. By-design refusals (documented policy — NOT gaps)

- **C-twin file-scope global rendering** stays oracle-side: global *definitions* are the
  oracle's job; the twin consumes them (Phase-3 linking design).
- **Foreign-ISA passthrough** enters only through the native-object gate (D-R6).
- **Discovery may veto, never steer** (D-R1): no runtime adaptation is coming; refusal IS
  the feature.

**Reading order for the next phase.** A1 → B1 first (they unlock the interrupt-driven half
of every collected datasheet), then A2 (unlocks the DMA half, including the 16750's DMA
signaling), with B2 tripping automatically as fixtures land. A3/A4/A5 proceed independently
on the ML/compiler tracks and gate nothing in the driver queue.

---

## Part VIII-A — The AMD AI driver/kernel roadmap (2026-07-05)

The AMD AI-inference driver strategy — bootstrap AMD's ROCm/XDNA stack via interop-not-fork, ride the resident AMDGPU-LLVM (GPU) and Peano/MLIR-AIE (NPU) backends, call real kernels through on-call Triton + AITER/CK, and own only BCIR's unique layer (K_BCIR cost, certificate-gated priors, `-bcir-verify`, StreamPack provenance, event-phase/DMA IR) — is developed in full in **[`BCIR_AMD_AI_DRIVER_ROADMAP.md`](BCIR_AMD_AI_DRIVER_ROADMAP.md)**: a 10-layer vertical stack, a 7-phase build order (deferred Linux-inheritance Phase 0 → device manifests/profiles → ROCm runtime binding → BCIR-Triton on-call compiler → BCIR enhancement layer → XDNA NPU class + hybrid router → serving/framework interop), the three device classes (CDNA/RDNA-iGPU/XDNA-NPU, never one), a per-project interop ledger (vLLM/SGLang/Lemonade/GAIA/TurnkeyML/TokenSpeed/Digest AI/Unsloth/torchtitan/bitsandbytes/LlamaIndex), and the PyTorch/JAX/TF supplement boundary. It refines the D15 CUDA/RDNA3 compute row below with the honest state (BCIR has no `amdgcn` codegen today; the AMD channel is a thin routing seam).

## Part VIII — The machine-code / HAL / ABI audit (2026-07-04): the MC-track

Wave 14 audited BCIR's machine-code capabilities against the classic HAL/ISA frame
(Microchip MCU16 HAL, the LinuxCNC HAL Handbook, two ISA texts, the x86 opcode
reference) with a three-rail repo sweep. The full audit — executive verdicts V1–V9, the
machine-model characterization, the toolchain and ABI ledgers, and the ranked gap
register — lives in **[`BCIR_MACHINE_CODE_HAL_ISA_AUDIT.md`](BCIR_MACHINE_CODE_HAL_ISA_AUDIT.md)**.

The headline verdicts:

- **No dedicated HAL backend is needed** — the registry/channel/manifest stack IS the
  HAL's schema + facade layers; the missing parts are resident drivers (runtime),
  a BSP-style name-binding table, and the `halcmd`-class operator tools.
- **HAL functions migrate into the BCIR ABI as `RuntimeChannel` v2** — an append-only,
  versioned hook vtable (open/claim, map, submit, sync, event delivery) with POSIX
  backing; laws stay compile-time, hooks observe-and-refuse (D-R1 at runtime).
- **The BCIR "ISA" is the claim vocabulary + StreamPack encoding** — no registers, no
  flags, no branch opcodes *by design*; the honest boundaries are the emit-only control
  tree (erased before planning) and affine-only addressing.
- **Toolchain**: loader/codec/verifier/compiler-driver are native and dual-rail; object
  code rides the resident compiler (gate intact); **disassembler, hex dump, pack-level
  linker, and peek/poke are the missing native tools**.

**The MC-track** (ranked; S/M/L = effort): MC1 disassembler + hex dump + listing (S) →
MC2 peek/poke with R11-governed pokes (S) → MC3 ROP v2 registry assembly with macros +
the BSP binding table (M) → MC4 carry-as-data + typed predicates (M) → MC6 HAM/semantic
swap composed from the wave-13 machinery (M) → MC7 pack-level linking + symbol section
(M) → MC8 RuntimeChannel v2 hook vtable, designed alongside the deep-driver analysis (L)
→ MC9 POSIX compat completion (M). MC5 (CFG-aware planning) is parked until a driver
fixture forces it. **Sequencing into the deep-driver phase:** MC1–MC3 first (the tools
driver bring-up uses daily), MC8's hook list derived from the first real driver's needs
— exactly as Part VII derived A1/A2 from the datasheets.

---

## Part IX — The comprehensive driver catalog + build order + BCIR-IPC (2026-07-04)

This part upgrades the uploaded "Binary Correspondence Driver Roadmap and Comprehensive
Driver Catalog" into the BCIR frame, with three additions the catalog left open: the
**per-driver blueprint contract** (the UART pattern, generalized so every future driver
is authored the same way), the **ML-seam-per-device-class mandate** (each hardware class
gets its own learned prior — the reason BCIR builds drivers at all), and the **BCIR-IPC
track** (the Linux-IPC research folded into a registry-first IPC that carries JIT
microkernels + modular POSIX compat). The catalog then sequences every named device into
build waves in dependency order.

> **Standing frame (unchanged from Parts I–VIII).** A BCIR driver is a *lowering backend +
> a small hot execution core*, not a runtime object subsystem: portable claims lower to
> device-native packets/register writes, then execute under phase ordering, hazard rules,
> and R11 generation guards. Higher drivers compile into the registries the lower ones
> stabilize — which is exactly why order matters. The wave-11..13 machinery makes this
> concrete: DeviceManifest + StridedView (D-R1..D-R4), event phases (A1/B1), and DMA
> descriptor rings (A2) are the substrate every device backend below compiles into.

### IX.1 The per-driver blueprint contract (generalize the UART U0–U9 pattern)

Every driver in the catalog is built in **its own research session** that first produces a
`BCIR_<DEVICE>_DRIVER_BLUEPRINT.md` (the UART one is the worked reference —
`BCIR_UART_DRIVER_BLUEPRINT.md`, 1042 lines, slices U0–U9). A blueprint is **normative
before any code**: it decides the design so the implementing session does not re-derive
it. Required sections, mirroring the UART blueprint's shape:

| § | Section | What it fixes |
|---|---|---|
| 0 | How to use (slice order, gates, house disciplines) | The build contract |
| 1 | **Normative device model** (register file / command format / descriptor layout, sourced from the spec PDFs) | The single source of truth for the generator, laws, sim, tests |
| 2 | **Variant matrix** (what the registry parameterizes across chip revisions) | One logical model, N placements |
| 3 | Field-reality quirks (errata not in any datasheet) | The traps |
| 4 | Architecture map — how each piece lands on **existing** BCIR machinery (table below) | No new subsystems |
| 5 | Build slices (`D<dev>0..N`, each one PR-sized commit, independently gated) | The increment plan |
| 6 | Performance model + measured-then-pinned gates (what "max performance" means, numerically) | The win, measured |
| 7 | **ML placement card** (the device's learned prior — §IX.2) | The optimization seam |
| 8 | Simulated-device behavioral contract (the sim the tests pin) | Hardware-free CI |
| 9 | Deferred items + why | Honest scope |
| 10 | Process requirements (six-artifact law, two-truth, prototype-then-port, registry registration, non-disturbance) | The house rules |
| 11 | Acceptance checklist (definition of done) | The gate |

**The architecture map — every driver lowers onto machinery that already exists** (this is
why "no dedicated HAL backend" holds, Part VIII V1):

| Driver concern | BCIR machinery it compiles into | Anchor |
|---|---|---|
| Register file / MMIO block | `RegMapContract` (UART U0) + `DeviceManifest` banks; MMIO domain under R3 | `device_manifest.py`; `uart_regs.h` |
| Device discovery / attestation | `DeviceManifest` digest + `probe_agree` (**D-R1** veto-not-steer) | `device_manifest.py:234` |
| Interrupts / ISR dispatch | **Event phases** (A1) + EV1–EV3 masking/ordering laws | `kbcir/events.py` |
| DMA / scatter-gather | `dma_descriptors` + `dma_transfer_module` (**A2**, D-R2/D-R3/D-R4-composed) | `kbcir/dma.py` |
| Memory mapping across banks/devices | `StridedView` (**D-R4**) + allocator tiers + HAM/semantic-swap (**MC6**) | `device_manifest.py`; `allocator.py` |
| Cost planning / device selection | `optimize` (K_BCIR) + `orchestrate` (tower) + learned priors (**D3**) | `realize.py`; `channels.py` |
| Deployment (the "JIT microkernel") | `hydrate` → StreamPack → `bcir_exec` → **RuntimeChannel v2** (MC8) | `gem/streampack.py`; `bcir_exec.h` |
| Command/response marshalling (TPM/NVMe) | StreamPack-ABI-style bounded length-prefixed records | `abi/streampack_abi.py` |
| Config/name binding (BSP) | ROP v2 binding table (**MC3**) — logical name → RID/claim recipe | Part VIII MC3 |

**A driver blueprint that needs a NEW subsystem is a design error** — stop and check
whether the concern maps onto the table above. The only sanctioned new machinery for the
whole catalog is the **BCIR-IPC substrate** (§IX.3) and the **RuntimeChannel v2 hook
vtable** (Part VIII MC8), because those are the runtime half BCIR genuinely lacks today.

### IX.2 The ML-seam-per-device-class mandate (why BCIR builds drivers)

Every device class gets **its own learned prior** — the same recipe D3 already ships
(`tile_prior.py` / `channel_prior.py`): a Q8-frozen logistic/table prior over *cheap*
features, trained offline on the exhaustive optimizer's own choices, gated by a
certificate (**guided == exhaustive, mismatches 0**), staleness-refused when the device
`cal_gen` bumps. This is section §7 of every blueprint and is non-negotiable — a driver
with no ML placement card is a transliterated Linux driver, not a BCIR driver.

| Device class | The learned prior (features → decision) | Precedent |
|---|---|---|
| UART family | RX trigger level + TX burst size (fill vs latency) | **Shipped** (UART U5) |
| Interrupt controllers (IOAPIC/GIC/PLIC) | IRQ→CPU affinity + coalescing threshold (latency × load) | new; mirrors channel_prior |
| Timers (HPET/TSC/RTC) | Per-device frequency-error + cross-device sync offset (PTP-style skew model) | new; a drift regressor |
| DMA engines | SG batch size + descriptor coalescing (setup cost vs fragmentation — extends A2's `dma_cost`) | A2 pricing |
| NVMe / storage | Queue depth + submission batching + read-ahead per workload class | new; tile_prior recipe |
| Block layer / cache | Hot/cold admission + eviction prior (ties directly to **MC6 semantic swap**) | MC6 |
| Network (e1000/virtio/mlx5) | Interrupt-coalescing timer (ITR) + ring size + RSS hash steering (throughput vs tail latency) | new |
| USB/xHCI | Transfer-ring scheduling + endpoint polling interval | new |
| GPU compute (CUDA/RDNA) | Kernel launch config + occupancy + memory-tier placement — **D3's home turf** (the tile/channel prior generalizes directly) | **D3 core** |
| IOMMU | IOTLB prefetch depth + domain-sharing prior | new |
| Filesystems (ext4/FAT32) | Readahead window + allocation-locality prior | new |

The certificate discipline means a mis-learned prior is **caught, never trusted** (a
poisoned table changes placements → the guided-vs-exhaustive diff is nonzero → refused),
so the ML seam adds performance without ever risking a correctness verdict (two-truth).

### IX.3 The BCIR-IPC track — Linux IPC research, slimmed and registry-first

**The research question restated:** guide BCIR IPC by Linux best practices; adopt what
serves max performance; drop the legacy kernel bloat; keep only ~2–3 generations of
backward support; reach JIT microkernels + modular POSIX compat. Where full compat is
infeasible natively, keep a dedicated Linux Master Kernel and migrate drivers behind a
communication interface.

**The Linux IPC inventory, triaged** (from Linux Device Drivers / IPC Linux / the syscall
references):

| Linux mechanism | Verdict for BCIR | Why |
|---|---|---|
| **io_uring** (SQ/CQ shared-memory rings) | **ADOPT as the ONE primitive** | Structurally identical to BCIR's `bcir_exec` submission + telemetry-ring completion + event-phase signal — BCIR already *is* an io_uring-shaped machine |
| POSIX IPC (`mq_*`, `sem_*`, `shm_*`, fd-based) | **ADOPT the model, reimplement native** | Clean fd lifecycle; maps to registry resources + generations |
| AF_UNIX sockets (STREAM/DGRAM/SEQPACKET, SCM_RIGHTS fd passing) | **ADOPT (the workhorse)** | The local-IPC transport apps expect; fd passing = capability handoff |
| `eventfd`/`signalfd`/`timerfd`/`pidfd` | **ADOPT ("everything is an fd")** | Uniform readiness; `eventfd` ≈ an event-phase completion flag |
| `futex` | **ADOPT (the sync primitive)** | Userspace-fast; the one blocking primitive under the rings |
| `mmap` / `memfd` shared memory | **ADOPT (zero-copy)** | The registry's shared-region story |
| **System V IPC** (msg/sem/shm, keyed) | **DROP** | Leaky global lifecycle, keyed namespace, no fd — the canonical legacy bloat |
| **Real-time + legacy signals** | **REPLACE with event phases (A1)** | Unreliable delivery, tiny payload, reentrancy hazards → EV1–EV3 give ordered, typed async entry |
| ptrace / legacy `/proc` IPC surfaces | **DROP / Master-Kernel-only** | Debug/legacy; not on the hot path |
| Namespaces / cgroups / seccomp | **Microkernel-scope, not IPC** | Isolation belongs to the microkernel model, layered on capabilities |

**The core insight:** BCIR's StreamPack + event phases + the generation-tagged registry
*are already an IPC substrate*. io_uring is a submission-ring + completion-ring shared-
memory ABI — the same shape as `bcir_exec`'s calling convention (Part VIII §5.2) plus the
telemetry ring plus event-phase completion (A1). So BCIR-IPC is not a new subsystem to
invent; it is **one primitive (the ring) expressed as registry resources + event phases,
with a thin measured POSIX shim on top.**

**The BCIR-IPC laws** (new, IPC-R1..IPC-R4 — to be authored as a six-artifact set in their
own wave, vacuous over the existing corpus by the non-disturbance discipline):

- **IPC-R1 — a channel is a resource pair.** An IPC channel is a (submission-ring,
  completion-ring) pair of registry resources with an owner and a generation; cross-
  microkernel reads are R11-generation-checked (R11 extended across address spaces — a
  stale peer view refuses, never races).
- **IPC-R2 — message passing is an explicit claim.** No implicit shared state: a message
  or ownership transfer is a claim (the `mem.move.*` shape), and handing a buffer across a
  channel bumps its `data_gen` (the receiver's stale pack refuses — zero-copy handoff with
  a proof, not a hope).
- **IPC-R3 — notification is an event phase.** No signals: readiness/completion is an
  event phase (A1), delivery ordered by EV3 against the interrupted flow. `eventfd` and
  friends are the fd view of the same event source.
- **IPC-R4 — the POSIX shim is measured and loadable.** Modular POSIX backward compat is a
  loadable translation layer, not kernel bloat: hot syscalls (measured via telemetry)
  lower to native BCIR-IPC claim recipes; the cold legacy tail delegates (Strategy 3).

**The three-strategy ladder** (the user's framing, made operational):

1. **Adopt-and-slim (default).** Reimplement the *modern* Linux IPC subset (fd-based POSIX
   IPC, io_uring rings, futex, the `*fd` family, AF_UNIX) as native BCIR-IPC registry +
   event-phase constructs under IPC-R1..IPC-R4. Frozen as a new ABI artifact (a `BCIR-IPC`
   wire format alongside StreamPack `BSPK` and telemetry `BTLM`). This is where max
   performance lives — no context switch to a monolith, generation-guarded zero-copy.
2. **Abstract-away (compat surface).** Keep the POSIX *interface* apps expect, backed by
   the Strategy-1 substrate through the C23 subset + the R12 ABI + the IR rules — legacy
   `read`/`write`/`sendmsg`/`epoll_wait` become claim recipes. Backward compat is a shim
   layer, bounded to ~2–3 generations, not a legacy kernel.
3. **Linux Master Kernel (fallback).** Where true native compat is infeasible or not worth
   it, keep a dedicated Linux kernel as a peer; BCIR microkernels talk to it over a
   communication interface (a virtio-style transport or a shared io_uring), migrating
   drivers one at a time while the Linux side retains the full legacy IPC pipeline.

**The recommendation:** Strategy 1 for everything modern and hot; Strategy 2 for the POSIX
surface applications require; Strategy 3 only for the long cold tail. The BCIR advantage
is that it can **measure** which syscalls/drivers are hot (the telemetry ring already
exists) and migrate/native-ize *only those*, leaving the cold legacy tail on the Master
Kernel indefinitely — a data-driven migration, not a big-bang rewrite.

**JIT microkernels, defined precisely.** A BCIR "microkernel" is a **StreamPack** (the hot
artifact) plus its event-phase handlers plus its RuntimeChannel v2 binding. "JIT" = the
pack is *hydrated on demand* from the planned claim graph and *replanned on measured cost*
(the L2 measured-replan path already exists, `kbcir/calibrate.py`). IPC between two
microkernels is a shared registry channel (IPC-R1) with generation-guarded handoff
(IPC-R2) and event-phase signaling (IPC-R3) — no syscall, no trap into a monolith. A
**driver microkernel** is exactly this: a device's lowered pack + its ISR event phases +
its RuntimeChannel binding, deployed and replaced as one versioned artifact.

### IX.4 Answers to the catalog's open questions

- **"What other legacy BUS systems do we need drivers for?"** Beyond ISA/port-I/O
  (shipped): **LPC / eSPI** (the TPM, SuperIO, and embedded-controller transport — eSPI is
  the modern LPC replacement, keep it, treat raw LPC as ~2-generation legacy),
  **SMBus / I²C** (SPD/SPD5 DIMM info, sensors, battery, PD controllers), **SPI** (boot
  flash, TPM-over-SPI), and **PCI Conventional** as *enumeration-legacy only* (folded into
  the PCIe ECAM walk — no separate conventional-PCI device stack). The companion specs the
  catalog flagged are confirmed required: **AHCI** (the SATA host-controller programming
  model) and **ATA/ACS** (the command set) travel with any SATA driver.
- **"What other network drivers do we need?"** Beyond **e1000/e1000e** (the baseline):
  **virtio-net first** (the highest-value target — every VM and cloud instance; it is also
  the natural Strategy-3 transport to a Linux Master Kernel), then **igb/igc** (2.5 GbE
  client/embedded), **r8169** (Realtek consumer ubiquity), and **mlx5** (datacenter /
  RDMA, the high-throughput ceiling), all over a shared **MDIO/MII PHY** abstraction.
  The **ITU telecom framers** (HDLC/E1/T1/SONET) are a niche subdriver class — low
  priority, scoped separately, not on the critical path.
- **Processor "drivers" (FPGA, ARM, RISC-V, AMD, CUDA, RDNA, IA-32/64).** These are
  **architecture backends**, not device drivers (the catalog's own note) — they are BCIR
  **channels + codegen targets**, already partly present (`arm64_neon`/`sve`, `riscv_rvv`,
  `nvidia_ptx` channels; x86/aarch64/riscv64/bpf codegen). They enter through the channel
  registry + native-object gate, not the driver waves. The FPGA track (AXI-UART + bitstream
  manager) is a genuine device driver and sits in the embedded wave.
- **Firmware-scope, not drivers.** DDR4/DDR5 (memory-controller training — the OS consumes
  the trained ACPI/UEFI map + optionally reads SPD over SMBus); UEFI PI (platform-init
  architecture — reference unless BCIR replaces firmware). BCIR boots as a **UEFI
  application** and calls `ExitBootServices` (a loader stub, in scope).
- **Subdriver-scope, not standalone drivers.** DisplayPort 2.1 and HDMI 2.1b/2.2 are
  connector protocols *inside* the GPU modeset/display-engine driver (link training +
  bandwidth negotiation + feature flags), not separate drivers. HID Usage Tables are a
  **data dictionary** consumed by the USB-HID class driver, not a driver.
- **Toolchain-scope.** The System V AMD64 ABI and the AArch64 PCS are not drivers but are
  required for correct BC23 codegen — already modeled by the R12 `TargetABI` matrix
  (Part VIII §5.3).

### IX.5 The build order — driver waves in dependency order

Each row is a driver (or driver family) that gets its own blueprint + research session
before its build slices. Waves are strictly ordered by the dependency facts in Part III;
within a wave, items are independent. `[fw]` = firmware-scope reference, `[sub]` =
subdriver, `[arch]` = architecture backend (channel, not a driver wave), `[net]` = network.

| Wave | Driver / module | Depends on | ML seam (§7) | Primary spec anchor |
|---|---|---|---|---|
| **D0 Boot** | UEFI boot handoff (loader stub, ExitBootServices) | — | boot-path timing | UEFI 2.11 |
| **D0 Boot** | UART 16550/16750 console (polled → IRQ) | ISA edge (done) | trigger/burst — **shipped** | PC16550D (U0–U9 done) |
| **D1 Substrate** | Physical memory manager + region registry (BCIR RES) | — | region-placement (allocator heat) | BCIR/BDI regions |
| **D1 Substrate** | Virtual memory + page tables + TLB shootdown (BCIR VM) | PMM | mapping-locality prior | Intel SDM / AMD64 paging |
| **D2 Discovery** | ACPI static-table parser (RSDP→XSDT→MADT/MCFG/HPET/SRAT/FADT; minimal AML policy) | VM | — (parser) | ACPI 6.6 |
| **D2 Discovery** | SMBIOS parser | — | — (parser) | SMBIOS 3.9.0 |
| **D3 Interrupts** | Local APIC / x2APIC (+ per-arch: GIC for ARM, PLIC/CLINT/AIA for RISC-V) | ACPI MADT | IPI-affinity prior | Intel SDM APIC; GICv3/4; RISC-V AIA |
| **D3 Interrupts** | I/O APIC (82093AA) + MSI/MSI-X interrupt allocator | LAPIC, ACPI | IRQ→CPU affinity + coalescing prior | 82093AA; PCIe base |
| **D4 Time** | HPET (+ TSC/LAPIC-timer policy, RTC/CMOS wall clock) — synchronized across devices | Interrupts, ACPI HPET | drift/skew (PTP-style) prior | IA-PC HPET 1.0a |
| **D5 Bus** | PCIe config + enumeration (ECAM via MCFG); BAR sizing; capability parse | ACPI MCFG, interrupts | — (enumeration) | PCIe base; ACPI MCFG |
| **D6 DMA** | DMA allocator + mapping API (bounce buffers first) | PCIe, VM | SG-batch prior (extends A2) | (A2 machinery; IOMMU overview) |
| **D6 DMA** | IOMMU (AMD-Vi / Intel VT-d) — device isolation, per-device address spaces | DMA, PCIe | IOTLB-prefetch + domain-share prior | AMD IOMMU; VT-d |
| **D7 IPC** | **BCIR-IPC substrate** (rings as resources, IPC-R1..R4) + POSIX shim (Strategy 1/2) | VM, event phases | hot-syscall migration prior | Linux IPC research (§IX.3) |
| **D8 Storage** | Block layer (queueing + cache + barriers) | DMA, interrupts | cache admit/evict prior (**MC6**) | NVMe/SATA interfaces |
| **D8 Storage** | NVMe admin + I/O queues (submit/completion rings) | PCIe, MSI-X, DMA | queue-depth + read-ahead prior | NVMe 2.0d → 2.3 |
| **D8 Storage** | SATA/AHCI + ATA/ACS command set (compatibility) | DMA, PCIe | — | SATA 3.5 + AHCI + ACS |
| **D9 FS** | GPT + MBR parsers → FAT32 (ESP) → ext2/3/4 (+ exFAT optional) | Block layer | readahead + alloc-locality prior | GPT (UEFI); fatgen103; ext4 kernel doc |
| **D10 Net** | e1000/e1000e; then **virtio-net**, igb/igc, r8169, mlx5 over MDIO/MII | PCIe, MSI-X, DMA, IOMMU | ITR-coalescing + ring-size + RSS prior | Intel e1000e; virtio; mlx5 |
| **D11 USB** | xHCI host controller → USB enumeration → USB-HID class | PCIe, MSI-X, DMA | transfer-ring + poll-interval prior | xHCI 1.2b; HID Usage Tables 1.7 |
| **D12 Security** | TPM 2.0 driver (CRB/TIS via ACPI) + event-log parser; TCG PFP mapping → CC verification contracts | ACPI, LPC/SPI | — (marshaller) | TCG PC Client PFP; TPM 2.0; CC PP |
| **D13 Virt** | SEV-SNP guest/hypervisor interface; BCIR virtual machines (VMCS/VMCB builders, virtio device models) | VM, IOMMU, IPC | — | AMD SEV-SNP ABI; Intel SDM VMX |
| **D14 Display** | GOP/framebuffer console (initial) → GPU modeset + connector mgmt (DP 2.1 / HDMI 2.1b `[sub]`) → BCIR-Wayland compositor | PCIe, DMA, VM | modeset-config + bandwidth prior | UEFI GOP; DP 2.1; HDMI announce |
| **D15 Compute** | CUDA backend (claims → native queue/command graph); RDNA3 backend (AQL/PM4) | PCIe, DMA, IOMMU, display | **occupancy/tile prior — D3 core** | CUDA guide; RDNA3 ISA |
| **D-embedded** | FPGA track: AXI-UART 16550 + FPGA manager / bitstream loader | UART, PCIe | — | AMD PG143 AXI-UART |
| **legacy-bus** | LPC/eSPI, SMBus/I²C, SPI (feed TPM/SPD/sensors) | ACPI | — | (platform buses) |
| **[arch]** | x86-64, ARM (GIC/A-profile), RISC-V, AMD64 backends — **channels, not driver waves** | native-object gate | K_BCIR tile/channel prior | Intel SDM; ARM ARM; RISC-V; AMD64 |
| **[fw]** | DDR4/5 (consume trained map + SPD), UEFI PI — **firmware-scope reference** | — | — | JEDEC; UEFI PI 1.9 |

**Critical-path facts to pin** (unchanged from Part III, extended): interrupts (D3) gate
every interrupt-driven device; PCIe enumeration (D5) is strictly before any PCIe device
(D8/D10/D11/D15); DMA + IOMMU (D6) gate every DMA-capable device; the **BCIR-IPC substrate
(D7) gates the microkernel deployment model** for everything above it (drivers become
JIT microkernels only once the ring substrate exists); the block layer (D8) gates
filesystems (D9); GPU modeset (D14) contains DP/HDMI as subdrivers and precedes GPU
compute's display interop (D15).

**Spec-currency note** (from the catalog, adopted): target the newest publicly-referenceable
revisions — UEFI 2.11, ACPI 6.6, SMBIOS 3.9.0, PCIe (revision-index anchor; base through
7.0), NVMe 2.3 set, SATA 3.5 + AHCI + ACS, xHCI 1.2b, HID 1.7, DP 2.1, HDMI 2.1b/2.2
(announcement-anchored), TPM PC Client PFP 1.06. Because a BCIR driver is *compiled*, a
newer optional capability is added as new claim forms + backend lowering rules — never a
driver rewrite.
