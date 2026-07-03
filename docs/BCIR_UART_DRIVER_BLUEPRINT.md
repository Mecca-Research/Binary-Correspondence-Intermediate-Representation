# BCIR UART-16550 Driver — the full build blueprint

> **What this document is.** The complete, execution-ready engineering blueprint for the first
> BCIR ML-compiled device driver: a 16550-family UART driver built registry-first, verified by
> the R-laws, priced by K_BCIR, tuned by frozen learned priors, and closed-loop-retuned from its
> own telemetry. It is written so that an implementing model (or engineer) can build every slice
> **without doing any high-level planning**: the device model is normative and sourced, every
> slice names its files, function signatures, law clauses, test names, gates, and commit shape,
> and every design decision is already made and justified. Follow it top to bottom.
>
> **Where it sits.** This is the Phase D slice-1 deep plan the
> [`BCIR_DRIVER_KERNEL_ROADMAP.md`](BCIR_DRIVER_KERNEL_ROADMAP.md) Part IV points at, expanded to
> a full program. It builds ONLY on machinery that is already merged (referenced by file path
> throughout); nothing here requires new core-IR invention beyond the named law additions.
>
> **Sources.** Normative device facts were extracted from eight vendor documents:
> TI/NSC **PC16550D** datasheet (SNLS378C, the canonical silicon), TI **TL16C550C** (SLLS177I)
> and **TL16C550D** (SLLS597E) (autoflow-control ACEs), Exar **ST16C550** (rev 5.0.1),
> Microsemi **Core16550 v3.4** handbook (APB FPGA IP) + the earlier Rev-4 handbook, and AMD/Xilinx
> **AXI UART 16550**: DS748 (v1.01a) + **PG143** (v2.0 product guide). Field-reality quirks
> (§3) come from the Linux 8250 driver corpus and public errata discussion; chip-lot-level
> errata sheets are explicitly **deferred to vendor support** (§9).

---

## 0. How to use this document (for the implementing model)

- Build the slices **in order** (U0 → U7). Each slice is one PR-sized commit and is
  independently gated; do not start slice N+1 until slice N's gates are green.
- Every slice section has: **Goal · Files · Design (already decided) · Tests · Gates · Commit.**
  "Design" is normative — do not re-derive it, and do not substitute alternatives.
- Follow the house disciplines (they are restated in §10 with file-path anchors): the
  six-artifact law pattern, two-truth quarantine (kbcir/driver-model code never imports
  `bcir.verify`; callers pass messages), prototype-then-port (oracle first, C twin second,
  MLIR clause with negatives), registry registration of every new test file
  (`bcir/tests/run_all.py::_MODULES` — the completeness guard `test_registry_complete.py` will
  fail the suite if you forget), measured-then-pinned convergence/performance gates, and
  non-disturbance (every new law vacuous on non-UART code).
- When a numeric gate is required (a performance win, a tolerance), **measure first, then pin
  the gate at ~3× headroom** — the repo convention. Never guess a threshold.
- The device model in §1 and the variant matrix in §2 are the single source of truth for the
  register generator, the laws, the simulator, and the tests. If an implementation detail is
  not in §1/§2, it is a design hole: stop and check the source PDFs, do not improvise silently.

---

## 1. The normative 16550 device model

Everything in this section is sourced from the eight documents above and is what `uart_model.py`
(U0) must encode. Where variants disagree, §2 carries the delta and the model carries the
*parameterization*.

### 1.1 Register file — 12 logical registers over 8 address slots

Three multiplexing dimensions (the driver's central protocol hazard):
1. **Direction**: RBR (read) / THR (write) share slot 0; IIR (read) / FCR (write) share slot 2.
2. **DLAB** (`LCR[7]`): with DLAB=1, slots 0/1 become DLL/DLM (divisor latches). RBR/THR/IER are
   unreachable while DLAB=1.
3. **Mode**: FCR bits above 0 are FIFO-mode constructs; IIR[7:6] read back FIFO-enabled state.

| Slot | DLAB | Read | Write | Reset (PC16550D) |
|---|---|---|---|---|
| 0 | 0 | RBR (received byte) | THR (transmit byte) | — |
| 0 | 1 | DLL | DLL | undefined (Core16550: 0x01) |
| 1 | 0 | IER | IER | 0x00 |
| 1 | 1 | DLM | DLM | undefined (Core16550: 0x00) |
| 2 | x | IIR | FCR | IIR=0x01, FCR=0x00 (Core16550: FCR=**0x01**) |
| 3 | x | LCR | LCR | 0x00 |
| 4 | x | MCR | MCR | 0x00 |
| 5 | x | LSR | (factory test only — treat as read-only) | **0x60** (THRE|TEMT) |
| 6 | x | MSR | (factory test only) | X on bits 7:4, 0 on 3:0 |
| 7 | x | SCR | SCR | undefined |

**Bit definitions (the model's field tables):**

- **IER**: 0=ERBFI (RX data available), 1=ETBEI (THRE), 2=ELSI (RX line status), 3=EDSSI
  (modem status), 4–7 = 0.
- **IIR** (read): bit0 = 0 when an interrupt is *pending* (active-low pending flag!);
  bits 3:1 = ID; bits 7:6 = FIFOs-enabled (both 1 in FIFO mode; 00 in 16450 mode).
  Priority ladder (highest first):

  | IIR[3:0] | Source | Priority | Cleared by |
  |---|---|---|---|
  | `0110` (0x06) | Receiver line status (OE/PE/FE/BI) | 1 | reading LSR |
  | `0100` (0x04) | RX data available (trigger reached) | 2 | FIFO drops below trigger |
  | `1100` (0x0C) | **Character timeout** (FIFO mode only) | 2 | reading RBR (one read) |
  | `0010` (0x02) | THRE (TX FIFO empty) | 3 | writing THR or reading IIR |
  | `0000` (0x00) | Modem status (ΔCTS/ΔDSR/TERI/ΔDCD) | 4 | reading MSR |

- **FCR** (write): 0=FIFO enable (**gates all other FCR bits** — on Core16550 an FCR write with
  bit0=0 does not program the other bits), 1=RX FIFO reset (self-clearing; shift register NOT
  cleared), 2=TX FIFO reset (self-clearing), 3=DMA mode select (RXRDY/TXRDY pin mode 0/1),
  4–5 reserved, 7:6=RX trigger level: `00`→1, `01`→4, `10`→8, `11`→14 bytes.
- **LCR**: 1:0=word length (`00`→5 … `11`→8 bits), 2=stop bits (0→1; 1→1.5 for 5-bit words,
  2 otherwise), 3=parity enable, 4=even parity select, 5=stick parity, 6=set break (forces SOUT
  low until cleared), 7=**DLAB**.
- **MCR**: 0=DTR, 1=RTS, 2=OUT1, 3=OUT2 (on PC/AT wiring OUT2 gates the IRQ line to the
  controller — treat "OUT2=1" as "interrupt output connected"), 4=LOOP (internal loopback:
  SOUT forced high, SIN disconnected, MCR[3:0] wired to MSR[7:4] — the built-in self-test path),
  **5=AFE (autoflow enable — TL16C550C/D and compatible clones ONLY; reserved-zero elsewhere)**.
- **LSR**: 0=DR (data ready: ≥1 byte in RX FIFO; cleared when FIFO empty), 1=OE (overrun),
  2=PE, 3=FE, 4=BI (break interrupt), 5=THRE (TX holding/FIFO empty), 6=TEMT (TX FIFO **and**
  shift register empty — the true line-idle signal), 7=error-in-RX-FIFO (FIFO mode: at least one
  PE/FE/BI anywhere in the FIFO; cleared when no more error bytes remain).
  In FIFO mode PE/FE/BI travel *per byte*: they describe the byte **at the head** of the FIFO at
  the time LSR is read; LSR read clears OE/PE/FE/BI.
- **MSR**: 0=ΔCTS, 1=ΔDSR, 2=TERI (trailing edge of RI), 3=ΔDCD, 4=CTS, 5=DSR, 6=RI, 7=DCD.
  Read clears the delta bits (0–3). In LOOP mode bits 7:4 mirror MCR 3:0.

### 1.2 FIFO semantics (FIFO mode, FCR0=1)

- 16-byte TX FIFO, 16-byte RX FIFO; RX carries **3 error bits per byte** (PE/FE/BI ride with the
  data through the FIFO; OE does not — it is a FIFO-full event).
- **Overrun rule** (precise, needed by the simulator): when a byte completes in the receive
  shift register while the FIFO is full, OE is set and **the shift-register byte is discarded —
  the FIFO contents are NOT corrupted** (ST16C550 states this explicitly; PC16550D concurs).
- **RX trigger interrupt**: RDA (0x04) asserts when FIFO level ≥ trigger; deasserts (and the IIR
  indication clears) as soon as level drops below trigger.
- **Character-timeout interrupt** (0x0C): fires when ALL of — (a) ≥1 byte in RX FIFO, (b) no new
  serial byte for 4 continuous character times (a character time = start + data + parity + stop
  bits at the current line format; the second stop bit counts when programmed), (c) no CPU read
  of the FIFO for 4 character times. One RBR read clears it and resets the timer. Timing is
  derived from RCLK, so it scales with baud. This is what makes trigger levels of 8/14 safe for
  bursty traffic: stragglers below the trigger are delivered at most ~4 char-times late.
- **THRE interrupt**: asserts when TX FIFO becomes empty; 1–16 bytes may be written per service.
  Subtle re-assert rule (PC16550D 8.4.1): the *first* THRE after enabling can be immediate, and
  the empty indication is delayed by (1 character − last stop bit) when fewer than two bytes
  have co-resided in the FIFO since the last THRE — do not build logic that depends on THRE
  edge timing; poll LSR before bursts instead (see §3 quirks).
- **Polled mode**: FCR0=1 with IER=0. No trigger/timeout indications exist; drive everything
  from LSR (DR, THRE, TEMT, error bits). RX/TX sides can be polled/interrupt independently.
- **FIFO reset bits** clear the FIFO counters, not the shift registers.

### 1.3 Baud generation

- divisor = reference_clock / (16 × baud), 16-bit (DLL low byte, DLM high byte), loaded LSB
  then MSB (PG143's programming sequence writes DLL first); loading either latch reloads the
  16-bit counter immediately. **Divisor 0 is forbidden** (PC16550D: "not recommended"; the law
  in U2 rejects it).
- Reference clock is variant-specific (§2): XIN crystal (silicon), PCLK/APB clock (Core16550),
  S_AXI_ACLK or external `xin` < ACLK/2 (AXI core). The driver takes `ref_clk_hz` as a
  parameter and computes the divisor + the achieved-baud error; the U2 law bounds the error
  (|error| ≤ 3% is the classic async tolerance bound; make it a named constant).
- PG143 worked example (use as a test vector): 100 MHz ACLK, 56 kbaud → divisor
  0x006F (111): 100e6 / (16 × 111) = 56306 baud, +0.55% error.
- PC16550D divisor tables (1.8432 MHz / 3.072 MHz / 18.432 MHz crystals) are the other test
  vectors: e.g. 1.8432 MHz → 115200 = divisor 1, 9600 = divisor 12, 300 = divisor 384.

### 1.4 Initialization contract (the canonical bring-up order)

PG143's programming sequence, generalized and hardened (this exact order is what U2's
`uart_init` implements and what the U1 protocol law checks):

1. Read LSR; **wait for TEMT** if the channel may be mid-character (reconfigure-only-at-idle).
2. Write LCR with DLAB=1 plus the final line format bits (one write).
3. Write DLL, then DLM (LSB first).
4. Write LCR again with DLAB=0, same format bits (closes the DLAB bracket).
5. Write FCR: enable FIFOs + reset both FIFOs + trigger level (one write; on Core16550-class
   parts bit0 must be 1 in this same write for the rest to take).
6. Write MCR: DTR/RTS/OUT2 as required (+AFE on autoflow parts, per §1.6).
7. Write IER last (0x00 for polled mode; enables for interrupt mode).
8. Read LSR, RBR, IIR, MSR once each to clear any stale status.

### 1.5 DMA signaling (documented, deferred)

RXRDY/TXRDY pins, mode 0 (single-transfer) / mode 1 (multi-transfer) selected by FCR3.
Pin-level handshakes only — no bus-master protocol exists in the family. Model the two modes in
`uart_model.py` for completeness; the driver slices do not use them (deferred, §9).

### 1.6 Autoflow control (TL16C550C/D — closes the wave-6 "missing piece")

- **Enable**: MCR5 (AFE) = 1 **and** MCR1 (RTS) = 1 → full autoflow (auto-RTS + auto-CTS).
  AFE=1 with RTS=0 → **auto-CTS only**.
- **Auto-CTS**: transmitter sends while CTS low; if CTS goes high before the middle of the last
  stop bit of the current byte, the current byte completes and the *next* byte is held; sending
  resumes when CTS returns low. (Prevents TX overrun of the far end with zero software.)
- **Auto-RTS**: RTS deasserts as the RX FIFO reaches the trigger level; for triggers 1/4/8 the
  peer may legally complete one in-flight byte (so effective high-water = trigger+1); RTS
  **re-asserts only when the FIFO has been emptied** (read down to empty) for triggers 1/4/8.
  For trigger 14, RTS deasserts at the first data bit of the 16th character and re-asserts as
  soon as ≥1 FIFO slot frees. **Consequence for the driver**: with autoflow at trigger 1/4/8 the
  service routine must drain to empty or throughput collapses; at trigger 14 partial drains are
  fine. This asymmetry is a *plan input* in U5 (it changes the optimal drain-batch size), and a
  simulator behavior in U3.
- **Without AFE hardware** (plain PC16550D/ST16C550/FPGA cores as configured): software flow
  control via MSR ΔCTS interrupt + MCR.RTS writes, with an inherent race window of up to one
  character time — the blueprint's software fallback (U5) documents the window instead of
  pretending it away.

### 1.7 Loopback self-test (MCR4)

LOOP=1: SOUT high, SIN internally fed from the transmitter, MCR[3:0] → MSR[7:4] (DTR→DSR,
RTS→CTS, OUT1→RI, OUT2→DCD), interrupts still operate. This is the driver's built-in
end-to-end self-test path — U2's `uart_selftest` uses it (write a pattern, read it back, check
MSR mirrors) and the simulator must implement it.

---

## 2. The variant matrix (what the registry must parameterize)

| Axis | PC16550D | TL16C550C/D | ST16C550 | Core16550 v3.4 (APB) | AXI UART 16550 v2.0 |
|---|---|---|---|---|---|
| Register stride | 1 byte (A2:A0) | 1 byte | 1 byte | **4 bytes** (PADDR 0x00–0x1C) | **4 bytes at base+0x1000** (0x1000–0x101C), 32-bit lanes |
| FCR at reset | 0x00 | 0x00 | 0x00 | **0x01 (FIFOs on)** | 0x00 |
| FCR readable | no (write-only) | no | no | no | **yes** (reads at slot 2, LCR7=1 documented) |
| FCR bit-gating | n/a | n/a | n/a | **other bits ignored unless bit0=1 in the same write** | per PC16550D |
| DLL/DLM at reset | undefined | undefined | undefined | **0x01 / 0x00** | per core |
| MCR5 | 0 (reserved) | **AFE** | 0 | 0 | 0 |
| Extra regs behind DLAB | none | none | **"enhanced feature register"** (EFR-class, 16650 lineage) | none | none |
| 16450 mode | runtime (FCR0=0) | runtime | runtime | FIFO-centric | **compile-time core generic** (FCR removed in 16450 builds) |
| Baud reference | XIN crystal (+ separate RCLK in) | XIN | XIN | PCLK (bus clock) | S_AXI_ACLK or ext. `xin` (< ACLK/2) |
| Interrupt output | INTR pin (edge behavior per source) | INTR | INTR | IRQ | **level-sensitive** IP2INTC_Irpt |
| LSR/MSR writable | factory test | factory test | factory test | read-only | **documented R/W** |

**Variant hazards the laws must catch** (these are the *reasons* U0/U1 exist):
- **H-A (stride)**: a driver compiled for stride-1 silently reads the wrong registers on a
  4-byte-strided core. → the RegMap contract (U0) makes the placement part of the attestation.
- **H-B (DLAB leakage)**: on ST16C550-class clones, DLAB=1 exposes *more than divisor latches*;
  a sloppy bracket that "just sets DLAB and pokes around" can hit the EFR and reconfigure the
  part. → the DLAB bracket law (U1) allows **only** DLL/DLM/LCR accesses inside the bracket.
- **H-C (FCR gating)**: on Core16550, `FCR = 0x06` (reset FIFOs, bit0=0) programs nothing.
  → U1 law: every FCR write must have bit0=1 unless the *variant record* says plain-16450
  writes are meaningful.
- **H-D (reset-state drift)**: FIFOs already on at reset (Core16550) — init must not *assume*
  16450 mode; the U2 init writes FCR unconditionally rather than reading back.
- **H-E (readable FCR)**: never *rely* on reading FCR even where possible — the portable driver
  shadows its last FCR write in RAM (a plain struct field, not a device read).

---

## 3. Field-reality quirks (research; not in any datasheet)

From the Linux 8250 driver corpus and public errata discussions — these harden the design:

1. **FIFO presence must be probed, never assumed.** The original NS 16550 had a broken FIFO
   (the 16550A is the fixed part); clones vary. The classic identification algorithm: write
   FCR=0xC1 (enable + trigger 14), read IIR — IIR[7:6]=`11` → 16550A-class working FIFOs,
   `10` → broken-FIFO 16550, `00` → 16450/8250; then SCR write/read-back distinguishes
   8250 (no SCR) from 16450. U2's `uart_probe` implements exactly this and records the result
   in the driver state; U3's simulator answers as a 16550A.
2. **THRE interrupt re-arm bugs (Linux `UART_BUG_TXEN`/THRE quirks).** Numerous clones fail to
   re-assert the THRE interrupt after it is cleared once, or assert it spuriously; Linux probes
   this at init and falls back to a timer. **Design consequence (normative for this driver):
   TX never depends on THRE *interrupt* edges.** The polled driver reads LSR.THRE before every
   burst; the future IRQ driver (§9) must treat THRE interrupts as a hint and re-check LSR —
   this rule is written into U1's law commentary so the invariant survives into the IRQ slice.
3. **Modem-status quirks (`UART_BUG_NOMSR`)**: some integrations wire no modem lines and float
   MSR; the driver must gate all MSR-derived logic behind the variant record's `has_modem`
   flag (autoflow/software-flow slices only run when true).
4. **Busy-detect clones (DesignWare 8250)**: some clones reject LCR writes while busy (a
   vendor-specific "busy" interrupt). Our reconfigure-only-at-TEMT law (U1) makes the portable
   driver immune by construction — it never writes LCR mid-character.
5. **IIR ghost reads**: reading IIR in a race window can return "no interrupt pending" (0x01)
   even though a source is active; robust service loops re-poll LSR rather than trusting one
   IIR sample. (Matters for the deferred IRQ slice; documented now.)
6. **Chip-lot errata sheets** (specific date-code silicon bugs) are **not publicly indexed** —
   deferred to vendor support (§9). The defensive-sequencing rules above are the mitigation.

Sources: [Linux 8250 THRE-test patch discussion](https://patchew.org/linux/20260224121639.579404-1-alban.bedel@lht.dlh.de/),
[Serial Programming / 8250 UART Programming (Wikibooks)](https://en.wikibooks.org/wiki/Serial_Programming/8250_UART_Programming),
[Linux 8250 Kconfig (quirk inventory)](https://github.com/torvalds/linux/blob/master/drivers/tty/serial/8250/Kconfig),
[8250_dw.c (DesignWare busy quirk)](https://github.com/torvalds/linux/blob/master/drivers/tty/serial/8250/8250_dw.c),
[Linux 8250 core (UART_BUG_* flags)](https://docs.huihoo.com/doxygen/linux/kernel/3.7/8250_8c_source.html).

---

## 4. Architecture: how each piece maps onto existing BCIR machinery

**The thesis this driver demonstrates** (write it into every commit message):
*legality is law* (R-laws + the new UART protocol laws), *performance is a priced plan*
(K_BCIR over MMIO cost), *tuning is a frozen learned prior* (L1, the D3/tile-prior pattern),
*and the device's measured behavior closes the loop* (L2, calibloop + DurableLog) — with the
hot path containing **zero** learned inference (L0).

| Blueprint piece | Existing machinery to reuse (file-path anchors) |
|---|---|
| Register model + header generation | new `bcir/frontends/devices/uart_model.py`; generation pattern follows `bcir/frontends/models/manifest.py` style (dep-free stdlib, deterministic output) |
| RegMap contract | the R12 call-ABI contract pattern end to end: `bcir/frontends/cfront/abi.py::AbiContract/verify_abi_contract`, `CompileResult.abi_contracts` wiring in `pipeline.py`, MLIR `bcir.abi_contract` op + verifier clause in `mlir/lib/passes/BCIRVerifyPass.cpp`, negatives in `mlir/test/passes/verify_abi_contract.mlir` |
| Protocol laws (oracle) | checker style: `verify_uart_protocol(unit) -> list[str]` next to the existing op-level checkers (`check_attention` precedent — caller-passed messages, no verifier import); wired as advisory diagnostics in `pipeline.py` exactly like `abi_diagnostics` |
| Protocol laws (MLIR) | clause block in `BCIRVerifyPass.cpp` (the R22/R23 adjacency-walk pattern); ops in `mlir/include/BCIR/BCIRCoreOps.td` if a record op is needed |
| Volatile/MMIO ordering | already law: the §5.14 Phase-2 volatile qualifier + R3 MMIO discipline; the CMSIS fixture (`runtime/c/cmsis_gpio.h` + `cfront_driver_gpio.c`, `bcir/tests/test_driver_gpio.py`) shows the exact test idioms (MMIO-domain claims, store-only atomicity checks, ABI contracts) |
| Two-TU build + link | `bcir-cc --linkable` (C twin, `runtime/c/bcir_cc.c::cc_emit_linkable`) and oracle `--linkable` (`bcir/frontends/cfront/emit.py::emit_linkable`); the `#link` stanza in `tools/c/check_runtime.sh` is the gate template |
| Planned streams + overlap | `bcir/kbcir/train_graph.py::train_run_module/schedule_train_run/PipelineCertificate` (D1.5) is the exact template for `uart_service_module` + `schedule_uart_service`; `bcir/gem/schedule.py::schedule_eft/execute_tokens/durations_from` |
| Learned prior (L1) | `bcir/kbcir/tile_prior.py` **verbatim as the pattern**: features → logistic → Q8 freeze → certificate (guided == exhaustive, mismatches 0) → persisted envelope tied to `cal_gen` (`save_tile_prior/load_tile_prior`) |
| Telemetry + closed loop (L2) | `bcir/telemetry.py::DataDNA/Broker/TelemetryRing/DurableLog/load_durable_log`; `bcir/kbcir/calibloop.py::close_loop/CalibrationCertificate` |
| Provenance | `bcir/kbcir/provenance.py::build_manifest/replay` (R13), the 0.4b `DecisionRecord` envelope + `bcir.run --replay` exit-code contract |
| Simulated device | new `runtime/c/sim16550.c` (host tool, the `bcir_train.c`/`test_train.c` posture: caller-owned state, deterministic, libc-only harness) |

**New law numbering.** Do **not** mint a new global R-number for the UART protocol rules in the
first pass. They enter as (a) op-level/unit-level checks on the oracle (advisory diagnostics,
like `abi_diagnostics`) and (b) an MLIR clause reported under **R3** (MMIO discipline — the
DLAB/FCR/ordering rules are refinements of "device access is ordered and well-formed") with the
message prefix `R3-uart:`. If the maintainers later want a numbered law (R24), it is a rename,
not a redesign. This keeps `gen_status`'s R-table stable and honors non-disturbance.

---

## 5. The build slices

### U0 — The device registry: one logical model, N placements, a RegMap contract

**Goal.** A single normative 16550 model that generates the vendor-style headers for every
placement, plus a compile-time RegMap contract verified like the R12 ABI contract — so a driver
binary *attests which register map it was laid out for* and a placement mismatch is caught by
law, not by debugging.

**Files.**
- `bcir/frontends/devices/__init__.py` — new package (dep-free stdlib; exports below).
- `bcir/frontends/devices/uart_model.py` — the model + generator.
- `bcir/frontends/devices/fixtures/` — golden generated headers (checked in for diffability).
- `runtime/c/uart16550_regs.h` — the generated header for the default placement (byte-strided),
  checked in with a `/* GENERATED by uart_model.py -- do not edit */` banner.
- `bcir/tests/test_uart_model.py` — registered in `run_all._MODULES`.

**Design.**
```python
@dataclass(frozen=True)
class UartReg:      # one logical register
    name: str       # "RBR", "THR", "IER", "IIR", "FCR", "LCR", "MCR", "LSR", "MSR", "SCR", "DLL", "DLM"
    slot: int       # 0..7 (the A2:A0 address)
    access: str     # "r" | "w" | "rw"
    dlab: int | None  # None = don't care; 0/1 = only reachable at that DLAB value
    fields: tuple   # ((bit_hi, bit_lo, field_name, doc), ...) from §1.1

@dataclass(frozen=True)
class UartPlacement:
    name: str            # "pc16550d" | "core16550_apb" | "axi16550"
    stride: int          # 1 | 4
    base_offset: int     # 0 | 0 | 0x1000
    reg_width: int       # 1 | 4 | 4 (bytes per register lane)
    fcr_reset: int       # 0x00 | 0x01 | 0x00
    fcr_gated: bool      # False | True | False   (H-C)
    has_afe: bool        # MCR5 autoflow exists
    has_modem: bool      # modem lines wired (H-NOMSR)
    ref_clk_hz: int      # the baud reference the divisor law uses
    irq_level: bool      # level-sensitive interrupt output (AXI) vs pin

UART16550: tuple[UartReg, ...]          # THE normative table (from §1.1)
PLACEMENTS: dict[str, UartPlacement]    # the three §2 placements

def emit_header(placement: UartPlacement) -> str
    # deterministic C header: one `#define UART_<REG>_OFF` per register (stride/base applied),
    # one `#define UART_<REG>_<FIELD>` + mask/shift per field, the trigger-level enum, and a
    # `uart_regmap_contract` static const struct literal (name, stride, base, fcr_reset,
    # ref_clk_hz) the driver TU embeds — the RegMapContract's C-side anchor.

@dataclass(frozen=True)
class RegMapContract:   # mirrors AbiContract exactly
    placement: str
    stride: int
    base_offset: int
    regs: tuple          # (name, offset, access, dlab) per register — the laid-out facts

def regmap_contract_for(placement: UartPlacement) -> RegMapContract
def verify_regmap_contract(contract, placement) -> list[str]   # [] == clean; lies caught
```
MLIR half: op `bcir.regmap_contract` (SymbolNameAttr `$sym_name`, StrAttr `$placement`,
I64Attr `$stride`, I64Attr `$base_offset`, DenseI64ArrayAttr `$reg_offsets`,
`hasVerifier=1`) in `BCIRCoreOps.td` next to `BCIR_AbiContractOp`; the `-bcir-verify` clause
checks the placement name against the normative matrix (unknown placement → error; offsets
must equal `base + slot*stride`) — copy the R12 clause structure (`kMatrix` static table)
verbatim, message prefix `R3-uart:`. Negatives in `mlir/test/passes/verify_regmap_contract.mlir`
(wrong-stride lie, unknown placement, legal record) + `check_passes.sh` stanza.

**Tests** (`test_uart_model.py`):
- `test_the_normative_table_matches_the_datasheet` — pin slot/access/dlab for all 12 registers
  and the exact FCR/LCR/LSR/IIR field tables from §1.1 (golden asserts).
- `test_headers_generate_deterministically_for_all_placements` — same input → identical bytes;
  the three placements produce offsets `{0,...,7}`, `{0x00..0x1C step 4}`, `{0x1000..0x101C}`.
- `test_generated_header_compiles_through_both_rails` — the header + a 5-line probe TU through
  `compile_unit` and (toolchain-gated) `bcir-cc`; clean.
- `test_a_tampered_regmap_contract_is_caught` — `dataclasses.replace` lies (stride, offset,
  placement) each produce messages; the clean contract produces [].
- `test_checked_in_header_is_current` — regenerate and byte-compare against
  `runtime/c/uart16550_regs.h` (the drift guard, like `gen_status --check`).

**Gates.** Suite + registry guard; `check_passes.sh` (new stanza); `check_links`.
**Commit.** `devices: U0 -- the UART16550 registry (one model, three placements, the RegMap contract)`

### U1 — The protocol laws

**Goal.** The five §1/§2/§3-derived legality rules as laws: caught on the oracle as advisory
diagnostics (exactly like `abi_diagnostics`), enforced on the MLIR rail under `R3-uart:`, C-twin
checked where the C rail lowers the same constructs. All vacuous on non-UART code (the checks
key on accesses to resources whose names come from the U0 header's register offsets).

**The laws** (message shapes are normative — tests grep them):
1. **DLAB bracket** — `R3-uart: DLL/DLM access outside a DLAB=1 bracket` and its dual
  `R3-uart: <REG> access inside a DLAB=1 bracket` for REG ∈ {RBR, THR, IER}. A bracket = the
  claim range between an LCR store whose value has bit7=1 and the next LCR store with bit7=0
  **in straight-line order** (loops/branches around a bracket → conservative: reject with
  `R3-uart: DLAB bracket must be straight-line`). Inside a bracket only DLL/DLM/LCR accesses
  are legal (hazard H-B: clone EFRs behind DLAB).
2. **Divisor sanity** — `R3-uart: divisor 0 is forbidden` (a DLL=0 store inside a bracket whose
  DLM store is also 0, or constant-folded equivalents); plus the achieved-baud bound when both
  divisor and `ref_clk_hz` are compile-time constants: `R3-uart: baud error 4.7% exceeds 3.0%`.
3. **FCR gating** — on a placement with `fcr_gated`: `R3-uart: FCR write with FIFO-enable clear
  programs nothing on <placement>` (hazard H-C).
4. **Reconfigure-only-at-idle** — an LCR/DLL/DLM store must be dominated (straight-line: preceded
  without an intervening THR store) by an LSR read; message
  `R3-uart: line reconfiguration without a dominating LSR (TEMT) check`. (This also immunizes
  against the DesignWare busy-detect quirk, §3.4.)
5. **Read-effect preservation** — LSR/IIR/MSR/RBR reads are *effectful* (they clear device
  state): mark them so CSE/dedup can never merge two reads or delete an "unused" one. Oracle:
  lower these MMIO loads with `hazard="mmio_read_clear"` (a new claim field, digest-excluded,
  vacuous elsewhere — the W1 extent-provenance precedent for adding claim fields); law:
  `R3-uart: two LSR reads merged` fires if claim counts show a merged/deleted read against the
  source count. MLIR: the existing volatile-load op already blocks reordering; add the clause
  asserting `bcir.load` ops on UART register resources carry the volatile qualifier —
  `R3-uart: non-volatile access to UART register <name>`.

**Files.** `bcir/frontends/devices/uart_laws.py` (oracle checker `verify_uart_protocol(lowered,
placement) -> list[str]`); wiring in `bcir/frontends/cfront/pipeline.py` (a
`uart_diagnostics` list on `CompileResult`, populated only when the unit includes a U0 header —
detect via the embedded `uart_regmap_contract` symbol); MLIR clause in `BCIRVerifyPass.cpp`;
negatives `mlir/test/passes/verify_uart_protocol.mlir`; oracle tests
`bcir/tests/test_uart_laws.py` (registered); `check_passes.sh` stanza; LangRef paragraph under
the R3 section.

**Tests.** One positive (the U2 driver compiles with zero `uart_diagnostics`) + one negative per
law on both rails (oracle message + MLIR `expected-error`), plus the vacuousness pin: compiling
`cfront_driver_gpio.c` and the whole existing corpus produces zero `uart_diagnostics`
(non-disturbance measured, not assumed).

**Commit.** `devices: U1 -- the UART protocol laws (DLAB bracket, divisor, FCR gating, idle-reconfig, read effects)`

### U2 — The polled driver + the simulated device + the linked, running binary

**Goal.** The headline: a real driver, compiled by BCIR on both rails, linked as emitted
artifacts only, running against a spec-exact simulated 16550, behavior-equivalence- and
law-gated. This is the first *running* BCIR-compiled driver.

**Files.**
- `runtime/c/uart16550_regs.h` — from U0 (already checked in).
- `runtime/c/uart_driver.c` — the driver TU (compiles through BOTH rails; stays inside the
  supported C subset — the CMSIS fixture proves every idiom used here already works):
  ```c
  typedef struct uart_state {            /* driver-private, RAM domain */
    uint32_t fcr_shadow;                 /* H-E: never read FCR back */
    uint32_t trigger;                    /* the configured trigger level (bytes) */
    uint32_t fifo_ok;                    /* probe result: 0 none, 1 broken, 2 ok (16550A) */
    uint32_t tx_burst;                   /* plan-chosen TX burst cap (<= 16) */
  } uart_state;

  uint32_t uart_probe(volatile uart_regs_t *u, uart_state *st);
      /* the §3.1 identification: FCR=0xC1, read IIR[7:6]; SCR probe; returns fifo_ok. */
  uint32_t uart_init(volatile uart_regs_t *u, uart_state *st,
                     uint32_t divisor, uint32_t lcr_format, uint32_t trigger_code);
      /* the §1.4 sequence exactly; returns 0 ok / nonzero law-shaped error code.  */
  uint32_t uart_tx(volatile uart_regs_t *u, uart_state *st,
                   const uint8_t *buf, uint32_t len);
      /* LSR-checked bursts: while remaining: poll LSR until THRE; write
         min(tx_burst, 16, remaining) bytes; returns bytes sent (== len unless timeout). */
  uint32_t uart_rx_drain(volatile uart_regs_t *u, uart_state *st,
                         uint8_t *buf, uint32_t cap, uint32_t *err_counts);
      /* while LSR.DR and cap: triage LSR7 -> per-byte OE/PE/FE/BI into err_counts[4];
         read RBR; returns bytes drained. ONE LSR read per iteration feeds both the DR
         test and the error triage (the CSE win the plan prices).                    */
  uint32_t uart_flush(volatile uart_regs_t *u, uint32_t spin_budget);
      /* poll LSR until TEMT or budget exhausted; returns spins used (bounded loop).  */
  uint32_t uart_selftest(volatile uart_regs_t *u, uart_state *st);
      /* MCR4 loopback: pattern out, pattern back, MSR mirror check (SS1.7).          */
  ```
- `runtime/c/sim16550.c` + `sim16550.h` — the simulated device (host-side, deterministic):
  a `sim16550` struct holding the two 16-deep FIFOs (RX bytes carry 3 error bits), shift-register
  emulation, LSR/IIR/MSR state machines, the trigger + 4-char-time timeout counters (time
  modeled in *character ticks* advanced by an explicit `sim_tick(dev, n_char_times)` — no wall
  clock, fully reproducible), the §1.2 overrun rule (discard shift byte, FIFO intact), the §1.7
  loopback path, and per-variant knobs from `UartPlacement` (fcr_gated, fcr_reset).
  Access entry: `uint32_t sim_read(sim16550*, uint32_t offset)` / `void sim_write(sim16550*,
  uint32_t offset, uint32_t val)` implementing the §1.1 mux (direction, DLAB, mode).
  **The sim is itself spec-tested** (see tests) — it is the reference the driver runs against,
  so its fidelity gates first.
- `runtime/c/uart_main.c` — the harness `main`: builds a sim, maps the driver's
  `volatile uart_regs_t *` onto it (the sim exposes a byte-array "MMIO window" whose
  reads/writes trap into sim_read/sim_write via a tiny accessor shim in the harness TU),
  runs: probe → init (PG143 vector: 100 MHz/56k/8E2) → selftest → TX a golden buffer →
  sim_tick loop injecting an RX script (bursts, gaps that trip the timeout, one parity error,
  one overrun) → drain → flush; prints a deterministic transcript (bytes, error counts, spins,
  LSR snapshots) to stdout.
- `bcir/tests/test_uart_driver.py` (registered) — orchestrates: compile driver TU through the
  oracle (clean, zero `uart_diagnostics`, MMIO domains present, RegMap + ABI contracts
  recorded); `--linkable`-emit driver via BOTH rails; host-compile sim + harness; **link
  emitted-driver + sim + harness into one binary**; run; compare the transcript against (a) the
  same binary built from the original driver source (behavior equivalence) and (b) a golden
  transcript checked in after first measurement.
- `tools/c/check_runtime.sh` — a `#uartdrv` stanza: the C-twin `--linkable` build of the same
  three-TU program runs and its transcript byte-matches the oracle-emitted build's.

**Sim spec-fidelity tests** (in `test_uart_driver.py`, run before any driver test):
`test_sim_reset_states` (IIR=0x01, LSR=0x60, per-variant FCR), `test_sim_trigger_and_timeout`
(trigger 8: RDA asserts at 8, clears at 7; timeout fires after exactly 4 idle char-ticks with
1 byte held, cleared by one read), `test_sim_overrun_discards_shift_byte_not_fifo`,
`test_sim_loopback_mirrors_mcr_to_msr`, `test_sim_dlab_mux_and_fcr_gating` (per placement),
`test_sim_thre_semantics` (write 16 → THRE clears; drain to empty → THRE sets; TEMT lags by
one char-tick).

**Driver tests**: `test_driver_compiles_clean_on_both_rails_with_contracts`,
`test_probe_identifies_the_sim_as_16550a`, `test_init_follows_the_canonical_sequence`
(assert the claim order: LSR-read < LCR(DLAB=1) < DLL < DLM < LCR(DLAB=0) < FCR < MCR < IER),
`test_tx_bursts_never_exceed_fifo_and_never_write_without_thre`,
`test_rx_drain_triage_counts_the_injected_errors_exactly`,
`test_emitted_to_emitted_binary_matches_reference_behavior` (the headline),
`test_selftest_passes_in_loopback`.

**Gates.** Full suite; `check_runtime.sh` incl. `#uartdrv`; `check_passes.sh`; doc gates.
**Commit.** `devices: U2 -- the polled UART driver, linked from emitted TUs, running against the spec-exact sim`

### U3 — The device model registered as resources + planned service streams

**Goal.** Lift the driver's steady state into the claim-graph world so K_BCIR can price it:
a `uart_service_module()` (mirroring `train_step_module`) whose claims are the service-loop
stages over registry resources — [poll-LSR] → [classify] → [drain-k] / [fill-k] → [account] —
with TX-fill and RX-drain as separate streams sharing only the LSR read.

**Files.** `bcir/kbcir/uart_graph.py` (cost-side; imports no verifier):
`UartServiceSpec(trigger, tx_burst, rx_batch, arrival_class)`, `uart_service_module(spec)`,
`service_run_module(spec, n_services)` (the D1.5 `train_run_module` pattern),
`schedule_uart_service(spec, n, h, theta) -> (PipelineCertificate, GemSchedule)` reusing
`durations_from/schedule_eft/execute_tokens` unchanged. MMIO claim costs: give the register
resources `Domain.MMIO` and let the calibrated `TargetProfile.base_overhead`/`gather_penalty`
price uncached device reads (that is what those axes are for); the RX-drain claim count = batch
size, the TX-fill count = burst size.
Tests `bcir/tests/test_uart_graph.py`: module is R-law clean (`verify(module) == []`),
plans/hydrates to a StreamPack (R10/R11 clean — the `test_train_graph` assertions), the token
DAG overlaps TX-fill with RX-account stages while the LSR-read claim stays shared
(slot assertions like `test_pipelined_run_respects_raw_...`), and the certificate orders
pipelined ≤ barriered ≤ serial with a measured-then-pinned win gate.

**Commit.** `kbcir: U3 -- the UART service loop as priced, pipelined claim streams`

### U4 — Telemetry: the driver emits DataDNA; the log is durable

**Goal.** Every service event becomes a `DataDNA` record; the harness publishes through
`Broker` to a `TelemetryRing` (small, backpressure counted) + `DurableLog`; the log replays.

**Design.** Field mapping (reuse `DataDNA` as-is — no schema change):
`segment_id="uart:<placement>"`, `claim_id=service sequence number`, `cycles=spin count`,
`bytes=bytes moved`, `misses=timeouts hit (normalized 0-100 over the window)`,
`thermal=RX-FIFO high-water (0-100 = level/16*100)`, `voltage=error count (normalized)`,
`utilization=FIFO occupancy at service entry (0-100)`, `provenance=R13 digest prefix of the
driver build`. The C harness prints one `telemetry: seq=.. spins=.. bytes=..` line per service;
the Python test parses and feeds the Broker (no C JSON writer needed — honest and small).
Tests in `test_uart_driver.py`: the durable log round-trips the harness run bit-for-bit;
ring overflow on a long run shows `stats.dropped > 0` counted.

**Commit.** `devices: U4 -- UART service telemetry through Broker/DurableLog (counted backpressure)`

### U5 — The learned trigger/burst prior (L1) + the measured replan (L2)

**Goal.** The ML statement. The (trigger ∈ {1,4,8,14}) × (tx_burst ∈ {1,2,4,8,16}) ×
(rx_batch ∈ {1,4,8,16}) candidate space is priced exactly by U3's planner per *arrival class*
(steady / bursty / interactive — synthetic arrival scripts defined in the sim); a logistic
prior over cheap features orders the candidates; the exact search verifies (mismatches MUST be
0); the frozen Q8 table persists tied to `cal_gen`; and `close_loop` re-plans when measured
telemetry (U4) disagrees with the assumed arrival class, certifying the win.

**Files.** `bcir/kbcir/uart_prior.py` — copy the `tile_prior.py` structure function-for-function:
`uart_features(spec, cand, target)` (trigger/16, burst/16, batch/16, arrival-class one-hots,
autoflow flag [the §1.6 asymmetry: with AFE at trigger≤8 the drain-to-empty rule makes small
batches expensive], MMIO cost ratio from the profile, bias),
`prior_samples/train_uart_prior/FrozenUartPrior.order/guided_plan_service` (admissible early
exit: the serial-cost lower bound — a candidate whose priced makespan equals the per-byte MMIO
floor `ceil(bytes * base_overhead)` is unbeatable; when the floor is unreachable the guided
search degenerates to exhaustive — pin that honestly, the tile-prior test pattern),
`UartPriorCertificate` (guided == exhaustive optimum per arrival class, mismatches 0, node
reduction recorded), `save_uart_prior/load_uart_prior` (envelope kind `"bcir.uart_prior"`,
schema 1, `cal_gen` + placement tied, stale/newer/wrong refusals).
L2 half: `replan_from_log(log_path, spec, h, theta)` — classify the DurableLog's measured
arrival pattern (inter-service gap histogram → class), re-run the exact search under the
measured class, emit a `CalibrationCertificate`-shaped record (reuse
`calibloop.rescore_plan` where it fits) whose win ≥ 0 by construction.
Tests `bcir/tests/test_uart_prior.py`: certificate admitted per class with real node reduction
(measure, gate at ~half), honest degeneration case, persistence + staleness refusals (copy the
tile-prior tests), and the L2 end-to-end: a bursty log re-plans trigger 14→8 (or whatever the
priced model actually chooses — pin after measuring) with win > 0 recorded.

**Commit.** `kbcir: U5 -- the UART trigger/burst prior (L1, certificate-gated) + measured replan (L2)`

### U6 — Autoflow + software flow control (variant-conditional)

**Goal.** §1.6 as driver capability: on `has_afe` placements enable AFE per the MCR5+MCR1 rule
and encode the drain-to-empty consequence in the plan (U5 feature already carries the flag);
on non-AFE `has_modem` placements implement software RTS/CTS with the documented one-char race
window; on `!has_modem` placements both are compile-time absent.
**Files.** extend `uart_driver.c` (`uart_flow_enable`, the drain-to-empty variant of
`uart_rx_drain`), `sim16550.c` (auto-RTS/auto-CTS per the §1.6 timing: deassert at trigger,
reassert per trigger class; CTS mid-stop-bit rule), tests (`test_autoflow_rts_hysteresis_matches_spec`
per trigger level, `test_auto_cts_holds_the_next_byte_not_the_current`,
`test_software_flow_race_window_is_at_most_one_char` — sim-measured).
**Commit.** `devices: U6 -- autoflow (AFE) + software flow control, variant-conditional`

### U7 — Provenance wrap + docs + the release artifact

**Goal.** The deliverable statement: one build = driver binary + R13 manifest + ABI & RegMap
contracts + plan `DecisionRecord` (replayable via `bcir.run --replay`'s exit-code contract) +
the frozen prior envelope + the DurableLog schema — the *proof-carrying driver*. Write
`docs/UART_DRIVER.md` (user-facing: how to build for each placement, how to retune, what each
certificate means), update `BCIR_DRIVER_KERNEL_ROADMAP.md` Part IV (slice 1 DONE with pointers)
and `BCIR_MASTER_ROADMAP.md` Phase D, regenerate STATUS.
**Commit.** `devices: U7 -- the proof-carrying UART driver artifact + docs`

---

## 6. Performance model and gates (what "MAX performance" means, measurably)

All measured on the deterministic sim (char-tick clock), so the numbers are exact and CI-stable:

| Lever | Baseline | Expected | Gate (measure first, pin at ~⅔ of measured) |
|---|---|---|---|
| TX FIFO bursts vs byte-at-a-time | 1 LSR poll + 1 THR write per byte | 1 poll + ≤16 writes | MMIO ops/byte ≤ 1.2 at burst 16 (vs 2.0 baseline) |
| RX trigger 8/14 + timeout vs trigger 1 | 1 service per byte | 1 service per ~8–14 bytes sustained | services/byte ≤ 0.15 at trigger 8, sustained script |
| Shared LSR read (CSE) | 2 LSR reads per drain iteration | 1 | pinned by claim-count assert in U2 |
| Pipelined service streams (U3) | serial stage sum | token-DAG overlap | pipelined ≤ barriered, win > 0, certificate |
| Tail latency under timeout | unbounded staleness at trigger>1 without timeout | ≤ 4 char-times | sim-measured exactly 4 ticks (spec), pinned |
| Prior-guided planning | exhaustive candidate walk | guided, exact optimum | mismatches == 0, node reduction ≥ pinned |
| Measured replan (L2) | stale plan under wrong arrival class | re-planned | certificate win ≥ 0, > 0 on the bursty script |

The honest boundary, stated in every doc: these are *simulated-device* wins (MMIO cost = model,
time = char-ticks). Real-silicon numbers wait for the rig (§9) — the same discipline as the
Clang-comparison perf budgets (strict correctness in CI, perf floors bare-metal-only).

## 7. ML placement summary (the compliance card)

- **L0 (hot path)**: `uart_tx`/`uart_rx_drain` contain zero learned inference, zero float — the
  compiled-out decisions are `tx_burst`, `trigger`, `rx_batch` constants baked by the plan.
- **L1 (plan time)**: `FrozenUartPrior` — Q8 integer table, deterministic order, certificate
  (guided == exhaustive, mismatches 0), persisted + `cal_gen`-tied, staleness refused loudly.
- **L2 (checkpoint)**: `replan_from_log` — measured arrival class → exact re-plan → certified
  win ≥ 0; runs offline, never in the service loop.
- **L3**: none in this program (no meta-policy changes).
- **Two-truth**: `uart_graph.py`/`uart_prior.py` import no verifier; all legality flows through
  `verify_uart_protocol` + the MLIR clauses with caller-passed messages.

## 8. Simulated-16550 behavioral contract (condensed; the sim tests in U2 pin each line)

Reset per §1.1 tables (variant FCR/DLL). Mux per §1.1. Trigger/RDA per §1.2. Timeout: an
internal counter reset by (byte arrival | RBR read), incremented by `sim_tick`; fires at 4 with
FIFO non-empty; cleared by one RBR read. THRE/TEMT: THRE = TX FIFO empty; TEMT = THRE ∧ shift
idle; shift takes 1 char-tick per byte. OE per §1.2 (discard shift byte). Errors: injected via
`sim_inject(dev, byte, err_bits)`; PE/FE/BI ride the FIFO per byte; LSR7 = any error byte
present. Loopback per §1.7. Autoflow per §1.6 (U6). IIR returns the highest-priority pending
source per the §1.1 ladder with bit0 active-low; FIFO-enabled bits reflect FCR0. LSR read
clears OE/PE/FE/BI (head-of-FIFO error bits latch into LSR at the read that exposes them).
The sim identifies as a 16550A (IIR[7:6]=11 after FCR enable) and has a functional SCR.

## 9. Deferred items (and why)

| Item | Why deferred | Unblocks when |
|---|---|---|
| Interrupt/ISR claim model + IRQ-driven driver | BCIR has no event-triggered phase model; polled slice needs none (roadmap Part 0 finding). Design note: THRE-as-hint rule (§3.2) is already written so the IRQ port cannot regress. | after U7; needs an `event.irq` phase-trigger design on the IR |
| Bus-master DMA | not in the 16550 family (pins only); platform DMA engines are their own drivers | a DMA-engine registry model |
| Real-silicon measurement (PMU/RAPL, a physical UART) | rig-gated, same as CT4 | `HARDWARE_VALIDATION.md` runbook rig |
| Chip-lot errata sheets (date-code silicon bugs) | not publicly indexed | **vendor support** (TI/Exar/Microsemi/AMD contacts) |
| 16750/64-byte-FIFO + sleep modes, Exar EFR feature set | out of the 550 contract; H-B law already fences the EFR hazard | a future variant record |
| OS integration (termios/tty) | Phase D slice 1 is freestanding | the POSIX layer track |

## 10. Process requirements (restated, with anchors)

1. **Six-artifact law pattern** for every law: LangRef text, oracle law + tests, MLIR clause +
   `-verify-diagnostics` negatives, C twin where the C rail lowers it, `check_passes.sh` stanza,
   `gen_status` sweep (automatic).
2. **Register every new test file** in `bcir/tests/run_all.py::_MODULES` —
   `test_registry_complete.py` fails the suite otherwise.
3. **Gates per slice**: touched tests + `python -m bcir.tests.run_all --tier quick` (expect the
   documented toolchain-gated failure count to stay constant; any new failure is yours) +
   `tools/c/check_runtime.sh` when `runtime/c` changes + `PATH=/usr/lib/llvm-18/bin
   tools/wsl/check_passes.sh` when `mlir/` changes + `gen_status > docs/STATUS.md` +
   `check_links.py` + `check_retired_paths.py`.
4. **Non-disturbance**: every new claim field digest-excluded by default (the R13 fixed fold
   list in `bcir/kbcir/provenance.py` makes this automatic); every law vacuous off-UART, proven
   by a corpus-sweep test, not asserted.
5. **Commit trailer** discipline per repo convention; one slice = one commit = one reviewable
   unit; PR after the wave with the gates listed.
6. **Measured-then-pinned**: run the number, then write the gate with headroom; never
   copy a gate from this document without re-measuring (the numbers here are *expected shapes*,
   not oracle truth).

## 11. Acceptance checklist (the definition of done for the program)

- [ ] U0: three placements generate; contracts round-trip; tampering caught on both rails.
- [ ] U1: five laws fire on negatives on both rails; whole existing corpus produces zero
      UART diagnostics.
- [ ] U2: the emitted-TUs-only binary runs the probe/init/selftest/TX/RX/flush transcript
      byte-identically to the reference build on both rails; sim passes its spec-fidelity suite.
- [ ] U3: service module is R-law clean, hydrates, pipelines with a certified win.
- [ ] U4: DurableLog replays a harness run bit-for-bit; ring backpressure counted.
- [ ] U5: prior certificate admitted (mismatches 0) per arrival class; persistence + staleness
      refusals pinned; L2 replan win > 0 on the bursty script.
- [ ] U6: autoflow hysteresis matches §1.6 per trigger level; software-flow race ≤ 1 char.
- [ ] U7: the proof-carrying artifact (manifest + contracts + DecisionRecord + prior envelope +
      log) replays via the 0.4b CLI contract; docs updated; STATUS regenerated.
