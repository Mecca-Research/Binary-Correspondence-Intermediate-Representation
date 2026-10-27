# BCIR Event Timers — the build blueprint (a proposal)

> **Status (2026-10-27): a proposal. Nothing in it has landed.** The repository has no timer model,
> timer law, timer simulator or timer driver. This document works toward the event-timer program's
> D0 research contract (driver roadmap §5.1). It supplies the normative model, provenance, the
> device schema, the variant and errata matrix, laws stated so that a test can drive each violation,
> a simulator contract and a telemetry signal set. The manifest shape, the threat and failure model
> and the performance baseline that D0 also asks for are not written yet. The build slices in §6
> are proposals, not milestones.
>
> Every device fact cites a page. Where the documents leave a point open, the text marks the
> choice as a *reading* and repeats it in §8. Arithmetic that the documents do not print is marked
> *derived* and collected in Appendix A.
>
> **What it is for.** BCIR's event phases (EV1–EV3, `bcir/kbcir/events.py`) order an asynchronous
> handler against program flow by hazards and masking. The IR has no source of *time*.
> `dma.done:<engine>` (`bcir/kbcir/dma.py`) arrives whenever it arrives, and the simulated 16550
> keeps time as a count of character times (UART blueprint §8). Event timers are where the event
> phases get a real time source.
>
> **Where it sits.**
> - It is item 5 of the native-kernel dependency track (driver roadmap §5.3: "Clock, timer,
>   interrupt-controller, and event-delivery substrate").
> - It meets §3.3's rule for a shared primitive: two independent driver classes need it, namely
>   the UART's timeouts and the DMA completion bound.
> - The HPET is an IA-PC platform timer, but this is the **event-timer program**: the HPET is one
>   placement beside the AVR RTC.
> - The Intel driver track (x2APIC, IOAPIC, platform records; branch
>   `claude/intel-platform-records`, paused 2026-10-27) stays paused. No slice here programs an
>   I/O APIC, a local APIC or an x2APIC register. Interrupt routes are data in the model.
>
> **Primary sources.** These are the user's documents: untrusted reference data, read in
> full.
> - `HPET`: IA-PC HPET (High Precision Event Timers) Specification, Revision 1.0a, October 2004,
>   33 pages. Its PDF pages and printed pages are the same.
> - `TB3213`: Microchip TB3213, "Getting Started with Real-Time Counter (RTC)", DS90003213C
>   (revision C, 02/2021), 26 pages, covering tinyAVR 0/1, megaAVR 0 and AVR DA.
>   Its PDF pages and printed pages are the same.
>
> **Supplementary sources.** This is code, fetched on 2026-10-27 at the user's request. Each file
> is authoritative only for what its project implements, and this document cites it by name and
> function or line of the fetched text:
> - `linux-x86-hpet-c`: Linux `arch/x86/kernel/hpet.c`, GPL-2.0-only.
> - `zephyr-hpet-c`: Zephyr `drivers/timer/hpet.c`, Apache-2.0.
> - `edk2-hpet-timer-dxe-c`, `edk2-hpet-register-h`, `edk2-hpet-table-h`: TianoCore EDK2,
>   BSD-2-Clause-Patent.
> - `qemu-hpet-c`: QEMU `hw/timer/hpet.c`, LGPL-2.1-or-later.
>
> These files ground the laws in real drivers and the simulator in a real model. They are cited,
> never copied into BCIR's tree. Where one of them disagrees with a document, §7.4 records it and
> nothing here picks a side silently.
>
> The network policy refused the ATtiny1614 data sheet, the MC146818A data sheet and the kernel's
> timekeeping and HPET guides. TB3213 is the AVR RTC's only source.
>
> **Repository files read (read-only).**
> - `bcir/kbcir/events.py`, `bcir/kbcir/dma.py`, `bcir/silicon.py`, `bcir/kbcir/microbench.py`.
> - `docs/BCIR_TARGET_ACCESS.md`, `docs/kernel/BCIR_DRIVER_KERNEL_ROADMAP.md`,
>   `docs/kernel/TELEMETRY_ENVELOPE_ABI.md`, `docs/security/laws.md`, and the UART blueprint.
> - On the UART branch (`claude/uart-driver`, its worktree, untouched):
>   `bcir/frontends/devices/uart_model.py` and `uart_laws.py`. This is the registry pattern that
>   this program copies.

---

## 0. Purpose and scope

### 0.1 What a timer adds

| Use | Today | With an event timer |
|---|---|---|
| Driver timeouts | A UART poll is bounded by a spin count (`uart_flush(u, spin_budget)`, UART blueprint U2). The 16550's own character timeout is internal to the device and counted in character times (UART §1.2, §8). | A deadline is counted in the timer's ticks, armed as a one-shot comparator and delivered as an event phase. The device's own timeout keeps its own clock on a shared simulated timeline (§4.5). |
| A DMA completion bound | `dma_transfer_module` waits for `dma.done:<engine>` with no bound (`dma.py`). | A second event source, `timer.expire:<placement>.t<n>`, races the completion. Outcome rules O1–O5 decide the result (§4.5), and a timeout never proves that the engine stopped. |
| Periodic telemetry sampling | Samples are taken wherever code happens to run. | A periodic comparator (HPET) or the PIT (AVR) paces sampling. Missed periods are counted, never silently coalesced (§5.4). |
| A measured clock | `silicon.py` reads OS clocks. The native rig records the clocksource and the timer quantum (`microbench.NativeEvidence`). | An elapsed-time clock whose rate comes from the counter's own declaration, with that provenance recorded (T4). It is never a TSC tick count and never core cycles (§5.2). |

### 0.2 Terms
- **Counter**: the free-running count, meaning the HPET main counter or the AVR RTC's CNT.
- **Tick**: one increment of a counter.
  - A counter's **period** is its seconds per tick.
  - Its **declaration** is where that period comes from (T4).
- **Comparator**: a match register and its compare logic. On the HPET this is "timer" n (HPET
  p.6). On the AVR it is CMP, plus PER for overflow.
- **Arm**: write a comparator's target.
- **Unmask**: EV2's explicit `irq.unmask:<source>` claim. Arming and unmasking are distinct acts,
  and the laws keep them apart.
- **Placement**: one documented part laid onto the one model, in the `uart_model.UartPlacement`
  sense. The placements are `hpet_spec` and `avr_rtc_tb3213`. `qemu_hpet` is a simulator
  profile.

### 0.3 Scope

**In scope:**
- the normative model of both families;
- the variant matrix;
- T-laws on the oracle rail;
- a deterministic simulator with specification and QEMU profiles;
- UART and DMA timeouts on the simulator;
- timer telemetry and a time-source record.

**Out of scope:**
- I/O APIC, x2APIC and local-APIC programming, which belong to the paused track.
- ACPI table parsing. That is item 4 of the native track; this document records the HPET table
  as data (§1.6).
- Any real-hardware binding. This guest exposes no HPET (§5.3), and there is no AVR rig.
- TSC or core-frequency calibration, which the measurement discipline refuses (§5.2).

### 0.4 Provenance discipline
- A fact is a document's statement at a cited page.
- Where a document is silent, the text says "not stated" and the model carries `None`.
- Where two statements disagree, §7 records both and the model takes neither as fact.
- A law that must decide such a point says which reading it enforces and why, and the choice
  becomes an open question in §8.
- Supplementary code is evidence of practice. It is never a source of normative device facts.

---

## 1. The normative timer model

### 1.0 One model

An event timer has the following parts:
- **A counter**: its width, its top, its period with that period's declaration, and its run or
  halt state.
- **Comparators.** Each is one of four kinds:
  - **match**: fires when the counter reaches the target;
  - **accumulate**: matches, then adds the period (the HPET periodic mode);
  - **top**: fires at the counter's top and resets it (the AVR overflow);
  - **tap**: fires every 2^k source cycles (the AVR PIT).
- **A status latch** per source, with its clear rule.
- **Delivery as data**: routes, messages and vectors.
- **Access-shape rules.**
- **Synchronization rules**, where writes cross clock domains.

The two families fire under different conventions, and the model keeps them apart in a field
called `fire_on`:
- The HPET fires when the counter equals the match value (HPET p.6, p.20).
- The AVR RTC fires "at the first count after the counter equals" the compare or period value
  (TB3213 p.7).

### 1.1 The HPET register file

Offsets from the block base (HPET p.10, Table 2; `edk2-hpet-register-h` repeats them):

| Offset | Register | Type | Notes |
|---|---|---|---|
| 000h | General Capabilities and ID (GCAP_ID) | read-only | writes "should not be attempted" (p.11) |
| 010h | General Configuration (GEN_CONF) | read-write | p.12–13 |
| 020h | General Interrupt Status (GINTR_STA) | read/write-clear | p.14 |
| 0F0h | Main Counter Value (MAIN_CNT) | read/write | p.15 |
| 100h + 20h·n | Timer n Configuration and Capability (Tn_CONF) | read/write | p.16–18 |
| 108h + 20h·n | Timer n Comparator Value (Tn_CMP) | read/write | p.19–22 |
| 110h + 20h·n | Timer n FSB Interrupt Route (Tn_FSB) | read/write | p.23 |
| 008h, 018h, 028h–0EFh, 0F8h, 118h + 20h·n | reserved | — | p.10 |

**The register space.**
- Every register is 64 bits wide (p.10–23).
- A block's register space is 1 KiB (p.8, p.27, p.30). On IA-64 it may be up to 64 KiB, with page
  protection (p.8; the ACPI table's page-protection field, p.31).
- The BIOS assigns the base, and the OS is "not expected" to move it (p.7).
- §7 E22 records that 32 timers at this stride do not fit in 1 KiB.

**GCAP_ID** (p.11–12). 32-bit reads are made at 00h or 04h; a 64-bit read at 00h.

| Bits | Field | Meaning |
|---|---|---|
| 63:32 | COUNTER_CLK_PERIOD | the tick period in femtoseconds, nonzero and at most 05F5E100h (10^8 fs = 100 ns) |
| 31:16 | VENDOR_ID | as a PCI vendor ID would be |
| 15 | LEG_RT_CAP | LegacyReplacement routing supported |
| 14 | reserved | |
| 13 | COUNT_SIZE_CAP | 0: the counter is 32 bits; 1: 64 bits, which "does not preclude" 32-bit operation |
| 12:8 | NUM_TIM_CAP | the **index of the last timer** ("three timers ... 02h") |
| 7:0 | REV_ID | never 00h |

**GEN_CONF** (p.12–13):

| Bits | Field | Meaning |
|---|---|---|
| 0 | ENABLE_CNF | "must be set to enable any of the timers to generate interrupts. If this bit is 0, then the main counter will halt (will not increment) and no interrupts will be caused by any of these timers" |
| 1 | LEG_RT_CNF | with ENABLE_CNF also set: timer 0 goes to IRQ0 (non-APIC) or IRQ2 (I/O APIC), timer 1 goes to IRQ8 in both, and timers 2–n follow their own routing (p.13) |
| 7:2, 63:16 | reserved | preserved by read-modify-write |
| 15:8 | reserved for the manufacturer | "OS-based drivers must not modify" them (p.12) |

**GINTR_STA** (p.14):
- Bit n is Tn_INT_STS for timers 0–31. Bits 63:32 are reserved and written as 0.
- In level mode, the hardware sets the bit while the timer's interrupt is active. Writing 1
  clears it, and writing 0 has no effect.
- In edge mode: "This bit should be ignored by software. Software should always write 0 to this
  bit."

**MAIN_CNT** (p.15): bits 63:0.
- A read returns the current value.
- Writes "should only be done while the counter is halted".
- A 32-bit counter returns 0 in the upper 32 bits.

**Tn_CONF** (p.16–18). The document prints bits 63:32 as "64:32" (§7 E9).

| Bits | Field | Access | Meaning |
|---|---|---|---|
| 63:32 | Tn_INT_ROUTE_CAP | RO | bit k is set when the timer can route to I/O (x)APIC input k |
| 31:16 | reserved | write 0 | |
| 15 | Tn_FSB_INT_DEL_CAP | RO | direct FSB (message) delivery is supported |
| 14 | Tn_FSB_EN_CNF | RW | deliver as FSB messages; Tn_INT_ROUTE_CNF is then ignored |
| 13:9 | Tn_INT_ROUTE_CNF | RW | the I/O APIC input, default 00h; an unsupported value reads back differently; "The software must only write valid values" |
| 8 | Tn_32MODE_CNF | RW | forces a 64-bit timer to behave as a 32-bit one; on a 32-bit timer it reads 0 and writes are ignored |
| 7 | reserved | write 0 | |
| 6 | Tn_VAL_SET_CNF | RW, self-clearing | periodic timers only: the next comparator write sets the accumulator |
| 5 | Tn_SIZE_CAP | RO | 1 = 64-bit timer, 0 = 32-bit |
| 4 | Tn_PER_INT_CAP | RO | periodic mode is supported |
| 3 | Tn_TYPE_CNF | RW if periodic-capable | 1 = periodic, 0 = non-periodic; otherwise it reads 0 and ignores writes |
| 2 | Tn_INT_ENB_CNF | RW | interrupt enable; at 0, "the timer will still operate and generate appropriate status bits, but will not cause an interrupt" |
| 1 | Tn_INT_TYPE_CNF | RW | 0 = edge, 1 = level (held until cleared in GINTR_STA) |
| 0 | reserved | write 0 | |

**Tn_FSB** (p.23):
- Bits 63:32 are Tn_FSB_INT_ADDR, the address the message is written to. The document prints the
  range as "64:32".
- Bits 31:0 are Tn_FSB_INT_VAL, the value written.

**Access shape** (p.10 §2.3.2 and each register's own rules):
- No access crosses a register boundary.
- A 32-bit access is made at +0 or +4 of a register; a 64-bit access at +0 only.
- Software should not write read-only registers.
- Software should not "read or write" unimplemented timers (p.17, p.23).
- There is no exclusive access: "if software attempts exclusive-access mechanisms ... the results
  are undefined" (p.10).
- §7 E23 records three different policies for reserved bits.

### 1.2 The HPET main counter
- **Width.** 32 or 64 bits, per COUNT_SIZE_CAP (p.11–12); 64 is recommended (p.9). A 64-bit
  comparator needs a 64-bit counter. A 32-bit comparator works with either (p.6).
- **Period.** COUNTER_CLK_PERIOD, in femtoseconds (p.11).
  - The minimum clock is 10 MHz (p.9), which is the 100 ns bound.
  - Accuracy: within 0.05% (500 ppm) over any 1 ms, and within any 100 µs at most 2 ticks early
    or late, which is under 0.2% or 2000 ppm (p.23; p.9 Table 1).
  - The period is given in femtoseconds "in order to achieve a resolution of 50 ppm" (p.11).
- **Monotonicity.** The counter counts up.
  - A second read is never less than the first unless the counter rolled over (p.7, p.23).
  - Two reads may be equal when they come faster than the clock (p.23).
  - The wordings differ (§7 E8).
- **Wrap.** The counter rolls over (p.7), and comparator arithmetic wraps "through 0" (p.20).
- **Run and halt.** ENABLE_CNF = 0 halts the counter and suppresses all timer interrupts (p.13).
  In the initial state, "Main Counter is Halted and Zeroed" (p.27).
- **Power.**
  - The registers are not preserved through S3, S4 or S5.
  - They are not used in S1, and their interrupts must be disabled before entering it.
  - The counter is permitted, but not required, to run in S1 and S2 (p.10–11).
  - The OS saves and restores the context (p.10).
  - Linux disables the HPET on PC10-capable Intel systems because it "stops counting in PC10"
    (`linux-x86-hpet-c` `hpet_is_pc10_damaged`; §7 D12).

### 1.3 HPET comparators

**Non-periodic (one-shot) mode.** Every timer supports it (p.21, p.25).
- A write loads the match value. When the counter equals that value, the interrupt "can be
  generated (if so enabled)". A match does not change the value (p.20).
- In 32-bit mode, "it will generate another interrupt when the main counter wraps around" (p.21,
  p.24). This sentence has two readings (§7 E26).
- **The warning** (p.21, p.25): "If the value written to the register is not sufficiently set far
  enough ahead of the current register value, then the counter may pass the value before it
  reaches the register and the interrupt will be missed."

**Periodic mode.** Support is optional per timer (Tn_PER_INT_CAP). At least 1 timer of 3 is
recommended (p.9). The comparator register is overloaded (p.22) and holds two hidden quantities:
the **accumulator** ACC (the next match) and the **period** PER (the last value written).
- A read returns ACC: "the current contents of the match register at which the next interrupt will
  occur" (p.22).
- A write sets PER: "writes to this register will program the periodic interval value to be added to
  the contents of the match register at the next interrupt" (p.22). Also: "A write to the register
  will not immediately be loaded into the comparator. It will only be added at the time the
  comparator triggers. There is no mechanism to immediately change the periodic rate" (p.25).
- With Tn_VAL_SET_CNF = 1, a write also sets ACC directly. The bit clears itself (p.17), and it
  must not be set on a non-periodic timer (p.18).
- On a match, ACC ← ACC + PER, wrapping in the timer's width (p.20). The worked example runs
  123h → 246h → 369h. The wrap example, FFFF0000h + 20000h → 00010000h, works only if 20000 is read
  as hexadecimal (§7 E11).
- Software keeps PER itself. A read tells "when the next periodic interrupt will be generated (not
  the rate at which it generates interrupts)" (p.21).
- *Reading:* a single write with VAL_SET sets both ACC and PER, because in the p.20 example the
  period equals the one value written.
  - QEMU implements exactly this. Every periodic write goes into `period`. A write with SETVAL,
    or any one-shot write, also goes into `cmp`. SETVAL clears after every comparator write that
    QEMU accepts, including a 32-bit half of a comparator in 64-bit mode. In 32-bit mode, QEMU
    drops a high-half write before reaching that point, so SETVAL stays set (`qemu-hpet-c`
    `hpet_ram_write`, the `HPET_TN_CMP` case).
  - Linux writes the accumulator with SETVAL and then the period in a second write, citing an
    AMD 81xx part that needs it (`linux-x86-hpet-c` `hpet_clkevt_set_state_periodic`).
  - Under this reading, both sequences end in the same state.
- At reset, every comparator holds all ones in its implemented bits (p.20, p.27; `qemu-hpet-c`
  `hpet_reset`).

**The write-ordering rules** that the documents state (the laws in §3 cite these numbers):

| # | Rule | Source |
|---|---|---|
| W1 | The main counter is written only while halted | HPET p.15 rule 2 |
| W2 | Periodic restart: clear ENABLE_CNF; write 0 to the main counter; set VAL_SET; write the comparator; set ENABLE_CNF | HPET p.21 |
| W3 | For periodic mode "the main counter is typically written with a value of 0 ... If the main counter is not set to 0, interrupts may not occur as expected" | HPET p.21, p.25 |
| W4 | A new periodic rate is written with "either the main counter ... halted or the comparator disabled" | HPET p.21 |
| W5 | The accumulator is set (VAL_SET) "with the main counter halted" | HPET p.21 |
| W6 | Driver enabling order: timer type; interrupt enable; comparator; overall enable last ("This starts the main counter") | HPET p.25 §2.4.4 |
| W7 | The BIOS or OS PnP code sets the routes (LEG_RT_CNF, each timer's route, edge or level). The OS establishes routing "prior to utilizing any given comparator" | HPET p.25, p.27 |
| W8 | A comparator target must be far enough ahead of the counter (the warning above) | HPET p.21, p.25 |
| W9 | Disable the timers' interrupts before entering S1 | HPET p.11 |

### 1.4 HPET interrupts
- **Enable gating.** An interrupt needs both ENABLE_CNF (p.13) and Tn_INT_ENB_CNF (p.18).
- **Status.**
  - In level mode, Tn_INT_STS latches, and writing 1 clears it (p.14).
  - The status latches whether or not Tn_INT_ENB_CNF is set (p.18; §7 E28).
  - "If another interrupt occurs before the interrupt is cleared, the interrupt will remain
    active" (p.18). Two events become one.
- **Handling** (p.26).
  - Edge mode with a unique interrupt needs no steps and no read.
  - A level-mode interrupt must be cleared "by reading the interrupt status register and writing
    a 1 back to the bit position for the interrupt to be cleared."
- **Routing.**
  1. LegacyReplacement (LEG_RT_CNF = 1): timer 0 → IRQ0 (8259) or IRQ2 (APIC), and timer 1 →
     IRQ8 in both. The 8254 and the RTC then cause no interrupts (the p.24 table; p.13).
  2. Standard: each timer routes by its Tn_INT_ROUTE_CNF, validated against Tn_INT_ROUTE_CAP
     (p.17, p.24).
  3. FSB: a direct message to Tn_FSB_INT_ADDR with the value Tn_FSB_INT_VAL.
     - It requires edge mode.
     - It has higher priority than the standard path and excludes it.
     - p.24 says it is usable "even when the LegacyReplacement mapping is used", which p.13
       contradicts (§7 E29).
  - EDK2 calls FSB delivery MSI (`edk2-hpet-register-h`).
- **Polarity and sharing.**
  - "The interrupts are all active high" (p.25).
  - Level-mode interrupts on the I/O APIC may be shared with PCI interrupts (p.25).
  - Edge-mode timers "should not be shared with any PCI interrupts" (p.24).

### 1.5 32-bit and 64-bit access
- **The torn read.** A 32-bit CPU that reads a 64-bit counter in two halves races the rollover:
  "An accuracy problem may be arise if just after reading one half, the other half rolls over"
  (p.26).
- **The remedies** (p.26):
  - Halt the counter, read both halves, and restart it. This has "the obvious drawback of shifting
    the counter timebase".
  - Set the 32-bit mode bit. §7 E15 notes that this bit belongs to a comparator, not to the
    counter.
  - Read high, then low, then high, and repeat until the two high reads agree.
- **64-bit platforms.** "On a 64-bit platform ... some platforms may break the 64 bit read into
  two 32 bit reads" (p.26).
- **32-bit software** should operate the timer in 32-bit mode (p.15 rule 5). A 64-bit comparator
  must offer a 32-bit mode on IA-32 platforms (p.9).
- **Writes.** Tn_32MODE_CNF is "typically needed if the software is not willing to halt the main
  counter to read or write a particular timer" (p.17). The specification states no rule for
  writing the periodic accumulator in 32-bit halves; T7(c) takes its rule from practice (§1.3).
- **Practice.**
  - Zephyr reads hi-lo-hi on 32-bit builds and uses one 64-bit read on 64-bit builds
    (`zephyr-hpet-c` `hpet_counter_get`). On 32-bit builds it writes a 64-bit comparator as two
    32-bit halves while the counter runs (`hpet_timer_comparator_set`).
  - Linux reads only the low 32 bits and registers a 32-bit clocksource (`linux-x86-hpet-c`
    `HPET_MASK`, `read_hpet`).
  - EDK2 reads all 64 bits at once, in its handler after halting the counter
    (`edk2-hpet-timer-dxe-c` `HpetRead`, `TimerInterruptHandler`).
  - QEMU computes every counter access from its virtual clock, so a +4 read is the high half of a
    new sample. It accepts 4-byte and 8-byte accesses (`qemu-hpet-c` `hpet_ram_read`,
    `hpet_ram_ops`).

### 1.6 Enumeration and reset
- **Initial state** (p.27 §3.1):
  - the counter is halted and zeroed;
  - the comparators hold all ones;
  - interrupts are disabled;
  - GEN_CONF[1:0] = 00;
  - legacy routing is off (the 8254 is on IRQ0 and the RTC on IRQ8).
- **The ACPI HPET table** (p.30–31; `edk2-hpet-table-h`):

  | Field | Byte offset | Size |
  |---|---|---|
  | Event Timer Block ID (GCAP_ID[31:0]) | 36 | 4 |
  | BASE_ADDRESS (a Generic Address Structure) | 40 | 12 |
  | HPET Number | 52 | 1 |
  | "Main Counter Minimum Clock_tick in Periodic Mode" | 53 | 2 |
  | Page protection and OEM attribute | 55 | 1 |

  - *Derived:* the table is 56 bytes in all.
  - The namespace device has `_HID` PNP0103 and a `_UID` that matches the HPET Number
    (p.31–32).
  - The HPET replaces only the RTC's periodic interrupt. With IRQ8 taken, the RTC alarm signals
    through SCI (p.29).
  - BCIR records these fields as data. Parsing them is item 4 of the native track.
- **Discovery in practice.**
  - EDK2 treats a GCAP_ID of 0 or of all ones as "not present" (`TimerDriverInitialize`).
  - Linux polls GEN_CONF while it reads all ones. This is for AMD SB700 systems, whose HPET is
    emulated by SMM code that takes time to start (`hpet_cfg_working`).
  - Linux checks the period's range (`hpet_enable`) and refuses a counter that does not advance
    (`hpet_counting`).

### 1.7 The AVR RTC (TB3213)
- **Two functions** (p.7): the RTC, a counter compared with a period register and a compare
  register, and the PIT, which "can be enabled independently". Both run "off the same counter
  inside the prescaler" (p.7).
- **Clock** (p.10; RTC.CLKSEL[1:0], reset 0). The selected clock is CLK_RTC. TB3213 states no
  accuracy for any source.

  | CLKSEL | Name | Source |
  |---|---|---|
  | 0x0 | INT32K | 32.768 kHz from OSCULP32K |
  | 0x1 | INT1K | 1.024 kHz from OSCULP32K |
  | 0x2 | TOSC32K | 32.768 kHz from XOSC32K, or an external clock on TOSC1 |
  | 0x3 | EXTCLK | the EXTCLK pin |

- **Prescaler** (p.11; RTC.CTRLA.PRESCALER[3:0], bits 6:3): DIV1 to DIV32768, that is,
  CLK_RTC / 2^code. The prescaler counter is 15 bits wide (p.7).
- **Counter** (p.7). CNT counts prescaled CLK_RTC cycles. The text: "It will generate a compare
  interrupt and/or event at the first count after the counter equals the Compare register value,
  and an overflow interrupt and/or event at the first count after the counter value equals the
  Period register value. The overflow will also reset the counter value to zero."
  - INTCTRL.OVF fires "when the Counter value (CNT) matches the Period value (PER) and wraps
    around to zero" (p.11).
  - *Derived:* the overflow period is PER + 1 counts. The example's PER = 511 at 32.768 kHz / 32
    gives 0.5 s, which is the "500 ms" of p.8 (p.18).
  - Not stated: CNT's width, CMP's width and CMP's reset value.
- **PER** (p.10). 16 bits, reset FFFFh, held as a PERL/PERH pair with the high byte at +0x01.
  TB3213 defers 16-bit access to a "CPU section" that it does not contain (§7 M2).
- **PIT** (p.7, p.13).
  - It interrupts every n CLK_RTC cycles, with n ∈ {4, 8, …, 32768} for interrupts and
    {64, …, 8192} for events.
  - RTC.PITCTRLA.PERIOD[3:0] (bits 6:3): 0x0 is off, 0x1 is 4 cycles, and so on up to 0xE at
    32768 cycles; 0xF is reserved. PERIOD "selects the bit from the 15-bit prescaler counter to
    be used as PIT period output". PITEN is bit 0.
  - The table's text is garbled in the extraction (§7 M4). *Derived:* code c gives 2^(c+1)
    cycles.
- **Synchronization** (p.9–11).
  - RTC.STATUS (read-only, reset 0) holds CMPBUSY (bit 3), PERBUSY (2), CNTBUSY (1) and CTRLABUSY
    (0). Each says its register is "busy synchronizing ... in the RTC clock domain".
  - For PER and for CTRLA (the prescaler): "there is a latency of two RTC clock cycles from
    updating the register until this has an effect. The application software needs to check that
    the [PERBUSY / CTRLABUSY] flag in RTC.STATUS is cleared before writing to this register"
    (p.10, p.11).
  - The same rule is not stated for CMP or CNT, and no busy flag is mentioned for the PIT
    registers (§7 M7).
- **Interrupts** (p.11–14).
  - RTC.INTCTRL has CMP (bit 1) and OVF (bit 0).
  - The ISR clears INTFLAGS.OVF by writing 1 (p.12). The code uses `RTC_OVF_bm` for both
    registers.
  - RTC.PITINTCTRL.PI is bit 0, and the ISR clears PITINTFLAGS.PI by writing 1 (p.14).
  - Global interrupts need `sei()` (p.12).
  - The vectors are `RTC_CNT_vect` (overflow) and `RTC_PIT_vect` (p.12, p.14). CMP's vector and the
    position of INTFLAGS.CMP are not shown.
- **Control** (p.10–11).
  - CTRLA.RUNSTDBY (bit 7) runs the RTC in standby, and CTRLA.RTCEN is bit 0. A CORREN bit is
    shown but not described (§7 M3).
  - RTC.DBGCTRL.DBGRUN (bit 0) runs the RTC in break debug mode (p.10).
- **Order** (p.8, p.18). "The source clock for the RTC counter must be configured before enabling
  the RTC peripheral and the desired actions". Example 7-1 runs:
  1. Disable XOSC32K. This is a configuration-change-protected write, `ccp_write_io` (p.8).
  2. Wait for XOSC32KS = 0.
  3. Select the crystal.
  4. Enable the oscillator.
  5. Wait in `while (RTC.STATUS > 0)`.
  6. Write PER, CLKSEL and DBGCTRL.
  7. Write CTRLA = prescaler | RTCEN | RUNSTDBY.
  8. Set INTCTRL.OVF.
  9. Call `sei()`.

  The PIT examples write PITINTCTRL and then PITCTRLA (p.20–21).
- **Oscillator** (p.8–9). XOSC32KCTRLA's SEL and CSUT "cannot be changed as long as the ENABLE bit
  is set or the XOSC32K Stable (XOSC32KS) bit ... is high". XOSC32KS "will only be available if
  the source is requested as the main clock or by another module".
- **Sleep** (p.15, p.21). The PIT wakes the CPU from power-down (Example 7-3). The document does
  not say which sleep modes keep the counter running, or what XOSC32KCTRLA.RUNSTDBY must be (§7
  M6).

### 1.8 What the drivers and the emulator add (evidence of practice, not device facts)

| Practice | Where | Used by |
|---|---|---|
| Arm, read the counter back, compare as a signed 32-bit difference, and take the expired path when the margin is under 128 ticks; the arming distance is at most 0x7FFFFFFF | `linux-x86-hpet-c` `hpet_clkevt_set_next_event`, `HPET_MIN_CYCLES`, `clockevents_config_and_register(..., 0x7FFFFFFF)` | T5 |
| Comparator writes are delayed by up to two HPET clocks (ATI, ICH9/10), longer on newer AMD; the first comparator readback after a write can be stale (ICH9/10) | the same function's comment | T5, §4.4 |
| A one-shot catch-up loop that counts lost periods | `linux-x86-hpet-c` `hpet_rtc_timer_reinit`, `hpet_cnt_ahead` | T5, §5.4 |
| Hi-lo-hi reads on 32-bit builds | `zephyr-hpet-c` `hpet_counter_get` | T3 |
| A core verify loop for an exact-equality comparator ("a target written after the counter has passed it is lost for a whole counter period") | `zephyr-hpet-c`, the comment above `TIMER_CORE_BACKEND_COMPARE_EXACT` | T5 |
| Halt, update the comparator and resume, in every interrupt | `edk2-hpet-timer-dxe-c` `TimerInterruptHandler` | T2, T4 |
| Acknowledge with exactly the timer's own bit before re-arming | `edk2-hpet-timer-dxe-c` `TimerInterruptHandler` (l.307–310) | T6 |
| Acknowledge only in level mode | `zephyr-hpet-c` `hpet_isr` | T6 |
| The timer's own enable bit used as the interrupt mask | `linux-x86-hpet-c` `hpet_msi_mask` / `hpet_msi_unmask` | §5.1 |
| The period read from GCAP_ID and range-checked | `linux-x86-hpet-c` `hpet_enable` | T4 |
| A counter that does not count is refused | `linux-x86-hpet-c` `hpet_counting` | T4, T8 |

---

## 2. The variant matrix

### 2.1 The two placements

| Axis | HPET per the specification | AVR RTC (TB3213) |
|---|---|---|
| Counter width | 32 or 64 (COUNT_SIZE_CAP), 64 recommended (p.9, p.11–12) | not stated (PER is 16 bits, p.10) |
| Counter top | 2^W − 1, rolls over (p.7, p.20) | PER: CNT resets to 0 at the first count after CNT = PER (p.7) |
| Tick period declared by | COUNTER_CLK_PERIOD, fs, 1..05F5E100h (p.11) | CLK_RTC by CLKSEL (p.10), over 2^PRESCALER (p.11); document-nominal |
| Slowest clock | 10 MHz (p.9) | the 1.024 kHz source with DIV32768 (p.10–11) |
| Accuracy | 500 ppm over ≥ 1 ms; ±2 ticks within 100 µs (p.23) | not stated |
| Equal consecutive reads | allowed within one tick (p.23) | not stated |
| Comparators | ≤ 32 per block, ≤ 8 blocks (p.7); 3 recommended (p.9) | CMP, PER and the PIT (p.7) |
| Comparator width | 32 or 64 per timer (Tn_SIZE_CAP, p.18) | not stated |
| Fires when | counter = match (p.6, p.20) | the first count after CNT = CMP or PER (p.7) |
| Periodic | optional per timer; ACC += PER (p.18, p.20) | overflow every PER + 1 counts; the PIT every 2^k cycles (p.7, p.13) |
| Periodic minimum | an ACPI table field that the platform writes (p.30–31) | not stated |
| Accumulator writes | VAL_SET, with the counter halted (p.17, p.21) | n/a |
| Missed-target warning | yes (p.21, p.25) | not stated |
| Write synchronization | none stated | BUSY flags; 2 RTC clocks for PER and CTRLA (p.9–11) |
| Status clear | level: write 1; edge: ignore and write 0 (p.14) | INTFLAGS.OVF and PITINTFLAGS.PI: write 1 (p.12, p.14) |
| Edge or level | per timer (p.18) | not stated |
| Delivery | legacy IRQ0/IRQ2 and IRQ8; the I/O APIC by capability; FSB messages (p.13, p.17, p.24) | the vectors RTC_CNT_vect and RTC_PIT_vect (p.12, p.14) |
| Polarity | all active high (p.25) | n/a |
| Access | 32-bit or 64-bit, register-aligned (p.10) | 8-bit registers; PER as a low/high pair (p.10) |
| Torn-read guidance | p.26 | deferred to a CPU section that TB3213 lacks (p.10) |
| Register space | 1 KiB (p.8, p.27, p.30); offsets on p.10 | offsets not stated |
| Reset | halted, zeroed, comparators all ones (p.27) | each figure's reset row (pp.8–15); PER FFFFh |
| Low power | not preserved in S3–S5; may stop in S1/S2 (p.10–11) | RUNSTDBY; the PIT wakes the CPU from power-down (p.11, p.15) |
| Debug | not stated | DBGRUN (p.10) |
| Discovery | ACPI HPET table, PNP0103 (p.30–32) | a fixed peripheral; TB3213 gives no addresses |
| Write protection | none; exclusive access undefined (p.10) | XOSC32KCTRLA.ENABLE behind CCP (p.8) |

### 2.2 The emulated HPET (what a KVM guest has when QEMU provides one)

| Axis | QEMU (`qemu-hpet-c`) | Against the specification |
|---|---|---|
| Counter | computed from QEMU's virtual clock at each read (`hpet_get_ticks`, `hpet_ram_read`) | a free-running counter; Zephyr reports that it reads backwards under QEMU SMP (§7 D4) |
| Period | GCAP_ID declares HPET_CLK_PERIOD × FS_PER_NS, a whole number of nanoseconds (`hpet_realize`; the constant's value is in a header that was not supplied) | any femtosecond value in range |
| Capabilities | every timer is periodic-capable and 64-bit-capable; FSB only with the `msi` property, which is off by default; vendor 8086h, revision 01h, legacy-capable, 64-bit counter (`hpet_reset`; `hpet_realize`, 0x8086a001) | per timer |
| Instances | at most 8 (`hpet_realize`) | 8 blocks (p.7) |
| One-shot 64-bit target already passed | armed at its past time, so not "missed" (`hpet_set_timer`, `hpet_calculate_cmp64`) | missed (p.21, p.25); §7 D6 |
| One-shot in 32-bit mode | an interrupt at the 32-bit wrap when the target lies beyond it (`wrap_flag`) | an ambiguous sentence (§7 E26) |
| Periodic | periods it missed are coalesced into one interrupt (the loop in `hpet_timer`); spacing clamped to at least 1 µs (`hpet_arm`) | one ACC step per match (p.20); §7 D8 |
| Counter write while running | traced and lost (`hpet_ram_write`, `HPET_COUNTER`) | "should only be done while ... halted"; the outcome is not stated (p.15) |
| Status | latched for level timers only; write-1-to-clear clears only set bits (`update_irq`, `HPET_STATUS`) | edge bits "should be ignored" (p.14) |
| Enable or unmask with status pending | the level interrupt is raised again (`hpet_ram_write`, the GEN_CONF and Tn_CONF cases) | not stated |
| Unimplemented timer | reads 0; writes ignored (`hpet_ram_read`, `hpet_ram_write`) | "should not be attempted" (p.17) |
| Access sizes | 4 and 8 bytes (`hpet_ram_ops`) | 32-bit or 64-bit (p.10) |

### 2.3 Hazards the laws exist for
- **TH-A (the rate is assumed).** A nominal crystal or a build-time constant is used instead of
  the counter's declaration. Zephyr does this when
  `CONFIG_TIMER_READS_ITS_FREQUENCY_AT_RUNTIME` is off (`sys_clock_driver_init`). → T4.
- **TH-B (exact equality).** A target the counter has already passed is missed for a whole
  counter period (p.21; Zephyr's COMPARE_EXACT comment; the comment in Linux's
  `hpet_clkevt_set_next_event`). → T5.
- **TH-C (torn reads).** Two 32-bit halves of a running 64-bit counter (p.26). → T3.
- **TH-D (clock domains).** AVR registers written while they are still synchronizing (TB3213
  p.10–11). → T1.
- **TH-E (writing back a write-1-to-clear value).** Writing back the status value just read clears
  the other timers' pending interrupts (p.14, p.26). → T6.
- **TH-F (names drift).** The same thing goes by several names:
  - FSB = MSI;
  - Tn_32MODE_CNF = TIMERn_32MODE_CNF = CounterSizeEnable;
  - NUM_TIM_CAP = "Number of Comparators" = NumberOfTimers, which is an index, not a count.

  → One registry generates every name (T0).
- **TH-G (the emulator is not the hardware).** QEMU fires a passed 64-bit target, coalesces
  periods and clamps their spacing (§2.2). A driver tested only on QEMU never meets TH-B. → The
  simulator's specification profile is normative, and its QEMU profile is a cross-check (§4.2).
- **TH-H (a halted counter is not a clock).** EDK2 halts the counter at every interrupt, Zephyr
  halts it in idle on a single CPU, and Linux halts it on resume (§1.8). → T4's continuity rule
  and T8.
- **TH-I (the window).** 32 timers do not fit in the 1 KiB the specification gives a block (§7
  E22). → T9.

---

## 3. The laws (T-laws)

**The common form.** Every law has the same parts:
- the rule;
- its sources;
- where it is checked;
- a **violation that a test drives**, and a **legal witness that the same test admits**
  (laws.md L22: every row admits a witness; L2: every check can fire);
- the whole message the test pins (L11).

**Where the checks run.**
- Checks on C units run on the claim graph, as advisory diagnostics prefixed `R3-timer:`. They
  refine R3's MMIO discipline. This follows the UART precedent (UART blueprint §4: no new global
  R-number).
- The checks are vacuous on any unit that does not embed the T0 header's `timer_regmap_contract`
  literal. That non-disturbance is measured over the whole corpus, not asserted.
- The registry and provider laws (T4, T8 and part of T9) return messages from the registry's own
  predicates.

**Module rules.**
- Each law module declares its scope in its docstring (L18).
- Every exit from a checker is a verdict (L1).
- Each law gets fault-table entries that inject its violation and must reach the code under test
  (L25).

### T1 — No write to a register that is still synchronizing (AVR RTC)
- **Rule.** A store to RTC.CTRLA or RTC.PER is legal only while that register's BUSY flag is
  clear. On the claim graph, every path to the store must pass a read of RTC.STATUS after the last
  store to the same register.
- **Extension (a design choice, §8 Q2).** The same rule applies to RTC.CMP and RTC.CNT. Their
  flags exist (p.9), but TB3213 does not state their write rule.
- **Sources.** TB3213 p.9 (the flags), p.10 (PER, and the latency of 2 RTC clocks), p.11 (CTRLA).
- **Checked by.**
  - T1, on the claim graph.
  - The simulator. A store while BUSY is a recorded violation, and the register becomes
    *undefined*: the simulator does not invent what the part does (§4.1).
- **Driving the violation.** `RTC.PER = 511;` with no read of RTC.STATUS before it. In the
  simulator: a second PER store one CLK_RTC cycle after the first.
- **Legal witness.** `RTC_init` in Example 7-1 (TB3213 p.18): one `while (RTC.STATUS > 0)` loop,
  then PER, CLKSEL, DBGCTRL and CTRLA. No register is stored twice after the read.
- **Message.** `R3-timer: RTC.<REG> store without a dominating RTC.STATUS check (<FLAG>)`.
- **Vacuity.**
  - HPET placements declare no synchronized registers. The law is vacuous there because the
    record says `sync_regs = ()`, not because the documents are silent.
  - TB3213 gives the PIT registers no flag, so they are not policed (§8 Q1).

### T2 — Halt, update, resume
- **Rule.**
  - (a) A MAIN_CNT store is legal only where a GEN_CONF store with ENABLE_CNF = 0 dominates it and
    no store with ENABLE_CNF = 1 comes in between (W1).
  - (b) An accumulator write (VAL_SET) needs the counter halted (W5).
  - (c) A periodic rate change needs the counter halted or the timer's interrupt disabled (W4).
    *Reading:* "the comparator disabled" means Tn_INT_ENB_CNF = 0, because no field disables a
    comparator and a disabled timer "will still operate" (p.18; §7 E27).
  - (d) The specification's enabling order (W6) is admitted as written. An order that stores the
    comparator after the overall enable is admitted only when that store passes T5.
  - (e) AVR: RTC.CLKSEL is stored only while RTCEN = 0 and PITEN = 0. This is a *reading* of
    TB3213 p.8's "configured before enabling" (§8 Q12).
- **Sources.** HPET p.13, p.15, p.18, p.21, p.25; TB3213 p.8.
- **Practice.**
  - EDK2 halts the counter around every comparator update (`TimerInterruptHandler` l.317–361).
  - Linux stops the counter before programming periodic mode and before zeroing it
    (`hpet_clkevt_set_state_periodic`, `hpet_restart_counter`).
- **Driving the violation.** A MAIN_CNT store after `GEN_CONF |= ENABLE_CNF`.
  - In the simulator, that store is a recorded violation.
  - In the specification profile, the counter then becomes undefined.
  - QEMU drops such a write (`hpet_ram_write`).
- **Legal witness.** Linux's `hpet_restart_counter`: stop the counter, write 0 to both halves,
  start it.
- **Messages.**
  - `R3-timer: main counter store while the counter may be running`
  - `R3-timer: accumulator store on timer <n> while the counter may be running`
  - `R3-timer: periodic rate change on timer <n> while the counter runs and its interrupt is enabled`
  - `R3-timer: RTC.CLKSEL store while the RTC or the PIT may be enabled`

### T3 — No torn 64-bit read, and no torn 64-bit comparator
- **Rule.** This applies to a placement whose main counter is 64 bits.
  - A value composed from a 32-bit read at 0F0h and a 32-bit read at 0F4h, of a counter that may
    be running, must come either from the hi-lo-hi retry or from a halted window (p.26). The
    hi-lo-hi retry reads high, low and high, and repeats until the two highs agree.
  - A single 64-bit read is admitted only on a placement that declares `atomic_64bit_read = True`.
    That declaration is evidence about the platform, and its default is `None`. p.26 warns:
    "some platforms may break the 64 bit read into two 32 bit reads".
  - Using the low 32 bits alone, as a 32-bit counter, is legal (p.15 rule 5; Linux's 32-bit
    clocksource).
  - A 64-bit comparator written as two 32-bit halves while the counter may run is admitted only
    in 32-bit mode, with the counter halted, or when T5's check follows the second half.
- **Why the comparator half is a design choice.** The documents cover torn reads (p.26). They do
  not cover a one-shot comparator written in halves, which Zephyr does on 32-bit builds (§7 D19).
- **Not a remedy.** Tn_32MODE_CNF is a comparator's mode (p.17). Whatever p.26 implies, it does
  not make the main counter 32 bits wide (§7 E15).
- **Driving the violation.** `lo = MAIN_LO; hi = MAIN_HI; t = (hi << 32) | lo;` on a running
  counter.
  - In the simulator, with `bus_split_ticks = 1` and the counter at 0000_0000_FFFF_FFFFh, reading
    low then high composes 0000_0001_FFFF_FFFFh.
  - Reading high then low composes 0 (*derived*, Appendix A).
- **Legal witness.** `zephyr-hpet-c` `hpet_counter_get` on a 32-bit build.
- **Messages.**
  - `R3-timer: torn read of the running 64-bit main counter (no hi-lo-hi check)`
  - `R3-timer: 64-bit main-counter read on a placement that does not declare atomic 64-bit reads`
  - `R3-timer: 64-bit comparator of timer <n> written in halves while the counter may run`

### T4 — The rate comes from the counter's own declaration
- **Rule.** Every conversion between ticks and time goes through one total predicate:
  `tick_period(placement, observed) -> Rate | None`. Following L14 (one predicate), the registry,
  the laws, the simulator and the providers all call it, and none re-derives it.
- **HPET.**
  - The period is COUNTER_CLK_PERIOD, read from GCAP_ID. It is admitted only in the range
    1..05F5E100h fs (p.11), with provenance `device-register`.
  - A device that declares anything else is **vetoed**. This follows the driver roadmap's rule
    that discovery may veto, never steer.
- **AVR.**
  - CLK_RTC is chosen by CLKSEL, with the nominal 32.768 kHz or 1.024 kHz that the document names.
    Its provenance is `document-nominal`, and its accuracy is not stated.
  - EXTCLK has provenance `external`: it is supplied, or it is `None`.
  - For CNT, the rate is divided by 2^PRESCALER. The PIT counts CLK_RTC cycles (p.7, p.10–11,
    p.13).
- **Arithmetic and continuity.**
  - The arithmetic is exact (`Fraction`).
  - A deadline is the ceiling of duration / period.
  - An interval is the wrap-aware difference in the counter's width.
  - A counter that does not advance while enabled is vetoed (Linux `hpet_counting`).
  - An interval that spans a halt or a reset of the counter is `None` (TH-H).
- **Sources.** HPET p.9, p.11, p.23, p.33; TB3213 p.7, p.10–11, p.13.
- **Practice.**
  - Linux reads the period, range-checks it and converts with FSEC_PER_SEC (`hpet_enable`).
  - EDK2 converts through CounterClockPeriod (`TimerDriverSetTimerPeriod`). It checks that the
    device is present but not the period's range (`TimerDriverInitialize`), so a zero period would
    reach a division.
  - Zephyr reads the period at run time only under `CONFIG_TIMER_READS_ITS_FREQUENCY_AT_RUNTIME`.
    Otherwise it uses a configured frequency (`sys_clock_driver_init`).
  - EDK2 computes a wrapped interval as `(mask − prev) + cur`, which is one tick short of
    `(cur − prev) mod 2^w` (`TimerInterruptHandler` l.379, and l.523, l.580, l.724; §7 D15).
- **Driving the violation.**
  1. A placement whose conversion reads `nominal_hz`.
  2. GCAP_ID periods of 0 and 05F5E101h.
  3. A float in a conversion.
  4. prev = 2^32 − 3 and cur = 1 on a 32-bit counter: the predicate gives 4, and EDK2's formula
     gives 3 (*derived*).
  5. An interval that spans a halt.
- **Legal witness.** A period of 05F5E100h (10 MHz, p.9 and p.11) is admitted. So is a QEMU-shaped
  period, a whole number of nanoseconds expressed in femtoseconds.
- **Messages.**
  - `T4: <placement> converts ticks with a rate the counter does not declare (<source>)`
  - `T4: COUNTER_CLK_PERIOD <v> fs is outside 1..0x05F5E100 (HPET p.11)`
  - `T4: the main counter did not advance while enabled`
  - `T4: interval spans a halt or reset of the counter`

### T5 — The comparator race
- **Rule.**
  - A one-shot comparator store made while the counter may run must be followed, in straight
    line, by a read of the **main counter**. Then comes a signed, wrap-aware comparison of the
    written target against that read, in the comparator's width. The margin must be at least the
    placement's `min_arm_ticks`.
  - A target inside the margin, or behind the counter, takes the expired path: the caller handles
    the event or re-arms.
  - The arming distance must be below 2^(w−1).
  - A store made with the counter halted is exempt. That is EDK2's form.
  - The check must not read the comparator back. Linux reports that on ICH9/10 "the first
    readout after write can return the old stale value".
  - Periodic mode: the period must exceed the placement's declared minimum. That minimum is the
    ACPI field (HPET p.30–31). A placement that declares none has no row (`None`), because a row
    nothing can satisfy is a defect (L22).
- **Sources.** HPET p.20–21, p.25, p.30–31.
- **Practice.**
  - Linux's `hpet_clkevt_set_next_event` does
    `cnt = counter + delta; write cnt; res = (s32)(cnt − counter); return res < HPET_MIN_CYCLES ? -ETIME : 0`.
    HPET_MIN_CYCLES is 128, the minimum programming delta is 1.5 × 128, and the maximum is
    0x7FFFFFFF.
  - Linux gives its reasons: comparator writes are delayed "up to two HPET clock cycles in certain
    chipsets (ATI, ICH9,10)", longer on newer AMD parts, and "a NMI or SMI hitting between the
    counter readout and the comparator write" can pass the target.
  - Linux's RTC emulation re-arms in a loop until it is ahead, counting the lost interrupts
    (`hpet_rtc_timer_reinit`).
  - Zephyr's timer core verifies its exact-equality comparator (the comment above
    `TIMER_CORE_BACKEND_COMPARE_EXACT`).
  - EDK2 halts the counter instead.
- **`min_arm_ticks`.** No document states a one-shot margin; Linux's 128 is a driver's constant.
  The placement field is `None` for the documents and a declared parameter for each simulator
  scenario (§8 Q3).
- **Driving the violation.**
  1. An arm with no read-back.
  2. A read-back compared as unsigned, which fails across the wrap.
  3. In the simulator's specification profile, with `arm_latency_ticks = 2`, an arm 1 tick ahead.
     No interrupt comes for 2^w ticks: never for a 64-bit timer, and for a 32-bit-mode timer
     2^32 ticks later or as the chosen wrap reading says (§7 E26).
- **Legal witness.** On the same scenario, Linux's sequence takes `-ETIME` and re-arms.
- **Messages.**
  - `R3-timer: one-shot comparator <n> stored while the counter may run, without a counter read-back check`
  - `R3-timer: arm of comparator <n> verified by reading the comparator back`
  - `R3-timer: periodic period <k> ticks does not exceed the declared minimum <P>`
  - `R3-timer: arming distance <d> ticks is not below 2^(w-1)`

### T6 — Acknowledge before re-arming, and write exactly the bits acknowledged
- **Rule.**
  - (a) In a level-mode timer's handler, its GINTR_STA write comes before any store to its
    comparator. Otherwise a second event before the clear merges with the first (HPET p.18).
  - (b) A GINTR_STA store writes 1 only at the level-mode timers being acknowledged, and 0
    everywhere else. That includes edge-mode timers ("Software should always write 0", p.14) and
    the reserved bits 63:32 (p.14). Storing a value read from GINTR_STA is refused, because it
    clears the other timers' pending interrupts.
  - (c) A branch on an edge-mode timer's status bit is refused (p.14: "should be ignored").
  - (d) AVR: an RTC handler clears its own flag (INTFLAGS.OVF, PITINTFLAGS.PI) by writing 1 before
    it returns (TB3213 p.12, p.14).
- **Practice.**
  - EDK2 writes `1 << mTimerIndex` and then re-arms (`TimerInterruptHandler` l.307–356). It does
    the same in edge (MSI) mode too, against p.14 (§7 D1).
  - Zephyr acknowledges only in level mode: "When edge trigger is selected, spec says only 0 can
    be written" (`hpet_isr`).
  - QEMU clears only the bits that are set (`hpet_ram_write`, `HPET_STATUS`).
- **Driving the violation.**
  - A handler that re-arms first. In the simulator, a second match inside the window yields one
    interrupt for two expiries.
  - `GINTR_STA = GINTR_STA;`
  - A store that sets an edge-mode timer's bit.
- **Legal witness.** EDK2's handler in its I/O APIC (level) mode; Zephyr's `hpet_isr`; TB3213's
  ISRs (p.19–21).
- **Messages.**
  - `R3-timer: comparator <n> re-armed before its level status is acknowledged`
  - `R3-timer: GINTR_STA store of a value read from GINTR_STA`
  - `R3-timer: GINTR_STA store sets edge-mode timer <n>'s bit (HPET p.14: write 0)`
  - `R3-timer: branch on edge-mode timer <n>'s status bit`
  - `R3-timer: RTC <source> handler returns without clearing its flag`
- **The EV3 bound.** GINTR_STA is one register shared by every timer in a block. Two timers'
  handlers writing their own bits are two handlers sharing one resource, which is the case
  `events.py` defers ("Handler-vs-handler sharing ... is likewise deferred"). The first slices
  therefore use one HPET timer per module, or edge mode, which writes no status (p.26). An EV3 rule
  for bit-disjoint write-1-to-clear stores would be a later slice (§6, notes under T4).

### T7 — The periodic accumulator's write rules
- **Rule.**
  - (a) VAL_SET is set only on a periodic-mode timer (p.18) whose Tn_PER_INT_CAP is 1 (p.18).
  - (b) An accumulator write happens with the counter halted (W5).
  - (c) Where the comparator is written in 32-bit halves, VAL_SET comes before each half. The bit
    clears itself (p.17): Linux says "after the first write" (`hpet_clkevt_set_state_periodic`),
    and QEMU clears SETVAL after each accepted half of a comparator in 64-bit mode (§1.3).
  - (d) The period is the driver's own shadow copy. In periodic mode a comparator read returns the
    next match, never the rate (p.21–22), so a period derived from such a read is refused.
  - (e) The first match is placed against the counter's value. Either the counter is zeroed (W2,
    W3), or the accumulator is set relative to the halted counter, as Linux does. Both are
    admitted (§7 D11).
- **Practice.**
  - Linux stops the counter; sets `cmp = now + delta`; sets ENABLE | PERIODIC | SETVAL | 32BIT;
    writes cmp; writes the period; and starts the counter (`hpet_clkevt_set_state_periodic`).
  - Linux also forces periodic mode on legacy channel 0 whatever Tn_PER_INT_CAP says
    (`hpet_legacy_clockevent_register`, §7 D10). Under T7(a) that is a violation. A part that
    under-reports its capability belongs in the registry's errata (the `uart_model.ERRATA`
    pattern), not in a law exception (§8 Q13).
- **Driving the violation.**
  - VAL_SET on a one-shot timer.
  - VAL_SET while the counter runs.
  - A 64-bit accumulator written in two halves with only one VAL_SET. The simulator then leaves
    the accumulator's high half stale and the period's high half new.
  - A period computed from a comparator read.
- **Legal witness.** The specification's usage model (p.21) and Linux's sequence.
- **Messages.**
  - `R3-timer: Tn_VAL_SET_CNF set on non-periodic timer <n>`
  - `R3-timer: periodic mode on timer <n> without Tn_PER_INT_CAP`
  - `R3-timer: accumulator half of timer <n> written without its own Tn_VAL_SET_CNF`
  - `R3-timer: period of timer <n> derived from a comparator read`

### T8 — A time this host cannot read is None
- **Rule.** A provider read or a conversion returns `None`, never 0 and never a nearby number, in
  each of these cases:
  - the clock is absent on the host (this guest has no HPET, §5.3);
  - the period is not declared (T4);
  - the device is absent or not ready: a GCAP_ID of 0 or of all ones (EDK2
    `TimerDriverInitialize`), or GEN_CONF reading all ones while SMM emulation starts (Linux
    `hpet_cfg_working`);
  - the counter does not count (Linux `hpet_counting`);
  - an interval spans a halt or a reset;
  - a register of an unimplemented timer is read (p.17). QEMU returns 0 there (`hpet_ram_read`).

  Telemetry omits an unreadable value. It never sends it as 0.
- **Sources.**
  - The module docstring of `bcir/silicon.py`: "A value this host cannot read is None -- never 0
    and never a nearby number".
  - The systems-engineer skill, §7: "record an unreadable counter as null, never zero".
- **Contrast.** Zephyr's `hpet_counter_get` returns 0 before the block is mapped (the `CONFIG_MMU`
  branch).
- **Driving the violation.** A provider on this guest that reports HPET ticks as 0, or a
  conversion with an undeclared period that returns 0.
- **Legal witness.** The same provider returning `None`, and its summary listing the HPET as
  unavailable.
- **Message.** `T8: <clock> unreadable on this host reported as <value>`.

### T9 — Access shape and the register map
- **Rule.**
  - Every HPET access is 32 or 64 bits and stays inside one register. A 32-bit access is made at
    +0 or +4, and a 64-bit access at +0 (p.10 and each register's rules).
  - There is no store to GCAP_ID (p.11).
  - There is no access to an unimplemented timer, meaning one whose index exceeds NUM_TIM_CAP
    (p.17, p.23).
  - There is no exclusive access: an atomic (`Lane.A`) claim on a timer register is refused (the
    p.10 note).
  - Every timer's registers lie inside the placement's declared window. A 32-timer record in a
    1 KiB window is refused (§7 E22).
  - Reserved bits follow each register's own text (§7 E23).
  - The AVR placement carries no offsets, because TB3213 states none. It therefore has no RegMap
    contract until a data sheet supplies them (§8 Q1).
- **Practice.** QEMU accepts only 4-byte and 8-byte accesses and ignores GCAP_ID writes
  (`hpet_ram_ops`, `hpet_ram_write`).
- **Driving the violation.** Each of these is refused:
  - a 16-bit access;
  - a 32-bit access at 0F2h;
  - a 64-bit access at 0F4h;
  - a store to GCAP_ID;
  - timer 3 on a three-timer block;
  - an atomic claim on GEN_CONF;
  - a 32-timer placement in 1 KiB.
- **Legal witness.** The access forms of p.10, and the drivers' own accesses.
- **Messages.**
  - `R3-timer: <w>-bit access at <off> is not a register-aligned 32/64-bit access`
  - `R3-timer: store to the read-only GCAP_ID`
  - `R3-timer: access to unimplemented timer <n>`
  - `R3-timer: atomic access to the timer block (HPET p.10: undefined)`
  - `T9: placement <p> lays timer <n> outside its <size>-byte window`

### T10 — Routes come from capabilities
- **Rule.**
  - Tn_INT_ROUTE_CNF takes only a value whose bit is set in Tn_INT_ROUTE_CAP (p.17).
  - FSB delivery needs Tn_FSB_INT_DEL_CAP (p.17) and edge mode (p.24).
  - Routes are data in the delivery record. Nothing here programs an interrupt controller.
- **Driving the violation.** Each of these is refused:
  - route 5 on a timer whose capability lacks bit 5;
  - FSB delivery on a timer without the capability;
  - FSB delivery in level mode.
- **Legal witness.** Route 16 on a timer whose Tn_INT_ROUTE_CAP sets bits 16, 18, 20, 22 and 24
  (the p.17 example); FSB delivery in edge mode on a capable timer. Both are admitted.
- **Messages.**
  - `R3-timer: route <r> is not in timer <n>'s Tn_INT_ROUTE_CAP`
  - `R3-timer: FSB delivery on timer <n> without Tn_FSB_INT_DEL_CAP`
  - `R3-timer: FSB delivery requires edge mode (timer <n> is level)`

---

## 4. A simulated timer's behavioral contract

### 4.1 Principles
1. **Time moves only by an explicit tick.** `sim_tick(sim, n)` advances n ticks of the
   placement's source clock. On the HPET these are main-counter ticks, whether or not the counter
   is enabled; on the AVR they are CLK_RTC cycles. There is no wall clock and no host timer. This
   is the UART simulator's discipline: "time modeled in *character ticks* advanced by an explicit
   `sim_tick(dev, n_char_times)` — no wall clock, fully reproducible" (UART blueprint U2 and §8).
2. **Integer state, exact time across devices.** Registers are integers of their own width.
   Seconds exist only through T4's predicate, as `Fraction`.
3. **A deterministic order.** Within one tick, the simulator:
   1. lands any pending writes whose time has come;
   2. advances the counter;
   3. evaluates comparators in index order;
   4. latches status;
   5. computes the interrupt lines;
   6. appends events to the transcript.
4. **Refuse to invent.** Some outcomes the documents leave open: a write while BUSY, a
   main-counter write while running, the 32-bit one-shot wrap, and the CMP and CNT
   synchronization latency. For these, the simulator either records a violation or requires the
   scenario to declare an assumption, which the transcript carries. It never picks a hardware
   behavior silently.
5. **Profiles are data.** `spec` (normative), `qemu` (the emulator's documented behavior, §2.2)
   and `tb3213` (AVR) are tables, not code forks. This is the rule U9 sets for the UART
   simulator's profiles.

### 4.2 The HPET placement
- **State.**
  - `now`; MAIN (W bits); ENABLE and LEG_RT; ISR (32 bits).
  - Per timer: the CONF fields; TARGET or ACC; PER; a pending comparator write (its value and the
    tick it lands at); the line state.
  - `undefined` marks.
- **Reset** (p.27):
  - the counter is 0 and halted;
  - the comparators hold all ones;
  - interrupts are disabled;
  - GEN_CONF[1:0] = 00;
  - the capabilities come from the placement record.
- **Access.**
  - `sim_read(off, width)` and `sim_write(off, width, value)` accept 32-bit and 64-bit accesses at
    register-aligned offsets; anything else is refused under T9.
  - A +4 access addresses bits 63:32, as QEMU's `hpet_ram_read` and `hpet_ram_write` do.
  - A 64-bit access is a single access unless the scenario sets `bus_split_ticks = k`. That makes
    it two 32-bit accesses k ticks apart (p.26's "some platforms").
- **Main counter.**
  - A read returns MAIN, which is the held value while the counter is halted.
  - A write while halted sets the addressed half.
  - A write while enabled is a T2 violation. The `spec` profile marks MAIN undefined; the `qemu`
    profile drops the write (`hpet_ram_write`).
- **Comparator write.**
  - One-shot: TARGET ← value.
  - Periodic: PER ← value; with VAL_SET, also ACC ← value. VAL_SET clears after the write (p.17;
    QEMU).
  - In 32-bit mode the high half is 0, and a write to the high half is ignored. p.26 says "The
    upper 32-bits will always be 0", and QEMU truncates the value and ignores the high-half
    write.
  - A write lands after `arm_latency_ticks`. The default is 0; Linux reports up to two HPET clocks
    on some chipsets. Until the write lands, a read returns the old value.
- **Tick.**
  - If ENABLE is set, MAIN ← MAIN + 1 mod 2^W.
  - Each timer is evaluated in index order. It matches when the counter's value, in the timer's
    width, equals TARGET (one-shot) or ACC (periodic).
  - A match happens on the tick that makes the values equal. *Reading:* a write that equals the
    current counter value does not fire until the counter comes back round to it. The documents do
    not say, and T5 makes drivers indifferent to the answer.
- **On a match.**
  - Periodic: ACC ← ACC + PER mod 2^w (p.20).
  - Level mode latches ISR[n] whether or not Tn_INT_ENB_CNF is set (p.18).
  - An interrupt is delivered if ENABLE and Tn_INT_ENB_CNF are both set:
    - as an FSB message (address and value) if Tn_FSB_EN_CNF is set;
    - otherwise, for timers 0 and 1 under LEG_RT, to IRQ0/IRQ2 or IRQ8;
    - otherwise, to the routed input.
  - Edge mode delivers one event per match. Level mode holds the line while ISR[n] is set.
- **The 32-bit one-shot wrap** (§7 E26). The scenario must declare a reading:
  - `oneshot32_wrap = "rematch"`: another match 2^32 ticks later;
  - `oneshot32_wrap = "wrap_interrupt"`: an interrupt when the low 32 bits roll over before the
    target is reached, which is QEMU's `wrap_flag`.

  A scenario that crosses a wrap without declaring a reading is refused.
- **Status writes.**
  - Writing 1 clears only bits that are set (QEMU).
  - A 1 at an edge-mode timer or at a reserved bit is a T6 violation.
  - An edge-mode status bit reads 0. This is QEMU's choice; the specification says the bit
    "should be ignored".
- **Enable or unmask with a status pending.** The documents do not say what happens.
  - `qemu` raises the line again (`hpet_ram_write`).
  - `spec` derives the line as ISR[n] ∧ Tn_INT_ENB_CNF ∧ ENABLE, and records the re-assertion as
    a reading (§8 Q7).
- **The `qemu` profile** adds four behaviors:
  - A 64-bit one-shot target that the counter has already passed fires at the next tick. QEMU arms
    it at its past time. This is a *reading* of QEMU's timer API, consistent with Zephyr's report
    of immediate firing (§7 D5, D6).
  - Periodic matches that the host missed are coalesced into one interrupt (`hpet_timer`).
  - Periodic spacing is at least 1 µs (`hpet_arm`). This needs the period in nanoseconds, which
    the profile states.
  - Reads of an unimplemented timer return 0.

### 4.3 The AVR RTC placement
- **State.**
  - `now`, counted in CLK_RTC cycles.
  - The 15-bit prescaler counter.
  - CNT, of width `cnt_bits`. TB3213 does not state the width, so a scenario declares it.
  - PER (16 bits, reset FFFFh).
  - CMP. Its width and reset are not stated, so it is undefined until written.
  - CTRLA, CLKSEL, INTCTRL, INTFLAGS, PITCTRLA, PITINTCTRL, PITINTFLAGS, STATUS and DBGCTRL, each
    with its figure's reset value (pp.8–15).
  - The pending synchronized writes.
- **Synchronized writes.**
  - A store to CTRLA or PER sets that register's BUSY flag and lands two CLK_RTC cycles later
    (p.10–11), which clears the flag.
  - CMP and CNT take their latency from the scenario; the placement holds `None` because the
    latency is not stated.
  - A store while BUSY is a T1 violation, and the register becomes undefined.
- **Tick.** One CLK_RTC cycle does the following, in order:
  1. Land the writes that are due.
  2. If RTCEN or PITEN is set, advance the prescaler counter. *Reading:* p.7 says the PIT runs off
     the prescaler and can be enabled alone.
  3. If RTCEN is set and the count for PRESCALER is due, make one count:
     - a compare event if CNT = CMP;
     - then, if CNT = PER, an overflow (CNT ← 0); otherwise CNT ← CNT + 1.

     This is the "first count after" rule (p.7).
  4. Latch the flags, and raise interrupts as INTCTRL enables them.
  5. PIT: if PITEN is set and PERIOD ≠ 0, latch PITINTFLAGS.PI each time the prescaler counter
     crosses a multiple of 2^(PERIOD+1).

  The length of the first PIT period is not stated. The simulator aligns it to the prescaler
  counter, and no test depends on it.
- **Not modeled.** Oscillator start-up (TB3213 gives no CSUT values, and XOSC32KS is a scenario
  input); sleep modes; CORREN; the event system.

### 4.4 The hazard knobs and their provenance

| Knob | Default | Grounded in |
|---|---|---|
| `bus_split_ticks` | `None` (64-bit accesses are single) | the HPET p.26 note |
| `arm_latency_ticks` | 0 | the `linux-x86-hpet-c` comment: up to two HPET clocks on ATI and ICH9/10, longer on newer AMD |
| `stale_readback` | off | the same comment (ICH9/10's first readout is stale) |
| `sync_latency` | CTRLA 2, PER 2, CMP `None`, CNT `None` | TB3213 p.10–11 |
| `oneshot32_wrap` | `None` | HPET p.21, p.24; `qemu-hpet-c` |
| `cnt_bits` (AVR) | `None` | not stated |

### 4.5 The coupled world and timeouts
- **One timeline.** `SimWorld` holds a set of devices, each with a tick period taken from T4's
  predicate.
  - `world_advance(world, dt)` adds the exact `Fraction` dt to each device's accumulator, ticks
    the device floor(acc / period) times, and keeps the exact remainder.
  - Events at the same instant are ordered by time, then device index, then the device's own
    order.
  - Nothing drifts and nothing is rounded.
- **UART coupling 1: the device's own timeout, on the shared clock.**
  - The 16550's character timeout (UART blueprint §1.2) stays the device's own: 4 character times
    with no arrival and no read.
  - The UART simulator's character tick becomes a device on the timeline, with period
    (bits per character × 16 × divisor) / ref_clk_hz (UART blueprint §1.2–1.3). A format with 1.5
    stop bits is a `Fraction`.
  - A test can then pin that the CTI event lands in the timer tick that contains the instant
    4 × that period after the last arrival.
- **UART coupling 2: driver deadlines.**
  - A bounded poll becomes "until the condition holds or the deadline passes", as in
    `uart_flush_until`.
  - An inter-character deadline covers the cases where the device has no timeout of its own:
    16450 mode, since the CTI exists in FIFO mode only (UART blueprint §1.2), and the TL16C750E
    with AFR.RCVEN cleared (§1.9).
- **DMA.** The completion phase of `dma_transfer_module` (`dma.done:<engine>`) races a timer phase
  (`timer.expire:<placement>.t<n>`). The world orders the scripted completion time against the
  simulated expiry.
- **Outcome rules.** These are normative for slices T4 and T5 in §6.
  - **O1: completion wins a tie.** When the waiter sees the deadline expire, it re-checks the
    completion once before it reports a timeout (UART: an LSR read; DMA: the DONE flag, read under
    both masks).
  - **O2: a timeout proves nothing about the device.** After a DMA timeout, the transfer's buffers
    stay device-owned until `dma.done` is observed. This is the ownership law that the UART
    blueprint's §12 proposes: "a cancellation is a transition request, never proof that the
    device stopped". That section is on the UART branch, not on main.
  - **O3: deadlines are ceilings in declared ticks**, counted from a counter read taken before the
    operation starts (T4), and armed under T5.
  - **O4: disarming is a program claim.** After completion, the waiter masks the timer source and
    parks or rewrites its comparator. A level status that latched in the meantime is acknowledged
    before the next arm (T6). Zephyr parks the comparator at UINT64_MAX, or at counter + 2^48 on
    QEMU because of the overflow it reports (`sys_clock_no_timeout`).
  - **O5: EV3's current bounds hold.** Each handler writes its own flag (DONE or EXPIRED). The
    consumer reads both inside a window masked for both sources. There is one HPET timer per
    module (T6's EV3 bound).

### 4.6 The transcript and the fidelity tests
A transcript is a deterministic list of `(tick, kind, detail)` records: accesses with their
values, matches, status latches, deliveries, violations and assumptions.

The fidelity tests run before any driver test, the order U2 uses:
- the reset state (p.27);
- p.20's sequence 123h → 246h → 369h;
- p.20's wrap with 20000h, with the decimal reading's result, FFFF4E20h, kept as the DocNote's
  witness;
- equality matching, and a target missed under `arm_latency_ticks`;
- truncation in 32-bit mode;
- both torn-read orders;
- write-1-to-clear and the edge-mode rule;
- VAL_SET before each half (T7(c));
- the legacy routes (p.13, p.24);
- an FSB message's address and value (p.23);
- the AVR overflow at PER + 1 counts, reproducing Example 7-1's 0.5 s;
- the PIT's 1 s at CYC32768 from 32.768 kHz (p.13);
- a write while BUSY;
- the `qemu` profile's four behaviors, each pinned to its function in `qemu-hpet-c`.

---

## 5. Where a timer meets BCIR's event phases and the measurement providers

### 5.1 Event phases
- **Source names.** `timer.expire:<placement>.t<n>` for HPET timer n. For the AVR:
  `timer.expire:<placement>.ovf`, `.cmp` and `.pit`. These follow the shape of
  `dma.done:<engine>`.
- **EV1.** A timer phase declares no phase dependencies.
- **EV2.** The source is unmasked by an explicit `irq.unmask:timer.expire:…` claim.
  - Its controller resource is the timer's **own** enable: Tn_CONF (Tn_INT_ENB_CNF) on the HPET,
    INTCTRL or PITINTCTRL on the AVR.
  - Linux masks an MSI-delivered HPET channel the same way, through ENABLE and FSB in Tn_CONF
    (`hpet_msi_mask` / `hpet_msi_unmask`).
  - No I/O APIC or local-APIC resource appears: those belong to the paused track.
  - `events.py`'s mask law (exactly one controller resource written, none read) holds if the driver
    keeps Tn_CONF in a RAM shadow, so that masking is a store rather than a read-modify-write. This
    is the same idea as the UART's FCR shadow (H-E).
- **EV3.** Program flow touches the resources that the handler writes (its flag, and the comparator
  it re-arms) only under mask or atomically. Masking at Tn_INT_ENB_CNF still lets the status latch
  (p.18).
- **Arming is not unmasking.** Writing a comparator is ordinary MMIO, under R3 and T5. Unmasking is
  EV2's claim. The laws keep the two apart.

### 5.2 The three clocks
The measurement discipline has three rules:
- Elapsed time, TSC ticks and core cycles are three different clocks.
- Level A evidence is timing-qualified; Level B evidence is microarchitecture-qualified.
- A frequency is unknown or externally supplied, never estimated from the TSC.

| Clock | What counts it | Rate provenance | What an event timer is to it | This guest |
|---|---|---|---|---|
| Elapsed time | a counter with a declared rate | HPET: the device's fs period (T4). AVR: the document's nominal CLK_RTC. The host's CLOCK_MONOTONIC: the OS, over its clocksource | an event timer **is** an elapsed-time clock, while it runs unhalted (TH-H) | CLOCK_MONOTONIC over `tsc`; no HPET |
| TSC ticks | the CPU's TSC | declared by the platform (here, the hypervisor), or `None` | none: a timer cannot serve TSC ticks. A TSC-to-timer ratio may be recorded as a measured ratio with its window, never as a frequency (§8 Q9) | `rdtsc` and `rdtscp`; perf `msr/tsc` |
| Core cycles | the PMU's cycles event | counted, never converted | none | `None`: there is no `cpu` event source |

Linux calibrates the TSC against the HPET when it can: "If HPET is functional it is useful for
calibrating TSC" (the comment above `hpet_is_pc10_damaged`). BCIR does not, because that would be
a frequency estimated from another clock.

**Levels.**
- A timer-backed measurement is at most Level A.
- Level B needs a PMU, and RAPL for energy (the metrics line in the UART blueprint's §12, on the
  UART branch).
- Calibration that freezes tables refuses a shared runner (`docs/BCIR_TARGET_ACCESS.md` §2).
- The native rig attests its tenancy from the host's own signals (`microbench.NativeEvidence`;
  laws.md L18, the S0-F instance).

### 5.3 What this guest exposes (read-only, 2026-10-27T18:57Z)

| Read | Value |
|---|---|
| `/sys/devices/system/clocksource/clocksource0/available_clocksource` | `tsc kvm-clock` |
| `.../current_clocksource` | `tsc` |
| `/dev/hpet` | absent |
| `/sys/firmware/acpi/tables/` | APIC, DSDT, FACP, MCFG; no HPET table |
| `/dev/rtc*`, `/sys/class/rtc` | absent |
| `/proc/interrupts` | no IRQ0 or IRQ8 row; `LOC` local timer interrupts on all 4 CPUs; `ttyS0` on IO-APIC pin 4, edge; virtio on PCI-MSIX |
| `/sys/devices/system/clockevents/clockevent{0..3}/current_device` | `lapic-deadline`; the broadcast device is empty |
| `/proc/cpuinfo` flags (cpu0, filtered) | `arat constant_tsc hypervisor nonstop_tsc rdtscp tsc tsc_adjust tsc_deadline_timer tsc_known_freq x2apic` |
| `/proc/cpuinfo` cpu MHz | 2800.198; on a KVM guest this is the TSC's rate, not a core frequency (the `silicon.py` docstring) |
| `/sys/bus/event_source/devices` | `breakpoint msr power software tracepoint uprobe`; no `cpu` |
| `/sys/bus/event_source/devices/msr/events` | `smi tsc` |
| `/sys/devices/system/cpu/cpu0/tsc_freq_khz` | absent |
| `uname -r`; `/proc/cmdline` | `6.18.44-fc-v165`; no `hpet=`, `clocksource=`, `tsc=`, `nohz` or `isolcpus` parameter; the command line names `--firecracker-init` |

**What this guest can offer:**
- Level A elapsed time from CLOCK_MONOTONIC, over the `tsc` clocksource.
- TSC ticks, through `rdtsc`/`rdtscp` and perf's `msr/tsc`.
- A TSC rate whose provenance is the hypervisor.
- The whole correctness program (registry, laws, simulator and parity), which needs no hardware.

**What it cannot offer:**
- An HPET of any kind: no clocksource, no device node and no ACPI table.
- The legacy PIT or RTC interrupts.
- Core cycles, which are `None`.
- Level B evidence.
- Frequency control or isolation (`docs/BCIR_TARGET_ACCESS.md` §2).
- Device binding: no VFIO, UIO or IOMMU.
- An AVR.

The kernel's event timer here is the local APIC in TSC-deadline mode, which is the paused x2APIC
track's device. Linux would not use HPET per-CPU clock events on this CPU anyway:
`hpet_select_clockevents` returns early when the CPU has ARAT (`linux-x86-hpet-c` l.704–706), and
`arat` is present. Had QEMU provided an HPET, it would be §2.2's model, not the specification's.

### 5.4 Telemetry
**The constraint.** In `TelemetryEnvelopeV0`, the `clock` codes are 0 none, 1 monotonic,
2 realtime, 3 boot and 4 cycles, with units of ns, µs or cycles (`TELEMETRY_ENVELOPE_ABI.md`). No
code names TSC ticks or a device timer's ticks. `cycles` must not carry either, because that would
merge two of the three clocks.

**The proposal.**
- Timer quantities travel as `sample` values under new BCIR-range signals, with unit `count` and
  provenance 0 (measured), 1 (modeled) or 2 (simulated).
- They are timestamped by the host's monotonic clock where one exists.
- A device-clock timestamp waits for an envelope version bump with append-only codes (§8 Q8).

**Candidate signals:**
- `timer.ticks`: cumulative; monotonic only for a counter that is never halted.
- `timer.expiries`.
- `timer.arm_late`: the number of times T5's expired path was taken.
- `timer.lost_periods`: Linux counts "Lost RTC interrupts" (`hpet_rtc_timer_reinit`); QEMU
  coalesces them silently.
- `timer.halts`.

HPET reads are slow MMIO: "Reading the HPET counter is a very slow operation" (the comment in
Linux's `read_hpet`). A sampling plan therefore prices them as uncached device reads, the way UART
U3 prices its register reads.

---

## 6. The build slices T0–T7 (a proposal, not milestones)

> Each slice below is a proposed, gateable, PR-sized step, in the repository's form: Goal, Files,
> Design, Tests, Gates, Commit. None is scheduled and none has a checklist entry. Each would be
> re-read against the code of its day before it is built. The gates named are the repository's
> (AGENTS.md; the systems-engineer skill §6). None was run for this document.

### T0 — The event-timer registry
- **Goal.** One normative model, two placements, generated views and a RegMap contract. This is
  the `uart_model` pattern (U0 on the UART branch).
- **Files.**
  - `bcir/frontends/devices/timer_model.py`
  - `bcir/frontends/devices/fixtures/hpet_<placement>.h`
  - `runtime/c/hpet_regs.h`: generated, for the default placement `hpet_spec`, with the
    generated-file banner.
  - `bcir/tests/test_timer_model.py`, registered in `run_all._MODULES`.
  - `tools/testing/faults/timer-model.json`
  - the row in the oracle inventory.

  The names avoid the paused track's `intel_platform` files.
- **Design.**

```python
@dataclass(frozen=True)
class Rate:                    # T4's declaration
    period_s: Fraction | None  # seconds per tick; None = not declared
    provenance: str            # "device-register" | "document-nominal" | "external" | "unknown"
    source: str                # the citation

@dataclass(frozen=True)
class TimerReg:
    name: str                  # "GCAP_ID", "GEN_CONF", "GINTR_STA", "MAIN_CNT", "TN_CONF", "TN_CMP",
                               # "TN_FSB"; AVR: "CTRLA", "STATUS", "INTCTRL", "INTFLAGS", "PER", "CMP",
                               # "CNT", "CLKSEL", "PITCTRLA", "PITINTCTRL", "PITINTFLAGS", "DBGCTRL"
    offset: int | None         # None: not stated (every AVR register)
    size: int                  # bytes
    per_timer: bool            # HPET: offset + 0x20 * n
    access: str                # "r" | "rw" | "w1c"
    fields: tuple              # ((hi, lo, name, access, reset, doc), ...)
    sync_flag: str = ""        # AVR: the RTC.STATUS flag that guards a store

@dataclass(frozen=True)
class TimerPlacement:
    name: str                  # "hpet_spec" | "avr_rtc_tb3213"
    family: str                # "hpet" | "avr_rtc"
    counter_bits: int | None
    timers: int                # comparators
    timer_bits: tuple          # per timer: 32 | 64 | None (read Tn_SIZE_CAP)
    periodic: tuple            # per timer: True | False | None (read Tn_PER_INT_CAP)
    fire_on: str               # "equal" | "count_after_equal"
    window_bytes: int | None   # 1024 for the HPET (p.8)
    routes: tuple              # per timer: allowed inputs, ("msg",), or None (read Tn_INT_ROUTE_CAP)
    rate: Rate
    nominal_hz: int | None     # recorded, never converted with
    min_periodic_ticks: int | None
    min_arm_ticks: int | None  # None: no document states one
    atomic_64bit_read: bool | None
    sync_regs: tuple           # (("CTRLA", 2), ("PER", 2), ("CMP", None), ("CNT", None)) or ()
    oneshot32_wrap: str | None # None: the documents are ambiguous
    errata: tuple = ()
```

  The functions:
  - `placement_problems(p)`: total, returning a list of findings, as in the `uart_model` contract.
  - T4's single predicate family: `tick_period(p, observed=None)`, `ticks_for(p, seconds)` (a
    ceiling), `seconds_of(p, ticks)` and `wrap_delta(bits, a, b)`.
  - `emit_header(p)` for HPET placements. An AVR placement is refused by name: "TB3213 states no
    register offsets".
  - `TimerRegMapContract` and `verify_timer_regmap_contract(contract, placement)`.
  - `DOC_NOTES` (§7, in the `DocNote` shape) and `PROVENANCE` (fact, document and page, reading).
- **Tests.**
  - The register table and fields against p.10–23 and TB3213 pp.8–15.
  - p.20's worked examples, pinned as hexadecimal, with the decimal reading kept as E11's witness.
  - T4's vetoes (0 and 05F5E101h) and its admission (05F5E100h).
  - `wrap_delta` against EDK2's formula.
  - The refusal of a 32-timer record in a 1 KiB window.
  - Tampered contracts, with one message per lie.
  - Header determinism and the drift guard.
  - Every fault, caught by its own test.
- **Gates.**
  - `python -m bcir.tests.run_all --tier quick -j 2`
  - `gen_status --check`
  - `check_links`
  - the oracle-inventory check
  - `git diff --check`
  - `tools/local/check_latest.sh` (AGENTS.md)
- **Commit.** `devices: T0 -- the event-timer registry (HPET and AVR RTC placements, the RegMap contract)`

### T1 — The T-laws on the oracle rail
- **Goal.** T1–T3, T5–T7, T9 and T10 on the claim graph, as advisory `R3-timer` diagnostics
  that are vacuous outside timer units.
- **Files.**
  - `bcir/frontends/devices/timer_laws.py`, with
    `verify_timer_protocol(lowered, placement, plans=None) -> list[str]` and
    `timer_unit_diagnostics(lowered, plans=None)`.
  - `pipeline.py`, gaining `CompileResult.timer_diagnostics`.
  - `bcir/tests/test_timer_laws.py`.
  - The fault entries.
- **Design.**
  - Registers are named from the claim graph by offset, width, direction and timer index.
  - T1 needs the facts machinery the UART laws built in `uart_laws._Facts`: offsets, folded
    constants, known bits and the path walk.
  - laws.md L14 argues for one shared module rather than a second copy. That would touch the UART
    branch, so it is a coordination point (§8 Q10), not a decision made here.
- **Tests.**
  - A violation and a legal witness for each law row, with each message pinned whole.
  - The whole corpus and the UART fixtures at zero timer diagnostics.
- **Commit.** `devices: T1 -- the event-timer protocol laws (sync, halt order, torn reads, the comparator race, acknowledgement, the accumulator)`

### T2 — The simulated timer (the oracle reference)
- **Goal.** §4's contract as a pure, deterministic Python model with the `spec`, `qemu` and
  `tb3213` profiles, tested against the specification first.
- **Files.** `bcir/frontends/devices/timer_sim.py`; `bcir/tests/test_timer_sim.py`.
- **Tests.**
  - The list in §4.6.
  - The refusal of an undeclared assumption.
  - Byte-identical transcripts across runs.
- **Commit.** `devices: T2 -- the simulated event timer (spec, QEMU and TB3213 profiles), advanced only by explicit ticks`

### T3 — The simulator's C twin
- **Goal.** `runtime/c/simtimer.{h,c}`: caller-owned state, deterministic, with transcripts
  identical to T2's. This follows the host-tool posture of the UART's `sim16550.c`.
- **Files.**
  - The two sources, added to `runtime/manifest.json`, the one source list.
  - A parity test over T2's scenario corpus.
- **Gates.** As for T0, plus `tools/c/check_runtime.sh`.
- **Commit.** `runtime: T3 -- the simulated event timer in C, transcript-identical to the oracle`

### T4 — A DMA completion bound on the event-phase rail
- **Goal.** A deadline that races `dma.done:<engine>`, law-clean, under rules O1–O5.
- **Files.**
  - `bcir/kbcir/deadline.py`, on the cost side, importing no verifier. It defines
    `dma_transfer_with_deadline_module(man, src, dst, *, engine, timer)`, which composes:
    - the phases of `dma_transfer_module`;
    - a comparator arm;
    - EV2's unmask on the timer's own enable;
    - a `timer.expire:…` event phase that writes EXPIRED (and, in level mode, acknowledges);
    - a consumer under both masks.
  - `bcir/tests/test_deadline.py`.
- **Tests.**
  - The module is clean under the R-laws and EV1–EV3.
  - EV2 fires if the timer source is never unmasked.
  - EV3 fires if EXPIRED is read outside the mask.
  - Scripted orders on `SimWorld`:
    - completion first → completed;
    - expiry first → timed out, with the buffers still device-owned (O2);
    - a tie → completed (O1).
- **Notes.** Two HPET timers sharing GINTR_STA stay out of scope until EV3 learns bit-disjoint
  write-1-to-clear stores. That is the handler-vs-handler case `events.py` defers, and the
  extension would be a slice of its own.
- **Commit.** `kbcir: T4 -- a DMA completion bound from an event timer (O1-O5, EV-clean)`

### T5 — UART timeouts on the simulator (after UART U2)
- **Goal.** §4.5's two couplings, once U2's `sim16550.c` and driver exist.
- **Files.**
  - `runtime/c/bcir_timer.{h,c}`: a small polled timer API, compiled through both rails and clean
    under T1. It provides `timer_now` (hi-lo-hi where needed), `timer_arm_oneshot` (arm, read the
    counter, run the signed check) and `timer_ack`.
  - `uart_flush_until`, and an inter-character deadline, in `runtime/c/uart_driver.c`.
  - The harness, running on `SimWorld`.
- **Tests.**
  - The CTI lands at exactly four character periods on the shared timeline.
  - `uart_flush_until` reports a timeout only after its final LSR read (O1).
  - The driver times out the modes that have no CTI.
- **Commit.** `devices: T5 -- UART timeouts on the simulated event timer`

### T6 — Telemetry and the time-source record
- **Goal.** §5.4's signals, and a provider record per clock that follows T8's `None`.
- **Files.**
  - A `TimeSource` record and `time_sources()`, proposed either in a new module or as an addition
    to `bcir/silicon.py` that changes no existing function.
  - The signal registry's new rows, with append-only IDs.
  - Tests.
- **Tests.** On this guest:
  - the HPET is `None`;
  - core cycles are `None`;
  - the TSC is present, with the hypervisor's provenance;
  - no timer quantity is encoded as `cycles`;
  - an unreadable sample is omitted, never sent as 0.
- **Commit.** `telemetry: T6 -- event-timer signals and time sources (None where the host cannot read)`

### T7 — The law-rail halves
- **Goal.** The parts that live on the law rail:
  - `bcir.timer_regmap_contract`;
  - `-bcir-verify` clauses under `R3-timer:`;
  - negatives in `mlir/test/passes/verify_timer_protocol.mlir`;
  - the `check_passes.sh` stanza;
  - the laws' C twin.

  These wait for the MLIR rail, as U0's and U1's halves do.
- **Commit.** `mlir: T7 -- the event-timer laws on the law rail`

### Deferred

| Item | Why | Unblocked when |
|---|---|---|
| A real HPET binding | this guest has none (§5.3), and its routes need the I/O APIC, which is the paused track | a host with an HPET exists and the Intel track resumes |
| The AVR on silicon | there is no rig, and TB3213 lacks offsets and widths | a data sheet and a board are available |
| Parsing the ACPI HPET table | it is item 4 of the native track | that track reaches it |
| Two handlers sharing GINTR_STA | it is EV3's deferred case | an EV3 extension slice lands |
| TSC or core-frequency calibration | the measurement discipline refuses it | never: it would be an estimate |

---

## 7. Errors and inconsistencies (provenance notes, never adopted as facts)

Each note records what is printed and why it is inconsistent. Where a law must decide, the note
also gives the reading that the model takes. T0 would carry these notes as `DocNote` data.

### 7.1 The HPET specification

| ID | Where | What is printed, and the inconsistency | The model's reading |
|---|---|---|---|
| E1 | p.1, p.4 | The revision history lists 0.98 (01/20/2002) before 0.98a (08/31/2001), so 0.98a predates 0.98. Revisions 1.0 and 1.0a are both dated 6/8/2004, yet the cover says October 2004. | Facts are cited as Revision 1.0a. |
| E2 | p.2 vs p.1 | A Revision 1.0a document calls itself "a preliminary draft provided for comment ... Readers should not rely on this specification in any way for product design purposes". | This is recorded as provenance. |
| E3 | p.11 | Figure 3 names bit 15 LEG_ROUTE_CAP, and the table names it LEG_RT_CAP. "femptoseconds" is misspelled three times. | LEG_RT_CAP. |
| E4 | p.11 | GCAP_ID is read-only, and "Writes to this register should not be attempted", yet bit 14 says "software should always write a 0 to these bits". | No writes (T9). |
| E5 | p.12 | Figure 4 (General Configuration) is labeled "Offset: 00h". The register is at 010h (p.10, and p.12's own addressing table). | 010h. |
| E6 | p.12 vs p.13, p.24 | LEG_RT_CNF is described as "0 – Doesn't support / 1 – Supports LegacyReplacement Route", which is capability wording for a configuration bit that forces the mapping. | A configuration bit (p.13, p.24). |
| E7 | p.14 | "Tn_INT_STS Timer n Interrupt Active (where xx is 31:3)" — "xx" is never defined. | n = 3..31. |
| E8 | p.7, p.15, p.23 | Three wordings for consecutive reads: never less "unless the counter has actually rolled over" (p.7); "No two consecutive reads will return the same value, except ... less than the resolution ... or if the counter has rolled over" (p.15); "may return the same value if the access latency ... is less than the clock period", and "unless the counter has rolled over and actually reached the same value" (p.23), which is garbled. | Non-decreasing between rollovers, with equal values allowed within one tick. |
| E9 | p.17, p.23 | Bit ranges are printed as "64:32" for Tn_INT_ROUTE_CAP and Tn_FSB_INT_ADDR, in 64-bit registers. | 63:32. |
| E10 | p.17, p.10, p.23, p.33 | "Tn_FSB_INT_DEL_CNF" names no field; the fields are Tn_FSB_EN_CNF (bit 14) and Tn_FSB_INT_DEL_CAP (bit 15). One register goes by three names: "the Tn_FSB_ROUTE register", "Timer N FSB Interrupt Route Register" and "Timer_N_FSB_Route Register". Also "prarticular", and "the I/O (x)" with a word missing. | One name per field (T0). |
| E11 | p.20 | The wrap example adds "20000" to FFFF0000h and gets 00010000h. Only 20000h gives that result; decimal 20000 (4E20h) gives FFFF4E20h with no wrap. | 20000h; the decimal reading is kept as the note's witness. |
| E12 | p.21 vs p.24 | §2.3.9.2.1 says "it will generate a value in the main counter that matches the value in the timer's comparator register". §2.4.3.1 says "it will generate an interrupt when the value in the main counter matches". | p.24's wording. |
| E13 | p.16, p.21, p.26 | Names that are not field names: GLOBAL_ENABLE_CNF, TIMER0_VAL_SET_CNF, TIMER0_COMPARATOR_VAL, TIMERn_32MODE_CNF; and in the p.16 diagram, "LEG_RT_EN", "T0_FSB_EN" and "Configuration and Capacities". | The register definitions' names. |
| E14 | p.25, p.33 | "Set the Overall Enable bit (Offset 04h, bit 0)", and "General Configuration Register Offset 04h". The register is at 010h (p.10, p.12). | 010h. |
| E15 | p.26 vs p.17, p.15 | p.26 offers Tn_32MODE_CNF as a way to read the 64-bit counter without tearing, saying "The upper 32-bits will always be 0". But the bit belongs to a timer (p.17); no field makes the main counter 32 bits wide; and p.15 rule 4's zero upper half applies to 32-bit counters. | It is a comparator mode. T3 does not accept it as a remedy for the counter. |
| E16 | p.27 | "General Configuration and Capability Register [Offset 0x010]". The register at 0x010 is the General Configuration Register. | GEN_CONF. |
| E17 | p.28 | The Compare 1 row reads "Comparator_0 replaces 8254 PIT Function", copied from Compare 0. Comparator 1 replaces the RTC's periodic interrupt (p.24, p.29). | Comparator 1 → RTC periodic. |
| E18 | p.30 vs p.31 | BASE_ADDRESS is "Lower 32-bit ... The lower 32-bit base address of Event Timer Block", for a 12-byte GAS whose bytes 5–12 hold a 64-bit address (Note #2). EDK2 keeps the name (`BaseAddressLower32Bit`, `edk2-hpet-table-h`). | A 64-bit address in a GAS. |
| E19 | p.30 vs p.12 | The ACPI table says "[12:8] = Number of Comparators in 1st Timer Block". NUM_TIM_CAP "indicates the last timer (i.e. if there are three timers, the value will be 02h)". | The last index. EDK2 and Linux both read it this way (§7 D16). |
| E20 | p.30–31 | The minimum periodic tick field has "Unit: Clock tick", yet Note #3 defines P = (Minimum Period) / (Main counter period), a time-to-ticks conversion. The field says the minimum "can be set without lost interrupts" (≥), and Note #3 says the value "must be greater than P" (>). | Ticks; the comparison is §8 Q4. |
| E21 | p.33 | "Timer N Configuration & Capabilities Register ... Offsets: Timer 0: 108h – 10Fh". Those are the comparator's offsets (p.20); the configuration register is at 100h–107h (p.10, p.17). | 100h + 20h·n. |
| E22 | p.7–8, p.10, p.17, p.27, p.30 | A 1 KiB register space ("160-3FFh Reserved for Timers 3-31") versus 32 timers at (20h × n) + 100h. Timer n's registers end at 11Fh + 20h·n, which is inside 3FFh only for n ≤ 23 (*derived*). The access rules' "1x0h" notation also covers only timers 0–7. | T9 refuses a record whose timers exceed its window. |
| E23 | p.10 vs p.11, p.12, p.14, p.17–18 | Three policies for reserved bits: "Software should perform read-modify-write operations on reserved bits" (p.10); "should always write a 0" (p.11, p.14, p.17–18); and GEN_CONF's "should not modify ... by doing a read-modify-write" (p.12). | Each register's own text (T9). |
| E24 | p.3 vs p.30, p.32 | The table of contents gives "3.2.4 Create ACPI 2.0 HPET Description Table" and "3.2.5 Describe Event Timer(s)"; the body gives "The ACPI 2.0 HPET Description Table" and "Describing Event Timer(s)". | Minor. |
| E25 | p.31–32 | `_UID` is "Optional", yet the HPET Number "must match" it (Note 4; §3.2.5.1 Note 2). Also "__STR". | Minor. |
| E26 | p.21, p.24 | "If the timer is set up for 32-bit mode, then it will generate another interrupt when the main counter wraps around." This reads either as a re-match 2^32 ticks later or as an interrupt at the wrap itself; QEMU implements the second (§7 D7). | No reading adopted; the simulator requires one per scenario (§8 Q6). |
| E27 | p.21 | "the comparator disabled": no field disables a comparator, and Tn_INT_ENB_CNF = 0 leaves the timer operating (p.18). | Tn_INT_ENB_CNF = 0 (T2(c)). |
| E28 | p.14 vs p.18 | The status bit is set "if the corresponding timer interrupt is active" (p.14). Yet at Tn_INT_ENB_CNF = 0 "the timer will still operate and generate appropriate status bits" (p.18). | It latches regardless; QEMU does the same (`update_irq`). |
| E29 | p.13 vs p.24, p.28 | "If the LegacyReplacement Route bit is set, the individual routing bits for timers 0 and 1 (APIC or FSB) will have no impact" (p.13), versus "The FSB interrupt delivery can be used even when the LegacyReplacement mapping is used" (p.24) and "does not preclude delivery of IRQ0/IRQ8 via FSB" (p.28). QEMU sides with p.24 (`update_irq` checks FSB first). | Not adopted; FSB under legacy routing stays a variant datum. |

### 7.2 TB3213

| ID | Where | What is printed, and the inconsistency |
|---|---|---|
| M1 | p.10 | Figure 3-7 is captioned "RTC.CLKSEL – Set the DBGRUN Bit". The register is RTC.DBGCTRL, which the code below the figure writes. |
| M2 | p.10 | "refer to Accessing 16-bit Registers in the CPU section". TB3213 has no such section; this is a data-sheet cross-reference. |
| M3 | p.7, p.11 | A "Correction counter" (Figure 2-1) and a CORREN bit (Figure 3-8) are shown and never described. |
| M4 | p.13 | In the extracted text, the PITCTRLA table is glyph-substituted: the bit row reads "7 6 ' v - 9 8 Y", and entries read "CYCv" and "CYC86". Decoding with the bit row (', v, -, 9, 8, Y = 5, 4, 3, 2, 1, 0) gives 4 … 32768 cycles. The substitution maps 1 and 8 to one glyph, and 2 and 9 to another, so the table alone is ambiguous. p.7's set {4, 8, 16, .. 32768} and the code's `RTC_PERIOD_CYC32768_gc` (p.13, p.20–21) settle it. The text cannot show whether the PDF or the extraction is at fault. |
| M5 | p.8 vs p.9 | SEL and CSUT are locked while "ENABLE ... or ... XOSC32KS ... is high" (p.8), but only "when the oscillator is enabled (ENABLE = 1)" (p.9). |
| M6 | p.10 vs pp.18–21 | XOSC32KCTRLA.RUNSTDBY "must be configured accordingly", yet no example writes it, including Example 7-3, which sleeps in power-down with the PIT as wake source. "Accordingly" is never defined. |
| M7 | pp.9–11, p.13 | Busy flags for CMP and CNT are defined, but the write rule is stated only for PER and CTRLA. No flag is mentioned for PITCTRLA. |
| M8 | p.1 vs pp.12–16 | Each use case is said to have two examples (bare metal on ATmega4809, MCC on AVR128DA48). Only the bare-metal ones are printed; the AVR128DA48 code is not in the brief. |

### 7.3 Across the documents

| ID | Where | Inconsistency |
|---|---|---|
| X1 | HPET p.25 vs EDK2 | "all active high", versus EDK2's "Asserted Low" (§7 D2). |

### 7.4 The drivers and the emulator against the documents

| ID | Source | Against | Note |
|---|---|---|---|
| D1 | `edk2-hpet-timer-dxe-c` `TimerInterruptHandler` l.310, with `TimerDriverInitialize` l.892 | HPET p.14 | EDK2 writes 1 to its timer's status bit in MSI mode, where it has cleared LevelTriggeredInterrupt (edge mode); p.14 says to write 0. Zephyr follows p.14 (`hpet_isr`). QEMU ignores the write (edge bits are never set). |
| D2 | `edk2-hpet-timer-dxe-c` l.904–907 | HPET p.25; within EDK2 | The comment says "Fixed Delivery Mode, Level Triggered, Asserted Low", while the call passes `IO_APIC_DELIVERY_MODE_LOWEST_PRIORITY`. "Asserted Low" differs from p.25's "all active high". |
| D3 | `zephyr-hpet-c` file comment and `HPET_COUNTER_CLK_PERIOD` | HPET p.11 | An override for parts whose COUNTER_CLK_PERIOD "is not in femtoseconds". |
| D4 | `zephyr-hpet-c` comment above `TIMER_CORE_COUNTER_NONMONOTONIC` | HPET p.7, p.15, p.23 | "Under QEMU SMP the shared counter can be observed reading backwards." QEMU's `hpet_ram_read` computes the counter from its virtual clock at each access; the supplied code does not show why it would go backwards. |
| D5 | `zephyr-hpet-c` `sys_clock_no_timeout` | the specification's comparator | QEMU "converts the distance to the target into nanoseconds in a signed 64-bit value, so an extreme target overflows and the timer fires immediately and forever". QEMU's `hpet_get_ns` (tick × HPET_CLK_PERIOD − offset) can overflow for large ticks (*reading*). |
| D6 | `qemu-hpet-c` `hpet_set_timer`, `hpet_calculate_cmp64` | HPET p.21, p.25 | A 64-bit one-shot target that the counter has passed is armed at its past time, not missed. |
| D7 | `qemu-hpet-c` `hpet_set_timer` (`wrap_flag`) | HPET p.21, p.24 | One reading of E26: an interrupt at the 32-bit wrap, and no re-match afterwards. |
| D8 | `qemu-hpet-c` `hpet_timer`, `hpet_arm` | HPET p.20 | Periods the host missed are coalesced into one interrupt, and periodic spacing is clamped to at least 1 µs. |
| D9 | `linux-x86-hpet-c` `hpet_legacy_clockevent_register` comment vs `qemu-hpet-c` `hpet_reset` | each other | Linux says QEMU supports periodic mode "despite the fact that both don't advertise it". The supplied QEMU sets HPET_TN_PERIODIC_CAP on every timer. The files cannot say which QEMU version Linux meant. |
| D10 | `linux-x86-hpet-c` `hpet_legacy_clockevent_register` | HPET p.18 | Linux forces periodic mode on legacy channel 0 whatever Tn_PER_INT_CAP says; the specification ignores Tn_TYPE_CNF writes when the capability is 0. |
| D11 | `linux-x86-hpet-c` `hpet_clkevt_set_state_periodic` | HPET p.21, p.25 (W2, W3); p.20 | Linux programs periodic mode without zeroing the counter. It also writes the period separately, because "HPET on AMD 81xx needs a second write" — whereas the p.20 example and QEMU set both values with one VAL_SET write. |
| D12 | `linux-x86-hpet-c` `hpet_is_pc10_damaged` | HPET p.10–11 | Linux: the HPET "stops counting in PC10", so it force-disables it; Linux calls its own check "overbroad". The specification's power rules name only the sleep states S1–S5 (p.10–11) and say nothing about processor idle states. |
| D13 | `linux-x86-hpet-c` `hpet_clkevt_set_next_event` | HPET p.20 | Comparator writes "delayed internally up to two HPET clock cycles" (ATI, ICH9/10), longer on newer AMD, and a stale first readback on ICH9/10. The specification states no write latency. |
| D14 | `edk2-hpet-timer-dxe-c` `TimerDriverSetTimerPeriod` l.568–571 vs l.86–92 | within EDK2 | The tick count is floored, while the protocol text says the period "will be rounded up". |
| D15 | `edk2-hpet-timer-dxe-c` l.379, l.523, l.580, l.724 | arithmetic | The wrapped interval `(mask − prev) + cur` is one tick short of `(cur − prev) mod 2^w` (*derived*, Appendix A8). |
| D16 | `edk2-hpet-register-h` / `edk2-hpet-table-h` `NumberOfTimers`; `edk2-hpet-timer-dxe-c` l.823, l.842; `linux-x86-hpet-c` `hpet_enable` | HPET p.30 vs p.12 | EDK2 names the field after p.30's label but loops 0..NumberOfTimers inclusive; Linux adds 1. Both read p.12, which supports E19's reading. |
| D17 | `zephyr-hpet-c` `hpet_counter_get` (`CONFIG_MMU`) | the measurement discipline | Returns 0 when the block is not mapped yet. This does not contradict the specification; it is the contrast for T8. |
| D18 | `zephyr-hpet-c` `hpet_counter_get` (`CONFIG_64BIT`); `edk2-hpet-timer-dxe-c` `HpetRead` | HPET p.26 note | One 64-bit read of the running counter, assuming the platform does not split it; p.26 warns that some platforms do. QEMU honors 8-byte accesses (`hpet_ram_ops`). |
| D19 | `zephyr-hpet-c` `hpet_timer_comparator_set` | HPET p.26 | On 32-bit builds, a 64-bit comparator is written in two halves while the counter runs. The documents' rules cover reads, not this case (T3). |

---

## 8. Open questions for the user

1. **Q1 — the AVR device.** Which device should the placement be: the ATmega4809 of the
   bare-metal examples, or the AVR128DA48 of the MCC ones? Its data sheet would settle register
   offsets, the widths of CNT and CMP, INTFLAGS.CMP, any PIT busy flag, the CMP/CNT
   synchronization latency, 16-bit access, and which sleep modes keep what running. TB3213 is the
   only source; the ATtiny1614 data sheet was refused.
2. **Q2 — T1's extension.** TB3213 states the busy rule for PER and CTRLA only. Should T1 extend it
   to CMP and CNT?
3. **Q3 — T5's form.** This document makes arm-then-verify (Linux) normative and admits arming with
   the counter halted (EDK2). What margin should the simulator scenarios use for `min_arm_ticks`?
   Linux's 128 is a driver constant; the documents state none.
4. **Q4 — the periodic minimum (E20).** Should a period be ≥ P or > P?
5. **Q5 — the canonical enabling order.** Should BCIR's own drivers follow the specification
   (overall enable last), which this document proposes?
6. **Q6 — the 32-bit one-shot wrap (E26).** Should it be a re-match, or an interrupt at the wrap?
   Until this is answered, the simulator requires a reading per scenario.
7. **Q7 — unmasking with a level status pending.** The documents do not say what happens, and QEMU
   raises the interrupt again. Should the specification profile adopt QEMU's behavior, or refuse
   such scenarios?
8. **Q8 — telemetry.** Should the timer signals get new IDs now (an append-only change), and when
    should the envelope gain a device-clock timestamp (a version bump)?
9. **Q9 — a TSC-to-timer ratio.** May such a ratio be recorded as a Level A measurement, provided
    it is never promoted to a frequency? Or is that out of bounds too?
10. **Q10 — sharing the claim-graph facts.** Should `uart_laws._Facts` move into a shared module for
    T1 (L14)? This touches the UART branch, so it needs its owner's agreement.
11. **Q11 — deep idle.** Linux says the HPET stops counting in PC10 (D12). Should the HPET count
    as an elapsed-time clock across deep idle only on platforms with evidence for it?
12. **Q12 — the conservative readings.** Are these acceptable: T2(e) (CLKSEL written only while the
    RTC and PIT are disabled); the simulator's refusal to invent outcomes the documents leave open;
    and an AVR CNT width that each scenario declares?
13. **Q13 — under-reported capabilities.** For a part that supports a mode without advertising it
    (D10), should the model use an erratum record (proposed) or a law exception?
14. **Q14 — simulation only.** On this guest, and until a host exposes an HPET and the Intel track
    resumes, the HPET placements are simulation-only. Confirm?

---

## Appendix A — Derived arithmetic (none of this is printed in the documents)

- **A1.** 05F5E100h = 100,000,000 fs = 100 ns, which corresponds to 10 MHz (HPET p.9, p.11).
- **A2.** 2 ticks × 100 ns / 100 µs = 0.2% = 2000 ppm, and 0.05% = 500 ppm (HPET p.9, p.23).
- **A3.** FFFF0000h + 20000h = 1_0001_0000h, which is 00010000h mod 2^32. By contrast,
  FFFF0000h + 4E20h (decimal 20000) = FFFF4E20h, with no wrap (E11).
- **A4.** 100h + 20h·n + 1Fh ≤ 3FFh holds exactly when n ≤ 23. Timers 24–31 would sit at
  400h–4FFh (E22).
- **A5.** 2^32 ticks of 100 ns = 429.4967296 s. 2^64 ticks of 100 ns ≈ 1.845 × 10^12 s ≈ 58,454
  years of 365.25 days.
- **A6.** 32768 Hz / 32 = 1024 Hz, and (511 + 1) / 1024 = 0.5 s (TB3213 p.8, p.18). CYC32768 at
  32768 Hz = 1 s (p.13).
- **A7.** Take a torn read at 0000_0000_FFFF_FFFFh with one tick between the halves; the true value
  at the second read is 0000_0001_0000_0000h. Reading low then high gives 0000_0001_FFFF_FFFFh.
  Reading high then low gives 0.
- **A8.** With prev = 2^32 − 3 and cur = 1 on a 32-bit counter, (cur − prev) mod 2^32 = 4, while
  (mask − prev) + cur = 3.
- **A9.** The ACPI HPET table is 36 + 4 + 12 + 1 + 2 + 1 = 56 bytes (HPET p.30–31).
- **A10.** PIT code c gives 2^(c+1) cycles: 0x1 → 4, and so on up to 0xE → 32768 (TB3213 p.7,
  p.13).
