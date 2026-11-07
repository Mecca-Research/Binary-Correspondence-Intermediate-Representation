# BCIR DMA — the controller blueprint (registry, laws, simulator, lowering)

> **Status (2026-10-27): a proposal. Nothing in it is built, and no slice is a milestone.**
> The driver/kernel roadmap's A2 row stands as written: descriptors are derived and priced from
> `StridedView` pairs, and "No DMA engine is programmed on hardware"
> ([`BCIR_DRIVER_KERNEL_ROADMAP.md`](BCIR_DRIVER_KERNEL_ROADMAP.md) §2.2). This document is the
> research contract that would come before a DMA engine is programmed: gate D0 of the roadmap's
> §5.1 maturity ladder. That gate covers the normative specification, the variant matrix,
> provenance, the manifest shape and the failure model.
>
> **What this document is.** The DMA-engine registry the UART blueprint is waiting for. Its §9
> defers bus-master DMA until "a DMA-engine registry model" exists, and its §1.5 documents
> RXRDY/TXRDY as deferred. The document then sets out the laws, the deterministic simulator and
> the lowering that would make §1.5 real and land §12's U11 buffer-ownership law. Like the UART
> blueprint, it is written so that an implementer can build each slice without high-level
> planning. Every device fact is cited. Every gap is named as a gap, never filled by a guess.
>
> **Sources.**
> - *Primary* (the user's three documents):
>   - **AN205266** Rev \*B, *FR, MB91460, Direct Memory Access* (Cypress/Infineon; the CY91460
>     DMAC);
>   - **AN234225** Rev \*B, *XMC7000: Using direct memory access (DMA) controllers* (Infineon;
>     the DW and DMAC engines);
>   - **SPNU194**, *TMS470R1x Direct Memory Access (DMA) Controller Reference Guide* (TI,
>     November 2002).
> - *16550 side:* **PC16550D** (TI/NSC SNLS378C), only the RXRDY/TXRDY, FCR3 and timing
>   statements that the UART registry already cites (`uart_model.DMA_MODES`).
> - *Supplementary*, fetched 2026-10-27 at the user's request:
>   - Infineon's PDL `cy_dma.h`/`cy_dma.c` (driver 2.90) and `cy_dmac.h`/`cy_dmac.c` (driver
>     1.40), mtb-pdl-cat1 master, cited as `infineon-cy-dma-h`, `infineon-cy-dma-c`,
>     `infineon-cy-dmac-h` and `infineon-cy-dmac-c`;
>   - ARM CMSIS `cachel1_armv7.h` V1.0.3 (`cmsis-cachel1-armv7-h`);
>   - QEMU's `hw/dma/pl330.c` (`qemu-pl330-c`). This is QEMU's model of the ARM PL330, not
>     ARM's manual.
> - *Refused* by the session's network policy: the PL330 TRM (DDI 0424A), ST AN4031 and
>   AN4839, and the kernel's dmaengine guides. Nothing below depends on them.
>
> **Citation form.**
> - `AN205266 p.N`, `AN234225 p.N`, `SPNU194 p.N` and `PC16550D p.N` give PDF page N of the
>   extracted text (its `===== PAGE N =====` marker). The printed page number is N−1 in
>   AN205266, N−8 in SPNU194 and N in AN234225.
> - Library and model sources are cited by name, with a function or line number (L).
> - BCIR code is cited by path and line.
>
> **Authority.** The sources do not carry equal weight:
> - SPNU194 is a reference guide.
> - Both application notes disclaim themselves. AN234225 p.42 says it is "given as a hint for
>   the implementation of the product only and shall in no event be regarded as a description
>   or warranty of a certain functionality". AN205266 p.23 says that its code is "provided only
>   for reference purposes".
> - The PDL is code: it is authoritative for what the PDL writes into a descriptor, not for what
>   the silicon does with it.
> - The QEMU file is a simulator.
>
> The registry records the authority with every fact (§6, D0). A fact that only an application
> note, the PDL or a model supports is marked so, and needs a TRM before hardware promotion
> (gate D7).

---

## 0. Purpose, scope, and how to use this document

**The first consumer is the 16550 UART driver.** UART blueprint §1.5 defines two pin
handshakes, RXRDY and TXRDY, with mode 0 or mode 1 selected by FCR3. They have been documented
and deferred. This program makes them real: a simulated DMA controller is paced by the
simulated 16550's RXRDY/TXRDY, and its buffers obey the ownership law that UART §12 proposes
for U11:

> AVAILABLE → RESERVED → DEVICE_OWNED → COMPLETING → RETIRED → AVAILABLE,
> DEVICE_OWNED(B) ⇒ ¬REUSE(B), and a cancellation is a transition request, never proof that
> the device stopped.

**The second consumer is A2.** `bcir/kbcir/dma.py` compiles a `StridedView` pair into
`Descriptor(src_off, dst_off, nbytes)` runs, prices them, and composes a planned module with an
event-phase completion. This program lowers that program onto the real formats of four
controllers:
- the XMC7000 DW and DMAC descriptor words;
- the TMS470R1x control packets;
- the CY91460 channel registers.

The lowering refuses, by name, anything a format cannot express.

**In scope.**
- One normative, capability-parameterized DMA model, with five placements:
  - the four hardware variants;
  - QEMU's PL330 model, as a simulator reference row only.
- The D-laws (§4).
- A deterministic, tick-timed C simulator, coupled to the UART simulator (§5).
- The lowering (§3).
- Telemetry.
- The PIO-versus-DMA choice as a priced, certified plan.

**Out of scope.**
- Any hardware binding: there is no rig, and the A2 row's boundary stands.
- IOMMUs.
- virtio queues: that is the roadmap's driver 3.
- DMA from on-chip UARTs (the XMC7000 SCB, the MB91460 UART2). The documents' own examples use
  them; this program's UART is the 16550.
- CRC, scatter, memory-copy and fly-by transfers, beyond registry rows and refusals.
- A PL330 microcode compiler.

**How to use it.** Build the slices in order (§6). Each slice is one PR-sized, independently
gated change, and its **Design** paragraph is normative. §1 and §2 are the single source of
truth for the registry, the laws, the simulator and the tests.

A field that §1/§2 record as **not stated** is a hole in the sources. The registry holds it as
`None` (null, never zero), and every law and generator that needs it refuses until a cited
source supplies it. Never fill one silently.

House method, restated (`.claude/skills/bcir-systems-engineer/SKILL.md`):
- registry first;
- legality before cost;
- prototype on the oracle, then port to the twin, then lock with parity;
- every new law vacuous off-DMA, measured over the whole corpus;
- tests that drive the failure (`docs/security/laws.md` L2);
- every rule with a refused and an admitted witness (L22).

**Names.** The slices are D0–D7, as the request asked. PR titles should spell them `DMA-D0` …
`DMA-D7`, because the roadmap already uses "D0–D7" for its maturity gates (§5.1) and "D-R1…D-R6"
for the driver laws (§10, item 4). The laws are named by mnemonic (D-OWN, D-DESC, D-WIDTH, D-COUNT,
D-TRIG, D-IDLE, D-REUSE, D-COH, D-PLAN, D-HEAD) and are never numbered D1…Dn. Fault-table labels
take a per-table prefix: DR (registry), DL (laws), DS (simulator), DU (UART coupling), DW
(lowering), DT (telemetry), DP (plan). §8 Q1 asks the user to confirm the scheme.

---

## 1. The normative DMA model

What follows is what the registry encodes (§6, D0). Each statement carries its source.
- "Not stated" means the named sources are silent. It is never a claim that the silicon lacks
  the feature.
- PDL statements are marked as the PDL's. They describe what Infineon's library writes and
  requires, which on CAT1C (the XMC7000 family's PDL category) is the closest available
  evidence after the TRM, which this session does not have.

### 1.1 Engines and channels

| | XMC7000 DW | XMC7000 DMAC | CY91460 DMAC | TMS470R1x DMA |
|---|---|---|---|---|
| Engine | one transfer engine shared by all channels (AN234225 p.3 Table 1, p.5) | one engine per channel (AN234225 p.3, p.5) | per-channel register sets on the CPU's bus: "the DMAC shares the same bus as of CPU" (AN205266 p.5); bus taken by hold request/acknowledge (p.11) | one controller between the CPU bus and the peripheral bus (SPNU194 p.11) |
| Channels | "multiple"; count per datasheet (AN234225 p.3, p.13). The PDL bounds by `CY_DW0_CH_NR`/`CY_DW1_CH_NR` per block (`infineon-cy-dma-h` L384-387). A comment says DW version 2 supports "upto 512 channels" (L699) | "a small number of channels" (AN234225 p.3); count per datasheet | 5 (AN205266 p.8, p.9) | 16, or 8 in the 16-packet configuration (SPNU194 p.10) |
| Blocks | several DW blocks, device specific (`infineon-cy-dma-h` L45); examples use DW1 (AN234225 p.16, p.32) | one DMAC block in the examples (AN234225 p.37) | one | one |
| Where the program lives | descriptors in memory outside the DMA block, read by the engine when it activates a channel (AN234225 p.5, p.8); "DMA will read descriptors from SRAM memory" (`infineon-cy-dma-h` L79) | the same (AN234225 p.8; `infineon-cy-dmac-h` L79) | the channel's registers DMACA, DMACB, DMASA, DMADA (AN205266 pp.6-8) | 32 (or 16) control packets in a command buffer at 0xFFFFF800–0xFFFFF9FF, inside the system-control register space (SPNU194 p.13) |

Two facts in SPNU194, read together, mean that no DMA can rewrite its own control packets. The
controller moves data "between any locations in the TMS470 memory map except the system control
registers" (p.10, p.11), and the command buffer sits in that space (p.13). The registry records
the consequence as a placement fact: no self-modifying packet chains.

### 1.2 Programs: descriptors, control packets, channel registers

**XMC7000 DW** (bitfields from the PDL, `infineon-cy-dma-h` L389-414 and L2074-2109; layout from
AN234225 p.8):

| Word | Offset | Contents |
|---|---|---|
| `ctl` | 0x00 | RETRIG [1:0], INTR_TYPE [3:2], TR_OUT_TYPE [5:4], TR_IN_TYPE [7:6], CH_DISABLE [24], SRC_TRANSFER_SIZE [26], DST_TRANSFER_SIZE [27], DATA_SIZE [29:28], DESCR_TYPE [31:30] |
| `src` | 0x04 | source address [31:0] |
| `dst` | 0x08 | destination address [31:0] |
| `xCtl` | 0x0C | SRC_X_INCR [11:0], DST_X_INCR [23:12], X_COUNT [31:24] |
| `yCtl` | 0x10 | SRC_Y_INCR [11:0], DST_Y_INCR [23:12], Y_COUNT [31:24] |
| `nextPtr` | 0x14 | ADDR [31:2] (`infineon-cy-dma-h` L2108-2109) |

- **Descriptor types.** Single (0), 1D (1), 2D (2), CRC (3) (`infineon-cy-dma-h` L293-299;
  AN234225 p.3).
- **Compaction.** "A word address is shifted forward if there is any unused descriptor word"
  (AN234225 p.8). The next pointer is therefore at 0x0C (single), 0x10 (1D and CRC) or
  0x14 (2D) (`infineon-cy-dma-c` L99-102, `Cy_DMA_Descriptor_SetNextDescriptor` L336-358),
  which agrees with AN234225 pp.8-9.
- **Counts.** 1…256 elements, held in the machine field as count−1
  (`Cy_DMA_Descriptor_SetXloopDataCount`, `infineon-cy-dma-h` L1298-1299).
- **Increments.** Signed, −2048…2047 (L257-260).
- **Data size.** Byte, halfword or word (L311-316).
- **Transfer size.** For each side, either "as specified by dataSize" or "a full word" (L328-332).
- **Channel state on completion.** The channel stays enabled or is disabled when the descriptor
  completes (L335-339).
- **Retrigger.** Immediately, after 4 or after 16 Clk_Slow cycles, or wait for trigger
  reactivation (L319-325).
- **CRC descriptors** use only the source X increment and the X count
  (`Cy_DMA_Descriptor_Init`, `infineon-cy-dma-c` L185-200).

**XMC7000 DMAC** (word layout from the PDL; field positions live in device headers that were
not fetched):

| Type | Words, in order (`infineon-cy-dmac-h` L359-408) | Next pointer at (`infineon-cy-dmac-c` L331-336) |
|---|---|---|
| single | ctl, src, dst, next | 0x0C |
| scatter | ctl, src, xSize, next (no dst word; X count at 0x08) | 0x0C |
| memory copy | ctl, src, dst, xSize, next | 0x10 |
| 1D | ctl, src, dst, xSize, xIncr, next | 0x14 |
| 2D | ctl, src, dst, xSize, xIncr, ySize, yIncr, next | 0x1C |

- **The `ctl` fields.** WAIT_FOR_DEACT (retrigger), INTR_TYPE, TR_OUT_TYPE, TR_IN_TYPE,
  DATA_PREFETCH, SRC_TRANSFER_SIZE, DST_TRANSFER_SIZE, CH_DISABLE, DATA_SIZE, DESCR_TYPE
  (`Cy_DMAC_Descriptor_Init`, `infineon-cy-dmac-c` L80-90). Their positions are *not stated*
  in the fetched sources.
- **Counts.** 1…65536, machine value count−1. Scatter takes 1…32768 [address, data] pairs,
  with machine value 2×pairs−1 (`infineon-cy-dmac-h` L192-197; `infineon-cy-dmac-c` L314).
- **Count units.** Bytes for memory copy, pairs for scatter, and the data size otherwise
  (`infineon-cy-dmac-c` L248-252).
- **Increments.** −32768…32767 (`infineon-cy-dmac-h` L199-202).
- **Data prefetch.** "Source data transfers are initiated as soon as the channel is enabled, the
  current descriptor pointer is NOT '0' and there is space available in the channel's data
  FIFO" (L428-430).
- **Ignored settings.** Data and transfer sizes are ignored for memory copy and scatter
  (L431-439).

AN234225's DMAC figures disagree with this layout for 1D (§7, AN-X2).

**CY91460** (AN205266 pp.6-8; one program per channel, no list):

| Register | Fields |
|---|---|
| DMACA | DENB [31], PAUS [30], STRG [29], IS [28:24], EIS [23:20], BLK [19:16] (0 means 16, p.6 fn 4), DTC [15:0] |
| DMACB | TYPE [31:30] (2-cycle; fly-by memory→I/O; fly-by I/O→memory; 11 "setting disabled"), MOD [29:28] (block/step; burst; demand; 11 disabled), WS [27:26] (byte; half; word; 11 disabled), SADM [25], DADM [24] (increment / decrement), DTCR [23], SADR [22], DADR [21] (reload; SADR and DADR only with DTCR = 1, p.7 fn 5), ERIE [20], EDIE [19], DSS [18:16], SASZ [15:8], DASZ [7:0] |
| DMASA, DMADA | source and destination address |
| DMACR (global) | DMAE [31], PM01 [28], DMAH [27:24] |

Two gaps in these tables:
- The ERIE/EDIE polarity contradicts the AN's own examples (§7, CY-1). The registry holds both
  polarities as null.
- The values of IS and EIS are "refer the hardware manual" (p.6 fn 3).

**TMS470R1x** control packet (SPNU194 pp.13-19; four 32-bit words, word writes only, aligned to
four words, p.13):

| Word | Offset | Fields |
|---|---|---|
| DMAC | 0x0 | NCPACK [28:24] (next packet), INTEN [15], TRSIZE [14:13] (8/16/32-bit; 11 reserved), DSTINC [11], SRCINC [9], DSTMOD [8:5], SRCMOD [4:1], DCHN [0] (pp.14-16) |
| DMASA | 0x4 | source address (p.17) |
| DMADA | 0x8 | destination address (p.18) |
| DMATC | 0xC | TRANSCOUNT [15:0]: 0…65535 transactions; 0 means 0 (p.19) |

Every packet word is "RWP": written in privilege mode only (p.14 ff.).

The module codes in SRCMOD/DSTMOD (p.16) are:
- 0000–0101: fine memory selects 0–5;
- 0110–1011: coarse memory selects 0–5;
- 1100: a module with its own decoding on the CPU bus;
- 1111: expansion-bus peripherals;
- 1101 and 1110: reserved.

### 1.3 Requests and triggers

**Sources.**
- **XMC7000 DW** takes a hardware trigger, a software trigger, or another channel's trigger
  output (tr_out), all through the trigger multiplexer (AN234225 p.4).
  - The PDL can also activate a channel directly with `Cy_DMA_Channel_SetSWTrigger`
    (TR_CMD.ACTIVATE), except on M4CPUSS DMA version 1 (`infineon-cy-dma-h` L78, L1998-2002).
- **XMC7000 DMAC** takes only a software trigger or tr_out. "Unlike DW, DMAC does not support
  hardware triggers" (AN234225 p.4, p.5). The PDL's generic text speaks of "a hardware trigger
  signal from another component" for both engines (`infineon-cy-dmac-h` L36-37); §7 records
  the disagreement as LIB-AN-3.
- **CY91460** takes:
  - a peripheral interrupt chosen by IS/EIS (AN205266 p.6). The DMAC clears the interrupt that
    initiated the request (p.9, p.10);
  - the software trigger STRG, "always valid regardless of" IS/EIS (p.6 fn 2);
  - the external DREQ pin, on channels 0–3 only (p.2).
- **TMS470R1x** has two modes:
  - request mode (RQEN = 1): "each DMA transaction … is performed only when the DMA controller
    receives a DMA request" (SPNU194 p.25), and it waits for "a rising edge of the DMA_REQ
    signal" (p.40);
  - non-request mode: it runs as soon as the channel is enabled (p.25, p.40).

  Which signal drives which channel's DMA_REQ is *not stated*.

**Sensitivity (level, pulse, edge).**
- **XMC.** The software trigger is a pulse, issued as `CY_TRIGGER_TWO_CYCLES` (AN234225 p.34,
  p.37). The SCB "keeps active level for DMA" (p.17). The descriptor's retrigger field sets how
  a held trigger is re-sampled: immediately, after 4 or 16 Clk_Slow cycles, or after
  deactivation (`infineon-cy-dma-h` L319-325, L454). The Clk_Slow frequency is *not stated*.
- **CY91460.** DREQ can be set to "low level or high level and … rising edge or falling edge",
  with a minimum width of 5 system clocks (AN205266 p.11). Demand mode is level-sensitive: high
  level with IS = 01110, low level with IS = 01111 (p.10). The example comments IS = 0xF as
  "low level or falling edge" (p.16). In step mode, "If there is one more transfer request
  during the transfer … it would be ignored" (p.9).
- **TMS470R1x.** Rising edge (SPNU194 p.40).

**Quantum per request.**
- **XMC.** The input trigger type sets the quantum (AN234225 p.10; `infineon-cy-dma-h`
  L302-308):
  - 0 — one element;
  - 1 — one X loop;
  - 2 — one descriptor;
  - 3 — the whole chain.
- **CY91460** (AN205266 pp.9-10):
  - step — one unit;
  - block — BLK units;
  - burst — BLK × DTC units from one request;
  - demand — one unit for each sample of the active level.
- **TMS470R1x.** One transaction per request (SPNU194 p.25).

**Output triggers.**
- **XMC.** tr_out types 0–3 mirror the input types (AN234225 p.10), and can trigger the same
  channel or others (p.5).
- **CY91460.** DACKX acknowledges a request, and DEOP is asserted while the last data item of
  the last block is transferred, never for internal-to-internal transfers (AN205266 p.11).
- **TMS470R1x.** None stated.

**Priority and arbitration.**
- **XMC DW.** Four levels. A preemptable channel can be preempted "between single transfers"
  (AN234225 p.3, p.4); PRIO is CH_CTL [9:8] and PREEMPTABLE is CH_CTL [11] (p.7).
- **XMC DMAC.** Four levels, not preemptable "because it would degrade the overall memory
  bandwidth" (p.4).
- **CY91460.** PM01 selects fixed priority (ch0 > ch1) or "alternating priority (ch1 > ch0)"
  (AN205266 p.8). The order among channels 2–4 is *not stated*.
- **TMS470R1x.**
  - CHPR selects fixed priority (channel 0 highest) or rotating, where the most recently served
    channel goes to the bottom (SPNU194 p.32).
  - CHSS sets 1–16 transactions per channel before switching (pp.32-33).
  - BMSS sets the CPU:DMA turns as 1:1 or 1:4 when both want the bus (p.32).
  - Appendix A states "rotating" unconditionally (p.40); see §7, TI-13.

**Halts, pauses and holds.**
- **CY91460** (AN205266):
  - DMAE disables every channel (p.8).
  - A non-zero DMAH halts every channel. It is a 4-bit counter, which an ISR increments and
    decrements to hold the DMAC off (p.8, p.11).
  - PAUS pauses one channel (p.6).
  - An NMI, or an interrupt above the hold-suppress level, cancels the transfer "at a transfer
    unit boundary (one block)" until the request is cleared (p.11).
  - An external bus request (BRQ) holds the DMAC off the external bus (p.12).
- **TMS470R1x** (SPNU194):
  - DMA_HALT and DMA_STOP each finish the current transaction and stop until cleared
    (pp.33-34).
  - HIRQ and HFIQ set DMA_HALT whenever the CPU enters IRQ or FIQ mode (pp.31-32).
  - SEN decides whether a channel runs while the debugger suspends the CPU (p.24).
- **XMC.** Nothing beyond enable and disable is stated.

### 1.4 Transfer geometry

**Element width.**
- **XMC.** 8, 16 or 32 bits (AN234225 p.3; `infineon-cy-dma-h` L311-316). Each side also has a
  separate *transfer* size, "as specified by dataSize" or "a full word"
  (`infineon-cy-dma-h` L328-332). How a byte is placed on a word transfer (zero extension, which
  lane) is *not stated*.
- **CY91460.** Byte, halfword or word; WS = 11 is "setting disabled" (AN205266 p.7).
- **TMS470R1x.** 8, 16 or 32 bits; TRSIZE = 11 is reserved (SPNU194 p.15).

**Counts and 2D.**

| | X / Y | Count range | Encoding |
|---|---|---|---|
| XMC DW | X and Y | 1…256 | count−1 |
| XMC DMAC | X and Y | 1…65536 | count−1 |
| CY91460 | 1D only | BLK 1…16 × DTC 16-bit | BLK: 0 means 16; DTC = 0 *not stated* |
| TMS470R1x | 1D only | 0…65535 transactions | 0 means 0 transfers (p.19) |

The AN234225 pseudo-code bound `X_IDX <= COUNT` (p.9) agrees with the PDL's count−1 machine
field; the AN never states the encoding.

**Increments: fixed, increment, decrement, wrap.**
- **XMC.** Signed X and Y increments for source and destination (DW ±2048, DMAC ±32768; see
  §1.2). An increment of 0 holds an address fixed: the TX FIFO is written at one address
  (AN234225 p.15). The *unit* of the increment (elements or bytes) is *not stated* in the AN or
  the PDL text (§8 Q5).
- **XMC, 2D.** How a Y increment composes with the X walk is *not stated* either: it may be
  measured from the row's first element, or added after the row's last X step. The PDL says
  only "after each X-loop transfer" and "after each Y-loop transfer" (`infineon-cy-dma-h`
  L468-472). The AN's 2D pseudo-code says "DST_INCR/SRC_INCR depend on X/Y_INCR" (p.9), and
  its 2D example gives no increment values (pp.27-29). The two readings place every row after
  the first at different addresses (§3.5, §8 Q5).
- **CY91460.**
  - SASZ and DASZ are 8-bit step widths "for the transfer source/destination address … in each
    transfer operation" (AN205266 p.8).
  - SADM and DADM select increment or decrement (p.7).
  - The unit is bytes according to the example ("Source Address increment by 4" with word
    transfers, p.16); the table states none.
  - A step of 0 is "Source Address Same" (p.19).
  - Reload of the count and addresses gives whole-buffer repetition (p.7, p.9).
  - There is an end code "Address error (underflow/overflow)" (p.8). What under- or overflows
    is *not stated*.
- **TMS470R1x.**
  - SRCINC and DSTINC either hold the address constant or add 1, 2 or 4, matching TRSIZE
    (SPNU194 p.15).
  - An expansion-bus peripheral destination is "always incremented by four, irrespective of"
    TRSIZE (p.15).
  - The SRCINC sentence conditions the source's step on the *destination* (§7, TI-8).
  - There is no decrement and no wrap.

**Addresses and alignment.**
- **TMS470R1x.** Addresses are aligned to the transfer size by masking. The low two bits are
  ignored for 32-bit transfers and bit 0 for 16-bit ones (SPNU194 p.17, p.18): the hardware
  truncates silently. "In memory addressing, offset is sufficient … only the offset will be
  considered" (p.17); the width of that offset is *not stated*.
- **XMC.** The PDL lists source and destination misalignment as interrupt causes
  (`infineon-cy-dma-h` L285-286; `infineon-cy-dmac-h` L215-218). On CAT1C, descriptors and
  buffers must be 32-byte aligned for cache maintenance (§1.8).
- **CY91460.** *Not stated.*

**Chaining and linked descriptors.**
- **XMC.**
  - A next-descriptor pointer is stored in the descriptor (AN234225 p.4, p.21).
  - A NULL next pointer stops "the chain execution" (p.17).
  - "It is possible to have circular lists" (p.3), and double buffering is shown (p.21). The
    CRC listing chains a descriptor to itself (p.33).
  - No ownership handshake is stated: the engine re-enters a circular list whatever the CPU has
    done.
- **TMS470R1x.**
  - With DCHN = 1, the next block takes the packet NCPACK names, and "If the CPS bit of the new
    control packet … is set to zero, the DMA controller waits until the application program
    sets it to one" (SPNU194 p.16, p.40).
  - The chain ends at a packet with DCHN = 0 (p.36, pp.40-41).
  - Whether a chain may cycle is *not stated*.
  - The CPS bit is a hardware ownership handshake.
- **CY91460.** No chaining is stated, only reload (AN205266 p.7, p.9).

### 1.5 Completion, interrupts and error reporting

**XMC.**
- **Interrupt granularity.** Element, X loop, descriptor or chain, set per descriptor
  (AN234225 p.10). The examples count:
  - for types 0 and 1, 16 and 2 outputs over the completion of descriptor 0 (p.11);
  - for types 2 and 3, 2 and 1 outputs over the completion of the two-descriptor list
    (pp.12-13).
- **Channel mask.** Each channel has an interrupt mask, which must be set
  (`infineon-cy-dma-h` L83-85).
- **DW cause enumeration** (`infineon-cy-dma-h` L279-290):
  - completion;
  - source bus error and destination bus error;
  - source misalignment and destination misalignment;
  - current descriptor pointer NULL;
  - active channel disabled;
  - descriptor bus error.

  `Cy_DMA_Channel_GetStatus` reads the cause (L1846-1851).
- **DMAC.** A bitmask with the same meanings in bits 0–7 (`infineon-cy-dmac-h` L209-224).
  - Bit 6 reads "The channel is disabled and the data transfer engine is busy".
  - Bit 5 reads "The channel is enabled and the current descriptor pointer is '0'".
- **Clearing.** Both engines clear by a write, followed by a "dummy reading [that] … provides a
  guarantee that interrupt is cleared at returning" (`infineon-cy-dma-h` L1870-1877;
  `infineon-cy-dmac-h` L1721-1729).
- **What the AN says about errors.** AN234225 describes no error cause. Its ISR treats any cause
  other than completion as `CY_ASSERT(false)` (p.34).

**CY91460.**
- **End codes in DSS** (AN205266 p.8):
  - 11 — normal end (EDIE);
  - 01 — address error (ERIE);
  - 10 — transfer stop request (ERIE);
  - DSS2 = 1 — "stopped temporarily" by DMAH, PAUS or an interrupt.
- **Channel enable.** DENB clears when the count is reached (p.6 fn 1). Whether DENB stays set
  under reload is *not stated*.
- **Clearing.** The ISR clears DSS by writing 0 (p.17).
- **Stop.** DEOTX stops a transfer from outside (p.11). A receive error on the UART raises a
  STOP request (p.2).

**TMS470R1x** (SPNU194):
- **Interrupts.** INTEN in each packet raises an interrupt when that packet's count reaches 0
  (p.14). IL sends each channel to interrupt line 0 or 1 (p.24).
- **Flags.**
  - IF[15:0] holds one flag per channel (p.26).
  - PINT is the OR of the IF flags (p.26).
  - TC is "set … when the transfer count … reaches zero for any control packet assigned to any
    channel", and software must clear it (p.26). TC is global, so it cannot tell which channel
    finished.
- **Offset registers.** DMAIO0 and DMAIO1 give the highest-priority pending channel's offset.
  Table 6's three printed rows show it as the channel number plus 1 (pp.27-28; TI-16). Reading
  one clears that channel's IF and deasserts the line (p.14, pp.26-28).
- **Packet status.** The controller clears the packet's CPS on completion (p.29, p.40) and the
  channel's DMAEN when no chain follows (p.36, p.41).
- **Errors.** The document defines no error condition, flag or interrupt. It refers only to
  "any DMA transfer error" in its recovery steps: set DMA_STOP, then clear CPS through DMACPSC
  (p.43).

### 1.6 Channel states and transitions

The registry normalizes every variant onto one state set. Each cell gives the documented
predicate, or *not stated*.

| BCIR state | XMC7000 DW / DMAC | CY91460 | TMS470R1x |
|---|---|---|---|
| OFF (block disabled) | DW_CTL / DMAC_CTL.ENABLED = 0, the reset value (AN234225 p.6); an enabled channel "is not operational until the DMA block is enabled" (`infineon-cy-dma-h` L81-82) | DMACR.DMAE = 0 (AN205266 p.8) | *not stated* per block; DMA_STOP or DMA_HALT hold every channel (SPNU194 pp.33-34) |
| IDLE (programmable) | CH_CTL.ENABLED = 0 (AN234225 p.13) | DENB = 0 (AN205266 p.6) | DMAEN = 0 (SPNU194 p.36), and the channel's packets have CPS = 0 (p.29) |
| ARMED (waiting for a request) | enabled, current pointer set (AN234225 p.7); trigger pending (block diagrams, pp.4-5) | DENB = 1, waiting for a request (AN205266 p.9) | DMAEN = 1 in request mode, waiting for a rising edge (SPNU194 p.40) |
| ACTIVE | DW: DW_STATUS.ACTIVE with CH_IDX (`Cy_DMA_GetActiveChannelIndex`); DMAC: the DMAC_ACTIVE bits (`Cy_DMAC_GetActiveChannel`) | transferring (AN205266 pp.9-10) | DMAAC.ACT_CHN = the channel (SPNU194 p.34) |
| WAIT (between quanta) | between triggers of types 0–2 (AN234225 p.10) | between blocks, or demand level inactive (AN205266 pp.9-10) | service-size slice ended, or the next chained packet has CPS = 0 (SPNU194 p.16, p.40) |
| HELD (paused or halted) | *not stated* | PAUS = 1, DMAH ≠ 0, hold cancelled by an interrupt, or BRQ; DSS2 = 1 (AN205266 p.6, p.8, pp.11-12) | DMA_HALT, including through HIRQ/HFIQ; debug suspend with SEN = 0 (SPNU194 p.24, pp.31-34) |
| STOPPING (stop requested) | disabling an active channel raises the cause "active channel disabled", whose DMAC text is "disabled and the data transfer engine is busy" (`infineon-cy-dmac-h` L221-222); the fate of in-flight data is *not stated* | DEOTX or a stop request (AN205266 p.2, p.11); granularity *not stated* | DMA_STOP = 1: "finishes the current transaction and updates" the packet (SPNU194 p.33) |
| DONE | completion cause at the descriptor's interrupt type; channel stays enabled or is disabled per CH_DISABLE (`infineon-cy-dma-h` L335-339) | DSS = 11, DENB cleared (AN205266 p.6 fn 1, p.8) | IF (if INTEN) and TC set; CPS cleared; DMAEN cleared unless chained (SPNU194 p.26, p.29, p.36) |
| FAULT | causes 2–8, or DMAC bits 1–7 (§1.5) | DSS = 01, or 10 (stop) (AN205266 p.8) | *not stated* (SPNU194 p.43) |

**Transitions.** Each is listed with what causes it, and the source. Where a variant has no
source, the edge does not exist for it.

| From → to | Cause |
|---|---|
| OFF → IDLE | block enable: XMC DW_CTL / DMAC_CTL (AN234225 p.6); CY91460 DMAE (AN205266 p.8) |
| IDLE → ARMED | channel enable: XMC CH_CTL.ENABLED (AN234225 p.13); CY91460 DENB (AN205266 p.6, p.16); TMS470 DMACCP.DMAEN, after packets and CPS (SPNU194 p.42) |
| ARMED → ACTIVE | a request or trigger (§1.3). A TMS470 channel in non-request mode goes straight to ACTIVE (SPNU194 p.40). A DMAC with prefetch reads its source on enable (`infineon-cy-dmac-h` L428-430) |
| ACTIVE → WAIT | the quantum is done: XMC trigger types 0–2 (AN234225 p.10); CY91460 step or block (AN205266 p.9); TMS470 CHSS slice (SPNU194 p.40) |
| WAIT → ACTIVE | the next request; for the TMS470, the next chained packet's CPS = 1 (SPNU194 p.16) |
| ACTIVE or WAIT → HELD → back | CY91460: PAUS, DMAH, interrupt hold-cancel, BRQ (AN205266 p.6, p.8, pp.11-12). TMS470: DMA_HALT, HIRQ/HFIQ, debug suspend (SPNU194 p.24, pp.31-34) |
| ACTIVE → DONE | count end: AN205266 pp.9-10; SPNU194 p.26, p.40; XMC descriptor end (AN234225 p.10) |
| DONE → IDLE | auto-disable: CY91460 DENB (AN205266 p.6 fn 1); TMS470 DMAEN when nothing is chained (SPNU194 p.36); XMC CH_DISABLE = 1 (`infineon-cy-dma-h` L335-339) |
| DONE → ARMED or ACTIVE | continuation: CY91460 reload (AN205266 p.9); TMS470 chain (SPNU194 p.16, p.40); XMC next descriptor (AN234225 p.17, p.21), or the same descriptor again through a circular list |
| any → STOPPING → IDLE | stop: CY91460 DEOTX (AN205266 p.11); TMS470 DMA_STOP (SPNU194 p.33); XMC disable (`infineon-cy-dma-h` L288). **When STOPPING ends is not stated for any variant** |
| ACTIVE → FAULT | the §1.5 codes |
| FAULT → IDLE | clear: XMC `Cy_DMA_Channel_ClearInterrupt` (L1870-1877); CY91460 DSS written 0 (AN205266 p.17); TMS470 recovery = STOP, then CPSC (SPNU194 p.43) |

**The observable "has stopped touching memory".**
- None of the three documents gives a register predicate that proves the engine has finished
  its in-flight transfer after a cancel:
  - TMS470's STOP "finishes the current transaction", but no flag reports that it has
    (SPNU194 p.33);
  - the XMC sources report that a disable happened mid-transfer, not when it ended;
  - the CY91460 granularity is unstated.
- This gap governs the cancellation half of D-OWN (§4) and is §8 Q3.

The PL330 model makes a contrast:
- DMAEND stalls until the channel's queued loads and stores have drained before the channel
  stops (`qemu-pl330-c` L643-663);
- DMAKILL discards them (L782-799).

Both are QEMU's behaviour, not ARM's, and the registry records them only in the reference row.

### 1.7 What software may touch while a channel is active

| | Rule or fact | Source |
|---|---|---|
| XMC DW/DMAC | "The DMA channel must be disabled during the configuration, and enabled after configuring the channel" | AN234225 p.13 |
| XMC DW/DMAC | A peripheral trigger is required *after* the channel is set up | AN234225 p.6 |
| XMC DW | Setting a descriptor writes CURR_PTR and clears X_IDX and Y_IDX | `infineon-cy-dma-h` L1612-1618 |
| XMC DMAC | Setting a descriptor does not touch the indices | `infineon-cy-dmac-h` L1405-1410 |
| XMC DMAC | With data prefetch, *source reads begin at enable*, before any trigger | `infineon-cy-dmac-h` L428-430 |
| XMC | Listings 1 and 3 rewrite the TX descriptor's source and re-enable the channel without showing it disabled. Whether it was disabled depends on the descriptor's channel state, which the text never gives | AN234225 p.17, p.26 |
| CY91460 | Both examples clear DENB before programming and set it last. No rule is stated | AN205266 p.16, p.19 |
| CY91460 | DMASA/DMADA "can't be read when the transfer is going on"; a read returns the address before the transfer | AN205266 p.8 |
| CY91460 | DMAH may be incremented and decremented by ISRs to hold the DMAC off | AN205266 p.11 |
| TMS470R1x | "The control packets have to be programmed before enabling the DMA channel" | SPNU194 p.13 |
| TMS470R1x | Required order: packets, then DMACC0/1, DMAGC and DMAGD, then CPS, then DMACCP. Steps 2 and 3 "should only be performed" after the earlier ones | SPNU194 p.42 |
| TMS470R1x | The controller writes the active packet back when it switches channel, on STOP or HALT, and at block end. A packet's registers in the command buffer are stale while it runs | SPNU194 p.13, pp.17-19 |
| TMS470R1x | Writing the next packet with CPS = 0 and then setting CPS, while the channel runs, is the documented chain handshake | SPNU194 p.16, p.40 |
| TMS470R1x | After an error: set DMA_STOP, then clear CPS through DMACPSC | SPNU194 p.43 |

The registry adopts no rule beyond these. The D-IDLE law (§4) turns the stated ones into
refusals and treats the unstated cases as refusals too. Writing DMACC or DMAGC while a channel
is active is one such case, *not stated*, and §8 Q4 asks about it.

### 1.8 Cache coherence

**XMC7000 (Cortex-M7).** The datasheets are named "32-bit Arm Cortex-M7 microcontroller"
(AN234225 p.40).

What the application note's code does:
- **CRC listing.** It disables the data cache outright: `SCB_DisableDCache(); // Disables D cache
  because DMA also reads descriptor in the SRAM` (AN234225 p.33).
- **DMAC listing.** Under `CY_CPU_CORTEX_M7 && ENABLE_CM7_DATA_CACHE`, it does two things
  (p.37):
  - it cleans the source buffer and the descriptor before enabling the channel and triggering;
  - it invalidates the destination after completion, before `memcmp` reads it.
- **UART listings.** They do no cache maintenance and do not say whether the cache is on
  (pp.16-30).

What the PDL requires, for both engines (`infineon-cy-dma-h` L79-80, L90-96;
`infineon-cy-dmac-h` L79-80, L90-96):
- "D cache needs to be cleaned before [the transfer / calling `Cy_DMAC_Channel_Enable`] and
  should be invalidated after".
- "Data Cache line is 32 bytes".
- Buffer and descriptor pointers must be aligned (32-byte, by `CY_ALIGN(32)`; one DW sentence
  says "32 bit", §7 LIB-2).
- "User needs to clean the following data elements from the cache and invalidate before
  accessing them: source and destination buffers and descriptor structure."

What CMSIS does (`cmsis-cachel1-armv7-h`):
- The Cortex-M7 line is "fixed to 32 bytes" (L45-47).
- `SCB_CleanDCache_by_Addr` and `SCB_InvalidateDCache_by_Addr` start from the 32-byte-aligned
  address and cover every line touched by [addr, addr+size), with a DSB before and a DSB+ISB
  after (L350-407).
- An invalidate therefore also discards dirty data in the parts of the first and last lines
  that lie outside the buffer.

The documents do not state:
- the write policy (write-back or write-through);
- whether any region is non-cacheable by default;
- whether the DMA is coherent with the cache. The cleaning requirement implies it is not.

**CY91460.** Nothing about a data cache is stated. AN205266 mentions only the CPU's prefetch
queue (p.5) and a chip-select prefetch enable bit (PFEN, p.15).

**TMS470R1x.** Nothing about any cache is stated.

The registry holds `dcache = None` for both, which is null and not "absent". D-COH therefore
refuses a hardware plan for them until a cited source settles it (§8 Q6). The simulator may
declare `dcache = False` as a SIMULATED assumption, labelled as such.

**PL330 (QEMU).** QEMU's DMA reads and writes go straight to the address space, and there is
no cache model (`qemu-pl330-c` L1104-1119, L1184, L1216). This is not evidence about any
silicon.

### 1.9 The 16550 side: RXRDY and TXRDY

These are the UART registry's facts. The DMA program reads them and never redefines them.
`uart_model.DMA_MODES` records mode 0 and mode 1's RXRDY, but not mode 1's TXRDY. Coupling
needs mode 1's TXRDY, so D3 asks the UART owner to add it with its page (§6, D3).

| Pin | Mode 0 (16450 mode, or FIFO mode with FCR3 = 0) | Mode 1 (FCR0 = 1, FCR3 = 1) | Source |
|---|---|---|---|
| RXRDY (active low) | active while at least 1 character is in the RX FIFO or holding register; inactive when none remains | active when the trigger level **or the timeout** is reached; inactive when none remains | PC16550D p.6 |
| TXRDY (active low) | active when the TX FIFO and holding register are empty; inactive after the first character is loaded | active when the TX FIFO is empty; inactive when it is "completely full" | PC16550D p.6 |

Further facts:
- In 16450 mode, "only DMA mode 0 is allowed" (PC16550D p.6).
- FCR3 switches the pins to mode 1 "if FCR0 = 1" (p.22). The UART's U1 law 3 already covers the
  write: on a placement with `fcr_gated`, an FCR store whose bit 0 is provably clear and whose
  other bits are not "programs nothing" (`uart_laws.py`, law 3).
- Timing (PC16550D p.9, p.10):
  - in FIFO mode the active RXRDY indication is delayed 3 RCLKs;
  - RXRDY goes inactive 290 ns after the read of RBR (tRXI);
  - TXRDY goes inactive 195 ns after the write (tWXI);
  - TXRDY goes active 8 BAUDOUT cycles after the start bit (tSXA).
- A placement whose `fcr_impl_mask` clears bit 3 has no FCR3. The UART blueprint's U8 adds two
  such placements, `lattice16550_lmmi` and `lattice16550_apb`, with mask 0xC6; they have no
  mode 1.

Three consequences follow from these facts and drive D-TRIG and D-HEAD:

1. **A mode-1 RXRDY assertion guarantees only one byte.** It may come from the timeout. A
   fixed burst of `trigger` bytes on each assertion can therefore over-read RBR.
2. **Either TXRDY assertion guarantees `fifo_depth` free slots in FIFO mode.** Both modes assert
   only when the TX FIFO is empty. In 16450 mode the guarantee is one slot.
3. **The pins are levels, and both are active low.**
   - A consumer that does one transfer per edge loses requests while the level stays asserted.
     CY91460 step mode ignores requests during a transfer (AN205266 p.9).
   - A rising-edge consumer, such as the TMS470 (SPNU194 p.40), would fire on *deassertion*
     unless the board inverts the signal.

---

## 2. The variant matrix

There is one row per controller, plus the PL330 reference row. The columns are every field the
registry parameterizes, grouped into nine tables. "n/s" means *not stated* in the cited sources;
the registry holds it as `None`. The PL330 row is QEMU's model (`qemu-pl330-c`). It is a
**simulator reference, not a hardware placement**: its cells describe what that file does, and
the ARM manual this session could not fetch may differ.

### 2.1 Identity and program storage

| Controller | Source and authority | Engine | Channels | Program storage | Program unit | Program slots | Alignment |
|---|---|---|---|---|---|---|---|
| XMC7000 DW | AN234225 (application note, "hint", p.42); PDL `cy_dma` 2.90 | one engine shared by all channels (p.3) | per datasheet (p.13); PDL `CY_DWx_CH_NR` (L384-387) | descriptors in SRAM (p.8; `infineon-cy-dma-h` L79) | 4–6 words by type (p.8; `infineon-cy-dma-c` L99-102) | unbounded (memory) | 32-byte on CAT1C (`infineon-cy-dma-h` L1606); next pointer ADDR [31:2] (L2108) |
| XMC7000 DMAC | AN234225; PDL `cy_dmac` 1.40 | one engine per channel (p.3) | per datasheet (p.13) | descriptors in SRAM (`infineon-cy-dmac-h` L79) | 4–8 words by type (L359-408) | unbounded | 32-byte on CAT1C (`infineon-cy-dmac-h` L1399) |
| CY91460 DMAC | AN205266 (application note, "reference purposes", p.23) | per-channel registers; bus shared with the CPU (p.5) | 5 (p.8) | channel registers (pp.6-8) | 4 registers | 1 per channel | n/a |
| TMS470R1x DMA | SPNU194 (reference guide) | one controller (p.11) | 16 or 8 (p.10) | command buffer 0xFFFFF800–9FF (p.13) | 4 words (p.13) | 32 or 16, shared by all channels (p.10, p.13) | 4-word; word writes only (p.13) |
| PL330 (QEMU) | `qemu-pl330-c` (a model) | microcoded: one manager and N channel threads, shared MFIFO, read and write queues (L229-276) | default 8 (L1649) | microcode at the channel PC (L1099-1122) | instructions of 1–6 bytes (L37, L1064-1089) | n/a | n/s |

### 2.2 Element width and counts

| Controller | Widths (bytes) | Separate src/dst size | X count | Y / 2D | Count encoding | Count 0 |
|---|---|---|---|---|---|---|
| XMC7000 DW | 1, 2, 4 (p.3; L311-316) | yes: per side "dataSize" or "word" (L328-332); semantics n/s | 1…256 (L252-255) | yes, 1…256 (L452-475) | count−1 (L1298-1299) | not encodable (L1298) |
| XMC7000 DMAC | 1, 2, 4 (`infineon-cy-dmac-h` L255-261) | yes (L434-439); ignored for memory copy and scatter | 1…65536; memory copy in bytes; scatter 1…32768 pairs (L192-197, L445-448) | yes, 1…65536 (L451) | count−1; scatter 2×pairs−1 (`infineon-cy-dmac-c` L314-320) | not encodable |
| CY91460 DMAC | 1, 2, 4; 11 disabled (p.7) | no (one WS) | BLK 1…16 (0 means 16) × DTC 16-bit (p.6); one request moves BLK, BLK×DTC or 1 (pp.9-10) | no | BLK: 0 means 16 (p.6 fn 4) | DTC = 0: n/s |
| TMS470R1x DMA | 1, 2, 4; 11 reserved (p.15) | no (one TRSIZE) | 0…65535 transactions (p.19) | no | plain | 0 means 0 transfers (p.19) |
| PL330 (QEMU) | 1 << CCR[3:1] and 1 << CCR[17:15] (L745, L919) | yes | burst CCR[7:4]+1 and CCR[21:18]+1 (L743, L918); loop counters lc0 and lc1 (L775-780) | by nested loops | burst n−1 | n/s |

### 2.3 Addressing

| Controller | Source step | Destination step | Range and unit | Decrement | Fixed address | Wrap / reload | Misalignment | Unreachable |
|---|---|---|---|---|---|---|---|---|
| XMC7000 DW | X and Y increments | X and Y increments | −2048…2047 (L257-260); unit n/s; Y composition n/s (L468-472) | signed field (L468-472) | increment 0 (p.15) | circular lists (p.3); none within a descriptor | interrupt causes 4 and 5 (L285-286) | n/s |
| XMC7000 DMAC | X and Y | X and Y | −32768…32767 (L199-202); unit n/s; Y composition n/s | signed | 0 | circular lists | interrupt bits 3 and 4 (L215-218) | n/s |
| CY91460 DMAC | SASZ 8-bit, with SADM (pp.7-8) | DASZ, with DADM | 0…255; bytes per the example (p.16) | yes (p.7) | SASZ/DASZ = 0 (p.19) | reload DTC, SA, DA (p.7) | n/s | the DMAC's own registers (p.4) |
| TMS470R1x DMA | constant or +TRSIZE (p.15) | constant or +TRSIZE; expansion-bus destination always +4 (p.15) | size only | no | INC = 0 | none | masked silently (p.17, p.18) | system-control registers, so the command buffer too (p.10, p.13) |
| PL330 (QEMU) | +size×burst if CCR[0] (L746, L751) | +size×burst if CCR[14] (L920, L925) | any, by DMAADDH/DMAADNH (L615-641) | DMAADNH (L638-641) | inc = 0 | by program | first beat shortened (L751, L1182) | n/s |

### 2.4 Chaining

| Controller | Mechanism | Termination | Ownership handshake | Circular |
|---|---|---|---|---|
| XMC7000 DW | next pointer in the descriptor (p.21) | NULL (p.17) | **none stated** | yes (p.3, p.21, p.33) |
| XMC7000 DMAC | next pointer (`infineon-cy-dmac-h` L44-51) | NULL (L452) | **none stated** | yes |
| CY91460 DMAC | none stated | n/a | n/a | reload only (p.9) |
| TMS470R1x DMA | DCHN + NCPACK (p.14, p.16) | a packet with DCHN = 0 (p.36, p.40) | **CPS**: the controller waits for CPS = 1 and clears it on completion (p.16, p.29, p.40) | n/s |
| PL330 (QEMU) | program control flow | DMAEND (L643-663) | n/a | by program |

### 2.5 Requests and triggers

| Controller | Sources | Software trigger | Sensitivity | Re-sample control | Quantum per request | Request during transfer | Pins / outputs |
|---|---|---|---|---|---|---|---|
| XMC7000 DW | hardware, software, tr_out via the trigger mux (p.4) | trigger mux (p.4); `Cy_DMA_Channel_SetSWTrigger` (L1998) | pulse (2 cycles, p.34); SCB level (p.17); otherwise n/s | retrigger immediately, 4 or 16 Clk_Slow, or reactivation (L319-325); Clk_Slow n/s | element, X loop, descriptor or chain (p.10) | n/s | tr_out types 0–3 (p.10) |
| XMC7000 DMAC | software, tr_out only (p.4, p.5) | trigger mux (p.37) | pulse (p.37) | same retrigger (`infineon-cy-dmac-h` L263-270) | same types | n/s | tr_out |
| CY91460 DMAC | peripheral interrupt via IS/EIS, STRG, DREQ on channels 0–3 (p.2, p.6) | STRG (p.6) | DREQ level or edge, at least 5 clocks (p.11); demand by level (p.10) | n/s | step 1, block BLK, burst BLK×DTC, demand 1 per sample (pp.9-10) | **ignored** in step mode (p.9) | DACKX, DEOP, DEOTX (p.11) |
| TMS470R1x DMA | DMA_REQ (request mode); routing n/s (p.25, p.40) | none (non-request mode runs at enable, p.40) | rising edge (p.40) | n/s | one transaction (p.25) | n/s | none |
| PL330 (QEMU) | peripheral lines modelled as "busy" levels (L1281-1289); events (L877-901) | DMAGO by debug registers (L1092-1097) | level (busy) | n/a | single or burst per DMAWFP (L1025-1041) | n/a | events and IRQs (L894-898) |

### 2.6 Arbitration

| Controller | Priority | Preemption | Bus sharing with the CPU | Halt and pause | Debug suspend |
|---|---|---|---|---|---|
| XMC7000 DW | 4 levels, PRIO [9:8] (p.3, p.7) | between single transfers when PREEMPTABLE (p.4) | n/s | n/s | n/s |
| XMC7000 DMAC | 4 levels (p.3) | none (p.4) | n/s | n/s | n/s |
| CY91460 DMAC | PM01: ch0 > ch1, or "alternating (ch1 > ch0)"; ch2–4 n/s (p.8) | n/s | hold request/acknowledge (p.11); CPU waits or runs from its prefetch queue (p.5) | DMAE, DMAH, PAUS, hold cancelled by interrupts, BRQ (p.6, p.8, pp.11-12) | n/s |
| TMS470R1x DMA | CHPR fixed or rotating (p.32); App. A says rotating (p.40) | slices of CHSS (1–16) transactions (pp.32-33) | BMSS 1:1 or 1:4 (p.32); concurrent unless contended (p.11) | DMA_HALT, DMA_STOP, HIRQ, HFIQ (pp.31-34) | SEN (p.24) |
| PL330 (QEMU) | one instruction per channel per cycle (L1167-1177) | n/a | untimed | n/a | debug interface (L1291-1335) |

### 2.7 Completion and errors

| Controller | Interrupt granularity | Lines | Observables | Clear | Errors | Auto-disable |
|---|---|---|---|---|---|---|
| XMC7000 DW | per descriptor: element, X loop, descriptor, chain (p.10) | per channel, e.g. `cpuss_interrupts_dw1_22` (p.16) | cause enum (L279-290); per-channel and block status (L692-740) | write, then a dummy read (L1870-1877) | bus (src, dst, descriptor), misalignment, NULL pointer, disabled while active (L283-289) | per descriptor, CH_DISABLE (L335-339) |
| XMC7000 DMAC | same (`infineon-cy-dmac-h` L246-253) | `cpuss_interrupts_dmac_0` (p.37) | interrupt bitmask (L209-224); current src, dst, indices (L1559-1645) | write mask, then a dummy read (L1721-1729) | the same set, as bits 1–7 | per descriptor |
| CY91460 DMAC | at end of count (pp.9-10) | one DMAC vector; the number is contradictory (§7, CY-11) | DSS (p.8); DENB (p.6 fn 1) | write DSS = 0 (p.17) | address error and stop request (p.8); ERIE/EDIE polarity contradictory (§7, CY-1) | DENB cleared at count (p.6 fn 1) |
| TMS470R1x DMA | per packet with INTEN (p.14) | 2 lines, IL per channel (p.24) | IF, PINT, TC (global), CPS, DMAEN, ACT_CHN (p.26, p.29, p.34, p.36) | read DMAIO, or write 0 to IF; clear TC (pp.26-28) | **none defined** (p.43) | DMAEN cleared if not chained (p.36) |
| PL330 (QEMU) | by program (DMASEV) | irq[event] and abort (L590-603, L894-898) | channel state and fault type (L1423-1454) | INTCLR (L1351-1360) | fault bits (L43-55); bus errors **never raised** (L1104-1119, L1184, L1216) | DMAEND (L643-663) |

### 2.8 Lifecycle

| Controller | Reprogramming rule | Programming order | Stop or abort | Observable "stopped" | Privilege |
|---|---|---|---|---|---|
| XMC7000 DW | disabled during configuration (p.13) | descriptor, channel, enable channel, enable block (p.6, p.15) | disable; cause 7 if active (L288) | **n/s** | n/s |
| XMC7000 DMAC | the same (p.13) | the same; clean caches before `Channel_Enable` (`infineon-cy-dmac-h` L79-80) | disable; bit 6 if the engine is busy (L221-222) | **n/s** | n/s |
| CY91460 DMAC | n/s; examples clear DENB first (p.16, p.19) | the example's order (p.16) | DEOTX, stop request (p.11, p.2) | DSS codes; granularity **n/s** | n/s |
| TMS470R1x DMA | packets before enable (p.13); order steps 1→2→3 (p.42) | p.42 | DMA_STOP (finishes the current transaction, p.33) | **n/s** (no flag) | RWP (p.14 ff.) |
| PL330 (QEMU) | DMAGO only on a stopped channel (L711-714) | by program | DMAKILL discards queued work (L782-799) | channel state stopped (L1433-1437) | ns checks (L679-682, L715-718) |

### 2.9 Platform: caches, buffering, special kinds

| Controller | Data cache | Line | Maintenance required | Bufferable writes | Prefetch | Special kinds | Worked examples |
|---|---|---|---|---|---|---|---|
| XMC7000 DW | CM7 D-cache (p.33, p.37, p.40) | 32 B (`infineon-cy-dma-h` L90; `cmsis-cachel1-armv7-h` L46) | clean source, destination and descriptor before; invalidate after (L79-80, L95-96) | channel B bit: a write "can complete without waiting for the destination to accept" (L484-485, `infineon-cy-dma-c` L279) | n/s | CRC (p.31) | trigger types (pp.10-13); UART TX/RX (pp.14-30); CRC (pp.31-34) |
| XMC7000 DMAC | CM7 D-cache | 32 B | clean before `Channel_Enable`; invalidate after (`infineon-cy-dmac-h` L79-80, L90-96) | B bit (L461-463) | DATA_PREFETCH (L428-430) | memory copy, scatter (L237-244) | memory copy (pp.35-37) |
| CY91460 DMAC | **n/s** | n/s | n/s | n/s | n/s | fly-by (p.10) | external bus (pp.13-17); UART echo (pp.18-19) |
| TMS470R1x DMA | **n/s** | n/s | n/s | n/s | n/s | none | memory to memory (pp.43-44) |
| PL330 (QEMU) | none modelled | n/a | n/a | n/a | n/a | DMASTZ, store zeros (L950-964) | none |

### 2.10 Hazards the laws exist for

These play the part of the UART blueprint's H-A…H-M. Each hazard names the law that catches it
(§4).

- **HD-A — a polarity the document contradicts.**
  - AN205266 p.7 prints ERIE and EDIE as "0 enabled / 1 disabled". Its two examples set
    EDIE = 1 for "IRQ when transfer finished" (p.16, p.19).
  - Any CY91460 program whose interrupt behaviour matters is undecidable until this is
    resolved.
  - Caught by D-DESC (the registry holds both polarities as null) and §8 Q2.
- **HD-B — silent truncation.**
  - A misaligned TMS470 address is masked rather than faulted (SPNU194 pp.17-18): a byte-offset
    word transfer silently moves different bytes.
  - Caught by D-WIDTH, which refuses before the hardware can truncate.
- **HD-C — a global flag read as a channel's completion.**
  - TMS470 TC is set by "any control packet assigned to any channel" (p.26).
  - Caught by D-REUSE: TC never counts as evidence for a channel.
- **HD-D — no stopped observable.**
  - No document says how to observe that a cancelled channel's in-flight unit has finished
    (§1.6).
  - Caught by D-OWN: a cancel needs the variant's documented predicate, or a declared bound.
    Null means refused.
- **HD-E — authority and missing bit positions.**
  - Two of the three primary sources are application notes. The DMAC's descriptor bit positions
    are only in device headers the session does not have.
  - Caught by D-DESC: facts carry their authority, and the encoder does not pack DMAC bits.
- **HD-F — a level producer meets an edge or fixed-burst consumer.**
  - RXRDY and TXRDY are levels (PC16550D p.6); CY91460 ignores requests during a step transfer
    (p.9); TMS470 fires on a rising edge (p.40).
  - The level also deasserts late: 290 ns after RBR is read and 195 ns after THR is written
    (PC16550D p.9, p.10).
  - Caught by D-TRIG.
- **HD-G — a circular list with no handshake.**
  - The XMC engine re-enters a circular list whether or not the CPU has drained the slot
    (p.3, p.21; no handshake is stated).
  - Ownership becomes a timing law. Caught by D-HEAD (ring form).
- **HD-H — halts that make service time unbounded.**
  - TMS470 HIRQ/HFIQ halt the DMA whenever the CPU enters IRQ or FIQ mode, until software
    clears DMA_HALT (pp.31-32). CY91460 interrupts above the hold-suppress level cancel
    transfers until the request is cleared (p.11).
  - Caught by D-HEAD, which needs a declared bound on these windows.
- **HD-I — descriptors and buffers in cached SRAM.**
  - Applies to the XMC CM7 with 32-byte lines (§1.8). Caught by D-COH.
- **HD-J — four count encodings.**
  - The encodings are count−1 (XMC), BLK 0 meaning 16 (CY91460), 0 meaning 0 (TMS470), and
    DTC = 0 unknown (CY91460).
  - Caught by D-COUNT: 0 is never emitted, and every encoding comes from the registry table.
- **HD-K — no hardware triggers on the XMC7000 DMAC.**
  - AN234225 p.4 and p.5 say so. Caught by D-TRIG: UART pacing on the DMAC is refused.
- **HD-L — dropped requests.**
  - CY91460 step mode ignores a request that arrives during a transfer (p.9). Caught by D-TRIG.
- **HD-M — a packet saved back on channel switch.**
  - TMS470 writes the running packet back on a switch, STOP, HALT or block end
    (p.13, pp.17-19). A CPU write to that packet races the write-back.
  - Caught by D-IDLE.
- **HD-N — reads at enable.**
  - With DATA_PREFETCH, the DMAC reads the source as soon as the channel is enabled
    (`infineon-cy-dmac-h` L428-430). Ownership begins at enable.
  - Caught by D-OWN. A port whose reads are destructive (RBR) never takes prefetch.
- **HD-O — bufferable completion.**
  - With the B bit set, "a write transaction can complete without waiting for the destination
    to accept" it (`infineon-cy-dma-h` L484-485).
  - Caught by D-REUSE: a buffer released on completion requires B = 0, the setting of the AN's
    own listing (AN234225 p.33).
- **HD-P — validation only by assertion.**
  - The PDL checks counts and increments only with `CY_ASSERT_L2/L3`
    (`infineon-cy-dma-c` L114-122, L147-149; `infineon-cy-dmac-c` L70-78). With assertions
    compiled out, `xCount − 1` and the increments are passed to `_VAL2FLD` unchecked. The macro
    is not in the fetched sources; if it shifts and masks, as its other uses imply, then a count
    of 0 is written as the machine value of 256.
  - Caught by D-COUNT: BCIR's encoder refuses, and never relies on a library's assertions.

---

## 3. How A2's descriptor ring maps onto each controller

### 3.1 What A2 emits today

`bcir/kbcir/dma.py` (L40-128) refuses three things:
- a view that `check_strided_view` rejects (its strides are ≥ 1, every extent fits the bank,
  the offset is aligned to the bank, and the native tile is honoured;
  `bcir/kbcir/device_manifest.py` L263-335);
- a shape or element-width mismatch;
- a same-bank overlap.

It then walks both views as contiguous byte runs and emits `Descriptor(src_off, dst_off,
nbytes)`:
- one descriptor per run;
- adjacent descriptors merged while both sides stay contiguous;
- a non-unit innermost stride broken into one descriptor per element ("scatter is scatter",
  L1-10).

`dma_cost` prices the program as `move_cost` plus `_DESC_SETUP = 16` per descriptor (L37,
L131-161). `_DESC_SETUP` is a modelled placeholder, not a measured value.

`dma_transfer_module` (L168-248) builds a planned module in four phases:
- phase 0 writes the ring;
- phase 1 unmasks `dma.done:<engine>` and performs the kick;
- event phase 10 carries the `mem.move.*` claim and the done flag;
- phase 2 reaps the flag inside a masked window.

### 3.2 The canonical program: one normal form between A2 and the formats

A2's `Descriptor` drops what every controller needs: the element width, the geometry, the
trigger, the completion granularity, and any notion of a port. The lowering therefore does
*not* take A2's flattened runs as input. It takes the **view pair**, builds one canonical
program, and keeps A2 as the **semantic oracle**:

```python
@dataclass(frozen=True)
class PortView:            # the peripheral endpoint A2 cannot spell (additive; StridedView unchanged)
    device: str            # e.g. "uart16550:<placement>"
    register: str          # e.g. "RBR" / "THR" — resolved through that device's registry
    lane_bytes: int        # the register lane width (uart_model placement reg_width)
    direction: str         # "read" (device produces) | "write" (device consumes)
    destructive_read: bool # RBR: True (a read pops the FIFO; uart_laws.READ_CLEAR)
    request: str           # the request signal, e.g. "RXRDY" / "TXRDY" (§1.9)

@dataclass(frozen=True)
class Endpoint:            # exactly one of view / port
    view: StridedView | None
    port: PortView | None

@dataclass(frozen=True)
class DmaDesc:             # one controller-neutral descriptor
    kind: str              # "single" | "1d" | "2d"
    width: int             # 1 | 2 | 4 bytes per element
    src: Endpoint; dst: Endpoint
    x_count: int; x_src_inc: int; x_dst_inc: int        # increments in ELEMENTS, signed
    y_count: int = 1; y_src_inc: int = 0; y_dst_inc: int = 0

@dataclass(frozen=True)
class DmaProgram:
    placement: str; channel: int; priority: int
    trigger: "TriggerBinding"        # source, sensitivity, quantum (§4 D-TRIG)
    completion: "CompletionSpec"     # granularity + observable (§4 D-REUSE)
    circular: bool
    descs: tuple                     # tuple[DmaDesc, ...]
```

Four rules govern it:

- **`fold(src, dst, placement)`.** This builds the smallest `DmaDesc` list the placement can
  express, directly from the views' shape and strides:
  - a dense pair becomes one 1D descriptor;
  - A2's one-run-per-row case becomes **one 2D descriptor** where the placement has Y
    (`x_count` = row elements, `y_count` = rows, Y increments = row strides). In the normal
    form, element (x, y) lies at base + (x·x_inc + y·y_inc)·width: the Y increment is measured
    from the row's first element. That is BCIR's definition of `DmaDesc`, not a hardware fact.
    An encoder must translate it into its format's own composition, which no fetched source
    states for the XMC (§1.4, §3.5);
  - a non-unit innermost stride becomes a 1D descriptor with that X increment, where the
    format's increment range allows it.

  The fold is legal only where A2 itself admits the pair: `dma_descriptors(src, dst, man)` must
  not raise. The A2 refusals are never bypassed.
- **`walk(program)`** yields the program's byte runs in transfer order.
- **The differential.** For every memory↔memory pair A2 admits,
  `coalesce(walk(fold(s, d, p))) == dma_descriptors(s, d, man)` must hold byte for byte. A
  lowering bug cannot hide behind a decoder that shares its misreading: A2 computes its runs
  independently.
- **Port endpoints.** For a pair with one port, the walk's memory side must equal A2's runs for
  that view alone. The port side must be the port's one address, repeated.

### 3.3 TMS470R1x control packets (bit-exact)

A `DmaDesc` of kind `1d` lowers to one control packet when all of these hold:
- `x_src_inc` and `x_dst_inc` are 0 or 1 element;
- `1 ≤ x_count ≤ 65535`;
- both addresses are aligned to the width.

The packet's fields (SPNU194 pp.14-19):

| Packet field | Value |
|---|---|
| TRSIZE | 00, 01 or 10 for widths 1, 2, 4 |
| SRCINC / DSTINC | 1 for a memory run, 0 for a port |
| SRCMOD / DSTMOD | the bank's module code from the placement's bank binding (p.16) |
| INTEN | set where the completion spec asks for this packet |
| DCHN / NCPACK | link to the next packet; DCHN = 0 on the last |
| DMASA / DMADA | the bank's base plus the offset (memory: "offset is sufficient", p.17) |
| DMATC | `x_count` |

The program is emitted in the order SPNU194 requires (p.42): the packets, then DMACC (IL, SEN,
RQEN), DMAGC and DMAGD, then CPS, then DMACCP (DMAEN, CCPACK).

**Golden worked example.** SPNU194 B.1 (pp.43-44): 256 words from 0x4000 (fine memory select 2)
to 0x8000 (fine memory select 3), on channel 0 with packet 0, interrupt line 1, non-request
mode, suspend enable, fixed priority, BMSS 1:1 and CHSS 1. It must encode to:

| Register | Value |
|---|---|
| DMAC | 0x0000CA64 |
| DMASA | 0x00004000 |
| DMADA | 0x00008000 |
| DMATC | 0x00000100 |
| DMACC0 | 0x0000000C |
| DMAGC | 0 |
| DMAGD | 0 |
| DMACPS | 0x00000001 |
| DMACCP0 | 0x00000040 |

That is the document's own table, decoded field by field and consistent with pp.14-16, 24, 32
and 36. It is pinned byte for byte.

**What the format cannot express, and the lowering's refusal** (message prefix `R12-dma:`):

- **A non-unit innermost stride.** SRCINC adds only TRSIZE (p.15). The fallback is one packet
  per element, which is legal only while the program's packets fit the packet budget assigned
  to the channel. The budget is at most 32 (or 16), shared by all channels (p.10, p.13). Past
  the budget the lowering reports, for example:
  `tms470r1x cannot express innermost stride 2 elements: 256 per-element packets exceed the
  channel's packet budget 8 (SPNU194 p.13)`.

  A software-refill ring is a different plan, through the CPS handshake. It is priced by
  D-PLAN, never substituted silently.
- **2D.** Lowered as one packet per row, under the same budget.
- **Decrement.** Not used by A2, so never needed.
- **Count 0.** Refused (D-COUNT).
- **A count above 65535.** Split across chained packets, under the budget.
- **Misalignment.** Refused (D-WIDTH; HD-B).
- **Widths other than 1, 2 or 4.** Refused.
- **SRCINC = 1 with an expansion-bus peripheral at either end.** Refused while TI-8 (§7) is
  unresolved.
- **Any address in the system-control space**, which includes the command buffer. Refused
  (p.10, p.13).
- **A memory offset wider than the module's decode width.** Refused while that width is null
  (p.17).

### 3.4 CY91460 channel registers (bit-exact, except two fields the document leaves open)

A `DmaDesc` of kind `1d` lowers to one channel programming when all of these hold:
- the step `x_*_inc × width` fits in 0…255 bytes, the reading of SASZ and DASZ in bytes given
  by the example (p.16);
- `x_count = BLK × DTC`, with `1 ≤ BLK ≤ 16` and `1 ≤ DTC ≤ 65535`.

The register fields (AN205266 pp.6-8):

| Register field | Value |
|---|---|
| WS | the width |
| SASZ / DASZ | the step in bytes; 0 for a port |
| SADM / DADM | 0 (increment) |
| BLK, DTC | chosen from the trigger quantum (below) |
| MOD | step (BLK = 1), block (BLK = quantum), burst (whole count per request) or demand (level; BLK ignored), per the trigger binding (pp.9-10) |
| TYPE | 00 (2-cycle) |
| DMASA / DMADA | the addresses |

The encoder emits in the example's order: DENB = 0 first and DENB = 1 last (p.16).

Choosing BLK and DTC: the trigger quantum fixes BLK in block mode. A count that has no
factorization under these bounds splits into several programmings.

**Open fields.**
- **ERIE/EDIE.** The encoder refuses to emit either until §8 Q2 is answered (HD-A):
  `cy91460 interrupt-enable polarity is contradicted by AN205266 (p.7 vs p.16, p.19); supply the
  hardware manual's polarity`.
- **IS/EIS.** The encoder emits only the codes the document states:
  - 01110 (DREQ high level) and 01111 (DREQ low level / falling edge) (p.10, p.16);
  - the UART2 receive example's IS = 0x14 with EIS = 0x3 (p.19), flagged by CY-9.

  Every other request source is refused as *not stated*.

**What the format cannot express.**
- **Chaining.** There is none (§2.4). An A2 program of several descriptors becomes a sequence of
  programmings, each re-armed by software after DENB clears (p.6 fn 1). D-PLAN prices that plan,
  with one completion interrupt per descriptor.
- **2D.** One programming per row.
- **A step above 255 bytes.** Refused: `cy91460 cannot express a 260-byte step (SASZ is 8 bits,
  AN205266 p.8)`.
- **Counts that do not factor.** Split.
- **The fly-by transfer type.** Never emitted; A2 has no such transfer.
- **Demand mode without an external-area endpoint.** Refused, since demand 2-cycle transfers
  need an external address (p.10).
- **DREQ on channel 4.** Refused (p.2).

**Golden examples.**
- **External bus** (pp.13-16). Words; SASZ = DASZ = 4; BLK = 4, DTC = 4; block mode; IS = 0xF.
  It is pinned on every field except EDIE.
- **UART echo** (pp.18-19). Bytes; SASZ = DASZ = 0; BLK = 1, DTC = 0x32; block mode; IS = 0x14
  with EIS = 0x3. It is pinned the same way.

### 3.5 XMC7000 DW and DMAC descriptor words

**DW (bit-exact)**, from `infineon-cy-dma-h` L389-414 and L2074-2109:
- The kinds `single`, `1d` and `2d` map to DESCR_TYPE 0, 1, 2.
- `x_count` and `y_count` fit 1…256 and are encoded count−1.
- The increments fit −2048…2047.
- The words follow the type's compaction (§1.2): the next pointer goes at 0x0C, 0x10 or 0x14.
- TR_IN_TYPE comes from the trigger quantum, and INTR_TYPE and TR_OUT_TYPE from the completion
  spec.
- CH_DISABLE is set on the last descriptor of a non-circular program. Circular programs leave
  it clear.
- SRC and DST transfer size are always "as dataSize" until §8 Q5 is answered.
- RETRIG comes from the trigger binding (D-TRIG).

**The increment unit is not stated** (§1.4). The DW encoder therefore emits non-zero
increments only for width 1, where elements and bytes coincide. That covers the AN's own byte
examples (AN234225 p.15, p.33) and every UART transfer. A wider element with a non-zero
increment is refused: `xmc7000_dw increment unit (elements or bytes) is not stated (AN234225,
cy_dma.h); supply the TRM's DESCR_X_CTL definition`.

**The A2 shape, recovered — once the Y composition is stated.** A2's one-run-per-row program,
of up to 256 rows of up to 256 elements, is **one** 2D descriptor in the normal form. A2 emits
one descriptor per row: 16 in `test_dma.py`'s gapped case. D-PLAN's prices must reflect that,
since A2's per-descriptor setup charge would otherwise overprice the controller that can
express the shape.

Encoding that 2D descriptor needs the XMC's Y composition, which is not stated (§1.4). Until
§8 Q5 answers it:
- the DW and DMAC encoders refuse a `2d` descriptor with `y_count` > 1:
  `<placement> Y-increment composition is not stated (cy_dma.h L468-472, AN234225 p.9); supply
  the TRM's 2D address rule`;
- the lowering falls back to a chain of 1D descriptors, one per row: A2's own shape, legal and
  encodable;
- the simulator's XMC profiles declare a composition as a SIMULATED parameter, and D4 runs the
  fold under both readings, so the switch is one registry fact when the answer comes.

**DMAC (word-level only).**
- The words follow `cy_stc_dmac_descriptor_t` and the per-type compacted layouts
  (`infineon-cy-dmac-h` L359-408), with the next pointer at the offsets of `infineon-cy-dmac-c`
  L331-336.
- The bit positions of the fields inside `ctl`, `xSize`, `xIncr`, `ySize` and `yIncr` are in
  the device headers (`DMAC_CH_V2_DESCR_*`), which were not fetched. The encoder therefore
  emits a **field-named, unpacked** descriptor and refuses to pack it:
  `xmc7000_dmac field positions are not in the cited sources (cy_dmac.h uses DMAC_CH_V2_DESCR_*
  from the device header)`.
- AN234225's DMAC 1D figure puts the next pointer at +0x10 with no increment word, where the
  PDL has 0x14. The registry records the disagreement (§7, AN-X2) and uses the PDL's layout,
  by explicit choice and citation, pending §8 Q5.

**What the XMC formats cannot express.**
- **DW.** Counts above 256 in one dimension; the lowering splits, and folds into Y only once
  the Y composition is stated.
- **Increments outside ±2048.** The lowering splits.
- **DMAC.** UART pacing (HD-K).
- **CRC, scatter and memory copy** have no A2 source. They are registry rows only.

### 3.6 The refusal table

| Construct (from A2 or the plan) | TMS470R1x | CY91460 | XMC DW | XMC DMAC |
|---|---|---|---|---|
| Unit-stride 1D run | packet | programming | 1D | 1D (unpacked) |
| Gappy outer stride (A2: one run per row) | packet per row, within budget | programming per row | **one 2D** once §8 Q5 states the Y composition; until then a 1D chain, one per row | the same (unpacked) |
| Non-unit innermost stride | packet per element, within budget | step ≤ 255 B | X increment (width 1 only) | X increment (unpacked) |
| Fixed port endpoint | INC = 0 | step 0 | increment 0 | increment 0 |
| Element width 8 (A2 allows any positive `elem_bytes`) | refused, or re-typed (D-WIDTH) | refused, or re-typed | refused, or re-typed | refused, or re-typed |
| Count 0 / count above the field | refused / split | refused / split | refused / split (into Y after §8 Q5) | refused / split |
| Circular ring | a finite chain, re-armed by software through CPS (priced); a cyclic NCPACK is n/s | reload of one buffer only | circular list (D-HEAD) | circular list |
| UART RXRDY/TXRDY pacing | routing n/s: **declared boundary** | DREQ demand mode (ch0–3) | trigger routing n/s: **declared boundary** | **refused** (no hardware triggers) |

A **declared boundary** is a placement capability the registry marks unsupported, with its
reason. It is not a law row that admits nothing, which L22 forbids. It reports
UNAVAILABLE-SKIPPED with its reason (L1), and it never fails silently.

### 3.7 What A2 must grow, additively

| Addition | Why | Non-disturbance |
|---|---|---|
| `PortView` | peripheral FIFOs are a fixed MMIO address, which `StridedView` cannot spell: strides ≥ 1, and a bank is required. A zero stride in memory would be the aliasing shape A2's overlap rule refuses, so ports stay a separate type | new type; `check_strided_view` unchanged |
| A bank binding: base address, and TMS470's module code, per placement | A2's `MemoryBank` has no address (`device_manifest.py` L51-60), so no absolute descriptor address can be computed | lives in the DMA registry; `DeviceManifest` and its digest unchanged |
| Element width carried into the lowering | `Descriptor` drops `elem_bytes` | comes from the views; `Descriptor` unchanged |
| Count splitting, Y folding | count fields are finite (§2.2) | lowering only |
| Circularity, and completion granularity | A2 has one completion per module | `DmaProgram` fields |
| Ownership and cache claims (`dma.reserve`, `dma.submit`, `dma.cancel`, `dma.stopped`, `dma.reap`, `dma.retire`, `dma.release`, `cache.clean`, `cache.invalidate`) | §4 D-OWN, D-COH | explicit op-string claims, as EV's `irq.mask`/`irq.unmask` are (`events.py` L41-65); vacuous where absent |
| `dma_transfer_module` emits the ownership claims; A2's `dma.reap` gains its `:<engine>` suffix | the canonical spelling. No test pins the bare `dma.reap` (`grep` over `bcir/`: only `dma.py` and `test_dma.py` use the builder) | `test_dma.py` keeps passing; one new negative is added (D1) |

---

## 4. The laws (D-laws)

**Homes and names.** Like U1, the laws do not mint a new R-number in the first pass. Each one
reports under the existing law it refines, with a message prefix:

| Prefix | Existing law | D-laws |
|---|---|---|
| `R5-dma:` | hazard legality | D-OWN, D-REUSE, D-COH |
| `R12-dma:` | lowering legality | D-DESC, D-WIDTH, D-COUNT |
| `R3-dma:` | device-access discipline | D-TRIG, D-IDLE |
| `R9-dma:` | plan legality | D-HEAD, D-PLAN |

They enter in two places:
- **Oracle diagnostics.** `bcir/kbcir/dma_own.py` holds the claim-graph laws. It imports no
  verifier, and is wired into the canonical `verify(module)` from the first slice. L2's S0-A
  lesson is that EV1–EV3 were vacuous until they were wired in. `bcir/frontends/devices/dma_laws.py`
  holds the program and binding laws.
- **MLIR clauses** under those R-laws, landed in D7.

Renumbering later is a rename, not a redesign (UART blueprint §4).

**Disciplines every law obeys.**
- **L1.** Every exit is PASS, FAIL, INVALID-VACUOUS or UNAVAILABLE-SKIPPED. A `DmaProgram` that
  names no buffer, or a module whose `dma.submit` claims name no resource, is INVALID-VACUOUS,
  never clean.
- **L2.** Each law is seen to fire, by its fault table, before it guards anything.
- **L11.** Each witness pins the whole message and the law that must produce it, per rail.
- **L14.** There is one predicate per repeated question. `bcir/frontends/devices/dma_predicates.py`
  holds the shared, total predicates: `guaranteed_items`, `stopped_predicate`, `aligned`,
  `encode_count` and `line_span`. The laws, the lowering and the simulator-profile generator
  all call them.
- **L18.** Each law module states its scope in its docstring.
- **L22.** Every row has a refused witness and an admitted witness. A placement where a row
  could admit nothing is a declared boundary, not a law row (§3.6).

Non-disturbance is measured, not assumed: every law is vacuous on any module or unit without
`dma.*` claims or a DMA format contract, over the whole corpus (§6, D1).

### D-OWN — buffer ownership (the central law)

**Statement.** Every buffer B a DMA program names has, for each generation `g`, exactly one
state. Only these transitions exist:

| From | By (an explicit claim) | To | Guard |
|---|---|---|---|
| AVAILABLE | `dma.reserve:B` | RESERVED | no live descriptor names any byte of B; `g := g+1` |
| RESERVED | CPU accesses to B (a TX fill) | RESERVED | — |
| RESERVED | `dma.submit:<ch>` naming B (the enable or kick) | DEVICE_OWNED | D-DESC, D-IDLE and D-COH hold at this claim |
| DEVICE_OWNED | `dma.cancel:<ch>` | DEVICE_OWNED (cancel requested) | a request, never proof the device stopped |
| DEVICE_OWNED | `dma.reap:<ch>`, observing the placement's completion predicate (D-REUSE) for B's generation | COMPLETING | — |
| DEVICE_OWNED, cancel requested | `dma.stopped:<ch>`, observing the placement's stopped predicate | COMPLETING (contents undefined) | the predicate is documented, or carries a declared bound in ticks and its source |
| COMPLETING | `cache.invalidate:B` and status checks | COMPLETING | — |
| COMPLETING | `dma.retire:B` | RETIRED | the completion status is recorded |
| RETIRED | `dma.release:B` | AVAILABLE | — |

While B is DEVICE_OWNED, four things are illegal: any CPU claim that reads or writes any byte of
B; a second `dma.submit` naming any byte of B, on any channel; `dma.reserve:B`; and
`dma.release:B`.

Three refinements follow from the sources:
- **A masked completion line exempts nothing.** EV3 treats an `irq.mask:<src>` window as
  exclusion (`bcir/kbcir/events.py` L16-21, L124-148). That holds for a CPU handler. It does not
  hold for a bus master: masking `dma.done` delays the *notification*, not the transfer.
- **Prefetch starts ownership at enable.** On a placement with prefetch, the enable claim *is*
  the submit, because source reads begin at enable (HD-N). A port with `destructive_read` is
  never a prefetch source.
- **A stale completion is refused.** A `dma.reap` whose generation is not B's live generation
  is the duplicate or missing event of the roadmap's edge corpus (§4.2).

**Why.**
- UART blueprint §12 (U11) states the law.
- §1.6 and HD-D show that no document provides the cancellation proof.
- A2's module carries no ownership claims today. EV3, as written, examines only resources an
  event phase *writes* (`events.py` L124-128), so a program store to `SRCB` between the kick and
  the reap of `dma_transfer_module` draws no EV message. Whether some other R-law catches it has
  not been run here. D1's first witness measures it before anything depends on it.

**Messages** (normative shapes):
- `R5-dma: claim <id> (<op>) writes <B>[<a>:<b>) while DEVICE_OWNED by <ch> since claim <k> (<op>); reap it first`
- `R5-dma: claim <id> (<op>) reads <B> inside irq.mask:<src>; masking the completion line does not stop the engine`
- `R5-dma: <B> submitted to <ch> at claim <id> while DEVICE_OWNED by <ch2> since claim <k>`
- `R5-dma: claim <id> (dma.release:<B>) after dma.cancel:<ch> with no dma.stopped:<ch>; a cancel is a request, not proof`
- `R5-dma: dma.stopped:<ch> on <placement> asserts no documented predicate (<source> reports none) and declares no bound`
- `R5-dma: dma.reap:<ch> observes generation <g> of <B>; its live generation is <g2> (a stale completion)`
- `R5-dma: <port> is a destructive-read source on a prefetching channel (<placement>, cy_dmac.h DATA_PREFETCH)`

**Witnesses** (`bcir/tests/test_dma_own.py`). These are refused:
- `test_cpu_store_to_the_source_between_kick_and_reap_is_refused`. This is A2's own module
  with one store added, the gap EV3 leaves.
- `test_reading_the_destination_inside_the_masked_window_is_refused`.
- `test_a_buffer_submitted_twice_is_refused`.
- `test_cancel_then_reuse_without_an_observed_stop_is_refused`.
- `test_the_vendor_echo_shape_is_refused_without_an_rx_reap`. This is AN234225 Listings 1 and 2:
  `buffer_a` is the RX destination and the TX source, with no proof that RX has released it
  (pp.16-21).
- `test_prefetch_from_rbr_is_refused`.
- `test_a_stale_generation_reap_is_refused`.

These are admitted:
- `test_the_canonical_cycle_is_clean`;
- `test_the_upgraded_a2_module_is_clean`;
- `test_ping_pong_done_right_is_clean`. This is AN234225 pp.21-26's double buffer, with each
  half reaped before TX reads it.

**Faults** (`dma-laws.json`):
- DL1: the CPU-access check skips reads.
- DL2: masked windows exempted.
- DL3: cancel treated as a stop.
- DL4: overlap tested on whole buffers rather than byte ranges.
- DL5: generation ignored.
- DL6: prefetch not modelled.

**Scope (L18).** Program phases are walked in module list order, the first-slice bound EV3
already records (`events.py` L26-31). A handler phase may touch B only after its own `dma.reap`
of B's channel.

### D-DESC — descriptor well-formedness

**Statement.** A lowered program is well formed when all of these hold:
1. Each descriptor's kind is one the placement supports (§2.2): no CRC on the DMAC, no memory
   copy or scatter on the DW.
2. Every word the kind uses is present, at the kind's compacted offsets (§1.2).
3. No reserved or "setting disabled" encoding is emitted: CY91460 TYPE, MOD or WS = 11
   (AN205266 p.7); TMS470 TRSIZE = 11 (SPNU194 p.15), and SRCMOD/DSTMOD 1101 or 1110 (p.16).
4. Every link either terminates (NULL, or DCHN = 0) or belongs to a program declared circular.
   Every link targets a descriptor of the same program, and no descriptor or packet is reachable
   from two channels. The TMS470 saves the running packet back on a switch (p.13), so a packet
   shared between channels is clobbered.
5. Placement-specific requirements hold:
   - TMS470: NCPACK is within the packet configuration (0–31 or 0–15).
   - XMC DW: the next pointer has ADDR [31:2] (`infineon-cy-dma-h` L2108).
   - XMC on CAT1C: every descriptor is 32-byte aligned (L1606).
6. Every field value comes from the registry's encoding tables. Any field the registry holds as
   null is refused: CY91460 ERIE/EDIE, and IS/EIS beyond the stated codes; XMC DMAC packing.
7. A DMA-programming unit's format contract attests the registry's placement (the RegMap
   contract pattern, §6 D0).

**Messages:**
- `R12-dma: <placement> descriptor <i> kind <k> is not supported (<source>)`
- `R12-dma: <placement> <field>=<v> is a reserved/disabled encoding (<source>)`
- `R12-dma: descriptor <i> links to <target> outside the program`
- `R12-dma: control packet <n> is reachable from channels <a> and <b> (SPNU194 p.13 saves it back on switch)`
- `R12-dma: tms470r1x NCPACK=<n> names no packet in the <k>-packet configuration (SPNU194 p.10, p.14)`
- `R12-dma: <placement> <field> is not stated in the cited sources; supply <what>`

**Witnesses.** Refused:
- a TMS470 DCHN = 1 packet whose NCPACK is 20, in the 16-packet configuration. NCPACK is a
  5-bit field (bits 28:24, p.14), so every value up to 31 encodes, and only the configuration
  (p.10) refuses 20;
- a TMS470 packet reachable from channels 0 and 1;
- CY91460 WS = 11;
- an XMC DW cycle without the circular flag;
- a DMAC CRC descriptor;
- a CY91460 EDIE emission (HD-A).

Admitted: the SPNU194 B.1 encoding (§3.3), pinned byte for byte.

**Faults:**
- DL7: kind support read from the engine family, not the placement. The DMAC CRC descriptor
  is admitted.
- DL8: the reserved-encoding table skipped. CY91460 WS = 11 is admitted.
- DL9: a link cycle taken as termination. The DW cycle without the circular flag is admitted.
- DL10: reachability computed one channel at a time. The packet shared by channels 0 and 1 is
  admitted.
- DL11: NCPACK checked against its field width, not the packet configuration. NCPACK 20 is
  admitted in the 16-packet configuration.
- DL12: a null registry field defaulted. The CY91460 EDIE emission is admitted.

### D-WIDTH — width and alignment

**Statement.**
1. **Width.** The element width is 1, 2 or 4 bytes (all four hardware placements, §2.2).
2. **Element alignment.** Every element address, on both sides, is aligned to the width. The
   TMS470 would mask the low bits silently (HD-B, SPNU194 pp.17-18). The XMC reports
   misalignment as an error (§1.5). The CY91460 does not say. All three are refused alike,
   statically.
3. **Equal sides.** Source and destination widths are equal ("as dataSize") until the XMC
   transfer-size widening semantics are stated (§8 Q5).
4. **Ports at their lane width.** A port endpoint is transferred at its lane width,
   `PortView.lane_bytes`. Take a 16550 placed on 4-byte lanes: the UART registry's
   `core16550_apb` and `axi16550` have `reg_width` 4. A byte-wide transfer to such a port is
   refused unless the controller placement states byte-lane semantics on word transfers, which
   none does (§1.4).
5. **Re-typing.** A memory↔memory transfer may be re-typed to a narrower legal width w that
   divides every run length and every alignment, which preserves the content. A2's
   `elem_bytes = 8` becomes w = 4, with the re-type recorded in the plan. A port is never
   re-typed.

**Messages:**
- `R12-dma: tms470r1x source 0x4002 is not 4-aligned for TRSIZE=32-bit; the controller would mask bits 1:0 (SPNU194 p.17)`
- `R12-dma: port <dev>.<reg> has 4-byte lanes; a 1-byte transfer is not stated for <placement>`

**Witnesses.** Refused:
- a TMS470 source address of 0x4002 under TRSIZE = 32-bit;
- the lane mismatch;
- an XMC DW 1D descriptor with byte data and a word destination transfer size. AN234225's CRC
  listing uses that setting (p.33); §5.6 pins it as an encoding only, and no A2 program lowers
  to it.

Admitted:
- an address of 0x4000 (B.1);
- byte transfers to `pc16550d` (lane 1);
- an `elem_bytes = 8` memory copy re-typed to 4.

**Faults:**
- DL13: alignment checked on the destination side only. The 0x4002 source is admitted.
- DL14: the port's lane width ignored. The 1-byte transfer to a 4-byte lane is admitted.
- DL15: the equal-sides clause skipped. The byte-data, word-destination descriptor is admitted.

### D-COUNT — counts and wrap

**Statement.**
1. **Never zero.** Every count is at least 1. No encoding of 0 is ever emitted:
   - XMC cannot encode it (count−1), and the PDL guards it only by assertion (HD-P);
   - the TMS470's table maps 0000h to a block size of "0" (SPNU194 p.19) and states no
     completion behaviour for it;
   - the CY91460's DTC = 0 is unstated (HD-J).
2. **Within the field.** Each count fits its field (§2.2). Larger counts are split, or folded
   into Y.
3. **Encodings from the registry.** Every encoding comes from the registry (count−1;
   CY91460 BLK 16 → 0).
4. **Within the view.** The address walk stays inside the endpoint's view, so inside its bank,
   for every element. The §3.2 differential against A2's runs is the proof.
5. **Wrap only as declared.** Wrapping happens only as declared circularity (D-HEAD ring form)
   or as CY91460 reload. Reload of the addresses requires DTCR = 1 (AN205266 p.7 fn 5).

**Messages:**
- `R12-dma: count 0 is never encoded (<placement>)`
- `R12-dma: x_count 300 exceeds <placement>'s field 1…256; split or fold into Y`
- `R12-dma: cy91460 SADR=1 requires DTCR=1 (AN205266 p.7 fn 5)`

**Witnesses.** Refused:
- a count of 0;
- a TMS470 count of 65536 emitted as one packet (TRANSCOUNT is 16 bits, p.19);
- CY91460 SADR without DTCR;
- a normal-form 2D walk of bytes, under §3.2's composition: X count 4 with increment +1, Y
  count 2 with increment −4, starting at offset 5 of an 8-byte view. Its first element (5) and
  its last (4) lie inside the view; its first row's last element (8) does not.

Admitted:
- 65536 elements split across two chained TMS470 packets;
- 512 DW elements split into two chained 1D descriptors of 256;
- the same 2D walk started at offset 4.

**Faults:**
- DL16: the zero check dropped. Count 0 is admitted, and count−1 encodes it as all ones.
- DL17: the field bound checked off by one. The 65536-element TMS470 packet is admitted.
- DL18: the SADR→DTCR dependency dropped. SADR without DTCR is admitted.
- DL19: the walk checked at its first and last elements only. The 2D walk from offset 5 is
  admitted.

### D-TRIG — trigger legality

**Statement**, for a binding of a producer, such as a `PortView`'s request signal, to a channel's
request input:
1. **Source.** The source is one the placement accepts. The XMC DMAC takes no hardware trigger
   (HD-K). CY91460 DREQ exists on channels 0–3 only (AN205266 p.2). TMS470 routing is not
   stated, which makes it a declared boundary.
2. **Sensitivity and polarity.** These must match. RXRDY and TXRDY are active-low levels
   (PC16550D p.6). The consumer is one of:
   - CY91460 demand mode, on the low level IS = 01111 (p.10);
   - an edge mode;
   - the TMS470's rising edge (SPNU194 p.40);
   - XMC, whose sensitivity is not stated beyond retrigger.

   A mismatch is refused unless the board placement declares an inversion, with its source.
3. **Quantum.** The quantum per request is at most `guaranteed_items(port, mode)`. That
   predicate, shared by the laws, the lowering and the simulator, returns:
   - for RX, 1 in mode 0 and 1 in mode 1, because the timeout asserts RXRDY with a single byte
     (§1.9);
   - for TX, `fifo_depth` with the FIFOs on (either mode), and 1 in 16450 mode (PC16550D p.6).
4. **Re-sampling.** A level consumer must not re-sample before the producer has deasserted:
   `T_resample ≥ L_deassert`, where
   - `L_deassert` is tRXI = 290 ns or tWXI = 195 ns (PC16550D pp.9-10), converted at the
     declared clock;
   - `T_resample` is the placement's. On the XMC that is RETRIG: 0, 4 or 16 Clk_Slow, or
     reactivation (`infineon-cy-dma-h` L319-325). The re-sample interval of the CY91460's
     demand mode, and the Clk_Slow frequency, are not stated, so these are null: refused for
     hardware, a declared parameter in the simulator.
5. **Edges against levels.** An edge-per-request consumer driven by a level producer is
   refused. Requests are lost while the level stays asserted: CY91460 ignores them in step mode
   (p.9), and the TMS470 needs a new rising edge (p.40).
6. **Mode 1.** Mode 1 requires FCR0 = 1 in the same FCR write (PC16550D p.22) and a placement
   whose `fcr_impl_mask` has bit 3. 16450 mode allows only mode 0 (p.6). The FCR0 half is the
   UART U1 law 3's (§1.9); D-TRIG takes that law's verdict and does not re-derive it (L14).
   D-TRIG checks the mask.
7. **Pulse widths.** CY91460 DREQ needs at least 5 system clocks (p.11). The XMC software
   trigger is two cycles (AN234225 p.34).
8. **Order.** The channel is enabled *before* the producer can assert. AN234225 states the rule
   for the XMC7000: "A peripheral trigger is required after setting the corresponding DMA
   channel" (p.6). The other two documents state no such order. BCIR applies the rule to every
   placement as a design choice, because an edge consumer enabled late misses the first edge,
   and a level consumer loses nothing by the order. The program must therefore enable the
   channel, then set FCR3 and enable the FIFOs.

**Messages:**
- `R3-dma: RXRDY guarantees 1 byte in mode 1 (timeout, PC16550D p.6); quantum 8 per request over-reads RBR`
- `R3-dma: xmc7000_dmac accepts no hardware trigger (AN234225 p.4, p.5)`
- `R3-dma: tms470r1x requests on a rising edge (SPNU194 p.40); RXRDY is active low: declare an inversion`
- `R3-dma: <placement> re-samples a level after <t> ticks < deassert latency <L> ticks`
- `R3-dma: <placement> step mode ignores requests during a transfer (AN205266 p.9); a level producer loses requests`
- `R3-dma: cy91460 DREQ exists on channels 0-3 (AN205266 p.2); channel <c> has none`
- `R3-dma: <uart placement> has no FCR3 (fcr_impl_mask <m>); RXRDY/TXRDY stay in mode 0`
- `R3-dma: cy91460 DREQ pulse of <n> system clocks is shorter than 5 (AN205266 p.11)`
- `R3-dma: claim <id> sets FCR3 before <ch> is enabled; the channel is set up before its trigger (AN234225 p.6)`

**Witnesses.** Refused:
- mode-1 RX with a quantum of 8;
- TX in 16450 mode with a quantum of 16;
- DMAC pacing;
- a TMS470 rising edge without inversion;
- CY91460 DREQ on channel 4;
- CY91460 step mode on RXRDY;
- CY91460 demand with a declared re-sample shorter than tRXI;
- mode 1 on a UART placement whose `fcr_impl_mask` clears bit 3. The UART blueprint's U8
  Lattice rows are the first such placements (mask 0xC6 = 0xCF without bits 0 and 3), so this
  witness lands with U8;
- a scripted DREQ pulse of 4 system clocks, in the simulator;
- enable after FCR3.

Admitted:
- CY91460 demand on IS = 01111, quantum 1, with a declared re-sample of at least L;
- TX mode 1 with a quantum of 16;
- a scripted DREQ pulse of 5 system clocks.

**Faults:**
- DL20: `guaranteed_items` returns the trigger level in mode 1. The mode-1 RX quantum of 8 is
  admitted.
- DL21: 16450 mode treated as FIFO mode. The TX quantum of 16 is admitted.
- DL22: the DMAC's hardware-trigger refusal dropped. DMAC pacing is admitted.
- DL23: the polarity check skipped. The TMS470 rising edge without an inversion is admitted.
- DL24: the DREQ channel range unchecked. DREQ on channel 4 is admitted.
- DL25: step mode treated as demand mode. Step mode on RXRDY is admitted.
- DL26: the re-sample inequality skipped. The short re-sample is admitted.
- DL27: `fcr_impl_mask` not consulted. Mode 1 without FCR3 is admitted.
- DL28: the pulse-width check skipped. The 4-clock pulse is admitted.
- DL29: the order check skipped. Enable after FCR3 is admitted.

### D-IDLE — reprogram only an idle or disabled channel

**Statement.**
1. **Idle first.** A write to a channel's configuration, or to any descriptor or packet
   reachable from it, needs the channel IDLE (§1.6). Idleness must be established by a
   dominating observation, or by the program's own disable plus the stop of anything in flight
   (D-OWN):
   - XMC: CH_CTL.ENABLED = 0 (AN234225 p.13);
   - TMS470: DMAEN = 0 and the packet's CPS = 0 (SPNU194 p.29, p.36);
   - CY91460: DENB = 0, the practice of both examples (AN205266 p.16, p.19).

   On the XMC DW, writing CURR_PTR also clears the indices (`infineon-cy-dma-h` L1612-1618),
   so it is legal only while disabled.
2. **The TMS470 refill exception.** Writing a packet that has CPS = 0 and is not the channel's
   current packet (CCPACK), then setting its CPS, is the documented chain refill
   (pp.16, 29, 40). It is admitted.
3. **TMS470 order.** Packet words, then DMACC, DMAGC and DMAGD, then CPS, then DMACCP (p.42).
   Error recovery is STOP, then CPSC (p.43).
4. **Unstated cases are refused.** DMACC or DMAGC written while any channel is active is not
   stated, and is refused (§8 Q4). TMS470 writes need privilege (RWP). The law records this, and
   checks it where the claim graph carries privilege (§8 Q4).

**Messages:**
- `R3-dma: claim <id> writes <ch>'s descriptor while the channel is enabled (AN234225 p.13)`
- `R3-dma: DMACPS set for packet <n> before its four words (SPNU194 p.42)`
- `R3-dma: DMAGC written while channel <c> is active: not stated in SPNU194`
- `R3-dma: packet <n> is <ch>'s current packet (CCPACK); the controller saves it back on a switch (SPNU194 p.13)`
- `R3-dma: claim <id> writes <ch>'s CURR_PTR while enabled; the write clears X_IDX and Y_IDX (cy_dma.h L1612-1618)`

**Witnesses.** Refused:
- AN234225 Listing 1/3's re-enable shape (pp.17, 26). It rewrites `TxDma_Descriptor_0` and
  calls `SetDescriptor` with no proven disable. It is admitted only when the previous descriptor
  completed with CH_DISABLE = 1 and that completion was reaped.
- CPS before the packet words;
- a refill of the channel's current packet;
- DMAGC written while channel 2 is active;
- an XMC CURR_PTR write while the channel is enabled.

Admitted: the TMS470 refill of a non-current packet with CPS = 0.

**Faults:**
- DL30: a disable accepted that does not dominate the write. The Listing 1/3 re-enable shape is
  admitted.
- DL31: the TMS470 order unchecked. CPS before the packet words is admitted.
- DL32: the refill exception widened to the current packet. The CCPACK refill is admitted.
- DL33: the global registers exempted. DMAGC during an active channel is admitted.
- DL34: CURR_PTR treated as a status register. The CURR_PTR write while enabled is admitted.

### D-REUSE — completion before reuse

**Statement.**
1. **The right observation.** The observation that moves B to COMPLETING must be the placement's
   **per-channel** completion predicate, at least at the granularity of the descriptor that
   owns B:
   - **XMC.** The cause COMPLETION, on a descriptor whose INTR_TYPE is at least DESCR. An
     element or X-loop completion releases nothing unless the plan tracks the indices.
   - **TMS470.** IF with INTEN = 1, or that packet's CPS = 0, or DMAEN = 0 when nothing is
     chained. **TC never counts** (HD-C).
   - **CY91460.** That channel's DSS = 11, or 01/10 with contents undefined.
2. **No bufferable release.** A program that releases B on completion requires B = 0 on the
   XMC channel (HD-O), unless the placement states that completion waits for acceptance. It
   states no such thing.
3. **Error completions.** An error completion allows release, never consumption of the data.
4. **Generation.** The completion must match B's live generation (D-OWN).

**Messages:**
- `R5-dma: TC is set by any packet on any channel (SPNU194 p.26); it does not show channel <c> completed`
- `R5-dma: <ch> interrupt type <t> completes elements, not descriptor <i>; <B> stays DEVICE_OWNED`
- `R5-dma: <ch> is bufferable; completion does not show the last write was accepted (cy_dma.h)`
- `R5-dma: claim <id> consumes <B> after an error completion (<code>); its contents are undefined`

**Witnesses.** Refused:
- TMS470 TC polling with two channels active;
- an XMC type-0 interrupt releasing a whole buffer;
- a bufferable release;
- a read of the destination's data after a CY91460 DSS = 01 completion.

Admitted:
- a CY91460 DSS = 11 reap;
- a TMS470 per-channel IF;
- an XMC DESCR completion with B = 0;
- a release, without a read, after DSS = 01.

**Faults:**
- DL35: TC accepted as a channel's completion. The TC poll is admitted.
- DL36: the interrupt type ignored. The type-0 release is admitted.
- DL37: the bufferable flag ignored. The bufferable release is admitted.
- DL38: an error completion treated as success. The read after DSS = 01 is admitted.

### D-COH — coherence

**Statement**, on a placement whose record has `dcache = True` (the XMC CM7 with its D-cache
enabled):
1. **Clean before.** Before the submit or enable claim, `cache.clean` covers every line of the
   descriptor words, every DMA-read buffer and every DMA-written buffer: "source and destination
   buffers and descriptor structure" (`infineon-cy-dma-h` L95-96; `infineon-cy-dmac-h` L95-96).
   For the DMAC the clean comes before `Channel_Enable` (L79-80).
2. **Invalidate after.** After the reap, and before the first CPU read of a DMA-written buffer,
   `cache.invalidate` covers its lines.
3. **Whole lines.** Every DMA buffer and descriptor occupies whole 32-byte lines: base 32-aligned
   and size padded. This is the PDL's requirement (L90-94), and it is a necessity: CMSIS's
   by-address operations act on every line the range touches (`cmsis-cachel1-armv7-h` L350-407),
   so an invalidate discards a neighbour's dirty bytes.
4. **No CPU writes in between.** No CPU write to a DMA-written buffer's lines occurs between its
   clean and its invalidate. A dirty line evicted after the DMA's write overwrites it. D-OWN
   already forbids the write; D-COH names the mechanism.
5. **The alternative.** A plan may instead run with the cache disabled (AN234225 p.33), or on a
   region declared non-cacheable with a cited MPU configuration, which is not stated anywhere
   (§8 Q6). Either makes the maintenance claims vacuous.
6. **Null cache facts.** A placement whose `dcache` is null (TMS470, CY91460) has its hardware
   plans refused: `cache facts are not stated (SPNU194 / AN205266)`. Simulator profiles declare
   `dcache = False` as SIMULATED.
7. **Barriers.** The maintenance claims carry CMSIS's barriers: a DSB before, and a DSB and ISB
   after (L365, L373-374).

**Messages:**
- `R5-dma: <B> is DMA-written but not cleaned before claim <id> (dma.submit) on <placement> (cy_dmac.h: clean source and destination buffers and descriptor)`
- `R5-dma: <B> [<a>,<b>) shares cache line 0x<l> with <other>; an invalidate would discard its data (cachel1_armv7.h)`
- `R5-dma: claim <id> reads DMA-written <B> before cache.invalidate`

**Witnesses.** Refused: AN234225 p.37's DMAC sequence as printed, on two counts.
- `dstData` is zeroed by the CPU, and only the source and descriptor are cleaned.
- Neither 8-byte array is 32-byte aligned.

Also refused:
- a CPU read of an RX buffer after its reap with no `cache.invalidate`;
- a `cache.clean` of length 0 offered as covering a buffer. CMSIS does nothing when the size is
  not positive (`if ( dsize > 0 )`, L361 and L391);
- a TMS470 hardware plan, whose `dcache` is null.

Admitted:
- the same sequence corrected;
- p.33's global disable;
- the TMS470 simulator profile, with `dcache = False` declared SIMULATED.

**Faults:**
- DL39: the clean checked on the source and descriptor only. The uncleaned `dstData` is
  admitted.
- DL40: coverage judged by the claim's presence, not its line span. The zero-length clean is
  admitted.
- DL41: line exclusivity checked at element alignment. The unaligned 8-byte arrays are
  admitted.
- DL42: the invalidate check skipped. The read without `cache.invalidate` is admitted.
- DL43: a null `dcache` read as False. The TMS470 hardware plan is admitted.

### D-PLAN — PIO versus DMA as a priced plan

**Statement.**
1. **Candidates.** The execution strategy is K_BCIR's choice among legal candidates:
   - polled PIO (UART U2);
   - IRQ-driven PIO (U10);
   - every DMA candidate: placement × mode × quantum × trigger type × descriptor shape × circular
     or not.
2. **Legality first.** Each candidate passes D-OWN…D-COH and D-HEAD before it is priced.
3. **Priced from a table with provenance.** Prices come from a cost table whose provenance is
   declared: SIMULATED from D2/D5, or MEASURED on silicon. The table covers descriptor setup,
   completion interrupts per KiB, bus transactions, CPU time and contention.
   - A2's `_DESC_SETUP = 16` is a modelled placeholder, and a decision priced with it is not
     certifiable.
   - A placement that folds A2's runs into 2D must be priced on its own descriptor count, not
     A2's.
4. **A certificate.** The decision carries the candidates, their verdicts, their prices with
   provenance, and the argmin.
5. **No fixed threshold.** The vendors' remarks are notes, not rules: AN205266 p.4 ("For a
   single transfer the DMA does not offer a real advantage") and AN234225 p.4 (DW bandwidth,
   DMAC latency).

**Messages:**
- `R9-dma: strategy decision has no certificate`
- `R9-dma: price table <t> has no provenance`
- `R9-dma: _DESC_SETUP placeholder used to certify a choice`
- `R9-dma: certificate chooses <c> at price <p>, but its own table prices <c2> at <p2> < <p>`
- `R9-dma: certificate prices candidate <c>, which <law> refuses; legality precedes cost`

**Witnesses.** D1 checks the certificate's shape. Refused:
- a strategy decision with no certificate;
- a certificate without provenance;
- a certificate priced with `_DESC_SETUP`;
- a certificate whose choice is not the argmin of its own prices;
- a certificate that prices a candidate a D-law refuses.

Admitted: a certificate with SIMULATED provenance, labelled so.

The selection itself lands with D6. Its witness is
`test_the_choice_follows_the_table_not_a_threshold`: two calibrated tables that should flip the
decision at different transfer sizes. A selector containing a constant byte threshold fails one
of them (D6's fault DP1).

**Faults:**
- DL44: a decision without a certificate admitted.
- DL45: the provenance field not required. The certificate without provenance is admitted.
- DL46: the placeholder accepted. The `_DESC_SETUP` certificate is admitted.
- DL47: the argmin not recomputed from the certificate's own prices. The mismatched choice is
  admitted.
- DL48: prices checked before verdicts. The priced illegal candidate is admitted.

### D-HEAD — FIFO headroom in DMA terms

**Statement (RX: UART to memory).**

  (D − L_assert) ≥ ⌈R × T_service_worst⌉ + N_inflight + M_safety

The terms:
- **D** is the placement's `fifo_depth`, from the UART registry.
- **L_assert** is the occupancy at assertion: the trigger byte value in mode 1, and 1 in mode 0
  (§1.9).
- **R** is the bytes per tick, from the line format and baud. The UART registry's character time
  gives it.
- **T_service_worst** is the sum of five terms:
  - *T_detect*: RXRDY's 3-RCLK assertion delay (PC16550D p.9), and the CY91460 DREQ minimum
    width of 5 clocks (AN205266 p.11).
  - *T_arb*: the competing channels' maximal quanta. On the TMS470 that is CHSS transactions per
    competing channel, plus BMSS's CPU turns (SPNU194 pp.32-33). On the CY91460 it is
    higher-priority block units. On the XMC DW it is one single transfer when preemptable
    (AN234225 p.4); the DMAC is never preempted.
  - *T_halt*: TMS470 HIRQ/HFIQ windows (pp.31-32); CY91460 DMAH, PAUS, hold-cancel and BRQ
    windows (AN205266 p.8, pp.11-12).
  - *T_fetch*: the descriptor fetch (XMC).
  - *T_xfer*: the transfer itself.
- **N_inflight** is the bytes that can land before the first slot frees and that no term of
  R × T counts: at least the receive-shift-register byte (UART blueprint §1.2's overrun rule).
- **M_safety** is a declared margin.

Every term is a number with provenance. A single null term refuses the plan. Two examples:
- on the CY91460 profile, T_halt with interrupt hold-cancel in force and no declared bound on
  the ISR (AN205266 p.11);
- on any TMS470 binding, T_halt with HIRQ = 1 and no declared bound on IRQ duration.

**The sustained form:** quantum ÷ cycle time ≥ R.

**TX.** Overflow is D-TRIG's business (quantum ≤ guaranteed free slots). Underrun loses no data.
It is priced by D-PLAN, never a legality condition.

**The ring form.** A circular list with no handshake (HD-G) must satisfy

  T_consume_worst ≤ (N_slots − 1) × T_fill_min, where T_fill_min = slot_bytes × char_time_min.

With the TMS470 CPS handshake, the controller waits instead (p.16). The FIFO inequality must
then also hold with T_halt including the CPU's refill latency.

**Messages:**
- `R9-dma: RX headroom 2 bytes < ceil(R*T)=3 + N_inflight 1 + M 1 at trigger 14 (<placement>)`
- `R9-dma: T_halt is unbounded (interrupts above the hold-suppress level cancel transfers until cleared, AN205266 p.11); declare a bound`
- `R9-dma: T_halt is unbounded (HIRQ=1 halts DMA on every IRQ entry, SPNU194 p.31); declare a bound`
- `R9-dma: ring of 2 slots: T_consume_worst <t> > (N-1)*T_fill_min <f>`

**Witnesses** (on the CY91460 profile, the supported UART binding, §3.6). Refused:
- trigger 14, mode 1, 115200 baud, with hold-cancel in force and no ISR bound;
- the same with a declared bound, when the inequality fails by one byte;
- a 2-slot ring whose consumer takes longer than one slot's fill time, and less than two.

Admitted:
- the same RX at trigger 8, with a declared bound that satisfies the inequality;
- the same consumer on a 4-slot ring.

**Faults:**
- DL49: a null term read as 0. The plan without an ISR bound is admitted.
- DL50: N_inflight omitted. The plan that fails by one byte is admitted.
- DL51: L_assert read as 1 in mode 1. The trigger-14 plan is admitted.
- DL52: the ring form computed with N_slots, not N_slots − 1. The 2-slot ring is admitted.

---

## 5. The simulated controller's behavioral contract

### 5.1 Shape and determinism

The simulator is `runtime/c/simdma.c` and `simdma.h`. It is a host tool in the
`sim16550.c`/`bcir_train.c` posture: state is caller-owned, it uses libc only, and it is fully
deterministic. It has no wall clock, no threads, and no randomness. Fault injection is scripted,
by tick and address.

There is one simulator. Each placement is a **profile**, a data table generated by D0 from the
registry (`runtime/c/simdma_profiles.h`), never an `#ifdef` fork. A profile field the registry
holds as null is a profile field with no value. A scenario that needs it must supply a declared,
SIMULATED value, or the simulator refuses it with `SIMDMA_UNSTATED` and the field's name.

**Time.** One tick is one controller clock. `simdma_tick(dev, n)` advances by n ticks. Each tick:
1. sample every request input (levels), and latch edges per channel sensitivity;
2. apply holds and halts;
3. arbitrate;
4. advance each engine by at most one bus transaction.

The engines are DW (one shared engine), DMAC (one per channel), CY91460 and TMS470 (one each).
Whether the DMAC's per-channel engines share one master port is not stated. The profile
serializes them, as a declared SIMULATED assumption.

Costs are profile parameters, SIMULATED and never hardware facts: `t_rd`, `t_wr`, and
`t_desc_word` for each descriptor word fetched (XMC). An engine's transaction is a read followed
by a write, which is AN205266's "2-cycle" transfer (p.10).

### 5.2 Per-variant request semantics (each line is a fidelity pin)

**XMC DW and DMAC.**
- A trigger starts the TR_IN_TYPE quantum: an element, an X loop, a descriptor, or the chain
  (AN234225 p.10).
- After a quantum, RETRIG decides the re-sample: immediately, after 4 or 16 Clk_Slow ticks
  (Clk_Slow ratio a profile parameter), or after the trigger has deactivated and reactivated
  (`infineon-cy-dma-h` L319-325).
- On completing a descriptor, the channel disables itself if CH_DISABLE is set (L335-339), and
  otherwise follows the next pointer. A NULL next pointer stops the chain (AN234225 p.17).
- Preemption: a preemptable DW channel yields between single transfers (p.4); the DMAC never
  yields (p.4).
- With prefetch, the DMAC reads the source at enable (`infineon-cy-dmac-h` L428-430).

**CY91460** (AN205266):
- Modes follow pp.9-10:
  - step moves 1 unit per request;
  - block moves BLK units;
  - burst moves BLK × DTC units;
  - demand moves 1 unit per tick that samples the active level.
- A request that arrives during a step transfer is **dropped** and counted (p.9).
- A DREQ pulse shorter than 5 ticks is recorded as a violation, and is not served (p.11).
- At count end, DSS = 11 and DENB clears (p.6 fn 1). With reload, the channel waits for the next
  request (p.9).
- PAUS and DMAH hold, and set DSS2 (p.8). An injected "high interrupt" holds at the next block
  boundary (p.11).

**TMS470R1x** (SPNU194):
- In request mode, a rising edge moves one transaction (p.25, p.40). Non-request mode runs at
  enable (p.40).
- Arbitration follows CHPR, with CHSS-transaction slices (pp.32-33). BMSS inserts CPU turns,
  as a CPU-demand script supplied by the scenario (p.32).
- The chain waits on CPS = 0 (p.16).
- STOP and HALT finish the current transaction, write the packet back, and hold (p.33).
- HIRQ and HFIQ are scenario inputs that set HALT (pp.31-32).
- At packet end: TC is set; IF is set if INTEN; CPS clears; DMAEN clears when nothing is
  chained (p.26, p.29, p.36).

### 5.3 Completion, errors, and the behaviours the documents leave open

- **Completion** sets exactly the observables of §2.7 for the profile.
- **XMC errors.** Profile semantics exist for the XMC causes (§1.5). The scenario injects a bus
  error by address; the simulator models misalignment, a NULL pointer, and a channel disabled
  while active.
- **CY91460 errors.** DSS = 01 is raised when the address walk leaves the 32-bit space. That is
  a declared SIMULATED assumption: the source says only "Address error (underflow/overflow)"
  (§1.4). DSS = 10 is raised on DEOTX or a scripted stop request.
- **TMS470 errors.** No semantics are stated (p.43). An injected bus error in the TMS470 profile
  returns `SIMDMA_UNSTATED`. The simulator never invents an error.
- **Outcome selectors.** Where a behaviour is not stated, the simulator exposes an explicit,
  scripted selector rather than choosing: the fate of the in-flight element on an XMC disable;
  the granularity of a CY91460 DEOTX stop; a TMS470 count of 0; whether CY91460 DENB stays set
  under reload; the XMC Y-increment composition (from the row's first element, or after its
  last X step, §1.4). **The laws must hold for every outcome the sources permit.** D3's
  cancellation tests run both selector values, and D4 runs every 2D program under both
  compositions. This is how a gap is made testable without guessing it.

### 5.4 Memory and the CM7 cache model

**Memory.** The simulator's RAM is a byte array; the bank map comes from the placement's bank
binding. Ports are callbacks into the peripheral simulator, by offset.

**The XMC cache.** For XMC profiles with `dcache` declared on, a 32-byte-line, write-back data
cache model sits in front of the CPU's accesses. The DMA reads and writes backing memory
directly: not coherent, as the sources' cleaning requirement implies.

`clean`, `invalidate` and `clean_invalidate` act by line, exactly as CMSIS's by-address
operations do (`cmsis-cachel1-armv7-h` L350-437). Eviction is scripted, as "evict line X at
tick T", never by policy. D-COH's violations therefore become **observable corruption**:
- AN234225 p.37's unclean `dstData` line, evicted after the DMAC's write, overwrites it;
- an invalidate over a line shared with a dirty neighbour loses the neighbour's bytes.

### 5.5 Coupling to the 16550 simulator (UART U2's `sim16550`)

**The coupled harness.** `runtime/c/uart_dma_main.c` owns one `sim16550` and one `simdma`, with
one time base.
- `ticks_per_char`, an integer of at least 1, converts the UART simulator's character ticks
  (`sim_tick(dev, n_char_times)`, UART blueprint §5 U2) into DMA ticks.
- RX bytes arrive only at character-tick boundaries, from the scenario script
  (`sim_inject(dev, byte, err_bits)`).

**Request signals.** D3 adds `sim_rxrdy(dev)` and `sim_txrdy(dev)` to `sim16550`, coordinated
with the UART owner. They are derived exactly from §1.9:

| Signal | Mode 0 | Mode 1 |
|---|---|---|
| RXRDY | active while RX FIFO + RBR ≥ 1; inactive at empty | active from trigger *or* timeout; inactive at empty |
| TXRDY | active while TX FIFO + THR empty; inactive after the first load | active when TX FIFO empty; inactive when full |

16450 mode forces mode 0 (PC16550D p.6). The pins are active low: the harness converts them to
"asserted" booleans, and applies a polarity inversion only if the board placement declares one.

**Latencies** are converted at the profile's declared clock:
- assertion is delayed `L_assert_rx` ticks, the 3-RCLK RXRDY delay (PC16550D p.9);
- deassertion follows a read of RBR by `L_deassert_rx` ticks (tRXI = 290 ns) and a write of THR
  by `L_deassert_tx` ticks (tWXI = 195 ns) (PC16550D pp.9-10).

**Transfers.** The DMA reads RBR and writes THR through `sim_read`/`sim_write` at the placement's
offsets, so the UART simulator's register effects apply unchanged: a read of RBR pops the FIFO;
LSR, PE, FE and BI travel with each byte. The harness records two **violation events** in the
transcript, never silently:
- **Over-read:** RBR read while the RX FIFO is empty. The UART simulator's own contract decides
  what the read returns; the harness flags the read whatever it returns.
- **Over-write:** THR written while the TX FIFO is full.

Overrun is the UART simulator's own OE (§1.2's rule: the shift-register byte is discarded).

**Mode-specific programs the coupling exercises:**
- **RX, mode 0 or 1.** Level-sensitive, one element per request, with re-sample ≥ L. On the
  CY91460 that is demand mode; on the XMC DW it is trigger type 0 with RETRIG. The circular RX
  ring is drained by the reap/retire/release cycle.
- **TX, mode 0 or 1.** Quantum ≤ 16 per assertion, with FIFOs on.
- **The refusal scenarios.** Each D-TRIG and D-HEAD witness is replayed through the simulator:
  - mode-1 RX with a burst equal to the trigger, *with a timeout-only arrival pattern*;
  - TX bursts in 16450 mode;
  - edge-consumer step mode on a held RXRDY.

  Each shows the predicted defect (over-read, over-write, lost request or overrun) at the
  predicted tick. Every law is thereby **proven against the behaviour it predicts**, not only
  against its own text.

### 5.6 The documents' worked examples as fidelity pins

| Pin | Source | Expected |
|---|---|---|
| TMS470 B.1 memory copy | SPNU194 pp.43-44 | 256 words 0x4000→0x8000; TC set, IF0 set (INTEN = 1, line 1); CPS0 clears; DMAEN0 clears |
| CY91460 external bus | AN205266 pp.13-16 | 4 falling DREQ edges × 4 words = 16 words; DSS = 11 after the 4th; DENB clears; the completion interrupt is the profile's declared EDIE reading (HD-A, flagged) |
| CY91460 UART echo | AN205266 pp.18-19 | 50 request events × 1 byte, RDR→TDR; DSS = 11 after the 50th; a scripted receive error raises the stop request (DSS = 10). The link from p.2 "STOP request … receive error" to p.8 "10 Transfer stop request" is the registry's *reading*, recorded as such |
| XMC trigger types | AN234225 pp.10-13 | descriptor 0 an 8 × 2 2D descriptor, as derived from p.11's counts; descriptor 1 the same, as a declared test assumption. Type 0: 16 triggers and 16 outputs complete descriptor 0. Type 1: 2 and 2. Type 2: 2 triggers and 2 outputs complete the list. Type 3: 1 and 1 |
| XMC DW CRC descriptor config | AN234225 pp.32-33 | the descriptor words encode per `infineon-cy-dma-h` L389-414: retrigger 4 cycles, interrupt and tr_out X loop, tr_in descriptor, byte data, destination transfer size word, CRC, X count 5 (machine 4), source increment 1, next pointer self. The encoding is pinned; CRC computation is out of scope |
| XMC DMAC memory copy | AN234225 p.37 | 8 bytes, software trigger, completion bit 0; with the cache model on, the corrected D-COH sequence reads correctly, and the printed one fails under a scripted eviction |
| PL330 | — | **no fidelity target.** QEMU's behaviours are reference points for the state set (§1.6) only |

### 5.7 What the simulator refuses

The simulator refuses rather than run:
- reserved or disabled encodings (D-DESC);
- a null profile field a scenario needs, without a declared value;
- a DREQ on channel 4;
- a DMAC hardware trigger;
- an injected TMS470 bus error;
- a CY91460 EDIE or ERIE with no declared polarity.

Each refusal is a status code with a name, never a crash (L1).

---

## 6. The build slices (a proposal: D0–D7, not milestones)

> **Status: proposal.** None of these slices is scheduled, and none has a checklist entry. Each
> one would be one PR-sized, independently gated change, landed in order. As U0 and U1 did, a
> slice may land on the oracle rail first, with its law-rail half stated as pending. **No slice
> reaches the roadmap's hardware gate (§5.1, D7).** That gate needs a rig and the TRMs (§8), and
> each slice's labels must name the highest gate it has completed.
>
> **The gates every slice runs** (AGENTS.md; `CONTRIBUTING.md`):
> - the touched tests;
> - `python -m bcir.tests.run_all --tier quick -j 2`;
> - `bash tools/c/check_runtime.sh` when `runtime/c` changes;
> - `tools/build/manifest.py --check` when a C source is added;
> - `bash tools/wsl/check_passes.sh`, or the `mlir` preset's `ctest -L mlir`, when `mlir/`
>   changes, on a coherent LLVM/MLIR 23 only;
> - `python tools/docs/check_links.py`;
> - `python tools/docs/gen_status.py`, regenerated **last**;
> - `git diff --check`;
> - `bash tools/local/check_latest.sh`, for the Clang/LLVM 23, Node 24 and Python 3.15 judgement.
>
> Heavy gates run serially, with at most two workers. Every new test module is registered in
> `bcir/tests/run_all.py::_MODULES`. Every fault table gets its row in
> `tools/testing/faults/README.md`, and is seen RED before GREEN (L2). Every C source enters
> `runtime/manifest.json`.

### D0 — The DMA registry

**Goal.** One normative, capability-parameterized model:
- the four hardware placements, plus the PL330 reference row;
- every fact cited, with its authority;
- generated views, and a format contract;
- the §7 notes recorded as provenance, never as facts.

The slice changes no existing module.

**Files.**
- `bcir/frontends/devices/dma_model.py`: the registry and its generators. Stdlib only, and
  deterministic.
- `bcir/frontends/devices/dma_predicates.py`: the shared, total predicates of §4 (L14).
- `bcir/frontends/devices/fixtures/dma_<placement>.h`: golden headers, shipped as package data
  and read through `importlib.resources`, as the UART fixtures are.
- `runtime/c/dma_formats.h` (encodings) and `runtime/c/simdma_profiles.h` (simulator profiles):
  generated, and checked by `--check`.
- `bcir/tests/test_dma_model.py`: registered.
- `tools/testing/faults/dma-model.json`.
- *Coordinated, optional:* `bcir/frontends/devices/provenance.py`. It would hoist `DocNote` and
  `Erratum` out of `uart_model.py`, which would re-export them. Two driver classes now use them,
  which meets the roadmap's §3.3 condition 1 for a shared primitive. This needs the UART owner's
  agreement. Otherwise D0 imports `uart_model.DocNote`, which needs U0 on `main` first (§8 Q11).

**Design** (normative):

```python
@dataclass(frozen=True)
class Fact:                 # one cited value; value None == not stated (null, never zero)
    value: object
    sources: tuple          # ("SPNU194 p.19", "infineon-cy-dma-h L252-255", ...)
    authority: str          # "reference-guide" | "application-note" | "vendor-library" | "simulator-model"
    note: str = ""

@dataclass(frozen=True)
class DmaField:             # one bit field of a descriptor word / packet word / channel register
    word: str; hi: int; lo: int; name: str
    encodings: tuple        # ((value, meaning, legal: bool), ...)  -- reserved/disabled marked illegal
    source: str

@dataclass(frozen=True)
class DescriptorFormat:
    name: str               # "tms470_packet" | "cy91460_channel" | "xmc_dw" | "xmc_dmac"
    words_by_kind: dict     # kind -> tuple of word names (compaction, §1.2)
    next_offset_by_kind: dict
    fields: tuple           # DmaField, ...
    packable: bool          # False for xmc_dmac: field positions not in the cited sources

@dataclass(frozen=True)
class BankBinding:          # what A2's MemoryBank lacks (§3.7)
    bank: str; base: int | None; module_code: int | None

@dataclass(frozen=True)
class DmaPlacement:         # one row of §2: every column is a Fact
    name: str; role: str    # "hardware" | "reference" (PL330)
    fmt: DescriptorFormat
    # ...one Fact per §2.1-§2.9 column (engine, channels, widths, x_count, y_count, increments,
    #    increment_unit, y_composition, decrement, fixed, wrap, chaining, handshake, circular, sources,
    #    sensitivity, resample, quantum, request_while_busy, priority, preemption, halts,
    #    granularity, observables, errors, auto_disable, reprogram_rule, order, stop,
    #    stopped_observable, privilege, dcache, line_bytes, maintenance, bufferable,
    #    prefetch, special_kinds) ...
```

The registry exports:
- `PLACEMENTS`;
- `DOC_NOTES`, the §7 notes, each a `DocNote` with `where`, `printed`, `corrected` and `note`;
- `PROVENANCE`, a tuple of (fact, sources, reading), as in `uart_model.PROVENANCE`.

The views:
- `emit_header(p)`;
- `emit_formats()`;
- `emit_sim_profiles()`;
- `format_contract_for(p)` and `verify_format_contract(c, p) -> list[str]`. These are the
  `RegMapContract` pattern: total, with type-exact comparison.
- `placement_problems(p) -> list[str]`: total.
- A command line with `--check` and `--write`.

**A generator asked for a null field raises a refusal that names the field and the missing
source. It never substitutes a default.**

**Tests.**
- `test_every_fact_carries_a_source_and_an_authority`
- `test_the_tms470_packet_fields_match_spnu194` (pp.14-19, bit by bit)
- `test_the_cy91460_register_fields_match_an205266` (pp.6-8)
- `test_the_dw_fields_match_the_pdl` (`infineon-cy-dma-h` L389-414, L2074-2109)
- `test_the_dmac_layouts_match_the_pdl_and_refuse_packing`
- `test_null_fields_refuse_by_name`. It covers the ERIE/EDIE polarity, the IS/EIS codes, the
  DMAC positions, the increment unit, the Y composition, `dcache` for TMS470 and CY91460, the TMS470 request
  routing, and the stopped predicates.
- `test_the_reference_row_is_never_a_hardware_placement`
- `test_headers_generate_deterministically`
- `test_checked_in_views_are_current`
- `test_a_tampered_format_contract_is_caught`
- `test_each_document_error_is_provenance_and_no_field_derives_from_one`
- `test_the_golden_headers_ship_in_the_wheel` (L21)

**Faults** (`dma-model.json`):
- DR1: TRSIZE shifted one bit.
- DR2: DMATC 0 admitted.
- DR3: BLK 16 encoded as 16.
- DR4: DW X_COUNT written without the −1.
- DR5: DW 1D next pointer at 0x14.
- DR6: EDIE polarity filled from p.7's table.
- DR7: TMS470 `dcache` filled as False.
- DR8: PL330 marked hardware.
- DR9: a DocNote dropped.
- DR10: the header ignores the bank base.
- DR11: a Fact without a source admitted.

**Commit.** `devices: DMA-D0 -- the DMA registry (four controllers + a PL330 reference row, cited facts, null never zero)`

### D1 — The D-laws on the oracle

**Goal.** Ten laws on the oracle:
- the claim-graph laws D-OWN, D-REUSE and D-COH, wired into the canonical verifier;
- the program and binding laws D-DESC, D-WIDTH, D-COUNT, D-TRIG, D-IDLE, D-HEAD, and D-PLAN's
  certificate shape.

A2's module carries the ownership claims. Every law is vacuous everywhere else.

**Files.**
- `bcir/kbcir/dma_program.py`: the §3.2 IR. Cost-side; it imports no verifier.
- `bcir/kbcir/dma_own.py`: D-OWN, D-REUSE, D-COH. Cost-side, with caller-passed messages.
- `bcir/frontends/devices/dma_laws.py`: the rest.
- Wiring into `verify(module)`, beside the EV laws.
- `bcir/kbcir/dma.py`: `dma_transfer_module` emits `dma.reserve`, `dma.submit`, `dma.reap:<engine>`,
  `dma.retire` and `dma.release`. `_DESC_SETUP` keeps its value, with its docstring marking it a
  placeholder.
- `bcir/tests/test_dma_own.py` and `bcir/tests/test_dma_laws.py`: registered.
- `bcir/tests/test_dma.py`: one new negative, the `SRCB` store.
- `tools/testing/faults/dma-laws.json`.
- LangRef paragraphs under R3, R5, R9 and R12.

**Design.** As §4.

**The first measurement.** Before the law lands, the PR records whether any existing R-law
already refuses a store to `SRCB` between the kick and the reap. The answer goes in the PR body.
If one does, D-OWN's message is checked to arrive first and to name the buffer (L11).

**Tests.**
- Every §4 witness, by name.
- `test_the_dma_laws_are_vacuous_on_the_existing_corpus`: the whole corpus, and every existing
  module, verify byte-identically.
- `test_the_ownership_law_is_part_of_the_canonical_verifier` (L2).
- `test_every_law_row_has_a_refused_and_an_admitted_witness` (L22).
- `test_declared_boundaries_report_unavailable_with_their_reason` (L1).

**Faults:** DL1–DL52. Each disables one clause of one law, and each is seen RED against that
law's refused witness before GREEN (L2).

**Commit.** `kbcir: DMA-D1 -- the D-laws on the oracle (ownership, descriptor, width, count, trigger, idle, reuse, coherence, headroom, plan)`

### D2 — The simulator

**Goal.** §5 in C. The documents' examples are pinned, the CM7 line cache is modelled, and every
unstated behaviour is either refused or put behind an explicit selector.

**Files.**
- `runtime/c/simdma.c`, `runtime/c/simdma.h`, and `runtime/c/simdma_profiles.h` (from D0).
- `runtime/c/simdma_main.c`: the harness.
- `runtime/manifest.json` entries.
- `tools/c/sections/simdma.sh`: a migrated runtime-gate section, which runs as a BCIR Make task
  and as CTest `c-section-simdma`.
- `bcir/tests/test_simdma.py`: builds the harness and compares golden transcripts. It is
  toolchain-gated, and a skip is recorded as a skip (L21).
- `tools/testing/faults/dma-sim.json`.

**Tests.**
- One test per §5.6 pin.
- `test_unstated_behaviour_is_refused_by_name`
- `test_outcome_selectors_cover_every_permitted_branch`
- `test_requests_during_a_step_transfer_are_dropped_and_counted`
- `test_tms470_stop_finishes_the_current_transaction`
- `test_the_cache_model_shows_the_unclean_destination_corruption`
- `test_transcripts_are_byte_identical_across_runs` (L24)

**Faults:**
- DS1: step mode serves a mid-transfer request.
- DS2: DSS set before the last unit.
- DS3: TC cleared by the controller.
- DS4: RETRIG ignored.
- DS5: an eviction applied to the DMA's view.
- DS6: a null profile field defaulted to 0.
- DS7: CH_DISABLE ignored.

**Commit.** `runtime: DMA-D2 -- the deterministic DMA simulator (four profiles, tick time, CM7 line cache, the documents' examples pinned)`

### D3 — UART TX/RX over DMA against the simulated 16550

**Preconditions.** UART U2 has landed (`sim16550`, `uart_driver.c`), and so have D1 and D2. This
slice realizes UART §12's U11. The UART owner decides whether U11 is closed by it (§8 Q10).

**Goal.** RXRDY/TXRDY (§1.5) made real:
- the coupled harness of §5.5;
- the DMA-driven TX and RX paths;
- ownership held under injected cancellation;
- the headroom law checked against the simulator's measured service times.

**Files.**
- `runtime/c/sim16550.c/.h`: adds `sim_rxrdy` and `sim_txrdy`, plus the deassert-latency
  parameters. This is the UART owner's file; if the owner prefers, the harness can wrap it
  thinly instead.
- `uart_model.DMA_MODES`: mode 1's TXRDY added, citing PC16550D p.6. This is the owner's change.
- `runtime/c/uart_dma.c`: `uart_tx_dma`, `uart_rx_dma_ring` and `uart_dma_cancel`. It contains
  no learned inference and no floating point (UART blueprint §7, L0).
- `runtime/c/uart_dma_main.c`.
- `bcir/kbcir/uart_dma_graph.py`: the DMA service as a claim graph carrying ownership and cache
  claims. It is the subject D-OWN, D-COH and D-HEAD verify.
- `bcir/tests/test_uart_dma.py`.
- `tools/testing/faults/dma-uart.json`.
- Manifest entries.

**Design.**
- **Profile order.** The **CY91460 profile comes first.** Its DREQ demand mode is the only
  documented external level request input (§2.5, §3.6): channels 0–3, low level, quantum 1, a
  declared re-sample interval. The **XMC DW** comes second: the PDL gives the RETRIG semantics,
  and trigger routing is a declared simulator assumption. TMS470 and the DMAC are declared
  boundaries, each with its refusal test.
- **RX.** A circular ring. D-HEAD's ring form sizes it. Each slot cycles reserve → submit →
  reap → retire → release.
- **TX.** One buffer per message: submit after the fill, reap before the refill.
- **Cancellation.** The variant's stop (DEOTX, disable, or STOP), then its stopped predicate, or
  the declared bound. Both outcome-selector branches run.

**Tests.**
- `test_rx_over_dma_in_modes_0_and_1_delivers_the_script_exactly`
- `test_tx_over_dma_never_overwrites_a_full_fifo`
- `test_each_d_trig_refusal_shows_its_defect_in_the_simulator`
- `test_ownership_holds_under_injected_cancellation_on_every_outcome`
- `test_the_headroom_law_predicts_the_first_overrun_tick`. Where the inequality fails by one
  byte, the simulator's first OE must arrive at the predicted tick.
- `test_dma_and_polled_drivers_move_identical_bytes`: behavioural equivalence with U2's driver,
  on one script.

**Faults:**
- DU1: mode-1 RXRDY deasserts at trigger−1.
- DU2: a deassert latency of 0.
- DU3: the TX quantum ignores 16450 mode.
- DU4: cancel releases on request.
- DU5: a ring slot reused before its reap.
- DU6: FCR3 written before the channel is enabled.

**Commit.** `devices: DMA-D3 -- UART TX/RX over DMA against the simulated 16550 (RXRDY/TXRDY, ownership under cancellation, headroom)`

### D4 — A2's lowering to the descriptor formats

**Goal.** §3 in code:
- fold the view pairs, and `PortView`s, into a `DmaProgram`;
- encode it per placement, and decode it;
- check the differential against A2;
- pin the goldens and the refusal table.

The simulator runs the lowered bytes.

**Files.**
- `bcir/kbcir/dma_lower.py`: fold, walk, encode and decode. Cost-side.
- `runtime/c/simdma.c`: decodes encoded TMS470, CY91460 and DW programs through `dma_formats.h`.
- `bcir/tests/test_dma_lower.py`.
- `tools/testing/faults/dma-lower.json`.

**Design.** §3.2–§3.7.
- The differential runs over a **generated** corpus of view pairs (shapes, strides, offsets,
  widths and banks), not over curated pins.
- The goldens pin the documents' own examples.
- Each refusal-table row is a test that pins the whole message (L11).

**Tests.**
- `test_fold_agrees_with_a2_on_the_generated_corpus`
- `test_the_spnu194_example_encodes_byte_for_byte`
- `test_the_cy91460_examples_encode_except_the_unresolved_polarity`
- `test_a2s_row_runs_fold_into_one_normal_form_2d_descriptor`
- `test_xmc_2d_encoding_is_refused_until_the_y_composition_is_stated` (the 1D-chain fallback is
  encoded instead)
- `test_the_simulator_runs_each_2d_fold_under_both_compositions`
- `test_every_refusal_row_fires_with_its_message`
- `test_encode_decode_round_trips_and_negative_increments_sign_extend` (LIB-3)
- `test_the_simulator_runs_the_lowered_bytes_and_moves_a2s_runs`

**Faults:**
- DW1: the DW count is written without the −1.
- DW2: TMS470 SRCINC set on a port.
- DW3: CY91460 BLK 16 written as 16.
- DW4: the fold drops the last row.
- DW5: the decoder does not sign-extend.
- DW6: the TMS470 packet budget is exceeded silently.
- DW7: XMC DMAC words are packed.
- DW8: an XMC 2D descriptor is encoded with `y_count` > 1 while the composition is null.

**Commit.** `kbcir: DMA-D4 -- lower A2's programs to TMS470 packets, CY91460 registers and XMC descriptors (A2 as the oracle)`

### D5 — Telemetry

**Goal.** DMA evidence carried on the existing rails. The signals:
- bytes moved;
- descriptors completed;
- completion interrupts;
- request-to-grant ticks (maximum);
- dropped requests;
- over-reads, over-writes and overruns, which are 0 in any legal plan;
- cancellations, and stop-wait ticks;
- cache lines maintained.

On the simulator their provenance is **SIMULATED**.

**Files.**
- `bcir/signal_registry.py`: new `MetricDefinition`s. Their BCIR-range IDs are allocated by the
  registry when the slice lands; none is fixed here.
- `runtime/c/bcir_signal_table.h`: regenerated by `python -m bcir.signal_table --emit`.
- The harness: it emits TelemetryEnvelopeV0 over the v0 live ring, or, as U4 did, transcript
  lines that the Python side feeds to the Broker.
- `bcir/tests/test_dma_telemetry.py`.
- `tools/testing/faults/dma-telemetry.json`.

**Tests.**
- `test_dma_signals_are_registered_with_units_and_provenance`
- `test_the_d3_run_round_trips_through_the_durable_log`
- `test_ring_overflow_is_counted`
- `test_simulated_values_are_never_labelled_measured`

**Faults:**
- DT1: MEASURED provenance on the simulator.
- DT2: dropped requests not counted.
- DT3: a signal ID hard-coded outside the registry.

**Commit.** `telemetry: DMA-D5 -- DMA signals (SIMULATED provenance) through the envelope and live ring`

### D6 — The priced strategy plan

**Goal.** D-PLAN's selection:
- the candidates are polled, IRQ-driven, and the DMA variants;
- each is filtered by legality, then priced from D5's SIMULATED tables;
- the result is certified.

A2's placeholder is retired from every certified choice.

**Files.**
- `bcir/kbcir/dma_plan.py`.
- `bcir/tests/test_dma_plan.py`.
- `tools/testing/faults/dma-plan.json`.

**Design.**
- The candidate set is small, and it is searched exhaustively.
- Prices come from a cost-table envelope tied to `cal_gen` and the placement. This is the
  tile-prior pattern: kind `bcir.dma_cost_table`, schema 1, with stale, newer and
  wrong-placement refusals.
- A learned ordering is allowed only behind a guided-equals-exhaustive certificate, with 0
  mismatches (U5's pattern).

**Tests.**
- `test_the_choice_follows_the_table_not_a_threshold`
- `test_illegal_candidates_are_never_priced`
- `test_the_certificate_records_provenance`
- `test_a_stale_table_is_refused`

**Faults:**
- DP1: a constant byte threshold.
- DP2: an illegal candidate priced.
- DP3: `_DESC_SETUP` used for a certificate.

**Commit.** `kbcir: DMA-D6 -- PIO versus DMA as a priced, certified plan`

### D7 — The law rail

**Goal.** The six-artifact pattern for every D-law:
- MLIR clauses under R3, R5, R9 and R12, with `-verify-diagnostics` negatives;
- a `bcir.dma_format_contract` op, mirroring `bcir.regmap_contract`. That op is the UART U0
  contract's law-rail half, which the UART blueprint records as pending; D7 follows it, and
  does not precede it;
- a C-twin form, where the C rail lowers DMA programming;
- LangRef text;
- a `check_passes.sh` stanza;
- a structural parity corpus in which each case declares its law, per rail (L11).

**Files.**
- `mlir/lib/passes/BCIRVerifyPass.cpp`.
- `mlir/include/BCIR/BCIRCoreOps.td`.
- `mlir/test/passes/verify_dma_*.mlir`.
- The stanza in `tools/wsl/check_passes.sh`.
- `docs/BCIR_LANGREF.md`.
- The C twin.

**Gates.**
- `ctest -L mlir` on a coherent LLVM/MLIR 23. On a host without one this is an honest,
  documented skip; CI's MLIR job owns it.
- The latest-toolchain check.
- `gen_status`, regenerated last.

**Commit.** `mlir: DMA-D7 -- the D-laws on the law rail (R3/R5/R9/R12 dma clauses, negatives, parity)`

---

## 7. Errors and inconsistencies inside the documents (provenance, never facts)

Each entry gives the page, what is printed, and why it cannot be adopted as printed. Where a
registry field depends on it, the entry also gives what the registry does. These become
`DocNote`s in D0, exactly as SLLZ058A's misprints did in `uart_model.ERRATA`. No field takes its
value from a misprint. Where one reading is chosen, the choice is explicit and cited.

### 7.1 AN205266 (CY91460 DMAC)

| ID | Where | Printed | Reading / why | Registry |
|---|---|---|---|---|
| CY-1 | p.7 against p.16, p.19 | ERIE/EDIE: "0 … enabled, 1 … disabled" | Both examples set `DMACB0_EDIE = 1; // IRQ when transfer finished`. The table and the code contradict each other, and the names say "Enable" | polarity null; ERIE/EDIE emission refused (HD-A, Q2) |
| CY-2 | p.9, p.10 | "DMABA" | no register DMABA exists; the bits (MOD, WS, DSS) are DMACB's (p.7) | read as DMACB |
| CY-3 | p.9 | "BLK[1:0]" | BLK is 4 bits, [3:0] (p.6) | 4 bits |
| CY-4 | p.9, step transfer item 3 | after one transfer "the transfer is stopped with DSS[1:0] … B'11" | contradicts block mode (DSS = 11 only at DTC = 0, items 6-7) and the UART example (BLK = 1, DTC = 50, continuing for 50 receptions, pp.18-19) | the simulator follows the count rule; the conflict is pinned by a test |
| CY-5 | p.9, block transfer item 3 | "The BLK … become 0" after a block | p.6 fn 4 makes BLK = 0 mean 16; how the block size is restored for the next block is not said | simulator: BLK is held per programming (declared); hardware: n/s |
| CY-6 | p.9, burst item 1 | "In case of block transfer…" | the section is about burst | — |
| CY-7 | p.2 against p.7, p.9 | "Bulk" transfer | "Burst" in the register table and §2.4.1.3 | "burst" |
| CY-8 | p.2 | "DMA is always triggered by a resource interrupt" | the same page lists external pins and software requests, and p.6 has STRG | sources as §1.3 |
| CY-9 | p.6 against p.19 | IS "01110 – 10010" | the UART example uses IS = 0x14 (10100), outside the printed range, and the reset value 00000 is outside it too | only the stated codes are emitted; flagged |
| CY-10 | p.10 | "transfer destination address register DMASA is ignored" | DMASA is the source register (p.8) | read as source |
| CY-11 | p.18 against p.20 | `#pragma intvect DMA_IRQ 76` and `#pragma intvect DMAIRQHandler 142`, both with `ICR63 = 20` | one controller, two vector numbers | vector n/s (Q2) |
| CY-12 | p.11 | "DMAH[2:0] bits of the DMACB register would transfer stop request" | DMAH is DMACR [27:24], a 4-bit counter (p.8); the stop end code is DSS = 10 in DMACB (p.8) | read as DSS |
| CY-13 | p.10 | fly-by I/O→memory "uses a write operation as its unit … issues a fly-by transfer (read) request" | the unit and the request contradict each other | fly-by is never emitted |
| CY-14 | p.8 | DMASA/DMADA "can't be read when the transfer is going on", then what a read returns | self-contradictory wording; the registry keeps the second sentence's behaviour | as §1.7 |
| CY-15 | p.8 | PM01 "Alternating priority (ch1 > ch0)" | "alternating" printed as a fixed order; channels 2–4 omitted | n/s beyond the two |
| CY-16 | p.11 | "By default the DMAC has the access of D-Bus" | the same paragraph: the DMAC requests the bus (DHREQ) and the CPU grants it (DHACK) | hold model as §1.3 |
| CY-17 | p.7 against p.16, p.19 | field MOD | the code writes `MD1`/`MD0` | — |
| CY-18 | p.19 | `DMACA0_EIS = 0x3` | no semicolon: the listing does not compile as printed | — |
| CY-19 | p.21 against pp.18-19 | software example "91460_dma_uart0" | the example uses UART2 | — |
| CY-20 | p.2, p.13, p.18, p.11 | sections 3.4 and 3.5 with no 3.1–3.3; the first example's "Vectors.c" printed under §3.5; "LVL bits if HRCL register" | layout and typing errors | — |

### 7.2 SPNU194 (TMS470R1x DMA)

| ID | Where | Printed | Reading / why | Registry |
|---|---|---|---|---|
| TI-1 | p.13 | "data chaining (DMAC.1)" | DCHN is bit 0 (p.14, p.16, p.36) | bit 0 |
| TI-2 | p.14, p.18 | "DISTINC" | "DSTINC" (p.15) | DSTINC |
| TI-3 | p.17, p.18 | "DMACW.9", "DMACW.11" | the register is DMAC (p.14) | — |
| TI-4 | p.5, p.13, p.19 | "DMATCW"; "Transfer count size reg"; the Table 3 caption against its header "Block Size…" | the register is DMATC (p.19) | — |
| TI-5 | p.16 | "CPS (DMACPSR.31:0)" | DMACPS (p.21 table, p.29) | — |
| TI-6 | p.24 | "REQEN x" | "RQENx" (p.21, p.24 diagram, p.25) | — |
| TI-7 | p.36 | "DMENx" | "DMAENx" | — |
| TI-8 | p.15 | SRCINC: "If the *destination* address is peripheral, the *source* address is always incremented by four" | the source's step conditioned on the destination; its sibling DSTINC rule names an "expansion bus peripheral" destination | SRCINC = 1 refused when either end is an expansion-bus peripheral (§3.3) |
| TI-9 | p.17, p.18 | "bits 0 to 32 form the … address" | 33 bits in a 32-bit field | [31:0] |
| TI-10 | p.30 | "DMACPS.32:0" | 33 bits | [31:0] |
| TI-11 | p.31 | "Bits 31:7 Reserved" | bit 7 is HIRQ (same page; p.22) | [31:8] reserved |
| TI-12 | p.33 | Table 8, "Channel Service Size Bit Encoding for BMSS" | the table encodes CHSS | — |
| TI-13 | p.40 against p.32, p.43 | "the DMA controller uses a rotating priority scheme" | CHPR selects fixed or rotating (p.32), and the B.1 example configures fixed (p.43) | CHPR governs |
| TI-14 | p.43, p.44 against p.7, p.14 | "shown in Table 1" / "Table 1. Address and Data Values" | the list of tables numbers it Table 11; Table 1 is the NCPACK encoding (p.14) | — |
| TI-15 | p.27 | "The interrupt priority is fixed … as shown in Table 6" | Table 6 gives offsets per channel, not an order | interrupt order n/s |
| TI-16 | pp.27-28 | Table 6 lists channels 0, 14 and 15 only, split across two pages | the pattern offset = channel + 1 is read from three rows | recorded as an inference |
| TI-17 | p.43 | "If any DMA transfer error occurs…" | no error condition, flag or interrupt is defined anywhere | errors n/s |
| TI-18 | p.34 | ACT_CHN "R-1" | reset value 1 for a 4-bit field; what ACT_CHN reads when idle is not stated | n/s |
| TI-19 | p.10, p.26, p.32 | "contoller", "fo", "when and FIQ" | typing errors | — |

*Checked and consistent:* the B.1 example's values (p.44), decoded field by field against
pp.14-16, 24, 32 and 36 (§3.3).

### 7.3 AN234225 (XMC7000 DW and DMAC)

| ID | Where | Printed | Reading / why | Registry |
|---|---|---|---|---|
| AN-X1 | p.7 | X_IDX "(bit7:0)" and "DW_CH_STRUCT_CH_IDX_Y_IDX (bit15:0)" in the same CH_IDX | the fields overlap; the name breaks the REG.FIELD pattern. The PDL returns Y_IDX as `uint8_t` (`Cy_DMA_Channel_GetCurrentYIndex`, `infineon-cy-dma-h` L1793-1798), consistent with an 8-bit Y index | Y_IDX position n/s (device header) |
| AN-X2 | p.9 (DMAC 1D column) | next pointer at "+10", and no increment word | its own 1D pseudo-code needs X increments (p.9); the PDL's 1D layout has `xIncr` at 0x10 and next at 0x14 (`cy_stc_dmac_dscr_1d_t`, `infineon-cy-dmac-h` L400-408; `Cy_DMAC_Descriptor_SetNextDescriptor`, `infineon-cy-dmac-c` L335). The AN's column matches its memory-copy column | **explicit choice: the PDL layout**, cited, pending the TRM (Q5) |
| AN-X3 | p.8 | the DMAC figure, as extracted: X count +0c, Y count +10, next +14, X control +18, Y control +1c | the PDL orders xSize, xIncr, ySize, yIncr, next (L359-369). The extraction may have scrambled the figure | PDL order, by explicit choice (Q5) |
| AN-X4 | p.14 against p.15 | "The number of bytes to be transmitted is 1" | step 5 transfers d0 … d15 | — |
| AN-X5 | p.15, p.35 | "DW repeats from (3) to (4)", "DMAC repeats (3) (4)" | step (3) is the descriptor read; the repeated steps are (4)–(5) | — |
| AN-X6 | p.14 against p.17 | TX "starts the transfer with a software trigger from the CPU" | the code keeps the channel disabled "because TX FIFO is empty and SCB keeps active level for DMA": a peripheral level trigger | the trigger kind of the example is n/s |
| AN-X7 | p.18, p.23, p.27 | "DW does not increment the source address and the destination address by 0x04" | ambiguous; the next sentence writes at +0x04, which contradicts "8-bit data elements" unless the element is widened, which is not stated | increment unit n/s (Q5) |
| AN-X8 | p.23 against p.21 | descriptor 1 writes to "RAM base address 0" | p.21: descriptor 1's destination is "RAM base address 1" | — |
| AN-X9 | p.27 against p.28 | "The number of bytes … is 32" | "low 8 bytes … high 8 bytes" is 16 | — |
| AN-X10 | p.31 against pp.32-33, p.41 | CRC figure "P-DMA Ch.8", DATA_SIZE 32-bit | the listing: `DW_CHANNEL (7)`, `dataSize = CY_DMA_BYTE`; and revision \*A says P-DMA was replaced by DW "in all instances" | the listing is the golden (§5.6) |
| AN-X11 | p.35 against p.37 | memory copy "Ch.7", "36 bytes", "from flash to RAM" | the listing: channel 0, `DATACNT 8`, RAM `srcData` to RAM `dstData` | — |
| AN-X12 | p.37 | `sizeof(Descriptor_0)` | undeclared; the descriptor is `DMAC_Descriptor_0` | — |
| AN-X13 | p.17, p.20, p.30 | `configure_rx_dma(rx_dma_uart_buffer_a, rx_dma_uart_buffer_b, …)` | `rx_dma_uart_buffer_b` is not declared in Listings 1, 2 and 4 | — |
| AN-X14 | pp.16-29 | "/* Initialize descriptor 1 */" over `RxDma_Descriptor_0` | Listing 3 never initializes a descriptor 1, nor uses `buffer_b` in `configure_rx_dma`, although the narrative chains two (pp.21-23) | — |
| AN-X15 | p.7, p.13 against pp.33-34 | CRC registers listed as channel configuration; "the DMA channel must be disabled during the configuration" | the CRC listing enables the channel at initialization (`.enable = true`) and calls `Cy_DMA_Crc_Init` in its loop without disabling it | D-IDLE treats CRC registers as channel configuration (Q4) |
| AN-X16 | pp.32-34 | `for(uint32_t i = 1; i < TEST_PATTERN_NUM; i++)` | skips `stcCrcConfig[0]` | — |
| AN-X17 | p.6 | "See section 0 for more details" (twice) | broken cross-references | — |
| AN-X18 | p.7, p.13 | `DMAC_CH_CH_CTL.PRIO` against `DMAC_CH_CTL.ENABLED`; `DW_CH_STRUCT_CH_CURR_PTR` against `DMAC_CH_CURR.PTR` | inconsistent register naming | — |
| AN-X19 | pp.9-10 | `X_IDX+)`; `IDX` and `X_IDX` mixed; `DST_ADDR[DST_INCR]` indexes by the increment | pseudo-code errors | geometry from the PDL |
| AN-X20 | p.18, p.28, p.39 | "transmit data from RX-FIFO"; "reads the port states"; a glossary "ADC" entry after the ADC example was removed (p.41) | wording carried over from earlier examples | — |

### 7.4 PDL against AN234225, and inside the PDL

| ID | Where | What | Registry |
|---|---|---|---|
| LIB-AN-1 | AN234225 p.9, p.8 against `infineon-cy-dmac-*` | DMAC 1D next pointer and the 8-word order: see AN-X2 and AN-X3 | PDL, by explicit choice |
| LIB-AN-2 | AN234225 p.7 against `infineon-cy-dma-h` L1793-1798 | Y_IDX "bit15:0" against a `uint8_t` Y index: see AN-X1 | n/s |
| LIB-AN-3 | AN234225 p.4, p.5 against `infineon-cy-dmac-h` L36-37 | "DMAC does not support hardware triggers" against the generic "basing on a hardware trigger signal from another component". The library text spans devices; the AN is about XMC7000 | the AN governs XMC7000 (HD-K); both recorded |
| LIB-AN-4 | AN234225 p.37 against `infineon-cy-dmac-h` L90-96 | the listing's buffers are not 32-byte aligned, and `dstData` (written by the CPU) is not cleaned before enable. The PDL requires both | D-COH's refused witness |
| LIB-AN-5 | AN234225 pp.16-30 against `infineon-cy-dma-h` L79-80 | the UART listings do no cache maintenance; the PDL requires it on CM7. The AN does not say whether the cache is on | D-COH applies whenever `dcache` is true |
| LIB-AN-6 | AN234225 p.9 against `infineon-cy-dma-h` L1298-1299 | `X_IDX <= COUNT` against the user count 1…256 held as count−1. **Consistent:** the PDL resolves the AN's unstated encoding | count−1 |
| LIB-1 | `infineon-cy-dmac-h` L367-368 | comments give `yIncr` at "0x00000010" and `nextPtr` at "0x00000014"; the struct layout puts them at 0x18 and 0x1C | layout from the struct |
| LIB-2 | `infineon-cy-dma-h` L92, L805 against L850, L1606, `infineon-cy-dma-c` L330 and `infineon-cy-dmac-h` L92 | "32 bit aligned" against "32 byte aligned"; with `CY_ALIGN(32)` and a 32-byte line, "byte" is the consistent reading | 32-byte, line-exclusive (D-COH) |
| LIB-3 | `infineon-cy-dma-h` L1371-1376 and the other increment getters; `infineon-cy-dmac-h` L1165-1170 | a signed 12-bit or 16-bit field is returned as `(int32_t) _FLD2VAL(...)`, with no visible sign extension. Unless `_FLD2VAL`, which is not in the fetched sources, sign-extends, −1 reads back as 4095 (DW) | BCIR's decoder sign-extends; D4 pins the round trip |
| LIB-4 | `infineon-cy-dmac-h` L1625-1638 | `Cy_DMAC_Channel_GetCurrentYloopIndex` is documented as returning "the current transfer X-loop index" | — |
| LIB-5 | `infineon-cy-dma-c` L114-122, L147-149; `infineon-cy-dmac-c` L70-78 | parameter checks by assertion only (HD-P) | the encoder refuses (D-COUNT) |

### 7.5 QEMU's PL330 model (`qemu-pl330-c`)

These notes are about the model. None is about ARM's PL330, whose manual this session could not
read.

| ID | Where | What |
|---|---|---|
| QM-1 | L395-401 | `pl330_fifo_get` claims to verify each byte's tag, but compares `s->tag[s->head]` on every iteration, never the tag at `get_idx` |
| QM-2 | L43-55 against L1104-1119, L1184, L1216 | DATA_READ_ERR, DATA_WRITE_ERR and INSTR_FETCH_ERR are defined and never raised; the model ignores the result of every memory access |
| QM-3 | L1472-1473 | FSRD returns `s->manager.state ? 1 : 0`, so an executing manager reads as faulted |
| QM-4 | L1239-1271 (comment L1243-1244) | execution runs to quiescence, untimed ("Is it all right to execute everything or should we do per-cycle simulation?"). It cannot serve as a timing oracle for D-HEAD |
| QM-5 | L1281-1289, L1043-1048, L683 | peripheral requests are modelled as "busy" levels, so with no line asserted DMAWFP never waits; DMAFLUSHP does nothing |
| QM-6 | L1467-1471 | INTCLR reads 0, with the comment "Documentation says that we can't read this register but linux kernel does it" |

---

## 8. Open questions for the user

1. **Names.** May the slices be titled `DMA-D0` … `DMA-D7`? The roadmap's D0–D7 gates and
   D-R1…D-R6 laws already use "D". May the laws keep their mnemonic names under the existing
   R-laws' prefixes (`R3-dma`, `R5-dma`, `R9-dma`, `R12-dma`), or should they take a new number?
2. **CY91460.** Can you supply the MB91460 hardware manual, for three things the application
   note leaves contradictory or open?
   - the ERIE/EDIE polarity (CY-1);
   - the IS/EIS request-source table (p.6 fn 3; CY-9);
   - the DMA interrupt vector (CY-11).

   Until then, no CY91460 program with interrupt behaviour can be emitted.
3. **Stopped predicates.** For each controller, which register state proves that a cancelled
   channel has finished its in-flight transfer? None of the documents says (§1.6; HD-D). Without
   it, D-OWN's cancellation path needs a declared time bound per placement.
4. **Writes while active.** Are writes to TMS470 DMACC and DMAGC, or to XMC CRC registers,
   allowed while a channel is active? (§1.7; AN-X15.) And how should TMS470 privileged-mode
   writes (RWP) be represented in the claim graph?
5. **XMC7000 TRM.** Can you supply the XMC7000 architecture and registers TRMs, or the PDL's
   device headers (`DMAC_CH_V2_DESCR_*`, `DW_CH_STRUCT_CH_IDX_*`)? They are needed for:
   - the DMAC's descriptor bit positions;
   - the increment unit (elements or bytes);
   - how a Y increment composes with the X walk, which decides whether A2's row runs fold into
     one 2D descriptor (§1.4, §3.5);
   - how a transfer size of "word" combines with byte data;
   - whether the DMAC 1D layout is the PDL's or the AN's (AN-X2, AN-X3).
6. **Caches.** Do the TMS470R1x and MB91460 parts have a data cache? Neither document says.
   For the XMC7000, is the CM7 D-cache enabled in your target configuration, with which write
   policy, and is any MPU region non-cacheable?
7. **First target.** Is the first consumer an **external 16550** on an MCU bus (only the CY91460
   documents an external request input: DREQ, channels 0–3), or an on-chip UART (XMC7000 SCB,
   MB91460 UART2)? The documents' own examples use the on-chip UARTs, and their request
   semantics are not the 16550's.
8. **TMS470 routing.** Which DMA_REQ line serves which channel on the target device, and is
   RXRDY's polarity inverted on the board (SPNU194 p.40: rising edge)?
9. **PL330.** Should the PL330 stay a simulator-reference row, or become a placement? A
   placement would need the ARM TRM (DDI 0424A), which the session could not fetch.
10. **U11.** Does D3 close the UART blueprint's U11 (DMA integration), or run beside it? D3 needs
    two small changes in the UART owner's files (`sim_rxrdy`/`sim_txrdy`, and mode 1's TXRDY in
    `DMA_MODES`).
11. **Shared provenance.** May `DocNote` and `Erratum` move from `uart_model.py` into a shared
    `bcir/frontends/devices/provenance.py`, with `uart_model` re-exporting them? Or should D0
    wait for U0 on `main` and import them from it?
12. **Library versions.** The PDL used here is mtb-pdl-cat1 master (drivers 2.90 and 1.40),
    fetched 2026-10-27. AN234225 cites PDL release v3.3.0 (p.16, p.20). Should D0 pin a specific
    PDL release for the registry's library-sourced facts?
