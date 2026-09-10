# Hardware — from transistors to instruction sets

> **Status: PLANNED.** This folder has a scope and no lessons yet. Nothing here
> is teaching material, and nothing in it is gated. See
> [`../ROADMAP.md`](../ROADMAP.md) Phase 2 for when it opens, and
> [`../CORPUS_STANDARD.md`](../CORPUS_STANDARD.md) for what it must deliver.

## Scope

How a machine is built, from device physics up to the instruction set a
compiler targets. This is the floor of the corpus: everything above it
eventually bottoms out here.

### In scope

- Circuit fundamentals; CMOS logic, timing, and power
- VLSI design flow and the abstraction ladder from cells to blocks
- SPICE modelling and what a simulation can and cannot establish
- Verilog and VHDL: synthesis versus simulation semantics
- Vendor instruction sets, mnemonic families, and encoding formats
- Memory and register architecture: hierarchies, banks, ports, coherence

### Out of scope

- Fabrication process engineering
- Analog/RF design beyond what digital timing requires
- Board-level and mechanical design

## Reference manuals to break down first

A subject folder opens only after its primary references are inventoried, so
the lessons cite normative sources rather than recollection. For this subject:

- The vendor architecture manuals for each ISA taught (volume, revision,
  and date recorded per lesson)
- IEEE 1364 / 1800 (Verilog, SystemVerilog) and IEEE 1076 (VHDL)
- A SPICE reference for the simulator actually used
- The memory-device standards relevant to the hierarchies taught

## Depends on

Nothing. This is the base of the ladder and can open in parallel with any other phase.

## Verification plan

- HDL sources elaborate under an open toolchain, or the lesson says which
  toolchain and skips honestly
- Encoding tables round-trip: assemble a mnemonic, disassemble it, compare
- Timing and power figures are labelled `measured`, `modelled`, or
  `from the datasheet`, never mixed in one table

## Corpus obligations

Like every subject, this folder is complete only at Tier 3: chapters whose
factual claims are gated, chunks that trace to a line, and distillation records
each backed by a gate. Prose alone does not close it.
