# Low-level — representation, encoding, and the lost art of machine code

> **Status: PLANNED.** This folder has a scope and no lessons yet. Nothing here
> is teaching material, and nothing in it is gated. See
> [`../ROADMAP.md`](../ROADMAP.md) Phase 4 for when it opens, and
> [`../CORPUS_STANDARD.md`](../CORPUS_STANDARD.md) for what it must deliver.

## Scope

What a program or a datum becomes once it leaves source form. The
representation layer: intermediate forms, wire encodings, bytecode, and
hand-written machine code on modern hardware.

### In scope

- Hardware abstraction layers and what they may and may not hide
- Intermediate representations as a family, and why they differ
- Machine code programming on current hardware — a subject largely abandoned,
  and one BCIR is unusually well placed to revive
- Character and wire encodings: ASCII, ASN.1, EXI, XDR, Protocol Buffers,
  Thrift, FlatBuffers, MessagePack, CBOR, JSON
- Bytecode targets: Java bytecode, WASM, CIL, SPIR-V
- Networking as an encoding and framing problem

### Out of scope

- Cryptographic protocol design
- Application protocols above the framing layer

## Reference manuals to break down first

A subject folder opens only after its primary references are inventoried, so
the lessons cite normative sources rather than recollection. For this subject:

- ITU-T X.680–X.697 for ASN.1 and its encoding rules — already the strongest
  reference base in this repository
- RFC 8949 (CBOR), RFC 4506 (XDR), the Protocol Buffers and FlatBuffers
  specifications, W3C EXI
- The JVM, WASM, ECMA-335 (CIL), and SPIR-V specifications
- The Unicode standard for the character-encoding material

## Depends on

`hardware/` for encodings and addressing; `llvm/` for the IR family lessons.

## Verification plan

- Every encoder is checked against the standard's own worked examples, never
  only against its own decoder — a round trip passes when both share a
  misreading
- Every decoder is fuzzed at the trust boundary
- Canonical forms assert both halves: the bad spelling is refused AND the
  encoder's own output still round-trips

## Corpus obligations

Like every subject, this folder is complete only at Tier 3: chapters whose
factual claims are gated, chunks that trace to a line, and distillation records
each backed by a gate. Prose alone does not close it.
