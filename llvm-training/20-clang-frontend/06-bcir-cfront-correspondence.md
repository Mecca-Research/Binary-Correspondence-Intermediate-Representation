# The BCIR C-Front Correspondence

## Why this chapter exists

This repository contains a second C front end:
[`runtime/c/bcir_cfront.c`](../../runtime/c/bcir_cfront.c) and its supporting
translation units, documented in
[`../../docs/languages/CFRONT_GUIDE.md`](../../docs/languages/CFRONT_GUIDE.md).
It is not a Clang replacement and does not try to be. Understanding what it
shares with Clang — and what it deliberately does not — is the difference
between using it correctly and misreading its output.

**This chapter is review material, not a machine-checked claim set.** The
checked claims in this chapter family are the Clang ones in
[`03`](03-c-lowering-rules.md), [`04`](04-cxx-lowering-rules.md), and
[`05`](05-abi-and-target-lowering.md). Statements here about the BCIR rail are
sourced from its own documentation and should be re-read there before being
relied on: this corpus is a training pack, and the `runtime/c` rail owns its own
truth.

## The same three stages, a different third one

| Stage | Clang | BCIR C-front |
| --- | --- | --- |
| Preprocess/parse | full C++ and C | a driver-oriented subset of C23 |
| Semantic check | Sema, full language | verified C plus BCIR verifier-law status (R1–R18 range per the guide) |
| Lowering | LLVM IR for a target | BCIR claims/resources, with an explicit LLVM-backend fallback |

The first two stages answer the same kind of question and are meant to agree.
The third does not: Clang's output is a realization for one target, and BCIR's
output is a *plan* that has not yet chosen one. That is the whole architectural
difference in one row.

## Three things it copies from Clang on purpose

1. **Clang-style diagnostics.** Same shape, same severity vocabulary, plus a
   machine-readable form for editors and CI. The point is comparability: the
   same input can be put to both front ends and the answers diffed.
2. **An explicit target ABI matrix.** Type layout for another target's ABI is a
   first-class operation, not a compile flag side effect — for the reason
   [`05-abi-and-target-lowering.md`](05-abi-and-target-lowering.md) documents:
   layout and argument classification cannot be inferred from the source.
3. **Graceful degradation as a reported signal.** When the input leaves the
   supported subset, the rail reports a *fallback-to-LLVM signal* rather than
   emitting something approximate. An unsupported construct produces a verdict,
   not a guess.

That third property is the one worth transplanting into any tool you build.
"Fails honestly" is a design requirement, not a courtesy: a front end that
half-lowers an unsupported construct produces output that passes every
structural check and is wrong.

## Three things it deliberately does not do

- **No C++.** The object-model machinery in
  [`04-cxx-lowering-rules.md`](04-cxx-lowering-rules.md) — vtables, comdat
  merging, EH cleanups, mangling — is exactly the surface the repository keeps
  behind a narrow C ABI seam.
- **No general instruction selection.** BCIR is a planning, verification,
  artifact, and runtime layer *above* a resident toolchain; the native-object
  question is governed by a written gate
  ([`../../docs/BCIR_NATIVE_OBJECT_GATE.md`](../../docs/BCIR_NATIVE_OBJECT_GATE.md)),
  not by drift.
- **No silent extension of the subset.** Constructs outside the documented
  subset are refused and recorded, and the guide's own "Known limits" section is
  the list.

## Where the two front ends must agree, and how that is proved

The repository's method is **differential parity, not assertion**
([`../../docs/PARITY.md`](../../docs/PARITY.md)): a claim that two rails agree
is worth what its generated counterexample search cost. For a C front end that
means agreement is demonstrated on:

- **acceptance** — does each front end accept or reject the same input, for the
  same reason;
- **type layout** — do sizes, offsets, and alignments match for a named target
  ABI (`clang -Xclang -fdump-record-layouts` is the Clang-side oracle);
- **observable behaviour** — does the compiled program compute the same thing.

The corpus-side lesson from [`03-c-lowering-rules.md`](03-c-lowering-rules.md)
applies directly: the constructs most likely to disagree are the ones the
frontend *erases* — bit-fields, unions, packed layout, integer promotion, and
the exact placement of `nsw`. A front end can be correct on every test in a
corpus and wrong on a construct the corpus never contained.

## What a training reader should take from the comparison

- **Read the erasures as an interface question.** Every fact the frontend erases
  is a fact a correspondence IR must carry explicitly if anything downstream
  needs it. Bit-field width, union active member, `restrict`, and signed-overflow
  assumptions are all gone from IR unless someone encoded them.
- **A second front end is a differential oracle, not a competitor.** Its value
  is that it disagrees with Clang somewhere, and that the disagreement is
  findable. That is why the rail is fuzzed against Clang rather than only tested
  against expectations.
- **Subset boundaries belong in the tool, not in the documentation.** A
  documented limit that the tool does not enforce is a limit that will be
  exceeded.

## See also

- [`../../docs/languages/CFRONT_GUIDE.md`](../../docs/languages/CFRONT_GUIDE.md) — the rail's own reference
- [`../../docs/languages/C_MEMORY_DISCIPLINE.md`](../../docs/languages/C_MEMORY_DISCIPLINE.md) — the three runtime memory classes
- [`../../docs/PARITY.md`](../../docs/PARITY.md) — how cross-rail agreement is enforced
- [`../bcir-mapping/README.md`](../bcir-mapping/README.md) — BCIR-to-LLVM lowering idioms
- [`03-c-lowering-rules.md`](03-c-lowering-rules.md) — the Clang rules this compares against
