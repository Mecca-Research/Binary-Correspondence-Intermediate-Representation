# Languages — high-level programming languages

> **Status: PLANNED.** This folder has a scope and no lessons yet. Nothing here
> is teaching material, and nothing in it is gated. See
> [`../ROADMAP.md`](../ROADMAP.md) Phase 6 for when it opens, and
> [`../CORPUS_STANDARD.md`](../CORPUS_STANDARD.md) for what it must deliver.

## Scope

General-purpose languages above the systems layer, taught with the same
discipline: what the language guarantees, what the implementation adds,
and where the two are confused.

### In scope

- Python, Java, JavaScript, TypeScript, Rust, Go, Swift, PHP, R, Fortran,
  Visual Basic, Delphi, CSS
- Per language: semantics, memory and concurrency model, toolchain, and the
  boundary where its abstractions stop holding

### Out of scope

- Framework-specific material, which belongs to `backends/`
- Database query languages, which belong to `data/`

## Reference manuals to break down first

A subject folder opens only after its primary references are inventoried, so
the lessons cite normative sources rather than recollection. For this subject:

- The normative specification or reference for each language at a pinned
  version, recorded per lesson

## Depends on

`systems/` for the execution model each runtime rests on.

## Verification plan

- Every example runs under a pinned interpreter or compiler
- Semantic claims that differ between implementations name the implementation
- Version-dependent behaviour is pinned, never described as universal

## Corpus obligations

Like every subject, this folder is complete only at Tier 3: chapters whose
factual claims are gated, chunks that trace to a line, and distillation records
each backed by a gate. Prose alone does not close it.
