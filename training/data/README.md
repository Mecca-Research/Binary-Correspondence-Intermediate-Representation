# Data — databases and query languages

> **Status: PLANNED.** This folder has a scope and no lessons yet. Nothing here
> is teaching material, and nothing in it is gated. See
> [`../ROADMAP.md`](../ROADMAP.md) Phase 7 for when it opens, and
> [`../CORPUS_STANDARD.md`](../CORPUS_STANDARD.md) for what it must deliver.

## Scope

How data is stored, queried, and kept consistent, from the query language
down to the storage engine.

### In scope

- SQL: the language, the standard, and where engines diverge
- NoSQL families and the consistency models they actually provide
- Storage engines, indexing, transactions, and isolation levels
- Query planning, and why a plan is a cost decision

### Out of scope

- Data-warehouse product operations
- Analytics and modelling practice

## Reference manuals to break down first

A subject folder opens only after its primary references are inventoried, so
the lessons cite normative sources rather than recollection. For this subject:

- ISO/IEC 9075 (SQL) at the parts taught
- The reference manual for each engine used, at a pinned version

## Depends on

`languages/` for the client side; `low-level/` for on-disk encoding.

## Verification plan

- Every query runs against a pinned engine in a reproducible fixture
- Isolation-level claims are demonstrated with an interleaving that exhibits
  the anomaly, not merely asserted from the standard's table
- Query-plan lessons show the engine's own plan output

## Corpus obligations

Like every subject, this folder is complete only at Tier 3: chapters whose
factual claims are gated, chunks that trace to a line, and distillation records
each backed by a gate. Prose alone does not close it.
