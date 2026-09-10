# Backends — application runtimes and frameworks

> **Status: PLANNED.** This folder has a scope and no lessons yet. Nothing here
> is teaching material, and nothing in it is gated. See
> [`../ROADMAP.md`](../ROADMAP.md) Phase 8 for when it opens, and
> [`../CORPUS_STANDARD.md`](../CORPUS_STANDARD.md) for what it must deliver.

## Scope

The application layer: server runtimes, frameworks, and the composition
patterns built on everything below.

### In scope

- Node.js and its execution and module model
- React and component-model frameworks
- Service composition, API boundaries, and deployment shapes

### Out of scope

- Product design and UX
- Vendor platform operations

## Reference manuals to break down first

A subject folder opens only after its primary references are inventoried, so
the lessons cite normative sources rather than recollection. For this subject:

- The runtime and framework documentation at pinned major versions
- The relevant web platform specifications where behaviour is normative

## Depends on

`languages/` and `data/`.

## Verification plan

- Examples run against pinned dependency versions with a lockfile
- Framework behaviour that changes across majors is pinned and dated
- No example depends on a network service at gate time

## Corpus obligations

Like every subject, this folder is complete only at Tier 3: chapters whose
factual claims are gated, chunks that trace to a line, and distillation records
each backed by a gate. Prose alone does not close it.
