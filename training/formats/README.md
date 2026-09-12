# Formats — text, markup, and document encoding

> **Status: PLANNED.** This folder has a scope and no lessons yet. Nothing here
> is teaching material, and nothing in it is gated. See
> [`../ROADMAP.md`](../ROADMAP.md) Phase 5 for when it opens, and
> [`../CORPUS_STANDARD.md`](../CORPUS_STANDARD.md) for what it must deliver.
>
> The folder's *data* half — a pinned Unicode 17.0 source tree and the database
> built from it — is scheduled ahead of its lessons in
> [`../UNICODE_ROADMAP.md`](../UNICODE_ROADMAP.md). Landing it satisfies this
> folder's first opening condition (references inventoried, versioned, dated)
> and does not by itself open the folder as a subject.

## Scope

How text and documents are represented, and where that representation
leaks into program behaviour.

### In scope

- Unicode: encoding forms, normalisation, collation, and the security
  consequences of confusables
- XML and its schema languages; HTML and its parsing algorithm
- Document-level concerns: escaping, injection boundaries, canonicalisation

### Out of scope

- Binary wire formats, which belong to `low-level/`
- Rendering, layout, and styling, which belong to `backends/`

## Reference manuals to break down first

A subject folder opens only after its primary references are inventoried, so
the lessons cite normative sources rather than recollection. For this subject:

- The Unicode Standard and UAX #15, #31, #39
- W3C XML 1.0, XML Schema, and the HTML parsing specification

## Depends on

`low-level/` for the encoding foundations.

## Verification plan

- Parser lessons are checked against the specification's own conformance
  suites where one exists
- Never parse a format with a host-language parser: the format's grammar is
  enforced before any conversion, a rule this repository has paid for

## Corpus obligations

Like every subject, this folder is complete only at Tier 3: chapters whose
factual claims are gated, chunks that trace to a line, and distillation records
each backed by a gate. Prose alone does not close it.
