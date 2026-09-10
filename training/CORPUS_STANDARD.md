# The Corpus Standard — three tiers every subject must reach

This is the contract for `training/`. It applies to every subject folder, the
existing `llvm/` one and every folder the [`ROADMAP.md`](ROADMAP.md) adds after
it. A subject is not finished when its prose is written; it is finished when it
reaches Tier 3.

## Why three tiers

The corpus serves three consumers, and they need the same knowledge in three
different shapes:

| Consumer | Needs | Tier |
| --- | --- | --- |
| A person learning | narrative, worked examples, a reading order | 1 — prose |
| An agent doing a task | a retrievable passage, scoped and traceable | 2 — retrieval |
| A model being trained | supervised examples with verified answers | 3 — distillation |

Writing only Tier 1 and hoping the rest follows is how a corpus becomes a
document nobody can compute over. Tier 2 and Tier 3 are **derived**, not
hand-written — which is the only way they stay in step with the prose.

```
     Tier 1  prose            Tier 2  retrieval          Tier 3  distillation
   ┌──────────────────┐     ┌───────────────────┐      ┌────────────────────┐
   │ chapters, tables │────▶│ chunks + spans    │      │ (system,user,      │
   │ examples, gates  │     │ heading trails    │      │  assistant) records │
   │                  │     │ embedding-ready   │      │ each gate-backed    │
   └──────────────────┘     └───────────────────┘      └────────────────────┘
      hand-written             build_chunks.py          build_distillation.py
      and gated                deterministic            deterministic
```

## Tier 1 — prose

What already exists in `llvm/`: numbered chapters, checked examples, exercises,
and the gates that verify them.

**The Tier-1 rule, inherited from the LLVM corpus and unchanged:** a statement
of fact belongs to a gate, or it is prose. Chapters may explain, motivate, and
argue; but any claim a reader could act on — an ABI rule, an instruction
semantic, a tool's output — must have something in `tools/` that re-checks it.

## Tier 2 — retrieval

Built by [`tools/build_chunks.py`](tools/build_chunks.py), schema in
[`schema/chunk-v1.json`](schema/chunk-v1.json).

A chunk is one embedding-ready unit of corpus text. Requirements:

- **Content-addressed identity.** `chunk_id` is a sha256 over
  `(subject, source_path, heading_trail, text)`. Editing one section does not
  renumber its neighbours; reordering a file does not invalidate the rest.
- **Traceable to a line.** Every chunk carries `source_path`, an inclusive
  `span`, and the `source_sha256` of the whole file, so a retrieved passage can
  be opened where it came from and shown stale when the source moves on.
- **Code is never split.** A fenced block stays whole even when it exceeds the
  size budget. A truncated program teaches the wrong thing, and half a function
  retrieved into a context window is worse than no example.
- **Context comes from the heading trail, not token overlap.** Each chunk is
  prefixed with `[subject] Chapter > Section > Subsection`. Deterministic, and
  it does not duplicate neighbouring text into the index.
- **`verified_by` is honest.** It lists the gates covering that source file, and
  it is conservative: a chunk claiming verification it does not have is worse
  than one marked unverified, because a consumer weights the first higher.

### The embedding rule

**A builder never writes a vector.** `embedding` is `null` and
`embedding_spec.model` is `null` until a named model fills both. The verifier
enforces the pair: a vector with no model named is rejected.

This is the same discipline the performance chapter applies to an unreadable
hardware counter — record `null`, never `0`. A fabricated vector is
indistinguishable downstream from a measured one, and it silently poisons every
similarity computed against it.

Filling the vectors is a separate, explicitly-named step:

```
chunks.jsonl (embedding: null)  ──[named model, recorded in embedding_spec]──▶  chunks.jsonl (embedding: [...])
```

## Tier 3 — distillation

Built by [`tools/build_distillation.py`](tools/build_distillation.py), schema in
[`schema/distill-v1.json`](schema/distill-v1.json).

Chat-format `(system, user, assistant)` records, ready for supervised
fine-tuning without a converter. One rule governs the tier:

> **A record exists only if a gate in this repository checks its answer.**

Every record carries `verified_by: {gate, claim}`, and the verifier rejects any
record whose gate is not a real file. A distillation record whose assistant turn
nothing verifies is a hallucination with provenance attached — and it is *worse*
than no record, because the provenance makes a consumer trust it more.

The consequence is deliberate: **the size of Tier 3 is bounded by how much of a
subject is actually checked, not by how much of it is written.** To get more
training data, gate more claims. That is the pressure the corpus wants.

### Record kinds

| Task | Question | Answer comes from | Verified by |
| --- | --- | --- | --- |
| `exercise` | write the artifact | the reference solution | the solution verifier |
| `repair` | why is this rejected? | the fixture's declared defect | the invalid-fixture gate |
| `prediction` | what does this transform produce? | the checked-in golden | the opt-diff gate |
| `claim` | show the evidence for a stated fact | real tool output in a checked snapshot | the claim gate |
| `review` | is this conclusion supported? | the pinned verdict and its reason | the analysis self-test |

`claim` and `review` are the most valuable and the hardest to fake: their
answers are lines of real tool output and machine-pinned verdicts, including the
verdicts that **refuse** to conclude. A corpus that only ever trains on
confident answers teaches confidence.

### Splits and leakage

Split assignment hashes the **source file**, not the record. Every record
derived from one file lands in the same split, so a test record can never be a
memorised train record. The verifier fails on any source that appears in two
splits, and on a corpus that landed entirely in one.

## The gate

[`tools/verify_corpus_records.py`](tools/verify_corpus_records.py) builds both
tiers twice and enforces:

1. **Anti-vacuity** — a run that examined nothing fails. A gate that passes on
   an empty corpus is green for not running.
2. **No fabricated embeddings** — null, or a finite vector with a named model
   and matching `dim`.
3. **Gate-backing** — every distillation record names a gate that exists.
4. **Provenance** — sources exist, digests are current, spans lie inside their
   file.
5. **No split leakage** — one source, one split.
6. **Determinism** — two builds of the same tree are byte-identical.

Each of those was proved able to fail by injecting the defect it guards and
watching the gate fire.

## What a new subject folder must deliver

A subject is **Tier-1 complete** when its chapters exist and its factual claims
are gated. It is **corpus-complete** when:

- [ ] `training/<subject>/` chapters, examples, and a `tools/` gate set
- [ ] every factual claim reachable from a gate
- [ ] `build_chunks.py --subject <name>` produces chunks with no unverified
      claims of verification
- [ ] `build_distillation.py --subject <name>` produces gate-backed records in
      at least two splits
- [ ] `verify_corpus_records.py` passes, and each new gate has been broken once
      on purpose to prove it fires
- [ ] the subject's `README.md` states its verification boundary — what is
      checked, what is reviewed, and what is neither

The builders discover subjects automatically: any directory under `training/`
that is not `tools/` or `schema/` is treated as a subject. A new folder joins
both tiers as soon as it has content, with no registration step.

## What this standard is not

- It is not an embedding pipeline. It produces embedding-*ready* records and
  refuses to invent vectors; choosing and running a model is a separate step
  with its own provenance.
- It is not a fine-tuning harness. It produces records in a standard chat
  format; training with them is out of scope for this repository.
- It does not make unverified prose worthless. Tier 1 carries the explanation
  and the argument, and much of it will never be gateable. It just does not
  become Tier-3 training data.
