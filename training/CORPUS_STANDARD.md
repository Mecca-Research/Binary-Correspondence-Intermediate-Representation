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
   ┌──────────────────┐     ┌───────────────────┐      ┌─────────────────────┐
   │ chapters, tables │────▶│ chunks + spans    │      │ (system,user,       │
   │ examples, gates  │     │ heading trails    │      │  assistant) records │
   │                  │     │ embedding-ready   │      │ each gate-backed    │
   └──────────────────┘     └─────────┬─────────┘      └─────────────────────┘
      hand-written             build_chunks.py          build_distillation.py
      and gated                deterministic            deterministic
                                      │
                                      ▼
                            ┌───────────────────┐
                            │ vectors, attributed│
                            │ + exact search     │
                            └───────────────────┘
                              embed_chunks.py
                              search_chunks.py
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

**A chunker never writes a vector.** `embedding` is `null` and
`embedding_spec.model` is `null` until a named model fills both. The verifier
enforces the pair: a vector with no model named is rejected.

This is the same discipline the performance chapter applies to an unreadable
hardware counter — record `null`, never `0`. A fabricated vector is
indistinguishable downstream from a measured one, and it silently poisons every
similarity computed against it.

Filling the vectors is a separate, explicitly-named step —
[`tools/embed_chunks.py`](tools/embed_chunks.py), with the set manifest in
[`schema/embedding-set-v1.json`](schema/embedding-set-v1.json):

```
chunks.jsonl            embed_chunks.py            embeddings/<model>/
(embedding: null)  ──▶  [a named Provider]   ──▶   manifest.json   model, revision, coverage
                                                   index.jsonl     chunk_id ─ row ─ text digest
                                                   vectors.f32     float32, row-major
                                                   vectors.q15     BCIR's Q15 code space
```

**Vectors live beside the chunks, not inside them.** A chunk is a deterministic
function of the corpus alone; a vector is a function of the corpus *and* a
model. Storing them together would make every chunk record change whenever a
model did, and would forbid holding two models' vectors for one corpus. The
manifest is the binding, and `--inline` joins them for consumers that want the
single-file shape.

### The attribution rule

Naming the model is necessary and **not sufficient**. A hashed-n-gram vector and
a trained sentence encoder are both "named", and they behave nothing alike: the
first scores near-duplicate text highly and a paraphrase not at all. So every
set and every filled `embedding_spec` also declares `semantics`:

| `semantics` | Means | Retrieval behaviour |
| --- | --- | --- |
| `lexical` | a specified function of the surface text | matches wording |
| `learned` | trained weights | matches meaning |

This is `measured` / `modelled` / `estimated` applied one level up: a consumer
that cannot tell the two apart cannot weight them, and will discover the
difference in its own retrieval quality instead.

`revision` is required alongside, for the same reason a benchmark pins its
toolchain: an unpinned model is one whose vectors nobody can reproduce.

### The refusal rule

A model the environment cannot load is a **skip that writes nothing** — never a
substituted model, and never a set carrying the requested model's name over
another model's vectors. The CI job that installs a model passes
`--require-provider`, which turns that skip into a failure: absence is expected
on a laptop and is a defect in the job that exists to provide it.

### Searching the vectors

A rail that builds vectors nothing queries has not been shown to work, so
retrieval is part of the standard, not an example
([`tools/search_chunks.py`](tools/search_chunks.py)).

**The arithmetic is BCIR's, not a lookalike.** `runtime/c/bcir_ai_kernels.c`
already carries what this needs and BCIR already gates it, so the corpus calls
it rather than writing a second set:

| Kernel | Used for |
| --- | --- |
| `bcir_ai_q15_topk` | exact integer squared-L2 top-k — the search |
| `bcir_ai_quantize_q8_f64` | the BCIRQ8 view, produced by BCIR's own bridge |
| `bcir_ai_q8_rows_dot_f64` | scoring that view — its header names embedding projections as its purpose |

Vectors are projected into BCIR's symmetric **Q15** code space: signed int16,
`-32768` excluded so negation stays lossless, and codes rounded **half away from
zero** — the rule `bcir.kbcir.quantize` specifies, *not* Python's banker's
rounding. Reusing a convention has to include its rounding, or the claim is
false in exactly the place nobody looks; the gate checks that against BCIR's own
function, on the half-integers where the two rules differ.

For unit vectors,

```
||q - p||²  =  2 - 2·cos(q, p)
```

so an exact squared-L2 ranking *is* a cosine ranking, computed in integers with
no floating-point tie-break to disagree about across hosts.

Three backends, and the difference between them is the point:

| Backend | Arithmetic | Held to |
| --- | --- | --- |
| `reference` | exact Q15, pure Python | the definition |
| `native` | exact Q15, `bcir_ai_q15_topk` | **equals** `reference`, exactly |
| `q8` | BCIRQ8, 8 bits/coordinate | still retrieves an exact match; the rest measured |

`reference` and `native` are one contract with two implementations, so the gate
requires them to agree **exactly** — same rows, same integer distances. That is
the differential this repository prefers to an assertion.

`q8` is deliberately outside that equality. Requiring a lossier view to rank
identically would be requiring quantization not to quantize. It is held to the
one thing loss must not break — a query that *is* some chunk's vector must still
rank that chunk first — and its agreement beyond that is measured and reported,
never asserted.

> The direction of the dependency is unchanged: `training/` is never a build
> dependency of BCIR. BCIR is imported lazily and only for the native backend,
> and the reference path is always sufficient on its own. The corpus may call
> the implementation it teaches; the implementation still knows nothing about
> the corpus.

### Measuring whether retrieval works

Everything above is a statement about the *index*: the vectors are attributed,
aligned, non-degenerate, and searched correctly. An index can satisfy all of
that and still rank the wrong chapter first. Measuring the difference needs
judgments, and where those come from decides whether the number means anything.

**No query is written for the evaluation.** A query set authored by whoever also
tuned the retriever measures the author's memory of the corpus. Every judgment
in [`tools/build_eval_queries.py`](tools/build_eval_queries.py) is instead
derived from a binding the corpus already made for another purpose — an index
row that says which chapters cover a concept, a prose link whose text describes
what it points at, a section's own heading trail.

**Every family declares its bias**, because a pooled score hides which part of
the corpus was easy. Three consequences are enforced rather than hoped for:

- **A control never contributes to a quality score.** `heading` queries are
  verbatim in their targets, so that family is a **positive control for the
  harness**. It is reported in its own block and excluded from `overall` — and
  from the lift and shuffle gates that read `overall`, which would otherwise be
  partly measuring a self-match. If it does not score near-perfectly, retrieval
  is broken and every other number in the run is noise.
- **Index rows written alongside the evaluation are their own family.**
  `index-authored` names the rows added by the same change as the evaluator,
  after gaps and scores had been measured. They can encode knowledge of the
  retriever, so they are separated rather than pooled — and reporting them apart
  is what lets a reader see they behave *differently* from inherited rows.
- **Recall means recall.** A judgment may name many documents, and finding one
  of fifteen is not full recall. Recall at a cutoff is the fraction of a
  judgment's targets inside it; `hit@10` reports the weaker "found anything"
  question beside it, so neither is mistaken for the other.

**A family that produces nothing must say why.** A length filter once deleted an
entire source of judgments here and the report simply did not mention it; a
zero-count family now carries a measured reason or the gate fails.

**Artifacts are bound before anything is scored.** Chunks, vectors and judgments
are three files built at three moments; all three carry the same corpus digest
and a mismatch is refused. Source paths and counts do not catch a chapter
rewritten in place, which leaves both unchanged while every judgment in it goes
stale.

**A memory is compared against its own population.** When circular queries are
withheld, the baselines are recomputed over exactly the queries the memory was
allowed to answer. Pooling a baseline over every query while the memory answered
a subset compares two different populations, and the comparison then moves for
reasons that have nothing to do with the memory. The gate checks that every
ranker in that block scored the same number of queries, because a lift is
otherwise an arithmetic between two different question sets.

**A query is projected into the space it will be scored in.** A concept memory
records the model, revision and dimension it was built with, and a query reaches
it through that model — not through whichever embedding set happens to be open.
Today both are `lexical-hash-v1` at 512 dimensions and the distinction costs
nothing; the day one adopts a learned model and the other does not, mismatched
coordinates would rank on nothing and no score would look wrong. So `recall`
refuses a query of the wrong width outright, and the gate exercises the path
with a memory built at a dimension the chunk set does not use.

**A memory says what it was built from.** Its manifest carries a digest of its
own entries and of the index tree they were read from, and loading refuses a
combination that never coexisted. A memory left in a build directory while its
indexes keep being edited answers every question about a table that no longer
exists, and nothing in its own files would show it.

**Absolute scores are reported, never gated.** Freezing today's recall into a
threshold would tune the bar to today's corpus and then cite it as evidence.
[`tools/verify_retrieval.py`](tools/verify_retrieval.py) asserts properties
instead: the control scores near-perfectly, the model clears both a seeded
random ranker and a query-ignoring constant ranker by a wide margin, recall is
monotonic in the cutoff, the run is deterministic — and a **shuffled index must
collapse to the noise floor**, because an evaluation that scores a scrambled
index as highly as a real one measures nothing.

### The concept memory, and what it makes transparent

Direct vector retrieval answers with chunks and a cosine. It cannot say *why*
those chunks: the similarity lives in 512 coordinates that mean nothing to a
reader. [`tools/build_index_memory.py`](tools/build_index_memory.py) adds a
second path through a named intermediate —

```
query ──▶ concept ──▶ documents
          ^^^^^^^ written by a person, in a table, under version control
```

— so every answer arrives with a legible reason: the concept that matched, the
index row that defines it, and the chapters that row names. You can read it,
disagree with it, and fix it by editing a table.

> **What this makes transparent, precisely: the memory, not the model.** Nothing
> here opens up a learned model's weights or explains what a network computes.
> It replaces one opaque hop with two hops through an auditable intermediate,
> which makes *retrieval* reviewable in the way the rest of this repository is
> reviewable. Claiming more than that would be the kind of overclaim this corpus
> exists to refuse.

Two rules keep it honest:

- **The indexes must cover what is retrievable.** A document reachable by search
  but named by no index has no concept leading to it. The gate checks that,
  with directory `README`s and top-level navigation pages as a declared
  exclusion — indexing an index is circular.
- **A memory may not be scored on queries derived from its own rows.** The
  `index` query family comes from the very table the memory embeds; scoring it
  there is asking it to find its own keys. The exclusion is enforced **per
  query** from the memory's declared sources, so a new index file cannot quietly
  reintroduce the circle, and so a family is never discarded wholesale over a
  handful of circular members.

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

## The edge into BCIR's training stack

A corpus that emits records nothing consumes is a database that happens to sit
next to a training system. BCIR already owns every component the corpus would
otherwise reimplement, so [`tools/export_training_examples.py`](tools/export_training_examples.py)
builds edges rather than a second stack:

| BCIR component | What the corpus feeds it |
| --- | --- |
| `hosted.training.data` | source files as `RawDocument`s — it prepares, splits, and reports |
| `hosted.training.bpe` | the prepared corpus — its tokenizer, trained on it |
| `hosted.training.contracts` | Tier-3 records as `SFTExample`s under its schema |
| `hosted.training.pipeline` | its append-only, content-addressed ledger |

Nothing here reimplements a tokenizer, a split policy, a provenance digest, or
an example schema. Every field BCIR's `RawDocument` asks for — identity, text,
source, licence, source digest — the corpus already carried, which is why this
is an edge and not an adapter.

Three rules make the edge safe to consume:

- **One source file, one `RawDocument`.** BCIR's splitter hashes each document
  independently, so handing it *chunks* puts fragments of one chapter on both
  sides of the split — measured here, 89 of 285 sources — and a held-out
  chapter's neighbouring paragraphs end up in training. A document is a file
  with a licence and a digest; a chunk is a retrieval fragment of one, and the
  two are not interchangeable because both carry text.
- **The split is the file.** `SFTExample` has no split field, so examples are
  written one file per split rather than merged with a field a consumer must
  remember to honour. A record filed under the wrong name is a gate failure.
- **The provenance digest covers the whole record.** `provenance_sha256` hashes
  the entire canonical record, not its identity: the gate that verifies an
  answer, its sources, split and difficulty are part of what the example *is*.
  A digest over identity alone stays unchanged when the justification moves, and
  the gate checks both halves — every digest recomputes from its record, and
  perturbing any covered field changes it.
- **Nothing shrinks in silence.** Every document offered is either prepared or
  refused for a reason BCIR names, the manifest records both, and the gate
  refuses a run where the two do not reconcile. A stale chunk build naming a
  source that no longer exists is refused outright rather than exported one
  chapter short.

### Preferences decided by a verifier, not by a rater

Preference data normally needs human labels, which this corpus cannot produce
honestly. It has something better: artifacts whose status a **gate** already
decided. A checked-in solution is IR the assembler accepts; an invalid fixture
is IR the assembler is proven to refuse. "Accepted is preferred to rejected" is
then a legality verdict — the same order this repository applies everywhere,
where legality precedes cost.

Two rules keep that label true:

- **The prompt asks what both sides answer.** It asks for a module the verifier
  accepts, and claims no topical relationship between the two responses. A pair
  whose rejected side does not attempt the prompt's task would be a misleading
  label dressed as data.
- **Only genuinely-refused IR may lose.** This corpus declares some fixtures
  invalid that the assembler nevertheless *accepts* — `semantic-only` and
  `assemble-valid-semantically-risky`, where the defect is meaning rather than
  form. Putting those on the losing side would teach the opposite of the
  intended lesson, so they are excluded by name and the count is reported.

### What the corpus does not claim

BCIR's ledger enforces a real training DAG, and in it `sft` means *a model was
trained*, not *examples exist*. The corpus records `data` and `tokenizer` —
the two artifacts it genuinely produces — and no stage downstream of them. It
still runs no training: these are inputs, and saying otherwise would be the
overclaim the rest of this standard exists to refuse.

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

[`tools/verify_embeddings.py`](tools/verify_embeddings.py) gates the vectors:

1. **Anti-vacuity** — a set covering nothing fails.
2. **Attribution** — model, pinned revision, dimension, and `semantics` present.
3. **No degenerate vector** — finite, right length, non-zero, unit where claimed.
4. **Row alignment** — a chunk's own text must retrieve *that* chunk at distance
   exactly 0. This is the check that catches an off-by-one between the index and
   the vector block: every structural check passes over that defect, and it
   silently returns the neighbouring chunk for the life of the index.
5. **Discrimination** — a model that maps every chunk to nearly one direction
   passes everything above while being useless, so a set whose vectors do not
   separate fails.
6. **Binding and staleness** — the digest of the text actually embedded still
   matches that chunk today.
7. **Declared coverage** — partial is legal, silent partial is not.
8. **Determinism, where claimed** — a set declaring `deterministic: true` is
   gated on byte-identity across two builds. A learned model honestly declaring
   `false` is gated on its invariants instead: contracting for bit-identity
   across arbitrary hardware is a promise the rail cannot keep, and gating on an
   unkeepable promise produces flaky red rather than evidence.
9. **The native differential** — reference ranking versus `bcir_ai_q15_topk`,
   exactly equal.
10. **The rounding convention** — codes round half away from zero, checked
    against BCIR's own function *and* for whether the probes can tell that rule
    from banker's rounding at all. They differ only on exact half-integers, and
    no corpus coordinate lands on one, so a probe set drawn from corpus data
    would pass against either rule and check nothing.
11. **The BCIRQ8 view** — BCIR's quantizer and its embedding-projection kernel
    over the same vectors, held to retrieving an exact match rather than to the
    exact ranking.

Each check in both gates was proved able to fail by injecting the defect it
guards and watching the gate fire.

## What a new subject folder must deliver

A subject is **Tier-1 complete** when its chapters exist and its factual claims
are gated. It is **corpus-complete** when:

- [ ] `training/<subject>/` chapters, examples, and a `tools/` gate set
- [ ] every factual claim reachable from a gate
- [ ] `build_chunks.py --subject <name>` produces chunks with no unverified
      claims of verification
- [ ] `embed_chunks.py --subject <name>` produces an attributed embedding set,
      and `search_chunks.py` retrieves that subject's material for a question it
      should answer
- [ ] `build_distillation.py --subject <name>` produces gate-backed records in
      at least two splits
- [ ] `verify_corpus_records.py` and `verify_embeddings.py` pass, and each new
      gate has been broken once on purpose to prove it fires
- [ ] the subject's `README.md` states its verification boundary — what is
      checked, what is reviewed, and what is neither

The builders discover subjects automatically: any directory under `training/`
that is not `tools/` or `schema/` is treated as a subject. A new folder joins
both tiers as soon as it has content, with no registration step.

## What this standard is not

- It is not an embedding *service*, and it ships no weights. It defines how a
  vector must be attributed, provides a hermetic baseline model so the pipeline
  is gateable anywhere, and refuses to invent a vector or substitute a model it
  could not load. Choosing and obtaining a trained model stays the consumer's
  step, with its own provenance.
- It is not a vector database. It emits a flat, row-major, digest-bound set that
  any index can ingest, plus an exact reference search; it does not build an
  approximate-nearest-neighbour structure or serve queries.
- It is not a fine-tuning harness. It produces records in a standard chat
  format; training with them is out of scope for this repository.
- It does not make unverified prose worthless. Tier 1 carries the explanation
  and the argument, and much of it will never be gateable. It just does not
  become Tier-3 training data.
