# Training corpus — expansion roadmap

## Mission

Grow `training/` from a single LLVM context pack into a **computer-science and
programming training corpus for humans, agents, and new models** — bottom-up
from transistors to application backends, with every subject reaching the same
three-tier standard.

The corpus is not part of BCIR. Nothing here is a build dependency of `bcir/`,
`mlir/`, or `runtime/`, and the direction of reference is one-way: the corpus
may cite the implementation as a worked example; the implementation never
imports the corpus.

## What already changed

`llvm-training/` became `training/`, and the existing corpus moved to
`training/llvm/` — which now owns LLVM, MLIR/IRDL, and every other
compiler-infrastructure subject. Everything above the subject folders is shared:

```
training/
├── CORPUS_STANDARD.md      the three-tier contract every subject must reach
├── ROADMAP.md              this file
├── schema/                 chunk-v1.json, embedding-set-v1.json, eval-set-v1.json,
│                           distill-v1.json
├── tools/                  build_chunks.py, embed_chunks.py, search_chunks.py,
│                           bcir_native.py, build_index_memory.py,
│                           build_eval_queries.py, evaluate_retrieval.py,
│                           build_distillation.py, verify_corpus_records.py,
│                           verify_embeddings.py, verify_retrieval.py
├── llvm/                   Phase 1 — compilers (the existing corpus)
├── hardware/               Phase 2 — transistors to instruction sets
├── systems/                Phase 3 — C/C++/CUDA, build systems, drivers, kernels
├── low-level/              Phase 4 — representation, encodings, machine code
├── formats/                Phase 5 — Unicode, XML, HTML
├── languages/              Phase 6 — high-level languages
├── data/                   Phase 7 — databases and query languages
└── backends/               Phase 8 — runtimes and frameworks
```

The seven new folders carry a scope statement and nothing else. That is
deliberate: an empty folder that reads like a chapter is a lie, and the builders
skip a subject with no content rather than emitting an empty record file.

## The standard, in one paragraph

Every subject must reach **Tier 3**. Tier 1 is prose whose factual claims are
gated. Tier 2 is deterministic, line-traceable retrieval chunks that never
fabricate an embedding vector. Tier 3 is chat-format supervised records, each
one backed by a gate that checks its answer — no gate, no record. The full
contract is [`CORPUS_STANDARD.md`](CORPUS_STANDARD.md); the machinery is in
`tools/` and is already running over `llvm/`.

The load-bearing consequence: **Tier 3 grows only as fast as verification does.**
A subject with beautiful prose and no gates produces no training data. That is
the intended pressure.

---

## Phase 0 — corpus infrastructure

*Status: the first slice has landed.*

| Slice | State |
| --- | --- |
| 0.1 Restructure to `training/<subject>/` | landed |
| 0.2 Chunk schema + deterministic chunker | landed |
| 0.3 Distillation schema + gate-backed builder | landed |
| 0.4 Record verifier (anti-vacuity, provenance, leakage, determinism) | landed |
| 0.5 Embedding step: a named model fills `embedding`, provenance recorded | landed |
| 0.6 Retrieval evaluation: does a chunk set answer the questions it should? | landed |
| 0.7 Extract the grader/dataset machinery from `llvm/` to `training/` | open |

**0.5 closed with** `tools/embed_chunks.py`, `tools/search_chunks.py`,
`tools/verify_embeddings.py`, and `schema/embedding-set-v1.json`. Vectors are
written only by a named `Provider`, land in a sidecar set rather than inside the
chunk records, stay out of git, and carry a pinned `revision` and a `semantics`
declaration — `lexical` or `learned` — because naming a model is necessary and
not sufficient to say what a vector means. A model that cannot be loaded is a
skip that writes nothing.

Retrieval came with it, because a rail that builds vectors nothing queries has
not been shown to work: the vectors are projected into BCIR's own symmetric Q15
code space and searched by `bcir_ai_q15_topk` from `runtime/c` — the same
exact-integer top-k kernel BCIR uses for its optimization memory — with a
pure-Python reference as the definition and an exact differential between the
two. The corpus is still never a build dependency of BCIR; the import is lazy
and the reference path always suffices.

**Still open in this area, deliberately:** the corpus ships **no trained model**.
The hermetic `lexical-hash-v1` baseline exists so the pipeline is gateable on any
host, and it declares itself lexical precisely so nobody mistakes it for
semantic retrieval. Running a learned model is implemented and unproven here —
see 0.6.

**0.6 closed with** `tools/build_eval_queries.py`, `tools/evaluate_retrieval.py`,
`tools/build_index_memory.py`, `tools/verify_retrieval.py`, and
`schema/eval-set-v1.json`.

Judgments are **derived, never authored for the evaluation**: an index row
already says which chapters cover a concept, a prose link already describes what
it points at. Every family declares its bias, one family is trivial on purpose
as a harness control, and a family that yields nothing must carry a measured
reason — a length filter deleted an entire source of judgments on the first run
and the report did not mention it.

No absolute score is gated. The gate asserts properties instead: the control
scores near-perfectly, the model clears both a random and a query-ignoring
baseline by a wide margin, recall is monotonic, the run is deterministic, and a
**shuffled index must collapse to the noise floor** — an evaluation that cannot
tell a scrambled index from a working one measures nothing.

**The indexes grew to make this possible**, and the growth is now a checked
property: every retrievable teaching document must be named by some index, with
directory `README`s and navigation pages as a declared exclusion. Five index
files were added covering backend/JIT, performance and evidence, frontends and
production lowering, concurrency and atomics, and exercises by skill.

**An index-backed concept memory** turns those rows into a second retrieval
path: `query → concept → documents`, where the concept is human-written and the
answer carries the index row that produced it. It makes the *memory* auditable —
not a learned model's internals, and the corpus says so plainly. A memory is
never scored on queries derived from its own rows; the exclusion is enforced per
query from the memory's declared sources.

**Corrected after review.** An adversarial review of the first cut found ten
defects, and several were places where the code contradicted this document:
the harness control was pooled into the headline score it is declared never to
measure; `recall` was really a hit rate for the 119 judgments that name more
than one document; index rows authored alongside the evaluator were counted as
inherited judgments; the memory's lift was measured against a baseline drawn
from a different population; and the corpus/vector/judgment artifacts were not
bound, so a stale combination scored silently. Each is now fixed and gated.

Two of the corrections changed conclusions, which is the point of measuring:
the `index-authored` rows score *worse* than the inherited ones rather than
better, and over the whole population the memory may answer, direct retrieval
outranks it — the memory's advantage is real but confined to descriptive
queries.

Proving each new check could fail found three more holes, all of the same shape:
a fix that was correct by construction and therefore ungated. Querying the
memory in its own space produced identical numbers to querying it in the chunk
set's, because today the two spaces are the same one — so `recall` now refuses a
query of the wrong width outright and the gate exercises it against a memory
built at another dimension. The provenance digest's *scope* was an assertion
until the gate recomputed it from the records and perturbed each covered field.
And the memory's entry-count check could not see a swap that preserved the
count, or an index edited afterwards, so the memory now carries digests of both
and refuses a combination that never coexisted. Reading the diff back
adversarially added two more: the export claimed to report what BCIR refused and
recorded none of it, and a chunk build naming a deleted source would have
exported one chapter short without a word. Fifteen injected faults, all caught.

**Still open in this area, deliberately.** The judged set is derived from one
author's corpus, so it shares vocabulary with what it judges; an independent
judge would be better and does not exist here. The concept memory is scoreable
on only the minority of queries that are not circular, which is a small sample
and reported as one. And retrieval quality is still measured against a
**lexical** baseline: `0.5` left the learned rail implemented and unexercised,
and this phase gives it the metric it was missing rather than the model.

### The edge into BCIR's training stack

Landed alongside 0.6, and the reason the corpus stops being a passive database.
`tools/export_training_examples.py` feeds BCIR's own `hosted.training` stack —
its provenance-preserving corpus preparation, its byte-fallback tokenizer, its
`SFTExample`/`PreferenceExample` contracts, and its content-addressed pipeline
ledger — instead of emitting JSON nothing in this repository consumes. Those
modules are deliberately tensor-framework free, so the edge is gated in CI with
no torch on the runner.

Preference pairs are decided by a **verifier rather than a rater**: a checked-in
solution assembles, an invalid fixture is proven refused. Fixtures declared
invalid that nevertheless assemble are excluded by name and counted, because
putting valid IR on the losing side of a pair is a false label no schema catches.

The corpus records only `data` and `tokenizer` in BCIR's ledger. A first cut
appended `sft` and the ledger rejected it — correctly, since there `sft` means a
model was trained. These are training inputs; the corpus runs no training stage,
and Phase 0.7 is where the shared machinery for one would live.

## Phase 1 — complete `llvm/`

*Status: LLVM 15–18 material is complete and gated. The version and MLIR gaps are open.*

| Slice | Work |
| --- | --- |
| 1.1 | Raise the corpus to **LLVM 23**: install the toolchain, re-verify every example, and record where 15→23 changed a documented rule |
| 1.2 | Re-derive the frontend chapters against the **latest Clang**, refreshing the checked claim set |
| 1.3 | Study the **LLVM 23 LangRef** and close the delta: new instructions, attributes, and intrinsics the corpus does not yet teach |
| 1.4 | **Comprehensive MLIR**: dialects, regions, interfaces, the pass infrastructure, bytecode, and the parts `14-` and `18-` only introduce |
| 1.5 | **IRDL**: dialect definition as data, and what it makes checkable that ODS does not |
| 1.6 | **BCIR's own approach**: how a correspondence IR relates to LLVM IR and MLIR — the cost model, the verifier laws, and why legality precedes optimization |
| 1.7 | Update `SEMVER.md`: the baseline moves, and the new baseline is enforced by the assembler check already in the frontend gate |

1.1 and 1.2 are prerequisites for the rest: re-deriving chapters against a
toolchain the corpus does not yet run would produce claims no gate checks.

**Version discipline.** The corpus currently declares LLVM 15 as its floor and
enforces it by assembling snapshots with the oldest available assembler. Raising
the ceiling to 23 does not raise the floor automatically — moving the floor
invalidates checked-in artifacts, so it is its own slice with its own
regeneration pass.

## Phases 2–8 — the subject ladder

Each phase opens a folder that already has a scope statement. The ordering is
dependency-driven, bottom-up, and phases may overlap where their dependencies
allow.

| Phase | Folder | Opens after | Theme |
| --- | --- | --- | --- |
| 2 | `hardware/` | nothing | circuits, CMOS, VLSI, SPICE, Verilog/VHDL, ISAs, memory architecture |
| 3 | `systems/` | 2 | C, C++, CUDA, Make/CMake, shell, drivers, microkernel, Linux |
| 4 | `low-level/` | 2, and 1 for the IR family | HALs, IRs, machine code, wire encodings, bytecode targets, networking |
| 5 | `formats/` | 4 | Unicode, XML, HTML |
| 6 | `languages/` | 3 | Python, Java, JS/TS, Rust, Go, Swift, PHP, R, Fortran, VB, Delphi, CSS |
| 7 | `data/` | 6, 4 | SQL, NoSQL, storage engines, query planning |
| 8 | `backends/` | 6, 7 | Node.js, React, service composition |

### The gate on opening a folder

A subject folder moves from PLANNED to open only when:

1. Its **reference manuals are inventoried** — named, versioned, dated. Each
   folder's README already lists its primary sources. Lessons cite normative
   references, not recollection.
2. A **verification plan** exists: what tool checks the claims, and what it
   does when that tool is absent. Each README states one.
3. The **first slice is gateable**. A phase does not open with a survey chapter;
   it opens with material something can check.

That order is what keeps the corpus from becoming an encyclopedia of confident
prose. It is the same rule the LLVM corpus already lives under, applied before
the writing starts rather than after.

### Where this corpus can be unusually good

Three subjects where BCIR's existing work is a real advantage, not a claim:

- **Machine code programming on modern hardware** (`low-level/`). Largely
  abandoned as a taught subject. This repository already reasons about
  instruction selection, ABI, and encoding as first-class concerns.
- **Wire encodings** (`low-level/`). The ASN.1 portfolio here spans X.680–X.697
  with canonical-form discipline and fuzzed decoders. That is a stronger
  foundation than most encoding courses start from.
- **Intermediate representations as a family** (`low-level/`, `llvm/`). BCIR is
  a correspondence IR with a cost model and verifier laws; teaching IRs
  comparatively is the thing this repository is structurally best placed to do.

## Governance carried over

These are inherited from the LLVM corpus and apply to every subject:

- **A claim belongs to a gate, or it is prose.** Chapters may explain and argue
  freely; a fact a reader could act on needs a check.
- **Prove every gate can fail.** Break the thing it guards, watch it fire,
  restore. A gate that has never failed has not been shown to work.
- **Honest skips.** A missing tool is a skip that says so, never a pass. A skip
  is never counted as evidence.
- **Counts live in generated output**, never in hand-written prose.
- **Declared non-goals are first-class.** Saying what a folder will not cover,
  with the reason, is part of finishing it.
- **Measured, modelled, estimated** are three different words with three
  different meanings, and a table that mixes them labels every row.

## Non-goals

- **Not a fine-tuning harness.** The corpus emits standard-format records;
  training with them happens elsewhere.
- **Not an embedding service, and it ships no weights.** It defines how a vector
  must be attributed, provides a hermetic baseline so the pipeline is gateable
  anywhere, and refuses to invent a vector or to substitute a model it could not
  load.
- **Not a replacement for primary references.** Every subject cites its
  normative sources; the corpus teaches how to read them, and does not
  paraphrase them into a substitute.
- **Not a certification syllabus.** The ordering is dependency-driven, not
  curriculum-driven, and it optimises for what can be checked.
- **Not part of BCIR.** Neither may depend on the other.

## How to open a subject folder

1. Inventory its reference manuals into the folder's README; pin versions.
2. Write the first gateable slice: material plus the tool that checks it.
3. Break the gate on purpose; confirm it fires; restore.
4. Run `python3 training/tools/verify_corpus_records.py` — the builders pick the
   subject up automatically, with no registration step.
5. Record the folder's verification boundary in its README: what is checked,
   what is reviewed, and what is neither.
6. Move the phase row here from PLANNED to open, and say what closed it.
