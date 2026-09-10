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

*Status: complete. Every slice below has landed and is gated.*

| Slice | State |
| --- | --- |
| 0.1 Restructure to `training/<subject>/` | landed |
| 0.2 Chunk schema + deterministic chunker | landed |
| 0.3 Distillation schema + gate-backed builder | landed |
| 0.4 Record verifier (anti-vacuity, provenance, leakage, determinism) | landed |
| 0.5 Embedding step: a named model fills `embedding`, provenance recorded | landed |
| 0.6 Retrieval evaluation: does a chunk set answer the questions it should? | landed |
| 0.7 Extract the grader/dataset machinery from `llvm/` to `training/` | landed |
| 0.8 Invert the distillation rail; inventory BCIR's ML components | landed |

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
and 0.7 below shares the grading and dataset machinery a training stage would
be evaluated with, without claiming to run one.

**0.7 closed with** `tools/grading.py`, `tools/dataset_export.py`,
`tools/subject_profile.py`, `tools/safe_process.py` and `tools/verify_grading.py`,
plus `llvm/tools/llvm_profile.py` — the LLVM half of what used to be one 669-line
grader and one 280-line exporter under `llvm/tools/`.

The motive was not tidiness. Three tools under `llvm/tools/` each carried their
own `find_tool`, two more their own `model_visible_prompt`, and two their own
`normalized_text`. All of the copies agreed and nothing made them agree; a
fourth `find_tool` turned up in `verify-mlir-lowering.py` when the new gate first
ran, with a *different* search order behind the same name. That is the shape a
second subject would have multiplied.

What a subject now declares is small enough to enumerate: its answer kinds and
extensions, the tools that prove an answer well-formed and their argument
arrays, how binaries are discovered on a host, how a reference solution is
hidden from a model, and whether a submission may be run. Everything else —
confinement, the attempt-tree policy, point allocation, structural and rubric
matching, skip-versus-fail, the report shape, split manifests, checksums,
deterministic JSONL — is inherited.

Two things make the extraction checkable rather than declared. `verify_grading.py`
refuses a subject identifier in the shared rail's code, refuses a second
definition of any policy predicate anywhere under `training/`, and grades a
**synthetic non-LLVM subject** end to end through the real kernel — an
abstraction exercised only by its original caller has not been shown to be one.
And the extraction was proved behaviour-preserving before it landed: the
grader's self-test report, its report over the incomplete-attempt fixtures, its
text rendering, and all three dataset export modes are byte-identical to the
pre-extraction tool, modulo a random temporary-directory name.

Ten faults were injected, one per new check, and all ten fired. One of them found
a defect in the gate itself: with the skip check broken the gate raised an
`IndexError` where a verdict belonged, which is the same fail-open shape the
rails exist to refuse.

**Still open in this area, deliberately.** `build_distillation.py` sits in the
shared `tools/` directory and still knows LLVM: its five record extractors read
`.ll` fixtures, opt goldens and frontend claims by name. That is the same
boundary problem in the opposite direction, and closing it is a separate slice —
the extractors belong to the subject, discovered through a profile, the way
answer kinds now are.

**0.8 closed the other half of the boundary, and corrected the record.**

0.7 moved shared machinery out of the subject. `build_distillation.py` was the
same problem pointing the other way: it sat in the shared `tools/` directory and
knew that invalid fixtures are named `*.invalid.ll.txt`, that optimizer goldens
live under `07-optimization/examples/`, and what an LLVM system prompt says. Its
five extractors and that prompt now live in `llvm/tools/distill_sources.py`,
found by convention; `tools/distillation.py` owns the record contract, the split
policy and the identity digest, and knows nothing about any subject. The 79
records it produces are byte-identical to before.

The contract is enforceable rather than documented: a record cannot be
constructed without naming a gate and a claim, a subject cannot declare an empty
`SOURCES` or two sources with one task name, an empty system prompt is refused,
and the split is re-derived per record from its own source file so two records
from one chapter cannot straddle the boundary. Seven faults injected, seven
caught.

**What BCIR already provides for machine learning, measured.** This roadmap
previously implied the training-time components were out of reach without torch.
That was wrong twice over, and `tools/ml_components.py` now records what is
actually there, with `tools/verify_ml_components.py` resolving every symbol
*and calling it*:

- **BCIR's substrate is tensor-framework free.** Attention, LayerNorm, RMSNorm
  with its gradient, RoPE, the full transformer block with SwiGLU and causal
  masking, activations with an exactness predicate, losses with their gradients,
  reverse-mode autodiff on an explicit tape, SGD/momentum/RMSProp/Adam, a seeded
  training loop, the precision framework (intervals, ULP distance, proved
  quantization and reduction bounds), BCIRQ8, Q4 packing with SmoothQuant, GRU
  and LSTM cells, a frozen-and-hardened MoE gate, and the byte-latent specs —
  all of it runs on flat Python lists with no torch and no numpy. Seventeen
  components are exercised on every run, not merely imported: a normalized row
  really has zero mean, RoPE really preserves each pair's norm, an analytic
  gradient really matches a central difference, a compensated reduction really
  beats a naive one (naive 10 vs exact 11) with a tighter proved bound (10 → 1
  ULP).
- **The hosted stages run where torch is installed** — including here. BCIR's
  `train_sft` consumes this corpus's own `SFTExample` (4 examples, 2 steps,
  loss 4.218 → 3.973) and `train_dpo` its own `PreferenceExample`, after
  refusing a reference model whose parameters can still move. The corpus's
  export is therefore not a file nothing reads: it is the input to a stage that
  demonstrably trains. Note that `stages.py` needs torch to **import**, not
  merely to run — a first draft of the inventory claimed otherwise, having
  measured only on a host where torch was installed, and CI corrected it within
  two minutes. A property measured on one host is a property of that host until
  a second one disagrees.
- **Two components are declared and not exercised, each saying why** in a field
  the gate checks rather than a sentence it would have to read: PPO needs a
  rollout and a reward source the corpus does not produce; bounded reasoning
  search needs a generator, which is the model the corpus has not trained.

  A third was listed here as needing "a teacher this repository does not ship".
  Measured, that was false twice over. The stage constructs and calls no teacher
  at all: its targets are a cosine Gram matrix, and the corpus's own lexical
  provider produces the vectors it is built from. And the pairing was blocked by
  a defect, not by a missing model -- `relational_embedding_targets` returned
  diagonal entries a few ULP above 1.0, which `train_embedding_distillation`
  refuses outright, so BCIR's only cosine-target constructor was incompatible
  with its only consumer for essentially every real input. Both shipped call
  sites escaped it by using exactly-representable toy vectors. Fixed, and
  embedding distillation is now exercised on real corpus text.

The corpus can now feed supervised, preference and embedding-distillation
training end to end. Whether the evaluation should then RUN on a learned
embedding set is a separate decision, and it has its own gate:
[`LEARNED_EMBEDDING_GATE.md`](LEARNED_EMBEDDING_GATE.md) owns the GO/STOP
criteria and the measurement behind them. The short version, including one
retraction:

- **Measured (2026-09-10).** A student trained inside BCIR's hosted stack,
  distilling the corpus's own lexical teacher over all 2013 chunks, reached
  overall MRR@10 0.221 against the baseline's 0.593. Its ceiling is the
  baseline: imitation cannot beat its teacher, and the baseline is the only
  teacher obtainable here.
- **Retracted.** An earlier version of this section named the evaluation's
  calibration as the obstacle -- the discrimination cap and the control floor
  being set around a lexical model. Tripling the training budget cleared the
  control floor (0.983 against 0.95) while overall MRR@10 moved by 0.014. The
  bar was not what was holding the learned rail back; the retrieval was.
  Recorded rather than quietly dropped.
- **Still true, and now the gate's G3 and G4.** The control's ARGUMENT is
  lexical -- "the query is its target's own heading wording, verbatim as a word
  sequence" is a guarantee for a model that ranks on shared wording and nothing
  for a learned encoder -- so the control has to
  be re-argued before its floor is applied to a learned set, which is a
  [`CORPUS_STANDARD.md`](CORPUS_STANDARD.md) decision, not a modelling one. And
  the document-level judgments mean the obvious self-supervised objective trains
  the very operation the evaluator performs: index files are themselves chunk
  documents, so an objective over raw chunk text can memorize 84% of the judged
  mapping without ever constructing a pair, and contamination of that kind would
  leave every existing gate GREEN -- the shuffle control still collapses, the
  lift gates still clear, the digests still match.
- **What the experiment left in the repository.** `relational_reference_loss` in
  `bcir/hosted/training/providers.py`: the loss a student reaches by ignoring the
  teacher entirely, which is the number that says whether a distillation run
  learned anything. The stage reported `0.5135 -> 0.0383` for a run whose
  held-out error never beat the 0.0161 a teacher-free model reaches. The
  ML-component gate now computes both on real corpus text, on every host, and
  refuses a teacher whose Gram carries no structure to distil.

## Phase 1 — complete `llvm/`

*Status: LLVM 15–18 material is complete and gated. The version and MLIR gaps are open.*

| Slice | Work |
| --- | --- |
| 1.1 | Raise the corpus to **LLVM 23**: install the toolchain, re-verify every example, and record where 15→23 changed a documented rule |
| 1.2 | Re-derive the frontend chapters against the **latest Clang**, refreshing the checked claim set |
| 1.3 | Study the **LLVM 23 LangRef** and close the delta: new instructions, attributes, and intrinsics the corpus does not yet teach |
| 1.4 | **Comprehensive MLIR**: dialects, regions, interfaces, the pass infrastructure, bytecode, and the parts `14-` and `18-` only introduce |
| 1.5 | **IRDL**: dialect definition as data, and what it makes checkable that ODS does not |
| 1.6 | ~~**BCIR's own approach**~~ — **landed**: [`llvm/22-bcir-approach/`](llvm/22-bcir-approach) teaches the correspondence, the twelve cost axes, and legality before cost, with every table regenerated from `bcir/` |
| 1.7 | Update `SEMVER.md`: the baseline moves, and the new baseline is enforced by the assembler check already in the frontend gate |

1.1 and 1.2 are prerequisites for **1.3–1.5**, which re-derive chapters against a
specific LLVM release: doing that on a toolchain the corpus does not run would
produce claims no gate checks. Slice 1.6 is not in that class — its claims are
about this repository's own rails, so it landed first and no toolchain bump can
invalidate it.

**A toolchain is reachable here after all.** `apt.llvm.org`, `releases.llvm.org`,
`github.com/llvm` and the `llvm.org`/`mlir.llvm.org` documentation are refused by
this environment's proxy, which is why 1.1 read as blocked. `conda.anaconda.org`
is not refused, and conda-forge ships LLVM/MLIR **23.1.1** — `llvm-as`, `opt`,
`llc`, `lli`, `clang`, `FileCheck`, `mlir-opt`, `mlir-tblgen` and
`mlir-irdl-to-cpp`, extractable without installing conda. Measured against that
toolchain, with nothing changed in the corpus: all 167 standalone examples
assemble and verify, all 18 invalid fixtures are still rejected, the exercise,
mapping, adversarial and opaque-pointer gates pass, and the 15 MLIR examples tier-
grade clean under `LLVM_SUFFIX=-23`. The 15→23 delta, read from the tools rather
than from a changelog the proxy will not serve: `opt`'s base pass names go
273 → 318 → 453 and `llc`'s targets 41 → 44 → 48, and **no corpus material names
any of the 28 pass names that disappeared**. What 1.1 still needs is the CI half —
the corpus job installs Ubuntu's default LLVM, not 23.

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
