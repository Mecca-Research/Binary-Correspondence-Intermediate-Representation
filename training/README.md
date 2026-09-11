# `training/` — a corpus for humans, agents, and model training

> **Scope:** this directory is a **training corpus**, not part of the BCIR IR.
> The IR is realized by the executable oracle in [`../bcir/`](../bcir) and the
> MLIR law in [`../mlir/`](../mlir). Nothing here is built into those
> components, and neither may depend on the other. See
> [`../AGENTS.md`](../AGENTS.md) and
> [`../docs/BCIR_Repo_Structure.md`](../docs/BCIR_Repo_Structure.md).

One body of knowledge, delivered in three shapes:

| If you are | Start at | You get |
| --- | --- | --- |
| a person learning | the subject's `README.md` and curriculum | chapters, worked examples, exercises |
| an agent doing a task | `tools/search_chunks.py`, or the subject's index | a scoped passage that traces to a line |
| training a model | `tools/export_training_examples.py` | BCIR `SFTExample`s and verifier-decided preference pairs |

The reference for all of it — the table, the predicate grammar, the artifact
formats, the planner, the retrieval backends and the ML rails — is
[`TRAINING_LANGREF.md`](TRAINING_LANGREF.md), which is *normative*: the tools
under [`tools/`](tools) are its conformance oracle, and
`tools/verify_langref.py` reconciles the two on every CI run.

The contract the three audiences share is
[`CORPUS_STANDARD.md`](CORPUS_STANDARD.md). The plan for growing it is
[`ROADMAP.md`](ROADMAP.md). Which embedding provider the retrieval evaluation
runs on is decided in
[`LEARNED_EMBEDDING_GATE.md`](LEARNED_EMBEDDING_GATE.md). How the retrieval rail
became a database — measured against nine proven engines, then built out as the
S1–S19 ladder — and *why* each mechanism is there is
[`DATABASE_ROADMAP.md`](DATABASE_ROADMAP.md).

A subject brings its own material and its own gates; it does not bring its own
grader. Grading, dataset export and bounded tool execution live in
[`tools/`](tools) and are shared, with each subject declaring only what is
genuinely its own — see *What a subject owns, and what it inherits* in the
standard.

## Subjects

| Folder | Status | Covers |
| --- | --- | --- |
| [`llvm/`](llvm) | **open** | LLVM IR, Clang frontend, MLIR bridge and lowering, backend/JIT, optimization, performance methodology, BCIR mapping |
| [`hardware/`](hardware) | planned | circuits, CMOS, VLSI, SPICE, Verilog/VHDL, ISAs, memory architecture |
| [`systems/`](systems) | planned | C, C++, CUDA, Make/CMake, shell, drivers, microkernel, Linux |
| [`low-level/`](low-level) | planned | HALs, IRs, machine code, wire encodings, bytecode targets, networking |
| [`formats/`](formats) | planned | Unicode, XML, HTML |
| [`languages/`](languages) | planned | high-level programming languages |
| [`data/`](data) | planned | SQL, NoSQL, storage engines, query planning |
| [`backends/`](backends) | planned | Node.js, React, service composition |

A **planned** folder holds a scope statement and no lessons. That is not a
placeholder for content that exists elsewhere — there is nothing there yet, and
the folder says so.

## The three tiers

```
Tier 1  prose          hand-written chapters; a factual claim belongs to a gate
Tier 2  retrieval      deterministic chunks, line-traceable, embedding-ready
        + vectors      attributed embedding sets, searchable exactly
        + judgments    derived queries that measure whether retrieval works
        + memory       index concepts, so an answer carries its reason
Tier 3  distillation   (system, user, assistant) records, each backed by a gate
```

Tier 2 and Tier 3 are **derived**, never hand-written — the only way they stay
in step with the prose.

Four rules do most of the work:

- **A chunk never carries a fabricated embedding.** `embedding` is `null` until
  a named model fills it. A vector with no model behind it is indistinguishable
  downstream from a real one.
- **An attributed vector says what kind of similarity it has.** Every set
  declares `semantics`: `lexical` (a specified function of the surface text) or
  `learned` (trained weights). Naming the model is necessary and not sufficient
  — the two are not interchangeable, and a consumer that cannot tell them apart
  discovers the difference in its own retrieval quality.
- **Retrieval is measured, and never against questions written for it.** Every
  judgment is derived from a binding the corpus already made — an index row, a
  prose link — and every query family declares why it is easier or harder than a
  real question. Scores are reported; the gate asserts properties, including
  that a shuffled index collapses to noise.
- **A distillation record exists only if a gate checks its answer.** No gate, no
  record. Tier 3 therefore grows only as fast as verification does, which is the
  intended pressure.
- **The corpus feeds BCIR, it does not shadow it.** Source files go through
  BCIR's own corpus preparation, its tokenizer, and its example contracts;
  preference pairs are decided by a verifier rather than a rater. The corpus
  records the two artifacts it produces — `data` and `tokenizer` — and claims no
  training stage, because it runs none.
- **Every artifact says what it was built from, and a mismatch is refused.**
  Chunks, vectors, judgments and the concept memory carry digests binding them
  to the corpus and to each other. A chapter rewritten in place changes neither
  a path nor a count, so a stale combination would otherwise score in silence.

## Building the records

```bash
# Tier 2 — retrieval chunks (all subjects, or one)
python3 training/tools/build_chunks.py --out build/training/chunks
python3 training/tools/build_chunks.py --subject llvm --stats

# Tier 2 vectors — a named model fills them; nothing else may
python3 training/tools/embed_chunks.py --chunks build/training/chunks \
                                       --out build/training/embeddings

# the catalog: where each row's bytes are, and which rows a predicate admits
python3 training/tools/catalog.py --out build/training/catalog --stats

# ask the corpus something
python3 training/tools/search_chunks.py --query "what does musttail require of a call"
python3 training/tools/search_chunks.py --query "opaque pointers" --backend both
python3 training/tools/search_chunks.py --query "opaque pointers" --backend q8

# ... and narrow it, project it, count it, or ask how it will be answered
python3 training/tools/search_chunks.py --query "jit" --where "source_path^=training/llvm/12-backend-jit"
python3 training/tools/search_chunks.py --query "sparsity" --where "kind=code" --select "source_path,title"
python3 training/tools/search_chunks.py --query "lowering" --backend auto --objective latency --explain

# aggregates, answered from the index rather than by scanning
python3 training/tools/search_chunks.py --count --where "subject=llvm" --where "kind=code"
python3 training/tools/search_chunks.py --count --group-by kind
python3 training/tools/search_chunks.py --count --distinct language
python3 training/tools/search_chunks.py --count --stats char_count --where "kind=code"

# a relational scan: no query, no vectors -- rows by predicate, ordered by a column
python3 training/tools/search_chunks.py --where "kind=code" --order-by "char_count:desc" \
                                        --top-k 5 --select "source_path,char_count"

# a value that contains a comma is quoted, whole. Unquoted, a comma separates a set:
#   --where 'subject=llvm,data'      is  subject IN (llvm, data)
#   --where 'kind="code,prose"'      is  kind = the one value "code,prose"
# Inside quotes \" is a quote and \\ is a backslash, and those are the only escapes.
# A quote inside an unquoted value, or text after a closing quote, is refused rather
# than guessed -- and `=` and `!=` read one value grammar, so they stay complements.
python3 training/tools/search_chunks.py --count --where 'language="c++"'

# rebuild only what changed, and keep the corpus you measured against
python3 training/tools/embed_chunks.py --chunks build/training/chunks \
                                       --out build/training/embeddings --incremental
python3 training/tools/generations.py --publish --label "before the rewrite"
python3 training/tools/generations.py --list

# what the planner decided, pinned, so a change to it cannot be silent. A moved
# decision is a finding, not a failure: re-record deliberately and read the diff.
python3 training/tools/plan_baseline.py --compare
python3 training/tools/plan_baseline.py --record

# how good is that retrieval, really? (judged queries, derived not authored)
python3 training/tools/build_eval_queries.py --out build/training/eval
python3 training/tools/evaluate_retrieval.py

# an index-backed concept memory: retrieval you can read
python3 training/tools/build_index_memory.py --out build/training/memory
python3 training/tools/build_index_memory.py --recall "is my speedup real or just noise"

# Tier 3 — gate-backed distillation records
python3 training/tools/build_distillation.py --out build/training/distill

# feed BCIR's own training stack: its corpus prep, tokenizer, example contracts
python3 training/tools/export_training_examples.py --out build/training/export

# The gates: schema, provenance, split leakage, determinism, anti-vacuity;
# then attribution, row alignment, discrimination, and the native differential
python3 training/tools/verify_corpus_records.py
python3 training/tools/verify_embeddings.py
python3 training/tools/verify_retrieval.py
python3 training/tools/verify_training_export.py

# the database layer: kernel cache, lazy columns, row locator, predicate, planner,
# parts, generations -- each check asserting both that the cheap path agrees with
# the expensive one and that it actually skipped the work
python3 training/tools/verify_database.py

# the shared rail stays shared, and the ML inventory stays true
python3 training/tools/verify_grading.py
python3 training/tools/verify_ml_components.py
```

The arithmetic is BCIR's own. `--backend both` runs the pure-Python reference
*and* `bcir_ai_q15_topk` from `runtime/c`, and requires them to agree exactly;
`--backend q8` scores a BCIRQ8 view — built by `bcir_ai_quantize_q8_f64` and
read by `bcir_ai_q8_rows_dot_f64`, the kernel whose header names embedding
projections as its purpose — at half the bytes per coordinate.

The corpus is never a build dependency of BCIR: those imports are lazy, reached
only when a native backend is asked for, and the reference backend is always
sufficient on its own.

Both builders discover subjects automatically: any directory under `training/`
that is not `tools/` or `schema/` is a subject, and a new folder joins both
tiers as soon as it has content. There is no registration step.

Outputs are build artifacts, not committed: the verifier builds twice and
compares digests, so determinism is checked rather than assumed.

## Schemas

| File | Tier | Consumer |
| --- | --- | --- |
| [`schema/chunk-v1.json`](schema/chunk-v1.json) | 2 | retrieval indexes, RAG pipelines |
| [`schema/embedding-set-v1.json`](schema/embedding-set-v1.json) | 2 | vector indexes |
| [`schema/eval-set-v1.json`](schema/eval-set-v1.json) | 2 | retrieval evaluation |
| [`schema/distill-v1.json`](schema/distill-v1.json) | 3 | supervised fine-tuning |

All four are published contracts. The verifiers check the invariants that would
corrupt training data if violated; the schemas are what an external consumer
validates against.

## Verification boundary

- **Checked:** every gate listed in a subject's `tools/`, plus the record,
  embedding, retrieval, and training-export gates above — which also check that
  every retrievable teaching document is named by some index, and that no
  preference pair puts assembler-valid IR on its rejected side. `llvm/` alone carries example assembly, opaque-pointer
  conformance, exercise solutions, invalid fixtures, optimizer goldens, MLIR
  registry tiers, Clang lowering claims, benchmark analysis, and a built
  out-of-tree pass plugin.
- **Reviewed, not checked:** narrative, design argument, and any material whose
  toolchain the corpus does not assume. Each subject's README says which is
  which.
- **Implemented, not yet exercised:** the `learned` embedding provider. Its
  refusal path is *written* — an unloadable model writes nothing, and
  `--require-provider` turns that skip into a failure — but nothing exercises it:
  no CI job, script or test passes that flag, and no trained model has been run
  end-to-end through the rail here, so the `deterministic: false` branch of the
  embedding gate has never been taken. Written and checked are different words,
  and this entry used to use the wrong one. Phase 0.6 built the metric
  a learned model would be judged by; it did not obtain the model, and this
  corpus still reports a **lexical** baseline only.
- **Neither:** nothing. A claim that is neither checked nor declared reviewed is
  a defect.

## License

BCIR Non-Commercial License v1.0, `LicenseRef-BCIR-NC-1.0` — see the root
[`LICENSE`](../LICENSE). Each subject folder carries its own attribution notice
for vendored or derived material.
