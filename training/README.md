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
| training a model | `tools/build_distillation.py` | chat-format records whose answers are gate-checked |

The contract those three share is [`CORPUS_STANDARD.md`](CORPUS_STANDARD.md).
The plan for growing it is [`ROADMAP.md`](ROADMAP.md).

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
Tier 3  distillation   (system, user, assistant) records, each backed by a gate
```

Tier 2 and Tier 3 are **derived**, never hand-written — the only way they stay
in step with the prose.

Three rules do most of the work:

- **A chunk never carries a fabricated embedding.** `embedding` is `null` until
  a named model fills it. A vector with no model behind it is indistinguishable
  downstream from a real one.
- **An attributed vector says what kind of similarity it has.** Every set
  declares `semantics`: `lexical` (a specified function of the surface text) or
  `learned` (trained weights). Naming the model is necessary and not sufficient
  — the two are not interchangeable, and a consumer that cannot tell them apart
  discovers the difference in its own retrieval quality.
- **A distillation record exists only if a gate checks its answer.** No gate, no
  record. Tier 3 therefore grows only as fast as verification does, which is the
  intended pressure.

## Building the records

```bash
# Tier 2 — retrieval chunks (all subjects, or one)
python3 training/tools/build_chunks.py --out build/training/chunks
python3 training/tools/build_chunks.py --subject llvm --stats

# Tier 2 vectors — a named model fills them; nothing else may
python3 training/tools/embed_chunks.py --chunks build/training/chunks \
                                       --out build/training/embeddings

# ask the corpus something
python3 training/tools/search_chunks.py --query "what does musttail require of a call"
python3 training/tools/search_chunks.py --query "opaque pointers" --backend both
python3 training/tools/search_chunks.py --query "opaque pointers" --backend q8

# Tier 3 — gate-backed distillation records
python3 training/tools/build_distillation.py --out build/training/distill

# The gates: schema, provenance, split leakage, determinism, anti-vacuity;
# then attribution, row alignment, discrimination, and the native differential
python3 training/tools/verify_corpus_records.py
python3 training/tools/verify_embeddings.py
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
| [`schema/distill-v1.json`](schema/distill-v1.json) | 3 | supervised fine-tuning |

All three are published contracts. The verifiers check the invariants that would
corrupt training data if violated; the schemas are what an external consumer
validates against.

## Verification boundary

- **Checked:** every gate listed in a subject's `tools/`, plus the record and
  embedding gates above. `llvm/` alone carries example assembly, opaque-pointer
  conformance, exercise solutions, invalid fixtures, optimizer goldens, MLIR
  registry tiers, Clang lowering claims, benchmark analysis, and a built
  out-of-tree pass plugin.
- **Reviewed, not checked:** narrative, design argument, and any material whose
  toolchain the corpus does not assume. Each subject's README says which is
  which.
- **Implemented, not yet exercised:** the `learned` embedding provider. Its
  refusal path is checked — an unloadable model writes nothing, and
  `--require-provider` turns that skip into a failure — but no trained model has
  been run end-to-end through the rail here, so the `deterministic: false`
  branch of the embedding gate has never been taken. It is named rather than
  implied, and closing it belongs with the retrieval evaluation in
  [`ROADMAP.md`](ROADMAP.md) 0.6.
- **Neither:** nothing. A claim that is neither checked nor declared reviewed is
  a defect.

## License

BCIR Non-Commercial License v1.0, `LicenseRef-BCIR-NC-1.0` — see the root
[`LICENSE`](../LICENSE). Each subject folder carries its own attribution notice
for vendored or derived material.
