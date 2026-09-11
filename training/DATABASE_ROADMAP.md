# The retrieval rail as a database — comparative analysis and the S-ladder

## The question

`training/` holds a corpus, a chunk builder, an embedding set, three retrieval
backends, an index-backed concept memory, an evaluation harness and a
distillation export. That is storage and ranking. It is not yet a database: it
executes one plan very well and cannot be asked for a different one.

Nine proven projects were read against it to decide what to take:
ClickHouse, faiss, DuckDB, TileDB, ScyllaDB, Ray, Dask, webdataset and the
repository at `mindsdb/mindshub`. The question put to each was narrow — *which
mechanism, specifically, would modernize this rail* — and the answer had to
name a file, a line and a number.

## Verdict

| Decision | Outcome |
| --- | --- |
| Fork | none of the nine |
| Vendor source | none of the seven that permit it |
| Reimplement | one mechanism, in its cheap half (**S2**) |
| Build | six slices, all native, priced against the measurement below |

The constraint is not licence. ClickHouse is Apache-2.0; faiss, DuckDB, TileDB
and MindsHub are MIT; Dask and webdataset are BSD-3 — every one of those
permits incorporation into a work under any licence with the notice retained,
and `LicenseRef-BCIR-NC-1.0` restricts outsiders commercializing BCIR without
placing any constraint on what BCIR takes in. Only ScyllaDB is excluded on
licence: since 2025 it ships under the ScyllaDB Software License Agreement
v1.1, which is source-available rather than open source. `mindsdb/mindshub` is
excluded on shape — it now describes an agent workspace, not a database.

The constraint is scale, and it is measured.

## The measurement that decides it

One query against `build/training/embeddings/lexical-hash-v1` (2,012 rows x 512
dims), in-process, warm, best of 3-5, on the session host at the commit that
introduced this file. Every row is metric class `wall` — indicative, never
gating, per `docs/PERFORMANCE_AUDIT.md`'s classification.

| Phase | Best (ms) | What it does |
| --- | ---: | --- |
| `NativeAIKernels.build()` | 281.9 | runs the C compiler, every process |
| `EmbeddingSet(root)` | 77.1 | of which 61.95 computes `row_squares` for all rows |
| `load_chunk_texts()` | 47.7 | parses every chunk record to render `top_k` snippets |
| `topk_reference` | 54.1 | exact Q15 squared-L2 in pure Python |
| `topk_q8` | 4.76 | BCIRQ8 rows-dot through BCIR's C kernel |
| **`topk_native`** | **0.89** | exact integer top-k through `bcir_ai_q15_topk` |

CLI wall clock, best of 5: bare interpreter 13.9 ms; `--backend reference`
241.0 ms; `--backend native` 443.1 ms; `--backend q8` 766.9 ms.

Two facts follow, and together they are the whole analysis:

1. **The answer costs 0.2% of the command.** The scan is 0.89 ms. Nothing any
   of the nine projects does to a scan is worth anything here.
2. **The backend with the fastest scan has the slowest command**, because
   `NativeAIKernels.build` invokes the compiler unconditionally and
   `os.replace`s a byte-identical library. There is no freshness check.

## What the rail already does better

Each item below is a property the compared project measurably lacks, found by
agents reading that project's source.

- **Two implementations of one ranking, compared for bit equality.**
  `--backend both` requires identical indices *and* identical squared
  distances; exact integer arithmetic and the total tie order
  `(squared_distance, index)` are what make that expressible. faiss cannot: its
  reduction order varies with SIMD width and thread count.
- **Refusal over a silently wrong answer.** The eligibility mask is validated
  to 0/1 and anything else returns `BCIR_AI_INVALID_ARGUMENT`; the rail raises
  `BackendUnavailable` rather than switching backends. faiss's
  `IDSelectorBitmap::is_member` returns *false* for an out-of-range id.
- **Retrieval that states its reason.** `build_index_memory.py` routes
  query -> human-authored named concept -> documents and prints the concept that
  fired. The table is version-controlled and corrected by editing a row. None
  of the nine has this.
- **Atomic publish, already correct.** A single `os.replace(temporary, target)`
  for both the embedding set and the compiled kernel. TileDB's commit-marker
  protocol would be a second, weaker convention beside it — and TileDB cannot
  build the same array twice, because its names carry wall-clock time and a
  random label.
- **Gates that run in the shipping path.** DuckDB's `RowGroup::Verify()` is
  wrapped in `#ifdef DEBUG`; release builds trust their statistics.
- **Determinism as a build property.** `-ffp-contract=off`, sequential f64
  accumulation, no OpenMP. Ray Data defaults `preserve_order` to `False`.

## The prototype evaluated against SQL operations

The left column is what the audit found; the right is what the S-ladder left
behind it.

| Operation | Was | Is | How |
| --- | --- | --- | --- |
| `ORDER BY <distance> LIMIT k` | strong | strong | three backends, two bit-compared on real data — and the comparison now survives filtering |
| `EXPLAIN` | strong | strong | `--recall` names the concept that fired; `--explain` prints every candidate plan, its twelve-axis cost, and why one won |
| `BEGIN` / `COMMIT` | strong | strong | single-writer atomic publish, now for the catalog and every generation too |
| `WHERE <predicate>` | absent | **built** | four operators, ANDed, resolved through an inverted index and applied inside the kernel scan |
| `SELECT <columns>` | absent | **built** | `--select` projects named columns; nothing else is materialized |
| `JOIN` | absent | **built** | the catalog binds the chunk table's columns to the embedding set's row number, so `kind`, `title` and `heading_trail` are one lookup |
| `GROUP BY` / aggregate | absent | **built** | `--count`, `--group-by`, `--distinct` and `--stats` (MIN/MAX/SUM/AVG over a packed numeric column) — answered from statistics when unfiltered, by intersecting postings when not |
| `ORDER BY <column>` | absent | **built** | `--order-by col[:desc]` without a query: a relational scan, total and stable, with absent measurements sorting last in both directions |
| `LIMIT` / `OFFSET` | partial | **built** | `--top-k` and `--offset`, a window on one order rather than two orders |
| `INSERT` / incremental | absent | **built** | parts as the coarse filter, per-row text digests as the fine one |
| `AS OF` / time travel | absent | **built** | content-addressed generations; an old one is reopened and reproduces its answer |
| prepared / cached plan | absent | **built** | the kernel is content-addressed and compiled once |
| cost-based planning | elsewhere | **built** | `plan.py`, on the same twelve axes, with legality decided first |

What has *not* changed is the shape of the strengths: the rail is still strong
wherever an answer must be trustworthy. The predicate narrows without touching
the ranking's definition, the planner refuses a lossy backend for an exactness
request at any price, and every new artifact publishes atomically or not at all.

## The S-ladder — landed

All six landed, gated by `training/tools/verify_database.py` (222 checks). Every
check was injected and watched to fire before its fix went in, because a slice
without a failable gate cannot be shown to have landed (`docs/security/laws.md`
L2, L11).

Measured after the slices, on the 2,215-row corpus, warm, best of 3–5. Metric
class `wall` — indicative, never gating.

| Slice | Before | After | |
| --- | ---: | ---: | --- |
| S1 kernel build, per process | 532.6 ms | 1.90 ms | 280x |
| S2 `EmbeddingSet(root)` | 77.06 ms | 10.90 ms | 7x |
| S3 text for `k` results | 45.10 ms | 1.06 ms | 43x |
| S4 scan at a narrow predicate | 60.4 ms | 0.21 ms | 288x |
| S5 rebuild after a one-row edit | 4,876 ms | 1,073 ms | 4.5x |
| S6 reading a past generation | not possible | addressable | — |

The query layer those slices needed is in `training/tools/plan.py`: legality
first, then a price on the same twelve axes `bcir/asn1/selection.py` prices an
encoding rule on. `search_chunks.py` gained `--where`, `--select`, `--count`,
`--group-by`, `--distinct`, `--stats`, `--order-by`, `--offset`, `--explain`,
`--backend auto` and `--objective`, all vacuous by default so the pre-slice invocations produce byte-identical output.

### S1 — content-addressed kernel cache

Hash the C source, the compiler identity and the flag vector; skip the compiler
when that hash matches what produced the library already on disk. Keep the
atomic replace exactly as it is.

*Payoff:* −281.9 ms per process, 63.6% of the native command.
*Gate:* touch the source, assert a rebuild; touch nothing, assert byte identity
and that no compiler was invoked.

### S2 — lazy derived columns

Make `row_squares` a cached property on `EmbeddingSet`, and mirror it in
`build_index_memory.py`, which holds a second copy of the identical recompute.
This is the one nominated mechanism that survived adversarial review, and it
survived in the form needing no schema edit, no new artifact and no migration:
`row_squares` has two consumers and the `native` and `q8` backends are neither.

Persisting the column stays deferred. It only ever helps the backend that reads
it, and that backend stops being the one you run somewhere above 10-20k rows —
after which the persisted column is permanent dead weight in the format.

*Payoff:* −61.95 ms for `native` and `q8`; ~97% of it for the discrimination
gate, which indexes 48 sampled rows.
*Gate:* assert the column is not computed when the selected backend never reads
it.

### S3 — late materialization of chunk text

Build a byte-offset index over the chunk files at embed time and seek to the
`k` rows the ranking selected, instead of parsing every record to render five.

*Payoff:* −47.5 ms, and it stops growing with the corpus.
*Gate:* assert the `k`-row path reads fewer bytes than the file and returns
text identical to the full-scan path.

### S4 — a predicate that reaches the kernel

Let the caller pass a mask instead of hard-coding all-ones, and give
`search_chunks` a `--where` over `source_path` prefix — the one live filter
axis, since `subject` is 97.3% a single value (1,958 of 2,012 rows are `llvm`;
the largest three-level directory prefix is 13.2%).

Keep the mask representation. Compacting surviving rows would renumber indices
and break the bit-exact reference-vs-native differential, which is the property
worth most in the rail.

*Payoff:* the scan falls from 60.4 ms to 0.21 ms at a narrow predicate.
*Gate:* a predicate matching nothing must score zero rows, not all rows — the
anti-vacuity case, per L2.

### S5 — incremental sets with a containment rule

Adopt ClickHouse's *model*, not its code: an embedding set is a list of parts,
each named by the row range and content hash it covers; a rebuild re-embeds
only parts whose inputs changed; a part wholly contained in another is dropped.

The reuse is two-level, and the levels answer different questions. A part whose
bytes are unchanged cannot hold a changed row, so nothing in it needs looking
at; within the parts that did move, each row's own text digest decides whether
its vector is reused. `--incremental` reports both, because a chapter reformatted
without changing a word moves a part and reuses every row in it, and only the
pair shows that.

Reuse is refused across providers, revisions, dimensions, and any set whose
`vectors.f32` no longer matches the digest its manifest records: a stored vector
is sound only when the function that produced it is the function that would
produce the new one, and nothing in the bytes reveals that on its own.

**This slice was mispredicted, and the correction belongs on the record.** The
analysis said it would pay nothing today — "0 today; unbounded at the first
non-local provider" — on the grounds that a local rebuild costs 3.30 s. The
rebuild actually costs 4,876 ms and the incremental path 1,073 ms: a 4.5x saving
with the cheapest possible provider. The reasoning was right about the shape and
wrong about the number, which is the argument for measuring the thing rather
than the thing it resembles.

*Payoff:* 4,876 ms → 1,073 ms after a one-row edit; unbounded at the first
non-local provider.
*Gate:* an unchanged corpus re-embeds zero rows; a one-row edit re-embeds
exactly one; and the incremental result is byte-identical to a full rebuild in
`vectors.f32`, `vectors.q15`, `index.jsonl` and `manifest.json`.

### S6 — generations, and reading an old one

Only once S5 exists. Keep the previous generation addressable so a retrieval
result can be reproduced against the corpus as it stood. This applies BCIR's
existing artifact discipline — immutable within a generation, promoted at
quiescent boundaries — to the training rail rather than inventing a second one.

*Payoff:* reproducibility of a past answer.
*Gate:* an evaluation pinned to a generation produces identical scores after
the corpus moves.

## What is verified where

`verify_database.py` runs in the **LLVM training corpus** job, which is
ubuntu-only — as the whole `training/` rail has always been. The database layer
inherits that boundary, so it is worth stating rather than leaving a reader to
assume otherwise: the catalog, the planner, the predicate path, incremental
rebuilds and generations are exercised on Linux and on no other host.

The one piece that *is* cross-host is `NativeAIKernels`, because
**Host portability (windows-latest)** runs `bcir/tests/test_native_ai.py`. That
is how S1's stamp tests found a real defect this repository had carried for as
long as the class existed: the docstring claimed the object owned the
dynamic-library handle, and there was no way to give it back. On Windows a loaded
module keeps its file locked, so a temporary build directory could not be
removed, `build`'s own `os.replace` could not run over a library the process had
already loaded, and neither could an unlink. Nothing on POSIX is locked, which is
why only the other host in the matrix could surface it.

Two consequences worth keeping:

- A rail tested on one host is tested against one operating system's opinions
  about files. The defect was not in the new code; it was in code the new code
  used in a way nothing had used it before.
- The fix — `close()`, plus `kernel32.FreeLibrary` when CPython's private
  `_ctypes.FreeLibrary` is absent — cannot be run on the development host. It is
  exercised by substituting both doors and asserting the order and the handle,
  and confirmed by CI. That is a weaker verification than the rest of this
  document rests on, and it is labelled as such rather than reported alongside
  measurements taken here.

## Why this points inward

`bcir/asn1/selection.py` already chooses an encoding rule by minimizing over
the twelve-axis cost vector, and spells its objectives as axis names:
`WIRE_SIZE = "memory"`, `ENCODE_LATENCY = "compute.encode"`,
`DECODE_LATENCY = "compute.decode"`. It encodes, decodes and compares — and
only then reports cost.

Choosing between the exact Q15 scan, the C kernel and the lossy Q8 view is the
same shape of decision: a legality check, then a minimization over the same
vector. `compile` prices S1. `memory` prices S3. `accuracy` prices the Q8 view
against the exact one. None of the nine projects has an accuracy axis at all,
because none was built to choose between two answers of different quality.

What makes the six slices a database rather than six optimizations is that each
becomes a term that planner can price.

## Method, and what would overturn each verdict

Seven agents read one project each — source, not documentation — and nominated
mechanisms with file-and-line citations; fourteen nominations then went to
fourteen independent refuters instructed to default to refuting. Twenty-one
agents, no errors, roughly 2.06M subagent tokens.

Two caveats belong on the record. The refuter instruction biases the outcome:
14-of-14 refuted is a property of the prompt as much as of the mechanisms, and
the verdicts are load-bearing only because each cites a file and a number. And
the brief those agents worked from carried two wrong figures — a corpus of
2,215 rows and a 44 ms command. The built set holds 2,012 rows; 2,215 is what a
rebuild would produce today, which is itself the evidence for S5. Every number
in this document was re-measured directly.

| Verdict | Reopens when |
| --- | --- |
| faiss stays study-only | the corpus passes ~50,000 rows, where an IVF structure begins to pay |
| ClickHouse parts stay deferred (S5) | the embedding provider stops being a local feature hasher |
| Ray and Dask stay study-only | a parallel stage acquires a payload large enough to exhaust memory |
| webdataset stays skipped | training input passes roughly 10^5 samples (today: 1.89 MiB in 10 files) |
| TileDB time travel stays deferred (S6) | S5 lands |

## One finding outside this scope

A refuter checking whether the kernel's filter path is tested found that it is,
thoroughly — but that the branch beside it is not. The short-return path in
`runtime/c/bcir_ai_kernels.c`, where fewer rows are admitted than `top_k`
requests, has never executed under a gate: every test admits more rows than it
asks for. That is a coverage hole in shipped C, unrelated to any of the nine
projects, and it belongs in the oracle's test registry rather than in this
ladder.
