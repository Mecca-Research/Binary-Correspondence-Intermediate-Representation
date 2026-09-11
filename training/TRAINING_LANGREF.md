# BCIR Training Rail Language Reference — v1.0.0 (normative)

The `training/` tree is a **corpus that is also a database**. It holds teaching
material about compilers, systems and machine learning; it indexes that material
as a typed, planned, content-addressed table; and it exposes both a relational
surface and a vector-retrieval surface over the same rows, priced on BCIR's own
twelve cost axes.

This document is the reference for the table, the query surface, the artifact
formats, and the machine-learning rails built on them. It covers the system as it
stands after PR #782, and records what preceded it.

> **Source-of-truth rule.** The formats, grammars, laws and defaults in this
> document are normative. The Python under `training/tools/` is the **executable
> conformance oracle** for them — it must agree with this document, never
> override it. Where the two disagree, this document is the defect report and the
> code is the thing to fix. `training/tools/verify_langref.py` reconciles the two
> mechanically, so the pairing is checked rather than asserted
> (`docs/security/laws.md` L15 — a mirror list will drift).
>
> Two things are deliberately **not** here. Counts of rows, subjects, files and
> tests live in generated output, never in prose — a number written beside the
> code that produces it is a number that will disagree with it. And durations live
> in `build/training/perf/`, host-local, because a millisecond is a statement about
> a machine (§15).

---

## 0. Stance

The training rail answers one question the rest of BCIR does not: **what does the
system know, and can it show its work?** Everything else follows from taking that
literally.

- **A subject teaches what a gate can check.** Prose that no tool verifies is not
  corpus material; it is a claim. Every tier below is defined by what verifies it.
- **Legality before cost, as everywhere in BCIR.** A query plan is refused by a
  structural rule or it is priced — never priced into legality (§7).
- **An artifact is named by its content, not by when it was written.** Two hosts
  building the same corpus produce the same bytes, or the build is wrong (§14).
- **A wrong answer must be impossible to reach quietly.** Where a cheaper path
  exists, it is taken only when a checked property says it returns the identical
  rows; otherwise the expensive path runs.

The rail is *above* BCIR's kernels and reuses them rather than reinventing them:
retrieval scores through `bcir_ai_q15_topk`, the lossy view quantizes through
`bcir_ai_quantize_q8_f64`, and query plans are priced with `bcir.kbcir.cost`'s
twelve-axis `CostVector`. Nothing in `training/` is a second convention for
something BCIR already specifies.

---

## 1. What was here before

Before PR #768 the tree held `llvm-training/`: a twenty-chapter LLVM IR
curriculum (`00-foundations` … `19-hardware-aware`) with worked examples,
exercises, an autograder, invalid-example tripwires, opt-diff goldens and a CMake
aggregate gate. It was **prose and fixtures only** — there was no `training/`
directory, no chunk format, no vectors, no catalog, no planner, and no tool under
`training/tools/`.

That curriculum is not superseded. It moved to `training/llvm/` in PR #769 and
remains the largest single subject, with its own gates still running (§15). What
PR #768 onward added is everything *around* it: a corpus contract that other
subjects can satisfy, and the database and ML rails described here.

---

## 2. The three tiers

`training/CORPUS_STANDARD.md` is the contract; this is its normative summary.

| Tier | Artifact | What it is | What checks it |
|---|---|---|---|
| 1 | Markdown under `training/<subject>/` | prose a person reads | the subject's own gates |
| 2 | `*.chunks.jsonl` + embedding sets | retrievable, attributed passages | `verify_database`, `verify_embeddings`, `verify_retrieval` |
| 3 | distillation records | question/answer pairs whose answers come from tools | `verify_corpus_records`, `verify_training_export` |

**Tier 2 is derived, never authored.** A chunk is produced from a Tier-1 document
by `build_chunks.py`; editing a chunk file by hand is a defect, because its
`source_sha256` and `span` bind it to bytes it no longer describes.

**Tier 3 answers come from verifiers, not from opinion.** The five record kinds
are `exercise`, `repair`, `prediction`, `claim` and `review`. `claim` and `review`
are the load-bearing ones: their answers are lines of real tool output and pinned
verdicts — *including verdicts that refuse to conclude*, because a corpus that
only ever trains on confident answers teaches confidence.

---

## 3. The declared table

One table, declared once, in `training/tools/schema.py`. Every tool reads columns
through it; none may invent a column, a type, or a filterability rule of its own.

### 3.1 Roles

A column's **role** decides what may be done to it, and is the only thing that
decides it:

| Role | Meaning | Filterable | Orderable |
|---|---|---|---|
| `key` | the row's identity | no | no |
| `indexed` | an inverted index exists | yes | yes (by value, then row) |
| `path` | a filesystem path, prefix-searchable | yes (`eq`, `prefix`, presence) | no |
| `numeric` | a measurement, packed and zone-mapped | yes (all ten operators) | yes (sorted index) |
| `text` | free text carried for display | **no** | no |
| `structural` | provenance and format fields | no | no |

A predicate against a `text` column is **refused**, loudly. This is deliberate:
the alternative — accepting it and scanning — turns a typo into a slow correct
answer today and a silently wrong one after the next schema change.

### 3.2 Columns

| Column | Type | Role | Required | Nullable | Domain / minimum |
|---|---|---|---|---|---|
| `schema` | string | structural | yes | no | `bcir-training/chunk/v1` |
| `chunk_id` | string | key | yes | no | — |
| `corpus` | string | structural | yes | no | `training` |
| `subject` | string | indexed | yes | no | — |
| `source_path` | string | path | yes | no | — |
| `source_sha256` | string | structural | yes | no | — |
| `span` | object | structural | yes | no | — |
| `heading_trail` | array | text | yes | no | — |
| `title` | string | text | yes | no | — |
| `kind` | string | indexed | yes | no | `prose`, `code`, `mixed`, `table` |
| `language` | string | indexed | **no** | **yes** | — |
| `text` | string | text | yes | no | — |
| `char_count` | integer | numeric | yes | no | ≥ 1 |
| `token_estimate` | integer | numeric | yes | no | ≥ 1 |
| `embedding` | array | structural | yes | yes | — |
| `embedding_spec` | object | structural | yes | no | — |
| `provenance` | object | structural | yes | no | — |
| `verified_by` | array | structural | yes | no | — |

### 3.3 Constraints

Three constraint kinds, all checked at build and all refusals rather than
coercions:

1. **Type.** A value of the wrong JSON type is refused. It is *not* read as
   absent — that conflation let a measurement that was a string be counted as an
   unmeasured row, which is a wrong number reported with exit 0.
2. **NOT NULL.** `required` says the key must be present; `nullable` says whether
   `null` is an admissible value. `language` is the only column that is optional
   and nullable, because not every passage is in a language.
3. **Domain.** Where a column declares a domain, a value outside it is refused.
   The domain is stated once, here and in `schema.py`, and the JSON Schema under
   `training/schema/chunk-v1.json` must agree — a drift between the Python domain
   and the JSON enum is itself a gated defect.

---

## 4. Physical layout

### 4.1 The catalog directory

A catalog directory holds a pointer and a set of immutable publications:

```
build/training/catalog/
  CURRENT              the publication id currently readable
  sets/<publication>/  one complete, immutable set of artifacts
    catalog.json       manifest: schema, files, statistics, parts, zone maps
    postings.json      inverted index, per indexed column
    ids.txt            the key column, one per line, in row order
    locator.bin        where each row's bytes are, for late materialization
    numeric.bin        packed measurement columns
    order.bin          a sorted index per measurement column
```

`catalog.json` declares `schema = "bcir-training/catalog/v4"`.

### 4.2 Binary formats

All three binaries are little-endian by contract, and a reader on a big-endian
host byte-swaps on load. All are row-major in **catalog row order** (§4.3).

| File | Record | Bytes | Meaning |
|---|---|---|---|
| `locator.bin` | `<HQI` | 14 | file index (u16), byte offset (u64), byte length (u32) |
| `numeric.bin` | `<q` | 8 | one measurement, two's-complement signed |
| `order.bin` | `<I` | 4 | a row number |

- **`numeric.bin` null sentinel** is `NUMERIC_NULL` = `-(2**63)`. It is a sentinel,
  not a value: no comparison admits it, and it is not a minimum.
- **`order.bin`** stores, per measurement column, the measured rows first in
  ascending `(value, row)` order and then the unmeasured rows in row order. The
  split point is that column's own `count` in the statistics, so the index needs
  no length of its own to disagree with. A row number must fit in `u32`, so
  `MAX_INDEXABLE_ROWS` caps a catalog at `2**32 - 1` rows.
- **`postings.json`** maps value → ascending row list, per indexed column. Rows
  carrying no value for an indexed column are listed under the reserved key
  `"\0null"`, which cannot collide with a real value because a NUL cannot appear
  in one.

### 4.3 Schema registry

Every artifact the rail writes declares its schema in its own first field. The
identifiers are the compatibility surface: a consumer reads the identifier before
it reads anything else, and refuses what it does not know.

| Identifier | Artifact |
|---|---|
| `bcir-training/chunk/v1` | a Tier-2 chunk record |
| `bcir-training/catalog/v4` | the catalog manifest |
| `bcir-training/embedding-set/v1` | an embedding set manifest |
| `bcir-training/generation/v1` | a generation manifest |
| `bcir-training/distill/v1` | a Tier-3 distillation record |
| `bcir-training/eval-query/v1` | a judged retrieval query |
| `bcir-training/eval-set/v1` | a judged retrieval query set |
| `bcir-training/index-memory/v1` | an index-backed concept memory |
| `bcir-training/export/v1` | an export into BCIR's training stack |
| `bcir-training/plan-baseline/v1` | recorded planner decisions |
| `bcir-training/perf-suite/v1` | a recorded performance report |

A version is bumped when a reader that understood the old identifier would
misread the new bytes — never for an addition a reader may ignore. `catalog/v4`
is at v4 for that reason: the statistics, zone maps and sorted index each changed
what a reader must understand.

### 4.4 Row order

**The** row order of the corpus is `catalog.row_sort_key`, and it is imported by
every writer rather than re-derived. Row *i* of an embedding set must be row *i*
of the catalog: that is what lets a ranked row number become bytes on disk with
no lookup. Two writers arriving at the same order by two independent rules is not
agreement; it is a coincidence waiting to end.

### 4.5 Parts

Rows are grouped into **parts** of at most `MAX_BLOCK_ROWS = 128` rows. A part is
the unit of three separate things:

- **zone maps** — per part, per measurement column, a `{count, min, max}` summary
  that lets a range predicate rule the part out without reading a row;
- **incremental rebuild** — the unit of "what changed" (§11);
- **the containment rule** — a part's rows are contiguous in row order.

Parts are bounded by row count and **not** by subject. Subject-shaped parts made
the zone map useless on the one column that mattered, because a subject spans the
whole measurement range.

---

## 5. The predicate language

A predicate is `column <op> value`. A selection is a **conjunction** of
predicates, resolved against the catalog.

### 5.1 Declared scope

A conjunction of `column op value` terms, and nothing else. **No disjunction
across columns, no arithmetic, no user expressions.** The boundary is declared so
that the answer to the next soundness question is to point at it, rather than to
add an expression evaluator nobody asked for.

`BETWEEN` is not an operator because it does not need to be: two comparisons on
one column intersect to the same interval.

### 5.2 Operators

Ten, and the gate asserts that every one of them is exercised (§15):

| Operator | Spelling | Applies to | Meaning |
|---|---|---|---|
| `eq` | `=` | indexed, path, numeric | equal |
| `ne` | `!=` | indexed, path, numeric | **complement** — see §5.5 |
| `in` | `=a,b,c` | indexed, path | any of |
| `prefix` | `^=` | path | starts with |
| `lt` `le` `gt` `ge` | `<` `<=` `>` `>=` | numeric | existential comparison — see §5.5 |
| `isnull` | `!=?` | any filterable | carries no value |
| `notnull` | `=?` | any filterable | carries some value |

The parse takes the **leftmost** spelling and, at that position, the **longest**.
Scanning in table order would read `title=a!=b` as `!=` with the column `title=a`,
which is a confusing error where `title = "a!=b"` is the answer.

### 5.3 Values and quoting

A value may be wrapped in double quotes so it can contain characters the grammar
would otherwise read as syntax — above all the comma that separates a set.

- Inside quotes, `\"` is a quote and `\\` is a backslash. **Those are the only two
  escapes**; a backslash before anything else is refused rather than dropped.
- A value is quoted **whole or not at all**: a quote is syntax only as the first
  character, the matching close quote must be the last character, and a quote
  anywhere inside an unquoted value is refused.

The alternative — quotes significant wherever they appear, as a shell reads them —
makes `title=say "hi"` silently ask about `say hi`. That is the same silent
misreading the comma rule exists to remove, spelled with a different character.

A measurement is compared against an integer in ASCII digits and nothing else.
Python's own `int()` accepts PEP 515 underscores and Unicode digits; a wire
grammar must not.

### 5.4 Presence

`?` is reserved on **every** filterable column as the right-hand side meaning "a
value, any value": `column=?` is SQL's `IS NOT NULL` and `column!=?` is `IS NULL`.
It is reserved everywhere rather than only where nothing could collide, so the
grammar answers the same question the same way on every column
(`docs/security/laws.md` L12). The cost is stated rather than hidden: a column
holding the literal string `?` cannot be matched by equality.

### 5.5 Two declared divergences from SQL

Both are about rows that carry no value, and both are gated so neither can drift
into the other.

**A comparison is existential.** A row with no measurement satisfies neither
`char_count >= 500` nor `char_count < 500`, so the two do **not** partition the
table. That is SQL's rule, and it is the one this layout makes easiest to lose,
because the null sentinel is an ordinary negative integer to the packed column.

**`!=` is complement, which is *not* SQL's rule.** `language != c` admits a row
with no language, where SQL would return UNKNOWN and drop it. Retrieval wants the
complement far more often than it wants three-valued logic, and the surprising
half of SQL's answer is one term away: `language!=c` with `language=?` is exactly
`<>`.

---

## 6. Statistics and estimates

Every selection carries an **estimate** and a flag saying whether that estimate is
a count or a bound. The flag is the load-bearing part: a bound reported as a count
is a wrong number that no correctness gate can see, because the rows are right.

- **A single indexed equality** is exact — the posting list length *is* the answer.
- **A conjunction** is estimated from stored pairwise joint statistics where the
  pair was counted at build time, and is otherwise a bound. It is exact only when
  one term determines it, or when the estimate is zero (nothing can be admitted by
  a superset of an empty set).
- **Joint statistics** are stored for indexed column pairs whose cross product
  does not exceed `MAX_JOINT_CELLS = 4096`. The pair key is the two column names
  **sorted** and NUL-separated, so the writer and the reader cannot disagree about
  which way round the table is stored.
- **A range** is estimated from zone maps: parts ruled out contribute nothing, and
  parts that survive contribute their count as a bound.

---

## 7. The query planner

Retrieval is a **planned** query. The planner enumerates candidates, refuses the
illegal ones by structural rule, prices the rest on twelve axes, and minimizes one
named axis. This is the same shape BCIR already uses to choose an ASN.1 encoding
rule, over the same cost vector.

### 7.1 Legality, decided first and never from a measurement

Four rules, each a structural property of a plan. A plan that fails one is not a
more expensive plan — it is not a plan.

| Rule | Refuses |
|---|---|
| `exactness` | a lossy backend for a request that asked for the exact ranking |
| `availability` | a backend this host cannot reach |
| `coverage` | a plan that cannot see every row the predicate admitted |
| `materialization` | a plan that cannot produce the text the caller asked for |

No amount of speed makes the Q8 view legal for a request that asked for the exact
ranking. That is `accuracy`, and it is a legality question first and a priced axis
second.

### 7.2 Objectives

Each objective names the cost axis it minimizes:

| Objective | Axis | Means |
|---|---|---|
| `EXACTNESS` | `accuracy` | the exact Q15 ranking, whatever it costs |
| `LATENCY` | `compute` | least work |
| `FOOTPRINT` | `memory` | least resident |
| `STARTUP` | `compile` | least time before the first answer |

`EXACTNESS` is not "no objective": it is the named objective whose answer is the
definition — the degenerate case this rail pins.

Ties break by the scalarized cost and then by declaration order, so the same
request against the same catalog picks the same plan on every host. A planner
whose choice wobbles cannot be gated.

### 7.3 The cost model

Constants are calibrated wall-clock measurements on one host. They exist to
**order** plans; nothing gates on them, and no optimality is claimed from them
(TMSAO-4: heuristic, no claim). A host where they are wrong picks a slower plan,
never a wrong answer — which is why legality is decided without them.

Two constants are asked rather than assumed, because each names an artifact the
caller may already have:

- `kernel_cached` — is the compiled kernel on disk, or must it be built?
- `derived_cached` — does the embedding set **store** its row-squares column, or
  must the reader rebuild it? (§8.4)

A constant that is wrong by a factor which **reorders** the plans is not an
approximation, it is a defect: the planner is then choosing confidently in the
wrong direction. `training/plans/baseline-v1.json` pins the resulting decision for
a fixed query set across three availability configurations and both derived-column
states, so a constant that starts reordering plans is a finding (§16, S18).

---

## 8. Retrieval

### 8.1 Embedding sets

An embedding set is a directory declaring `bcir-training/embedding-set/v1`:

```
manifest.json   model, revision, dim, coverage, parts, digests
index.jsonl     one row per vector, in catalog row order
vectors.f32     the unit vectors as stored
vectors.q15     the BCIR-native Q15 codes
squares.u32     ‖p‖² per row — a materialized derived column (optional)
```

A provider declares its own `semantics`, so no consumer has to find that out by
experiment: `lexical-hash-v1` declares `semantics = "lexical"`, a sentence
transformer declares `"learned"`. Both declare `normalize = "l2"`.

### 8.2 The Q15 code space

Codes are BCIR's symmetric Q15: signed `int16`, `Q15_SCALE` = `32767`, with `-32768`
forbidden so the space stays symmetric about zero and negation is lossless — the
same discipline `runtime/c` applies to Q8's `-128` and Q4's `-8`. Quantization
rounds half away from zero, matching BCIR's own rule rather than Python's
banker's rounding.

Quantization is valid only for L2-normalized vectors. For two unit vectors,
`‖a-b‖² = 2 - 2·cos(a,b)` is strictly decreasing, so an exact squared-L2 ranking
**is** a cosine ranking — which is what lets retrieval run on `bcir_ai_q15_topk`
and be compared for bit equality rather than within a tolerance.

### 8.3 Backends

| Backend | Arithmetic | Exact? | Needs a kernel |
|---|---|---|---|
| `reference` | Q15 integers in pure Python | yes — **the definition** | no |
| `native` | the same ranking through `bcir_ai_q15_topk` | yes, bit-identical | yes |
| `q8` | BCIRQ8 cosine through `bcir_ai_q8_rows_dot_f64` | **no** — lossier by construction | yes |
| `both` | `reference` and `native`, required to agree exactly | yes | yes |

`q8` is a different arithmetic, not a third spelling of the same one. No gate
asserts its ranking equals the exact one; what is worth knowing is how often it
agrees, and that is measured and reported rather than assumed.

`q8` also takes no eligibility mask — it is a rows-dot, not a top-k — so a
selection filters *after* its scan. The planner prices that honestly, which is one
reason a narrow predicate makes the native backend win.

### 8.4 Derived columns

`‖p‖²` per row is needed because `topk_reference` scores with the expanded form
`‖q‖² + ‖p‖² - 2(q·p)`, leaving one dot product as the only per-row work. The term
is neither optional nor constant: the codes come from unit vectors so every square
is *near* `scale²`, but quantization moves each one, and rows closer together than
that spread would reorder if it were dropped.

So it is **stored**, not dropped — `squares.u32`, `uint32` little-endian, in row
order. A stored derived column is a new way to be wrong in the worst shape: right
length, right dtype, every value wrong, exit 0. Three properties are therefore
checked before it is trusted, and any failure falls back to computing:

1. `derived_from` — the digest of the codes these squares were computed over —
   names the codes this manifest describes;
2. the column's own `sha256` matches its bytes;
3. its row count is the set's.

A set built before the column existed has no such file, loads, and returns the
identical ranking. That is the non-disturbance rule, not a special case.

---

## 9. The relational surface

### 9.1 ORDER BY

Ordering is total and stable. A numeric column orders by its packed values with
**absent last in both directions** — absent is not small, and it is not large
either; sorting it as though it were is how a "shortest chunks" query comes back
full of rows that were never measured.

Equal values keep row order in **both** directions. A single key with
`reverse=True` would reverse the tiebreak along with the key, and a page boundary
falling inside a run of equal values would then repeat or skip rows.

Two strategies produce identical rows, so the choice is about time only:

| Strategy | How | Chosen when |
|---|---|---|
| `index` | read `order.bin`, keep the admitted rows | no predicate, **or** admitted share ≥ `ORDERED_INDEX_SHARE` |
| `sort` | sort the admitted rows directly | admitted share < `ORDERED_INDEX_SHARE` |

`ORDERED_INDEX_SHARE = 0.70`. The choice is returned as a *value* rather than made
inside a branch, so a gate can assert which path ran — otherwise the decision
would be the one thing about the slice that nothing could observe.

### 9.2 Range resolution

The sibling decision, with its own measured constant: `SORTED_SCAN_SHARE = 0.70`
chooses between seeking the sorted index and scanning the packed column. Below it
the seek wins by up to 90×; over the whole table the scan wins by about 6×,
because a scan produces rows in order for free while a seek has to sort what it
found.

### 9.3 Aggregates and grouping

`COUNT`, `MIN`, `MAX`, `SUM` and `AVG` over a measurement column, `GROUP BY` over
an indexed column, and `HAVING` over the group summary. An **unfiltered** group
count is answered from statistics the build already wrote and reads no row at all;
a filtered one intersects postings. Which of the two ran is reported, because "it
was fast" and "it read nothing" are different claims.

---

## 10. Generations and publication

### 10.1 Content addressing

A generation is identified by the digest of the corpus it was built from, so the
same corpus always produces the same generation id on any host. Generations are
immutable: a rebuild of the same corpus re-publishes identical bytes rather than a
new version.

### 10.2 Atomic publication (ACID across artifacts)

Six artifacts must become visible together or not at all. There is no log to
replay, so the ordering *is* the durability argument:

1. every artifact is written to a temporary file in the destination directory,
   `flush`ed, and **`fsync`ed** — a rename that reaches the directory before the
   file's bytes reach the disk publishes a name for content a crash can lose;
2. each is `os.replace`d into the publication directory, which is atomic;
3. the directory itself is synced, where the host has a way to say that (POSIX
   needs it; Windows has no directory handle and an already-ordered rename, so the
   call is a declared per-host no-op rather than an omission);
4. **only then** is `CURRENT` swapped to point at the completed set.

A reader therefore never observes a partial publication, and a crash at any step
leaves `CURRENT` naming the previous complete set. Superseded publications are
pruned, never the current one.

---

## 11. Incremental builds

A rebuild re-embeds only what changed, and the **containment rule** is what makes
that safe: a derived set may reuse a row only if the part that row belongs to is
byte-identical to the part the reuse was recorded against. Reuse is reported —
how many rows were reused, how many recomputed, and which parts changed — because
a reuse rate nobody prints is a correctness claim nobody checks.

---

## 12. Subsystem interfaces

Five modules under `training/tools/db/`, one per kind of question. They exist so
that a caller depends on the question it is asking rather than on the whole
engine, and so that two callers asking the same question cannot answer it two
ways.

| Module | Answers | Deliberately does not |
|---|---|---|
| `engine` | how to load a module and open a catalog | decide anything about the data |
| `relational` | rows, in an order, without vectors | rank |
| `retrieval` | which rows a search may consider, and what they say | score, or know what a kernel is |
| `analytics` | how many, grouped by what, aggregated how | return rows |
| `ingest` | row order, parts, and what changed | query |

The split between `relational` and `retrieval` is the load-bearing one: both
return rows in an order and they mean different things by "order". A tool that
quietly reordered a ranked result would be answering a different question than it
was asked, so the two do not share an entry point.

`engine` exists because every tool here runs as a script from the repository
root: a plain `import catalog` resolves differently depending on how the tool was
invoked, and each tool used to answer that for itself, differently.

---

## 13. The machine-learning rails

| Rail | Module | Produces |
|---|---|---|
| chunking | `build_chunks.py` | Tier-2 chunk records |
| embedding | `embed_chunks.py` | embedding sets (§8.1) |
| retrieval eval | `build_eval_queries.py`, `evaluate_retrieval.py` | a judged query set and measured recall |
| concept memory | `build_index_memory.py` | an index-backed memory whose retrieval is *readable* |
| distillation | `distillation.py`, `build_distillation.py` | Tier-3 records (§2) |
| grading | `grading.py` | subject-independent autograding |
| dataset export | `dataset_export.py`, `export_training_examples.py` | deterministic JSONL into BCIR's own training stack |
| substrate survey | `ml_components.py` | what BCIR already provides for ML, as data |

Four rules hold across all of them:

- **One `RawDocument` per source, not per chunk.** A document split into chunks is
  still one document; counting it as many inflates every corpus statistic.
- **Held-out records stay held out.** A record in the evaluation split must not
  reach the supervised export, and the gate checks the split rather than trusting
  the exporter.
- **The baseline is matched.** A retrieval metric compared against an unmatched
  control measures the mismatch.
- **Preferences are decided by a verifier, not by a rater.** Where the corpus
  emits a preference pair, the preferred side is the one a tool accepted.

The shared rail owns the machinery; a subject owns its content and declares itself
through `subject_profile.py`. A subject that wants different grading does not fork
the grader — it declares what makes it different.

---

## 14. Host independence

**An artifact's bytes do not depend on the host that wrote it**
(`docs/security/laws.md` L24). This is the one property the rail claims over every
system it has been compared against, and it is enforced in four places:

1. **Line endings.** Every text artifact writer pins `newline="\n"`. Python's text
   mode otherwise translates to the platform's ending, which moved the corpus
   fingerprint, every recorded `sha256`, every part digest, and `locator.bin`'s
   offsets on Windows — an identical corpus with a different content address.
2. **Byte order.** Every binary artifact is little-endian by contract, byte-swapped
   on a big-endian reader.
3. **No timestamps.** The manifest records size and digest, never `mtime`. A
   modification time makes `catalog.json` unreproducible across directories, and
   the cost of digesting the corpus on every load is paid deliberately (§15).
4. **Canonical JSON.** Sorted keys, fixed separators, trailing newline.

---

## 15. Measurement discipline

Every measured row carries a **class**, and the class decides what it may be used
for:

| Class | Meaning | May gate |
|---|---|---|
| `exact` | a deterministic count — modules imported, bytes read, rows scanned, parts pruned, which strategy ran | **yes** |
| `ratio` | one measurement over another in the same process | within a 25% band |
| `wall` | an absolute duration | **never** |

`training/tools/perf_suite.py` measures eight families — `startup`, `catalog`,
`resolve`, `aggregate`, `order`, `fetch`, `build`, `memory` — and `--compare`
holds `exact` rows to equality and the rest to the band. A recorded baseline is
host-local and is not tracked: comparing one machine's milliseconds against
another's reports a finding per row, and not one of them is about the code. The
decisions that must hold across hosts are pinned by `plan_baseline.py`, in exact
terms that do not involve a clock.

This host has no PMU, so there are no cycle or cache-miss rows and none are faked.

---

## 16. The gates

| Gate | Verifies |
|---|---|
| `verify_database.py` | the table, the grammar, the planner, the artifacts, the publication rules |
| `verify_embeddings.py` | attribution, alignment, non-degeneracy, digests, the Q15 rounding rule, the derived column, determinism across two builds |
| `verify_retrieval.py` | the judged query set and the measured retrieval properties |
| `verify_corpus_records.py` | Tier-3 record shape, splits and leakage |
| `verify_training_export.py` | the export is deterministic and the held-out split is respected |
| `verify_grading.py` | the grading kernel, against its own fixtures |
| `verify_ml_components.py` | the ML substrate survey against the hosted stages |
| `verify_langref.py` | **this document** against the code it describes |

`verify_database.py` runs named check groups, among them: `interfaces-boundary`,
`interfaces`, `host-byte-order`, `S1`, `S2`, `aggregates`, `line endings`,
`schema`, `constraints`, `S7-grammar`, `quoting`, `estimates`, `coverage`,
`S7-ranges`, `S7-presence`, `S7-groups`, `ordering`, `planner`, `order strategy`,
`explain`, `S5`, `S5-incremental`, `S6`, `S13`, `S18`.

### 16.1 What a gate here must do

- **Every exit is a verdict.** A missing tool, a timeout, an unreadable input and a
  malformed configuration all return the structured report. A traceback in place of
  a report is itself a defect.
- **Anti-vacuity is a state, not a comment.** A gate that can pass without
  examining anything is broken while green. Ask of every checker: *what input makes
  every loop in it iterate zero times?* — then feed it that input in a test.
- **Prove the gate can fail.** For each new check, inject the defect it guards and
  watch it fire (RED) before landing the fix/gate pair (GREEN). A test asserting
  only the honest path passes against every defect.
- **A witness must hit its own law.** A check that asserts "a predicate forces a
  sort" using a predicate that admits most of the table is testing the presence of
  a WHERE clause, not the law it was written for.

---

## 17. Conformance

An implementation of this rail conforms when:

1. it produces byte-identical artifacts from an identical corpus on any host (§14);
2. its `reference` and `native` backends agree **exactly**, not within a tolerance;
3. every legality refusal in §7.1 is structural and decided before any price;
4. every estimate is accompanied by whether it is a count or a bound (§6);
5. a publication is visible in whole or not at all (§10.2);
6. every gate in §16 passes with zero unexplained skips, and each check has been
   seen to fail against its own injected defect.

---

## 18. Provenance

The slice ladder, and the pull request each slice landed in. Slice IDs are the
vocabulary the code comments and the roadmap both use.

### 18.1 Before the ladder

| PR | What it did |
|---|---|
| — | `llvm-training/`: a 20-chapter LLVM IR curriculum, examples, exercises, autograder, opt-diff goldens (§1) |
| #768 | closed the curriculum's ROADMAP gaps with checked material |

### 18.2 The corpus becomes a corpus

| PR | What it did |
|---|---|
| #769 | restructured `llvm-training/` into `training/llvm/`, added the multi-subject layout and the three model-training tiers |
| #770 | measured retrieval, added the index-backed concept memory, opened the edges into BCIR's training stack |
| #771 | fixed ten review findings and gated each fix |
| #772 | extracted the grading and dataset-export kernels so subjects share them (0.7) |
| #773 | inverted the distillation rail to subject-owned record sources; surveyed BCIR's ML substrate (0.8) |
| #774 | trained a learned embedding model from the corpus and reported the verdict honestly against a matched control |

### 18.3 The corpus teaches the IR it belongs to

| PR | What it did |
|---|---|
| #775 | the corpus teaches BCIR itself (Phase 1.6) |
| #776 | Phase 1 slices 1.2 and 1.5; fixed a fail-open in the mapping gate |
| #777 | Phase 1 complete — the corpus at LLVM 23, the LangRef delta, comprehensive MLIR, an enforced SEMVER baseline |
| #778 | cross-referenced the corpus against the LLVM and MLIR surfaces and closed the gaps |

### 18.4 The retrieval rail becomes a database

| Slice | What it added | PR |
|---|---|---|
| S1 | content-addressed kernel cache | #779 |
| S2 | lazy derived columns | #779 |
| S3 | late materialization of chunk text | #779 |
| S4 | a predicate that reaches the kernel | #779 |
| S5 | incremental sets with a containment rule | #779 |
| S6 | generations, and reading an old one | #779 |
| S7 | zone maps, range and presence operators, grouped aggregates and HAVING | #779 |
| S8 | a sorted index per measurement column | #779 |
| S9 | parts stop being subject-shaped | #780 |
| S10 | a measured threshold for the seek/scan choice | #780 |
| S11 | the database layer becomes host independent | #780 |
| S12 | one interface module per subsystem | #780 |
| S13 | ACID publish across artifacts | #781 |
| S14 | type-safe table access from one declared schema | #781 |
| S15 | exact joint statistics for conjunctions | #781 |
| S16 | operator-coverage closure in the gate | #781 |
| S17 | domain and NOT NULL constraints | #781 |
| S18 | plan-regression detection | #781 |
| S19 | the row-squares column is stored, not rebuilt | #782 |

### 18.5 Defects found by the rail's own analyses

Landed with their fixes, each caught by a named check thereafter:

| Defect | Symptom | PR |
|---|---|---|
| comma/quoting | a value containing a comma silently became two values | #781 |
| conjunction estimates | a bound was reported as a count for most indexed pairs | #781 |
| catalog reproducibility | `catalog.json` differed across directories; the determinism gate built twice from one directory, so it was vacuous | #781 |
| type-as-null | a measurement of the wrong type was counted as unmeasured | #781 |
| line endings | the corpus had different bytes, and a different content address, on Windows | #781 |
| q8 setup mispricing | one cost constant 37.5× low chose a measurably slower plan on a partial build | #782 |
| ORDER BY threshold | a wide predicate was sorted row by row beside an unused index | #782 |
| EXPLAIN verdict | a plan forced by a flag was reported as the planner's choice | #782 |

---

## 19. Reading order

- `training/CORPUS_STANDARD.md` — the three-tier contract, and what a new subject
  must deliver.
- `training/DATABASE_ROADMAP.md` — why each database mechanism exists, the
  comparative analyses against existing systems, and the measurements.
- `training/ROADMAP.md` — what is planned next.
- `docs/BCIR_LANGREF.md` — the IR this rail indexes, and the cost vector it prices
  with.
- `docs/security/laws.md` — the gate-authoring laws §15.1 summarizes.
