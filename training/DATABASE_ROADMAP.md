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
| Build | twelve slices, all native, priced against the measurement below |

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
| `WHERE <predicate>` | absent | **built** | ten operators, ANDed: equality, set, prefix, the four comparisons, and the two presence tests — resolved through an inverted index or a sorted one and applied inside the kernel scan |
| `BETWEEN` / range scan | absent | **built** | two comparisons on one column intersect to one interval; answered by two binary searches over `order.bin`, priced beforehand from per-part zone maps without reading it |
| `IS NULL` / `IS NOT NULL` | absent | **built** | `col!=?` and `col=?` over any indexed or measurement column; a row with no value satisfies no comparison, and `!=` deliberately admits it |
| `HAVING` | absent | **built** | `--having rows>=10`, `avg<500` and the rest, compared as the exact rational SUM/COUNT rather than a rounded decimal |
| `SELECT <columns>` | absent | **built** | `--select` projects named columns; nothing else is materialized |
| `JOIN` | absent | **built** | the catalog binds the chunk table's columns to the embedding set's row number, so `kind`, `title` and `heading_trail` are one lookup |
| `GROUP BY` / aggregate | absent | **built** | `--count`, `--group-by`, `--distinct` and `--stats` (MIN/MAX/SUM/AVG over a packed numeric column), and the two compose: MIN/MAX/SUM/AVG *per group* — answered from statistics when unfiltered, by intersecting postings when not |
| `ORDER BY <column>` | absent | **built** | `--order-by col[:desc]` without a query: total and stable, absent measurements last in both directions, and with no predicate a slice of the sorted index rather than a sort |
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

All twelve landed, gated by `training/tools/verify_database.py`, which prints its
own check count rather than having one written here. Every check was injected and
watched to fire before its fix went in — twenty-eight defects across S7 and S8
alone — because a slice without a failable gate cannot be shown to have landed
(`docs/security/laws.md` L2, L11). Two of those twenty-eight matter more than
the rest: widening a zone map leaves every answer correct and is caught only by
comparing the map against its own cells, and switching pruning off leaves every
correctness check green, so only an anti-vacuity floor notices.

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
| S7 `WHERE char_count >= n` | not expressible | 0.16 ms | — |
| S8 the same, through the index | 0.29 ms | 0.004 ms | 80x |
| S8 `ORDER BY char_count` | 0.755 ms | 0.035 ms | 21x |
| S9 rows an edit invalidates | 2,160 | 128 | 17x |
| S9 cells read for a narrow range | 2,160 | 624 | 3.5x |
| S10 `rows_in_range` over the table | 0.183 ms | 0.022 ms | 8x |
| S12 `search_chunks.py` | 1,113 lines | 921 lines | — |

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

### S7 — comparisons, the rows that carry no value, and grouped aggregates

The packed measurement columns could be aggregated but not *filtered*:
`OPERATORS` held `eq`, `ne`, `in`, `prefix`, so `WHERE char_count >= 500` was
unexpressible over a column the catalog already stored, summarised and could
answer MIN/MAX/SUM over. Three smaller gaps sat beside it — `GROUP BY` refused
to compose with an aggregate, there was no `HAVING`, and `IS NULL` had no
spelling even though a missing measurement was a state the ordering already had
to reason about.

Four comparisons (`>=`, `<=`, `>`, `<`) collapse to one closed integer interval,
because the columns are integral and `> n` is `>= n + 1` exactly. Two presence
tests spell the rest: `column=?` is `IS NOT NULL` and `column!=?` is `IS NULL`,
with `?` reserved on every column so the grammar answers the same way everywhere.
Each part of the catalog gained a **zone map** — the MIN/MAX/COUNT/NULLS of each
measurement column over its rows — produced by the same function that produces
the whole-corpus statistics, so a part cannot summarise its rows by a rule the
catalog does not use on all of them.

Two divergences from SQL are declared rather than discovered, and both are gated:

- A comparison is *existential*. A row with no measurement satisfies neither
  `char_count >= 500` nor `char_count < 500`, so the two do not partition the
  table. This is SQL's rule, and the one this layout makes easiest to lose: the
  null sentinel is an ordinary negative integer to the packed column, so an
  interval left open at the bottom would otherwise sweep up every gap.
- `!=` is *complement*, which is **not** SQL's rule. `language != c` admits the
  1,532 rows carrying no language, where SQL returns UNKNOWN and drops them.
  Retrieval wants the complement far more often than it wants three-valued
  logic, and the other reading is one term away: `language!=c` with `language=?`
  is exactly `<>`. Both are properties in the gate, so neither can drift into
  the other.

*Payoff:* a comparison at all — 38.5 ms re-reading every chunk record becomes
0.16 ms. The zone map's own share of that is **1.29×**; see below.
*Gate:* soundness by fetching each returned row's own bytes, completeness by a
direct pass over the chunk files, and the pruned answer against a full scan of
the packed column — with a floor that fails if the zone map never ruled a part
out, because pruning changes cost and never the answer.

### S8 — a sorted index per measurement column

The zone map fires — it skipped 7 of 8 parts on `char_count >= 5000` — and it
is still worth only 1.29×, because the parts are subject-shaped: one of them
holds 2,160 of 2,215 rows, so ruling out the other seven leaves 97% of the cells
to read. That is a *mis-assignment*, not a bound: the structure is correct and
the corpus does not give it anything to prune.

So the rows themselves get ordered. `order.bin` holds, per measurement column,
every row number in ascending order — measured rows first by (value, row), then
the unmeasured ones in row order, the split being the column's own `count`. Two
binary searches then answer a comparison exactly, and a row carrying no value is
excluded by the *layout* rather than by a branch that could be forgotten on one
path. 4 bytes per row per column: 17.3 KiB for this corpus, read on first use
like every other sidecar.

| `char_count >=` | rows | share | scan | seek | |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10000 | 7 | 0.3% | 0.296 ms | 0.003 ms | 97x |
| 5000 | 16 | 0.7% | 0.292 ms | 0.004 ms | 80x |
| 2000 | 99 | 4.5% | 0.174 ms | 0.004 ms | 46x |
| 1000 | 454 | 20.5% | 0.192 ms | 0.013 ms | 15x |
| 500 | 1387 | 62.6% | 0.232 ms | 0.035 ms | 6.6x |
| 200 | 2122 | 95.8% | 0.266 ms | 0.088 ms | 3.0x |
| 101 | 2215 | 100.0% | 0.137 ms | 0.082 ms | 1.7x |

There is no threshold in the code because the measurement does not support one:
the index wins at every selectivity this corpus can reach, narrowing to 1.7× but
never crossing. The crossover would reappear if the window had to come back
sorted — `k log k` against `n` — so it deliberately comes back in value order,
and the one caller that needs row order sorts once.

`ORDER BY <column>` with no predicate becomes a slice of the same index rather
than a sort of every row: 0.755 ms to 0.035 ms ascending. Descending is one
stable sort on the negated value — 0.716 ms to 0.261 ms — because reversing the
list would reverse the ties with it, and a page boundary inside a run of equal
values has to land in the same place whichever direction was asked for. With a
predicate it keeps the sort, since walking 2,215 index entries to find the 16 a
narrow predicate admits costs more than sorting those 16. That choice is
returned as a value (`_order_strategy`) rather than made inside a branch,
because both paths return identical rows — so a gate comparing their output
passes whichever one ran, and the decision would otherwise be the one thing
about this slice nothing could observe.

The zone map keeps a job the index cannot do: it lives in the manifest, so a
comparison can be *priced* without reading any artifact at all, which is the
discipline the whole catalog is built on. Estimation reads four integers per
part and returns a bound marked as a bound; evaluation reads the index and
returns the rows.

*Payoff:* 1.7×–97× on a comparison, 21× on `ORDER BY … LIMIT k`, exact range
counts without listing a row.
*Gate:* the index is a permutation, its measured prefix is ascending by
(value, row), its tail is exactly the unmeasured rows — checked on the corpus
*and* on a table built with a gap, because every corpus this gate builds
measures every row and the split-point claims would otherwise be about the empty
set. Then seek and scan must agree on every interval: two implementations of one
lookup, which is the only claim a binary search can make about itself that does
not come out of the same arithmetic.

### S9 — parts stop being subject-shaped

A part was one chunk file, and this corpus keeps 2,160 of its 2,215 rows in one.
A part is the unit of two different things — the grain an incremental rebuild
re-embeds at, and the span one zone map summarises — and a part per file served
neither.

Each file is now cut into blocks of at most `MAX_BLOCK_ROWS` rows, and a part's
content digest covers its own rows rather than its whole file. A block never
spans two files, so a part still belongs to exactly one source and a file's rows
are still a contiguous run of blocks. The size is 128, picked from a table that
lives in the constant's own docstring so the trade is re-derivable rather than
asserted: `catalog.json` is the one artifact read on every load, so its growth is
paid by every caller, while the rebuild grain and the estimate are paid only by
callers that rebuild or plan.

*Payoff:* an appended row used to move a 2,160-row part and now moves one
128-row part, so the coarse filter went from naming most of the corpus to naming
one block; `range_scan` reads 624 cells where it read 2,160, skipping 19 parts
of 24. `catalog.json` grew 24.7 → 29.9 KiB and its parse went 0.077 → 0.094 ms.
*Gate:* parts tile the rows and none is wider than the block size; `part_of`
finds every row's own part and refuses every row outside the set; and one
appended row moves at most two parts, checked by mutating the file cut into the
*most* blocks rather than the first one alphabetically.

**What this did not fix, measured rather than hoped.** Finer blocks do not
rescue the zone map as a selectivity estimate. Over this corpus the bound calls a
25%-selective predicate 98% selective at *every* block size from 32 rows to a
whole file, because an open interval keeps any block that holds one large value
however few of its rows are in range:

| block rows | parts | truth → bound at 5% | at 25% | at 55% | worst |
| ---: | ---: | ---: | ---: | ---: | ---: |
| whole file | 8 | 5% → 98% | 25% → 98% | 55% → 98% | 19.3× over |
| 128 | 24 | 5% → 80% | 25% → 98% | 55% → 98% | 15.9× over |
| 64 | 41 | 5% → 63% | 25% → 98% | 55% → 98% | 12.4× over |
| 32 | 75 | 5% → 47% | 25% → 95% | 55% → 98% | 9.3× over |

So `count_range` is gone, and `plan.estimate` prices a comparison from the sorted
index, exactly. An estimate 16× over the truth is worse than the artifact read it
saves, and the predicate being priced is about to read that index anyway. The
zone map keeps the job it is good at — ruling blocks out of the scan — and loses
the one it was not.

### S10 — a threshold for the seek, because there is one after all

S8 recorded that the sorted index has no crossover and therefore needs no
threshold. That was wrong, and the reason is in the entry point rather than in
the measurement: `rows_in_range` *sorts* the window it seeks, which puts
`k log k` against a linear pass that is already in order. S8's own reopening
condition named this exactly — "a caller needs the seek window back in row
order" — and the caller was `rows_in_range` all along.

| share of table | seek + sort | scan | |
| ---: | ---: | ---: | --- |
| 1% | 0.005 ms | 0.053 ms | seek |
| 25% | 0.026 ms | 0.110 ms | seek |
| 55% | 0.081 ms | 0.120 ms | seek |
| 70% | — | — | char_count crosses |
| 80% | — | — | token_estimate crosses |
| 100% | 0.183 ms | 0.022 ms | scan, by 8× |

`SORTED_SCAN_SHARE` is 0.70, the earlier of the two crossings, so neither column
is pushed past its own. The share is computed from the *exact* count, not the
zone map's bound — deciding on the bound sends predicates the seek wins by 20×
down the scan, which is what the S9 table above measures.

*Payoff:* 4× at 25% selectivity, 8× over the whole table, in both directions
from a decision that costs two binary searches.
*Gate:* both strategies must be chosen across the interval sweep, and both must
return identical rows. The second half is why the first is needed: a threshold
stuck on one path is invisible in every answer, so only an anti-vacuity floor
notices.

### S11 — the database layer, on more than one host

The catalog, the planner, the predicate path, incremental rebuilds and
generations ran in the ubuntu-only LLVM training corpus job. They now also run in
**Host portability** — the job whose purpose is host adaptation — on
ubuntu-latest and windows-latest, and in the aarch64 oracle job, all three with
`--require-native`.

Running somewhere new is not the same as being pinned there, so the two host
assumptions the layer makes are checks now. `check_byte_order` decodes every
packed artifact a second time with an explicitly little-endian struct and
requires it to match the `array` fast path, whose swap branch no host in this
matrix executes. Its scope is declared and narrower than it looks: both readings
read the file as little-endian, so a file *written* big-endian passes here and is
caught instead by the differential against the corpus, whose JSON has no byte
order to get wrong. That division was confirmed, not assumed.

One real divergence fixed: `generations.publish` renames a staged directory into
place, and POSIX and Windows disagree about what that does if the destination
appears in the window between the check and the rename. The outcome is now
decided in the code rather than by the host.

### S12 — an interface per subsystem

`catalog.py` and `plan.py` are the engine. `training/tools/db/` is what the rest
of the tree talks to, split by *consumer need* rather than by engine structure,
because those are different shapes:

| module | the subsystem it serves |
| --- | --- |
| `db/engine.py` | opening the engine: one module loader, one way to open a catalog |
| `db/relational.py` | which rows a predicate admits, in what order, which page |
| `db/analytics.py` | how many, grouped by what, aggregated how, filtered by HAVING |
| `db/retrieval.py` | the rows a ranked search may consider, and their text and columns |
| `db/ingest.py` | the canonical row order, what a part is, and which parts moved |

`generations.py` is the history interface already and is unchanged — it was
exactly this shape before the package existed.

The functions moved verbatim out of `search_chunks.py`, which lost 192 lines and
is now a command-line front end over the interfaces. Behaviour is unchanged: 14
of 16 pre-existing invocations are byte-identical, the two that differ print the
S9 estimate correction, and the embedding rail produces `vectors.f32`,
`vectors.q15` and `index.jsonl` byte-for-byte as before.

*Why it exists:* the sibling-module loader was carried in three copies that
disagreed. One returned any cached module that happened to share the name; one
executed a fresh module and then discarded it in favour of whatever
`setdefault` had kept; one checked that the cached module came from the file it
meant. Only the last is correct, and it is now the only one.

*Gate:* a boundary check reads every tool under `training/tools/` and refuses a
second door to the engine — importing `catalog` or `plan` directly, or loading
either by path — with `plan.py` and `generations.py` declared as the engine's own
insiders. Its scope is stated in its docstring: aliases, `__import__` and
anything assembled at run time belong to a linter, not to this rail. Reaching the
engine *through* `db.engine` is not a violation; the package is the door, however
a caller knocks on it.

**One finding left open rather than folded in.** Ten more tools under
`training/tools/` each carry their own copy of that loader. None of them loads
`catalog` or `plan` — they load other sibling tools — so none crosses the
database boundary this slice is about, and widening the slice to catch them would
have made it a tools refactor wearing a database slice's name. `db.engine.load`
is generic, so the migration is mechanical; it is recorded here rather than done
quietly.

## What is verified where

`verify_database.py` used to run only in the **LLVM training corpus** job, which is
ubuntu-only, as the whole `training/` rail has always been. Since S11 it also runs
in **Host portability** on ubuntu-latest and windows-latest, and in the **aarch64
oracle** job, each with `--require-native`. The catalog, the planner, the predicate
path, incremental rebuilds and generations are therefore exercised on two operating
systems and two architectures rather than on one host.

What is still not covered is a big-endian host, and the swap branch in each packed
reader exists for exactly that. `check_byte_order` is what stands in for it: on a
little-endian host it confirms the files are what the format declares, and on a
big-endian one the swap is the only thing that could make it pass. That is a
narrower claim than "tested there", and it is labelled as one.

Two things do cross that boundary by construction rather than by coverage. Every
sidecar is packed little-endian explicitly and byteswapped on read, the way the
embedding set's `vectors.q15` already was, so `numeric.bin` and `order.bin` do
not depend on the host's byte order; and the gate requires two builds of one
chunk table to produce identical bytes, which is what would catch a sort whose
order depended on anything but the rows. Neither is a substitute for running on a
second host. Both were also run under the declared Python floor, 3.11, where the
artifacts came out byte-identical to 3.12's.

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
| ~~the zone map stays worth 1.29x (S7)~~ | **reopened and acted on in S9.** Parts are bounded blocks now. The scan reads 3.5x fewer cells; the *estimate* did not improve enough at any block size to be worth pricing on, so it is priced from the index instead |
| ~~the sorted index stays threshold-free (S8)~~ | **reopened and acted on in S10.** The caller needing row order was `rows_in_range` itself; the crossover is at 70% of the table and is now a named strategy |
| the block size stays 128 (S9) | `catalog.json` stops being read on every load, which is what makes its growth the binding constraint; or a corpus arrives whose files are already small, where splitting buys nothing |
| the ten remaining loader copies stay out of scope (S12) | any of them starts loading `catalog` or `plan`, at which point it is a database boundary question rather than a tools one |

## One finding outside this scope

A refuter checking whether the kernel's filter path is tested found that it is,
thoroughly — but that the branch beside it is not. The short-return path in
`runtime/c/bcir_ai_kernels.c`, where fewer rows are admitted than `top_k`
requests, has never executed under a gate: every test admits more rows than it
asks for. That is a coverage hole in shipped C, unrelated to any of the nine
projects, and it belongs in the oracle's test registry rather than in this
ladder.

---

# Second comparative analysis — six projects, the feature checklist, and S13–S18

The first analysis asked whether to fork or to study, across nine projects, and
answered *study*. This one asks a narrower and harder question: with S1–S12
landed, **what is still missing that a proven system already has**, and what
does this rail do that they do not?

Six projects, chosen for the capabilities they are known for rather than for
resemblance: Apache Spark, Chroma, Convex, SpacetimeDB, RushDB, Supabase. The
brief was explicit that licence is not a constraint — we are reading designs,
not vendoring code — so nothing was dropped on licence grounds. Sixteen agents
read source, nominated mechanisms, and then adversarially refuted each other's
nominations; 1.88M tokens, 660 tool uses, 62 minutes. Everything an agent
claimed that this document repeats has been re-derived by hand against this
tree, and the two places where the agents and the code disagreed are called out
below.

## Fit: one partial, five dropped

| Project | Fit | Why |
| --- | --- | --- |
| **apache/spark** | **PARTIAL** | Catalyst's *structure* transfers — rule-based rewrite over a typed plan tree, with legality gating each rule. Tungsten does not: whole-stage codegen amortises a JIT over 10^8 rows, and this table holds 2,215. |
| chroma-core/chroma | DROP | A vector store with a server, a WAL and a compactor. The vector half we already do exactly (Q15, bit-identical); the rest is the distributed-service problem this rail deliberately does not have. |
| get-convex/convex-backend | DROP | Its value is a reactive transaction log driving subscriptions. There is no subscriber here — no server, no client, no session. |
| ClockworkLabs/SpacetimeDB | DROP | Same shape: the database *is* the server, and the design is inseparable from the WASM module host. |
| rush-db/rushdb | DROP | A graph layer over Neo4j. The corpus has no graph; `heading_trail` is a tree, and a tree is already a path prefix, which we index. |
| supabase/supabase | DROP | A composition of Postgres, PostgREST, GoTrue and Realtime. The composition is the product; there is no mechanism to lift that is not "run Postgres". |

The one mechanism that survived from all of it is in the smallest project on
the list, and it is not a database mechanism at all: **PostgREST's rule for
quoting a value that contains the separator**. That is the whole harvest of 22
nominations, and it turned out to be the fix for a live wrong answer in our own
predicate grammar — see below.

Two findings from reading Spark are worth recording because they cut against
the expectation:

- **Spark has no transaction manager either.**
  `sql/catalyst/.../TransactionUtils.scala` is a 38-line shim. The ACID story
  in that ecosystem lives in Delta Lake, which is a *separate project* whose
  mechanism is exactly the one S13 implements: immutable files plus one
  atomically-swapped pointer. We did not import that from Delta; we arrived at
  it from BCIR's own generation discipline, which is the same idea and predates
  this analysis. That is a convergence, not a borrowing, and it is the reason
  S13 was a small change rather than a new subsystem.
- **Catalyst independently validates legality-before-cost.**
  `Expression.deterministic` gates whether a rewrite may fire at all, and
  `CostBasedJoinReorder` bails out entirely when row counts are unavailable
  rather than guessing. That is `K_BCIR`'s rule — legality decides, cost
  ranks, and a missing statistic is a refusal rather than an assumption —
  reached by a different project for different reasons.

## What this rail does better

Three, and each is checkable rather than asserted:

1. **Every statistic is exact, and a bound says so.** Catalyst estimates
   selectivity from histograms with an independence assumption; the error is
   unbounded and unreported. Here a count is a count and a bound is labelled
   `[bound]` — and S15 below turns almost all of the remaining bounds into
   counts. The one thing this rail will not do is what every surveyed planner
   does by default: report a guess in the same shape as a fact.
2. **Byte-identical artifacts across hosts.** Every surveyed system names
   artifacts with a timestamp or a UUID. TileDB's `generate_timestamped_name()`
   is the clearest case: the same array built twice is two different arrays.
   Here the name *is* the content, so two hosts building the same corpus agree
   on the bytes and on the names — which is what makes a generation digest mean
   anything.

   This claim was **false on Windows** until defect 5 below, and it is worth
   being blunt about that rather than quietly fixing it: the property this
   ladder puts first, over every system it has been compared against, did not
   hold on one of the three hosts in its own CI matrix, and no gate could see
   it. Fixes 2 and 5 are what make it true.
3. **The gate proves it can fail.** Every claim in this ladder was landed by
   injecting its own defect and watching its own named check fire. None of the
   six projects gates its planner's *decisions* at all; S18 does.

## The fourteen-feature checklist, answered

Asked for honestly, including the ones that are "no" and will stay "no".

| Feature | Status | Notes |
| --- | --- | --- |
| ACID transactions with relational tables | **added (S13)** | A: a publish is one pointer swap. C: S14/S17 constraints. I: a reader is pinned to the set it opened. D: `fsync` before the swap. Single-writer; there is no concurrent-writer isolation because there is no concurrent writer. |
| Type-safe APIs and table access | **added (S14)** | One declared table; an unknown column is refused with the list of real ones. |
| Application logic inside the engine | **has** | Not stored procedures — the cost model, legality laws and plan selection *are* engine-resident logic, and `bcir_ai_q15_topk` is a kernel the engine dispatches to. |
| Deterministic reducers with atomic writes | **has** | Every build is a pure function of the chunk bytes; every publish is atomic (S13). This is SpacetimeDB's property, reached without its runtime. |
| Interactive performance without extra caches | **has** | 76 ms cold, no daemon, no warm cache, no second tier. There is nothing to cache *into*. |
| Self-host or managed cloud | **has (self-host)** | It is a directory. Managed cloud is not a goal. |
| Built-in real-time sync | **no, by design** | Requires a server and a subscriber. Neither exists, and adding them would be the largest change in this tree to serve no caller. |
| Auto-updating queries | **no, by design** | Same reason: a query that updates needs something to update *to*. |
| SDKs for real-time apps | **no, by design** | Same. |
| In-memory with client mirroring | **no, by design** | The whole table is 356 KiB of artifacts; it is already effectively in memory after one read. There is no client to mirror to. |
| Client cache mirroring | **no, by design** | Same. |
| Unified runtime (state + logic + sync) | **partial, by design** | State and logic are unified in one process. Sync is the half that is deliberately absent. |
| Built-in authentication | **no, out of scope** | A single-process reader of a repository directory authenticates through the filesystem. Adding an auth layer over `git`-tracked content would be theatre. |
| Postgres wire format | **no, out of scope** | The wire protocol's value is existing clients. None of them would understand `--query` ranking over Q15 vectors, which is the reason this rail exists. |

The five "by design" rows all reduce to one fact, and it is worth stating once:
**this database has no server, and nearly every modern database feature on that
list is a consequence of having one.** That is not a gap to close later. A
corpus that ships inside a repository and is read by a build step has no client
to sync to, and a sync mechanism with no subscriber is cost with no benefit.
The reopening condition is written down below.

## Four defects the analysis found in our own rail

The survey's most valuable output was not a feature to import. It was four
defects in code that was already gated, three of which produced *wrong answers
with exit 0*.

### 1. A value containing the separator asked a different question

PostgREST's rule exists because without it a value containing a comma does not
fail — it silently becomes two values. On `origin/main`, measured:

```
language!=llvm,markdown   ->  ne ('llvm,markdown',)   admits 2215 of 2215 rows
kind="code,prose"         ->  in ('"code', 'prose"')  admits    0 of 2215 rows
```

The first is the serious one. A caller excluding two languages got **every row
in the table**, exit 0, no warning — because `=` read the comma as a set
separator and `!=` read it as an ordinary character, so the complement of a
value nothing carries was everything. That is two value grammars under one
operator table, which is exactly the disagreement `docs/security/laws.md` L12
forbids.

*A correction to an earlier framing of this defect.* It was first reported here
as affecting the 103 titles and 393 heading trails in this corpus that contain
commas. That overstates it: `title` and `heading_trail` are `text` columns and
are **not filterable at all**, so a predicate naming one is refused loudly with
the list of indexed columns. The live silent failure is the one measured above,
on an indexed column, and it is worse than the overstated version because it
returns rows rather than an error.

Fixed by adopting PostgREST's rule exactly: a value is quoted whole or not at
all, `\"` and `\\` are the only escapes, and everything the grammar cannot read
back unchanged is refused rather than guessed — a quote inside an unquoted
value, text after a closing quote, an unknown escape. The arity check now
happens *after* the split, so `=` and `!=` read one value grammar and stay
complements for every value. `EXPLAIN` prints through `spell_value`, so a
printed predicate parses back to the predicate that printed it; that round trip
is gated over 15 values × 3 operators plus sets.

### 2. A bound was reported as a count, for most of the table

`estimate_exact` asked whether each *term* was counted exactly, where the claim
being made is about their *intersection*. Measured on `origin/main`:

| conjunction | priced `[exact]` while wrong |
| --- | --- |
| one indexed column | 0 of 29 (0%) |
| two indexed columns | **191 of 236 (80%)** |
| three indexed columns | **526 of 544 (96%)** |

`kind=code AND language=asm` claimed one row where there are none. A planner
that is confidently wrong about selectivity is the failure mode the catalog's
whole "exact statistics, no histograms" design exists to avoid, and the label
had quietly opted out of it. S15 below does not just fix the label — it makes
the number exact.

Two smaller instances of the same shape were fixed with it: `subject=llvm,llvm`
summed the marginal twice and estimated 4,320 rows of a 2,215-row table, still
labelled exact; and an `IN` list now sums over distinct values.

### 3. `catalog.json` was not reproducible across directories

Built from two directories holding byte-identical chunk files with modification
times 25 years apart: `postings.json`, `locator.bin`, `numeric.bin`,
`order.bin` and `ids.txt` were byte-identical, and **`catalog.json` differed**
— on `mtime_ns` and on `chunk_dir`. The generation *id* was safe (it digests
names and content only), but `tree_digest` covers `catalog.json`, so a
generation's verification digest was not reproducible across checkouts.

**The determinism gate could not see this**, because it built twice from *one*
directory, where the paths and mtimes are equal by construction — a check whose
two sides cannot differ in the way the artifact actually differed. That is L2,
and it is the most instructive thing in this whole analysis: the gate was green
for the entire life of the defect.

Fixed by removing the mtime and the directory from the manifest, and the gate by
building from two different directories with different stamps and requiring both
to actually differ before comparing.

Removing the mtime cost more than expected, and the cost was worth paying in
full rather than working around. The mtime was the cache key for a freshness
fast path: a file whose size *and* mtime matched its record skipped its digest.
Keeping the fast path on size alone was measured and rejected — it widens the
declared scope from "size and nanosecond mtime deliberately preserved" to *any
same-size edit*, which on JSONL is every single-character correction anyone will
ever make. So the fast path is gone and every load digests every chunk file:
**0.24 ms → 3.39 ms**, one path instead of two (L12), and no scope note at all.

### 4. A type error in a measurement was silently a null

`char_count = "not-an-int"` built a clean catalog, exited 0, and appeared
downstream as one more null among the genuine ones. The gap and the error shared
a spelling, and nothing afterwards could tell them apart. S17 generalises the
fix from type to value.

### 5. The corpus had different bytes on Windows

Found by CI, on the first push that pinned a corpus fingerprint into the
repository. The Windows host-portability job reported:

```
S18: plan baseline: recorded against a different corpus
  recorded: e7fe3a4fde6dad9364297425b16671ceac6fec6fe416db263484117fdec1fb10
  now:      e173870819daa383e263b35ce267fe30f4e7f2e030302c5f32c7f2591adbda3f
```

`build_chunks.py` wrote chunk files with `open("w", encoding="utf-8")` and no
`newline=`, so Python's text mode translated every `\n` to the platform's line
ending. On Windows every chunk file was CRLF. Reproduced exactly — taking the
Linux corpus and replacing `\n` with `\r\n` reproduces both digests above,
byte for byte.

Nothing reads those files as text and cares. But everything about them is
**digested**, and all of it moved:

| | with CRLF |
| --- | --- |
| corpus `fingerprint` | differs |
| every recorded per-file `sha256` | differs |
| every part's content digest | differs |
| `locator.bin` | differs — byte offsets shift with every line before them |
| `generation_id`, `tree_digest` | differ, being functions of the above |
| `postings.json`, `ids.txt`, `numeric.bin` | identical — they hold logical content, not bytes |

So two hosts disagreed about the content address of an identical corpus, and a
generation published on one could never be verified on the other. This is the
property stated above as the thing this rail does better than everything it was
compared against; it had never been true on Windows, and it took pinning a
fingerprint into the repo for anything to notice.

The repository had already *declared* the rule — `.gitattributes` opens with
`* text=auto eol=lf` — for everything it tracks. What it could not reach is
`build/`, which is generated rather than tracked. The fix extends the same rule
to the tools that generate it: all 17 text-mode artifact writers under
`training/tools/` now pin `newline="\n"`, and `check_line_endings` holds it
there with both halves it needs — a static read of every non-`verify_` module
for a write that lets the host choose, and a read of the bytes actually on disk
(L11: a witness must hit the law it exists to test). The RED witness for the
dynamic half pins CRLF explicitly, which the static half accepts, so neither
half is redundant.

**And then the rest of the repository.** The first pass fixed `training/tools/`
and recorded the same shape at 37 sites under `bcir/` and 29 under `tools/` as
out of scope. That audit is now done, and it was worth doing: the sweep also
found a subtree the first pass had missed entirely — `training/llvm/tools/`,
whose generators were never scanned because the first scan walked
`training/tools/` and not `training/`.

**77 sites pinned across 42 files**, in `bcir/`, `tools/`, `training/` and
`.claude/`. Sixty-seven were classified for exposure against their actual
consumers; **fifteen were exposed**, each with a cited consumer:

| exposure | sites | evidence |
| --- | --- | --- |
| a generator writing a **tracked** file | 6 | three MLIR corpora from `bcir/kbcir/differential.py`, plus `runtime/c/bcir_q8_tables.h`, `mlir/test/passes/structural_corpus.mlir`, `.claude/context/BCIR_DIGEST.md` |
| generated source whose **object bytes** enter an artifact bundle | 2 | `bcir/codegen/codegen.py` → `add_codegen_result` → `add_native_object` |
| bytes read back and **`sha256`'d into a manifest** | 2 | `tools/models/run_hardware_rl_gate.py`, `run_hosted_model_gate.py` |
| a report that is **committed** | 5 | `docs/security/audit-2026-09-04/*.json` |

The sharpest evidence is in CI itself: `.github/workflows/ci.yml` regenerates
two of those MLIR corpora and gates them with `git diff --exit-code`. That is
already a byte gate on a regenerated tracked file — a trip-wire that would have
caught this the first time it ran on a host that chose CRLF, and never did,
because it runs on ubuntu.

Fifty-one sites were judged **not exposed** — a throwaway temp source consumed
by a compiler in the same process, a human-read report, a log, a sysfs write —
and **pinned anyway**. The rule is total inside its declared scope, with no
allowlist, because an exemption list of "the sites we judged safe" is a second
thing to maintain and is wrong the first time somebody digests one of them
(L15). Pinning costs nothing: no tree here generates a `.bat`, `.cmd` or
`.ps1`, the only artifacts that would want CRLF. What is excluded is structural
and decidable — a test or gate module, whose fixtures are written and read in
one process on one host.

The gate is `bcir/tests/test_line_endings.py`, in the oracle suite so it runs on
every host in the matrix, with three halves: the source rule over four trees,
a witness for the matcher itself, and — derived rather than curated — *no
tracked text file in the repository carries a carriage return*, which is
`.gitattributes`'s own `* text=auto eol=lf` checked instead of trusted, across
all 1,998 tracked files. `verify_database.py` kept only the half no other rail
can do, the built corpus on disk; its copy of the source scan is gone, because
two scanners for one rule had already begun to differ about what counts as a
text write (L15).

The finding is registered as **L24 — an artifact's bytes do not depend on the
host that wrote them** (`docs/security/laws.md`).

## The S-ladder — S13 to S18

### S13 — a publish is one transaction

`write()` was six `os.replace` calls into one live directory, manifest last.
Each was atomic on its own, which is a much weaker claim than it reads as:
between the second and the third the directory held four old artifacts and two
new ones, and a reader holding the old manifest that then read a new sidecar
found a digest mismatch. That failed *loudly* — the manifest vouches for every
artifact — so it was never a wrong answer. It was a rebuild that could not be
done while anything was reading.

Now a catalog root is one mutable file and an immutable tree:

```
build/training/catalog/
  CURRENT                                  <- the only thing that ever changes
  sets/7c68989215e5e4dc5f6bd4b41efc8cf1/   <- named by the digest of its own contents
      catalog.json  postings.json  locator.bin  ids.txt  numeric.bin  order.bin
```

Artifacts are written into a staging directory nothing has read from because
nothing knew the name, `fsync`ed, moved into place under their content-addressed
name with one rename, and then `CURRENT` is swapped. A reader holds the old name
or the new one, and both name a complete set. Because the name is the content,
republishing an unchanged catalog finds the set already there and only moves the
pointer; nothing inside a published set is ever rewritten. The set the pointer
replaced is kept, so a reader that resolved `CURRENT` an instant before the swap
finishes its work; older sets are pruned.

Gated by cutting the publish at **every** `os.replace` it makes and asking what a
reader resolves to. Across all nine interruption points a reader saw exactly two
things — the complete old catalog or the complete new one — and never an
incomplete set, a torn set, or an error. The RED witness is a one-line reorder
that moves the pointer before the set it names exists.

ACID, stated precisely and without overclaiming: **A** is the pointer swap,
**C** is S14/S17, **I** is that a `Catalog` binds its set directory at load so a
rebuild landing mid-read cannot move it, **D** is `fsync` on every artifact and
on the directories before the swap (on POSIX; Windows has no directory handle to
sync and its rename is already ordered, which is declared in `_sync_directory`
rather than swallowed). There is no multi-writer isolation and this does not
claim any: there is one writer.

### S14 — one declared table, read by two rails

`training/schema/chunk-v1.json` declared eighteen fields and their types.
`catalog.py` named six of them in three Python tuples and said nothing about
what they hold. The chunk rows carried whatever the builder put there. **Nothing
reconciled any pair of the three** (L15), so renaming a field in the JSON while
the Python kept indexing the old name was a green run that produced a postings
list of one entry — every row, null.

`training/tools/schema.py` is now the Python rail: one `Column` per field with
its type, its role (`key`, `indexed`, `path`, `numeric`, `text`, `structural`),
whether it is required, whether it admits null, and its constraints. The
catalog's three tuples are derived from it. The JSON Schema stays what an
external consumer validates against and is **not** generated from it — the two
are read out of their own sources and reconciled by `check_schema`, which
refuses any disagreement about which fields exist, which are required, which
admit null, what type each holds, and what values each admits.

A table is a *parameter* of a build rather than a global, and that is load-bearing
rather than tidy. The chunk contract requires every row to carry a `char_count`
of at least 1 — which is true, and enforced — and it therefore means no
schema-valid corpus can ever leave a measurement absent. The packed column must
still spell absence and comparisons must still answer for it, so the checks that
cover that declare a relaxed table that admits it. The alternative was to weaken
the contract so a fixture could reach a state the contract forbids, which is a
hole dressed as a test (L2).

### S15 — exact joint statistics, because these columns are small

Every surveyed planner estimates conjunction selectivity with sketches and an
independence assumption, because the joint distribution is expensive in the
general case. That reasoning does not transfer here, and noticing why is the
whole slice: **an indexed column is by role one whose distinct-value map is a
handful of entries.** `kind` has 4 values, `subject` 8, `language` 17. The full
joint distribution over any pair is bounded by the product of two handfuls, and
the corpus is static, so it is computed once at build.

The entire joint distribution over all three pairs is **61 non-empty cells, 978
bytes — 3.1% of `catalog.json`**, which grew 30.0 → 30.6 KiB. Total artifacts
355.9 → 356.5 KiB.

Every operator that is a statement about *which values* a column holds folds to
one shape — a set of index keys — after which there is no polarity left to reason
about: `!=` is the complement, `IS NULL` is the null key alone, `IS NOT NULL` is
everything else, `IN` is the set as written. One rule instead of sixteen
operator-pair formulas (L14). Two columns are then a cross product of pair
lookups; three or more have no stored statistic, so the answer is the **tightest
pair**, which assumes nothing — each pair counts a strictly larger set — and is
far tighter than the tightest marginal.

| conjunction | main | this branch |
| --- | --- | --- |
| 1 indexed column (29) | 29 exact | 29 exact |
| 2 indexed columns (236) | 45 exact, **191 falsely exact** | **236 exact**, 0 wrong |
| 3 indexed columns (544) | 18 exact, **526 falsely exact** | 511 exact, 33 tight bounds, **0 loose bounds** |

`subject=llvm AND kind=code AND language=llvm` went from a bound of 75 to a
bound of 27, which is the exact answer. The 511 "exact" at three columns are the
empty conjunctions: an upper bound of zero is a count of zero whatever the terms
were, and that is the case a planner acts on.

A cap (`MAX_JOINT_CELLS`) bounds the cross product where the work is committed
rather than asserting something about the data (L3); above it the estimator
declines and the bound stands.

### S16 — an operator nothing exercises is an operator nothing checks

The gate ran 1,010 checks, and an eleventh operator could have been added to
`plan.OPERATORS` and passed every one of them. Nothing reconciled the declared
language against the tables that exercise it, so coverage was whatever somebody
remembered to add.

`check_operator_coverage` now reconciles `plan.OPERATORS` against three rails,
because an operator can pass one and fail another: the grammar table says it can
be **written**; resolution against the corpus says it can be **answered**, with
its estimate bounding its own answer; and `plan_baseline.QUERIES` says its
**decision is pinned**. An operator missing from any of them fails the gate,
naming which. The grammar table was hoisted to a module constant so the coverage
check and the grammar check read one source rather than two that agree by habit
(L15). RED witness: adding `"between"` to the operator tuple.

### S17 — domain and NOT NULL constraints

"This is an integer" was the only thing a build ever checked, and the
interesting failures are not type failures. A `kind` of `"prose "` with a
trailing space types fine, indexes fine, and silently creates a fifth kind that
no query written against the documented four will ever match. A `char_count` of
`0` types fine and makes every average wrong.

Constraints are declared on the column — required, nullable, enumerated domain,
minimum — and enforced against the row **as it came off disk**, before the
catalog projects out the six columns it indexes. Checking the projection would
check the six copied out and say nothing about the twelve left behind, which is
where a renamed field hides. Refusing an undeclared field is the other half of
the same rule: a rename otherwise reads as one field going absent *and* an
unindexed one appearing — two findings for one cause.

Verified end to end: 17 distinct violations each refuse with exit 1, naming the
offending chunk and column, writing no catalog; the two legal shapes (`language`
null, `language` absent) still build. The trial list is *derived from the table*
and then reconciled against it, so a column that declares a constraint with no
trial fails the check for lack of coverage.

### S18 — the decision, not just the answer

Every other gate here checks that an answer is *right*. None checked that it was
reached the same way — and both real estimator defects this tree has had were
invisible to correctness gates by construction. The zone map calling a
25%-selective predicate 98% selective, and the conjunction bound reported as a
count, both produced correct rows the entire time they were wrong.

`training/plans/baseline-v1.json` pins 26 requests: the parsed predicates, the
admitted rows, the estimate and whether it was a count or a bound, the plan
chosen under each of the four objectives, how many candidates were illegal, the
range strategy and how many parts were pruned, the order strategy, and the
groups kept after `HAVING`. No wall time — that is a property of the runner, not
the plan — and no scalarized cost, which moves whenever the model is recalibrated
without the *decision* changing.

A moved decision is a finding, not a failure of the tool: improving an estimate
moves the file and so does breaking one, and the gate's job is to refuse to let
either happen without somebody reading the diff. The file carries the corpus
fingerprint, so comparing against a different corpus is refused rather than
reported as 26 regressions — which makes adding corpus content a deliberate
re-record, and that is the point.

## What was scouted and declined

71 mechanisms outside the given checklist were scouted: 14 already present, 12
worth building (the six above plus the four defects, now all landed), 6 worth
reopening later, **53 not at this scale**. Two declines are worth recording
because the measurement is the interesting part:

- **Approximate nearest neighbour (LSH, IVF, HNSW).** Exact all-pairs cosine
  over this corpus is 2,452,005 pairs in **58.9 s of pure Python**. Every ANN
  structure trades exactness for time we do not need to save, on a rail whose
  `accuracy` axis is a legality question first.
- **Approximate quantiles (t-digest, KLL).** `order.bin` already holds every
  measurement column sorted. An exact quantile is an O(1) index into it and
  costs **zero new bytes**. Sketching an answer we can look up would be strictly
  worse on every axis.

## What this slice cost

Measured on this host, interleaved A/B against `origin/main`, 15 paired runs,
median. `[wall, indicative]` — these gate nothing.

| | main | this branch | delta |
| --- | --- | --- | --- |
| `--count --where kind=code` | 69.8 ms | 76.3 ms | +6.5 ms (+9%) |
| `--count --stats char_count` | 68.3 ms | 75.7 ms | +7.4 ms (+11%) |
| `--count --group-by subject` | 73.7 ms | 78.4 ms | +4.6 ms (+6%) |
| `catalog.build` (2,215 rows) | 58.2 ms | 99.9 ms | +41.7 ms |
| `Catalog.load` | 0.24 ms | 3.39 ms | +3.15 ms |
| artifacts on disk | 355.9 KiB | 356.5 KiB | +0.6 KiB |
| gate checks | 1,010 | 1,812 | +802 |

The command delta decomposes into the freshness digest (+3.15 ms, the price of
fix 2) and the `schema` import (+1.7 ms by `-X importtime`); the rest is noise
at this resolution. The build delta is per-row schema validation over 2,215 rows
× 18 columns, paid once per rebuild and never by a query.

**Every one of those numbers is a cost, and every one buys a property that was
false before.** The 9% is worth stating plainly rather than hiding: it is what
reproducible artifacts and a refusal-on-malformed-input cost on a corpus this
size, and it is the kind of trade this ladder should keep making.

## Reopening conditions

| Verdict | Reopens when |
| --- | --- |
| the five sync/client features stay absent | something other than a build step reads this corpus — a second process, a service, an editor — at which point "no subscriber" stops being true and the whole row changes |
| Postgres wire format stays out of scope | a caller appears that wants relational access *without* ranked retrieval, since that is the half an existing client could use |
| built-in authentication stays out of scope | the corpus stops being repository content, i.e. when filesystem permissions stop being the access control |
| joint statistics stay pairwise (S15) | a query shape appears whose three-column bound actually costs something; the tightest-pair bound was exact on every non-empty triple in this corpus |
| the freshness digest stays unconditional (fix 2) | digesting the corpus dominates a command, at which point the answer is a design that stays *one* path, not a second one |
| the line-ending rule stays scoped to `training/tools/` (fix 5) | a text artifact under `bcir/` or `tools/` is digested, or compared across hosts; 66 unpinned writes are waiting there, and only the byte-written frozen formats are safe by construction |
| ANN indexing stays declined | the corpus passes ~50,000 rows, where 58.9 s of exact all-pairs becomes minutes |
| Spark's Catalyst stays study-only | a rewrite rule appears that is worth expressing as a rule rather than as code, i.e. when there are enough of them to need a driver |
| the previous published set is kept, no more (S13) | a reader can hold a catalog across more than one republish, which today it cannot: there is one writer and it runs to completion |

## Method, and what would overturn it

Sixteen agents read source in six cloned repositories, nominated mechanisms with
file-and-line citations, and then adversarially refuted each other's
nominations; 22 nominations, one survivor. That ratio is a property of the
refuter instruction as much as of the mechanisms, and it is why the survivor is
the only agent output this document treats as a finding rather than as a lead.

Everything else here was re-derived by hand: the four defects were reproduced
against `origin/main` in a separate worktree, every measurement above was taken
on this host with the comparison interleaved, and every claim is gated. The RED
sweep is the load-bearing part — eight defects injected, each caught by its own
named check, listed in the pull request.

Two things would overturn the central verdict. If a second reader of this corpus
ever appears, five "by design" rows become gaps and the server question reopens
properly rather than by accretion. And if the corpus outgrows the scale that
makes exact statistics cheap — the joint distribution is small *because* an
indexed column is small — S15's argument stops holding, and the honest answer
then is a bound that says it is a bound, which is where this started.
