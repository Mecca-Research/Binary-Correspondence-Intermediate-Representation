# Constructing a small language model within BCIR — roadmap

> **Status: planning document, no slice landed.** This roadmap fleshes out the
> proposal *Constructing SLM within BCIR* into something this repository can
> build: it names the existing machinery each part wires into, gives the
> mathematical schema of the hybrid SQL/vector lookup the model consults at
> inference time, specifies the token-level pipeline that serializes Unicode 17.0
> character records for the tokenizer, and lays the work out as gateable slices
> in the style of [`DATABASE_ROADMAP.md`](DATABASE_ROADMAP.md). It is a companion
> roadmap in the repository's authority order (LangRef → generated evidence and
> implementation → current-state audit → `docs/BCIR_MASTER_ROADMAP.md` →
> companion roadmaps → research notes) and it does not redefine the spine, the
> two-truth quarantine, or any frozen ABI.
>
> **It depends on [`UNICODE_ROADMAP.md`](UNICODE_ROADMAP.md).** Every slice here
> that reads a character property names the Unicode slice it needs; the Unicode
> roadmap depends on nothing here. Within the ML program it extends **Phase F1**
> of `docs/machine-learning/BCIR_ML_AI_INTEGRATION_ROADMAP.md` — sequence-interface
> promotion, whose pin list already names "the Unicode database" and whose §8.5
> records "Built-in tables (Unicode DB → the F1 tokenizer …) ABI machinery
> exists; F1 unstarted" — and it sits in capability track CT3 (frontends) as that
> document's §4 places F1. No new capability-track number is created.

## 0. The proposal, restated with its claims labeled

The proposal's thesis is that a small model (the proposal says 500M–3B
parameters) can carry less of language's *structure* in its weights if that
structure is supplied deterministically: the Unicode Character Database as a
runtime substrate (code points, categories, scripts, operators, radical-stroke
decomposition), the collation, confusability and IDNA algorithms as exact
engines, a relational lexical graph (roots, derivations, calques, lexical
fields) as tables the model queries, and a training curriculum that first
teaches the substrate, then morphology over trees, then graph traversal, then
register and rhetoric. The mechanisms it names — tabular pre-alignment records,
tree-constrained attention masks, recursive composition
`h_node = tanh(W [h_left || h_right] + b)`, a hybrid SQL/vector retrieval loss —
are each concrete enough to build and gate.

This repository's rule is that a claim carries its label. So, before anything is
built:

| Claim in the proposal | Label here |
|---|---|
| A 500M–3B model "can attain deep reasoning fluency … on a fraction of typical computational budgets" | **hypothesis**; the evaluation slices (§7) are how it would be measured; nothing in this repository has trained a model at that scale (`docs/BCIR_MASTER_ROADMAP.md` §2.3: the 32M, byte-native and progressive models are untrained at useful scale) |
| "159,801 characters" in Unicode 17.0 | an expectation the Unicode roadmap's pin gate asserts; not restated as a fact here |
| The hybrid SQL/vector engine "replaces statistical guesswork" | the *lookup* is exact and deterministic (§2 gives the rules); what the model does with it is learned and stays labeled learned |
| Deterministic morphological base forms via UCA | **exact**: level-1 collation keys are computed, not learned (UNICODE_ROADMAP §6.2) |
| MMLU / ARC / GSM8K / MATH / CoLA results | **external benchmarks**; run only when their data is admitted as a cache/build product with provenance, never committed; no published number is restated as BCIR-measured (`docs/machine-learning/THIRD_PARTY_MODELS.md`) |
| Any throughput or cost number in this document | **modeled** unless a row says *measured* with a host and provenance; the planner's constants are labeled modeled and never gate (`training/tools/plan.py:23-27`) |

Everything BCIR emits is TMSAO-4 (heuristic, no optimality claim) until the
GEM+ lower-bound stack lands; that applies to every plan the lookup chooses.

## 1. What exists today, by name

The proposal describes the model as if it began from nothing. It does not. The
inventory below was read out of the tree for this document (file and line for
every item); §3–§5 wire into it, and the slice ladder in §6 names what is missing.

### 1.1 Four text-entry rails, none of them Unicode-versioned

- `bcir/hosted/training/bpe.py` — `BytePairTokenizer`: four specials
  (`<pad>=0, <bos>=1, <eos>=2, <unk>=3`) plus 256 byte tokens (id = byte + 4),
  canonical merges, NFC+LF normalization *by the host interpreter*, canonical
  JSON under schema `bcir.byte_bpe.v1` with a closed key set (`from_json`
  refuses any other), content-addressed by `sha256` of that JSON, and threaded
  through `token_source_from_corpus` → `CorpusManifest.tokenizer_sha256` →
  checkpoint → export → the BCIRQ8 header (`weights_io.py`, bytes 184..216).
- `bcir/kbcir/byte_latent.py` — `ByteVocabularySpec` (260 ids: octets 0..255,
  `bos=256, eos=257, pad=258, mask=259`, frozen), `encode_raw_sequence` (strict
  UTF-8, **no normalization**, per-scalar `UnicodeByteReference` spans),
  `decode_raw_bytes` (strict or replace); the hosted `byte_latent.py` models
  patch on byte indices with no code-point or grapheme awareness.
- `bcir/frontends/models/tokenizer.py` and `spm.py` — stdlib readers for HF
  `tokenizer.json` (byte-level BPE, no normalization, lossless by construction)
  and SentencePiece `tokenizer.model` (score-merge loop; only the `▁` space rule,
  no NFKC), each digest-tied to `ModelManifest.tokenizer_digest`.
- `bcir/kbcir/sequence_interfaces.py` — the dependency-free contracts of the F1
  portfolio: `canonical_substring_candidate`, `UnigramPiece`/`segment_unigram`
  (exact byte Viterbi), `ContinuedBPEExpansionPlan`, `FrozenTokenCodeSpec`,
  `CrossTokenizerProjection`.

Three incompatible special-id layouts coexist (bpe: pad 0 / bos 1 / eos 2 / unk 3,
bytes from 4; byte-latent: bytes 0..255, bos 256 / eos 257 / pad 258 / mask 259;
the hosted gate: bytes 0..255, unk 256 / `<s>` 257 / `</s>` 258 / `<pad>` 259) and
the C twin (`runtime/c/bcir_q8_model.[ch]`) contains **no tokenizer** — it
consumes ids and carries a 32-byte `tokenizer_sha256` it does not verify.

### 1.2 One production-shaped model, and a laboratory beside it

- `bcir/hosted/models/model.py` — `HostedLlama`: pre-norm RMSNorm decoder over
  the frozen `DecoderSpec` (`bcir/frontends/models/decode.py:46`), half-split
  RoPE, GQA, `F.scaled_dot_product_attention(..., is_causal=True)` **hard-coded
  with no mask argument** (`model.py:74`), SwiGLU without biases, optional tied
  embeddings, `hidden_states()` as the sanctioned boundary for auxiliary heads,
  `forward(input_ids, targets, loss_mask)` with a per-token loss mask. Its
  parameter formula is `decoder_param_count` (`decode.py:229`):
  `vocab·d + n_layers·(2d + 2d² + 2d·kv_dim + 3·d·d_ff) + d [+ vocab·d if untied]`.
- `train.py` — deterministic AdamW (warmup + cosine), CPU float32 only, exact
  resume by batch index; `checkpoint.py` — content-addressed generations
  `step-XXXXXXXX-<model12>-<optimizer12>` committed by `latest.json`;
  `export.py` — HF-layout export that must re-ingest strictly through
  `hf_ingest` and refuses a tokenizer whose bytes do not match
  `corpus_manifest.tokenizer_sha256`.
- `bcir/hosted/training/` — `StageTrainSpec`, SFT / reward / DPO / PPO /
  reasoning-SFT stages, `relational_embedding_targets` + `relational_gram_loss`
  + `HostedEmbeddingStudent` + `train_embedding_distillation` (frozen
  cosine-matrix distillation with a torch-free floor), `TeacherProvider` /
  `RecordedTeacherProvider.from_jsonl` (frozen targets, never a gradient
  channel), the append-only `TrainingPipelineLedger` with a fixed stage DAG, and
  `bounded_reasoning_search(prompt_ids, budget, propose, verify)` — torch-free
  test-time search with pluggable `verify`.
- The three 2026-07-22 laboratories: `adaptive_transformer.py` (tied depth,
  variable width, an additive `[L, L]` mask materialized from the dependency-free
  `ReferenceWindowSpec.additive_mask` in `bcir/kbcir/adaptive_transformer.py`
  — the **only** attention-mask seam in the tree), `byte_latent.py`
  (`_SelfAttention.forward(values, *, causal, mask)`), and
  `sequence_interfaces.py` (`_GrowthBlock.forward(values, mask)`,
  `HostedProgressiveLanguageModel`). None may enter the stable Llama
  train-to-C path silently (`docs/machine-learning/BCIR_WHOLE_MODEL_REFERENCE.md`).
- The hosted gate trains 64 CPU steps of a tiny spec (`tools/models/run_hosted_model_gate.py`);
  `run_real_model_gate.py` pins TinyLlama → BCIRQ8 → standalone C parity. Both
  prove composition, not model quality, and say so.

The C training rail (`runtime/c/bcir_train.c`) trains a logistic readout, not a
transformer; there is no dependency-free transformer trainer, no KV-cache in
`HostedLlama.greedy`, no attention mask on the stable model, and no tree,
dependency-parse or recursive-composition structure anywhere.

### 1.3 The native data plane: seven kernels, one ABI

`runtime/c/bcir_ai_kernels.c` exports, under `BCIR_AI_ABI_VERSION 1`:
`bcir_ai_quantize_q8_f64`, `bcir_ai_quantize_q4_f64`, `bcir_ai_q8_matvec_f64`
(+ `_prevalidated`), `bcir_ai_q8_rows_dot_f64` (+ `_prevalidated`),
`bcir_ai_q8_values_validate`, and `bcir_ai_q15_topk` — exact-integer Q15 squared-L2
top-k over int16 rows with a per-row 0/1 eligibility mask. Every kernel borrows
its buffers, takes explicit capacities, forbids overlap, allocates nothing,
validates before writing, and returns 0 / −1 / −2; rounding is half-away-from-zero
literally as `floor(x + 0.5)` mirroring `bcir.kbcir.quantize._round_half_away`.
Memory class `hosted_tool`; header markers "borrowed", "must not overlap",
"allocate no memory" are enforced by `tools/c/check_memory_discipline.py`;
Python reaches them through `bcir/kbcir/native_ai.py` (single-TU build,
content-addressed build stamp) and the training corpus through exactly three
wrappers in `training/tools/bcir_native.py` (`quantize_q8`, `q8_rows_dot`,
`q15_topk`), which turn a host failure into `BackendUnavailable` (an honest skip)
and a kernel refusal into `KernelRejected` (never a skip). The kernels are never a
fallback.

Absent from the ABI, and therefore slices here: softmax, tanh, any activation,
LayerNorm (RMSNorm exists only as the freestanding twin `bcir_rmsnorm` in
`bcir_decode.h`, unbound from Python), matmul beyond matrix-vector, masked
attention (`bcir_gqa_attention_row` is causal-only, unbound), gather/scatter over
quantized storage, generic top-k over logits, and any string or code-point
primitive.

### 1.4 The database that already does hybrid lookup

`training/tools/` is a database (`DATABASE_ROADMAP.md` S1–S19): one declared
table (`schema.COLUMNS`, `bcir-training/chunk/v1`), a content-addressed catalog
set behind a `CURRENT` pointer (`bcir-training/catalog/v4`), postings on indexed
columns with exact per-value and pairwise joint statistics, packed `<q` numeric
columns with a sorted index and per-part zone maps, a predicate language of ten
operators whose legality is decided by column role alone, a planner
(`plan.candidates / choose / explain`) that refuses by four structural rules
before pricing on the `bcir.kbcir.cost.CostVector`, and an embedding set
(`bcir-training/embedding-set/v1`: manifest, `index.jsonl`, `vectors.f32`,
`vectors.q15`, `squares.u32`, four digests) bound to the catalog by row order.
**The hybrid path already runs end to end**: `db.retrieval.eligible(catalog,
terms)` → `Selection.rows` → `bcir_native.q15_topk(query, codes, rows=,
eligible=rows)` → `catalog.fetch(rows)`, priced and explained by `plan.py`
(`search_chunks.py:874-1056`). What it lacks for this program — a second table, a
second vector space per row, exact fusion of two rankings, a Unicode-versioned
provider — is exactly what §2 specifies and §6 schedules.

## 2. The hybrid SQL/vector lookup at inference time — the mathematical schema

The lookup is not a new engine. It is the database rail of §1.4 run over the
Unicode character table (and, later, the lexical-graph tables), with the
vector stage expressed through the kernel that already exists and the three
algorithmic engines folded into *stored columns* at ingest, so that at query
time they are ordinary predicates and never code. Every statement below is one
a gate can check; §6 names the slice that checks it.

### 2.1 The request

A lookup is a request `Q = (P, q, k, F, Θ)`:

- **`P`** is a conjunction of `column op value` terms in the grammar of
  `plan.parse_predicate` over the character table `U` — the ten operators of
  `TRAINING_LANGREF.md` §5, unchanged, with legality decided by column *role*
  alone (`schema.require_filterable`). A measurement has no equality operator,
  so equality on a code point is spelled as a closed interval,
  `code_point>=N AND code_point<=N`, which `plan.ranges` folds to one interval
  and `Catalog.seek_range` answers exactly.
- **`q`** is an ordered tuple of `(space, vector)` pairs. A *space* names one
  embedding set (`bcir-training/embedding-set/v1`) bound to the same catalog
  generation as `U`; the *vector* is `int16^{d_s}` with every code in
  `[-32767, 32767]`, produced by that set's own `embed_chunks.Provider`
  (name, revision, dim read off the provider object, as the manifest rule
  requires). One pair is the common case; two pairs — the stroke space `A` and
  the radical space `B` — are the Han case.
- **`k`** is the requested count, `1 ≤ k`.
- **`F`** is the fusion rule when `|q| = 2`, one of `cascade(b)` or `sum`
  (§2.4). Fusion is part of the *request* because it changes the answer; a
  plan may never choose it.
- **`Θ`** is `bcir.kbcir.cost.Theta` — live pressures in `0..100`. It is a
  **pricing input only**: `plan.legality` takes no `Θ` and gains none. The
  witness is that the legal set and every refusal string are identical under
  `Theta.cool()` and `Theta.hot()`.

The invariant that makes the lookup a *database* operation rather than a
model: **every legal exact plan for `Q` returns the same sequence of
`(row, distance)` pairs.** Plans differ in cost; the answer is a function of
`Q` and the generation alone.

### 2.2 Relations, rows, and the eligibility mask

The character table is a second declared `schema.Table` (`UNICODE_TABLE`, row
schema `bcir-training/unicode-character/v1`, UNICODE_ROADMAP §5.2), built by
`catalog.build(chunk_dir, table=UNICODE_TABLE)` into the same six artifacts and
published by `generations.publish` as one generation `G` per Unicode version.
`N = rows_total(G)`. Row order is **one** function, imported by every writer:

```
unicode_row_sort_key(record) = (code_point,)
```

so row `i` is the `i`-th assigned code point in numeric order, and row `i` of
every embedding set bound to `G` is that same character (`search_chunks._same_rows`,
checked per set).

Relational selection is `plan.select(catalog, P)`:

```
E(P) = { i ∈ [0, N) : row_i ⊨ P }
```

computed exactly — postings intersection for indexed terms, `seek_range` for
numeric intervals, existential comparisons, `!=` as complement — while the
*estimate* is computed first from statistics and may be a bound. "Pricing may
use a bound; answering never does" (`plan.py`) is preserved verbatim: the price
reads `Selection.admitted`, the kernel reads the mask. The mask is

```
m = 1_E ∈ {0,1}^N        m = bcir_native.eligibility_mask(N, E)
```

one byte per row; a row outside `[0, N)` is a refusal before the kernel sees it.

### 2.3 Distances and the single-space rank

For a set `s` with codes `V_s ∈ int16^{N×d_s}` (`vectors.q15`, little-endian,
`-32768` forbidden, produced by `embed_chunks.quantize_q15` with the
repository's half-away-from-zero rounding):

```
d_s(i) = Σ_{j<d_s} (q_s[j] − V_s[i,j])²                       (uint64, exact)
```

Because `d_s ≤ 4096 = BCIR_AI_MAX_DIMENSION` and every difference is at most
`65534`, `d_s(i) ≤ 4096 · 65534² < 2^64`: the sum never wraps, which is what
makes `reference == native` a **bit-equality** gate rather than a tolerance.

```
TopK_s(E, k) = the first min(k, |E|) elements of E under the total order (d_s(i), i)
```

That is literally the contract of `bcir_ai_q15_topk` ("sorted by distance,
then index"; fewer than `top_k` when the mask admits fewer). Since `i` is
catalog row order, **ties break by ascending code point** — the ranking is a
function of the artifacts alone. (The kernel header's remark about SHA-256
order describes its optimization-memory caller; the Unicode sets store
code-point order, and the LangRef says so where the set is declared.)

### 2.4 Sharding above the kernel's ceiling, and fusion of two spaces

`BCIR_AI_MAX_PATTERNS = 65536` rows per call, and the 17.0 repertoire exceeds
it (a fact the pin gate asserts, not this sentence). A set is therefore stored
as shards `S_1..S_r` of at most 65536 contiguous rows in row order, and

```
TopK_s(E, k) = first k of  ⋃_j TopK_s(E ∩ S_j, k)   under (d_s, i)
```

with each shard's mask slice passed to its call and every returned index
offset by the shard base before the merge. The identity is exact: any element
of the global top-`k` is in the top-`k` of its own shard, because a row that
beat it within the shard would beat it globally. The reference backend computes
the unsharded set in Python and the parity gate compares `(i, d)` pairs exactly.

Two spaces per row are joined by **nothing but the row index** — the rail
joins no two sets, and this design does not ask it to. The two fusion rules,
neither learned and neither weighted:

```
cascade(b):  F(Q) = TopK_B( TopK_A(E, min(b·k, |E|)),  k )
sum:         F(Q) = first k of E under ( d_A(i) + d_B(i),  i )
```

`cascade(b)` is a second kernel call whose mask is the indicator of the first
result; `b` is a pinned integer constant registered in
`verify_langref.check_constants` and changed only at L3. It is exact with
respect to the lexicographic order (`d_A` then `d_B`) restricted to the fan-out
set and claims nothing about a global two-space optimum. `sum` is exact integer
arithmetic in mixed units (the two spaces have different dimensions) and says
so; its native form needs every `d_B(i)` for `i ∈ E`, so it is legal on the
native backend only when `|E| ≤ BCIR_AI_MAX_TOP_K = 1024` (the reference
backend has no such bound). Any other arithmetic across spaces is refused.

### 2.5 The Q8 view: rank-then-filter, priced for what it scans

`bcir_ai_q8_rows_dot_f64` takes no mask. Under the Q8 view

```
score(i) = Σ_j h[j] · (code_{ij} · 2^{e_g(j)})     (f64, ascending j)
R_q8 = first k of E under (−score(i), i)   after scoring all N rows
```

it is lossier by construction, is refused under `Objective.EXACTNESS`, and is
charged `rows_total` on `compute` and `k × accuracy_penalty_per_result` on
`accuracy` exactly as `plan.price` charges it today. It is a candidate for
requests that permit an inexact ranking, never a fallback, and no gate asserts
that its ranking equals the exact one — how often it agrees is measured and
reported, not assumed.

### 2.6 The engines are columns at ingest, and callbacks in search

The three algorithmic engines run at **ingest** (`training/tools/unicode/`,
UNICODE_ROADMAP §6) and produce stored columns of `U`; a plan reads columns and
never runs an engine.

| Engine | Stored columns (role) | Exact query it enables |
|---|---|---|
| UTS #10 | `uca_primary` (numeric), defined once in UNICODE_ROADMAP §5.2 and only cited here: the first non-zero level-1 weight of an explicit DUCET entry, 0 when level-1 ignorable, or the packed implicit pair `(AAAA << 16) \| BBBB` for ideographs (bases per UTS #10 Table 16, UNICODE_ROADMAP §6.2); `uca_variable` (indexed) | `ORDER BY uca_primary` through `relational.ordered_rows` is UCA level-1 order — the "morphological base form" equivalence, computed; `uca_primary>=X AND uca_primary<=Y` is a range of base letters |
| UTS #39 | `confusable_class` (numeric): the smallest code point whose skeleton equals this character's; `identifier_status`, `identifier_type` (indexed) | `confusable(a, b) ⇔ class(a) = class(b)`, spelled as a closed interval on `confusable_class` — two binary searches |
| UTS #46 | `idna_status` (indexed, domain `{valid, ignored, mapped, deviation, disallowed}` — the five statuses since UTS #46 revision 31; `UseSTD3ASCIIRules` is a validity criterion, not a status); `idna_mapping` (text, materialized, never filtered) | `idna_status=valid` narrows a lookup to characters a label may carry |

The *string-level* engines — the full multi-level sort key of a string, the
skeleton of a string, `ToASCII`/`ToUnicode` of a label — are not columns. They
are the `verify` callbacks of `bounded_reasoning_search(prompt_ids, budget,
propose, verify)`: the model proposes, the engine verifies, and acceptance is
a verdict of the engine, never a score. A verify closure is built from the
loaded generation, refuses to be built from the host's `unicodedata`, and
records the generation digest in the search report.

### 2.7 Legality — structural rules, appended, none of them a number

`plan.LEGALITY_RULES` is a tuple of rule names, extended append-only; every
existing caller of `plan.legality` is unchanged because the new parameters are
keyword-only with defaults. A rule is a property of the plan and the artifacts'
manifests; none reads a measurement or `Θ`.

| Rule | Refuses |
|---|---|
| `exactness` (existing) | a backend outside `EXACT_BACKENDS` under `Objective.EXACTNESS` |
| `availability` (existing) | a backend whose library did not load — `BackendUnavailable` is a skip, never a candidate |
| `coverage` (existing) | a selection naming a row outside `[0, rows_total)` |
| `materialization` (existing) | a text reading that contradicts the request |
| `binding` | a space whose `ids[i] ≠ catalog.ids[i]` for some `i`, or whose set was built from another corpus digest |
| `space` | a query vector whose provider `(name, revision, dim)` is not the set's manifest `(model, revision, dim)` |
| `generation` | a plan whose catalog, sets, projection tensor (§2.9) or engine pins name different generations |
| `bounds` | a call with `pattern_count > 65536`, `dim > 4096`, `k ∉ [1, min(1024, \|E\|)]`, or a Q8 view over `N·dim > 2^26` — kernel limits as refusals *before* cost, so the same request can never reach C as a `KernelRejected` |
| `fanout` | `sum` on `native`/`both` when `\|E\| > 1024`; `cascade(b)` when `b·k > 1024` |
| `engine` | a predicate on an engine column that the generation did not store, or any plan that would compute an engine value at query time |

Each rule lands with one fixture request that trips exactly it and no other,
and the RED witness runs the fixture against the unextended `legality()`,
where it is reported legal.

### 2.8 Cost — the twelve axes, and which numbers are exact

Every legal candidate is priced on `bcir.kbcir.cost.CostVector` by `plan.price`,
extended for the second call and the shards:

| Axis | What is charged | Exact / modeled |
|---|---|---|
| `compute` | `kmacs = ⌈scan_rows · dim / 1024⌉` per call (`scan_rows = \|E ∩ S_j\|` per shard; `b·k` or `\|E\|` for the second call; `rows_total` for the Q8 view) × `ns_per_kmac[backend]` | count exact; ns modeled (`plan.py` labels its constants modeled, class `wall`, never gating) |
| `memory` | `scan_rows · dim · 2` code bytes + `N` mask bytes per masked call + `k · 1024` seek bytes | byte counts exact; the unit modeled |
| `compile` | `ns_load_cached_kernel` or `ns_compile_kernel` for compiled backends | `kernel_cached` is an exact fact from the build stamp; ns modeled |
| `accuracy` | `0` for exact backends; `k · accuracy_penalty_per_result` for the Q8 view | modeled |
| `verification` | `\|E\| · verification_units_per_row` for `both` | modeled |
| `fabric`, `sync`, `reliability`, `security` | zero, left zero rather than invented | — |
| `thermal`, `power`, `contention` | zero from `price`; `Θ` enters only through `CostVector.couple(theta_factor(Θ))` with a frozen Q8 factor table (`256 = ×1.0`) — coupling scales an axis, it cannot create a price on a zero one | modeled |

`plan.choose(plans, Objective)` minimizes the objective's axis; `plan.explain`
prints every candidate's per-axis and scalarized cost. The chosen plan is the
cheapest *legal* plan under a modeled price — TMSAO-4 — and every
`(Q, Objective, availability)` decision the SLM depends on is pinned in the
plan baseline (`training/plans/baseline-v1.json`, `plan_baseline.py`) so that a
constant change that reorders plans is a reviewed diff, not a silent reorder.

### 2.9 How the model asks: a frozen query projection

The model's hidden state `h ∈ f64^{d_model}` (from `HostedLlama.hidden_states()`
or the laboratory model's equivalent) becomes a query vector through a frozen
projection `T_q^s` per space:

```
u = q8_rows_dot(h, T_q^s)          (bcir_native.q8_rows_dot; T_q^s a BCIRQ8 tensor, d_s rows)
q_s = quantize_q15(u / ‖u‖)        (embed_chunks.quantize_q15)
```

`T_q^s` is learned at L2 (`train_embedding_distillation` over
`relational_embedding_targets`, the existing frozen-Gram distillation) and
frozen into the generation with its SHA-256; a projection whose row count is
not `d_s`, or whose digest is not the generation's, is refused by `space` and
`generation`. The projection uses `q8_rows_dot` and the Python Q15 quantizer
exactly because those are what exist — the roadmap does not assume a C Q15
quantizer or a wrapped `q8_matvec`.

### 2.10 The certificate: a DER lookup envelope

Because the retrieval-augmented loss of §5 must be replayable, every lookup
the training stack consumes is recorded as a `LookupEnvelope`: a DER-encoded
record (a new `bcir/asn1` module under the BCIR arc, with its own arc number
allocated in `docs/BCIR_ASN1_X690_ABI.md`'s registry, not invented here)
carrying the generation id, the catalog and set digests, the request, every
candidate with its `legal`/`refusal` verdict and its 12-axis modeled cost, and
the selected plan — content-addressed over inputs only, so it replays
byte-identically on another host. DER out, BER in; laws A1–A5; a measured
sidecar, if a calibration run produces one, lives outside the digest and is
labeled measured. The envelope records legality separately from cost, which is
the `selection_envelope` pattern of `bcir/asn1/manifest.py` applied to a
lookup.

### 2.11 Placement, restated for the lookup alone

- **L0** (`runtime/c/bcir_llama.c`): nothing — no catalog read, no postings, no
  `q15_topk` call. The lookup's result reaches the decoder only as prompt ids
  assembled before `bcir_llama_generate_greedy` runs. Witness: a grep of
  `runtime/c` for lookup symbols outside `bcir_ai_kernels.*`, in the style of
  `check_memory_discipline.py`.
- **L1** (`training/tools/unicode/lookup.py` through `bcir_native`): the
  planner, the frozen Q15 shards, the frozen projection, the engine columns,
  the pinned `b`, the `CostModel` and factor tables. Everything here is exact
  integer arithmetic or frozen Q8/Q15 data.
- **L2**: the hosted model and the projection student; the plan baseline as a
  replay gate; `bounded_reasoning_search` with engine callbacks.
- **L3**: changing `b`, any cost constant, the factor table, promoting a
  learned set past `LEARNED_EMBEDDING_GATE.md` (STOP today), re-pinning a
  Unicode version.

## 3. The token-level data pipeline

### 3.1 What a token is, and what travels beside it

The SLM's tokens are the repository's own byte-fallback BPE ids, under a new
schema `bcir.byte_bpe.v2`, and **beside** every token travels a row of exact
character features read from the Unicode generation. The tokenizer stays a
tokenizer; the substrate is a side channel bound to it by content address. This
is the design that keeps the whole existing provenance chain —
`BytePairTokenizer.digest` → `token_source_from_corpus` →
`CorpusManifest.tokenizer_sha256` → checkpoint → `export_hf_checkpoint` → the
BCIRQ8 header — intact, and it is the one member of the F1 portfolio that this
roadmap adds; it does not declare itself the base of every model.

`bcir.byte_bpe.v2` differs from v1 in exactly three declared ways, and
`from_json` keeps refusing any other key set:

| Field | v1 | v2 |
|---|---|---|
| `normalization` | `"NFC+LF"` (tables unrecorded) | `"NFC+LF@<ucd_version>:<generation digest>"` — normalization by the generation's tables (UNICODE_ROADMAP §8), never the host's |
| `alignment` | absent (merges may split a scalar) | `"scalar"`: no merge crosses a code-point boundary, admitted by `canonical_substring_candidate` at training time and gated at load |
| `records` | absent | the SHA-256 of the generation's `records.der` (§3.2) |

The special-id layout is v1's — `<pad>=0, <bos>=1, <eos>=2, <unk>=3`, bytes at
`4..259`, merges from `260` — because that layout is the one the
checkpoint/export/BCIRQ8 chain already carries; `<unk>` is never emitted. The
curriculum's control tokens (`[REGISTER: …]`, `[CADENCE: …]`, §5) are ordinary
specials appended to `special_tokens`, never split. The two other layouts the
tree carries (`ByteVocabularySpec`: bytes `0..255`, `bos 256 …`; the hosted
gate: `unk 256, <s> 257 …`) are a finding this roadmap records and SLM-11
resolves with one declared `SpecialLayout` imported by every rail (L14), not
by this tokenizer silently adopting a fourth.

### 3.2 The character record and its serialization

One `UnicodeCharacterRecord` per assigned code point in the generation:

```
UnicodeCharacterRecord ::= SEQUENCE {
  codePoint        [0] INTEGER (0..1114111),
  generalCategory  [1] ENUMERATED { ... the 30 UCD values ... },
  script           [2] INTEGER,             -- index into the generation's script table
  block            [3] INTEGER,             -- index into Blocks.txt order
  ccc              [4] INTEGER (0..255),
  bidiClass        [5] ENUMERATED { ... },
  decomposition    [6] SEQUENCE OF INTEGER OPTIONAL,   -- canonical mapping, code points
  decompositionType[7] ENUMERATED { ... } OPTIONAL,
  numeric          [8] SEQUENCE { num INTEGER, den INTEGER } OPTIONAL,
  uca              [9] SEQUENCE { primary INTEGER, secondary INTEGER, tertiary INTEGER, variable BOOLEAN },
  confusableClass  [10] INTEGER,
  identifier       [11] SEQUENCE { status ENUMERATED {...}, type BIT STRING },
  idna             [12] SEQUENCE { status ENUMERATED {...}, mapping SEQUENCE OF INTEGER OPTIONAL },
  han              [13] SEQUENCE { radical INTEGER, residualStrokes INTEGER, totalStrokes INTEGER } OPTIONAL,
  age              [14] SEQUENCE { major INTEGER, minor INTEGER },
  name             [15] UTF8String
}
```

The exact module — component tags, enumerations spelled from
`PropertyValueAliases.txt`, and the OPTIONAL/DEFAULT discipline R24 checks —
is UC-8's deliverable; the sketch above fixes the *fields*. It is encoded with
`Module.encode` (DER, canonical), decoded by `Module.decode` (DER by default,
BER accepted), and the generation's `records.der` is a DER `SEQUENCE OF` in
code-point order whose SHA-256 is `records_sha256` in the generation manifest.
The catalog's JSONL rows carry the same fields as canonical JSON, and a gate
asserts DER-decoded record == JSON row, field for field (two rails, one truth).

Why DER for the pipeline and JSONL for the catalog: the tokenizer lives in
`bcir/` (`bcir.kbcir.sequence_interfaces`, `bcir.hosted.training.bpe`) and
`bcir/` may not import `training/`. The record therefore crosses as **bytes
with a canonical form** that `bcir`'s own DER decoder reads and digests (laws
A1–A5), and later a C twin can read the same bytes. The catalog reads JSONL
because that is what `catalog.build` reads.

The record table needed at tokenization time is a **generation-tagged build
product** (`unicode-records-<generation>.der`), loaded by path and refused on
digest mismatch — never committed, like BCIRQ8 weights. A fixture of a few
hundred Latin records ships as package data so the contract's tests run from
the installed wheel (L21) without the full table.

### 3.3 From bytes to tokens with features

```
bytes  --strict UTF-8 (encode_raw_sequence; invalid bytes refused)-->  scalars with spans
       --NFC by the generation's tables (UC-7)-->                      normalized scalars
       --record lookup by code point in records.der-->                 records (unassigned: refused,
                                                                        or a declared 'unassigned' record kind
                                                                        under an explicit policy flag; PUA likewise)
       --scalar-aligned BPE (v2)-->                                     token ids over the normalized bytes
       --per-token reduction-->                                         one feature row per token
```

The per-token feature row is a pure function of (token byte span, the records
of the scalars it covers) and is byte-exact reproducible:

```
feature(t) = ( gc(first scalar), script(t) or MIXED, block(first), max ccc,
               uca_primary(first), confusable_class(first), worst idna_status,
               han(first) if any, n_scalars(t) )
```

Because merges are scalar-aligned, every token covers whole scalars and the
"first scalar" is well defined; a v1 tokenizer, which may split a scalar, is
refused by the interface with a message naming `alignment`. The proposal's
tabular record `[Unicode Category, UCA Key, Root ID, Derivational Suffix,
Grammatical Case]` is this row joined with the lexical-graph tables of §4.4;
before those tables exist the row carries the Unicode columns alone, and that
is what the phase-1 curriculum trains on.

The whole pipeline is one new dependency-free contract,
`UnicodeRecordInterface` in `bcir/kbcir/sequence_interfaces.py`, with a hosted
twin that materializes feature tensors; it is a *member* of the sequence-interface
portfolio and carries the portfolio's multi-objective evidence (exact round
trip, fertility, fragment rate) rather than a single "best tokenizer" score.

### 3.4 Provenance, end to end

| Digest | Where it is recorded | What it vouches for |
|---|---|---|
| generation id `g-<digest16>` | `generations.publish` manifest | the vendored pins, the row set, `records.der`, every engine column |
| `records_sha256` | generation manifest; `bcir.byte_bpe.v2` `records` field | the DER record table the tokenizer reads |
| `tokenizer_sha256` | `CorpusManifest`; BCIRQ8 header bytes 184..216 | the v2 JSON, which now names the generation and the record table |
| `policy_sha256` | `DataPreparationSpec` | the corpus cleaning policy, now including the generation digest |
| stage digests | `TrainingPipelineLedger` (`data`, `tokenizer`, and the new `prealign` stage) | every curriculum input |

A model whose tokenizer names generation `g₁` cannot be evaluated over
features from `g₂`: the interface refuses the mismatch, and the envelope of
§2.10 records which generation answered every lookup.

### 3.5 What the C side needs, and when

Today the C decoder consumes ids and carries a `tokenizer_sha256` it does not
verify. Nothing in this roadmap changes that: the SLM's inference with
features runs in the hosted laboratory, and the stable C decoder is untouched.
The day a C consumer of records exists — a decoder taking feature inputs — the
record lookup gets a freestanding C twin (a DER reader over `records.der` and
a binary search by code point, in the memory class the consumer dictates),
gated by parity against the Python decoder, exactly as every other twin here
was landed. Until then it is declared, not built, and the roadmap says so.

### 3.6 The byte-native alternative

If a later decision prefers the byte-native laboratory as the SLM body, the
same records serve it through `BytePatchSpec.method = "grapheme"`: patch
boundaries from the generation's UAX #29 tables (UC-8) instead of byte
indices, with the record row attached per patch. `encode_raw_sequence` already
refuses normalization and records spans; the only addition is the boundary
policy. This is one slice (SLM-7b) and is not on the critical path.

## 4. The model mechanisms, wired into the laboratory

None of the four mechanisms touches `DecoderSpec`, BCIRQ8, or the standalone C
decoder. They live where the tree already keeps research architectures — the
dependency-free contracts in `bcir/kbcir/` and their import-quarantined hosted
twins — and the promotion of any of them into the stable path is a separate,
later gate (§9).

### 4.1 Tabular pre-alignment: feature embeddings and the router

The feature row of §3.3 becomes model input through a `FeatureEmbedding`
module of the laboratory model: one learned embedding table per categorical
column (general category, script, block, IDNA status, identifier status),
summed with the token embedding; numeric columns (`uca_primary`, stroke counts)
enter through a **deterministic** code — the fixed ±1 codebook by bits that
`FrozenTokenCodeSpec` already defines for token ids, applied to the UCA
primary weight — so the collation identity is a fixed geometry the model cannot
unlearn. The "tabular router" the proposal trains first is `HostedSmallModel`
(family `mlp`, `SmallModelSpec`) via `train_small_supervised`, over feature
rows, predicting morphological-compatibility and case-alignment labels derived
deterministically from the tables; it is a confirmation model, gated on
properties, and its weights are frozen targets for the main model's
pre-alignment loss.

### 4.2 Tree-constrained attention

A dependency-free `TreeMaskSpec` in `bcir/kbcir/` follows the one mask seam
the tree has, `ReferenceWindowSpec`: given `head[i]` for each token (the
dependency parse) and a depth budget `h`,

```
visible(i) = { i } ∪ ancestors_h(i) ∪ descendants_h(i) ∪ specials
additive_mask(L)[i, j] = 0 if j ∈ visible(i) else −1.0e30
```

materialized by the hosted `_mask()` and passed as `attn_mask` to
`_AdaptiveAttention.forward` for the first `n_tree_layers` layers; the
remaining layers use the reference-window mask. For decoding, `visible(i)` is
intersected with `j ≤ i` — a tree edge into the future is not a legal
dependency at generation time. The admitted set is gated against a reference
walker over the parse; the RED witness drops the depth bound.

### 4.3 Recursive composition

```
h_node = tanh( W_merge [h_left ‖ h_right] + b )
```

applied bottom-up over the parse in post-order, batched by tree level. `tanh`
is in the closed primitive set of the dependency-free autodiff
(`bcir.kbcir.autodiff.Tape`), so the cell's gradient exists on the oracle rail
and the hosted twin is gated against it (the E4 recurrent-cell precedent, the
`bcir_train.c` tolerance discipline). The composed phrase vector is an
auxiliary input to the pre-alignment loss and to a root-prediction head over
`hidden_states()` (the `HostedRewardModel` pattern: a `Linear` head, deep-copied
policy).

### 4.4 The lexical graph as tables

The proposal's four tiers — root core, morphology, drift and calque, lexical
field — are relations, and they are declared as tables of the same rail:
`roots(root_id, form, gloss)`, `derivations(lemma, root_id, suffix, stratum)`,
`edges(src_lemma, dst_lemma, kind ∈ {cognate, doublet, calque, field})`. Each
is a declared `schema.Table`, published as a generation, queried by the same
planner, and its edges become exact targets for `relational_gram_loss`. The
graph-walk tuples of curriculum phase 3 are generated from these tables by a
deterministic query generator (the `build_eval_queries.py` pattern). What the
tables are *filled from* is the open question §9 records: no open etymological
dataset with roots, sound laws and strata is named here, and the slice that
fills them begins with a GO/STOP note on a source.

## 5. The curriculum, mapped onto the stages that exist

The proposal's four phases become four ledger stages of the existing
`TrainingPipelineLedger` DAG, each a `StageTrainSpec` with its own content
address, and none of them a new kind of thing:

| Proposal phase | Objective | Data it consumes (all content-addressed) | Stage kind and where it plugs in |
|---|---|---|---|
| 1 — Orthographic and radical grounding | character composition, stroke logic, symbol mechanics | the Unicode character table (UC-2..UC-5), the record stream of §3, operator truth tables generated from `PropList`/`Math` properties | a **pre-alignment stage** (new kind in the ledger DAG) training the tabular router of §4.1 against exact targets; loss = cross-entropy over record columns, plus a **collation-distance loss** whose target is the exact level-1/2/3 key difference (UNICODE_ROADMAP §6.2) |
| 2 — Morphological tree induction | affixes, dependency structure, root families | lemmatized text with dependency trees (an admitted external corpus, provenance-pinned, cache-only), the lexical-graph tables of §2.3 | `train_sft` over the adaptive laboratory model with the **tree mask** of §4.2 and an auxiliary **root-prediction head** on `hidden_states()` |
| 3 — Relational graph traversals | semantic radiation, doublets, analogies | graph-walk tuples generated from the lexical-graph tables by a deterministic query generator (like `build_eval_queries.py` builds the retrieval evaluation) | `train_sft` with the **retrieval-augmented loss** of §4.4 and `relational_gram_loss` over frozen graph-edge targets |
| 4 — Compositional styling and rhetoric | cadence, register, argument | stratified multi-register text (admitted, pinned) | `train_sft` / `train_dpo` with prefix control tokens (`[REGISTER: …]`, `[CADENCE: …]`) that are ordinary specials of the tokenizer of §3 |

The order is a dependency, not a preference: phase 2 needs the table phase 1
grounds, phase 3 needs the graph phase 2's root head learned to name, phase 4
needs a model that composes. Each stage records its inputs' digests in the
ledger; a stage whose inputs' generation changed is stale and re-runs — the
cached-artifact validity law of the ML roadmap §8.4, applied to a curriculum.

## 6. The build slices

One gateable slice per PR, slice id in the title, RED before GREEN, a fault
entry in `tools/testing/faults/training-slm.json` per new check, exact gate
output in the PR body. Ids are `SLM-n`; Unicode prerequisites are `UC-n`
from [`UNICODE_ROADMAP.md`](UNICODE_ROADMAP.md) §10. Where a slice extends an
existing gate, it names it; where it adds one, the gate joins the CI job that
owns that rail (§10).

### SLM-1 — the record interface and `bcir.byte_bpe.v2`

`UnicodeRecordInterface` in `bcir/kbcir/sequence_interfaces.py`: strict UTF-8
through `encode_raw_sequence`, NFC by the generation's tables, record lookup in
`records.der`, scalar-aligned BPE under the v2 schema, one feature row per
token (§3.3). The v1 layout of specials is kept; `from_json` refuses v1 for
this interface by name (`alignment` absent). A Latin fixture of records ships
as package data so the tests run from the wheel.

*Payoff:* every token the SLM sees carries exact, versioned character features,
and the tokenizer's digest names the tables that normalized it.
*Gate:* exact round trip over a corpus containing every admitted block; no
token span splits a scalar; the feature row of every token is a function of
the record table (two builds, two hosts, byte-identical); a v1 tokenizer is
refused with `alignment` in the message; a generation-digest mismatch between
the tokenizer JSON and the record table is refused. *RED:* drop the
`canonical_substring_candidate` admission — a merge crosses a scalar and the
alignment gate fires; point the JSON at a stale generation — the digest gate
fires. *Depends on:* UC-7, UC-8.

### SLM-2 — the lookup request, six appended legality rules, and the price

`LookupRequest` and `candidates_for / choose / run / explain` in
`training/tools/unicode/lookup.py`; `plan.LEGALITY_RULES` extended append-only
with `binding, space, generation, bounds, fanout, engine`; `plan.price`
extended for the second call and per-shard masks; `Θ` coupling through a
frozen factor table; every `(Q, Objective, availability)` decision pinned in
the plan baseline.

*Payoff:* the lookup is a planned query with a legality verdict and a priced
plan, on the same twelve axes as everything else BCIR prices.
*Gate:* check group `legality` — one fixture per rule trips exactly that rule;
the legal set and refusal strings are identical under `Theta.cool()` and
`Theta.hot()`; check group `pricing` — `fabric/sync/reliability/security` are
zero on every candidate, the Q8 view charges `rows_total`, a sharded plan's
compute is the sum of its shards' exact counts; `plan_baseline --compare` is
clean. *RED:* run each fixture against the unextended `legality()` — all are
reported legal; halve `ns_per_kmac_q8` in a copy of `CostModel` — a
`LATENCY` decision moves and `--compare` fires. *Depends on:* UC-3, UC-5, UC-6.

### SLM-3 — exact fusion: shards, cascade, sum

Sharding above `BCIR_AI_MAX_PATTERNS`, the exact merge, `cascade(b)` and
`sum`, on both backends, with `b` registered in `verify_langref.check_constants`.

*Payoff:* a whole-repertoire ranking through a kernel bounded at 65536 rows,
bit-identical to an unbounded one; two spaces composed without a learned
weight.
*Gate:* check group `fusion` — native sharded == reference unsharded on
`(row, distance)` pairs for `N > 65536` (synthetic shards) and `N ≤ 65536`;
`cascade(b)` equals the reference two-stage order; `sum` equals the reference
integer order; a masked-out row never appears in any shard's result.
*RED:* drop the mask from one shard's call — a masked-out row appears;
offset one shard's indices by the wrong base — the parity fails on the first
cross-shard pair. *Depends on:* SLM-2 (synthetic sets suffice; UC-10 supplies
the real two spaces).

### SLM-4 — the `LookupEnvelope`, DER out, BER in, replayed

The `BCIR-Lookup` ASN.1 module (arc allocated in the X.690 ABI registry), the
writer and the replayer, the A1–A5 gates, and the content address over inputs.

*Payoff:* every lookup the training stack consumed can be replayed on any host
and refused if the generation moved.
*Gate:* check group `envelope` — DER re-encodes byte-identically from its own
decode; a BER encoding decodes and replays; replay on a second build directory
reproduces the selected plan and every candidate verdict; R24 static rules
pass on the module. *RED:* flip the `legal` bit of one candidate in the DER —
replay fails naming that candidate; reorder two components — the canonical
re-encode differs. *Depends on:* SLM-2.

### SLM-5 — the frozen query projection and the L0 witness

`T_q^s` as a BCIRQ8 tensor learned by `train_embedding_distillation`, frozen
with its digest into the generation; `query_projection.project(h, T_q)` via
`q8_rows_dot` + `quantize_q15`; the hosted call site; a witness that
`runtime/c` contains no lookup symbol and that nothing in the lookup path
imports `torch`.

*Payoff:* the model asks the database with the kernels that exist, and the hot
path stays free of both models and lookups.
*Gate:* projection deterministic across two runs and two hosts (byte-equal
`q_s`); a `T_q` whose row count is not `d_s` is refused by `space`; the grep
witness over `runtime/c` passes; `import torch` under `training/tools/unicode/`
is refused by the import witness. *RED:* plant a `bcir_lookup_stub` symbol in a
`runtime/c` fixture — the grep witness fires; add `import torch` to `lookup.py`
in a fixture copy — the import witness fires. *Depends on:* SLM-2.

### SLM-6 — engines as `verify` callbacks in bounded reasoning search

`collate / security / idna` closures built from a loaded generation and passed
as `verify` to `bounded_reasoning_search`; a fixture `propose`.

*Payoff:* an accept/reject verdict the model cannot argue with, from the
standard's own algorithms, inside the existing bounded search.
*Gate:* check group `reasoning` — with a fixture proposer emitting one
IDNA-disallowed and one valid candidate, the valid one is selected regardless
of scores (permute the scores; the selection is unchanged); a closure built
from the host's `unicodedata` is refused. *RED:* replace the engine call with
`score > 0.5` in a test double — the permuted-score witness fires. *Depends on:*
UC-5, UC-6.

### SLM-7 — tabular pre-alignment: feature embeddings, the router, the stage

`FeatureEmbedding` in the laboratory model (learned tables per categorical
column; the frozen ±1 code for `uca_primary`); the tabular router as
`HostedSmallModel(mlp)` over feature rows; a new ledger stage kind `prealign`
in `TrainingPipelineLedger`'s DAG; the collation-distance loss against exact
level-1/2/3 differences.

*Payoff:* curriculum phase 1 exists as a stage with a content address.
*Gate:* the frozen code for `uca_primary` is byte-identical across hosts; the
router's labels are derived from the tables by a deterministic function (two
builds equal); the stage refuses a feature tensor whose generation differs
from the tokenizer's; the existing `--require-torch` gate in the hosted-model
job runs it. *RED:* seed the router's labels from the model's own predictions —
the determinism gate fires. *Depends on:* SLM-1.

### SLM-7b — grapheme-aware patching for the byte-native rail (optional)

`BytePatchSpec.method = "grapheme"` from UAX #29 tables; records attached per
patch. Not on the critical path.

*Gate:* every patch boundary is a grapheme boundary per `GraphemeBreakTest.txt`
rules; byte spans reproduce. *RED:* patch on byte index — a combining mark is
split from its base. *Depends on:* UC-8.

### SLM-8 — tree-constrained attention

`TreeMaskSpec` in `bcir/kbcir/` (visible set, additive mask, decode-time
causal intersection) and the hosted `_mask()` for the first `n_tree_layers`
layers of the laboratory model.

*Payoff:* early heads attend along dependencies before attending broadly —
the proposal's mechanism, as a mask the labs already know how to take.
*Gate:* the admitted set equals a reference walker's closure over the parse
for every token; at decode time no visible key is in the future; the
laboratory gate's report is timestamp-free and identical across two one-thread
runs. *RED:* drop the depth bound — the closure differs on any tree deeper than
`h`. *Depends on:* nothing but the laboratory.

### SLM-9 — the recursive composition cell and the root head

`tanh(W [h_left ‖ h_right] + b)` bottom-up over the parse; the oracle cell on
`bcir.kbcir.autodiff.Tape` and the hosted twin gated against it; the
root-prediction head over `hidden_states()`.

*Payoff:* explicit compositionality without a deeper transformer.
*Gate:* hosted gradient matches the oracle `Tape` gradient within the
training-rail tolerance on fixed fixtures; post-order batching gives the same
composed vectors as a sequential walk. *RED:* swap `h_left` and `h_right` in
the twin — the fixture gradient differs. *Depends on:* SLM-8 (the parse
plumbing).

### SLM-10 — the lexical graph as tables

`roots`, `derivations`, `edges` as declared tables of the rail, published as
generations; the graph-walk query generator; the retrieval-augmented loss
consuming envelopes; a **GO/STOP note on the source** written before any
data enters.

*Payoff:* curriculum phases 2–3 have exact targets and replayable retrieval.
*Gate:* every edge in a training tuple exists in the generation; the loss is a
deterministic function of (generation, tokenizer, checkpoint) — two runs equal;
the source's license passes the dependency audit. *RED:* inject an edge absent
from the tables into a tuple — the membership gate fires. *Depends on:* SLM-2,
SLM-4, and the GO decision.

### SLM-11 — the curriculum as ledger stages, and one special layout

Phases 1–4 as `StageTrainSpec` stages in the ledger DAG; the reconciliation of
the three special-id layouts into one declared `SpecialLayout` imported by
`bpe.py`, `kbcir/byte_latent.py` and the hosted gate (L14); the laboratory-scale
end-to-end run in the hosted-model CI job.

*Payoff:* the proposal's four phases are four content-addressed stages that a
stale input re-runs; the tree stops carrying three spellings of `<pad>`.
*Gate:* a stage whose input generation changed is reported stale and re-runs;
one layout module, and a witness that no rail spells a special id literally.
*RED:* hard-code `pad = 0` in one rail — the literal witness fires. *Depends
on:* SLM-7, SLM-8, SLM-9.

### SLM-12 — the evaluation harness

External benchmarks admitted as pinned cache products; the three linguistic
measures built from the lexical-graph tables; the timestamp-free report with
four digests.

*Payoff:* a number that carries its provenance, or no number.
*Gate:* a report without the dataset, model, tokenizer and generation digests
is refused; a benchmark set whose digest changed is refused; the linguistic
measures are gated on properties (every item traces to a table row). *RED:*
drop one digest from the report — the harness refuses to write it. *Depends
on:* SLM-10, SLM-11.

### SLM-13 — GO/STOP: scale, and promotion into the stable path

A written gate, in the style of `docs/BCIR_NATIVE_OBJECT_GATE.md`, with GO
criteria (property gates green at laboratory scale; a measured advantage of the
substrate-fed model over a matched baseline on the linguistic measures, with
provenance; a GPU training rig with a recorded capability audit) and STOP
criteria (no measured advantage; a mechanism that only works at a scale the
repository cannot run). Only after GO does any mechanism approach
`DecoderSpec` — and then through the labs' promotion order, not around it.

*Depends on:* SLM-12.

## 7. Evaluation

Two kinds of evaluation, kept apart because they answer different questions.

**Property gates** (this repository's kind): every mechanism has a gate that can
fail. The lookup returns exactly the rows the predicate admits and ranks them by
exact integer distance; the tree mask admits exactly the dependency paths; the
recursive cell's gradient matches its closed form; the record stream round-trips
byte-exact; the tokenizer's digest ties to the tables' generation. These are
listed per slice in §6 and proved RED in each slice's PR.

**Benchmarks** (the proposal's kind): MMLU and ARC-Challenge, GSM8K and MATH,
CoLA, plus the proposal's three linguistic measures — analogy and etymological
cross-transfer, neologism decomposition into Unicode-anchored roots and affixes,
and a register-modulation score. All are *external* datasets: they are admitted
the way `run_real_model_gate.py` admits TinyLlama — pinned revision, SHA-256,
license reviewed, cache/build product never committed — and run by a harness
that emits a timestamp-free report with the dataset digest, the model
generation, the tokenizer digest and the Unicode generation. A number without
those four digests is not a result. The proposal's linguistic measures need
judged sets that do not exist yet; they are built the way `eval-set-v1.json`
was built for retrieval (derived from structure the repository already has —
here the lexical-graph tables — and gated on properties, not on a score), and the
register score is defined against the stratum labels of the lexical graph
(Germanic / Norman-French / Renaissance Latin-Greek) rather than by a judge model.

No benchmark result is a gate. A benchmark that regresses between two model
generations is a finding recorded in the ledger; a gate that fails blocks the
slice. The distinction keeps a learned score from ever becoming a verdict.

## 8. Placement on the L0–L3 ladder, and what is exact

The two-truth quarantine (`docs/BCIR_LANGREF.md` §13; `bcir/tests/test_hot_cold.py`)
governs BCIR's own plan execution: no learned inference on L0, decisions
compiled out. The SLM is not on that path — it is a hosted model, opt-in, in
the import-quarantined laboratory — but its components still have to be placed,
because the same three questions (learned or exact? frozen or live? who
actuates?) decide what a gate can promise.

| Component | Placement | What is exact | What is learned, and where frozen |
|---|---|---|---|
| Relational selection (predicates over the character and lexeme tables) | data, exact | row sets, counts, joint statistics, sort keys, skeletons, IDNA status | nothing |
| Vector ranking (Q15 top-k under the mask) | L1 — frozen tables | the ranking over the given codes (integer arithmetic, declared tie-break) | the embeddings: produced by a provider, quantized to Q15 with the repository's rounding rule, published as a set with four digests, never recomputed at inference |
| Plan choice (which backend, which materialization) | plan time, priced | legality (four structural rules + the new ones of §2.2) | the cost constants (modeled, indicative, never gating) |
| The model's forward pass | hosted laboratory | shapes, dtypes, the mask's admitted set | the weights |
| Test-time search (`bounded_reasoning_search`) | L2 — bounded, replayable | the `verify` callbacks (engines) | the `propose` callback (the model) |
| Curriculum and evaluation | L3 — human-actuated | the ledger, the digests | which generation is promoted |

A learned quantity crosses into an exact one at exactly one kind of place: a
frozen, digest-recorded table (the Q15 set, the tokenizer JSON, a generation).
Nothing learned decides legality, nothing learned is recomputed inside a lookup,
and nothing in the lookup path imports `torch`.

## 9. Risk register, and what this roadmap refuses to claim

- **Scale.** Nothing here trains at 500M parameters. The hosted rail is CPU
  float32 with a 64-step gate; the adaptive laboratory refuses more than
  50 M parameter elements (`adaptive_transformer.py:32-34`). A training run at
  the proposal's scale needs the GPU path that exists in `train.py` but is not
  CI-gated, on hardware the repository does not have; the roadmap's slices are
  sized so that every mechanism is proved at laboratory scale and *nothing*
  about the mechanisms changes with scale. The scale claim itself stays a
  hypothesis with an evaluation plan.
- **External data.** Dependency trees, lemmatized corpora, register-stratified
  literature, etymological graphs, benchmark sets: all external, all under
  their own licenses, all admitted only as pinned cache/build products with the
  audit's license review, none committed. The proposal's Tier 1–4 lexical
  graph (PIE roots, sound laws, strata) has no open dataset this roadmap can
  name today; §4.4 defines the tables it would fill and the gates over them,
  and the slice that fills them starts with a GO/STOP decision on a source.
- **Stroke vectors.** The UCD has radical and stroke *counts*, not stroke
  *sequences* (UNICODE_ROADMAP §7.4). "Sub-character stroke vectors" means, in
  this roadmap, vectors over UCD-derived structure until an external IDS source
  passes its GO/STOP gate.
- **The stable model does not change.** Tree masks and recursive cells live in
  the adaptive laboratory; `DecoderSpec`, BCIRQ8 and the standalone C decoder
  are untouched by every slice below. Promotion into the stable path is a later
  decision with its own gate, taken only after the laboratory model has a
  measured advantage on the property gates *and* the benchmarks with provenance.
- **Two-truth.** No learned organ enters BCIR's plan execution; the SLM's lookup
  engine is `training/tools` plus the C kernels, and `training/` is never a
  build dependency of BCIR.
- **No paper result is restated as BCIR-measured performance.** The proposal's
  citations are context, not evidence.
- **Everything BCIR emits is TMSAO-4.** The planner's chosen plan is the
  cheapest *legal* plan under a modeled price, not an optimum.

## 10. Dependencies, ownership, and where the gates run

| Slice | Needs from the Unicode roadmap | Rail it lands on | CI job that owns its gate |
|---|---|---|---|
| SLM-1 | UC-7 (repository NFC, `byte_bpe.v2`), UC-8 (`records.der`, UAX #29) | `bcir/kbcir`, `bcir/hosted/training` | `hosted-model` (`--require-torch` for the hosted twin); the quick tier for the contract |
| SLM-2, SLM-3, SLM-4, SLM-5 | UC-3 (second table), UC-5 (UCA columns), UC-6 (security/IDNA columns), UC-10 (two spaces) | `training/tools`, `bcir/asn1` (envelope), `bcir/tests` (witnesses) | `training-llvm` (`verify_unicode.py`), `host-portability` (`--require-native`) |
| SLM-6 | UC-5, UC-6 | `bcir/hosted/training` (`reasoning.py` callbacks) | `hosted-model` |
| SLM-7, SLM-8, SLM-9, SLM-11 | UC-8 for SLM-7b | `bcir/kbcir`, `bcir/hosted/training` | `hosted-model` |
| SLM-10 | — (its own GO/STOP) | `training/tools`, `training/<lexicon>` | `training-llvm` |
| SLM-12, SLM-13 | — | `tools/models` | `hosted-model`; the GO/STOP doc under `docs/` |

The two roadmaps share one CI discipline: a new verifier joins the job that
runs its siblings; a `--require-X` flag is passed exactly where the job
installed X; every skip is named. The first Unicode slice (UC-1) is the first
thing this program lands, and nothing in `SLM-*` is started before UC-3.
