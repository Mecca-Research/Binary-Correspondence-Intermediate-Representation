# Learned Embedding Provider — the Decision Gate

> **Naming.** G1–G4 below are this gate's GO *criteria* and S1–S3 its STOP criteria. They are
> not corpus phases (Phase 0.x lives in [`ROADMAP.md`](ROADMAP.md)), and they are unrelated to
> the GO/STOP criteria of BCIR's [native-object gate](../docs/BCIR_NATIVE_OBJECT_GATE.md),
> which reuses the same letters for a different decision.

> **Status (2026-09-10): STOP — measured, not inferred.** The corpus keeps `lexical-hash-v1`
> as the provider its retrieval evaluation runs on. A student trained inside BCIR's own hosted
> stack, distilling the corpus's own lexical teacher over the whole corpus, reached overall
> MRR@10 0.221 against that baseline's 0.593 — and its ceiling *is* the baseline, because
> imitating a teacher cannot beat the teacher. The obstacle turned out not to be the
> evaluation's calibration, which an earlier reading of this corpus blamed; see §2.
> This document is the gate: what would flip the decision, and what aborts an attempt.

This gate owns the GO/STOP criteria for admitting an embedding set that declares
`semantics: "learned"` as the provider the retrieval evaluation runs on.
[`ROADMAP.md`](ROADMAP.md) schedules corpus work, [`CORPUS_STANDARD.md`](CORPUS_STANDARD.md)
defines what any set must declare, and `training/tools/verify_retrieval.py` enforces the
properties every run must satisfy. Those may rank or schedule an experiment; none of them
admits a learned provider without this gate.

Nothing here forbids *building* learned sets. `embed_chunks.py` accepts any provider that
satisfies its protocol, the set format is provider-neutral, and the corpus has been able to
train an embedding student since the hosted stages were wired in. What is gated is the
promotion of such a set to the thing the evaluation — and therefore every claim about this
corpus's retrieval — is computed on.

## 1. The decision

The corpus answers questions by retrieving chunks, and something has to turn text into
vectors. Two candidates:

1. **`lexical-hash-v1` (chosen).** Signed-hash n-grams over word unigrams, bigrams and
   character 4-grams, `1 + ln(count)` weighting, L2 normalization. Hermetic: no weights, no
   download, no training, and every vector is a function of its own text alone, so adding a
   chapter cannot change any existing vector. It declares `semantics: "lexical"` precisely so
   nobody mistakes it for meaning: it scores near-duplicate text highly and a paraphrase not
   at all.
2. **A learned encoder.** A pooled decoder trained to place semantically related text nearby.
   It is what would make retrieval answer a question the corpus does not already contain the
   words for. It costs weights, a training budget, and — this is the part that decided the
   question — a *teacher*, or an objective that can stand in for one.

## 2. The measurement (2026-09-10)

Run because the arguments for and against were both plausible, and the experiment is cheap.

**Method.** `HostedLlama(DecoderSpec(vocab_size=4096, d_model=192, n_heads=4, n_layers=3,
d_ff=512, activation="silu_gate"))` wrapped in `stages.HostedEmbeddingStudent(policy, 256)`,
trained with `stages.train_embedding_distillation` on targets from
`providers.relational_embedding_targets` built from `lexical-hash-v1` over the corpus's own
2013 chunks — 96 sampled chunks per round, 3 steps per round, byte-level tokenization
truncated to 192 bytes. The trained student then encoded every chunk, the set was written
with `embed_chunks.write_set`, and `evaluate_retrieval.py` scored it over the judged query
set (445 scored queries plus the 60-query harness control), against the same corpus digest
as the baseline run.

| | 24 rounds | 72 rounds | `lexical-hash-v1` |
|---|---|---|---|
| harness control `heading` recall@5 (floor 0.95) | 0.917 | 0.983 | 1.000 |
| overall recall@5 | 0.239 | 0.246 | 0.694 |
| overall MRR@10 | 0.207 | 0.221 | 0.593 |
| mean abs cosine (cap 0.60) | 0.192 | 0.127 | 0.106 |
| held-out relational MSE | 0.0465 | 0.0214 | — |
| orthogonal reference on the same targets | 0.0161 | 0.0161 | — |
| off-diagonal Gram correlation with the teacher | 0.146 | 0.207 | 1.000 |

Four readings, in the order they change the decision:

- **The bar was not the obstacle.** The first run missed the harness control's 0.95 floor
  (0.917) and cleared the discrimination cap easily; tripling the budget cleared the control
  too (0.983). An earlier draft of [`ROADMAP.md`](ROADMAP.md) named the evaluation's
  calibration as the blocker. Measured, that was wrong, and it is retracted here rather than
  quietly dropped.
- **Retrieval quality is the obstacle.** Tripling the training budget bought +0.014 MRR@10
  against a 0.386 gap to the baseline. Two points are a slope, not a curve, and no
  extrapolation is claimed — but nothing in them suggests the gap is a budget away.
- **The student never beat a student that ignored the teacher.** On held-out chunk pairs its
  relational MSE stayed above `relational_reference_loss` — the loss reached by any mutually
  orthogonal embedding, which encodes nothing about the teacher at all — at both budgets.
  The stage's own progress line reads `0.5135 -> 0.0383`, which looks like learning; the
  reference for those targets is 0.0161. That gap between what a number looks like and what
  it means is the one durable thing this experiment produced, and it is now a function in
  `bcir/hosted/training/providers.py` with a gate that reads every run against it.
- **The ceiling is the teacher.** Distillation reproduces its teacher's geometry; a perfect
  student of `lexical-hash-v1` scores exactly what `lexical-hash-v1` scores. The only teacher
  this repository can obtain offline is the baseline itself, so this objective cannot improve
  retrieval here — at best it reproduces the baseline at the cost of weights.

**Two honesty notes.** The experiment was optimistic by construction: chunk text carries the
`[subject] Heading > Trail` prefix that *is* the control's query text, so the control numbers
above are an upper bound on what an uncontaminated student would score. And evaluating a
locally trained set at all required projecting queries with a provider the corpus cannot name
— `embed_chunks.build_provider` resolves `lexical-hash-v1` and sentence-transformers ids and
nothing else — so the run went through an in-process patch that is deliberately **not** in
the repository. <!-- claim: learned-provider-unregistrable -->

No gate re-runs this measurement; it is dated evidence, not a check. What *is* gated is what
the experiment produced on the way: `relational_reference_loss` and the shared
`relational_gram_loss` it is the trivial solution of, exercised by
`training/tools/verify_ml_components.py` on real corpus text on every host; and the harness
control's own premise, which the run made worth checking and which
`training/tools/verify_retrieval.py` now checks per query. The general form is registered as
L23 in [`../docs/security/laws.md`](../docs/security/laws.md).

## 3. GO criteria — what would warrant a learned provider

Promote a learned set only when **all** of these hold, and record the measurement that flips
each:

- **G1 — A teacher that is not the baseline.** The set is trained against semantics the
  lexical provider does not have: a pinned-revision encoder obtainable under this
  environment's network policy, a human-judged pair set, or an objective that is not teacher
  imitation. *Flipped by:* a set built by `embed_chunks.py` whose provider resolves offline
  and whose `revision` names exactly which weights produced it.
- **G2 — It beats the baseline where quality is measured.** Overall recall@5 and MRR@10
  exceed `lexical-hash-v1` on the same corpus digest and the same query manifest, by more
  than the run-to-run spread. *Flipped by:* two `evaluate_retrieval.py` runs, published
  together. The control family is not evidence for this criterion and never can be.
- **G3 — The control still means what it says.** The harness control is a positive control:
  its query is its target's own heading wording -- verbatim as a word sequence under the
  provider's own tokenizer, which `verify_retrieval.py` checks -- so a low score means the
  harness is broken. That argument is lexical. Before a learned set is scored against it, either the control's query
  text must be excluded from the encoder's view of the target chunk, or the control must be
  re-argued for a provider that does not match substrings — a
  [`CORPUS_STANDARD.md`](CORPUS_STANDARD.md) change, decided by a human, not a side effect of
  landing a model. Note what is and is not enforced today: the control's *premise* is now
  checked per query (`check_control_premise` in `verify_retrieval.py` requires each control
  query's word sequence to be a contiguous run inside its target's heading line), but the
  floor's *domain* is not. `verify_retrieval.py` builds its own lexical set every run, and
  `evaluate_retrieval.py` — the only tool that takes a `--set`, and the one that produced the
  numbers in §2 — applies no floor at all. So nothing would stop a learned set from being
  scored against a control whose argument does not cover it. Closing that is part of G3.
- **G4 — The objective cannot memorize the judged mapping.** The judgments are
  document-level, index files are themselves chunk documents, and an objective over raw chunk
  text can reach 84% of the judged mapping without ever constructing a pair. Contamination of
  that kind leaves every existing gate GREEN. *Flipped by:* a training-data filter, gated,
  that provably excludes judged targets from the objective.

If any of G1–G4 fails, the lexical baseline stays. As of today G1 fails outright — this
repository ships no teacher other than the baseline, and `huggingface.co` is refused by the
environment's proxy policy — so the gate holds without needing the others.

## 4. STOP criteria — if a learned-provider experiment is taken

An attempt is **time-boxed** and abandoned the moment any of these trips:

- **S1 — It never beats ignoring the teacher.** Held-out relational MSE stays at or above
  `relational_reference_loss` for its own targets. A run that cannot reach the floor a
  teacher-free model reaches has not learned the teacher, whatever its loss curve looks like.
- **S2 — The slope is not there.** Tripling the training budget moves overall MRR@10 by less
  than a tenth of the gap to the baseline. (Measured here: +0.014 against 0.386.)
- **S3 — Admission needs the bar moved.** The set clears only if `MAX_MEAN_ABS_COSINE` or
  `CONTROL_RECALL_AT_5` is loosened. Tuning a threshold to admit the thing it measures is what
  this standard exists to refuse; the deliverable is then the negative result, not the set.

## 5. Current verdict and evidence boundary

**The lexical baseline stays; the gate stands.** The corpus can train an embedding student,
does so on real corpus text in CI wherever torch is present, and gets a set that is worse at
the task by a factor of roughly three — for a structural reason, not a tuning one.

What this measurement does **not** establish: nothing about learned embeddings in general,
about larger students, about other objectives, or about other corpora. One architecture, one
objective, two budgets, one corpus, two evaluation runs. It establishes that *this* objective
— distilling the provider already in use — is bounded above by the provider already in use,
and that the corpus's own numbers say so.

Revisit when G1 becomes reachable: an offline-obtainable encoder with non-lexical semantics,
or a judged pair set this corpus can produce for itself. Until then the honest state is a
lexical baseline that declares itself lexical, which is worth more than a learned set that
declares itself learned and retrieves worse.
