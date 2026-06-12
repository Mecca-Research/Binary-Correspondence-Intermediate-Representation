# Curated exercise evaluation dataset

This directory defines a **small curated evaluation dataset** exported from the
numbered exercises and declarative grading manifests in `llvm-training/`. It is
not a scaled fine-tuning corpus, a data-generation pipeline, or a replacement
for the teaching material. The tracked prompts, reference artifacts, and
autograder manifests remain the source of truth.

## Controlled variants

[`VARIANT_POLICY.md`](VARIANT_POLICY.md) defines the reviewed, fixed-seed
variant-generation contract, rejection gates, semantic-family leakage rules, and
small initial budget. Generated variant JSON Lines files are review artifacts and
are not part of the curated export unless maintainers explicitly promote them.

## Tracked contracts

- [`schema-v1.json`](schema-v1.json) is the JSON Schema for one JSON Lines
  record. Schema version `1.0.0` covers stable IDs, normalized prompt and answer
  data, verification and rubric contracts, source paths, version assumptions,
  artifacts, splits, checksums, licensing, and provenance.
- [`splits-v1.json`](splits-v1.json) is the deterministic split manifest. It
  assigns every stable exercise ID to `train`, `validation`, or `test` by
  concept family and source lineage, never randomly at export time.
- A leakage group is indivisible. Exercises derived from the same prompt family,
  seed, or closely related source lineage must remain in one split. When a new
  relationship creates a transitive leakage risk, merge the affected groups
  rather than placing near-duplicate variants in separate splits.

The split manifest's normalized UTF-8 content is SHA-256 hashed into every
record. Prompt and reference-solution content receive independent SHA-256
checksums. Line endings are normalized to LF before hashing and export; all
other whitespace, including fenced code blocks and source-artifact indentation,
is preserved.

## Reproducible export

The exporter writes one compact JSON object per line, ordered by stable exercise
ID. Output defaults to stdout and excludes solution content unless explicitly
requested:

```sh
python3 llvm-training/tools/export-exercise-dataset.py \
  --without-solutions --split all \
  --output /tmp/llvm-training-eval-v1.jsonl
```

To create an evaluator-only snapshot that embeds reference answers:

```sh
python3 llvm-training/tools/export-exercise-dataset.py \
  --include-solutions --split test \
  --output /tmp/llvm-training-eval-v1-test-with-solutions.jsonl
```

`--split` accepts `train`, `validation`, `test`, or `all`. The option filters the
checked-in assignments; it never invents or randomizes assignments. The two
solution modes change only whether reference content is embedded. Solution
paths, kinds, and checksums remain available in solution-free exports so an
evaluation harness can identify the contract without receiving the answer.

Generated `*.jsonl` files under this directory are ignored by Git. Maintainers
may intentionally publish a versioned release snapshot, but routine changes
should be reviewed through the tracked source exercises, manifests, schema, and
split manifest rather than by committing regenerated bulk output.

## Verification

Run the dependency-free verifier from the repository root:

```sh
python3 llvm-training/tools/verify-dataset-export.py
```

It exports the dataset repeatedly in a temporary directory and checks:

- JSON Schema conformance for every record;
- stable, unique IDs and ID-sorted output;
- exact split membership and non-overlapping, indivisible leakage groups;
- prompt, solution, and split-manifest checksums;
- parity between solution-including and solution-free modes;
- byte-for-byte deterministic regeneration; and
- existence of every source, artifact, and provenance reference.

## Current scale and intended use

Dataset schema v1 exports 42 stable-ID records. This deliberately small,
curated set supports held-out agent evaluation, grader regression tests, and
curriculum analysis. It is not a statistically representative benchmark and is
not a production-scale pretraining or fine-tuning dataset. Model-visible exports
must use the default `--without-solutions`; solution-bearing exports are limited
to trusted oracle/reviewer workflows.
