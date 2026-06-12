# Controlled exercise variant policy

This policy governs generated problem/reference-solution pairs derived from the
numbered LLVM training exercises. Generated records are **review candidates**,
not automatically published training data. The tracked parent exercises,
per-exercise grading manifests, and curated split manifest remain authoritative.

## Scope and initial budget

`llvm-training/tools/generate-exercise-variants.py` starts with a deliberately
small default budget of **five accepted records per invocation**, at most one
from each reviewed generator family. Maintainers must review the prompt,
reference solution, parameter values, oracle report, and hashes before promoting
any generated record. Raising the budget or adding a family is a policy change,
not a routine regeneration.

The initial typed parameter vocabulary is:

- integer widths and constant values;
- array lengths, struct field positions, and legal alignments;
- branch shapes;
- metadata payload values; and
- register/claim identifiers.

Only parent exercises with strong executable LLVM oracles are eligible. The
initial families derive from exercises 001, 002, 006, 011, and 014. They emit
LLVM IR reference solutions, not Markdown review answers.

## Reproducibility contract

A fixed integer `--seed` is mandatory. The generator derives a family-local
random stream from the seed, generator version, family name, and attempt number,
so adding an unrelated family cannot perturb existing families. Every accepted
record contains:

- the fixed seed and generator version;
- the permanent parent exercise ID;
- the complete typed parameter object;
- the prompt and reference solution rendered from that same object;
- the grading manifest rendered from that same object;
- prompt, solution, manifest, and normalized-IR SHA-256 hashes; and
- the semantic family, concept family, leakage group, and inherited split.

JSON objects are serialized with sorted keys and compact separators. Runs with
the same toolchain, generator version, arguments, and source tree must be
byte-for-byte identical. CI checks a small fixed seed set rather than expanding
the full combinatorial space.

## Oracle and rejection gates

The generated reference solution is graded through the same
`grade-exercises.py` manifest engine used for learner attempts. A record is
accepted only when every artifact, assembly, verification, normalization,
structural, opaque-pointer, and semantic-execution check passes. Missing LLVM
oracle tools are a hard generation error, not a skipped acceptance gate.

Candidates are rejected when any of these conditions holds:

1. `llvm-as`, `opt -passes=verify`, normalization, or `lli` semantics fail;
2. the typed parameters match an existing semantic-equivalence key;
3. the reference text hash or normalized IR structure hash duplicates an
   accepted candidate;
4. a family-specific triviality rule fires (for example, cancelling constants,
   identical branch outcomes, an out-of-range field, or indistinguishable
   success/failure values);
5. target triples, data layouts, target CPU/features, or target-specific
   namespaces introduce an unsupported target assumption; or
6. legacy typed-pointer syntax violates the corpus opaque-pointer policy.

Prior generated JSON Lines files may be supplied with repeatable `--existing`
arguments. Their semantic keys, solution hashes, and normalized-structure hashes
seed all three deduplication sets before new candidates are sampled.

The normalized-structure hash is computed after `opt -S`, removal of comments
and source filenames, deterministic renaming of local values and metadata node
numbers, and whitespace normalization. It complements rather than replaces the
exact text hashes.

Each run emits deterministic acceptance and rejection counts. Rejection reasons
are aggregated in the optional JSON report; they must not depend on timestamps,
temporary paths, or nondeterministic diagnostics.

## Negative variants

Negative variants are allowed only when the expected parser/verifier diagnostic
class or semantic classification is deterministic across the supported LLVM
versions. Their typed parameter object must state `polarity: negative` and carry
the expected classification or diagnostic contract. The current reviewed
catalog contains no negative family. Adding one requires:

- an explicit invalid input separate from the valid reference repair or answer;
- a version-tolerant deterministic diagnostic matcher;
- a manifest check proving the expected rejection/classification; and
- tests on every LLVM version claimed by the dataset.

Merely producing malformed IR is not sufficient.

## Leakage control and splits

A generated record inherits its parent's `split`, `concept_family`, and
`leakage_group` from `splits-v1.json`. Its `semantic_family` identifies the
specific generator family and equivalence key. Generated siblings and their
parent are indivisible: they must remain in that inherited split. Promotion must
never assign siblings independently or use random split selection.

If a new generator relates parents from different leakage groups, maintainers
must first merge those groups in the curated split manifest and review the
transitive leakage impact.

## Human review and Markdown prohibition

Generated records are marked `generated-unreviewed` and
`human_review_required: true`. The generator must not auto-generate Markdown
review solutions. Such a family may be added only after a deterministic rubric
exists, the rubric is exercised by the attempt grader, and a human reviews every
published answer for semantic completeness. Keyword coverage alone is not a
semantic oracle.

## Usage

From the repository root, with `llvm-as`, `opt`, and `lli` available:

```sh
python3 llvm-training/tools/generate-exercise-variants.py \
  --seed 20260612 \
  --budget 5 \
  --output /tmp/llvm-training-variants.jsonl \
  --report /tmp/llvm-training-variants-report.json
```

Generated JSON Lines files remain build artifacts and are ignored by Git unless
maintainers intentionally publish a reviewed, versioned snapshot.
