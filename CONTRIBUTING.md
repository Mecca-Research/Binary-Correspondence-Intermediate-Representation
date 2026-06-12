# Contributing

Thanks for improving BCIR and the LLVM training material. This repository keeps
runtime code, design notes, and agent-readable examples side by side, so new
contributions should make it obvious which files are runnable, which files are
expected failures, and which files are documentation-only.

## Repository layout

This repo holds two separate things; keep contributions on the right side of
the line (see [`AGENTS.md`](AGENTS.md) and
[`docs/BCIR_Repo_Structure.md`](docs/BCIR_Repo_Structure.md)):

- **`bcir/`** — the executable conformance oracle (Python): the K_BCIR optimizer,
  GEM, M5 transduction, ROP/MAP front-ends, lowering (AOT/JIT), telemetry, verifier.
- **`mlir/`** — the IR law: the TableGen/ODS dialect family, the compiled
  `bcir-opt`, and the IRDL projection. `tools/` holds the validation scripts.
- **`llvm-training/`** — a separate LLVM/MLIR training corpus for agents. It is
  not part of the IR and the IR does not depend on it. The naming conventions
  below are about this corpus.

## Documentation and example naming conventions

Use filenames as part of the contract with readers and verification scripts.
Prefer lowercase, hyphen-separated topic names, and keep related prompt,
solution, input, and expected-output files adjacent.

### LLVM IR examples: `.ll`

Use `.ll` for complete, known-good LLVM IR modules that are expected to parse,
assemble, and pass the LLVM verifier under a modern LLVM toolchain using opaque
pointers.

- Place standalone training modules under a chapter-local `examples/` directory,
  for example `llvm-training/04-memory/examples/global-counter.ll`.
- Use descriptive topic names such as `loop-metadata.ll`,
  `mixed-stride-byte-offset.ll`, or `register-binding.ll`.
- For optimization snapshots, make the role obvious in the filename: use
  `name-before.ll`, `name-after-mem2reg.ll`, `name-after-simplifycfg.ll`, or
  `name-after-o2.ll`.
- Do not use `.ll` for deliberately broken IR unless the filename includes
  `invalid`; prefer `.invalid.ll.txt` for expected-failure fixtures.

### Exercise solutions: `.solution.ll`

Use `.solution.ll` for checked-in LLVM IR reference answers to exercises, for
example `llvm-training/exercises/001-add.solution.ll`.

- Pair the solution with a sibling `NNN-topic.prompt.md` file.
- Keep the numeric exercise prefix stable once published so links and learner
  progress references do not drift.
- A `.solution.ll` file must be a complete known-good module and is verified by
  `llvm-training/tools/verify-exercises.sh`.
- If a written answer is more appropriate than LLVM IR, use `NNN-topic.solution.md`
  instead of forcing a synthetic `.ll` file.

### Invalid LLVM IR fixtures: `.invalid.ll.txt`

Use `.invalid.ll.txt` for intentionally broken LLVM IR that should be rejected by
`llvm-as`, the LLVM verifier, or a documented pass, for example
`016-fix-phi-predecessor.invalid.ll.txt`.

- Keep invalid fixtures as text so broad known-good `.ll` sweeps cannot assemble
  them accidentally.
- Include the failure mode in the prompt, nearby prose, or a local README:
  parser error, verifier error, semantic-only hazard, or pass-specific failure.
- If LLVM accepts the file but the lesson is still intentionally invalid at a
  semantic level, add the marker required by
  `llvm-training/tools/verify-invalid-fixtures.sh` so the fixture is not mistaken
  for an accidentally valid parser/verifier failure.

### MLIR examples: `.mlir`

Use `.mlir` for MLIR dialect sketches, LLVM-dialect examples, and MLIR-to-LLVM
review fixtures.

- Put standalone bridge examples under `llvm-training/14-mlir-bridge/examples/`
  when they are part of the chapter corpus.
- Name files for the dialect boundary or lowering concept they demonstrate, such
  as `bcir-dialect-sketch.mlir` or `llvm-dialect-lowering.mlir`.
- `.mlir` files are not LLVM IR modules and should not be listed in the
  standalone LLVM IR manifest.
- Run `llvm-training/tools/verify-mlir-examples.sh`; it skips cleanly when
  `mlir-opt` is unavailable and parses examples when the MLIR toolchain exists.

### Binary-analysis evidence: `.csv`

Use `.csv` for checked-in trace, counter, or BCSA feature samples, especially
under `llvm-training/15-binary-analysis/examples/`.

- Name files by schema family and scenario, for example
  `dynamic-trace-sample.csv`, `perf-counter-sample.csv`, or
  `bcsa-feature-variant-wide.csv`.
- Keep headers stable and include at least one representative data row.
- If you introduce a new CSV schema family, update the schema expectations in
  `llvm-training/tools/verify-csv-schema.sh` or document why the file is outside
  that fixture-scoped check.
- CSV artifacts are evidence inputs, not LLVM IR; do not add them to
  `llvm-training/examples/README.md`.

### Prompt and solution Markdown files

Use Markdown for learner-facing prompts, written diagnoses, and review answers.

- Exercise prompts should use `NNN-topic.prompt.md`.
- Written exercise answers should use `NNN-topic.solution.md`.
- BCIR mapping prompts that are not numbered exercises may use descriptive names
  such as `bcir-operation.prompt.md`, `diagnostic-metadata.prompt.md`, or
  `ham-hint.prompt.md` beside the examples they discuss.
- A prompt should state the task, the command or review procedure to run, and the
  expected observation. A solution should be concise but specific enough for an
  agent or human reviewer to compare behavior.

## Updating the standalone example manifest

`llvm-training/examples/README.md` is the top-level manifest for known-good
standalone LLVM IR examples. Add or update entries there whenever a contribution
adds, removes, or renames a complete `*/examples/*.ll` file that should be part
of the repository-wide assembly and verifier guarantee.

Do **not** add these files to the manifest:

- `.ll.txt` or `.invalid.ll.txt` expected-failure fixtures;
- `.mlir` MLIR bridge files;
- `.csv` evidence artifacts;
- Markdown prompts or written solutions;
- illustrative fenced snippets that remain embedded in prose.

After changing standalone `.ll` examples, run
`llvm-training/tools/verify-manifest.sh` to confirm the manifest matches the
files discovered by the verification policy.

## BCIR mapping examples

BCIR mapping examples live in `llvm-training/bcir-mapping/` and should connect a
BCIR concept to concrete LLVM IR lowering behavior.

- Put runnable lowered LLVM IR in `llvm-training/bcir-mapping/examples/*.ll` and
  include it in `llvm-training/examples/README.md`.
- Use nearby Markdown prompts for review tasks that ask contributors or agents
  to explain a lowering rather than assemble a new module.
- Preserve source-level BCIR concepts in the file name when possible:
  `vertex-edge-attribute.ll`, `claim-resource-lookup.ll`,
  `hardware-aware-gem-lowering.ll`, `bcir-op-runtime-wrapper.ll`, or
  `mixed-stride-byte-offset.ll`.
- If you add source-like `.bcir.txt` fragments, keep them non-empty, include the
  required BCIR markers or operation keywords, and provide expected lowered LLVM
  IR companions as described by `llvm-training/tools/verify-bcir-mapping.sh`.
- If you add real `.bcir` assembler fixtures under `bcir-mapping/examples/`, add
  a sibling `<name>.generated.ll` expected output unless you are intentionally
  refreshing generated outputs with `UPDATE_BCIR_MAPPING=1`.

## Metadata-preservation examples

Metadata-preservation examples are most useful when they demonstrate both the
payload and why it must survive lowering or optimization.

- Prefer descriptive names such as `debug-location-preserved.ll`,
  `diagnostic-metadata-preservation.ll`, or `profile-branch-weights.ll`.
- Keep debug locations, diagnostic metadata, branch weights, TBAA, loop metadata,
  and BCIR-specific tags minimal but complete enough for `llvm-as` and
  `opt -passes=verify`.
- If the example is a before/after optimization snapshot, name the files with the
  pass relationship and document how to compare or regenerate them.
- Add a prompt when the lesson is a review task, such as checking that a lowering
  carried `!dbg`, diagnostic tags, or profile metadata through to the final IR.

## Verification scripts to run

Run the smallest set that covers the files you changed, and prefer the checked-in
scripts over ad-hoc command loops. Execute scripts from the repository root.

| Change type | Required checks |
|---|---|
| Any known-good `llvm-training/**/examples/*.ll` file | `./llvm-training/tools/verify-examples.sh` and `./llvm-training/tools/verify-manifest.sh` |
| Exercise `.solution.ll` files | `./llvm-training/tools/verify-exercises.sh` |
| Invalid fixtures (`.invalid.ll.txt` or other expected failures) | `./llvm-training/tools/verify-invalid-fixtures.sh` |
| Opaque-pointer migration or typed-pointer teaching material | `./llvm-training/tools/verify-opaque-pointers.sh` |
| Optimization before/after golden examples | `./llvm-training/tools/verify-opt-diff.sh` |
| MLIR bridge examples | `./llvm-training/tools/verify-mlir-examples.sh` |
| CSV binary-analysis fixtures | `./llvm-training/tools/verify-csv-schema.sh` |
| BCIR mapping examples or fragments | `./llvm-training/tools/verify-bcir-mapping.sh` |
| Portable backend smoke coverage | `./llvm-training/tools/smoke-llc.sh` |
| Interpreter smoke coverage | `./llvm-training/tools/smoke-lli.sh` |
| BOLT fixture changes | `./llvm-training/tools/smoke-bolt.sh` |

For IR changes outside the training corpus, run the relevant gate:

```bash
# bcir/ (the oracle) -- no third-party deps:
python -m bcir.tests.run_all

# mlir/ (the dialect law) -- needs libmlir-NN-dev + llvm-NN-dev:
bash tools/wsl/tblgen_check.sh        # ODS generators
bash tools/wsl/build_mlir.sh          # build bcir-opt
bash tools/wsl/check_ods_examples.sh  # pretty ODS corpus via bcir-opt
bash tools/irdl/check_corpus.sh       # IRDL projection on stock mlir-opt
```

Some training scripts intentionally skip when optional tools such as `llvm-as`,
`opt`, `llc`, `lli`, `mlir-opt`, or `llvm-bolt` are not installed. A clean skip
is acceptable only when the script says the tool is optional for that check;
missing required fixtures or manifest drift should fail the contribution.

## Future helper scripts

If generated indexes or manifests grow beyond the current checked-in scripts,
prefer adding explicit repository-root tools such as
`llvm-training/tools/generate-example-index.sh` or
`llvm-training/tools/verify-generated-indexes.sh`. New scripts should be
executable, documented in `llvm-training/tools/README.md`, fail closed when
required fixtures disappear, and be wired into CI when they protect repository
health.

## LLVM training dataset stability

The files under `llvm-training/dataset/` describe a small curated evaluation
set exported from tracked exercises and grading manifests. Do not describe or
expand it as a scaled fine-tuning corpus. Generated JSON Lines files are build
artifacts and stay out of version control unless maintainers intentionally
publish a versioned release snapshot.

Treat the dataset contract as an API:

- **IDs are permanent.** Once an exercise ID is exported, do not rename, reuse,
  or renumber it. Retire an exercise explicitly rather than assigning its ID to
  different content.
- **Semantic changes require an export-version decision.** Changes to the task,
  accepted answer kind, required artifacts, verification behavior, rubric, or
  split meaning must be reviewed for a schema/export version bump. Editorial
  fixes that do not alter meaning still change content checksums and should be
  called out in review.
- **Solutions cannot silently change scoring behavior.** A reference-solution
  update must be accompanied by review of the declarative checks and points.
  If the accepted behavior changes, update the rubric intentionally and make
  the compatibility impact explicit.
- **Provenance and licensing are mandatory.** Every exported record must retain
  repository-relative source and manifest paths, source-lineage/leakage data,
  and an SPDX license identifier backed by the repository license.
- **Splits are curated, not random.** Assign new exercises in
  `llvm-training/dataset/splits-v1.json` by concept family and source lineage.
  Prompt templates, generated variants, and exercises derived from one seed
  belong to one indivisible leakage group and cannot cross evaluation splits.

After changing any numbered exercise, autograder manifest, split assignment, or
dataset tool, run:

```sh
python3 llvm-training/tools/verify-exercise-manifests.py
python3 llvm-training/tools/verify-dataset-export.py
```

## Grader and dataset contribution requirements

Changes to numbered training exercises or evaluation tooling must preserve the
closed-loop contract:

- **Exercise manifests:** every numbered prompt has exactly one schema-valid
  manifest. Prompt, solution, artifact, tool, minimum-version, timeout,
  determinism, difficulty, license, and tool-absence fields must be accurate.
- **Scoring rubrics:** every check has a stable ID, dimension, deterministic
  pass/fail condition, and explicit points. Points sum exactly to the manifest
  score. Skipped optional checks earn no points and reduce score confidence.
- **Dataset provenance:** exports retain repository-relative source and manifest
  paths, SPDX license, source lineage/leakage group, split assignment, exporter
  identity, and SHA-256 checksums. Never mix unrelated or externally sourced
  material without documenting its license and lineage.
- **Stable IDs:** exercise IDs, exported IDs, leakage-group IDs, and published
  check IDs are append-only API identifiers. Do not renumber or reuse them;
  retire obsolete entries explicitly and review schema/version implications.
- **Grader self-tests:** reference answers must earn full raw scores with the
  required toolchain, partial fixtures must demonstrate nontrivial partial
  credit, malformed attempts must fail safely, and deterministic export tests
  must remain byte-identical.
- **Held-out isolation:** model-visible evaluation bundles use solution-free
  export/preparation paths. Reference content and `*.solution.*` siblings may be
  used by trusted oracle/self-test code only, never supplied to the evaluated
  model.

Required integration checks for such a change are:

```bash
python3 llvm-training/tools/verify-exercise-manifests.py
python3 -m unittest discover -s llvm-training/autograder/tests -p 'test_*.py'
python3 llvm-training/tools/grade-exercises.py --self-test --format json
python3 llvm-training/tools/verify-dataset-export.py
```
