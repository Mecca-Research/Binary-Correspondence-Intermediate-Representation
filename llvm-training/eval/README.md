# Provider-neutral exercise evaluation

`llvm-training/tools/run-eval.py` connects four deliberately separate stages:
prompt preparation, provider-neutral generation, deterministic grading, and
aggregate reporting. It uses the executable exercise set in
`llvm-training/autograder/exercises.json` and joins topic/difficulty metadata
from the matching files in `llvm-training/autograder/manifests/`.

The runner has no hosted-provider dependency and does not read credentials. A
caller chooses a run directory with `--output-dir`; generated data is never
written into tracked exercise or corpus paths.

## Quick start

Run the complete loop with a local generator:

```bash
python3 llvm-training/tools/run-eval.py run \
  --output-dir /tmp/bcir-eval/my-run \
  --generator-id my-agent-local \
  --generator-command 'my-agent --input {input} --output {output}' \
  --generation-parameters '{"temperature": 0, "seed": 17}'
```

The command template is parsed with Python's `shlex.split` and executed as an
argument vector with `shell=False`. It must contain both `{input}` and
`{output}`. The placeholders are replaced within their individual arguments;
they are not shell-expanded.

Select a subset by repeating `--exercise`:

```bash
python3 llvm-training/tools/run-eval.py run \
  --output-dir /tmp/bcir-eval/smoke \
  --exercise 001 --exercise 025 \
  --fixture-adapter reference
```

## Modes

- `prepare` copies each prompt, discovers non-solution sibling/link context,
  bundles that context, and writes the generator input JSON.
- `generate` invokes the configured adapter. If prepared inputs are absent, it
  prepares them first.
- `grade` grades attempts already in the run directory, or a caller-provided
  `--attempts-dir`, and then refreshes the report.
- `report` rebuilds `results.jsonl` and `summary.json` from prior grade records.
- `run` performs prepare, generate, grade, and report in order.

For grading attempts produced elsewhere, use the conventional layout
`<attempts-dir>/<exercise-id>/answer.<ext>`:

```bash
python3 llvm-training/tools/run-eval.py grade \
  --output-dir /tmp/bcir-eval/external-grade \
  --attempts-dir /srv/attempts \
  --exercise 001
```

## Generator adapter contract

For every exercise, the runner passes the adapter an input JSON document with:

```json
{
  "contract_version": "1.0",
  "exercise_id": "001",
  "prompt_text": "...",
  "context_paths": ["/absolute/run/path/prepared/001/context/..."],
  "output_contract": {
    "attempt_directory": "/absolute/run/path/attempts/001",
    "primary_artifact_path": "/absolute/run/path/attempts/001/answer.ll",
    "answer_kind": "llvm-ir",
    "required_artifacts": ["answer.ll"]
  },
  "generation_parameters": {}
}
```

The adapter writes output JSON to `{output}`. A completed result has this shape:

```json
{
  "contract_version": "1.0",
  "generated_artifact_paths": ["/absolute/run/path/attempts/001/answer.ll"],
  "model": {
    "identity": "local-model-or-agent-name",
    "revision": "optional-revision"
  },
  "token_counts": {
    "input": 123,
    "output": 45,
    "total": 168
  },
  "status": "completed"
}
```

`token_counts` may be `null` when unavailable. `status` is `completed`,
`failed`, or `skipped`. Every listed artifact must exist below the assigned
attempt directory, and a completed response must list the primary artifact.
The adapter may add provider-neutral metadata fields; the runner preserves them.

The runner records generator identity separately from generation parameters, so
hosted APIs, local inference servers, CLI agents, and deterministic fixtures can
all implement the same contract. Credentials, endpoints, and provider SDKs are
intentionally outside this repository and must be supplied by the caller's
adapter environment.

## Deterministic CI fixture adapters

Three built-in adapters exercise the pipeline without a model or network:

- `--fixture-adapter reference` copies the checked-in reference answer;
- `--fixture-adapter empty` creates a zero-byte primary artifact;
- `--fixture-adapter partial` copies the known incomplete autograder fixture.

The partial adapter is defined for the executable grader set. These adapters
identify themselves as deterministic fixtures in the reproducibility manifest
and report no token counts.

## Resume and regeneration

Generation is resumable by default. An attempt is reused when its
`generator-output.json` says `completed`, names the expected primary artifact,
and that artifact still exists. The runner prints `[resume]` and does not invoke
the adapter again. Use `--force-regenerate` to delete and recreate selected
attempt directories.

A run directory is tied to the exercise IDs recorded in its reproducibility
manifest. Later `generate`, `grade`, and `report` calls reuse those IDs unless
`--exercise` is supplied explicitly.

## Run artifacts

A run directory contains:

```text
<output-dir>/
  reproducibility-manifest.json
  prepared/<id>/input.json
  prepared/<id>/prompt.md
  prepared/<id>/context/...
  attempts/<id>/answer.<ext>
  attempts/<id>/generator-output.json
  grades/<id>.json
  results.jsonl
  summary.json
```

The reproducibility manifest records the repository commit and dirty state,
exercise-registry/schema version and hashes, tool paths and versions, generator
identity, generation parameters, grader version/hash, and attempt hashes.
`results.jsonl` contains one complete record per exercise. `summary.json`
contains:

- pass rate, mean score, and median score;
- mean score grouped by topic and difficulty;
- hard/infrastructure-failure rate;
- skipped-check rate;
- required-tool availability and observed tool invocations;
- separate `generation_quality` and `infrastructure` sections.

A low-scoring or invalid generated answer is a `quality_failure`. Missing
attempts, generator failures, malformed adapter output, and grader execution
failures are `infrastructure_failure`. This distinction prevents broken
orchestration from being reported as poor model quality. The process exits `2`
when infrastructure failures occur; ordinary quality failures remain valid
evaluation results and do not make the orchestration command fail.

## Security boundary

**Generated attempts and adapter metadata are untrusted content.** Never source
generated files as shell scripts, evaluate them, or interpolate their contents
into a command string. Any tool that consumes an attempt must use an
argument-safe subprocess API with an explicit argument vector and `shell=False`
(or the equivalent in another language). The runner follows this rule for both
the local generator adapter and the existing grader. Run model-generated code
only inside an appropriately isolated sandbox with resource limits suitable for
your threat model.
