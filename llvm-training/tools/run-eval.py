#!/usr/bin/env python3
"""Provider-neutral generation, grading, and reporting for LLVM training exercises."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import sys
from typing import Any

RUNNER_SCHEMA_VERSION = "1.0"
GENERATOR_CONTRACT_VERSION = "1.0"
ANSWER_EXTENSIONS = {
    "llvm-ir": ".ll",
    "mlir": ".mlir",
    "markdown-review": ".md",
    "pass-output": ".ll",
    "diagnostic": ".md",
}
FIXTURE_NAMES = ("reference", "empty", "partial")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo_root, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def find_tool(name: str) -> str | None:
    suffix = os.environ.get("LLVM_SUFFIX", "")
    candidates = ([name + suffix] if suffix else []) + [name]
    candidates.extend(f"{name}-{version}" for version in range(30, 9, -1))
    return next((path for candidate in candidates if (path := shutil.which(candidate))), None)


def tool_version(path: str) -> str:
    result = subprocess.run(
        [path, "--version"], text=True, capture_output=True, check=False, timeout=5
    )
    lines = (result.stdout or result.stderr).splitlines()
    return lines[0].strip() if lines else "version unavailable"


def answer_path(attempts_dir: Path, entry: dict[str, Any]) -> Path:
    return attempts_dir / entry["id"] / ("answer" + ANSWER_EXTENSIONS[entry["answer_kind"]])


def repository_path(repo_root: Path, value: str) -> Path:
    path = (repo_root / value).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"repository path escapes the checkout: {value}") from exc
    return path


def discover_context_paths(repo_root: Path, metadata: dict[str, Any]) -> list[str]:
    prompt = repository_path(repo_root, metadata["prompt"])
    solution = metadata.get("solution")
    excluded = {metadata["prompt"], solution}
    candidates: set[str] = set()

    prefix = prompt.name.removesuffix(".prompt.md")
    for sibling in prompt.parent.glob(prefix + ".*"):
        relative = sibling.resolve().relative_to(repo_root.resolve()).as_posix()
        if sibling.is_file() and relative not in excluded:
            candidates.add(relative)

    text = prompt.read_text(encoding="utf-8")
    for match in re.findall(r"(?:llvm-training/)?[A-Za-z0-9_./-]+\.(?:ll|mlir|txt|md|c|cpp)", text):
        value = match.strip("`'\"()[]{}.,:;")
        possibilities = [repo_root / value, prompt.parent / value]
        for candidate in possibilities:
            if candidate.is_file():
                relative = candidate.resolve().relative_to(repo_root.resolve()).as_posix()
                if relative not in excluded and ".solution." not in relative:
                    candidates.add(relative)
                break
    return sorted(candidates)


def load_catalog(repo_root: Path, registry_path: Path, manifests_dir: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    registry = read_json(registry_path)
    grader_entries = {entry["id"]: entry for entry in registry["exercises"]}
    metadata_entries: dict[str, dict[str, Any]] = {}
    for path in sorted(manifests_dir.glob("*.json")):
        metadata = read_json(path)
        metadata["_manifest_path"] = path.relative_to(repo_root).as_posix()
        metadata_entries[metadata["id"]] = metadata

    missing = sorted(set(grader_entries) - set(metadata_entries))
    if missing:
        raise ValueError("gradeable exercises lack metadata manifests: " + ", ".join(missing))

    catalog: dict[str, dict[str, Any]] = {}
    for exercise_id, grader in grader_entries.items():
        metadata = metadata_entries[exercise_id]
        if metadata["prompt"] != grader["prompt_path"] or metadata["solution"] != grader["reference_solution_path"]:
            raise ValueError(f"manifest and grader registry disagree for exercise {exercise_id}")
        item = dict(grader)
        item["metadata"] = metadata
        item["context_paths"] = discover_context_paths(repo_root, metadata)
        catalog[exercise_id] = item
    return str(registry["schema_version"]), catalog


def select_exercises(catalog: dict[str, dict[str, Any]], selected: list[str] | None) -> list[dict[str, Any]]:
    ids = selected or list(catalog)
    unknown = sorted(set(ids) - set(catalog))
    if unknown:
        raise ValueError("unknown or non-gradeable exercise ID(s): " + ", ".join(unknown))
    if len(ids) != len(set(ids)):
        raise ValueError("exercise IDs must not be repeated")
    return [catalog[exercise_id] for exercise_id in ids]


def parse_generation_parameters(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    candidate = Path(value)
    try:
        data = read_json(candidate) if candidate.is_file() else json.loads(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid --generation-parameters JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("--generation-parameters must be a JSON object or a path to one")
    return data


def generator_identity(args: argparse.Namespace) -> dict[str, Any]:
    if args.fixture_adapter:
        return {"adapter": "deterministic-fixture", "identity": args.fixture_adapter}
    if args.generator_command:
        executable = shlex.split(args.generator_command)[0]
        return {
            "adapter": "local-command",
            "identity": args.generator_id or executable,
            "command_template": args.generator_command,
        }
    return {"adapter": "none", "identity": args.generator_id or "not-configured"}


def collect_tools(entries: list[dict[str, Any]], grader: Path) -> dict[str, dict[str, Any]]:
    names = {tool for entry in entries for tool in entry["required_tools"]}
    tools: dict[str, dict[str, Any]] = {
        "python": {"path": sys.executable, "version": sys.version.split()[0]},
        "grader": {"path": str(grader), "version": "schema 1.0", "sha256": sha256(grader)},
    }
    for name in sorted(names):
        path = find_tool(name)
        tools[name] = {
            "available": path is not None,
            "path": path,
            "version": tool_version(path) if path else None,
        }
    return tools


def make_manifest(
    args: argparse.Namespace,
    repo_root: Path,
    registry_schema: str,
    registry_path: Path,
    schema_path: Path,
    grader: Path,
    entries: list[dict[str, Any]],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "generator_contract_version": GENERATOR_CONTRACT_VERSION,
        "created_at": utc_now(),
        "repository": {
            "root": str(repo_root),
            "commit": git_value(repo_root, "rev-parse", "HEAD"),
            "dirty": bool(git_value(repo_root, "status", "--porcelain")),
        },
        "exercise_manifest": {
            "schema_version": registry_schema,
            "schema_path": schema_path.relative_to(repo_root).as_posix(),
            "schema_sha256": sha256(schema_path),
            "registry_path": registry_path.relative_to(repo_root).as_posix(),
            "registry_sha256": sha256(registry_path),
        },
        "exercise_ids": [entry["id"] for entry in entries],
        "tools": collect_tools(entries, grader),
        "generator": generator_identity(args),
        "generation_parameters": parameters,
        "grader": {"version": "1.0", "path": str(grader), "sha256": sha256(grader)},
        "attempt_hashes": {},
    }


def prepare(args: argparse.Namespace, repo_root: Path, entries: list[dict[str, Any]], parameters: dict[str, Any]) -> None:
    prepared_root = args.output_dir / "prepared"
    for entry in entries:
        exercise_id = entry["id"]
        metadata = entry["metadata"]
        directory = prepared_root / exercise_id
        context_root = directory / "context"
        directory.mkdir(parents=True, exist_ok=True)
        prompt_source = repository_path(repo_root, entry["prompt_path"])
        prompt_copy = directory / "prompt.md"
        shutil.copyfile(prompt_source, prompt_copy)
        bundled_context: list[str] = []
        for relative in entry["context_paths"]:
            source = repository_path(repo_root, relative)
            target = context_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            bundled_context.append(str(target.resolve()))
        attempt = answer_path(args.output_dir / "attempts", entry)
        generator_input = {
            "contract_version": GENERATOR_CONTRACT_VERSION,
            "exercise_id": exercise_id,
            "prompt_text": prompt_source.read_text(encoding="utf-8"),
            "context_paths": bundled_context,
            "output_contract": {
                "attempt_directory": str(attempt.parent.resolve()),
                "primary_artifact_path": str(attempt.resolve()),
                "answer_kind": entry["answer_kind"],
                "required_artifacts": [attempt.name],
                "generator_output_schema": {
                    "generated_artifact_paths": "array of paths below attempt_directory",
                    "model": "object with provider-neutral identity/revision fields",
                    "token_counts": "object or null",
                    "status": "completed, failed, or skipped",
                },
            },
            "generation_parameters": parameters,
            "exercise_metadata": {
                "title": metadata["title"],
                "difficulty": metadata["difficulty"],
                "topics": [tag for tag in metadata["tags"] if tag != "llvm-training"],
                "manifest_path": metadata["_manifest_path"],
            },
        }
        write_json(directory / "input.json", generator_input)


def validate_artifact_path(path_value: str, attempt_directory: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = attempt_directory / path
    path = path.resolve()
    try:
        path.relative_to(attempt_directory.resolve())
    except ValueError as exc:
        raise ValueError(f"generated artifact escapes attempt directory: {path_value}") from exc
    if not path.is_file():
        raise ValueError(f"generated artifact does not exist: {path}")
    return path


def validate_generator_output(output: Any, attempt_directory: Path, primary_artifact: Path) -> dict[str, Any]:
    if not isinstance(output, dict):
        raise ValueError("generator output must be a JSON object")
    if output.get("status") not in {"completed", "failed", "skipped"}:
        raise ValueError("generator output status must be completed, failed, or skipped")
    paths = output.get("generated_artifact_paths")
    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
        raise ValueError("generated_artifact_paths must be an array of strings")
    resolved = [validate_artifact_path(path, attempt_directory) for path in paths]
    if output["status"] == "completed" and primary_artifact.resolve() not in resolved:
        raise ValueError("completed generator output must list the primary artifact")
    if "model" not in output or not isinstance(output["model"], dict):
        raise ValueError("generator output model must be an object")
    if output.get("token_counts") is not None and not isinstance(output["token_counts"], dict):
        raise ValueError("generator output token_counts must be an object or null")
    output["generated_artifact_paths"] = [str(path) for path in resolved]
    return output


def fixture_generate(
    name: str, repo_root: Path, entry: dict[str, Any], primary_artifact: Path
) -> dict[str, Any]:
    primary_artifact.parent.mkdir(parents=True, exist_ok=True)
    if name == "reference":
        shutil.copyfile(repository_path(repo_root, entry["reference_solution_path"]), primary_artifact)
    elif name == "empty":
        primary_artifact.write_text("", encoding="utf-8")
    elif name == "partial":
        fixture = repo_root / "llvm-training" / "autograder" / "fixtures" / "incomplete" / "attempts" / entry["id"] / primary_artifact.name
        if not fixture.is_file():
            raise ValueError(f"no known partial fixture for exercise {entry['id']}")
        shutil.copyfile(fixture, primary_artifact)
    return {
        "contract_version": GENERATOR_CONTRACT_VERSION,
        "generated_artifact_paths": [str(primary_artifact.resolve())],
        "model": {"identity": f"fixture:{name}", "revision": "deterministic-v1"},
        "token_counts": None,
        "status": "completed",
    }


def completed_attempt_is_reusable(output_path: Path, primary_artifact: Path) -> bool:
    if not output_path.is_file() or not primary_artifact.is_file():
        return False
    try:
        output = read_json(output_path)
        return output.get("status") == "completed" and any(
            Path(path).resolve() == primary_artifact.resolve()
            for path in output.get("generated_artifact_paths", [])
        )
    except (OSError, json.JSONDecodeError, TypeError):
        return False


def generate(args: argparse.Namespace, repo_root: Path, entries: list[dict[str, Any]]) -> int:
    if not args.fixture_adapter and not args.generator_command:
        raise ValueError("generate and run require --generator-command or --fixture-adapter")
    failures = 0
    for entry in entries:
        exercise_id = entry["id"]
        input_path = args.output_dir / "prepared" / exercise_id / "input.json"
        if not input_path.is_file():
            raise ValueError(f"prepared input is missing for exercise {exercise_id}; run prepare first")
        primary_artifact = answer_path(args.output_dir / "attempts", entry)
        output_path = primary_artifact.parent / "generator-output.json"
        if not args.force_regenerate and completed_attempt_is_reusable(output_path, primary_artifact):
            print(f"[resume] {exercise_id}: keeping completed attempt")
            continue
        if primary_artifact.parent.exists():
            shutil.rmtree(primary_artifact.parent)
        primary_artifact.parent.mkdir(parents=True)
        started = utc_now()
        try:
            if args.fixture_adapter:
                output = fixture_generate(args.fixture_adapter, repo_root, entry, primary_artifact)
                command_record = None
                stderr = ""
                returncode = 0
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    token.replace("{input}", str(input_path.resolve())).replace("{output}", str(output_path.resolve()))
                    for token in shlex.split(args.generator_command)
                ]
                if not any("{input}" in token for token in shlex.split(args.generator_command)) or not any(
                    "{output}" in token for token in shlex.split(args.generator_command)
                ):
                    raise ValueError("--generator-command must contain {input} and {output} placeholders")
                result = subprocess.run(
                    command,
                    cwd=repo_root,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=args.generator_timeout,
                )
                command_record = command
                stderr = result.stderr
                returncode = result.returncode
                if result.returncode != 0:
                    raise RuntimeError(f"generator exited with status {result.returncode}: {result.stderr.strip()}")
                if not output_path.is_file():
                    raise RuntimeError("generator did not write its output JSON")
                output = read_json(output_path)
            output = validate_generator_output(output, primary_artifact.parent, primary_artifact)
            output.update({"started_at": started, "finished_at": utc_now()})
            if command_record:
                output["adapter_command"] = command_record
                output["adapter_stderr"] = stderr
                output["adapter_returncode"] = returncode
            write_json(output_path, output)
            print(f"[generated] {exercise_id}: {output['status']}")
            if output["status"] != "completed":
                failures += 1
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            failures += 1
            write_json(
                output_path,
                {
                    "contract_version": GENERATOR_CONTRACT_VERSION,
                    "generated_artifact_paths": [],
                    "model": {"identity": generator_identity(args)["identity"]},
                    "token_counts": None,
                    "status": "failed",
                    "started_at": started,
                    "finished_at": utc_now(),
                    "infrastructure_error": str(exc),
                },
            )
            print(f"[generation-failed] {exercise_id}: {exc}", file=sys.stderr)
    return failures


def grade_one(grader: Path, repo_root: Path, registry: Path, entry: dict[str, Any], attempt: Path) -> tuple[dict[str, Any] | None, str | None]:
    command = [
        sys.executable,
        str(grader),
        "--registry",
        str(registry),
        "--exercise",
        entry["id"],
        "--answer",
        str(attempt),
        "--format",
        "json",
    ]
    result = subprocess.run(command, cwd=repo_root, text=True, capture_output=True, check=False)
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None, f"grader did not emit valid JSON (exit {result.returncode}): {result.stderr.strip()}"
    if not report.get("results"):
        return None, "grader emitted no exercise result"
    return report, None


def grade(
    args: argparse.Namespace,
    repo_root: Path,
    registry: Path,
    grader: Path,
    entries: list[dict[str, Any]],
) -> int:
    attempts_dir = args.attempts_dir.resolve() if args.attempts_dir else args.output_dir / "attempts"
    grades_dir = args.output_dir / "grades"
    grades_dir.mkdir(parents=True, exist_ok=True)
    infrastructure_failures = 0
    for entry in entries:
        exercise_id = entry["id"]
        attempt = answer_path(attempts_dir, entry)
        generation_output_path = attempts_dir / exercise_id / "generator-output.json"
        generation = read_json(generation_output_path) if generation_output_path.is_file() else {
            "status": "external",
            "model": {"identity": "external-attempt"},
            "token_counts": None,
            "generated_artifact_paths": [str(attempt)],
        }
        record: dict[str, Any] = {
            "runner_schema_version": RUNNER_SCHEMA_VERSION,
            "exercise_id": exercise_id,
            "title": entry["metadata"]["title"],
            "topics": [tag for tag in entry["metadata"]["tags"] if tag != "llvm-training"],
            "difficulty": entry["metadata"]["difficulty"],
            "attempt_path": str(attempt),
            "attempt_sha256": sha256(attempt) if attempt.is_file() else None,
            "generation": generation,
            "graded_at": utc_now(),
        }
        if not attempt.is_file():
            infrastructure_failures += 1
            record.update({
                "outcome": "infrastructure_failure",
                "infrastructure_error": f"attempt artifact is missing: {attempt}",
                "grade": None,
            })
            report = None
            error = None
        else:
            report, error = grade_one(grader, repo_root, registry, entry, attempt)
            if error:
                infrastructure_failures += 1
                record.update({"outcome": "infrastructure_failure", "infrastructure_error": error, "grade": None})
        if report is not None:
            grade_result = report["results"][0]
            failed_checks = [check for check in grade_result["checks"] if check["status"] == "fail"]
            skipped_checks = [check for check in grade_result["checks"] if check["status"] == "skip"]
            generation_failed = generation.get("status") == "failed"
            if generation_failed:
                outcome = "infrastructure_failure"
                infrastructure_failures += 1
            else:
                outcome = "passed" if not failed_checks else "quality_failure"
            record.update(
                {
                    "outcome": outcome,
                    "grade": grade_result,
                    "grader_toolchain": report.get("toolchain", {}),
                    "failed_check_count": len(failed_checks),
                    "skipped_check_count": len(skipped_checks),
                }
            )
        write_json(grades_dir / f"{exercise_id}.json", record)
        print(f"[graded] {exercise_id}: {record['outcome']}")
    return infrastructure_failures


def load_grade_records(output_dir: Path, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for entry in entries:
        path = output_dir / "grades" / f"{entry['id']}.json"
        if not path.is_file():
            raise ValueError(f"grade result is missing for exercise {entry['id']}: {path}")
        records.append(read_json(path))
    return records


def grouped_scores(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[float]] = {}
    for record in records:
        if not record.get("grade") or record.get("outcome") == "infrastructure_failure":
            continue
        values = record[key] if isinstance(record[key], list) else [record[key]]
        for value in values:
            groups.setdefault(value, []).append(float(record["grade"]["score_percent"]))
    return {
        name: {"count": len(scores), "mean_score": round(statistics.fmean(scores), 2)}
        for name, scores in sorted(groups.items())
    }


def build_summary(records: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    graded = [record for record in records if record.get("grade")]
    quality_graded = [record for record in graded if record.get("outcome") != "infrastructure_failure"]
    scores = [float(record["grade"]["score_percent"]) for record in quality_graded]
    passed = sum(record["outcome"] == "passed" for record in records)
    infrastructure = sum(record["outcome"] == "infrastructure_failure" for record in records)
    quality_failures = sum(record["outcome"] == "quality_failure" for record in records)
    skipped = sum(record.get("skipped_check_count", 0) for record in records)
    checks = sum(len(record["grade"]["checks"]) for record in graded)

    required_occurrences: dict[str, int] = {}
    for exercise_id in manifest["exercise_ids"]:
        grade_record = next((record for record in records if record["exercise_id"] == exercise_id), None)
        if grade_record and grade_record.get("grade"):
            for check in grade_record["grade"]["checks"]:
                command = check.get("command") or []
                if command:
                    required_occurrences[Path(command[0]).name] = required_occurrences.get(Path(command[0]).name, 0) + 1
    declared = {
        name: info for name, info in manifest["tools"].items()
        if name not in {"python", "grader"}
    }
    available_count = sum(bool(info.get("available")) for info in declared.values())
    toolchain_coverage = {
        "available_tools": available_count,
        "required_tools": len(declared),
        "coverage_rate": round(available_count / len(declared), 4) if declared else 1.0,
        "by_tool": declared,
        "observed_check_tool_invocations": required_occurrences,
    }
    return {
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "exercise_count": len(records),
        "graded_count": len(graded),
        "pass_rate": round(passed / len(records), 4) if records else 0.0,
        "mean_score": round(statistics.fmean(scores), 2) if scores else None,
        "median_score": round(statistics.median(scores), 2) if scores else None,
        "score_by_topic": grouped_scores(records, "topics"),
        "score_by_difficulty": grouped_scores(records, "difficulty"),
        "hard_failure_rate": round(infrastructure / len(records), 4) if records else 0.0,
        "skipped_check_rate": round(skipped / checks, 4) if checks else 0.0,
        "toolchain_coverage": toolchain_coverage,
        "outcomes": {
            "passed": passed,
            "quality_failures": quality_failures,
            "infrastructure_failures": infrastructure,
        },
        "generation_quality": {
            "evaluated_attempts": passed + quality_failures,
            "passed": passed,
            "failed": quality_failures,
        },
        "infrastructure": {
            "hard_failures": infrastructure,
            "generation_failures": sum(record.get("generation", {}).get("status") == "failed" for record in records),
        },
    }


def update_attempt_hashes(manifest: dict[str, Any], records: list[dict[str, Any]]) -> None:
    manifest["attempt_hashes"] = {
        record["exercise_id"]: {
            "path": record["attempt_path"],
            "sha256": record.get("attempt_sha256"),
            "generation_status": record.get("generation", {}).get("status"),
        }
        for record in records
    }
    manifest["updated_at"] = utc_now()


def report(args: argparse.Namespace, entries: list[dict[str, Any]]) -> int:
    manifest_path = args.output_dir / "reproducibility-manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"reproducibility manifest is missing: {manifest_path}")
    manifest = read_json(manifest_path)
    records = load_grade_records(args.output_dir, entries)
    results_path = args.output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    summary = build_summary(records, manifest)
    write_json(args.output_dir / "summary.json", summary)
    update_attempt_hashes(manifest, records)
    write_json(manifest_path, manifest)
    print(f"[report] wrote {results_path} and {args.output_dir / 'summary.json'}")
    return int(summary["outcomes"]["infrastructure_failures"])


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "generate", "grade", "report", "run"))
    parser.add_argument("--output-dir", type=Path, required=True, help="caller-owned run directory")
    parser.add_argument("--exercise", action="append", help="stable exercise ID (repeatable; defaults to all gradeable exercises)")
    parser.add_argument("--generator-command", help="argument-safe command template containing {input} and {output}")
    parser.add_argument("--generator-id", help="provider-neutral generator identity recorded in the manifest")
    parser.add_argument("--generator-timeout", type=float, default=300.0)
    parser.add_argument("--generation-parameters", help="JSON object or path to a JSON object")
    parser.add_argument("--fixture-adapter", choices=FIXTURE_NAMES)
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--attempts-dir", type=Path, help="grade attempts from this NNN/answer.ext tree")
    parser.add_argument("--registry", type=Path, help="override the executable grader registry")
    parser.add_argument("--manifests-dir", type=Path, help="override per-exercise metadata manifests")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    training_root = Path(__file__).resolve().parents[1]
    repo_root = training_root.parent
    registry = (args.registry or training_root / "autograder" / "exercises.json").resolve()
    manifests_dir = (args.manifests_dir or training_root / "autograder" / "manifests").resolve()
    schema_path = training_root / "autograder" / "schema" / "exercise.schema.json"
    grader = training_root / "tools" / "grade-exercises.py"
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.generator_command and args.fixture_adapter:
            raise ValueError("choose either --generator-command or --fixture-adapter")
        registry_schema, catalog = load_catalog(repo_root, registry, manifests_dir)
        manifest_path = args.output_dir / "reproducibility-manifest.json"
        selected = args.exercise
        if selected is None and args.mode in {"generate", "grade", "report"} and manifest_path.is_file():
            selected = read_json(manifest_path).get("exercise_ids")
        entries = select_exercises(catalog, selected)
        parameters = parse_generation_parameters(args.generation_parameters)
        if args.mode in {"prepare", "run"} or not manifest_path.exists():
            write_json(
                manifest_path,
                make_manifest(args, repo_root, registry_schema, registry, schema_path, grader, entries, parameters),
            )

        infrastructure_failures = 0
        if args.mode in {"prepare", "run"}:
            prepare(args, repo_root, entries, parameters)
        if args.mode in {"generate", "run"}:
            if not all((args.output_dir / "prepared" / entry["id"] / "input.json").is_file() for entry in entries):
                prepare(args, repo_root, entries, parameters)
            infrastructure_failures += generate(args, repo_root, entries)
            manifest = read_json(manifest_path)
            manifest["attempt_hashes"] = {
                entry["id"]: {
                    "path": str(answer_path(args.output_dir / "attempts", entry)),
                    "sha256": sha256(answer_path(args.output_dir / "attempts", entry))
                    if answer_path(args.output_dir / "attempts", entry).is_file() else None,
                    "generation_status": (
                        read_json(args.output_dir / "attempts" / entry["id"] / "generator-output.json").get("status")
                        if (args.output_dir / "attempts" / entry["id"] / "generator-output.json").is_file() else "missing"
                    ),
                }
                for entry in entries
            }
            manifest["updated_at"] = utc_now()
            write_json(manifest_path, manifest)
        if args.mode in {"grade", "run"}:
            infrastructure_failures += grade(args, repo_root, registry, grader, entries)
            infrastructure_failures += report(args, entries)
        elif args.mode == "report":
            infrastructure_failures += report(args, entries)
        return 2 if infrastructure_failures else 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"run-eval: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
