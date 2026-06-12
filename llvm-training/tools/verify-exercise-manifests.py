#!/usr/bin/env python3
"""Validate declarative manifests for the graded llvm-training exercise set."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

CHECK_TYPES = {
    "file_exists", "non_empty", "llvm_assemble", "llvm_verify", "mlir_parse",
    "contains_symbol", "contains_instruction", "contains_metadata", "forbids_pattern",
    "run_and_compare", "expected_diagnostic", "rubric_terms", "custom_script",
}
ANSWER_KINDS = {
    "llvm_ir", "markdown_review", "optimizer_prediction", "mlir_review",
    "diagnostic", "adversarial_analysis",
}
TOOLS = {"FileCheck", "clang", "lli", "llc", "llvm-as", "mlir-opt", "opt", "python3"}
TOOL_FOR_CHECK = {
    "llvm_assemble": "llvm-as",
    "llvm_verify": "opt",
    "mlir_parse": "mlir-opt",
}
REQUIRED_FIELDS = {
    "id", "title", "prompt", "solution", "answer_kind", "required_tools",
    "minimum_tool_versions", "tool_absence_policy", "checks", "score", "tags",
    "difficulty", "license", "determinism", "timeout_seconds",
}
CHECK_FIELDS = {
    "id", "type", "points", "path", "tool", "symbol", "instruction", "metadata",
    "pattern", "command", "expected_stdout", "diagnostic", "terms", "mode",
}
PATH_CHECKS = {
    "file_exists", "non_empty", "llvm_assemble", "llvm_verify", "mlir_parse",
    "contains_symbol", "contains_instruction", "contains_metadata", "forbids_pattern",
    "expected_diagnostic", "rubric_terms",
}
VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){0,2}$")
ID_RE = re.compile(r"^(?:[0-9]{3}|adv-[a-z0-9][a-z0-9-]*)$")
CHECK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
TAG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def load_json(path: Path, errors: list[str]) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path}: cannot load JSON: {exc}")
        return None


def repository_path(repo_root: Path, value: Any, label: str, errors: list[str]) -> Path | None:
    if not isinstance(value, str) or not value.startswith("llvm-training/"):
        errors.append(f"{label}: expected a repository-relative path below llvm-training/")
        return None
    candidate = (repo_root / value).resolve()
    try:
        candidate.relative_to(repo_root.resolve())
    except ValueError:
        errors.append(f"{label}: path escapes the repository: {value!r}")
        return None
    if not candidate.is_file():
        errors.append(f"{label}: referenced file does not exist: {value}")
    return candidate


def require_string(value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value:
        errors.append(f"{label}: expected a non-empty string")


def validate_check(
    check: Any,
    manifest_path: Path,
    index: int,
    repo_root: Path,
    declared_tools: set[str],
    errors: list[str],
) -> float:
    label = f"{manifest_path}: checks[{index}]"
    if not isinstance(check, dict):
        errors.append(f"{label}: expected an object")
        return 0.0
    unknown = set(check) - CHECK_FIELDS
    if unknown:
        errors.append(f"{label}: unknown fields: {', '.join(sorted(unknown))}")
    missing = {"id", "type", "points"} - set(check)
    if missing:
        errors.append(f"{label}: missing fields: {', '.join(sorted(missing))}")
        return 0.0
    check_id = check.get("id")
    if not isinstance(check_id, str) or not CHECK_ID_RE.fullmatch(check_id):
        errors.append(f"{label}.id: invalid check ID")
    check_type = check.get("type")
    if check_type not in CHECK_TYPES:
        errors.append(f"{label}.type: unsupported check type {check_type!r}")
    points = check.get("points")
    if isinstance(points, bool) or not isinstance(points, (int, float)) or points < 0 or not math.isfinite(points):
        errors.append(f"{label}.points: expected a finite non-negative number")
        points = 0.0

    if check_type in PATH_CHECKS:
        repository_path(repo_root, check.get("path"), f"{label}.path", errors)
    required_by_type = {
        "contains_symbol": ("symbol",),
        "contains_instruction": ("instruction",),
        "contains_metadata": ("metadata",),
        "forbids_pattern": ("pattern",),
        "run_and_compare": ("tool", "command", "expected_stdout"),
        "expected_diagnostic": ("tool", "diagnostic"),
        "rubric_terms": ("terms", "mode"),
        "custom_script": ("tool", "command"),
    }
    for field in required_by_type.get(check_type, ()):
        if field not in check:
            errors.append(f"{label}: {check_type} requires {field!r}")

    for field in ("symbol", "instruction", "metadata", "pattern", "diagnostic"):
        if field in check:
            require_string(check[field], f"{label}.{field}", errors)
    if "command" in check and (
        not isinstance(check["command"], list)
        or not check["command"]
        or not all(isinstance(part, str) for part in check["command"])
    ):
        errors.append(f"{label}.command: expected a non-empty string array")
    if "expected_stdout" in check and not isinstance(check["expected_stdout"], str):
        errors.append(f"{label}.expected_stdout: expected a string")
    if "terms" in check and (
        not isinstance(check["terms"], list)
        or not check["terms"]
        or not all(isinstance(term, str) and term for term in check["terms"])
        or len(check["terms"]) != len(set(check["terms"]))
    ):
        errors.append(f"{label}.terms: expected a non-empty array of unique strings")
    if "mode" in check and check["mode"] not in {"all", "any"}:
        errors.append(f"{label}.mode: expected 'all' or 'any'")

    implicit_tool = TOOL_FOR_CHECK.get(check_type)
    explicit_tool = check.get("tool")
    for tool in [implicit_tool, explicit_tool]:
        if tool is not None and tool not in declared_tools:
            errors.append(f"{label}: tool {tool!r} is used but absent from required_tools")
    if explicit_tool is not None and explicit_tool not in TOOLS:
        errors.append(f"{label}.tool: unsupported tool {explicit_tool!r}")
    return float(points)


def validate_manifest(path: Path, data: Any, repo_root: Path, errors: list[str]) -> str | None:
    if not isinstance(data, dict):
        errors.append(f"{path}: manifest root must be an object")
        return None
    unknown = set(data) - (REQUIRED_FIELDS | {"mlir_validation"})
    missing = REQUIRED_FIELDS - set(data)
    if unknown:
        errors.append(f"{path}: unknown fields: {', '.join(sorted(unknown))}")
    if missing:
        errors.append(f"{path}: missing fields: {', '.join(sorted(missing))}")
        return None

    manifest_id = data.get("id")
    if not isinstance(manifest_id, str) or not ID_RE.fullmatch(manifest_id):
        errors.append(f"{path}: invalid exercise id {manifest_id!r}")
        manifest_id = None
    require_string(data.get("title"), f"{path}: title", errors)
    repository_path(repo_root, data.get("prompt"), f"{path}: prompt", errors)
    if data.get("solution") is not None:
        repository_path(repo_root, data.get("solution"), f"{path}: solution", errors)
    if data.get("answer_kind") not in ANSWER_KINDS:
        errors.append(f"{path}: unsupported answer_kind {data.get('answer_kind')!r}")

    mlir_validation = data.get("mlir_validation")
    if data.get("answer_kind") == "mlir_review":
        if not isinstance(mlir_validation, dict):
            errors.append(f"{path}: mlir_review exercises require mlir_validation tier metadata")
        else:
            allowed = {"expected_tier", "classification", "registry_sources"}
            unknown_mlir = set(mlir_validation) - allowed
            if unknown_mlir:
                errors.append(f"{path}: unknown mlir_validation fields: {', '.join(sorted(unknown_mlir))}")
            tier = mlir_validation.get("expected_tier")
            classification = mlir_validation.get("classification")
            expected_class = {
                0: "file-and-rubric-review",
                1: "unregistered-dialect-sketch",
                2: "registered-dialect",
                3: "conversion-pipeline",
                4: "translated-llvm-ir",
            }.get(tier)
            if expected_class is None or classification != expected_class:
                errors.append(f"{path}: mlir_validation classification does not match expected_tier")
            sources = mlir_validation.get("registry_sources")
            if not isinstance(sources, list) or not all(isinstance(source, str) for source in sources):
                errors.append(f"{path}: mlir_validation.registry_sources must be an array of paths")
            else:
                for source in sources:
                    repository_path(repo_root, source, f"{path}: mlir_validation.registry_sources", errors)
    elif mlir_validation is not None:
        errors.append(f"{path}: mlir_validation is only valid for mlir_review exercises")

    tools = data.get("required_tools")
    if not isinstance(tools, list) or not all(isinstance(tool, str) and tool in TOOLS for tool in tools):
        errors.append(f"{path}: required_tools must contain only declared tool names")
        tools = []
    elif len(tools) != len(set(tools)):
        errors.append(f"{path}: required_tools contains duplicates")
    declared_tools = set(tools)

    versions = data.get("minimum_tool_versions")
    if not isinstance(versions, dict):
        errors.append(f"{path}: minimum_tool_versions must be an object")
        versions = {}
    if set(versions) != declared_tools:
        missing_versions = declared_tools - set(versions)
        extra_versions = set(versions) - declared_tools
        if missing_versions:
            errors.append(f"{path}: missing minimum versions for: {', '.join(sorted(missing_versions))}")
        if extra_versions:
            errors.append(f"{path}: versions declared for unused tools: {', '.join(sorted(extra_versions))}")
    for tool, version in versions.items():
        if tool not in TOOLS or not isinstance(version, str) or not VERSION_RE.fullmatch(version):
            errors.append(f"{path}: invalid minimum version for {tool!r}: {version!r}")

    if data.get("tool_absence_policy") not in {"hard_failure", "unscored_skip", "reduced_confidence_score"}:
        errors.append(f"{path}: invalid tool_absence_policy")
    if data.get("difficulty") not in {"beginner", "intermediate", "advanced"}:
        errors.append(f"{path}: invalid difficulty")
    if data.get("license") != "Apache-2.0":
        errors.append(f"{path}: license must be Apache-2.0")
    if data.get("determinism") not in {"deterministic", "toolchain_dependent", "review_rubric"}:
        errors.append(f"{path}: invalid determinism")
    timeout = data.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 300:
        errors.append(f"{path}: timeout_seconds must be an integer from 1 through 300")
    tags = data.get("tags")
    if not isinstance(tags, list) or not tags or not all(isinstance(tag, str) and TAG_RE.fullmatch(tag) for tag in tags):
        errors.append(f"{path}: tags must be a non-empty array of lowercase slug strings")
    elif len(tags) != len(set(tags)):
        errors.append(f"{path}: tags contains duplicates")

    checks = data.get("checks")
    if not isinstance(checks, list) or not checks:
        errors.append(f"{path}: checks must be a non-empty array")
        checks = []
    check_ids: set[str] = set()
    points = 0.0
    for index, check in enumerate(checks):
        points += validate_check(check, path, index, repo_root, declared_tools, errors)
        if isinstance(check, dict) and isinstance(check.get("id"), str):
            if check["id"] in check_ids:
                errors.append(f"{path}: duplicate check id {check['id']!r}")
            check_ids.add(check["id"])
    score = data.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or score <= 0 or not math.isfinite(score):
        errors.append(f"{path}: score must be a finite positive number")
    elif not math.isclose(points, float(score), abs_tol=1e-9):
        errors.append(f"{path}: check points total {points:g}, expected score {float(score):g}")
    return manifest_id


def validate_schema(schema_path: Path, data: Any, errors: list[str]) -> None:
    if not isinstance(data, dict):
        errors.append(f"{schema_path}: schema root must be an object")
        return
    if data.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        errors.append(f"{schema_path}: expected JSON Schema draft 2020-12")
    check_enum = set(
        data.get("$defs", {}).get("check", {}).get("properties", {}).get("type", {}).get("enum", [])
    )
    if check_enum != CHECK_TYPES:
        errors.append(f"{schema_path}: check type enum is out of sync with the validator")
    required = set(data.get("required", []))
    if required != REQUIRED_FIELDS:
        errors.append(f"{schema_path}: required field list is out of sync with the validator")


def main() -> int:
    script = Path(__file__).resolve()
    repo_root = script.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, default=repo_root / "llvm-training/autograder/schema/exercise.schema.json")
    parser.add_argument("--manifests", type=Path, default=repo_root / "llvm-training/autograder/manifests")
    args = parser.parse_args()

    errors: list[str] = []
    schema = load_json(args.schema, errors)
    if schema is not None:
        validate_schema(args.schema, schema, errors)
    manifest_paths = sorted(args.manifests.glob("*.json")) if args.manifests.is_dir() else []
    if not manifest_paths:
        errors.append(f"{args.manifests}: no exercise manifests found")

    ids: dict[str, Path] = {}
    prompts: dict[str, Path] = {}
    for path in manifest_paths:
        data = load_json(path, errors)
        if data is None:
            continue
        manifest_id = validate_manifest(path, data, repo_root, errors)
        if manifest_id:
            if manifest_id in ids:
                errors.append(f"duplicate exercise id {manifest_id!r}: {ids[manifest_id]} and {path}")
            ids[manifest_id] = path
        if isinstance(data, dict) and isinstance(data.get("prompt"), str):
            prompt = data["prompt"]
            if prompt in prompts:
                errors.append(f"prompt has multiple manifests: {prompt} ({prompts[prompt]} and {path})")
            prompts[prompt] = path

    graded_prompts = {
        path.relative_to(repo_root).as_posix()
        for path in (repo_root / "llvm-training/exercises").glob("[0-9][0-9][0-9]-*.prompt.md")
    }
    missing = sorted(graded_prompts - set(prompts))
    extra = sorted(set(prompts) - graded_prompts)
    for prompt in missing:
        errors.append(f"graded exercise is missing a manifest: {prompt}")
    for prompt in extra:
        errors.append(f"manifest prompt is not a numbered graded exercise: {prompt}")

    if errors:
        print("exercise manifest validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"validated {len(manifest_paths)} exercise manifests against {args.schema.relative_to(repo_root)}")
    print(f"covered {len(graded_prompts)} numbered graded exercise prompts with unique IDs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
