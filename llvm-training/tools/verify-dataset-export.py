#!/usr/bin/env python3
"""Regenerate and verify the deterministic LLVM training dataset export."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def normalized_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(normalized_text(path))


def type_matches(value: Any, expected: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "null": value is None,
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
    }[expected]


def validate_schema_instance(
    value: Any, schema: dict[str, Any], location: str, errors: list[str]
) -> None:
    if "$ref" in schema:
        errors.append(f"{location}: unsupported $ref in repository-local validator")
        return
    expected_type = schema.get("type")
    if expected_type is not None:
        choices = expected_type if isinstance(expected_type, list) else [expected_type]
        if not any(type_matches(value, choice) for choice in choices):
            errors.append(f"{location}: expected type {choices}, got {type(value).__name__}")
            return
    if "const" in schema and value != schema["const"]:
        errors.append(f"{location}: expected constant {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{location}: value {value!r} is not in enum")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{location}: string is too short")
        pattern = schema.get("pattern")
        if pattern and re.search(pattern, value) is None:
            errors.append(f"{location}: string does not match {pattern!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{location}: value is below minimum")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(f"{location}: value is not above exclusive minimum")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{location}: array has too few items")
        if schema.get("uniqueItems"):
            serialized = [json.dumps(item, sort_keys=True) for item in value]
            if len(serialized) != len(set(serialized)):
                errors.append(f"{location}: array items are not unique")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                validate_schema_instance(item, item_schema, f"{location}[{index}]", errors)
    if isinstance(value, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                errors.append(f"{location}: missing required property {key!r}")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                validate_schema_instance(item, properties[key], f"{location}.{key}", errors)
            elif additional is False:
                errors.append(f"{location}: unexpected property {key!r}")
            elif isinstance(additional, dict):
                validate_schema_instance(item, additional, f"{location}.{key}", errors)


def read_jsonl(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(normalized_text(path).splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"{path}:{line_number}: invalid JSON: {error}")
            continue
        if not isinstance(value, dict):
            errors.append(f"{path}:{line_number}: record must be an object")
            continue
        records.append(value)
    return records


def run_export(exporter: Path, output: Path, split: str, include: bool) -> None:
    command = [sys.executable, str(exporter), "--output", str(output), "--split", split]
    command.append("--include-solutions" if include else "--without-solutions")
    subprocess.run(command, check=True)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    exporter = repo_root / "llvm-training/tools/export-exercise-dataset.py"
    schema_path = repo_root / "llvm-training/dataset/schema-v1.json"
    split_path = repo_root / "llvm-training/dataset/splits-v1.json"
    errors: list[str] = []

    try:
        schema = load_json(schema_path)
        split_manifest = load_json(split_path)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        print(f"dataset export verification failed: {error}", file=sys.stderr)
        return 1
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        errors.append("dataset schema must declare JSON Schema draft 2020-12")
    if schema.get("type") != "object" or not isinstance(schema.get("properties"), dict):
        errors.append("dataset schema must describe an object with properties")

    assignments = {item["id"]: item for item in split_manifest.get("exercises", [])}
    if len(assignments) != len(split_manifest.get("exercises", [])):
        errors.append("split manifest contains duplicate exercise IDs")
    manifest_checksum = sha256_text(normalized_text(split_path))

    group_splits: dict[str, set[str]] = {}
    for item in assignments.values():
        group_splits.setdefault(item["leakage_group"], set()).add(item["split"])
    for group, splits in group_splits.items():
        if len(splits) != 1:
            errors.append(f"leakage group {group!r} spans splits: {sorted(splits)}")
    group_entries = split_manifest.get("leakage_groups", [])
    declared_groups = {group["id"]: group for group in group_entries}
    if len(declared_groups) != len(group_entries):
        errors.append("split manifest contains duplicate leakage group IDs")
    for group_id, splits in group_splits.items():
        group = declared_groups.get(group_id)
        if group is None:
            errors.append(f"assignment references undeclared leakage group {group_id!r}")
            continue
        if group.get("split") not in splits:
            errors.append(f"leakage group {group_id!r} split disagrees with assignments")
        assigned_ids = {
            stable_id
            for stable_id, item in assignments.items()
            if item.get("leakage_group") == group_id
        }
        if set(group.get("exercise_ids", [])) != assigned_ids:
            errors.append(f"leakage group {group_id!r} exercise list disagrees with assignments")
        assigned_families = {
            assignments[stable_id].get("concept_family") for stable_id in assigned_ids
        }
        if assigned_families != {group.get("concept_family")}:
            errors.append(f"leakage group {group_id!r} concept family disagrees with assignments")
    for group_id in sorted(set(declared_groups) - set(group_splits)):
        errors.append(f"declared leakage group {group_id!r} has no exercise assignments")

    try:
        with tempfile.TemporaryDirectory(prefix="llvm-training-dataset-") as temporary:
            temp = Path(temporary)
            without_a = temp / "all-without-a.jsonl"
            without_b = temp / "all-without-b.jsonl"
            with_solutions = temp / "all-with.jsonl"
            run_export(exporter, without_a, "all", False)
            run_export(exporter, without_b, "all", False)
            run_export(exporter, with_solutions, "all", True)
            if without_a.read_bytes() != without_b.read_bytes():
                errors.append("repeated exports are not byte-for-byte deterministic")

            records = read_jsonl(without_a, errors)
            included_records = read_jsonl(with_solutions, errors)
            ids = [record.get("id") for record in records]
            if ids != sorted(ids):
                errors.append("records are not ordered by stable exercise ID")
            if len(ids) != len(set(ids)):
                errors.append("export contains duplicate stable exercise IDs")
            if set(ids) != set(assignments):
                errors.append("export IDs do not exactly match the split manifest")
            included_by_id = {record.get("id"): record for record in included_records}

            split_union: set[str] = set()
            for split in ("train", "validation", "test"):
                output = temp / f"{split}.jsonl"
                run_export(exporter, output, split, False)
                split_records = read_jsonl(output, errors)
                actual = {record.get("id") for record in split_records}
                expected = {
                    stable_id for stable_id, item in assignments.items() if item["split"] == split
                }
                if actual != expected:
                    errors.append(f"{split} export does not match split manifest")
                if split_union & actual:
                    errors.append(f"{split} export overlaps another split")
                split_union |= actual

            for index, record in enumerate(records, 1):
                stable_id = record.get("id", f"record-{index}")
                validate_schema_instance(record, schema, stable_id, errors)
                assignment = assignments.get(stable_id)
                if assignment and any(
                    record.get(key) != assignment.get(key)
                    for key in ("split", "leakage_group", "concept_family")
                ):
                    errors.append(f"{stable_id}: split metadata differs from manifest")
                prompt = record.get("prompt")
                checksums = record.get("checksums", {})
                if isinstance(prompt, str) and checksums.get("prompt") != sha256_text(prompt):
                    errors.append(f"{stable_id}: prompt checksum mismatch")
                if checksums.get("split_manifest") != manifest_checksum:
                    errors.append(f"{stable_id}: split manifest checksum mismatch")
                if record.get("reference_solution", {}).get("content") is not None:
                    errors.append(f"{stable_id}: --without-solutions leaked solution content")
                included = included_by_id.get(stable_id, {})
                solution = included.get("reference_solution", {}).get("content")
                if not isinstance(solution, str) or not solution:
                    errors.append(f"{stable_id}: --include-solutions omitted reference content")
                elif checksums.get("solution") != sha256_text(solution):
                    errors.append(f"{stable_id}: solution checksum mismatch")
                for path in record.get("source_files", []):
                    if not (repo_root / path).is_file():
                        errors.append(f"{stable_id}: missing source reference {path}")
                provenance_manifest = record.get("provenance", {}).get("exercise_manifest")
                if (
                    not isinstance(provenance_manifest, str)
                    or not (repo_root / provenance_manifest).is_file()
                ):
                    errors.append(f"{stable_id}: missing provenance manifest reference")
    except (OSError, subprocess.CalledProcessError) as error:
        errors.append(f"could not regenerate exports: {error}")

    if errors:
        print("dataset export verification failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(
        f"Verified {len(assignments)} deterministic dataset records across train, validation, and test splits."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
