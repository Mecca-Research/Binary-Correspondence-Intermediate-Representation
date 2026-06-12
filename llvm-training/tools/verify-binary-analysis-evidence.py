#!/usr/bin/env python3
"""Validate Chapter 15 evidence-manifest coverage and provenance contracts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[2]
CHAPTER = REPO_ROOT / "llvm-training/15-binary-analysis"
MANIFEST_PATH = CHAPTER / "evidence-manifest.json"
CLASSIFICATIONS = {"deterministic", "host-sensitive", "schematic"}
REQUIRED_STATIC_KINDS = {
    "symbol", "instruction-class-count", "basic-block-count", "call-edge", "object-section"
}
GENERATED_HEADER = [
    "fixture", "evidence_kind", "scope", "name", "value", "unit", "classification"
]


def relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def repo_file(value: Any, label: str, errors: list[str], nullable: bool = False) -> Path | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value.startswith("llvm-training/"):
        errors.append(f"{label}: expected a repository-relative llvm-training/ path")
        return None
    path = (REPO_ROOT / value).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError:
        errors.append(f"{label}: path escapes repository: {value}")
        return None
    if not path.is_file():
        errors.append(f"{label}: file does not exist: {value}")
    return path


def nonempty_string_list(value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        errors.append(f"{label}: expected a non-empty string array")


def validate_generated_csv(path: Path, entry: dict[str, Any], errors: list[str]) -> set[str]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        errors.append(f"{relative(path)}: cannot read CSV: {exc}")
        return set()
    if reader.fieldnames != GENERATED_HEADER:
        errors.append(f"{relative(path)}: generated header must be {','.join(GENERATED_HEADER)}")
        return set()
    if not rows:
        errors.append(f"{relative(path)}: generated CSV has no evidence rows")
        return set()
    kinds: set[str] = set()
    for line, row in enumerate(rows, 2):
        if row["fixture"] != entry["id"]:
            errors.append(f"{relative(path)}:{line}: fixture id does not match manifest")
        if row["classification"] != "deterministic":
            errors.append(f"{relative(path)}:{line}: generated row must be deterministic")
        kinds.add(row["evidence_kind"])
        try:
            if int(row["value"]) < 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"{relative(path)}:{line}: value must be a non-negative integer")
    return kinds


def validate_provenance(path: Path, entry: dict[str, Any], errors: list[str]) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{relative(path)}: cannot load provenance JSON: {exc}")
        return
    expected = {
        "fixture_id": entry["id"],
        "classification": "deterministic",
        "source_fixture": entry["source_fixture"],
        "checked_in_csv": entry["checked_in_csv"],
    }
    for key, value in expected.items():
        if data.get(key) != value:
            errors.append(f"{relative(path)}: {key} does not match manifest")
    target = data.get("target")
    if not isinstance(target, dict) or target.get("triple") != entry["target_triple"]:
        errors.append(f"{relative(path)}: target metadata does not match manifest")
    toolchain = data.get("toolchain")
    if not isinstance(toolchain, dict) or not all(isinstance(toolchain.get(key), str) and toolchain[key] for key in ("clang", "python")):
        errors.append(f"{relative(path)}: toolchain must record non-empty clang and python versions")


def main() -> int:
    errors: list[str] = []
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot load {relative(MANIFEST_PATH)}: {exc}", file=sys.stderr)
        return 1
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("evidence"), list):
        print("error: evidence manifest must have schema_version 1 and an evidence array", file=sys.stderr)
        return 1

    entries_by_csv: dict[str, dict[str, Any]] = {}
    ids: set[str] = set()
    generated_kinds: set[str] = set()
    for index, raw_entry in enumerate(manifest["evidence"]):
        label = f"{relative(MANIFEST_PATH)}:evidence[{index}]"
        if not isinstance(raw_entry, dict):
            errors.append(f"{label}: expected an object")
            continue
        entry = raw_entry
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id:
            errors.append(f"{label}.id: expected a non-empty string")
        elif entry_id in ids:
            errors.append(f"{label}.id: duplicate id {entry_id}")
        else:
            ids.add(entry_id)
        classification = entry.get("classification")
        if classification not in CLASSIFICATIONS:
            errors.append(f"{label}.classification: expected one of {sorted(CLASSIFICATIONS)}")
        csv_path = repo_file(entry.get("checked_in_csv"), f"{label}.checked_in_csv", errors)
        if isinstance(entry.get("checked_in_csv"), str):
            if entry["checked_in_csv"] in entries_by_csv:
                errors.append(f"{label}.checked_in_csv: duplicate manifest coverage")
            entries_by_csv[entry["checked_in_csv"]] = entry

        if classification == "deterministic":
            source = repo_file(entry.get("source_fixture"), f"{label}.source_fixture", errors)
            provenance = repo_file(entry.get("provenance_path"), f"{label}.provenance_path", errors)
            nonempty_string_list(entry.get("build_command"), f"{label}.build_command", errors)
            nonempty_string_list(entry.get("optimization_flags"), f"{label}.optimization_flags", errors)
            nonempty_string_list(entry.get("collection_command"), f"{label}.collection_command", errors)
            for field in ("target_triple", "expected_artifact_type"):
                if not isinstance(entry.get(field), str) or not entry[field]:
                    errors.append(f"{label}.{field}: expected a non-empty string")
            if source is not None and source.suffix != ".s":
                errors.append(f"{label}.source_fixture: deterministic fixture must be assembly source")
            if csv_path is not None and csv_path.is_file():
                generated_kinds.update(validate_generated_csv(csv_path, entry, errors))
            if provenance is not None and provenance.is_file():
                validate_provenance(provenance, entry, errors)
        else:
            if entry.get("provenance_path") is not None:
                errors.append(f"{label}.provenance_path: non-generated evidence must use null")

    missing_kinds = REQUIRED_STATIC_KINDS - generated_kinds
    if missing_kinds:
        errors.append(
            "deterministic fixture set is missing static evidence kinds: "
            + ", ".join(sorted(missing_kinds))
        )

    discovered = {
        relative(path) for path in CHAPTER.rglob("*.csv")
        if path.is_file()
    }
    covered = set(entries_by_csv)
    for path in sorted(discovered - covered):
        errors.append(f"{path}: checked-in CSV has no evidence-manifest entry")
    for path in sorted(covered - discovered):
        errors.append(f"{path}: manifest CSV path is not a checked-in Chapter 15 CSV")

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        print(f"Binary-analysis evidence validation failed with {len(errors)} error(s).", file=sys.stderr)
        return 1
    print(f"Validated {len(discovered)} Chapter 15 CSV(s) across {len(ids)} provenance classifications.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
