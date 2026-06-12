#!/usr/bin/env python3
"""Export graded exercises as a deterministic JSON Lines evaluation dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0.0"
STABLE_ID_RE = re.compile(r"^[0-9]{3}$")
COMMAND_HEADINGS = {"Verification command", "Expected verification command", "Pass command"}


def normalized_text(path: Path) -> str:
    """Decode UTF-8 and normalize line endings without changing other whitespace."""
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def model_visible_prompt(text: str) -> str:
    """Remove reference-solution paths from held-out/model-visible prompt text."""
    return re.sub(
        r"(?:llvm-training/exercises/)?[^\s`\"']*\.solution\.(ll|mlir|md)",
        lambda match: f"answer.{match.group(1)}",
        text,
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(normalized_text(path))


def markdown_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    heading: str | None = None
    for line in text.splitlines(keepends=True):
        match = re.match(r"^##\s+(.+?)\s*\n?$", line)
        if match:
            heading = match.group(1)
            sections.setdefault(heading, [])
        elif heading is not None:
            sections[heading].append(line)
    return {name: "".join(lines) for name, lines in sections.items()}


def expected_observation(prompt: str, stable_id: str) -> str:
    sections = markdown_sections(prompt)
    for name in ("Expected observation", "Expected behavior", "Expected diagnostic observation"):
        content = sections.get(name)
        if content and content.strip():
            return content
    raise ValueError(f"{stable_id}: no reference solution file or expected-observation section")


def verification_description(prompt: str, manifest: dict[str, Any]) -> str:
    sections = markdown_sections(prompt)
    selected = [f"## {name}\n{body}" for name, body in sections.items() if name in COMMAND_HEADINGS]
    if selected:
        return "\n".join(selected)
    check_types = ", ".join(check["type"] for check in manifest["checks"])
    return f"Apply the manifest's machine-readable checks: {check_types}."


def role_for(path: str, manifest: dict[str, Any]) -> str:
    if path == manifest["prompt"]:
        return "prompt"
    if manifest.get("solution") and path == manifest["solution"]:
        return "reference_solution"
    if ".invalid." in path:
        return "broken_input"
    if ".candidate." in path:
        return "candidate"
    if ".input." in path:
        return "input"
    return "grading_reference"


def source_paths(manifest: dict[str, Any]) -> list[str]:
    paths = {manifest["prompt"]}
    if manifest.get("solution"):
        paths.add(manifest["solution"])
    for check in manifest["checks"]:
        path = check.get("path")
        if isinstance(path, str):
            paths.add(path)
    return sorted(paths)


def load_split_manifest(path: Path) -> tuple[dict[str, dict[str, str]], str]:
    text = normalized_text(path)
    data = json.loads(text)
    assignments: dict[str, dict[str, str]] = {}
    for item in data.get("exercises", []):
        stable_id = item.get("id")
        if stable_id in assignments:
            raise ValueError(f"duplicate split assignment for {stable_id}")
        assignments[stable_id] = item
    return assignments, sha256_text(text)


def build_record(
    repo_root: Path,
    manifest_path: Path,
    split_item: dict[str, str],
    split_manifest_checksum: str,
    include_solutions: bool,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    numeric_id = manifest.get("id")
    if not isinstance(numeric_id, str) or not STABLE_ID_RE.fullmatch(numeric_id):
        raise ValueError(f"{manifest_path}: invalid exercise id {numeric_id!r}")
    stable_id = f"llvm-training-exercise-{numeric_id}"
    if split_item.get("id") != stable_id:
        raise ValueError(f"{stable_id}: split manifest ID mismatch")

    paths = source_paths(manifest)
    for relative in paths:
        if not (repo_root / relative).is_file():
            raise ValueError(f"{stable_id}: missing referenced source file: {relative}")

    source_prompt = normalized_text(repo_root / manifest["prompt"])
    prompt = model_visible_prompt(source_prompt)
    solution_path = manifest.get("solution")
    if solution_path:
        solution_kind = "file"
        solution_text = normalized_text(repo_root / solution_path)
    else:
        solution_kind = "prompt_expected_observation"
        solution_text = expected_observation(source_prompt, stable_id)

    required_tools = list(manifest["required_tools"])
    llvm_assumption = "LLVM >= 15 with opaque pointers"
    uses_mlir = (
        manifest["answer_kind"] == "mlir_review"
        or "mlir-opt" in required_tools
        or any(path.endswith(".mlir") for path in paths)
    )
    mlir_assumption = "MLIR >= 15" if uses_mlir else None

    artifacts = [
        {"path": path, "role": role_for(path, manifest), "required": True}
        for path in paths
    ]
    if not solution_path:
        artifacts.append({
            "path": manifest["prompt"],
            "role": "grading_reference",
            "required": True,
        })
    artifacts = sorted(
        {json.dumps(item, sort_keys=True): item for item in artifacts}.values(),
        key=lambda item: (item["path"], item["role"]),
    )

    relative_manifest = manifest_path.relative_to(repo_root).as_posix()
    return {
        "schema_version": SCHEMA_VERSION,
        "id": stable_id,
        "title": manifest["title"],
        "prompt": prompt,
        "answer_kind": manifest["answer_kind"],
        "reference_solution": {
            "included": include_solutions,
            "kind": solution_kind,
            "path": solution_path,
            "content": solution_text if include_solutions else None,
        },
        "source_files": paths,
        "verification": {
            "description": verification_description(prompt, manifest),
            "required_tools": required_tools,
            "minimum_tool_versions": manifest["minimum_tool_versions"],
            "tool_absence_policy": manifest["tool_absence_policy"],
            "determinism": manifest["determinism"],
            "timeout_seconds": manifest["timeout_seconds"],
        },
        "rubric": {
            "maximum_score": manifest["score"],
            "checks": manifest["checks"],
        },
        "topic_tags": sorted(set(manifest["tags"])),
        "difficulty": manifest["difficulty"],
        "version_assumptions": {
            "llvm": llvm_assumption,
            "mlir": mlir_assumption,
            "tools": manifest["minimum_tool_versions"],
        },
        "expected_artifacts": artifacts,
        "split": split_item["split"],
        "leakage_group": split_item["leakage_group"],
        "concept_family": split_item["concept_family"],
        "checksums": {
            "algorithm": "sha256",
            "prompt": sha256_text(prompt),
            "solution": sha256_text(solution_text),
            "split_manifest": split_manifest_checksum,
        },
        "license": {"spdx": manifest["license"], "file": "LICENSE"},
        "provenance": {
            "repository": "Binary-Correspondence-Intermediate-Representation",
            "exercise_manifest": relative_manifest,
            "source_lineage": split_item["concept_family"],
            "exporter": "llvm-training/tools/export-exercise-dataset.py",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("-"), help="JSONL output path, or - for stdout")
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="all")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--include-solutions", action="store_true", help="embed reference solution content")
    modes.add_argument("--without-solutions", action="store_true", help="omit reference solution content (default)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    manifests_dir = repo_root / "llvm-training/autograder/manifests"
    split_path = repo_root / "llvm-training/dataset/splits-v1.json"
    include_solutions = bool(args.include_solutions)

    try:
        assignments, split_checksum = load_split_manifest(split_path)
        manifest_paths = sorted(manifests_dir.glob("*.json"))
        records = []
        seen: set[str] = set()
        for manifest_path in manifest_paths:
            numeric_id = load_json(manifest_path).get("id")
            stable_id = f"llvm-training-exercise-{numeric_id}"
            if stable_id in seen:
                raise ValueError(f"duplicate stable ID: {stable_id}")
            seen.add(stable_id)
            split_item = assignments.get(stable_id)
            if split_item is None:
                raise ValueError(f"{stable_id}: missing split assignment")
            record = build_record(
                repo_root, manifest_path, split_item, split_checksum, include_solutions
            )
            if args.split == "all" or record["split"] == args.split:
                records.append(record)
        extra_assignments = sorted(set(assignments) - seen)
        if extra_assignments:
            raise ValueError(f"split manifest has unknown exercise IDs: {', '.join(extra_assignments)}")
        records.sort(key=lambda record: record["id"])
        output = "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records)
        if args.output == Path("-"):
            sys.stdout.write(output)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8", newline="\n")
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, KeyError, TypeError) as error:
        print(f"dataset export failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
