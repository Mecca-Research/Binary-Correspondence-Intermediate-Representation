#!/usr/bin/env python3
"""Export graded exercises as a deterministic JSON Lines evaluation dataset.

What makes an evaluation dataset trustworthy is not its schema, it is that four
things hold at once, and none of them is about the subject being taught:

  * **A prompt a model reads never names its own reference solution.** The
    redaction is the subject's pattern, applied here so no exporter can forget
    it.
  * **Every record declares its split, and the split comes from a manifest that
    is itself checksummed** into the record. A leakage group that moved after
    export is then visible as a digest change rather than invisible.
  * **Every referenced file exists**, and an exercise with no split assignment
    -- or a split assignment with no exercise -- is a hard error. Half a dataset
    that exports cleanly is the failure mode worth refusing.
  * **The bytes are a function of the tree.** Sorted keys, sorted records, LF
    endings, no timestamps: two exports of one checkout are identical, so a diff
    means the corpus changed.

The subject supplies its manifests, its split file, its id prefix and its
version assumptions. Everything above is shared, because it would otherwise be
written again, slightly differently, by the second subject to need a dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from subject_profile import SubjectProfile  # noqa: E402

SCHEMA_VERSION = "1.0.0"
STABLE_ID_RE = re.compile(r"^[0-9]{3}$")
COMMAND_HEADINGS = {"Verification command", "Expected verification command", "Pass command"}


def normalized_text(path: Path) -> str:
    """Decode UTF-8 and normalize line endings without changing other whitespace."""
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


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
    profile: SubjectProfile,
    repo_root: Path,
    manifest_path: Path,
    split_item: dict[str, str],
    split_manifest_checksum: str,
    include_solutions: bool,
    version_assumptions: Callable[[dict[str, Any], list[str]], dict[str, Any]],
    exporter: str,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    numeric_id = manifest.get("id")
    if not isinstance(numeric_id, str) or not STABLE_ID_RE.fullmatch(numeric_id):
        raise ValueError(f"{manifest_path}: invalid exercise id {numeric_id!r}")
    stable_id = f"{profile.dataset_id_prefix}{numeric_id}"
    if split_item.get("id") != stable_id:
        raise ValueError(f"{stable_id}: split manifest ID mismatch")

    paths = source_paths(manifest)
    for relative in paths:
        if not (repo_root / relative).is_file():
            raise ValueError(f"{stable_id}: missing referenced source file: {relative}")

    source_prompt = normalized_text(repo_root / manifest["prompt"])
    prompt = profile.model_visible(source_prompt)
    solution_path = manifest.get("solution")
    if solution_path:
        solution_kind = "file"
        solution_text = normalized_text(repo_root / solution_path)
    else:
        solution_kind = "prompt_expected_observation"
        solution_text = expected_observation(source_prompt, stable_id)

    artifacts = [
        {"path": path, "role": role_for(path, manifest), "required": True} for path in paths
    ]
    if not solution_path:
        artifacts.append(
            {
                "path": manifest["prompt"],
                "role": "grading_reference",
                "required": True,
            }
        )
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
            "required_tools": list(manifest["required_tools"]),
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
        "version_assumptions": version_assumptions(manifest, paths),
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
            "exporter": exporter,
        },
    }


def parse_args(description: str, argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output", type=Path, default=Path("-"), help="JSONL output path, or - for stdout"
    )
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="all")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--include-solutions", action="store_true", help="embed reference solution content"
    )
    modes.add_argument(
        "--without-solutions", action="store_true", help="omit reference solution content (default)"
    )
    return parser.parse_args(argv)


def export(
    profile: SubjectProfile,
    *,
    split: str,
    include_solutions: bool,
    version_assumptions: Callable[[dict[str, Any], list[str]], dict[str, Any]],
    exporter: str,
) -> str:
    """Every manifest, bound to its split assignment, as canonical JSON Lines."""
    repo_root = profile.root.parent.parent
    manifests_dir = profile.root / "autograder" / "manifests"
    split_path = profile.root / "dataset" / "splits-v1.json"

    assignments, split_checksum = load_split_manifest(split_path)
    manifest_paths = sorted(manifests_dir.glob("*.json"))
    if not manifest_paths:
        raise ValueError(
            f"no exercise manifest under {manifests_dir}; an empty dataset is a defect"
        )
    records = []
    seen: set[str] = set()
    for manifest_path in manifest_paths:
        numeric_id = load_json(manifest_path).get("id")
        stable_id = f"{profile.dataset_id_prefix}{numeric_id}"
        if stable_id in seen:
            raise ValueError(f"duplicate stable ID: {stable_id}")
        seen.add(stable_id)
        split_item = assignments.get(stable_id)
        if split_item is None:
            raise ValueError(f"{stable_id}: missing split assignment")
        record = build_record(
            profile,
            repo_root,
            manifest_path,
            split_item,
            split_checksum,
            include_solutions,
            version_assumptions,
            exporter,
        )
        if split == "all" or record["split"] == split:
            records.append(record)
    extra_assignments = sorted(set(assignments) - seen)
    if extra_assignments:
        raise ValueError(f"split manifest has unknown exercise IDs: {', '.join(extra_assignments)}")
    records.sort(key=lambda record: record["id"])
    return "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records
    )


def main(
    profile: SubjectProfile,
    description: str,
    *,
    version_assumptions: Callable[[dict[str, Any], list[str]], dict[str, Any]],
    exporter: str,
    argv: list[str] | None = None,
) -> int:
    args = parse_args(description, argv if argv is not None else sys.argv[1:])
    try:
        output = export(
            profile,
            split=args.split,
            include_solutions=bool(args.include_solutions),
            version_assumptions=version_assumptions,
            exporter=exporter,
        )
        if args.output == Path("-"):
            sys.stdout.write(output)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8", newline="\n")
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, KeyError, TypeError) as error:
        print(f"dataset export failed: {error}", file=sys.stderr)
        return 1
    return 0
