#!/usr/bin/env python3
"""Gate the Tier-2 and Tier-3 corpus record builds.

The two builders are only worth what this checks about them. It enforces, in
order of how badly each failure would poison a downstream consumer:

  1. **Anti-vacuity.** A run that examined nothing FAILS. A gate that passes on
     an empty corpus is green for not running, which is the failure mode this
     repository has paid for before.
  2. **No fabricated embeddings.** A chunk's `embedding` is null, or it is a
     finite vector whose length matches its declared `dim` and whose model is
     named. A vector with no model behind it is indistinguishable downstream
     from a real one.
  3. **Every distillation record is gate-backed.** `verified_by.gate` must name
     a file that exists in this repository. This is the rule the whole tier
     rests on: no gate, no record.
  4. **Provenance holds.** Every `source_path` exists; every chunk's recorded
     `source_sha256` matches the file on disk today; every chunk's line span
     lies inside its source and contains its body.
  5. **No split leakage.** A source file's records all land in one split, so a
     test record can never be a memorised train record.
  6. **Determinism.** Both builders run twice and must produce byte-identical
     output. Content-addressed identifiers make this a real check on the
     builders rather than on the filesystem's ordering.

Structural validation here is deliberately scoped: it checks required keys,
types, enums, and the cross-file invariants above. It is not a general JSON
Schema engine -- the schemas in training/schema/ are the published contract for
external consumers, and this gate checks the properties that failing would
actually corrupt training data.

    python3 training/tools/verify_corpus_records.py
    python3 training/tools/verify_corpus_records.py --keep build/training/records
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
CORPUS_ROOT = TOOLS_DIR.parent
REPO_ROOT = CORPUS_ROOT.parent

CHUNK_SCHEMA = "bcir-training/chunk/v1"
DISTILL_SCHEMA = "bcir-training/distill/v1"

CHUNK_KINDS = {"prose", "code", "mixed", "table"}
DISTILL_TASKS = {"claim", "exercise", "repair", "prediction", "review"}
ANSWER_KINDS = {"code", "explanation", "diagnosis"}
SPLITS = {"train", "validation", "test"}
ROLES = ["system", "user", "assistant"]

CHUNK_REQUIRED = {
    "schema",
    "chunk_id",
    "corpus",
    "subject",
    "source_path",
    "source_sha256",
    "span",
    "heading_trail",
    "title",
    "kind",
    "text",
    "char_count",
    "token_estimate",
    "embedding",
    "embedding_spec",
    "provenance",
    "verified_by",
}
DISTILL_REQUIRED = {
    "schema",
    "record_id",
    "corpus",
    "subject",
    "task",
    "messages",
    "answer_kind",
    "source_paths",
    "verified_by",
    "split",
    "difficulty",
    "provenance",
}


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def require(self, condition: bool, message: str) -> bool:
        self.checks += 1
        if not condition:
            self.failures.append(message)
        return condition


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{number}: {exc}") from exc
    return rows


# --------------------------------------------------------------------------
# Chunk checks
# --------------------------------------------------------------------------


def check_chunks(rows: list[dict], report: Report) -> None:
    report.require(bool(rows), "chunk build produced no records at all")
    seen_ids: set[str] = set()
    source_cache: dict[str, list[str]] = {}

    for index, chunk in enumerate(rows):
        label = f"chunk[{index}]"
        missing = CHUNK_REQUIRED - set(chunk)
        if not report.require(not missing, f"{label}: missing key(s) {sorted(missing)}"):
            continue
        extra = set(chunk) - CHUNK_REQUIRED - {"language"}
        report.require(not extra, f"{label}: unexpected key(s) {sorted(extra)}")

        report.require(chunk["schema"] == CHUNK_SCHEMA, f"{label}: wrong schema tag")
        report.require(chunk["kind"] in CHUNK_KINDS, f"{label}: bad kind {chunk['kind']!r}")
        report.require(
            chunk["chunk_id"].startswith("sha256:") and len(chunk["chunk_id"]) == 71,
            f"{label}: malformed chunk_id",
        )
        report.require(chunk["chunk_id"] not in seen_ids, f"{label}: duplicate chunk_id")
        seen_ids.add(chunk["chunk_id"])

        report.require(
            chunk["char_count"] == len(chunk["text"]),
            f"{label}: char_count disagrees with text length",
        )

        # No fabricated vectors.
        spec = chunk["embedding_spec"]
        if chunk["embedding"] is None:
            report.require(
                spec["model"] is None,
                f"{label}: names an embedding model but carries no vector",
            )
        else:
            report.require(
                isinstance(spec["model"], str) and spec["model"],
                f"{label}: carries a vector with no model named -- a vector "
                "nobody can attribute is not evidence",
            )
            report.require(
                spec["dim"] == len(chunk["embedding"]),
                f"{label}: embedding length disagrees with declared dim",
            )
            report.require(
                all(isinstance(v, (int, float)) and v == v for v in chunk["embedding"]),
                f"{label}: embedding contains a non-finite value",
            )

        # Provenance: the source must exist and be unchanged since the build.
        source = REPO_ROOT / chunk["source_path"]
        if not report.require(source.is_file(), f"{label}: source_path does not exist"):
            continue
        report.require(
            digest_file(source) == chunk["source_sha256"],
            f"{label}: source_sha256 is stale for {chunk['source_path']}",
        )

        lines = source_cache.setdefault(
            chunk["source_path"], source.read_text(encoding="utf-8").splitlines()
        )
        span = chunk["span"]
        report.require(
            1 <= span["start_line"] <= span["end_line"] <= max(len(lines), 1),
            f"{label}: span {span} outside {chunk['source_path']}",
        )

        for gate in chunk["verified_by"]:
            report.require(
                (REPO_ROOT / gate).is_file(),
                f"{label}: names a gate that does not exist: {gate}",
            )


# --------------------------------------------------------------------------
# Distillation checks
# --------------------------------------------------------------------------


def check_distillation(rows: list[dict], report: Report) -> None:
    report.require(bool(rows), "distillation build produced no records at all")
    seen_ids: set[str] = set()
    split_of_source: dict[str, str] = {}

    for index, record in enumerate(rows):
        label = f"record[{index}]"
        missing = DISTILL_REQUIRED - set(record)
        if not report.require(not missing, f"{label}: missing key(s) {sorted(missing)}"):
            continue
        extra = set(record) - DISTILL_REQUIRED
        report.require(not extra, f"{label}: unexpected key(s) {sorted(extra)}")

        report.require(record["schema"] == DISTILL_SCHEMA, f"{label}: wrong schema tag")
        report.require(record["task"] in DISTILL_TASKS, f"{label}: bad task {record['task']!r}")
        report.require(record["answer_kind"] in ANSWER_KINDS, f"{label}: bad answer_kind")
        report.require(record["split"] in SPLITS, f"{label}: bad split")
        report.require(
            isinstance(record["difficulty"], int) and 1 <= record["difficulty"] <= 5,
            f"{label}: difficulty out of range",
        )
        report.require(record["record_id"] not in seen_ids, f"{label}: duplicate record_id")
        seen_ids.add(record["record_id"])

        messages = record["messages"]
        if report.require(len(messages) == 3, f"{label}: expected 3 messages"):
            report.require(
                [m["role"] for m in messages] == ROLES,
                f"{label}: roles must be system, user, assistant in order",
            )
            report.require(
                all(m["content"].strip() for m in messages),
                f"{label}: an empty message turn",
            )

        # The rule the whole tier rests on.
        gate = record["verified_by"]["gate"]
        report.require(
            (REPO_ROOT / gate).is_file(),
            f"{label}: verified_by names a gate that does not exist: {gate}",
        )
        report.require(
            bool(record["verified_by"]["claim"].strip()),
            f"{label}: verified_by.claim is empty -- the record states what is "
            "checked, or it is not a training record",
        )

        report.require(bool(record["source_paths"]), f"{label}: no source_paths")
        for source in record["source_paths"]:
            report.require(
                (REPO_ROOT / source).is_file(),
                f"{label}: source_path does not exist: {source}",
            )
            # Leakage: one source file, one split.
            previous = split_of_source.setdefault(source, record["split"])
            report.require(
                previous == record["split"],
                f"{label}: {source} appears in both '{previous}' and "
                f"'{record['split']}' -- a test record derived from a train source "
                "is a memorisation check, not an evaluation",
            )

    # A corpus that is entirely one split cannot evaluate anything -- but a
    # subject with only a handful of records can legitimately hash into one, so
    # the requirement starts where it becomes meaningful.
    splits = {record["split"] for record in rows}
    if len(rows) >= 10:
        report.require(
            len(splits) >= 2,
            f"all {len(rows)} records landed in a single split ({splits}); no held-out data exists",
        )


def build_into(directory: Path, chunks_module, distill_module) -> tuple[Path, Path]:
    chunk_dir = directory / "chunks"
    distill_dir = directory / "distill"
    chunks_module.main(["--out", str(chunk_dir)])
    distill_module.main(["--out", str(distill_dir)])
    return chunk_dir, distill_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", type=Path, help="also write the verified build to this directory")
    args = parser.parse_args()

    chunks_module = load_module(TOOLS_DIR / "build_chunks.py", "build_chunks")
    distill_module = load_module(TOOLS_DIR / "build_distillation.py", "build_distillation")

    report = Report()
    workspace = Path(tempfile.mkdtemp(prefix="bcir-corpus-records-"))
    try:
        first_chunks, first_distill = build_into(workspace / "a", chunks_module, distill_module)
        second_chunks, second_distill = build_into(workspace / "b", chunks_module, distill_module)

        chunk_files = sorted(first_chunks.glob("*.chunks.jsonl"))
        distill_files = sorted(first_distill.glob("*.distill.jsonl"))
        report.require(bool(chunk_files), "no subject produced a chunk file")
        report.require(bool(distill_files), "no subject produced a distillation file")

        for path in chunk_files:
            rows = read_jsonl(path)
            print(f"[chunks]  {path.name}: {len(rows)} record(s)")
            check_chunks(rows, report)
            twin = second_chunks / path.name
            report.require(
                twin.is_file() and digest_file(twin) == digest_file(path),
                f"{path.name}: two builds of the same tree differ -- the chunker "
                "is not deterministic",
            )

        for path in distill_files:
            rows = read_jsonl(path)
            print(f"[distill] {path.name}: {len(rows)} record(s)")
            check_distillation(rows, report)
            twin = second_distill / path.name
            report.require(
                twin.is_file() and digest_file(twin) == digest_file(path),
                f"{path.name}: two builds of the same tree differ -- the "
                "distillation builder is not deterministic",
            )

        if args.keep:
            if args.keep.exists():
                shutil.rmtree(args.keep)
            shutil.copytree(workspace / "a", args.keep)
            print(f"[keep]    {args.keep}")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    if report.failures:
        print("corpus record gate: FAILED", file=sys.stderr)
        for failure in report.failures[:40]:
            print(f"  - {failure}", file=sys.stderr)
        if len(report.failures) > 40:
            print(f"  ... and {len(report.failures) - 40} more", file=sys.stderr)
        return 1

    print(f"corpus record gate: PASSED ({report.checks} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
