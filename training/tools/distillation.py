#!/usr/bin/env python3
"""Shaping a Tier-3 record: the part that is not about any subject.

`build_distillation.py` used to hold both halves -- the rule that a record is a
(system, user, assistant) turn bound to the gate that checks its answer, *and*
five extractors that knew where LLVM keeps its invalid fixtures, its optimizer
goldens and its frontend snapshots. The second half was subject code living in
the shared directory: the same boundary problem the grader had, pointing the
other way.

What stays here is the record contract:

  * **A record exists only because a gate checks its answer**, so `verified_by`
    is a required argument rather than an optional field. There is no way to
    build a record through this module without naming a gate and the claim it
    proves.
  * **Identity is content.** `record_id` digests the subject, task and messages,
    so the same three turns are the same record however they were derived.
  * **The split is assigned by SOURCE FILE, not by record.** Two records drawn
    from one chapter can never straddle a split boundary, which is the same
    one-source-one-split invariant the corpus export keeps at the other tier.
  * **The system prompt belongs to the subject.** It frames what the assistant
    is answering as, and that is a statement about the material, not about
    record-keeping.

A subject supplies `SOURCES` -- named extractors -- and a `SYSTEM_PROMPT`, in
`<subject>/tools/distill_sources.py`. The builder finds that file by convention
and knows nothing else about the subject; a folder without one contributes no
Tier-3 records, which is the honest state of every subject that is still a scope
statement.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
CORPUS_ROOT = TOOLS_DIR.parent
REPO_ROOT = CORPUS_ROOT.parent

SCHEMA = "bcir-training/distill/v1"
LICENSE = "LicenseRef-BCIR-NC-1.0"
BUILDER = "training/tools/build_distillation.py"

# The vocabulary a subject's own manifests already use.
DIFFICULTY_BY_NAME = {"beginner": 1, "intermediate": 3, "advanced": 4, "expert": 5}

# Deterministic split weights. Assignment is by SOURCE FILE, not by record, so
# two records derived from one file can never straddle a split boundary.
SPLIT_BUCKETS = [("train", 80), ("validation", 10), ("test", 10)]

# The conventional name of a subject's record-source module.
SOURCES_MODULE = "distill_sources.py"


def canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assign_split(source_path: str) -> str:
    """Hash the source path into a split. Same file -> same split, always."""
    position = int(digest_text(source_path)[:8], 16) % 100
    cursor = 0
    for name, weight in SPLIT_BUCKETS:
        cursor += weight
        if position < cursor:
            return name
    return SPLIT_BUCKETS[-1][0]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclasses.dataclass(frozen=True)
class RecordSource:
    """One named way a subject turns checked artifacts into records.

    `task` names the shape of the question, and is what the stats report counts.
    `extract` receives the builder and returns records; it is the only place
    that knows where this subject keeps anything.
    """

    task: str
    extract: Callable[["RecordBuilder"], list[dict]]


class RecordBuilder:
    """The handle an extractor gets: paths, reading, and record construction.

    An extractor never imports this module's constants or reaches for the repo
    root itself. Everything it needs arrives here, which is what keeps a
    subject's source file about *its material* rather than about the record
    format.
    """

    def __init__(self, subject_root: Path, system_prompt: str) -> None:
        self.subject_root = subject_root
        self.subject = subject_root.name
        self.repo_root = REPO_ROOT
        if not system_prompt.strip():
            raise SystemExit(
                f"build_distillation: {self.subject} declares an empty SYSTEM_PROMPT; "
                "the system turn frames every answer and cannot be blank"
            )
        self.system_prompt = system_prompt

    # -- paths and reading -------------------------------------------------

    def read(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def relpath(self, path: Path) -> str:
        return path.resolve().relative_to(REPO_ROOT).as_posix()

    def gate(self, name: str) -> str:
        """The repository path of one of this subject's gates."""
        return self.relpath(self.subject_root / "tools" / name)

    def tool(self, name: str) -> Path:
        return self.subject_root / "tools" / name

    def load_module(self, path: Path, name: str):
        return load_module(path, name)

    def difficulty(self, name: str, default: int = 3) -> int:
        return DIFFICULTY_BY_NAME.get(name, default)

    # -- the record --------------------------------------------------------

    def record(
        self,
        *,
        task: str,
        user: str,
        assistant: str,
        answer_kind: str,
        source_paths: list[str],
        gate: str,
        claim: str,
        difficulty: int,
    ) -> dict:
        if not gate.strip() or not claim.strip():
            raise SystemExit(
                f"build_distillation: {self.subject}/{task} produced a record with no "
                "gate or no claim; a record whose answer nothing checks is a "
                "hallucination with provenance attached"
            )
        if not source_paths:
            raise SystemExit(
                f"build_distillation: {self.subject}/{task} produced a record with no "
                "source path, so it can be traced to nothing"
            )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ]
        identity = canonical_json({"subject": self.subject, "task": task, "messages": messages})
        return {
            "schema": SCHEMA,
            "record_id": f"sha256:{digest_text(identity)}",
            "corpus": "training",
            "subject": self.subject,
            "task": task,
            "messages": messages,
            "answer_kind": answer_kind,
            "source_paths": sorted(set(source_paths)),
            "verified_by": {"gate": gate, "claim": claim},
            "split": assign_split(sorted(set(source_paths))[0]),
            "difficulty": difficulty,
            "provenance": {"license": LICENSE, "builder": BUILDER},
        }


def subject_sources(subject_root: Path):
    """A subject's record sources, or None if it declares none.

    Found by convention rather than registration: a folder that is still a scope
    statement has no `tools/distill_sources.py` and contributes nothing, which
    is what it should contribute.
    """
    path = subject_root / "tools" / SOURCES_MODULE
    if not path.is_file():
        return None
    module = load_module(path, f"{subject_root.name}_distill_sources")
    for attribute in ("SOURCES", "SYSTEM_PROMPT"):
        if not hasattr(module, attribute):
            raise SystemExit(
                f"build_distillation: {path} defines no {attribute}; a subject that "
                "declares record sources must declare both"
            )
    if not module.SOURCES:
        raise SystemExit(
            f"build_distillation: {path} declares an empty SOURCES; a subject with no "
            "record source should not have the file at all, rather than a file that "
            "silently produces nothing"
        )
    return module


def build_subject(subject_root: Path) -> tuple[list[dict], dict[str, int]]:
    """Every record this subject's own sources produce, deterministically ordered."""
    module = subject_sources(subject_root)
    if module is None:
        return [], {}
    builder = RecordBuilder(subject_root, module.SYSTEM_PROMPT)
    records: list[dict] = []
    counts: dict[str, int] = {}
    seen: set[str] = set()
    for source in module.SOURCES:
        if source.task in seen:
            raise SystemExit(
                f"build_distillation: {subject_root.name} declares the task "
                f"{source.task!r} twice; the stats would silently merge them"
            )
        seen.add(source.task)
        produced = source.extract(builder)
        counts[source.task] = len(produced)
        records.extend(produced)
    records.sort(key=lambda r: (r["task"], r["source_paths"][0], r["record_id"]))
    return records, counts
