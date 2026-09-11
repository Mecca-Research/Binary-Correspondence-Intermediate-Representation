#!/usr/bin/env python3
"""Build Tier-3 distillation records from the training corpus.

Tier 3 is supervised training data: chat-format (system, user, assistant) turns
a fine-tuning pipeline can consume directly.

One rule decides what may become a record:

    **A record exists only if a gate in this repository checks its answer.**

That is the whole design. A distillation record whose assistant turn nothing
verifies is a hallucination with provenance attached -- and it is *worse* than
no record, because the provenance makes a consumer trust it more. So every
record carries `verified_by: {gate, claim}`, and the verifier rejects any record
that does not.

The consequence is that this builder is small and its output is bounded by how
much of the corpus is actually checked, not by how much of it is written. That
is the intended pressure: to get more training data, gate more claims.

**Where the records come from is the subject's business.** This file walks
subject folders, asks each one for its record sources, writes what comes back,
and reports. It does not know where any subject keeps its fixtures. A subject
that has sources declares them in `<subject>/tools/distill_sources.py`; one that
is still a scope statement has no such file and contributes nothing, which is
the honest answer. `training/tools/distillation.py` owns the record contract
they share.

    python3 training/tools/build_distillation.py --out build/training/distill
    python3 training/tools/build_distillation.py --stats
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
CORPUS_ROOT = TOOLS_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from distillation import build_subject, canonical_json  # noqa: E402

# Not subject folders: the shared machinery and the published contracts.
NON_SUBJECT_DIRS = {"tools", "schema"}


def subject_roots(selected: str | None) -> list[Path]:
    roots = [
        path
        for path in sorted(CORPUS_ROOT.iterdir())
        if path.is_dir() and not path.name.startswith(".") and path.name not in NON_SUBJECT_DIRS
    ]
    if selected:
        roots = [path for path in roots if path.name == selected]
        if not roots:
            raise SystemExit(f"no such subject: {selected}")
    return roots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject")
    parser.add_argument("--out", type=Path, help="directory for <subject>.distill.jsonl")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args(argv)

    total = 0
    for root in subject_roots(args.subject):
        records, counts = build_subject(root)
        total += len(records)

        # A subject folder that is still just a scope statement produces
        # nothing; write no file rather than an empty one for the gate to
        # special-case.
        if args.out and records:
            args.out.mkdir(parents=True, exist_ok=True)
            destination = args.out / f"{root.name}.distill.jsonl"
            with destination.open("w", encoding="utf-8", newline="\n") as handle:
                for record in records:
                    handle.write(canonical_json(record) + "\n")
            print(f"[write] {destination} ({len(records)} record(s))")

        if args.stats or not args.out:
            splits: dict[str, int] = {}
            for record in records:
                splits[record["split"]] = splits.get(record["split"], 0) + 1
            by_task = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()) if v)
            by_split = ", ".join(f"{k}={v}" for k, v in sorted(splits.items()))
            print(f"{root.name}: {len(records)} record(s) [{by_task}] splits[{by_split}]")

    print(f"build_distillation: {total} gate-backed record(s) total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
