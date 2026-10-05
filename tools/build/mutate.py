#!/usr/bin/env python3
"""Apply one manifest mutation: the fault-injection witnesses, spelled once and applied once.

A gate must be able to fail (docs/security/laws.md L2), so several runtime-gate sections grade a
mutant -- the twin rebuilt with one law removed -- and require it to turn a row red. Each mutant
is a `variants` entry of runtime/manifest.json naming its harness (`of`) and one edit to one
runtime source (`mutation`: file, find, replace). tools/c/check_runtime.sh and the CMake build
both generate the mutant source with this tool, so the injected fault has one spelling (the
manifest) and one applier (here), and the two builds cannot inject different faults.

    mutate.py --variant test_kplan_mutant --out build/mutants/bcir_kplan.c

The edit is exact text, applied only when the anchor occurs exactly once: an anchor that is gone
(the law's line changed) or that repeats (the edit would be ambiguous) fails loudly instead of
grading an unmutated or a wrongly mutated copy, and so does an edit that changes nothing.

Exit 0 with the mutant written; 1 when the mutation does not apply; 2 when the manifest or the
variant is unusable. Nothing is written unless the mutation applies.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "runtime" / "manifest.json"
C_DIR = ROOT / "runtime" / "c"


class MutationError(Exception):
    """The mutation does not apply (exit 1)."""


class UnusableError(Exception):
    """The manifest or the variant cannot be read as a mutation (exit 2)."""


def mutation_of(manifest: dict, variant: str) -> dict:
    variants = manifest.get("variants")
    if not isinstance(variants, dict) or variant not in variants:
        raise UnusableError(f"no manifest variant named {variant!r}")
    mutation = variants[variant].get("mutation") if isinstance(variants[variant], dict) else None
    if not isinstance(mutation, dict):
        raise UnusableError(f"variant {variant} carries no mutation")
    for key in ("file", "find", "replace"):
        if not isinstance(mutation.get(key), str) or not mutation[key]:
            raise UnusableError(f"variant {variant}: mutation.{key} is not a non-empty string")
    if "/" in mutation["file"] or "\\" in mutation["file"] or not mutation["file"].endswith(".c"):
        raise UnusableError(
            f"variant {variant}: mutation.file {mutation['file']!r} is not a runtime/c basename"
        )
    return mutation


def mutant_bytes(manifest: dict, variant: str, c_dir: Path = C_DIR) -> bytes:
    """The variant's mutated source, or MutationError naming why the edit does not apply."""
    mutation = mutation_of(manifest, variant)
    source = c_dir / mutation["file"]
    try:
        original = source.read_bytes()
    except OSError as exc:
        raise UnusableError(f"variant {variant}: {exc}") from exc
    find = mutation["find"].encode("utf-8")
    replace = mutation["replace"].encode("utf-8")
    count = original.count(find)
    if count != 1:
        raise MutationError(
            f"variant {variant}: the anchor occurs {count} time(s) in {mutation['file']}, not once "
            f"(the law's line changed, or the anchor is ambiguous)"
        )
    mutated = original.replace(find, replace, 1)
    if mutated == original:
        raise MutationError(f"variant {variant}: the edit changes nothing in {mutation['file']}")
    return mutated


def write_atomically(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", required=True, help="a `variants` entry carrying a mutation")
    parser.add_argument("--out", required=True, type=Path, help="where to write the mutated source")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--c-dir", type=Path, default=C_DIR, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise UnusableError(f"{args.manifest} is not a JSON object")
        data = mutant_bytes(manifest, args.variant, args.c_dir)
    except (OSError, ValueError, UnusableError) as exc:
        print(f"mutate: UNUSABLE: {exc}", file=sys.stderr)
        return 2
    except MutationError as exc:
        print(f"mutate: FAIL: {exc}", file=sys.stderr)
        return 1
    write_atomically(args.out, data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
