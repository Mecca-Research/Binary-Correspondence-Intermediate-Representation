#!/usr/bin/env python3
"""The train rail's corpus witness (TRAIN-RAIL): every tracked Markdown file of `docs/` and
`training/` is in what the rail trained on, or this says which is not.

The one-way corpus law (AGENTS.md section 2, docs/research/BCIR_TRAINING_ACCELERATION_ROADMAP.md
section 5) lets the IR's build read the training corpus as data, through the corpus recipe
tools/models/configs/bcir-corpus-860k.json. `train.corpus.unselected` is the law's exact row: the
tracked `*.md` files under the two corpus roots that the recipe's selection does not reach. Before
the rail, the lab recipe read `docs/` alone and every file of `training/` was outside the IR's
training data; the row counts them, and the rail holds it at zero.

    train_rail.py check [--recipe RECIPE] [--corpus-json OUT/corpus.json]

Without --corpus-json the recipe's own selection is computed (build_lab_model.corpus_documents),
so the row reads without a training run; with it, the files are read from a built model's
corpus.json -- what the rail did train on. Exit 0 at zero, 1 above it, 2 when the row cannot be
read (docs/security/laws.md L1): every exit is a verdict.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# The two corpus roots the law names, in the order the recipe lists them.
ROOTS = ("docs", "training")
DEFAULT_RECIPE = ROOT / "tools" / "models" / "configs" / "bcir-corpus-860k.json"
_LIMIT = 1 << 20  # a corpus.json is a file manifest, not a corpus


def _builder():
    """tools/models/build_lab_model.py, loaded beside this file (the two live in tools/, not in
    the package)."""
    path = Path(__file__).resolve().parent / "build_lab_model.py"
    spec = importlib.util.spec_from_file_location("bcir_tools_build_lab_model", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def tracked_markdown(root: Path, roots: Iterable[str] = ROOTS) -> list[str]:
    """Every tracked `*.md` under the corpus roots, as `git ls-files` spells it (sorted, unique)."""
    patterns = [p for r in roots for p in (f"{r}/*.md", f"{r}/**/*.md")]
    listed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", *patterns],
        capture_output=True,
        check=True,
        timeout=120,
    ).stdout
    return sorted({p for p in listed.decode("utf-8").split("\0") if p})


def unselected(root: Path, selected: Iterable[str], roots: Iterable[str] = ROOTS) -> list[str]:
    """The tracked Markdown files of the roots that `selected` (repo-relative paths) leaves out."""
    chosen = set(selected)
    return [path for path in tracked_markdown(root, roots) if path not in chosen]


def selected_by_recipe(root: Path, recipe_path) -> list[str]:
    """The files the recipe's corpus selects, computed as the builder computes them (one
    predicate: `corpus_documents`), without a training run."""
    builder = _builder()
    recipe, _spec, _sha = builder.load_recipe(recipe_path)
    _documents, files = builder.corpus_documents(root, recipe["corpus"])
    return [entry["path"] for entry in files]


def selected_by_card(corpus_json) -> list[str]:
    """The files a built model's corpus.json records: what the rail trained on."""
    path = Path(corpus_json)
    if path.stat().st_size > _LIMIT:
        raise ValueError(f"{path}: larger than a file manifest ({_LIMIT} bytes)")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    files = manifest["files"] if isinstance(manifest, dict) else None
    if not isinstance(files, list) or not all(
        isinstance(entry, dict) and isinstance(entry.get("path"), str) for entry in files
    ):
        raise ValueError(f"{path}: not a corpus manifest ({{files: [{{path, ...}}]}})")
    return [entry["path"] for entry in files]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="read train.corpus.unselected; exit 1 above zero")
    check.add_argument("--recipe", default=str(DEFAULT_RECIPE))
    check.add_argument(
        "--corpus-json", help="a built model's corpus.json (default: the recipe's selection)"
    )
    args = parser.parse_args(argv)
    try:
        if args.corpus_json:
            selected = selected_by_card(args.corpus_json)
            source = args.corpus_json
        else:
            selected = selected_by_recipe(ROOT, args.recipe)
            source = args.recipe
        left = unselected(ROOT, selected)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(f"train-rail: UNUSABLE: {exc}", file=sys.stderr)
        return 2
    print(
        f"train-rail: corpus files={len(selected)} roots={','.join(ROOTS)} "
        f"unselected={len(left)} ({source})"
    )
    for path in left[:20]:
        print(f"  unselected {path}")
    if len(left) > 20:
        print(f"  ... {len(left) - 20} more")
    return 1 if left else 0


if __name__ == "__main__":
    sys.exit(main())
