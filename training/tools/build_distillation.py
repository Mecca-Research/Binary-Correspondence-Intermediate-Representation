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

Record sources, each one hermetic -- derived from checked-in artifacts, needing
no toolchain, so the build is reproducible on any host:

  exercise   a graded exercise manifest: prompt -> reference solution
             (gate: verify-exercises.sh proves the solution assembles/verifies)
  repair     an intentionally invalid fixture -> why LLVM rejects it
             (gate: verify-invalid-fixtures.sh proves it IS rejected)
  prediction a before/after optimizer golden -> the post-pass IR
             (gate: verify-opt-diff.sh proves the golden still reproduces)
  claim      a structural claim asserted against a checked-in compiler snapshot
             (gate: verify-frontend-lowering.py proves the claim holds)
  review     a benchmark sample file -> the analysis verdict and its reason
             (gate: verify-benchmark-analysis.py pins that verdict)

    python3 training/tools/build_distillation.py --out build/training/distill
    python3 training/tools/build_distillation.py --stats
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
CORPUS_ROOT = TOOLS_DIR.parent
REPO_ROOT = CORPUS_ROOT.parent

SCHEMA = "bcir-training/distill/v1"
LICENSE = "LicenseRef-BCIR-NC-1.0"
BUILDER = "training/tools/build_distillation.py"

SYSTEM_PROMPT = (
    "You are working with LLVM IR and compiler toolchains. Answer precisely, "
    "and prefer the exact spelling a tool would produce over a paraphrase."
)

DIFFICULTY_BY_NAME = {"beginner": 1, "intermediate": 3, "advanced": 4, "expert": 5}

# Deterministic split weights. Assignment is by SOURCE FILE, not by record, so
# two records derived from one file can never straddle a split boundary.
SPLIT_BUCKETS = [("train", 80), ("validation", 10), ("test", 10)]


def canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def relpath(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def assign_split(source_path: str) -> str:
    """Hash the source path into a split. Same file -> same split, always."""
    position = int(sha256_text(source_path)[:8], 16) % 100
    cursor = 0
    for name, weight in SPLIT_BUCKETS:
        cursor += weight
        if position < cursor:
            return name
    return SPLIT_BUCKETS[-1][0]


def make_record(
    *,
    subject: str,
    task: str,
    user: str,
    assistant: str,
    answer_kind: str,
    source_paths: list[str],
    gate: str,
    claim: str,
    difficulty: int,
) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user.strip()},
        {"role": "assistant", "content": assistant.strip()},
    ]
    identity = canonical_json({"subject": subject, "task": task, "messages": messages})
    return {
        "schema": SCHEMA,
        "record_id": f"sha256:{sha256_text(identity)}",
        "corpus": "training",
        "subject": subject,
        "task": task,
        "messages": messages,
        "answer_kind": answer_kind,
        "source_paths": sorted(set(source_paths)),
        "verified_by": {"gate": gate, "claim": claim},
        "split": assign_split(sorted(set(source_paths))[0]),
        "difficulty": difficulty,
        "provenance": {"license": LICENSE, "builder": BUILDER},
    }


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


def from_exercises(subject_root: Path) -> list[dict]:
    """Graded exercises: the prompt asks, the reference solution answers."""
    manifests = sorted((subject_root / "autograder" / "manifests").glob("*.json"))
    gate = relpath(subject_root / "tools" / "verify-exercises.sh")
    records: list[dict] = []

    for manifest_path in manifests:
        manifest = json.loads(read(manifest_path))
        solution = manifest.get("solution")
        prompt = manifest.get("prompt")
        if not solution or not prompt:
            continue  # review exercises with no canonical answer are not SFT data
        solution_file = REPO_ROOT / solution
        prompt_file = REPO_ROOT / prompt
        if not solution_file.is_file() or not prompt_file.is_file():
            continue
        if not solution.endswith((".ll", ".md")):
            continue

        answer = read(solution_file).strip()
        body = read(prompt_file).strip()
        kind = "code" if solution.endswith(".ll") else "explanation"
        records.append(
            make_record(
                subject=subject_root.name,
                task="exercise",
                user=f"{body}\n\nProduce the complete artifact.",
                assistant=(f"```llvm\n{answer}\n```" if kind == "code" else answer),
                answer_kind=kind,
                source_paths=[prompt, solution],
                gate=gate,
                claim=(
                    f"exercise {manifest['id']}: the reference solution is the graded "
                    "answer, and the gate proves it assembles and verifies"
                ),
                difficulty=DIFFICULTY_BY_NAME.get(manifest.get("difficulty", ""), 3),
            )
        )
    return records


def from_invalid_fixtures(subject_root: Path) -> list[dict]:
    """Intentionally invalid IR: the fixture's own header states the defect."""
    gate = relpath(subject_root / "tools" / "verify-invalid-fixtures.sh")
    records: list[dict] = []

    for fixture in sorted(subject_root.rglob("*.invalid.ll.txt")):
        text = read(fixture)
        first = text.splitlines()[0].strip() if text.splitlines() else ""
        # The convention is a leading comment naming the intended failure.
        match = re.match(r";\s*(?:Intentionally invalid|Invalid)\s*[:\-]\s*(.+)", first)
        if not match:
            continue
        reason = match.group(1).strip().rstrip(".")
        body = "\n".join(
            line for line in text.splitlines() if not line.strip().startswith("; Intentionally")
        ).strip()
        records.append(
            make_record(
                subject=subject_root.name,
                task="repair",
                user=(
                    "This LLVM IR module is rejected by the toolchain. State why.\n\n"
                    f"```llvm\n{body}\n```"
                ),
                assistant=f"It is rejected because {reason[0].lower() + reason[1:]}.",
                answer_kind="diagnosis",
                source_paths=[relpath(fixture)],
                gate=gate,
                claim="the gate proves LLVM rejects this fixture",
                difficulty=3,
            )
        )
    return records


def from_opt_goldens(subject_root: Path) -> list[dict]:
    """Before/after optimizer pairs pinned by the opt-diff gate."""
    gate = relpath(subject_root / "tools" / "verify-opt-diff.sh")
    examples = subject_root / "07-optimization" / "examples"
    records: list[dict] = []
    if not examples.is_dir():
        return records

    for after in sorted(examples.glob("opt-diff-*.after-*.ll")):
        stem, _, pass_name = after.name.partition(".after-")
        pass_name = pass_name[: -len(".ll")]
        before = examples / f"{stem}-before.ll"
        if not before.is_file():
            continue
        records.append(
            make_record(
                subject=subject_root.name,
                task="prediction",
                user=(
                    f"Run `opt -S -passes={pass_name}` on this module and give the "
                    f"resulting IR.\n\n```llvm\n{read(before).strip()}\n```"
                ),
                assistant=f"```llvm\n{read(after).strip()}\n```",
                answer_kind="code",
                source_paths=[relpath(before), relpath(after)],
                gate=gate,
                claim=f"the gate re-runs {pass_name} and diffs against this golden",
                difficulty=4,
            )
        )
    return records


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def from_frontend_claims(subject_root: Path) -> list[dict]:
    """Structural claims about compiler output, matched against the snapshot.

    The claims are the gate's own declarative table, and the snapshots are
    checked in. Matching one against the other yields a question whose answer is
    a line of real, verified compiler output -- no toolchain needed at build
    time, because the gate already proved the claim against fresh output.
    """
    tool = subject_root / "tools" / "verify-frontend-lowering.py"
    if not tool.is_file():
        return []
    module = load_module(tool, "verify_frontend_lowering")
    gate = relpath(tool)
    records: list[dict] = []

    for case in module.CASES:
        if not case.snapshot:
            continue
        snapshot = module.EXAMPLES / case.snapshot
        if not snapshot.is_file():
            continue
        snapshot_text = read(snapshot)

        for claim in case.claims:
            if not claim.present:
                continue  # "must NOT appear" is not a question with an answer
            haystack = snapshot_text
            if claim.scope:
                body = module.function_body(snapshot_text, claim.scope)
                if body is None:
                    continue
                haystack = body
            found = re.search(claim.pattern, haystack)
            if found is None:
                continue  # normalization removed it; do not invent an answer

            # Expand the match to the full source line(s): a bare regex hit
            # fenced as ```llvm reads as a complete construct and is not one.
            line_start = haystack.rfind("\n", 0, found.start()) + 1
            line_end = haystack.find("\n", found.end())
            if line_end == -1:
                line_end = len(haystack)
            evidence = haystack[line_start:line_end].strip()
            records.append(
                make_record(
                    subject=subject_root.name,
                    task="claim",
                    user=(
                        f"Compiling `{case.source}` with `clang {case.opt_level} "
                        f"-S -emit-llvm -target {case.triple}`: show the emitted IR "
                        f"that demonstrates that {claim.description}."
                    ),
                    assistant=f"```llvm\n{evidence}\n```",
                    answer_kind="code",
                    source_paths=[relpath(snapshot), relpath(module.EXAMPLES / case.source)],
                    gate=gate,
                    claim=claim.description,
                    difficulty=4,
                )
            )
    return records


def from_benchmark_verdicts(subject_root: Path) -> list[dict]:
    """Benchmark fixtures whose verdicts the analysis self-test pins."""
    self_test = subject_root / "tools" / "verify-benchmark-analysis.py"
    analyzer = subject_root / "tools" / "analyze-benchmark-samples.py"
    if not self_test.is_file() or not analyzer.is_file():
        return []

    checker = load_module(self_test, "verify_benchmark_analysis")
    engine = load_module(analyzer, "analyze_benchmark_samples")
    gate = relpath(self_test)
    records: list[dict] = []

    for name, (verdict, _needle) in sorted(checker.EXPECTED.items()):
        fixture = checker.FIXTURES / name
        if not fixture.is_file():
            continue
        document = engine.load(fixture)
        result = engine.analyze(
            document,
            alpha=0.05,
            iterations=2000,
            seed=20260101,
            min_samples=engine.DEFAULT_MIN_SAMPLES,
            min_effect=engine.DEFAULT_MIN_EFFECT,
        )
        baseline = document["series"]["baseline"]["samples"]
        candidate = document["series"]["candidate"]["samples"]
        records.append(
            make_record(
                subject=subject_root.name,
                task="review",
                user=(
                    "A benchmark compared two builds of the workload "
                    f"`{document.get('workload', 'unnamed')}` "
                    f"(unit: {document.get('unit', 'unit')}, metric class: "
                    f"{document.get('metric_class', 'wall')}).\n\n"
                    f"baseline ({len(baseline)} samples, in run order): "
                    f"{', '.join(f'{v:g}' for v in baseline)}\n"
                    f"candidate ({len(candidate)} samples, in run order): "
                    f"{', '.join(f'{v:g}' for v in candidate)}\n\n"
                    "Is the difference real? Give the verdict and the reason."
                ),
                assistant=f"Verdict: {result.verdict}.\n\n{result.reason}",
                answer_kind="explanation",
                source_paths=[relpath(fixture)],
                gate=gate,
                claim=f"the self-test pins this fixture's verdict as '{verdict}'",
                difficulty=5,
            )
        )
    return records


SOURCES = (
    ("exercise", from_exercises),
    ("repair", from_invalid_fixtures),
    ("prediction", from_opt_goldens),
    ("claim", from_frontend_claims),
    ("review", from_benchmark_verdicts),
)


def build_subject(subject_root: Path) -> tuple[list[dict], dict[str, int]]:
    records: list[dict] = []
    counts: dict[str, int] = {}
    for name, source in SOURCES:
        produced = source(subject_root)
        counts[name] = len(produced)
        records.extend(produced)
    records.sort(key=lambda r: (r["task"], r["source_paths"][0], r["record_id"]))
    return records, counts


def subject_roots(selected: str | None) -> list[Path]:
    roots = [
        path
        for path in sorted(CORPUS_ROOT.iterdir())
        if path.is_dir() and not path.name.startswith(".") and path.name not in {"tools", "schema"}
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
            with destination.open("w", encoding="utf-8") as handle:
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
