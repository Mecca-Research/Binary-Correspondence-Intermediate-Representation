#!/usr/bin/env python3
"""Where LLVM keeps the artifacts a gate has already checked.

Tier 3 grows only as fast as verification does: a record exists because some
gate in this repository proves its answer. This file is the whole of what that
means for `training/llvm/` -- five places where a checked artifact can be turned
into a question whose answer is already known to be right, and the gate that
knows it.

None of this belongs in the shared builder. `training/tools/distillation.py`
owns the record contract, the split policy and the identity digest; it does not
know that an invalid fixture is named `*.invalid.ll.txt`, that optimizer goldens
live under `07-optimization/examples/`, or that a frontend claim is matched
against a checked-in Clang snapshot. Those are facts about this subject, and a
second subject will have entirely different ones.

Every source here is **hermetic**: it reads checked-in artifacts and needs no
toolchain, so the build reproduces on any host. The gates that make the answers
true are the ones that need `llvm-as`, `opt` and `clang`, and they run
separately.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SUBJECT_ROOT = Path(__file__).resolve().parents[1]
SHARED_TOOLS = SUBJECT_ROOT.parent / "tools"
if str(SHARED_TOOLS) not in sys.path:
    sys.path.insert(0, str(SHARED_TOOLS))

from distillation import RecordBuilder, RecordSource  # noqa: E402

SYSTEM_PROMPT = (
    "You are working with LLVM IR and compiler toolchains. Answer precisely, "
    "and prefer the exact spelling a tool would produce over a paraphrase."
)


def from_exercises(builder: RecordBuilder) -> list[dict]:
    """Graded exercises: the prompt asks, the reference solution answers."""
    manifests = sorted((builder.subject_root / "autograder" / "manifests").glob("*.json"))
    gate = builder.gate("verify-exercises.sh")
    records: list[dict] = []

    for manifest_path in manifests:
        manifest = json.loads(builder.read(manifest_path))
        solution = manifest.get("solution")
        prompt = manifest.get("prompt")
        if not solution or not prompt:
            continue  # review exercises with no canonical answer are not SFT data
        solution_file = builder.repo_root / solution
        prompt_file = builder.repo_root / prompt
        if not solution_file.is_file() or not prompt_file.is_file():
            continue
        if not solution.endswith((".ll", ".md")):
            continue

        answer = builder.read(solution_file).strip()
        body = builder.read(prompt_file).strip()
        kind = "code" if solution.endswith(".ll") else "explanation"
        records.append(
            builder.record(
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
                difficulty=builder.difficulty(manifest.get("difficulty", "")),
            )
        )
    return records


def from_invalid_fixtures(builder: RecordBuilder) -> list[dict]:
    """Intentionally invalid IR: the fixture's own header states the defect."""
    gate = builder.gate("verify-invalid-fixtures.sh")
    records: list[dict] = []

    for fixture in sorted(builder.subject_root.rglob("*.invalid.ll.txt")):
        text = builder.read(fixture)
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
            builder.record(
                task="repair",
                user=(
                    "This LLVM IR module is rejected by the toolchain. State why.\n\n"
                    f"```llvm\n{body}\n```"
                ),
                assistant=f"It is rejected because {reason[0].lower() + reason[1:]}.",
                answer_kind="diagnosis",
                source_paths=[builder.relpath(fixture)],
                gate=gate,
                claim="the gate proves LLVM rejects this fixture",
                difficulty=3,
            )
        )
    return records


def from_opt_goldens(builder: RecordBuilder) -> list[dict]:
    """Before/after optimizer pairs pinned by the opt-diff gate."""
    gate = builder.gate("verify-opt-diff.sh")
    examples = builder.subject_root / "07-optimization" / "examples"
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
            builder.record(
                task="prediction",
                user=(
                    f"Run `opt -S -passes={pass_name}` on this module and give the "
                    f"resulting IR.\n\n```llvm\n{builder.read(before).strip()}\n```"
                ),
                assistant=f"```llvm\n{builder.read(after).strip()}\n```",
                answer_kind="code",
                source_paths=[builder.relpath(before), builder.relpath(after)],
                gate=gate,
                claim=f"the gate re-runs {pass_name} and diffs against this golden",
                difficulty=4,
            )
        )
    return records


def from_frontend_claims(builder: RecordBuilder) -> list[dict]:
    """Structural claims about compiler output, matched against the snapshot.

    The claims are the gate's own declarative table, and the snapshots are
    checked in. Matching one against the other yields a question whose answer is
    a line of real, verified compiler output -- no toolchain needed at build
    time, because the gate already proved the claim against fresh output.
    """
    tool = builder.tool("verify-frontend-lowering.py")
    if not tool.is_file():
        return []
    module = builder.load_module(tool, "verify_frontend_lowering")
    gate = builder.relpath(tool)
    records: list[dict] = []

    for case in module.CASES:
        if not case.snapshot:
            continue
        snapshot = module.EXAMPLES / case.snapshot
        if not snapshot.is_file():
            continue
        snapshot_text = builder.read(snapshot)

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
                builder.record(
                    task="claim",
                    user=(
                        f"Compiling `{case.source}` with `clang {case.opt_level} "
                        f"-S -emit-llvm -target {case.triple}`: show the emitted IR "
                        f"that demonstrates that {claim.description}."
                    ),
                    assistant=f"```llvm\n{evidence}\n```",
                    answer_kind="code",
                    source_paths=[
                        builder.relpath(snapshot),
                        builder.relpath(module.EXAMPLES / case.source),
                    ],
                    gate=gate,
                    claim=claim.description,
                    difficulty=4,
                )
            )
    return records


def from_benchmark_verdicts(builder: RecordBuilder) -> list[dict]:
    """Benchmark fixtures whose verdicts the analysis self-test pins."""
    self_test = builder.tool("verify-benchmark-analysis.py")
    analyzer = builder.tool("analyze-benchmark-samples.py")
    if not self_test.is_file() or not analyzer.is_file():
        return []

    checker = builder.load_module(self_test, "verify_benchmark_analysis")
    engine = builder.load_module(analyzer, "analyze_benchmark_samples")
    gate = builder.relpath(self_test)
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
            builder.record(
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
                source_paths=[builder.relpath(fixture)],
                gate=gate,
                claim=f"the self-test pins this fixture's verdict as '{verdict}'",
                difficulty=5,
            )
        )
    return records


SOURCES = (
    RecordSource("exercise", from_exercises),
    RecordSource("repair", from_invalid_fixtures),
    RecordSource("prediction", from_opt_goldens),
    RecordSource("claim", from_frontend_claims),
    RecordSource("review", from_benchmark_verdicts),
)
