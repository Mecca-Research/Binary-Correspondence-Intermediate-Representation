#!/usr/bin/env python3
"""Everything the shared training rails need to know about LLVM, and no more.

The grader, the dataset exporter and the evaluation runner live in
`training/tools/` and know nothing about compilers. This file is the whole of
what makes them grade *this* subject: five answer kinds, the tools that prove an
answer well-formed, how a host that ships `opt-18` instead of `opt` is searched,
how a prompt's reference-solution path is hidden from a model, and the `lli`
harness that runs a submission.

Adding a second subject means writing a file like this one -- not copying a
grader.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Mapping

SUBJECT_ROOT = Path(__file__).resolve().parents[1]
SHARED_TOOLS = SUBJECT_ROOT.parent / "tools"
if str(SHARED_TOOLS) not in sys.path:
    sys.path.insert(0, str(SHARED_TOOLS))

from subject_profile import AnswerKind, SemanticHarness, SubjectProfile, ToolCheck  # noqa: E402

# LLVM tools are frequently installed with a version suffix and no unsuffixed
# symlink. The sweep runs newest-first so a host with several majors grades on
# the newest, and LLVM_SUFFIX pins one explicitly when that matters.
TOOL_SUFFIX_ENV = "LLVM_SUFFIX"
TOOL_VERSIONS = tuple(range(30, 9, -1))

SOLUTION_PATH = re.compile(r"(?:training/llvm/exercises/)?[^\s`\"']*\.solution\.(ll|mlir|md)")


def _hide_solution(match: re.Match[str]) -> str:
    return f"answer.{match.group(1)}"


ASSEMBLE = ToolCheck(
    id="llvm.assemble",
    dimension="validity",
    tool="llvm-as",
    argv=lambda tool, answer: [tool, str(answer), "-o", os.devnull],
)
VERIFY = ToolCheck(
    id="llvm.verify",
    dimension="validity",
    tool="opt",
    argv=lambda tool, answer: [tool, "-passes=verify", str(answer), "-o", os.devnull],
)
# Structural assertions read canonical IR rather than whatever the submission
# happened to spell, so a correct answer is not failed for its formatting.
NORMALIZE = ToolCheck(
    id="llvm.normalize",
    dimension="normalization",
    tool="opt",
    argv=lambda tool, answer: [tool, "-S", str(answer), "-o", "-"],
    normalizes=True,
    label="opt -S",
    pass_message="opt -S produced normalized textual IR",
    missing_message="opt unavailable; structural checks use original text",
)
MLIR_PARSE = ToolCheck(
    id="mlir.parse",
    dimension="validity",
    tool="mlir-opt",
    argv=lambda tool, answer: [tool, str(answer), "-o", os.devnull],
)

ANSWER_KINDS: Mapping[str, AnswerKind] = {
    "llvm-ir": AnswerKind("llvm-ir", ".ll", (ASSEMBLE, VERIFY, NORMALIZE)),
    "mlir": AnswerKind("mlir", ".mlir", (MLIR_PARSE,)),
    "markdown-review": AnswerKind("markdown-review", ".md", rubric_coverage_only=True),
    "pass-output": AnswerKind("pass-output", ".ll"),
    "diagnostic": AnswerKind("diagnostic", ".md", rubric_coverage_only=True),
}


def _refuse_own_main(source: str) -> str | None:
    """The harness supplies @main; a submission that also defines one collides."""
    if re.search(r"\bdefine\s+i32\s+@main\s*\(", source):
        return "submission defines @main; semantic harness requires that name"
    return None


def _wrap_for_lli(source: str, vector: Mapping[str, object], index: int) -> str:
    """Wrap a submission in a module whose exit status IS the verdict."""
    return_type = vector.get("return_type", "i32")
    args = vector.get("args", [])
    typed = ", ".join(f"{item['type']} {item['value']}" for item in args)
    expected = vector["expected"]
    predicate = vector.get("predicate", "eq")
    call_name = f"%actual{index}"
    cmp_name = f"%ok{index}"
    wrapper = (
        f"  {call_name} = call {return_type} @{vector['function']}({typed})\n"
        f"  {cmp_name} = icmp {predicate} {return_type} {call_name}, {expected}\n"
    )
    return (
        source
        + "\n\ndefine i32 @main() {\nentry:\n"
        + wrapper
        + f"  %exit = select i1 %ok{index}, i32 0, i32 1\n  ret i32 %exit\n}}\n"
    )


SEMANTIC = SemanticHarness(
    tool="lli",
    allowlist_field="safe_deterministic_lli",
    build=_wrap_for_lli,
    suffix=".ll",
    refuse=_refuse_own_main,
    denied_message="lli execution is not allowlisted for this exercise",
)

PROFILE = SubjectProfile(
    name="llvm",
    root=SUBJECT_ROOT,
    kinds=ANSWER_KINDS,
    tool_suffix_env=TOOL_SUFFIX_ENV,
    tool_versions=TOOL_VERSIONS,
    # `opt` normalizes IR for every kind that has structural assertions, so it is
    # probed even when no exercise lists it among its required tools.
    always_probe=("opt",),
    semantic=SEMANTIC,
    solution_pattern=SOLUTION_PATH,
    solution_replacement=_hide_solution,
    dataset_id_prefix="training-llvm-exercise-",
)
