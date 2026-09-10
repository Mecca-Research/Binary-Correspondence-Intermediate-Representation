#!/usr/bin/env python3
"""What a subject must tell the shared rail about itself.

`training/tools/` owns the machinery every subject needs -- confining an
untrusted answer, bounding a tool run, allocating points, checking text,
shaping a report, exporting a dataset. None of that is about LLVM, and none of
it should be written a second time when `hardware/` or `systems/` opens.

What *is* about LLVM is small and enumerable: which answer kinds exist, what
file extension each one uses, which tool proves an answer is well-formed and
with what argument array, how tool binaries are discovered on a host that ships
`opt-18` rather than `opt`, and how a runnable answer is wrapped into a program.
This module is the vocabulary for saying exactly that much and no more.

The split is load-bearing rather than tidy. Before it, three tools under
`training/llvm/tools/` each carried their own copy of `find_tool`, and two more
carried their own `model_visible_prompt` and `normalized_text`. The copies
happened to agree; nothing made them agree, and the next edit to one of them
would have been the drift. One predicate, called from every site, is the rule
this repository already applies to its verifier laws.

A profile is data, not a plugin system: a subject constructs one, hands it to a
kernel, and the kernel does the rest. There is no registry, no discovery, and no
import of a subject by the shared rail -- the dependency points one way, the
same direction `training/` itself points at BCIR.
"""

from __future__ import annotations

import dataclasses
import os
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class ToolCheck:
    """One external-tool assertion: what to run, and what its verdict is called.

    `argv` builds the argument array from the resolved tool path and the answer
    file. It is a callable rather than a template because the shape differs per
    tool -- an assembler takes an output path, a verifier takes a pass pipeline
    -- and a template language for that would be a worse version of Python.

    A `normalizes` check contributes its stdout as the text every later
    structural assertion reads. That is how canonicalizing a submission before
    pattern-matching it stays a subject's decision: the kernel has no opinion
    about what normalized form means, only that at most one check produces it.
    """

    id: str
    dimension: str
    tool: str
    argv: Callable[[str, Path], list[str]]
    normalizes: bool = False
    # How this check refers to itself when it fails. Empty means the generic
    # "tool exited 1"; a subject that runs `opt -S` says so, because a reader of
    # the report should not have to guess which invocation produced the failure.
    label: str = ""
    pass_message: str = "tool accepted answer"
    missing_message: str = ""

    def absent(self) -> str:
        return self.missing_message or f"required tool not found: {self.tool}"

    def failed(self, returncode: int, timed_out: bool) -> str:
        subject = self.label or "tool"
        return f"{subject} timed out" if timed_out else f"{subject} exited {returncode}"


@dataclasses.dataclass(frozen=True)
class AnswerKind:
    """A shape of answer a subject accepts, and how a file of it is named.

    `rubric_coverage_only` marks a kind whose score measures deterministic
    rubric coverage rather than semantic proof -- prose review, a diagnostic
    explanation. The distinction is reported on every result, because a
    percentage that means "covered the required terms" and one that means "the
    verifier accepted it" must not be read as the same number.
    """

    name: str
    extension: str
    checks: tuple[ToolCheck, ...] = ()
    rubric_coverage_only: bool = False


@dataclasses.dataclass(frozen=True)
class SemanticHarness:
    """Running a submission, which only a subject knows how to do safely.

    Execution is the one grading step that cannot be subject-neutral: it needs a
    language, a calling convention, and an interpreter. It is also the step with
    real blast radius, so it is opt-in twice over -- a subject must supply a
    harness at all, and each exercise must carry `allowlist_field` set true.

    `refuse` gets the submission text and returns a reason to reject it before
    anything runs, or None. It exists so a subject can refuse a submission that
    would collide with the harness it is about to be wrapped in.
    """

    tool: str
    allowlist_field: str
    build: Callable[[str, Mapping[str, object], int], str]
    suffix: str
    refuse: Callable[[str], str | None] | None = None
    denied_message: str = "execution is not allowlisted for this exercise"


@dataclasses.dataclass(frozen=True)
class SubjectProfile:
    """One subject's whole contribution to the shared grading and export rails."""

    name: str
    root: Path
    kinds: Mapping[str, AnswerKind]
    # Hosts ship `opt-18` as often as `opt`. The env var lets a caller pin one
    # toolchain; the version sweep finds a suffixed binary when nothing else is
    # on PATH. Both are subject policy: nothing about grading requires either.
    tool_suffix_env: str = ""
    tool_versions: Sequence[int] = ()
    always_probe: tuple[str, ...] = ()
    semantic: SemanticHarness | None = None
    # Applied to prompt text before a model sees it. A prompt that names its own
    # reference solution hands the answer to the thing being evaluated.
    solution_pattern: re.Pattern[str] | None = None
    solution_replacement: Callable[[re.Match[str]], str] | None = None
    # Prefix for the stable identifier a dataset record carries.
    dataset_id_prefix: str = ""

    def kind(self, name: str) -> AnswerKind:
        try:
            return self.kinds[name]
        except KeyError:
            raise ValueError(f"unsupported answer kind for {self.name}: {name}") from None

    def extension_for(self, kind_name: str) -> str:
        return self.kind(kind_name).extension

    def find_tool(self, name: str, *, major: int | None = None) -> str | None:
        """Resolve a tool name to a path, or None. The one copy of this rule.

        Two search orders, one predicate. With no `major`, an unsuffixed binary
        on PATH wins and the version sweep is the fallback for hosts that ship
        only `opt-18`. With a `major`, that exact version wins instead -- a
        caller that names one is asserting the version matters, so silently
        answering with a different one would defeat the point of asking.
        """
        suffix = os.environ.get(self.tool_suffix_env, "") if self.tool_suffix_env else ""
        candidates = [name + suffix] if suffix else []
        if major is not None:
            candidates += [f"{name}-{major}", name]
        else:
            candidates += [name]
            candidates += [f"{name}-{version}" for version in self.tool_versions]
        return next((path for c in candidates if (path := shutil.which(c))), None)

    def model_visible(self, text: str) -> str:
        """Strip reference-solution paths from text a model is allowed to read."""
        if self.solution_pattern is None:
            return text
        replacement = self.solution_replacement or (lambda match: match.group(0))
        return self.solution_pattern.sub(replacement, text)
