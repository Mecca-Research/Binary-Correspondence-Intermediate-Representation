#!/usr/bin/env python3
"""Verify the Clang frontend-lowering claims made by training/llvm/20-clang-frontend/.

The chapter states, as facts, things like "a 12-byte struct is passed in two
registers on x86-64 SysV and as `[2 x i64]` on AArch64" and "a C bit-field does
not exist in LLVM IR; the frontend lowers it to shifts and masks". Those are
checkable, and a documentation corpus that states them without checking them is
one toolchain release away from being wrong.

This tool compiles each checked-in source with `clang -S -emit-llvm` and asserts
structural claims against the result. Claims are deliberately *structural*
(a signature shape, an instruction family, an attribute) rather than byte-exact,
because exact value names, attribute spellings, and instruction order change
between Clang releases while the lowering rule does not.

    python3 training/llvm/tools/verify-frontend-lowering.py
    python3 training/llvm/tools/verify-frontend-lowering.py --require-tools
    python3 training/llvm/tools/verify-frontend-lowering.py --update

`--update` rewrites the normalized `.ll` snapshots next to each source.
Without `--require-tools` a missing or incapable Clang is an explicit skip, and
the exit status says which of pass/skip/fail happened -- a skip is never
reported as a pass.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = TRAINING_ROOT.parent.parent
EXAMPLES = TRAINING_ROOT / "20-clang-frontend" / "examples"

# SEMVER.md declares LLVM 18 as the corpus baseline. Snapshots are checked
# against the OLDEST assembler at or above it that the host provides, not the
# newest -- the newest is the one that cannot catch this class of defect.
#
# The baseline was 15 until Phase 1.7. It moved because a declared floor that
# nothing assembles is not a floor: CI's training job installs Ubuntu's default
# LLVM (18) and nothing in CI ever had an `llvm-as-15`, so the search below
# resolved to 18 and the corpus's LLVM 15 claim was checked on no machine that
# gates anything. 18 is what CI installs and can therefore enforce, it is the
# current Ubuntu LTS default so it is what a reader most likely has, and it keeps
# the reason the floor existed at all -- opaque pointers are the default from 15
# onward, so every release at or above the new floor still satisfies it.
#
# --require-baseline turns "no assembler at the declared major" from a silent
# pass into a failure, and belongs to the job that installs that major.
BASELINE_LLVM_MAJOR = 18

# Clang emits these but they are version noise, not lowering facts. Removing
# them also keeps the snapshots assembling on the corpus's LLVM >= 18 baseline,
# since attribute-group spellings (`memory(none)` and friends) are newer than
# the baseline while the instructions they annotate are not.
_DROP_LINE = re.compile(r"^(?:; ModuleID|source_filename|attributes #|!|; Function Attrs:)")
_ATTR_GROUP_REF = re.compile(r"\s#\d+\b")
_METADATA_SUFFIX = re.compile(r",\s*![a-zA-Z.]+ ![0-9]+")

# Inline parameter/return attributes that postdate the corpus's declared
# LLVM >= 18 baseline (SEMVER.md). Stripping attribute *groups* is not enough:
# these ride on the parameter itself, so they survive that pass and then fail to
# parse on a baseline assembler. That is not hypothetical -- CI's
# host-portability job runs an older llvm-as than the training job, so a
# snapshot that has only ever met the newest assembler passes locally and fails
# there.
#
# Dropping them costs the lesson nothing: they are refinements on a call whose
# sret/byval shape the snapshot already shows, and the chapter discusses them in
# prose where no assembler has to parse them.
#
# This list is a DECLARED SCOPE, not a claim of completeness. An attribute newer
# than the list is caught by the baseline assembly check in
# `check_snapshot_baseline`, which asks a real assembler instead of guessing.
# An attribute's SPELLING is not stable across majors either. `dead_on_return`
# entered as a bare flag and takes an argument by LLVM 23 -- clang 23.1.1 emits
# `dead_on_return(4)` on the destructor call in `cxx-object-model.cpp` -- so
# stripping only the bare name leaves `(4)` behind and the LLVM 15 assembler
# stops at `expected type`. Both lists are therefore stripped both ways: the
# parenthesized form first, then the bare one. Which list a name is in records
# how it was FIRST seen, not a promise about how it will be spelled next.
_POST_BASELINE_FLAG_ATTRS = ("dead_on_unwind", "dead_on_return", "writable")
_POST_BASELINE_CALL_ATTRS = ("initializes", "range", "captures", "nofpclass")
# Not every post-baseline spelling is a parameter attribute. Clang 23.1.1 emits
# `getelementptr inbounds nuw` on the member access in `cxx-object-model.cpp`;
# the no-wrap flags on a GEP arrived after the corpus's LLVM 18 floor, and the
# floor's assembler stops at `expected type` on the flag. They are stripped for
# the same reason the attributes above are: a refinement on an address
# computation the snapshot already shows, in a chapter about name mangling and
# object layout. The order matters -- `nusw` before `nuw`, or the shorter name
# eats the prefix of the longer one.
_POST_BASELINE_GEP_FLAGS = ("nusw", "nuw")
_POST_BASELINE_GEP_RE = re.compile(
    r"\b(getelementptr(?: inbounds)?) (?:(?:" + "|".join(_POST_BASELINE_GEP_FLAGS) + r") )+"
)
_POST_BASELINE_FLAG_RE = re.compile(r"\b(?:" + "|".join(_POST_BASELINE_FLAG_ATTRS) + r")\b[ ]?")
# Only these carry parameter attributes; restricting the substitution keeps it
# from touching an identifier or a comment that happens to share a keyword.
_ATTRIBUTE_BEARING_LINE = re.compile(r"^(?:define|declare)\b|\b(?:call|invoke) |\bgetelementptr\b")


@dataclass(frozen=True)
class Claim:
    """One structural fact the chapter asserts about emitted IR."""

    description: str
    pattern: str
    present: bool = True
    scope: str | None = None  # limit the search to one function body


@dataclass(frozen=True)
class Case:
    source: str
    triple: str
    opt_level: str = "-O0"
    snapshot: str | None = None
    extra_args: tuple[str, ...] = ()
    claims: tuple[Claim, ...] = field(default=())

    @property
    def label(self) -> str:
        return f"{self.source} [{self.triple} {self.opt_level}]"


CASES: tuple[Case, ...] = (
    Case(
        source="struct-layout.c",
        triple="x86_64-unknown-linux-gnu",
        snapshot="struct-layout.ll",
        claims=(
            Claim(
                "natural padding is left to the data layout, not materialized",
                r"%struct\.Padded = type \{ i8, i32, i8 \}",
            ),
            Claim(
                "__attribute__((packed)) selects the packed struct kind",
                r"%struct\.Packed = type <\{ i8, i32, i8 \}>",
            ),
            Claim(
                "a union becomes storage for its largest member",
                r"%union\.Word = type \{ i32 \}",
            ),
            Claim(
                "a packed field load drops to align 1",
                r"load i32, ptr %\d+, align 1",
                scope="packed_value",
            ),
            Claim(
                "a bit-field read is a shift and a mask",
                r"lshr i32 .*\n\s*%\d+ = and i32",
                scope="bits_count",
            ),
            Claim(
                "a bit-field write is a read-modify-write",
                r"\bor i32\b",
                scope="bits_set_count",
            ),
            Claim(
                "IR has no bit-field concept at all",
                r"bitfield",
                present=False,
            ),
            Claim(
                "array parameters with a static bound carry dereferenceable",
                r"dereferenceable\(32\)",
                scope="array_element",
            ),
        ),
    ),
    Case(
        source="abi-boundary.c",
        triple="x86_64-unknown-linux-gnu",
        snapshot="abi-boundary.ll",
        claims=(
            Claim(
                "an 8-byte struct is coerced into one integer register",
                r"@take_small\(i64 ",
            ),
            Claim(
                "a 12-byte struct is split across two registers",
                r"@take_medium\(i64 [^,]+, i32 ",
            ),
            Claim(
                "a 64-byte struct is passed byval on SysV",
                r"@take_large\(ptr[^)]*byval\(%struct\.Large\)",
            ),
            Claim(
                "a large return becomes an sret out-parameter",
                r"@return_large\(ptr[^)]*sret\(%struct\.Large\)",
            ),
            Claim(
                "narrow integers are sign/zero extended by the caller on SysV",
                r"signext i8 @narrow\(i8 noundef signext ",
            ),
            Claim(
                "_Bool is zero-extended, not sign-extended",
                r"i1 noundef zeroext ",
                scope="narrow",
            ),
        ),
    ),
    Case(
        source="abi-boundary.c",
        triple="aarch64-unknown-linux-gnu",
        claims=(
            Claim(
                "the same 12-byte struct becomes an array of registers on AArch64",
                r"@take_medium\(\[2 x i64\] ",
            ),
            Claim(
                "AArch64 does not use byval for the 64-byte struct",
                r"@take_large\(ptr[^)]*byval",
                present=False,
            ),
            Claim(
                "AArch64 does not ask the caller to extend narrow integers",
                r"@narrow\(i8 noundef signext",
                present=False,
            ),
            Claim(
                "the sret rule is shared between the two ABIs",
                r"@return_large\(ptr[^)]*sret\(%struct\.Large\)",
            ),
        ),
    ),
    Case(
        source="control-flow-lowering.c",
        triple="x86_64-unknown-linux-gnu",
        snapshot="control-flow-lowering.ll",
        claims=(
            Claim(
                "&& is lowered to a branch, because the right operand may not run",
                r"br i1 ",
                scope="guarded_load",
            ),
            Claim(
                "&& is never lowered to a bitwise and of both operands",
                r"\band i1\b",
                present=False,
                scope="guarded_load",
            ),
            Claim(
                "a dense-enough switch stays a switch instruction",
                r"switch i32 ",
                scope="dispatch",
            ),
            Claim(
                "volatile survives as an instruction flag, not a type",
                r"load volatile i32",
                scope="read_twice",
            ),
            Claim(
                "at -O0 every local is a stack slot",
                r"alloca ",
                scope="loop_sum",
            ),
        ),
    ),
    Case(
        source="control-flow-lowering.c",
        triple="x86_64-unknown-linux-gnu",
        opt_level="-O1",
        claims=(
            Claim(
                "at -O1 the stack slots are promoted away",
                r"alloca ",
                present=False,
                scope="loop_sum",
            ),
            Claim(
                "volatile accesses are still not merged at -O1",
                r"load volatile i32",
                scope="read_twice",
            ),
        ),
    ),
    Case(
        source="cxx-object-model.cpp",
        triple="x86_64-unknown-linux-gnu",
        snapshot="cxx-object-model.ll",
        claims=(
            Claim(
                "C++ names reach IR Itanium-mangled",
                r"@_Z10total_areaRK5Shape\(",
            ),
            Claim(
                "overloads become distinct symbols",
                r"@_Z5scalei\(",
            ),
            Claim(
                "overload resolution happens before IR, not in it",
                r"@_Z5scaled\(",
            ),
            Claim(
                "a template instantiation is emitted linkonce_odr in a comdat",
                r"define linkonce_odr[^\n]*@_Z5twiceIiET_S0_",
            ),
            Claim(
                "a virtual call indexes a vtable of function pointers",
                # Tolerates both spellings: an assertions build keeps the
                # frontend's %vtable/%vfn names, a release build numbers them.
                r"getelementptr inbounds ptr, ptr %\w+, i64 \d+",
                scope="_Z10total_areaRK5Shape",
            ),
            Claim(
                "the virtual call target is a loaded value, not a symbol",
                r"call noundef i32 %\w+\(",
                scope="_Z10total_areaRK5Shape",
            ),
            Claim(
                "a temporary's destructor is called by the frontend",
                r"@_ZN5GuardD1Ev",
                scope="_Z7guardedv",
            ),
        ),
    ),
)


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def find_clang(name: str) -> str | None:
    if suffix := os.environ.get("LLVM_SUFFIX"):
        if found := shutil.which(f"{name}{suffix}"):
            return found
    if found := shutil.which(name):
        return found
    for major in range(30, 14, -1):
        if found := shutil.which(f"{name}-{major}"):
            return found
    return None


def strip_balanced_call(line: str, name: str) -> str:
    """Remove `name(...)` including nested parens, e.g. `initializes((0, 4))`."""
    while (match := re.search(rf"\b{re.escape(name)}\(", line)) is not None:
        depth = 0
        for index in range(match.end() - 1, len(line)):
            if line[index] == "(":
                depth += 1
            elif line[index] == ")":
                depth -= 1
                if depth == 0:
                    tail = index + 1
                    if tail < len(line) and line[tail] == " ":
                        tail += 1
                    line = line[: match.start()] + line[tail:]
                    break
        else:  # unbalanced: leave it alone rather than corrupt the line
            return line
    return line


def strip_post_baseline_attrs(line: str) -> str:
    """Drop inline attributes newer than the corpus's LLVM >= 18 baseline."""
    if not _ATTRIBUTE_BEARING_LINE.search(line):
        return line
    # Instruction lines are indented; the tidy-up below collapses runs of
    # spaces, so hold the indent aside rather than letting it be eaten.
    indent = line[: len(line) - len(line.lstrip(" "))]
    body = line[len(indent) :]

    # Parenthesized forms first: a flag that has since gained an argument must
    # lose the argument with it, or the bare-name substitution orphans `(...)`.
    for name in _POST_BASELINE_CALL_ATTRS + _POST_BASELINE_FLAG_ATTRS:
        body = strip_balanced_call(body, name)
    body = _POST_BASELINE_FLAG_RE.sub("", body)
    body = _POST_BASELINE_GEP_RE.sub(r"\1 ", body)

    # A removed attribute can leave `(  `, ` )`, ` ,` or a doubled space behind.
    # The comma case is not hypothetical: LLVM replaced `nocapture` with
    # `captures(none)`, so stripping the newer spelling from
    # `ptr noalias nocapture writeonly, ...` leaves `writeonly ,` -- normalized
    # text that no assembler minds and no human would write.
    body = re.sub(r"\( +", "(", body)
    body = re.sub(r" +\)", ")", body)
    body = re.sub(r" +,", ",", body)
    return indent + re.sub(r"  +", " ", body)


def normalize(text: str) -> str:
    """Strip version noise so a snapshot reads as lowering, not as build config."""
    out: list[str] = []
    for line in text.splitlines():
        if _DROP_LINE.match(line):
            continue
        line = _METADATA_SUFFIX.sub("", line)
        line = _ATTR_GROUP_REF.sub("", line)
        line = strip_post_baseline_attrs(line)
        line = line.rstrip()
        if not line and out and not out[-1]:
            continue
        out.append(line)
    while out and not out[-1]:
        out.pop()
    return "\n".join(out) + "\n"


def find_baseline_assembler() -> tuple[str, int] | None:
    """The OLDEST `llvm-as` on PATH at or above the corpus baseline.

    A snapshot checked only against the newest assembler is a snapshot nobody
    has checked: CI runs jobs on different LLVM majors, and the strictest one
    decides whether the corpus's declared baseline actually holds.

    Both `llvm-as-N` and a bare `llvm-as` are considered -- the latter by asking it
    its version, because a name is not a version and a host may ship only the
    unsuffixed binary.

    When the host offers only a recent assembler this check cannot bite, and it
    says nothing rather than claiming a baseline it did not test. The backstop
    is CI's host-portability job, which runs `verify-examples.sh` over the same
    snapshots on an older LLVM than the training job installs -- that is where
    this defect was caught the first time.
    """
    candidates: list[tuple[int, str]] = []
    for major in range(BASELINE_LLVM_MAJOR, 31):
        if found := shutil.which(f"llvm-as-{major}"):
            candidates.append((major, found))

    # The unsuffixed `llvm-as` counts too, and asking it is not optional. Searching only
    # for versioned names says "nothing at or above the baseline" on a host whose ONLY
    # assembler is `llvm-as` at exactly the baseline major -- a perfectly good baseline
    # host, reported as having no baseline at all. That is a false negative on the plain
    # skip path and, once --require-baseline exists, a false FAILURE on the job that owns
    # the floor. A name is not a version; the binary is asked which major it is.
    if plain := shutil.which("llvm-as"):
        try:
            proc = subprocess.run(
                [plain, "--version"], capture_output=True, text=True, check=False, timeout=60
            )
        except (OSError, subprocess.TimeoutExpired):
            proc = None
        if proc is not None:
            match = re.search(r"LLVM version (\d+)\.", proc.stdout)
            if match and (major := int(match.group(1))) >= BASELINE_LLVM_MAJOR:
                candidates.append((major, plain))

    if not candidates:
        return None
    # Oldest wins; a suffixed name is preferred at the same major because it is the one a
    # reader can reproduce from the chapter's own commands.
    major, binary = min(candidates, key=lambda pair: (pair[0], pair[1].endswith("llvm-as")))
    return binary, major


_SNAPSHOT_MAJOR_RE = re.compile(r"^; Produced by clang (\d+)\.", re.MULTILINE)


def snapshot_major(path: Path) -> int | None:
    """The clang major a snapshot records, or None for one written before this."""
    match = _SNAPSHOT_MAJOR_RE.search(path.read_text(encoding="utf-8"))
    return int(match.group(1)) if match else None


def clang_major(clang: str) -> int | None:
    """The major of the clang actually being run, read from the binary itself."""
    try:
        completed = subprocess.run(
            [clang, "--version"], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"clang version (\d+)\.", completed.stdout)
    return int(match.group(1)) if match else None


def body_delta(existing: str, rendered: str) -> int:
    """Differing line count between two snapshots, headers excluded."""
    strip = lambda text: text.split("\n\n", 1)[1] if "\n\n" in text else text  # noqa: E731
    left, right = strip(existing).splitlines(), strip(rendered).splitlines()
    return sum(
        1
        for line in difflib.unified_diff(left, right, lineterm="", n=0)
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    )


def check_snapshot_baseline(rendered: str, label: str) -> str | None:
    """Assemble a normalized snapshot at the baseline; return a failure message.

    This is the total check behind the declared strip list: an attribute newer
    than that list fails here, named by a real assembler, instead of reaching a
    CI job on an older toolchain than the one that produced it.
    """
    assembler = find_baseline_assembler()
    if assembler is None:
        return None
    binary, major = assembler
    # `major` is the oldest assembler at or above the floor, which is not always the floor
    # itself: on a host carrying only LLVM 23 this check runs at 23. Saying "the LLVM 23
    # baseline" there would be a lie about which claim was tested, so the declared floor is
    # named separately whenever the two differ.
    at_floor = major == BASELINE_LLVM_MAJOR
    with tempfile.NamedTemporaryFile("w", suffix=".ll", delete=False) as handle:
        handle.write(rendered)
        temporary = Path(handle.name)
    try:
        completed = subprocess.run(
            [binary, str(temporary), "-o", os.devnull],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"{label}: baseline assembler {binary} could not be run ({exc})"
    finally:
        temporary.unlink(missing_ok=True)

    if completed.returncode == 0:
        return None
    detail = completed.stderr.strip().splitlines()
    first = detail[0] if detail else "(no diagnostic)"
    return (
        f"{label}: does not assemble at LLVM {major}"
        + (
            ""
            if at_floor
            else f" (the oldest assembler here; the declared baseline is {BASELINE_LLVM_MAJOR})"
        )
        + f": {first}\n"
        f"       If this is a parameter attribute newer than LLVM "
        f"{BASELINE_LLVM_MAJOR}, add it to _POST_BASELINE_FLAG_ATTRS or "
        f"_POST_BASELINE_CALL_ATTRS; if it is an instruction flag, to "
        f"_POST_BASELINE_GEP_FLAGS or its equivalent -- the assembler rejects any "
        f"construct newer than the floor, not only parameter attributes, so read "
        f"the named line before assuming which. Then regenerate with --update."
    )


def function_body(ir: str, name: str) -> str | None:
    """Return the text of one function definition, brace to brace."""
    marker = re.search(rf"^define[^\n]*@{re.escape(name)}\(", ir, re.MULTILINE)
    if marker is None:
        return None
    end = ir.find("\n}", marker.start())
    if end == -1:
        return None
    return ir[marker.start() : end + 2]


def compile_case(clang: str, clangxx: str, case: Case) -> tuple[str, str]:
    source = EXAMPLES / case.source
    driver = clangxx if source.suffix == ".cpp" else clang
    cmd = [
        driver,
        case.opt_level,
        "-S",
        "-emit-llvm",
        "-fno-discard-value-names" if source.suffix == ".cpp" else "-Wall",
        "-target",
        case.triple,
        str(source),
        "-o",
        "-",
    ]
    cmd.extend(case.extra_args)
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=180)
    if completed.returncode != 0:
        raise RuntimeError(f"clang failed for {case.label}:\n{completed.stderr.strip()}")
    return completed.stdout, " ".join(cmd)


def check_case(ir: str, case: Case) -> list[str]:
    failures: list[str] = []
    for claim in case.claims:
        haystack = ir
        if claim.scope is not None:
            body = function_body(ir, claim.scope)
            if body is None:
                failures.append(
                    f"{claim.description}: function '{claim.scope}' not found in output"
                )
                continue
            haystack = body
        found = re.search(claim.pattern, haystack) is not None
        if found != claim.present:
            expectation = "expected" if claim.present else "expected NOT"
            failures.append(
                f"{claim.description}: {expectation} to match /{claim.pattern}/"
                + (f" in '{claim.scope}'" if claim.scope else "")
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-tools",
        action="store_true",
        help="fail instead of skipping when clang is unavailable",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite the normalized .ll snapshots from the current clang",
    )
    parser.add_argument(
        "--require-baseline",
        action="store_true",
        help=f"fail unless an llvm-as at the declared baseline major "
        f"({BASELINE_LLVM_MAJOR}) is present; pass this from the CI job that installs it",
    )
    args = parser.parse_args()

    # The floor claim needs an owner. Without this flag the baseline check is allowed to
    # find nothing and say nothing, which is how the corpus came to declare LLVM 15 while
    # no CI job had an llvm-as-15 to test it with. The job that installs the baseline
    # toolchain passes --require-baseline, and there absence is a defect, not a skip.
    if args.require_baseline:
        found = find_baseline_assembler()
        if found is None or found[1] != BASELINE_LLVM_MAJOR:
            got = "nothing at or above it" if found is None else f"only llvm-as-{found[1]}"
            print(
                f"frontend lowering gate: FAILED (--require-baseline: SEMVER.md declares "
                f"LLVM {BASELINE_LLVM_MAJOR} as the corpus baseline, but this host has "
                f"{got}. The job passing this flag is the one that is supposed to install "
                f"llvm-as-{BASELINE_LLVM_MAJOR}; a floor nothing assembles is not a floor)",
                file=sys.stderr,
            )
            return 1

    clang = find_clang("clang")
    clangxx = find_clang("clang++")
    if clang is None or clangxx is None:
        message = "clang/clang++ not on PATH"
        if args.require_tools:
            print(f"frontend lowering gate: FAILED ({message})", file=sys.stderr)
            return 1
        print(f"frontend lowering gate: SKIPPED ({message})")
        return 0

    version = subprocess.run(
        [clang, "--version"], capture_output=True, text=True, check=False
    ).stdout.splitlines()
    banner = version[0].strip() if version else "clang (unknown version)"
    print(f"frontend lowering gate: using {banner}")
    local_major = clang_major(clang)

    failed = 0
    claims_checked = 0

    for case in CASES:
        try:
            ir, cmd = compile_case(clang, clangxx, case)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            # A target the local clang was not built for is a capability gap,
            # not a lowering defect. Say so rather than reporting either a pass
            # or a lowering failure.
            text = str(exc)
            if "unable to create target" in text or "unknown target" in text:
                print(f"[skip] {case.label}: target unsupported by this clang")
                continue
            print(f"[FAIL] {case.label}: {text}", file=sys.stderr)
            failed += 1
            continue

        failures = check_case(ir, case)
        claims_checked += len(case.claims)
        if failures:
            failed += 1
            print(f"[FAIL] {case.label}", file=sys.stderr)
            print(f"       command: {cmd}", file=sys.stderr)
            for failure in failures:
                print(f"       - {failure}", file=sys.stderr)
        else:
            print(f"[ ok ] {case.label}: {len(case.claims)} claim(s)")

        if case.snapshot:
            snapshot_path = EXAMPLES / case.snapshot
            header = (
                f"; Normalized `clang {case.opt_level} -S -emit-llvm -target "
                f"{case.triple}` output for {case.source}.\n"
                f"; Produced by clang {local_major}. A host on that major must\n"
                "; reproduce this file byte for byte; a different major reports its\n"
                "; delta instead, because lowering legitimately moves between releases.\n"
                "; Regenerate with:\n"
                ";   python3 training/llvm/tools/verify-frontend-lowering.py --update\n"
                "; Attribute groups, module flags, the ident string, and parameter\n"
                "; attributes newer than LLVM 18 are stripped: they are build\n"
                "; configuration rather than lowering, and their spellings move between\n"
                "; releases faster than the corpus's LLVM >= 18 baseline allows.\n"
                "\n"
            )
            rendered = header + normalize(ir)

            # Ask the oldest available assembler, not the newest. This is the
            # check that catches an attribute the strip list does not know about.
            if baseline_failure := check_snapshot_baseline(rendered, relpath(snapshot_path)):
                print(f"[FAIL] {baseline_failure}", file=sys.stderr)
                failed += 1
            elif args.update:
                snapshot_path.write_text(rendered)
                print(f"[write] {relpath(snapshot_path)}")
            elif not snapshot_path.exists():
                print(
                    f"[FAIL] missing snapshot {relpath(snapshot_path)}; run with --update",
                    file=sys.stderr,
                )
                failed += 1
            else:
                # Without this, nothing ever compared the checked-in snapshot to what
                # the toolchain produces: the claims are checked against freshly
                # compiled IR and the baseline check assembles the fresh rendering, so
                # a snapshot could rot into fiction while the gate stayed green. It is
                # enforced only on the major that produced the file -- a different
                # major reports its delta, because clang 23 spells `nocapture` as
                # `captures(none)` and materializes a bool load as `icmp ne` where 18
                # used `trunc`, and neither is drift for anyone to fix.
                recorded = snapshot_major(snapshot_path)
                existing = snapshot_path.read_text(encoding="utf-8")
                if recorded is None:
                    # An unstamped snapshot is one nothing can enforce: without a
                    # producer there is no major to match, so the identity check
                    # below would quietly never apply. Losing the stamp must cost
                    # something, or dropping one line disables the check.
                    print(
                        f"[FAIL] {relpath(snapshot_path)}: records no producing clang; "
                        "rerun with --update so a host on that major can be held to it",
                        file=sys.stderr,
                    )
                    failed += 1
                elif recorded == local_major:
                    if drift := body_delta(existing, rendered):
                        print(
                            f"[FAIL] {relpath(snapshot_path)}: clang {local_major} produced "
                            f"this snapshot, and no longer reproduces it ({drift} line(s) "
                            "differ); rerun with --update and read what changed",
                            file=sys.stderr,
                        )
                        failed += 1
                elif drift := body_delta(existing, rendered):
                    print(
                        f"[note] {relpath(snapshot_path)}: recorded under clang {recorded}, "
                        f"read here with clang {local_major}: {drift} line(s) differ"
                    )

    if failed:
        print(
            f"frontend lowering gate: FAILED ({failed} case(s), {claims_checked} claim(s) checked)",
            file=sys.stderr,
        )
        return 1

    print(f"frontend lowering gate: PASSED ({len(CASES)} case(s), {claims_checked} claim(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
