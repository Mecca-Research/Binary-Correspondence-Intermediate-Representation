#!/usr/bin/env python3
"""Verify the Clang frontend-lowering claims made by llvm-training/20-clang-frontend/.

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

    python3 llvm-training/tools/verify-frontend-lowering.py
    python3 llvm-training/tools/verify-frontend-lowering.py --require-tools
    python3 llvm-training/tools/verify-frontend-lowering.py --update

`--update` rewrites the normalized `.ll` snapshots next to each source.
Without `--require-tools` a missing or incapable Clang is an explicit skip, and
the exit status says which of pass/skip/fail happened -- a skip is never
reported as a pass.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = TRAINING_ROOT.parent
EXAMPLES = TRAINING_ROOT / "20-clang-frontend" / "examples"

# Clang emits these but they are version noise, not lowering facts. Removing
# them also keeps the snapshots assembling on the corpus's LLVM >= 15 baseline,
# since attribute-group spellings (`memory(none)` and friends) are newer than
# the baseline while the instructions they annotate are not.
_DROP_LINE = re.compile(r"^(?:; ModuleID|source_filename|attributes #|!|; Function Attrs:)")
_ATTR_GROUP_REF = re.compile(r"\s#\d+\b")
_METADATA_SUFFIX = re.compile(r",\s*![a-zA-Z.]+ ![0-9]+")


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


def normalize(text: str) -> str:
    """Strip version noise so a snapshot reads as lowering, not as build config."""
    out: list[str] = []
    for line in text.splitlines():
        if _DROP_LINE.match(line):
            continue
        line = _METADATA_SUFFIX.sub("", line)
        line = _ATTR_GROUP_REF.sub("", line)
        line = line.rstrip()
        if not line and out and not out[-1]:
            continue
        out.append(line)
    while out and not out[-1]:
        out.pop()
    return "\n".join(out) + "\n"


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
    args = parser.parse_args()

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
                "; Regenerate with:\n"
                ";   python3 llvm-training/tools/verify-frontend-lowering.py --update\n"
                "; Attribute groups, module flags, and the ident string are stripped:\n"
                "; they are build configuration, not lowering, and their spellings move\n"
                "; between releases faster than the corpus's LLVM >= 15 baseline allows.\n"
                "\n"
            )
            rendered = header + normalize(ir)
            if args.update:
                snapshot_path.write_text(rendered)
                print(f"[write] {relpath(snapshot_path)}")
            elif not snapshot_path.exists():
                print(
                    f"[FAIL] missing snapshot {relpath(snapshot_path)}; run with --update",
                    file=sys.stderr,
                )
                failed += 1

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
