#!/usr/bin/env python3
"""Exercise the MLIR infrastructure claims of chapter 24 against a real mlir-opt.

Chapter 24 teaches how MLIR is put together rather than what any one dialect does: the
region/block structure, the two assembly syntaxes, interfaces, the pass manager's
anchoring rules, the cast a partial lowering leaves behind, and bytecode. Every one of
those is a claim about *behaviour*, so every one of them is run here rather than asserted.

The interesting checks are the negative ones. A pass pipeline that names a function-scoped
pass at module level must FAIL, and a pipeline whose anchor op name does not exist must be
ACCEPTED while doing nothing -- that asymmetry is the trap the chapter exists to warn
about, and a gate that only tested the happy path would not notice if it went away.

Without mlir-opt this reports a skip and names its owner: CI's `LLVM training corpus
(LLVM 23)` job installs the toolchain, and passes --require-tools so that absence there is
a failure rather than a quiet pass.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llvm_toolchain import find_llvm_tool  # noqa: E402

TOOLS = Path(__file__).resolve().parent
TRAINING_ROOT = TOOLS.parent
EXAMPLES = TRAINING_ROOT / "24-mlir-infrastructure" / "examples"

# MLIR bytecode files begin with this magic. Checked rather than described, because the
# chapter tells a reader to recognise a .mlirbc by it.
BYTECODE_MAGIC = b"ML\xefR"


class Report:
    def __init__(self) -> None:
        self.findings: list[str] = []
        self.checks = 0

    def require(self, condition: object, message: str) -> bool:
        self.checks += 1
        if not condition:
            self.findings.append(message)
            return False
        return True

    def verdict(self, label: str) -> int:
        if self.findings:
            print(f"{label}: FAILED", file=sys.stderr)
            for finding in self.findings:
                print(f"  - {finding}", file=sys.stderr)
            return 1
        print(f"{label}: PASSED ({self.checks} checks)")
        return 0


def run(mlir_opt: str, args: list[str], stdin_path: Path | None = None) -> tuple[int, str]:
    """Run mlir-opt, returning (exit code, stdout+stderr). Never raises on tool failure."""
    cmd = [mlir_opt, *args]
    if stdin_path is not None:
        cmd.append(str(stdin_path))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, f"{exc}"
    return proc.returncode, proc.stdout + proc.stderr


def check_generic_form(report: Report, mlir_opt: str, work: Path) -> None:
    """Custom syntax and generic syntax are two spellings of one module."""
    source = EXAMPLES / "regions-and-blocks.mlir"
    code, generic = run(mlir_opt, ["--mlir-print-op-generic"], source)
    if not report.require(
        code == 0, f"mlir-opt could not print {source.name} generically: {generic[:200]}"
    ):
        return

    report.require(
        '"func.func"()' in generic and '"builtin.module"()' in generic,
        "the generic form does not spell operations as quoted names with an operand list; "
        "chapter 24 teaches that shape as the one an unregistered dialect must use",
    )
    # Inherent attributes print inside <{...}> and discardable ones outside it. The chapter
    # draws that distinction, so the syntax it rests on is checked.
    report.require(
        "<{" in generic,
        "the generic form contains no `<{...}>` group, so the inherent/discardable "
        "attribute distinction chapter 24 teaches is not visible in this output",
    )

    # The two syntaxes must round-trip into the same module, or they are not two spellings
    # of one thing.
    generic_file = work / "generic.mlir"
    generic_file.write_text(generic, encoding="utf-8")
    code_a, custom_from_generic = run(mlir_opt, [], generic_file)
    code_b, custom_direct = run(mlir_opt, [], source)
    report.require(
        code_a == 0 and code_b == 0 and custom_from_generic == custom_direct,
        "parsing the generic form back does not reproduce the module the custom form "
        "parses to; the two syntaxes are supposed to be interchangeable",
    )
    print("[syntax]  custom and generic forms round-trip to the same module")


def check_unregistered_dialect(report: Report, mlir_opt: str, work: Path) -> None:
    """Generic syntax is necessary to carry an unknown dialect; it is not sufficient.

    Chapter 01 first said a generic-form file "can be parsed by an mlir-opt that knows
    nothing about the dialect at all". It cannot: the parser refuses until the caller
    either passes --allow-unregistered-dialect or registers the dialect. A reader following
    that sentence would have run a command that rejects the file the chapter said it would
    read, so the corrected claim is checked here rather than trusted.
    """
    source = work / "unregistered.mlir"
    source.write_text(
        '"builtin.module"() ({\n  "mydialect.myop"() : () -> ()\n}) : () -> ()\n',
        encoding="utf-8",
    )

    code, out = run(mlir_opt, [], source)
    report.require(
        code != 0 and "unregistered dialect" in out,
        "generic syntax alone parsed an operation from a dialect mlir-opt does not know. "
        "Chapter 01 teaches that it does NOT, and that --allow-unregistered-dialect or a "
        "registered dialect is required -- if MLIR has changed, rewrite the chapter "
        f"(got exit {code}: {out[:160]})",
    )

    code, out = run(mlir_opt, ["--allow-unregistered-dialect"], source)
    report.require(
        code == 0,
        f"--allow-unregistered-dialect did not parse the generic-form module: {out[:200]}",
    )
    print("[unreg]   generic syntax needs --allow-unregistered-dialect, as the chapter says")


def check_pass_anchoring(report: Report, mlir_opt: str) -> None:
    """A pass runs on the op it is written for, and the pipeline says which."""
    source = EXAMPLES / "regions-and-blocks.mlir"

    # A function-scoped pass named at module level is a configuration error.
    code, out = run(mlir_opt, ["--pass-pipeline=builtin.module(affine-scalrep)"], source)
    report.require(
        code != 0 and "failed to add" in out,
        "naming the function-scoped pass `affine-scalrep` at module level was accepted; "
        "chapter 24 teaches that a pass must be nested under the operation it runs on, and "
        f"this is the error that teaches it (got exit {code}: {out[:160]})",
    )

    # Nested correctly, it is accepted.
    code, out = run(mlir_opt, ["--pass-pipeline=builtin.module(func.func(affine-scalrep))"], source)
    report.require(
        code == 0,
        f"`affine-scalrep` nested under func.func was rejected: {out[:200]}",
    )

    # The trap: an anchor naming an operation that does not exist is ACCEPTED and simply
    # never matches, so a typo silently runs nothing. If MLIR ever starts rejecting this,
    # the chapter's warning becomes wrong and should be rewritten -- so the current
    # behaviour is pinned rather than assumed.
    code, out = run(mlir_opt, ["--pass-pipeline=builtin.module(no.such_op(canonicalize))"], source)
    report.require(
        code == 0,
        "a pipeline anchored on the non-existent operation `no.such_op` was rejected. "
        "That is an improvement in MLIR, but chapter 24 currently warns that such a "
        "pipeline is accepted and silently runs nothing -- update the chapter",
    )
    print("[passes]  anchoring is enforced; a non-existent anchor is still accepted silently")


def check_interfaces(report: Report, mlir_opt: str) -> None:
    """Generic passes work through interfaces, not through dialect knowledge."""
    source = EXAMPLES / "interfaces-inlining.mlir"

    # Anti-vacuity, and it was needed: checking only the POST-inline state passes happily on
    # an example with no call in it at all. Injecting exactly that -- replacing the call with
    # plain arithmetic -- left this check green while it proved nothing about interfaces. The
    # precondition has to be asserted before the result means anything.
    before = source.read_text(encoding="utf-8")
    if not report.require(
        "call @callee" in before,
        f"{source.name} contains no call to @callee, so running --inline over it cannot "
        f"demonstrate anything about CallOpInterface; the example lost its subject",
    ):
        return

    code, out = run(mlir_opt, ["--inline"], source)
    if not report.require(code == 0, f"--inline failed on {source.name}: {out[:200]}"):
        return
    report.require(
        "call @callee" not in out,
        "--inline left the call in place; chapter 24 uses this as its demonstration that "
        "the inliner reaches func.func through CallOpInterface/CallableOpInterface rather "
        "than by knowing the func dialect",
    )
    report.require(
        "arith.addi" in out,
        "--inline removed the call but the callee's body did not appear in the caller",
    )
    print("[iface]   --inline reaches func.func purely through its interfaces")


def check_unrealized_casts(report: Report, mlir_opt: str) -> None:
    """A cancelling cast pair reconciles; a lone cast survives and is the signal."""
    source = EXAMPLES / "unrealized-casts.mlir"

    # Same anti-vacuity precondition as the inlining check, for the same reason: the two
    # assertions below are about what SURVIVES reconciliation, and both are satisfied
    # trivially by an example containing no casts at all. Assert the subject exists first.
    before = source.read_text(encoding="utf-8")
    pair_g = (
        before.split("func.func @g")[-1].split("func.func")[0] if "func.func @g" in before else ""
    )
    lone_f = before.split("func.func @f")[-1] if "func.func @f" in before else ""
    ok = report.require(
        pair_g.count("unrealized_conversion_cast") >= 2,
        f"{source.name}: @g no longer contains a cancelling pair of casts, so 'the pair "
        f"folds' is checked against nothing",
    )
    ok &= report.require(
        lone_f.count("unrealized_conversion_cast") == 1,
        f"{source.name}: @f no longer contains exactly one unpaired cast, so 'a lone cast "
        f"survives' is checked against nothing",
    )
    if not ok:
        return

    code, out = run(mlir_opt, ["--reconcile-unrealized-casts"], source)
    if not report.require(code == 0, f"--reconcile-unrealized-casts failed: {out[:200]}"):
        return

    body_g = out.split("func.func @g")[-1].split("func.func")[0] if "func.func @g" in out else ""
    report.require(
        "unrealized_conversion_cast" not in body_g,
        "the cancelling i32->i64->i32 cast pair in @g survived reconciliation; chapter 24 "
        "teaches that a pair which cancels is folded away",
    )
    body_f = out.split("func.func @f")[-1] if "func.func @f" in out else ""
    report.require(
        "unrealized_conversion_cast" in body_f,
        "the lone cast in @f was removed by reconciliation; chapter 24 teaches that a cast "
        "with no partner SURVIVES, and that surviving cast is how an unfinished lowering "
        "announces itself",
    )
    print("[casts]   a cancelling pair folds; a lone cast survives, as the chapter claims")


def check_bytecode(report: Report, mlir_opt: str, work: Path) -> None:
    """Bytecode is a second serialization of the same module, not a different module."""
    source = EXAMPLES / "regions-and-blocks.mlir"
    bc = work / "module.mlirbc"
    code, out = run(mlir_opt, ["--emit-bytecode", "-o", str(bc)], source)
    if not report.require(code == 0 and bc.is_file(), f"--emit-bytecode failed: {out[:200]}"):
        return

    report.require(
        bc.read_bytes().startswith(BYTECODE_MAGIC),
        f"the emitted bytecode does not begin with {BYTECODE_MAGIC!r}; chapter 24 tells a "
        f"reader to recognise a .mlirbc by that magic",
    )

    code_text, from_text = run(mlir_opt, [], source)
    code_bc, from_bytecode = run(mlir_opt, [], bc)
    report.require(
        code_text == 0 and code_bc == 0 and from_text == from_bytecode,
        "reading the module back from bytecode does not produce the same textual module; "
        "bytecode is supposed to be a different encoding of the same IR, not a lossy one",
    )
    print("[bytes]   bytecode carries the magic and round-trips to the same module")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-tools",
        action="store_true",
        help="fail rather than skip when mlir-opt is absent (pass this from the CI job "
        "that installs it)",
    )
    args = parser.parse_args(argv)

    mlir_opt = find_llvm_tool("mlir-opt")
    if mlir_opt is None:
        if args.require_tools:
            print(
                "error: --require-tools was passed but mlir-opt is absent; the job that "
                "installs the MLIR toolchain is the one that owns these checks",
                file=sys.stderr,
            )
            return 1
        print(
            "[skip]    mlir-opt is absent; chapter 24's behaviour checks belong to the CI "
            "job that installs the MLIR toolchain",
            file=sys.stderr,
        )
        return 0

    report = Report()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        check_generic_form(report, mlir_opt, work)
        check_unregistered_dialect(report, mlir_opt, work)
        check_pass_anchoring(report, mlir_opt)
        check_interfaces(report, mlir_opt)
        check_unrealized_casts(report, mlir_opt)
        check_bytecode(report, mlir_opt, work)

    # Anti-vacuity: every check above is guarded by an early return on a tool failure, so a
    # broken mlir-opt could otherwise produce a green run with almost nothing examined.
    report.require(
        report.checks >= 12,
        f"only {report.checks} checks ran; chapter 24 has more claims than that, so the "
        f"run stopped early and this verdict covers less than it appears to",
    )
    return report.verdict("mlir infrastructure gate")


if __name__ == "__main__":
    sys.exit(main())
