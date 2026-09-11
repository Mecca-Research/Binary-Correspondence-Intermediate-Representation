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
import re
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

# The chapter that displays the serialization sizes, and the examples it measures. The
# sizes are generated into it rather than typed, because a size typed into prose is a
# number measured once on a module nobody can name afterwards -- which is exactly how this
# chapter came to quote figures from a scratch file instead of its own example.
SIZE_CHAPTER = TRAINING_ROOT / "24-mlir-infrastructure" / "04-bytecode-and-partial-lowering.md"
CHAPTER_README = TRAINING_ROOT / "24-mlir-infrastructure" / "README.md"
SIZED_EXAMPLES = (
    "regions-and-blocks.mlir",
    "interfaces-inlining.mlir",
    "unrealized-casts.mlir",
)
# Both sides of the size table are measured with locations stripped. That is not tidying:
# `mlir-opt`'s default TEXT print drops locations while bytecode keeps them, so comparing
# the two as they come out of the tool compares a lossy encoding against a lossless one --
# and, because a location holds the source file's path, it also makes the byte count depend
# on where the repository happens to be checked out. check_location_cost() below pins both
# halves of that, so this pipeline is a fix with a witness rather than a workaround.
STRIP_LOCATIONS = "--pass-pipeline=builtin.module(strip-debuginfo)"

_SIZE_BLOCK = re.compile(
    r"(?P<open><!-- generated: serialization-sizes -->\n)(?P<body>.*?)(?P<close><!-- /generated -->)",
    re.DOTALL,
)


class Report:
    def __init__(self) -> None:
        self.findings: list[str] = []
        self.checks = 0
        self.completed: set[str] = set()

    def require(self, condition: object, message: str) -> bool:
        self.checks += 1
        if not condition:
            self.findings.append(message)
            return False
        return True

    def done(self, name: str) -> None:
        """Record that a check function reached its end rather than returning early."""
        self.completed.add(name)

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
    report.done("generic_form")


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
    report.done("unregistered_dialect")


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
    report.done("pass_anchoring")


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
    report.done("interfaces")


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
    report.done("unrealized_casts")


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
    report.done("bytecode")


# --------------------------------------------------------------------------
# The one table in chapter 24 that is a measurement, and is therefore generated
# --------------------------------------------------------------------------


def measure_serializations(mlir_opt: str, work: Path) -> dict[str, tuple[int, int]] | None:
    """Canonical text bytes and bytecode bytes for each example the chapter measures.

    Text is measured after a round trip through mlir-opt rather than from the file on
    disk: the source carries comments and whatever whitespace its author typed, and
    comparing that against bytecode would be comparing a comment budget against an
    encoding. What the chapter is contrasting is two serializations of the same module,
    so both sides are asked of the same tool -- and, per STRIP_LOCATIONS, of the same
    module rather than of two modules that differ in what they remember.
    """
    sizes: dict[str, tuple[int, int]] = {}
    for name in SIZED_EXAMPLES:
        source = EXAMPLES / name
        if not source.is_file():
            return None
        text_out = work / f"{name}.txt"
        bc_out = work / f"{name}.bc"
        code_text, _ = run(mlir_opt, [STRIP_LOCATIONS, "-o", str(text_out)], source)
        code_bc, _ = run(mlir_opt, [STRIP_LOCATIONS, "--emit-bytecode", "-o", str(bc_out)], source)
        if code_text != 0 or code_bc != 0:
            return None
        sizes[name] = (text_out.stat().st_size, bc_out.stat().st_size)
    return sizes


def render_size_block(sizes: dict[str, tuple[int, int]]) -> str:
    """The size table, built from a live measurement so no byte count is ever typed."""
    rows = [
        "| example | text | bytecode | bytecode is |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, (text_bytes, bc_bytes) in sizes.items():
        if bc_bytes < text_bytes:
            verdict = f"**smaller** by {text_bytes - bc_bytes} bytes"
        elif bc_bytes > text_bytes:
            verdict = f"**larger** by {bc_bytes - text_bytes} bytes"
        else:
            verdict = "the same size"
        rows.append(
            f"| [`examples/{name}`](examples/{name}) | {text_bytes} | {bc_bytes} | {verdict} |"
        )
    smaller = sum(1 for t, b in sizes.values() if b < t)
    larger = sum(1 for t, b in sizes.values() if b > t)
    rows.append("")
    rows.append(
        f"Measured on this chapter's own {len(sizes)} examples by the gate, with source "
        f"locations stripped from both sides first: bytecode is smaller on {smaller} of "
        f"them and larger on {larger}."
    )
    return "\n".join(rows) + "\n"


def sync_size_block(report: Report | None, sizes: dict[str, tuple[int, int]], update: bool) -> None:
    """The chapter's size table must be what this mlir-opt actually produces."""
    if not SIZE_CHAPTER.is_file():
        if report:
            report.require(False, f"{SIZE_CHAPTER} is missing; the sizes are shown nowhere")
        return
    text = SIZE_CHAPTER.read_text(encoding="utf-8")
    match = _SIZE_BLOCK.search(text)
    if match is None:
        if report:
            report.require(
                False,
                f"{SIZE_CHAPTER.name} has no `<!-- generated: serialization-sizes -->` "
                f"block; the chapter's size claim would then be typed prose, which is how "
                f"it came to quote a module that is not in this corpus",
            )
        return
    fresh = render_size_block(sizes)
    if update:
        if match.group("body") != fresh:
            SIZE_CHAPTER.write_text(
                text[: match.start("body")] + fresh + text[match.end("body") :], encoding="utf-8"
            )
            print(f"[update]  {SIZE_CHAPTER.name}")
        return
    if report:
        report.require(
            match.group("body") == fresh,
            f"{SIZE_CHAPTER.name}: the serialization-sizes block is stale; this mlir-opt "
            f"measures:\n" + fresh,
        )


def check_location_cost(report: Report, mlir_opt: str, work: Path) -> None:
    """Why the size table strips locations, pinned rather than asserted in a comment.

    Two facts, and the table above is wrong without both. Bytecode records source
    locations that the default textual print does not, and a location holds the path of
    the file it came from -- so the same module emitted from two different paths produces
    two different byte counts. A size table built on that would be measuring the checkout
    directory, and would fail on any machine whose path length differs from its author's.
    """
    source = EXAMPLES / SIZED_EXAMPLES[0]
    if not report.require(source.is_file(), f"{source} is missing"):
        return

    # Same bytes, two names of deliberately different length, in one directory so that the
    # difference between the two paths is exactly the difference between the two names.
    NAME_PADDING = 40
    short = work / "m.mlir"
    long = work / ("m" + "a" * NAME_PADDING + ".mlir")
    body = source.read_bytes()
    short.write_bytes(body)
    long.write_bytes(body)

    raw: list[int] = []
    stripped: list[int] = []
    for index, path in enumerate((short, long)):
        raw_out = work / f"loc-raw-{index}.bc"
        strip_out = work / f"loc-strip-{index}.bc"
        code_raw, _ = run(mlir_opt, ["--emit-bytecode", "-o", str(raw_out)], path)
        code_strip, _ = run(
            mlir_opt, [STRIP_LOCATIONS, "--emit-bytecode", "-o", str(strip_out)], path
        )
        if not report.require(
            code_raw == 0 and code_strip == 0, f"could not emit bytecode from {path.name}"
        ):
            return
        raw.append(raw_out.stat().st_size)
        stripped.append(strip_out.stat().st_size)

    # Not merely "different": one byte of bytecode per byte of path, because the path is
    # interned in the string table and paid for once however many locations cite it. That
    # exact equality is the claim the chapter makes, so it is the claim checked here -- an
    # inequality would also pass if the cost were quadratic in the path, which would mean
    # something quite different about the format.
    report.require(
        raw[1] - raw[0] == NAME_PADDING,
        f"a {NAME_PADDING}-character longer filename changed the bytecode by "
        f"{raw[1] - raw[0]} bytes rather than {NAME_PADDING}; the chapter says the source "
        f"path costs its own length exactly, interned once, and that is no longer true",
    )
    report.require(
        stripped[0] == stripped[1],
        f"stripping locations does not make the size path-independent ({stripped[0]} vs "
        f"{stripped[1]} bytes for the same module under two names); the generated size "
        f"table would then differ per checkout and fail on someone else's machine",
    )

    # The other half of the asymmetry: text drops what bytecode keeps.
    code_text, text_out = run(mlir_opt, [], source)
    if report.require(code_text == 0, "could not print the example textually"):
        report.require(
            "loc(" not in text_out,
            "the default textual print now emits loc(...), so text and bytecode no longer "
            "disagree about locations; the size table's strip step was justified by that "
            "disagreement and needs revisiting",
        )
    print(
        f"[loc]     bytecode embeds the source path ({raw[0]} vs {raw[1]} bytes by name "
        f"alone); stripping makes it {stripped[0]} either way"
    )
    report.done("location_cost")


def check_prose_counts(report: Report, mlir_opt: str, work: Path) -> None:
    """Two counts chapter 24 states in prose, held to what the toolchain says.

    Neither belongs in a generated block -- each is one number inside an English sentence
    that would read badly as a table -- but both are measurements of a live toolchain and
    both will drift: MLIR gains dialects every release, and the location count changes if
    anyone edits the example. So the sentence is pinned instead, phrase and all, and the
    failure names the sentence to write.
    """
    code, out = run(mlir_opt, ["--show-dialects"])
    if not report.require(code == 0 and "Available Dialects:" in out, "--show-dialects failed"):
        return
    listed = out.split("Available Dialects:", 1)[1]
    dialects = [d.strip() for d in listed.replace("\n", ",").split(",") if d.strip()]

    code_v, version = run(mlir_opt, ["--version"])
    match = re.search(r"LLVM version (\d+)\.", version)
    if not report.require(code_v == 0 and match, "could not read mlir-opt's version"):
        return
    assert match is not None
    major = match.group(1)

    if report.require(CHAPTER_README.is_file(), f"{CHAPTER_README} is missing"):
        sentence = f"MLIR {major} registers {len(dialects)} dialects"
        report.require(
            sentence in CHAPTER_README.read_text(encoding="utf-8"),
            f"24-mlir-infrastructure/README.md should say {sentence!r}; it is a count of "
            f"what this mlir-opt registers, and MLIR gains dialects every release",
        )

    # The location count behind "25 locations all cite the same path".
    source = EXAMPLES / SIZED_EXAMPLES[0]
    bc = work / "prose-count.bc"
    code_bc, _ = run(mlir_opt, ["--emit-bytecode", "-o", str(bc)], source)
    if not report.require(code_bc == 0, "could not emit bytecode for the location count"):
        return
    code_loc, printed = run(mlir_opt, ["--mlir-print-debuginfo"], bc)
    if not report.require(code_loc == 0, "could not print the module with debug info"):
        return
    locations = len(re.findall(r"^#loc", printed, re.MULTILINE))
    if report.require(SIZE_CHAPTER.is_file(), f"{SIZE_CHAPTER} is missing"):
        phrase = f"{locations} locations in that module"
        report.require(
            phrase in SIZE_CHAPTER.read_text(encoding="utf-8"),
            f"04-bytecode-and-partial-lowering.md should say {phrase!r}; it is a count of "
            f"what this example carries, and editing the example changes it",
        )
    print(f"[prose]   {len(dialects)} dialects and {locations} locations, as the chapters say")
    report.done("prose_counts")


def check_serialization_sizes(report: Report, mlir_opt: str, work: Path) -> None:
    """Both directions of the chapter's claim must be visible in its own examples."""
    sizes = measure_serializations(mlir_opt, work)
    if not report.require(
        sizes is not None,
        "could not measure text and bytecode sizes for chapter 24's examples",
    ):
        return
    assert sizes is not None

    # The chapter's claim is "bytecode is not automatically smaller". A corpus where every
    # example pointed the same way would make that claim unillustrated -- true, but taken
    # on trust. It is worth knowing if that ever happens, because then the chapter needs a
    # different example rather than a different sentence.
    report.require(
        any(bc < txt for txt, bc in sizes.values()) and any(bc > txt for txt, bc in sizes.values()),
        "every example in chapter 24 now serializes the same direction, so the chapter's "
        "'bytecode is not automatically smaller' claim is no longer demonstrated by its "
        "own examples; add one that points the other way rather than softening the text",
    )
    sync_size_block(report, sizes, update=False)
    print(
        "[sizes]   "
        + ", ".join(f"{n.split('.')[0]} {t}/{b}" for n, (t, b) in sizes.items())
        + " (text/bytecode)"
    )
    report.done("serialization_sizes")


# The check functions main() runs, named so that a run which stops early is detectable as
# a missing name rather than as a count nobody maintains. A numeric floor alone rots: the
# one here read `>= 12` while the honest path ran 22, so eight checks could have vanished
# without the anti-vacuity guard noticing.
CHECK_NAMES = (
    "generic_form",
    "unregistered_dialect",
    "pass_anchoring",
    "interfaces",
    "unrealized_casts",
    "bytecode",
    "location_cost",
    "serialization_sizes",
    "prose_counts",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-tools",
        action="store_true",
        help="fail rather than skip when mlir-opt is absent (pass this from the CI job "
        "that installs it)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite the chapter's generated serialization-size block from a live measurement",
    )
    args = parser.parse_args(argv)

    mlir_opt = find_llvm_tool("mlir-opt")
    if mlir_opt is None:
        if args.update:
            # --update without a toolchain would rewrite nothing and exit 0, which reads
            # as "the block is current". The block IS the measurement; refusing is the
            # only honest answer.
            print(
                "error: --update needs mlir-opt, because the block it writes is a "
                "measurement rather than a rendering of checked-in data",
                file=sys.stderr,
            )
            return 1
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

    if args.update:
        with tempfile.TemporaryDirectory() as tmp:
            sizes = measure_serializations(mlir_opt, Path(tmp))
            if sizes is None:
                print("error: could not measure chapter 24's examples", file=sys.stderr)
                return 1
            sync_size_block(None, sizes, update=True)
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
        check_location_cost(report, mlir_opt, work)
        check_serialization_sizes(report, mlir_opt, work)
        check_prose_counts(report, mlir_opt, work)

    # Anti-vacuity, as a state rather than a count. Every check above returns early on a
    # tool failure, so a broken mlir-opt could otherwise produce a green run with almost
    # nothing examined. Each one records its own name on reaching the end; a name missing
    # here says exactly which claim went unexamined, which a threshold cannot.
    missing = sorted(set(CHECK_NAMES) - report.completed)
    report.require(
        not missing,
        f"these checks did not run to completion: {', '.join(missing)}; the run stopped "
        f"early and this verdict covers less than it appears to",
    )
    return report.verdict("mlir infrastructure gate")


if __name__ == "__main__":
    sys.exit(main())
