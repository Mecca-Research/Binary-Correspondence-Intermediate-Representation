#!/usr/bin/env python3
"""Hold the corpus to the LLVM surface it claims to teach, across two releases.

The corpus tracks a moving language. Between two LLVM majors instructions arrive,
attributes arrive and are withdrawn, and intrinsics graduate out of
`llvm.experimental.*` under a new name. None of that is visible to a gate that only
assembles examples: IR written for the older major keeps assembling on the newer one,
so a corpus can fall a whole release behind while every check stays green.

This reads the surface from the toolchain's own generated definitions -- `Instruction.def`,
`Attributes.td` and `IntrinsicEnums.inc`, the files TableGen writes and LangRef documents --
rather than from a changelog, and requires every item that moved between the two majors to
have a disposition: a chapter that teaches it, or a written reason it is out of scope.

Three modes, because no single CI job can see both majors at once:

  --emit-surface N     write reference/llvm-surface-N.json from the installed LLVM N
  --require-surface N  fail, rather than skip, when LLVM N's headers are absent; the CI
                       job that installs them passes this
  --update             rewrite the generated blocks the chapters display
  (default)            derive the delta from the two checked-in snapshots and check that
                       every item in it is disposed. The drift check against a real
                       toolchain runs here too for whichever majors are installed, and
                       says so when one is not -- needs no toolchain to run at all

That split is deliberate. The job that installs LLVM 18 owns the 18 snapshot, the job that
installs 23 owns the 23 one, and the delta itself is pure data both can be checked against.

Target-prefixed intrinsic names (`llvm.aarch64.*`, `llvm.x86.*`, ...) are excluded from
every set. LLVM 18 emitted four SVE `pmov` intrinsics into the common enum and LLVM 23 does
not, so comparing the raw files would report them as withdrawn from the language. They were
never part of the target-independent surface LangRef describes, and a delta that called them
a LangRef change would be wrong.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_text import code_spans, corpus_code, teaching_text  # noqa: E402

TOOLS = Path(__file__).resolve().parent
TRAINING_ROOT = TOOLS.parent
REFERENCE = TRAINING_ROOT / "reference"
DISPOSITIONS = REFERENCE / "langref-delta-dispositions.json"
ATTRIBUTE_DISPOSITIONS = REFERENCE / "langref-attribute-dispositions.json"
INTRINSIC_DISPOSITIONS = REFERENCE / "langref-intrinsic-dispositions.json"

# The two majors the corpus spans: the baseline it enforces, and the release it tracks.
BASELINE_MAJOR = 18
CURRENT_MAJOR = 23

# Names under these prefixes belong to a target, not to the language. See the module
# docstring: LLVM 18 leaked four of them into the common enum and LLVM 23 does not.
_TARGET_PREFIXES = (
    "aarch64",
    "amdgcn",
    "arm",
    "bpf",
    "dx",
    "dxil",
    "hexagon",
    "loongarch",
    "mips",
    "nvvm",
    "ppc",
    "r600",
    "riscv",
    "s390",
    "spv",
    "ve",
    "wasm",
    "x86",
    "xcore",
)
_TARGET_INTRINSIC = re.compile(r"^llvm\.(?:" + "|".join(_TARGET_PREFIXES) + r")\.")

# `Br` became `CondBr`/`UncondBr` between 18 and 23. That is a split of the C++ opcode
# enum, not a language change -- IR spells both `br` before and after -- so the gate must
# not report it as an arriving or departing instruction.
INTERNAL_OPCODE_SPLITS = {"Br": ("CondBr", "UncondBr")}

_HANDLE_INST = re.compile(r"^HANDLE_[A-Z_]*INST *\( *[0-9]+ *, *([A-Za-z0-9]+) *,", re.MULTILINE)
# Attributes.td wraps a declaration whose name is long:
#
#     def NoCreateUndefOrPoison
#         : EnumAttr<"nocreateundeforpoison", IntersectAnd, [FnAttr]>;
#
# so the pattern cannot require the `def NAME : Kind<...>` to sit on one line. Demanding
# that cost five attributes on LLVM 23 and four on 18 -- `allocalign`, `allockind`,
# `disable_sanitizer_instrumentation`, `hot`, and `nocreateundeforpoison` -- and the last
# of those is a genuine 18->23 arrival that the delta therefore never reported. A
# whitespace-tolerant pattern is the fix; `\s` spans the newline where `<space>` did not.
_ATTR_DEF = re.compile(
    r'def\s+[A-Za-z0-9_]+\s*:\s*(?:Enum|Int|Type|Str|ConstantRange|ComplexStr)Attr<"([^"]+)"'
)
# The enum names are C++ identifiers (`vector_reduce_add`); the dotted LangRef spelling
# only appears in the trailing comment TableGen writes beside each one, which is why this
# reads the comment rather than the identifier -- and why it is not anchored to the line
# start: the comment follows the enumerator on the same line.
_INTRINSIC_COMMENT = re.compile(r"// (llvm\.[a-z0-9_.]+)")

# Opcodes that exist in the enum but are not language constructs.
_NON_LANGUAGE_OPCODES = {"UserOp1", "UserOp2"}


class Report:
    """Findings accumulate; the process exits on the count, never on the first one."""

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


# --------------------------------------------------------------------------
# Reading the surface out of an installed toolchain
# --------------------------------------------------------------------------


def include_dir(major: int) -> Path | None:
    """Where LLVM `major` keeps its headers, asked of llvm-config rather than guessed."""
    for candidate in (f"llvm-config-{major}", "llvm-config"):
        exe = shutil.which(candidate)
        if exe is None:
            continue
        try:
            proc = subprocess.run(
                [exe, "--version", "--includedir"], capture_output=True, text=True, timeout=60
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        lines = proc.stdout.split()
        if len(lines) < 2 or not lines[0].startswith(f"{major}."):
            continue
        path = Path(lines[1])
        if (path / "llvm/IR/Instruction.def").is_file():
            return path
    # Distribution layout, when llvm-config is absent but the headers are installed.
    fallback = Path(f"/usr/lib/llvm-{major}/include")
    return fallback if (fallback / "llvm/IR/Instruction.def").is_file() else None


# Constructs no LLVM this corpus supports can be missing. These are canaries rather than
# expected counts: a count drifts every release and would need maintaining, while `Ret`
# leaving LLVM would mean something other than a broken regex. They were chosen by
# intersecting the checked-in snapshots rather than from memory -- `Br` looked like an
# obvious candidate and is absent from 23, which splits it into `CondBr`/`UncondBr`.
#
# This exists because the drift check below compares a stored surface against a live one
# read by THE SAME extractor. If an extractor silently matched nothing -- as the attribute
# regex nearly did, by requiring a declaration to fit on one line -- then `--emit-surface`
# writes an empty snapshot, and the drift check compares [] against [] and passes. A
# comparison between a generated file and its own generator says nothing about the
# generator; only an outside fact does.
_SURFACE_CANARIES = {
    "instructions": ("Add", "Alloca", "Call", "Load", "Ret", "Store"),
    "attributes": ("byval", "noalias", "nounwind", "sret"),
    "intrinsics": ("llvm.assume", "llvm.memcpy", "llvm.trap"),
}


def surface_defects(surface: dict[str, list[str]], origin: str) -> list[str]:
    """Every way an extraction can have silently found nothing, or nearly nothing."""
    problems = []
    for kind, canaries in _SURFACE_CANARIES.items():
        found = set(surface.get(kind, ()))
        missing = [name for name in canaries if name not in found]
        if missing:
            problems.append(
                f"{origin}: the {kind} surface holds {len(found)} item(s) and none of "
                f"{', '.join(missing)} -- no supported LLVM omits those, so this is a "
                f"broken extraction rather than a real surface"
            )
    return problems


def read_surface(include: Path) -> dict[str, list[str]]:
    """The target-independent language surface, as the toolchain itself declares it."""
    ir = include / "llvm/IR"

    opcodes = {
        name
        for name in _HANDLE_INST.findall((ir / "Instruction.def").read_text(encoding="utf-8"))
        if name not in _NON_LANGUAGE_OPCODES
    }
    attributes = set(_ATTR_DEF.findall((ir / "Attributes.td").read_text(encoding="utf-8")))
    intrinsics = {
        name
        for name in _INTRINSIC_COMMENT.findall(
            (ir / "IntrinsicEnums.inc").read_text(encoding="utf-8")
        )
        if not _TARGET_INTRINSIC.match(name)
    }
    return {
        "instructions": sorted(opcodes),
        "attributes": sorted(attributes),
        "intrinsics": sorted(intrinsics),
    }


def snapshot_path(major: int) -> Path:
    return REFERENCE / f"llvm-surface-{major}.json"


def write_snapshot(major: int) -> int:
    include = include_dir(major)
    if include is None:
        print(
            f"error: no LLVM {major} headers found; --emit-surface needs llvm-{major}-dev",
            file=sys.stderr,
        )
        return 1
    surface = read_surface(include)
    defects = surface_defects(surface, f"LLVM {major} headers at {include}")
    if defects:
        for defect in defects:
            print(f"error: {defect}", file=sys.stderr)
        print(
            "error: refusing to write a snapshot from an extraction that found nothing; "
            "fix the extractor before regenerating, or the drift check will compare the "
            "break against itself and pass",
            file=sys.stderr,
        )
        return 1
    payload = {
        "llvm_major": major,
        "source": "Instruction.def + Attributes.td + IntrinsicEnums.inc, as installed",
        "note": (
            "Generated by training/llvm/tools/verify-langref-delta.py --emit-surface. "
            "Target-prefixed intrinsics are excluded: they belong to a target, not to the "
            "language LangRef describes."
        ),
        "counts": {kind: len(items) for kind, items in surface.items()},
        **surface,
    }
    snapshot_path(major).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"[emit] {snapshot_path(major).name}: "
        + ", ".join(f"{len(v)} {k}" for k, v in surface.items())
    )
    return 0


def check_stored_surfaces(report: Report) -> None:
    """The checked-in snapshots must look like real LLVM surfaces.

    `check_snapshot` compares a stored surface against a live one read by the same
    extractor, so it cannot see a break that affected both. This one asks an outside
    question instead, and it needs no toolchain -- the snapshots are repository content,
    so a host with no LLVM headers at all still owns this check.
    """
    for major in (BASELINE_MAJOR, CURRENT_MAJOR):
        path = snapshot_path(major)
        stored = json.loads(path.read_text(encoding="utf-8"))
        for defect in surface_defects(stored, path.name):
            report.require(False, defect)
        # The declared counts are what the chapters and the coverage checks read. A count
        # that disagrees with the list beside it makes every figure downstream a guess.
        for kind, count in stored.get("counts", {}).items():
            report.require(
                count == len(stored.get(kind, ())),
                f"{path.name}: counts[{kind}] says {count} but the list holds "
                f"{len(stored.get(kind, ()))}",
            )
    print("[stored]  both surface snapshots carry their canaries and agree with their counts")


def check_snapshot(major: int, report: Report, required: bool) -> None:
    """The checked-in snapshot must still be what this toolchain says."""
    include = include_dir(major)
    if include is None:
        if required:
            report.require(
                False,
                f"--require-surface {major} was passed but no LLVM {major} headers are "
                f"installed; the job that installs llvm-{major}-dev is the one that owns "
                f"this check, so absence here is a defect rather than a skip",
            )
        else:
            print(
                f"[skip]    no LLVM {major} headers; the snapshot drift check for {major} "
                f"belongs to the CI job that installs them",
                file=sys.stderr,
            )
        return

    stored = json.loads(snapshot_path(major).read_text(encoding="utf-8"))
    live = read_surface(include)
    for defect in surface_defects(live, f"LLVM {major} headers at {include}"):
        report.require(False, defect)
    for kind, items in live.items():
        report.require(
            stored[kind] == items,
            f"llvm-surface-{major}.json is stale for {kind}: the installed LLVM {major} "
            f"adds {sorted(set(items) - set(stored[kind]))[:8]} and drops "
            f"{sorted(set(stored[kind]) - set(items))[:8]}; regenerate with "
            f"--emit-surface {major}",
        )
    print(f"[drift]   llvm-surface-{major}.json matches the installed LLVM {major}")


# --------------------------------------------------------------------------
# The delta, and whether the corpus has an answer for every item in it
# --------------------------------------------------------------------------


def compute_delta() -> dict[str, dict[str, list[str]]]:
    """What arrived and what left between the two checked-in snapshots."""
    old = json.loads(snapshot_path(BASELINE_MAJOR).read_text(encoding="utf-8"))
    new = json.loads(snapshot_path(CURRENT_MAJOR).read_text(encoding="utf-8"))

    delta: dict[str, dict[str, list[str]]] = {}
    for kind in ("instructions", "attributes", "intrinsics"):
        before, after = set(old[kind]), set(new[kind])
        arrived, departed = after - before, before - after
        if kind == "instructions":
            # A split of one opcode into two is not two arrivals and one departure.
            for original, parts in INTERNAL_OPCODE_SPLITS.items():
                if original in departed and set(parts) <= arrived:
                    departed.discard(original)
                    arrived.difference_update(parts)
        delta[kind] = {"arrived": sorted(arrived), "departed": sorted(departed)}
    return delta


# --------------------------------------------------------------------------
# The opcodes 23-version-movement/03 exists to run, and the claim that nothing else ran them
# --------------------------------------------------------------------------

CONVERSIONS_CHAPTER = TRAINING_ROOT / "23-version-movement" / "03-conversions-and-shifts.md"
CONVERSIONS_EXAMPLE = (
    TRAINING_ROOT / "23-version-movement" / "examples" / "conversions-and-shifts.ll"
)

# The opcodes that chapter says the corpus named but never ran. This list is the tool's,
# not the chapter's: the chapter used to carry a count typed beside a list, and the two
# disagreed (it said nine over eight names) because nothing compared them.
UNDEMONSTRATED_OPCODES = (
    "ashr",
    "fneg",
    "fpext",
    "fptrunc",
    "fptosi",
    "fptoui",
    "sitofp",
    "uitofp",
)
_NUMBER_WORDS = {
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine",
    10: "Ten",
    11: "Eleven",
    12: "Twelve",
}


def _opcode_re(opcode: str) -> re.Pattern[str]:
    """`opcode` used as an instruction: not as part of `llvm.fptosi.sat`, not in a word.

    A type must follow, which is what separates the instruction `fptosi float ...` from
    the intrinsic family prefix `llvm.fptosi.sat.i32.f32`.
    """
    return re.compile(
        r"(?:^|[^A-Za-z0-9_.])" + re.escape(opcode) + r" +(?:i\d|f(?:loat|p128|16|128)|"
        r"double|half|bfloat|x86_fp80|ppc_fp128|<|ptr)"
    )


# LLVM's `Instruction.def` names opcode CLASSES (`Add`, `CondBr`, `GetElementPtr`); IR
# spells them as mnemonics. Most are the class name lowercased, and these are not.
# `Br` split into `CondBr`/`UncondBr` between 18 and 23, which is why both map to `br`.
_MNEMONICS = {
    "AddrSpaceCast": "addrspacecast",
    "AtomicCmpXchg": "cmpxchg",
    "AtomicRMW": "atomicrmw",
    "BitCast": "bitcast",
    "CallBr": "callbr",
    "CatchPad": "catchpad",
    "CatchRet": "catchret",
    "CatchSwitch": "catchswitch",
    "CleanupPad": "cleanuppad",
    "CleanupRet": "cleanupret",
    "CondBr": "br",
    "ExtractElement": "extractelement",
    "ExtractValue": "extractvalue",
    "FCmp": "fcmp",
    "FNeg": "fneg",
    "FPExt": "fpext",
    "FPToSI": "fptosi",
    "FPToUI": "fptoui",
    "FPTrunc": "fptrunc",
    "GetElementPtr": "getelementptr",
    "ICmp": "icmp",
    "IndirectBr": "indirectbr",
    "InsertElement": "insertelement",
    "InsertValue": "insertvalue",
    "IntToPtr": "inttoptr",
    "LandingPad": "landingpad",
    "PHI": "phi",
    "PtrToAddr": "ptrtoaddr",
    "PtrToInt": "ptrtoint",
    "SExt": "sext",
    "SIToFP": "sitofp",
    "ShuffleVector": "shufflevector",
    "UIToFP": "uitofp",
    "UncondBr": "br",
    "VAArg": "va_arg",
    "ZExt": "zext",
}


def check_instruction_coverage(report: Report) -> None:
    """Every instruction LangRef defines is named somewhere in the corpus, in code.

    The attribute and intrinsic surfaces earn their disposition tables by being large
    enough that not naming something can be the right answer. This one is not: 67
    instructions IS the language, and an instruction the corpus never writes down is a
    hole rather than a judgement. It is at 67 of 67 -- so the only thing this check can
    do is keep it there, which is the point, because nothing was keeping it there.
    """
    surface = json.loads(snapshot_path(CURRENT_MAJOR).read_text(encoding="utf-8"))
    opcodes = surface["instructions"]
    if not report.require(
        len(opcodes) >= 40,
        f"only {len(opcodes)} instructions in the LLVM {CURRENT_MAJOR} snapshot; the "
        f"extraction is broken and this check would examine almost nothing",
    ):
        return
    text = corpus_code(TRAINING_ROOT)
    for opcode in opcodes:
        mnemonic = _MNEMONICS.get(opcode, opcode.lower())
        # Word-bounded: `ret` must not be credited to `returned`, nor `and` to `landingpad`.
        report.require(
            re.search(rf"(?<![A-Za-z0-9_.]){re.escape(mnemonic)}(?![A-Za-z0-9_])", text),
            f"instruction {opcode!r} (spelt {mnemonic!r} in IR) is defined by LLVM "
            f"{CURRENT_MAJOR} and written nowhere in the corpus as code; unlike an "
            f"attribute or an intrinsic there is no honest reason not to name one of the "
            f"{len(opcodes)} instructions the language has",
        )
    stale = sorted(k for k in _MNEMONICS if k not in opcodes)
    report.require(
        not stale,
        f"the mnemonic table maps opcode(s) LLVM {CURRENT_MAJOR} no longer defines: "
        f"{stale}; a mapping nothing uses hides the next rename",
    )
    print(f"[opcode]  all {len(opcodes)} instruction opcodes are written in the corpus as code")


def check_intrinsic_coverage(report: Report) -> None:
    """Every intrinsic is either named by the corpus or belongs to a class that answers.

    This is the attribute check one surface over, with one difference forced by the
    subject: 524 intrinsics is too many to disposition one at a time and, more to the
    point, most of them do not deserve an individual answer. `llvm.vp.*` is 90 operations
    that are the same operation ninety times, and the vectorization chapter teaching the
    mask/`%evl` model and naming two of them is BETTER teaching than a list of ninety --
    which is the case that made a family-level answer necessary.

    So the unit is a class, and the assignment is per intrinsic. It is per intrinsic
    because LLVM's `llvm.<family>.*` prefixes are not reliably semantic: `llvm.get.*`
    spans the floating-point environment, the stack, and vector shape, and a table keyed
    on prefixes would have had to write one reason covering all three, which is how a
    disposition becomes a rubber stamp.
    """
    if not report.require(INTRINSIC_DISPOSITIONS.is_file(), f"{INTRINSIC_DISPOSITIONS} is missing"):
        return
    surface = json.loads(snapshot_path(CURRENT_MAJOR).read_text(encoding="utf-8"))
    intrinsics = surface["intrinsics"]
    table = json.loads(INTRINSIC_DISPOSITIONS.read_text(encoding="utf-8"))
    classes, assignment = table["classes"], table["intrinsics"]

    if not report.require(
        len(intrinsics) >= 300,
        f"only {len(intrinsics)} intrinsics in the LLVM {CURRENT_MAJOR} snapshot; the "
        f"language defines far more, so the extraction is broken and every loop below "
        f"would iterate over almost nothing",
    ):
        return

    text = corpus_code(TRAINING_ROOT)
    named = {n for n in intrinsics if n in text}
    for name in intrinsics:
        if name in named:
            continue
        report.require(
            name in assignment,
            f"intrinsic {name!r} is defined by LLVM {CURRENT_MAJOR}, named nowhere in the "
            f"corpus in code, and belongs to no class in "
            f"langref-intrinsic-dispositions.json; an intrinsic nobody teaches and nobody "
            f"declared out of scope is a gap nobody has found yet",
        )

    for name, cls in assignment.items():
        if not report.require(
            name in intrinsics,
            f"langref-intrinsic-dispositions.json assigns {name!r} to a class, but LLVM "
            f"{CURRENT_MAJOR} does not define it; drop the entry or regenerate the snapshot",
        ):
            continue
        report.require(
            cls in classes,
            f"{name!r} is assigned to class {cls!r}, which the table does not define",
        )

    for cls, entry in classes.items():
        members = sorted(n for n, c in assignment.items() if c == cls)
        if not report.require(
            members, f"class {cls!r} is defined and holds no intrinsic; drop it or assign one"
        ):
            continue
        status = entry.get("status")
        if not report.require(
            status in ("taught", "referenced", "declared"),
            f"class {cls!r} has status {status!r}, which is none of taught/referenced/declared",
        ):
            continue
        if status == "declared":
            report.require(
                len((entry.get("reason") or "").split()) >= 12,
                f"class {cls!r} is declared out of scope with no reason, or with one too "
                f"short to be one; a class of intrinsics dismissed in a few words reads "
                f"exactly like a class nobody looked at",
            )
            continue
        # taught and referenced both owe a citation, and the SAME citation rule the
        # dialect registry uses: the named file must name a member of the class, in code.
        where = entry.get("where") or ""
        chapter = TRAINING_ROOT / where
        if not report.require(
            where and chapter.is_file(),
            f"class {cls!r} is {status} but cites {where!r}, which is not a file",
        ):
            continue
        code = code_spans(chapter.read_text(encoding="utf-8"))
        cited = sorted(n for n in members if n in code)
        report.require(
            cited,
            f"{where} is cited as {status} for the {cls!r} intrinsic class but names none "
            f"of its {len(members)} members in a code block or inline code span",
        )

    covered = len(named) + sum(1 for n in assignment if n not in named)
    report.require(
        covered == len(intrinsics),
        f"{covered} of {len(intrinsics)} intrinsics are accounted for; the remainder are "
        f"neither named nor classified",
    )
    print(
        f"[intrin]  {len(named)}/{len(intrinsics)} intrinsics named in code; the rest are "
        f"covered by {len(classes)} class(es)"
    )


def check_undemonstrated_opcodes(report: Report) -> None:
    """The premise of 03-conversions-and-shifts.md, checked instead of trusted.

    Its opening sentence is a claim about the whole corpus -- these opcodes are named in
    the syntax table and run nowhere -- and that claim expires the moment another chapter
    runs one. Three things have to hold: the chapter lists exactly these opcodes, its own
    example runs every one of them, and no other example does.
    """
    if not report.require(
        CONVERSIONS_CHAPTER.is_file() and CONVERSIONS_EXAMPLE.is_file(),
        "23-version-movement/03-conversions-and-shifts.md or its example is missing",
    ):
        return

    chapter = CONVERSIONS_CHAPTER.read_text(encoding="utf-8")
    word = _NUMBER_WORDS.get(len(UNDEMONSTRATED_OPCODES), str(len(UNDEMONSTRATED_OPCODES)))
    sentence = f"{word} instruction opcodes appear in this corpus only as entries"
    report.require(
        sentence in chapter,
        f"03-conversions-and-shifts.md should open with {sentence!r} -- there are "
        f"{len(UNDEMONSTRATED_OPCODES)} opcodes in the list this gate holds it to, and a "
        f"count typed next to a list is a count that drifts away from it",
    )
    for opcode in UNDEMONSTRATED_OPCODES:
        report.require(
            f"`{opcode}`" in chapter,
            f"03-conversions-and-shifts.md does not name `{opcode}`, which this gate "
            f"holds it responsible for teaching",
        )

    example = CONVERSIONS_EXAMPLE.read_text(encoding="utf-8")
    for opcode in UNDEMONSTRATED_OPCODES:
        report.require(
            _opcode_re(opcode).search(example),
            f"{CONVERSIONS_EXAMPLE.name} does not run `{opcode}`; the chapter's whole "
            f"premise is that it demonstrates what the corpus only listed",
        )

    # And the other half: if some other example started running one of these, the chapter's
    # "named, never demonstrated" framing is no longer true and needs rewriting, not
    # quietly leaving in place.
    for path in sorted(TRAINING_ROOT.rglob("*.ll")):
        if path == CONVERSIONS_EXAMPLE:
            continue
        body = path.read_text(encoding="utf-8", errors="replace")
        for opcode in UNDEMONSTRATED_OPCODES:
            report.require(
                not _opcode_re(opcode).search(body),
                f"{path.relative_to(TRAINING_ROOT)} runs `{opcode}`, so "
                f"03-conversions-and-shifts.md's claim that the corpus never demonstrates "
                f"it is now false; drop the opcode from the chapter's list and from "
                f"UNDEMONSTRATED_OPCODES, or move the demonstration",
            )
    print(
        f"[opcodes] {len(UNDEMONSTRATED_OPCODES)} opcodes run in "
        f"{CONVERSIONS_EXAMPLE.name} and nowhere else in the corpus"
    )


def check_attribute_coverage(report: Report) -> None:
    """Every attribute in the language is either named by the corpus or answered for.

    The delta table above only ever saw the 131 items that MOVED between LLVM 18 and 23.
    That left the standing surface unexamined: 32 of the 99 attributes LLVM 23 defines
    were named nowhere in the corpus and nothing noticed, because nothing was looking.

    Each disposition carries `emitted_by`: the recipe actually run against clang to
    produce the attribute. That field is the difference between "we decided this is build
    configuration" and "we assumed it was". Twelve of them record "not reproduced", which
    is a finding rather than a gap -- an attribute the language defines and the toolchain
    never emits is one a reader will not meet by reading output.
    """
    if not report.require(ATTRIBUTE_DISPOSITIONS.is_file(), f"{ATTRIBUTE_DISPOSITIONS} is missing"):
        return
    surface = json.loads(snapshot_path(CURRENT_MAJOR).read_text(encoding="utf-8"))
    attributes = surface["attributes"]
    entries = json.loads(ATTRIBUTE_DISPOSITIONS.read_text(encoding="utf-8"))["attributes"]

    # Anti-vacuity: an empty surface would make every loop below iterate zero times.
    if not report.require(
        len(attributes) >= 50,
        f"only {len(attributes)} attributes in the LLVM {CURRENT_MAJOR} snapshot; the "
        f"language defines far more, so the snapshot is broken and this check would pass "
        f"having examined almost nothing",
    ):
        return

    # `corpus_code`, not `teaching_text`: several attribute names are ordinary English
    # words, and a raw-text search credited `returned` and `ssp` to sentences that merely
    # used them. Neither was named in code anywhere, and neither was dispositioned -- so
    # the looser predicate was hiding a real gap behind a better-looking number.
    text = corpus_code(TRAINING_ROOT)
    unnamed = [a for a in attributes if a not in text]
    for name in unnamed:
        report.require(
            name in entries,
            f"attribute {name!r} is defined by LLVM {CURRENT_MAJOR} and named nowhere in "
            f"the corpus, and langref-attribute-dispositions.json does not mention it; an "
            f"attribute nobody teaches and nobody declared out of scope is a gap nobody "
            f"has found yet",
        )

    for name, entry in entries.items():
        if not report.require(
            name in attributes,
            f"langref-attribute-dispositions.json disposes of {name!r}, which LLVM "
            f"{CURRENT_MAJOR} does not define; drop the entry or regenerate the snapshot",
        ):
            continue
        report.require(
            (entry.get("emitted_by") or "").strip(),
            f"{name!r} records no `emitted_by`; the recipe that produces an attribute is "
            f"what separates a checked disposition from a guess",
        )
        status = entry.get("status")
        if not report.require(
            status in ("taught", "declared"),
            f"{name!r} has status {status!r}, which is neither 'taught' nor 'declared'",
        ):
            continue
        if status == "declared":
            report.require(
                (entry.get("reason") or "").strip(),
                f"{name!r} is declared out of scope with no reason",
            )
            continue
        where = entry.get("where") or ""
        chapter = TRAINING_ROOT / where
        if not report.require(
            where and chapter.is_file(),
            f"{name!r} is marked taught by {where!r}, which is not a file under "
            f"{TRAINING_ROOT.name}/",
        ):
            continue
        report.require(
            name in code_spans(chapter.read_text(encoding="utf-8")),
            f"{where} is cited as teaching {name!r} but never writes it in a code block "
            f"or inline code span",
        )

    taught = sum(1 for e in entries.values() if e.get("status") == "taught")

    # The chapter states how many attributes it does NOT teach, in words. That is a count
    # of live data sitting in prose, which is how the bytecode figures in chapter 24 came
    # to be wrong, so it is pinned here rather than trusted.
    declared = len(entries) - taught
    words = {
        11: "Eleven",
        12: "Twelve",
        13: "Thirteen",
        14: "Fourteen",
        15: "Fifteen",
        16: "Sixteen",
        17: "Seventeen",
        18: "Eighteen",
        19: "Nineteen",
        20: "Twenty",
        21: "Twenty-one",
        22: "Twenty-two",
        23: "Twenty-three",
        24: "Twenty-four",
        25: "Twenty-five",
        26: "Twenty-six",
        27: "Twenty-seven",
        28: "Twenty-eight",
        29: "Twenty-nine",
        30: "Thirty",
    }
    chapter = TRAINING_ROOT / "13-advanced-ir" / "04-attributes.md"
    if chapter.is_file():
        sentence = (
            f"{words.get(declared, str(declared))} further attributes exist in LLVM {CURRENT_MAJOR}"
        )
        body = chapter.read_text(encoding="utf-8")
        report.require(
            sentence in body,
            f"13-advanced-ir/04-attributes.md should say {sentence!r}: {declared} "
            f"attributes are dispositioned as out of scope, and the chapter states that "
            f"count in prose",
        )
        # The same sentence carries a SECOND count of live data: how many of those could
        # not be produced at all. The first version of it said thirteen while the table
        # held fourteen, which is the argument for pinning it rather than the argument
        # against -- one pinned number in a sentence does not protect the other.
        unproduced = sum(
            1 for e in entries.values() if e.get("emitted_by", "").startswith("not reproduced")
        )
        phrase = f"and {words.get(unproduced, str(unproduced)).lower()} that"
        report.require(
            phrase in body,
            f"13-advanced-ir/04-attributes.md should say {phrase!r}: {unproduced} "
            f"dispositions record that no recipe produced the attribute",
        )

    reproduced = sum(
        1 for e in entries.values() if not e.get("emitted_by", "").startswith("not reproduced")
    )
    print(
        f"[attrs]   {len(attributes) - len(unnamed)}/{len(attributes)} attributes named; "
        f"{len(entries)} dispositioned ({taught} taught, {len(entries) - taught} declared), "
        f"{reproduced} with a reproduced emission recipe"
    )


def check_dispositions(report: Report) -> None:
    """Every item that moved needs an answer: a chapter that teaches it, or a reason."""
    delta = compute_delta()
    dispositions = json.loads(DISPOSITIONS.read_text(encoding="utf-8"))
    entries = dispositions["items"]

    moved: list[tuple[str, str, str]] = [
        (kind, direction, name)
        for kind, sides in delta.items()
        for direction, names in sides.items()
        for name in names
    ]

    # Anti-vacuity, per surface rather than in total. A single `if moved:` is satisfied by
    # any one kind moving, so the intrinsic churn alone -- over a hundred names -- would
    # keep this gate green while the instruction and attribute snapshots had quietly become
    # identical and were being compared to nothing. Each surface answers for itself.
    for kind, sides in delta.items():
        report.require(
            sides["arrived"] or sides["departed"],
            f"no {kind} moved between the LLVM {BASELINE_MAJOR} and {CURRENT_MAJOR} "
            f"snapshots, so this gate compared that surface against nothing; either a "
            f"snapshot is wrong or {kind} stopped moving and this check has lost its "
            f"subject there",
        )

    for kind, direction, name in moved:
        entry = entries.get(name)
        if not report.require(
            entry is not None,
            f"{kind[:-1]} {name!r} {direction} between LLVM {BASELINE_MAJOR} and "
            f"{CURRENT_MAJOR} but langref-delta-dispositions.json does not mention it; "
            f"every item that moves needs a chapter that teaches it or a stated reason "
            f"it is out of scope",
        ):
            continue

        status = entry.get("status")
        if not report.require(
            status in ("taught", "declared"),
            f"{name!r} has disposition {status!r}, which is neither 'taught' nor 'declared'",
        ):
            continue

        if status == "declared":
            report.require(
                (entry.get("reason") or "").strip(),
                f"{name!r} is declared out of scope with no reason; a declared gap without "
                f"a reason is an omission wearing a label",
            )
            continue

        where = entry.get("where") or ""
        chapter = TRAINING_ROOT / where
        if not report.require(
            where and chapter.is_file(),
            f"{name!r} is marked taught by {where!r}, which is not a file under "
            f"{TRAINING_ROOT.name}/",
        ):
            continue

        # A citation that does not mention the thing is not a citation -- but a bare
        # substring search is barely better. Several of these names are ordinary English
        # words (`range`, `flatten`, `noext`), and a chapter that happens to use one in a
        # sentence would satisfy a substring test while teaching nothing about the
        # attribute. What a reader searches for is the IR spelling, and a chapter that
        # teaches an IR construct writes it as code, so only code counts here: fenced
        # blocks and inline `backticked` spans.
        text = chapter.read_text(encoding="utf-8")
        code = code_spans(text)
        spellings = {name, name.lower()}
        report.require(
            any(spelling in code for spelling in spellings),
            f"{where} is cited as teaching {name!r}, but names it nowhere in a code block "
            f"or inline code span -- only, at best, in prose. An attribute whose spelling "
            f"a chapter never shows is an attribute that chapter does not teach",
        )

    # Stale entries are as bad as missing ones: they claim coverage of a surface that no
    # longer moves, and hide the fact that nothing checks them any more.
    named = {name for _, _, name in moved}
    for name in entries:
        report.require(
            name in named,
            f"langref-delta-dispositions.json disposes of {name!r}, which does not appear "
            f"in the {BASELINE_MAJOR}->{CURRENT_MAJOR} delta; drop the entry or fix the "
            f"snapshots",
        )

    counts = {
        kind: f"{len(sides['arrived'])} arrived, {len(sides['departed'])} departed"
        for kind, sides in delta.items()
    }
    taught = sum(1 for e in entries.values() if e.get("status") == "taught")
    print(
        f"[delta]   LLVM {BASELINE_MAJOR} -> {CURRENT_MAJOR}: "
        + "; ".join(f"{k} {v}" for k, v in counts.items())
    )
    print(f"[cover]   {taught} taught, {len(entries) - taught} declared out of scope")


# --------------------------------------------------------------------------
# The chapter's one table, which is computed rather than typed
# --------------------------------------------------------------------------

CHAPTER = TRAINING_ROOT / "23-version-movement" / "01-reading-the-delta.md"
VP_CHAPTER = TRAINING_ROOT / "09-vectorization" / "03-vector-predication.md"


def _block_re(name: str) -> re.Pattern[str]:
    return re.compile(
        r"(?P<open><!-- generated: " + re.escape(name) + r" -->\n)"
        r"(?P<body>.*?)(?P<close><!-- /generated -->)",
        re.DOTALL,
    )


_BLOCK = _block_re("langref-delta")
_VP_BLOCK = _block_re("vp-family")


def render_block() -> str:
    """The delta as a table, built from the snapshots so no number is ever typed."""
    delta = compute_delta()
    entries = json.loads(DISPOSITIONS.read_text(encoding="utf-8"))["items"]
    taught = sum(1 for e in entries.values() if e.get("status") == "taught")
    rows = [
        "| surface | arrived | departed |",
        "| --- | ---: | ---: |",
    ]
    for kind in ("instructions", "attributes", "intrinsics"):
        rows.append(f"| {kind} | {len(delta[kind]['arrived'])} | {len(delta[kind]['departed'])} |")
    total = sum(len(s["arrived"]) + len(s["departed"]) for s in delta.values())
    rows.append("")
    rows.append(
        f"Of those {total}, **{taught}** are taught by a chapter and "
        f"**{total - taught}** are declared out of scope with a stated reason."
    )
    return "\n".join(rows) + "\n"


def render_vp_family() -> str:
    """How large the vector-predication family is, counted from the current surface.

    09-vectorization/03-vector-predication.md exists because the corpus used LLVM's term
    of art for this family while teaching a different mechanism. Its size is the argument
    for taking it seriously, so the number is computed here rather than typed there --
    this repository does not allow a count to live in prose, and a family that grows every
    release is exactly the kind that would rot.
    """
    surface = json.loads(snapshot_path(CURRENT_MAJOR).read_text(encoding="utf-8"))["intrinsics"]
    stable = [n for n in surface if n.startswith("llvm.vp.")]
    staging = [n for n in surface if n.startswith("llvm.experimental.vp.")]
    length = [n for n in surface if n.endswith(".get.vector.length")]
    total = len(stable) + len(staging) + len(length)
    helper = f"`{length[0]}`" if length else "no length helper"
    sentence = (
        f"In LLVM {CURRENT_MAJOR} the family is **{len(stable)}** intrinsics spelled "
        f"`llvm.vp.*`, **{len(staging)}** still staged as `llvm.experimental.vp.*`, and "
        f"the length helper {helper} — **{total}** names, out of {len(surface)} "
        f"target-independent intrinsics in the whole language."
    )
    # Wrapped to the corpus's prose width so the block reads like the text around it.
    words, lines, line = sentence.split(" "), [], ""
    for word in words:
        if line and len(line) + 1 + len(word) > 88:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}" if line else word
    lines.append(line)
    return "\n".join(lines) + "\n"


def sync_named_block(
    chapter: Path, name: str, fresh: str, report: Report | None, update: bool
) -> None:
    """Hold one `<!-- generated: NAME -->` block to what the surfaces produce.

    `sync_block` and `sync_vp_block` below predate this and say the same thing twice with
    different nouns; a third copy would have been the defect this repository names L14, so
    new blocks come through here.
    """
    if not chapter.is_file():
        if report:
            report.require(False, f"{chapter} is missing; the {name} block has no chapter")
        return
    text = chapter.read_text(encoding="utf-8")
    match = _block_re(name).search(text)
    if match is None:
        if report:
            report.require(
                False,
                f"{chapter.name} has no `<!-- generated: {name} -->` block, so the table is "
                f"computed and shown nowhere; a table nothing displays is a check over nothing",
            )
        return
    if update:
        if match.group("body") != fresh:
            chapter.write_text(
                text[: match.start("body")] + fresh + text[match.end("body") :], encoding="utf-8"
            )
            print(f"[update]  {chapter.name} ({name})")
        return
    if report:
        report.require(
            match.group("body") == fresh,
            f"{chapter.name}: the {name} block is stale; the surfaces now produce:\n" + fresh,
        )


def render_surface_coverage() -> str:
    """Where each LangRef surface stands, counted rather than claimed."""
    surface = json.loads(snapshot_path(CURRENT_MAJOR).read_text(encoding="utf-8"))
    text = corpus_code(TRAINING_ROOT)
    attrs = json.loads(ATTRIBUTE_DISPOSITIONS.read_text(encoding="utf-8"))["attributes"]
    table = json.loads(INTRINSIC_DISPOSITIONS.read_text(encoding="utf-8"))

    rows = [
        "| surface | in LLVM 23 | named in code | answered another way | how |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    opcodes = surface["instructions"]
    named_ops = sum(
        1
        for o in opcodes
        if re.search(
            rf"(?<![A-Za-z0-9_.]){re.escape(_MNEMONICS.get(o, o.lower()))}(?![A-Za-z0-9_])", text
        )
    )
    rows.append(
        f"| instructions | {len(opcodes)} | {named_ops} | {len(opcodes) - named_ops} | "
        f"nothing left over: 67 instructions is the language |"
    )
    a_named = sum(1 for a in surface["attributes"] if a in text)
    rows.append(
        f"| attributes | {len(surface['attributes'])} | {a_named} | {len(attrs)} | "
        f"one disposition each, with the clang recipe that emits it |"
    )
    i_named = sum(1 for n in surface["intrinsics"] if n in text)
    classes = table["classes"]
    rows.append(
        f"| intrinsics | {len(surface['intrinsics'])} | {i_named} | "
        f"{len(table['intrinsics']) - sum(1 for n in table['intrinsics'] if n in text)} | "
        f"{len(classes)} classes, each taught, referenced or declared |"
    )
    by_status = collections.Counter(v["status"] for v in classes.values())
    rows.append("")
    rows.append("| intrinsic class | status | members | held by |")
    rows.append("| --- | --- | ---: | --- |")
    for name, entry in sorted(classes.items()):
        members = sum(1 for c in table["intrinsics"].values() if c == name)
        held = f"`{entry['where']}`" if entry["status"] != "declared" else "a written reason"
        rows.append(f"| `{name}` | {entry['status']} | {members} | {held} |")
    rows.append("")
    rows.append(
        "Classes: **"
        + "**, **".join(f"{by_status[s]} {s}" for s in ("taught", "referenced", "declared"))
        + f"**. Every one of the {len(table['intrinsics'])} classified intrinsics belongs "
        f"to exactly one of them."
    )
    return "\n".join(rows) + "\n"


def sync_vp_block(report: Report | None, update: bool) -> None:
    """The VP chapter's one computed sentence."""
    if not VP_CHAPTER.is_file():
        if report:
            report.require(False, f"{VP_CHAPTER} is missing")
        return
    text = VP_CHAPTER.read_text(encoding="utf-8")
    match = _VP_BLOCK.search(text)
    if match is None:
        if report:
            report.require(
                False,
                f"{VP_CHAPTER.name} has no `<!-- generated: vp-family -->` block, so its "
                f"claim about the family's size is a hand-written count",
            )
        return
    fresh = render_vp_family()
    if update:
        if match.group("body") != fresh:
            VP_CHAPTER.write_text(
                text[: match.start("body")] + fresh + text[match.end("body") :], encoding="utf-8"
            )
            print(f"[update]  {VP_CHAPTER.name}")
        return
    if report:
        report.require(
            match.group("body") == fresh,
            f"{VP_CHAPTER.name}: the vp-family block is stale; the surface now produces:\n" + fresh,
        )


def sync_block(report: Report | None, update: bool) -> None:
    if not CHAPTER.is_file():
        if report:
            report.require(False, f"{CHAPTER} is missing; the delta has no chapter to show it")
        return
    text = CHAPTER.read_text(encoding="utf-8")
    match = _BLOCK.search(text)
    if match is None:
        if report:
            report.require(
                False,
                f"{CHAPTER.name} has no `<!-- generated: langref-delta -->` block, so the "
                f"delta is computed and shown nowhere; a table nothing displays is a check "
                f"over nothing",
            )
        return
    fresh = render_block()
    if update:
        if match.group("body") != fresh:
            CHAPTER.write_text(
                text[: match.start("body")] + fresh + text[match.end("body") :], encoding="utf-8"
            )
            print(f"[update]  {CHAPTER.name}")
        return
    if report:
        report.require(
            match.group("body") == fresh,
            f"{CHAPTER.name}: the langref-delta block is stale; the snapshots now produce:\n"
            + fresh,
        )


_GRADUATION_ROW = re.compile(
    r"^\s*(?P<old>llvm\.[a-z0-9_.]+)?\s*->\s*(?P<new>llvm\.[a-z0-9_.]+)\s*$", re.MULTILINE
)


def check_rename_table(report: Report) -> None:
    """Every name in the chapter's rename table must exist in the release it claims.

    SCOPE, stated here so it does not creep: this checks the `old -> new` rows in the
    graduation block of 23-version-movement/01-reading-the-delta.md, and nothing else. It
    does NOT scan prose for intrinsic names. That was tried and it is the wrong check: a
    corpus legitimately writes overloaded spellings (`llvm.memcpy.p0.p0.i64`), family
    prefixes as prose (`llvm.experimental.constrained`), and BCIR's own `llvm.bcir.*`
    namespace, none of which appear in a snapshot -- and relaxing the rule far enough to
    admit them also admits the very defect this exists to catch.

    That defect was real and was in this gate's own chapter: the table read
    `llvm.experimental.vector.splice -> llvm.vector.splice`, which is the obvious guess
    and is false. The intrinsic became TWO, `llvm.vector.splice.left` and `.right`, and
    plain `llvm.vector.splice` exists in no LLVM release. A rename is a hypothesis formed
    by reading two lists side by side, and a hypothesis belongs in a checked table.

    So: a row's left side must be an intrinsic LLVM 18 had, and its right side one LLVM
    23 has. Exact membership, because in this table every name is a whole claim.
    """
    old_surface = set(
        json.loads(snapshot_path(BASELINE_MAJOR).read_text(encoding="utf-8"))["intrinsics"]
    )
    new_surface = set(
        json.loads(snapshot_path(CURRENT_MAJOR).read_text(encoding="utf-8"))["intrinsics"]
    )

    if not CHAPTER.is_file():
        return
    rows = _GRADUATION_ROW.findall(CHAPTER.read_text(encoding="utf-8"))

    # Anti-vacuity: no rows means this check examined nothing, and the table it guards is
    # exactly where a wrong rename hides.
    report.require(
        rows,
        f"{CHAPTER.name} has no `llvm.x -> llvm.y` rename rows, so the rename table check "
        f"examined nothing; either the graduation table was removed or its shape changed",
    )

    for old, new in rows:
        if old:
            report.require(
                old in old_surface,
                f"{CHAPTER.name}: the rename table claims {old!r} existed in LLVM "
                f"{BASELINE_MAJOR}; it is not in that release's surface snapshot",
            )
        report.require(
            new in new_surface,
            f"{CHAPTER.name}: the rename table claims {new!r} exists in LLVM "
            f"{CURRENT_MAJOR}; it is not in that release's surface snapshot. A graduated "
            f"intrinsic does not always keep its name -- look the new spelling up rather "
            f"than deriving it from the old one",
        )
    print(f"[renames] {len(rows)} rename row(s) name intrinsics that really exist")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-surface",
        type=int,
        metavar="MAJOR",
        help="regenerate reference/llvm-surface-MAJOR.json from the installed LLVM MAJOR",
    )
    parser.add_argument(
        "--update", action="store_true", help="rewrite the chapter's generated block"
    )
    parser.add_argument(
        "--require-surface",
        type=int,
        action="append",
        default=[],
        metavar="MAJOR",
        help="fail rather than skip when LLVM MAJOR's headers are absent (pass this from "
        "the CI job that installs them)",
    )
    args = parser.parse_args(argv)

    if args.emit_surface is not None:
        return write_snapshot(args.emit_surface)

    if args.update:
        sync_block(None, update=True)
        sync_vp_block(None, update=True)
        sync_named_block(CHAPTER, "surface-coverage", render_surface_coverage(), None, True)
        return 0

    report = Report()

    # A --require-surface for a major this gate does not track would otherwise be accepted
    # and then never evaluated: the loop below only visits the two snapshot majors, so the
    # flag would silently do nothing and the run would go green having checked exactly what
    # it was told not to skip. A requirement that can be ignored is not a requirement.
    known_majors = (BASELINE_MAJOR, CURRENT_MAJOR)
    for major in args.require_surface:
        if major not in known_majors:
            print(
                f"error: --require-surface {major} names a major this gate does not track "
                f"(it spans {BASELINE_MAJOR} and {CURRENT_MAJOR}); the flag would have been "
                f"accepted and never evaluated",
                file=sys.stderr,
            )
            return 2

    check_stored_surfaces(report)
    for major in known_majors:
        check_snapshot(major, report, required=major in args.require_surface)
    check_dispositions(report)
    check_instruction_coverage(report)
    check_attribute_coverage(report)
    check_intrinsic_coverage(report)
    check_undemonstrated_opcodes(report)
    check_rename_table(report)
    sync_block(report, update=False)
    sync_vp_block(report, update=False)
    sync_named_block(CHAPTER, "surface-coverage", render_surface_coverage(), report, False)
    return report.verdict("langref delta gate")


if __name__ == "__main__":
    sys.exit(main())
