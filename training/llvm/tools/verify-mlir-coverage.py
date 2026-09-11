#!/usr/bin/env python3
"""Hold the corpus to the MLIR dialect surface, and make its coverage a tracked fact.

MLIR's operation set is two orders of magnitude larger than LLVM's: 47 registered
dialects and ~2500 operations against LangRef's 67 instructions. A corpus cannot teach
that op by op, and [`../24-mlir-infrastructure/README.md`](../24-mlir-infrastructure)
says so in as many words -- it teaches the framework and refuses the dialect tour.

The risk that refusal creates is the one this gate exists for. "We teach the framework,
not the dialects" is indistinguishable, from inside a green CI run, from "we never
noticed that `linalg` has 99 operations and the corpus names one of them." Both look
like silence. So every dialect carries a **disposition** -- a chapter that teaches it, or
a written reason it is out of scope -- exactly as every item in the LLVM 18->23 delta
does in [`verify-langref-delta.py`](verify-langref-delta.py). A gap with a reason is a
decision; a gap without one is an accident nobody has found yet.

TWO DERIVATIONS, DIFFED. The op set is read out of the installed MLIR two independent
ways, because a single extraction that quietly under-reports would make coverage look
better than it is:

  rail 1  `mlir-tblgen --gen-op-doc -dialect=N` -- LLVM's own op-documentation generator
  rail 2  `getOperationName()` in the generated `*.h.inc` -- what the C++ actually returns

Neither is a superset of the other by construction, and their disagreements are
informative rather than noise. Rail 1 misses operations defined outside a dialect's home
directory: every `transform.<dialect>.*` extension, all of `arm_sme`, and `builtin`,
whose dialect lives in `include/mlir/IR/` rather than `include/mlir/Dialect/`. Rail 2
would over-report if it matched any string literal that looks like an op name -- an
earlier version of it counted `amdgpu.buffer.oob.mode`, an *attribute* mnemonic, as an
operation -- which is why it anchors on `getOperationName()`, the accessor only an
operation has. Rail 2 is expected to contain rail 1. The reverse containment failing is
a real finding and fails this gate: it means an op TableGen documents has no generated
accessor, or the extraction broke.

Modes, following the split verify-langref-delta.py uses for the same reason -- the job
that installs the toolchain owns the claims only a toolchain can check:

  --emit-surface       write reference/mlir-surface-23.json from the installed MLIR
  --require-tools      fail, rather than skip, when mlir-opt/mlir-tblgen are absent
  --update             rewrite the generated block the chapter displays
  (default)            check the checked-in snapshot's dispositions, and additionally
                       check it against a real toolchain wherever one is installed

The dialect list is read from `mlir-opt --show-dialects`, not from a documentation page.
That matters more here than it looks: which dialects a build registers is a build
decision, and the two builds this corpus runs on disagree -- a conda-forge build of
23.1.1 registers 47 and Ubuntu's 23.1.2 registers 50. So the snapshot records what one
named build reported, the drift check reports differences rather than failing on them,
and no count of dialects is ever written into prose. See
`24-mlir-infrastructure/README.md` for the same lesson stated for readers.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_text import code_spans, teaching_text  # noqa: E402
from llvm_toolchain import find_llvm_tool  # noqa: E402

TOOLS = Path(__file__).resolve().parent
TRAINING_ROOT = TOOLS.parent
REFERENCE = TRAINING_ROOT / "reference"
SURFACE = REFERENCE / "mlir-surface-23.json"
DISPOSITIONS = REFERENCE / "mlir-dialect-dispositions.json"
CHAPTER = TRAINING_ROOT / "24-mlir-infrastructure" / "05-the-dialect-landscape.md"

# An operation's generated class declares its name through getOperationName(); an
# attribute or a type declares getMnemonic() instead. Anchoring here is what keeps
# `amdgpu.buffer.oob.mode` -- an attribute -- from being counted as an operation.
_OP_ACCESSOR = re.compile(
    r"getOperationName\(\)\s*\{\s*return\s*::llvm::StringLiteral"
    r'\("([a-z_][a-z0-9_]*\.[a-z0-9_.]+)"\)',
    re.S,
)
_OP_DOC = re.compile(r"^### `([a-z_][a-z0-9_]*\.[a-z0-9_.]+)`", re.M)
_DIALECT_DECL = re.compile(r'let\s+name\s*=\s*"([a-z_][a-z0-9_]*)"')

_BLOCK = re.compile(
    r"(?P<open><!-- generated: dialect-coverage -->\n)(?P<body>.*?)(?P<close><!-- /generated -->)",
    re.DOTALL,
)

# Four answers a dialect can give for itself, and each carries a different obligation:
#
#   taught      a chapter teaches it        -> `where`, and that chapter must name a real
#                                              operation of the dialect, in code
#   referenced  named in passing somewhere  -> same citation requirement, lower claim
#   planned     a gap the corpus intends to close -> `slice`, naming the work that closes
#                                              it, so an intention cannot masquerade as
#                                              coverage indefinitely
#   declared    out of scope                -> `reason`, in words
#
# `planned` exists because the other three could not express the honest state of this
# corpus. Marking `tosa` "declared out of scope" would be false -- it is squarely in
# scope and simply not written yet -- and marking it "taught" would be a lie. A gap the
# corpus has decided to close is a third thing, and it needs to be visible as one.
VALID_STATUSES = ("taught", "referenced", "planned", "declared")
CITED_STATUSES = ("taught", "referenced")


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
        self.completed.add(name)

    def verdict(self, label: str) -> int:
        if self.findings:
            print(f"{label}: FAILED", file=sys.stderr)
            for finding in self.findings:
                print(f"  - {finding}", file=sys.stderr)
            return 1
        print(f"{label}: PASSED ({self.checks} checks)")
        return 0


# --------------------------------------------------------------------------
# Reading the surface out of an installed MLIR
# --------------------------------------------------------------------------


def include_dir(mlir_opt: str) -> Path | None:
    """The include tree beside the tool, which is where the generated headers live."""
    candidate = Path(mlir_opt).resolve().parent.parent / "include"
    return candidate if (candidate / "mlir").is_dir() else None


def registered_dialects(mlir_opt: str) -> tuple[list[str], str] | None:
    """What this build registers, and the version banner that identifies the build."""
    try:
        shown = subprocess.run(
            [mlir_opt, "--show-dialects"], capture_output=True, text=True, timeout=120
        )
        version = subprocess.run(
            [mlir_opt, "--version"], capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if shown.returncode != 0 or "Available Dialects:" not in shown.stdout:
        return None
    listed = shown.stdout.split("Available Dialects:", 1)[1]
    names = sorted({d.strip() for d in listed.replace("\n", ",").split(",") if d.strip()})
    banner = ""
    for line in version.stdout.splitlines():
        if "LLVM version" in line:
            banner = line.strip()
            break
    return names, banner


def ops_from_headers(include: Path) -> dict[str, list[str]]:
    """Rail 2: every operation whose generated class returns its own name."""
    ops: dict[str, set[str]] = {}
    for header in include.glob("mlir/**/*.h.inc"):
        for name in _OP_ACCESSOR.findall(header.read_text(errors="replace")):
            ops.setdefault(name.split(".", 1)[0], set()).add(name)
    return {k: sorted(v) for k, v in sorted(ops.items())}


def ops_from_tblgen(include: Path, tblgen: str, dialects: list[str]) -> dict[str, list[str]]:
    """Rail 1: LLVM's own op-doc generator, asked one dialect at a time.

    `--gen-op-doc` refuses a .td that defines more than one dialect unless told which to
    document, so each run names its dialect. Only the directories that declare a dialect
    are searched for it, which is why this rail misses operations contributed to a
    dialect from elsewhere in the tree -- see the module docstring.
    """
    home: dict[str, set[Path]] = {}
    for td in include.glob("mlir/Dialect/**/*.td"):
        for ns in _DIALECT_DECL.findall(td.read_text(errors="replace")):
            home.setdefault(ns, set()).add(td.parent)

    ops: dict[str, list[str]] = {}
    for ns in dialects:
        found: set[str] = set()
        for directory in sorted(home.get(ns, ())):
            for td in sorted(directory.glob("*.td")):
                try:
                    run = subprocess.run(
                        [tblgen, "--gen-op-doc", f"-dialect={ns}", "-I", str(include), str(td)],
                        capture_output=True,
                        text=True,
                        timeout=180,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    continue
                if run.returncode == 0:
                    found.update(_OP_DOC.findall(run.stdout))
        if found:
            ops[ns] = sorted(found)
    return ops


def read_surface(mlir_opt: str, tblgen: str | None) -> dict | None:
    registered = registered_dialects(mlir_opt)
    include = include_dir(mlir_opt)
    if registered is None or include is None:
        return None
    dialects, banner = registered
    headers = ops_from_headers(include)
    tblgen_ops = ops_from_tblgen(include, tblgen, dialects) if tblgen else {}

    # Dialects the headers ship operations for but the tool does not register, and the
    # reverse. Both are real conditions rather than extraction bugs: `dlti` registers
    # with no operations at all because it is an attribute-only dialect, and a build can
    # ship headers for a dialect it does not link in.
    unregistered = sorted(set(headers) - set(dialects))
    opless = sorted(set(dialects) - set(headers))
    return {
        "measured_with": banner,
        "registered_dialects": dialects,
        "ops": headers,
        "tblgen_ops": tblgen_ops,
        "unregistered_with_ops": unregistered,
        "registered_without_ops": opless,
    }


def write_surface(mlir_opt: str, tblgen: str | None) -> int:
    surface = read_surface(mlir_opt, tblgen)
    if surface is None:
        print("error: could not read the MLIR surface", file=sys.stderr)
        return 1
    SURFACE.write_text(json.dumps(surface, indent=1) + "\n", encoding="utf-8")
    total = sum(len(v) for v in surface["ops"].values())
    print(
        f"[emit] {SURFACE.name}: {len(surface['registered_dialects'])} registered dialects, "
        f"{len(surface['ops'])} with operations, {total} operations "
        f"({surface['measured_with']})"
    )
    return 0


# --------------------------------------------------------------------------
# Coverage: which operations the corpus actually names
# --------------------------------------------------------------------------


def corpus_text() -> str:
    """The teaching corpus, minus the files that would answer for themselves.

    `corpus_text.teaching_text` excludes the corpus's own Python, which matters here more
    than anywhere else: Python's `math` module shares five operation names with MLIR's
    `math` dialect, and a benchmark script importing it made that dialect look a sixth
    covered when no chapter mentions it.
    """
    return teaching_text(TRAINING_ROOT)


def coverage(surface: dict, text: str) -> dict[str, tuple[int, int]]:
    """Per dialect: (operations in the build, operations named anywhere in the corpus)."""
    return {
        ns: (len(names), sum(1 for n in names if n in text)) for ns, names in surface["ops"].items()
    }


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def check_rails(report: Report, surface: dict) -> None:
    """The two derivations must not contradict each other."""
    tblgen_ops = surface.get("tblgen_ops") or {}
    if not tblgen_ops:
        print("[rails]   no tblgen rail in this snapshot; the header rail stands alone")
        report.done("rails")
        return

    contradictions = []
    for ns, names in tblgen_ops.items():
        missing = sorted(set(names) - set(surface["ops"].get(ns, [])))
        if missing:
            contradictions.append((ns, missing[:3]))
    report.require(
        not contradictions,
        "mlir-tblgen documents operations the generated headers have no accessor for: "
        + "; ".join(f"{ns}: {names}" for ns, names in contradictions[:4])
        + ". The header rail is supposed to contain the tblgen rail, so one of the two "
        "extractions is broken rather than merely incomplete",
    )
    covered = sum(len(v) for v in tblgen_ops.values())
    total = sum(len(v) for v in surface["ops"].values())
    print(
        f"[rails]   tblgen {covered} ops over {len(tblgen_ops)} dialects is contained in "
        f"headers {total} ops over {len(surface['ops'])}"
    )
    report.done("rails")


def check_dispositions(report: Report, surface: dict, cover: dict) -> None:
    """Every dialect answers for itself: taught by a chapter, or out of scope with a reason."""
    if not report.require(DISPOSITIONS.is_file(), f"{DISPOSITIONS} is missing"):
        return
    entries = json.loads(DISPOSITIONS.read_text(encoding="utf-8"))["dialects"]

    subjects = sorted(set(surface["registered_dialects"]) | set(surface["ops"]))
    # Anti-vacuity: an empty or near-empty surface would make every loop below iterate
    # zero times and this gate pass having disposed of nothing.
    if not report.require(
        len(subjects) >= 20,
        f"only {len(subjects)} dialects in the snapshot; MLIR ships dozens, so the "
        f"surface extraction has failed and this gate would check almost nothing",
    ):
        return

    for ns in subjects:
        entry = entries.get(ns)
        if not report.require(
            entry is not None,
            f"dialect {ns!r} is in the MLIR surface but carries no disposition; every "
            f"dialect needs a chapter that teaches it or a stated reason it is out of "
            f"scope, or the corpus cannot tell a decision from an oversight",
        ):
            continue
        status = entry.get("status")
        if not report.require(
            status in VALID_STATUSES,
            f"{ns!r} has status {status!r}, which is not one of {VALID_STATUSES}",
        ):
            continue
        if status == "declared":
            report.require(
                (entry.get("reason") or "").strip(),
                f"{ns!r} is declared out of scope with no reason; a declared gap without "
                f"a reason is an omission wearing a label",
            )
            continue
        if status == "planned":
            report.require(
                (entry.get("slice") or "").strip(),
                f"{ns!r} is planned with no slice named; 'we intend to cover this' with "
                f"nothing attached is how a gap stays open forever while looking handled",
            )
            continue
        where = entry.get("where") or ""
        chapter = TRAINING_ROOT / where
        if not report.require(
            where and chapter.is_file(),
            f"{ns!r} is marked {status} by {where!r}, which is not a file under "
            f"{TRAINING_ROOT.name}/",
        ):
            continue
        # A citation counts only if the chapter names a REAL operation of that dialect,
        # in code. Every weaker rule tried here admitted something false: a prose search
        # credits "the tensor's shape." to the `shape` dialect; a `<ns>.<word>` pattern
        # over code credits `shape.ll` (a filename) and, worse, credits the `x86` and
        # `vector` dialects for `llvm.x86.pclmulqdq` and `llvm.vector.reduce.or.v4i1`,
        # which are LLVM *intrinsics* that merely contain the dialect's name. Matching
        # against the snapshot's own operation list has no such failure mode.
        ops = set(surface["ops"].get(ns, ()))
        if not report.require(
            ops,
            f"{ns!r} is marked {status} but ships no operations, so no chapter can name "
            f"one; an operation-less dialect is dispositioned 'declared' with the reason",
        ):
            continue
        code = code_spans(chapter.read_text(encoding="utf-8"))
        named = sorted(op for op in ops if op in code)
        report.require(
            named,
            f"{where} is cited as {status} for the {ns!r} dialect but names none of its "
            f"{len(ops)} operations in a code block or inline code span; a prose mention "
            f"of an ordinary word, a filename like `{ns}.ll`, or an LLVM intrinsic that "
            f"happens to contain the name is not a citation",
        )

    stale = [ns for ns in entries if ns not in subjects]
    for ns in stale:
        report.require(
            False,
            f"mlir-dialect-dispositions.json disposes of {ns!r}, which this MLIR does not "
            f"ship; drop the entry or regenerate the snapshot",
        )

    counts = {s: sum(1 for e in entries.values() if e.get("status") == s) for s in VALID_STATUSES}
    named = sum(hit for _, hit in cover.values())
    total = sum(total for total, _ in cover.values())
    print(
        "[cover]   "
        + ", ".join(f"{counts[s]} {s}" for s in VALID_STATUSES)
        + f"; {named}/{total} operations named"
    )
    report.done("dispositions")


def check_misalignments(report: Report, surface: dict) -> None:
    """Headers and registry disagreeing is a fact to record, not a failure to hide."""
    entries = json.loads(DISPOSITIONS.read_text(encoding="utf-8"))
    recorded = entries.get("misalignments", {})
    for ns in surface["unregistered_with_ops"]:
        report.require(
            ns in recorded,
            f"the generated headers ship operations for {ns!r} but this mlir-opt does not "
            f"register it, and no misalignment entry explains that; record it or explain "
            f"why the build differs",
        )
    for ns in surface["registered_without_ops"]:
        report.require(
            ns in recorded,
            f"{ns!r} is registered but ships no operations, and no misalignment entry "
            f"says why; an attribute-only dialect is a legitimate answer, an extraction "
            f"failure is not, and they look identical here",
        )
    for ns, note in recorded.items():
        report.require(
            (note or "").strip(),
            f"misalignment entry {ns!r} has no explanation",
        )
    print(
        f"[align]   {len(surface['unregistered_with_ops'])} unregistered-with-ops, "
        f"{len(surface['registered_without_ops'])} registered-without-ops, all recorded"
    )
    report.done("misalignments")


def check_drift(report: Report, surface: dict, mlir_opt: str, tblgen: str | None) -> None:
    """The snapshot against a live toolchain, reported rather than enforced.

    Enforcing equality here would be wrong: which dialects a build registers is a build
    decision, and the two builds this corpus runs on disagree. What IS enforced is that
    the snapshot is not a fiction -- every dialect it records must still exist somewhere
    in the live build, in the registry or in the headers.
    """
    live = read_surface(mlir_opt, tblgen)
    if not report.require(live is not None, "could not read the live MLIR surface"):
        return
    assert live is not None
    live_names = set(live["registered_dialects"]) | set(live["ops"])
    snap_names = set(surface["registered_dialects"]) | set(surface["ops"])
    vanished = sorted(snap_names - live_names)
    report.require(
        not vanished,
        f"the snapshot records dialects this MLIR does not have anywhere: {vanished[:8]}; "
        f"regenerate with --emit-surface",
    )
    added = sorted(live_names - snap_names)
    if added:
        print(f"[drift]   this build adds {len(added)} dialect(s) the snapshot lacks: {added}")
    else:
        print("[drift]   the snapshot's dialects all exist in this build")
    report.done("drift")


# --------------------------------------------------------------------------
# The chapter's one table, computed rather than typed
# --------------------------------------------------------------------------


def render_block(surface: dict, cover: dict) -> str:
    entries = json.loads(DISPOSITIONS.read_text(encoding="utf-8"))["dialects"]
    buckets: dict[str, list[tuple[str, int, int]]] = {s: [] for s in VALID_STATUSES}
    for ns in sorted(set(surface["registered_dialects"]) | set(surface["ops"])):
        total, hit = cover.get(ns, (0, 0))
        buckets.setdefault(entries.get(ns, {}).get("status", "declared"), []).append(
            (ns, total, hit)
        )

    rows = [
        "| disposition | dialects | operations | operations the corpus names |",
        "| --- | ---: | ---: | ---: |",
    ]
    for status in VALID_STATUSES:
        group = buckets.get(status, [])
        rows.append(
            f"| {status} | {len(group)} | {sum(t for _, t, _ in group)} | {sum(h for _, _, h in group)} |"
        )
    total_ops = sum(t for g in buckets.values() for _, t, _ in g)
    total_hit = sum(h for g in buckets.values() for _, _, h in g)
    dialects = sum(len(g) for g in buckets.values())
    rows.append(f"| **all** | **{dialects}** | **{total_ops}** | **{total_hit}** |")
    rows.append("")
    rows.append("Dialects this corpus teaches, with what it names of each:")
    rows.append("")
    rows.append("| dialect | operations | named here |")
    rows.append("| --- | ---: | ---: |")
    for ns, total, hit in sorted(buckets.get("taught", []), key=lambda r: -r[2]):
        rows.append(f"| `{ns}` | {total} | {hit} |")
    return "\n".join(rows) + "\n"


def sync_block(report: Report | None, surface: dict, cover: dict, update: bool) -> None:
    if not CHAPTER.is_file():
        if report:
            report.require(False, f"{CHAPTER} is missing; the coverage table is shown nowhere")
        return
    text = CHAPTER.read_text(encoding="utf-8")
    match = _BLOCK.search(text)
    if match is None:
        if report:
            report.require(
                False,
                f"{CHAPTER.name} has no `<!-- generated: dialect-coverage -->` block; a "
                f"coverage table nothing displays is a measurement nobody reads",
            )
        return
    fresh = render_block(surface, cover)
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
            f"{CHAPTER.name}: the dialect-coverage block is stale; the snapshot and the "
            f"corpus now produce:\n" + fresh,
        )
        report.done("block")


CHECK_NAMES = ("rails", "dispositions", "misalignments", "block")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-surface",
        action="store_true",
        help="write reference/mlir-surface-23.json from the installed MLIR",
    )
    parser.add_argument(
        "--require-tools",
        action="store_true",
        help="fail rather than skip the live-toolchain checks (pass this from the CI job "
        "that installs MLIR)",
    )
    parser.add_argument(
        "--update", action="store_true", help="rewrite the chapter's generated block"
    )
    args = parser.parse_args(argv)

    mlir_opt = find_llvm_tool("mlir-opt")
    tblgen = find_llvm_tool("mlir-tblgen")

    if args.emit_surface:
        if mlir_opt is None:
            print("error: --emit-surface needs mlir-opt", file=sys.stderr)
            return 1
        return write_surface(mlir_opt, tblgen)

    if not SURFACE.is_file():
        print(
            f"error: {SURFACE} is missing; run --emit-surface on a host with MLIR", file=sys.stderr
        )
        return 1
    surface = json.loads(SURFACE.read_text(encoding="utf-8"))
    cover = coverage(surface, corpus_text())

    if args.update:
        sync_block(None, surface, cover, update=True)
        return 0

    report = Report()
    check_rails(report, surface)
    check_dispositions(report, surface, cover)
    check_misalignments(report, surface)
    sync_block(report, surface, cover, update=False)

    # The live-toolchain check is the one half that needs MLIR present. Everything above
    # runs off the checked-in snapshot, so a host without a toolchain still verifies every
    # disposition rather than skipping the gate wholesale.
    if mlir_opt is None:
        if args.require_tools:
            report.require(
                False,
                "--require-tools was passed but mlir-opt is absent; the job that installs "
                "the MLIR toolchain is the one that owns the drift check",
            )
        else:
            print(
                "[skip]    mlir-opt is absent; the snapshot drift check belongs to the CI "
                "job that installs MLIR",
                file=sys.stderr,
            )
    else:
        check_drift(report, surface, mlir_opt, tblgen)

    missing = sorted(set(CHECK_NAMES) - report.completed)
    report.require(
        not missing,
        f"these checks did not run to completion: {', '.join(missing)}; the run stopped "
        f"early and this verdict covers less than it appears to",
    )
    return report.verdict("mlir coverage gate")


if __name__ == "__main__":
    sys.exit(main())
