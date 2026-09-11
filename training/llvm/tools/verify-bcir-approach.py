#!/usr/bin/env python3
"""Hold the `22-bcir-approach/` chapters to numbers BCIR itself produces.

    python3 training/llvm/tools/verify-bcir-approach.py            # check
    python3 training/llvm/tools/verify-bcir-approach.py --update    # refresh the blocks

WHY THIS EXISTS. This subject teaches what K_BCIR prices, what the verifier laws refuse,
and why legality is decided before cost -- claims about *this repository*, not about a
compiler somewhere. A chapter that quotes a score, a chosen lane width, or a policy weight
is making a checkable statement, so nothing here is quoted by hand: every table between a
`<!-- generated: NAME -->` / `<!-- /generated -->` pair is recomputed from `bcir/` and
compared byte for byte. Edit BCIR's cost model and the chapter fails until someone reruns
this with `--update` and reads what changed.

Three checks do not regenerate anything, because their subject is agreement between rails:

  * `axis-order`   -- the 12 axis names, in order, must be the same list in the oracle
    (`bcir/kbcir/cost.py`), the MLIR attribute (`mlir/include/BCIR/BCIRAttrs.td`) and the
    prose (`docs/PARITY.md`). Three copies exist; this makes the third and fourth
    copies checked rather than trusted.
  * `law-rails`    -- every R-law a chapter names must be findable on BOTH rails. The
    oracle's R24/R25 are NOT in `bcir/verify/`: they are enforced at encode time under
    `bcir/asn1/`, pinned to the law rail by `bcir/tests/test_asn1_law_parity.py`. A
    checker that reads only `bcir/verify/` concludes the oracle lacks two laws it has.
  * `no-dead-passes` -- no chapter may name an `opt` pass the declared toolchain major
    does not have. LLVM 15 -> 23 removed 28 pass names; a chapter written against an old
    pipeline goes stale silently, and `opt` is the authority on what exists.

A gate that only regenerated tables would pass over a chapter whose PROSE contradicted
them, so the prose claims that carry weight are checked too, by name, in `PROSE_CLAIMS`.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llvm_toolchain import find_llvm_tool  # noqa: E402

SUBJECT = Path(__file__).resolve().parents[1]
REPO = SUBJECT.parents[1]
CHAPTERS = SUBJECT / "22-bcir-approach"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BLOCK = re.compile(
    r"(?P<open><!-- generated: (?P<name>[a-z0-9-]+) -->\n)(?P<body>.*?)(?P<close><!-- /generated -->)",
    re.DOTALL,
)


class Report:
    """Every exit is a verdict: findings accumulate, nothing raises past the caller."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def require(self, condition: object, message: str) -> bool:
        self.checks += 1
        if not condition:
            self.failures.append(message)
        return bool(condition)


# --------------------------------------------------------------------------
# The generated blocks, each recomputed from BCIR
# --------------------------------------------------------------------------

THERMAL_CAPS = (2000, 1088, 1000, 700, 500, 300)
SUBSTRATES = ("x86-64-avx512", "x86-64-avx2", "aarch64-neon", "nvptx")


def _bcir():
    """The five handles every block needs.

    `bcir.kbcir.weights` is resolved through `importlib` on purpose: the package binds
    the name `weights` to the FUNCTION of that name, so both `from bcir.kbcir import
    weights` and `import bcir.kbcir.weights as w` hand back the callable -- attribute
    lookup finds the function before the submodule -- and every policy lookup on it
    fails. `import_module` returns what is in `sys.modules`, which is the module.
    """
    import importlib

    policy_weights = importlib.import_module("bcir.kbcir.weights")
    from bcir import kbcir
    from bcir.examples import vector_add
    from bcir.kbcir import cost, realize

    return kbcir, vector_add, cost, realize, policy_weights


def _profiles(realize):
    return {
        "x86-64-avx512": realize.HProfile(
            name="x86-64-avx512",
            triple="x86_64-avx512",
            lane_widths=(1, 8, 16),
            isa_features=frozenset({"avx512"}),
        ),
        "x86-64-avx2": realize.HProfile(),
        "aarch64-neon": realize.HProfile(name="aarch64-neon", triple="aarch64", lane_widths=(1, 4)),
        "nvptx": realize.HProfile(name="nvptx", triple="nvptx64", lane_widths=(1, 32), warp=32),
    }


def block_thermal_ladder() -> str:
    kbcir, vector_add, cost, realize, _ = _bcir()
    module = vector_add(1024)
    h = _profiles(realize)["x86-64-avx512"]
    theta = realize.Theta()
    thermal = cost.DIMS.index("thermal")
    lines = [
        "```text",
        "thermal cap   chosen    score   thermal used",
    ]
    unbounded = realize.optimize(module, h, theta)
    used = kbcir.plan_resources(unbounded, theta).v[thermal]
    lines.append(
        f"{'(none)':>11}   {unbounded.steps[0].candidate.name:<8} {unbounded.score:>6}   {used:>12}"
    )
    for cap in THERMAL_CAPS:
        try:
            result = kbcir.optimize_constrained(
                module, h, theta, budget=kbcir.Budget(caps=((thermal, cap),))
            )
        except kbcir.Infeasible:
            lines.append(f"{cap:>11}   {'--':<8} {'--':>6}   Infeasible")
            continue
        used = kbcir.plan_resources(result, theta).v[thermal]
        lines.append(
            f"{cap:>11}   {result.steps[0].candidate.name:<8} {result.score:>6}   {used:>12}"
        )
    lines.append("```")
    return "\n".join(lines) + "\n"


def block_substrates() -> str:
    _, vector_add, _, realize, _ = _bcir()
    module = vector_add(1024)
    profiles = _profiles(realize)
    rows = ["| substrate | lane widths | chosen | K_BCIR score |", "| --- | --- | --- | --- |"]
    for name in SUBSTRATES:
        h = profiles[name]
        result = realize.optimize(module, h, realize.Theta())
        widths = ", ".join(str(w) for w in h.lane_widths)
        rows.append(
            f"| `{name}` | {widths} | `{result.steps[0].candidate.name}` | {result.score} |"
        )
    return "\n".join(rows) + "\n"


def block_theta() -> str:
    _, vector_add, _, realize, _ = _bcir()
    module = vector_add(1024)
    h = _profiles(realize)["x86-64-avx512"]
    rows = ["| Theta | chosen | K_BCIR score |", "| --- | --- | --- |"]
    for label, theta in (
        ("cool (all zero)", realize.Theta()),
        ("thermal=900, power=900", realize.Theta(thermal=900, power=900)),
    ):
        result = realize.optimize(module, h, theta)
        rows.append(f"| {label} | `{result.steps[0].candidate.name}` | {result.score} |")
    return "\n".join(rows) + "\n"


def _producer_counts(cost) -> dict[str, int]:
    """Files under non-test `bcir/` that put a nonzero in each axis, by keyword.

    Deliberately keyword-only, and the chapter says so: `realize._cost` builds the
    planner's innermost cost vector POSITIONALLY, so this scan cannot see it. That is the
    point being taught -- a survey of an interface is not a survey of its callers.
    """
    counts = {dim: 0 for dim in cost.DIMS}
    for path in sorted((REPO / "bcir").rglob("*.py")):
        if "tests" in path.relative_to(REPO / "bcir").parts:
            continue
        text = path.read_text(encoding="utf-8")
        for dim in cost.DIMS:
            for match in re.finditer(rf"\b{dim}\s*=\s*([^,)\s][^,)]*)", text):
                value = match.group(1).strip()
                if value not in ("0", "0,", "None") and "def " not in value:
                    counts[dim] += 1
                    break
    return counts


def block_axes() -> str:
    _, _, cost, _, weights = _bcir()
    counts = _producer_counts(cost)
    policies = (weights.PERF, weights.THROUGHPUT, weights.ENERGY, weights.SAFE)
    header = "| # | axis | files that produce it | " + " | ".join(p.name for p in policies) + " |"
    rows = [header, "| ---: | --- | ---: | " + " | ".join("---:" for _ in policies) + " |"]
    for index, dim in enumerate(cost.DIMS):
        weights_row = " | ".join(str(p.base[index]) for p in policies)
        rows.append(f"| {index} | `{dim}` | {counts[dim]} | {weights_row} |")
    return "\n".join(rows) + "\n"


def check_irdl_quotes(report) -> None:
    """The IRDL the chapter quotes must still be the IRDL BCIR ships.

    22-bcir-approach/04-dialect-as-data.md now shows real operation definitions lifted
    from `mlir/irdl/bcir.irdl.mlir` -- the point being that IRDL is readable, which only
    works if a reader is shown the actual thing. A quotation is a copy, and a copy drifts:
    if BCIR renames an operation or changes a constraint, the chapter keeps teaching the
    old one and nothing notices. So every construct the chapter quotes is checked against
    the file it was quoted from.

    This reads BCIR's rail rather than importing from it. The training corpus is never a
    build dependency of BCIR, and reading a file to check a quotation does not make it one.
    """
    projection = REPO / "mlir/irdl/bcir.irdl.mlir"
    chapter = CHAPTERS / "04-dialect-as-data.md"
    if not report.require(
        projection.is_file(), f"{projection} is missing; the chapter quotes a file that is gone"
    ):
        return
    if not report.require(chapter.is_file(), f"{chapter} is missing"):
        return

    source = projection.read_text(encoding="utf-8")
    quoted = chapter.read_text(encoding="utf-8")

    # The constraint and declaration operations the chapter names as IRDL's vocabulary.
    for op in (
        "irdl.dialect",
        "irdl.type",
        "irdl.operation",
        "irdl.region",
        "irdl.regions",
        "irdl.operands",
        "irdl.results",
        "irdl.any",
        "irdl.is",
    ):
        if op in quoted:
            report.require(
                op in source,
                f"04-dialect-as-data.md teaches `{op}` but BCIR's IRDL projection no longer "
                f"uses it; the chapter is quoting a vocabulary the file has moved past",
            )

    # The specific operation definitions it reproduces. Matched with a boundary rather
    # than as a substring: `irdl.operation @loadX` contains `irdl.operation @load`, so a
    # plain `in` test would call a renamed operation present and report nothing. That is
    # the same substring trap that credited the `x86` dialect for `llvm.x86.pclmulqdq`,
    # and it survived into this check until its own RED proof refused to fire.
    for name in ("@module", "@resource", "@load"):
        if re.search(rf"irdl\.operation {re.escape(name)}\b", quoted):
            report.require(
                re.search(rf"irdl\.operation {re.escape(name)}\b", source),
                f"04-dialect-as-data.md reproduces `irdl.operation {name}` but the "
                f"projection no longer declares it; requote or drop the example",
            )

    # And the claim that the projection uses no compiled-C++ escape hatch, which is the
    # whole argument for it being loadable by a stock tool.
    if "irdl.c_pred" in quoted:
        report.require(
            "irdl.c_pred" not in source.replace("irdl.c_pred (it requires", "")
            or source.count("irdl.c_pred") <= 1,
            "04-dialect-as-data.md says the projection uses no irdl.c_pred, but the "
            "projection now contains one; a definition with a C++ predicate is not "
            "loadable by a stock mlir-opt, which is the property the chapter claims",
        )
    print("[irdl]    every IRDL construct the chapter quotes is still in BCIR's projection")


def block_irdl() -> str:
    """BCIR's ODS dialect against its IRDL projection, counted from the files.

    Deliberately toolchain-free: `tools/irdl/check_inventory.py` reconciles three
    text sources -- the ODS dialect, the IRDL projection and the manifest of
    operations declared unprojected -- and needs no mlir-opt, so this table is the
    same on a host with the MLIR toolchain and one without. The round-trip through
    stock `mlir-opt` is checked separately, where a toolchain exists.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "bcir_irdl_inventory", REPO / "tools/irdl/check_inventory.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    counts = module.audit(REPO)["counts"]
    corpus = sorted((REPO / "mlir/test/irdl").glob("*.mlir"))
    rows = [
        "| | count |",
        "| --- | ---: |",
        f"| operations the ODS dialect defines | {counts.get('ods', 0)} |",
        f"| operations the IRDL projection declares | {counts.get('irdl', 0)} |",
        f"| projected under the same name | {counts.get('projected_exact', 0)} |",
        f"| projected under a renamed spelling (IRDL admits no dots) "
        f"| {counts.get('projected_renamed', 0)} |",
        f"| declared unprojected, each with a stated reason "
        f"| {counts.get('unprojected_declared', 0)} |",
        f"| generic-syntax corpus files the projection is validated against | {len(corpus)} |",
    ]
    return "\n".join(rows) + "\n"


BLOCKS = {
    "irdl-projection": block_irdl,
    "thermal-ladder": block_thermal_ladder,
    "substrates": block_substrates,
    "theta": block_theta,
    "axes": block_axes,
}


# --------------------------------------------------------------------------
# The cross-rail checks, which regenerate nothing
# --------------------------------------------------------------------------


def check_axis_order(report: Report) -> None:
    from bcir.kbcir import cost

    attrs = (REPO / "mlir/include/BCIR/BCIRAttrs.td").read_text(encoding="utf-8")
    block = re.search(
        r"def BCIR_CostVectorAttr\b.*?let parameters = \(ins(?P<params>.*?)\);", attrs, re.DOTALL
    )
    report.require(block, "BCIRAttrs.td no longer declares BCIR_CostVectorAttr parameters")
    if not block:
        return
    law_order = tuple(re.findall(r"\$([a-z_]+)", block.group("params")))
    report.require(
        law_order == cost.DIMS,
        f"the law rail's cost-vector axis order {law_order} is not the oracle's {cost.DIMS}",
    )

    parity = (REPO / "docs/PARITY.md").read_text(encoding="utf-8")
    prose = re.search(r"same dimension order:\s*`([^`]+)`", parity, re.DOTALL)
    report.require(prose, "docs/PARITY.md no longer states the cost-vector axis order")
    if prose:
        stated = tuple(name.strip() for name in prose.group(1).replace("\n", " ").split(","))
        report.require(
            stated == cost.DIMS,
            f"docs/PARITY.md states the axis order as {stated}, the oracle has {cost.DIMS}",
        )


# Deliberately NOT `R(2[0-5]|1[0-9]|[1-9])`. Baking the valid range into the extractor
# makes the range unfalsifiable: a chapter naming R26 would not be recognized as naming a
# law at all, so the check that laws exist on both rails would pass over the one case it
# most needs to catch. Extract any R-number; let the rails decide which are real.
LAW_PATTERN = re.compile(r"\bR(\d+)\b")


def check_law_rails(report: Report, chapters: list[Path]) -> None:
    """Every R-law a chapter names must be findable on both rails.

    The oracle's R24/R25 are enforced at encode time under `bcir/asn1/`, not in
    `bcir/verify/`; a checker that reads only the latter reports two laws missing that the
    oracle has, and freezes that error into whatever prose it is guarding.
    """
    named: set[str] = set()
    for chapter in chapters:
        named.update(f"R{number}" for number in LAW_PATTERN.findall(chapter.read_text("utf-8")))
    report.require(named, "no chapter names an R-law; this subject exists to teach them")

    oracle_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((REPO / "bcir/verify").rglob("*.py"))
        + sorted((REPO / "bcir/asn1").rglob("*.py"))
    )
    law_text = "\n".join(
        path.read_text(encoding="utf-8")
        for pattern in ("mlir/lib/**/*.cpp", "mlir/include/**/*.td", "mlir/lib/**/*.h")
        for path in sorted(REPO.glob(pattern))
    )
    for law in sorted(named, key=lambda name: int(name[1:])):
        report.require(
            re.search(rf"\b{law}\b", oracle_text),
            f"{law} is named by a chapter but no oracle module under bcir/verify or "
            f"bcir/asn1 mentions it",
        )
        report.require(
            re.search(rf"\b{law}\b", law_text),
            f"{law} is named by a chapter but the MLIR law rail does not mention it",
        )


def check_irdl_roundtrip(report: Report) -> None:
    """Run BCIR's IRDL corpus through stock mlir-opt, where one exists.

    This is the claim `04-dialect-as-data.md` is built on -- that a dialect defined
    as data is loadable by a tool that was never built with it -- so the chapter
    does not get to assert it. `tools/irdl/check_corpus.sh` owns the round-trip;
    this asks it for a verdict. Where no mlir-opt is installed the corpus job says
    so by name rather than passing quietly: the MLIR rail job owns that toolchain.
    """
    mlir_opt = find_llvm_tool("mlir-opt")
    if mlir_opt is None:
        print(
            "[skip]    mlir-opt is absent; the IRDL round-trip is the MLIR rail job's",
            file=sys.stderr,
        )
        return
    try:
        completed = subprocess.run(
            ["bash", str(REPO / "tools/irdl/check_corpus.sh")],
            capture_output=True,
            text=True,
            timeout=900,
            cwd=REPO,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        report.require(False, f"the IRDL corpus check did not run: {exc}")
        return
    output = completed.stdout + completed.stderr
    report.require(
        completed.returncode == 0,
        f"stock mlir-opt no longer round-trips BCIR's IRDL corpus: {output.strip()[-400:]}",
    )
    # Anti-vacuity: the script exits 0 when it finds no mlir-opt of its own, so a
    # green exit code alone would not mean any file was parsed.
    report.require(
        "all corpus files round-trip" in output,
        "the IRDL corpus check exited 0 without reporting a round-trip; it resolved "
        "no mlir-opt of its own and checked nothing",
    )


def check_no_dead_passes(report: Report, chapters: list[Path]) -> None:
    """No chapter may name an `opt` pass the declared toolchain does not have."""
    opt = find_llvm_tool("opt")
    if opt is None:
        print(
            "[skip]    opt is absent; the dead-pass check is the toolchain job's", file=sys.stderr
        )
        return
    try:
        listing = subprocess.run(
            [opt, "--print-passes"], capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        report.require(False, f"opt --print-passes did not run: {exc}")
        return
    available = {
        line.strip().split("<")[0]
        for line in listing.stdout.splitlines()
        if line.startswith(("  ", "\t")) and line.strip()
    }
    report.require(len(available) > 50, f"opt listed only {len(available)} passes; unusable")

    examined = 0
    for chapter in chapters:
        for pipeline in re.findall(r"-passes=([A-Za-z0-9,<>()_\-]+)", chapter.read_text("utf-8")):
            examined += 1
            for name in re.split(r"[,()]", pipeline):
                base = name.split("<")[0].strip()
                if not base or base in ("default", "module", "function", "loop", "cgscc"):
                    continue
                report.require(
                    base in available or base.startswith("bcir-"),
                    f"{chapter.name}: names pass {base!r}, which this opt does not have",
                )

    # Anti-vacuity, as a state rather than a comment. This loop currently examines a
    # single reference -- `opt -passes=verify` in chapter 01 -- so one edit to one
    # sentence would leave the check iterating zero times and passing over anything.
    # A gate that can go green without looking at its subject is broken while green.
    report.require(
        examined,
        "no chapter names an opt pass pipeline, so the dead-pass check examined nothing; "
        "either a chapter lost its `-passes=` reference or this check no longer belongs here",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="rewrite the generated blocks")
    args = parser.parse_args(argv)

    report = Report()
    if not CHAPTERS.is_dir():
        print(f"error: {CHAPTERS.relative_to(REPO)} does not exist", file=sys.stderr)
        return 1
    chapters = sorted(CHAPTERS.glob("*.md"))
    report.require(chapters, "the subject has no chapters, so this gate would check nothing")

    seen: set[str] = set()
    for chapter in chapters:
        text = chapter.read_text(encoding="utf-8")
        updated = text
        for match in BLOCK.finditer(text):
            name = match.group("name")
            seen.add(name)
            if not report.require(name in BLOCKS, f"{chapter.name}: unknown block {name!r}"):
                continue
            expected = BLOCKS[name]()
            if args.update:
                updated = updated.replace(
                    match.group(0), match.group("open") + expected + match.group("close")
                )
            else:
                report.require(
                    match.group("body") == expected,
                    f"{chapter.name}: block {name!r} is stale; BCIR now produces:\n{expected}",
                )
        if args.update and updated != text:
            chapter.write_text(updated, encoding="utf-8")
            print(f"[update]  {chapter.relative_to(REPO)}")

    missing = sorted(set(BLOCKS) - seen)
    report.require(
        not missing,
        f"{len(missing)} generated block(s) are computed but no chapter shows them: {missing}; "
        "a block nothing displays is a check over nothing",
    )
    check_axis_order(report)
    check_law_rails(report, chapters)
    check_no_dead_passes(report, chapters)
    check_irdl_roundtrip(report)
    check_irdl_quotes(report)

    if report.failures:
        print("bcir-approach gate: FAILED", file=sys.stderr)
        for failure in report.failures[:40]:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"bcir-approach gate: PASSED ({report.checks} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
