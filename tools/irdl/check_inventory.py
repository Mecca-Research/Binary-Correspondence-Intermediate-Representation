#!/usr/bin/env python3
"""ODS -> IRDL inventory gate (S0-9): the projection's supported subset is a MANIFEST, and the
parity between the ODS dialect, the IRDL projection and that manifest is checked both ways.

    python tools/irdl/check_inventory.py                       # report, exit 0 PASS / 1 FAIL / 2 vacuous
    python tools/irdl/check_inventory.py --json-out build/irdl-inventory.json

WHY. `tools/irdl/check_corpus.sh` round-trips the corpus through stock mlir-opt, which proves
the projection LOADS and accepts the corpus; it says nothing about which of the dialect's
operations the projection covers. The 2026-07/08 assessment found the coverage incomplete with
no gate: an operation added to ODS never had to be projected or declared unprojected, and an
operation renamed or removed from ODS could leave an orphan in the projection. This tool reads
the three sources themselves (laws.md L14: never a fourth list that drifts) and refuses drift:

  * every ODS operation is projected, or declared unprojected in the manifest with a reason;
  * a manifest entry for an operation the projection DOES cover is stale;
  * a manifest entry for an operation ODS does not define is a ghost;
  * an IRDL operation no ODS operation maps to is an orphan;
  * two ODS operations that spell the same IRDL name collide.

THE NAMING RULE. IRDL requires an operation name of lowercase letters, digits and underscores
(mlir-opt refuses `irdl.operation @"gem.stream_pack"`), so the projection spells a dotted ODS
name with underscores: `gem.stream_pack` -> `gem_stream_pack`. The rule is applied here, not
tabulated per operation; the manifest records it so a reader of the projection knows why its
names differ from the dialect's.

THE VERDICT (laws.md L1/L2): PASS, FAIL (every finding names its operation and its kind), or
INVALID/VACUOUS when an inventory is empty -- a gate over nothing has audited nothing. No
toolchain is needed: the inputs are text, so this runs on every host and in the quick tier.
The text is read the way its compilers read it: comments are not declarations (`active_text`),
and a manifest whose root is not an object is a finding, never a traceback."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INCLUDE_DIR = Path("mlir/include/BCIR")
IRDL_FILE = Path("mlir/irdl/bcir.irdl.mlir")
MANIFEST = Path("mlir/irdl/MANIFEST.json")
SCHEMA = "bcir-irdl-manifest.v1"
NAMING_RULE = "dots-to-underscores"

# `def BCIR_FooOp : BCIR_Op<"foo.bar", [...]>` -- every operation of the dialect is declared
# through the one BCIR_Op class (asserted by the gate: a second class would be a second
# spelling of "operation" this parser would miss).
_ODS_OP = re.compile(r'def\s+\w+\s*:\s*(\w+)<"([a-z][a-z0-9_.]*)"')
_IRDL_OP = re.compile(r'irdl\.operation\s+@"?([A-Za-z0-9_.]+)"?')
_ODS_OP_CLASS = "BCIR_Op"


def active_text(text: str) -> str:
    """`text` with every `//` line comment and `/* */` block comment blanked (string literals
    untouched, newlines kept). ODS (TableGen) and IRDL (MLIR) share C's comment syntax, and the
    two parsers below read only what their compilers would: an `irdl.operation` or an ODS `def`
    disabled by commenting it out is text, not an operation. The gate used to count it as one --
    a projection with a commented-out operation reported PASS, and for an operation with no
    corpus witness the round-trip gate would not have exposed the missing definition either
    (laws.md L2: a gate must be able to fire on the change it exists to catch)."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':  # a string literal: copied verbatim, escapes included
            out.append(c)
            i += 1
            while i < n:
                ch = text[i]
                out.append(ch)
                i += 1
                if ch == "\\" and i < n:
                    out.append(text[i])
                    i += 1
                elif ch == '"':
                    break
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":  # a line comment
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":  # a block comment (TableGen nests them)
            depth, i = 1, i + 2
            while i < n and depth:
                if text.startswith("/*", i):
                    depth += 1
                    i += 2
                elif text.startswith("*/", i):
                    depth -= 1
                    i += 2
                else:
                    if text[i] == "\n":
                        out.append("\n")
                    i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def projected_name(ods_name: str) -> str:
    """The IRDL spelling of an ODS operation name (the one naming rule)."""
    return ods_name.replace(".", "_")


def ods_operations(include_dir: Path) -> tuple[dict[str, str], list[dict]]:
    """Operation name -> defining file, plus findings for declarations the parser refuses."""
    ops: dict[str, str] = {}
    findings: list[dict] = []
    for td in sorted(include_dir.glob("*.td")):
        text = active_text(td.read_text(encoding="utf-8"))
        for match in _ODS_OP.finditer(text):
            op_class, name = match.group(1), match.group(2)
            if op_class != _ODS_OP_CLASS:
                findings.append(
                    {
                        "kind": "unknown-op-class",
                        "operation": name,
                        "detail": f"{td.name} declares it through {op_class}, not {_ODS_OP_CLASS}",
                    }
                )
                continue
            if name in ops:
                findings.append(
                    {
                        "kind": "duplicate-ods-op",
                        "operation": name,
                        "detail": f"declared in {ops[name]} and {td.name}",
                    }
                )
                continue
            ops[name] = td.name
    return ops, findings


def irdl_operations(irdl_file: Path) -> tuple[list[str], list[dict]]:
    names = _IRDL_OP.findall(active_text(irdl_file.read_text(encoding="utf-8")))
    findings = [
        {
            "kind": "duplicate-irdl-op",
            "operation": name,
            "detail": "declared twice in the projection",
        }
        for name in sorted({n for n in names if names.count(n) > 1})
    ]
    return names, findings


def load_manifest(path: Path) -> tuple[dict, list[dict]]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {}, [{"kind": "manifest-unreadable", "operation": "", "detail": str(exc)}]
    if not isinstance(manifest, dict):  # valid JSON whose root is not an object: `[]`, `null`
        return {}, [
            {
                "kind": "manifest-shape",
                "operation": "",
                "detail": f"the manifest's top level must be a JSON object (got {type(manifest).__name__})",
            }
        ]
    findings: list[dict] = []
    if manifest.get("schema") != SCHEMA:
        findings.append(
            {"kind": "manifest-schema", "operation": "", "detail": f"expected {SCHEMA!r}"}
        )
    if manifest.get("naming_rule") != NAMING_RULE:
        findings.append(
            {
                "kind": "manifest-naming-rule",
                "operation": "",
                "detail": f"expected {NAMING_RULE!r} (the rule this gate applies)",
            }
        )
    unprojected = manifest.get("unprojected")
    if not isinstance(unprojected, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v.strip()
        for k, v in (unprojected or {}).items()
    ):
        findings.append(
            {
                "kind": "manifest-shape",
                "operation": "",
                "detail": "`unprojected` must map every operation name to a non-empty reason",
            }
        )
        manifest["unprojected"] = {}
    return manifest, findings


def audit(
    root: Path | str | None = None,
    *,
    include_dir: Path | None = None,
    irdl_file: Path | None = None,
    manifest_path: Path | None = None,
) -> dict:
    """Reconcile the three sources. Returns {state, findings, counts, ...}; never raises on
    a malformed input -- an unreadable source is a finding (L1)."""
    base = Path(root) if root is not None else ROOT
    include_dir = include_dir or base / INCLUDE_DIR
    irdl_file = irdl_file or base / IRDL_FILE
    manifest_path = manifest_path or base / MANIFEST

    findings: list[dict] = []
    try:
        ods, ods_findings = ods_operations(include_dir)
    except OSError as exc:
        return _report(
            "INVALID/VACUOUS", [{"kind": "ods-unreadable", "operation": "", "detail": str(exc)}], {}
        )
    try:
        irdl_names, irdl_findings = irdl_operations(irdl_file)
    except OSError as exc:
        return _report(
            "INVALID/VACUOUS",
            [{"kind": "irdl-unreadable", "operation": "", "detail": str(exc)}],
            {},
        )
    manifest, manifest_findings = load_manifest(manifest_path)
    findings += ods_findings + irdl_findings + manifest_findings

    counts = {"ods": len(ods), "irdl": len(irdl_names)}
    if not ods or not irdl_names:
        return _report(
            "INVALID/VACUOUS",
            findings
            + [
                {
                    "kind": "empty-inventory",
                    "operation": "",
                    "detail": f"ods={len(ods)} irdl={len(irdl_names)}",
                }
            ],
            counts,
        )

    irdl = set(irdl_names)
    unprojected: dict[str, str] = manifest.get("unprojected", {})

    # The naming rule must be injective over the dialect, or a projection could stand for
    # two operations at once.
    by_projected: dict[str, list[str]] = {}
    for name in ods:
        by_projected.setdefault(projected_name(name), []).append(name)
    for spelled, names in sorted(by_projected.items()):
        if len(names) > 1:
            findings.append(
                {
                    "kind": "collision",
                    "operation": ", ".join(sorted(names)),
                    "detail": f"both spell {spelled!r} under the {NAMING_RULE} rule",
                }
            )

    exact = renamed = 0
    mapped: set[str] = set()
    for name in sorted(ods):
        spelled = projected_name(name)
        if spelled in irdl:
            mapped.add(spelled)
            if spelled == name:
                exact += 1
            else:
                renamed += 1
            if name in unprojected:
                findings.append(
                    {
                        "kind": "stale-unprojected",
                        "operation": name,
                        "detail": f"the manifest declares it unprojected, but the projection defines @{spelled}",
                    }
                )
        elif name not in unprojected:
            findings.append(
                {
                    "kind": "undeclared",
                    "operation": name,
                    "detail": f"ODS declares it ({ods[name]}); the projection has no @{spelled} and "
                    "the manifest does not declare it unprojected",
                }
            )
    for name in sorted(unprojected):
        if name not in ods:
            findings.append(
                {
                    "kind": "ghost",
                    "operation": name,
                    "detail": "the manifest names an operation ODS does not define",
                }
            )
    for spelled in sorted(irdl - mapped):
        findings.append(
            {
                "kind": "orphan",
                "operation": spelled,
                "detail": "the projection defines it, but no ODS operation spells it under the naming rule",
            }
        )

    counts.update(
        {
            "projected_exact": exact,
            "projected_renamed": renamed,
            "unprojected_declared": len([n for n in unprojected if n in ods]),
        }
    )
    return _report("FAIL" if findings else "PASS", findings, counts)


def _report(state: str, findings: list[dict], counts: dict) -> dict:
    return {"state": state, "findings": findings, "counts": counts, "naming_rule": NAMING_RULE}


def render(report: dict) -> str:
    counts = report["counts"]
    lines = [
        f"check_inventory: {report['state']} ods={counts.get('ods', 0)} irdl={counts.get('irdl', 0)} "
        f"projected={counts.get('projected_exact', 0)}+{counts.get('projected_renamed', 0)}(renamed) "
        f"unprojected={counts.get('unprojected_declared', 0)} findings={len(report['findings'])}"
    ]
    for finding in report["findings"]:
        lines.append(f"  {finding['kind']:<20} {finding['operation']:<32} {finding['detail']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root", default=str(ROOT), help="repository root (default: this checkout)"
    )
    parser.add_argument("--json-out", help="write the report as JSON")
    args = parser.parse_args(argv)
    report = audit(Path(args.root))
    print(render(report))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return {"PASS": 0, "FAIL": 1}.get(report["state"], 2)


if __name__ == "__main__":
    raise SystemExit(main())
