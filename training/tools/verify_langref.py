#!/usr/bin/env python3
"""Reconcile `training/TRAINING_LANGREF.md` against the code it describes.

The LangRef is normative and the tools are its conformance oracle, which is a
useful arrangement exactly as long as somebody checks that the two still say the
same thing. Nobody does that by reading. A document restating a list that lives in
code is a mirror list, and a mirror list will drift (`docs/security/laws.md` L15)
-- silently, because both halves keep working while they disagree, and the reader
is the only thing that breaks.

So the document's normative tables are parsed and compared against the objects
they describe. The comparison is **two-sided**: a column in the code and missing
from the document is a finding, and so is a column in the document that the code
does not have. A one-sided check would let the document quietly describe a system
that no longer exists.

Declared scope: the tables and inline constants this file names. Prose is not
parsed and is not claimed to be checked -- the rule is that anything *enumerable*
is reconciled, and anything argued in sentences is reviewed by a person. Extending
the scope means adding a reconciler here, not teaching this one to read English.

    python3 training/tools/verify_langref.py
    python3 training/tools/verify_langref.py --langref path/to/other.md
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
CORPUS_ROOT = TOOLS_DIR.parent
REPO_ROOT = CORPUS_ROOT.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

DEFAULT_LANGREF = CORPUS_ROOT / "TRAINING_LANGREF.md"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2

#: The smallest number of reconciled facts that counts as having checked the
#: document. A parser that silently matched nothing would otherwise report a clean
#: run over a file it never understood, which is the vacuous-pass shape this tree
#: refuses everywhere else (L2).
MIN_RECONCILED = 60


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def require(self, condition: bool, message: str) -> bool:
        self.checks += 1
        if not condition:
            self.failures.append(message)
        return bool(condition)

    def same(self, label: str, document, code) -> bool:
        """Two-sided set comparison, reported as what each side has that the other lacks."""
        document, code = set(document), set(code)
        if document == code:
            return self.require(True, "")
        missing = sorted(code - document)
        extra = sorted(document - code)
        parts = []
        if missing:
            parts.append(f"the code has {missing} and the document does not")
        if extra:
            parts.append(f"the document has {extra} and the code does not")
        return self.require(False, f"{label}: " + "; ".join(parts))


# --------------------------------------------------------------------------
# Reading the document
# --------------------------------------------------------------------------


def sections(text: str) -> dict[str, str]:
    """The document split by heading, so a table is looked up where it is declared.

    Keyed by the heading's number (`3.2`, `7.1`) where it has one, because the
    titles are prose and the numbers are the stable handle.
    """
    found: dict[str, str] = {}
    current = None
    for line in text.splitlines():
        heading = re.match(r"^#{2,3}\s+(\d+(?:\.\d+)?)[.\s]", line)
        if heading:
            current = heading.group(1)
            found[current] = ""
        elif current is not None:
            found[current] += line + "\n"
    return found


def table_rows(block: str) -> list[list[str]]:
    """Every data row of every markdown table in a block, as stripped cells."""
    rows = []
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells):
            continue  # the separator row
        rows.append(cells)
    return rows


def ticked(cell: str) -> list[str]:
    """Every `code-spanned` token in a cell -- how the document names an identifier."""
    return re.findall(r"`([^`]+)`", cell)


def first_ticked(cell: str) -> str | None:
    names = ticked(cell)
    return names[0] if names else None


# --------------------------------------------------------------------------
# The reconcilers
# --------------------------------------------------------------------------


def check_columns(report: Report, blocks: dict[str, str], schema) -> None:
    """SS3.2's column table is the declared table, or the document is wrong."""
    rows = [row for row in table_rows(blocks.get("3.2", "")) if len(row) >= 6]
    documented = {}
    for name_cell, type_cell, role_cell, required, nullable, domain in (
        (row[0], row[1], row[2], row[3], row[4], row[5]) for row in rows
    ):
        name = first_ticked(name_cell)
        if name is None or name == "Column":
            continue
        documented[name] = {
            "type": type_cell.strip(),
            "role": role_cell.strip().strip("*"),
            "required": "yes" in required.lower(),
            "nullable": "yes" in nullable.lower(),
            "domain": domain,
        }
    report.require(
        len(documented) >= 8,
        f"anti-vacuity: only {len(documented)} column row(s) parsed from section 3.2",
    )
    report.same("3.2 columns", documented, (c.name for c in schema.COLUMNS))

    for column in schema.COLUMNS:
        stated = documented.get(column.name)
        if stated is None:
            continue
        report.require(
            stated["type"] == column.type,
            f"3.2 {column.name}: document says type {stated['type']!r}, code says {column.type!r}",
        )
        report.require(
            stated["role"] == column.role,
            f"3.2 {column.name}: document says role {stated['role']!r}, code says {column.role!r}",
        )
        report.require(
            stated["required"] == bool(column.required),
            f"3.2 {column.name}: document says required={stated['required']}, "
            f"code says {bool(column.required)}",
        )
        report.require(
            stated["nullable"] == bool(column.nullable),
            f"3.2 {column.name}: document says nullable={stated['nullable']}, "
            f"code says {bool(column.nullable)}",
        )
        # A declared domain or minimum must appear in the row: the whole point of
        # printing it is that a reader can rely on it without opening schema.py.
        for value in getattr(column, "domain", None) or ():
            report.require(
                str(value) in stated["domain"],
                f"3.2 {column.name}: domain value {value!r} is not in the document's row",
            )
        if getattr(column, "minimum", None) is not None:
            report.require(
                str(column.minimum) in stated["domain"],
                f"3.2 {column.name}: minimum {column.minimum} is not in the document's row",
            )


def check_roles(report: Report, blocks: dict[str, str], schema) -> None:
    rows = table_rows(blocks.get("3.1", ""))
    documented = [first_ticked(row[0]) for row in rows if first_ticked(row[0])]
    report.same("3.1 roles", documented, schema.ROLES)


def check_operators(report: Report, blocks: dict[str, str], plan) -> None:
    """SS5.2 names every operator, and a row may name several (`lt` `le` `gt` `ge`)."""
    documented: list[str] = []
    for row in table_rows(blocks.get("5.2", "")):
        documented += [name for name in ticked(row[0]) if name in plan.OPERATORS]
    report.same("5.2 operators", documented, plan.OPERATORS)
    text = blocks.get("5.2", "")
    for operator in plan.RANGE_OPERATORS:
        report.require(f"`{operator}`" in text, f"5.2: range operator {operator!r} is not named")


def check_objectives(report: Report, blocks: dict[str, str], plan) -> None:
    rows = table_rows(blocks.get("7.2", ""))
    documented = {}
    for row in rows:
        name, axis = first_ticked(row[0]), first_ticked(row[1]) if len(row) > 1 else None
        if name and name.isupper():
            documented[name] = axis
    report.same("7.2 objectives", documented, (o.name for o in plan.Objective))
    for objective in plan.Objective:
        stated = documented.get(objective.name)
        report.require(
            stated == objective.value,
            f"7.2 {objective.name}: document says axis {stated!r}, code says {objective.value!r}",
        )


def check_legality(report: Report, blocks: dict[str, str], plan) -> None:
    documented = [
        first_ticked(row[0]) for row in table_rows(blocks.get("7.1", "")) if first_ticked(row[0])
    ]
    report.same("7.1 legality rules", documented, plan.LEGALITY_RULES)


def check_backends(report: Report, blocks: dict[str, str], plan) -> None:
    rows = table_rows(blocks.get("8.3", ""))
    documented, exact = [], []
    for row in rows:
        name = first_ticked(row[0])
        if name not in plan.BACKENDS:
            continue
        documented.append(name)
        # Column 3 says whether the backend is exact. "no" anywhere in it means no.
        if len(row) > 2 and "no" not in row[2].lower():
            exact.append(name)
    report.same("8.3 backends", documented, plan.BACKENDS)
    report.same("8.3 exact backends", exact, plan.EXACT_BACKENDS)


def check_binary_formats(report: Report, blocks: dict[str, str], catalog) -> None:
    """SS4.2's struct codes and widths are the ones the reader actually unpacks."""
    expected = {
        "locator.bin": ("<HQI", catalog.LOCATOR_ENTRY_BYTES),
        "numeric.bin": ("<q", catalog.NUMERIC_CELL_BYTES),
        "order.bin": ("<I", catalog.ORDER_ENTRY_BYTES),
    }
    rows = table_rows(blocks.get("4.2", ""))
    seen = set()
    for row in rows:
        name = first_ticked(row[0])
        if name not in expected:
            continue
        seen.add(name)
        code, width = expected[name]
        stated_code = first_ticked(row[1]) if len(row) > 1 else None
        report.require(
            stated_code == code,
            f"4.2 {name}: document says struct {stated_code!r}, code uses {code!r}",
        )
        report.require(
            struct.calcsize(code) == width,
            f"4.2 {name}: {code!r} is {struct.calcsize(code)} bytes, the module says {width}",
        )
        stated_width = re.search(r"\b(\d+)\b", row[2]) if len(row) > 2 else None
        report.require(
            stated_width is not None and int(stated_width.group(1)) == width,
            f"4.2 {name}: document says {row[2] if len(row) > 2 else '(nothing)'} bytes, "
            f"code says {width}",
        )
    report.same("4.2 binary artifacts", seen, expected)


def check_artifact_names(report: Report, blocks: dict[str, str], catalog) -> None:
    """Every filename SS4.1 draws is the constant the builder writes."""
    block = blocks.get("4.1", "")
    for attribute in (
        "CATALOG_FILE",
        "POSTINGS_FILE",
        "LOCATOR_FILE",
        "IDS_FILE",
        "NUMERIC_FILE",
        "ORDER_FILE",
        "CURRENT_FILE",
        "SETS_DIR",
    ):
        value = getattr(catalog, attribute)
        report.require(
            value in block,
            f"4.1: {attribute} is {value!r} and the layout in the document does not show it",
        )


#: How the document is allowed to write a number: a decimal, or a power of two
#: with an optional subtraction, optionally parenthesised and negated. Anything
#: else is not a spelling this reconciler claims to read (declared scope).
_NUMBER = re.compile(r"-?\(?2\*\*(\d+)\)?(?:\s*-\s*(\d+))?|-?\d+(?:\.\d+)?")


def _value_of(spelling: str) -> float | None:
    """One of the document's number spellings, as a number. None if it is not one."""
    found = _NUMBER.fullmatch(spelling.strip())
    if found is None:
        return None
    exponent, subtracted = found.group(1), found.group(2)
    if exponent is None:
        try:
            return float(spelling)
        except ValueError:
            return None
    value = float(2 ** int(exponent))
    if subtracted is not None:
        value -= int(subtracted)
    return -value if spelling.strip().startswith("-") else value


def _numbers_near(text: str, name: str, window: int = 220) -> list[float]:
    """Every number the document writes shortly after naming `name`.

    A window rather than a strict grammar because the document is prose: the value
    may follow as `= 0.70`, as `is 0.70`, or inside a sentence. What matters is
    that the *document's own number* is read back and compared, never a number
    typed into this file -- that would make this checker a second mirror of the
    constant, which is the defect it exists to prevent.
    """
    numbers = []
    for match in re.finditer(re.escape(name), text):
        for token in _NUMBER.finditer(text[match.end() : match.end() + window]):
            value = _value_of(token.group(0))
            if value is not None:
                numbers.append(value)
    return numbers


def check_constants(report: Report, text: str, catalog, plan, embed) -> None:
    """Every constant the document states by value is the value in the code.

    Both halves are read from their own side: the code's value from the module,
    the document's from the sentence that names it.
    """
    cases = (
        ("MAX_BLOCK_ROWS", catalog.MAX_BLOCK_ROWS),
        ("ORDERED_INDEX_SHARE", catalog.ORDERED_INDEX_SHARE),
        ("SORTED_SCAN_SHARE", catalog.SORTED_SCAN_SHARE),
        ("MAX_JOINT_CELLS", plan.MAX_JOINT_CELLS),
        ("MAX_INDEXABLE_ROWS", catalog.MAX_INDEXABLE_ROWS),
        ("Q15_SCALE", embed.Q15_SCALE),
        ("NUMERIC_NULL", catalog.NUMERIC_NULL),
    )
    for name, value in cases:
        if not report.require(name in text, f"constants: the document never names {name}"):
            continue
        nearby = _numbers_near(text, name)
        report.require(
            any(abs(number - float(value)) < 1e-9 for number in nearby),
            f"constants: {name} is {value} in the code, and the document writes "
            f"{[int(n) if n.is_integer() else n for n in nearby] or 'no number'} beside it",
        )
    report.require(
        r"\0null" in text,
        f"constants: the reserved posting key is {catalog.NULL_KEY!r} and the document does not "
        "show it",
    )
    report.require(
        embed.Q15_KERNEL in text,
        f"constants: the ranking kernel is {embed.Q15_KERNEL} and the document does not name it",
    )


def check_gates(report: Report, blocks: dict[str, str]) -> None:
    """Every gate SS16 tabulates exists, and every gate on disk is tabulated."""
    documented = set()
    for row in table_rows(blocks.get("16", "")):
        name = first_ticked(row[0])
        if name and name.endswith(".py"):
            documented.add(name)
    on_disk = {path.name for path in TOOLS_DIR.glob("verify_*.py")}
    report.require(
        len(on_disk) >= 5, f"anti-vacuity: only {len(on_disk)} verifier(s) found on disk"
    )
    report.same("16 gates", documented, on_disk)


def check_interfaces(report: Report, text: str) -> None:
    """Every subsystem interface module is named somewhere in the document."""
    modules = sorted(
        path.stem for path in (TOOLS_DIR / "db").glob("*.py") if not path.stem.startswith("_")
    )
    report.require(len(modules) >= 4, f"anti-vacuity: {len(modules)} interface module(s) found")
    for module in modules:
        report.require(
            f"`{module}`" in text or f"db/{module}.py" in text,
            f"interfaces: `{module}` is a subsystem interface and the document never names it",
        )


def check_schema_ids(report: Report, text: str) -> None:
    """Every `bcir-training/*/vN` identifier in the tools is one the document knows."""
    declared = set()
    for path in sorted(TOOLS_DIR.glob("*.py")):
        if path.name == Path(__file__).name:
            continue
        declared |= set(
            re.findall(r'"(bcir-training/[a-z0-9-]+/v[0-9]+)"', path.read_text(encoding="utf-8"))
        )
    report.require(len(declared) >= 6, f"anti-vacuity: {len(declared)} schema id(s) discovered")
    documented = set(re.findall(r"bcir-training/[a-z0-9-]+/v[0-9]+", text))
    missing = sorted(declared - documented)
    report.require(
        not missing,
        f"schema ids: declared in the tools and absent from the document: {missing}",
    )


def check_provenance(report: Report, blocks: dict[str, str]) -> None:
    """The slice ladder is contiguous: S1..Sn with nothing skipped."""
    slices = set()
    for key in ("18.4",):
        for row in table_rows(blocks.get(key, "")):
            found = re.fullmatch(r"S(\d+)", (first_ticked(row[0]) or row[0]).strip())
            if found:
                slices.add(int(found.group(1)))
    report.require(len(slices) >= 10, f"anti-vacuity: {len(slices)} slice row(s) parsed")
    if slices:
        gaps = sorted(set(range(1, max(slices) + 1)) - slices)
        report.require(
            not gaps,
            f"18.4: the slice ladder skips {['S%d' % n for n in gaps]}, so a slice landed "
            "without a row or a row names a slice that did not",
        )


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--langref", type=Path, default=DEFAULT_LANGREF)
    args = parser.parse_args(argv)

    try:
        text = args.langref.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"verify_langref: {args.langref} is unreadable: {exc}", file=sys.stderr)
        return EXIT_USAGE

    # The engine is reached through its package, not around it: `catalog` and `plan`
    # have exactly one door, and a checker that opened a second one while asserting
    # that the document matches the code would be violating a rule it is standing
    # next to (`verify_database.py::check_interface_boundary`). `schema` and
    # `embed_chunks` are not behind that door and are imported directly.
    from db import engine

    import embed_chunks
    import schema

    catalog = engine.catalog()
    plan = engine.planner()

    report = Report()
    blocks = sections(text)
    report.require(
        len(blocks) >= 12,
        f"anti-vacuity: only {len(blocks)} numbered section(s) parsed from {args.langref.name}",
    )

    check_roles(report, blocks, schema)
    check_columns(report, blocks, schema)
    check_artifact_names(report, blocks, catalog)
    check_binary_formats(report, blocks, catalog)
    check_operators(report, blocks, plan)
    check_legality(report, blocks, plan)
    check_objectives(report, blocks, plan)
    check_backends(report, blocks, plan)
    check_constants(report, text, catalog, plan, embed_chunks)
    check_interfaces(report, text)
    check_schema_ids(report, text)
    check_gates(report, blocks)
    check_provenance(report, blocks)

    report.require(
        report.checks >= MIN_RECONCILED,
        f"anti-vacuity: {report.checks} fact(s) reconciled, fewer than the {MIN_RECONCILED} "
        "this document is known to state -- the parser matched almost nothing",
    )

    if report.failures:
        print("langref gate: FAILED", file=sys.stderr)
        for failure in report.failures[:40]:
            print(f"  - {failure}", file=sys.stderr)
        if len(report.failures) > 40:
            print(f"  ... and {len(report.failures) - 40} more", file=sys.stderr)
        return EXIT_FAILED

    print(
        f"langref gate: PASSED ({report.checks} fact(s) reconciled between "
        f"{args.langref.relative_to(REPO_ROOT)} and training/tools/)"
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
