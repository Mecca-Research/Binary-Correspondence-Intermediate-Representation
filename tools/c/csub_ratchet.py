#!/usr/bin/env python3
"""The C subset's ratchet (CSUB-1, docs/research/BCIR_C_SUBSET_AUDIT_2026-11-10.md §4): `bcir-cc`, the verified C
compiler's C rail, over every production translation unit of the runtime, held to a record of what it makes of each.

The units are the ones `runtime/c/MEMORY_CLASSIFICATION.txt` lists -- the inventory every production unit is already
held to -- so a unit added to the runtime is added here too, or the gate fails. The record, `runtime/c/csub_ratchet.json`,
says of each unit either `clean` (it compiles with R1-R18 clean) or `blocked`, with the first refusal `bcir-cc` gives
for it, the construct that refusal is about and the rung of the CSUB ladder that lifts it (`CSUB-2` .. `CSUB-6`, or
`outside` for what the subset leaves to the fallback contract by design). The gate holds the record exact:

  * a unit recorded clean that no longer compiles clean is a shrink, and fails;
  * a unit recorded blocked that now compiles clean fails until the record promotes it -- the set grows by being
    recorded, so the distance to the next rung is always the recorded one;
  * a blocked unit whose first refusal changed fails until the record names the new one;
  * a unit the inventory lists and the record does not, or the record lists and the inventory does not, fails;
  * a record that is malformed -- an unknown verdict or rung, a blocked unit with no construct, a duplicated key --
    is INVALID.

Every exit is a verdict (L1): 0 PASS, 1 FAIL, 2 INVALID (the record, the inventory or the compiler unusable). A unit
whose compile crashes, times out or prints nothing to say why it refused is a FAIL naming the unit, never a skip.
`--update` rewrites the record from the compiler's verdicts, keeping each still-accurate construct and rung, and
marks a newly blocked unit's rung `?` for a person to fill in -- which `--check` refuses.

    python3 tools/c/csub_ratchet.py --cc build/.../bcir-cc            # check (the gate)
    python3 tools/c/csub_ratchet.py --cc build/.../bcir-cc --update   # rewrite the record
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNTIME = os.path.join(ROOT, "runtime", "c")
INVENTORY = os.path.join(RUNTIME, "MEMORY_CLASSIFICATION.txt")
RECORD = os.path.join(RUNTIME, "csub_ratchet.json")
SCHEMA = "bcir.csub-ratchet/1"
RUNGS = ("CSUB-2", "CSUB-3", "CSUB-4", "CSUB-5", "CSUB-6", "outside")
VERDICTS = ("clean", "blocked")
#: The longest one compile may take: the largest unit compiles in well under a second.
TIMEOUT_S = 120
#: The longest refusal kept (`bcir-cc`'s diagnostics are one short line; a longer one is cut, never dropped).
MAX_REASON = 300


class Invalid(Exception):
    """The record, the inventory or the compiler cannot be used: the gate's verdict is INVALID (exit 2)."""


def _no_duplicates(pairs):
    seen = {}
    for k, v in pairs:
        if k in seen:
            raise Invalid(f"the record repeats the key {k!r}")
        seen[k] = v
    return seen


def read_inventory(path: str = INVENTORY) -> dict:
    """unit -> its memory class, from the inventory every production unit is held to."""
    units = {}
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except OSError as e:
        raise Invalid(f"cannot read the inventory {path}: {e}") from None
    for n, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2 or not parts[1].endswith(".c"):
            raise Invalid(f"{os.path.basename(path)}:{n}: not `<class> <unit>.c`: {line!r}")
        if parts[1] in units:
            raise Invalid(f"{os.path.basename(path)}:{n}: {parts[1]} is listed twice")
        units[parts[1]] = parts[0]
    if not units:
        raise Invalid(f"the inventory {path} lists no unit")
    return units


def read_record(path: str = RECORD) -> dict:
    """The record's units, each checked for shape: {unit: entry}."""
    try:
        text = open(path, encoding="utf-8").read()
    except OSError as e:
        raise Invalid(f"cannot read the record {path}: {e}") from None
    try:
        doc = json.loads(text, object_pairs_hook=_no_duplicates)
    except json.JSONDecodeError as e:
        raise Invalid(f"the record is not JSON: {e}") from None
    if (
        not isinstance(doc, dict)
        or doc.get("schema") != SCHEMA
        or not isinstance(doc.get("units"), dict)
    ):
        raise Invalid(f"the record is not a {SCHEMA} document with a `units` object")
    units = doc["units"]
    for unit, entry in units.items():
        if not isinstance(entry, dict) or entry.get("verdict") not in VERDICTS:
            raise Invalid(f"{unit}: the verdict must be one of {VERDICTS}")
        if entry["verdict"] == "clean":
            if set(entry) != {"verdict"}:
                raise Invalid(f"{unit}: a clean unit records its verdict only")
            continue
        if set(entry) != {"verdict", "blocker", "construct", "rung"}:
            raise Invalid(
                f"{unit}: a blocked unit records its verdict, blocker, construct and rung"
            )
        for key in ("blocker", "construct"):
            if not isinstance(entry[key], str) or not entry[key].strip():
                raise Invalid(f"{unit}: the {key} must be a nonempty string")
        if entry["rung"] not in RUNGS:
            raise Invalid(f"{unit}: the rung {entry['rung']!r} is none of {RUNGS}")
    return units


def compile_unit(cc: str, unit: str, root: str = RUNTIME) -> tuple:
    """`bcir-cc`'s verdict on one unit: ("clean", "") or ("blocked", its first refusal) -- or ("broken", why) where
    the compiler crashed, timed out or refused without saying why."""
    path = os.path.join(root, unit)
    try:
        run = subprocess.run(
            [cc, "-I", root, "-o", os.devnull, path],
            capture_output=True,
            timeout=TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "broken", f"no verdict in {TIMEOUT_S} s"
    except OSError as e:
        raise Invalid(f"cannot run the compiler {cc}: {e}") from None
    if run.returncode == 0:
        return "clean", ""
    text = run.stderr.decode("utf-8", "replace") or run.stdout.decode("utf-8", "replace")
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    prefix = path + ": "
    reason = first[len(prefix) :] if first.startswith(prefix) else first
    if run.returncode < 0 or run.returncode not in (1, 2) or not reason:
        return "broken", f"exit {run.returncode} with no refusal: {first[:MAX_REASON]!r}"
    return "blocked", reason[:MAX_REASON]


def check(inventory: dict, record: dict, verdicts: dict) -> list:
    """Every way the compiler's verdicts and the record disagree: one line each, empty when they agree."""
    fails = []
    for unit in sorted(set(record) - set(inventory)):
        fails.append(f"{unit}: recorded, but the inventory lists no such production unit")
    for unit in sorted(inventory):
        got, why = verdicts[unit]
        entry = record.get(unit)
        if got == "broken":
            fails.append(f"{unit}: bcir-cc gave no verdict ({why})")
        elif entry is None:
            fails.append(
                f"{unit}: not recorded -- record it {'clean' if got == 'clean' else 'blocked by: ' + why}"
            )
        elif entry["verdict"] == "clean" and got != "clean":
            fails.append(f"{unit}: recorded clean, now refused (a shrink): {why}")
        elif entry["verdict"] == "blocked" and got == "clean":
            fails.append(f"{unit}: now compiles R1-R18 clean -- promote it to clean in the record")
        elif entry["verdict"] == "blocked" and why != entry["blocker"]:
            fails.append(
                f"{unit}: the first refusal changed: recorded {entry['blocker']!r}, now {why!r}"
            )
    return fails


def update(inventory: dict, record: dict, verdicts: dict) -> dict:
    """The record the verdicts say: each still-accurate construct and rung kept, a new blocker's rung `?`."""
    out = {}
    for unit in sorted(inventory):
        got, why = verdicts[unit]
        if got == "clean":
            out[unit] = {"verdict": "clean"}
            continue
        old = record.get(unit, {})
        same = old.get("verdict") == "blocked" and old.get("blocker") == why
        out[unit] = {
            "verdict": "blocked",
            "blocker": why,
            "construct": old.get("construct", "?") if same else "?",
            "rung": old.get("rung", "?") if same else "?",
        }
    return out


def write_record(path: str, units: dict) -> None:
    doc = {
        "schema": SCHEMA,
        "about": (
            "What bcir-cc (the C rail of the verified C compiler, default target x86_64-linux, -I runtime/c) makes of "
            "each production unit runtime/c/MEMORY_CLASSIFICATION.txt lists: clean (R1-R18 clean), or blocked, with "
            "its first refusal, the construct it is about and the CSUB rung that lifts it. Held exact by "
            "tools/c/csub_ratchet.py (docs/research/BCIR_C_SUBSET_AUDIT_2026-11-10.md section 4)."
        ),
        "units": units,
    }
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2, sort_keys=False)
        fh.write("\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--cc", required=True, help="the bcir-cc binary")
    ap.add_argument("--record", default=RECORD)
    ap.add_argument("--inventory", default=INVENTORY)
    ap.add_argument("--root", default=RUNTIME, help="the directory the units are in")
    ap.add_argument("--update", action="store_true", help="rewrite the record from the verdicts")
    args = ap.parse_args(argv)
    try:
        if not (os.path.isfile(args.cc) and os.access(args.cc, os.X_OK)):
            raise Invalid(f"{args.cc} is not an executable bcir-cc")
        inventory = read_inventory(args.inventory)
        record = {} if args.update and not os.path.exists(args.record) else read_record(args.record)
        verdicts = {unit: compile_unit(args.cc, unit, args.root) for unit in sorted(inventory)}
    except Invalid as e:
        print(f"csub-ratchet: INVALID: {e}")
        return 2
    clean = sum(1 for v, _ in verdicts.values() if v == "clean")
    if args.update:
        units = update(inventory, record, verdicts)
        write_record(args.record, units)
        open_rungs = [u for u, e in units.items() if e.get("rung") == "?"]
        print(f"csub-ratchet: recorded {clean} clean of {len(units)} units")
        for unit in open_rungs:
            print(f"  TODO {unit}: name the construct and the rung of {units[unit]['blocker']!r}")
        return 1 if open_rungs else 0
    fails = check(inventory, record, verdicts)
    for line in fails:
        print(f"  FAIL {line}")
    blocked = len(inventory) - clean
    if fails:
        print(
            f"csub-ratchet: FAIL ({len(fails)} disagreements; {clean} clean, {blocked} blocked of {len(inventory)})"
        )
        return 1
    print(
        f"csub-ratchet: PASS ({clean} clean, {blocked} blocked of {len(inventory)} units, each as recorded)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
