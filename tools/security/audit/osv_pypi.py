#!/usr/bin/env python3
"""Offline OSV cross-check of PyPI package versions against the OSV bulk export.

    python tools/security/audit/osv_pypi.py --osv-dir WORK --report pip-audit.json ... \
        [--pin name==version ...] [--out results.json]

Report-only audit tooling (not a gate); the gate is pip-audit inside
tools/security/audit_dependencies.py. This is the SECOND source: PyPI's advisory API
(pip-audit's default) and OSV's export both descend from the same upstream records, but
they are evaluated by different code, so agreement is evidence and disagreement is a
question. Needs `packaging` (pip-audit's environment has it). The export
(https://osv-vulnerabilities.storage.googleapis.com/PyPI/all.zip) is downloaded into
--osv-dir when absent.

Evaluation follows OSV: an ECOSYSTEM/SEMVER range's [introduced, fixed) or
[introduced, last_affected] intervals, and the explicit `versions` list. A feed value that
is not PEP 440 (PYSEC torch records say `2.6.0-NA`, `2.6.0-cu124`) is read by its dotted
numeric prefix when one exists and is otherwise recorded as unevaluable — never compared as
text, because text order puts 2.13 below 2.6 and produced nine false positives on the first
pass of the 2026-09-04 audit. GIT ranges are never evaluated (commit hashes are not
versions). Exit 1 when any pair is affected.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import urllib.request
import zipfile

try:
    from packaging.utils import canonicalize_name
    from packaging.version import InvalidVersion, Version
except ModuleNotFoundError:
    print("osv_pypi: needs the `packaging` distribution (run inside pip-audit's environment)",
          file=sys.stderr)
    raise SystemExit(2) from None

EXPORT = "https://osv-vulnerabilities.storage.googleapis.com/PyPI/all.zip"
DOWNLOAD_CAP = 512 << 20
_PREFIX = re.compile(r"^\s*v?(\d+(?:\.\d+)+)(?:[-+_].*)?$")
UNEVALUABLE: set[str] = set()


def version(text: str) -> Version | None:
    try:
        return Version(text)
    except InvalidVersion:
        m = _PREFIX.match(text)
        if m:
            try:
                return Version(m.group(1))
            except InvalidVersion:
                pass
        UNEVALUABLE.add(text)
        return None


def cmp(a: str, b: str) -> int | None:
    va, vb = version(a), version(b)
    if va is None or vb is None:
        return None
    return (va > vb) - (va < vb)


def affected(entry: dict, ver: str) -> bool:
    if any(cmp(x, ver) == 0 for x in entry.get("versions") or []):
        return True
    for r in entry.get("ranges", []):
        if r.get("type") not in ("ECOSYSTEM", "SEMVER"):
            continue
        intro = None
        for e in r.get("events", []):
            if "introduced" in e:
                intro = e["introduced"]
            elif "fixed" in e:
                if intro is not None and (cmp(ver, intro) or 0) >= 0:
                    upper = cmp(ver, e["fixed"])
                    if upper is not None and upper < 0:
                        return True
                intro = None
            elif "last_affected" in e:
                if intro is not None and (cmp(ver, intro) or 0) >= 0:
                    upper = cmp(ver, e["last_affected"])
                    if upper is not None and upper <= 0:
                        return True
                intro = None
        if intro is not None and (cmp(ver, intro) or 0) >= 0:
            return True
    return False


def load_export(osv_dir: str) -> tuple[dict, int, int]:
    path = os.path.join(osv_dir, "PyPI.all.zip")
    if not os.path.exists(path):
        os.makedirs(osv_dir, exist_ok=True)
        with urllib.request.urlopen(EXPORT, timeout=120) as response:
            blob = response.read(DOWNLOAD_CAP + 1)
        if len(blob) > DOWNLOAD_CAP:
            raise SystemExit(f"osv_pypi: export exceeds {DOWNLOAD_CAP} bytes; refusing")
        with open(path, "wb") as handle:
            handle.write(blob)
    index: dict[str, list] = collections.defaultdict(list)
    withdrawn = 0
    archive = zipfile.ZipFile(path)
    for name in archive.namelist():
        record = json.loads(archive.read(name))
        if record.get("withdrawn"):
            withdrawn += 1
            continue
        for entry in record.get("affected", []):
            package = entry.get("package", {})
            if package.get("ecosystem") == "PyPI":
                index[canonicalize_name(package["name"])].append((record, entry))
    return index, len(archive.namelist()), withdrawn


def pairs_from(reports: list[str], pins: list[str]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for path in reports:
        with open(path, encoding="utf-8") as handle:
            for dep in json.load(handle).get("dependencies", []):
                if isinstance(dep.get("version"), str):
                    pairs.add((dep["name"], dep["version"]))
    for pin in pins:
        name, _, ver = pin.partition("==")
        pairs.add((name.strip(), ver.strip()))
    return pairs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--osv-dir", required=True)
    ap.add_argument("--report", action="append", default=[], help="pip-audit --format json output")
    ap.add_argument("--pin", action="append", default=[], help="extra name==version to check")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    index, total, withdrawn = load_export(args.osv_dir)
    rows = []
    findings = 0
    for name, ver in sorted(pairs_from(args.report, args.pin)):
        entries = index.get(canonicalize_name(name), [])
        hits = [(rec["id"], rec.get("aliases", []), rec.get("summary", "")[:100])
                for rec, entry in entries if affected(entry, ver)]
        fixed_here = sorted({
            rec["id"] for rec, entry in entries for r in entry.get("ranges", [])
            if r.get("type") in ("ECOSYSTEM", "SEMVER")
            for e in r.get("events", []) if "fixed" in e and cmp(e["fixed"], ver) == 0
        })
        rows.append({"name": name, "version": ver, "advisories_on_record": len({rec["id"] for rec, _ in entries}),
                     "affected_by": [{"id": i, "aliases": a, "summary": s} for i, a, s in hits],
                     "fixed_exactly_at_this_version": fixed_here})
        findings += bool(hits)
    print(f"osv_pypi: export {total} advisories ({withdrawn} withdrawn skipped), "
          f"{len(index)} packages; {len(rows)} pairs checked; {findings} affected")
    for r in rows:
        if r["advisories_on_record"]:
            tail = f"  first clean release for {r['fixed_exactly_at_this_version']}" if r["fixed_exactly_at_this_version"] else ""
            print(f"  {r['name']:22s} {r['version']:12s} on record {r['advisories_on_record']:3d}  "
                  f"affected {[h['id'] for h in r['affected_by']] or 'none'}{tail}")
    if UNEVALUABLE:
        print(f"  unevaluable feed values recorded (never text-compared): {len(UNEVALUABLE)}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"source": EXPORT, "advisories": total, "withdrawn_skipped": withdrawn,
                       "unevaluable_feed_values": sorted(UNEVALUABLE), "rows": rows}, handle, indent=1)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
