#!/usr/bin/env python3
"""Ubuntu advisory match for the apt packages the workflows install, from the OSV export.

    python tools/security/audit/osv_ubuntu.py --osv-dir WORK --release 24.04 \
        --binary clang --binary nodejs ... [--package source=version ...] [--out results.json]

Report-only audit tooling (not a gate). For each binary package, `apt-cache` on this host
gives the source package and the candidate version the archive resolves today (run it on an
up-to-date host of the same release as the runner image); `--package` states a source
package and version directly. Every OSV record for that source in `Ubuntu:<release>:LTS`
is classified with dpkg's own version comparator: fixed at or below the candidate,
AFFECTING the candidate (a fix exists above it), or OPEN (no fix in the release, with
Ubuntu's priority). The export
(https://osv-vulnerabilities.storage.googleapis.com/Ubuntu:<release>:LTS/all.zip) is
downloaded into --osv-dir when absent. Exit 1 when anything is AFFECTING.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import urllib.request
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from proc_bounds import run_bounded  # noqa: E402

DOWNLOAD_CAP = 1 << 30
APT_TIMEOUT = 60.0
APT_CAP = 1 << 20


def apt_text(args: list[str]) -> str | None:
    outcome = run_bounded(["apt-cache", *args], timeout=APT_TIMEOUT, cap=APT_CAP)
    if not outcome["launched"] or outcome["timed_out"] or outcome["returncode"] != 0:
        return None
    return outcome["stdout"].decode("utf-8", "replace")


def resolve_binary(binary: str) -> tuple[str, str] | None:
    """(source package, candidate version) for a binary package name, from apt-cache."""
    policy = apt_text(["policy", binary])
    show = apt_text(["show", binary])
    if policy is None or show is None:
        return None
    candidate = None
    for line in policy.splitlines():
        if line.strip().startswith("Candidate:"):
            candidate = line.split(":", 1)[1].strip()
            break
    if not candidate or candidate == "(none)":
        return None
    source = binary
    for line in show.splitlines():
        if line.startswith("Source:"):
            source = line.split(":", 1)[1].strip().split(" ")[0]
            break
    return source, candidate


def dpkg_lt(a: str, b: str) -> bool | None:
    outcome = run_bounded(
        ["dpkg", "--compare-versions", a, "lt", b], timeout=APT_TIMEOUT, cap=APT_CAP
    )
    if not outcome["launched"] or outcome["timed_out"] or outcome["returncode"] not in (0, 1):
        return None
    return outcome["returncode"] == 0


def load_export(osv_dir: str, release: str) -> tuple[zipfile.ZipFile, str]:
    ecosystem = f"Ubuntu:{release}:LTS"
    path = os.path.join(osv_dir, f"Ubuntu{release.replace('.', '')}.all.zip")
    if not os.path.exists(path):
        os.makedirs(osv_dir, exist_ok=True)
        url = f"https://osv-vulnerabilities.storage.googleapis.com/{ecosystem}/all.zip"
        with urllib.request.urlopen(url, timeout=600) as response:
            blob = response.read(DOWNLOAD_CAP + 1)
        if len(blob) > DOWNLOAD_CAP:
            raise SystemExit(f"osv_ubuntu: export exceeds {DOWNLOAD_CAP} bytes; refusing")
        with open(path, "wb") as handle:
            handle.write(blob)
    return zipfile.ZipFile(path), ecosystem


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--osv-dir", required=True)
    ap.add_argument("--release", default="24.04")
    ap.add_argument(
        "--binary", action="append", default=[], help="binary package, resolved through apt-cache"
    )
    ap.add_argument("--package", action="append", default=[], help="source=version stated directly")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    wanted: dict[str, dict] = {}
    unresolved: list[str] = []
    for binary in args.binary:
        resolved = resolve_binary(binary)
        if resolved is None:
            unresolved.append(binary)
            continue
        source, candidate = resolved
        wanted.setdefault(source, {"version": candidate, "binaries": []})["binaries"].append(binary)
    for item in args.package:
        source, _, ver = item.partition("=")
        wanted.setdefault(source.strip(), {"version": ver.strip(), "binaries": []})
    archive, ecosystem = load_export(args.osv_dir, args.release)
    index: dict[str, list] = collections.defaultdict(list)
    for name in archive.namelist():
        record = json.loads(archive.read(name))
        if record.get("withdrawn"):
            continue
        for entry in record.get("affected", []):
            package = entry.get("package", {})
            if package.get("ecosystem") == ecosystem and package.get("name") in wanted:
                index[package["name"]].append((record, entry))
    out: dict[str, dict] = {}
    affecting_total = 0
    for source, info in sorted(wanted.items()):
        ver = info["version"]
        buckets: dict[str, list] = {"fixed": [], "affecting": [], "open": [], "unevaluable": []}
        for record, entry in index.get(source, []):
            status = "open"
            for r in entry.get("ranges", []):
                fixed = next((e["fixed"] for e in r.get("events", []) if "fixed" in e), None)
                if fixed is None:
                    status = "open"
                else:
                    below = dpkg_lt(ver, fixed)
                    status = "unevaluable" if below is None else ("affecting" if below else "fixed")
            buckets[status].append(
                {
                    "id": record["id"],
                    "aliases": [a for a in record.get("aliases", []) if a.startswith("CVE-")],
                    "priority": (entry.get("ecosystem_specific") or {}).get("ubuntu_priority"),
                    "summary": record.get("summary", "")[:100],
                }
            )
        affecting_total += len(buckets["affecting"])
        out[source] = {
            "version": ver,
            "binaries": info["binaries"],
            "advisories_on_record": len(index.get(source, [])),
            "fixed_at_or_below": len(buckets["fixed"]),
            "affecting": buckets["affecting"],
            "open_no_fix": buckets["open"],
            "unevaluable": buckets["unevaluable"],
        }
        by_priority = collections.Counter(x["priority"] for x in buckets["open"])
        print(
            f"  {source:20s} {ver:28s} on record {len(index.get(source, [])):3d}  fixed<= {len(buckets['fixed']):3d}  "
            f"AFFECTING {len(buckets['affecting']):2d}  open {len(buckets['open']):2d} {dict(by_priority) if buckets['open'] else ''}"
        )
    print(
        f"osv_ubuntu: {ecosystem}; {len(out)} source packages; {affecting_total} affecting; "
        f"unresolved binaries: {unresolved or 'none'}"
    )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(
                {"ecosystem": ecosystem, "unresolved_binaries": unresolved, "packages": out},
                handle,
                indent=1,
            )
            handle.write("\n")
    return 1 if affecting_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
