#!/usr/bin/env python3
"""How current every GitHub Action and pre-commit hook pin is, against upstream's tags.

    python tools/security/audit/actions_currency.py [--root .] [--out report.json]

Report-only audit tooling (not a gate). Reads every `uses: owner/repo@<sha>` in
.github/workflows/*.yml and every `repo:`/`rev:` pair in .pre-commit-config.yaml, asks
upstream for its tags (`git ls-remote --tags`, bounded), and answers: is the pinned commit a
release, which release, what is the latest release, the latest within the pinned major, and
how many majors behind the pin is. Exit 1 when any upstream could not be queried, so a
report with a hole in it never reads as a clean one.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from proc_bounds import run_bounded  # noqa: E402

_USES = re.compile(r"^\s*(?:-\s*)?uses:\s*([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(?:/[^@\s]+)?@([0-9a-f]{40})"
                   r"(?:\s*#\s*(\S+))?", re.M)
_HOOK_REPO = re.compile(r"^\s*-\s*repo:\s*https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)\s*$"
                        r"\n\s*rev:\s*(\S+)", re.M)
_RELEASE = re.compile(r"^v?\d+\.\d+\.\d+$")
LS_REMOTE_TIMEOUT = 90.0
LS_REMOTE_CAP = 4 << 20


def semver_key(tag: str) -> tuple | None:
    m = re.match(r"^v?(\d+)(?:\.(\d+))?(?:\.(\d+))?(.*)$", tag)
    if not m:
        return None
    pre = m.group(4) or ""
    return (int(m.group(1)), int(m.group(2) or 0), int(m.group(3) or 0), 0 if not pre else -1, pre)


def pins(root: str) -> list[dict]:
    found: list[dict] = []
    for path in sorted(glob.glob(os.path.join(root, ".github", "workflows", "*.yml"))):
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        for repo, sha, comment in _USES.findall(text):
            found.append({"repo": repo, "pin": sha, "comment": comment or None,
                          "where": os.path.relpath(path, root)})
    hooks = os.path.join(root, ".pre-commit-config.yaml")
    if os.path.exists(hooks):
        with open(hooks, encoding="utf-8") as handle:
            for repo, rev in _HOOK_REPO.findall(handle.read()):
                found.append({"repo": repo, "pin": rev, "comment": "rev", "where": ".pre-commit-config.yaml"})
    return found


def upstream_tags(repo: str) -> dict[str, str] | None:
    """tag -> commit sha (annotated tags peeled), or None when upstream could not be read."""
    outcome = run_bounded(["git", "ls-remote", "--tags", f"https://github.com/{repo}"],
                          timeout=LS_REMOTE_TIMEOUT, cap=LS_REMOTE_CAP)
    if not outcome["launched"] or outcome["timed_out"] or outcome["overflow"] or outcome["returncode"] != 0:
        return None
    raw: dict[str, str] = {}
    peeled: dict[str, str] = {}
    for line in outcome["stdout"].decode("utf-8", "replace").splitlines():
        parts = line.split()
        if len(parts) != 2 or not parts[1].startswith("refs/tags/"):
            continue
        sha, tag = parts[0], parts[1][len("refs/tags/"):]
        if tag.endswith("^{}"):
            peeled[tag[:-3]] = sha
        else:
            raw[tag] = sha
    for tag, sha in raw.items():
        peeled.setdefault(tag, sha)
    return peeled


def assess(repo: str, pin: str, tags: dict[str, str]) -> dict:
    releases = {t: s for t, s in tags.items() if _RELEASE.match(t)}
    latest = max(releases, key=semver_key) if releases else None
    if re.fullmatch(r"[0-9a-f]{40}", pin):
        pin_tags = sorted((t for t, s in tags.items() if s == pin), key=lambda t: (len(t), t))
        pin_sha = pin
    else:
        pin_tags = [pin] if pin in tags else []
        pin_sha = tags.get(pin)
    pin_release = next((t for t in pin_tags if _RELEASE.match(t)), pin_tags[0] if pin_tags else None)
    same_major = None
    majors_behind = None
    if pin_release and latest:
        major = semver_key(pin_release)[0]
        in_major = [t for t in releases if semver_key(t)[0] == major]
        same_major = max(in_major, key=semver_key) if in_major else None
        majors_behind = semver_key(latest)[0] - major
    return {
        "repo": repo, "pin": pin, "pin_sha": pin_sha, "is_release_commit": bool(pin_tags),
        "pin_release": pin_release, "pin_tags": pin_tags,
        "latest_release": latest, "latest_release_sha": releases.get(latest) if latest else None,
        "latest_in_pinned_major": same_major,
        "latest_in_pinned_major_sha": releases.get(same_major) if same_major else None,
        "majors_behind": majors_behind,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.getcwd())
    ap.add_argument("--out", help="write the JSON report here")
    args = ap.parse_args(argv)
    root = os.path.abspath(args.root)
    found = pins(root)
    by_repo: dict[tuple[str, str], list[dict]] = {}
    for item in found:
        by_repo.setdefault((item["repo"], item["pin"]), []).append(item)
    tag_cache: dict[str, dict[str, str] | None] = {}
    rows: list[dict] = []
    unavailable = 0
    for (repo, pin), items in sorted(by_repo.items()):
        if repo not in tag_cache:
            tag_cache[repo] = upstream_tags(repo)
        tags = tag_cache[repo]
        if tags is None:
            unavailable += 1
            rows.append({"repo": repo, "pin": pin, "state": "UNAVAILABLE", "uses": len(items),
                         "comment": items[0]["comment"], "where": sorted({i["where"] for i in items})})
            continue
        row = assess(repo, pin, tags)
        row.update({"state": "OK", "uses": len(items), "comment": items[0]["comment"],
                    "where": sorted({i["where"] for i in items})})
        rows.append(row)
    for r in rows:
        if r["state"] != "OK":
            print(f"  {r['repo']:36s} {r['pin'][:12]:12s} UNAVAILABLE (upstream tags not readable)")
            continue
        print(f"  {r['repo']:36s} {r['pin'][:12]:12s} {str(r['comment']):8s} uses={r['uses']:2d} "
              f"release={str(r['pin_release']):9s} latest={str(r['latest_release']):9s} "
              f"same-major={str(r['latest_in_pinned_major']):9s} majors-behind={r['majors_behind']} "
              f"release-commit={r['is_release_commit']}")
    print(f"actions_currency: {len(rows)} pins, {unavailable} unavailable")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"pins": rows}, handle, indent=1)
    return 1 if unavailable else 0


if __name__ == "__main__":
    raise SystemExit(main())
