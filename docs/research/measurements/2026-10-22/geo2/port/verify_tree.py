"""GEO-0: every bundle's MANIFEST_SHA256.json against the bytes origin/main holds, read from git
(no checkout): per file ok / differs / missing, a differing file's cause classified when its bytes
are the manifest's plus one trailing newline (GitHub's web rename adds one), and the files the
tree holds that the manifest does not list.
usage: python3 -I verify_tree.py REPO REV OUT.json"""

import hashlib, json, subprocess, sys

repo, rev, out = sys.argv[1], sys.argv[2], sys.argv[3]
runs = ("gp-global", "wedge-next", "wedge-integrated", "wedge-query-aware", "wedge-pr818")
listing = subprocess.run(
    ["git", "-C", repo, "ls-tree", "-r", "-l", rev, "docs/research/"],
    capture_output=True,
    text=True,
    check=True,
).stdout.splitlines()
tree = {}
for line in listing:
    meta, path = line.split("\t", 1)
    _, kind, oid, size = meta.split()
    tree[path] = (oid, int(size))


def blob(oid):
    return subprocess.run(
        ["git", "-C", repo, "cat-file", "blob", oid], capture_output=True, check=True
    ).stdout


MANIFESTS = ("MANIFEST_SHA256.json", "SOURCE_MANIFEST.json", "manifest.json")
report = {"rev": rev, "bundles": {}}
for run in runs:
    root = f"docs/research/{run}/"
    name = next(n for n in MANIFESTS if root + n in tree)
    man = json.loads(blob(tree[root + name][0]))["files"]
    # three spellings across the bundles: {path: sha256}, [{path, bytes, sha256}], {path: {bytes, sha256}}
    if isinstance(man, list):
        files = {e["path"]: e for e in man}
    else:
        files = {k: (v if isinstance(v, dict) else {"sha256": v}) for k, v in man.items()}
    res = {
        "manifest": name,
        "manifest_entries": len(files),
        "ok": 0,
        "newline_only": [],
        "differs": [],
        "missing": [],
        "unlisted": [],
    }
    for rel, e in files.items():
        got = tree.get(root + rel)
        if got is None:
            res["missing"].append({"path": rel, "bytes": e.get("bytes")})
            continue
        b = blob(got[0])
        if hashlib.sha256(b).hexdigest() == e["sha256"] and len(b) == e.get("bytes", len(b)):
            res["ok"] += 1
        elif b.endswith(b"\n") and hashlib.sha256(b[:-1]).hexdigest() == e["sha256"]:
            res["newline_only"].append(rel)
        else:
            res["differs"].append(rel)
    listed = set(files) | {name}
    res["unlisted"] = sorted(
        p[len(root) :] for p in tree if p.startswith(root) and p[len(root) :] not in listed
    )
    res["tree_files"] = sum(1 for p in tree if p.startswith(root))
    res["tree_bytes"] = sum(s for p, (_, s) in tree.items() if p.startswith(root))
    report["bundles"][run] = res
    print(
        f"{run}: {res['ok']}/{res['manifest_entries']} byte-identical, newline-only "
        f"{len(res['newline_only'])}, differ {len(res['differs'])}, missing {len(res['missing'])} "
        f"{[m['path'] for m in res['missing']][:6]}, unlisted {len(res['unlisted'])} {res['unlisted'][:4]}",
        flush=True,
    )
with open(out, "w", encoding="utf-8", newline="\n") as f:
    json.dump(report, f, indent=1, sort_keys=True)
    f.write("\n")
