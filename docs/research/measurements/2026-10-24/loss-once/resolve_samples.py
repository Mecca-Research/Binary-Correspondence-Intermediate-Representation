"""Resolves sampler.c's raw file: each instruction pointer to its mapping's file and that file's function (nm's
defined symbols, sorted), then the counts by function and by file. usage: resolve_samples.py RAW [TOP]"""

import bisect
import collections
import subprocess
import sys

raw, top = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40
maps, ips = [], []
for ln in open(raw, encoding="utf-8", errors="replace"):
    if ln.startswith("map "):
        parts = ln[4:].split()
        if len(parts) >= 6 and "x" in parts[1]:
            lo, hi = (int(x, 16) for x in parts[0].split("-"))
            maps.append((lo, hi, int(parts[2], 16), parts[5]))
    elif ln.startswith("ip "):
        ips.append(int(ln[3:], 16))
maps.sort()
syms = {}


def table(path):
    if path not in syms:
        rows = []
        try:
            out = subprocess.run(
                ["nm", "--defined-only", "-n", path], capture_output=True, text=True, check=False
            ).stdout
        except OSError:
            out = ""
        for x in out.splitlines():
            p = x.split()
            if len(p) >= 3 and p[1] in "tTwWiI":
                rows.append((int(p[0], 16), p[2]))
        if not rows:  # a stripped file: its dynamic symbols
            try:
                out = subprocess.run(
                    ["nm", "-D", "--defined-only", "-n", path],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout
            except OSError:
                out = ""
            for x in out.splitlines():
                p = x.split()
                if len(p) >= 3 and p[1] in "tTwWiI":
                    rows.append((int(p[0], 16), p[2]))
            rows.sort()
        syms[path] = ([r[0] for r in rows], [r[1] for r in rows])
    return syms[path]


starts = [m[0] for m in maps]
by_fn, by_file = collections.Counter(), collections.Counter()
# the load base of a file: its first mapping's start less that mapping's offset
base = {}
for lo, _, off, path in maps:
    base.setdefault(path, lo - off)
for ip in ips:
    k = bisect.bisect_right(starts, ip) - 1
    if k < 0 or ip >= maps[k][1]:
        by_fn["?"] += 1
        by_file["?"] += 1
        continue
    path = maps[k][3]
    name = path.rsplit("/", 1)[-1]
    vaddr = ip - base[path] if path.endswith(".so") or ".so." in path else ip
    addrs, names = table(path)
    j = bisect.bisect_right(addrs, vaddr) - 1
    fn = names[j] if j >= 0 else "?"
    by_fn[f"{name}:{fn}"] += 1
    by_file[name] += 1
n = len(ips)
print(f"{n} samples")
for k, v in by_file.most_common(12):
    print(f"  {100 * v / n:5.1f}%  {k}")
print("functions:")
for k, v in by_fn.most_common(top):
    print(f"  {100 * v / n:5.1f}%  {k}")
