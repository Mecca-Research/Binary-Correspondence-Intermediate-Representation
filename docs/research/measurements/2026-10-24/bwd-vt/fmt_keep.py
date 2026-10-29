"""ab_keep's JSON lines read: per row the backward's, the forward's and the whole call's proto/base ratio beside A/A."""

import json, math, statistics, sys

T95 = (
    0,
    0,
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
)


def ex(r):
    logs = [math.log(x) for x in r]
    n = len(logs)
    t = T95[n] if n <= 24 else 1.96 + 2.4 / (n - 1)
    m = statistics.fmean(logs)
    h = t * statistics.stdev(logs) / math.sqrt(n)
    return math.exp(m), math.exp(m - h), math.exp(m + h)


for line in open(sys.argv[1]):
    line = line.strip()
    if not line.startswith("{"):
        continue
    r = json.loads(line)
    a = r["arms"]
    out = []
    for part in ("backward", "forward", "total"):
        if part == "total":
            b = [x + y for x, y in zip(a["base"]["forward"]["ms"], a["base"]["backward"]["ms"])]
            p = [x + y for x, y in zip(a["proto"]["forward"]["ms"], a["proto"]["backward"]["ms"])]
            q = [x + y for x, y in zip(a["A/A"]["forward"]["ms"], a["A/A"]["backward"]["ms"])]
        else:
            b, p, q = a["base"][part]["ms"], a["proto"][part]["ms"], a["A/A"][part]["ms"]
        e = ex([y / x for x, y in zip(b, p)])
        c = ex([y / x for x, y in zip(b, q)])
        out.append(
            f"{part} {statistics.median(b):7.2f} ms x{e[0]:.3f} ({e[1]:.3f}-{e[2]:.3f}) A/A {c[0]:.3f} ({c[1]:.3f}-{c[2]:.3f})"
        )
    print(
        r.get("compiler", "?"),
        r["shape"],
        "th",
        r["threads"],
        "ct",
        r["contract"],
        "|",
        " | ".join(out),
    )
