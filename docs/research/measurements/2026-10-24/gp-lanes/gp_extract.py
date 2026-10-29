"""Copies the study fork's scalar geometric-product kernels verbatim (each function by name, from its
signature to its closing brace at column 0) into one file a check program includes.
usage: gp_extract.py FORK.c OUT.inc"""

import re
import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text(encoding="utf-8")
NAMES = (
    "wedge_ill_conditioned",
    "wedge_precise_forward",
    "wedge_precise_backward",
    "gp_pair_forward",
    "gp_pair_backward",
    "gp_query_unit",
    "gp_query_contract_pair",
    "gp_query_contract_vjp",
)
out = [
    "/* extracted verbatim by gp_extract.py */",
    "#define GP_RIDGE 1e-4",
    "#define GP_QUERY_EPS 1e-6",
    "#if GP_PROJECTOR",
    "#define GP_P ((GP_RANK*(GP_RANK+1))/2)",
    "#else",
    "#define GP_P ((GP_RANK*(GP_RANK-1))/2)",
    "#endif",
]
for name in NAMES:
    m = re.search(r"^(static )?(void|int|double) " + name + r"\(", src, re.M)
    if not m:
        raise SystemExit(f"{name} not found")
    end = src.index("\n}\n", m.start()) + 3
    text = src[m.start() : end]
    if not text.startswith("static"):
        text = "static " + text
    out.append(text)
for d in ("#define GP_RIDGE 1e-4", "#define GP_QUERY_EPS 1e-6"):
    if d not in src:
        raise SystemExit(f"{d!r} is not the fork's")
Path(sys.argv[2]).write_text("\n".join(out) + "\n", encoding="utf-8")
print("extracted", len(NAMES), "functions")
