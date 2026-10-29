#!/usr/bin/env python3
"""GEO-AMX's builds: GEO-2b's level-2 overlays of the PR818 study's three models -- guard_compact (the global
head), query_guard_compact (the query-aware head) and attention (the attention baseline) -- each compiled against
main's headers as level 2 was (port/build_main.py's graft, unchanged) and linked with a tensor unit of this branch,
as ../../2026-10-22/infer-attn/build_level3.py builds the attention model (level 3). Every tensor entry an overlay
calls has the same prototype in main's header and the branch's (checked here, or nothing is built), so each
overlay's code is level 2's and only the tensor unit differs.

usage: python3 -I build_amx.py LEVEL2_WORK OUT_WORK TENSOR_SRC MAIN_SRC
  writes OUT_WORK/build/<name>.c, <name>.so and <name>-build-amx.json for the three names
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

LEVEL2, OUT, TENSOR, MAIN = (Path(a) for a in sys.argv[1:5])
NAMES = ("guard_compact", "query_guard_compact", "attention")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def protos(header):
    text = Path(header).read_text(encoding="utf-8")
    pat = r"BCIR_DT_API\s+([^;{]*?)\b(bcir_tensor_[a-z0-9_]+)\s*\(([^;]*?)\)\s*;"
    return {
        m.group(2): (" ".join(m.group(1).split()), " ".join(m.group(3).split()))
        for m in re.finditer(pat, text, re.S)
    }


main, branch = protos(MAIN / "bcir_tensor.h"), protos(TENSOR / "bcir_tensor.h")
build = OUT / "build"
build.mkdir(parents=True, exist_ok=True)
flags = ["-std=c11", "-O3", "-fPIC", "-ffp-contract=off", "-fno-math-errno", "-pthread"]
for name in NAMES:
    source = (LEVEL2 / "build" / f"{name}.c").read_text(encoding="utf-8")
    called = sorted(set(re.findall(r"\b(bcir_tensor_[a-z0-9_]+)\s*\(", source)))
    declared = [f for f in called if f in main]
    differ = [f for f in declared if branch.get(f) != main[f]]
    if differ or not declared:
        raise SystemExit(
            f"{name}: the overlay's tensor entries differ between the headers: {differ}"
        )
    (build / f"{name}.c").write_text(source, encoding="utf-8")
    level2 = json.loads((LEVEL2 / "build" / f"{name}-build-main.json").read_text(encoding="utf-8"))
    cc = level2["command"][0]
    defines = [a for a in level2["command"] if a.startswith("-D")]
    obj = build / f"{name}.o"
    cmds = [
        [
            cc,
            *flags,
            "-c",
            "-I",
            str(TENSOR),
            str(TENSOR / "bcir_tensor.c"),
            "-o",
            str(build / "bcir_tensor.o"),
        ],
        [cc, *flags, "-c", "-I", str(MAIN), *defines, str(build / f"{name}.c"), "-o", str(obj)],
        [
            cc,
            "-shared",
            "-pthread",
            str(build / "bcir_tensor.o"),
            str(obj),
            "-lm",
            "-o",
            str(build / f"{name}.so"),
        ],
    ]
    for cmd in cmds:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"{' '.join(cmd)}\n{r.stderr}")
    record = {
        "commands": cmds,
        "entries_checked": declared,
        "overlay_sha256": sha(build / f"{name}.c"),
        "level2_overlay_sha256": sha(LEVEL2 / "build" / f"{name}.c"),
        "tensor_c_sha256": sha(TENSOR / "bcir_tensor.c"),
        "native_sha256": sha(build / f"{name}.so"),
    }
    (build / f"{name}-build-amx.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )
    print(name, "built:", build / f"{name}.so")
