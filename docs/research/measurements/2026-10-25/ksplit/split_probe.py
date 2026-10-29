"""KSPLIT: a copy of the tensor unit whose split rule (mm_ksplit) logs every call it weighs -- a blocked call with A
transposed and B not, at least BCIR_MM_KSPLIT_BLOCKS K blocks deep, in a job lent a room -- to stderr as one line,
"KSPLIT m n k blocks partials_bytes room_left_bytes split|kept", before it decides; every value the unit computes is
its own (the line is written, nothing else changes). Run under step_ab, each such call of the step shows whether
its partials fit what the job's room had left. usage: split_probe.py SRC_bcir_tensor.c DST_bcir_tensor.c"""

import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text(encoding="utf-8")


def once(text, old, new):
    if text.count(old) != 1:
        raise SystemExit(f"anchor found {text.count(old)} times: {old[:70]!r}")
    return text.replace(old, new)


src = once(
    src,
    "  raw=blocks*mn*sizeof(float);\n  if (raw>*left) return tasks;\n",
    "  raw=blocks*mn*sizeof(float);\n"
    '  fprintf(stderr,"KSPLIT %zu %zu %zu %zu %zu %zu %s\\n",j->x.m,j->x.n,j->x.k,blocks,raw,*left,'
    'raw>*left ? "kept" : "split");\n'
    "  if (raw>*left) return tasks;\n",
)
Path(sys.argv[2]).write_text("#include <stdio.h>\n" + src, encoding="utf-8")
