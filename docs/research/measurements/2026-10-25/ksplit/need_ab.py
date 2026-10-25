"""KSPLIT: a copy of ../../2026-10-22/attn-rate/step_ab.c that lends each state the exact workspace its plan
needs at THREADS -- bcir_decoder_workspace_need, what the Python trainer lends (native.memory_plan) -- where
step_ab lends the workspace for any pool (bcir_decoder_workspace_bytes, the need at no thread count named);
everything else is step_ab's. usage: need_ab.py SRC_step_ab.c DST.c"""

import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text(encoding="utf-8")


def once(text, old, new):
    if text.count(old) != 1:
        raise SystemExit(f"anchor found {text.count(old)} times: {old[:70]!r}")
    return text.replace(old, new)


src = once(
    src,
    "typedef size_t (*wsb_fn)(const bcir_decoder_plan *, uint32_t);",
    "typedef size_t (*wsb_fn)(const bcir_decoder_plan *, uint32_t, unsigned);",
)
src = once(src, 'sym(h,"bcir_decoder_workspace_bytes")', 'sym(h,"bcir_decoder_workspace_need")')
src = once(
    src,
    "s->workspace_bytes=L[j].wsb(&s->plan,s->contract);",
    "s->workspace_bytes=L[j].wsb(&s->plan,s->contract,threads);",
)
Path(sys.argv[2]).write_text(src, encoding="utf-8")
