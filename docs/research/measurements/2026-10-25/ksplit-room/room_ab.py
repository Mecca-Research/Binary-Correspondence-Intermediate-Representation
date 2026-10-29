"""KSPLIT-ROOM: a copy of ../../2026-10-22/attn-rate/step_ab.c that lends state A the need without the split room
(bcir_decoder_workspace_unsplit, what the tree before lent: the attention's copies) and states B and B again the
need (bcir_decoder_workspace_need), both at the run's thread count -- so given one library twice it measures the
room alone, every instruction the same; everything else is step_ab's. usage: room_ab.py SRC_step_ab.c DST.c"""

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
src = once(src, "plan_fn plan; wsb_fn wsb; ", "plan_fn plan; wsb_fn wsb,unsplit; ")
src = once(
    src,
    'sym(h,"bcir_decoder_workspace_bytes");',
    'sym(h,"bcir_decoder_workspace_need");'
    ' *(void **)&l.unsplit=sym(h,"bcir_decoder_workspace_unsplit");',
)
src = once(
    src,
    "s->workspace_bytes=L[j].wsb(&s->plan,s->contract);",
    "s->workspace_bytes=j ? L[j].wsb(&s->plan,s->contract,threads) : L[j].unsplit(&s->plan,s->contract,threads);",
)
Path(sys.argv[2]).write_text(src, encoding="utf-8")
