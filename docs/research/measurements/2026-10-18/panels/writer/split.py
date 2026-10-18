"""PANELS-WRITER split: two flags on the shipped writer of a strip-writer copy (patch_v4.py's output),
for timing only -- nt bit 8, the row segment's values and no panel; bit 4, the A panels and no second
panel. Neither writes whole panels, so pw_split.c compares no bytes for them.
    python3 split.py V4_TENSOR_C OUT_TENSOR_C"""

import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()
old = """      panel_values(j,i,c0,w,v0,v1);
      if (j->bad) bad|=!span_finite_run(j->variant,v0,w);
      panel_a(j,0,i,c0,w,v0);
      if (two) panel_a(j,1,i,c0,w,v1);
    }
    panel_second(j,0,b*32);
    if (two) panel_second(j,1,b*32);"""
new = """      panel_values(j,i,c0,w,v0,v1);
      if (j->bad) bad|=!span_finite_run(j->variant,v0,w);
      if (j->nt&8) { panel_sink+=v0[w-1]+(two ? v1[w-1] : 0.0f); continue; }  /* split: values only */
      panel_a(j,0,i,c0,w,v0);
      if (two) panel_a(j,1,i,c0,w,v1);
    }
    if (j->nt&4) continue;  /* split: no second panel */
    panel_second(j,0,b*32);
    if (two) panel_second(j,1,b*32);"""
assert src.count(old) == 1
src = src.replace(old, new)
head = "static void panel_task(void *ctx, size_t t) {\n"
assert src.count(head) == 1
src = src.replace(head, "static volatile float panel_sink;\n" + head)
Path(sys.argv[2]).write_text(src)
print("split flags inserted")
