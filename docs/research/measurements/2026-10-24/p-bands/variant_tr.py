"""Variants of the transposed products' walk on a unit copy: band rows (BCIR_ATT_BAND) and a software prefetch
of P's lines PFD floats ahead along each band row (hint T0/T1/T2), issued by the first block of each 16-float
line pair; no value changes (each element's terms in the same order). usage: variant_tr.py IN OUT BAND PFD HINT"""

import sys
from pathlib import Path

src, out, band, pfd, hint = (
    Path(sys.argv[1]).read_text(),
    Path(sys.argv[2]),
    int(sys.argv[3]),
    int(sys.argv[4]),
    sys.argv[5],
)
old = "#define BCIR_ATT_BAND 32\n"
assert src.count(old) == 1
src = src.replace(old, f"#define BCIR_ATT_BAND {band}\n", 1)
if pfd:
    old = (
        "  for (i=diag ? j0+7 : ib;i<ie;i++) {  /* every row of the block takes term i */\n"
        "    const float *xi=x+i*ldx+c0,*mi=m+i*ldm+j0;\n"
    )
    assert src.count(old) == 1
    new = old + f"    if (!(j0&15)) _mm_prefetch((const char *)(mi+{pfd}),_MM_HINT_{hint});\n"
    src = src.replace(old, new, 1)
out.write_text(src)
print(out, "band", band, "pfd", pfd, hint)
