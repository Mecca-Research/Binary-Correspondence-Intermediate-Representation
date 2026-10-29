"""variants of the banded walk on an (instrumented) prototype copy: the walk's band rows and a software prefetch
of each layout band's lines PFD keys ahead of the block. usage: var_btr.py IN OUT BAND PFD"""

import sys
from pathlib import Path

s = Path(sys.argv[1]).read_text()
band = int(sys.argv[3])
pfd = int(sys.argv[4])
o = "  size_t c0,j0,ib,band=t<BCIR_ATT_BAND_FROM ? t : BCIR_ATT_BAND;\n  for (c0=0;c0<d;c0+=48) {\n    size_t nc=d-c0<48 ? d-c0 : 48;\n    __mmask16 m0=att512_lanes(nc),m1=att512_lanes(nc>16 ? nc-16 : 0),m2=att512_lanes(nc>32 ? nc-32 : 0);\n    for (ib=0;ib<t;ib+=band) {\n      size_t ie=t-ib<band ? t : ib+band;\n      for (j0=0;j0<ie;j0+=8) {\n        int zero=first && j0>=ib;\n"
assert s.count(o) == 1
n = o.replace(
    "band=t<BCIR_ATT_BAND_FROM ? t : BCIR_ATT_BAND;", f"band=t<BCIR_ATT_BAND_FROM ? t : {band};"
)
if pfd:
    n += (
        "        { size_t lb,l; for (lb=ib>>4;lb<(ie+15)>>4;lb++) for (l=0;l<8;l++)\n"
        f"            _mm_prefetch((const char *)(pb+lb*16*t+(j0+{pfd})*16+l*16),_MM_HINT_T0); }}\n"
    )
s = s.replace(o, n)
Path(sys.argv[2]).write_text(s)
