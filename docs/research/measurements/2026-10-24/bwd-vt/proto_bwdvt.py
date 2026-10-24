"""BWD-VT-0's prototype: in the backward's contiguous task (ATT-CONTIG's region) on the AVX-512 instance, V's slot holds
the KV head's values transposed -- V is read only by the probability-adjoint tiles -- so every tile reads the whole
panel and none transposes its values again for each row block (DP-PANEL's per-tile panel). usage: SRC OUT"""

import sys
from pathlib import Path

t = Path(sys.argv[1]).read_text()


def once(old, new):
    global t
    assert t.count(old) == 1, (old[:70], t.count(old))
    t = t.replace(old, new)


once(
    """  att_rows_in(kc,a->k+(r*t*nk+kh)*d,t,d,nk*d);
  att_rows_in(vc,a->v+(r*t*nk+kh)*d,t,d,nk*d);
  for (i=v1;i<t;i++) { memset(dkc+i*d,0,d*sizeof(float)); memset(dvc+i*d,0,d*sizeof(float)); }""",
    """  att_rows_in(kc,a->k+(r*t*nk+kh)*d,t,d,nk*d);
#ifdef BCIR_MM_DISPATCH
  if (a->variant==4) att512_panel(t,d,a->v+(r*t*nk+kh)*d,nk*d,vc);  /* BWD-VT: the dP tiles' panel, whole */
  else
#endif
  att_rows_in(vc,a->v+(r*t*nk+kh)*d,t,d,nk*d);
  for (i=v1;i<t;i++) { memset(dkc+i*d,0,d*sizeof(float)); memset(dvc+i*d,0,d*sizeof(float)); }""",
)
once(
    """  {
#ifdef BCIR_MM_DISPATCH
    _Alignas(64) float panel[BCIR_ATT_PANEL];
    if (a->variant==4 && d*t<=BCIR_ATT_WHOLE) { att512_panel(t,d,vc,d,panel); vt=panel; }
    else if (a->variant==4 && d*BCIR_ATT_COLS<=BCIR_ATT_PANEL) tp=panel;
#endif
    for (h=first;h<first+a->group;h++) {""",
    """  {
#ifdef BCIR_MM_DISPATCH
    if (a->variant==4) vt=vc;  /* BWD-VT */
#endif
    for (h=first;h<first+a->group;h++) {""",
)
Path(sys.argv[2]).write_text(t)
print("BWD-VT prototype:", sys.argv[2])
