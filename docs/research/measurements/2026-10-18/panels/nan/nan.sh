#!/bin/bash
# The panels harness's NaN rule, four cases on the writers' generic path (an AVX2-capped build):
# a writer whose NaN elements differ from the reference's only in their sign -- a NaN the contract
# allows, the case the AArch64 build met -- under 5804ad5's harness (bits) and this one's (a quiet
# NaN); and PN34, the writer storing a NaN element as a signaling NaN, under a rule that takes any
# NaN and under this one's. Serial, from scratch copies; the tree is not edited.
#
#   bash nan.sh OUT     (from the repository root)
set -u
OUT=$1; C=runtime/c; mkdir -p "$OUT"
git show 5804ad5:$C/test_decoder_train.c > "$OUT/harness_5804ad5.c"
cp $C/test_decoder_train.c "$OUT/harness_quiet.c"
python3 - "$OUT" <<'PY'
import sys
from pathlib import Path
out, c = Path(sys.argv[1]), Path("runtime/c")
t = (c / "bcir_tensor.c").read_text()
old = "  for (q=0;q<w;q++) j->pa[k][panel_a_at(j->c,i,c0+q)]=bf16_of(v[q]);"
assert t.count(old) == 1
body = "  for (q=0;q<w;q++) { uint16_t h=bf16_of(v[q]); j->pa[k][panel_a_at(j->c,i,c0+q)]=(uint16_t)((h&0x7fc0u)==0x7fc0u ? %s : h); }"
(out / "tensor_sign.c").write_text(t.replace(old, body % "(h^0x8000u)"))
(out / "tensor_pn34.c").write_text(t.replace(old, body % "((h&0xffbfu)|1u)"))
h = (out / "harness_quiet.c").read_text()
q = "static int bf16_quiet_nan(unsigned short h) { return (h&0x7fc0u)==0x7fc0u; }"
assert h.count(q) == 1
any_nan = "static int bf16_quiet_nan(unsigned short h) { return (h&0x7f80u)==0x7f80u && (h&0x7fu)!=0u; }"
(out / "harness_anynan.c").write_text(h.replace(q, any_nan))
PY
P="-DBCIR_DT_DIVISION_PROBE -DBCIR_DT_WORKSPACE_PROBE -DBCIR_DT_ATTENTION_PROBE -DBCIR_DT_PANELS_PROBE"
one() {  # name harness tensor
  gcc -std=c2x $P -DBCIR_TENSOR_MAX_VARIANT=2 -O2 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off -I $C \
    "$OUT/$2" "$OUT/$3" $C/bcir_decoder_train.c -o "$OUT/$1" -lm -pthread > "$OUT/$1.build" 2>&1 \
    || { echo "$1: BUILD-FAIL"; return; }
  if "$OUT/$1" > "$OUT/$1.log" 2>&1; then v=PASS; else v="FAIL($?)"; fi
  echo "$1: $v | $(grep -m1 -E 'panels:|Assertion' "$OUT/$1.log" | sed "s|$OUT/||" | cut -c1-160)"
}
one sign-5804ad5 harness_5804ad5.c tensor_sign.c
one sign-quiet harness_quiet.c tensor_sign.c
one pn34-anynan harness_anynan.c tensor_pn34.c
one pn34-quiet harness_quiet.c tensor_pn34.c
