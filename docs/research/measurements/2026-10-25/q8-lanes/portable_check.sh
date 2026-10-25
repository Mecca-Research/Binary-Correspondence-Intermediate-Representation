#!/usr/bin/env bash
# Q8-LANES' portable instance against the tree before. (1) Each compiler's instructions for the portable loops: the
# tree after's q8_matvec_portable and q8_rows_dot_portable against the tree before's prevalidated entries, every
# instruction compared but the alignment padding (nop forms) and GCC's endbr64 (an exported function's, not a static
# one's). (2) GCC's kernel A/B (q8_ab.c's small shapes, 15 rounds) at its default loop alignment and with loops
# aligned to 32 bytes, each in both link orders: whether the instructions placed differently read differently.
# Each build -std=c11 -O2 -ffp-contract=off.   usage: portable_check.sh BEFORE AFTER OUT_DIR Q8_AB_C -> JSON lines
set -euo pipefail
before="$1/runtime/c"; after="$2/runtime/c"; out="$3"; ab="$4"
mkdir -p "${out}"
flags=(-std=c11 -O2 -ffp-contract=off)
listing() {  # object function -> its instructions, one a line, without addresses, padding or endbr64
  objdump -d --no-show-raw-insn "$1" | awk -v f="<$2" 'index($0, f) && /^[0-9a-f]+ </ {on = 1; next} on && /^$/ {exit} on' |
    sed -e 's/^ *[0-9a-f]*:\t//' -e 's/[0-9a-f]* <[^>]*>//' -e 's/#.*//' -e 's/ *$//' |
    grep -v -E '^(nop|xchg|cs nop|data16|endbr64)|^$' || true
}
for cc in clang gcc; do
  "${cc}" "${flags[@]}" -I "${before}" -c "${before}/bcir_ai_kernels.c" -o "${out}/before-${cc}.o"
  "${cc}" "${flags[@]}" -I "${after}" -c "${after}/bcir_ai_kernels.c" -o "${out}/after-${cc}.o"
  for pair in bcir_ai_q8_matvec_f64_prevalidated:q8_matvec_portable bcir_ai_q8_rows_dot_f64_prevalidated:q8_rows_dot_portable; do
    listing "${out}/before-${cc}.o" "${pair%%:*}>" > "${out}/b.s"
    listing "${out}/after-${cc}.o" "${pair#*:}" > "${out}/a.s"
    differ=$(diff "${out}/b.s" "${out}/a.s" | grep -c '^[<>]' || true)
    printf '{"check":"instructions","cc":"%s","before":"%s","after":"%s","before_instructions":%d,"after_instructions":%d,"differing_lines":%d}\n' \
      "${cc}" "${pair%%:*}" "${pair#*:}" "$(wc -l < "${out}/b.s")" "$(wc -l < "${out}/a.s")" "${differ}"
  done
done
renames=()
for f in abi_version quantize_q8_f64 quantize_q4_f64 q8_matvec_f64 q8_values_validate q8_matvec_f64_prevalidated \
         q8_rows_dot_f64 q8_rows_dot_f64_prevalidated q15_topk; do renames+=("-Dbcir_ai_${f}=before_${f}"); done
for align in default 32; do
  extra=(); [ "${align}" = default ] || extra=("-falign-loops=${align}")
  gcc "${flags[@]}" "${extra[@]}" "${renames[@]}" -I "${before}" -c "${before}/bcir_ai_kernels.c" -o "${out}/b.o"
  gcc "${flags[@]}" "${extra[@]}" -I "${after}" -c "${after}/bcir_ai_kernels.c" -o "${out}/a.o"
  gcc "${flags[@]}" "${extra[@]}" -I "${after}" -c "${ab}" -o "${out}/m.o"
  gcc "${out}/m.o" "${out}/a.o" "${out}/b.o" -o "${out}/ab-after-first" -lm
  gcc "${out}/m.o" "${out}/b.o" "${out}/a.o" -o "${out}/ab-before-first" -lm
  for order in after-first before-first; do
    "${out}/ab-${order}" 15 200 m:128,128,32 m:128,64,32 m:128,344,32 m:344,128,32 r:128,1024,32 |
      sed "s/^{/{\"check\":\"gcc layout\",\"loop_alignment\":\"${align}\",\"link_order\":\"${order}\",/"
  done
done
