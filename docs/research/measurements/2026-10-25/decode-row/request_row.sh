#!/usr/bin/env bash
# DECODE-ROW's whole requests, after measure_row.sh: the qualification's single runs are one 48-token request a process
# and read a few percent either way between runs, so this times whole requests apart -- CRC-SLICE's request_ab.py
# (../crc-slice/, as committed): the standalone C decoder from each tree (HEAD~1 the tree before, HEAD DECODE-ROW), a
# process a request (the model's load, a 5-token prompt, 8, 48 or 200 new tokens), 24 rounds in rotated order with an
# A/A control, clang and gcc, both lab models; every run's tokens and final logits the tree before's bytes. Run from
# <REPO>; <SCRATCH>, <LAB> and <LAB2> as in measure_row.sh. Writes requests.json beside itself.
set -euo pipefail
repo="<REPO>"; s="<SCRATCH>/decode-row-requests"; lab="<LAB>"; lab2="<LAB2>"
here="${repo}/docs/research/measurements/2026-10-25/decode-row"
crc="${repo}/docs/research/measurements/2026-10-25/crc-slice"
rm -rf "${s}"; mkdir -p "${s}"
for t in before after; do
  git -C "${repo}" worktree add --detach "${s}/${t}" "$([ "${t}" = before ] && echo HEAD~1 || echo HEAD)" >/dev/null
done
flags=(-std=c11 -O2 -ffp-contract=off -Wall -Wextra -Werror)
llama() {  # tree cc name
  local c="$1/runtime/c"
  "$2" "${flags[@]}" -I "${c}" "${c}/bcir_ai_kernels.c" "${c}/bcir_q8_model.c" "${c}/bcir_decode.c" \
    "${c}/bcir_llama.c" "${c}/bcir_llama_cli.c" -o "${s}/$3" -lm
}
for cc in clang gcc; do
  llama "${s}/before" "${cc}" "llama-before-${cc}"
  llama "${s}/after" "${cc}" "llama-after-${cc}"
  for m in 1 2; do
    model="$([ "${m}" = 1 ] && echo "${lab}" || echo "${lab2}")"
    python3 "${crc}/request_ab.py" 24 "${model}/model.bcirq8" "${s}/requests-${cc}-${m}.json" 8,48,200 \
      before="${s}/llama-before-${cc}" after="${s}/llama-after-${cc}" > /dev/null
  done
done
python3 -c 'import json, sys; json.dump({f"{cc} model {m}": json.load(open(f"{sys.argv[1]}/requests-{cc}-{m}.json")) for cc in ("clang", "gcc") for m in (1, 2)}, open(sys.argv[2], "w"), indent=1)' \
  "${s}" "${here}/requests.json"
for t in before after; do git -C "${repo}" worktree remove --force "${s}/${t}"; done
