#!/usr/bin/env bash
# CRC-SLICE's records, run from <REPO> on the committed tree -- HEAD CRC-SLICE, HEAD~1 the tree before -- one step at a
# time with nothing else running; <SCRATCH> a scratch directory, <LAB> the qualification's lab model (BCIR-Corpus-860K)
# and <LAB2> a second one trained apart. Writes crc.json, trust.json, requests.json, qualification.json, note-crc.json
# and hashes.json beside itself (explore.json and explore_trust_parts.c are NEXT-6's survey, on the tree before).
set -euo pipefail
repo="<REPO>"; s="<SCRATCH>/crc-slice"; lab="<LAB>"; lab2="<LAB2>"
here="${repo}/docs/research/measurements/2026-10-25/crc-slice"
rm -rf "${s}"; mkdir -p "${s}"
for t in before after; do
  git -C "${repo}" worktree add --detach "${s}/${t}" "$([ "${t}" = before ] && echo HEAD~1 || echo HEAD)" >/dev/null
done
B="${s}/before/runtime/c"; A="${s}/after/runtime/c"
flags=(-std=c11 -O2 -ffp-contract=off -Wall -Wextra -Werror)
lines() { python3 -c 'import json, sys; json.dump([json.loads(l) for l in sys.stdin], open(sys.argv[1], "w"), indent=1)' "$1"; }

# the qualification's native pack (its program, plan and hydration are the same bytes on both trees)
python3 "${here}/dump_pack.py" "${s}/after" "${lab}" "${s}/pack.bin"

# crc.json: crc_ab.c -- the tree before's bcir_runtime.c and bcir_q8_model.c, every global renamed before_*, against the
# tree after's, with zlib's crc32 beside them, clang and gcc
for cc in clang gcc; do
  "${cc}" "${flags[@]}" -I "${B}" -c "${B}/bcir_runtime.c" -o "${s}/b-rt-${cc}.o"
  "${cc}" "${flags[@]}" -I "${B}" -c "${B}/bcir_q8_model.c" -o "${s}/b-qm-${cc}.o"
  nm --defined-only -g "${s}/b-rt-${cc}.o" "${s}/b-qm-${cc}.o" | awk 'NF == 3 {print $3 " before_" $3}' | sort -u \
    > "${s}/renames-${cc}.txt"
  objcopy --redefine-syms="${s}/renames-${cc}.txt" "${s}/b-rt-${cc}.o" "${s}/b-rt-${cc}-r.o"
  objcopy --redefine-syms="${s}/renames-${cc}.txt" "${s}/b-qm-${cc}.o" "${s}/b-qm-${cc}-r.o"
  "${cc}" "${flags[@]}" -I "${A}" "${here}/crc_ab.c" "${A}/bcir_runtime.c" "${A}/bcir_q8_model.c" \
    "${s}/b-rt-${cc}-r.o" "${s}/b-qm-${cc}-r.o" -o "${s}/crc_ab-${cc}" -lz -lm
done
for cc in clang gcc; do
  for model in "${lab}" "${lab2}"; do
    "${s}/crc_ab-${cc}" 15 "${model}/model.bcirq8" 16384 131072 609997 912840 1048576 8388608 |
      sed "s/^{/{\"cc\":\"${cc}\",\"model_sha256\":\"$(sha256sum "${model}/model.bcirq8" | cut -c1-16)\",/"
  done
done | lines "${here}/crc.json"

# trust.json: trust_parts.c built against each tree, the two in alternation, five times each
for t in before after; do
  for cc in clang gcc; do
    c="${s}/${t}/runtime/c"
    "${cc}" "${flags[@]}" -I "${c}" "${here}/trust_parts.c" "${c}/bcir_runtime.c" "${c}/bcir_exec.c" -o "${s}/trust-${t}-${cc}"
  done
done
for run in 1 2 3 4 5; do
  for cc in clang gcc; do
    for t in $([ $((run % 2)) = 1 ] && echo "before after" || echo "after before"); do
      "${s}/trust-${t}-${cc}" "${s}/pack.bin" 9 5 | sed "s/^{/{\"cc\":\"${cc}\",\"tree\":\"${t}\",\"run\":${run},/"
    done
  done
done | lines "${here}/trust.json"

# requests.json: request_ab.py, bcir-llama from each tree, each compiler, both lab models
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
    python3 "${here}/request_ab.py" 24 "${model}/model.bcirq8" "${s}/requests-${cc}-${m}.json" 1,8,64 \
      before="${s}/llama-before-${cc}" after="${s}/llama-after-${cc}"
  done
done
python3 -c 'import json, sys; json.dump({f"{cc} model {m}": json.load(open(f"{sys.argv[1]}/requests-{cc}-{m}.json")) for cc in ("clang", "gcc") for m in (1, 2)}, open(sys.argv[2], "w"), indent=1)' \
  "${s}" "${here}/requests.json"

# qualification.json: run_qualification.py from each tree, its traffic under callgrind
for t in before after; do
  (cd "${s}/${t}" && python3 tools/models/run_qualification.py --model-dir "${lab}" --output "${s}/qual-${t}.json" \
    --require-callgrind > "${s}/qual-${t}.log")
done
python3 "${here}/qual_crc.py" "${s}/qual-before.json" "${s}/qual-after.json" > "${here}/qualification.json"

# note-crc.json: the 2026-10-10 performance note's own CRC stage (bcir_crc32 against Python's zlib) on each tree
for t in before after; do
  python3 "${s}/${t}/docs/research/measurements/2026-10-10/run_performance_tests.py" --stage crc --output "${s}/note-${t}" \
    > /dev/null
done
python3 -c 'import json, sys; json.dump({t: json.load(open(f"{sys.argv[1]}/note-{t}/crc.json")) for t in ("before", "after")}, open(sys.argv[2], "w"), indent=1)' \
  "${s}" "${here}/note-crc.json"

# hashes.json: the committed tools' sha256 and the measured sources'
python3 - "${s}" "${here}" <<'EOF'
import hashlib, json, sys
from pathlib import Path
s, here = Path(sys.argv[1]), Path(sys.argv[2])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
tools = ["measure_crc.sh", "crc_ab.c", "trust_parts.c", "request_ab.py", "qual_crc.py", "dump_pack.py", "explore_trust_parts.c"]
units = ["bcir_runtime.c", "bcir_runtime.h", "bcir_q8_model.c", "bcir_exec.c", "bcir_llama_cli.c"]
record = {"tools": {t: sha(here / t) for t in tools},
          "sources": {f"{t}/{u}": sha(s / t / "runtime" / "c" / u) for t in ("before", "after") for u in units},
          "pack.bin": sha(s / "pack.bin")}
record["sources"]["after/bcir_crc32.h"] = sha(s / "after" / "runtime" / "c" / "bcir_crc32.h")
(here / "hashes.json").write_text(json.dumps(record, indent=1) + "\n")
EOF
for t in before after; do git -C "${repo}" worktree remove --force "${s}/${t}"; done
