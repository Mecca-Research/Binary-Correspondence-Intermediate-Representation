#!/usr/bin/env bash
# DECODE-ROW's records, run from <REPO> on the committed tree -- HEAD DECODE-ROW, HEAD~1 the tree before -- one step at a
# time with nothing else running; <SCRATCH> a scratch directory, <LAB> the qualification's lab model (BCIR-Corpus-860K)
# and <LAB2> a second one trained apart. Writes kernels.json, decode.json, ops.json, bits.json, qualification.json,
# qualification-repeats.json and hashes.json beside itself (explore.json and gqa_bench.c are NEXT-7's survey, on the
# tree before). The decode A/B, the operation timer and the bit check are Q8-LANES' instruments as committed there
# (../q8-lanes/: decode_ab.py, llama_ops.py, bits.sh), the qualification's rows CRC-SLICE's (../crc-slice/qual_crc.py).
set -euo pipefail
repo="<REPO>"; s="<SCRATCH>/decode-row"; lab="<LAB>"; lab2="<LAB2>"
here="${repo}/docs/research/measurements/2026-10-25/decode-row"
q8="${repo}/docs/research/measurements/2026-10-25/q8-lanes"; crc="${repo}/docs/research/measurements/2026-10-25/crc-slice"
rm -rf "${s}"; mkdir -p "${s}"
for t in before after; do
  git -C "${repo}" worktree add --detach "${s}/${t}" "$([ "${t}" = before ] && echo HEAD~1 || echo HEAD)" >/dev/null
done
B="${s}/before/runtime/c"; A="${s}/after/runtime/c"
flags=(-std=c11 -O2 -ffp-contract=off -Wall -Wextra -Werror)
lines() { python3 -c 'import json, sys; json.dump([json.loads(l) for l in sys.stdin], open(sys.argv[1], "w"), indent=1)' "$1"; }

# kernels.json: attn_ab.c -- the tree before's bcir_decode.c, every global renamed before_*, against the tree after's
for cc in clang gcc; do
  "${cc}" "${flags[@]}" -I "${B}" -c "${B}/bcir_decode.c" -o "${s}/b-dec-${cc}.o"
  nm --defined-only -g "${s}/b-dec-${cc}.o" | awk 'NF == 3 {print $3 " before_" $3}' | sort -u > "${s}/renames-${cc}.txt"
  objcopy --redefine-syms="${s}/renames-${cc}.txt" "${s}/b-dec-${cc}.o" "${s}/b-dec-${cc}-r.o"
  "${cc}" "${flags[@]}" -I "${A}" "${here}/attn_ab.c" "${A}/bcir_decode.c" "${s}/b-dec-${cc}-r.o" -o "${s}/attn_ab-${cc}" \
    -lm
done
for cc in clang gcc; do "${s}/attn_ab-${cc}" 15 | sed "s/^{/{\"cc\":\"${cc}\",/"; done | lines "${here}/kernels.json"

# decode.json: decode_ab.py, bcir-llama from each tree, each compiler, both lab models
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
    python3 "${q8}/decode_ab.py" 24 "${model}/model.bcirq8" "${s}/decode-${cc}-${m}.json" \
      before="${s}/llama-before-${cc}" after="${s}/llama-after-${cc}"
  done
done
python3 -c 'import json, sys; json.dump({f"{cc} model {m}": json.load(open(f"{sys.argv[1]}/decode-{cc}-{m}.json")) for cc in ("clang", "gcc") for m in (1, 2)}, open(sys.argv[2], "w"), indent=1)' \
  "${s}" "${here}/decode.json"

# ops.json: llama_ops.py on each tree (clang, 200 tokens, the median of 5 runs)
for t in before after; do
  python3 "${q8}/llama_ops.py" "${s}/${t}" "${lab}/model.bcirq8" "${s}/ops-${t}" clang 5 | sed "s/^{/{\"tree\":\"${t}\",/"
done | lines "${here}/ops.json"

# bits.json: bits.sh over both lab models (7 builds of the tree after against the tree before's bytes)
bash "${q8}/bits.sh" "${s}/before" "${s}/after" "${s}/bits" "${lab}/model.bcirq8" "${lab2}/model.bcirq8" \
  | lines "${here}/bits.json"

# qualification.json: run_qualification.py from each tree, its traffic under callgrind
for t in before after; do
  (cd "${s}/${t}" && python3 tools/models/run_qualification.py --model-dir "${lab}" --output "${s}/qual-${t}.json" \
    --require-callgrind > "${s}/qual-${t}.log")
done
python3 "${crc}/qual_crc.py" "${s}/qual-before.json" "${s}/qual-after.json" > "${here}/qualification.json"

# qualification-repeats.json: three runs of each tree in alternation without callgrind, each run's C latency rows
for run in 1 2 3; do
  for t in $([ $((run % 2)) = 1 ] && echo "before after" || echo "after before"); do
    (cd "${s}/${t}" && python3 tools/models/run_qualification.py --model-dir "${lab}" --output "${s}/q-${t}-${run}.json" \
      --no-callgrind > /dev/null)
    python3 -c 'import json, sys; d = json.load(open(sys.argv[1])); c = d["latency"]["c"]; print(json.dumps({"tree": sys.argv[2], "run": int(sys.argv[3]), "commit": d["commit"], "status": d["status"], "monolithic_median_ms": round(c["monolithic_median_ms"], 3), "gem_median_ms": round(c["gem_median_ms"], 3), "gem_over_monolithic_median": round(c["gem_over_monolithic_median"], 4), "per_token_ms_median": round(c["per_token_ms"]["median"], 4), "time_to_first_token_ms": round(c["time_to_first_token_ms"], 3)}))' \
      "${s}/q-${t}-${run}.json" "${t}" "${run}"
  done
done | lines "${here}/qualification-repeats.json"

# hashes.json: the tools' sha256 (this directory's and the ones run from Q8-LANES' and CRC-SLICE's) and the sources'
python3 - "${s}" "${here}" "${q8}" "${crc}" <<'PY'
import hashlib, json, sys
from pathlib import Path
s, here, q8, crc = (Path(a) for a in sys.argv[1:5])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
record = {
    "tools": {t: sha(here / t) for t in ("measure_row.sh", "attn_ab.c", "gqa_bench.c")},
    "tools run from ../q8-lanes": {t: sha(q8 / t) for t in ("decode_ab.py", "llama_ops.py", "bits.sh")},
    "tools run from ../crc-slice": {"qual_crc.py": sha(crc / "qual_crc.py")},
    "sources": {f"{t}/{u}": sha(s / t / "runtime" / "c" / u)
                for t in ("before", "after")
                for u in ("bcir_decode.c", "bcir_decode.h", "bcir_llama.c", "bcir_llama_ops.h", "bcir_decoder_gem.c")},
}
(here / "hashes.json").write_text(json.dumps(record, indent=1) + "\n")
PY
for t in before after; do git -C "${repo}" worktree remove --force "${s}/${t}"; done
