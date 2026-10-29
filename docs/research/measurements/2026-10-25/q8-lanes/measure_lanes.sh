#!/usr/bin/env bash
# Q8-LANES' records, run from <REPO> on the committed tree -- HEAD Q8-LANES' second commit (the portable loops in
# functions of their own), HEAD~1 its first, HEAD~2 the tree before -- one step at a time with nothing else running;
# <SCRATCH> a scratch directory, <LAB> the qualification's lab model (BCIR-Corpus-860K) and <LAB2> a second one
# trained apart. Writes kernels.json, decode.json, ops.json, bits.json, portable.json, qualification.json,
# faults.json and hashes.json beside itself. kernels-b444807.json and decode-b444807.json are its first run, made when
# the first commit was HEAD and the tree before HEAD~1, and stopped after its decode A/B: its kernel A/B read the
# portable instance slower than the tree before's loops, which the second commit fixes.
set -euo pipefail
repo="<REPO>"; s="<SCRATCH>/q8-lanes"; lab="<LAB>"; lab2="<LAB2>"
here="${repo}/docs/research/measurements/2026-10-25/q8-lanes"
rm -rf "${s}"; mkdir -p "${s}"
for t in before after without; do
  git -C "${repo}" worktree add --detach "${s}/${t}" "$([ "${t}" = before ] && echo HEAD~2 || echo HEAD)" >/dev/null
done
B="${s}/before/runtime/c"; A="${s}/after/runtime/c"
flags=(-std=c11 -O2 -ffp-contract=off -Wall -Wextra -Werror)
lines() { python3 -c 'import json, sys; json.dump([json.loads(l) for l in sys.stdin], open(sys.argv[1], "w"), indent=1)' "$1"; }

# kernels.json: q8_ab.c against the tree before's unit, its entry points renamed before_*
renames=()
for f in abi_version quantize_q8_f64 quantize_q4_f64 q8_matvec_f64 q8_values_validate q8_matvec_f64_prevalidated \
         q8_rows_dot_f64 q8_rows_dot_f64_prevalidated q15_topk; do renames+=("-Dbcir_ai_${f}=before_${f}"); done
for cc in clang gcc; do
  "${cc}" "${flags[@]}" "${renames[@]}" -I "${B}" -c "${B}/bcir_ai_kernels.c" -o "${s}/before-${cc}.o"
  "${cc}" "${flags[@]}" -I "${A}" "${here}/q8_ab.c" "${A}/bcir_ai_kernels.c" "${s}/before-${cc}.o" \
    -o "${s}/q8_ab-${cc}" -lm
done
for cc in clang gcc; do
  { "${s}/q8_ab-${cc}" 15 200 m:128,128,32 m:128,64,32 m:128,344,32 m:344,128,32 r:128,1024,32
    "${s}/q8_ab-${cc}" 10 3 m:2048,2048,32 m:2048,256,32 m:2048,5632,32 m:5632,2048,32 r:2048,32000,32
  } | sed "s/^{/{\"cc\":\"${cc}\",/"
done | lines "${here}/kernels.json"

# decode.json: decode_ab.py, bcir-llama from each tree, each compiler
llama() {  # tree cc name [define]
  local c="$1/runtime/c"
  "$2" "${flags[@]}" "${@:4}" -I "${c}" "${c}/bcir_ai_kernels.c" "${c}/bcir_q8_model.c" "${c}/bcir_decode.c" \
    "${c}/bcir_llama.c" "${c}/bcir_llama_cli.c" -o "${s}/$3" -lm
}
for cc in clang gcc; do
  llama "${s}/before" "${cc}" "llama-before-${cc}"
  llama "${s}/after" "${cc}" "llama-after-${cc}"
  llama "${s}/after" "${cc}" "llama-avx2-${cc}" -DBCIR_AI_MAX_INSTANCE=2
  llama "${s}/after" "${cc}" "llama-portable-${cc}" -DBCIR_AI_MAX_INSTANCE=1
  python3 "${here}/decode_ab.py" 24 "${lab}/model.bcirq8" "${s}/decode-${cc}.json" before="${s}/llama-before-${cc}" \
    after="${s}/llama-after-${cc}" after-avx2="${s}/llama-avx2-${cc}" after-portable="${s}/llama-portable-${cc}"
done
python3 -c 'import json, sys; json.dump({cc: json.load(open(f"{sys.argv[1]}/decode-{cc}.json")) for cc in ("clang", "gcc")}, open(sys.argv[2], "w"), indent=1)' \
  "${s}" "${here}/decode.json"

# ops.json: llama_ops.py on each tree
for t in before after; do
  python3 "${here}/llama_ops.py" "${s}/${t}" "${lab}/model.bcirq8" "${s}/ops-${t}" clang 5 | sed "s/^{/{\"tree\":\"${t}\",/"
done | lines "${here}/ops.json"

# bits.json: bits.sh over both lab models
bash "${here}/bits.sh" "${s}/before" "${s}/after" "${s}/bits" "${lab}/model.bcirq8" "${lab2}/model.bcirq8" \
  | lines "${here}/bits.json"

# portable.json: portable_check.sh -- the portable loops' instructions against the tree before's, GCC's layout A/B
bash "${here}/portable_check.sh" "${s}/before" "${s}/after" "${s}/portable" "${here}/q8_ab.c" | lines "${here}/portable.json"

# qualification.json: run_qualification.py from each tree, its traffic under callgrind
for t in before after; do
  (cd "${s}/${t}" && python3 tools/models/run_qualification.py --model-dir "${lab}" --output "${s}/qual-${t}.json" \
    --require-callgrind > "${s}/qual-${t}.log")
done
python3 "${here}/qual_summary.py" "${s}/qual-before.json" "${s}/qual-after.json" > "${here}/qualification.json"

# faults.json: q8-lanes.json swept on the tree after, then on the witnesses before Q8-LANES (without_the_cases.py)
(cd "${s}/after" && python3 tools/testing/red_sweep.py --faults tools/testing/faults/q8-lanes.json \
  --json-out "${s}/faults-q8-lanes.json" > "${s}/faults-q8-lanes.txt")
python3 "${here}/without_the_cases.py" "${s}/without" "${s}/q8-lanes-without.json"
(cd "${s}/without" && python3 tools/testing/red_sweep.py --faults "${s}/q8-lanes-without.json" \
  --json-out "${s}/faults-without.json" > "${s}/faults-without.txt") || true  # its faults the old witnesses miss
test -s "${s}/faults-without.json"
python3 - "${s}" "${here}/faults.json" "$(command -v python3)" "${repo}" <<'EOF'
import json, sys
s, out, python, repo = sys.argv[1:5]
def scrub(v):  # the sweeps' strings carry no host paths: the interpreter, the scratch and the repository named
    if isinstance(v, dict):
        return {k: scrub(x) for k, x in v.items()}
    if isinstance(v, list):
        return [scrub(x) for x in v]
    if isinstance(v, str):
        if v in (python, sys.executable):
            return "@python"
        return v.replace(s, "<" "SCRATCH>/q8-lanes").replace(repo, "<" "REPO>")  # split: the run copy's fill keeps them
    return v
json.dump({"q8_lanes": scrub(json.load(open(f"{s}/faults-q8-lanes.json"))),
           "without_the_cases": scrub(json.load(open(f"{s}/faults-without.json")))}, open(out, "w"), indent=1)
EOF

# hashes.json: the committed tools' sha256 and the measured sources' (the first commit's unit, its two records')
python3 - "${s}" "${here}" "${repo}" <<'EOF'
import hashlib, json, subprocess, sys
from pathlib import Path
s, here, repo = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
first = subprocess.run(["git", "-C", repo, "show", "HEAD~1:runtime/c/bcir_ai_kernels.c"], capture_output=True, check=True).stdout
tools = ["measure_lanes.sh", "q8_ab.c", "decode_ab.py", "llama_ops.py", "bits.sh", "portable_check.sh", "qual_summary.py",
         "without_the_cases.py"]
units = ["bcir_ai_kernels.c", "bcir_ai_kernels.h", "test_ai_kernels.c", "bcir_llama.c", "bcir_decode.c", "bcir_q8_model.c",
         "bcir_llama_cli.c"]
record = {"tools": {t: sha(here / t) for t in tools},
          "sources": {f"{t}/{u}": sha(s / t / "runtime" / "c" / u) for t in ("before", "after") for u in units}}
record["sources"]["first/bcir_ai_kernels.c"] = hashlib.sha256(first).hexdigest()
(here / "hashes.json").write_text(json.dumps(record, indent=1) + "\n")
EOF
for t in before after without; do git -C "${repo}" worktree remove --force "${s}/${t}"; done
