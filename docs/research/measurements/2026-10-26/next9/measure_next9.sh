#!/usr/bin/env bash
# NEXT-9's records, run from <REPO> on the committed tree -- HEAD the tree CONTRACT-OFF left (e9eb43a), which NEXT-9
# leaves as it is -- one step at a time with nothing else running; <SCRATCH> a scratch directory and <LAB> the
# qualification's lab model (BCIR-Corpus-860K). The tree before is HEAD; the tree after is HEAD with q8_hoist.patch
# applied: Q8-HOIST's prototype, the vector matvecs' x times a group's scale formed once a row and group, each such
# product checked exact, and a call that formed an inexact one computed again as before. Q8-ROWS2, the head's two
# blocks, is rows_bench.c alone. Writes copies.json, matvec.json, rows.json, prototype.json, decode.json, ops.json,
# inprocess.json, switched.json, clock.json and hashes.json beside itself; Q8-LANES' decode A/B
# and operation timer (../../2026-10-25/q8-lanes/decode_ab.py and llama_ops.py) run as committed.
set -euo pipefail
repo="<REPO>"; s="<SCRATCH>/next9"; lab="<LAB>"
here="${repo}/docs/research/measurements/2026-10-26/next9"
lanes="${repo}/docs/research/measurements/2026-10-25/q8-lanes"
rm -rf "${s}"; mkdir -p "${s}"
for t in before after; do git -C "${repo}" worktree add --detach "${s}/${t}" HEAD >/dev/null; done
git -C "${s}/after" apply "${here}/q8_hoist.patch"
B="${s}/before/runtime/c"; A="${s}/after/runtime/c"
flags=(-std=c11 -O2 -ffp-contract=off -Wall -Wextra -Werror)
lines() { python3 -c 'import json, sys; json.dump([json.loads(l) for l in sys.stdin], open(sys.argv[1], "w"), indent=1)' "$1"; }

# copies.json: the benches' kernels the tree before's and the prototype's, byte for byte
python3 "${here}/copies.py" "${s}/before" "${here}" > "${here}/copies.json"

# matvec.json: hoist_bench.c -- the committed block matvec against the hoist's ceiling, its loop form and the
# prototype; rows.json: rows_bench.c -- the committed row-dot block against two blocks interleaved; three runs of each,
# each compiler (one run on this VM can be disturbed through all its rounds)
for cc in clang gcc; do
  "${cc}" "${flags[@]}" "${here}/hoist_bench.c" -o "${s}/hoist_bench-${cc}" -lm
  "${cc}" "${flags[@]}" "${here}/rows_bench.c" -o "${s}/rows_bench-${cc}" -lm
done
for run in 1 2 3; do
  for cc in clang gcc; do
    "${s}/hoist_bench-${cc}" 15 | sed "s/^{/{\"run\":${run},\"cc\":\"${cc}\",/"
  done
done | lines "${here}/matvec.json"
for run in 1 2 3; do
  for cc in clang gcc; do
    "${s}/rows_bench-${cc}" 15 | sed "s/^{/{\"run\":${run},\"cc\":\"${cc}\",/"
  done
done | lines "${here}/rows.json"

# prototype.json: the prototype's own witnesses on the tree after -- the kernel unit's C harness built as
# test_native_ai_c_harness_c11_and_c23 builds it (C11; C23; C23 capped at AVX2; C23 with the dispatch compiled out;
# each with the instance probe), each compiler, its summary lines; then that test with the compiler required
for cc in clang gcc; do
  n=0
  for build in "c11" "c2x" "c2x -DBCIR_AI_MAX_INSTANCE=2" "c2x -DBCIR_AI_NO_DISPATCH"; do
    read -r std opt <<< "${build}"
    exe="${s}/ai-kernels-${cc}-${n}"
    # shellcheck disable=SC2086  # opt is one option or none
    "${cc}" "-std=${std}" -DBCIR_AI_INSTANCE_PROBE ${opt} -O2 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off \
      -I "${A}" "${A}/test_ai_kernels.c" "${A}/bcir_ai_kernels.c" "${A}/bcir_q4_kernel.c" -o "${exe}" -lm
    "${exe}" > "${exe}.txt"
    python3 - "${exe}.txt" "${cc}" "${build}" <<'PY'
import json, re, sys
text = open(sys.argv[1]).read()
lanes = re.search(r"Q8-LANES: instances (\d+), (\d+) vector comparisons equal", text)
hoist = re.search(r"Q8-HOIST: (\d+) matvecs hoisted, (\d+) computed again term by term", text)
print(json.dumps({"cc": sys.argv[2], "build": "-std=" + sys.argv[3], "all_gates_passed": "native AI kernels: all gates passed" in text,
                  "instances": int(lanes.group(1)), "vector_comparisons": int(lanes.group(2)),
                  "matvecs_hoisted": int(hoist.group(1)), "matvecs_computed_again": int(hoist.group(2))}))
PY
    n=$((n + 1))
  done
done > "${s}/harness.jsonl"
(cd "${s}/after" && python3 tools/testing/check_tests.py --require bcir.tests.test_native_ai:_CC \
  bcir.tests.test_native_ai:test_native_ai_c_harness_c11_and_c23) > "${s}/witness.txt"
python3 - "${s}/harness.jsonl" "${s}/witness.txt" "${here}/prototype.json" <<'PY'
import json, sys
harness = [json.loads(line) for line in open(sys.argv[1])]
assert all(h["all_gates_passed"] for h in harness), harness
witness = {"test": "bcir.tests.test_native_ai:test_native_ai_c_harness_c11_and_c23", "required": "bcir.tests.test_native_ai:_CC",
           "exit": 0, "output": open(sys.argv[2]).read().strip().splitlines()}
json.dump({"harness": harness, "witness": witness}, open(sys.argv[3], "w"), indent=1)
PY

# decode.json: decode_ab.py -- bcir-llama from each tree, each compiler, 24 rounds
llama() {  # tree cc name
  local c="$1/runtime/c"
  "$2" "${flags[@]}" -I "${c}" "${c}/bcir_ai_kernels.c" "${c}/bcir_q8_model.c" "${c}/bcir_decode.c" \
    "${c}/bcir_llama.c" "${c}/bcir_llama_cli.c" -o "${s}/$3" -lm
}
for cc in clang gcc; do
  llama "${s}/before" "${cc}" "llama-before-${cc}"
  llama "${s}/after" "${cc}" "llama-after-${cc}"
  python3 "${lanes}/decode_ab.py" 24 "${lab}/model.bcirq8" "${s}/decode-${cc}.json" before="${s}/llama-before-${cc}" \
    after="${s}/llama-after-${cc}" > "${s}/decode-${cc}.log"
done
python3 -c 'import json, sys; json.dump({cc: json.load(open(f"{sys.argv[1]}/decode-{cc}.json")) for cc in ("clang", "gcc")}, open(sys.argv[2], "w"), indent=1)' \
  "${s}" "${here}/decode.json"

# ops.json: llama_ops.py -- each tree's decoder with its operations timed, each compiler, three repeats in
# alternating order, each five generations of 200 new tokens
for cc in clang gcc; do
  for rep in 1 2 3; do
    for t in $([ $((rep % 2)) = 1 ] && echo "before after" || echo "after before"); do
      python3 "${lanes}/llama_ops.py" "${s}/${t}" "${lab}/model.bcirq8" "${s}/ops-${t}-${cc}" "${cc}" 5 \
        | sed "s/^{/{\"cc\":\"${cc}\",\"tree\":\"${t}\",\"repeat\":${rep},/"
    done
  done
done | lines "${here}/ops.json"

# inprocess.json: hoist_ab.c, one process -- the tree before's decoder units, every global renamed before_*, beside the
# tree after's and a second copy of the tree before's renamed before2_* (the placement control, linked next to the
# after arm's units), each compiler: at the qualification's flags (plain), and with every function and loop of all
# four arms aligned to 64 bytes (aligned), so that where a hot loop lands moves less between the arms
units=(bcir_ai_kernels bcir_q8_model bcir_decode bcir_llama)
for cc in clang gcc; do
  for layout in plain aligned; do
    extra=(); if [ "${layout}" = aligned ]; then extra=(-falign-functions=64 -falign-loops=64); fi
    copies=()
    for prefix in before before2; do
      objs=()
      for u in "${units[@]}"; do
        "${cc}" "${flags[@]}" "${extra[@]}" -I "${B}" -c "${B}/${u}.c" -o "${s}/${prefix}-${u}-${cc}-${layout}.o"
        objs+=("${s}/${prefix}-${u}-${cc}-${layout}.o")
      done
      nm --defined-only -g "${objs[@]}" | awk -v p="${prefix}_" 'NF == 3 {print $3 " " p $3}' | sort -u \
        > "${s}/renames-${prefix}-${cc}-${layout}.txt"
      for o in "${objs[@]}"; do
        objcopy --redefine-syms="${s}/renames-${prefix}-${cc}-${layout}.txt" "${o}" "${o%.o}-r.o"
      done
    done
    for prefix in before2 before; do for u in "${units[@]}"; do copies+=("${s}/${prefix}-${u}-${cc}-${layout}-r.o"); done; done
    srcs=(); for u in "${units[@]}"; do srcs+=("${A}/${u}.c"); done
    "${cc}" "${flags[@]}" "${extra[@]}" -I "${A}" "${here}/hoist_ab.c" "${srcs[@]}" "${copies[@]}" \
      -o "${s}/hoist_ab-${cc}-${layout}" -lm
  done
done
for run in 1 2 3; do  # three runs of each build: one run's reading moves more between runs than its interval says
  for cc in clang gcc; do
    for layout in plain aligned; do
      "${s}/hoist_ab-${cc}-${layout}" 24 3 "${lab}/model.bcirq8" \
        | sed "s/^{/{\"run\":${run},\"cc\":\"${cc}\",\"layout\":\"${layout}\",/"
    done
  done
done | lines "${here}/inprocess.json"

# switched.json: switch_ab.c, one binary -- the tree after's decoder with switch.py's kernel unit, the block matvec's
# path chosen at run time (off, on, the A/A control, the committed kernel's copy, the ceiling), three runs, each
# compiler
python3 "${here}/switch.py" "${A}/bcir_ai_kernels.c" "${s}/bcir_ai_kernels_switched.c"
for cc in clang gcc; do
  "${cc}" "${flags[@]}" -I "${A}" "${here}/switch_ab.c" "${s}/bcir_ai_kernels_switched.c" "${A}/bcir_q8_model.c" \
    "${A}/bcir_decode.c" "${A}/bcir_llama.c" -o "${s}/switch_ab-${cc}" -lm
  "${cc}" "${flags[@]}" -I "${A}" "${here}/clock_probe.c" "${s}/bcir_ai_kernels_switched.c" -o "${s}/clock_probe-${cc}" -lm
done
for run in 1 2 3; do
  for cc in clang gcc; do
    "${s}/switch_ab-${cc}" 24 3 "${lab}/model.bcirq8" | sed "s/^{/{\"run\":${run},\"cc\":\"${cc}\",/"
  done
done | lines "${here}/switched.json"

# clock.json: clock_probe.c -- the core's cycle time while the matvecs run, back to back (pass: a 20,000-step chain
# after each pass of 28) and with a decode's duty cycle (matvec: 10,000 steps after each matvec), three runs, each
# compiler
for run in 1 2 3; do
  for cc in clang gcc; do
    "${s}/clock_probe-${cc}" 24 40 20000 pass | sed "s/^{/{\"run\":${run},\"cc\":\"${cc}\",/"
    "${s}/clock_probe-${cc}" 24 40 280000 matvec | sed "s/^{/{\"run\":${run},\"cc\":\"${cc}\",/"
  done
done | lines "${here}/clock.json"

# hashes.json: the tools' sha256 (this directory's and those run from Q8-LANES') and the measured sources'
python3 - "${s}" "${here}" "${lanes}" <<'PY'
import hashlib, json, sys
from pathlib import Path
s, here, lanes = (Path(a) for a in sys.argv[1:4])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
record = {
    "tools": {t: sha(here / t) for t in ("measure_next9.sh", "copies.py", "hoist_bench.c", "rows_bench.c", "hoist_ab.c",
                                         "q8_hoist.patch", "switch.py", "switch_ab.c", "clock_probe.c")},
    "tools run from ../../2026-10-25/q8-lanes": {t: sha(lanes / t) for t in ("decode_ab.py", "llama_ops.py")},
    "sources": {f"{t}/{u}": sha(s / t / "runtime" / "c" / u)
                for t in ("before", "after")
                for u in ("bcir_ai_kernels.c", "test_ai_kernels.c", "bcir_q4_kernel.c", "bcir_llama.c", "bcir_decode.c",
                          "bcir_q8_model.c", "bcir_llama_cli.c")},
    "switched/bcir_ai_kernels.c": sha(s / "bcir_ai_kernels_switched.c"),
}
(here / "hashes.json").write_text(json.dumps(record, indent=1) + "\n")
PY
for t in before after; do git -C "${repo}" worktree remove --force "${s}/${t}"; done
