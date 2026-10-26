#!/usr/bin/env bash
# Q8-BOUND's records, run from <REPO> on the committed tree -- HEAD Q8-BOUND, HEAD~1 the tree before (NEXT-9's
# records) -- one step at a time with nothing else running; <SCRATCH> a scratch directory and <LAB> the
# qualification's lab model (BCIR-Corpus-860K). Writes witness.json, kernels.json, trees.json, switched.json,
# clock.json, decode.json, ops.json, inprocess.json, qualification.json, qualification-repeats.json, faults.json and
# hashes.json beside itself. Q8-LANES' kernel and decode A/Bs, operation timer and qualification summary
# (../../2026-10-25/q8-lanes/q8_ab.c, decode_ab.py, llama_ops.py and qual_summary.py) and NEXT-9's in-process A/B
# (../next9/hoist_ab.c) run as committed.
set -euo pipefail
repo="<REPO>"; s="<SCRATCH>/q8-bound"; lab="<LAB>"
here="${repo}/docs/research/measurements/2026-10-26/q8-bound"
lanes="${repo}/docs/research/measurements/2026-10-25/q8-lanes"
next9="${repo}/docs/research/measurements/2026-10-26/next9"
rm -rf "${s}"; mkdir -p "${s}"
for t in before after without; do
  git -C "${repo}" worktree add --detach "${s}/${t}" "$([ "${t}" = before ] && echo HEAD~1 || echo HEAD)" >/dev/null
done
# the vacuity control: the tree after's kernels under the witnesses there were before Q8-BOUND
git -C "${s}/without" checkout HEAD~1 -- runtime/c/test_ai_kernels.c bcir/tests/test_native_ai.py
B="${s}/before/runtime/c"; A="${s}/after/runtime/c"
flags=(-std=c11 -O2 -ffp-contract=off -Wall -Wextra -Werror)
lines() { python3 -c 'import json, sys; json.dump([json.loads(l) for l in sys.stdin], open(sys.argv[1], "w"), indent=1)' "$1"; }

# witness.json: the tree after's kernel harness (test_ai_kernels.c) in the four builds
# test_native_ai_c_harness_c11_and_c23 makes (C11; C23; C23 capped at AVX2; C23 with the dispatch compiled out; each
# with the instance probe), each compiler, its summary lines; then the fault tables' command with the compiler required
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
bound = re.search(r"Q8-BOUND: (\d+) matvecs hoisted, (\d+) computed again", text)
print(json.dumps({"cc": sys.argv[2], "build": "-std=" + sys.argv[3], "all_gates_passed": "native AI kernels: all gates passed" in text,
                  "instances": int(lanes.group(1)), "vector_comparisons": int(lanes.group(2)),
                  "matvecs_hoisted": int(bound.group(1)), "matvecs_computed_again": int(bound.group(2))}))
PY
    n=$((n + 1))
  done
done > "${s}/harness.jsonl"
tests=(bcir.tests.test_native_ai:test_native_ai_c_harness_c11_and_c23
       bcir.tests.test_native_ai:test_native_q8_instances_match_the_increasing_coordinate_oracle
       bcir.tests.test_native_ai:test_native_q8_matrix_kernels_match_increasing_coordinate_oracle_exactly)
(cd "${s}/after" && python3 tools/testing/check_tests.py --require bcir.tests.test_native_ai:_CC "${tests[@]}") \
  > "${s}/witness.txt"
python3 - "${s}/harness.jsonl" "${s}/witness.txt" "${here}/witness.json" "${tests[@]}" <<'PY'
import json, sys
harness = [json.loads(line) for line in open(sys.argv[1])]
assert all(h["all_gates_passed"] for h in harness), harness
witness = {"tests": sys.argv[4:], "required": "bcir.tests.test_native_ai:_CC", "exit": 0,
           "output": open(sys.argv[2]).read().strip().splitlines()}
json.dump({"harness": harness, "witness": witness}, open(sys.argv[3], "w"), indent=1)
PY

# the switched unit: the tree after's kernel unit with Q8-BOUND's choice behind a run-time switch, its calls counted
python3 "${here}/bound_switch.py" "${A}/bcir_ai_kernels.c" "${s}/bcir_ai_kernels_switched.c"

# kernels.json: bound_kernels.c, one binary -- each vector instance's matvec with Q8-BOUND's choice off and on, at the
# 860K decoder's shapes, a ragged one (the AVX-512 runs) and TinyLlama's, three runs, each compiler
for cc in clang gcc; do
  "${cc}" "${flags[@]}" -I "${A}" "${here}/bound_kernels.c" "${s}/bcir_ai_kernels_switched.c" -o "${s}/bound_kernels-${cc}" -lm
done
for run in 1 2 3; do
  for cc in clang gcc; do
    { "${s}/bound_kernels-${cc}" 15 200 128,128,32 128,64,32 128,344,32 344,128,32 128,100,32
      "${s}/bound_kernels-${cc}" 10 3 2048,2048,32 2048,256,32 2048,5632,32 5632,2048,32
    } | sed "s/^{/{\"run\":${run},\"cc\":\"${cc}\",/"
  done
done | lines "${here}/kernels.json"

# trees.json: Q8-LANES' q8_ab.c against the tree before's unit, its entry points renamed before_* -- the tree after's
# instances against the tree before's widest, the row dots Q8-BOUND leaves as they were the control -- three runs
renames=()
for f in abi_version quantize_q8_f64 quantize_q4_f64 q8_matvec_f64 q8_values_validate q8_matvec_f64_prevalidated \
         q8_rows_dot_f64 q8_rows_dot_f64_prevalidated q15_topk q8_instances q8_matvec_f64_instance \
         q8_rows_dot_f64_instance; do renames+=("-Dbcir_ai_${f}=before_${f}"); done
for cc in clang gcc; do
  "${cc}" "${flags[@]}" "${renames[@]}" -I "${B}" -c "${B}/bcir_ai_kernels.c" -o "${s}/before-${cc}.o"
  "${cc}" "${flags[@]}" -I "${A}" "${lanes}/q8_ab.c" "${A}/bcir_ai_kernels.c" "${s}/before-${cc}.o" \
    -o "${s}/q8_ab-${cc}" -lm
done
for run in 1 2 3; do
  for cc in clang gcc; do
    { "${s}/q8_ab-${cc}" 15 200 m:128,128,32 m:128,64,32 m:128,344,32 m:344,128,32 r:128,1024,32
      "${s}/q8_ab-${cc}" 10 3 m:2048,2048,32 m:2048,256,32 m:2048,5632,32 m:5632,2048,32 r:2048,32000,32
    } | sed "s/^{/{\"run\":${run},\"cc\":\"${cc}\",/"
  done
done | lines "${here}/trees.json"

# switched.json: bound_ab.c, one binary -- the tree after's decoder with the switched unit, Q8-BOUND's choice off, on
# and off again (the A/A control), on the widest instance and capped at AVX2, three runs, each compiler
for cc in clang gcc; do
  for widest in 4 2; do
    "${cc}" "${flags[@]}" "-DBCIR_AI_MAX_INSTANCE=${widest}" -I "${A}" "${here}/bound_ab.c" \
      "${s}/bcir_ai_kernels_switched.c" "${A}/bcir_q8_model.c" "${A}/bcir_decode.c" "${A}/bcir_llama.c" \
      -o "${s}/bound_ab-${cc}-${widest}" -lm
  done
done
for run in 1 2 3; do
  for cc in clang gcc; do
    for widest in 4 2; do
      "${s}/bound_ab-${cc}-${widest}" 24 3 "${lab}/model.bcirq8" \
        | sed "s/^{/{\"run\":${run},\"cc\":\"${cc}\",\"widest\":${widest},/"
    done
  done
done | lines "${here}/switched.json"

# clock.json: bound_clock.c -- the core's cycle time while the 860K decoder's 28 matvecs run through the switched
# unit, back to back (pass) and with a decode's duty cycle (matvec), three runs, each compiler
for cc in clang gcc; do
  "${cc}" "${flags[@]}" -I "${A}" "${here}/bound_clock.c" "${s}/bcir_ai_kernels_switched.c" -o "${s}/bound_clock-${cc}" -lm
done
for run in 1 2 3; do
  for cc in clang gcc; do
    "${s}/bound_clock-${cc}" 24 40 20000 pass | sed "s/^{/{\"run\":${run},\"cc\":\"${cc}\",/"
    "${s}/bound_clock-${cc}" 24 40 280000 matvec | sed "s/^{/{\"run\":${run},\"cc\":\"${cc}\",/"
  done
done | lines "${here}/clock.json"

# decode.json: Q8-LANES' decode_ab.py -- bcir-llama from each tree, each compiler, 24 rounds, on the widest instance
# and with both trees capped at AVX2
llama() {  # tree cc name [define]
  local c="$1/runtime/c"
  "$2" "${flags[@]}" "${@:4}" -I "${c}" "${c}/bcir_ai_kernels.c" "${c}/bcir_q8_model.c" "${c}/bcir_decode.c" \
    "${c}/bcir_llama.c" "${c}/bcir_llama_cli.c" -o "${s}/$3" -lm
}
for cc in clang gcc; do
  for t in before after; do
    llama "${s}/${t}" "${cc}" "llama-${t}-${cc}"
    llama "${s}/${t}" "${cc}" "llama-${t}-avx2-${cc}" -DBCIR_AI_MAX_INSTANCE=2
  done
  python3 "${lanes}/decode_ab.py" 24 "${lab}/model.bcirq8" "${s}/decode-${cc}.json" before="${s}/llama-before-${cc}" \
    after="${s}/llama-after-${cc}" > "${s}/decode-${cc}.log"
  python3 "${lanes}/decode_ab.py" 24 "${lab}/model.bcirq8" "${s}/decode-avx2-${cc}.json" \
    before-avx2="${s}/llama-before-avx2-${cc}" after-avx2="${s}/llama-after-avx2-${cc}" > "${s}/decode-avx2-${cc}.log"
done
python3 -c 'import json, sys; json.dump({f"{cc} {w}": json.load(open(f"{sys.argv[1]}/decode-{x}{cc}.json")) for cc in ("clang", "gcc") for w, x in (("widest", ""), ("avx2", "avx2-"))}, open(sys.argv[2], "w"), indent=1)' \
  "${s}" "${here}/decode.json"

# ops.json: Q8-LANES' llama_ops.py -- each tree's decoder with its operations timed, each compiler, three repeats in
# alternating order, each five generations of 200 new tokens: where a token's time goes after Q8-BOUND
for cc in clang gcc; do
  for rep in 1 2 3; do
    for t in $([ $((rep % 2)) = 1 ] && echo "before after" || echo "after before"); do
      python3 "${lanes}/llama_ops.py" "${s}/${t}" "${lab}/model.bcirq8" "${s}/ops-${t}-${cc}" "${cc}" 5 \
        | sed "s/^{/{\"cc\":\"${cc}\",\"tree\":\"${t}\",\"repeat\":${rep},/"
    done
  done
done | lines "${here}/ops.json"

# inprocess.json: NEXT-9's hoist_ab.c, one process -- the tree before's decoder units, every global renamed before_*,
# beside the tree after's and a second copy of the tree before's renamed before2_* (the placement control, linked
# next to the after arm's units), each compiler, at the qualification's flags (plain) and with every function and
# loop of all four arms aligned to 64 bytes (aligned), three runs
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
    "${cc}" "${flags[@]}" "${extra[@]}" -I "${A}" "${next9}/hoist_ab.c" "${srcs[@]}" "${copies[@]}" \
      -o "${s}/hoist_ab-${cc}-${layout}" -lm
  done
done
for run in 1 2 3; do
  for cc in clang gcc; do
    for layout in plain aligned; do
      "${s}/hoist_ab-${cc}-${layout}" 24 3 "${lab}/model.bcirq8" \
        | sed "s/^{/{\"run\":${run},\"cc\":\"${cc}\",\"layout\":\"${layout}\",/"
    done
  done
done | lines "${here}/inprocess.json"

# qualification.json: run_qualification.py from each tree, its traffic under callgrind (where the AVX2 instance runs)
for t in before after; do
  (cd "${s}/${t}" && python3 tools/models/run_qualification.py --model-dir "${lab}" --output "${s}/qual-${t}.json" \
    --require-callgrind > "${s}/qual-${t}.log")
done
python3 "${lanes}/qual_summary.py" "${s}/qual-before.json" "${s}/qual-after.json" > "${here}/qualification.json"

# qualification-repeats.json: three runs of each tree in alternation without callgrind, each run's C latency rows
for run in 1 2 3; do
  for t in $([ $((run % 2)) = 1 ] && echo "before after" || echo "after before"); do
    (cd "${s}/${t}" && python3 tools/models/run_qualification.py --model-dir "${lab}" --output "${s}/q-${t}-${run}.json" \
      --no-callgrind > /dev/null)
    python3 -c 'import json, sys; d = json.load(open(sys.argv[1])); c = d["latency"]["c"]; print(json.dumps({"tree": sys.argv[2], "run": int(sys.argv[3]), "commit": d["commit"], "status": d["status"], "per_token_ms_median": round(c["per_token_ms"]["median"], 4), "tokens_per_second": round(c["tokens_per_second"], 1), "time_to_first_token_ms": round(c["time_to_first_token_ms"], 3), "monolithic_median_ms": round(c["monolithic_median_ms"], 3), "gem_median_ms": round(c["gem_median_ms"], 3), "gem_over_monolithic_median": round(c["gem_over_monolithic_median"], 4), "matvec_ms": round(c["per_operation_ms"]["dec.matvec"], 3), "matvec_add_ms": round(c["per_operation_ms"]["dec.matvec_add"], 3)}))' \
      "${s}/q-${t}-${run}.json" "${t}" "${run}"
  done
done | lines "${here}/qualification-repeats.json"

# faults.json: q8-bound.json and q8-lanes.json swept on the tree after, then q8-bound.json under the witnesses there
# were before Q8-BOUND (the tree after's kernels, the tree before's test_ai_kernels.c and test_native_ai.py)
for table in q8-bound q8-lanes; do
  (cd "${s}/after" && python3 tools/testing/red_sweep.py --faults "tools/testing/faults/${table}.json" \
    --json-out "${s}/faults-${table}.json" > "${s}/faults-${table}.txt")
done
(cd "${s}/without" && python3 tools/testing/red_sweep.py --faults tools/testing/faults/q8-bound.json \
  --json-out "${s}/faults-without.json" > "${s}/faults-without.txt") || true  # its faults the old witnesses miss
test -s "${s}/faults-without.json"
python3 - "${s}" "${here}/faults.json" "$(command -v python3)" "${repo}" <<'PY'
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
        return v.replace(s, "<" "SCRATCH>/q8-bound").replace(repo, "<" "REPO>")  # split: the run copy's fill keeps them
    return v
json.dump({"q8_bound": scrub(json.load(open(f"{s}/faults-q8-bound.json"))),
           "q8_lanes": scrub(json.load(open(f"{s}/faults-q8-lanes.json"))),
           "q8_bound_under_the_witnesses_before": scrub(json.load(open(f"{s}/faults-without.json")))},
          open(out, "w"), indent=1)
PY

# hashes.json: the tools' sha256 (this directory's and those run from Q8-LANES' and NEXT-9's) and the measured sources'
python3 - "${s}" "${here}" "${lanes}" "${next9}" <<'PY'
import hashlib, json, sys
from pathlib import Path
s, here, lanes, next9 = (Path(a) for a in sys.argv[1:5])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
record = {
    "tools": {t: sha(here / t) for t in ("measure_bound.sh", "bound_switch.py", "bound_kernels.c", "bound_ab.c",
                                         "bound_clock.c")},
    "tools run from ../../2026-10-25/q8-lanes": {t: sha(lanes / t) for t in ("q8_ab.c", "decode_ab.py", "llama_ops.py",
                                                                               "qual_summary.py")},
    "tools run from ../next9": {"hoist_ab.c": sha(next9 / "hoist_ab.c")},
    "sources": {f"{t}/{p}/{u}": sha(s / t / p / u)
                for t in ("before", "after", "without")
                for p, u in (("runtime/c", "bcir_ai_kernels.c"), ("runtime/c", "test_ai_kernels.c"),
                             ("runtime/c", "bcir_q4_kernel.c"), ("runtime/c", "bcir_llama.c"), ("runtime/c", "bcir_decode.c"),
                             ("runtime/c", "bcir_q8_model.c"), ("runtime/c", "bcir_llama_cli.c"),
                             ("bcir/tests", "test_native_ai.py"), ("tools/testing/faults", "q8-bound.json"))
                if (s / t / p / u).exists()},
    "switched/bcir_ai_kernels.c": sha(s / "bcir_ai_kernels_switched.c"),
}
(here / "hashes.json").write_text(json.dumps(record, indent=1) + "\n")
PY
for t in before after without; do git -C "${repo}" worktree remove --force "${s}/${t}"; done
