#!/usr/bin/env bash
# HEAD-PACK's records, run from <REPO> on the committed tree -- HEAD HEAD-PACK, HEAD~1 the tree before (Q8-PAIR's
# records) -- one step at a time with nothing else running; <SCRATCH> a scratch directory and <LAB> the
# qualification's lab model (BCIR-Corpus-860K). Writes witness.json, kernels.json, switched.json, big.json,
# decode.json, ops.json, qualification.json, qualification-repeats.json, faults.json and hashes.json beside itself.
# Q8-LANES' decode A/B, operation timer and qualification summary (../../2026-10-25/q8-lanes/decode_ab.py,
# llama_ops.py and qual_summary.py) and Q8-PAIR's synthetic decoder (../q8-pair/big_model.py) run as committed.
set -euo pipefail
repo="<REPO>"; s="<SCRATCH>/head-pack"; lab="<LAB>"
here="${repo}/docs/research/measurements/2026-10-26/head-pack"
lanes="${repo}/docs/research/measurements/2026-10-25/q8-lanes"
pair="${repo}/docs/research/measurements/2026-10-26/q8-pair"
rm -rf "${s}"; mkdir -p "${s}"
for t in before after without; do
  git -C "${repo}" worktree add --detach "${s}/${t}" "$([ "${t}" = before ] && echo HEAD~1 || echo HEAD)" >/dev/null
done
# the vacuity control: the tree after's code under the witnesses there were before HEAD-PACK
git -C "${s}/without" checkout HEAD~1 -- runtime/c/test_ai_kernels.c runtime/c/test_memory_discipline.c \
  bcir/tests/test_native_ai.py bcir/tests/test_model_weights_io.py
B="${s}/before/runtime/c"; A="${s}/after/runtime/c"
flags=(-std=c11 -O2 -ffp-contract=off -Wall -Wextra -Werror)
lines() { python3 -c 'import json, sys; json.dump([json.loads(l) for l in sys.stdin], open(sys.argv[1], "w"), indent=1)' "$1"; }

# witness.json: the tree after's kernel harness (test_ai_kernels.c) in the four builds
# test_native_ai_c_harness_c11_and_c23 makes (C11; C23; C23 capped at AVX2; C23 with the dispatch compiled out; each
# with the instance probe), without the probe, and in the build test_native_ai_c_harness_under_address_sanitizer makes,
# each compiler, its summary lines; then the fault table's command with the compiler and the sanitizer required
for cc in clang gcc; do
  n=0
  for build in "c11" "c2x" "c2x -DBCIR_AI_MAX_INSTANCE=2" "c2x -DBCIR_AI_NO_DISPATCH" "c11 noprobe" "c11 asan"; do
    read -r std opt <<< "${build}"
    exe="${s}/ai-kernels-${cc}-${n}"
    if [ "${opt}" = asan ]; then
      "${cc}" "-std=${std}" -DBCIR_AI_INSTANCE_PROBE -O1 -g -fno-omit-frame-pointer -ffp-contract=off \
        -fsanitize=address,undefined -fno-sanitize-recover=all \
        -I "${A}" "${A}/test_ai_kernels.c" "${A}/bcir_ai_kernels.c" "${A}/bcir_q4_kernel.c" -o "${exe}" -lm
      ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 "${exe}" > "${exe}.txt"
    elif [ "${opt}" = noprobe ]; then
      "${cc}" "-std=${std}" -O2 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off \
        -I "${A}" "${A}/test_ai_kernels.c" "${A}/bcir_ai_kernels.c" "${A}/bcir_q4_kernel.c" -o "${exe}" -lm
      "${exe}" > "${exe}.txt"
    else
      # shellcheck disable=SC2086  # opt is one option or none
      "${cc}" "-std=${std}" -DBCIR_AI_INSTANCE_PROBE ${opt} -O2 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off \
        -I "${A}" "${A}/test_ai_kernels.c" "${A}/bcir_ai_kernels.c" "${A}/bcir_q4_kernel.c" -o "${exe}" -lm
      "${exe}" > "${exe}.txt"
    fi
    python3 - "${exe}.txt" "${cc}" "${build}" <<'PY'
import json, re, sys
text = open(sys.argv[1]).read()
std, _, opt = sys.argv[3].partition(" ")
row = {"cc": sys.argv[2],
       "build": "-std=" + std + {"asan": " -fsanitize=address,undefined", "noprobe": " without the probe"}.get(
           opt, " " + opt if opt else ""),
       "all_gates_passed": "native AI kernels: all gates passed" in text}
lanes = re.search(r"Q8-LANES: instances (\d+), (\d+) vector comparisons equal", text)
row["instances"], row["vector_comparisons"] = int(lanes.group(1)), int(lanes.group(2))
pack = re.search(r"HEAD-PACK: (\d+) packed row dots, (\d+) passes of 32 rows, (\d+) blocks", text)
if pack:  # the probe builds
    pair = re.search(r"Q8-PAIR: (\d+) matvecs two rows a pass, (\d+) in runs", text)
    bound = re.search(r"Q8-BOUND: (\d+) matvecs hoisted, (\d+) computed again", text)
    row.update(matvecs_hoisted=int(bound.group(1)), matvecs_computed_again=int(bound.group(2)),
               matvecs_two_rows_a_pass=int(pair.group(1)), matvecs_in_runs=int(pair.group(2)),
               packed_row_dots=int(pack.group(1)), passes_of_32_rows=int(pack.group(2)), blocks=int(pack.group(3)))
print(json.dumps(row))
PY
    n=$((n + 1))
  done
done > "${s}/harness.jsonl"
mapfile -t tests < <(python3 -c 'import json, sys; print("\n".join(x for x in json.load(open(sys.argv[1]))["command"] if x.startswith("bcir.tests.") and x.split(":")[1].startswith("test_")))' \
  "${s}/after/tools/testing/faults/head-pack.json")
(cd "${s}/after" && python3 tools/testing/check_tests.py --require bcir.tests.test_native_ai:_CC \
  --require bcir.tests.test_native_ai:_ASAN "${tests[@]}") > "${s}/witness.txt"
(cd "${s}/after" && MEMORY_DISCIPLINE_SANITIZE=1 bash tools/c/check_memory_discipline.sh) > "${s}/memory.txt" 2>&1
python3 - "${s}/harness.jsonl" "${s}/witness.txt" "${s}/memory.txt" "${here}/witness.json" "${tests[@]}" <<'PY'
import json, sys
harness = [json.loads(line) for line in open(sys.argv[1])]
assert all(h["all_gates_passed"] for h in harness), harness
memory = open(sys.argv[3]).read()
assert "memory-discipline: all gates passed" in memory, memory[-2000:]
witness = {"tests": sys.argv[5:], "required": ["bcir.tests.test_native_ai:_CC", "bcir.tests.test_native_ai:_ASAN"],
           "exit": 0, "output": open(sys.argv[2]).read().strip().splitlines(),
           "memory_discipline": [l for l in memory.splitlines() if l.startswith(("memory-discipline", "native AI"))]}
json.dump({"harness": harness, "witness": witness}, open(sys.argv[4], "w"), indent=1)
PY

# kernels.json: headpack_kernels.c, one binary -- each vector instance's row dots over the rows as they are against
# the same rows packed: the 860K decoder's head (in L2), a 4 MiB head (in L3), a GPT-2-sized head and TinyLlama-1.1B's
# (each one head again and again, from L3, and a sequence of heads past L3, each from memory), three runs, each compiler
for cc in clang gcc; do
  "${cc}" "${flags[@]}" -DBCIR_AI_INSTANCE_PROBE -I "${A}" "${here}/headpack_kernels.c" "${A}/bcir_ai_kernels.c" \
    -o "${s}/headpack_kernels-${cc}" -lm
done
for run in 1 2 3; do
  for cc in clang gcc; do
    k="${s}/headpack_kernels-${cc}"
    "${k}" 15 128,1024,32,1,32 | sed "s/^{/{\"run\":${run},\"cc\":\"${cc}\",\"head\":\"860K\",\"from\":\"L2\",/"
    "${k}" 9 512,8192,32,1,1 | sed "s/^{/{\"run\":${run},\"cc\":\"${cc}\",\"head\":\"4 MiB\",\"from\":\"L3\",/"
    "${k}" 9 768,50257,32,1,1 | sed "s/^{/{\"run\":${run},\"cc\":\"${cc}\",\"head\":\"GPT-2\",\"from\":\"L3\",/"
    "${k}" 5 768,50257,32,10,1 | sed "s/^{/{\"run\":${run},\"cc\":\"${cc}\",\"head\":\"GPT-2\",\"from\":\"DRAM\",/"
    "${k}" 7 2048,32000,32,1,1 | sed "s/^{/{\"run\":${run},\"cc\":\"${cc}\",\"head\":\"TinyLlama\",\"from\":\"L3\",/"
    "${k}" 5 2048,32000,32,6,1 | sed "s/^{/{\"run\":${run},\"cc\":\"${cc}\",\"head\":\"TinyLlama\",\"from\":\"DRAM\",/"
  done
done | lines "${here}/kernels.json"

# switched.json: headpack_decode.c, one binary -- the tree after's decoder on the 860K decoder, its head unpacked,
# packed and unpacked again (the A/A control), on the widest instance and capped at AVX2, 200-token generations, three
# runs, each compiler
for cc in clang gcc; do
  for widest in 4 2; do
    "${cc}" "${flags[@]}" -DBCIR_AI_INSTANCE_PROBE "-DBCIR_AI_MAX_INSTANCE=${widest}" -I "${A}" \
      "${here}/headpack_decode.c" "${A}/bcir_ai_kernels.c" "${A}/bcir_q8_model.c" "${A}/bcir_decode.c" \
      "${A}/bcir_llama.c" -o "${s}/headpack_decode-${cc}-${widest}" -lm
  done
done
for run in 1 2 3; do
  for cc in clang gcc; do
    for widest in 4 2; do
      "${s}/headpack_decode-${cc}-${widest}" 24 3 "${lab}/model.bcirq8" 8 200 64 \
        | sed "s/^{/{\"run\":${run},\"cc\":\"${cc}\",\"widest\":${widest},/"
    done
  done
done | lines "${here}/switched.json"

# big.json: Q8-PAIR's synthetic decoder at TinyLlama-1.1B's widths, six layers (its weights past this host's L3), and
# headpack_decode.c on it: 4- and 16-token generations, five rounds, three runs, each compiler, widest and capped at
# AVX2; the artifact itself is not kept
python3 "${pair}/big_model.py" "${s}/big.bcirq8" 6 > "${s}/big-model.json"
for run in 1 2 3; do
  for cc in clang gcc; do
    for widest in 4 2; do
      "${s}/headpack_decode-${cc}-${widest}" 5 1 "${s}/big.bcirq8" 4 16 1024 \
        | sed "s/^{/{\"run\":${run},\"cc\":\"${cc}\",\"widest\":${widest},/"
    done
  done
done > "${s}/big.jsonl"
python3 - "${s}/big-model.json" "${s}/big.jsonl" "${here}/big.json" <<'PY'
import json, sys
json.dump({"model": json.load(open(sys.argv[1])), "runs": [json.loads(l) for l in open(sys.argv[2])]},
          open(sys.argv[3], "w"), indent=1)
PY
rm -f "${s}/big.bcirq8"

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
# alternating order, each five generations of 200 new tokens: where a token's time goes after HEAD-PACK
for cc in clang gcc; do
  for rep in 1 2 3; do
    for t in $([ $((rep % 2)) = 1 ] && echo "before after" || echo "after before"); do
      python3 "${lanes}/llama_ops.py" "${s}/${t}" "${lab}/model.bcirq8" "${s}/ops-${t}-${cc}" "${cc}" 5 \
        | sed "s/^{/{\"cc\":\"${cc}\",\"tree\":\"${t}\",\"repeat\":${rep},/"
    done
  done
done | lines "${here}/ops.json"

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
    python3 -c 'import json, sys; d = json.load(open(sys.argv[1])); c = d["latency"]["c"]; o = c["per_operation_ms"]; print(json.dumps({"tree": sys.argv[2], "run": int(sys.argv[3]), "commit": d["commit"], "status": d["status"], "per_token_ms_median": round(c["per_token_ms"]["median"], 4), "tokens_per_second": round(c["tokens_per_second"], 1), "time_to_first_token_ms": round(c["time_to_first_token_ms"], 3), "monolithic_median_ms": round(c["monolithic_median_ms"], 3), "gem_median_ms": round(c["gem_median_ms"], 3), "gem_over_monolithic_median": round(c["gem_over_monolithic_median"], 4), "head_ms": round(o["dec.head"], 3), "matvec_ms": round(o["dec.matvec"], 3), "matvec_add_ms": round(o["dec.matvec_add"], 3)}))' \
      "${s}/q-${t}-${run}.json" "${t}" "${run}"
  done
done | lines "${here}/qualification-repeats.json"

# faults.json: head-pack.json, q8-pair.json and q8-lanes.json swept on the tree after, then head-pack.json under the
# witnesses there were before HEAD-PACK (the tree after's code, the tree before's test_ai_kernels.c,
# test_memory_discipline.c, test_native_ai.py and test_model_weights_io.py): its command reduced to the tests those
# witnesses had, each fault caught by any of them
for table in head-pack q8-pair q8-lanes; do
  (cd "${s}/after" && python3 tools/testing/red_sweep.py --faults "tools/testing/faults/${table}.json" \
    --json-out "${s}/faults-${table}.json" > "${s}/faults-${table}.txt")
done
python3 - "${s}/after/tools/testing/faults/head-pack.json" "${s}/head-pack-without.json" <<'PY'
import json, sys
table = json.load(open(sys.argv[1]))
table["command"] = ["@python", "tools/testing/check_tests.py", "--require", "bcir.tests.test_native_ai:_CC",
                    "--require", "bcir.tests.test_native_ai:_ASAN"] + [
    f"bcir.tests.{t}" for t in (
        "test_native_ai:test_native_ai_c_harness_c11_and_c23",
        "test_native_ai:test_native_ai_c_harness_under_address_sanitizer",
        "test_native_ai:test_native_q8_instances_match_the_increasing_coordinate_oracle",
        "test_native_ai:test_native_q8_matrix_kernels_match_increasing_coordinate_oracle_exactly",
        "test_model_weights_io:test_standalone_c_q8_decoder_matches_python_for_tied_and_untied_heads",
        "test_model_weights_io:test_the_decoders_rotation_is_bcir_ropes_rows_bit_for_bit")]
for fault in table["faults"]:
    fault["expects"] = "test_"
json.dump(table, open(sys.argv[2], "w"), indent=1)
PY
(cd "${s}/without" && python3 tools/testing/red_sweep.py --faults "${s}/head-pack-without.json" \
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
        return v.replace(s, "<" "SCRATCH>/head-pack").replace(repo, "<" "REPO>")  # split: the run copy's fill keeps them
    return v
json.dump({"head_pack": scrub(json.load(open(f"{s}/faults-head-pack.json"))),
           "q8_pair": scrub(json.load(open(f"{s}/faults-q8-pair.json"))),
           "q8_lanes": scrub(json.load(open(f"{s}/faults-q8-lanes.json"))),
           "head_pack_under_the_witnesses_before": scrub(json.load(open(f"{s}/faults-without.json")))},
          open(out, "w"), indent=1)
PY

# hashes.json: the tools' sha256 (this directory's and those run from Q8-LANES' and Q8-PAIR's) and the measured sources'
python3 - "${s}" "${here}" "${lanes}" "${pair}" <<'PY'
import hashlib, json, sys
from pathlib import Path
s, here, lanes, pair = (Path(a) for a in sys.argv[1:5])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
record = {
    "tools": {t: sha(here / t) for t in ("measure_headpack.sh", "headpack_kernels.c", "headpack_decode.c")},
    "tools run from ../../2026-10-25/q8-lanes": {t: sha(lanes / t) for t in ("decode_ab.py", "llama_ops.py",
                                                                               "qual_summary.py")},
    "tools run from ../q8-pair": {"big_model.py": sha(pair / "big_model.py")},
    "sources": {f"{t}/{p}/{u}": sha(s / t / p / u)
                for t in ("before", "after", "without")
                for p, u in (("runtime/c", "bcir_ai_kernels.c"), ("runtime/c", "bcir_ai_kernels.h"),
                             ("runtime/c", "test_ai_kernels.c"), ("runtime/c", "bcir_q4_kernel.c"),
                             ("runtime/c", "bcir_llama.c"), ("runtime/c", "bcir_decode.c"),
                             ("runtime/c", "bcir_q8_model.c"), ("runtime/c", "bcir_q8_model.h"),
                             ("runtime/c", "bcir_llama_cli.c"), ("runtime/c", "test_memory_discipline.c"),
                             ("bcir/tests", "test_native_ai.py"), ("bcir/tests", "test_model_weights_io.py"),
                             ("bcir/kbcir", "native_ai.py"), ("tools/testing/faults", "head-pack.json"))
                if (s / t / p / u).exists()},
}
(here / "hashes.json").write_text(json.dumps(record, indent=1) + "\n")
PY
for t in before after without; do git -C "${repo}" worktree remove --force "${s}/${t}"; done
