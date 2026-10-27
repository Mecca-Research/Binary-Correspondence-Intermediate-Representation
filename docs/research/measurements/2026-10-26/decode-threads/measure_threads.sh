#!/usr/bin/env bash
# DECODE-THREADS' records, run from <REPO> on the committed tree -- HEAD DECODE-THREADS with each vector kernel's whole
# call an instance of its own, 4bac89e the tree before (the paste builder's bounds, on HEAD-PACK's records) -- one step
# at a time with nothing else running; <SCRATCH> a scratch directory and <LAB> the qualification's lab model
# (BCIR-Corpus-860K). Writes witness.json, decode.json,
# qualification.json, faults.json and hashes.json beside itself. Q8-LANES' qualification summary
# (../../2026-10-25/q8-lanes/qual_summary.py) runs as committed.
set -euo pipefail
repo="<REPO>"; s="<SCRATCH>/decode-threads"; lab="<LAB>"; before=4bac89e66d708b6963cac110e1fc999372bbb44d
here="${repo}/docs/research/measurements/2026-10-26/decode-threads"
lanes="${repo}/docs/research/measurements/2026-10-25/q8-lanes"
rm -rf "${s:?}"; mkdir -p "${s}"
for t in before after without; do
  git -C "${repo}" worktree add --detach "${s}/${t}" "$([ "${t}" = before ] && echo "${before}" || echo HEAD)" >/dev/null
done
# the vacuity control: the tree after's code under the witnesses there were before DECODE-THREADS
git -C "${s}/without" checkout "${before}" -- runtime/c/test_ai_kernels.c bcir/tests/test_native_ai.py \
  bcir/tests/test_model_weights_io.py bcir/tests/test_decoder_gem.py
A="${s}/after/runtime/c"
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
parts = re.search(r"DECODE-THREADS: (\d+) column parts on a named vector instance equal", text)
row["column_parts"] = int(parts.group(1))
pairs = re.search(r"DECODE-THREADS: (\d+) of them two rows a pass", text)
if pairs:  # the probe builds
    pack = re.search(r"HEAD-PACK: (\d+) packed row dots, (\d+) passes of 32 rows, (\d+) blocks", text)
    pair = re.search(r"Q8-PAIR: (\d+) matvecs two rows a pass, (\d+) in runs", text)
    row.update(column_parts_two_rows_a_pass=int(pairs.group(1)), matvecs_two_rows_a_pass=int(pair.group(1)),
               matvecs_in_runs=int(pair.group(2)), packed_row_dots=int(pack.group(1)))
print(json.dumps(row))
PY
    n=$((n + 1))
  done
done > "${s}/harness.jsonl"
mapfile -t tests < <(python3 -c 'import json, sys; print("\n".join(x for x in json.load(open(sys.argv[1]))["command"] if x.startswith("bcir.tests.") and x.split(":")[1].startswith("test_")))' \
  "${s}/after/tools/testing/faults/decode-threads.json")
(cd "${s}/after" && python3 tools/testing/check_tests.py --require bcir.tests.test_native_ai:_CC \
  --require bcir.tests.test_native_ai:_ASAN "${tests[@]}") > "${s}/witness.txt"
python3 - "${s}/harness.jsonl" "${s}/witness.txt" "${here}/witness.json" "${tests[@]}" <<'PY'
import json, sys
harness = [json.loads(line) for line in open(sys.argv[1])]
assert all(h["all_gates_passed"] for h in harness), harness
witness = {"tests": sys.argv[4:], "required": ["bcir.tests.test_native_ai:_CC", "bcir.tests.test_native_ai:_ASAN"],
           "exit": 0, "output": open(sys.argv[2]).read().strip().splitlines()}
json.dump({"harness": harness, "witness": witness}, open(sys.argv[3], "w"), indent=1)
PY

# decode.json: threads_decode.c, one binary -- the tree after's decoder lent no runner, the trainer's pool at 2, 3
# and 4 threads (the default thresholds) and no runner again (the A/A control) -- each compiler, on the widest
# instance and capped at AVX2, three runs, on the lab model and on synthetic decoders 256, 512 and 1,024 wide and at
# TinyLlama-1.1B's widths (synthetic_model.py; the artifacts themselves are not kept, their sha256 is)
for cc in clang gcc; do
  for widest in 4 2; do
    "${cc}" "${flags[@]}" "-DBCIR_AI_MAX_INSTANCE=${widest}" -I "${A}" "${here}/threads_decode.c" \
      "${A}/bcir_ai_kernels.c" "${A}/bcir_q8_model.c" "${A}/bcir_decode.c" "${A}/bcir_llama.c" "${A}/bcir_tensor.c" \
      -o "${s}/threads_decode-${cc}-${widest}" -lm -pthread
  done
done
python3 "${here}/synthetic_model.py" "${s}/d256.bcirq8" 6 8192 256 8 4 688 > "${s}/model-d256.json"
python3 "${here}/synthetic_model.py" "${s}/d512.bcirq8" 8 16384 512 8 4 1344 > "${s}/model-d512.json"
python3 "${here}/synthetic_model.py" "${s}/d1024.bcirq8" 12 32000 1024 16 4 2816 > "${s}/model-d1024.json"
python3 "${here}/synthetic_model.py" "${s}/tinyllama.bcirq8" 6 32000 2048 32 4 5632 > "${s}/model-tinyllama.json"
for run in 1 2 3; do
  for cc in clang gcc; do
    for widest in 4 2; do
      td="${s}/threads_decode-${cc}-${widest}"
      tag="\"run\":${run},\"cc\":\"${cc}\",\"widest\":${widest}"
      "${td}" 24 3 "${lab}/model.bcirq8" 8 200 64 | sed "s/^{/{${tag},\"model\":\"860K\",/"
      "${td}" 12 1 "${s}/d256.bcirq8" 8 48 64 | sed "s/^{/{${tag},\"model\":\"d256\",/"
      "${td}" 8 1 "${s}/d512.bcirq8" 8 32 64 | sed "s/^{/{${tag},\"model\":\"d512\",/"
      "${td}" 5 1 "${s}/d1024.bcirq8" 4 16 256 | sed "s/^{/{${tag},\"model\":\"d1024\",/"
      "${td}" 5 1 "${s}/tinyllama.bcirq8" 4 16 1024 | sed "s/^{/{${tag},\"model\":\"TinyLlama\",/"
    done
  done
done > "${s}/decode.jsonl"
python3 - "${s}" "${here}/decode.json" <<'PY'
import json, sys
s, out = sys.argv[1:3]
models = {name: json.load(open(f"{s}/model-{name}.json")) for name in ("d256", "d512", "d1024", "tinyllama")}
json.dump({"models": models, "runs": [json.loads(l) for l in open(f"{s}/decode.jsonl")]}, open(out, "w"), indent=1)
PY
for m in d256 d512 d1024 tinyllama; do rm -f "${s:?}/${m}.bcirq8"; done

# qualification.json: run_qualification.py from each tree (its C rail without a runner, as before), its traffic under
# callgrind (where the AVX2 instance runs)
for t in before after; do
  (cd "${s}/${t}" && python3 tools/models/run_qualification.py --model-dir "${lab}" --output "${s}/qual-${t}.json" \
    --require-callgrind > "${s}/qual-${t}.log")
done
python3 "${lanes}/qual_summary.py" "${s}/qual-before.json" "${s}/qual-after.json" > "${here}/qualification.json"

# faults.json: decode-threads.json swept on the tree after, then under the witnesses there were before DECODE-THREADS
# (the tree after's code, the tree before's test_ai_kernels.c, test_native_ai.py, test_model_weights_io.py and
# test_decoder_gem.py): its command reduced to the tests those witnesses had, each fault caught by any of them. Two of
# them cannot build against the tree after, so they are left out: the tree before's standalone parity builds the CLI
# without the pool's unit, and its head witness calls bcir_llama_op_head without the workspace DECODE-THREADS added.
(cd "${s}/after" && python3 tools/testing/red_sweep.py --faults tools/testing/faults/decode-threads.json \
  --json-out "${s}/faults-decode-threads.json" > "${s}/faults-decode-threads.txt")
python3 - "${s}/after/tools/testing/faults/decode-threads.json" "${s}/decode-threads-without.json" <<'PY'
import json, sys
table = json.load(open(sys.argv[1]))
table["command"] = ["@python", "tools/testing/check_tests.py", "--require", "bcir.tests.test_native_ai:_CC",
                    "--require", "bcir.tests.test_native_ai:_ASAN"] + [
    f"bcir.tests.{t}" for t in (
        "test_native_ai:test_native_ai_c_harness_c11_and_c23",
        "test_native_ai:test_native_ai_c_harness_under_address_sanitizer",
        "test_native_ai:test_native_q8_instances_match_the_increasing_coordinate_oracle",
        "test_native_ai:test_native_q8_matrix_kernels_match_increasing_coordinate_oracle_exactly",
        "test_model_weights_io:test_the_decoders_rotation_is_bcir_ropes_rows_bit_for_bit",
        "test_decoder_gem:test_the_c_interpreter_is_the_monolithic_runner_bit_for_bit")]
for fault in table["faults"]:
    fault["expects"] = "test_"
json.dump(table, open(sys.argv[2], "w"), indent=1)
PY
(cd "${s}/without" && python3 tools/testing/red_sweep.py --faults "${s}/decode-threads-without.json" \
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
        return v.replace(s, "<" "SCRATCH>/decode-threads").replace(repo, "<" "REPO>")  # split: the run copy's fill
    return v
json.dump({"decode_threads": scrub(json.load(open(f"{s}/faults-decode-threads.json"))),
           "decode_threads_under_the_witnesses_before": scrub(json.load(open(f"{s}/faults-without.json")))},
          open(out, "w"), indent=1)
PY

# hashes.json: the tools' sha256 (this directory's and Q8-LANES' summary) and the measured sources'
python3 - "${s}" "${here}" "${lanes}" <<'PY'
import hashlib, json, sys
from pathlib import Path
s, here, lanes = (Path(a) for a in sys.argv[1:4])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
record = {
    "tools": {t: sha(here / t) for t in ("measure_threads.sh", "threads_decode.c", "synthetic_model.py")},
    "tools run from ../../2026-10-25/q8-lanes": {"qual_summary.py": sha(lanes / "qual_summary.py")},
    "sources": {f"{t}/{p}/{u}": sha(s / t / p / u)
                for t in ("before", "after", "without")
                for p, u in (("runtime/c", "bcir_ai_kernels.c"), ("runtime/c", "bcir_ai_kernels.h"),
                             ("runtime/c", "test_ai_kernels.c"), ("runtime/c", "bcir_llama.c"),
                             ("runtime/c", "bcir_llama.h"), ("runtime/c", "bcir_llama_ops.h"),
                             ("runtime/c", "bcir_llama_cli.c"), ("runtime/c", "bcir_decoder_gem.c"),
                             ("runtime/c", "bcir_tensor.c"), ("bcir/tests", "test_native_ai.py"),
                             ("bcir/tests", "test_model_weights_io.py"), ("bcir/tests", "test_decoder_gem.py"),
                             ("bcir/kbcir", "native_ai.py"), ("tools/testing/faults", "decode-threads.json"))
                if (s / t / p / u).exists()},
}
json.dump(record, open(here / "hashes.json", "w"), indent=1)
PY
for t in before after without; do git -C "${repo}" worktree remove --force "${s}/${t}"; done
git -C "${repo}" worktree prune
echo "records written"
