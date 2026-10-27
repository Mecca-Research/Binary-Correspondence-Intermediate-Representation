#!/usr/bin/env bash
# KERNEL-ALIGN's records, run from <REPO> on the committed tree -- HEAD KERNEL-ALIGN (the decoder's units, bcir_models,
# built with the JCC-erratum mitigation on x86-64 from the manifest, every build that times them reading it through
# bcir.toolchain.unit_options) and dbe444c the tree before -- one step at a time with nothing else running; <SCRATCH> a
# scratch directory and <LAB> the qualification's lab model (BCIR-Corpus-860K). The slice moves no source of the
# kernels: each tree's units compiled as its manifest gives them is where the two differ. Writes placement.json,
# decode.json, gemm.json, code.json, qualification.json, qualification-repeats.json, witness.json, faults.json and
# hashes.json beside itself.
set -euo pipefail
repo="<REPO>"; s="<SCRATCH>/kernel-align"; lab="<LAB>"
before=dbe444c7b23b751f4b1faf4ba0971f902080e781
here="${repo}/docs/research/measurements/2026-10-27/kernel-align"
threads="${repo}/docs/research/measurements/2026-10-26/decode-threads"  # synthetic_model.py
mmrows="${repo}/docs/research/measurements/2026-10-23/mm-rows"         # ab_gemm.c
lanes="${repo}/docs/research/measurements/2026-10-25/q8-lanes"         # qual_summary.py
rm -rf "${s:?}"; mkdir -p "${s}"
for t in before after without; do
  git -C "${repo}" worktree add --detach "${s}/${t}" "$([ "${t}" = before ] && echo "${before}" || echo HEAD)" >/dev/null
done
# the vacuity control: the tree after's code under the witnesses there were before KERNEL-ALIGN
git -C "${s}/without" checkout "${before}" -- bcir/tests/test_toolchain.py bcir/tests/test_build_manifest.py \
  bcir/tests/test_make_rails.py bcir/tests/test_native_ai.py bcir/tests/test_qualification.py
B="${s}/before"; A="${s}/after"
for u in bcir_ai_kernels.c bcir_q8_model.c bcir_llama.c bcir_decode.c bcir_tensor.c; do
  cmp "${B}/runtime/c/${u}" "${A}/runtime/c/${u}"  # the slice moves no source of the units it times
done
lines() { python3 -c 'import json, sys; json.dump([json.loads(l) for l in sys.stdin], open(sys.argv[1], "w"), indent=1)' "$1"; }
scrub() {  # the records' strings carry no host paths: the scratch, the lab and the repository named
  sed -e "s#${s}#<""SCRATCH>/kernel-align#g" -e "s#${lab}#<""LAB>#g" -e "s#${repo}#<""REPO>#g"
}
# the mitigation as the tree after's manifest gives the kernel unit under a compiler (non-empty on x86-64)
mitigation() {
  python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); from bcir.toolchain import unit_options; print(" ".join(f for f in unit_options(["bcir_ai_kernels.c"], sys.argv[2]) if "32B" in f))' \
    "${A}" "$1"
}

# placement.json: align_ab over copies of the kernel unit in one binary, each compiler: the tree before's build (its
# manifest's options) at the four placements its 16-byte alignment admits, the tree after's (its manifest's: the
# mitigation) at the two its 32-byte alignment admits, and the alternatives measured beside them -- loops aligned to
# 32 and 64 bytes, every function to 64, loops to 32 with the mitigation -- each at every placement its alignment
# admits; on the widest instance, AVX2 and the portable loops, 101 rounds
for cc in clang gcc; do
  m="$(mitigation "${cc}")"; test -n "${m}"
  python3 "${here}/build_align.py" "${A}/runtime/c/bcir_ai_kernels.c" "${cc}" "${s}/align-${cc}" \
    d0:0 d16:16 d32:32 d48:48 m0:0:"${m}" m32:32:"${m}" l32a:0:-falign-loops=32 l32b:32:-falign-loops=32 \
    l64:0:-falign-loops=64 a64:0:-falign-functions=64 lm0:0:"-falign-loops=32 ${m}" lm32:32:"-falign-loops=32 ${m}" |
    sed "s/^{/{\"cc\":\"${cc}\",/" > "${s}/align-build-${cc}.jsonl"
  for i in 4 2 1; do "${s}/align-${cc}" "${i}" 101 | sed "s/^{/{\"cc\":\"${cc}\",/"; done > "${s}/align-runs-${cc}.jsonl"
done
cat "${s}"/align-build-*.jsonl | lines "${s}/align-build.json"
cat "${s}"/align-runs-*.jsonl | lines "${s}/align-runs.json"
python3 -c 'import json, sys; json.dump({"builds": json.load(open(sys.argv[1])), "runs": json.load(open(sys.argv[2]))}, open(sys.argv[3], "w"), indent=1)' \
  "${s}/align-build.json" "${s}/align-runs.json" "${here}/placement.json"

# code.json: each source of bcir_models compiled as each tree's manifest gives it (its text's size and alignment, the
# kernel unit's hot kernels' NOPs and padding), and the branches of every hot loop of every copy placement.json timed
for cc in clang gcc; do
  python3 "${here}/kernel_code.py" objects "${cc}" "${B}" "${A}" "${s}/objects-${cc}" | sed "s/^{/{\"cc\":\"${cc}\",/"
done > "${s}/code-objects.jsonl"
for cc in clang gcc; do
  python3 "${here}/kernel_code.py" layout "${s}/align-${cc}" d0 d16 d32 d48 m0 m32 l32a l32b l64 a64 lm0 lm32 |
    sed "s/^{/{\"cc\":\"${cc}\",/"
done > "${s}/code-layout.jsonl"
python3 -c 'import json, sys; json.dump({"objects": [json.loads(l) for l in open(sys.argv[1])], "layout": [json.loads(l) for l in open(sys.argv[2])]}, open(sys.argv[3], "w"), indent=1)' \
  "${s}/code-objects.jsonl" "${s}/code-layout.jsonl" "${here}/code.json"

# decode.json: decode_place over copies of the decoder's four units in one binary, each compiler -- the tree before's
# build at its four placements, the tree after's at its two, and the tree after's kernel unit alone with the tree
# before's other units at two -- on the lab model and two synthetic decoders (DECODE-THREADS' synthetic_model.py; the
# artifacts are not kept, their sha256 is), three runs a model, 31 rounds
python3 "${threads}/synthetic_model.py" "${s}/d256.bcirq8" 6 8192 256 8 4 688 > "${s}/model-d256.json"
python3 "${threads}/synthetic_model.py" "${s}/d512.bcirq8" 8 16384 512 8 4 1344 > "${s}/model-d512.json"
for cc in clang gcc; do
  python3 "${here}/build_decode.py" "${B}" "${A}" "${cc}" "${s}/decode-${cc}" D0:0:before D16:16:before \
    D32:32:before D48:48:before M0:0:after M32:32:after K0:0:kernels K32:32:kernels |
    sed "s/^{/{\"cc\":\"${cc}\",/" > "${s}/decode-build-${cc}.jsonl"
  for run in 1 2 3; do
    for model in "${lab}/model.bcirq8:4:200" "${s}/d256.bcirq8:2:72" "${s}/d512.bcirq8:1:40"; do
      IFS=: read -r path reps long <<< "${model}"
      "${s}/decode-${cc}" 31 "${reps}" "${path}" 8 "${long}" 64 | sed "s/^{/{\"cc\":\"${cc}\",\"run\":${run},/"
    done
  done > "${s}/decode-runs-${cc}.jsonl"
done
cat "${s}"/decode-build-*.jsonl | lines "${s}/decode-build.json"
cat "${s}"/decode-runs-*.jsonl | scrub | lines "${s}/decode-runs.json"
python3 -c 'import json, sys; json.dump({"builds": json.load(open(sys.argv[1])), "runs": json.load(open(sys.argv[2])), "synthetic_models": {"d256": json.load(open(sys.argv[3])), "d512": json.load(open(sys.argv[4]))}}, open(sys.argv[5], "w"), indent=1)' \
  "${s}/decode-build.json" "${s}/decode-runs.json" "${s}/model-d256.json" "${s}/model-d512.json" "${here}/decode.json"

# gemm.json: the trainer's GEMM (bcir_tensor.c, a unit the slice leaves as it was) as a shared library at the four
# placements its build admits and with the mitigation at two, MM-ROWS' ab_gemm.c (as committed) against the first
# placement in one process, one thread: four shapes in cache, 48 rounds of 20 calls -- whether the unit the slice
# leaves alone is placement-bound too
gcc -std=c11 -O2 -Wall -Wextra -o "${s}/ab_gemm" "${mmrows}/ab_gemm.c" -ldl -lm
for cc in clang gcc; do
  m="$(mitigation "${cc}")"
  for p in 0 16 32 48; do
    python3 "${here}/build_so.py" "${A}/runtime/c/bcir_tensor.c" "${cc}" "${s}/t-${cc}-d${p}.so" "${p}" >> "${s}/gemm-build.txt"
  done
  for p in 0 32; do
    python3 "${here}/build_so.py" "${A}/runtime/c/bcir_tensor.c" "${cc}" "${s}/t-${cc}-m${p}.so" "${p}" ${m} >> "${s}/gemm-build.txt"
  done
  for v in d16 d32 d48 m0 m32; do
    "${s}/ab_gemm" "${s}/t-${cc}-d0.so" "${s}/t-${cc}-${v}.so" 48 1 20 g:0,0,512,128,128 g:1,0,128,128,512 \
      g:0,1,512,128,128 g:0,0,256,64,256 | sed "s/^{/{\"cc\":\"${cc}\",\"base_arm\":\"d0\",\"proto_arm\":\"${v}\",/"
  done
done > "${s}/gemm.jsonl"
python3 -c 'import json, sys; json.dump({"builds": open(sys.argv[1]).read().splitlines(), "runs": [json.loads(l) for l in open(sys.argv[2])]}, open(sys.argv[3], "w"), indent=1)' \
  "${s}/gemm-build.txt" "${s}/gemm.jsonl" "${here}/gemm.json"

# qualification.json and qualification-repeats.json: the qualification from each tree -- its bcir-qualify built as
# that tree builds it -- under callgrind once (its summary by Q8-LANES' qual_summary.py), then six runs a tree without
# callgrind, the order rotated a run, each run's C latency rows
for t in before after; do
  (cd "${s}/${t}" && python3 tools/models/run_qualification.py --model-dir "${lab}" --output "${s}/q-${t}.json" \
    --require-callgrind > /dev/null)
done
python3 "${lanes}/qual_summary.py" "${s}/q-before.json" "${s}/q-after.json" | scrub > "${here}/qualification.json"
for run in 1 2 3 4 5 6; do
  if [ $((run % 2)) = 1 ]; then order="before after"; else order="after before"; fi
  for t in ${order}; do
    (cd "${s}/${t}" && python3 tools/models/run_qualification.py --model-dir "${lab}" \
      --output "${s}/q-${t}-${run}.json" --no-callgrind > /dev/null)
    python3 -c 'import json, sys; d = json.load(open(sys.argv[1])); c = d["latency"]["c"]; o = c["per_operation_ms"]; print(json.dumps({"tree": sys.argv[2], "run": int(sys.argv[3]), "commit": d["commit"], "status": d["status"], "per_token_ms_median": round(c["per_token_ms"]["median"], 4), "tokens_per_second": round(c["tokens_per_second"], 1), "monolithic_median_ms": round(c["monolithic_median_ms"], 3), "gem_median_ms": round(c["gem_median_ms"], 3), "head_ms": round(o["dec.head"], 3), "matvec_ms": round(o["dec.matvec"], 3), "matvec_add_ms": round(o["dec.matvec_add"], 3)}))' \
      "${s}/q-${t}-${run}.json" "${t}" "${run}"
  done
done | lines "${here}/qualification-repeats.json"

# witness.json: the fault table's command on the tree after (every witness passing, its capabilities required)
mapfile -t cmd < <(python3 -c 'import json, sys; print("\n".join(json.load(open(sys.argv[1]))["command"][1:]))' \
  "${A}/tools/testing/faults/kernel-align.json")
(cd "${A}" && python3 "${cmd[@]}") > "${s}/witness.txt"
python3 -c 'import json, sys; json.dump({"command": ["@python", *sys.argv[2:-1]], "exit": 0, "output": open(sys.argv[1]).read().strip().splitlines()}, open(sys.argv[-1], "w"), indent=1)' \
  "${s}/witness.txt" "${cmd[@]}" "${s}/witness.json"
scrub < "${s}/witness.json" > "${here}/witness.json"

# faults.json: kernel-align.json swept on the tree after, then under the witnesses there were before KERNEL-ALIGN
# (the tree after's code, the tree before's test_toolchain.py, test_build_manifest.py, test_make_rails.py,
# test_native_ai.py and test_qualification.py): its command reduced to the tests those witnesses had that read what
# the faults change -- the manifest's reconciliation, its injected violations and its reading of a build's option, the
# rails' plan, the compiler pick and the native AI library's build and stamp (the qualification's builder test needs
# the thorough tier's model, so none of its) -- each fault caught by any of them
(cd "${A}" && python3 tools/testing/red_sweep.py --faults tools/testing/faults/kernel-align.json \
  --json-out "${s}/faults-kernel-align.json" > "${s}/faults-kernel-align.txt")
python3 - "${A}/tools/testing/faults/kernel-align.json" "${s}/kernel-align-without.json" <<'PY'
import json, sys
table = json.load(open(sys.argv[1]))
table["command"] = ["@python", "tools/testing/check_tests.py", "--require", "bcir.tests.test_native_ai:_CC"] + [
    f"bcir.tests.{t}" for t in (
        "test_build_manifest:test_the_manifest_reconciles_with_the_tree",
        "test_build_manifest:test_every_injected_violation_is_a_finding",
        "test_build_manifest:test_a_build_names_the_option_beside_the_source",
        "test_make_rails:test_the_rails_build_plans_the_manifests_targets",
        "test_toolchain:test_the_compiler_pick_honours_cc_and_never_substitutes",
        "test_native_ai:test_native_build_reuses_the_library_when_nothing_changed",
        "test_native_ai:test_native_build_recompiles_when_the_recorded_inputs_do_not_match",
        "test_native_ai:test_native_build_stamp_covers_every_translation_unit_input")]
for fault in table["faults"]:
    fault["expects"] = "test_"
json.dump(table, open(sys.argv[2], "w"), indent=1)
PY
(cd "${s}/without" && python3 tools/testing/red_sweep.py --faults "${s}/kernel-align-without.json" \
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
        return v.replace(s, "<" "SCRATCH>/kernel-align").replace(repo, "<" "REPO>")  # split: the run copy's fill
    return v
json.dump({"kernel_align": scrub(json.load(open(f"{s}/faults-kernel-align.json"))),
           "kernel_align_under_the_witnesses_before": scrub(json.load(open(f"{s}/faults-without.json")))},
          open(out, "w"), indent=1)
PY

# hashes.json: the tools' sha256 (this directory's and those it runs from other records) and the measured sources'
python3 - "${s}" "${here}" "${threads}" "${mmrows}" "${lanes}" <<'PY'
import hashlib, json, sys
from pathlib import Path
s, here, threads, mmrows, lanes = (Path(a) for a in sys.argv[1:6])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
record = {
    "tools": {t: sha(here / t) for t in ("measure_align.sh", "align_ab.c", "build_align.py", "decode_place.c",
                                        "build_decode.py", "build_so.py", "kernel_code.py")},
    "tools run from other records": {"../../2026-10-26/decode-threads/synthetic_model.py": sha(threads / "synthetic_model.py"),
                                     "../../2026-10-23/mm-rows/ab_gemm.c": sha(mmrows / "ab_gemm.c"),
                                     "../../2026-10-25/q8-lanes/qual_summary.py": sha(lanes / "qual_summary.py")},
    "synthetic_models": {m: sha(s / f"{m}.bcirq8") for m in ("d256", "d512")},
    "sources": {f"{t}/{p}/{u}": sha(s / t / p / u)
                for t in ("before", "after", "without")
                for p, u in (("runtime/c", "bcir_ai_kernels.c"), ("runtime/c", "bcir_q8_model.c"),
                             ("runtime/c", "bcir_llama.c"), ("runtime/c", "bcir_decode.c"), ("runtime/c", "bcir_tensor.c"),
                             ("runtime", "manifest.json"), ("bcir", "toolchain.py"), ("cmake", "BCIRManifest.cmake"),
                             ("tools/build", "manifest.py"), ("tools/build", "bcirfile.py"),
                             ("bcir/kbcir", "native_ai.py"), ("bcir/frontends/models", "model_microbench.py"),
                             ("tools/models", "run_qualification.py"), ("tools/testing/faults", "kernel-align.json"))
                if (s / t / p / u).exists()},
}
json.dump(record, open(here / "hashes.json", "w"), indent=1)
PY
for t in before after without; do git -C "${repo}" worktree remove --force "${s}/${t}"; done
git -C "${repo}" worktree prune
echo "records written"
