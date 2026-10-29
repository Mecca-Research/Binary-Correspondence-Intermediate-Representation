#!/usr/bin/env bash
# KERNEL-ALIGN's longer decode pass, run after measure_align.sh from <REPO> on the committed tree -- HEAD KERNEL-ALIGN
# and dbe444c the tree before -- one step at a time with nothing else running; <SCRATCH> a scratch directory and <LAB>
# the qualification's lab model (BCIR-Corpus-860K). decode.json's runs, 31 rounds each, give a copy's token time a 95%
# interval about x0.92-1.08 wide, as wide as the placement effect they measure; this pass repeats their design at 101
# rounds a run on the two decoders whose token costs least -- the lab model and DECODE-THREADS' d256 synthetic decoder
# (the artifact not kept, its sha256 is) -- three runs a model: decode_place over copies of the decoder's four units in
# one binary, each compiler, built by build_decode.py with decode.json's arms (the tree before's build at its four
# placements, the tree after's at its two, the tree after's kernel unit alone with the tree before's other units at
# two). Writes decode-long.json and hashes-decode-long.json beside itself.
set -euo pipefail
repo="<REPO>"; s="<SCRATCH>/kernel-align-decode-long"; lab="<LAB>"
before=dbe444c7b23b751f4b1faf4ba0971f902080e781
here="${repo}/docs/research/measurements/2026-10-27/kernel-align"
threads="${repo}/docs/research/measurements/2026-10-26/decode-threads"  # synthetic_model.py
rm -rf "${s:?}"; mkdir -p "${s}"
for t in before after; do
  git -C "${repo}" worktree add --detach "${s}/${t}" "$([ "${t}" = before ] && echo "${before}" || echo HEAD)" >/dev/null
done
B="${s}/before"; A="${s}/after"
for u in bcir_ai_kernels.c bcir_q8_model.c bcir_llama.c bcir_decode.c; do
  cmp "${B}/runtime/c/${u}" "${A}/runtime/c/${u}"  # the slice moves no source of the units it times
done
lines() { python3 -c 'import json, sys; json.dump([json.loads(l) for l in sys.stdin], open(sys.argv[1], "w"), indent=1)' "$1"; }
scrub() {  # the records' strings carry no host paths: the scratch, the lab and the repository named
  sed -e "s#${s}#<""SCRATCH>/kernel-align-decode-long#g" -e "s#${lab}#<""LAB>#g" -e "s#${repo}#<""REPO>#g"
}

# decode-long.json: decode.json's arms and models but d512, 101 rounds a run
python3 "${threads}/synthetic_model.py" "${s}/d256.bcirq8" 6 8192 256 8 4 688 > "${s}/model-d256.json"
for cc in clang gcc; do
  python3 "${here}/build_decode.py" "${B}" "${A}" "${cc}" "${s}/decode-${cc}" D0:0:before D16:16:before \
    D32:32:before D48:48:before M0:0:after M32:32:after K0:0:kernels K32:32:kernels |
    sed "s/^{/{\"cc\":\"${cc}\",/" > "${s}/decode-build-${cc}.jsonl"
  for run in 1 2 3; do
    for model in "${lab}/model.bcirq8:4:200" "${s}/d256.bcirq8:2:72"; do
      IFS=: read -r path reps long <<< "${model}"
      "${s}/decode-${cc}" 101 "${reps}" "${path}" 8 "${long}" 64 | sed "s/^{/{\"cc\":\"${cc}\",\"run\":${run},/"
    done
  done > "${s}/decode-runs-${cc}.jsonl"
done
cat "${s}"/decode-build-*.jsonl | lines "${s}/decode-build.json"
cat "${s}"/decode-runs-*.jsonl | scrub | lines "${s}/decode-runs.json"
python3 -c 'import json, sys; json.dump({"builds": json.load(open(sys.argv[1])), "runs": json.load(open(sys.argv[2])), "synthetic_models": {"d256": json.load(open(sys.argv[3]))}}, open(sys.argv[4], "w"), indent=1)' \
  "${s}/decode-build.json" "${s}/decode-runs.json" "${s}/model-d256.json" "${here}/decode-long.json"

# hashes-decode-long.json: the tools' sha256 (this directory's and the one it runs from other records), the synthetic
# decoder's and the measured sources'
python3 - "${s}" "${here}" "${threads}" <<'PY'
import hashlib, json, sys
from pathlib import Path
s, here, threads = (Path(a) for a in sys.argv[1:4])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
record = {
    "tools": {t: sha(here / t) for t in ("measure_decode_long.sh", "decode_place.c", "build_decode.py")},
    "tools run from other records": {"../../2026-10-26/decode-threads/synthetic_model.py": sha(threads / "synthetic_model.py")},
    "synthetic_models": {"d256": sha(s / "d256.bcirq8")},
    "sources": {f"{t}/{p}/{u}": sha(s / t / p / u)
                for t in ("before", "after")
                for p, u in (("runtime/c", "bcir_ai_kernels.c"), ("runtime/c", "bcir_q8_model.c"),
                             ("runtime/c", "bcir_llama.c"), ("runtime/c", "bcir_decode.c"),
                             ("runtime", "manifest.json"), ("bcir", "toolchain.py"))},
}
json.dump(record, open(here / "hashes-decode-long.json", "w"), indent=1)
PY
for t in before after; do git -C "${repo}" worktree remove --force "${s}/${t}"; done
git -C "${repo}" worktree prune
echo "records written"
