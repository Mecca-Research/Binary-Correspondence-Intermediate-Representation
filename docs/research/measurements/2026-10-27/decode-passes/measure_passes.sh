#!/usr/bin/env bash
# DECODE-PASSES' records, run from <REPO> on the committed tree -- 661e9c3 DECODE-PASSES (the C decoder's embedding row
# and norm gammas dequantized a range at a time by bcir_q8_tensor_values, a group's scale built from its exponent's
# bits, and the greedy argmax over four lanes and a first-equal scan) and 0311c60 the tree before -- one step at a
# time with nothing else running; <SCRATCH> a scratch directory and <LAB> the qualification's lab model
# (BCIR-Corpus-860K). The slice moves bcir_q8_model.c and bcir_llama.c (and their headers) and no kernel. Writes
# decode.json, argmax.json, qualification.json, qualification-repeats.json, witness.json, host.json and hashes.json
# beside itself; faults.json holds the two fault sweeps run before the commit, the table and the same faults under the
# witnesses before (the table vacuity_table.py writes; README.json).
set -euo pipefail
repo="<REPO>"; s="<SCRATCH>/decode-passes"; lab="<LAB>"
before=0311c60f9b5c8ab3768507f45ad61223ea32027e
after=661e9c353aeb77eef2a9fd9702e2d9aa66873393
here="${repo}/docs/research/measurements/2026-10-27/decode-passes"
align="${repo}/docs/research/measurements/2026-10-27/kernel-align"    # decode_place.c, build_decode.py
threads="${repo}/docs/research/measurements/2026-10-26/decode-threads" # synthetic_model.py
lanes="${repo}/docs/research/measurements/2026-10-25/q8-lanes"         # qual_summary.py
rm -rf "${s:?}"; mkdir -p "${s}"
for t in before after; do
  git -C "${repo}" worktree add --detach "${s}/${t}" "$([ "${t}" = before ] && echo "${before}" || echo "${after}")" >/dev/null
done
B="${s}/before"; A="${s}/after"
for u in bcir_ai_kernels.c bcir_decode.c; do
  cmp "${B}/runtime/c/${u}" "${A}/runtime/c/${u}"  # the slice moves no kernel
done
lines() { python3 -c 'import json, sys; json.dump([json.loads(l) for l in sys.stdin], open(sys.argv[1], "w"), indent=1)' "$1"; }
scrub() {  # the records' strings carry no host paths: the scratch, the lab and the repository named
  sed -e "s#${s}#<""SCRATCH>/decode-passes#g" -e "s#${lab}#<""LAB>#g" -e "s#${repo}#<""REPO>#g"
}

# host.json: the host as /proc and /sys present it, the three clocks a record here can and cannot read
python3 - "${here}/host.json" <<'PY'
import json, os, sys
def first(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return None
info = {}
for line in (first("/proc/cpuinfo") or "").splitlines():
    if ":" in line:
        k, v = (x.strip() for x in line.split(":", 1))
        info.setdefault(k, v)
flags = set(info.get("flags", "").split())
want = ("hypervisor", "constant_tsc", "nonstop_tsc", "tsc_known_freq", "tsc_deadline_timer", "x2apic", "arat",
        "aperfmperf", "arch_perfmon", "avx2", "avx512f", "avx512bw", "avx512dq", "avx512vl", "avx512_vnni", "amx_tile")
sources = sorted(os.listdir("/sys/bus/event_source/devices")) if os.path.isdir("/sys/bus/event_source/devices") else []
msr_events = sorted(os.listdir("/sys/bus/event_source/devices/msr/events")) if "msr" in sources else []
record = {
    "brand": info.get("model name"), "vendor": info.get("vendor_id"),
    "signature": "%02X_%02XH" % (int(info.get("cpu family", "0")), int(info.get("model", "0"))),
    "family": int(info.get("cpu family", "0")), "model": int(info.get("model", "0")),
    "stepping": int(info.get("stepping", "0")),
    "microcode": {"presented": info.get("microcode"),
                  "trust": "hypervisor-presented" if "hypervisor" in flags else "native"},
    "flags": {f: f in flags for f in want},
    "cpus": os.cpu_count(),
    "clocksource": first("/sys/devices/system/clocksource/clocksource0/current_clocksource"),
    "perf_event_paranoid": first("/proc/sys/kernel/perf_event_paranoid"),
    "event_sources": sources, "msr_events": msr_events,
    "powercap": os.path.isdir("/sys/class/powercap"),
    "clocks": {
        "elapsed": "CLOCK_MONOTONIC: every time in these records",
        "reference_ticks": "the TSC (invariant: constant_tsc and nonstop_tsc); its rate as the kernel detected it at "
                           "boot, not read by these tools; no tick count is derived from a time here",
        "core_cycles": None,
        "core_cycles_reason": "no hardware PMU event source (a KVM guest): core cycles, instructions retired and the "
                              "decoded-µop-cache events cannot be counted here, and none is inferred from the TSC",
        "effective_frequency": None,
        "effective_frequency_reason": "neither APERF/MPERF (the aperfmperf flag absent; the msr event source offers "
                                      "only tsc and smi) nor a core-cycle counter",
        "energy": None, "energy_reason": "no RAPL (no powercap; the power event source lists no events)"},
    "level": "A (timing-qualified): correctness witnesses and elapsed time against an A/A control; no "
             "microarchitectural counter (Level B) is available on this host",
}
json.dump(record, open(sys.argv[1], "w"), indent=1)
PY

# decode.json: decode_place over copies of the decoder's four units in one binary, each compiler, the tree before's
# build at two placements and the tree after's at two, 101 rounds a run, three runs a model: the lab model, and
# DECODE-THREADS' synthetic decoders 256 and 512 wide (the artifacts not kept, their sha256 are)
python3 "${threads}/synthetic_model.py" "${s}/d256.bcirq8" 6 8192 256 8 4 688 > "${s}/model-d256.json"
python3 "${threads}/synthetic_model.py" "${s}/d512.bcirq8" 6 8192 512 8 4 1376 > "${s}/model-d512.json"
for cc in clang gcc; do
  python3 "${align}/build_decode.py" "${B}" "${A}" "${cc}" "${s}/decode-${cc}" D0:0:before D32:32:before \
    P0:0:after P32:32:after | sed "s/^{/{\"cc\":\"${cc}\",/" > "${s}/decode-build-${cc}.jsonl"
  for run in 1 2 3; do
    for model in "${lab}/model.bcirq8:4:200" "${s}/d256.bcirq8:2:72" "${s}/d512.bcirq8:1:48"; do
      IFS=: read -r path reps long <<< "${model}"
      "${s}/decode-${cc}" 101 "${reps}" "${path}" 8 "${long}" 64 | sed "s/^{/{\"cc\":\"${cc}\",\"run\":${run},/"
    done
  done > "${s}/decode-runs-${cc}.jsonl"
done
cat "${s}"/decode-build-*.jsonl | lines "${s}/decode-build.json"
cat "${s}"/decode-runs-*.jsonl | scrub | lines "${s}/decode-runs.json"
python3 -c 'import json, sys; json.dump({"builds": json.load(open(sys.argv[1])), "runs": json.load(open(sys.argv[2])), "synthetic_models": {"d256": json.load(open(sys.argv[3])), "d512": json.load(open(sys.argv[4]))}}, open(sys.argv[5], "w"), indent=1)' \
  "${s}/decode-build.json" "${s}/decode-runs.json" "${s}/model-d256.json" "${s}/model-d512.json" "${here}/decode.json"

# argmax.json: the two trees' argmax loops, copied verbatim into argmax_ab.c, held to one pick over 200,000 cases and
# timed on 1,024 and 32,000 logits beside an A/A control, 101 rounds, each compiler
for cc in clang gcc; do
  "${cc}" -std=c11 -O2 -ffp-contract=off -Wall -Wextra -Werror "${here}/argmax_ab.c" -o "${s}/argmax-${cc}" -lm
  "${s}/argmax-${cc}" 101 | sed "s/^{/{\"cc\":\"${cc}\",/"
done | lines "${here}/argmax.json"

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
    python3 -c 'import json, sys; d = json.load(open(sys.argv[1])); c = d["latency"]["c"]; o = c["per_operation_ms"]; print(json.dumps({"tree": sys.argv[2], "run": int(sys.argv[3]), "commit": d["commit"], "status": d["status"], "per_token_ms_median": round(c["per_token_ms"]["median"], 4), "tokens_per_second": round(c["tokens_per_second"], 1), "monolithic_median_ms": round(c["monolithic_median_ms"], 3), "gem_median_ms": round(c["gem_median_ms"], 3), "per_operation_ms": {k: round(v, 4) for k, v in sorted(o.items())}}))' \
      "${s}/q-${t}-${run}.json" "${t}" "${run}"
  done
done | lines "${here}/qualification-repeats.json"

# witness.json: the fault table's command on the tree after (every witness passing, its capabilities required)
mapfile -t cmd < <(python3 -c 'import json, sys; print("\n".join(json.load(open(sys.argv[1]))["command"][1:]))' \
  "${A}/tools/testing/faults/decode-passes.json")
(cd "${A}" && python3 "${cmd[@]}") > "${s}/witness.txt"
python3 -c 'import json, sys; json.dump({"command": ["@python", *sys.argv[2:-1]], "exit": 0, "output": open(sys.argv[1]).read().strip().splitlines()}, open(sys.argv[-1], "w"), indent=1)' \
  "${s}/witness.txt" "${cmd[@]}" "${s}/witness.json"
scrub < "${s}/witness.json" > "${here}/witness.json"

# hashes.json: the tools' sha256 (this directory's and the ones it runs from other records), the lab and synthetic
# models' and the measured sources'
python3 - "${s}" "${here}" "${align}" "${threads}" "${lanes}" "${lab}" <<'PY'
import hashlib, json, sys
from pathlib import Path
s, here, align, threads, lanes, lab = (Path(a) for a in sys.argv[1:7])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
record = {
    "tools": {t: sha(here / t) for t in ("measure_passes.sh", "argmax_ab.c", "vacuity_table.py")},
    "tools run from other records": {
        "../kernel-align/decode_place.c": sha(align / "decode_place.c"),
        "../kernel-align/build_decode.py": sha(align / "build_decode.py"),
        "../../2026-10-26/decode-threads/synthetic_model.py": sha(threads / "synthetic_model.py"),
        "../../2026-10-25/q8-lanes/qual_summary.py": sha(lanes / "qual_summary.py")},
    "models": {"lab": sha(lab / "model.bcirq8"), "d256": sha(s / "d256.bcirq8"), "d512": sha(s / "d512.bcirq8")},
    "sources": {f"{t}/{p}/{u}": sha(s / t / p / u)
                for t in ("before", "after")
                for p, u in (("runtime/c", "bcir_ai_kernels.c"), ("runtime/c", "bcir_q8_model.c"),
                             ("runtime/c", "bcir_q8_model.h"), ("runtime/c", "bcir_llama.c"),
                             ("runtime/c", "bcir_llama_ops.h"), ("runtime/c", "bcir_decode.c"),
                             ("runtime", "manifest.json"))},
}
json.dump(record, open(here / "hashes.json", "w"), indent=1)
PY
for t in before after; do git -C "${repo}" worktree remove --force "${s}/${t}"; done
git -C "${repo}" worktree prune
echo "records written"
