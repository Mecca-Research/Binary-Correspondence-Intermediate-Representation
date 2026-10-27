#!/usr/bin/env bash
# NEXT-12's records, run from <REPO> on the committed tree (HEAD) one step at a time with nothing else running;
# <SCRATCH> a scratch directory and <LAB> the qualification's lab model (BCIR-Corpus-860K). The register-blocked AVX-512
# matvec -- four 8-column accumulators held in registers down every input row, each output's terms in Q8-PAIR's order
# -- measured and classified rather than discarded: against the committed kernel in three regimes (block_ab.c), its
# effect on the core's clock on this host (block_clock.c, NEXT-9's instrument), and in the whole decoder (decode_place
# over the tree and the tree with q8_block.patch applied, the patch make_block_patch.py writes). Writes host.json,
# block.json, clock.json, decode.json, bandwidth.json and hashes.json beside itself. No source of the tree changes.
set -euo pipefail
repo="<REPO>"; s="<SCRATCH>/next12-records"; lab="<LAB>"
here="${repo}/docs/research/measurements/2026-10-27/next12"
align="${repo}/docs/research/measurements/2026-10-27/kernel-align"    # decode_place.c, build_decode.py
threads="${repo}/docs/research/measurements/2026-10-26/decode-threads" # synthetic_model.py
rm -rf "${s:?}"; mkdir -p "${s}"
for t in base block; do git -C "${repo}" worktree add --detach "${s}/${t}" HEAD >/dev/null; done
B="${s}/base"; K="${s}/block"
git -C "${K}" apply "${here}/q8_block.patch"
python3 "${here}/make_block_patch.py" "${B}/runtime/c/bcir_ai_kernels.c" "${s}/bcir_ai_kernels.block.c" > /dev/null
cmp "${s}/bcir_ai_kernels.block.c" "${K}/runtime/c/bcir_ai_kernels.c"  # the patch is the generator's text
for u in bcir_q8_model.c bcir_llama.c bcir_decode.c; do cmp "${B}/runtime/c/${u}" "${K}/runtime/c/${u}"; done
lines() { python3 -c 'import json, sys; json.dump([json.loads(l) for l in sys.stdin], open(sys.argv[1], "w"), indent=1)' "$1"; }
scrub() {  # the records' strings carry no host paths: the scratch, the lab and the repository named
  sed -e "s#${s}#<""SCRATCH>/next12-records#g" -e "s#${lab}#<""LAB>#g" -e "s#${repo}#<""REPO>#g"
}
# the kernel unit's options as the tree's manifest gives them under a compiler (M17 and M18 on x86-64)
opts() {
  python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); from bcir.toolchain import unit_options; print(" ".join(unit_options(["bcir_ai_kernels.c"], sys.argv[2])))' \
    "${B}" "$1"
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

# block.json: block_ab.c beside the tree's kernel unit, built with the unit's options, each compiler, 101 rounds a
# regime: hot (each of the 860K decoder's seven shapes alone), token (its 28 matrices in turn), memory (four layers
# of TinyLlama-1.1B's widths in turn, about 180 MiB of codes)
for cc in clang gcc; do
  # shellcheck disable=SC2046
  "${cc}" -std=c11 -O2 $(opts "${cc}") -Wall -Wextra -Werror -I "${B}/runtime/c" "${here}/block_ab.c" \
    "${B}/runtime/c/bcir_ai_kernels.c" -o "${s}/block_ab-${cc}" -lm
  for regime in hot token memory; do "${s}/block_ab-${cc}" 101 "${regime}" | sed "s/^{/{\"cc\":\"${cc}\",/"; done
done | lines "${here}/block.json"

# clock.json: block_clock.c (NEXT-9's clock probe with the blocked kernel an arm) -- the core's cycle time while the
# matvecs run, back to back (pass: a 20,000-step chain after each pass of 28) and with a decode's duty cycle (matvec:
# 10,000 steps after each matvec), 24 rounds of 40 passes as NEXT-9 ran it, three runs, each compiler
for cc in clang gcc; do
  # shellcheck disable=SC2046
  "${cc}" -std=c11 -O2 $(opts "${cc}") -Wall -Wextra -Werror -I "${B}/runtime/c" "${here}/block_clock.c" \
    "${B}/runtime/c/bcir_ai_kernels.c" -o "${s}/block_clock-${cc}" -lm
done
for run in 1 2 3; do
  for cc in clang gcc; do
    "${s}/block_clock-${cc}" 24 40 20000 pass | sed "s/^{/{\"run\":${run},\"cc\":\"${cc}\",/"
    "${s}/block_clock-${cc}" 24 40 280000 matvec | sed "s/^{/{\"run\":${run},\"cc\":\"${cc}\",/"
  done
done | lines "${here}/clock.json"

# decode.json: decode_place over copies of the decoder's four units in one binary, each compiler: the tree's build
# at two placements and the blocked tree's at two (every token and final logit of every copy the first copy's bytes,
# so the blocked kernel's outputs are held to the committed one's in the whole decoder), three runs a model: the lab
# model and DECODE-THREADS' 256-wide synthetic decoder at 101 rounds, and a two-layer synthetic decoder at
# TinyLlama-1.1B's widths (2,048 wide, 32 heads of 64, 4 KV heads, 5,632 FF, 32,000 tokens) at 31 rounds
python3 "${threads}/synthetic_model.py" "${s}/d256.bcirq8" 6 8192 256 8 4 688 > "${s}/model-d256.json"
python3 "${threads}/synthetic_model.py" "${s}/tl2.bcirq8" 2 32000 2048 32 4 5632 > "${s}/model-tl2.json"
for cc in clang gcc; do
  python3 "${align}/build_decode.py" "${B}" "${K}" "${cc}" "${s}/decode-${cc}" D0:0:before D32:32:before \
    K0:0:after K32:32:after | sed "s/^{/{\"cc\":\"${cc}\",/" > "${s}/decode-build-${cc}.jsonl"
  for run in 1 2 3; do
    for model in "${lab}/model.bcirq8:101:4:200:64" "${s}/d256.bcirq8:101:2:72:64" "${s}/tl2.bcirq8:31:1:24:256"; do
      IFS=: read -r path rounds reps long cap <<< "${model}"
      "${s}/decode-${cc}" "${rounds}" "${reps}" "${path}" 8 "${long}" "${cap}" | sed "s/^{/{\"cc\":\"${cc}\",\"run\":${run},/"
    done
  done > "${s}/decode-runs-${cc}.jsonl"
done
cat "${s}"/decode-build-*.jsonl | lines "${s}/decode-build.json"
cat "${s}"/decode-runs-*.jsonl | scrub | lines "${s}/decode-runs.json"
python3 -c 'import json, sys; json.dump({"builds": json.load(open(sys.argv[1])), "runs": json.load(open(sys.argv[2])), "synthetic_models": {"d256": json.load(open(sys.argv[3])), "tl2": json.load(open(sys.argv[4]))}}, open(sys.argv[5], "w"), indent=1)' \
  "${s}/decode-build.json" "${s}/decode-runs.json" "${s}/model-d256.json" "${s}/model-tl2.json" "${here}/decode.json"

# bandwidth.json: bw.c, the host's read bandwidth from memory at 1, 2 and 4 threads, contiguous parts of 1 GiB and
# column ranges of 5,632-byte rows (the best of five passes each), to read the memory regime's times against
for cc in gcc; do
  "${cc}" -std=c11 -O2 -Wall -Wextra -Werror -pthread "${here}/bw.c" -o "${s}/bw-${cc}"
  "${s}/bw-${cc}" | python3 -c 'import json, sys; [print(json.dumps({"cc": sys.argv[1], "line": l.strip()})) for l in sys.stdin if l.strip()]' "${cc}"
done | lines "${here}/bandwidth.json"

# hashes.json: the tools' sha256 (this directory's and the ones it runs from other records), the models' and the
# measured sources'
python3 - "${s}" "${here}" "${align}" "${threads}" "${lab}" <<'PY'
import hashlib, json, sys
from pathlib import Path
s, here, align, threads, lab = (Path(a) for a in sys.argv[1:6])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
record = {
    "tools": {t: sha(here / t) for t in ("measure_next12.sh", "block_ab.c", "block_clock.c", "make_block_patch.py",
                                         "q8_block.patch", "bw.c")},
    "tools run from other records": {
        "../kernel-align/decode_place.c": sha(align / "decode_place.c"),
        "../kernel-align/build_decode.py": sha(align / "build_decode.py"),
        "../../2026-10-26/decode-threads/synthetic_model.py": sha(threads / "synthetic_model.py")},
    "models": {"lab": sha(lab / "model.bcirq8"), "d256": sha(s / "d256.bcirq8"), "tl2": sha(s / "tl2.bcirq8")},
    "sources": {f"{t}/runtime/c/{u}": sha(s / t / "runtime" / "c" / u)
                for t in ("base", "block")
                for u in ("bcir_ai_kernels.c", "bcir_q8_model.c", "bcir_llama.c", "bcir_decode.c")},
}
json.dump(record, open(here / "hashes.json", "w"), indent=1)
PY
for t in base block; do git -C "${repo}" worktree remove --force "${s}/${t}"; done
git -C "${repo}" worktree prune
echo "records written"
