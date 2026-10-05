#!/usr/bin/env bash
# Validate the freestanding C StreamPack runtime: it compiles with no libc, and a
# Python-encoded StreamPack round-trips through the C decoder (ABI parity).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C="${ROOT}/runtime/c"
CC_WAS_SET="${CC+x}"
CC="${CC:-$(command -v clang || command -v cc || true)}"
if [ -z "${CC}" ]; then
  echo "no C compiler (clang/cc); skipping runtime check." >&2
  exit 0
fi

# Older distributions can expose an unversioned clang that predates the final
# `-std=c23` spelling while also installing a newer versioned clang.  Prefer an
# explicitly supplied CC, but otherwise upgrade the default to the highest
# versioned clang that accepts the spelling used throughout this gate.  This
# keeps the C23 checks meaningful instead of failing on tool-name discovery.
if [ "${CC_WAS_SET}" != x ] &&
   ! printf 'int main(void){return 0;}\n' | "${CC}" -std=c23 -x c -c -o /dev/null - >/dev/null 2>&1; then
  best=""; best_major=-1
  old_ifs="${IFS}"; IFS=:
  for directory in ${PATH}; do
    IFS="${old_ifs}"
    for candidate in "${directory}"/clang-[0-9]*; do
      [ -x "${candidate}" ] || continue
      major="${candidate##*/clang-}"
      [[ "${major}" =~ ^[0-9]+$ ]] || continue
      [ "${major}" -gt "${best_major}" ] || continue
      if printf 'int main(void){return 0;}\n' | "${candidate}" -std=c23 -x c -c -o /dev/null - >/dev/null 2>&1; then
        best="${candidate}"; best_major="${major}"
      fi
    done
    IFS=:
  done
  IFS="${old_ifs}"
  [ -z "${best}" ] || CC="${best}"
fi

echo "[c-runtime] memory classes + allocator/context/channel fault sweep"
if CC="${CC}" bash "${ROOT}/tools/c/check_memory_discipline.sh" 2>&1 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "  FAIL: memory-discipline gate"; exit 1
fi

echo "[c-runtime] freestanding compile (-ffreestanding -nostdlib), C11 + C23"
for std in c11 c23; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -c "${C}/bcir_runtime.c" -o /dev/null \
    || { echo "  FAIL: runtime not freestanding-clean under -std=${std}"; exit 1; }
done
echo "  PASS freestanding (C11 + C23; ABI static_assert holds)"

tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

echo "[c-runtime] x86 interrupt-frame ABI header (C11 + C23)"
for std in c11 c23; do
  "${CC}" -ffreestanding -std=${std} -Wall -Wextra -Werror -I "${C}" \
    "${C}/test_x86_interrupt.c" -o "${tmp}/test_x86_interrupt_${std}" \
    || { echo "  FAIL: x86 interrupt-frame ABI under -std=${std}"; exit 1; }
done
bash "${ROOT}/tools/c/sections/x86_interrupt.sh" "${tmp}/test_x86_interrupt_c11" "${tmp}/test_x86_interrupt_c23" || exit 1

echo "[c-runtime] build harness (C23) + Python->C ABI parity"
"${CC}" -std=c23 -O2 "${C}/bcir_runtime.c" "${C}/test_runtime.c" -I "${C}" -o "${tmp}/test_runtime" \
  || { echo "  FAIL: harness build"; exit 1; }
bash "${ROOT}/tools/c/sections/runtime.sh" "${tmp}/test_runtime" || exit 1

echo "[c-runtime] BCAB artifact bundle: freestanding reader + Python/C selection parity"
for std in c11 c23; do
  for unit in bcir_artifact_bundle.c bcir_sha256.c; do
    "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -Werror -I "${C}" \
      -c "${C}/${unit}" -o /dev/null \
      || { echo "  FAIL: BCAB reader (${unit}) not freestanding-clean under -std=${std}"; exit 1; }
  done
done
"${CC}" -std=c23 -O2 -Wall -Wextra -Werror -I "${C}" "${C}/bcir_sha256.c" "${C}/test_sha256.c" \
  -o "${tmp}/test_sha256" || { echo "  FAIL: SHA-256/HMAC vector harness build"; exit 1; }
"${CC}" -std=c23 -O2 -Wall -Wextra -Werror -I "${C}" \
  "${C}/bcir_artifact_bundle.c" "${C}/bcir_sha256.c" "${C}/bcir_runtime.c" \
  "${C}/test_artifact_bundle.c" -o "${tmp}/test_artifact_bundle" \
  || { echo "  FAIL: BCAB C parity harness build"; exit 1; }
"${CC}" -std=c23 -O2 -Wall -Wextra -Werror -I "${C}" \
  "${C}/bcir_asn1.c" "${C}/test_asn1.c" -o "${tmp}/test_artifact_asn1" \
  || { echo "  FAIL: BCAB ASN.1 C validation harness build"; exit 1; }
bash "${ROOT}/tools/c/sections/artifact_bundle.sh" "${tmp}/test_sha256" "${tmp}/test_artifact_bundle" "${tmp}/test_artifact_asn1" || exit 1

echo "[c-runtime] ETL binary-record decoder: freestanding compile (C11 + C23)"
# bcir_binrec.c is the C twin of bcir/etl/binary.py (a second binary trust boundary).
for std in c11 c23; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -I "${C}" -c "${C}/bcir_binrec.c" -o /dev/null \
    || { echo "  FAIL: bcir_binrec not freestanding-clean under -std=${std}"; exit 1; }
done
echo "  PASS bcir_binrec freestanding (C11 + C23)"

echo "[c-runtime] StreamPack executor: freestanding compile (C11 + C23) + Python->C parity"
for std in c11 c23; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -I "${C}" -c "${C}/bcir_exec.c" -o /dev/null \
    || { echo "  FAIL: bcir_exec not freestanding-clean under -std=${std}"; exit 1; }
done
"${CC}" -std=c23 -O2 "${C}/bcir_exec.c" "${C}/bcir_runtime.c" "${C}/test_exec.c" -I "${C}" -o "${tmp}/test_exec" \
  || { echo "  FAIL: executor harness build"; exit 1; }
bash "${ROOT}/tools/c/sections/executor.sh" "${tmp}/test_exec" || exit 1

echo "[c-runtime] StreamPack encoder: freestanding compile (C11 + C23) + byte-identical re-encode"
for std in c11 c23; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -I "${C}" -c "${C}/bcir_encode.c" -o /dev/null \
    || { echo "  FAIL: bcir_encode not freestanding-clean under -std=${std}"; exit 1; }
done
"${CC}" -std=c23 -O2 "${C}/bcir_encode.c" "${C}/bcir_runtime.c" "${C}/test_encode.c" -I "${C}" -o "${tmp}/test_encode" \
  || { echo "  FAIL: encoder harness build"; exit 1; }
bash "${ROOT}/tools/c/sections/encoder.sh" "${tmp}/test_encode" || exit 1

echo "[c-runtime] ExecutionPlanV1 (G11): freestanding decode + Python->C->Python byte-identical round trip + plan/pack binding"
"${CC}" -std=c23 -O2 -Wall -Wextra -Werror "${C}/bcir_runtime.c" "${C}/test_execution_plan.c" -I "${C}" \
  -o "${tmp}/test_execution_plan" || { echo "  FAIL: plan harness build"; exit 1; }
bash "${ROOT}/tools/c/sections/execution_plan.sh" "${tmp}/test_execution_plan" || exit 1

echo "[c-runtime] ControlRecordV1 (G14): freestanding plane + Python->C->Python round trip + declared refusal statuses + identical two-rail plane traces"
for std in c11 c23; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -Werror -I "${C}" \
    -c "${C}/bcir_control_plane.c" -o /dev/null \
    || { echo "  FAIL: control plane not freestanding-clean under -std=${std}"; exit 1; }
done
ctl_sources=("${C}/bcir_ring.c" "${C}/bcir_sha256.c" "${C}/bcir_runtime.c" "${C}/test_control_plane.c")
"${CC}" -std=c23 -O2 -Wall -Wextra -Werror -I "${C}" "${C}/bcir_control_plane.c" "${ctl_sources[@]}" \
  -o "${tmp}/test_control_plane" || { echo "  FAIL: control harness build"; exit 1; }
python3 "${ROOT}/tools/build/mutate.py" --variant test_control_plane_mutant --out "${tmp}/bcir_control_plane_mutant.c" \
  || { echo "  FAIL: the deferral-law fault injection did not apply (the law's line changed)"; exit 1; }
"${CC}" -std=c23 -O2 -I "${C}" "${tmp}/bcir_control_plane_mutant.c" "${ctl_sources[@]}" \
  -o "${tmp}/test_control_plane_mutant" || { echo "  FAIL: mutant harness build"; exit 1; }
bash "${ROOT}/tools/c/sections/control_plane.sh" "${tmp}/test_control_plane" "${tmp}/test_control_plane_mutant" || exit 1

echo "[c-runtime] live SPSC ring + TelemetryEnvelopeV0 (G15): freestanding ring/envelope + generated signal table + two-rail scenario traces + concurrency under ThreadSanitizer"
for std in c11 c23; do
  for unit in bcir_ring.c bcir_telemetry_envelope.c; do
    "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -Wconversion -Wpedantic -Werror \
      -I "${C}" -c "${C}/${unit}" -o /dev/null \
      || { echo "  FAIL: ${unit} not freestanding-clean under -std=${std}"; exit 1; }
  done
done
ring_sources=("${C}/bcir_telemetry_envelope.c" "${C}/bcir_runtime.c" "${C}/test_ring.c")
"${CC}" -std=c23 -O2 -Wall -Wextra -Werror -pthread -I "${C}" "${C}/bcir_ring.c" "${ring_sources[@]}" \
  -o "${tmp}/test_ring" || { echo "  FAIL: ring harness build"; exit 1; }
for opt in O0 O3; do
  "${CC}" -std=c23 -${opt} -Wall -Wextra -Werror -pthread -I "${C}" "${C}/bcir_ring.c" "${ring_sources[@]}" \
    -o "${tmp}/test_ring_${opt}" || { echo "  FAIL: ring harness build at -${opt}"; exit 1; }
done
python3 "${ROOT}/tools/build/mutate.py" --variant test_ring_torn --out "${tmp}/bcir_ring_torn.c" \
  || { echo "  FAIL: the seqlock fault injection did not apply (the re-check's line changed)"; exit 1; }
"${CC}" -std=c23 -O2 -pthread -I "${C}" "${tmp}/bcir_ring_torn.c" "${ring_sources[@]}" \
  -o "${tmp}/test_ring_torn" || { echo "  FAIL: ring mutant harness build"; exit 1; }
bash "${ROOT}/tools/c/sections/ring.sh" "${tmp}/test_ring" "${tmp}/test_ring_O0" "${tmp}/test_ring_O3" "${tmp}/test_ring_torn" || exit 1
# The ThreadSanitizer leg: the ring harness built with -fsanitize=thread, and the same build of a
# ring whose relaxed atomic stores are made plain (the manifest's sanitizer variants), judged by
# tools/c/sections/ring_tsan.sh -- the stress runs must report no data race, and the plain-store
# ring must be REPORTED as racing. TSan needs the compiler-rt runtime. The x86 C runtime job
# installs it and sets BCIR_REQUIRE_TSAN=1, so there an unavailable TSan is a failure (L2: the job
# that installed the tool owns its absence); on a host or runner without it (the aarch64 job
# installs clang/lld/llvm only) this leg is an explicit, reported skip -- the concurrent rows above
# still ran there, natively. Available means a trivial TSan program builds AND runs:
# tools/build/sanitizer.py, the one predicate the CMake configure and the section-parity gate ask.
if python3 "${ROOT}/tools/build/sanitizer.py" --cc "${CC}" thread >/dev/null 2>&1; then
  "${CC}" -std=c11 -O1 -g -fsanitize=thread -pthread -I "${C}" "${C}/bcir_ring.c" "${ring_sources[@]}" \
    -o "${tmp}/test_ring_tsan" \
    || { echo "  FAIL: the ThreadSanitizer build of the ring harness failed (a trivial TSan program built)"; exit 1; }
  python3 "${ROOT}/tools/build/mutate.py" --variant test_ring_plain --out "${tmp}/bcir_ring_plain.c" \
    || { echo "  FAIL: the plain-store fault injection did not apply (st_rlx changed)"; exit 1; }
  "${CC}" -std=c11 -O1 -g -fsanitize=thread -pthread -I "${C}" "${tmp}/bcir_ring_plain.c" "${ring_sources[@]}" \
    -o "${tmp}/test_ring_plain" \
    || { echo "  FAIL: the ThreadSanitizer build of the plain-store ring failed"; exit 1; }
  bash "${ROOT}/tools/c/sections/ring_tsan.sh" "${tmp}/test_ring_tsan" "${tmp}/test_ring_plain" || exit 1
elif [ "${BCIR_REQUIRE_TSAN:-0}" = "1" ]; then
  echo "  FAIL: ThreadSanitizer is required here (BCIR_REQUIRE_TSAN=1), but a trivial TSan program does not build and run"; exit 1
else
  echo "  SKIP live ring under ThreadSanitizer: UNAVAILABLE on this host (no TSan runtime); CI's x86 C runtime job requires it"
fi

echo "[c-runtime] data-plane hand-off (G16): freestanding pack table + shard manifest + per-step freeze + oracle/C traces"
for std in c11 c23; do
  for unit in bcir_handoff.c bcir_shard_manifest.c bcir_hydrate.c; do
    "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -Wconversion -Wpedantic -Werror \
      -I "${C}" -c "${C}/${unit}" -o /dev/null \
      || { echo "  FAIL: ${unit} not freestanding-clean under -std=${std}"; exit 1; }
  done
done
handoff_sources=("${C}/bcir_handoff.c" "${C}/bcir_shard_manifest.c" "${C}/bcir_hydrate.c" "${C}/bcir_plan.c" "${C}/bcir_control_plane.c" "${C}/bcir_sha256.c" "${C}/bcir_runtime.c" "${C}/bcir_ring.c" "${C}/bcir_telemetry_envelope.c" "${C}/test_handoff.c")
"${CC}" -std=c23 -O2 -Wall -Wextra -Werror -I "${C}" "${handoff_sources[@]}" -o "${tmp}/test_handoff" \
  || { echo "  FAIL: hand-off harness build"; exit 1; }
"${CC}" -std=c23 -O0 -Wall -Wextra -Werror -I "${C}" "${handoff_sources[@]}" -o "${tmp}/test_handoff_o0" \
  || { echo "  FAIL: -O0 hand-off harness build"; exit 1; }
"${CC}" -std=c23 -O3 -Wall -Wextra -Werror -I "${C}" "${handoff_sources[@]}" -o "${tmp}/test_handoff_o3" \
  || { echo "  FAIL: -O3 hand-off harness build"; exit 1; }
python3 "${ROOT}/tools/build/mutate.py" --variant test_handoff_mutant --out "${tmp}/bcir_handoff_mutant.c" \
  || { echo "  FAIL: the dispatch-generation fault injection did not apply (the law's line changed)"; exit 1; }
mutant_handoff=("${tmp}/bcir_handoff_mutant.c" "${handoff_sources[@]:1}")
"${CC}" -std=c23 -O2 -I "${C}" "${mutant_handoff[@]}" -o "${tmp}/test_handoff_mutant" \
  || { echo "  FAIL: hand-off mutant harness build"; exit 1; }
bash "${ROOT}/tools/c/sections/handoff.sh" "${tmp}/test_handoff" "${tmp}/test_handoff_o0" "${tmp}/test_handoff_o3" "${tmp}/test_handoff_mutant" || exit 1

echo "[c-runtime] native K_BCIR planner (G17): freestanding compile (C11 + C23) + byte parity with the oracle"
for std in c11 c23; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -Wconversion -Wpedantic -Werror \
    -I "${C}" -c "${C}/bcir_kplan.c" -o /dev/null \
    || { echo "  FAIL: bcir_kplan.c not freestanding-clean under -std=${std}"; exit 1; }
done
kplan_sources=("${C}/bcir_kplan.c" "${C}/bcir_runtime.c" "${C}/test_kplan.c")
"${CC}" -std=c23 -O2 -Wall -Wextra -Werror -I "${C}" "${kplan_sources[@]}" -o "${tmp}/test_kplan" \
  || { echo "  FAIL: planner harness build"; exit 1; }
for opt in O0 O3; do
  "${CC}" -std=c23 -${opt} -Wall -Wextra -Werror -I "${C}" "${kplan_sources[@]}" -o "${tmp}/test_kplan_${opt}" \
    || { echo "  FAIL: -${opt} planner harness build"; exit 1; }
done
python3 "${ROOT}/tools/build/mutate.py" --variant test_kplan_mutant --out "${tmp}/bcir_kplan_mutant.c" \
  || { echo "  FAIL: the carry fault injection did not apply (the line changed)"; exit 1; }
"${CC}" -std=c23 -O2 -I "${C}" "${tmp}/bcir_kplan_mutant.c" "${C}/bcir_runtime.c" "${C}/test_kplan.c" \
  -o "${tmp}/test_kplan_mutant" || { echo "  FAIL: planner mutant harness build"; exit 1; }
bash "${ROOT}/tools/c/sections/kplan.sh" "${tmp}/test_kplan" "${tmp}/test_kplan_O0" "${tmp}/test_kplan_O3" "${tmp}/test_kplan_mutant" || exit 1

echo "[c-runtime] UART telemetry frame (#telemetry-frame): freestanding compile (C11 + C23) + byte-identical re-encode"
for std in c11 c23; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -I "${C}" -c "${C}/bcir_telemetry_frame.c" -o /dev/null \
    || { echo "  FAIL: bcir_telemetry_frame not freestanding-clean under -std=${std}"; exit 1; }
done
"${CC}" -std=c23 -O2 "${C}/bcir_telemetry_frame.c" "${C}/bcir_runtime.c" "${C}/test_telemetry_frame.c" -I "${C}" -o "${tmp}/test_tframe" \
  || { echo "  FAIL: telemetry-frame harness build"; exit 1; }
bash "${ROOT}/tools/c/sections/telemetry_frame.sh" "${tmp}/test_tframe" || exit 1

echo "[c-runtime] frozen Q8 table (#embed / fallback): build + self-check (C11 + C23)"
# Drift gate: the committed runtime/c/{q8_tiers.bin,bcir_q8_tables.h} must equal a
# fresh emission from the oracle (bcir.kbcir.cost.MemoryHierarchy.default()).
python3 -m bcir.abi.q8_tables --emit >/dev/null || { echo "  FAIL: q8 emit"; exit 1; }
if ! git -C "${ROOT}" diff --quiet -- runtime/c/q8_tiers.bin runtime/c/bcir_q8_tables.h 2>/dev/null; then
  echo "  FAIL: Q8 table drifted from the oracle (regenerate: python -m bcir.abi.q8_tables --emit)"; exit 1
fi
for std in c11 c23; do
  "${CC}" -std=${std} -Wall -Wextra -I "${C}" "${C}/test_q8_tables.c" -o "${tmp}/q8_${std}" \
    || { echo "  FAIL: Q8 table build under -std=${std}"; exit 1; }
done
bash "${ROOT}/tools/c/sections/q8_tables.sh" "${tmp}/q8_c11" "${tmp}/q8_c23" || exit 1

echo "[c-runtime] plug-in C frontend (bcir_cfront): IR freestanding + Python<->C parity"
# bcir_cir.h is the freestanding BCIR claim-graph IR; bcir_cfront.c is the host compiler
# tool (the C twin of bcir/frontends/cfront -- it lowers driver/register-map C to that IR).
printf '#include "bcir_cir.h"\nint probe(void){return (int)sizeof(bcir_claim)+(int)BCIR_OP_LOAD+(int)BCIR_DOM_MMIO;}\n' > "${tmp}/cir_probe.c"
for std in c11 c23; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -I "${C}" -c "${tmp}/cir_probe.c" -o /dev/null \
    || { echo "  FAIL: bcir_cir.h not freestanding-clean under -std=${std}"; exit 1; }
done
echo "  PASS bcir_cir.h freestanding IR (C11 + C23)"
# bcir_cfront verifies (bcir_verify.c: R1-R8+R18 / provenance digest, with R10-R11 reaching
# bcir_sp_validate in bcir_runtime.c), so the host tool links both. bcir_verify.c is a host tool
# (its diagnostic path uses snprintf), not freestanding.
CFRONT_SRCS="${C}/bcir_cfront.c ${C}/bcir_cpp.c ${C}/bcir_verify.c ${C}/bcir_runtime.c"
"${CC}" -std=c23 -O2 -Wall -Wextra ${CFRONT_SRCS} "${C}/test_cfront.c" -I "${C}" -o "${tmp}/test_cfront" 2>/dev/null \
  || "${CC}" -std=c11 -O2 ${CFRONT_SRCS} "${C}/test_cfront.c" -I "${C}" -o "${tmp}/test_cfront" \
  || { echo "  FAIL: C frontend build"; exit 1; }
bash "${ROOT}/tools/c/sections/cfront.sh" "${tmp}/test_cfront" || exit 1

echo "[c-runtime] target-ABI matrix (bcir_cfront --target): sizeof data model == oracle (#abi)"
bash "${ROOT}/tools/c/sections/cfront_abi.sh" "${tmp}/test_cfront" || exit 1

echo "[c-runtime] full C compile->execute loop (cfront -> plan -> hydrate -> exec, no Python)"
# bcir_plan.c + bcir_hydrate.c are freestanding (the driver-embeddable planner + StreamPack
# writer that feed the existing bcir_exec.c) -- the loop closes with no Python.
for f in bcir_plan.c bcir_hydrate.c; do
  for std in c11 c23; do
    "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -I "${C}" -c "${C}/${f}" -o /dev/null \
      || { echo "  FAIL: ${f} not freestanding-clean under -std=${std}"; exit 1; }
  done
done
LOOP_SRCS="${C}/bcir_cfront.c ${C}/bcir_cpp.c ${C}/bcir_plan.c ${C}/bcir_hydrate.c ${C}/bcir_exec.c ${C}/bcir_runtime.c ${C}/bcir_verify.c ${C}/test_cfront_loop.c"
"${CC}" -std=c23 -O2 -I "${C}" ${LOOP_SRCS} -o "${tmp}/loop" 2>/dev/null \
  || "${CC}" -std=c11 -O2 -I "${C}" ${LOOP_SRCS} -o "${tmp}/loop" \
  || { echo "  FAIL: loop build"; exit 1; }
bash "${ROOT}/tools/c/sections/cfront_loop.sh" "${tmp}/loop" || exit 1

echo "[c-runtime] multi-channel lowering decision (bcir_channel): channel.json -> backend pick"
"${CC}" -std=c23 -O2 -Wall -Wextra -I "${C}" "${C}/bcir_channel.c" "${C}/test_channel.c" -o "${tmp}/tch" 2>/dev/null \
  || "${CC}" -std=c11 -O2 -I "${C}" "${C}/bcir_channel.c" "${C}/test_channel.c" -o "${tmp}/tch" \
  || { echo "  FAIL: channel router build"; exit 1; }
bash "${ROOT}/tools/c/sections/channel.sh" "${tmp}/tch" || exit 1

echo "[c-runtime] bcir-cc compiler driver: compile a driver (sibling header) + emit artifacts"
# bcir_cc.c is the cc-like driver over the full C pipeline (bcir_cpp_run_ex -I/-D -> bcir_cfront ->
# plan -> hydrate). It must compile a driver with sibling headers via a normal compile command.
"${CC}" -std=c23 -O2 -Wall -Wextra -I "${C}" "${C}/bcir_cc.c" "${C}/bcir_cpp.c" "${C}/bcir_cfront.c" \
  "${C}/bcir_verify.c" "${C}/bcir_runtime.c" "${C}/bcir_plan.c" "${C}/bcir_hydrate.c" -o "${tmp}/bcir-cc" 2>/dev/null \
  || "${CC}" -std=c11 -O2 -I "${C}" "${C}/bcir_cc.c" "${C}/bcir_cpp.c" "${C}/bcir_cfront.c" \
       "${C}/bcir_verify.c" "${C}/bcir_runtime.c" "${C}/bcir_plan.c" "${C}/bcir_hydrate.c" -o "${tmp}/bcir-cc" \
  || { echo "  FAIL: bcir-cc build"; exit 1; }
bash "${ROOT}/tools/c/sections/bcir_cc.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] bcir-cc --emit-c links the runtime: self-contained masked unit (#emitlink)"
CC="${CC}" bash "${ROOT}/tools/c/sections/emitlink.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] bcir-cc masked emit + recovery override: the recorded two-truth crossing (#recover)"
CC="${CC}" bash "${ROOT}/tools/c/sections/recover.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] bcir-cc WRITE-guard: an OOB store fails-fast, never clamps (#writeguard)"
CC="${CC}" bash "${ROOT}/tools/c/sections/writeguard.sh" "${tmp}/bcir-cc" || exit 1

# Atomic OOB ring (#atomicring, §5.12): the counter and payload publication must both be synchronized.
# An atomic fetch-add alone gives writers distinct slots but still lets a reporter race a non-atomic struct
# update (C undefined behaviour / torn audit records). The hosted implementation serializes each short
# publish/snapshot section. This test hammers writers while a reporter snapshots concurrently, then asserts
# the total equals N*M. If the freestanding build has no atomics the contract is single-threaded and this
# hosted concurrency test is skipped.
echo "[c-runtime] OOB ring counter is atomic under concurrent events (#atomicring)"
{ echo '#include <stdio.h>'; echo '#include "bcir_quarantine.h"'
  cat <<'DRV'
#if BCIR_OOB_COUNTER_ATOMIC
#include <pthread.h>
#define NT 8
#define NM 50000
static void *hammer(void *arg){ (void)arg;
  for(long k=0;k<NM;k++) bcir_oob_record_event(1u, (uint64_t)k, 64u, "race:a");
  return 0; }
static void *observe(void *arg){ (void)arg; FILE *f=tmpfile(); if(!f) return (void *)1;
  for(int k=0;k<200;k++){ bcir_quarantine_report(f); rewind(f); }
  return fclose(f) ? (void *)1 : 0; }
int main(void){
  pthread_t th[NT], reader;
  bcir_quarantine_report(NULL); bcir_decide_report(NULL); /* public null handles are harmless */
  if(pthread_create(&reader,0,observe,0)) return 2;
  for(int i=0;i<NT;i++) pthread_create(&th[i],0,hammer,0);
  for(int i=0;i<NT;i++) pthread_join(th[i],0);
  void *reader_rc=0; pthread_join(reader,&reader_rc); if(reader_rc) return 3;
  unsigned long got = bcir_oob_count, want = (unsigned long)NT*NM;
  printf(got==want ? "EXACT %lu\n" : "LOST %lu/%lu\n", got, want);
  return got==want ? 0 : 1; }
#else
int main(void){ printf("SKIP (no atomics; single-threaded contract)\n"); return 0; }
#endif
DRV
} > "${tmp}/race_main.c"
"${CC}" -std=c23 -O2 -pthread -I "${C}" "${tmp}/race_main.c" "${C}/bcir_quarantine.c" -o "${tmp}/race_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 -pthread -I "${C}" "${tmp}/race_main.c" "${C}/bcir_quarantine.c" -o "${tmp}/race_h" \
  || { echo "  FAIL: atomic-ring test build"; exit 1; }
race_out="$("${tmp}/race_h")"; race_rc=$?
{ [ "${race_rc}" = "0" ] && { printf '%s' "${race_out}" | grep -q "^EXACT" || printf '%s' "${race_out}" | grep -q "^SKIP"; }; } \
  && echo "  PASS atomicring: ${race_out} (concurrent OOB events do not scramble the total)" \
  || { echo "  FAIL: OOB ring counter raced (${race_out})"; exit 1; }

# rid->extent tamper-evidence (#extentassert, §5.12): the guard trusts the inline `n`, and the freestanding
# unit has NO registry to resolve `rid`->extent (a DOCUMENTED limit; see bcir_quarantine.h). The feasible
# lightweight check, for a KNOWN-EXTENT array, ties `n` to the array's true storage via a COMPILE-TIME
# BCIR_EXTENT_ASSERT(arr, n) == _Static_assert(n == sizeof(arr)/sizeof(arr[0])): a correct extent compiles,
# a TAMPERED `n` fails to compile -- so the extent is tamper-evident with no runtime cost and no registry.
echo "[c-runtime] rid->extent tamper-evidence: BCIR_EXTENT_ASSERT (freestanding lightweight check) (#extentassert)"
{ echo '#include "bcir_quarantine.h"'; echo 'static unsigned a[8];'
  echo 'int main(void){ BCIR_EXTENT_ASSERT(a, 8u); return (int)a[0]; }'; } > "${tmp}/ext_ok.c"
"${CC}" -std=c23 -I "${C}" "${tmp}/ext_ok.c" "${C}/bcir_quarantine.c" -o "${tmp}/ext_ok" 2>/dev/null \
  || "${CC}" -std=c2x -I "${C}" "${tmp}/ext_ok.c" "${C}/bcir_quarantine.c" -o "${tmp}/ext_ok" \
  || { echo "  FAIL: BCIR_EXTENT_ASSERT rejected a CORRECT extent"; exit 1; }
{ echo '#include "bcir_quarantine.h"'; echo 'static unsigned a[8];'
  echo 'int main(void){ BCIR_EXTENT_ASSERT(a, 99u); return (int)a[0]; }'; } > "${tmp}/ext_bad.c"
if "${CC}" -std=c23 -I "${C}" "${tmp}/ext_bad.c" "${C}/bcir_quarantine.c" -o "${tmp}/ext_bad" 2>/dev/null; then
  echo "  FAIL: BCIR_EXTENT_ASSERT did NOT catch a tampered extent (n=99 vs storage 8)"; exit 1
fi
echo "  PASS extentassert: correct extent compiles; tampered n=99 fails to compile (tamper-evident) (#extentassert)"

echo "[c-runtime] Clang-style diagnostic renderer (bcir_diag): caret layout == oracle (#diag)"
"${CC}" -std=c23 -O2 -Wall -Wextra -I "${C}" "${C}/bcir_diag.c" "${C}/test_diag.c" -o "${tmp}/test_diag" 2>/dev/null \
  || "${CC}" -std=c11 -O2 -I "${C}" "${C}/bcir_diag.c" "${C}/test_diag.c" -o "${tmp}/test_diag" \
  || { echo "  FAIL: bcir_diag build"; exit 1; }
bash "${ROOT}/tools/c/sections/diag.sh" "${tmp}/test_diag" || exit 1

echo "[c-runtime] fallback contract (bcir-cc --fallback): route-to-LLVM decision == oracle (#fallback)"
bash "${ROOT}/tools/c/sections/fallback.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] constant-expression evaluator: enum/case/static folds == reference (#cexpr)"
CC="${CC}" bash "${ROOT}/tools/c/sections/cexpr.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] R21 lifetime policy (bcir-cc --r21): verdict + exit-code parity vs the oracle (#r21policy)"
bash "${ROOT}/tools/c/sections/r21policy.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] project mode (bcir-cc multi-file): verdict line + exit code == oracle (#project)"
bash "${ROOT}/tools/c/sections/project.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] Phase 3 linking (bcir-cc prototypes): emitted caller + host linker == reference (#link)"
CC="${CC}" bash "${ROOT}/tools/c/sections/link.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] effect footprint + escape analysis (bcir-cc --emit-effects/--emit-escape) == oracle over the corpus, generated units and forms (#effects, G10)"
bash "${ROOT}/tools/c/sections/effects.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] automatic link-flag emission (bcir-cc --emit-link-flags): derived flags == oracle (#linkflags)"
bash "${ROOT}/tools/c/sections/linkflags.sh" "${tmp}/bcir-cc" || exit 1
echo "[c-runtime] B2 FFTW link-flag rule (bcir_cfront_link_flags twin): fftwf_* -> -lfftw3 (#linkflags-fftw)"
"${CC}" -std=c23 -O2 -Wall -Wextra -I "${C}" "${C}/test_link_flag_rules.c" "${C}/bcir_cfront.c" "${C}/bcir_cpp.c" \
  "${C}/bcir_verify.c" "${C}/bcir_runtime.c" -o "${tmp}/test_link_flag_rules" 2>/dev/null \
  || "${CC}" -std=c11 -O2 -I "${C}" "${C}/test_link_flag_rules.c" "${C}/bcir_cfront.c" "${C}/bcir_cpp.c" \
       "${C}/bcir_verify.c" "${C}/bcir_runtime.c" -o "${tmp}/test_link_flag_rules" \
  || { echo "  FAIL: link-flag rules harness build"; exit 1; }
bash "${ROOT}/tools/c/sections/linkflags_fftw.sh" "${tmp}/test_link_flag_rules" || exit 1

echo "[c-runtime] LAPACK link-flag rule (bcir_cfront_link_flags twin): LAPACKE_*/sgesv_ -> -llapack (#linkflags-lapack)"
bash "${ROOT}/tools/c/sections/linkflags_lapack.sh" "${tmp}/test_link_flag_rules" || exit 1

echo "[c-runtime] E1 OLS portable fallback (emit_lapack_ols_c): normal-equations recovers a known x (#ols)"
# Emit the fallback OLS kernel from the oracle, append a main that fits y = 2x + 1 (m=8, n=2 -> [1, x] rows,
# recovers [c0=1, c1=2]), and assert the recovered coefficients match to float round-off.
python3 "${ROOT}/tools/build/emit_kernel.py" kernel_ols > "${tmp}/ols_kernel.c" || { echo "  FAIL: python OLS emit"; exit 1; }
"${CC}" -std=c11 -O2 -Wall -Wextra "${tmp}/ols_kernel.c" -lm -o "${tmp}/ols" 2>/dev/null \
  || "${CC}" -std=c23 -O2 "${tmp}/ols_kernel.c" -lm -o "${tmp}/ols" \
  || { echo "  FAIL: OLS fallback build"; exit 1; }
bash "${ROOT}/tools/c/sections/ols.sh" "${tmp}/ols" || exit 1

echo "[c-runtime] E2 PCA portable fallback (emit_lapack_eigh_c): Jacobi recovers a known spectrum (#pca)"
python3 "${ROOT}/tools/build/emit_kernel.py" kernel_pca > "${tmp}/eigh_kernel.c" || { echo "  FAIL: python PCA eigh emit"; exit 1; }
"${CC}" -std=c11 -O2 -Wall -Wextra "${tmp}/eigh_kernel.c" -lm -o "${tmp}/eigh" 2>/dev/null \
  || "${CC}" -std=c23 -O2 "${tmp}/eigh_kernel.c" -lm -o "${tmp}/eigh" \
  || { echo "  FAIL: PCA eigh fallback build"; exit 1; }
bash "${ROOT}/tools/c/sections/pca.sh" "${tmp}/eigh" || exit 1

echo "[c-runtime] E3 Transformer layernorm (emit_layernorm_c): per-row normalize to mean~0/var~1 (#layernorm)"
python3 "${ROOT}/tools/build/emit_kernel.py" kernel_layernorm > "${tmp}/ln_kernel.c" || { echo "  FAIL: python layernorm emit"; exit 1; }
"${CC}" -std=c11 -O2 -Wall -Wextra "${tmp}/ln_kernel.c" -lm -o "${tmp}/ln" 2>/dev/null \
  || "${CC}" -std=c23 -O2 "${tmp}/ln_kernel.c" -lm -o "${tmp}/ln" \
  || { echo "  FAIL: layernorm kernel build"; exit 1; }
bash "${ROOT}/tools/c/sections/layernorm.sh" "${tmp}/ln" || exit 1

echo "[c-runtime] E4 recurrent LSTM cell (emit_lstm_cell_c): 1x1 cell matches hand-computed forward (#lstm)"
python3 "${ROOT}/tools/build/emit_kernel.py" kernel_lstm > "${tmp}/lstm_kernel.c" || { echo "  FAIL: python lstm emit"; exit 1; }
"${CC}" -std=c11 -O2 -Wall -Wextra "${tmp}/lstm_kernel.c" -lm -o "${tmp}/lstm" 2>/dev/null \
  || "${CC}" -std=c23 -O2 "${tmp}/lstm_kernel.c" -lm -o "${tmp}/lstm" \
  || { echo "  FAIL: lstm kernel build"; exit 1; }
bash "${ROOT}/tools/c/sections/lstm.sh" "${tmp}/lstm" || exit 1

echo "[c-runtime] E5 classical-ML RBF-SVM predict (emit_svm_rbf_predict_c): decision function matches reference (#classical #svm)"
python3 "${ROOT}/tools/build/emit_kernel.py" kernel_svm > "${tmp}/svm_rbf_kernel.c" || { echo "  FAIL: python svm-rbf emit"; exit 1; }
"${CC}" -std=c11 -O2 -Wall -Wextra "${tmp}/svm_rbf_kernel.c" -lm -o "${tmp}/svm_rbf" 2>/dev/null \
  || "${CC}" -std=c23 -O2 "${tmp}/svm_rbf_kernel.c" -lm -o "${tmp}/svm_rbf" \
  || { echo "  FAIL: svm-rbf kernel build"; exit 1; }
bash "${ROOT}/tools/c/sections/svm.sh" "${tmp}/svm_rbf" || exit 1

echo "[c-runtime] E5 classical-ML decision-tree predict (emit_tree_predict_c): exact threshold traversal, NO libm (#classical #tree)"
python3 "${ROOT}/tools/build/emit_kernel.py" kernel_tree > "${tmp}/tree_kernel.c" || { echo "  FAIL: python tree emit"; exit 1; }
# the tree kernel is EXACT -- no libm needed (we still link -lm only for the harness fabsf-free main; not required).
"${CC}" -std=c11 -O2 -Wall -Wextra "${tmp}/tree_kernel.c" -o "${tmp}/tree" 2>/dev/null \
  || "${CC}" -std=c23 -O2 "${tmp}/tree_kernel.c" -o "${tmp}/tree" \
  || { echo "  FAIL: tree kernel build"; exit 1; }
bash "${ROOT}/tools/c/sections/tree.sh" "${tmp}/tree" || exit 1

echo "[c-runtime] E6 unsupervised K-means assign (emit_kmeans_assign_c): nearest-centroid argmin, NO libm (#kmeans)"
python3 "${ROOT}/tools/build/emit_kernel.py" kernel_kmeans > "${tmp}/kmeans_kernel.c" || { echo "  FAIL: python kmeans emit"; exit 1; }
# the K-means assign kernel is EXACT -- no libm needed (no -lm; the EXACT half of the Area-B pattern).
"${CC}" -std=c11 -O2 -Wall -Wextra "${tmp}/kmeans_kernel.c" -o "${tmp}/kmeans" 2>/dev/null \
  || "${CC}" -std=c23 -O2 "${tmp}/kmeans_kernel.c" -o "${tmp}/kmeans" \
  || { echo "  FAIL: kmeans assign kernel build"; exit 1; }
bash "${ROOT}/tools/c/sections/kmeans.sh" "${tmp}/kmeans" || exit 1

echo "[c-runtime] GSL link-flag rule (bcir_cfront_link_flags twin): gsl_* -> -lgsl (#linkflags-gsl)"
bash "${ROOT}/tools/c/sections/linkflags_gsl.sh" "${tmp}/test_link_flag_rules" || exit 1

echo "[c-runtime] SLEEF link-flag rule (bcir_cfront_link_flags twin): Sleef_* -> -lsleef (#linkflags-sleef)"
bash "${ROOT}/tools/c/sections/linkflags_sleef.sh" "${tmp}/test_link_flag_rules" || exit 1

echo "[c-runtime] libcerf link-flag rule (bcir_cfront_link_flags twin): erfcx* -> -lcerf (#linkflags-cerf)"
bash "${ROOT}/tools/c/sections/linkflags_cerf.sh" "${tmp}/test_link_flag_rules" || exit 1

echo "[c-runtime] scalable IR (bcir-cc, no fixed BCIR_MAX_*): cap-busting unit compiles + matches oracle (#scale)"
bash "${ROOT}/tools/c/sections/scale.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] integer promotions + UAC (bcir-cc): signed/mixed-width emit == Clang full-range (#intpromote)"
CC="${CC}" bash "${ROOT}/tools/c/sections/intpromote.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] designated initializers (bcir-cc): dispatch-table emit == Clang (#designated)"
CC="${CC}" bash "${ROOT}/tools/c/sections/designated.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] local aggregate init (bcir-cc): struct/union {.field=v} emit == Clang (#aggregate)"
CC="${CC}" bash "${ROOT}/tools/c/sections/aggregate.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] scalable parser state (bcir-cc, no fixed gv/s/td/ec/env caps): cap-busting unit == oracle (#pscale)"
bash "${ROOT}/tools/c/sections/pscale.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] array element stores (bcir-cc): a[i]=v emit == Clang (#astore)"
CC="${CC}" bash "${ROOT}/tools/c/sections/astore.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] extern linkage (bcir-cc): extern global emit == Clang (#extern)"
CC="${CC}" bash "${ROOT}/tools/c/sections/extern.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] thread-local storage (bcir-cc): _Thread_local emit == Clang (#threadlocal)"
CC="${CC}" bash "${ROOT}/tools/c/sections/threadlocal.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] multi-declarator locals (bcir-cc): emit == Clang (#multidecl)"
CC="${CC}" bash "${ROOT}/tools/c/sections/multidecl.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] comma-operator for-step (bcir-cc): emit == Clang (#commastep)"
CC="${CC}" bash "${ROOT}/tools/c/sections/commastep.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] multi-declarator struct members (bcir-cc): layout + access == Clang (#structmulti)"
CC="${CC}" bash "${ROOT}/tools/c/sections/structmulti.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] empty statements (bcir-cc): emit == Clang (#emptystmt)"
CC="${CC}" bash "${ROOT}/tools/c/sections/emptystmt.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] store through a pointer (bcir-cc): emit == Clang (#ptrstore)"
CC="${CC}" bash "${ROOT}/tools/c/sections/ptrstore.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] nested struct member access (bcir-cc): layout + access == Clang (#nestmember)"
CC="${CC}" bash "${ROOT}/tools/c/sections/nestmember.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] reused loop-counter names (bcir-cc): unique emit == Clang (#loopreuse)"
CC="${CC}" bash "${ROOT}/tools/c/sections/loopreuse.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] for-loop variable scope (bcir-cc): emit == Clang (#loopscope)"
CC="${CC}" bash "${ROOT}/tools/c/sections/loopscope.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] bare-block variable scope (bcir-cc): emit == Clang (#blockscope)"
CC="${CC}" bash "${ROOT}/tools/c/sections/blockscope.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] struct member arrays (bcir-cc): layout + access == Clang (#memberarray)"
CC="${CC}" bash "${ROOT}/tools/c/sections/memberarray.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] multi-dimensional local arrays (bcir-cc): emit == Clang (#localmd)"
CC="${CC}" bash "${ROOT}/tools/c/sections/localmd.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] the 'signed' type specifier (bcir-cc): emit == Clang (#signedty)"
CC="${CC}" bash "${ROOT}/tools/c/sections/signedty.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] signed comparison vs literal (bcir-cc): emit == Clang (#signedcmp)"
CC="${CC}" bash "${ROOT}/tools/c/sections/signedcmp.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] unary on a wide operand (bcir-cc): emit == Clang (#longunary)"
CC="${CC}" bash "${ROOT}/tools/c/sections/longunary.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] store into a _Bool normalizes to 0/1 (bcir-cc): emit == Clang (#boolnorm)"
CC="${CC}" bash "${ROOT}/tools/c/sections/boolnorm.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] integer-promotion + float unary -/~ (bcir-cc): emit == Clang (#unarypromote)"
CC="${CC}" bash "${ROOT}/tools/c/sections/unarypromote.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] float -> signed-int cast (bcir-cc): emit == Clang (#floatsigncast)"
CC="${CC}" bash "${ROOT}/tools/c/sections/floatsigncast.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] integer cast to a signed type (bcir-cc): emit == Clang (#intsigncast)"
CC="${CC}" bash "${ROOT}/tools/c/sections/intsigncast.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] cast to _Bool normalizes the full value (bcir-cc): emit == Clang (#boolcast)"
CC="${CC}" bash "${ROOT}/tools/c/sections/boolcast.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] signed bitfield read sign-extends (bcir-cc): emit == Clang (#signedbf)"
CC="${CC}" bash "${ROOT}/tools/c/sections/signedbf.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] signed sub-int storage read sign-extends (bcir-cc): emit == Clang (#signedload)"
CC="${CC}" bash "${ROOT}/tools/c/sections/signedload.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] enum used as a type (bcir-cc): emit == Clang (#enumtype)"
CC="${CC}" bash "${ROOT}/tools/c/sections/enumtype.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] pointer locals -- T *p = &x (bcir-cc): emit == Clang (#ptrlocal)"
CC="${CC}" bash "${ROOT}/tools/c/sections/ptrlocal.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] pointer values -- return p + i (bcir-cc): emit == Clang (#ptrvalue)"
CC="${CC}" bash "${ROOT}/tools/c/sections/ptrvalue.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] pointer struct fields -- s->p = q (bcir-cc): emit == Clang (#ptrfield)"
CC="${CC}" bash "${ROOT}/tools/c/sections/ptrfield.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] pointer-to-pointer -- int **pp, **pp (bcir-cc): emit == Clang (#ptr2ptr)"
CC="${CC}" bash "${ROOT}/tools/c/sections/ptr2ptr.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] deref-through a loaded pointer field -- *(s->p) / s->mid->leaf->x / s->p[i] (#fieldderef)"
CC="${CC}" bash "${ROOT}/tools/c/sections/fieldderef.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] pointer-element signedness -- signed vs unsigned pointee load/store (#ptrsign)"
CC="${CC}" bash "${ROOT}/tools/c/sections/ptrsign.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] funcptr dispatch through a loaded pointer -- d->ops->fn(args) (#fnptrchain)"
CC="${CC}" bash "${ROOT}/tools/c/sections/fnptrchain.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] per-declarator pointer/array in a multi-declarator decl -- int *p, q; (#multiptr)"
CC="${CC}" bash "${ROOT}/tools/c/sections/multiptr.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] faithful char types -- char / signed char / unsigned char (#chartypes)"
CC="${CC}" bash "${ROOT}/tools/c/sections/chartypes.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] compound literals -- (type){init} by value / scalar / &(literal) / .field (#complit)"
CC="${CC}" bash "${ROOT}/tools/c/sections/complit.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] typeof -- typeof(type-name) / typeof(variable) / typeof(expr) (#typeof)"
CC="${CC}" bash "${ROOT}/tools/c/sections/typeof.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] variadic functions -- va_list / va_start / va_arg / va_end / va_copy (#variadic)"
CC="${CC}" bash "${ROOT}/tools/c/sections/variadic.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] wide/float compound assignment -- OP= / ++ / -- keeps width (#compoundwide)"
CC="${CC}" bash "${ROOT}/tools/c/sections/compoundwide.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] external variadic calls -- snprintf / vsnprintf passthrough (#extvariadic)"
CC="${CC}" bash "${ROOT}/tools/c/sections/extvariadic.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] long double -- 80/128-bit extended float, +l libm variants (#longdouble)"
CC="${CC}" bash "${ROOT}/tools/c/sections/longdouble.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] _Generic -- C11 type-generic selection (#generic)"
CC="${CC}" bash "${ROOT}/tools/c/sections/generic.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] nested designated initializers -- .a.b / .v[i] / .m[i][j] (#designate)"
CC="${CC}" bash "${ROOT}/tools/c/sections/designate.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] nested member access at a non-first offset (#nestoffset)"
CC="${CC}" bash "${ROOT}/tools/c/sections/nestoffset.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] address-of a (nested) struct member -- &s.f / &t.q.a / &t.q (#addrmember)"
CC="${CC}" bash "${ROOT}/tools/c/sections/addrmember.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] _Atomic local objects -- _Atomic int / _Atomic(int) (#atomiclocal)"
CC="${CC}" bash "${ROOT}/tools/c/sections/atomiclocal.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] GCC/Clang integer builtins -- popcount/clz/ctz/bswap/abs (#builtins)"
CC="${CC}" bash "${ROOT}/tools/c/sections/builtins.sh" "${tmp}/bcir-cc" || exit 1

echo "[c-runtime] GCC statement expressions -- ({ ...; e; }) (#stmtexpr)"
CC="${CC}" bash "${ROOT}/tools/c/sections/stmtexpr.sh" "${tmp}/bcir-cc" || exit 1

# Inline assembly (#inlineasm): GNU `asm`/`__asm__` is an ISA-NEUTRAL trusted opaque effect edge (ASM1) --
# the template is re-emitted VERBATIM as a `__asm__ [__volatile__] (...)` statement; BCIR owns only the
# calling side (operands + constraints + clobbers + ordering). This is a PYTHON-frontend feature (the C twin
# bcir_cfront.c does not parse asm yet), so the probe emits via `compile_unit` (not bcir-cc --emit-c). It
# proves the emit compiles under BOTH the default CC and -- when present -- a second compiler, AND that an
# `asm volatile("" ::: "memory")` compiler barrier wrapped around a store/load does NOT change the observable
# value (it only constrains ordering). Uses the reserved __asm__/__volatile__ spellings + the ISA-neutral
# `"=r"(out) : "0"(in)` tied-register copy + empty-template `"memory"` barrier, so it compiles on any ISA.
echo "[c-runtime] inline assembly: ISA-neutral trusted opaque edge, verbatim emit + memory barrier (#inlineasm)"
cat > "${tmp}/cfront_asm.c" <<'ASMC'
unsigned asm_copy(unsigned x){ unsigned y = 0; __asm__("" : "=r"(y) : "0"(x)); return y; }
unsigned asm_barrier(unsigned *p, unsigned x){
  *p = x;                                  /* store ... */
  __asm__ __volatile__("" ::: "memory");   /* an ordering fence between the store and the load */
  return *p;                               /* ... load: the barrier must NOT change the value (== x) */
}
void asm_basic(void){ __asm__("nop"); __asm__ __volatile__("" ::: "memory"); }
ASMC
FX="${tmp}/cfront_asm.c" python3 - > "${tmp}/asm_emit.c" <<'PY' || { echo "  FAIL: python asm emit"; exit 1; }
import os, re
from bcir.frontends.cfront import compile_unit
from bcir.frontends.cfront.emit import emit_function
r = compile_unit(open(os.environ['FX']).read(), check_clang=False)
assert r.is_clean, [(d.law, d.message) for d in r.diagnostics]
out = []
for lf in r.lowered.functions.values():
    t = re.sub(r"/\*.*?\*/\n?", "", emit_function(lf), flags=re.S)   # drop the attestation comment
    assert "__asm__" in t, "the asm edge was eliminated from the emit"
    out.append(t)
print("\n\n".join(out))
PY
grep -q '__asm__ __volatile__ ("" :  :  : "memory")' "${tmp}/asm_emit.c" \
  && echo "  PASS inlineasm: emit carries the verbatim __asm__ __volatile__ memory barrier" \
  || { echo "  FAIL: inlineasm emit missing the verbatim barrier"; cat "${tmp}/asm_emit.c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  cat "${tmp}/asm_emit.c"
  cat <<'DRV'
int main(void){
  unsigned buf = 0;
  for(unsigned x=0; x<5000u; x++){
    if(bcir_asm_copy(x) != x){ printf("COPY@%u\n", x); return 1; }
    if(bcir_asm_barrier(&buf, x) != x){ printf("BARRIER@%u\n", x); return 1; }
  }
  bcir_asm_basic();
  puts("MATCH"); return 0;
}
DRV
} > "${tmp}/asm_harness.c"
asm_ok=1; asm_seen=""
for cc in "${CC}" "$(command -v gcc)" "$(command -v clang)"; do
  [ -n "${cc}" ] && [ -x "${cc}" ] || continue
  case " ${asm_seen} " in *" ${cc} "*) continue;; esac     # de-dup (CC may already be gcc/clang)
  asm_seen="${asm_seen} ${cc}"
  "${cc}" -std=c11 -pedantic -O2 "${tmp}/asm_harness.c" -o "${tmp}/asm_h" 2>/dev/null \
    || { echo "  FAIL: inlineasm harness build (${cc})"; asm_ok=0; break; }
  ar="$("${tmp}/asm_h")"
  [ "${ar}" = "MATCH" ] || { echo "  FAIL: inlineasm behaviour (${cc}: ${ar})"; asm_ok=0; break; }
done
[ "${asm_ok}" = "1" ] \
  && echo "  PASS inlineasm: __asm__ copy + memory-barrier store/load == value-preserving (ISA-neutral, every CC)" \
  || exit 1

# Port-mapped I/O intrinsics (#portio, ASM2): inb/inw/inl / outb/outw/outl lower to a typed, isolated,
# BARRIERED I/O-port edge; the per-ISA `in`/`out` instruction is emitted behind `--target` (x86 emits the
# real instruction as `__asm__ __volatile__`; non-x86 has no port I/O -> the honest unsupported diagnostic).
# HONEST BOUNDARY: executing `in`/`out` from userspace TRAPS (it needs iopl/ioperm + ring-0), so this probe
# ASSEMBLES the emitted asm (`-c`, assemble-only) to prove it is valid x86 the toolchain accepts -- it NEVER
# links or runs it. The emitted text must carry the right `in`/`out` instruction + operands.
echo "[c-runtime] port-mapped I/O intrinsics: typed barriered edge, x86 in/out emit, assemble-only (#portio)"
cat > "${tmp}/cfront_portio.c" <<'PIOC'
unsigned pio_inb(unsigned port){ return inb(port); }      /* read u8  from a port */
unsigned pio_inw(unsigned port){ return inw(port); }      /* read u16 */
unsigned pio_inl(unsigned port){ return inl(port); }      /* read u32 */
void pio_outb(unsigned v, unsigned port){ outb(v, port); } /* Linux out(value, port): value first, port second */
void pio_outw(unsigned v, unsigned port){ outw(v, port); }
void pio_outl(unsigned v, unsigned port){ outl(v, port); }
PIOC
FX="${tmp}/cfront_portio.c" python3 - > "${tmp}/portio_emit.c" <<'PY' || { echo "  FAIL: python portio emit"; exit 1; }
import os, re
from bcir.frontends.cfront import compile_unit
from bcir.frontends.cfront.emit import emit_function
from bcir.model import Domain
r = compile_unit(open(os.environ['FX']).read(), check_clang=False)   # default target x86_64-linux
assert r.is_clean, [(d.law, d.message) for d in r.diagnostics]
# the edges are typed + isolated + barriered (the IR-level contract): MMIO domain, barriered hazard.
pio = [c for lf in r.lowered.functions.values() for c in lf.claims if c.op.startswith('c.portio.')]
assert len(pio) == 6, f"expected 6 port-I/O edges, got {len(pio)}"
assert all(c.hazard == 'barriered' for c in pio), "a port-I/O edge is not barriered"
assert all(c.domain == Domain.MMIO for c in pio), "a port-I/O edge is not isolated in the MMIO I/O domain"
out = ["#include <stdint.h>"]
for lf in r.lowered.functions.values():
    t = re.sub(r"/\*.*?\*/\n?", "", emit_function(lf), flags=re.S)   # drop the attestation comment
    assert "__asm__ __volatile__" in t, "the port-I/O edge was eliminated from the emit"
    out.append(t)
print("\n\n".join(out))
PY
# the emit carries the real x86 in/out instructions + the standard <asm/io.h> operand constraints.
{ grep -q '"inb %w1, %b0" : "=a"' "${tmp}/portio_emit.c" \
  && grep -q '"outb %b0, %w1" :  : "a"' "${tmp}/portio_emit.c" \
  && grep -q '"inl %w1, %k0" : "=a"' "${tmp}/portio_emit.c" \
  && grep -q '"Nd"' "${tmp}/portio_emit.c"; } \
  && echo "  PASS portio: emit carries the real x86 inb/outb/inl + the <asm/io.h> =a/a/Nd constraints" \
  || { echo "  FAIL: portio emit missing the real in/out instruction"; cat "${tmp}/portio_emit.c"; exit 1; }
# ASSEMBLE-ONLY (-c): prove the emitted x86 asm is valid the toolchain accepts. NEVER linked/run (the
# `in`/`out` instructions are privileged). Every available CC must assemble it.
# The emitted asm is x86 `in`/`out`, which a non-x86 host's native assembler cannot accept (and a clang
# cross-compile `-target x86_64-linux-gnu` cannot satisfy without an x86 sysroot the ARM CI lane lacks). So
# the assemble check runs ONLY on an x86 host and SKIPS on a non-x86 host (e.g. the aarch64 lane) -- the
# emit-text check above is arch-independent, and the x86 asm's validity is proven on the x86 CI lanes.
case "$(uname -m)" in x86_64|amd64|x86|i386|i486|i586|i686) pio_host_x86=1;; *) pio_host_x86=0;; esac
if [ "${pio_host_x86}" = "1" ]; then
  pio_ok=1; pio_seen=""
  for cc in "${CC}" "$(command -v gcc)" "$(command -v clang)"; do
    [ -n "${cc}" ] && [ -x "${cc}" ] || continue
    case " ${pio_seen} " in *" ${cc} "*) continue;; esac   # de-dup (CC may already be gcc/clang)
    pio_seen="${pio_seen} ${cc}"
    "${cc}" -std=c11 -pedantic -c "${tmp}/portio_emit.c" -o "${tmp}/portio_${cc##*/}.o" 2>/dev/null \
      || { echo "  FAIL: portio emit did not ASSEMBLE under ${cc}"; pio_ok=0; break; }
    [ -f "${tmp}/portio_${cc##*/}.o" ] || { echo "  FAIL: portio object not produced by ${cc}"; pio_ok=0; break; }
  done
  [ "${pio_ok}" = "1" ] \
    && echo "  PASS portio: emitted x86 in/out ASSEMBLES under every CC (-c, assemble-only; execution is privileged)" \
    || exit 1
else
  echo "  SKIP portio assemble: non-x86 host cannot assemble x86 in/out (validity proven on the x86 lanes; emit-text + fallback checks ran)"
fi
# the non-x86 honest diagnostic: ARM/RISC-V have no port I/O -> route to the LLVM fallback (the oracle).
pio_fb="$(python3 -c "
from bcir.frontends.cfront.pipeline import compile_with_fallback
r = compile_with_fallback('unsigned f(void){ return inb(0x60); }', check_clang=False, target='aarch64-linux')
print('FB' if (r.needs_fallback and 'requires an x86 target' in r.fallback) else 'NO')")" \
  || { echo "  FAIL: portio non-x86 fallback probe"; exit 1; }
[ "${pio_fb}" = "FB" ] \
  && echo "  PASS portio: a non-x86 target (aarch64) has no port I/O -> honest unsupported diagnostic (LLVM fallback)" \
  || { echo "  FAIL: portio non-x86 did not fall back honestly"; exit 1; }

# Memory-fence (hardware barrier) intrinsics (#barrier, ASM3): __sync_synchronize / __atomic_thread_fence /
# the C11 atomic_thread_fence (all FULL/seq_cst, op `c.fence`) + the x86-conventional _mm_mfence (full) /
# _mm_lfence (acquire) / _mm_sfence (release) lower to a typed, KINDED, barriered BARRIER edge; the per-ISA
# instruction is emitted behind `--target` (x86 mfence/lfence/sfence; aarch64 dmb ish/ishld/ishst; riscv64
# fence rw,rw / r,rw / rw,w), always with the required `"memory"` compiler-barrier clobber. Every ISA has a
# fence, so a target outside the three families keeps the portable __atomic_thread_fence default (no
# unsupported-diagnostic path -- unlike port I/O). The emit text is arch-independent (checked on every host);
# the emitted NATIVE-arch barrier is ASSEMBLED on its own lane (x86 assembles mfence/lfence/sfence; the
# aarch64 lane assembles dmb ish/ishld/ishst) -- non-native emits are NOT assembled (no cross sysroot).
echo "[c-runtime] memory-fence intrinsics: typed kinded barriered edge, per-ISA emit, native assemble (#barrier)"
# the native target for THIS host's arch (the only one we can assemble without a cross sysroot).
case "$(uname -m)" in
  x86_64|amd64|x86|i386|i486|i586|i686) brr_native="x86_64-linux"; brr_family="x86";;
  aarch64|arm64) brr_native="aarch64-linux"; brr_family="arm";;
  riscv64) brr_native="riscv64-linux"; brr_family="riscv";;
  *) brr_native=""; brr_family="";;
esac
cat > "${tmp}/cfront_barrier.c" <<'BRRC'
void b_full(void){ __sync_synchronize(); }          /* full (seq_cst) fence -> c.fence */
void b_full11(void){ atomic_thread_fence(5); }       /* C11 <stdatomic.h> full fence -> c.fence */
void b_mfence(void){ _mm_mfence(); }                 /* x86 mfence -> c.fence (full) */
void b_lfence(void){ _mm_lfence(); }                 /* x86 lfence -> c.fence.acquire (load fence) */
void b_sfence(void){ _mm_sfence(); }                 /* x86 sfence -> c.fence.release (store fence) */
BRRC
# emit for the NATIVE target (so the asm assembles); a host outside the three families emits x86_64-linux for
# the TEXT checks but assembles nothing.
BRR_TARGET="${brr_native:-x86_64-linux}" FX="${tmp}/cfront_barrier.c" python3 - > "${tmp}/barrier_emit.c" <<'PY' || { echo "  FAIL: python barrier emit"; exit 1; }
import os, re
from bcir.frontends.cfront import compile_unit
from bcir.frontends.cfront.emit import emit_function
from bcir.model import Opcode
r = compile_unit(open(os.environ['FX']).read(), check_clang=False, target=os.environ['BRR_TARGET'])
assert r.is_clean, [(d.law, d.message) for d in r.diagnostics]
# the edges are typed + kinded + barriered (the IR-level contract): BARRIER opcode, barriered hazard.
fen = [c for lf in r.lowered.functions.values() for c in lf.claims if c.op == 'c.fence' or c.op.startswith('c.fence.')]
assert len(fen) == 5, f"expected 5 fence edges, got {len(fen)}"
assert all(c.hazard == 'barriered' for c in fen), "a fence edge is not barriered"
assert all(c.opcode == Opcode.BARRIER for c in fen), "a fence edge is not the BARRIER opcode"
ops = sorted({c.op for c in fen})
assert ops == ['c.fence', 'c.fence.acquire', 'c.fence.release'], f"unexpected fence op set: {ops}"
out = ["#include <stdint.h>"]
for lf in r.lowered.functions.values():
    t = re.sub(r"/\*.*?\*/\n?", "", emit_function(lf), flags=re.S)   # drop the attestation comment
    assert '__asm__ __volatile__' in t and ':::' in t and '"memory"' in t, "the fence edge lost its barrier emit"
    out.append(t)
print("\n\n".join(out))
PY
# the emit carries the per-ISA fence mnemonics + the required "memory" compiler-barrier clobber, per family.
case "${brr_family}" in
  arm)   m_full="dmb ish"; m_acq="dmb ishld"; m_rel="dmb ishst";;
  riscv) m_full="fence rw,rw"; m_acq="fence r,rw"; m_rel="fence rw,w";;
  *)     m_full="mfence"; m_acq="lfence"; m_rel="sfence";;     # x86 + the default-text host
esac
{ grep -q "\"${m_full}\" ::: \"memory\"" "${tmp}/barrier_emit.c" \
  && grep -q "\"${m_acq}\" ::: \"memory\"" "${tmp}/barrier_emit.c" \
  && grep -q "\"${m_rel}\" ::: \"memory\"" "${tmp}/barrier_emit.c"; } \
  && echo "  PASS barrier: emit carries the per-ISA ${m_full}/${m_acq}/${m_rel} + the required \"memory\" clobber" \
  || { echo "  FAIL: barrier emit missing the per-ISA fence instruction"; cat "${tmp}/barrier_emit.c"; exit 1; }
# ASSEMBLE-ONLY (-c) the NATIVE-arch fence: prove the emitted barrier is valid asm the toolchain accepts. The
# emit is native for THIS host (target = brr_native), so a native gcc/clang assembles it without a cross
# sysroot. Non-native fence emits are NOT assembled (the aarch64 lane has no x86 sysroot, and vice-versa) --
# their validity is proven on their own CI lane, the emit-text check above is arch-independent.
if [ -n "${brr_native}" ]; then
  brr_ok=1; brr_seen=""
  for cc in "${CC}" "$(command -v gcc)" "$(command -v clang)"; do
    [ -n "${cc}" ] && [ -x "${cc}" ] || continue
    case " ${brr_seen} " in *" ${cc} "*) continue;; esac       # de-dup (CC may already be gcc/clang)
    brr_seen="${brr_seen} ${cc}"
    "${cc}" -std=c11 -pedantic -c "${tmp}/barrier_emit.c" -o "${tmp}/barrier_${cc##*/}.o" 2>/dev/null \
      || { echo "  FAIL: barrier emit did not ASSEMBLE under ${cc}"; brr_ok=0; break; }
    [ -f "${tmp}/barrier_${cc##*/}.o" ] || { echo "  FAIL: barrier object not produced by ${cc}"; brr_ok=0; break; }
  done
  [ "${brr_ok}" = "1" ] \
    && echo "  PASS barrier: emitted native ${brr_family} fence ASSEMBLES under every CC (-c, assemble-only)" \
    || exit 1
else
  echo "  SKIP barrier assemble: host arch outside the x86/aarch64/riscv64 families (emit-text check ran)"
fi
# the non-native fence emit is arch-independent TEXT (no assemble needed): the aarch64 dmb-ish family + the
# riscv64 fence family are emitted correctly even off-arch, proving the per-ISA table is keyed off --target.
brr_text="$(python3 -c "
import re
from bcir.frontends.cfront import compile_unit
from bcir.frontends.cfront.emit import emit_function
def body(src, target):
    r = compile_unit(src, check_clang=False, target=target)
    lf = r.lowered.functions['f']
    return re.sub(r'/\*.*?\*/\n?', '', emit_function(lf), flags=re.S)
ok = True
ok &= 'dmb ish' in body('void f(void){ _mm_mfence(); }', 'aarch64-linux')
ok &= 'dmb ishld' in body('void f(void){ _mm_lfence(); }', 'aarch64-linux')
ok &= 'dmb ishst' in body('void f(void){ _mm_sfence(); }', 'aarch64-linux')
ok &= 'fence rw,rw' in body('void f(void){ _mm_mfence(); }', 'riscv64-linux')
ok &= 'fence r,rw' in body('void f(void){ _mm_lfence(); }', 'riscv64-linux')
ok &= 'fence rw,w' in body('void f(void){ _mm_sfence(); }', 'riscv64-linux')
print('OK' if ok else 'NO')")" \
  || { echo "  FAIL: barrier per-ISA text probe"; exit 1; }
[ "${brr_text}" = "OK" ] \
  && echo "  PASS barrier: the aarch64 dmb-ish + riscv64 fence families emit correctly off-arch (per-ISA --target keying)" \
  || { echo "  FAIL: barrier per-ISA off-arch emit text is wrong"; exit 1; }

# X.691 PER decoding primitives (#per, roadmap phase C): the C twin of clause 11. PER is
# NOT self-delimiting (X.691 7.2), so unlike the X.690 twin there is no schema-free
# structure walk -- clause 11's whole-number and length decoders ARE the schema-free layer,
# and they are the ones that take an attacker-supplied width, octet count or fragment header
# and move a cursor with it. The dual-rail differential lives in bcir/tests/test_c_per.py;
# what is checked HERE is the same discipline the other twins get: strict warnings as
# errors, and a genuinely freestanding translation unit.
echo "[c-runtime] X.691 PER primitives: strict-warning and freestanding build (#per)"
if "${CC}" -std=c23 -O2 -Wall -Wextra -Werror -I "${C}" \
     "${C}/bcir_per.c" "${C}/test_per.c" -o "${tmp}/test_per"; then
  for std in c11 c23; do
    "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -Werror -I "${C}" \
      -c "${C}/bcir_per.c" -o /dev/null \
      || { echo "  FAIL: bcir_per is not freestanding-clean under -std=${std}"; exit 1; }
  done
  # A decoder whose answers depend on the optimiser is not a decoder. Build the same twin
  # at -O0 and -O3 and require identical output on the same campaign: a signed-overflow or
  # shift-past-width bug typically only diverges at one of the two.
  per_ok=1
  "${CC}" -std=c23 -O0 -I "${C}" "${C}/bcir_per.c" "${C}/test_per.c" -o "${tmp}/test_per_O0" \
    || per_ok=0
  "${CC}" -std=c23 -O3 -I "${C}" "${C}/bcir_per.c" "${C}/test_per.c" -o "${tmp}/test_per_O3" \
    || per_ok=0
  if [ "${per_ok}" -eq 1 ]; then
    python3 - "${tmp}" <<'PERPY' > "${tmp}/per_cases.txt"
import sys, random
sys.path.insert(0, ".")
from bcir.asn1.per import (BitWriter, PerVariant, _encode_constrained,
                           _encode_semi_constrained, _encode_unconstrained,
                           _encode_normally_small)
rng = random.Random(20260726)
for variant, flag in ((PerVariant.UNALIGNED, 0), (PerVariant.ALIGNED, 1)):
    for lb, ub in ((0, 255), (0, 256), (0, 65535), (0, 1 << 40), (-5, 5)):
        for _ in range(40):
            v = rng.randint(lb, ub)
            w = BitWriter(variant); _encode_constrained(w, v, lb, ub)
            print(f"constrained {lb} {ub} {flag} {w.to_bytes().hex()}")
    for _ in range(40):
        v = rng.randint(-(1 << 40), 1 << 40)
        w = BitWriter(variant); _encode_unconstrained(w, v)
        print(f"unconstrained {flag} {w.to_bytes().hex()}")
    for _ in range(40):
        v = rng.randint(0, 1 << 20)
        w = BitWriter(variant); _encode_semi_constrained(w, v, 0)
        print(f"semi 0 {flag} {w.to_bytes().hex()}")
    for v in (0, 63, 64, 300):
        w = BitWriter(variant); _encode_normally_small(w, v)
        print(f"small {flag} {w.to_bytes().hex()}")
PERPY
    "${tmp}/test_per_O0" < "${tmp}/per_cases.txt" > "${tmp}/per_O0.txt"
    "${tmp}/test_per_O3" < "${tmp}/per_cases.txt" > "${tmp}/per_O3.txt"
    if cmp -s "${tmp}/per_O0.txt" "${tmp}/per_O3.txt"; then
      echo "  PASS X.691 PER twin (freestanding, -Werror, -O0 == -O3 over $(wc -l < "${tmp}/per_cases.txt") cases)"
    else
      echo "  FAIL: the PER twin's answers depend on the optimisation level"
      diff "${tmp}/per_O0.txt" "${tmp}/per_O3.txt" | head -10
      exit 1
    fi
  else
    echo "  SKIP PER optimisation-parity (a build failed)"
  fi
else
  echo "  FAIL: the X.691 PER twin does not build warning-clean"
  exit 1
fi

# X.693 XER lexical layer (#xer, roadmap phase E-adjacent): the C twin of the tag scanner
# and the xmlcstring escaper. XER is text, so there is no bit cursor to get wrong -- but
# there is a byte cursor, and it is driven entirely by attacker-supplied content before any
# type is consulted. The dual-rail differential lives in bcir/tests/test_c_xer.py; what is
# checked HERE is the same discipline every other twin gets: strict warnings as errors, a
# genuinely freestanding translation unit, and answers that do not depend on the optimiser.
echo "[c-runtime] X.693 XER lexical layer: strict-warning and freestanding build (#xer)"
if "${CC}" -std=c23 -O2 -Wall -Wextra -Werror -I "${C}" \
     "${C}/bcir_xer.c" "${C}/test_xer.c" -o "${tmp}/test_xer"; then
  for std in c11 c23; do
    "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -Werror -I "${C}" \
      -c "${C}/bcir_xer.c" -o /dev/null \
      || { echo "  FAIL: bcir_xer is not freestanding-clean under -std=${std}"; exit 1; }
  done
  xer_ok=1
  "${CC}" -std=c23 -O0 -I "${C}" "${C}/bcir_xer.c" "${C}/test_xer.c" -o "${tmp}/test_xer_O0" \
    || xer_ok=0
  "${CC}" -std=c23 -O3 -I "${C}" "${C}/bcir_xer.c" "${C}/test_xer.c" -o "${tmp}/test_xer_O3" \
    || xer_ok=0
  if [ "${xer_ok}" -eq 1 ]; then
    python3 - <<'XERPY' > "${tmp}/xer_cases.txt"
import sys
sys.path.insert(0, ".")
from bcir.asn1.xer import _CONTROL_ELEMENT

# Documents that reach every branch of the scanner, plus the truncations one octet short of
# each excluded construct -- the inputs where a bounds check that is off by one shows up.
docs = ["<a>", "</a>", "<a/>", "<PersonnelRecord>", "</ChildInformation>", "<_XMLThing/>",
        "<a >", "<a\t/>", "<nul/>", "<BIT_STRING>", "<x-y.z/>", "<!-- c -->",
        "<![CDATA[x]]>", "<!DOCTYPE a>", "<?xml?>", '<a b="1">', "<a:b>", "<", "</",
        "<a", "<a/", "<!", "<!-", "<![CDATA", "<?", "<1a>", "<>", "< a>", "a",
        "<a><b/></a>", "  <a>x</a>", "<a\xc3\xa9>"]
for doc in docs:
    raw = doc.encode("utf-8", "surrogatepass")
    for pos in range(len(raw) + 2):
        print(f"tag {raw.hex() or '-'} {pos}")
        print(f"space {raw.hex() or '-'} {pos}")

strings = ["", "a", "a<b>&c", "\t\n\r", "John P Smith", "&&&", "é中\U0001f600",
           "".join(chr(code) for code in sorted(_CONTROL_ELEMENT))]
for text in strings:
    raw = text.encode()
    print(f"escape {raw.hex() or '-'}")
    print(f"unescape 1 {raw.hex() or '-'}")
    print(f"unescape 0 {raw.hex() or '-'}")
for text in ("a&#233;b", "a&#xEE;b", "a&amp;b", "a&nbsp;b", "a<nul/>b", "a&#;b"):
    print(f"unescape 1 {text.encode().hex()}")
    print(f"unescape 0 {text.encode().hex()}")
for raw in (b"\xc0\x80", b"\xe0\x80\x80", b"\xed\xa0\x80", b"\xf5\x80\x80\x80", b"\x80",
            b"\xc3", b"\xc3\xa9", b"\xf0\x9f\x98\x80"):
    for pos in range(len(raw) + 1):
        print(f"utf8 {raw.hex()} {pos}")
XERPY
    "${tmp}/test_xer_O0" < "${tmp}/xer_cases.txt" > "${tmp}/xer_O0.txt"
    "${tmp}/test_xer_O3" < "${tmp}/xer_cases.txt" > "${tmp}/xer_O3.txt"
    if cmp -s "${tmp}/xer_O0.txt" "${tmp}/xer_O3.txt"; then
      echo "  PASS X.693 XER twin (freestanding, -Werror, -O0 == -O3 over $(wc -l < "${tmp}/xer_cases.txt") cases)"
    else
      echo "  FAIL: the XER twin's answers depend on the optimisation level"
      diff "${tmp}/xer_O0.txt" "${tmp}/xer_O3.txt" | head -10
      exit 1
    fi
  else
    echo "  SKIP XER optimisation-parity (a build failed)"
  fi
else
  echo "  FAIL: the X.693 XER twin does not build warning-clean"
  exit 1
fi

# X.697 bounded JER reader (#jer, JSON roadmap phase J3): the C twin of
# bcir/asn1/jer_bounded.py -- 4.3's limits, 7.6.2's encoding, and the ECMA-404 grammar as an
# event stream. Unlike the XER twin this one has no `json.loads` behind it, so it is a real
# parser and not only a lexer. The dual-rail differential lives in bcir/tests/test_c_jer.py;
# what is checked HERE is the discipline every other twin gets: strict warnings as errors, a
# genuinely freestanding translation unit, and answers that do not depend on the optimiser.
#
# The -O0 == -O3 comparison earns its place on this file specifically. The reader's hot path
# is signed/unsigned arithmetic on attacker-supplied lengths and a saturating exponent
# accumulator; if any of it were undefined behaviour the optimiser would be entitled to
# choose differently at -O3 than at -O0, and the answers would diverge exactly where a
# malicious document lives.
echo "[c-runtime] X.697 bounded JER reader: strict-warning and freestanding build (#jer)"
if "${CC}" -std=c23 -O2 -Wall -Wextra -Werror -I "${C}" \
     "${C}/bcir_jer.c" "${C}/test_jer.c" "${C}/bcir_runtime.c" -o "${tmp}/test_jer"; then
  for std in c11 c23; do
    # bcir_crc32 is DECLARED here and DEFINED in bcir_runtime.c (the same discipline
    # bcir_telemetry_frame.h follows), so this is a compile and not a link.
    "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -Werror -I "${C}" \
      -c "${C}/bcir_jer.c" -o /dev/null \
      || { echo "  FAIL: bcir_jer is not freestanding-clean under -std=${std}"; exit 1; }
  done
  jer_ok=1
  "${CC}" -std=c23 -O0 -I "${C}" "${C}/bcir_jer.c" "${C}/test_jer.c" "${C}/bcir_runtime.c" \
    -o "${tmp}/test_jer_O0" || jer_ok=0
  "${CC}" -std=c23 -O3 -I "${C}" "${C}/bcir_jer.c" "${C}/test_jer.c" "${C}/bcir_runtime.c" \
    -o "${tmp}/test_jer_O3" || jer_ok=0
  if [ "${jer_ok}" -eq 1 ]; then
    python3 - <<'JERPY' > "${tmp}/jer_cases.txt"
import sys
sys.path.insert(0, ".")
from bcir.asn1.jer_bounded import STRICT_LIMITS, frame

# Documents that reach every branch of both the bounding pass and the parser, plus the
# forms permissive readers accept and ECMA-404 does not: trailing commas, missing
# separators, leading zeros, the non-JSON constants, and the surrogate cases.
docs = [b"", b" ", b"null", b"true", b"false", b"0", b"-0", b"10", b"01", b"1.5",
        b"-0.5e+3", b"1E-2", b"1.", b".5", b"-", b"1e", b"+1", b'""', b'"a"',
        rb'"\n\t\r\b\f\/\\\""', rb'"A"', '"\U0001f600"'.encode(), rb'"\ud800"',
        rb'"\udc00"', rb'"\uZZZZ"', rb'"\q"', b'"a', b"[]", b"{}", b"[1,2,3]",
        b'{"a":1,"b":2}', b'{"a":{"b":[1,[2,[3]]]}}', b"[1,]", b'{"a":1,}', b"[,]",
        b"[1 2]", b'{"a" 1}', b'{"a":}', b"[", b"]", b"{", b"}", b"1 2", b"[]]",
        b"nan", b"NaN", b"Infinity", b"undefined", b"'a'", b'{"a":1,"a":2}',
        b"\x80", b'"\x80"', b'"\xc0\x80"', b'"\xed\xa0\x80"', b"\xef\xbb\xbf{}",
        b'"\x00"', b'"\x1f"', "[\"é\", \"中\"]".encode(),
        b"[" * 80 + b"]" * 80, b"1" * (STRICT_LIMITS.integer_digits + 1),
        b"1e" + str(STRICT_LIMITS.exponent_magnitude + 1).encode()]
for doc in docs:
    payload = doc.hex() or "-"
    for strict in (0, 1):
        print(f"scan {strict} {payload}")
        print(f"parse {strict} {payload}")
    print(f"utf8doc {payload}")
    for at in range(4):
        print(f"refuse {at} {payload}")

for raw in (b"", b"a", rb"\n", rb"A", "\U0001f600".encode(), rb"\ud800", rb"\q",
            "é中".encode()):
    payload = raw.hex() or "-"
    for cap in (0, 3, 65536):
        print(f"unescape {cap} {payload}")

for raw in (b"\xc0\x80", b"\xe0\x80\x80", b"\xed\xa0\x80", b"\xf5\x80\x80\x80", b"\x80",
            b"\xc3", b"\xc3\xa9", b"\xf0\x9f\x98\x80", b"\xf4\x90\x80\x80"):
    for at in range(len(raw) + 1):
        print(f"utf8 {raw.hex()} {at}")

good = frame(b'{"a":1}', sequence=42, generation=7)
for cut in range(0, len(good) + 1):
    print(f"unframe {good[:cut].hex() or '-'}")
bad = bytearray(good)
bad[-1] ^= 1
print(f"unframe {bytes(bad).hex()}")

for field in ("input_bytes", "depth", "nodes", "members", "elements", "string_bytes",
              "number_bytes", "integer_digits", "exponent_magnitude", "work"):
    for value in (1, 1 << 40):
        print(f"tighten {field} {value}")
JERPY
    "${tmp}/test_jer_O0" < "${tmp}/jer_cases.txt" > "${tmp}/jer_O0.txt"
    "${tmp}/test_jer_O3" < "${tmp}/jer_cases.txt" > "${tmp}/jer_O3.txt"
    if cmp -s "${tmp}/jer_O0.txt" "${tmp}/jer_O3.txt"; then
      echo "  PASS X.697 JER twin (freestanding, -Werror, -O0 == -O3 over $(wc -l < "${tmp}/jer_cases.txt") cases)"
    else
      echo "  FAIL: the JER twin's answers depend on the optimisation level"
      diff "${tmp}/jer_O0.txt" "${tmp}/jer_O3.txt" | head -10
      exit 1
    fi
  else
    echo "  SKIP JER optimisation-parity (a build failed)"
  fi
else
  echo "  FAIL: the X.697 JER twin does not build warning-clean"
  exit 1
fi

# Plan-driven ASN.1 encoder (#emit, E2): the write-side twin of bcir/asn1/emit.py. #682
# established that only X.690 can be encoded WITHOUT a type -- X.697 22.2 puts member
# identifiers in a JER document and an identifier exists only in the schema -- so every
# emitter here is schema-DIRECTED and all four read one format-neutral value stream. That is
# what makes their costs comparable at all. -O0 == -O3 earns its place twice over: the file
# does long division on octet arrays for arbitrary-width integers, and it compares
# attacker-supplied counts against remaining stream lengths.
echo "[c-runtime] plan-driven ASN.1 encoder: strict-warning and freestanding build (#emit)"
if "${CC}" -std=c23 -O2 -Wall -Wextra -Werror -I "${C}" \
     "${C}/bcir_emit.c" "${C}/test_emit.c" -o "${tmp}/test_emit"; then
  for std in c11 c23; do
    "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -Werror -I "${C}" \
      -c "${C}/bcir_emit.c" -o /dev/null \
      || { echo "  FAIL: bcir_emit is not freestanding-clean under -std=${std}"; exit 1; }
  done
  emit_ok=1
  "${CC}" -std=c23 -O0 -I "${C}" "${C}/bcir_emit.c" "${C}/test_emit.c" \
    -o "${tmp}/test_emit_O0" || emit_ok=0
  "${CC}" -std=c23 -O3 -I "${C}" "${C}/bcir_emit.c" "${C}/test_emit.c" \
    -o "${tmp}/test_emit_O3" || emit_ok=0
  if [ "${emit_ok}" -eq 1 ]; then
    python3 - <<'EMITPY' > "${tmp}/emit_cases.txt"
import sys
sys.path.insert(0, ".")
from bcir.asn1.codec import NULL, Oid
from bcir.asn1.constraints import Extensible, Size, ValueRange
from bcir.asn1.emit import flatten
from bcir.asn1.encode_plan import compile_encode_plan
from bcir.asn1.schema import Choice, Component, Primitive, Sequence, SequenceOf
from bcir.asn1.tags import Universal

I = Primitive(Universal.INTEGER)
S = Primitive(Universal.UTF8_STRING)
B = Primitive(Universal.BOOLEAN)
N = Primitive(Universal.NULL)
O = Primitive(Universal.OCTET_STRING)
D = Primitive(Universal.OBJECT_IDENTIFIER)


def seq(*components, name="X"):
    return Sequence(tuple(components), name=name)


CH = Choice((Component("num", I, tag=0), Component("txt", S, tag=1)), name="C")

# Version 4: an ENUMERATED needs its enumeration, because X.697 22.2 spells the value as the
# IDENTIFIER of its item and X.691 14.1 indexes the root. A hyphen exercises the descriptor's
# `name:number|...` field against an identifier X.680 12.4 permits.
ENUM = (("five", 5), ("two-hundred", 200), ("minus-one", -1))

CASES = [
    (seq(Component("v", I)), {"v": -1}),
    (seq(Component("v", I)), {"v": 2 ** 64 + 7}),
    (seq(Component("v", I)), {"v": 2 ** 400 + 12345}),
    (seq(Component("v", I)), {"v": 0}),
    (seq(Component("v", I)), {"v": 128}),
    (seq(Component("a", B), Component("b", B)), {"a": True, "b": False}),
    (seq(Component("v", N)), {"v": NULL}),
    (seq(Component("a", I), Component("v", N), Component("b", I)),
     {"a": 1, "v": NULL, "b": 2}),
    (seq(Component("v", O)), {"v": b"\x00\xff\x10"}),
    (seq(Component("v", O)), {"v": b""}),
    (seq(Component("v", S)), {"v": ""}),
    (seq(Component("v", S)), {"v": "a\nb\tc\"d\\e"}),
    (seq(Component("v", S)), {"v": "x" * 300}),
    (seq(Component("v", D)), {"v": Oid((1, 3, 6, 1, 4, 1, 62596, 1))}),
    (seq(Component("v", D)), {"v": Oid((2, 999, 1234567))}),
    (seq(Component("a", I), Component("b", I, optional=True)), {"a": 1, "b": 2}),
    (seq(Component("a", I), Component("b", I, optional=True)), {"a": 1}),
    (seq(Component("a", I), Component("b", B, default=False)), {"a": 1}),
    # Version 5: a DEFAULT component whose value EQUALS the default must be OMITTED (X.690
    # 11.5, X.696 31.9, CJER). Every row above supplies one that differs, which is exactly
    # how three emitters shipped emitting it.
    (seq(Component("a", I), Component("b", B, default=False)), {"a": 1, "b": False}),
    (seq(Component("a", I), Component("b", I, default=7)), {"a": 1, "b": 7}),
    (seq(Component("a", I), Component("b", S, default="hi")), {"a": 1, "b": "hi"}),
    (seq(*[Component("c%d" % i, I, default=i) for i in range(12)]),
     dict(("c%d" % i, (i if i % 2 else 99)) for i in range(12))),
    (seq(*[Component("c%d" % i, I, optional=True) for i in range(12)]),
     dict(("c%d" % i, i) for i in range(0, 12, 2))),
    (seq(*[Component("c%d" % i, I, optional=True) for i in range(12)]), {}),
    (seq(Component("v", SequenceOf(I, "SEQ"))), {"v": []}),
    (seq(Component("v", SequenceOf(I, "SEQ"))), {"v": [1, 2, 3]}),
    (seq(Component("v", SequenceOf(I, "SEQ"))), {"v": list(range(300))}),
    (seq(Component("in", seq(Component("a", I), name="In"))), {"in": {"a": 5}}),
    (seq(Component("a", I, tag=0), Component("b", S, tag=1)), {"a": 1, "b": "x"}),
    (seq(Component("a", I, tag=0, explicit=True)), {"a": 1}),
    (seq(Component("a", I, tag=100, explicit=True)), {"a": 1}),
    (seq(Component("a", I, tag=100)), {"a": 1}),
    (seq(Component("v", CH, tag=5, explicit=True)), {"v": ("num", 7)}),
    (seq(Component("v", CH, tag=5, explicit=True)), {"v": ("txt", "hi")}),
    (seq(Component("s", seq(Component("a", I), name="In"), tag=3)), {"s": {"a": 9}}),
    (seq(Component("v", SequenceOf(I, "SEQ"), tag=4)), {"v": [1, 2]}),
]

# Plan version 3's cases. Every row above is UNCONSTRAINED, and that is precisely how an OER
# emitter ignoring constraints passed this gate: X.696 10.3 gives a constrained INTEGER a
# fixed-width form with no length determinant, and nothing here ever asked for one. An
# ENUMERATED is here for the same reason -- 11 is not 10, and nothing asked for that either.
CASES += [
    (seq(Component("v", Primitive(Universal.INTEGER, "I",
                                  constraint=ValueRange(0, 255)))), {"v": 42}),
    (seq(Component("v", Primitive(Universal.INTEGER, "I",
                                  constraint=ValueRange(0, 2 ** 64 - 1)))), {"v": 2 ** 63}),
    (seq(Component("v", Primitive(Universal.INTEGER, "I",
                                  constraint=ValueRange(-128, 127)))), {"v": -5}),
    (seq(Component("v", Primitive(Universal.INTEGER, "I",
                                  constraint=ValueRange(0, None)))), {"v": 300}),
    (seq(Component("v", Primitive(Universal.INTEGER, "I",
                                  constraint=Extensible(ValueRange(0, 255))))), {"v": 42}),
    (seq(Component("v", Primitive(Universal.OCTET_STRING, "O",
                                  constraint=Size(ValueRange(3, 3))))), {"v": b"abc"}),
    (seq(Component("v", Primitive(Universal.IA5_STRING, "A",
                                  constraint=Size(ValueRange(3, 3))))), {"v": "abc"}),
    (seq(Component("v", Primitive(Universal.UTF8_STRING, "U",
                                  constraint=Size(ValueRange(3, 3))))), {"v": "abc"}),
    (seq(Component("v", Primitive(Universal.ENUMERATED, "E", enumeration=ENUM))), {"v": 5}),
    (seq(Component("v", Primitive(Universal.ENUMERATED, "E", enumeration=ENUM))), {"v": 200}),
    (seq(Component("v", Primitive(Universal.ENUMERATED, "E", enumeration=ENUM))), {"v": -1}),
    (seq(Component("v", Primitive(Universal.ENUMERATED, "E", enumeration=ENUM,
                                  enum_extensible=True))), {"v": 5}),
    (Sequence((Component("a", I),), name="X", extensible=True), {"a": 1}),
]

for index, (kind, value) in enumerate(CASES):
    plan = compile_encode_plan(kind, module="Gate", type_name="c%d" % index)
    stream = flatten(plan, value)
    print("plan %s" % plan.serialize().hex())
    for rules in ("der", "ber", "jer", "coer", "cper-a", "cper-u", "bper-a", "bper-u"):
        print("emit %s %s" % (rules, stream.hex() or "-"))
EMITPY
    "${tmp}/test_emit_O0" < "${tmp}/emit_cases.txt" > "${tmp}/emit_O0.txt"
    "${tmp}/test_emit_O3" < "${tmp}/emit_cases.txt" > "${tmp}/emit_O3.txt"
    if ! diff -q "${tmp}/emit_O0.txt" "${tmp}/emit_O3.txt" >/dev/null; then
      echo "  FAIL: the plan-driven encoder answers differently at -O0 and -O3"
      exit 1
    fi
    emit_cases=$(grep -c '^emit ' "${tmp}/emit_cases.txt")
    if grep -q '^err ' "${tmp}/emit_O0.txt"; then
      echo "  FAIL: the encoder refused a case from its own reference corpus"
      exit 1
    fi
    echo "  ok: -O0 == -O3 over ${emit_cases} encode cases (8 candidates x 1 plan each)"
  fi
else
  echo "  FAIL: the plan-driven ASN.1 encoder does not build warning-clean"
  exit 1
fi

# X.696 OER decoder (#oer, ASN.1 build-out phase D): the encoding rule with the best decode
# cost -- octet-aligned throughout, most fields fixed-width words a target loads directly --
# and therefore the one a driver-side or DMA-fed path actually wants. It is SCHEMA-DIRECTED
# because 6.2 leaves no choice: "without knowledge of the type of the value encoded, it is
# not possible to determine the structure of the encoding". The dual-rail differential lives
# in bcir/tests/test_c_oer.py; what is checked here is the usual discipline plus -O0 == -O3,
# which earns its place because the decoder sign-extends by hand and compares widths against
# attacker-supplied lengths.
# X.691 PER, decoded against a PLAN (#per, ASN.1 build-out phase H). 6.2's NOTE is why this
# has to be plan-driven: PER "does not include the identifier of the type being encoded" and
# "the abstract syntax is required in order to decode" -- no tags, no lengths except where a
# clause asks, fields at odd bit widths, so there is no schema-free structural pass at all.
# 7.2 bars the schema-FREE decode and says nothing against this one.
#
# The corpus is generated by BCIR's OWN PER encoder and read back by the C twin. Both variants
# are covered, because ALIGNED and UNALIGNED differ at every field boundary and a decoder
# correct on one can be wrong on the other in a way no single-variant corpus would show.
#
# WHAT THIS GATE DOES AND DOES NOT PROVE. The -O0/-O3 comparison below is a MISCOMPILATION
# check and nothing more: both columns come from the same decoder, so a decoder that is wrong
# about a rule agrees with itself perfectly. This gate once carried a comment calling that a
# "real differential", and two defects lived behind the claim -- a mid-octet string reported at
# the octet containing it, and 16.6's short string refused in ALIGNED PER. The comparison
# against what the encoder actually encoded lives in bcir/tests/test_c_per_plan.py, which
# decodes the same records and checks every field VALUE, locating each string by its reported
# bit offset. Keep that file in mind before trusting this one about correctness.
echo "[c-runtime] X.691 plan-driven PER decoder: strict-warning and freestanding build (#per)"
if "${CC}" -O2 -Wall -Wextra -Werror -I "${C}" \
     "${C}/bcir_per.c" "${C}/bcir_per_plan.c" "${C}/test_per_plan.c" -o "${tmp}/test_per_plan"; then
  for std in c11 c2x; do
    "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -Werror -I "${C}" \
      -c "${C}/bcir_per_plan.c" -o /dev/null \
      || { echo "  FAIL: bcir_per_plan is not freestanding-clean under -std=${std}"; exit 1; }
  done
  per_ok=1
  "${CC}" -O0 -I "${C}" "${C}/bcir_per.c" "${C}/bcir_per_plan.c" "${C}/test_per_plan.c" \
    -o "${tmp}/test_per_O0" || per_ok=0
  "${CC}" -O3 -I "${C}" "${C}/bcir_per.c" "${C}/bcir_per_plan.c" "${C}/test_per_plan.c" \
    -o "${tmp}/test_per_O3" || per_ok=0
  if [ "${per_ok}" -eq 1 ]; then
    python3 - <<'PERPY' > "${tmp}/per_cases.txt"
import sys
sys.path.insert(0, ".")
from bcir.asn1.constraints import Size, ValueRange
from bcir.asn1.per import PerVariant, encode_per
from bcir.asn1.schema import Component, Primitive, Sequence
from bcir.asn1.tags import Universal

byte = Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(0, 255))
word = Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(0, 65535))
flag = Primitive(Universal.BOOLEAN, "BOOLEAN")
text = Primitive(Universal.OCTET_STRING, "OCTET STRING", constraint=Size(ValueRange(3, 3)))

# kind:bounds:lb:ub:fixed:optional -- every property bcir_per_field carries.
PLAN = "0:2:0:255:0:0,1:0:0:0:0:0,3:0:0:0:3:0"
record = Sequence((Component("id", byte), Component("flag", flag),
                   Component("name", text)), name="R")
for variant in (PerVariant.UNALIGNED, PerVariant.ALIGNED):
    aligned = 1 if variant is PerVariant.ALIGNED else 0
    for ident in (0, 1, 42, 254, 255):
        for truth in (True, False):
            raw = encode_per(record, {"id": ident, "flag": truth, "name": b"abc"},
                             variant=variant)
            print(f"sequence {raw.hex()} {aligned} 0 {PLAN}")

# A two-field record whose first component is optional, so 18.2's bit-map is exercised in
# both of its states rather than only in the one a happy path reaches.
opt = Sequence((Component("id", word, optional=True), Component("flag", flag)), name="O")
OPLAN = "0:2:0:65535:0:1,1:0:0:0:0:0"
for variant in (PerVariant.UNALIGNED, PerVariant.ALIGNED):
    aligned = 1 if variant is PerVariant.ALIGNED else 0
    for value in ({"flag": True}, {"id": 7, "flag": False}, {"id": 65535, "flag": True}):
        raw = encode_per(opt, value, variant=variant)
        print(f"sequence {raw.hex()} {aligned} 0 {OPLAN}")

# 16.6: a string of two octets or fewer is placed with NO alignment in EITHER variant, so it
# begins at bit 1 here. The twin refused exactly this shape in ALIGNED PER until the sweep,
# and no corpus reached it -- every other record above has a string of three octets.
short = Primitive(Universal.OCTET_STRING, "OCTET STRING", constraint=Size(ValueRange(2, 2)))
brief = Sequence((Component("flag", flag), Component("s", short)), name="S")
SPLAN = "1:0:0:0:0:0,3:0:0:0:2:0"
for variant in (PerVariant.UNALIGNED, PerVariant.ALIGNED):
    aligned = 1 if variant is PerVariant.ALIGNED else 0
    raw = encode_per(brief, {"flag": True, "s": b"hi"}, variant=variant)
    print(f"sequence {raw.hex()} {aligned} 0 {SPLAN}")

# Truncation and refusal: every prefix of a valid encoding, plus the 18.1 extension bit.
raw = encode_per(record, {"id": 42, "flag": True, "name": b"abc"},
                 variant=PerVariant.UNALIGNED).hex()
for cut in range(0, len(raw) - 1, 2):
    print(f"sequence {raw[:cut]} 0 0 {PLAN}")
print(f"sequence 80{raw} 0 1 {PLAN}")
PERPY
    "${tmp}/test_per_O0" < "${tmp}/per_cases.txt" > "${tmp}/per_O0.txt"
    "${tmp}/test_per_O3" < "${tmp}/per_cases.txt" > "${tmp}/per_O3.txt"
    if cmp -s "${tmp}/per_O0.txt" "${tmp}/per_O3.txt"; then
      cases=$(wc -l < "${tmp}/per_cases.txt")
      # A corpus that decoded nothing would compare equal too, so the gate checks that the
      # differential actually produced answers rather than a column of refusals.
      oks=$(grep -c '^OK' "${tmp}/per_O0.txt" || true)
      if [ "${oks}" -lt 20 ]; then
        echo "  FAIL: only ${oks} of ${cases} PER cases decoded; the corpus is not exercising the decoder"
        exit 1
      fi
      echo "  PASS X.691 plan-driven PER twin (freestanding, -Werror, -O0 == -O3 over ${cases} cases, ${oks} decoded)"
    else
      echo "  FAIL: PER decoder differs between -O0 and -O3"; exit 1
    fi
  else
    echo "  FAIL: PER decoder did not build at -O0/-O3"; exit 1
  fi
else
  echo "  FAIL: PER plan decoder did not build with -Werror"; exit 1
fi

echo "[c-runtime] X.696 OER decoder: strict-warning and freestanding build (#oer)"
if "${CC}" -std=c23 -O2 -Wall -Wextra -Werror -I "${C}" \
     "${C}/bcir_oer.c" "${C}/test_oer.c" -o "${tmp}/test_oer"; then
  for std in c11 c23; do
    "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -Werror -I "${C}" \
      -c "${C}/bcir_oer.c" -o /dev/null \
      || { echo "  FAIL: bcir_oer is not freestanding-clean under -std=${std}"; exit 1; }
  done
  oer_ok=1
  "${CC}" -std=c23 -O0 -I "${C}" "${C}/bcir_oer.c" "${C}/test_oer.c" \
    -o "${tmp}/test_oer_O0" || oer_ok=0
  "${CC}" -std=c23 -O3 -I "${C}" "${C}/bcir_oer.c" "${C}/test_oer.c" \
    -o "${tmp}/test_oer_O3" || oer_ok=0
  if [ "${oer_ok}" -eq 1 ]; then
    python3 - <<'OERPY' > "${tmp}/oer_cases.txt"
import sys
sys.path.insert(0, ".")
from bcir.asn1.constraints import Size, ValueRange
from bcir.asn1.oer import OerRules, encode_length, encode_oer
from bcir.asn1.schema import Component, Primitive, Sequence
from bcir.asn1.tags import Universal

for value in (0, 1, 126, 127, 128, 255, 256, 65535, 65536, 1 << 24):
    raw = encode_length(value)
    for pos in range(len(raw) + 2):
        print(f"length {raw.hex()} {pos}")
# The malformed and BASIC-OER spellings a decoder must tell apart.
for raw in ("80", "8100", "820080", "82ff", "ff"):
    print(f"length {raw} 0")

byte = Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(0, 255))
word = Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(-32768, 32767))
wide = Primitive(Universal.INTEGER, "INTEGER")
text = Primitive(Universal.UTF8_STRING, "UTF8String")
for kind, width, signed, values in ((byte, 1, 0, (0, 1, 255)),
                                    (word, 2, 1, (-32768, -1, 0, 32767)),
                                    (wide, 0, 1, (0, -1, 128, -129, 2 ** 40))):
    for value in values:
        raw = encode_oer(kind, value, rules=OerRules.CANONICAL)
        print(f"integer {raw.hex()} 0 {width} {signed}")
for raw in ("00", "ff", "0000", "ffff", "ffffffffffffffff"):
    for width in (1, 2, 4, 8, 0, 3):
        for signed in (0, 1):
            print(f"integer {raw} 0 {width} {signed}")

for raw in ("00", "80", "40", "c0", "ff", "81"):
    for count in (0, 1, 2, 3, 8):
        print(f"preamble {raw} 0 {count}")

record = Sequence((Component("id", byte), Component("delta", word),
                   Component("label", text), Component("note", text, optional=True)),
                  name="Record")
plan = "0:1:0:0:0,0:2:1:0:0,4:0:0:0:0,4:0:0:1:0"
for value in ({"id": 7, "delta": -3, "label": "abc", "note": "n"},
              {"id": 0, "delta": 0, "label": ""},
              {"id": 255, "delta": 32767, "label": "x" * 200}):
    raw = encode_oer(record, value, rules=OerRules.CANONICAL)
    for cut in range(len(raw) + 1):
        print(f"sequence {raw[:cut].hex() or '-'} {plan}")
# Plans the checker must refuse before reading an octet.
print(f"sequence 00 9:0:0:0:0")
print(f"sequence 00 0:3:0:0:0")
OERPY
    "${tmp}/test_oer_O0" < "${tmp}/oer_cases.txt" > "${tmp}/oer_O0.txt"
    "${tmp}/test_oer_O3" < "${tmp}/oer_cases.txt" > "${tmp}/oer_O3.txt"
    if cmp -s "${tmp}/oer_O0.txt" "${tmp}/oer_O3.txt"; then
      echo "  PASS X.696 OER twin (freestanding, -Werror, -O0 == -O3 over $(wc -l < "${tmp}/oer_cases.txt") cases)"
    else
      echo "  FAIL: the OER twin's answers depend on the optimisation level"
      diff "${tmp}/oer_O0.txt" "${tmp}/oer_O3.txt" | head -10
      exit 1
    fi
  else
    echo "  SKIP OER optimisation-parity (a build failed)"
  fi
else
  echo "  FAIL: the X.696 OER twin does not build warning-clean"
  exit 1
fi

# The native ASN.1 decode microbench (#asn1bench, JSON roadmap J6 follow-on): the harness
# that makes a `measured` cost table possible, and therefore the reason select_certified can
# decide a timing objective at all instead of refusing every one. It is a MEASUREMENT tool,
# so what is gated here is that it builds warning-clean and answers a corpus -- the numbers
# themselves are deliberately NOT a CI assertion, because a shared runner's timings are not
# evidence about a target and pinning them would invent the false precision J6 refuses.
echo "[c-runtime] native ASN.1 decode microbench: strict-warning build (#asn1bench)"
if "${CC}" -std=c23 -O2 -Wall -Wextra -Werror -I "${C}" \
     "${C}/bcir_asn1_bench.c" "${C}/bcir_asn1.c" "${C}/bcir_jer.c" "${C}/bcir_xer.c" \
     "${C}/bcir_runtime.c" "${C}/bcir_emit.c" "${C}/bcir_oer.c" \
     "${C}/bcir_per.c" "${C}/bcir_per_plan.c" -o "${tmp}/asn1_bench"; then
  printf 'rounds 1 7 8\ncase DER der 3009020102040461\ncase JER jer 7b2261223a317d\nrun\n' \
    > "${tmp}/bench_cases.txt"
  if "${tmp}/asn1_bench" < "${tmp}/bench_cases.txt" | grep -q '^done 2$'; then
    echo "  PASS native ASN.1 microbench (builds -Werror, answers a two-case corpus)"
  else
    echo "  FAIL: the native ASN.1 microbench did not complete its corpus"
    exit 1
  fi
else
  echo "  FAIL: the native ASN.1 microbench does not build warning-clean"
  exit 1
fi

# DER -> native StreamPack fast path (#asn1fast, roadmap phase D): reconstruct the native
# artifact from its X.690 DER projection in freestanding C, with no Python anywhere in the
# reconstruction path, and assert BYTE IDENTITY against what the Python encoder produced.
# That is law A3 (additive: the native octets survive the round trip) proven on the C rail.
# Byte identity, not equivalence -- the fast path has to re-derive the StreamPack VERSION
# from content the way bcir/abi::encode does, emit the reserved stride_k the projection
# deliberately omits, and recompute the CRC.
echo "[c-runtime] DER -> native StreamPack fast path: byte-identical reconstruction (#asn1fast)"
if "${CC}" -std=c23 -O2 -Wall -Wextra -I "${C}" "${C}/bcir_asn1_streampack.c" \
     "${C}/bcir_asn1.c" "${C}/bcir_runtime.c" "${C}/test_asn1_streampack.c" \
     -o "${tmp}/test_asn1_sp"; then
  for std in c11 c23; do
    "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -I "${C}" \
      -c "${C}/bcir_asn1_streampack.c" -o /dev/null \
      || { echo "  FAIL: bcir_asn1_streampack not freestanding-clean under -std=${std}"; exit 1; }
  done
  python3 - "${tmp}" <<'ASN1FASTPY' || { echo "  FAIL: could not project the corpus"; exit 1; }
import os, sys
from bcir.abi import encode
from bcir.asn1.streampack import encode_pack
from bcir.examples import PROGRAMS
from bcir.gem import hydrate
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
d = sys.argv[1]
host, theta = TargetProfile.x86_avx512(), Theta.cool()
for name, build in sorted(PROGRAMS.items()):
    module = build()
    pack = hydrate(module, optimize(module, host, theta))
    open(os.path.join(d, name + ".proj.der"), "wb").write(encode_pack(pack))
    open(os.path.join(d, name + ".native.bin"), "wb").write(encode(pack))
ASN1FASTPY
  fast_ok=0; fast_bad=0
  for proj in "${tmp}"/*.proj.der; do
    base="$(basename "${proj}" .proj.der)"
    if out="$("${tmp}/test_asn1_sp" "${proj}" "${tmp}/${base}.native.bin" 2>&1)" \
         && [ "${out%% *}" = "OK" ]; then
      fast_ok=$((fast_ok + 1))
    else
      echo "  FAIL ${base}: ${out}"; fast_bad=$((fast_bad + 1))
    fi
  done
  # A malformed or BER-only projection must be refused, never partially reconstructed.
  python3 - "${tmp}" <<'ASN1NEGPY'
import os, sys
d = sys.argv[1]
der = open(os.path.join(d, "vector_add.proj.der"), "rb").read()
for n in (0, 1, 5, len(der) // 2, len(der) - 1):
    open(os.path.join(d, f"neg{n}.bad.der"), "wb").write(der[:n])
if der[1] < 0x80:      # the same value with a non-minimal length: legal BER, not DER
    open(os.path.join(d, "nonmin.bad.der"), "wb").write(
        bytes([der[0], 0x81, der[1]]) + der[2:])
ASN1NEGPY
  for bad in "${tmp}"/*.bad.der; do
    if "${tmp}/test_asn1_sp" "${bad}" >/dev/null 2>&1; then
      echo "  FAIL: $(basename "${bad}") was accepted by the fast path"; fast_bad=$((fast_bad + 1))
    fi
  done
  if [ "${fast_bad}" -eq 0 ] && [ "${fast_ok}" -ge 10 ]; then
    echo "  PASS DER -> native byte-identical on ${fast_ok} corpus programs; malformed + BER-only refused"
  else
    echo "  FAIL: fast path (${fast_ok} ok, ${fast_bad} bad)"; exit 1
  fi
else
  echo "  SKIP fast path (harness did not build)"
fi

# Sanitizer + memory-stress harness for the cfront C twin (#sanitize): the dual-rail parity gates above
# compare the twin's OUTPUT against the oracle, but never build bcir_cfront.c under AddressSanitizer/UBSan
# and never run Valgrind -- so a memory bug (buffer overflow, use-after-free, UB) INSIDE the compiler's own
# lexer/parser/lowering/emit would go uncaught. sanitize_cfront.sh closes that: it builds the SAME
# `test_cfront` driver (same source list as above) under ASan/UBSan with clang and/or gcc, runs every
# cfront_*.c fixture + a bounded seeded fuzz campaign (valid + malformed) through it, and runs a bounded
# Valgrind pass over a fixture subset. Each stage self-skips when its tool is absent (SKIP, not FAIL). The
# heavy Valgrind stage can be dropped with SANITIZE_SKIP_VALGRIND=1 while ASan/UBSan stay always-on.
#
# BCIR_SKIP_CFRONT_SANITIZE=1 suppresses this nested call for a CALLER THAT RUNS THE SAME
# HARNESS ITSELF. The CI c-runtime job does exactly that (a dedicated step, so a sanitizer
# diagnostic is attributed to the sanitizer rather than buried in this script's output), and
# without the opt-out the whole 300-valid/400-malformed campaign ran twice in one job. Anyone
# invoking check_runtime.sh directly still gets the harness, valgrind included.
if [ "${BCIR_SKIP_CFRONT_SANITIZE:-0}" = "1" ]; then
  echo "[c-runtime] SKIP cfront sanitizer harness (BCIR_SKIP_CFRONT_SANITIZE=1; the caller runs it)"
else
echo "[c-runtime] cfront twin under ASan/UBSan + Valgrind (sanitize_cfront.sh)"
if CLANG="${CC}" bash "${ROOT}/tools/c/sanitize_cfront.sh" 2>&1 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -eq 0 ]; then
  echo "  PASS cfront sanitizer/valgrind harness"
else
  echo "  FAIL: cfront sanitizer/valgrind harness reported a diagnostic"; exit 1
fi
fi

# StreamPack SEMANTIC trust boundary (R10/R11 + lane/width/dispatch range, ported into the
# freestanding C decoder/executor): the CRC + bounds decode is memory-safe, but a CRC-VALID pack
# can still be semantically corrupt (a dangling/redirected claim_id, a swapped/undeclared RID, an
# out-of-range lane/width, an unresolved prefetch, a stale generation, a tampered dispatch). Each
# used to execute SILENTLY in C. check_streampack_semantic.sh crafts each as a CRC-FIXED pack and
# asserts the C rail (bcir_sp_verify_semantic / bcir_sp_execute_checked) now REJECTS it, plus a
# C-decode == Python-decode differential (the lane-asymmetry class) over v1/v2/v3 packs.
echo "[c-runtime] StreamPack semantic trust boundary (check_streampack_semantic.sh)"
if CC="${CC}" bash "${ROOT}/tools/c/check_streampack_semantic.sh" 2>&1 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -eq 0 ]; then
  echo "  PASS StreamPack semantic-corruption rejection + C/Python decode differential"
else
  echo "  FAIL: a CRC-valid semantically-corrupt pack was not rejected on the C rail"; exit 1
fi

# The hosted SIMD rail (#jersimd, J5): an optional C++17 UTF-8 accept-scanner behind the
# SCALAR C ABI, with SSE2/AVX2/NEON tiers, runtime feature detection and scalar fallback.
# 4.1 is the constraint: "the scalar rail is authoritative for native parser correctness.
# SIMD is an optimization candidate, not a separate semantic implementation." So the vector
# pass answers only "is this block entirely ASCII?" and hands everything else to
# bcir_jer_validate_utf8 ITSELF -- there is no second UTF-8 implementation to keep in step.
# check_jer_simd.sh proves every tier returns an identical status AND byte offset to the
# scalar rail over 489 documents, and that an unavailable tier degrades rather than faults.
# The two-host advantage clause of J5's gate is deliberately NOT checked on a shared runner;
# 8 refuses "noisy timing thresholds" there. Self-skips if no C++ compiler is present.
# The hosted structural index (#jerindex, J5's second half): `bcir_jer_index_scan` rebuilds
# `bcir_jer_scan`'s DISPATCH on the exported cursor and reuses its token scanners verbatim, so
# it is a second dispatch loop rather than a second scanner. Checked here as a build gate --
# warning-clean under C++17, and the C core still freestanding with the cursor exported. The
# equivalence differential itself lives in test_cpp_jer_index.py, where it can sweep 4.3's
# work ceiling across every failure position.
# Target ABI portability (#targetabi). The freestanding core is compiled for triples this
# host cannot run -- Android's among them -- and no source may hand-declare a libc function.
# #699 shipped a bench that did not build under Termux because it did exactly that, and the
# existing cross-compile gates target aarch64-linux-GNU, which is the right architecture and
# the wrong libc.
echo "[c-runtime] target ABI: the freestanding core builds for Android and 32-bit (#targetabi)"
if bash "${ROOT}/tools/c/check_target_abi.sh" 2>&1 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -eq 0 ]; then
  echo "  PASS target ABI sweep (freestanding core portable; no hand-declared libc prototypes)"
else
  echo "  FAIL: the freestanding core is not portable, or a source declares a libc function"; exit 1
fi

echo "[c-runtime] hosted structural index: the cursor seam and its vector pass (#jerindex)"
if bash "${ROOT}/tools/cpp/check_jer_index.sh" 2>&1 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -eq 0 ]; then
  echo "  PASS hosted structural index (freestanding core preserved, tier is real not scalar)"
else
  echo "  FAIL: the hosted structural index did not build, or a tier degraded to scalar"; exit 1
fi

echo "[c-runtime] hosted SIMD rail: scalar-identical status/offset at every tier (#jersimd)"
if bash "${ROOT}/tools/cpp/check_jer_simd.sh" 2>&1 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -eq 0 ]; then
  echo "  PASS hosted SIMD rail (same corpus + same trace at every tier, no unsupported-CPU fault)"
else
  echo "  FAIL: the hosted SIMD rail diverged from the scalar rail or did not build"; exit 1
fi

# The C<->C++ hand-off seam scaffold (#cpphandoff, docs/languages/CPP_HANDOFF_BOUNDARY.md): the boundary
# between the deterministic single-node C/IR rail and the C++ layer ABOVE it (dynamic graph
# topology + distributed MPI/NCCL orchestration). check_handoff.sh compiles the STANDALONE C++17
# scaffold (runtime/cpp/, NOT part of the MLIR build), hands a StreamPack the C/IR path produces
# through the single-node Orchestrator, and asserts the seam ROUND-TRIPS (the C++ dispatch order ==
# the direct C/IR decode of the same artifact) -- plus that a corrupted artifact is REJECTED at the
# boundary (admit() carries the C verifier's verdict; the two-truth quarantine holds across the seam).
# The dynamic-graph + distributed backends are documented STUBS behind the same interface (no real
# MPI/NCCL dependency). Self-skips (exit 0) if no C++ compiler is present, like the gates above.
echo "[c-runtime] C<->C++ hand-off seam scaffold round-trip (tools/cpp/check_handoff.sh)"
if bash "${ROOT}/tools/cpp/check_handoff.sh" 2>&1 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -eq 0 ]; then
  echo "  PASS C<->C++ hand-off seam compiles + round-trips (single-node Orchestrator == direct C/IR)"
else
  echo "  FAIL: the C++ hand-off seam scaffold did not compile or round-trip"; exit 1
fi

# The SYCL backend differential oracle (#sycldiff, docs/kernel/SYCL_INTEROP.md): SYCL is a backend CHANNEL +
# a differential oracle, NEVER on the legality path. check_sycl.sh emits the single-source C++ SAXPY
# kernel (emit_sycl_saxpy_c) and proves it reproduces BCIR's own deterministic reference (a*x+y) to
# float round-off: the PORTABLE scalar C++ fallback always (the real reference-verification work, no
# SYCL needed -- the #sycl-fallback marker), and the SYCL DEVICE parallel_for (-DBCIR_USE_SYCL -fsycl,
# the #sycl-device marker) IF a real SYCL compiler is detected (icpx/acpp/clang++ -fsycl that compiles+
# links a probe). SYCL is a compiler MODE, NOT a c.call.libm: -l<lib> edge (no link-flag rule). Lives
# above the G8 C++ boundary; self-skips (exit 0) if no C++ compiler, like check_handoff.sh.
echo "[c-runtime] SYCL backend differential oracle (tools/cpp/check_sycl.sh)"
if bash "${ROOT}/tools/cpp/check_sycl.sh" 2>&1 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -eq 0 ]; then
  echo "  PASS SYCL SAXPY differential (portable C++ fallback == BCIR reference; device path when -fsycl present)"
else
  echo "  FAIL: the SYCL backend differential oracle did not compile or agree"; exit 1
fi

echo "[c-runtime] ok"
