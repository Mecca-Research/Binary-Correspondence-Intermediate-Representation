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

# The gates this script delegates to -- the manifest's `delegated` entries, each a script that builds
# and judges binaries of its own -- run here in turn. BCIR_SKIP_DELEGATED_GATES=1 skips them for a
# caller that runs each one itself: the CTest `c-runtime` entry, beside which CMake registers every
# delegated gate as an entry of its own, so `ctest` runs each once (BUILD-2j). M16 of
# tools/build/manifest.py holds these calls, the manifest and the switch in step. Anyone invoking
# check_runtime.sh directly still gets every gate.
echo "[c-runtime] memory classes + allocator/context/channel fault sweep"
if [ "${BCIR_SKIP_DELEGATED_GATES:-0}" = "1" ]; then
  echo "  SKIP memory-discipline gate (BCIR_SKIP_DELEGATED_GATES=1; CTest runs it as c-memory-discipline)"
elif CC="${CC}" bash "${ROOT}/tools/c/check_memory_discipline.sh" 2>&1 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -ne 0 ]; then
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

echo "[c-runtime] OOB ring counter is atomic under concurrent events (#atomicring)"
"${CC}" -std=c23 -O2 -pthread -I "${C}" "${C}/test_oob_counter.c" "${C}/bcir_quarantine.c" -o "${tmp}/test_oob_counter" 2>/dev/null \
  || "${CC}" -std=c2x -O2 -pthread -I "${C}" "${C}/test_oob_counter.c" "${C}/bcir_quarantine.c" -o "${tmp}/test_oob_counter" \
  || { echo "  FAIL: atomic-ring test build"; exit 1; }
bash "${ROOT}/tools/c/sections/atomicring.sh" "${tmp}/test_oob_counter" || exit 1

echo "[c-runtime] rid->extent tamper-evidence: BCIR_EXTENT_ASSERT (freestanding lightweight check) (#extentassert)"
"${CC}" -std=c23 -I "${C}" "${C}/test_extent_assert.c" "${C}/bcir_quarantine.c" -o "${tmp}/test_extent_assert" 2>/dev/null \
  || "${CC}" -std=c2x -I "${C}" "${C}/test_extent_assert.c" "${C}/bcir_quarantine.c" -o "${tmp}/test_extent_assert" \
  || { echo "  FAIL: BCIR_EXTENT_ASSERT rejected a CORRECT extent"; exit 1; }
CC="${CC}" bash "${ROOT}/tools/c/sections/extentassert.sh" "${tmp}/test_extent_assert" || exit 1

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

echo "[c-runtime] inline assembly: ISA-neutral trusted opaque edge, verbatim emit + memory barrier (#inlineasm)"
CC="${CC}" bash "${ROOT}/tools/c/sections/inlineasm.sh" || exit 1

echo "[c-runtime] port-mapped I/O intrinsics: typed barriered edge, x86 in/out emit, assemble-only (#portio)"
CC="${CC}" bash "${ROOT}/tools/c/sections/portio.sh" || exit 1

echo "[c-runtime] memory-fence intrinsics: typed kinded barriered edge, per-ISA emit, native assemble (#barrier)"
CC="${CC}" bash "${ROOT}/tools/c/sections/barrier.sh" || exit 1

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
  # shift-past-width bug typically only diverges at one of the two. A twin that builds at -O2
  # but not at -O0 or -O3 is a defect, and the CMake build of the manifest variants fails on
  # it, so the gate fails too: these sweeps used to SKIP on a failed build (laws.md L12, L21).
  per_ok=1
  "${CC}" -std=c23 -O0 -I "${C}" "${C}/bcir_per.c" "${C}/test_per.c" -o "${tmp}/test_per_O0" \
    || per_ok=0
  "${CC}" -std=c23 -O3 -I "${C}" "${C}/bcir_per.c" "${C}/test_per.c" -o "${tmp}/test_per_O3" \
    || per_ok=0
  if [ "${per_ok}" -eq 1 ]; then
    bash "${ROOT}/tools/c/sections/per.sh" "${tmp}/test_per_O0" "${tmp}/test_per_O3" || exit 1
  else
    echo "  FAIL: the X.691 PER twin did not build at -O0/-O3"; exit 1
  fi
else
  echo "  FAIL: the X.691 PER twin does not build warning-clean"
  exit 1
fi

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
    bash "${ROOT}/tools/c/sections/xer.sh" "${tmp}/test_xer_O0" "${tmp}/test_xer_O3" || exit 1
  else
    echo "  FAIL: the X.693 XER twin did not build at -O0/-O3"; exit 1
  fi
else
  echo "  FAIL: the X.693 XER twin does not build warning-clean"
  exit 1
fi

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
    bash "${ROOT}/tools/c/sections/jer.sh" "${tmp}/test_jer_O0" "${tmp}/test_jer_O3" || exit 1
  else
    echo "  FAIL: the X.697 JER twin did not build at -O0/-O3"; exit 1
  fi
else
  echo "  FAIL: the X.697 JER twin does not build warning-clean"
  exit 1
fi

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
    bash "${ROOT}/tools/c/sections/asn1_emit.sh" "${tmp}/test_emit_O0" "${tmp}/test_emit_O3" || exit 1
  else
    echo "  FAIL: the plan-driven ASN.1 encoder did not build at -O0/-O3"; exit 1
  fi
else
  echo "  FAIL: the plan-driven ASN.1 encoder does not build warning-clean"
  exit 1
fi

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
    -o "${tmp}/test_per_plan_O0" || per_ok=0
  "${CC}" -O3 -I "${C}" "${C}/bcir_per.c" "${C}/bcir_per_plan.c" "${C}/test_per_plan.c" \
    -o "${tmp}/test_per_plan_O3" || per_ok=0
  if [ "${per_ok}" -eq 1 ]; then
    bash "${ROOT}/tools/c/sections/per_plan.sh" "${tmp}/test_per_plan_O0" "${tmp}/test_per_plan_O3" || exit 1
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
    bash "${ROOT}/tools/c/sections/oer.sh" "${tmp}/test_oer_O0" "${tmp}/test_oer_O3" || exit 1
  else
    echo "  FAIL: the X.696 OER twin did not build at -O0/-O3"; exit 1
  fi
else
  echo "  FAIL: the X.696 OER twin does not build warning-clean"
  exit 1
fi

echo "[c-runtime] native ASN.1 decode microbench: strict-warning build (#asn1bench)"
if "${CC}" -std=c23 -O2 -Wall -Wextra -Werror -I "${C}" \
     "${C}/bcir_asn1_bench.c" "${C}/bcir_asn1.c" "${C}/bcir_jer.c" "${C}/bcir_xer.c" \
     "${C}/bcir_runtime.c" "${C}/bcir_emit.c" "${C}/bcir_oer.c" \
     "${C}/bcir_per.c" "${C}/bcir_per_plan.c" -o "${tmp}/asn1_bench"; then
  bash "${ROOT}/tools/c/sections/asn1bench.sh" "${tmp}/asn1_bench" || exit 1
else
  echo "  FAIL: the native ASN.1 microbench does not build warning-clean"
  exit 1
fi

echo "[c-runtime] DER -> native StreamPack fast path: byte-identical reconstruction (#asn1fast)"
if "${CC}" -std=c23 -O2 -Wall -Wextra -I "${C}" "${C}/bcir_asn1_streampack.c" \
     "${C}/bcir_asn1.c" "${C}/bcir_runtime.c" "${C}/test_asn1_streampack.c" \
     -o "${tmp}/test_asn1_sp"; then
  for std in c11 c23; do
    "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -I "${C}" \
      -c "${C}/bcir_asn1_streampack.c" -o /dev/null \
      || { echo "  FAIL: bcir_asn1_streampack not freestanding-clean under -std=${std}"; exit 1; }
  done
  bash "${ROOT}/tools/c/sections/asn1fast.sh" "${tmp}/test_asn1_sp" || exit 1
else
  echo "  FAIL: the DER -> StreamPack fast-path harness did not build"; exit 1
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
if [ "${BCIR_SKIP_DELEGATED_GATES:-0}" = "1" ]; then
  echo "  SKIP StreamPack semantic trust boundary (BCIR_SKIP_DELEGATED_GATES=1; CTest runs it as c-streampack-semantic)"
elif CC="${CC}" bash "${ROOT}/tools/c/check_streampack_semantic.sh" 2>&1 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -eq 0 ]; then
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
if [ "${BCIR_SKIP_DELEGATED_GATES:-0}" = "1" ]; then
  echo "  SKIP target ABI sweep (BCIR_SKIP_DELEGATED_GATES=1; CTest runs it as c-target-abi)"
elif bash "${ROOT}/tools/c/check_target_abi.sh" 2>&1 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -eq 0 ]; then
  echo "  PASS target ABI sweep (freestanding core portable; no hand-declared libc prototypes)"
else
  echo "  FAIL: the freestanding core is not portable, or a source declares a libc function"; exit 1
fi

echo "[c-runtime] hosted structural index: the cursor seam and its vector pass (#jerindex)"
if [ "${BCIR_SKIP_DELEGATED_GATES:-0}" = "1" ]; then
  echo "  SKIP hosted structural index (BCIR_SKIP_DELEGATED_GATES=1; CTest runs it as cpp-jer-index)"
elif bash "${ROOT}/tools/cpp/check_jer_index.sh" 2>&1 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -eq 0 ]; then
  echo "  PASS hosted structural index (freestanding core preserved, tier is real not scalar)"
else
  echo "  FAIL: the hosted structural index did not build, or a tier degraded to scalar"; exit 1
fi

echo "[c-runtime] hosted SIMD rail: scalar-identical status/offset at every tier (#jersimd)"
if [ "${BCIR_SKIP_DELEGATED_GATES:-0}" = "1" ]; then
  echo "  SKIP hosted SIMD rail (BCIR_SKIP_DELEGATED_GATES=1; CTest runs it as cpp-jer-simd)"
elif bash "${ROOT}/tools/cpp/check_jer_simd.sh" 2>&1 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -eq 0 ]; then
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
if [ "${BCIR_SKIP_DELEGATED_GATES:-0}" = "1" ]; then
  echo "  SKIP C<->C++ hand-off seam (BCIR_SKIP_DELEGATED_GATES=1; CTest runs it as cpp-handoff)"
elif bash "${ROOT}/tools/cpp/check_handoff.sh" 2>&1 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -eq 0 ]; then
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
if [ "${BCIR_SKIP_DELEGATED_GATES:-0}" = "1" ]; then
  echo "  SKIP SYCL backend differential oracle (BCIR_SKIP_DELEGATED_GATES=1; CTest runs it as cpp-sycl)"
elif bash "${ROOT}/tools/cpp/check_sycl.sh" 2>&1 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -eq 0 ]; then
  echo "  PASS SYCL SAXPY differential (portable C++ fallback == BCIR reference; device path when -fsycl present)"
else
  echo "  FAIL: the SYCL backend differential oracle did not compile or agree"; exit 1
fi

echo "[c-runtime] ok"
