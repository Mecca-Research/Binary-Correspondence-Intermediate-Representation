#!/usr/bin/env bash
# decoder_train_tsan: the native trainer's deterministic thread pool under ThreadSanitizer (NDT-THREADS):
# every pool kernel and three complete training steps at one to four threads race-free, and an
# injected partition fault reported as a race
#
# One section of tools/c/check_runtime.sh, run against harness binaries built elsewhere: BCIR Make
# builds them (the decoder-training harness built with -fsanitize=thread, and the same build of a
# tensor unit whose attention backward runs one query head per task, so the query heads of one KV
# head -- which accumulate into the same dk and dv -- are tasks two threads may claim) and runs this script as
# a task, whose verdict the gate shows; the CMake project builds the same binaries from the same
# manifest (sanitizer `variants`) and runs this script as the `c-section-decoder_train_tsan` CTest
# entry; tools/build/section_parity.py holds the two builds' outputs byte-identical. Whether
# ThreadSanitizer is available here is the builder's question, answered by tools/build/sanitizer.py
# before these binaries exist (BCIR_REQUIRE_TSAN makes its absence a failure where a job installed
# the runtime, laws.md L2). The harness's pool_identity holds the partitions' RESULTS to the serial
# kernels; this section holds their MEMORY DISCIPLINE: a partition that lets two tasks write one
# element is a data race whether or not a given run's timing changes the sum (laws.md L11: the
# witness hits the law it exists to test), and ThreadSanitizer finds it from the happens-before
# order alone. Since NDT-POOL the pool's threads claim tasks as they come, so which of the fault's
# conflicting pairs two threads split is the run's timing, never the arithmetic; the harness runs
# the backward in many jobs at two to four threads, and the first of them reports it (every run
# observed: the report comes before the harness's first PASS line).
#
#   usage: decoder_train_tsan.sh <test_decoder_train_tsan> <test_decoder_train_race>
set -uo pipefail

[ $# -eq 2 ] || { echo "usage: $(basename "$0") <test_decoder_train_tsan> <test_decoder_train_race>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
TSAN="$(abs "$1")"; [ -x "${TSAN}" ] || { echo "  FAIL: TSAN: ${TSAN} is not an executable harness"; exit 2; }
RACE="$(abs "$2")"; [ -x "${RACE}" ] || { echo "  FAIL: RACE: ${RACE} is not an executable harness"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# ASLR is disabled for the runs (setarch -R) because older TSan runtimes reject the high-entropy
# mappings newer kernels hand out (as the ring's section does).
norandom=()
if command -v setarch >/dev/null 2>&1 && setarch "$(uname -m)" -R true 2>/dev/null; then
  norandom=(setarch "$(uname -m)" -R)
fi
if ! "${norandom[@]}" "${TSAN}" > "${tmp}/tsan.log" 2>&1 || grep -q "WARNING: ThreadSanitizer" "${tmp}/tsan.log"; then
  echo "  FAIL: ThreadSanitizer or the decoder laws failed on the pool:"; tail -40 "${tmp}/tsan.log"; exit 1
fi
grep -q "PASS native pool kernels" "${tmp}/tsan.log" \
  || { echo "  FAIL: the ThreadSanitizer harness did not run the pool identity"; tail -20 "${tmp}/tsan.log"; exit 1; }
echo "  PASS native trainer's pool under ThreadSanitizer (GEMM, attention, elementwise, rows, the update, the pool's claims and three training steps at 1..4 threads: no data race, every result the serial kernels')"
# the first report ends the run (TSan exits 66): the harness's own assertions, which the racing
# sums fail, are not reached, so the verdict is the sanitizer's report alone
TSAN_OPTIONS="halt_on_error=1${TSAN_OPTIONS:+:$TSAN_OPTIONS}" "${norandom[@]}" "${RACE}" > "${tmp}/race.log" 2>&1
status=$?
grep -q "WARNING: ThreadSanitizer" "${tmp}/race.log" \
  || { echo "  FAIL: ThreadSanitizer passed a pool whose attention backward splits one KV head's query heads over two threads (exit ${status})"; tail -20 "${tmp}/race.log"; exit 1; }
echo "  PASS ThreadSanitizer fires on an injected partition fault (one KV head's query heads on two threads race on its dk/dv)"
