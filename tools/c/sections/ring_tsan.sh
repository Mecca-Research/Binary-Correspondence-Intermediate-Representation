#!/usr/bin/env bash
# ring_tsan: live SPSC ring under ThreadSanitizer: backpressure + overwrite stress race-free, and an injected race reported
#
# One section of tools/c/check_runtime.sh, run against harness binaries built elsewhere: the gate
# compiles them (the ring harness built with -fsanitize=thread, and the same build of a ring whose
# relaxed atomic stores are made plain) and calls this script; the CMake project builds the same
# binaries from runtime/manifest.json (sanitizer `variants`) and runs this script as the
# `c-section-ring_tsan` CTest entry. The body is the gate's section text moved here (BUILD-2,
# docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and tools/build/section_parity.py holds
# the two builds' outputs byte-identical. Whether ThreadSanitizer is available here -- a trivial
# TSan program builds AND runs -- is the builder's question, answered by tools/build/sanitizer.py
# before these binaries exist (BCIR_REQUIRE_TSAN makes its absence a failure where a job installed
# the runtime, laws.md L2).
#
#   usage: ring_tsan.sh <test_ring_tsan> <test_ring_plain>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
[ $# -eq 2 ] || { echo "usage: $(basename "$0") <test_ring_tsan> <test_ring_plain>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
TSAN="$(abs "$1")"; [ -x "${TSAN}" ] || { echo "  FAIL: TSAN: ${TSAN} is not an executable harness"; exit 2; }
PLAIN="$(abs "$2")"; [ -x "${PLAIN}" ] || { echo "  FAIL: PLAIN: ${PLAIN} is not an executable harness"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# ThreadSanitizer models C11 atomics: the stress runs must report no data race, and a ring whose
# relaxed atomic stores are made plain must be REPORTED as racing (not merely fail). ASLR is
# disabled for the runs (setarch -R) because older TSan runtimes reject the high-entropy mappings
# newer kernels hand out.
norandom=()
if command -v setarch >/dev/null 2>&1 && setarch "$(uname -m)" -R true 2>/dev/null; then
  norandom=(setarch "$(uname -m)" -R)
fi
ring_tsan() {  # <binary> <log prefix> -> exit 0 when both stress runs are race-free and clean
  for mode in bp ow; do
    if ! "${norandom[@]}" "$1" --stress "${mode}" 40000 7 > "$2.${mode}.log" 2>&1; then return 1; fi
    if grep -q "WARNING: ThreadSanitizer" "$2.${mode}.log"; then return 1; fi
  done
  return 0
}
if ! ring_tsan "${TSAN}" "${tmp}/test_ring_tsan"; then
  echo "  FAIL: ThreadSanitizer or the stress laws failed on the ring:"; cat "${tmp}/test_ring_tsan".*.log | tail -40; exit 1
fi
echo "  PASS live ring under ThreadSanitizer (backpressure + overwrite stress: no data race, nothing torn or unaccounted)"
if ring_tsan "${PLAIN}" "${tmp}/test_ring_plain"; then
  echo "  FAIL: ThreadSanitizer passed a ring whose shared words are written with plain stores"; exit 1
fi
grep -q "WARNING: ThreadSanitizer" "${tmp}/test_ring_plain".*.log \
  || { echo "  FAIL: the plain-store ring failed, but not with a ThreadSanitizer race report"; exit 1; }
echo "  PASS ThreadSanitizer fires on an injected race (relaxed atomic stores made plain)"
