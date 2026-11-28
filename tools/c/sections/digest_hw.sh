#!/usr/bin/env bash
# digest_hw: SHA-256 and CRC-32 on the CPU's own instructions (bcir_digest_hw.c): test_digest_hw's vectors and
# its paths against the portable code, and every path the probe reports against hashlib and zlib
#
# One section of tools/c/check_runtime.sh, run against a binary built elsewhere: BCIR Make builds it from
# runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the CMake project builds it
# from the same manifest and runs this script as the `c-section-digest_hw` CTest entry, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. The comparison with hashlib and zlib
# is the Python test's own (bcir/tests/test_c_digest_hw.py), run here against the built harness. Which paths ran is
# the host's, the same for both builds; their throughput is not, so it is `test_digest_hw --bench`'s, never this
# section's.
#
#   usage: digest_hw.sh <test_digest_hw>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: $(basename "$0") <test_digest_hw>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }

out="$("${HARNESS}" 2>&1)" || { echo "  FAIL: hardware digest contracts"; echo "${out}"; exit 1; }
[ "$(printf '%s\n' "${out}" | tail -n 1)" = "test_digest_hw: OK" ] || { echo "  FAIL: unexpected digest harness output"; echo "${out}"; exit 1; }
paths="$(printf '%s\n' "${out}" | sed -n '1p')"
echo "  PASS hardware digests: FIPS 180-4's vectors and zlib's check value on every path, each path == the portable code (${paths})"
out="$("${PYTHON}" - "${HARNESS}" <<'PY' 2>&1
import sys
from bcir.tests.test_c_digest_hw import check_paths
seen = check_paths(sys.argv[1])
print(f"{len(seen['ran'])} masks, {sum(seen['ran'].values())} requests")
PY
)" || { echo "  FAIL: test_digest_hw != hashlib and zlib"; echo "${out}" | tail -20; exit 1; }
echo "  PASS test_digest_hw == hashlib and zlib: ${out}"
