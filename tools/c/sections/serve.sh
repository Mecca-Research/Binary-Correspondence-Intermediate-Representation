#!/usr/bin/env bash
# serve: the serving control plane (bcir_serve.c, N10): test_serve's contracts, and its page tables and constrained
# steps == the oracle's (bcir/frontends/models/paged_kv.py's PagedKV, serve.py's check_token_dfa and _masked_argmax)
#
# One section of tools/c/check_runtime.sh, run against a binary built elsewhere: BCIR Make builds it from
# runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the CMake project builds it
# from the same manifest and runs this script as the `c-section-serve` CTest entry, and tools/build/section_parity.py
# holds the two builds' outputs byte-identical. The comparison with the oracle is the Python test's own
# (bcir/tests/test_c_serve.py), run here against the built harness.
#
#   usage: serve.sh <test_serve>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: $(basename "$0") <test_serve>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }

out="$("${HARNESS}" 2>&1)" || { echo "  FAIL: serving control plane contracts"; echo "${out}"; exit 1; }
[ "${out}" = "test_serve: OK" ] || { echo "  FAIL: unexpected serving harness output"; echo "${out}"; exit 1; }
echo "  PASS serving control plane: too few page records, page arithmetic past 64 bits, the schema's layout, order and transition bound, every NULL"
out="$("${PYTHON}" - "${HARNESS}" <<'PY' 2>&1
import sys
from bcir.tests.test_c_serve import check_driver
seen = check_driver(sys.argv[1])
refused = sum(v for k, v in seen.items() if k.startswith("kv.err"))
print(f"{seen['tables'] + refused} page tables ({refused} refused), {seen['rows']} rows, {seen['evicted']} evictions; "
      f"{seen['schemas'] + seen['refused']} schemas ({seen['refused']} refused), {seen['steps']} steps")
PY
)" || { echo "  FAIL: test_serve != the oracle"; echo "${out}" | tail -20; exit 1; }
echo "  PASS test_serve == the oracle: ${out}"
