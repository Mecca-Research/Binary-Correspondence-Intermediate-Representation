#!/usr/bin/env bash
# csub_ratchet: bcir-cc over every production unit, held to runtime/c/csub_ratchet.json (CSUB-1)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds them from
# runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the CMake project builds them
# from the same manifest and runs this script as the `c-section-csub_ratchet` CTest entry. The C subset's ratchet
# (docs/research/BCIR_C_SUBSET_AUDIT_2026-11-10.md section 4): each unit runtime/c/MEMORY_CLASSIFICATION.txt lists
# compiles R1-R18 clean or is refused for exactly the first blocker the record names, with its construct and the
# rung that lifts it -- so the set only grows by being recorded, and a unit added to the runtime is recorded too.
#
#   usage: csub_ratchet.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
[ $# -eq 1 ] || { echo "usage: $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }

out="$("${PYTHON}" "${ROOT}/tools/c/csub_ratchet.py" --cc "${BCIR_CC}" 2>&1)"; rc=$?
if [ "${rc}" -ne 0 ]; then
  echo "${out}"
  echo "  FAIL: the C subset's ratchet (tools/c/csub_ratchet.py exited ${rc})"
  exit 1
fi
echo "  PASS ${out#csub-ratchet: PASS }"
