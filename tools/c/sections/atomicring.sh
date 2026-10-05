#!/usr/bin/env bash
# atomicring: OOB ring counter is atomic under concurrent events (#atomicring)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-atomicring` CTest entry. The body is the gate's section
# text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. The binary is the
# test_oob_counter harness, the probe program the gate used to write out from a heredoc.
#
#   usage: atomicring.sh <test_oob_counter>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: atomicring.sh <test_oob_counter>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Atomic OOB ring (#atomicring, §5.12): the counter and payload publication must both be synchronized.
# An atomic fetch-add alone gives writers distinct slots but still lets a reporter race a non-atomic struct
# update (C undefined behaviour / torn audit records). The hosted implementation serializes each short
# publish/snapshot section. This test hammers writers while a reporter snapshots concurrently, then asserts
# the total equals N*M. If the freestanding build has no atomics the contract is single-threaded and this
# hosted concurrency test is skipped.
race_out="$("${HARNESS}")"; race_rc=$?
{ [ "${race_rc}" = "0" ] && { printf '%s' "${race_out}" | grep -q "^EXACT" || printf '%s' "${race_out}" | grep -q "^SKIP"; }; } \
  && echo "  PASS atomicring: ${race_out} (concurrent OOB events do not scramble the total)" \
  || { echo "  FAIL: OOB ring counter raced (${race_out})"; exit 1; }
