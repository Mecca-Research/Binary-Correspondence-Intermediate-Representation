#!/usr/bin/env bash
# asn1bench: native ASN.1 decode microbench: strict-warning build (#asn1bench)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the `c-section-asn1bench`
# CTest entry. The body is the gate's section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md),
# so the two run one text, and tools/build/section_parity.py holds the two builds' outputs
# byte-identical. The binary is the manifest tool bcir_asn1_bench, which BCIR Make builds -Werror.
#
#   usage: asn1bench.sh <bcir_asn1_bench>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: asn1bench.sh <bcir_asn1_bench>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BENCH="$(abs "$1")"; [ -x "${BENCH}" ] || { echo "  FAIL: BENCH: ${BENCH} is not an executable bcir_asn1_bench"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# The native ASN.1 decode microbench (#asn1bench, JSON roadmap J6 follow-on): the harness
# that makes a `measured` cost table possible, and therefore the reason select_certified can
# decide a timing objective at all instead of refusing every one. It is a MEASUREMENT tool,
# so what is gated here is that it builds warning-clean and answers a corpus -- the numbers
# themselves are deliberately NOT a CI assertion, because a shared runner's timings are not
# evidence about a target and pinning them would invent the false precision J6 refuses.
printf 'rounds 1 7 8\ncase DER der 3009020102040461\ncase JER jer 7b2261223a317d\nrun\n' \
  > "${tmp}/bench_cases.txt"
if "${BENCH}" < "${tmp}/bench_cases.txt" | grep -q '^done 2$'; then
  echo "  PASS native ASN.1 microbench (builds -Werror, answers a two-case corpus)"
else
  echo "  FAIL: the native ASN.1 microbench did not complete its corpus"
  exit 1
fi
