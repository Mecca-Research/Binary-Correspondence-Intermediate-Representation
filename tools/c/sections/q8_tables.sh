#!/usr/bin/env bash
# q8_tables: frozen Q8 table (#embed / fallback): build + self-check (C11 + C23)
#
# One section of tools/c/check_runtime.sh, run against harness binaries built elsewhere: BCIR Make
# builds them (the harness once per C standard, from the real source) and runs this script as a
# task, whose verdict the gate shows; the CMake project builds the same binaries from the same
# manifest (the harness and its C11 `variant`) and runs this script as the `c-section-q8_tables`
# CTest entry. The body is the gate's section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md),
# so the two run one text, and tools/build/section_parity.py holds the two builds' outputs
# byte-identical. The table's drift check is bcir/tests/test_q8_embed.py's: it holds the committed
# files to a fresh emission without writing into the tree, where the gate's regenerate-and-diff,
# retired with its compile lines (BUILD-8), wrote tracked files.
#
#   usage: q8_tables.sh <test_q8_tables_c11> <test_q8_tables>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
[ $# -eq 2 ] || { echo "usage: $(basename "$0") <test_q8_tables_c11> <test_q8_tables>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
C11="$(abs "$1")"; [ -x "${C11}" ] || { echo "  FAIL: C11: ${C11} is not an executable harness"; exit 2; }
C23="$(abs "$2")"; [ -x "${C23}" ] || { echo "  FAIL: C23: ${C23} is not an executable harness"; exit 2; }
declare -A BY_STD=([c11]="${C11}" [c23]="${C23}")

# runtime/c/bcir_q8_tables.h bakes runtime/c/q8_tiers.bin with C23 #embed where the compiler has
# it and a generated array where it does not; either way the harness self-checks the table.
for std in c11 c23; do
  "${BY_STD[${std}]}" | grep -q "^OK q8" || { echo "  FAIL: Q8 self-check under -std=${std}"; exit 1; }
done
echo "  PASS Q8 table (#embed-guarded; fallback self-check OK under C11 + C23)"
