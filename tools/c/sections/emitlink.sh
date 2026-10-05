#!/usr/bin/env bash
# emitlink: bcir-cc --emit-c links the runtime: self-contained masked unit (#emitlink)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-emitlink` CTest entry. The body is the gate's section text
# moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> emitlink.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> emitlink.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# The bcir-cc driver links the runtime (#emitlink, §5.12): a masked (bounds-promoted) access emits
# `a[BCIR_CHK(...)]`, which references the bounds-quarantine runtime ABI. The driver's --emit-c output is
# a SELF-CONTAINED translation unit -- it pulls in bcir_quarantine.h -- so `--emit-c | cc -I runtime/c -
# runtime/c/bcir_quarantine.c` compiles AND links on its own (no hand-supplied guard). In-bounds the guard
# is transparent (returns the value); out-of-bounds it calls the weak handler, which records the provenance
# (naming the `<func>:<array>` site) and aborts. A unit with no masked access pulls in nothing.
printf 'unsigned el_pick(unsigned i){ unsigned a[8]; for(unsigned k=0u;k<8u;k++) a[k]=k*3u; return a[i]; }\n' \
  > "${tmp}/el.c"
"${BCIR_CC}" --emit-c "${tmp}/el.c" > "${tmp}/el_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
grep -q '#include "bcir_quarantine.h"' "${tmp}/el_emit.c" \
  || { echo "  FAIL: emit not self-contained (no runtime include for a masked unit)"; exit 1; }
{ echo '#include <stdio.h>'; echo '#include <stdlib.h>'
  cat "${tmp}/el_emit.c"
  echo 'int main(int c, char **v){ (void)c; printf("%u\n", bcir_el_pick((unsigned)atoi(v[1]))); return 0; }'
} > "${tmp}/el_main.c"
# compile + link the driver's emit against ONLY the runtime (-I runtime/c, link bcir_quarantine.c) -- proof
# the output is standalone-linkable, no injected stub.
"${CC}" -std=c23 -O2 -I "${C}" "${tmp}/el_main.c" "${C}/bcir_quarantine.c" -o "${tmp}/el_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 -I "${C}" "${tmp}/el_main.c" "${C}/bcir_quarantine.c" -o "${tmp}/el_h" \
  || { echo "  FAIL: --emit-c output did not link against the runtime"; exit 1; }
elr_in="$("${tmp}/el_h" 3)"                                   # in-bounds: a[3] = 9
[ "${elr_in}" = "9" ] || { echo "  FAIL: in-bounds masked access (got '${elr_in}', want 9)"; exit 1; }
elr_oob="$("${tmp}/el_h" 99 2>&1 1>/dev/null)"; elr_rc=$?     # out-of-bounds: the weak handler aborts
{ [ "${elr_rc}" != "0" ] && printf '%s' "${elr_oob}" | grep -q "el_pick:a"; } \
  && echo "  PASS emitlink: --emit-c links the runtime; in-bounds value + OOB quarantine (site el_pick:a)" \
  || { echo "  FAIL: OOB did not quarantine via the linked runtime (rc=${elr_rc}: ${elr_oob})"; exit 1; }
