#!/usr/bin/env bash
# writeguard: bcir-cc WRITE-guard: an OOB store fails-fast, never clamps (#writeguard)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the
# `c-section-writeguard` CTest entry. The body is the gate's section text moved here (BUILD-2,
# docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and tools/build/section_parity.py holds the
# two builds' outputs byte-identical. It compiles what bcir-cc emits with CC, which the caller
# names: BCIR Make passes the gate's, the CTest entry the configured C compiler, and the
# section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> writeguard.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> writeguard.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# WRITE-guard adversarial test (#writeguard, §5.12): a clamped OOB *store* silently redirects the write onto
# a valid element (a[extent-1]) -- data corruption disguised as recovery. So a STORE index site emits the
# WRITE guard `BCIR_CHK_W`, whose handler is `noreturn` and NEVER clamps: under the SAME confident clamp
# policy that recovers a read, an OOB store must ABORT (not corrupt a[7]). The emit names the store site
# `BCIR_CHK_W(...)` and the load site `BCIR_CHK(...)` -- both rails identically (gated in test_c_cfront.py).
printf 'unsigned el_sw(unsigned i, unsigned j, unsigned v){ unsigned a[8]; for(unsigned k=0u;k<8u;k++) a[k]=k*3u; a[i]=v; return a[j]; }\n' \
  > "${tmp}/sw.c"
"${BCIR_CC}" --emit-c "${tmp}/sw.c" > "${tmp}/sw_emit.c" || { echo "  FAIL: --emit-c (write-guard)"; exit 1; }
# the STORE sites use BCIR_CHK_W, the LOAD site uses the read BCIR_CHK -- distinguished per site.
{ grep -q 'a\[BCIR_CHK_W(.*) *\] *= *v;' "${tmp}/sw_emit.c" \
  && grep -q '= a\[BCIR_CHK(.*"el_sw:a")\]' "${tmp}/sw_emit.c"; } \
  || { echo "  FAIL: store site not WRITE-guarded / load site not READ-guarded"; cat "${tmp}/sw_emit.c"; exit 1; }
{ echo '#include <stdio.h>'; echo '#include <stdlib.h>'; echo '#include "bcir_quarantine_recover.h"'
  cat "${tmp}/sw_emit.c"
  cat <<'DRV'
int main(int c, char **v){
  /* the SAME confident clamp policy used for the read-recovery test above (threshold 500, conf 900). */
  static const bcir_recover_rule clampall[] = {{"el_sw:a", BCIR_RECOVER_CLAMP, 900}};
  bcir_recover_set_policy(clampall, 1, 500);
  int mode = c > 1 ? atoi(v[1]) : 0;
  if (mode == 0) {                          /* an OOB READ still clamps to a[7] = 7*3 = 21 */
    printf("%u\n", bcir_el_sw(0u, 99u, 555u));
  } else {                                  /* an OOB STORE must abort BEFORE the redirect corrupts a[7] */
    (void)bcir_el_sw(99u, 7u, 777u);
    printf("NO-ABORT a[7]=%u\n", 777u);     /* reaching here means the store was silently clamped -> BUG */
  }
  return 0; }
DRV
} > "${tmp}/sw_main.c"
"${CC}" -std=c23 -O2 -I "${C}" "${tmp}/sw_main.c" "${C}/bcir_quarantine.c" "${C}/bcir_quarantine_recover.c" -o "${tmp}/sw_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 -I "${C}" "${tmp}/sw_main.c" "${C}/bcir_quarantine.c" "${C}/bcir_quarantine_recover.c" -o "${tmp}/sw_h" \
  || { echo "  FAIL: write-guard override build"; exit 1; }
sw_read="$("${tmp}/sw_h" 0)"                                   # OOB read: clamps to a[7] = 21
[ "${sw_read}" = "21" ] || { echo "  FAIL: OOB read did not clamp (got '${sw_read}', want 21)"; exit 1; }
sw_wr="$("${tmp}/sw_h" 1 2>&1 1>/dev/null)"; sw_rc=$?          # OOB store: must abort, never reach NO-ABORT
{ [ "${sw_rc}" != "0" ] && ! printf '%s' "${sw_wr}" | grep -q "NO-ABORT" \
  && printf '%s' "${sw_wr}" | grep -q "out-of-bounds store cannot be clamped"; } \
  && echo "  PASS writeguard: OOB read clamps (a[7]=21) but OOB store ABORTS (no a[7] corruption) (#writeguard)" \
  || { echo "  FAIL: OOB store did not fail-fast (rc=${sw_rc}: ${sw_wr})"; exit 1; }
