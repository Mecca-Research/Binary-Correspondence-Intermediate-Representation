#!/usr/bin/env bash
# recover: bcir-cc masked emit + recovery override: the recorded two-truth crossing (#recover)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the `c-section-recover`
# CTest entry. The body is the gate's section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md),
# so the two run one text, and tools/build/section_parity.py holds the two builds' outputs
# byte-identical. It compiles what bcir-cc emits with CC, which the caller names: BCIR Make passes
# the gate's, the CTest entry the configured C compiler, and the section-parity gate the same one to
# both runs.
#
#   usage: CC=<compiler> recover.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> recover.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# The ML-layer / debugger RECOVERY override (#recover, §5.12): the same masked emit, linked against the
# reference strong override (bcir_quarantine_recover.c) instead of relying on the weak abort default. A
# frozen per-site policy proposes (action, confidence); the crossing collapses it at a frozen threshold into
# a CLASSICAL action (clamp / abort) and RECORDS the decide -- the only sanctioned two-truth crossing. An
# admitted clamp survives on a valid element (el_pick fills a[k]=k*3, so a[7]=21); an under-confident
# proposal fail-fasts.
# The masked unit #emitlink emits: the gate wrote it in that section's temp directory,
# and a section script makes its own inputs.
printf 'unsigned el_pick(unsigned i){ unsigned a[8]; for(unsigned k=0u;k<8u;k++) a[k]=k*3u; return a[i]; }\n' \
  > "${tmp}/el.c"
"${BCIR_CC}" --emit-c "${tmp}/el.c" > "${tmp}/el_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdio.h>'; echo '#include <stdlib.h>'; echo '#include "bcir_quarantine_recover.h"'
  cat "${tmp}/el_emit.c"
  cat <<'DRV'
int main(int c, char **v){
  static const bcir_recover_rule confident[] = {{"el_pick:a", BCIR_RECOVER_CLAMP, 900}};
  static const bcir_recover_rule underconf[] = {{"el_pick:a", BCIR_RECOVER_CLAMP, 300}};
  int abort_mode = c > 1 && v[1][0] == '1';
  bcir_recover_set_policy(abort_mode ? underconf : confident, 1, 500);  /* threshold 500 */
  printf("%u\n", bcir_el_pick(99u));            /* index 99 out of [0,8): the handler decides */
  bcir_decide_report(stdout);
  return 0; }
DRV
} > "${tmp}/rec_main.c"
"${CC}" -std=c23 -O2 -I "${C}" "${tmp}/rec_main.c" "${C}/bcir_quarantine.c" "${C}/bcir_quarantine_recover.c" -o "${tmp}/rec_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 -I "${C}" "${tmp}/rec_main.c" "${C}/bcir_quarantine.c" "${C}/bcir_quarantine_recover.c" -o "${tmp}/rec_h" \
  || { echo "  FAIL: recovery override build"; exit 1; }
rec_clamp="$("${tmp}/rec_h" 0)"                                # admitted: confidence 900 >= threshold 500
{ printf '%s' "${rec_clamp}" | grep -q "^21$" \
  && printf '%s' "${rec_clamp}" | grep -q "admitted, clamp to index 7"; } \
  || { echo "  FAIL: admitted clamp recovery (got '${rec_clamp}')"; exit 1; }
rec_abrt="$("${tmp}/rec_h" 1 2>&1 1>/dev/null)"; rec_rc=$?     # rejected: confidence 300 < threshold 500
{ [ "${rec_rc}" != "0" ] && printf '%s' "${rec_abrt}" | grep -q "recovery rejected"; } \
  && echo "  PASS recover: frozen-policy decide -> admitted clamp (a[7]=21) / under-confident abort (#recover)" \
  || { echo "  FAIL: rejected path did not fail-fast (rc=${rec_rc}: ${rec_abrt})"; exit 1; }
