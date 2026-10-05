#!/usr/bin/env bash
# inlineasm: inline assembly: ISA-neutral trusted opaque edge, verbatim emit + memory barrier (#inlineasm)
#
# One section of tools/c/check_runtime.sh: BCIR Make runs this script as a task, whose verdict the
# gate shows, and the CMake project runs it as the `c-section-inlineasm` CTest entry. The body is
# the gate's section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text.
# It takes no binary: it judges what the Python oracle emits for a C source under CC, which the
# caller names -- BCIR Make passes the gate's, the CTest entry the configured C compiler -- and
# under the gcc and clang on PATH. The manifest marks it compiler_only, and
# tools/build/section_parity.py holds its output stable over two runs.
#
#   usage: CC=<compiler> inlineasm.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 0 ] || { echo "usage: CC=<compiler> inlineasm.sh" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> inlineasm.sh" >&2; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Inline assembly (#inlineasm): GNU `asm`/`__asm__` is an ISA-NEUTRAL trusted opaque effect edge (ASM1) --
# the template is re-emitted VERBATIM as a `__asm__ [__volatile__] (...)` statement; BCIR owns only the
# calling side (operands + constraints + clobbers + ordering). This is a PYTHON-frontend feature (the C twin
# bcir_cfront.c does not parse asm yet), so the probe emits via `compile_unit` (not bcir-cc --emit-c). It
# proves the emit compiles under BOTH the default CC and -- when present -- a second compiler, AND that an
# `asm volatile("" ::: "memory")` compiler barrier wrapped around a store/load does NOT change the observable
# value (it only constrains ordering). Uses the reserved __asm__/__volatile__ spellings + the ISA-neutral
# `"=r"(out) : "0"(in)` tied-register copy + empty-template `"memory"` barrier, so it compiles on any ISA.
cat > "${tmp}/cfront_asm.c" <<'ASMC'
unsigned asm_copy(unsigned x){ unsigned y = 0; __asm__("" : "=r"(y) : "0"(x)); return y; }
unsigned asm_barrier(unsigned *p, unsigned x){
  *p = x;                                  /* store ... */
  __asm__ __volatile__("" ::: "memory");   /* an ordering fence between the store and the load */
  return *p;                               /* ... load: the barrier must NOT change the value (== x) */
}
void asm_basic(void){ __asm__("nop"); __asm__ __volatile__("" ::: "memory"); }
ASMC
FX="${tmp}/cfront_asm.c" "${PYTHON}" - > "${tmp}/asm_emit.c" <<'PY' || { echo "  FAIL: python asm emit"; exit 1; }
import os, re
from bcir.frontends.cfront import compile_unit
from bcir.frontends.cfront.emit import emit_function
r = compile_unit(open(os.environ['FX']).read(), check_clang=False)
assert r.is_clean, [(d.law, d.message) for d in r.diagnostics]
out = []
for lf in r.lowered.functions.values():
    t = re.sub(r"/\*.*?\*/\n?", "", emit_function(lf), flags=re.S)   # drop the attestation comment
    assert "__asm__" in t, "the asm edge was eliminated from the emit"
    out.append(t)
print("\n\n".join(out))
PY
grep -q '__asm__ __volatile__ ("" :  :  : "memory")' "${tmp}/asm_emit.c" \
  && echo "  PASS inlineasm: emit carries the verbatim __asm__ __volatile__ memory barrier" \
  || { echo "  FAIL: inlineasm emit missing the verbatim barrier"; cat "${tmp}/asm_emit.c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  cat "${tmp}/asm_emit.c"
  cat <<'DRV'
int main(void){
  unsigned buf = 0;
  for(unsigned x=0; x<5000u; x++){
    if(bcir_asm_copy(x) != x){ printf("COPY@%u\n", x); return 1; }
    if(bcir_asm_barrier(&buf, x) != x){ printf("BARRIER@%u\n", x); return 1; }
  }
  bcir_asm_basic();
  puts("MATCH"); return 0;
}
DRV
} > "${tmp}/asm_harness.c"
asm_ok=1; asm_seen=""
for cc in "${CC}" "$(command -v gcc)" "$(command -v clang)"; do
  [ -n "${cc}" ] && [ -x "${cc}" ] || continue
  case " ${asm_seen} " in *" ${cc} "*) continue;; esac     # de-dup (CC may already be gcc/clang)
  asm_seen="${asm_seen} ${cc}"
  "${cc}" -std=c11 -pedantic -O2 "${tmp}/asm_harness.c" -o "${tmp}/asm_h" 2>/dev/null \
    || { echo "  FAIL: inlineasm harness build (${cc})"; asm_ok=0; break; }
  ar="$("${tmp}/asm_h")"
  [ "${ar}" = "MATCH" ] || { echo "  FAIL: inlineasm behaviour (${cc}: ${ar})"; asm_ok=0; break; }
done
[ "${asm_ok}" = "1" ] \
  && echo "  PASS inlineasm: __asm__ copy + memory-barrier store/load == value-preserving (ISA-neutral, every CC)" \
  || exit 1
