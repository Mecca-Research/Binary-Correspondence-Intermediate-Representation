#!/usr/bin/env bash
# barrier: memory-fence intrinsics: typed kinded barriered edge, per-ISA emit, native assemble (#barrier)
#
# One section of tools/c/check_runtime.sh: BCIR Make runs this script as a task, whose verdict the
# gate shows, and the CMake project runs it as the `c-section-barrier` CTest entry. The body is the
# gate's section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text. It
# takes no binary: it judges what the Python oracle emits for a C source under CC, which the caller
# names -- BCIR Make passes the gate's, the CTest entry the configured C compiler -- and under the
# gcc and clang on PATH. The manifest marks it compiler_only, and tools/build/section_parity.py
# holds its output stable over two runs.
#
#   usage: CC=<compiler> barrier.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 0 ] || { echo "usage: CC=<compiler> barrier.sh" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> barrier.sh" >&2; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Memory-fence (hardware barrier) intrinsics (#barrier, ASM3): __sync_synchronize / __atomic_thread_fence /
# the C11 atomic_thread_fence (all FULL/seq_cst, op `c.fence`) + the x86-conventional _mm_mfence (full) /
# _mm_lfence (acquire) / _mm_sfence (release) lower to a typed, KINDED, barriered BARRIER edge; the per-ISA
# instruction is emitted behind `--target` (x86 mfence/lfence/sfence; aarch64 dmb ish/ishld/ishst; riscv64
# fence rw,rw / r,rw / rw,w), always with the required `"memory"` compiler-barrier clobber. Every ISA has a
# fence, so a target outside the three families keeps the portable __atomic_thread_fence default (no
# unsupported-diagnostic path -- unlike port I/O). The emit text is arch-independent (checked on every host);
# the emitted NATIVE-arch barrier is ASSEMBLED on its own lane (x86 assembles mfence/lfence/sfence; the
# aarch64 lane assembles dmb ish/ishld/ishst) -- non-native emits are NOT assembled (no cross sysroot).
# the native target for THIS host's arch (the only one we can assemble without a cross sysroot).
case "$(uname -m)" in
  x86_64|amd64|x86|i386|i486|i586|i686) brr_native="x86_64-linux"; brr_family="x86";;
  aarch64|arm64) brr_native="aarch64-linux"; brr_family="arm";;
  riscv64) brr_native="riscv64-linux"; brr_family="riscv";;
  *) brr_native=""; brr_family="";;
esac
cat > "${tmp}/cfront_barrier.c" <<'BRRC'
void b_full(void){ __sync_synchronize(); }          /* full (seq_cst) fence -> c.fence */
void b_full11(void){ atomic_thread_fence(5); }       /* C11 <stdatomic.h> full fence -> c.fence */
void b_mfence(void){ _mm_mfence(); }                 /* x86 mfence -> c.fence (full) */
void b_lfence(void){ _mm_lfence(); }                 /* x86 lfence -> c.fence.acquire (load fence) */
void b_sfence(void){ _mm_sfence(); }                 /* x86 sfence -> c.fence.release (store fence) */
BRRC
# emit for the NATIVE target (so the asm assembles); a host outside the three families emits x86_64-linux for
# the TEXT checks but assembles nothing.
BRR_TARGET="${brr_native:-x86_64-linux}" FX="${tmp}/cfront_barrier.c" "${PYTHON}" - > "${tmp}/barrier_emit.c" <<'PY' || { echo "  FAIL: python barrier emit"; exit 1; }
import os, re
from bcir.frontends.cfront import compile_unit
from bcir.frontends.cfront.emit import emit_function
from bcir.model import Opcode
r = compile_unit(open(os.environ['FX']).read(), check_clang=False, target=os.environ['BRR_TARGET'])
assert r.is_clean, [(d.law, d.message) for d in r.diagnostics]
# the edges are typed + kinded + barriered (the IR-level contract): BARRIER opcode, barriered hazard.
fen = [c for lf in r.lowered.functions.values() for c in lf.claims if c.op == 'c.fence' or c.op.startswith('c.fence.')]
assert len(fen) == 5, f"expected 5 fence edges, got {len(fen)}"
assert all(c.hazard == 'barriered' for c in fen), "a fence edge is not barriered"
assert all(c.opcode == Opcode.BARRIER for c in fen), "a fence edge is not the BARRIER opcode"
ops = sorted({c.op for c in fen})
assert ops == ['c.fence', 'c.fence.acquire', 'c.fence.release'], f"unexpected fence op set: {ops}"
out = ["#include <stdint.h>"]
for lf in r.lowered.functions.values():
    t = re.sub(r"/\*.*?\*/\n?", "", emit_function(lf), flags=re.S)   # drop the attestation comment
    assert '__asm__ __volatile__' in t and ':::' in t and '"memory"' in t, "the fence edge lost its barrier emit"
    out.append(t)
print("\n\n".join(out))
PY
# the emit carries the per-ISA fence mnemonics + the required "memory" compiler-barrier clobber, per family.
case "${brr_family}" in
  arm)   m_full="dmb ish"; m_acq="dmb ishld"; m_rel="dmb ishst";;
  riscv) m_full="fence rw,rw"; m_acq="fence r,rw"; m_rel="fence rw,w";;
  *)     m_full="mfence"; m_acq="lfence"; m_rel="sfence";;     # x86 + the default-text host
esac
{ grep -q "\"${m_full}\" ::: \"memory\"" "${tmp}/barrier_emit.c" \
  && grep -q "\"${m_acq}\" ::: \"memory\"" "${tmp}/barrier_emit.c" \
  && grep -q "\"${m_rel}\" ::: \"memory\"" "${tmp}/barrier_emit.c"; } \
  && echo "  PASS barrier: emit carries the per-ISA ${m_full}/${m_acq}/${m_rel} + the required \"memory\" clobber" \
  || { echo "  FAIL: barrier emit missing the per-ISA fence instruction"; cat "${tmp}/barrier_emit.c"; exit 1; }
# ASSEMBLE-ONLY (-c) the NATIVE-arch fence: prove the emitted barrier is valid asm the toolchain accepts. The
# emit is native for THIS host (target = brr_native), so a native gcc/clang assembles it without a cross
# sysroot. Non-native fence emits are NOT assembled (the aarch64 lane has no x86 sysroot, and vice-versa) --
# their validity is proven on their own CI lane, the emit-text check above is arch-independent.
if [ -n "${brr_native}" ]; then
  brr_ok=1; brr_seen=""
  for cc in "${CC}" "$(command -v gcc)" "$(command -v clang)"; do
    [ -n "${cc}" ] && [ -x "${cc}" ] || continue
    case " ${brr_seen} " in *" ${cc} "*) continue;; esac       # de-dup (CC may already be gcc/clang)
    brr_seen="${brr_seen} ${cc}"
    "${cc}" -std=c11 -pedantic -c "${tmp}/barrier_emit.c" -o "${tmp}/barrier_${cc##*/}.o" 2>/dev/null \
      || { echo "  FAIL: barrier emit did not ASSEMBLE under ${cc}"; brr_ok=0; break; }
    [ -f "${tmp}/barrier_${cc##*/}.o" ] || { echo "  FAIL: barrier object not produced by ${cc}"; brr_ok=0; break; }
  done
  [ "${brr_ok}" = "1" ] \
    && echo "  PASS barrier: emitted native ${brr_family} fence ASSEMBLES under every CC (-c, assemble-only)" \
    || exit 1
else
  echo "  SKIP barrier assemble: host arch outside the x86/aarch64/riscv64 families (emit-text check ran)"
fi
# the non-native fence emit is arch-independent TEXT (no assemble needed): the aarch64 dmb-ish family + the
# riscv64 fence family are emitted correctly even off-arch, proving the per-ISA table is keyed off --target.
brr_text="$("${PYTHON}" -c "
import re
from bcir.frontends.cfront import compile_unit
from bcir.frontends.cfront.emit import emit_function
def body(src, target):
    r = compile_unit(src, check_clang=False, target=target)
    lf = r.lowered.functions['f']
    return re.sub(r'/\*.*?\*/\n?', '', emit_function(lf), flags=re.S)
ok = True
ok &= 'dmb ish' in body('void f(void){ _mm_mfence(); }', 'aarch64-linux')
ok &= 'dmb ishld' in body('void f(void){ _mm_lfence(); }', 'aarch64-linux')
ok &= 'dmb ishst' in body('void f(void){ _mm_sfence(); }', 'aarch64-linux')
ok &= 'fence rw,rw' in body('void f(void){ _mm_mfence(); }', 'riscv64-linux')
ok &= 'fence r,rw' in body('void f(void){ _mm_lfence(); }', 'riscv64-linux')
ok &= 'fence rw,w' in body('void f(void){ _mm_sfence(); }', 'riscv64-linux')
print('OK' if ok else 'NO')")" \
  || { echo "  FAIL: barrier per-ISA text probe"; exit 1; }
[ "${brr_text}" = "OK" ] \
  && echo "  PASS barrier: the aarch64 dmb-ish + riscv64 fence families emit correctly off-arch (per-ISA --target keying)" \
  || { echo "  FAIL: barrier per-ISA off-arch emit text is wrong"; exit 1; }
