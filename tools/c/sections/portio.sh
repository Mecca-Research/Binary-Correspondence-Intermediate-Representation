#!/usr/bin/env bash
# portio: port-mapped I/O intrinsics: typed barriered edge, x86 in/out emit, assemble-only (#portio)
#
# One section of tools/c/check_runtime.sh: BCIR Make runs this script as a task, whose verdict the
# gate shows, and the CMake project runs it as the `c-section-portio` CTest entry. The body is the
# gate's section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text. It
# takes no binary: it judges what the Python oracle emits for a C source under CC, which the caller
# names -- BCIR Make passes the gate's, the CTest entry the configured C compiler -- and under the
# gcc and clang on PATH. The manifest marks it compiler_only, and tools/build/section_parity.py
# holds its output stable over two runs.
#
#   usage: CC=<compiler> portio.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 0 ] || { echo "usage: CC=<compiler> portio.sh" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> portio.sh" >&2; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Port-mapped I/O intrinsics (#portio, ASM2): inb/inw/inl / outb/outw/outl lower to a typed, isolated,
# BARRIERED I/O-port edge; the per-ISA `in`/`out` instruction is emitted behind `--target` (x86 emits the
# real instruction as `__asm__ __volatile__`; non-x86 has no port I/O -> the honest unsupported diagnostic).
# HONEST BOUNDARY: executing `in`/`out` from userspace TRAPS (it needs iopl/ioperm + ring-0), so this probe
# ASSEMBLES the emitted asm (`-c`, assemble-only) to prove it is valid x86 the toolchain accepts -- it NEVER
# links or runs it. The emitted text must carry the right `in`/`out` instruction + operands.
cat > "${tmp}/cfront_portio.c" <<'PIOC'
unsigned pio_inb(unsigned port){ return inb(port); }      /* read u8  from a port */
unsigned pio_inw(unsigned port){ return inw(port); }      /* read u16 */
unsigned pio_inl(unsigned port){ return inl(port); }      /* read u32 */
void pio_outb(unsigned v, unsigned port){ outb(v, port); } /* Linux out(value, port): value first, port second */
void pio_outw(unsigned v, unsigned port){ outw(v, port); }
void pio_outl(unsigned v, unsigned port){ outl(v, port); }
PIOC
FX="${tmp}/cfront_portio.c" "${PYTHON}" - > "${tmp}/portio_emit.c" <<'PY' || { echo "  FAIL: python portio emit"; exit 1; }
import os, re
from bcir.frontends.cfront import compile_unit
from bcir.frontends.cfront.emit import emit_function
from bcir.model import Domain
r = compile_unit(open(os.environ['FX']).read(), check_clang=False)   # default target x86_64-linux
assert r.is_clean, [(d.law, d.message) for d in r.diagnostics]
# the edges are typed + isolated + barriered (the IR-level contract): MMIO domain, barriered hazard.
pio = [c for lf in r.lowered.functions.values() for c in lf.claims if c.op.startswith('c.portio.')]
assert len(pio) == 6, f"expected 6 port-I/O edges, got {len(pio)}"
assert all(c.hazard == 'barriered' for c in pio), "a port-I/O edge is not barriered"
assert all(c.domain == Domain.MMIO for c in pio), "a port-I/O edge is not isolated in the MMIO I/O domain"
out = ["#include <stdint.h>"]
for lf in r.lowered.functions.values():
    t = re.sub(r"/\*.*?\*/\n?", "", emit_function(lf), flags=re.S)   # drop the attestation comment
    assert "__asm__ __volatile__" in t, "the port-I/O edge was eliminated from the emit"
    out.append(t)
print("\n\n".join(out))
PY
# the emit carries the real x86 in/out instructions + the standard <asm/io.h> operand constraints.
{ grep -q '"inb %w1, %b0" : "=a"' "${tmp}/portio_emit.c" \
  && grep -q '"outb %b0, %w1" :  : "a"' "${tmp}/portio_emit.c" \
  && grep -q '"inl %w1, %k0" : "=a"' "${tmp}/portio_emit.c" \
  && grep -q '"Nd"' "${tmp}/portio_emit.c"; } \
  && echo "  PASS portio: emit carries the real x86 inb/outb/inl + the <asm/io.h> =a/a/Nd constraints" \
  || { echo "  FAIL: portio emit missing the real in/out instruction"; cat "${tmp}/portio_emit.c"; exit 1; }
# ASSEMBLE-ONLY (-c): prove the emitted x86 asm is valid the toolchain accepts. NEVER linked/run (the
# `in`/`out` instructions are privileged). Every available CC must assemble it.
# The emitted asm is x86 `in`/`out`, which a non-x86 host's native assembler cannot accept (and a clang
# cross-compile `-target x86_64-linux-gnu` cannot satisfy without an x86 sysroot the ARM CI lane lacks). So
# the assemble check runs ONLY on an x86 host and SKIPS on a non-x86 host (e.g. the aarch64 lane) -- the
# emit-text check above is arch-independent, and the x86 asm's validity is proven on the x86 CI lanes.
case "$(uname -m)" in x86_64|amd64|x86|i386|i486|i586|i686) pio_host_x86=1;; *) pio_host_x86=0;; esac
if [ "${pio_host_x86}" = "1" ]; then
  pio_ok=1; pio_seen=""
  for cc in "${CC}" "$(command -v gcc)" "$(command -v clang)"; do
    [ -n "${cc}" ] && [ -x "${cc}" ] || continue
    case " ${pio_seen} " in *" ${cc} "*) continue;; esac   # de-dup (CC may already be gcc/clang)
    pio_seen="${pio_seen} ${cc}"
    "${cc}" -std=c11 -pedantic -c "${tmp}/portio_emit.c" -o "${tmp}/portio_${cc##*/}.o" 2>/dev/null \
      || { echo "  FAIL: portio emit did not ASSEMBLE under ${cc}"; pio_ok=0; break; }
    [ -f "${tmp}/portio_${cc##*/}.o" ] || { echo "  FAIL: portio object not produced by ${cc}"; pio_ok=0; break; }
  done
  [ "${pio_ok}" = "1" ] \
    && echo "  PASS portio: emitted x86 in/out ASSEMBLES under every CC (-c, assemble-only; execution is privileged)" \
    || exit 1
else
  echo "  SKIP portio assemble: non-x86 host cannot assemble x86 in/out (validity proven on the x86 lanes; emit-text + fallback checks ran)"
fi
# the non-x86 honest diagnostic: ARM/RISC-V have no port I/O -> route to the LLVM fallback (the oracle).
pio_fb="$("${PYTHON}" -c "
from bcir.frontends.cfront.pipeline import compile_with_fallback
r = compile_with_fallback('unsigned f(void){ return inb(0x60); }', check_clang=False, target='aarch64-linux')
print('FB' if (r.needs_fallback and 'requires an x86 target' in r.fallback) else 'NO')")" \
  || { echo "  FAIL: portio non-x86 fallback probe"; exit 1; }
[ "${pio_fb}" = "FB" ] \
  && echo "  PASS portio: a non-x86 target (aarch64) has no port I/O -> honest unsupported diagnostic (LLVM fallback)" \
  || { echo "  FAIL: portio non-x86 did not fall back honestly"; exit 1; }
