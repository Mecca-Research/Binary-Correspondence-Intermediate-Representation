#!/usr/bin/env bash
# Assemble-smoke-test for the asm-edge -convert-bcir-to-llvm lowerings (the ANTI-MASKING gate).
#
# The FileCheck pass tests (portio.mlir, inline_asm.mlir, creg.mlir, volatile_mmio.mlir) are TEXT-ONLY:
# they never feed the lowered IR to a real backend, so a lowering that PRINTS the right MLIR but emits
# LLVM IR that does NOT assemble passes silently (exactly what happened -- the GCC `%w1`/`=a,Nd` portio
# forms text-checked but `llc` rejected them: "invalid register name" / "couldn't allocate output
# register"). This harness pipes each asm-edge op's lowered IR through the REAL backend:
#
#     bcir-opt -convert-bcir-to-llvm <fixture>
#       | mlir-translate-20 --mlir-to-llvmir          # MLIR LLVM-dialect -> textual LLVM IR
#       | llc-20 -filetype=obj -o /dev/null           # MUST exit 0 (real object emission)
#
# and greps the `llc-20 -filetype=asm` output for the expected instruction (`inb %dx, %al`, `outb %al,
# %dx`, `mov %cr3`, ...). The fixture (mlir/test/passes/asm_lowering_smoke.mlir) wraps the bcir ops in
# `llvm.func` bodies so the post-lowering module is all-LLVM-dialect (there is no bundled func->llvm
# conversion in -convert-bcir-to-llvm), which mlir-translate --mlir-to-llvmir requires.
#
# Guarded on llc-20/mlir-translate-20 availability: skips CLEANLY when absent (like the FileCheck guard
# in check_passes.sh), but when present it MUST pass. Sourced by check_passes.sh; also runnable directly.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BO="${BCIR_OPT:-$(find "${ROOT}/build/mlir-build" -name bcir-opt -type f 2>/dev/null | head -1)}"
FIX="${ROOT}/mlir/test/passes/asm_lowering_smoke.mlir"

# Resolve the backend tools version-agnostically: prefer the versioned -20 names the brief names, then
# fall back to any installed major / unversioned name on PATH.
MT="$(command -v mlir-translate-20 || command -v mlir-translate-22 || command -v mlir-translate || true)"
LLC="$(command -v llc-20 || command -v llc-22 || command -v llc || true)"

if [ -z "${BO}" ]; then
  echo "[asm-smoke] bcir-opt not built; run tools/wsl/build_mlir.sh first (skipping)." >&2
  return 0 2>/dev/null || exit 0
fi
if [ -z "${MT}" ] || [ -z "${LLC}" ]; then
  echo "[asm-smoke] mlir-translate / llc not found; skipping assemble-smoke-test (text checks still ran)."
  return 0 2>/dev/null || exit 0
fi

# Lower the whole fixture once, translate to LLVM IR once, then (a) emit a real object (exit 0) and
# (b) emit asm for the per-op instruction greps. Any stage failing is a hard FAIL.
asm_smoke_fail=0
LL="$(mktemp)"; ASM="$(mktemp)"
trap 'rm -f "${LL}" "${ASM}"' RETURN 2>/dev/null || true

if ! "${BO}" -convert-bcir-to-llvm "${FIX}" 2>/tmp/asm_smoke_e | "${MT}" --mlir-to-llvmir >"${LL}" 2>>/tmp/asm_smoke_e; then
  echo "  FAIL asm-smoke: bcir-opt | mlir-translate did not produce LLVM IR"; cat /tmp/asm_smoke_e; asm_smoke_fail=1
fi

if [ "${asm_smoke_fail}" -eq 0 ]; then
  # (a) real object emission MUST exit 0 (the assemble gate).
  if "${LLC}" -filetype=obj "${LL}" -o /dev/null 2>/tmp/asm_smoke_e; then
    echo "  PASS asm-smoke: every asm-edge op produces a real .o (llc -filetype=obj exit 0)"
  else
    echo "  FAIL asm-smoke: llc -filetype=obj did NOT assemble the lowered IR"; cat /tmp/asm_smoke_e; asm_smoke_fail=1
  fi
  # (b) per-op instruction greps over the emitted asm (proves the RIGHT instruction, not just any .o).
  "${LLC}" -filetype=asm "${LL}" -o "${ASM}" 2>/dev/null
  check_instr() { # human-label, grep-ERE
    if grep -Eq "$2" "${ASM}"; then
      echo "    ok   $1 -> $(grep -Em1 "$2" "${ASM}" | sed 's/^[[:space:]]*//')"
    else
      echo "    MISS $1 (expected /$2/ in the emitted asm)"; asm_smoke_fail=1
    fi
  }
  check_instr "portio in.b"   'inb[[:space:]]+%dx, %al'
  check_instr "portio in.w"   'inw[[:space:]]+%dx, %ax'
  check_instr "portio in.l"   'inl[[:space:]]+%dx, %eax'
  check_instr "portio out.b"  'outb[[:space:]]+%al, %dx'
  check_instr "portio out.w"  'outw[[:space:]]+%ax, %dx'
  check_instr "portio out.l"  'outl[[:space:]]+%eax, %dx'
  check_instr "bcir.asm mov"  'movl[[:space:]]+%edi, %eax'
  check_instr "creg_read cr3" 'mov[a-z]?[[:space:]]+%cr3,'
  check_instr "creg_write cr3" 'mov[a-z]?[[:space:]]+%[a-z0-9]+, %cr3'
  check_instr "volatile_load"  'movl[[:space:]]+\(%rdi\), %eax'
  check_instr "volatile_store" 'movl[[:space:]]+%edi, \(%rsi\)'
fi

if [ "${asm_smoke_fail}" -eq 0 ]; then
  echo "  PASS asm-smoke: all asm-edge lowerings assemble to the expected instruction"
else
  echo "  FAIL asm-smoke"
fi
# Export the result for a sourcing parent (check_passes.sh) to fold into its own fail counter.
ASM_SMOKE_FAIL="${asm_smoke_fail}"
return "${asm_smoke_fail}" 2>/dev/null || exit "${asm_smoke_fail}"
