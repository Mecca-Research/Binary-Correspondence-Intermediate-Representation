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
#       | mlir-translate --mlir-to-llvmir             # one coherently resolved LLVM major
#       | llc -filetype=obj -o /dev/null              # MUST exit 0 (real object emission)
#
# and greps the resolved `llc -filetype=asm` output for the expected instruction (`inb %dx, %al`, `outb %al,
# %dx`, `mov %cr3`, ...). The fixture (mlir/test/passes/asm_lowering_smoke.mlir) wraps the bcir ops in
# `llvm.func` bodies so the post-lowering module is all-LLVM-dialect (there is no bundled func->llvm
# conversion in -convert-bcir-to-llvm), which mlir-translate --mlir-to-llvmir requires.
#
# Guarded on one coherent llc/mlir-translate installation: skips CLEANLY when absent (like the
# FileCheck guard in check_passes.sh), but when present it MUST pass. Sourced by check_passes.sh;
# also runnable directly.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BO="${BCIR_OPT:-$(find "${ROOT}/build/mlir-build" -name bcir-opt -type f 2>/dev/null | head -1)}"
FIX="${ROOT}/mlir/test/passes/asm_lowering_smoke.mlir"

# Resolve mlir-translate + llc as one coherent installation. This reuses the same
# LLVM_BIN/LLVM_SUFFIX/unversioned/highest-common-major policy as the Python JIT,
# WASM, and object pipelines; independently probing names could mix LLVM majors.
MT=""; LLC=""; OBJDUMP=""; LLVM_RESOLVE_MESSAGE="LLVM tool resolver did not return a result"
eval "$(PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python3 - <<'PY'
import os
from pathlib import Path
import re
import shlex

from bcir.toolchain import resolve_llvm_tools

resolved = resolve_llvm_tools(
    "mlir-translate", "llc", pipeline="MLIR asm-edge object gate")
if resolved.ok:
    mt = resolved.paths["mlir-translate"]
    llc = resolved.paths["llc"]
    match = re.fullmatch(r"llc(?P<suffix>-\d+)?(?P<exe>\.exe)?", Path(llc).name,
                         re.IGNORECASE)
    objdump = ""
    if match:
        candidate = Path(llc).with_name(
            "llvm-objdump" + (match.group("suffix") or "") + (match.group("exe") or ""))
        if candidate.is_file():
            objdump = str(candidate)
    message = resolved.message
else:
    mt = llc = objdump = ""
    message = resolved.message
for name, value in (("MT", mt), ("LLC", llc), ("OBJDUMP", objdump),
                    ("LLVM_RESOLVE_MESSAGE", message)):
    print(f"{name}={shlex.quote(value)}")
PY
)"

if [ -z "${BO}" ]; then
  echo "[asm-smoke] bcir-opt not built; run tools/wsl/build_mlir.sh first (skipping)." >&2
  return 0 2>/dev/null || exit 0
fi
if [ -z "${MT}" ] || [ -z "${LLC}" ]; then
  echo "[asm-smoke] ${LLVM_RESOLVE_MESSAGE}; skipping assemble-smoke-test (text checks still ran)."
  return 0 2>/dev/null || exit 0
fi

# Lower the whole fixture once, translate to LLVM IR once, then (a) emit a real object (exit 0) and
# (b) emit asm for the per-op instruction greps. Any stage failing is a hard FAIL.
asm_smoke_fail=0
LL="$(mktemp)"; ASM="$(mktemp)"; OBJ="$(mktemp)"; DIS="$(mktemp)"; ERR="$(mktemp)"

if ! "${BO}" -convert-bcir-to-llvm "${FIX}" 2>"${ERR}" | "${MT}" --mlir-to-llvmir >"${LL}" 2>>"${ERR}"; then
  echo "  FAIL asm-smoke: bcir-opt | mlir-translate did not produce LLVM IR"; cat "${ERR}"; asm_smoke_fail=1
fi

if [ "${asm_smoke_fail}" -eq 0 ]; then
  # (a) real object emission MUST exit 0 (the assemble gate). Retain the temporary object long enough
  # to disassemble it below; no generated artifact enters the worktree.
  if "${LLC}" -mtriple=x86_64-unknown-none-elf -filetype=obj "${LL}" -o "${OBJ}" 2>"${ERR}"; then
    echo "  PASS asm-smoke: every asm-edge op produces a real .o (llc -filetype=obj exit 0)"
  else
    echo "  FAIL asm-smoke: llc -filetype=obj did NOT assemble the lowered IR"; cat "${ERR}"; asm_smoke_fail=1
  fi
  # (b) per-op instruction greps over the emitted asm (proves the RIGHT instruction, not just any .o).
  "${LLC}" -mtriple=x86_64-unknown-none-elf -filetype=asm "${LL}" -o "${ASM}" 2>/dev/null
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
  check_instr "msr_read rdmsr" 'rdmsr'
  check_instr "msr_write wrmsr" 'wrmsr'
  check_instr "entry stack" 'lea[q]?[[:space:]]+bcir_stack_top\(%rip\), %rsp'
  check_instr "entry handoff" 'jmp[q]?[[:space:]]+bcir_kernel_main'
  check_instr "descriptor lgdt" 'lgdt[q]?[[:space:]]+\('
  check_instr "descriptor lidt" 'lidt[q]?[[:space:]]+\('
  check_instr "task register ltr" 'ltr[w]?[[:space:]]+%[a-z0-9]+'
  check_instr "segment far return" 'lretq'
  check_instr "interrupt return" 'iretq'
  check_instr "user transition swapgs" 'swapgs'

  # Pin the architecturally different exception frames: IRQ 32 gets a synthetic zero error code;
  # page fault 14 already has a CPU-pushed error code and must not get another one.
  irq32_body="$(sed -n '/^bcir_irq32:/,/^[[:space:]]*\.size[[:space:]]\+bcir_irq32/p' "${ASM}")"
  pf_body="$(sed -n '/^bcir_page_fault:/,/^[[:space:]]*\.size[[:space:]]\+bcir_page_fault/p' "${ASM}")"
  boot_body="$(sed -n '/^bcir_boot:/,/^[[:space:]]*\.size[[:space:]]\+bcir_boot/p' "${ASM}")"
  if printf '%s\n' "${boot_body}" | grep -Eq 'cli' &&
     printf '%s\n' "${boot_body}" | grep -Eq 'andq[[:space:]]+\$-16, %rsp' &&
     printf '%s\n' "${boot_body}" | grep -Eq 'pushq[[:space:]]+\$0'; then
    echo "    ok   entry invariant -> interrupts masked before aligned stack handoff"
  else
    echo "    MISS entry invariant (cli + alignment + sentinel required)"; asm_smoke_fail=1
  fi
  if printf '%s\n' "${irq32_body}" | grep -Eq 'pushq[[:space:]]+\$0' &&
     ! printf '%s\n' "${pf_body}" | grep -Eq 'pushq[[:space:]]+\$0'; then
    echo "    ok   interrupt frame normalization -> IRQ synthetic error / #PF hardware error"
  else
    echo "    MISS interrupt frame normalization (IRQ32 must push zero; #PF must not)"; asm_smoke_fail=1
  fi

  error_shape_fail=0
  for vector in 10 11 12 13 17 21 30; do
    body="$(sed -n "/^bcir_error${vector}:/,/^[[:space:]]*\\.size[[:space:]]\\+bcir_error${vector}/p" "${ASM}")"
    if [ -z "${body}" ] ||
       ! printf '%s\n' "${body}" | grep -Eq "pushq[[:space:]]+\\\$${vector}([^0-9]|$)"; then
      echo "    MISS vector ${vector}: trampoline or vector push is absent"; error_shape_fail=1
    elif printf '%s\n' "${body}" | grep -Eq 'pushq[[:space:]]+\$0'; then
      echo "    MISS vector ${vector}: hardware error code was double-normalized"; error_shape_fail=1
    fi
  done
  if [ "${error_shape_fail}" -eq 0 ]; then
    echo "    ok   hardware error-code table -> all 8 normal-op error vectors (#DF/#VC refused)"
  else
    echo "    MISS complete hardware error-code table"; asm_smoke_fail=1
  fi

  # Entry and exit swapgs must be paired from the original private RPL. The
  # mutable return CS is checked against it so a handler cannot change iret shape.
  if [ "$(printf '%s\n' "${irq32_body}" | grep -Ec 'swapgs')" -eq 2 ] &&
     printf '%s\n' "${irq32_body}" | grep -Eq 'movq[[:space:]]+144\(%rsp\), %r13' &&
     printf '%s\n' "${irq32_body}" | grep -Eq 'cmpq[[:space:]]+\$3, %r13' &&
     printf '%s\n' "${irq32_body}" | grep -Eq 'cmpq[[:space:]]+%r13, %rax' &&
     printf '%s\n' "${irq32_body}" | grep -Eq 'testq[[:space:]]+%r13, %r13' &&
     printf '%s\n' "${irq32_body}" | grep -Eq 'lfence' &&
     printf '%s\n' "${irq32_body}" | grep -Eq 'ud2' &&
     printf '%s\n' "${irq32_body}" | grep -Eq 'cli'; then
    echo "    ok   entry policy -> private RPL, paired swapgs, class guard, cli + lfence"
  else
    echo "    MISS immutable swapgs pairing"; asm_smoke_fail=1
  fi

  # (c) Inspect the actual encoded object when llvm-objdump is available. This catches a module-asm
  # spelling that survives `llc -filetype=asm` but encodes a different instruction form.
  if [ -n "${OBJDUMP}" ]; then
    if "${OBJDUMP}" -d "${OBJ}" >"${DIS}" 2>>"${ERR}"; then
      check_obj_instr() {
        if grep -Eq "$2" "${DIS}"; then
          echo "    obj  $1 -> $(grep -Em1 "$2" "${DIS}" | sed 's/^[[:space:]]*//')"
        else
          echo "    MISS object disassembly $1 (expected /$2/)"; asm_smoke_fail=1
        fi
      }
      check_obj_instr "lgdt" 'lgdt[q]?[[:space:]]'
      check_obj_instr "lidt" 'lidt[q]?[[:space:]]'
      check_obj_instr "ltr" 'ltr[w]?[[:space:]]'
      check_obj_instr "far return" 'lretq'
      check_obj_instr "interrupt return" 'iretq'
    else
      echo "  FAIL asm-smoke: llvm-objdump could not disassemble emitted object"; cat "${ERR}"; asm_smoke_fail=1
    fi
  else
    echo "    skip object disassembly: llvm-objdump unavailable (object emission still passed)"
  fi
fi

if [ "${asm_smoke_fail}" -eq 0 ]; then
  echo "  PASS asm-smoke: all asm-edge lowerings assemble to the expected instruction"
else
  echo "  FAIL asm-smoke"
fi
# This file is sourced by check_passes.sh. Explicit cleanup avoids installing or
# replacing an EXIT/RETURN trap in the parent shell.
rm -f "${LL}" "${ASM}" "${OBJ}" "${DIS}" "${ERR}"
# Export the result for a sourcing parent (check_passes.sh) to fold into its own fail counter.
ASM_SMOKE_FAIL="${asm_smoke_fail}"
return "${asm_smoke_fail}" 2>/dev/null || exit "${asm_smoke_fail}"
