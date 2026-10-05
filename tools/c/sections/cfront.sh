#!/usr/bin/env bash
# cfront: plug-in C frontend (bcir_cfront): IR freestanding + Python<->C parity
#
# One section of tools/c/check_runtime.sh, run against harness binaries built elsewhere: BCIR Make
# builds them from runtime/manifest.json and runs this script as a task, whose verdict the gate
# shows; the CMake project builds them from the same manifest and runs this script as the
# `c-section-cfront` CTest entry. The body is the gate's section text moved here (BUILD-2,
# docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and tools/build/section_parity.py holds the
# two builds' outputs byte-identical. The freestanding probe of bcir_cir.h is the freestanding
# section's.
#
#   usage: cfront.sh <test_cfront>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: $(basename "$0") <test_cfront>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# L1-L8 + type-model + casts + char literals + interleaved decls + funcptr dispatch + §5.8 + Phase D driver + str ops + hex-float + math.h (#320-#324) + ABI data model (#abi) + scalar global r/w (#globals) + effects (#effects) + integer promotions/UAC (#intpromote) + designated init (#designated) + local aggregate init (#aggregate) + restrict (#restrict) + array stores (#astore) + local arrays (#localarr)
FIXTURES="cfront_regmap.c cfront_array.c cfront_array2d.c cfront_widerow.c cfront_deref.c cfront_callgraph.c cfront_branch.c cfront_while.c cfront_for.c cfront_dowhile.c cfront_continue.c cfront_switch.c cfront_goto.c cfront_incdec.c cfront_macros.c cfront_ppinc.c cfront_structret.c cfront_packed.c cfront_typedef.c cfront_enum.c cfront_ternary.c cfront_sizeof.c cfront_cast.c cfront_alignof.c cfront_signed.c cfront_signedcmp.c cfront_longunary.c cfront_charlit.c cfront_strtab.c cfront_strconcat.c cfront_widelit.c cfront_static.c cfront_global.c cfront_compound.c cfront_logic.c cfront_float.c cfront_floatcast.c cfront_rmw.c cfront_bitfield.c cfront_bfcompound.c cfront_union.c cfront_interleave.c cfront_funcptr.c cfront_dispatch.c cfront_integration.c cfront_regdriver.c cfront_atomic.c cfront_cmpxchg.c cfront_atomic11.c cfront_atomic_xchg.c cfront_driver.c cfront_driver_uart.c cfront_strsizeof.c cfront_strval.c cfront_hexfloat.c cfront_mathh.c cfront_mathh_mixed.c cfront_mathh_long.c cfront_mathh_ptr.c cfront_calltyped.c cfront_comments.c cfront_abi.c cfront_global_rw.c cfront_effects.c cfront_intpromote.c cfront_dispatch_table.c cfront_agginit.c cfront_restrict.c cfront_arraystore.c cfront_localarray.c cfront_shiftassign.c cfront_extern.c cfront_switchfall.c cfront_ptrarith.c cfront_threadlocal.c cfront_multidecl.c cfront_commastep.c cfront_structmulti.c cfront_memberarray.c cfront_emptystmt.c cfront_ptrstore.c cfront_loopreuse.c cfront_loopscope.c cfront_blockscope.c cfront_localmd.c cfront_nestmember.c cfront_boolnorm.c cfront_unarypromote.c cfront_floatsigncast.c cfront_intsigncast.c cfront_boolcast.c cfront_signedbf.c cfront_signedload.c cfront_enumtype.c cfront_ptrlocal.c cfront_ptrvalue.c cfront_ptrfield.c cfront_ptr2ptr.c cfront_fieldderef.c cfront_ptrsign.c cfront_fnptrchain.c cfront_multiptr.c cfront_chartypes.c \
cfront_complit.c cfront_typeof.c cfront_structinit.c cfront_arraylit.c cfront_variadic.c cfront_compoundwide.c cfront_extvariadic.c cfront_longdouble.c cfront_generic.c cfront_designate.c cfront_nestoffset.c cfront_addrmember.c cfront_atomiclocal.c cfront_builtins.c cfront_stmtexpr.c \
cfront_bitint.c cfront_bitint_member.c cfront_bitint_mixed.c cfront_bitint_bitfield.c"
# + C23 `_BitInt(N)` (#bitint / #bitintmember / #bitintmixed / #bitintbitfield): exact-width bit-precise
# ints -- same-type + MIXED-WIDTH arithmetic (the wider `_BitInt` wins the C23 rank), PLAIN members, and
# `_BitInt(N) m:W` BITFIELDS; the result type + bitfield layout are verified == Clang in test_c_cfront.py.
# Precompute EVERY oracle summary in one python process (import compile_unit once) -- the old
# python-per-fixture loop paid ~0.3s of interpreter+import startup each (~30s over the fixture set).
"${PYTHON}" - "${C}" ${FIXTURES} > "${tmp}/py_sums.txt" <<'PY' || { echo "  FAIL: python lowering (batch)"; exit 1; }
import os, re, sys
from bcir.frontends.cfront import compile_unit
from bcir.model import Domain
from bcir.verify import cfront_structural_digest          # the cross-rail per-claim STRUCTURAL digest
cdir = sys.argv[1]
for fx in sys.argv[2:]:
    try:
        src = open(os.path.join(cdir, fx)).read()
        inc = {h: open(os.path.join(cdir, h)).read() for h in re.findall(r'#include\s+"([^"]+)"', src)
               if os.path.exists(os.path.join(cdir, h))}
        r = compile_unit(src, check_clang=False, includes=inc or None)
        fns = r.lowered.functions; lf = fns[next(reversed(fns))]
        mmio = sum(1 for c in lf.claims if c.op == 'c.load' and c.domain == Domain.MMIO)
        bf = sum(1 for c in lf.claims if c.op == 'c.bf.get'); kn = sum(1 for c in lf.claims if c.op == 'c.const')
        bo = sum(1 for c in lf.claims if c.op.startswith('c.bin.')); ca = sum(1 for c in lf.claims if c.op.startswith('c.call'))
        repro = sum(1 for f in fns.values() if getattr(f, 'reproducible', False))  # A1.3: matches the C twin's repro=N
        dg = cfront_structural_digest(r.lowered)           # byte-identical to the C twin's bcir_cfront_digest
        print(f"{fx}\tfuncs={len(fns)} claims={len(lf.claims)} mmio={mmio} bf={bf} const={kn} binop={bo} call={ca} repro={repro} ok={1 if r.is_clean else 0} digest={dg:016x}")
    except Exception as e:
        sys.stderr.write(f"oracle lowering failed for {fx}: {e}\n"); sys.exit(1)
PY
for fx in ${FIXTURES}; do
  c_sum="$("${HARNESS}" "${C}/${fx}" | sed -n '1p')" || { echo "  FAIL: C run ${fx}: ${c_sum}"; exit 1; }
  py_sum="$(awk -F'\t' -v f="${fx}" '$1==f{print $2; exit}' "${tmp}/py_sums.txt")"
  [ -n "${py_sum}" ] || { echo "  FAIL: no precomputed oracle summary for ${fx}"; exit 1; }
  [ "${c_sum}" = "${py_sum}" ] \
    && echo "  PASS parity ${fx} (oracle == C: ${c_sum})" \
    || { echo "  FAIL: parity ${fx} (C='${c_sum}' PY='${py_sum}')"; exit 1; }
done
