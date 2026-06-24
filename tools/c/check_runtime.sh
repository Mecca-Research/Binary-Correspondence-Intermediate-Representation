#!/usr/bin/env bash
# Validate the freestanding C StreamPack runtime: it compiles with no libc, and a
# Python-encoded StreamPack round-trips through the C decoder (ABI parity).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C="${ROOT}/runtime/c"
CC="${CC:-$(command -v clang || command -v cc || true)}"
if [ -z "${CC}" ]; then
  echo "no C compiler (clang/cc); skipping runtime check." >&2
  exit 0
fi

echo "[c-runtime] freestanding compile (-ffreestanding -nostdlib), C11 + C23"
for std in c11 c23; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -c "${C}/bcir_runtime.c" -o /dev/null \
    || { echo "  FAIL: runtime not freestanding-clean under -std=${std}"; exit 1; }
done
echo "  PASS freestanding (C11 + C23; ABI static_assert holds)"

tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT
echo "[c-runtime] build harness (C23) + Python->C ABI parity"
"${CC}" -std=c23 -O2 "${C}/bcir_runtime.c" "${C}/test_runtime.c" -I "${C}" -o "${tmp}/test_runtime" \
  || { echo "  FAIL: harness build"; exit 1; }
python3 -c "
from bcir.examples import vector_add
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.gem import hydrate
from bcir.abi import encode
m=vector_add(1024); pack=hydrate(m, optimize(m, TargetProfile.x86_avx512(), Theta.cool()))
open('${tmp}/pack.bin','wb').write(encode(pack))
" || { echo "  FAIL: python encode"; exit 1; }
out="$("${tmp}/test_runtime" "${tmp}/pack.bin")" || { echo "  FAIL: C decode"; echo "${out}"; exit 1; }
echo "${out}" | grep -q "^OK$" && echo "  PASS parity (Python encode -> C decode)" \
  || { echo "  FAIL: parity"; echo "${out}"; exit 1; }

echo "[c-runtime] ETL binary-record decoder: freestanding compile (C11 + C23)"
# bcir_binrec.c is the C twin of bcir/etl/binary.py (a second binary trust boundary).
for std in c11 c23; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -I "${C}" -c "${C}/bcir_binrec.c" -o /dev/null \
    || { echo "  FAIL: bcir_binrec not freestanding-clean under -std=${std}"; exit 1; }
done
echo "  PASS bcir_binrec freestanding (C11 + C23)"

echo "[c-runtime] StreamPack executor: freestanding compile (C11 + C23) + Python->C parity"
# bcir_exec.c is the C twin of bcir/gem/execute.py -- the deterministic phase-sliced
# executor that turns the StreamPack into a no-Python hot artifact.
for std in c11 c23; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -I "${C}" -c "${C}/bcir_exec.c" -o /dev/null \
    || { echo "  FAIL: bcir_exec not freestanding-clean under -std=${std}"; exit 1; }
done
"${CC}" -std=c23 -O2 "${C}/bcir_exec.c" "${C}/bcir_runtime.c" "${C}/test_exec.c" -I "${C}" -o "${tmp}/test_exec" \
  || { echo "  FAIL: executor harness build"; exit 1; }
python3 -c "
from bcir.examples import multi_histogram
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.gem import hydrate, execute
from bcir.abi import encode
m=multi_histogram(); r=optimize(m, TargetProfile.x86_avx512(), Theta.cool())
open('${tmp}/exec.bin','wb').write(encode(hydrate(m, r)))
open('${tmp}/exec.order','w').write(' '.join(str(i) for i in execute(m).order))
" || { echo "  FAIL: python encode"; exit 1; }
c_order="$("${tmp}/test_exec" "${tmp}/exec.bin" | sed -n 's/^order: //p')"
[ "${c_order}" = "$(cat "${tmp}/exec.order")" ] \
  && echo "  PASS executor parity (Python gem.execute == C bcir_sp_execute: ${c_order})" \
  || { echo "  FAIL: executor order parity (C='${c_order}' PY='$(cat "${tmp}/exec.order")')"; exit 1; }

echo "[c-runtime] StreamPack encoder: freestanding compile (C11 + C23) + byte-identical re-encode"
# bcir_encode.c is the C write-side twin of bcir/abi/streampack_abi.py -- the full C
# round-trip (a driver emits the artifact with no Python).
for std in c11 c23; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -I "${C}" -c "${C}/bcir_encode.c" -o /dev/null \
    || { echo "  FAIL: bcir_encode not freestanding-clean under -std=${std}"; exit 1; }
done
"${CC}" -std=c23 -O2 "${C}/bcir_encode.c" "${C}/bcir_runtime.c" "${C}/test_encode.c" -I "${C}" -o "${tmp}/test_encode" \
  || { echo "  FAIL: encoder harness build"; exit 1; }
"${tmp}/test_encode" "${tmp}/exec.bin" "${tmp}/reenc.bin" >/dev/null || { echo "  FAIL: C re-encode"; exit 1; }
if cmp -s "${tmp}/exec.bin" "${tmp}/reenc.bin"; then
  echo "  PASS encoder parity (C bcir_sp_reencode == Python encode, byte-identical)"
else
  echo "  FAIL: encoder bytes differ from the Python encoding"; exit 1
fi

echo "[c-runtime] frozen Q8 table (#embed / fallback): build + self-check (C11 + C23)"
# Drift gate: the committed runtime/c/{q8_tiers.bin,bcir_q8_tables.h} must equal a
# fresh emission from the oracle (bcir.kbcir.cost.MemoryHierarchy.default()).
python3 -m bcir.abi.q8_tables --emit >/dev/null || { echo "  FAIL: q8 emit"; exit 1; }
if ! git -C "${ROOT}" diff --quiet -- runtime/c/q8_tiers.bin runtime/c/bcir_q8_tables.h 2>/dev/null; then
  echo "  FAIL: Q8 table drifted from the oracle (regenerate: python -m bcir.abi.q8_tables --emit)"; exit 1
fi
for std in c11 c23; do
  "${CC}" -std=${std} -Wall -Wextra -I "${C}" "${C}/test_q8_tables.c" -o "${tmp}/q8_${std}" \
    || { echo "  FAIL: Q8 table build under -std=${std}"; exit 1; }
  "${tmp}/q8_${std}" | grep -q "^OK q8" || { echo "  FAIL: Q8 self-check under -std=${std}"; exit 1; }
done
echo "  PASS Q8 table (#embed-guarded; fallback self-check OK under C11 + C23)"

echo "[c-runtime] plug-in C frontend (bcir_cfront): IR freestanding + Python<->C parity"
# bcir_cir.h is the freestanding BCIR claim-graph IR; bcir_cfront.c is the host compiler
# tool (the C twin of bcir/frontends/cfront -- it lowers driver/register-map C to that IR).
printf '#include "bcir_cir.h"\nint probe(void){return (int)sizeof(bcir_claim)+(int)BCIR_OP_LOAD+(int)BCIR_DOM_MMIO;}\n' > "${tmp}/cir_probe.c"
for std in c11 c23; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -I "${C}" -c "${tmp}/cir_probe.c" -o /dev/null \
    || { echo "  FAIL: bcir_cir.h not freestanding-clean under -std=${std}"; exit 1; }
done
echo "  PASS bcir_cir.h freestanding IR (C11 + C23)"
# bcir_cfront verifies (bcir_verify.c: R1-R8+R18 / provenance digest, with R10-R11 reaching
# bcir_sp_validate in bcir_runtime.c), so the host tool links both. bcir_verify.c is a host tool
# (its diagnostic path uses snprintf), not freestanding.
CFRONT_SRCS="${C}/bcir_cfront.c ${C}/bcir_cpp.c ${C}/bcir_verify.c ${C}/bcir_runtime.c"
"${CC}" -std=c23 -O2 -Wall -Wextra ${CFRONT_SRCS} "${C}/test_cfront.c" -I "${C}" -o "${tmp}/test_cfront" 2>/dev/null \
  || "${CC}" -std=c11 -O2 ${CFRONT_SRCS} "${C}/test_cfront.c" -I "${C}" -o "${tmp}/test_cfront" \
  || { echo "  FAIL: C frontend build"; exit 1; }
# L1-L8 + type-model + casts + char literals + interleaved decls + funcptr dispatch + §5.8 + Phase D driver + str ops + hex-float + math.h (#320-#324) + ABI data model (#abi) + scalar global r/w (#globals) + effects (#effects) + integer promotions/UAC (#intpromote) + designated init (#designated) + local aggregate init (#aggregate) + restrict (#restrict) + array stores (#astore) + local arrays (#localarr)
FIXTURES="cfront_regmap.c cfront_array.c cfront_array2d.c cfront_widerow.c cfront_deref.c cfront_callgraph.c cfront_branch.c cfront_while.c cfront_for.c cfront_dowhile.c cfront_continue.c cfront_switch.c cfront_goto.c cfront_incdec.c cfront_macros.c cfront_ppinc.c cfront_structret.c cfront_packed.c cfront_typedef.c cfront_enum.c cfront_ternary.c cfront_sizeof.c cfront_cast.c cfront_alignof.c cfront_signed.c cfront_signedcmp.c cfront_longunary.c cfront_charlit.c cfront_strtab.c cfront_strconcat.c cfront_widelit.c cfront_static.c cfront_global.c cfront_compound.c cfront_logic.c cfront_float.c cfront_floatcast.c cfront_rmw.c cfront_bitfield.c cfront_bfcompound.c cfront_union.c cfront_interleave.c cfront_funcptr.c cfront_dispatch.c cfront_integration.c cfront_regdriver.c cfront_atomic.c cfront_cmpxchg.c cfront_atomic11.c cfront_atomic_xchg.c cfront_driver.c cfront_driver_uart.c cfront_strsizeof.c cfront_strval.c cfront_hexfloat.c cfront_mathh.c cfront_mathh_mixed.c cfront_mathh_long.c cfront_mathh_ptr.c cfront_calltyped.c cfront_comments.c cfront_abi.c cfront_global_rw.c cfront_effects.c cfront_intpromote.c cfront_dispatch_table.c cfront_agginit.c cfront_restrict.c cfront_arraystore.c cfront_localarray.c cfront_shiftassign.c cfront_extern.c cfront_switchfall.c cfront_ptrarith.c cfront_threadlocal.c cfront_multidecl.c cfront_commastep.c cfront_structmulti.c cfront_memberarray.c cfront_emptystmt.c cfront_ptrstore.c cfront_loopreuse.c cfront_loopscope.c cfront_blockscope.c cfront_localmd.c cfront_nestmember.c cfront_boolnorm.c cfront_unarypromote.c cfront_floatsigncast.c cfront_intsigncast.c cfront_boolcast.c cfront_signedbf.c cfront_signedload.c cfront_enumtype.c cfront_ptrlocal.c cfront_ptrvalue.c cfront_ptrfield.c cfront_ptr2ptr.c cfront_fieldderef.c cfront_ptrsign.c cfront_fnptrchain.c cfront_multiptr.c cfront_chartypes.c \
cfront_complit.c cfront_typeof.c cfront_structinit.c cfront_arraylit.c cfront_variadic.c cfront_compoundwide.c cfront_extvariadic.c cfront_longdouble.c cfront_generic.c cfront_designate.c cfront_nestoffset.c cfront_addrmember.c cfront_atomiclocal.c cfront_builtins.c cfront_stmtexpr.c"
# Precompute EVERY oracle summary in one python process (import compile_unit once) -- the old
# python-per-fixture loop paid ~0.3s of interpreter+import startup each (~30s over the fixture set).
python3 - "${C}" ${FIXTURES} > "${tmp}/py_sums.txt" <<'PY' || { echo "  FAIL: python lowering (batch)"; exit 1; }
import os, re, sys
from bcir.frontends.cfront import compile_unit
from bcir.model import Domain
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
        print(f"{fx}\tfuncs={len(fns)} claims={len(lf.claims)} mmio={mmio} bf={bf} const={kn} binop={bo} call={ca} ok={1 if r.is_clean else 0}")
    except Exception as e:
        sys.stderr.write(f"oracle lowering failed for {fx}: {e}\n"); sys.exit(1)
PY
for fx in ${FIXTURES}; do
  c_sum="$("${tmp}/test_cfront" "${C}/${fx}" | sed -n '1p')" || { echo "  FAIL: C run ${fx}: ${c_sum}"; exit 1; }
  py_sum="$(awk -F'\t' -v f="${fx}" '$1==f{print $2; exit}' "${tmp}/py_sums.txt")"
  [ -n "${py_sum}" ] || { echo "  FAIL: no precomputed oracle summary for ${fx}"; exit 1; }
  [ "${c_sum}" = "${py_sum}" ] \
    && echo "  PASS parity ${fx} (oracle == C: ${c_sum})" \
    || { echo "  FAIL: parity ${fx} (C='${c_sum}' PY='${py_sum}')"; exit 1; }
done

# Target-ABI matrix (#abi): the C twin's `--target` data model lays `long` / pointer / size_t-class
# types out exactly like the oracle's TargetABI, for every named target. The summary counts are
# target-invariant, so this compares the FOLDED sizeof constants (which carry the data-model widths):
# the C twin emits them as `= Nu;` literals; the oracle exposes them as the c.const immediates. The
# vectors must agree per target AND differ across LP64 / LLP64 / ILP32 (so the gate has teeth).
echo "[c-runtime] target-ABI matrix (bcir_cfront --target): sizeof data model == oracle (#abi)"
ABI_TARGETS="x86_64-linux aarch64-linux riscv64-linux x86_64-windows i386-linux"
# One python process for every target (was one per target): import compile_unit once.
python3 - "${C}" ${ABI_TARGETS} > "${tmp}/abi_sums.txt" <<'PY' || { echo "  FAIL: python ABI (batch)"; exit 1; }
import sys
from bcir.frontends.cfront import compile_unit
src = open(sys.argv[1] + "/cfront_abi.c").read()
for t in sys.argv[2:]:
    r = compile_unit(src, check_clang=False, target=t)
    lf = r.lowered.functions[next(reversed(r.lowered.functions))]
    print(f"{t}\t" + ','.join(str(c.imm[0]) for c in lf.claims if c.op == 'c.const'))
PY
abi_seen=""
for t in ${ABI_TARGETS}; do
  c_vals="$("${tmp}/test_cfront" --target "${t}" "${C}/cfront_abi.c" | sed -n '/----EMIT----/,$p' \
            | grep -oE '= [0-9]+u;' | grep -oE '[0-9]+' | paste -sd, -)" \
    || { echo "  FAIL: C ABI run ${t}"; exit 1; }
  py_vals="$(awk -F'\t' -v f="${t}" '$1==f{print $2; exit}' "${tmp}/abi_sums.txt")"
  [ "${c_vals}" = "${py_vals}" ] \
    && echo "  PASS ABI ${t} (sizeof model oracle == C: [${c_vals}])" \
    || { echo "  FAIL: ABI ${t} (C='[${c_vals}]' PY='[${py_vals}]')"; exit 1; }
  abi_seen="${abi_seen}${c_vals};"
done
# the three data models must produce distinct vectors (LP64 8/8/8, LLP64 long=4, ILP32 ptr=4 too).
echo "${abi_seen}" | grep -q "8,8,8,4,8;" && echo "${abi_seen}" | grep -q "4,8,8,4,8;" \
  && echo "${abi_seen}" | grep -q "4,4,4,4,8;" \
  && echo "  PASS ABI matrix spans LP64 / LLP64 / ILP32 (distinct data models)" \
  || { echo "  FAIL: ABI matrix did not span the three data models: ${abi_seen}"; exit 1; }

echo "[c-runtime] full C compile->execute loop (cfront -> plan -> hydrate -> exec, no Python)"
# bcir_plan.c + bcir_hydrate.c are freestanding (the driver-embeddable planner + StreamPack
# writer that feed the existing bcir_exec.c) -- the loop closes with no Python.
for f in bcir_plan.c bcir_hydrate.c; do
  for std in c11 c23; do
    "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -I "${C}" -c "${C}/${f}" -o /dev/null \
      || { echo "  FAIL: ${f} not freestanding-clean under -std=${std}"; exit 1; }
  done
done
LOOP_SRCS="${C}/bcir_cfront.c ${C}/bcir_cpp.c ${C}/bcir_plan.c ${C}/bcir_hydrate.c ${C}/bcir_exec.c ${C}/bcir_runtime.c ${C}/bcir_verify.c ${C}/test_cfront_loop.c"
"${CC}" -std=c23 -O2 -I "${C}" ${LOOP_SRCS} -o "${tmp}/loop" 2>/dev/null \
  || "${CC}" -std=c11 -O2 -I "${C}" ${LOOP_SRCS} -o "${tmp}/loop" \
  || { echo "  FAIL: loop build"; exit 1; }
for fx in cfront_regmap.c cfront_array.c cfront_array2d.c cfront_widerow.c cfront_deref.c cfront_callgraph.c cfront_typedef.c cfront_enum.c cfront_ternary.c cfront_sizeof.c cfront_cast.c cfront_alignof.c cfront_charlit.c cfront_strtab.c cfront_strconcat.c cfront_widelit.c cfront_static.c cfront_global.c cfront_compound.c cfront_logic.c cfront_atomic.c cfront_cmpxchg.c cfront_atomic11.c cfront_atomic_xchg.c cfront_driver.c cfront_driver_uart.c; do
  out="$("${tmp}/loop" "${C}/${fx}")" || { echo "  FAIL: loop ${fx}: ${out}"; exit 1; }
  case "${out}" in
    loop:*executed=*) echo "  PASS loop ${fx} (${out#loop: })" ;;
    *) echo "  FAIL: loop ${fx}: ${out}"; exit 1 ;;
  esac
done

echo "[c-runtime] multi-channel lowering decision (bcir_channel): channel.json -> backend pick"
# bcir_channel.c is the C twin of bcir/channels' routing seam -- it consumes channel.json and
# routes each claim to a backend; the Python<->C parity is gated in bcir/tests/test_c_channel.py.
"${CC}" -std=c23 -O2 -Wall -Wextra -I "${C}" "${C}/bcir_channel.c" "${C}/test_channel.c" -o "${tmp}/tch" 2>/dev/null \
  || "${CC}" -std=c11 -O2 -I "${C}" "${C}/bcir_channel.c" "${C}/test_channel.c" -o "${tmp}/tch" \
  || { echo "  FAIL: channel router build"; exit 1; }
CHJSON="${ROOT}/channels/example_cpu.channel.json ${ROOT}/channels/example_tpu.channel.json ${ROOT}/channels/example_pim.channel.json"
c_route="$(printf 'matmul.acc 4\ngather.load 5\nscalar.mov 0\n' | "${tmp}/tch" ${CHJSON} | sed -n 's/.*route=\([^|]*\).*/\1/p' | paste -sd, -)" \
  || { echo "  FAIL: channel route run"; exit 1; }
py_route="$(python3 -c "
from bcir.channel_plugin import load_manifest
from bcir.channels import route_claim
from bcir.model import Claim, Opcode, StrideClass
mans=[load_manifest('${ROOT}/channels/%s.channel.json'%n) for n in ('example_cpu','example_tpu','example_pim')]
cl=[('matmul.acc',StrideClass.TILE),('gather.load',StrideClass.RANDOM),('scalar.mov',StrideClass.SCALAR)]
print(','.join(route_claim(Claim(id=1,opcode=Opcode.LOAD,op=o,stride_class=s),mans).name for o,s in cl))
")" || { echo "  FAIL: python route"; exit 1; }
[ "${c_route}" = "${py_route}" ] \
  && echo "  PASS channel routing parity (Python route_claim == C bcir_channel_route: ${c_route})" \
  || { echo "  FAIL: channel routing parity (C='${c_route}' PY='${py_route}')"; exit 1; }

echo "[c-runtime] bcir-cc compiler driver: compile a driver (sibling header) + emit artifacts"
# bcir_cc.c is the cc-like driver over the full C pipeline (bcir_cpp_run_ex -I/-D -> bcir_cfront ->
# plan -> hydrate). It must compile a driver with sibling headers via a normal compile command.
"${CC}" -std=c23 -O2 -Wall -Wextra -I "${C}" "${C}/bcir_cc.c" "${C}/bcir_cpp.c" "${C}/bcir_cfront.c" \
  "${C}/bcir_verify.c" "${C}/bcir_runtime.c" "${C}/bcir_plan.c" "${C}/bcir_hydrate.c" -o "${tmp}/bcir-cc" 2>/dev/null \
  || "${CC}" -std=c11 -O2 -I "${C}" "${C}/bcir_cc.c" "${C}/bcir_cpp.c" "${C}/bcir_cfront.c" \
       "${C}/bcir_verify.c" "${C}/bcir_runtime.c" "${C}/bcir_plan.c" "${C}/bcir_hydrate.c" -o "${tmp}/bcir-cc" \
  || { echo "  FAIL: bcir-cc build"; exit 1; }
ccsum="$("${tmp}/bcir-cc" "${C}/cfront_driver_uart.c")" || { echo "  FAIL: bcir-cc compile"; echo "${ccsum}"; exit 1; }
case "${ccsum}" in
  *ok=1*) echo "  PASS bcir-cc compile (${ccsum##*: })" ;;
  *) echo "  FAIL: bcir-cc compile: ${ccsum}"; exit 1 ;;
esac

"${tmp}/bcir-cc" --emit-pack -o "${tmp}/uart.pack" "${C}/cfront_driver_uart.c" || { echo "  FAIL: bcir-cc --emit-pack"; exit 1; }
[ "$(head -c4 "${tmp}/uart.pack")" = "BSPK" ] \
  && echo "  PASS bcir-cc --emit-pack (valid StreamPack)" \
  || { echo "  FAIL: bcir-cc --emit-pack: bad magic"; exit 1; }

# The bcir-cc driver links the runtime (#emitlink, §5.12): a masked (bounds-promoted) access emits
# `a[BCIR_CHK(...)]`, which references the bounds-quarantine runtime ABI. The driver's --emit-c output is
# a SELF-CONTAINED translation unit -- it pulls in bcir_quarantine.h -- so `--emit-c | cc -I runtime/c -
# runtime/c/bcir_quarantine.c` compiles AND links on its own (no hand-supplied guard). In-bounds the guard
# is transparent (returns the value); out-of-bounds it calls the weak handler, which records the provenance
# (naming the `<func>:<array>` site) and aborts. A unit with no masked access pulls in nothing.
echo "[c-runtime] bcir-cc --emit-c links the runtime: self-contained masked unit (#emitlink)"
printf 'unsigned el_pick(unsigned i){ unsigned a[8]; for(unsigned k=0u;k<8u;k++) a[k]=k*3u; return a[i]; }\n' \
  > "${tmp}/el.c"
"${tmp}/bcir-cc" --emit-c "${tmp}/el.c" > "${tmp}/el_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
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

# The ML-layer / debugger RECOVERY override (#recover, §5.12): the same masked emit, linked against the
# reference strong override (bcir_quarantine_recover.c) instead of relying on the weak abort default. A
# frozen per-site policy proposes (action, confidence); the crossing collapses it at a frozen threshold into
# a CLASSICAL action (clamp / abort) and RECORDS the decide -- the only sanctioned two-truth crossing. An
# admitted clamp survives on a valid element (el_pick fills a[k]=k*3, so a[7]=21); an under-confident
# proposal fail-fasts.
echo "[c-runtime] bcir-cc masked emit + recovery override: the recorded two-truth crossing (#recover)"
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

# Clang-grade diagnostics (#diag): the C source-location model + caret renderer (bcir_diag.c, the C
# twin of cfront/diagnostics.py). Fed the SAME synthetic diagnostic (severity / message / byte span)
# over the same source, the C renderer's Clang-layout output (banner + source line + ^~~~ underline)
# is byte-identical to diagnostics.render() -- so the two rails share one diagnostic format,
# independent of which parser produced the error (messages aren't shared; the LAYOUT is).
echo "[c-runtime] Clang-style diagnostic renderer (bcir_diag): caret layout == oracle (#diag)"
"${CC}" -std=c23 -O2 -Wall -Wextra -I "${C}" "${C}/bcir_diag.c" "${C}/test_diag.c" -o "${tmp}/test_diag" 2>/dev/null \
  || "${CC}" -std=c11 -O2 -I "${C}" "${C}/bcir_diag.c" "${C}/test_diag.c" -o "${tmp}/test_diag" \
  || { echo "  FAIL: bcir_diag build"; exit 1; }
printf 'unsigned f(unsigned x){ return x + ; }\n'   > "${tmp}/da.c"
printf 'int main(void)\n{\n\treturn foo(1, 2);\n}\n' > "${tmp}/db.c"   # a tab-indented line
run_diag() {  # <src-file> <filename> <severity> <start> <end> <message>
  local src="$1" fn="$2" sev="$3" st="$4" en="$5" msg="$6"
  local c_out py_out
  c_out="$(printf '%s\t%s\t%s\t%s\n' "${sev}" "${st}" "${en}" "${msg}" | "${tmp}/test_diag" "${src}" "${fn}")"
  py_out="$(SRCF="${src}" FN="${fn}" SEV="${sev}" ST="${st}" EN="${en}" MSG="${msg}" python3 -c "
import os, sys
from bcir.frontends.cfront.diagnostics import SourceDiagnostic, Span, render
src=open(os.environ['SRCF']).read()
s,e=int(os.environ['ST']),int(os.environ['EN'])
span=None if (s==-1 and e==-1) else Span(s,e)
d=SourceDiagnostic(os.environ['SEV'], os.environ['MSG'], span=span)
sys.stdout.write(render(d, src, os.environ['FN']))")" || { echo "  FAIL: oracle render ${fn}"; exit 1; }
  [ "${c_out}" = "${py_out}" ] \
    && echo "  PASS diag ${fn} ${sev}@${st}:${en}" \
    || { echo "  FAIL: diag ${fn} (C != PY)"; printf '   C : %s\n   PY: %s\n' "${c_out}" "${py_out}"; exit 1; }
}
run_diag "${tmp}/da.c" u.c error    34 35 "expected ';'"          # a spanned single-caret error
run_diag "${tmp}/da.c" u.c error    -1 -1 "file-level problem"    # a spanless (no-caret) banner
run_diag "${tmp}/da.c" u.c warning   9 10 "odd parameter name"    # a warning severity
run_diag "${tmp}/db.c" m.c error    19 22 "implicit declaration of 'foo'"  # tab-indented line: caret aligns
run_diag "${tmp}/db.c" m.c error    34 34 "zero-width insertion point"     # zero-width span -> one caret
echo "  PASS diagnostic renderer is byte-identical to diagnostics.render()"
# the machine-readable (JSON) feed: bcir_diag_to_json == DiagnosticReport.to_json (json.dumps indent=2).
run_diag_json() {  # <src-file> <filename> <severity> <start> <end> <message>
  local src="$1" fn="$2" sev="$3" st="$4" en="$5" msg="$6"
  local c_out py_out
  c_out="$(printf '%s\t%s\t%s\t%s\n' "${sev}" "${st}" "${en}" "${msg}" | "${tmp}/test_diag" --json "${src}" "${fn}")"
  py_out="$(SRCF="${src}" FN="${fn}" SEV="${sev}" ST="${st}" EN="${en}" MSG="${msg}" python3 -c "
import os, sys
from bcir.frontends.cfront.diagnostics import SourceDiagnostic, Span, DiagnosticReport
src=open(os.environ['SRCF']).read()
s,e=int(os.environ['ST']),int(os.environ['EN'])
span=None if (s==-1 and e==-1) else Span(s,e)
d=SourceDiagnostic(os.environ['SEV'], os.environ['MSG'], span=span, phase='parse')
sys.stdout.write(DiagnosticReport([d], src, os.environ['FN']).to_json())")" || { echo "  FAIL: oracle json ${fn}"; exit 1; }
  [ "${c_out}" = "${py_out}" ] \
    && echo "  PASS diag-json ${fn} ${sev}@${st}:${en}" \
    || { echo "  FAIL: diag-json ${fn} (C != PY)"; printf '   C : %s\n   PY: %s\n' "${c_out}" "${py_out}"; exit 1; }
}
run_diag_json "${tmp}/da.c" u.c error    34 35 "expected ';'"               # spanned -> full location object
run_diag_json "${tmp}/da.c" u.c warning  -1 -1 "file-level problem"         # spanless -> just "file"
run_diag_json "${tmp}/db.c" m.c error    19 22 "implicit declaration of 'foo'"
echo "  PASS diagnostic JSON feed is byte-identical to DiagnosticReport.to_json()"
# fix-it hints: the verb (replace with / insert / remove) is derived from the fix-it's span +
# replacement, the replacement printed with Python repr() in text and JSON-escaped in the feed. A
# primary + one fix-it must match diagnostics.render() / to_json() byte-for-byte on both rails.
run_diag_fixit() {  # <verb-label> <mode:text|json> <fx-start> <fx-end> <replacement>
  local label="$1" mode="$2" fs="$3" fe="$4" repl="$5" c_out py_out jflag=""
  [ "${mode}" = json ] && jflag="--json"
  c_out="$(printf 'error\t34\t35\texpected token\n+\t%s\t%s\t%s\n' "${fs}" "${fe}" "${repl}" \
           | "${tmp}/test_diag" ${jflag} "${tmp}/da.c" u.c)"
  py_out="$(SRCF="${tmp}/da.c" MODE="${mode}" FS="${fs}" FE="${fe}" REPL="${repl}" python3 -c "
import os, sys
from bcir.frontends.cfront.diagnostics import SourceDiagnostic, Span, FixIt, DiagnosticReport, render
src=open(os.environ['SRCF']).read()
d=SourceDiagnostic('error','expected token', span=Span(34,35),
                   fixits=[FixIt(Span(int(os.environ['FS']),int(os.environ['FE'])), os.environ['REPL'])], phase='parse')
sys.stdout.write(DiagnosticReport([d], src, 'u.c').to_json() if os.environ['MODE']=='json' else render(d, src, 'u.c'))")" \
    || { echo "  FAIL: oracle fix-it ${label}/${mode}"; exit 1; }
  [ "${c_out}" = "${py_out}" ] \
    && echo "  PASS fix-it ${label}/${mode} (oracle == C)" \
    || { echo "  FAIL: fix-it ${label}/${mode}"; printf '   C : %s\n   PY: %s\n' "${c_out}" "${py_out}"; exit 1; }
}
for m in text json; do
  run_diag_fixit replace-with "${m}" 34 35 ";"     # span -> "replace with ';'"
  run_diag_fixit insert       "${m}" 34 34 ")"     # zero-width span -> "insert ')'"
  run_diag_fixit remove       "${m}" 34 36 ""      # empty replacement -> "remove ''"
done
echo "  PASS diagnostic fix-it hints are byte-identical across both rails"
# include / line-map origin: a diagnostic relocated to its origin file:line, with the #include chain
# printed as Clang "In file included from ...:" frames (text) / an "includedFrom" array (JSON). The
# primary error sits at offset 34 (column 35) of da.c; the origin relocates it to inc/b.h:42.
run_diag_origin() {  # <mode:text|json>
  local mode="$1" c_out py_out jflag=""
  [ "${mode}" = json ] && jflag="--json"
  c_out="$(printf 'error\t34\t35\tundeclared token\n@\t42\t0\tinc/b.h\n^\t10\t0\tmain.c\n^\t3\t0\tinc/a.h\n' \
           | "${tmp}/test_diag" ${jflag} "${tmp}/da.c" u.c)"
  py_out="$(SRCF="${tmp}/da.c" MODE="${mode}" python3 -c "
import os, sys, json
from bcir.frontends.cfront.diagnostics import SourceDiagnostic, Span, render, diagnostic_to_dict
src=open(os.environ['SRCF']).read()
d=SourceDiagnostic('error','undeclared token', span=Span(34,35), phase='parse')
origin=('inc/b.h', 42, [('main.c',10), ('inc/a.h',3)])
if os.environ['MODE']=='json':
    sys.stdout.write(json.dumps([diagnostic_to_dict(d, src, 'u.c', origin=origin)], indent=2))
else:
    sys.stdout.write(render(d, src, 'u.c', origin=origin))")" || { echo "  FAIL: oracle origin/${mode}"; exit 1; }
  [ "${c_out}" = "${py_out}" ] \
    && echo "  PASS origin/${mode} (#include chain oracle == C)" \
    || { echo "  FAIL: origin/${mode}"; printf '   C : %s\n   PY: %s\n' "${c_out}" "${py_out}"; exit 1; }
}
run_diag_origin text
run_diag_origin json
echo "  PASS diagnostic include-stack origin is byte-identical across both rails"
# parser error recovery: a panic-mode run reports EVERY error it resynchronizes past (not just the
# first). The oracle's diagnose() produces that multi-diagnostic DiagnosticReport; the C report
# renderer (bcir_diag_report_render / bcir_diag_to_json over the array) formats the identical report.
printf 'unsigned f(unsigned x) { return x + ; }\nunsigned g(unsigned y) { return y 7; }\n' > "${tmp}/rec_src.c"
RSRC="${tmp}/rec_src.c" RPP="${tmp}/rec_pp.c" RSPEC="${tmp}/rec.spec" RTXT="${tmp}/rec.txt" RJSON="${tmp}/rec.json" \
python3 -c "
import os
from bcir.frontends.cfront.pipeline import diagnose
rep=diagnose(open(os.environ['RSRC']).read(), filename='multi.c')
open(os.environ['RPP'],'w').write(rep.source)
lines=[]
for d in rep.diagnostics:
    s,e=(d.span.start,d.span.end) if d.span else (-1,-1)
    lines.append(f'{d.severity}\t{s}\t{e}\t{d.message}')
    for fx in d.fixits: lines.append(f'+\t{fx.span.start}\t{fx.span.end}\t{fx.replacement}')
    for nt in d.notes:
        ns,ne=(nt.span.start,nt.span.end) if nt.span else (-1,-1)
        lines.append(f'-\t{ns}\t{ne}\t{nt.message}')
open(os.environ['RSPEC'],'w').write('\n'.join(lines)+('\n' if lines else ''))
open(os.environ['RTXT'],'w').write(rep.render())
open(os.environ['RJSON'],'w').write(rep.to_json())
assert len(rep.diagnostics) >= 2, f'expected >=2 recovered diagnostics, got {len(rep.diagnostics)}'
" || { echo "  FAIL: oracle recovery report (expected >=2 diagnostics)"; exit 1; }
[ "$("${tmp}/test_diag" "${tmp}/rec_pp.c" multi.c < "${tmp}/rec.spec")" = "$(cat "${tmp}/rec.txt")" ] \
  && echo "  PASS recovery report text (multi-diagnostic oracle == C)" \
  || { echo "  FAIL: recovery report text"; exit 1; }
[ "$("${tmp}/test_diag" --json "${tmp}/rec_pp.c" multi.c < "${tmp}/rec.spec")" = "$(cat "${tmp}/rec.json")" ] \
  && echo "  PASS recovery report JSON (multi-diagnostic oracle == C)" \
  || { echo "  FAIL: recovery report JSON"; exit 1; }

# The total-compilation / fallback contract (#fallback, the C twin of pipeline.compile_with_fallback):
# a unit BCIR can compile + verify exits 0 (clean); a unit that compiles but fails the verifier (e.g.
# R18 recursion) exits 1 (DIRTY, NOT a fallback); a construct outside the supported subset exits 2
# ("fallback to LLVM backend: <phase>: <reason>"). The three-way outcome must agree with the oracle
# (needs_fallback=2 / not-clean=1 / clean=0) -- pinning the two rails' supported subset together.
echo "[c-runtime] fallback contract (bcir-cc --fallback): route-to-LLVM decision == oracle (#fallback)"
mkdir -p "${tmp}/fb"
printf 'unsigned f(unsigned x){ return x*2u + 1u; }\n'                  > "${tmp}/fb/ok.c"
printf 'unsigned f(unsigned n){ return f(n-1u); }\n'                    > "${tmp}/fb/recursion.c"
printf 'unsigned f(void){ _Complex double z; return 0u; }\n'           > "${tmp}/fb/complex.c"
printf 'unsigned f(unsigned n){ unsigned a[n][n]; return a[0][0]; }\n'  > "${tmp}/fb/vla.c"  # *multi-dim* VLA (1-D is native)
printf 'unsigned f(unsigned x){ return ({ unsigned y=x; y+1u; }); }\n'  > "${tmp}/fb/stmtexpr.c"
fb_seen=""
for fb in ok recursion complex vla stmtexpr; do
  "${tmp}/bcir-cc" --fallback "${tmp}/fb/${fb}.c" >/dev/null 2>&1; trc=$?
  orc="$(python3 -c "
from bcir.frontends.cfront.pipeline import compile_with_fallback
r=compile_with_fallback(open('${tmp}/fb/${fb}.c').read(), check_clang=False)
print(2 if r.needs_fallback else (0 if r.is_clean else 1))")" \
    || { echo "  FAIL: oracle fallback ${fb}"; exit 1; }
  [ "${trc}" = "${orc}" ] \
    && echo "  PASS fallback ${fb} (rc oracle == C: ${trc})" \
    || { echo "  FAIL: fallback ${fb} (C rc=${trc} PY rc=${orc})"; exit 1; }
  fb_seen="${fb_seen}${trc}"
done
# the gate must span all three outcomes (clean 0 / dirty 1 / fallback 2), else it has no teeth.
case "${fb_seen}" in
  *0*) case "${fb_seen}" in *1*) case "${fb_seen}" in *2*)
    echo "  PASS fallback contract spans clean / dirty / fallback" ;;
    *) echo "  FAIL: fallback gate never reached a fallback (2): ${fb_seen}"; exit 1 ;; esac ;;
    *) echo "  FAIL: fallback gate never reached a DIRTY (1): ${fb_seen}"; exit 1 ;; esac ;;
  *) echo "  FAIL: fallback gate never reached a clean (0): ${fb_seen}"; exit 1 ;;
esac

# Module-scope effect / commutation analysis (#effects, the C twin of pipeline.own_footprint +
# commute): per function the global names it reads/writes (callee effects folded in transitively),
# then the pairwise commute matrix (two readers commute; a writer conflicts with any reader/writer of
# the same global). bcir-cc --emit-effects must match the oracle's pipeline.effects / commute exactly,
# spanning a positive commute (read_a <> read_b over disjoint globals) and conflicts (a writer).
echo "[c-runtime] effect/commutation analysis (bcir-cc --emit-effects): footprints + commute == oracle (#effects)"
for fx in cfront_effects.c cfront_global_rw.c; do
  c_fx="$("${tmp}/bcir-cc" --emit-effects "${C}/${fx}")" || { echo "  FAIL: bcir-cc --emit-effects ${fx}"; exit 1; }
  py_fx="$(FX="${C}/${fx}" python3 -c "
import os
from bcir.frontends.cfront import compile_unit
r=compile_unit(open(os.environ['FX']).read(), check_clang=False)
fns=list(r.lowered.functions)
def names(rids): return sorted(r.lowered.resources[x].name for x in rids if x in r.lowered.resources)
out=[]
for n in fns:
    e=r.effects[n]
    out.append(f\"fn={n} reads={','.join(names(e.reads)) or '-'} writes={','.join(names(e.writes)) or '-'}\")
for i,a in enumerate(fns):
    for b in fns[i+1:]:
        out.append(f'commute {a} {b} = {1 if r.commute(a,b) else 0}')
print('\n'.join(out))")" || { echo "  FAIL: oracle effects ${fx}"; exit 1; }
  [ "${c_fx}" = "${py_fx}" ] \
    && echo "  PASS effects ${fx} (footprints + commute oracle == C)" \
    || { echo "  FAIL: effects ${fx} (C != PY)"; printf '   C :\n%s\n   PY:\n%s\n' "${c_fx}" "${py_fx}"; exit 1; }
done
# the gate must show both a commuting pair (1) and a conflict (0), else it has no teeth.
"${tmp}/bcir-cc" --emit-effects "${C}/cfront_effects.c" | grep -q "commute read_a read_b = 1" \
  && "${tmp}/bcir-cc" --emit-effects "${C}/cfront_effects.c" | grep -q "= 0" \
  && echo "  PASS effects analysis distinguishes commute (1) from conflict (0)" \
  || { echo "  FAIL: effects gate did not span commute + conflict"; exit 1; }

# Scalable IR (no fixed BCIR_MAX_*): a unit that busts every OLD ceiling -- 43 functions (> the old
# BCIR_MAX_FUNCS 16), many12 with 12 params (> 8), agg with 40 calls (> 32), big with 7500 claims
# (> the old 4096 per-function cap). The IR grows geometrically, so the twin compiles it clean and
# matches the oracle's structural counts; the old fixed arrays would have rejected it.
echo "[c-runtime] scalable IR (bcir-cc, no fixed BCIR_MAX_*): cap-busting unit compiles + matches oracle (#scale)"
python3 - "${tmp}/cfront_scale.c" <<'PY'
import sys
fns=[f"unsigned g{k}(void){{ return {k}u; }}" for k in range(40)]
ps=[chr(ord('a')+i) for i in range(12)]
fns.append("unsigned many12("+",".join(f"unsigned {p}" for p in ps)+"){ return "+"+".join(ps)+"; }")
fns.append("unsigned agg(void){ return "+"+".join(f"g{k}()" for k in range(40))+"; }")
body="\n".join("  acc = acc + 1u;" for _ in range(2500))
fns.append("unsigned big(unsigned acc){\n"+body+"\n  return acc;\n}")
open(sys.argv[1],"w").write("\n".join(fns)+"\n")
PY
c_scale="$("${tmp}/bcir-cc" --emit-claimgraph "${tmp}/cfront_scale.c" 2>&1 | grep -oE 'funcs=[0-9]+ claims=[0-9]+.*ok=[0-9]')"
case "${c_scale}" in
  "funcs=43 claims=7500"*"ok=1") echo "  PASS scale: 43 funcs / 7500-claim fn compile clean (old caps 16 / 4096 exceeded)";;
  *) echo "  FAIL: scale unit (twin: ${c_scale})"; exit 1;;
esac
py_scale="$(python3 -c "
from bcir.frontends.cfront import compile_unit
r=compile_unit(open('${tmp}/cfront_scale.c').read(), check_clang=False)
print('funcs=%d big=%d' % (len(r.lowered.functions), len(r.lowered.functions['big'].claims)))")"
[ "${py_scale}" = "funcs=43 big=7500" ] \
  && echo "  PASS scale: oracle agrees (${py_scale})" \
  || { echo "  FAIL: scale oracle mismatch (${py_scale})"; exit 1; }

# Integer promotions + usual arithmetic conversions (#intpromote): the C twin types each temp by its
# true (width, signedness), so a signed int divide / remainder / right-shift / comparison emits signed
# C (not the old flat uint32_t) and a mixed-width op widens per §6.3.1.8. The twin's --emit-c is
# behaviour-equivalent to Clang over the FULL signed range (negatives) -- exactly what the old model
# got wrong. Compile the emitted bcir_* beside the source + a full-range differential driver.
echo "[c-runtime] integer promotions + UAC (bcir-cc): signed/mixed-width emit == Clang full-range (#intpromote)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_intpromote.c" > "${tmp}/ip_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; cat "${C}/cfront_intpromote.c" "${tmp}/ip_emit.c"
  cat <<'DRV'
static uint64_t S=0x9E3779B97F4A7C15u;
static uint64_t nx(void){S=S*6364136223846793005u+1442695040888963407u;return S>>32;}
int main(void){
  for(int i=0;i<300000;i++){
    int a=(int)nx(), b=(int)nx(); long lb=(long)nx()<<3 | (long)nx();
    if(sdiv(a,b)!=bcir_sdiv(a,b)||smod(a,b)!=bcir_smod(a,b)||sshr(a)!=bcir_sshr(a)
     ||scmp(a,b)!=bcir_scmp(a,b)||udiv((unsigned)a,(unsigned)b)!=bcir_udiv((unsigned)a,(unsigned)b)
     ||wide(a,lb)!=bcir_wide(a,lb)||umix((unsigned)a,lb)!=bcir_umix((unsigned)a,lb)){
       printf("MISMATCH@%d\n",i);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/ip_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/ip_harness.c" -o "${tmp}/ip_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/ip_harness.c" -o "${tmp}/ip_h" \
  || { echo "  FAIL: intpromote harness build"; exit 1; }
ipr="$("${tmp}/ip_h")"
[ "${ipr}" = "MATCH" ] \
  && echo "  PASS intpromote: signed div/mod/shr/compare + mixed-width == Clang (full signed range)" \
  || { echo "  FAIL: intpromote behaviour (${ipr})"; exit 1; }
# teeth: the emit carries the real signed + wide integer types (the old model emitted only uint32_t).
{ grep -q "int32_t" "${tmp}/ip_emit.c" && grep -q "int64_t" "${tmp}/ip_emit.c"; } \
  && echo "  PASS intpromote: emit carries true signed (int32_t) + wide (int64_t) types" \
  || { echo "  FAIL: intpromote emit missing signed/wide types"; exit 1; }

# Designated initializers for a file-scope table (#designated): `static const T NAME[N] = {[i]=v,...}`
# (the driver opcode-dispatch / jump-table pattern, with a gap that zero-fills). Both rails now parse
# the designated initializer; the table is referenced by name (defined in the source), so the twin's
# --emit-c is Clang-behaviour-equivalent. Compile the emitted bcir_* beside the source + a driver that
# sweeps every opcode (incl. the zero-filled gap).
echo "[c-runtime] designated initializers (bcir-cc): dispatch-table emit == Clang (#designated)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_dispatch_table.c" > "${tmp}/dt_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; cat "${C}/cfront_dispatch_table.c" "${tmp}/dt_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned op=0; op<10u; op++)
    if(weigh(op)!=bcir_weigh(op)){printf("MISMATCH op=%u: %u vs %u\n",op,weigh(op),bcir_weigh(op));return 1;}
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/dt_harness.c"
"${CC}" -std=c23 -O2 -I "${C}" "${tmp}/dt_harness.c" "${C}/bcir_quarantine.c" -o "${tmp}/dt_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 -I "${C}" "${tmp}/dt_harness.c" "${C}/bcir_quarantine.c" -o "${tmp}/dt_h" \
  || { echo "  FAIL: designated harness build"; exit 1; }
dtr="$("${tmp}/dt_h")"
[ "${dtr}" = "MATCH" ] \
  && echo "  PASS designated: enum-indexed dispatch table (+ zero-fill gap) == Clang" \
  || { echo "  FAIL: designated behaviour (${dtr})"; exit 1; }

# Local aggregate initializers for a struct/union (#aggregate): `struct cfg c = {.field=v, ...}` lowers
# to a `= {0}` zero baseline + a c.store per initialized member (uninitialized members zero-fill). The
# twin's --emit-c is Clang-behaviour-equivalent. Compile the emitted bcir_* beside the source + a driver.
echo "[c-runtime] local aggregate init (bcir-cc): struct/union {.field=v} emit == Clang (#aggregate)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_agginit.c" > "${tmp}/ag_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'; cat "${C}/cfront_agginit.c" "${tmp}/ag_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned x=0; x<5000u; x++)
    if(config(x)!=bcir_config(x)||positional(x)!=bcir_positional(x)||overlap(x)!=bcir_overlap(x)){
      printf("MISMATCH x=%u\n",x);return 1;}
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/ag_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/ag_harness.c" -o "${tmp}/ag_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/ag_harness.c" -o "${tmp}/ag_h" \
  || { echo "  FAIL: aggregate harness build"; exit 1; }
agr="$("${tmp}/ag_h")"
[ "${agr}" = "MATCH" ] \
  && echo "  PASS aggregate: struct/union designated + positional init (+ zero-fill) == Clang" \
  || { echo "  FAIL: aggregate behaviour (${agr})"; exit 1; }
grep -q "= {0}" "${tmp}/ag_emit.c" \
  && echo "  PASS aggregate: emit carries the = {0} zero baseline" \
  || { echo "  FAIL: aggregate emit missing zero baseline"; exit 1; }

# Scalable parser state (#pscale): segment-1 made the IR arrays grow; this removes the twin's fixed
# *parser-state* caps too -- struct defs (was s[16]), file-scope globals (was gv[16]), typedefs (was
# td[64]), enum constants (was ec[256]) and locals (was env[256]) all grow geometrically (reused across
# compiles via a save/restore around the static CC). A unit that busts every old cap compiles clean on
# the twin and matches the oracle's structure -- real headers (many globals / structs / typedefs) lower.
echo "[c-runtime] scalable parser state (bcir-cc, no fixed gv/s/td/ec/env caps): cap-busting unit == oracle (#pscale)"
python3 - "${tmp}/pstress.c" <<'PY'
import sys
L=[f"struct S{k} {{ unsigned m0; unsigned m1; }};" for k in range(20)]      # 20 struct defs (> old s[16])
L+=[f"typedef unsigned U{k};" for k in range(20)]                            # 20 typedefs
L+=[f"static const unsigned G{k}[2] = {{ {k}u, {k+1}u }};" for k in range(25)]  # 25 globals (> old gv[16])
L.append("unsigned big(void){\n"+"\n".join(f"  unsigned v{i} = {i}u;" for i in range(300))+   # 300 locals (>256)
         "\n  return "+"+".join(f"v{i}" for i in range(300))+"; }")
L.append("unsigned useg(unsigned i){ return G0[i%2u] + G19[i%2u] + G24[i%2u]; }")
open(sys.argv[1],"w").write("\n".join(L)+"\n")
PY
c_ps="$("${tmp}/bcir-cc" --emit-claimgraph "${tmp}/pstress.c" 2>&1 | grep -oE 'funcs=[0-9]+ claims=[0-9]+.*ok=[0-9]' | tail -1)"
py_ps="$(python3 -c "
from bcir.frontends.cfront import compile_unit
from bcir.model import Domain
r=compile_unit(open('${tmp}/pstress.c').read(), check_clang=False)
fns=r.lowered.functions; lf=fns[next(reversed(fns))]
kn=sum(1 for c in lf.claims if c.op=='c.const'); bo=sum(1 for c in lf.claims if c.op.startswith('c.bin.'))
print(f'funcs={len(fns)} claims={len(lf.claims)} mmio=0 bf=0 const={kn} binop={bo} call=0 ok={1 if r.is_clean else 0}')")"
[ -n "${c_ps}" ] && [ "${c_ps}" = "${py_ps}" ] \
  && echo "  PASS pscale: 20 structs / 25 globals / 300 locals compile clean == oracle (${c_ps})" \
  || { echo "  FAIL: pscale (C='${c_ps}' PY='${py_ps}')"; exit 1; }

# Array element stores (#astore): a[i] = v / a[i] OP= v -- the driver buffer-fill / scatter idiom,
# lowered to a 3-read c.store and emitted as a[i] = v. A proper differential drives the source fn over
# buffer A and the twin's bcir_* over buffer B, then compares the buffers (these are void writers).
echo "[c-runtime] array element stores (bcir-cc): a[i]=v emit == Clang (#astore)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_arraystore.c" > "${tmp}/st_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; cat "${C}/cfront_arraystore.c" "${tmp}/st_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned t=0;t<3000u;t++){
    unsigned A[32]={0}, B[32]={0}; unsigned i=t%30u, j=(t*3u)%30u, v=t*7u+1u, base=t+5u;
    fill(A,i,base);   bcir_fill(B,i,base);
    scatter(A,i,j,v); bcir_scatter(B,i,j,v);
    accum(A,i,v);     bcir_accum(B,i,v);
    for(int k=0;k<32;k++) if(A[k]!=B[k]){printf("MISMATCH t=%u k=%d\n",t,k);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/st_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/st_harness.c" -o "${tmp}/st_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/st_harness.c" -o "${tmp}/st_h" \
  || { echo "  FAIL: astore harness build"; exit 1; }
str="$("${tmp}/st_h")"
[ "${str}" = "MATCH" ] \
  && echo "  PASS astore: a[i]=v / a[i] OP= v == Clang (source buffer == twin buffer)" \
  || { echo "  FAIL: astore behaviour (${str})"; exit 1; }

# `extern` linkage (#extern): a symbol declared here but defined in another TU. `extern T g;` is
# referenced by name (no storage emitted), so the twin's --emit-c is Clang-behaviour-equivalent once
# linked against the definition (supplied by the driver below).
echo "[c-runtime] extern linkage (bcir-cc): extern global emit == Clang (#extern)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_extern.c" > "${tmp}/ex_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo 'unsigned cfg_base = 0u;'   # the DEFINITION (other TU)
  cat "${C}/cfront_extern.c" "${tmp}/ex_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned t=0;t<3000u;t++){
    unsigned x=t*7u+1u;
    cfg_base = t & 255u;
    if(scaled(x)!=bcir_scaled(x) || offs(x)!=bcir_offs(x)){printf("read MISMATCH t=%u\n",t);return 1;}
    cfg_base=t; bump(3u);      unsigned src=cfg_base;     /* source writes the extern */
    cfg_base=t; bcir_bump(3u); unsigned twn=cfg_base;     /* twin writes the extern */
    if(src!=twn){printf("write MISMATCH t=%u\n",t);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/ex_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/ex_harness.c" -o "${tmp}/ex_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/ex_harness.c" -o "${tmp}/ex_h" \
  || { echo "  FAIL: extern harness build"; exit 1; }
exr="$("${tmp}/ex_h")"
[ "${exr}" = "MATCH" ] \
  && echo "  PASS extern: read + write a cross-TU global == Clang (linked to the definition)" \
  || { echo "  FAIL: extern behaviour (${exr})"; exit 1; }

# `_Thread_local` / `thread_local` (#threadlocal): a per-thread storage class. Recognized + consumed;
# the global is referenced by name (the source defines it). In the deterministic single-thread harness a
# thread-local behaves as a global, so the twin's --emit-c is Clang-behaviour-equivalent.
echo "[c-runtime] thread-local storage (bcir-cc): _Thread_local emit == Clang (#threadlocal)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_threadlocal.c" > "${tmp}/tl_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; cat "${C}/cfront_threadlocal.c" "${tmp}/tl_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned t=0;t<3000u;t++){
    unsigned x=t*7u+1u;
    tls_ctr = t & 255u;
    if(scaled(x)!=bcir_scaled(x) || offs(x)!=bcir_offs(x)){printf("read MISMATCH t=%u\n",t);return 1;}
    tls_ctr=t; bump(3u);      unsigned src=tls_ctr;
    tls_ctr=t; bcir_bump(3u); unsigned twn=tls_ctr;
    if(src!=twn){printf("write MISMATCH t=%u\n",t);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/tl_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/tl_harness.c" -o "${tmp}/tl_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/tl_harness.c" -o "${tmp}/tl_h" \
  || { echo "  FAIL: threadlocal harness build"; exit 1; }
tlr="$("${tmp}/tl_h")"
[ "${tlr}" = "MATCH" ] \
  && echo "  PASS threadlocal: read + write a _Thread_local global == Clang" \
  || { echo "  FAIL: threadlocal behaviour (${tlr})"; exit 1; }

# Multi-declarator local declarations (#multidecl): `T a = x, b, c = z;` -- the canonical two-variable
# loop init + grouped temporaries. Each declarator lowers to its own storage + copy (identical to
# separate decls), so the twin's --emit-c is Clang-behaviour-equivalent over both fixture functions.
echo "[c-runtime] multi-declarator locals (bcir-cc): emit == Clang (#multidecl)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_multidecl.c" > "${tmp}/md_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\bblend\b/blend_src/' -e 's/\bwindowed\b/windowed_src/' "${C}/cfront_multidecl.c"
  cat "${tmp}/md_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned n=0;n<6000u;n++){
    if(blend_src(n)!=bcir_blend(n)){printf("blend MISMATCH n=%u\n",n);return 1;}
    if(windowed_src(n)!=bcir_windowed(n)){printf("windowed MISMATCH n=%u\n",n);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/md_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/md_harness.c" -o "${tmp}/md_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/md_harness.c" -o "${tmp}/md_h" \
  || { echo "  FAIL: multidecl harness build"; exit 1; }
mdr="$("${tmp}/md_h")"
[ "${mdr}" = "MATCH" ] \
  && echo "  PASS multidecl: T a=x, b, c=z + two-variable for-init == Clang" \
  || { echo "  FAIL: multidecl behaviour (${mdr})"; exit 1; }

# Comma operator in the for-step (#commastep): `for(...; ...; i++, j--)` -- two-pointer / reversal loops
# and parallel-counter updates. Each comma-separated step element runs in order every iteration, so the
# twin's --emit-c (which loops the recorded step tokens on commas) is Clang-behaviour-equivalent.
echo "[c-runtime] comma-operator for-step (bcir-cc): emit == Clang (#commastep)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_commastep.c" > "${tmp}/cs_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\bspan_xor\b/span_xor_src/' -e 's/\btwin_acc\b/twin_acc_src/' "${C}/cfront_commastep.c"
  cat "${tmp}/cs_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned n=0;n<6000u;n++){
    if(span_xor_src(n)!=bcir_span_xor(n)){printf("span_xor MISMATCH n=%u\n",n);return 1;}
    if(twin_acc_src(n)!=bcir_twin_acc(n)){printf("twin_acc MISMATCH n=%u\n",n);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/cs_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/cs_harness.c" -o "${tmp}/cs_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/cs_harness.c" -o "${tmp}/cs_h" \
  || { echo "  FAIL: commastep harness build"; exit 1; }
csr="$("${tmp}/cs_h")"
[ "${csr}" = "MATCH" ] \
  && echo "  PASS commastep: i++,j-- + parallel compound-assign steps == Clang" \
  || { echo "  FAIL: commastep behaviour (${csr})"; exit 1; }

# Multi-declarator struct/union members (#structmulti): `unsigned x, y, z;` -- several members off one
# specifier (incl. multi-declarator bitfields). Each lays out as if on its own line, so offsets / size
# match Clang; the twin's --emit-c accesses members by byte offset into the source-defined struct, so a
# wrong offset would diverge. Differential checks sizeof + per-member round-trip.
echo "[c-runtime] multi-declarator struct members (bcir-cc): layout + access == Clang (#structmulti)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_structmulti.c" > "${tmp}/sm_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bpt_sum\b/pt_sum_src/' -e 's/\bfl_pack\b/fl_pack_src/' "${C}/cfront_structmulti.c"
  cat "${tmp}/sm_emit.c"
  cat <<'DRV'
int main(void){
  if(sizeof(struct Pt)!=12u||sizeof(struct Flags)!=4u){printf("LAYOUT Pt=%zu Flags=%zu\n",sizeof(struct Pt),sizeof(struct Flags));return 2;}
  for(unsigned a=0;a<6000u;a++){
    if(pt_sum_src(a)!=bcir_pt_sum(a)){printf("pt MISMATCH a=%u\n",a);return 1;}
    if(fl_pack_src(a)!=bcir_fl_pack(a)){printf("fl MISMATCH a=%u\n",a);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/sm_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/sm_harness.c" -o "${tmp}/sm_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/sm_harness.c" -o "${tmp}/sm_h" \
  || { echo "  FAIL: structmulti harness build"; exit 1; }
smr="$("${tmp}/sm_h")"
[ "${smr}" = "MATCH" ] \
  && echo "  PASS structmulti: unsigned x,y,z + multi-bitfield layout/access == Clang" \
  || { echo "  FAIL: structmulti behaviour (${smr})"; exit 1; }

# Empty statements (#emptystmt): a bare `;` -- the body of `for(...);` (work in the header) / `if(c);`
# and stray `;;`. Both rails consume it and emit no claim, so the twin's --emit-c is Clang-equivalent.
echo "[c-runtime] empty statements (bcir-cc): emit == Clang (#emptystmt)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_emptystmt.c" > "${tmp}/es_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\bcount_below\b/count_below_src/' -e 's/\bmasked\b/masked_src/' "${C}/cfront_emptystmt.c"
  cat "${tmp}/es_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned a=0;a<8000u;a++){
    if(count_below_src(a)!=bcir_count_below(a)){printf("count MISMATCH a=%u\n",a);return 1;}
    if(masked_src(a)!=bcir_masked(a)){printf("masked MISMATCH a=%u\n",a);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/es_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/es_harness.c" -o "${tmp}/es_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/es_harness.c" -o "${tmp}/es_h" \
  || { echo "  FAIL: emptystmt harness build"; exit 1; }
esr="$("${tmp}/es_h")"
[ "${esr}" = "MATCH" ] \
  && echo "  PASS emptystmt: for(...); / if(c); / stray ;; == Clang" \
  || { echo "  FAIL: emptystmt behaviour (${esr})"; exit 1; }

# Store through a pointer (#ptrstore): `*p = v` / `*p OP= v` / `*(p + i) = v` -- the write counterpart
# of the deref load. The twin parsed `*p` only as a read; now it lowers a deref store (offset-0 imm for
# `*p`, the indexed `p[i]` shape for `*(p + i)`). Verified on INDEPENDENT buffers (these mutate through p).
echo "[c-runtime] store through a pointer (bcir-cc): emit == Clang (#ptrstore)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_ptrstore.c" > "${tmp}/ps_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bstore_scaled\b/store_scaled_src/' -e 's/\baccum_at\b/accum_at_src/' \
      -e 's/\bset_then_bump\b/set_then_bump_src/' "${C}/cfront_ptrstore.c"
  cat "${tmp}/ps_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned v=0;v<4000u;v++){
    unsigned a=v,b=v; store_scaled_src(&a,v); bcir_store_scaled(&b,v);
    if(a!=b){printf("store MISMATCH v=%u\n",v);return 1;}
    unsigned b1[8],b2[8]; for(int k=0;k<8;k++){b1[k]=b2[k]=v+(unsigned)k;}
    unsigned i=v%8u; accum_at_src(b1,i,v); bcir_accum_at(b2,i,v);
    for(int k=0;k<8;k++) if(b1[k]!=b2[k]){printf("accum MISMATCH v=%u\n",v);return 1;}
    unsigned c=0,d=0; set_then_bump_src(&c,v); bcir_set_then_bump(&d,v);
    if(c!=d){printf("setbump MISMATCH v=%u\n",v);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/ps_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/ps_harness.c" -o "${tmp}/ps_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/ps_harness.c" -o "${tmp}/ps_h" \
  || { echo "  FAIL: ptrstore harness build"; exit 1; }
psr="$("${tmp}/ps_h")"
[ "${psr}" = "MATCH" ] \
  && echo "  PASS ptrstore: *p= / *p OP= / *(p+i)= mutate-through-pointer == Clang" \
  || { echo "  FAIL: ptrstore behaviour (${psr})"; exit 1; }

# Nested struct member access (#nestmember): `o.pos.lo` / `dev->ctrl.flags` -- a struct-in-struct. Both
# rails flatten the chain to one offset access; the twin emits member access by byte offset, so a wrong
# accumulated offset (or layout) would diverge. Differential checks sizeof + nested read/write/bitfield.
echo "[c-runtime] nested struct member access (bcir-cc): layout + access == Clang (#nestmember)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_nestmember.c" > "${tmp}/nm_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bouter_sum\b/outer_sum_src/' -e 's/\bdev_pack\b/dev_pack_src/' "${C}/cfront_nestmember.c"
  cat "${tmp}/nm_emit.c"
  cat <<'DRV'
int main(void){
  if(sizeof(struct Outer)!=12u||sizeof(struct Dev)!=8u){printf("LAYOUT O=%zu D=%zu\n",sizeof(struct Outer),sizeof(struct Dev));return 2;}
  for(unsigned a=0;a<8000u;a++){
    if(outer_sum_src(a)!=bcir_outer_sum(a)){printf("outer MISMATCH a=%u\n",a);return 1;}
    if(dev_pack_src(a)!=bcir_dev_pack(a)){printf("dev MISMATCH a=%u\n",a);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/nm_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/nm_harness.c" -o "${tmp}/nm_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/nm_harness.c" -o "${tmp}/nm_h" \
  || { echo "  FAIL: nestmember harness build"; exit 1; }
nmr="$("${tmp}/nm_h")"
[ "${nmr}" = "MATCH" ] \
  && echo "  PASS nestmember: o.pos.lo / dev->ctrl.bf layout + access == Clang" \
  || { echo "  FAIL: nestmember behaviour (${nmr})"; exit 1; }

# Reused local names across scopes (#loopreuse): two/three separate `for` loops each declaring the same
# counter name (`i`, `k`) are distinct flattened locals; the emit must give each a unique C identifier
# (`i`, `i_2`, ...) -- declaring both at function scope was a C redefinition (the unit was is_clean yet
# its emit did not compile, on BOTH rails). Differential confirms the disjoint-scope reuse == Clang.
echo "[c-runtime] reused loop-counter names (bcir-cc): unique emit == Clang (#loopreuse)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_loopreuse.c" > "${tmp}/lr_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\btwo_pass\b/two_pass_src/' -e 's/\btriple_pass\b/triple_pass_src/' "${C}/cfront_loopreuse.c"
  cat "${tmp}/lr_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned n=0;n<20000u;n++){
    if(two_pass_src(n)!=bcir_two_pass(n)){printf("two MISMATCH n=%u\n",n);return 1;}
    if(triple_pass_src(n)!=bcir_triple_pass(n)){printf("triple MISMATCH n=%u\n",n);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/lr_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/lr_harness.c" -o "${tmp}/lr_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/lr_harness.c" -o "${tmp}/lr_h" \
  || { echo "  FAIL: loopreuse harness build (emit did not compile)"; exit 1; }
lrr="$("${tmp}/lr_h")"
[ "${lrr}" = "MATCH" ] \
  && echo "  PASS loopreuse: two/three loops reusing a counter name == Clang" \
  || { echo "  FAIL: loopreuse behaviour (${lrr})"; exit 1; }

# For-loop variable scope (#loopscope): a `for(unsigned i = ...)` scopes `i` to the loop, so a post-loop
# read of `i` resolves to a same-named param / outer -- the for-init must not leak into the enclosing
# block. Both rails save/restore the name env around the loop; the differential pins the post-loop value.
echo "[c-runtime] for-loop variable scope (bcir-cc): emit == Clang (#loopscope)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_loopscope.c" > "${tmp}/lsc_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\bparam_shadow\b/param_shadow_src/' -e 's/\bouter_shadow\b/outer_shadow_src/' "${C}/cfront_loopscope.c"
  cat "${tmp}/lsc_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned x=0;x<20000u;x++){
    if(param_shadow_src(x)!=bcir_param_shadow(x)){printf("param MISMATCH x=%u\n",x);return 1;}
    if(outer_shadow_src(x)!=bcir_outer_shadow(x)){printf("outer MISMATCH x=%u\n",x);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/lsc_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/lsc_harness.c" -o "${tmp}/lsc_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/lsc_harness.c" -o "${tmp}/lsc_h" \
  || { echo "  FAIL: loopscope harness build"; exit 1; }
lscr="$("${tmp}/lsc_h")"
[ "${lscr}" = "MATCH" ] \
  && echo "  PASS loopscope: post-loop read resolves to the shadowed param/outer == Clang" \
  || { echo "  FAIL: loopscope behaviour (${lscr})"; exit 1; }

# Bare-block variable scope (#blockscope): a `{ unsigned x = ...; }` block scopes `x`, so a post-block
# read resolves to a same-named outer -- the block must not leak. Both rails save/restore the name env
# around every `{ ... }` (the general case of #loopscope; the oracle also lowers a bare block inline now,
# via a Block node, so its claim count matches the twin -- no spurious if(1) wrapper).
echo "[c-runtime] bare-block variable scope (bcir-cc): emit == Clang (#blockscope)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_blockscope.c" > "${tmp}/bs_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\bnested_shadow\b/ns_src/' -e 's/\bblock_then_use\b/bt_src/' "${C}/cfront_blockscope.c"
  cat "${tmp}/bs_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned a=0;a<5000u;a++){
    if(ns_src(a)!=bcir_nested_shadow(a)){printf("ns MISMATCH a=%u\n",a);return 1;}
    for(unsigned b=0;b<20u;b++) if(bt_src(a%500u,b)!=bcir_block_then_use(a%500u,b)){printf("bt MISMATCH\n");return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/bs_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/bs_harness.c" -o "${tmp}/bs_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/bs_harness.c" -o "${tmp}/bs_h" \
  || { echo "  FAIL: blockscope harness build"; exit 1; }
bsr="$("${tmp}/bs_h")"
[ "${bsr}" = "MATCH" ] \
  && echo "  PASS blockscope: post-block read resolves to the shadowed outer == Clang" \
  || { echo "  FAIL: blockscope behaviour (${bsr})"; exit 1; }

# Native struct member arrays (#memberarray): `s.arr[i]` -- a 1-D array member. The element lands at
# `&base + member_off + i*elem_size`; the twin emits this as a memcpy at that offset, so a wrong stride
# or layout would diverge. Differential checks sizeof + read/write/compound + a narrowing uint8 element.
echo "[c-runtime] struct member arrays (bcir-cc): layout + access == Clang (#memberarray)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_memberarray.c" > "${tmp}/mar_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bpkt_sum\b/pkt_src/' -e 's/\bbuf_pack\b/buf_src/' -e 's/\bgrid_pick\b/grid_src/' "${C}/cfront_memberarray.c"
  cat "${tmp}/mar_emit.c"
  cat <<'DRV'
int main(void){
  if(sizeof(struct Packet)!=28u||sizeof(struct Buf)!=12u||sizeof(struct Grid)!=52u){printf("LAYOUT bad\n");return 2;}
  for(unsigned i=0;i<6000u;i++)for(unsigned a=0;a<40u;a++){
    if(pkt_src(i,a)!=bcir_pkt_sum(i,a)){printf("pkt MISMATCH i=%u a=%u\n",i,a);return 1;}
    if(buf_src(i,a)!=bcir_buf_pack(i,a)){printf("buf MISMATCH i=%u a=%u\n",i,a);return 1;}
    if(grid_src(i,a)!=bcir_grid_pick(i,a)){printf("grid MISMATCH i=%u a=%u\n",i,a);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/mar_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/mar_harness.c" -o "${tmp}/mar_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/mar_harness.c" -o "${tmp}/mar_h" \
  || { echo "  FAIL: memberarray harness build"; exit 1; }
marr="$("${tmp}/mar_h")"
[ "${marr}" = "MATCH" ] \
  && echo "  PASS memberarray: s.arr[i] read/write/compound + uint8 element == Clang" \
  || { echo "  FAIL: memberarray behaviour (${marr})"; exit 1; }

# Multi-dimensional local arrays (#localmd): `T m[A][B]` up to 3 dims -- a flat resource of A*B elements
# with the per-dim flatten shape, so `m[i][j]` -> `m[i*B + j]` (declared `m[A*B]`). The twin emits the
# flattened `m[lin]`; a wrong stride or size would diverge. Differential checks 2-D + 3-D fill/read.
echo "[c-runtime] multi-dimensional local arrays (bcir-cc): emit == Clang (#localmd)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_localmd.c" > "${tmp}/lmd_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\bgrid_diag\b/grid_src/' -e 's/\bcube_sum\b/cube_src/' "${C}/cfront_localmd.c"
  cat "${tmp}/lmd_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned i=0;i<8000u;i++)for(unsigned a=0;a<30u;a++){
    if(grid_src(i,a)!=bcir_grid_diag(i,a)){printf("grid MISMATCH i=%u a=%u\n",i,a);return 1;}
    if(cube_src(i,a)!=bcir_cube_sum(i,a)){printf("cube MISMATCH i=%u a=%u\n",i,a);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/lmd_harness.c"
"${CC}" -std=c23 -O2 -I "${C}" "${tmp}/lmd_harness.c" "${C}/bcir_quarantine.c" -o "${tmp}/lmd_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 -I "${C}" "${tmp}/lmd_harness.c" "${C}/bcir_quarantine.c" -o "${tmp}/lmd_h" \
  || { echo "  FAIL: localmd harness build"; exit 1; }
lmdr="$("${tmp}/lmd_h")"
[ "${lmdr}" = "MATCH" ] \
  && echo "  PASS localmd: 2-D + 3-D local array fill/read == Clang" \
  || { echo "  FAIL: localmd behaviour (${lmdr})"; exit 1; }

# The `signed` type specifier (#signedty): `signed char` / `signed int` / `signed long` decl + cast. The
# twin recognized `unsigned` but not `signed` in its decl/cast type-start detection, so it rejected these
# (-> fallback) while the oracle accepted -- a rail disagreement, now fixed. Differential == Clang.
echo "[c-runtime] the 'signed' type specifier (bcir-cc): emit == Clang (#signedty)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_signed.c" > "${tmp}/sg_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\bsigned_roundtrip\b/sr_src/' -e 's/\bsigned_scale\b/ss_src/' "${C}/cfront_signed.c"
  cat "${tmp}/sg_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned a=0;a<8000u;a++)for(unsigned b=0;b<24u;b++){
    if(sr_src(a)!=bcir_signed_roundtrip(a)){printf("sr MISMATCH a=%u\n",a);return 1;}
    if(ss_src(a,b)!=bcir_signed_scale(a,b)){printf("ss MISMATCH a=%u b=%u\n",a,b);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/sg_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/sg_harness.c" -o "${tmp}/sg_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/sg_harness.c" -o "${tmp}/sg_h" \
  || { echo "  FAIL: signedty harness build"; exit 1; }
sgr="$("${tmp}/sg_h")"
[ "${sgr}" = "MATCH" ] \
  && echo "  PASS signedty: signed char/int/long decl + cast == Clang" \
  || { echo "  FAIL: signedty behaviour (${sgr})"; exit 1; }

# Signed comparison against an integer literal (#signedcmp): `x < 0` / `x >= 0` for a signed `x` (the abs
# / clamp / signum idiom). The twin hardcoded integer constants as uint32_t, so the comparison promoted
# to unsigned (a silent miscompile); now each constant carries its own type. Differential == Clang.
echo "[c-runtime] signed comparison vs literal (bcir-cc): emit == Clang (#signedcmp)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_signedcmp.c" > "${tmp}/sc_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\biabs\b/iabs_src/' -e 's/\bclamp_lo\b/clamp_src/' -e 's/\bsignum\b/signum_src/' "${C}/cfront_signedcmp.c"
  cat "${tmp}/sc_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned a=0;a<8000u;a++)for(unsigned b=0;b<40u;b++){
    if(iabs_src(a,b)!=bcir_iabs(a,b)){printf("iabs MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(clamp_src(a,b)!=bcir_clamp_lo(a,b)){printf("clamp MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(signum_src(a,b)!=bcir_signum(a,b)){printf("signum MISMATCH a=%u b=%u\n",a,b);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/sc_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/sc_harness.c" -o "${tmp}/sc_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/sc_harness.c" -o "${tmp}/sc_h" \
  || { echo "  FAIL: signedcmp harness build"; exit 1; }
scr="$("${tmp}/sc_h")"
[ "${scr}" = "MATCH" ] \
  && echo "  PASS signedcmp: signed x < 0 / x >= 0 == Clang" \
  || { echo "  FAIL: signedcmp behaviour (${scr})"; exit 1; }

# Unary `-` / `~` on a wide (long / long long) operand (#longunary): the result was forced to a 4-byte
# uint32, so negating a `long` truncated the sign (a -1 long became 4294967295), breaking `x < 0` / abs.
# Now both rails keep the promoted operand type. Differential == Clang.
echo "[c-runtime] unary on a wide operand (bcir-cc): emit == Clang (#longunary)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_longunary.c" > "${tmp}/lu_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\blong_abs\b/la_src/' -e 's/\blong_complement\b/lc_src/' "${C}/cfront_longunary.c"
  cat "${tmp}/lu_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned a=0;a<8000u;a++)for(unsigned b=0;b<40u;b++){
    if(la_src(a,b)!=bcir_long_abs(a,b)){printf("la MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(lc_src(a,b)!=bcir_long_complement(a,b)){printf("lc MISMATCH a=%u b=%u\n",a,b);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/lu_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/lu_harness.c" -o "${tmp}/lu_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/lu_harness.c" -o "${tmp}/lu_h" \
  || { echo "  FAIL: longunary harness build"; exit 1; }
lur="$("${tmp}/lu_h")"
[ "${lur}" = "MATCH" ] \
  && echo "  PASS longunary: -/~ on long/long long == Clang" \
  || { echo "  FAIL: longunary behaviour (${lur})"; exit 1; }

# Storing a value into a `_Bool` / `bool` object (#boolnorm): C normalizes any nonzero to 1 on the
# conversion to bool (§6.3.1.2). The twin emitted a bool local as a plain `uint8_t`, so the store kept
# the raw value (`_Bool x = 2` left x == 2): a silent miscompile. Now the twin emits `_Bool`, so the
# store normalizes -- on a decl init, compound assignment, array element, parameter, and return.
echo "[c-runtime] store into a _Bool normalizes to 0/1 (bcir-cc): emit == Clang (#boolnorm)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_boolnorm.c" > "${tmp}/bn_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\bbool_norm\b/bn_src/' -e 's/\bbool_mask\b/bm_src/' -e 's/\bbool_mod\b/bd_src/' \
      -e 's/\bbool_compound\b/bc_src/' -e 's/\bbool_array\b/ba_src/' "${C}/cfront_boolnorm.c"
  cat "${tmp}/bn_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned a=0;a<300u;a++)for(unsigned b=0;b<260u;b++){
    if(bn_src(a,b)!=bcir_bool_norm(a,b)){printf("bn MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(bm_src(a,b)!=bcir_bool_mask(a,b)){printf("bm MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(bd_src(a,b)!=bcir_bool_mod(a,b)){printf("bd MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(bc_src(a,b)!=bcir_bool_compound(a,b)){printf("bc MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(ba_src(a,b)!=bcir_bool_array(a,b)){printf("ba MISMATCH a=%u b=%u\n",a,b);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/bn_harness.c"
"${CC}" -std=c23 -O2 -I "${C}" "${tmp}/bn_harness.c" "${C}/bcir_quarantine.c" -o "${tmp}/bn_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 -I "${C}" "${tmp}/bn_harness.c" "${C}/bcir_quarantine.c" -o "${tmp}/bn_h" \
  || { echo "  FAIL: boolnorm harness build"; exit 1; }
bnr="$("${tmp}/bn_h")"
[ "${bnr}" = "MATCH" ] \
  && echo "  PASS boolnorm: store into _Bool normalizes to 0/1 == Clang" \
  || { echo "  FAIL: boolnorm behaviour (${bnr})"; exit 1; }

# Integer promotion + float on unary `-` / `~` (#unarypromote): a sub-int unsigned operand promotes to
# *signed* int, so `~(unsigned char)0` is -1 (the twin kept it unsigned -> wrong sign test); and `-x` on
# a float stays float (both rails forced a uint32 temp, truncating -2.5 to a huge integer). Now both rails
# take the promoted operand type. Differential == Clang.
echo "[c-runtime] integer-promotion + float unary -/~ (bcir-cc): emit == Clang (#unarypromote)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_unarypromote.c" > "${tmp}/up_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\bucomplement\b/uc_src/' -e 's/\bunegsign\b/un_src/' -e 's/\bushortshift\b/us_src/' \
      -e 's/\bfnegate\b/fn_src/' -e 's/\bdnegate\b/dn_src/' "${C}/cfront_unarypromote.c"
  cat "${tmp}/up_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned a=0;a<4000u;a++)for(unsigned b=0;b<70u;b++){
    if(uc_src(a,b)!=bcir_ucomplement(a,b)){printf("uc MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(un_src(a,b)!=bcir_unegsign(a,b)){printf("un MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(us_src(a,b)!=bcir_ushortshift(a,b)){printf("us MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(fn_src(a,b)!=bcir_fnegate(a,b)){printf("fn MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(dn_src(a,b)!=bcir_dnegate(a,b)){printf("dn MISMATCH a=%u b=%u\n",a,b);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/up_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/up_harness.c" -o "${tmp}/up_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/up_harness.c" -o "${tmp}/up_h" \
  || { echo "  FAIL: unarypromote harness build"; exit 1; }
upr="$("${tmp}/up_h")"
[ "${upr}" = "MATCH" ] \
  && echo "  PASS unarypromote: sub-int -> signed int + float -/~ == Clang" \
  || { echo "  FAIL: unarypromote behaviour (${upr})"; exit 1; }

# float / double -> SIGNED integer cast (#floatsigncast): a floating value converted to a signed integer
# must use a signed cast operator. Both rails canonicalized integer casts to the unsigned spelling, making
# it a float -> unsigned conversion -- UB for a negative value, target-divergent (the #395 aarch64 32-bit
# failure), and even on x86 a sub-int signed target lost the sign. Now emit a signed temp + signed operator.
echo "[c-runtime] float -> signed-int cast (bcir-cc): emit == Clang (#floatsigncast)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_floatsigncast.c" > "${tmp}/fs_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\bf2int\b/f2i_src/' -e 's/\bf2short\b/f2s_src/' -e 's/\bf2schar\b/f2c_src/' \
      -e 's/\bd2long\b/d2l_src/' -e 's/\bf2uint\b/f2u_src/' "${C}/cfront_floatsigncast.c"
  cat "${tmp}/fs_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned a=0;a<4000u;a++)for(unsigned b=0;b<90u;b++){
    if(f2i_src(a,b)!=bcir_f2int(a,b)){printf("f2i MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(f2s_src(a,b)!=bcir_f2short(a,b)){printf("f2s MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(f2c_src(a,b)!=bcir_f2schar(a,b)){printf("f2c MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(d2l_src(a,b)!=bcir_d2long(a,b)){printf("d2l MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(f2u_src(a,b)!=bcir_f2uint(a,b)){printf("f2u MISMATCH a=%u b=%u\n",a,b);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/fs_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/fs_harness.c" -o "${tmp}/fs_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/fs_harness.c" -o "${tmp}/fs_h" \
  || { echo "  FAIL: floatsigncast harness build"; exit 1; }
fsr="$("${tmp}/fs_h")"
[ "${fsr}" = "MATCH" ] \
  && echo "  PASS floatsigncast: float/double -> signed int == Clang" \
  || { echo "  FAIL: floatsigncast behaviour (${fsr})"; exit 1; }

# Pure-integer narrowing cast to a SIGNED integer used directly (#intsigncast): the twin typed every
# integer cast temp as unsigned, so a signed sub-int target lost its sign when the cast value was used
# without landing in a signed named local (`(signed char)(-5)` -> 251), and `(int)u` read back unsigned
# (a logical `>>`). Now the twin types the cast temp with the target's signedness, matching the oracle.
echo "[c-runtime] integer cast to a signed type (bcir-cc): emit == Clang (#intsigncast)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_intsigncast.c" > "${tmp}/is_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\bi2schar\b/i2c_src/' -e 's/\bi2short\b/i2s_src/' -e 's/\bi2int_shr\b/i2i_src/' \
      -e 's/\bi2schar_arith\b/i2a_src/' -e 's/\bi2long_div\b/i2l_src/' -e 's/\bi2uchar\b/i2u_src/' \
      "${C}/cfront_intsigncast.c"
  cat "${tmp}/is_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned a=0;a<5000u;a++)for(unsigned b=0;b<80u;b++){
    if(i2c_src(a,b)!=bcir_i2schar(a,b)){printf("i2c MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(i2s_src(a,b)!=bcir_i2short(a,b)){printf("i2s MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(i2i_src(a,b)!=bcir_i2int_shr(a,b)){printf("i2i MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(i2a_src(a,b)!=bcir_i2schar_arith(a,b)){printf("i2a MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(i2l_src(a,b)!=bcir_i2long_div(a,b)){printf("i2l MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(i2u_src(a,b)!=bcir_i2uchar(a,b)){printf("i2u MISMATCH a=%u b=%u\n",a,b);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/is_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/is_harness.c" -o "${tmp}/is_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/is_harness.c" -o "${tmp}/is_h" \
  || { echo "  FAIL: intsigncast harness build"; exit 1; }
isr="$("${tmp}/is_h")"
[ "${isr}" = "MATCH" ] \
  && echo "  PASS intsigncast: integer cast to a signed type == Clang" \
  || { echo "  FAIL: intsigncast behaviour (${isr})"; exit 1; }

# Cast to _Bool / bool (#boolcast): `(_Bool)x` is `x != 0` on the FULL value (§6.3.1.2), so `(_Bool)256`
# is 1 and `(_Bool)0.5` is 1. Both rails rendered it as `(uint8_t)x`, truncating to 8 bits before the bool
# test (`(_Bool)256` -> 0), and the twin did not normalize at all. Now both rails emit a real `_Bool` cast.
echo "[c-runtime] cast to _Bool normalizes the full value (bcir-cc): emit == Clang (#boolcast)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_boolcast.c" > "${tmp}/bx_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdbool.h>'; echo '#include <stdio.h>'
  sed -e 's/\bbcast_low\b/bx_lo_src/' -e 's/\bbcast_high\b/bx_hi_src/' -e 's/\bbcast_mask\b/bx_mk_src/' \
      -e 's/\bbcast_float\b/bx_fl_src/' -e 's/\bbcast_arith\b/bx_ar_src/' "${C}/cfront_boolcast.c"
  cat "${tmp}/bx_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned a=0;a<70000u;a+=7u)for(unsigned b=0;b<300u;b++){
    if(bx_lo_src(a,b)!=bcir_bcast_low(a,b)){printf("lo MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(bx_hi_src(a,b)!=bcir_bcast_high(a,b)){printf("hi MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(bx_mk_src(a,b)!=bcir_bcast_mask(a,b)){printf("mk MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(bx_fl_src(a,b)!=bcir_bcast_float(a,b)){printf("fl MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(bx_ar_src(a,b)!=bcir_bcast_arith(a,b)){printf("ar MISMATCH a=%u b=%u\n",a,b);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/bx_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/bx_harness.c" -o "${tmp}/bx_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/bx_harness.c" -o "${tmp}/bx_h" \
  || { echo "  FAIL: boolcast harness build"; exit 1; }
bxr="$("${tmp}/bx_h")"
[ "${bxr}" = "MATCH" ] \
  && echo "  PASS boolcast: (_Bool)x normalizes the full value == Clang" \
  || { echo "  FAIL: boolcast behaviour (${bxr})"; exit 1; }

# Signed bitfield read sign-extension (#signedbf): a `signed`/`int` bitfield of width N holds an N-bit
# two's-complement value, so reading it sign-extends from bit N-1 (`int x:4` of 1111 reads -1, not 15).
# Both rails extracted `(unit >> off) & mask` and stopped, zero-extending every read. Now both carry the
# field's signedness on c.bf.get and sign-extend a signed field. Unsigned fields (zero-extend) unchanged.
echo "[c-runtime] signed bitfield read sign-extends (bcir-cc): emit == Clang (#signedbf)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_signedbf.c" > "${tmp}/sb_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bbf_read\b/sb_rd_src/' -e 's/\bbf_signtest\b/sb_st_src/' -e 's/\bbf_arith\b/sb_ar_src/' \
      -e 's/\bbf_onebit\b/sb_ob_src/' -e 's/\bbf_unsigned\b/sb_un_src/' "${C}/cfront_signedbf.c"
  cat "${tmp}/sb_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned a=0;a<70000u;a+=3u)for(unsigned b=0;b<300u;b++){
    if(sb_rd_src(a,b)!=bcir_bf_read(a,b)){printf("rd MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(sb_st_src(a,b)!=bcir_bf_signtest(a,b)){printf("st MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(sb_ar_src(a,b)!=bcir_bf_arith(a,b)){printf("ar MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(sb_ob_src(a,b)!=bcir_bf_onebit(a,b)){printf("ob MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(sb_un_src(a,b)!=bcir_bf_unsigned(a,b)){printf("un MISMATCH a=%u b=%u\n",a,b);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/sb_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/sb_harness.c" -o "${tmp}/sb_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/sb_harness.c" -o "${tmp}/sb_h" \
  || { echo "  FAIL: signedbf harness build"; exit 1; }
sbr="$("${tmp}/sb_h")"
[ "${sbr}" = "MATCH" ] \
  && echo "  PASS signedbf: signed bitfield read sign-extends == Clang" \
  || { echo "  FAIL: signedbf behaviour (${sbr})"; exit 1; }

# Signed sub-int storage read sign-extension (#signedload): reading a `signed char`/`short` from a struct
# member, a member array, or a local array must sign-extend to int. The twin loaded a member via a
# zero-extending memcpy into a uint32 temp, and typed the array-element temp unsigned -- so a signed read
# came back as a large positive and its `< 0` test was always false. Now the twin types each load temp with
# the element's (width, signedness) and memcpy's the exact width. Unsigned elements unchanged.
echo "[c-runtime] signed sub-int storage read sign-extends (bcir-cc): emit == Clang (#signedload)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_signedload.c" > "${tmp}/sl_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bm_signtest\b/sl_ms_src/' -e 's/\bm_value\b/sl_mv_src/' -e 's/\bm_arr_signtest\b/sl_ma_src/' \
      -e 's/\bloc_signtest\b/sl_ls_src/' -e 's/\bloc_value\b/sl_lv_src/' -e 's/\buc_control\b/sl_uc_src/' \
      "${C}/cfront_signedload.c"
  cat "${tmp}/sl_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned a=0;a<70000u;a+=3u)for(unsigned b=0;b<300u;b++){
    if(sl_ms_src(a,b)!=bcir_m_signtest(a,b)){printf("ms MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(sl_mv_src(a,b)!=bcir_m_value(a,b)){printf("mv MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(sl_ma_src(a,b)!=bcir_m_arr_signtest(a,b)){printf("ma MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(sl_ls_src(a,b)!=bcir_loc_signtest(a,b)){printf("ls MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(sl_lv_src(a,b)!=bcir_loc_value(a,b)){printf("lv MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(sl_uc_src(a,b)!=bcir_uc_control(a,b)){printf("uc MISMATCH a=%u b=%u\n",a,b);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/sl_harness.c"
"${CC}" -std=c23 -O2 -I "${C}" "${tmp}/sl_harness.c" "${C}/bcir_quarantine.c" -o "${tmp}/sl_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 -I "${C}" "${tmp}/sl_harness.c" "${C}/bcir_quarantine.c" -o "${tmp}/sl_h" \
  || { echo "  FAIL: signedload harness build"; exit 1; }
slr="$("${tmp}/sl_h")"
[ "${slr}" = "MATCH" ] \
  && echo "  PASS signedload: signed sub-int member/array read sign-extends == Clang" \
  || { echo "  FAIL: signedload behaviour (${slr})"; exit 1; }

# `enum` as a type (#enumtype): `enum Tag` is a valid int-sized type specifier (decl / cast / parameter).
# The twin accepted it; the oracle's type parser recomputed the base from an empty keyword run after the
# enum branch already set it, raising "expected a type" -- rejecting every enum-as-type (a rail
# disagreement, oracle stricter than the twin and Clang). Fixed in the oracle; both rails lower enum -> int.
echo "[c-runtime] enum used as a type (bcir-cc): emit == Clang (#enumtype)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_enumtype.c" > "${tmp}/et_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\be_cast\b/et_c_src/' -e 's/\be_select\b/et_s_src/' -e 's/\be_param\b/et_p_src/' \
      -e 's/\bsign_mag\b/et_sm_src/' "${C}/cfront_enumtype.c"
  cat "${tmp}/et_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned a=0;a<3000u;a++)for(unsigned b=0;b<120u;b++){
    if(et_c_src(a,b)!=bcir_e_cast(a,b)){printf("c MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(et_s_src(a,b)!=bcir_e_select(a,b)){printf("s MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(et_p_src(a,b)!=bcir_e_param(a,b)){printf("p MISMATCH a=%u b=%u\n",a,b);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/et_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/et_harness.c" -o "${tmp}/et_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/et_harness.c" -o "${tmp}/et_h" \
  || { echo "  FAIL: enumtype harness build"; exit 1; }
etr="$("${tmp}/et_h")"
[ "${etr}" = "MATCH" ] \
  && echo "  PASS enumtype: enum decl/cast/param == Clang" \
  || { echo "  FAIL: enumtype behaviour (${etr})"; exit 1; }

# Pointer locals (#ptrlocal): `T *p = &x;` + read/write `*p`. The twin emitted a pointer local as a plain
# uint32_t -- an invalid pointer-to-integer assignment under C23 and an 8-byte-pointer truncation on a
# 64-bit target. Now it carries the pointee type on the resource and emits a real `T *p`, with the deref
# reading exactly the pointee width (a signed sub-int pointee sign-extends). The oracle was already correct.
echo "[c-runtime] pointer locals -- T *p = &x (bcir-cc): emit == Clang (#ptrlocal)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_ptrlocal.c" > "${tmp}/pl_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bpl_store\b/pl_st_src/' -e 's/\bpl_read\b/pl_rd_src/' -e 's/\bpl_schar\b/pl_sc_src/' \
      -e 's/\bpl_compound\b/pl_cp_src/' -e 's/\bpl_struct\b/pl_su_src/' -e 's/\bpl_reassign\b/pl_re_src/' \
      "${C}/cfront_ptrlocal.c"
  cat "${tmp}/pl_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned a=0;a<70000u;a+=3u)for(unsigned b=0;b<300u;b++){
    if(pl_st_src(a,b)!=bcir_pl_store(a,b)){printf("st MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(pl_rd_src(a,b)!=bcir_pl_read(a,b)){printf("rd MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(pl_sc_src(a,b)!=bcir_pl_schar(a,b)){printf("sc MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(pl_cp_src(a,b)!=bcir_pl_compound(a,b)){printf("cp MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(pl_su_src(a,b)!=bcir_pl_struct(a,b)){printf("su MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(pl_re_src(a,b)!=bcir_pl_reassign(a,b)){printf("re MISMATCH a=%u b=%u\n",a,b);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/pl_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/pl_harness.c" -o "${tmp}/pl_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/pl_harness.c" -o "${tmp}/pl_h" \
  || { echo "  FAIL: ptrlocal harness build"; exit 1; }
plr="$("${tmp}/pl_h")"
[ "${plr}" = "MATCH" ] \
  && echo "  PASS ptrlocal: T *p = &x read/write/compound/struct == Clang" \
  || { echo "  FAIL: ptrlocal behaviour (${plr})"; exit 1; }

# Pointer values returned by value (#ptrvalue): pointer arithmetic `p + i` as an rvalue yields a pointer
# that is returned. The twin typed that temp as uint32_t -- an invalid pointer-from-integer return and an
# 8-byte-pointer truncation on a 64-bit target. Now both rails carry the pointee type (`T *t = p + i`).
# Each function returns a pointer into a shared buffer; compare the returned offsets against Clang.
echo "[c-runtime] pointer values -- return p + i (bcir-cc): emit == Clang (#ptrvalue)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_ptrvalue.c" > "${tmp}/pv_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\bpv_advance\b/pv_adv_src/' -e 's/\bpv_commute\b/pv_com_src/' -e 's/\bpv_back\b/pv_bk_src/' \
      -e 's/\bpv_uadvance\b/pv_uadv_src/' -e 's/\bpv_sadvance\b/pv_sadv_src/' -e 's/\bpv_chain\b/pv_ch_src/' \
      "${C}/cfront_ptrvalue.c"
  cat "${tmp}/pv_emit.c"
  cat <<'DRV'
int main(void){
  static int ib[512]; static unsigned ub[512]; static short sb[512];
  for(unsigned n=0;n<190u;n++){
    if(pv_adv_src(ib,n)  !=bcir_pv_advance(ib,n))  {printf("adv MISMATCH n=%u\n",n);return 1;}
    if(pv_com_src(ib,n)  !=bcir_pv_commute(ib,n))  {printf("com MISMATCH n=%u\n",n);return 1;}
    if(pv_bk_src(ib,n)   !=bcir_pv_back(ib,n))     {printf("bk MISMATCH n=%u\n",n);return 1;}
    if(pv_uadv_src(ub,n) !=bcir_pv_uadvance(ub,n)) {printf("uadv MISMATCH n=%u\n",n);return 1;}
    if(pv_sadv_src(sb,n) !=bcir_pv_sadvance(sb,n)) {printf("sadv MISMATCH n=%u\n",n);return 1;}
    if(pv_ch_src(ib,n)   !=bcir_pv_chain(ib,n))    {printf("ch MISMATCH n=%u\n",n);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/pv_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/pv_harness.c" -o "${tmp}/pv_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/pv_harness.c" -o "${tmp}/pv_h" \
  || { echo "  FAIL: ptrvalue harness build"; exit 1; }
pvr="$("${tmp}/pv_h")"
[ "${pvr}" = "MATCH" ] \
  && echo "  PASS ptrvalue: returned p + i pointer == Clang (int/unsigned/short pointees)" \
  || { echo "  FAIL: ptrvalue behaviour (${pvr})"; exit 1; }

# Pointer values stored into / loaded from a struct field (#ptrfield): the twin modeled a pointer member
# as 4 bytes, so the struct LAYOUT was wrong (an adjacent field overlapped the high half of the pointer)
# and a store truncated the 8-byte pointer. Now the member occupies pointer_size and the store/load moves
# the full pointer with its real `T *` type. Each function stores then reads back; the scalar between two
# pointers (written first) survives only under the 8-byte layout, and the returned pointers round-trip.
echo "[c-runtime] pointer struct fields -- s->p = q (bcir-cc): emit == Clang (#ptrfield)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_ptrfield.c" > "${tmp}/pf_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bpf_store_return\b/pf_sr_src/' -e 's/\bpf_layout\b/pf_ly_src/' \
      -e 's/\bpf_long\b/pf_lg_src/' -e 's/\bpf_both\b/pf_bo_src/' "${C}/cfront_ptrfield.c"
  cat "${tmp}/pf_emit.c"
  cat <<'DRV'
int main(void){
  static int iv[300]; static long lv[300]; struct Node a, b;
  for(int i=0;i<290;i++){
    int k = i*7 - 11;
    if(pf_sr_src(&a,&iv[i])           != bcir_pf_store_return(&b,&iv[i]))     {printf("sr MISMATCH i=%d\n",i);return 1;}
    if(pf_ly_src(&a,&iv[i],k)         != bcir_pf_layout(&b,&iv[i],k))         {printf("ly MISMATCH i=%d\n",i);return 1;}
    if(pf_lg_src(&a,&lv[i])           != bcir_pf_long(&b,&lv[i]))             {printf("lg MISMATCH i=%d\n",i);return 1;}
    if(pf_bo_src(&a,&iv[i],&lv[i],k)  != bcir_pf_both(&b,&iv[i],&lv[i],k))    {printf("bo MISMATCH i=%d\n",i);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/pf_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/pf_harness.c" -o "${tmp}/pf_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/pf_harness.c" -o "${tmp}/pf_h" \
  || { echo "  FAIL: ptrfield harness build"; exit 1; }
pfr="$("${tmp}/pf_h")"
[ "${pfr}" = "MATCH" ] \
  && echo "  PASS ptrfield: 8-byte member layout + untruncated store/load == Clang" \
  || { echo "  FAIL: ptrfield behaviour (${pfr})"; exit 1; }

# Pointer-to-pointer (#ptr2ptr): `int **pp`, `**pp`, `*pp = q` (out-param), `**pp = v`, `int **pp = &p`.
# Both rails modeled `int **` as a single `int *` (no indirection depth), so `*pp` read the base width,
# `**pp` fell back, and a store truncated. Now the type carries a pointer DEPTH. A bespoke harness (the
# generic one would fill a pointee with random bytes, invalid to deref for a double pointer) builds real
# x / &x / &&x chains.
echo "[c-runtime] pointer-to-pointer -- int **pp, **pp (bcir-cc): emit == Clang (#ptr2ptr)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_ptr2ptr.c" > "${tmp}/pp_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bp2_read\b/p2_read_s/' -e 's/\bp2_get\b/p2_get_s/' -e 's/\bp2_set\b/p2_set_s/' \
      -e 's/\bp2_store_through\b/p2_st_s/' -e 's/\bp2_rmw\b/p2_rmw_s/' -e 's/\bp2_local\b/p2_local_s/' \
      "${C}/cfront_ptr2ptr.c"
  sed -e 's/\bbcir_p2_store_through\b/bcir_p2_st/' "${tmp}/pp_emit.c"
  cat <<'DRV'
int main(void){
  for(int i=-50;i<3000;i+=7){
    int x=i*3-7; int *px=&x; int **ppx=&px;
    if(p2_read_s(ppx)!=bcir_p2_read(ppx)){printf("read@%d\n",i);return 1;}
    if(p2_get_s(ppx)!=bcir_p2_get(ppx)){printf("get@%d\n",i);return 1;}
    int y=i+9; int *a1=px,*a2=px; int **b1=&a1,**b2=&a2;
    p2_set_s(b1,&y); bcir_p2_set(b2,&y);
    if(*b1!=*b2){printf("set@%d\n",i);return 1;}
    int s1=x,s2=x; int *p1=&s1,*p2=&s2; int **q1=&p1,**q2=&p2;
    if(p2_st_s(q1,i)!=bcir_p2_st(q2,i)||s1!=s2){printf("store@%d\n",i);return 1;}
    int u1=x,u2=x; int *r1=&u1,*r2=&u2; int **w1=&r1,**w2=&r2;
    if(p2_rmw_s(w1,i)!=bcir_p2_rmw(w2,i)||u1!=u2){printf("rmw@%d\n",i);return 1;}
    if(p2_local_s(i)!=bcir_p2_local(i)){printf("local@%d\n",i);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/pp_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/pp_harness.c" -o "${tmp}/pp_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/pp_harness.c" -o "${tmp}/pp_h" \
  || { echo "  FAIL: ptr2ptr harness build"; exit 1; }
ppr="$("${tmp}/pp_h")"
[ "${ppr}" = "MATCH" ] \
  && echo "  PASS ptr2ptr: int **pp / **pp / *pp=q / **pp=v / &p == Clang" \
  || { echo "  FAIL: ptr2ptr behaviour (${ppr})"; exit 1; }

# Deref-through a loaded pointer field (#fieldderef): `*(s->p)`, the chain `s->mid->k` / two-hop
# `s->mid->leaf->x`, and the subscript `s->p[i]` -- reads, writes, RMW. A member used as a base was
# resolved to the struct's address + the field type (so a deref read the struct's own bytes); now a
# pointer-valued field used as a base is loaded and the loaded pointer becomes the new base. A bespoke
# harness builds real Box->Mid->Leaf chains (the generic one would fill a pointee with random bytes).
echo "[c-runtime] deref-through a loaded pointer field -- *(s->p) / s->mid->leaf->x / s->p[i] (#fieldderef)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_fieldderef.c" > "${tmp}/fd_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bfd_read\b/fd_read_s/' -e 's/\bfd_write\b/fd_write_s/' -e 's/\bfd_qread\b/fd_qread_s/' \
      -e 's/\bfd_index\b/fd_index_s/' -e 's/\bfd_index_set\b/fd_index_set_s/' -e 's/\bfd_rmw\b/fd_rmw_s/' \
      -e 's/\bfd_chain1\b/fd_chain1_s/' -e 's/\bfd_chain1_set\b/fd_chain1_set_s/' -e 's/\bfd_chain1_rmw\b/fd_chain1_rmw_s/' \
      -e 's/\bfd_chain2\b/fd_chain2_s/' -e 's/\bfd_chain2_long\b/fd_chain2_long_s/' -e 's/\bfd_chain2_set\b/fd_chain2_set_s/' \
      "${C}/cfront_fieldderef.c"
  cat "${tmp}/fd_emit.c"
  cat <<'DRV'
int main(void){
  for(int i=-40;i<2000;i+=7){
    int buf[8]; for(int k=0;k<8;k++) buf[k]=i*k-3;
    long lq=(long)i*1000003L-7;
    struct Box b={0,&buf[0],i,&lq};
    if(fd_read_s(&b)!=bcir_fd_read(&b)){printf("read@%d\n",i);return 1;}
    if(fd_qread_s(&b)!=bcir_fd_qread(&b)){printf("qread@%d\n",i);return 1;}
    if(fd_index_s(&b,5)!=bcir_fd_index(&b,5)){printf("index@%d\n",i);return 1;}
    int w1[4]={0},w2[4]={0};
    struct Box c1={0,&w1[0],0,&lq},c2={0,&w2[0],0,&lq};
    fd_write_s(&c1,i); bcir_fd_write(&c2,i);
    if(w1[0]!=w2[0]){printf("write@%d\n",i);return 1;}
    fd_index_set_s(&c1,3,i); bcir_fd_index_set(&c2,3,i);
    if(w1[3]!=w2[3]){printf("iset@%d\n",i);return 1;}
    if(fd_rmw_s(&c1,i)!=bcir_fd_rmw(&c2,i)||w1[0]!=w2[0]){printf("rmw@%d\n",i);return 1;}
    struct Leaf lf1={i+1,(long)i*7+2},lf2={i+1,(long)i*7+2};
    struct Mid m1={&lf1,i+5},m2={&lf2,i+5};
    struct Box d1={&m1,&buf[0],0,&lq},d2={&m2,&buf[0],0,&lq};
    if(fd_chain1_s(&d1)!=bcir_fd_chain1(&d2)){printf("chain1@%d\n",i);return 1;}
    if(fd_chain2_s(&d1)!=bcir_fd_chain2(&d2)){printf("chain2@%d\n",i);return 1;}
    if(fd_chain2_long_s(&d1)!=bcir_fd_chain2_long(&d2)){printf("chain2l@%d\n",i);return 1;}
    fd_chain1_set_s(&d1,i*3); bcir_fd_chain1_set(&d2,i*3);
    if(m1.k!=m2.k){printf("c1set@%d\n",i);return 1;}
    if(fd_chain1_rmw_s(&d1,i)!=bcir_fd_chain1_rmw(&d2,i)||m1.k!=m2.k){printf("c1rmw@%d\n",i);return 1;}
    fd_chain2_set_s(&d1,i*2); bcir_fd_chain2_set(&d2,i*2);
    if(lf1.x!=lf2.x){printf("c2set@%d\n",i);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/fd_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/fd_harness.c" -o "${tmp}/fd_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/fd_harness.c" -o "${tmp}/fd_h" \
  || { echo "  FAIL: fieldderef harness build"; exit 1; }
fdr="$("${tmp}/fd_h")"
[ "${fdr}" = "MATCH" ] \
  && echo "  PASS fieldderef: *(s->p) / s->mid->leaf->x / s->p[i] (read+write+rmw) == Clang" \
  || { echo "  FAIL: fieldderef behaviour (${fdr})"; exit 1; }

# Pointer-element signedness (#ptrsign): a load / store / subscript through a pointer carries the
# pointee's SIGNEDNESS, not just its width -- a signed sub-int pointee sign-extends, an unsigned one
# zero-extends, and the loaded value drives signed-vs-unsigned divide / shift / comparison / UAC. A
# bespoke harness sweeps negative + boundary pointee values (a width-only model would diverge here).
echo "[c-runtime] pointer-element signedness -- signed vs unsigned pointee load/store (#ptrsign)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_ptrsign.c" > "${tmp}/psn_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bps_s8\b/ps_s8_s/' -e 's/\bps_u8\b/ps_u8_s/' -e 's/\bps_s16\b/ps_s16_s/' \
      -e 's/\bps_u16\b/ps_u16_s/' -e 's/\bps_s8_divrem\b/ps_s8_divrem_s/' -e 's/\bps_u8_div\b/ps_u8_div_s/' \
      -e 's/\bps_s8_shr\b/ps_s8_shr_s/' -e 's/\bps_u8_shr\b/ps_u8_shr_s/' -e 's/\bps_s8_cmp\b/ps_s8_cmp_s/' \
      -e 's/\bps_s64_div\b/ps_s64_div_s/' -e 's/\bps_u64_div\b/ps_u64_div_s/' -e 's/\bps_arith\b/ps_arith_s/' \
      -e 's/\bps_uac\b/ps_uac_s/' -e 's/\bps_field\b/ps_field_s/' -e 's/\bps_w8\b/ps_w8_s/' \
      "${C}/cfront_ptrsign.c"
  cat "${tmp}/psn_emit.c"
  cat <<'DRV'
int main(void){
  for(long i=-400;i<400;i++){
    int8_t sb=(int8_t)(i*7-3); uint8_t ub=(uint8_t)(i*5+1);
    int16_t ha[4]={(int16_t)(i*3),(int16_t)(-i),(int16_t)(i+9),(int16_t)(i*7)};
    uint16_t ua[4]={(uint16_t)(i*3),(uint16_t)(i),(uint16_t)(i+9),(uint16_t)(i*7)};
    long lv=i*1000000007L-7; unsigned long ul=(unsigned long)(i*2654435761UL+9);
    int8_t a[4]={(int8_t)i,(int8_t)(i-1),(int8_t)(i+2),(int8_t)(-i)};
    if(ps_s8_s(&sb)!=bcir_ps_s8(&sb)){printf("s8@%ld\n",i);return 1;}
    if(ps_u8_s(&ub)!=bcir_ps_u8(&ub)){printf("u8@%ld\n",i);return 1;}
    if(ps_s16_s(ha,2)!=bcir_ps_s16(ha,2)){printf("s16@%ld\n",i);return 1;}
    if(ps_u16_s(ua,2)!=bcir_ps_u16(ua,2)){printf("u16@%ld\n",i);return 1;}
    if(ps_s8_divrem_s(&sb)!=bcir_ps_s8_divrem(&sb)){printf("divrem@%ld\n",i);return 1;}
    if(ps_u8_div_s(&ub)!=bcir_ps_u8_div(&ub)){printf("udiv@%ld\n",i);return 1;}
    if(ps_s8_shr_s(&sb)!=bcir_ps_s8_shr(&sb)){printf("sshr@%ld\n",i);return 1;}
    if(ps_u8_shr_s(&ub)!=bcir_ps_u8_shr(&ub)){printf("ushr@%ld\n",i);return 1;}
    if(ps_s8_cmp_s(&sb)!=bcir_ps_s8_cmp(&sb)){printf("cmp@%ld\n",i);return 1;}
    if(ps_s64_div_s(&lv)!=bcir_ps_s64_div(&lv)){printf("s64@%ld\n",i);return 1;}
    if(ps_u64_div_s(&ul)!=bcir_ps_u64_div(&ul)){printf("u64@%ld\n",i);return 1;}
    if(ps_arith_s(a,2)!=bcir_ps_arith(a,2)){printf("arith@%ld\n",i);return 1;}
    if(ps_uac_s(&ub,(int)i)!=bcir_ps_uac(&ub,(int)i)){printf("uac@%ld\n",i);return 1;}
    struct Buf bb={&sb,&ub}; if(ps_field_s(&bb)!=bcir_ps_field(&bb)){printf("field@%ld\n",i);return 1;}
    signed char w1=0,w2=0; ps_w8_s(&w1,(int)i); bcir_ps_w8(&w2,(int)i);
    if(w1!=w2){printf("w8@%ld\n",i);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/psn_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/psn_harness.c" -o "${tmp}/psn_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/psn_harness.c" -o "${tmp}/psn_h" \
  || { echo "  FAIL: ptrsign harness build"; exit 1; }
psnr="$("${tmp}/psn_h")"
[ "${psnr}" = "MATCH" ] \
  && echo "  PASS ptrsign: signed/unsigned pointee load/store/divide/shift/cmp == Clang" \
  || { echo "  FAIL: ptrsign behaviour (${psnr})"; exit 1; }

# Funcptr dispatch through a loaded pointer (#fnptrchain): a function-pointer struct member reached
# THROUGH a loaded pointer-to-struct field -- `d->ops->fn(args)`, the two-hop `s->dev->ops->fn(args)`.
# The postfix pointer chain recognizes a `(` after a member as a fused indirect call on the loaded
# pointer base (`ptr->fn(args)`). A bespoke harness wires real operation tables (R18-opaque dispatch).
echo "[c-runtime] funcptr dispatch through a loaded pointer -- d->ops->fn(args) (#fnptrchain)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_fnptrchain.c" > "${tmp}/fcc_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bfc_add\b/fc_add_s/' -e 's/\bfc_combo\b/fc_combo_s/' -e 's/\bfc_twohop\b/fc_twohop_s/' \
      "${C}/cfront_fnptrchain.c"
  cat "${tmp}/fcc_emit.c"
  cat <<'DRV'
static int real_add(int a,int b){return a+b;}
static int real_sub(int a,int b){return a-b;}
static int real_mul(int a,int b){return a*b;}
int main(void){
  struct Ops ops={real_add,real_sub,real_mul};
  struct Dev dev={&ops,42};
  struct Sys sys={&dev,7};
  for(int i=-200;i<200;i++){
    int a=i*3-1,b=7-i;
    if(fc_add_s(&dev,a,b)!=bcir_fc_add(&dev,a,b)){printf("add@%d\n",i);return 1;}
    if(fc_combo_s(&dev,a,b)!=bcir_fc_combo(&dev,a,b)){printf("combo@%d\n",i);return 1;}
    if(fc_twohop_s(&sys,a,b)!=bcir_fc_twohop(&sys,a,b)){printf("twohop@%d\n",i);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/fcc_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/fcc_harness.c" -o "${tmp}/fcc_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/fcc_harness.c" -o "${tmp}/fcc_h" \
  || { echo "  FAIL: fnptrchain harness build"; exit 1; }
fccr="$("${tmp}/fcc_h")"
[ "${fccr}" = "MATCH" ] \
  && echo "  PASS fnptrchain: d->ops->fn(args) / s->dev->ops->fn(args) == Clang" \
  || { echo "  FAIL: fnptrchain behaviour (${fccr})"; exit 1; }

# Per-declarator pointer/array shape in a multi-declarator declaration (#multiptr): `int *p, q;` types
# p as `int*` and q as `int` (the `*` binds to the declarator); `int *p, *q;` types both as pointers
# (was rejected by the twin). The twin now parses the specifier once and applies each declarator's own
# `*`/`[]` on a fresh copy, for locals + struct members. A differential uses each trailing declarator
# AS a scalar (a wide store would clobber); also pins a `long m` member store moving 8 bytes, not 4.
echo "[c-runtime] per-declarator pointer/array in a multi-declarator decl -- int *p, q; (#multiptr)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_multiptr.c" > "${tmp}/mpt_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bmd_local_mixed\b/md_local_mixed_s/' -e 's/\bmd_local_two_ptr\b/md_local_two_ptr_s/' \
      -e 's/\bmd_local_ptr_arr\b/md_local_ptr_arr_s/' -e 's/\bmd_struct\b/md_struct_s/' \
      "${C}/cfront_multiptr.c"
  cat "${tmp}/mpt_emit.c"
  cat <<'DRV'
int main(void){
  for(int i=-300;i<300;i++){
    int a=i*3-1,b=7-i;
    if(md_local_mixed_s(i)!=bcir_md_local_mixed(i)){printf("mixed@%d\n",i);return 1;}
    if(md_local_two_ptr_s(a,b)!=bcir_md_local_two_ptr(a,b)){printf("twoptr@%d\n",i);return 1;}
    if(md_local_ptr_arr_s(i)!=bcir_md_local_ptr_arr(i)){printf("ptrarr@%d\n",i);return 1;}
    struct Mix m1,m2;
    if(md_struct_s(&m1,i)!=bcir_md_struct(&m2,i)){printf("struct@%d\n",i);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/mpt_harness.c"
"${CC}" -std=c23 -O2 -I "${C}" "${tmp}/mpt_harness.c" "${C}/bcir_quarantine.c" -o "${tmp}/mpt_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 -I "${C}" "${tmp}/mpt_harness.c" "${C}/bcir_quarantine.c" -o "${tmp}/mpt_h" \
  || { echo "  FAIL: multiptr harness build"; exit 1; }
mptr="$("${tmp}/mpt_h")"
[ "${mptr}" = "MATCH" ] \
  && echo "  PASS multiptr: int *p, q; / int *p, *q; / struct + wide member store == Clang" \
  || { echo "  FAIL: multiptr behaviour (${mptr})"; exit 1; }

# Faithful char types (#chartypes): the three distinct one-byte char types emit faithfully -- plain
# `char` -> `char` (impl-defined sign), `signed char` -> always signed, `unsigned char` -> always
# unsigned. Built under BOTH -fsigned-char AND -funsigned-char so plain char's platform sign is
# exercised both ways (the old emit collapsed `signed char` -> `char` / plain `char` -> int8_t, wrong
# on one of the two).
echo "[c-runtime] faithful char types -- char / signed char / unsigned char (#chartypes)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_chartypes.c" > "${tmp}/cht_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bct_plain_deref\b/ct_plain_deref_s/' -e 's/\bct_signed_deref\b/ct_signed_deref_s/' \
      -e 's/\bct_unsigned_deref\b/ct_unsigned_deref_s/' -e 's/\bct_plain_cmp\b/ct_plain_cmp_s/' \
      -e 's/\bct_signed_cmp\b/ct_signed_cmp_s/' -e 's/\bct_plain_div\b/ct_plain_div_s/' \
      -e 's/\bct_signed_div\b/ct_signed_div_s/' -e 's/\bct_unsigned_div\b/ct_unsigned_div_s/' \
      -e 's/\bct_roundtrip\b/ct_roundtrip_s/' -e 's/\bct_plain_widen\b/ct_plain_widen_s/' \
      "${C}/cfront_chartypes.c"
  cat "${tmp}/cht_emit.c"
  cat <<'DRV'
int main(void){
  for(int i=-200;i<200;i++){
    char pc=(char)(i*7-3); signed char sc=(signed char)(i*5+1); unsigned char uc=(unsigned char)(i*3+2);
    if(ct_plain_deref_s(&pc)!=bcir_ct_plain_deref(&pc)){printf("pd@%d\n",i);return 1;}
    if(ct_signed_deref_s(&sc)!=bcir_ct_signed_deref(&sc)){printf("sd@%d\n",i);return 1;}
    if(ct_unsigned_deref_s(&uc)!=bcir_ct_unsigned_deref(&uc)){printf("ud@%d\n",i);return 1;}
    if(ct_plain_cmp_s(pc)!=bcir_ct_plain_cmp(pc)){printf("pc@%d\n",i);return 1;}
    if(ct_signed_cmp_s(sc)!=bcir_ct_signed_cmp(sc)){printf("sc@%d\n",i);return 1;}
    if(ct_plain_div_s(&pc)!=bcir_ct_plain_div(&pc)){printf("pdv@%d\n",i);return 1;}
    if(ct_signed_div_s(&sc)!=bcir_ct_signed_div(&sc)){printf("sdv@%d\n",i);return 1;}
    if(ct_unsigned_div_s(&uc)!=bcir_ct_unsigned_div(&uc)){printf("udv@%d\n",i);return 1;}
    if(ct_roundtrip_s(&pc)!=bcir_ct_roundtrip(&pc)){printf("rt@%d\n",i);return 1;}
    if(ct_plain_widen_s(&pc)!=bcir_ct_plain_widen(&pc)){printf("pw@%d\n",i);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/cht_harness.c"
for cm in -fsigned-char -funsigned-char; do
  "${CC}" -std=c23 -O2 "${cm}" "${tmp}/cht_harness.c" -o "${tmp}/cht_h" 2>/dev/null \
    || "${CC}" -std=c2x -O2 "${cm}" "${tmp}/cht_harness.c" -o "${tmp}/cht_h" \
    || { echo "  FAIL: chartypes harness build (${cm})"; exit 1; }
  chtr="$("${tmp}/cht_h")"
  [ "${chtr}" = "MATCH" ] \
    && echo "  PASS chartypes (${cm}): char / signed char / unsigned char == Clang" \
    || { echo "  FAIL: chartypes behaviour (${cm}: ${chtr})"; exit 1; }
done

# Compound literals (#complit): `(type){init}` materialized as a nameless local, in rvalue position
# (by-value struct arg / scalar value / member init), under `&` (pointer to the temporary), and with
# direct postfix on the literal (`(struct P){...}.field`).
echo "[c-runtime] compound literals -- (type){init} by value / scalar / &(literal) / .field (#complit)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_complit.c" > "${tmp}/cl_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bcl_byval\b/cl_byval_s/' -e 's/\bcl_designated\b/cl_designated_s/' \
      -e 's/\bcl_partial\b/cl_partial_s/' -e 's/\bcl_scalar\b/cl_scalar_s/' \
      -e 's/\bcl_addr_scalar\b/cl_addr_scalar_s/' -e 's/\bcl_addr_struct\b/cl_addr_struct_s/' \
      -e 's/\bcl_nested\b/cl_nested_s/' \
      -e 's/\bcl_dot\b/cl_dot_s/' -e 's/\bcl_dot_desig\b/cl_dot_desig_s/' \
      -e 's/\bcl_dot_part\b/cl_dot_part_s/' -e 's/\bcl_dot_wide\b/cl_dot_wide_s/' \
      "${C}/cfront_complit.c"
  cat "${tmp}/cl_emit.c"
  cat <<'DRV'
int main(void){
  for(int a=-40;a<40;a++) for(int b=-7;b<7;b++){
    if(cl_byval_s(a,b)!=bcir_cl_byval(a,b)){printf("byval@%d,%d\n",a,b);return 1;}
    if(cl_designated_s(a,b)!=bcir_cl_designated(a,b)){printf("desig@%d,%d\n",a,b);return 1;}
    if(cl_partial_s(a)!=bcir_cl_partial(a)){printf("partial@%d\n",a);return 1;}
    if(cl_scalar_s(a)!=bcir_cl_scalar(a)){printf("scalar@%d\n",a);return 1;}
    if(cl_addr_scalar_s(a)!=bcir_cl_addr_scalar(a)){printf("as@%d\n",a);return 1;}
    if(cl_addr_struct_s(a,b)!=bcir_cl_addr_struct(a,b)){printf("ast@%d,%d\n",a,b);return 1;}
    if(cl_nested_s(a)!=bcir_cl_nested(a)){printf("nested@%d\n",a);return 1;}
    if(cl_dot_s(a,b)!=bcir_cl_dot(a,b)){printf("dot@%d,%d\n",a,b);return 1;}
    if(cl_dot_desig_s(a,b)!=bcir_cl_dot_desig(a,b)){printf("dotdes@%d,%d\n",a,b);return 1;}
    if(cl_dot_part_s(a)!=bcir_cl_dot_part(a)){printf("dotpart@%d\n",a);return 1;}
    if(cl_dot_wide_s(a)!=bcir_cl_dot_wide(a)){printf("dotwide@%d\n",a);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/cl_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/cl_harness.c" -o "${tmp}/cl_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/cl_harness.c" -o "${tmp}/cl_h" \
  || { echo "  FAIL: complit harness build"; exit 1; }
clr="$("${tmp}/cl_h")"
[ "${clr}" = "MATCH" ] \
  && echo "  PASS complit: by value / designators / &(int){v} / &(struct){...} / (struct){...}.field == Clang" \
  || { echo "  FAIL: complit behaviour (${clr})"; exit 1; }

# typeof (#typeof): `typeof(type-name)` / `typeof(variable)` / `typeof(expression)` resolves to the
# operand's type -- each case is built so the WRONG type (int vs long, signed vs unsigned, a short
# truncation) would diverge. The expression operand is type-inferred (oracle) / speculatively lowered
# then rolled back (twin), as it is unevaluated.
echo "[c-runtime] typeof -- typeof(type-name) / typeof(variable) / typeof(expr) (#typeof)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_typeof.c" > "${tmp}/to_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bto_width\b/to_width_s/' -e 's/\bto_sign\b/to_sign_s/' -e 's/\bto_typename\b/to_typename_s/' \
      -e 's/\bto_ptr\b/to_ptr_s/' -e 's/\bto_struct\b/to_struct_s/' -e 's/\bto_unqual\b/to_unqual_s/' \
      -e 's/\bto_ebinop\b/to_ebinop_s/' -e 's/\bto_ebinsign\b/to_ebinsign_s/' -e 's/\bto_ecast\b/to_ecast_s/' \
      -e 's/\bto_ederef\b/to_ederef_s/' -e 's/\bto_emember\b/to_emember_s/' -e 's/\bto_eindex\b/to_eindex_s/' \
      "${C}/cfront_typeof.c"
  cat "${tmp}/to_emit.c"
  cat <<'DRV'
int main(void){
  for(long a=-50;a<50;a++){
    if(to_width_s(a)!=bcir_to_width(a)){printf("width@%ld\n",a);return 1;}
    if(to_sign_s((unsigned)a)!=bcir_to_sign((unsigned)a)){printf("sign@%ld\n",a);return 1;}
    if(to_typename_s((int)a)!=bcir_to_typename((int)a)){printf("tn@%ld\n",a);return 1;}
    if(to_ptr_s((int)a)!=bcir_to_ptr((int)a)){printf("ptr@%ld\n",a);return 1;}
    if(to_struct_s((int)a)!=bcir_to_struct((int)a)){printf("struct@%ld\n",a);return 1;}
    if(to_unqual_s(a)!=bcir_to_unqual(a)){printf("unq@%ld\n",a);return 1;}
    if(to_ebinop_s(a)!=bcir_to_ebinop(a)){printf("ebinop@%ld\n",a);return 1;}
    if(to_ebinsign_s((unsigned)a)!=bcir_to_ebinsign((unsigned)a)){printf("ebinsign@%ld\n",a);return 1;}
    if(to_ecast_s((int)a)!=bcir_to_ecast((int)a)){printf("ecast@%ld\n",a);return 1;}
    if(to_ederef_s(a)!=bcir_to_ederef(a)){printf("ederef@%ld\n",a);return 1;}
    if(to_emember_s((int)a)!=bcir_to_emember((int)a)){printf("emember@%ld\n",a);return 1;}
    if(to_eindex_s(a)!=bcir_to_eindex(a)){printf("eindex@%ld\n",a);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/to_harness.c"
"${CC}" -std=c23 -O2 -I "${C}" "${tmp}/to_harness.c" "${C}/bcir_quarantine.c" -o "${tmp}/to_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 -I "${C}" "${tmp}/to_harness.c" "${C}/bcir_quarantine.c" -o "${tmp}/to_h" \
  || { echo "  FAIL: typeof harness build"; exit 1; }
tor="$("${tmp}/to_h")"
[ "${tor}" = "MATCH" ] \
  && echo "  PASS typeof: typeof type-name / variable / expr (a+b, (short)x, *p, s.f, arr[i]) == Clang" \
  || { echo "  FAIL: typeof behaviour (${tor})"; exit 1; }

# Variadic functions (#variadic): `f(T last, ...)` with <stdarg.h> -- a va_list cursor (va_start/va_arg/
# va_end), va_copy, a va_list parameter (vprintf-style forwarding), and a same-unit variadic call passing
# args past the fixed params (default promotions ride the real call). va_start/va_arg/va_end/va_copy lower
# as opaque builtins emitted verbatim; va_arg(ap, T) carries type T. Differential drives the int / float /
# va_copy / forwarding paths; a wrong emit (a truncated va_arg load, a dropped `...`) would diverge.
echo "[c-runtime] variadic functions -- va_list / va_start / va_arg / va_end / va_copy (#variadic)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_variadic.c" > "${tmp}/va_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'; echo '#include <stdarg.h>'
  sed -e 's/\bisum\b/isum_s/g' -e 's/\btwice\b/twice_s/g' -e 's/\bvsumv\b/vsumv_s/g' \
      -e 's/\bforward\b/forward_s/g' -e 's/\bnth\b/nth_s/g' -e 's/\bcaller\b/caller_s/g' \
      "${C}/cfront_variadic.c"
  cat "${tmp}/va_emit.c"
  cat <<'DRV'
int main(void){
  for(int a=-40;a<40;a++) for(int b=-40;b<40;b++){ int c=a-2*b;
    if(caller_s(a,b,c)!=bcir_caller(a,b,c)){printf("caller@%d,%d\n",a,b);return 1;}
    if(isum_s(4,a,b,c,a+b)!=bcir_isum(4,a,b,c,a+b)){printf("isum@%d,%d\n",a,b);return 1;}
    if(twice_s(3,a,b,c)!=bcir_twice(3,a,b,c)){printf("twice@%d,%d\n",a,b);return 1;}
    if(forward_s(3,a,b,c)!=bcir_forward(3,a,b,c)){printf("forward@%d,%d\n",a,b);return 1;}
    if(nth_s(2,(double)a,(double)b,(double)c)!=bcir_nth(2,(double)a,(double)b,(double)c)){printf("nth@%d,%d\n",a,b);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/va_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/va_harness.c" -o "${tmp}/va_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/va_harness.c" -o "${tmp}/va_h" \
  || { echo "  FAIL: variadic harness build"; exit 1; }
var="$("${tmp}/va_h")"
[ "${var}" = "MATCH" ] \
  && echo "  PASS variadic: va_list/va_start/va_arg/va_end/va_copy + va_list param + same-unit call == Clang" \
  || { echo "  FAIL: variadic behaviour (${var})"; exit 1; }

# Wide / floating compound assignment (#compoundwide): `OP=` / ++ / -- on a long/double lvalue keeps the
# operand width/float-ness (was truncated to a 4-byte uint32 at the compound-assign sites) -- across a
# local, a struct member, an array element and a pointer deref -- plus the float/wide-int variadic
# accumulation it unblocks. The driver spans values that overflow 32 bits, so a truncating result diverges.
echo "[c-runtime] wide/float compound assignment -- OP= / ++ / -- keeps width (#compoundwide)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_compoundwide.c" > "${tmp}/cw_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'; echo '#include <stdarg.h>'
  sed -e 's/\bl_local\b/l_local_s/g' -e 's/\bd_local\b/d_local_s/g' -e 's/\bl_inc\b/l_inc_s/g' \
      -e 's/\bl_member\b/l_member_s/g' -e 's/\bl_array\b/l_array_s/g' -e 's/\bl_ptr\b/l_ptr_s/g' \
      -e 's/\bd_vararg\b/d_vararg_s/g' -e 's/\bl_vararg\b/l_vararg_s/g' -e 's/\bdriver\b/driver_s/g' \
      "${C}/cfront_compoundwide.c"
  cat "${tmp}/cw_emit.c"
  cat <<'DRV'
int main(void){
  long V[]={0,1,-1,1000000000L,-1000000000L,5000000000L,-5000000000L,99999999999L};
  int n=(int)(sizeof V/sizeof V[0]);
  for(int i=0;i<n;i++) for(int j=0;j<n;j++){ long x=V[i],y=V[j];
    if(driver_s(x,y)!=bcir_driver(x,y)){printf("driver@%ld,%ld\n",x,y);return 1;}
    if(l_member_s(x,y)!=bcir_l_member(x,y)){printf("l_member@%ld,%ld\n",x,y);return 1;}
    if(l_ptr_s(x,y)!=bcir_l_ptr(x,y)){printf("l_ptr@%ld,%ld\n",x,y);return 1;}
    if(d_vararg_s(3,(double)x,(double)y,1.5)!=bcir_d_vararg(3,(double)x,(double)y,1.5)){printf("d_vararg@%ld,%ld\n",x,y);return 1;}
    if(l_vararg_s(3,x,y,x+y)!=bcir_l_vararg(3,x,y,x+y)){printf("l_vararg@%ld,%ld\n",x,y);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/cw_harness.c"
"${CC}" -std=c23 -O2 -I "${C}" "${tmp}/cw_harness.c" "${C}/bcir_quarantine.c" -o "${tmp}/cw_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 -I "${C}" "${tmp}/cw_harness.c" "${C}/bcir_quarantine.c" -o "${tmp}/cw_h" \
  || { echo "  FAIL: compoundwide harness build"; exit 1; }
cwr="$("${tmp}/cw_h")"
[ "${cwr}" = "MATCH" ] \
  && echo "  PASS compoundwide: long/double OP= / ++ / -- (local/member/array/ptr) + vararg accum == Clang" \
  || { echo "  FAIL: compoundwide behaviour (${cwr})"; exit 1; }

# External variadic calls (#extvariadic): the printf/scanf-family <stdio.h> variadics (snprintf/vsnprintf)
# emit verbatim, stay opaque to R18 (no bcir_ twin), return int; the format string passes through; a
# vsnprintf-forwarding wrapper hands its own va_list to the external. The differential compares the
# formatted BUFFER and the returned count against the real libc.
echo "[c-runtime] external variadic calls -- snprintf / vsnprintf passthrough (#extvariadic)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_extvariadic.c" > "${tmp}/ev_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'; echo '#include <stdarg.h>'
  sed -e 's/\bev_int\b/ev_int_s/g' -e 's/\bev_mix\b/ev_mix_s/g' -e 's/\bev_width\b/ev_width_s/g' \
      -e 's/\bev_fwd\b/ev_fwd_s/g' -e 's/\bev_call\b/ev_call_s/g' \
      "${C}/cfront_extvariadic.c"
  cat "${tmp}/ev_emit.c"
  cat <<'DRV'
int main(void){
  for(int x=-3000;x<3000;x++){ long y=(long)x*1234567L; char b1[64],b2[64]; int r1,r2;
    r1=ev_int_s(b1,x);   r2=bcir_ev_int(b2,x);   if(r1!=r2||strcmp(b1,b2)){printf("int@%d\n",x);return 1;}
    r1=ev_mix_s(b1,x,y); r2=bcir_ev_mix(b2,x,y); if(r1!=r2||strcmp(b1,b2)){printf("mix@%d\n",x);return 1;}
    r1=ev_width_s(b1,x); r2=bcir_ev_width(b2,x); if(r1!=r2||strcmp(b1,b2)){printf("width@%d\n",x);return 1;}
    r1=ev_call_s(b1,x,(int)(y&0xff)); r2=bcir_ev_call(b2,x,(int)(y&0xff));
    if(r1!=r2||strcmp(b1,b2)){printf("call@%d\n",x);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/ev_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/ev_harness.c" -o "${tmp}/ev_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/ev_harness.c" -o "${tmp}/ev_h" \
  || { echo "  FAIL: extvariadic harness build"; exit 1; }
evr="$("${tmp}/ev_h")"
[ "${evr}" = "MATCH" ] \
  && echo "  PASS extvariadic: snprintf/vsnprintf passthrough (buffer + count) == Clang" \
  || { echo "  FAIL: extvariadic behaviour (${evr})"; exit 1; }

# long double (#longdouble): the extended floating type -- the twin emits real `long double` C (closing a
# parse-gap vs the oracle). Differential over arithmetic, an `L` constant, double/int conversions, a
# `long double *`, the +l libm variants (sqrtl/fabsl), and += accumulation; a wrong width / a dropped
# `long double` would diverge from Clang's 80-bit result.
echo "[c-runtime] long double -- 80/128-bit extended float, +l libm variants (#longdouble)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_longdouble.c" > "${tmp}/ld_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'; echo '#include <math.h>'
  sed -e 's/\bld_arith\b/ld_arith_s/g' -e 's/\bld_promote\b/ld_promote_s/g' -e 's/\bld_to_int\b/ld_to_int_s/g' \
      -e 's/\bld_narrow\b/ld_narrow_s/g' -e 's/\bld_libm\b/ld_libm_s/g' -e 's/\bld_ptr\b/ld_ptr_s/g' \
      -e 's/\bld_acc\b/ld_acc_s/g' \
      "${C}/cfront_longdouble.c"
  cat "${tmp}/ld_emit.c"
  cat <<'DRV'
int main(void){
  for(int i=-300;i<300;i++){ long double a=(long double)i*0.3L, b=(long double)(i+5)*0.13L;
    if(ld_arith_s(a,b)!=bcir_ld_arith(a,b)){printf("arith@%d\n",i);return 1;}
    if(ld_promote_s((double)i*0.5,i)!=bcir_ld_promote((double)i*0.5,i)){printf("promote@%d\n",i);return 1;}
    if(ld_to_int_s(a,b)!=bcir_ld_to_int(a,b)){printf("toint@%d\n",i);return 1;}
    if(ld_narrow_s(a)!=bcir_ld_narrow(a)){printf("narrow@%d\n",i);return 1;}
    if(i>=0 && ld_libm_s(a)!=bcir_ld_libm(a)){printf("libm@%d\n",i);return 1;}
    if(ld_ptr_s(a)!=bcir_ld_ptr(a)){printf("ptr@%d\n",i);return 1;}
    if(ld_acc_s(i%17,a)!=bcir_ld_acc(i%17,a)){printf("acc@%d\n",i);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/ld_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/ld_harness.c" -o "${tmp}/ld_h" -lm 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/ld_harness.c" -o "${tmp}/ld_h" -lm \
  || { echo "  FAIL: longdouble harness build"; exit 1; }
ldr="$("${tmp}/ld_h")"
[ "${ldr}" = "MATCH" ] \
  && echo "  PASS longdouble: arith / L-const / conversions / long double * / sqrtl / += == Clang" \
  || { echo "  FAIL: longdouble behaviour (${ldr})"; exit 1; }

# _Generic (#generic): C11 generic selection on the controlling expression's static type. The differential
# drives int / long / unsigned / double / float / char / pointer controls -- a wrong type-match would pick
# the wrong arm and diverge. Both rails read the controlling type the same way (the twin off a
# speculatively-lowered value) so they select the same association.
echo "[c-runtime] _Generic -- C11 type-generic selection (#generic)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_generic.c" > "${tmp}/gn_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bg_int\b/g_int_s/g' -e 's/\bg_long\b/g_long_s/g' -e 's/\bg_uint\b/g_uint_s/g' \
      -e 's/\bg_double\b/g_double_s/g' -e 's/\bg_float\b/g_float_s/g' -e 's/\bg_char\b/g_char_s/g' \
      -e 's/\bg_ptr\b/g_ptr_s/g' -e 's/\bg_exprtype\b/g_exprtype_s/g' -e 's/\bg_default\b/g_default_s/g' \
      -e 's/\bg_compute\b/g_compute_s/g' \
      "${C}/cfront_generic.c"
  cat "${tmp}/gn_emit.c"
  cat <<'DRV'
int main(void){
  for(int i=-200;i<200;i++){ long b=(long)i*7777; int x=i;
    if(g_int_s(i)!=bcir_g_int(i)){puts("g_int");return 1;}
    if(g_long_s(b)!=bcir_g_long(b)){puts("g_long");return 1;}
    if(g_uint_s((unsigned)i)!=bcir_g_uint((unsigned)i)){puts("g_uint");return 1;}
    if(g_double_s((double)i)!=bcir_g_double((double)i)){puts("g_double");return 1;}
    if(g_float_s((float)i)!=bcir_g_float((float)i)){puts("g_float");return 1;}
    if(g_char_s((char)i)!=bcir_g_char((char)i)){puts("g_char");return 1;}
    if(g_ptr_s(&x)!=bcir_g_ptr(&x)){puts("g_ptr");return 1;}
    if(g_exprtype_s(i,b)!=bcir_g_exprtype(i,b)){puts("g_exprtype");return 1;}
    if(g_default_s((double)i)!=bcir_g_default((double)i)){puts("g_default");return 1;}
    if(g_compute_s(b)!=bcir_g_compute(b)){puts("g_compute");return 1;}
  }
  puts("MATCH");return 0;}
DRV
} > "${tmp}/gn_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/gn_harness.c" -o "${tmp}/gn_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/gn_harness.c" -o "${tmp}/gn_h" \
  || { echo "  FAIL: generic harness build"; exit 1; }
gnr="$("${tmp}/gn_h")"
[ "${gnr}" = "MATCH" ] \
  && echo "  PASS generic: _Generic selection (int/long/uint/double/float/char/ptr/default) == Clang" \
  || { echo "  FAIL: generic behaviour (${gnr})"; exit 1; }

# Nested / chained designated initializers (#designate): a designator list `.a.b` / `.v[i]` / `.m[i][j]`
# resolving to a cumulative byte offset. The differential drives nested struct chains, 1-D/2-D member
# arrays, out-of-order mixes and a 3-level chain -- a wrong offset would store to the wrong slot.
echo "[c-runtime] nested designated initializers -- .a.b / .v[i] / .m[i][j] (#designate)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_designate.c" > "${tmp}/de_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bdesig_chain\b/desig_chain_s/g' -e 's/\bdesig_memarr\b/desig_memarr_s/g' \
      -e 's/\bdesig_md\b/desig_md_s/g' -e 's/\bdesig_mix\b/desig_mix_s/g' -e 's/\bdesig_deep\b/desig_deep_s/g' \
      "${C}/cfront_designate.c"
  cat "${tmp}/de_emit.c"
  cat <<'DRV'
int main(void){
  for(int x=-300;x<300;x++){
    if(desig_chain_s(x)!=bcir_desig_chain(x)){printf("chain@%d\n",x);return 1;}
    if(desig_memarr_s(x)!=bcir_desig_memarr(x)){printf("memarr@%d\n",x);return 1;}
    if(desig_md_s(x)!=bcir_desig_md(x)){printf("md@%d\n",x);return 1;}
    if(desig_mix_s(x)!=bcir_desig_mix(x)){printf("mix@%d\n",x);return 1;}
    if(desig_deep_s(x)!=bcir_desig_deep(x)){printf("deep@%d\n",x);return 1;}
  }
  puts("MATCH");return 0;}
DRV
} > "${tmp}/de_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/de_harness.c" -o "${tmp}/de_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/de_harness.c" -o "${tmp}/de_h" \
  || { echo "  FAIL: designate harness build"; exit 1; }
der="$("${tmp}/de_h")"
[ "${der}" = "MATCH" ] \
  && echo "  PASS designate: .a.b / .v[i] / .m[i][j] / 3-level chain == Clang" \
  || { echo "  FAIL: designate behaviour (${der})"; exit 1; }

# Nested member access at a non-first offset (#nestoffset): `t.q.a` where `q` is not the first member --
# the oracle dropped q's byte offset (read+write), the twin over-aligned a nested struct member to its
# size. The differential drives read/write, a member array, a 3-level chain, and a non-first designated
# init; a dropped offset / wrong alignment would alias members and diverge.
echo "[c-runtime] nested member access at a non-first offset (#nestoffset)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_nestoffset.c" > "${tmp}/no_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bno_rw\b/no_rw_s/g' -e 's/\bno_memarr\b/no_memarr_s/g' -e 's/\bno_deep\b/no_deep_s/g' \
      -e 's/\bno_desig\b/no_desig_s/g' \
      "${C}/cfront_nestoffset.c"
  cat "${tmp}/no_emit.c"
  cat <<'DRV'
int main(void){
  for(int x=-300;x<300;x++){
    if(no_rw_s(x)!=bcir_no_rw(x)){printf("rw@%d\n",x);return 1;}
    if(no_memarr_s(x)!=bcir_no_memarr(x)){printf("memarr@%d\n",x);return 1;}
    if(no_deep_s(x)!=bcir_no_deep(x)){printf("deep@%d\n",x);return 1;}
    if(no_desig_s(x)!=bcir_no_desig(x)){printf("desig@%d\n",x);return 1;}
  }
  puts("MATCH");return 0;}
DRV
} > "${tmp}/no_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/no_harness.c" -o "${tmp}/no_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/no_harness.c" -o "${tmp}/no_h" \
  || { echo "  FAIL: nestoffset harness build"; exit 1; }
nor="$("${tmp}/no_h")"
[ "${nor}" = "MATCH" ] \
  && echo "  PASS nestoffset: non-first nested member read/write/array/chain/designated == Clang" \
  || { echo "  FAIL: nestoffset behaviour (${nor})"; exit 1; }

# Address-of a (nested) struct member (#addrmember): `&s.field` / `&t.q.a` / `&t.q` -> a typed
# `(T *)((char *)&base + off)`, used through a pointer and passed to a helper. A wrong offset/type would
# write the wrong slot or pass a bad address.
echo "[c-runtime] address-of a (nested) struct member -- &s.f / &t.q.a / &t.q (#addrmember)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_addrmember.c" > "${tmp}/am_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\baddone\b/addone_s/g' -e 's/\bam_first\b/am_first_s/g' -e 's/\bam_nested\b/am_nested_s/g' \
      -e 's/\bam_struct\b/am_struct_s/g' -e 's/\bam_arg\b/am_arg_s/g' \
      "${C}/cfront_addrmember.c"
  cat "${tmp}/am_emit.c"
  cat <<'DRV'
int main(void){
  for(int x=-200;x<200;x++){
    if(am_first_s(x)!=bcir_am_first(x)){puts("first");return 1;}
    if(am_nested_s(x)!=bcir_am_nested(x)){puts("nested");return 1;}
    if(am_struct_s(x)!=bcir_am_struct(x)){puts("struct");return 1;}
    if(am_arg_s(x)!=bcir_am_arg(x)){puts("arg");return 1;}
  }
  puts("MATCH");return 0;}
DRV
} > "${tmp}/am_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/am_harness.c" -o "${tmp}/am_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/am_harness.c" -o "${tmp}/am_h" \
  || { echo "  FAIL: addrmember harness build"; exit 1; }
amr="$("${tmp}/am_h")"
[ "${amr}" = "MATCH" ] \
  && echo "  PASS addrmember: &s.f / &t.q.a / &t.q (read/write/struct-ptr/arg) == Clang" \
  || { echo "  FAIL: addrmember behaviour (${amr})"; exit 1; }

# _Atomic local objects (#atomiclocal): `_Atomic int a;` / `_Atomic(int) a;` / const _Atomic / a
# pointer-to-atomic, as function locals (previously only the global form parsed). An unshared atomic local
# equals its plain type single-threaded; the differential drives compound-assign / shift / deref.
echo "[c-runtime] _Atomic local objects -- _Atomic int / _Atomic(int) (#atomiclocal)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_atomiclocal.c" > "${tmp}/at_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'; echo '#include <stdatomic.h>'
  sed -e 's/\ba_qual\b/a_qual_s/g' -e 's/\ba_paren\b/a_paren_s/g' -e 's/\ba_long\b/a_long_s/g' \
      -e 's/\ba_const\b/a_const_s/g' -e 's/\ba_ptr\b/a_ptr_s/g' \
      "${C}/cfront_atomiclocal.c"
  cat "${tmp}/at_emit.c"
  cat <<'DRV'
int main(void){
  for(int x=-300;x<300;x++){ long b=(long)x*100000;
    if(a_qual_s(x)!=bcir_a_qual(x)){puts("qual");return 1;}
    if(a_paren_s(x)!=bcir_a_paren(x)){puts("paren");return 1;}
    if(a_long_s(b)!=bcir_a_long(b)){puts("long");return 1;}
    if(a_const_s(x)!=bcir_a_const(x)){puts("const");return 1;}
    if(a_ptr_s(x)!=bcir_a_ptr(x)){puts("ptr");return 1;}
  }
  puts("MATCH");return 0;}
DRV
} > "${tmp}/at_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/at_harness.c" -o "${tmp}/at_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/at_harness.c" -o "${tmp}/at_h" \
  || { echo "  FAIL: atomiclocal harness build"; exit 1; }
atr="$("${tmp}/at_h")"
[ "${atr}" = "MATCH" ] \
  && echo "  PASS atomiclocal: _Atomic int / _Atomic(int) / const _Atomic / _Atomic(int)* == Clang" \
  || { echo "  FAIL: atomiclocal behaviour (${atr})"; exit 1; }

# GCC/Clang integer builtins (#builtins): __builtin_popcount/clz/ctz/ffs/parity/bswap/abs (+ l/ll), emitted
# verbatim (opaque to R18). The differential drives the bit-manip family; a wrong result type / a synthesized
# bcir_ twin would diverge or fail to link.
echo "[c-runtime] GCC/Clang integer builtins -- popcount/clz/ctz/bswap/abs (#builtins)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_builtins.c" > "${tmp}/bi_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bbi_pop\b/bi_pop_s/g' -e 's/\bbi_clz\b/bi_clz_s/g' -e 's/\bbi_ffs\b/bi_ffs_s/g' \
      -e 's/\bbi_bswap\b/bi_bswap_s/g' -e 's/\bbi_bswap64\b/bi_bswap64_s/g' -e 's/\bbi_abs\b/bi_abs_s/g' \
      "${C}/cfront_builtins.c"
  cat "${tmp}/bi_emit.c"
  cat <<'DRV'
int main(void){
  for(long i=-40000;i<40000;i+=3){ unsigned x=(unsigned)(i*131071); int xi=(int)i;
    unsigned long long w=(unsigned long long)x * 2654435761ULL + (unsigned)xi;
    if(bi_pop_s(x)!=bcir_bi_pop(x)){puts("pop");return 1;}
    if(bi_clz_s(x)!=bcir_bi_clz(x)){puts("clz");return 1;}
    if(bi_ffs_s(xi)!=bcir_bi_ffs(xi)){puts("ffs");return 1;}
    if(bi_bswap_s(x)!=bcir_bi_bswap(x)){puts("bswap");return 1;}
    if(bi_bswap64_s(w)!=bcir_bi_bswap64(w)){puts("bswap64");return 1;}
    if(bi_abs_s(xi)!=bcir_bi_abs(xi)){puts("abs");return 1;}
  }
  puts("MATCH");return 0;}
DRV
} > "${tmp}/bi_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/bi_harness.c" -o "${tmp}/bi_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/bi_harness.c" -o "${tmp}/bi_h" \
  || { echo "  FAIL: builtins harness build"; exit 1; }
bir="$("${tmp}/bi_h")"
[ "${bir}" = "MATCH" ] \
  && echo "  PASS builtins: popcount/clz/ctz/ffs/parity/bswap16-32-64/abs/labs/llabs == Clang" \
  || { echo "  FAIL: builtins behaviour (${bir})"; exit 1; }

# GCC statement expressions (#stmtexpr): `({ ...; e; })` -- a scoped compound statement whose value is the
# last expression. The differential drives the temporary idiom, a loop inside, nesting, and scope
# shadowing; a wrong value / leaked scope would diverge.
echo "[c-runtime] GCC statement expressions -- ({ ...; e; }) (#stmtexpr)"
"${tmp}/bcir-cc" --emit-c "${C}/cfront_stmtexpr.c" > "${tmp}/sx_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bse_simple\b/se_simple_s/g' -e 's/\bse_max\b/se_max_s/g' -e 's/\bse_embed\b/se_embed_s/g' \
      -e 's/\bse_loop\b/se_loop_s/g' -e 's/\bse_nest\b/se_nest_s/g' -e 's/\bse_scope\b/se_scope_s/g' \
      -e 's/\bse_void\b/se_void_s/g' \
      "${C}/cfront_stmtexpr.c"
  cat "${tmp}/sx_emit.c"
  cat <<'DRV'
int main(void){
  for(int a=-60;a<60;a++) for(int b=-25;b<25;b++){
    if(se_simple_s(a)!=bcir_se_simple(a)){puts("simple");return 1;}
    if(se_max_s(a,b)!=bcir_se_max(a,b)){puts("max");return 1;}
    if(se_embed_s(a)!=bcir_se_embed(a)){puts("embed");return 1;}
    if(se_loop_s(b)!=bcir_se_loop(b)){puts("loop");return 1;}
    if(se_nest_s(a)!=bcir_se_nest(a)){puts("nest");return 1;}
    if(se_scope_s(a)!=bcir_se_scope(a)){puts("scope");return 1;}
    if(se_void_s(a)!=bcir_se_void(a)){puts("void");return 1;}
  }
  puts("MATCH");return 0;}
DRV
} > "${tmp}/sx_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/sx_harness.c" -o "${tmp}/sx_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/sx_harness.c" -o "${tmp}/sx_h" \
  || { echo "  FAIL: stmtexpr harness build"; exit 1; }
sxr="$("${tmp}/sx_h")"
[ "${sxr}" = "MATCH" ] \
  && echo "  PASS stmtexpr: temporary / max / embedded / loop / nested / scope / void == Clang" \
  || { echo "  FAIL: stmtexpr behaviour (${sxr})"; exit 1; }

echo "[c-runtime] ok"
