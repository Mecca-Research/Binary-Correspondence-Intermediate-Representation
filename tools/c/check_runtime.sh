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
for fx in cfront_regmap.c cfront_array.c cfront_array2d.c cfront_widerow.c cfront_deref.c cfront_callgraph.c cfront_branch.c cfront_while.c cfront_for.c cfront_dowhile.c cfront_continue.c cfront_switch.c cfront_goto.c cfront_incdec.c cfront_macros.c cfront_ppinc.c cfront_structret.c cfront_packed.c cfront_typedef.c cfront_enum.c cfront_ternary.c cfront_sizeof.c cfront_cast.c cfront_alignof.c cfront_signed.c cfront_signedcmp.c cfront_longunary.c cfront_charlit.c cfront_strtab.c cfront_strconcat.c cfront_widelit.c cfront_static.c cfront_global.c cfront_compound.c cfront_logic.c cfront_float.c cfront_floatcast.c cfront_rmw.c cfront_bitfield.c cfront_bfcompound.c cfront_union.c cfront_interleave.c cfront_funcptr.c cfront_dispatch.c cfront_integration.c cfront_regdriver.c cfront_atomic.c cfront_cmpxchg.c cfront_atomic11.c cfront_atomic_xchg.c cfront_driver.c cfront_driver_uart.c cfront_strsizeof.c cfront_strval.c cfront_hexfloat.c cfront_mathh.c cfront_mathh_mixed.c cfront_mathh_long.c cfront_mathh_ptr.c cfront_calltyped.c cfront_comments.c cfront_abi.c cfront_global_rw.c cfront_effects.c cfront_intpromote.c cfront_dispatch_table.c cfront_agginit.c cfront_restrict.c cfront_arraystore.c cfront_localarray.c cfront_shiftassign.c cfront_extern.c cfront_switchfall.c cfront_ptrarith.c cfront_threadlocal.c cfront_multidecl.c cfront_commastep.c cfront_structmulti.c cfront_memberarray.c cfront_emptystmt.c cfront_ptrstore.c cfront_loopreuse.c cfront_loopscope.c cfront_blockscope.c cfront_localmd.c cfront_nestmember.c cfront_boolnorm.c cfront_unarypromote.c cfront_floatsigncast.c cfront_intsigncast.c cfront_boolcast.c cfront_signedbf.c cfront_signedload.c cfront_enumtype.c cfront_ptrlocal.c cfront_ptrvalue.c; do  # L1-L8 + type-model + casts + char literals + interleaved decls + funcptr dispatch + §5.8 + Phase D driver + str ops + hex-float + math.h (#320-#324) + ABI data model (#abi) + scalar global r/w (#globals) + effects (#effects) + integer promotions/UAC (#intpromote) + designated init (#designated) + local aggregate init (#aggregate) + restrict (#restrict) + array stores (#astore) + local arrays (#localarr)
  c_sum="$("${tmp}/test_cfront" "${C}/${fx}" | sed -n '1p')" || { echo "  FAIL: C run ${fx}: ${c_sum}"; exit 1; }
  py_sum="$(python3 -c "
import os, re
from bcir.frontends.cfront import compile_unit
from bcir.model import Domain
src=open('${C}/${fx}').read()
inc={h: open(os.path.join('${C}',h)).read() for h in re.findall(r'#include\s+\"([^\"]+)\"', src)
     if os.path.exists(os.path.join('${C}',h))}
r=compile_unit(src, check_clang=False, includes=inc or None)
fns=r.lowered.functions; lf=fns[next(reversed(fns))]
mmio=sum(1 for c in lf.claims if c.op=='c.load' and c.domain==Domain.MMIO)
bf=sum(1 for c in lf.claims if c.op=='c.bf.get'); kn=sum(1 for c in lf.claims if c.op=='c.const')
bo=sum(1 for c in lf.claims if c.op.startswith('c.bin.')); ca=sum(1 for c in lf.claims if c.op.startswith('c.call'))
print(f'funcs={len(fns)} claims={len(lf.claims)} mmio={mmio} bf={bf} const={kn} binop={bo} call={ca} ok={1 if r.is_clean else 0}')
")" || { echo "  FAIL: python lowering ${fx}"; exit 1; }
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
abi_seen=""
for t in x86_64-linux aarch64-linux riscv64-linux x86_64-windows i386-linux; do
  c_vals="$("${tmp}/test_cfront" --target "${t}" "${C}/cfront_abi.c" | sed -n '/----EMIT----/,$p' \
            | grep -oE '= [0-9]+u;' | grep -oE '[0-9]+' | paste -sd, -)" \
    || { echo "  FAIL: C ABI run ${t}"; exit 1; }
  py_vals="$(python3 -c "
from bcir.frontends.cfront import compile_unit
src=open('${C}/cfront_abi.c').read()
r=compile_unit(src, check_clang=False, target='${t}')
lf=r.lowered.functions[next(reversed(r.lowered.functions))]
print(','.join(str(c.imm[0]) for c in lf.claims if c.op=='c.const'))
")" || { echo "  FAIL: python ABI ${t}"; exit 1; }
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
printf 'unsigned f(unsigned n){ unsigned a[n]; return a[0]; }\n'        > "${tmp}/fb/vla.c"
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
"${CC}" -std=c23 -O2 "${tmp}/dt_harness.c" -o "${tmp}/dt_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/dt_harness.c" -o "${tmp}/dt_h" \
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
"${CC}" -std=c23 -O2 "${tmp}/lmd_harness.c" -o "${tmp}/lmd_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/lmd_harness.c" -o "${tmp}/lmd_h" \
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
"${CC}" -std=c23 -O2 "${tmp}/bn_harness.c" -o "${tmp}/bn_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/bn_harness.c" -o "${tmp}/bn_h" \
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
"${CC}" -std=c23 -O2 "${tmp}/sl_harness.c" -o "${tmp}/sl_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/sl_harness.c" -o "${tmp}/sl_h" \
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

echo "[c-runtime] ok"
