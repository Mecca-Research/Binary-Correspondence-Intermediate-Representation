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
for fx in cfront_regmap.c cfront_array.c cfront_array2d.c cfront_widerow.c cfront_deref.c cfront_callgraph.c cfront_branch.c cfront_while.c cfront_for.c cfront_dowhile.c cfront_continue.c cfront_switch.c cfront_goto.c cfront_incdec.c cfront_macros.c cfront_ppinc.c cfront_structret.c cfront_packed.c cfront_typedef.c cfront_enum.c cfront_ternary.c cfront_sizeof.c cfront_cast.c cfront_alignof.c cfront_charlit.c cfront_strtab.c cfront_strconcat.c cfront_widelit.c cfront_static.c cfront_global.c cfront_compound.c cfront_logic.c cfront_float.c cfront_floatcast.c cfront_rmw.c cfront_bitfield.c cfront_bfcompound.c cfront_union.c cfront_interleave.c cfront_funcptr.c cfront_dispatch.c cfront_integration.c cfront_regdriver.c cfront_atomic.c cfront_cmpxchg.c cfront_atomic11.c cfront_atomic_xchg.c cfront_driver.c cfront_driver_uart.c cfront_strsizeof.c cfront_strval.c cfront_hexfloat.c cfront_mathh.c cfront_mathh_mixed.c cfront_mathh_long.c cfront_mathh_ptr.c cfront_calltyped.c cfront_comments.c cfront_abi.c cfront_global_rw.c cfront_effects.c; do  # L1-L8 + type-model + casts + char literals + interleaved decls + funcptr dispatch + §5.8 + Phase D driver + str ops + hex-float + math.h (#320-#324) + ABI data model (#abi) + scalar global r/w (#globals) + effects (#effects)
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

echo "[c-runtime] ok"
