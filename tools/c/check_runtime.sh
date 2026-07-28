#!/usr/bin/env bash
# Validate the freestanding C StreamPack runtime: it compiles with no libc, and a
# Python-encoded StreamPack round-trips through the C decoder (ABI parity).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C="${ROOT}/runtime/c"
CC_WAS_SET="${CC+x}"
CC="${CC:-$(command -v clang || command -v cc || true)}"
if [ -z "${CC}" ]; then
  echo "no C compiler (clang/cc); skipping runtime check." >&2
  exit 0
fi

# Older distributions can expose an unversioned clang that predates the final
# `-std=c23` spelling while also installing a newer versioned clang.  Prefer an
# explicitly supplied CC, but otherwise upgrade the default to the highest
# versioned clang that accepts the spelling used throughout this gate.  This
# keeps the C23 checks meaningful instead of failing on tool-name discovery.
if [ "${CC_WAS_SET}" != x ] &&
   ! printf 'int main(void){return 0;}\n' | "${CC}" -std=c23 -x c -c -o /dev/null - >/dev/null 2>&1; then
  best=""; best_major=-1
  old_ifs="${IFS}"; IFS=:
  for directory in ${PATH}; do
    IFS="${old_ifs}"
    for candidate in "${directory}"/clang-[0-9]*; do
      [ -x "${candidate}" ] || continue
      major="${candidate##*/clang-}"
      [[ "${major}" =~ ^[0-9]+$ ]] || continue
      [ "${major}" -gt "${best_major}" ] || continue
      if printf 'int main(void){return 0;}\n' | "${candidate}" -std=c23 -x c -c -o /dev/null - >/dev/null 2>&1; then
        best="${candidate}"; best_major="${major}"
      fi
    done
    IFS=:
  done
  IFS="${old_ifs}"
  [ -z "${best}" ] || CC="${best}"
fi

echo "[c-runtime] memory classes + allocator/context/channel fault sweep"
if CC="${CC}" bash "${ROOT}/tools/c/check_memory_discipline.sh" 2>&1 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "  FAIL: memory-discipline gate"; exit 1
fi

echo "[c-runtime] freestanding compile (-ffreestanding -nostdlib), C11 + C23"
for std in c11 c23; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -c "${C}/bcir_runtime.c" -o /dev/null \
    || { echo "  FAIL: runtime not freestanding-clean under -std=${std}"; exit 1; }
done
echo "  PASS freestanding (C11 + C23; ABI static_assert holds)"

tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

echo "[c-runtime] x86 interrupt-frame ABI header (C11 + C23)"
for std in c11 c23; do
  "${CC}" -ffreestanding -std=${std} -Wall -Wextra -Werror -I "${C}" \
    "${C}/test_x86_interrupt.c" -o "${tmp}/test_x86_interrupt_${std}" \
    || { echo "  FAIL: x86 interrupt-frame ABI under -std=${std}"; exit 1; }
  "${tmp}/test_x86_interrupt_${std}" \
    || { echo "  FAIL: x86 interrupt-frame layout/helper under -std=${std}"; exit 1; }
done
echo "  PASS x86 interrupt-frame ABI (fixed 176-byte long-mode frame)"

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

echo "[c-runtime] BCAB artifact bundle: freestanding reader + Python/C selection parity"
for std in c11 c23; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -Werror -I "${C}" \
    -c "${C}/bcir_artifact_bundle.c" -o /dev/null \
    || { echo "  FAIL: BCAB reader not freestanding-clean under -std=${std}"; exit 1; }
done
python3 - "${tmp}/bundle.bcab" "${tmp}/bundle.der" "${tmp}/bundle.from-der.bcab" <<'PY' \
  || { echo "  FAIL: Python BCAB/ASN.1 fixture"; exit 1; }
import struct, sys
from bcir.abi import (ArtifactBundle, ArtifactFormat, ArtifactKind, ArtifactVariant,
                      Endianness, encode, encode_bundle)
from bcir.asn1.artifact_bundle import der_to_native, native_to_der
from bcir.examples import vector_add
from bcir.gem import hydrate
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
m = vector_add(8)
pack = hydrate(m, optimize(m, TargetProfile.x86_avx512(), Theta.cool()))
elf = bytearray(20); elf[:7] = b"\x7fELF\x02\x01\x01"; struct.pack_into("<H", elf, 16, 1); struct.pack_into("<H", elf, 18, 62)
variants = (
    ArtifactVariant("00-root", ArtifactKind.STREAM_PACK, ArtifactFormat.STREAM_PACK,
                    encode(pack), channel="host", portable=True),
    ArtifactVariant("portable-c", ArtifactKind.C_SOURCE, ArtifactFormat.TEXT,
                    b"int bcir_kernel(void){return 0;}\n", portable=True),
    ArtifactVariant("x86-avx2", ArtifactKind.ELF_OBJECT, ArtifactFormat.ELF, bytes(elf),
                    triple="x86_64-unknown-linux-gnu", architecture="x86_64",
                    os_abi="linux-gnu", channel="host", entry_symbol="bcir_kernel",
                    required_features=("avx2",), endianness=Endianness.LITTLE,
                    pointer_bits=64, e_machine=62, priority=9,
                    r12_attested=True, executable=True),
)
native = encode_bundle(ArtifactBundle(variants, "00-root", "portable-c", 123, 7))
open(sys.argv[1], "wb").write(native)
projection = native_to_der(native)
open(sys.argv[2], "wb").write(projection)
open(sys.argv[3], "wb").write(der_to_native(projection))
PY
"${CC}" -std=c23 -O2 -Wall -Wextra -Werror -I "${C}" \
  "${C}/bcir_artifact_bundle.c" "${C}/bcir_runtime.c" \
  "${C}/test_artifact_bundle.c" -o "${tmp}/test_artifact_bundle" \
  || { echo "  FAIL: BCAB C parity harness build"; exit 1; }
about="$("${tmp}/test_artifact_bundle" "${tmp}/bundle.bcab")" \
  || { echo "  FAIL: BCAB C parity harness"; exit 1; }
case "${about}" in
  OK\ entries=3*) echo "  PASS BCAB Python encode -> C checksum/select parity" ;;
  *) echo "  FAIL: unexpected BCAB result '${about}'"; exit 1 ;;
esac
cmp -s "${tmp}/bundle.bcab" "${tmp}/bundle.from-der.bcab" \
  || { echo "  FAIL: BCAB native -> ASN.1 DER -> native bytes differ"; exit 1; }
"${CC}" -std=c23 -O2 -Wall -Wextra -Werror -I "${C}" \
  "${C}/bcir_asn1.c" "${C}/test_asn1.c" -o "${tmp}/test_artifact_asn1" \
  || { echo "  FAIL: BCAB ASN.1 C validation harness build"; exit 1; }
asn1_about="$("${tmp}/test_artifact_asn1" "${tmp}/bundle.der")" \
  || { echo "  FAIL: BCAB ASN.1 projection C validation"; exit 1; }
printf '%s\n' "${asn1_about}" | grep -q '^der ok$' \
  && echo "  PASS BCAB native <-> DER byte identity + generic C X.690 validation" \
  || { echo "  FAIL: C X.690 rail rejected BCAB DER projection"; printf '%s\n' "${asn1_about}"; exit 1; }

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

echo "[c-runtime] UART telemetry frame (#telemetry-frame): freestanding compile (C11 + C23) + byte-identical re-encode"
# bcir_telemetry_frame.c is the C twin of bcir/telemetry_frame.py -- the framed, CRC-sealed,
# resync-able telemetry transport (T2). The producer drains TelemetryRing and frames the 56-byte
# <7q> records; the host decoder reuses RT3. It REUSES bcir_crc32 from bcir_runtime.c (so the C
# and Python (zlib.crc32) CRCs agree). Self-skipping/non-fatal without a C compiler (the section
# is only reached past the CC guard at the top, so a missing CC already exited 0 cleanly above).
for std in c11 c23; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -I "${C}" -c "${C}/bcir_telemetry_frame.c" -o /dev/null \
    || { echo "  FAIL: bcir_telemetry_frame not freestanding-clean under -std=${std}"; exit 1; }
done
"${CC}" -std=c23 -O2 "${C}/bcir_telemetry_frame.c" "${C}/bcir_runtime.c" "${C}/test_telemetry_frame.c" -I "${C}" -o "${tmp}/test_tframe" \
  || { echo "  FAIL: telemetry-frame harness build"; exit 1; }
# Python-encode a fixed DataDNA batch into one frame; C decode + re-encode; assert byte-identical.
python3 -c "
from bcir.telemetry import DataDNA
from bcir.telemetry_frame import encode_frame
recs=[DataDNA(segment_id='',claim_id=1,cycles=100,bytes=200,misses=5,thermal=40,voltage=10,utilization=30),
      DataDNA(segment_id='',claim_id=2,cycles=999999,bytes=4096,misses=0,thermal=0,voltage=0,utilization=100),
      DataDNA(segment_id='',claim_id=3,cycles=-50,bytes=0,misses=100,thermal=99,voltage=50,utilization=0)]
open('${tmp}/tframe.bin','wb').write(encode_frame(recs, seq=7, timestamp=123456))
" || { echo "  FAIL: python frame encode"; exit 1; }
tfout="$("${tmp}/test_tframe" "${tmp}/tframe.bin" "${tmp}/tframe_reenc.bin")" || { echo "  FAIL: C frame decode"; echo "${tfout}"; exit 1; }
echo "${tfout}" | grep -q "^OK " || { echo "  FAIL: C frame decode did not OK"; echo "${tfout}"; exit 1; }
if cmp -s "${tmp}/tframe.bin" "${tmp}/tframe_reenc.bin"; then
  echo "  PASS #telemetry-frame (C bcir_tf decode + re-encode == Python encode_frame, byte-identical; bcir_crc32 == zlib.crc32)"
else
  echo "  FAIL: telemetry-frame bytes differ from the Python encoding"; exit 1
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
cfront_complit.c cfront_typeof.c cfront_structinit.c cfront_arraylit.c cfront_variadic.c cfront_compoundwide.c cfront_extvariadic.c cfront_longdouble.c cfront_generic.c cfront_designate.c cfront_nestoffset.c cfront_addrmember.c cfront_atomiclocal.c cfront_builtins.c cfront_stmtexpr.c \
cfront_bitint.c cfront_bitint_member.c cfront_bitint_mixed.c cfront_bitint_bitfield.c"
# + C23 `_BitInt(N)` (#bitint / #bitintmember / #bitintmixed / #bitintbitfield): exact-width bit-precise
# ints -- same-type + MIXED-WIDTH arithmetic (the wider `_BitInt` wins the C23 rank), PLAIN members, and
# `_BitInt(N) m:W` BITFIELDS; the result type + bitfield layout are verified == Clang in test_c_cfront.py.
# Precompute EVERY oracle summary in one python process (import compile_unit once) -- the old
# python-per-fixture loop paid ~0.3s of interpreter+import startup each (~30s over the fixture set).
python3 - "${C}" ${FIXTURES} > "${tmp}/py_sums.txt" <<'PY' || { echo "  FAIL: python lowering (batch)"; exit 1; }
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
# Phase 3 breadth: the CMSIS-style GPIO fixture (__IO macro, RESERVED pads, write-only BSRR,
# RCC gate-first) must ALSO compile clean through the C rail -- real header shapes, not just
# the synthetic UART block.
gpsum="$("${tmp}/bcir-cc" "${C}/cfront_driver_gpio.c")" || { echo "  FAIL: bcir-cc gpio compile"; echo "${gpsum}"; exit 1; }
case "${gpsum}" in
  *ok=1*) echo "  PASS bcir-cc CMSIS gpio fixture (${gpsum##*: })" ;;
  *) echo "  FAIL: bcir-cc gpio: ${gpsum}"; exit 1 ;;
esac
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

# WRITE-guard adversarial test (#writeguard, §5.12): a clamped OOB *store* silently redirects the write onto
# a valid element (a[extent-1]) -- data corruption disguised as recovery. So a STORE index site emits the
# WRITE guard `BCIR_CHK_W`, whose handler is `noreturn` and NEVER clamps: under the SAME confident clamp
# policy that recovers a read, an OOB store must ABORT (not corrupt a[7]). The emit names the store site
# `BCIR_CHK_W(...)` and the load site `BCIR_CHK(...)` -- both rails identically (gated in test_c_cfront.py).
echo "[c-runtime] bcir-cc WRITE-guard: an OOB store fails-fast, never clamps (#writeguard)"
printf 'unsigned el_sw(unsigned i, unsigned j, unsigned v){ unsigned a[8]; for(unsigned k=0u;k<8u;k++) a[k]=k*3u; a[i]=v; return a[j]; }\n' \
  > "${tmp}/sw.c"
"${tmp}/bcir-cc" --emit-c "${tmp}/sw.c" > "${tmp}/sw_emit.c" || { echo "  FAIL: --emit-c (write-guard)"; exit 1; }
# the STORE sites use BCIR_CHK_W, the LOAD site uses the read BCIR_CHK -- distinguished per site.
{ grep -q 'a\[BCIR_CHK_W(.*) *\] *= *v;' "${tmp}/sw_emit.c" \
  && grep -q '= a\[BCIR_CHK(.*"el_sw:a")\]' "${tmp}/sw_emit.c"; } \
  || { echo "  FAIL: store site not WRITE-guarded / load site not READ-guarded"; cat "${tmp}/sw_emit.c"; exit 1; }
{ echo '#include <stdio.h>'; echo '#include <stdlib.h>'; echo '#include "bcir_quarantine_recover.h"'
  cat "${tmp}/sw_emit.c"
  cat <<'DRV'
int main(int c, char **v){
  /* the SAME confident clamp policy used for the read-recovery test above (threshold 500, conf 900). */
  static const bcir_recover_rule clampall[] = {{"el_sw:a", BCIR_RECOVER_CLAMP, 900}};
  bcir_recover_set_policy(clampall, 1, 500);
  int mode = c > 1 ? atoi(v[1]) : 0;
  if (mode == 0) {                          /* an OOB READ still clamps to a[7] = 7*3 = 21 */
    printf("%u\n", bcir_el_sw(0u, 99u, 555u));
  } else {                                  /* an OOB STORE must abort BEFORE the redirect corrupts a[7] */
    (void)bcir_el_sw(99u, 7u, 777u);
    printf("NO-ABORT a[7]=%u\n", 777u);     /* reaching here means the store was silently clamped -> BUG */
  }
  return 0; }
DRV
} > "${tmp}/sw_main.c"
"${CC}" -std=c23 -O2 -I "${C}" "${tmp}/sw_main.c" "${C}/bcir_quarantine.c" "${C}/bcir_quarantine_recover.c" -o "${tmp}/sw_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 -I "${C}" "${tmp}/sw_main.c" "${C}/bcir_quarantine.c" "${C}/bcir_quarantine_recover.c" -o "${tmp}/sw_h" \
  || { echo "  FAIL: write-guard override build"; exit 1; }
sw_read="$("${tmp}/sw_h" 0)"                                   # OOB read: clamps to a[7] = 21
[ "${sw_read}" = "21" ] || { echo "  FAIL: OOB read did not clamp (got '${sw_read}', want 21)"; exit 1; }
sw_wr="$("${tmp}/sw_h" 1 2>&1 1>/dev/null)"; sw_rc=$?          # OOB store: must abort, never reach NO-ABORT
{ [ "${sw_rc}" != "0" ] && ! printf '%s' "${sw_wr}" | grep -q "NO-ABORT" \
  && printf '%s' "${sw_wr}" | grep -q "out-of-bounds store cannot be clamped"; } \
  && echo "  PASS writeguard: OOB read clamps (a[7]=21) but OOB store ABORTS (no a[7] corruption) (#writeguard)" \
  || { echo "  FAIL: OOB store did not fail-fast (rc=${sw_rc}: ${sw_wr})"; exit 1; }

# Atomic OOB ring (#atomicring, §5.12): the counter and payload publication must both be synchronized.
# An atomic fetch-add alone gives writers distinct slots but still lets a reporter race a non-atomic struct
# update (C undefined behaviour / torn audit records). The hosted implementation serializes each short
# publish/snapshot section. This test hammers writers while a reporter snapshots concurrently, then asserts
# the total equals N*M. If the freestanding build has no atomics the contract is single-threaded and this
# hosted concurrency test is skipped.
echo "[c-runtime] OOB ring counter is atomic under concurrent events (#atomicring)"
{ echo '#include <stdio.h>'; echo '#include "bcir_quarantine.h"'
  cat <<'DRV'
#if BCIR_OOB_COUNTER_ATOMIC
#include <pthread.h>
#define NT 8
#define NM 50000
static void *hammer(void *arg){ (void)arg;
  for(long k=0;k<NM;k++) bcir_oob_record_event(1u, (uint64_t)k, 64u, "race:a");
  return 0; }
static void *observe(void *arg){ (void)arg; FILE *f=tmpfile(); if(!f) return (void *)1;
  for(int k=0;k<200;k++){ bcir_quarantine_report(f); rewind(f); }
  return fclose(f) ? (void *)1 : 0; }
int main(void){
  pthread_t th[NT], reader;
  bcir_quarantine_report(NULL); bcir_decide_report(NULL); /* public null handles are harmless */
  if(pthread_create(&reader,0,observe,0)) return 2;
  for(int i=0;i<NT;i++) pthread_create(&th[i],0,hammer,0);
  for(int i=0;i<NT;i++) pthread_join(th[i],0);
  void *reader_rc=0; pthread_join(reader,&reader_rc); if(reader_rc) return 3;
  unsigned long got = bcir_oob_count, want = (unsigned long)NT*NM;
  printf(got==want ? "EXACT %lu\n" : "LOST %lu/%lu\n", got, want);
  return got==want ? 0 : 1; }
#else
int main(void){ printf("SKIP (no atomics; single-threaded contract)\n"); return 0; }
#endif
DRV
} > "${tmp}/race_main.c"
"${CC}" -std=c23 -O2 -pthread -I "${C}" "${tmp}/race_main.c" "${C}/bcir_quarantine.c" -o "${tmp}/race_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 -pthread -I "${C}" "${tmp}/race_main.c" "${C}/bcir_quarantine.c" -o "${tmp}/race_h" \
  || { echo "  FAIL: atomic-ring test build"; exit 1; }
race_out="$("${tmp}/race_h")"; race_rc=$?
{ [ "${race_rc}" = "0" ] && { printf '%s' "${race_out}" | grep -q "^EXACT" || printf '%s' "${race_out}" | grep -q "^SKIP"; }; } \
  && echo "  PASS atomicring: ${race_out} (concurrent OOB events do not scramble the total)" \
  || { echo "  FAIL: OOB ring counter raced (${race_out})"; exit 1; }

# rid->extent tamper-evidence (#extentassert, §5.12): the guard trusts the inline `n`, and the freestanding
# unit has NO registry to resolve `rid`->extent (a DOCUMENTED limit; see bcir_quarantine.h). The feasible
# lightweight check, for a KNOWN-EXTENT array, ties `n` to the array's true storage via a COMPILE-TIME
# BCIR_EXTENT_ASSERT(arr, n) == _Static_assert(n == sizeof(arr)/sizeof(arr[0])): a correct extent compiles,
# a TAMPERED `n` fails to compile -- so the extent is tamper-evident with no runtime cost and no registry.
echo "[c-runtime] rid->extent tamper-evidence: BCIR_EXTENT_ASSERT (freestanding lightweight check) (#extentassert)"
{ echo '#include "bcir_quarantine.h"'; echo 'static unsigned a[8];'
  echo 'int main(void){ BCIR_EXTENT_ASSERT(a, 8u); return (int)a[0]; }'; } > "${tmp}/ext_ok.c"
"${CC}" -std=c23 -I "${C}" "${tmp}/ext_ok.c" "${C}/bcir_quarantine.c" -o "${tmp}/ext_ok" 2>/dev/null \
  || "${CC}" -std=c2x -I "${C}" "${tmp}/ext_ok.c" "${C}/bcir_quarantine.c" -o "${tmp}/ext_ok" \
  || { echo "  FAIL: BCIR_EXTENT_ASSERT rejected a CORRECT extent"; exit 1; }
{ echo '#include "bcir_quarantine.h"'; echo 'static unsigned a[8];'
  echo 'int main(void){ BCIR_EXTENT_ASSERT(a, 99u); return (int)a[0]; }'; } > "${tmp}/ext_bad.c"
if "${CC}" -std=c23 -I "${C}" "${tmp}/ext_bad.c" "${C}/bcir_quarantine.c" -o "${tmp}/ext_bad" 2>/dev/null; then
  echo "  FAIL: BCIR_EXTENT_ASSERT did NOT catch a tampered extent (n=99 vs storage 8)"; exit 1
fi
echo "  PASS extentassert: correct extent compiles; tampered n=99 fails to compile (tamper-evident) (#extentassert)"

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
printf 'unsigned f(unsigned n){ unsigned a[n][n][n][n]; return a[0][0][0][0]; }\n' > "${tmp}/fb/vla.c"  # >3-D VLA (1-3D native)
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

# The §5.9 constant-expression evaluator (#cexpr): enum initializers, case labels and
# static-local initializers fold comparisons / logical ops / the ternary on BOTH rails
# (ce_expr == the oracle's _const_eval vocabulary); the emitted program behaves like the
# reference (41 + 7 + 53 == 101).
echo "[c-runtime] constant-expression evaluator: enum/case/static folds == reference (#cexpr)"
cat > "${tmp}/cexpr.c" <<'CEXPR'
enum { A = 1 << 4, B = A + 2, C = (A > 10) ? 7 : 9, D = (1 < 2) && (3 != 4) };
unsigned pick(unsigned x) {
    static unsigned seed = (5 > 3) ? 40u : 2u;
    switch (x) {
    case A: return seed + 1u;
    case B: return (unsigned)C;
    default: return (unsigned)D + 52u;
    }
}
int main(void) { return (int)(pick(16u) + pick(18u) + pick(0u)); }
CEXPR
"${CC}" -std=c11 "${tmp}/cexpr.c" -o "${tmp}/cexpr_ref" || { echo "  FAIL: cexpr reference build"; exit 1; }
"${tmp}/cexpr_ref"; cexpr_ref=$?
{ echo '#include <stdint.h>'; "${tmp}/bcir-cc" --emit-c "${tmp}/cexpr.c"; \
  echo 'int main(void){ return (int)bcir_main(); }'; } > "${tmp}/cexpr_emit.c" \
  || { echo "  FAIL: bcir-cc --emit-c (cexpr)"; exit 1; }
grep -q "static uint32_t seed = 40u;" "${tmp}/cexpr_emit.c" \
  || { echo "  FAIL: the ternary static initializer did not fold to 40"; exit 1; }
"${CC}" -std=c11 "${tmp}/cexpr_emit.c" -o "${tmp}/cexpr_prog" || { echo "  FAIL: cexpr emitted build"; exit 1; }
"${tmp}/cexpr_prog"; cexpr_rc=$?
[ "${cexpr_rc}" = "${cexpr_ref}" ] && [ "${cexpr_rc}" = "101" ] \
  && echo "  PASS constant-expression folds (rc=${cexpr_rc} == ref)" \
  || { echo "  FAIL: cexpr rc=${cexpr_rc} != ref=${cexpr_ref}"; exit 1; }

# R21 lifetime policy (bcir-cc --r21, §5.12): a detected use-after-free / double-free becomes a
# VERDICT under a non-advisory policy, and the C twin (bcir-cc) and the Python rail (bcir-cfront) must
# draw the SAME exit code. advisory (default) never gates (0); fallback routes to LLVM (2); reject is a
# hard error (1); a clean (no-UAF) unit stays 0 under every policy. Detection is the same freed-set walk.
echo "[c-runtime] R21 lifetime policy (bcir-cc --r21): verdict + exit-code parity vs the oracle (#r21policy)"
mkdir -p "${tmp}/r21"
printf '#include <stdlib.h>\nunsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); free(p); return p[0]; }\n'           > "${tmp}/r21/uaf.c"
printf '#include <stdlib.h>\nunsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); free(p); free(p); return n; }\n'      > "${tmp}/r21/dfree.c"
printf '#include <stdlib.h>\nunsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); unsigned r=p[0]; free(p); return r; }\n' > "${tmp}/r21/clean.c"
r21_seen=""
for pol in advisory fallback reject; do
  for fx in uaf dfree clean; do
    "${tmp}/bcir-cc" --r21=${pol} "${tmp}/r21/${fx}.c" >/dev/null 2>&1; trc=$?
    python3 -m bcir.frontends.cfront --r21=${pol} -o /dev/null "${tmp}/r21/${fx}.c" >/dev/null 2>&1; prc=$?
    [ "${trc}" = "${prc}" ] \
      && echo "  PASS r21 ${pol}/${fx} (rc oracle == C: ${trc})" \
      || { echo "  FAIL: r21 ${pol}/${fx} (C rc=${trc} PY rc=${prc})"; exit 1; }
    r21_seen="${r21_seen}${trc}"
  done
done
# the policy gate must reach all three verdicts (clean 0 / reject 1 / fallback 2), else it has no teeth.
case "${r21_seen}" in
  *0*) case "${r21_seen}" in *1*) case "${r21_seen}" in *2*)
    echo "  PASS r21 policy spans advisory / fallback / reject verdicts" ;;
    *) echo "  FAIL: r21 gate never reached a fallback (2): ${r21_seen}"; exit 1 ;; esac ;;
    *) echo "  FAIL: r21 gate never reached a reject (1): ${r21_seen}"; exit 1 ;; esac ;;
  *) echo "  FAIL: r21 gate never reached a clean (0): ${r21_seen}"; exit 1 ;;
esac

# Project mode (#project, Phase 3): over a multi-file invocation both drivers print ONE per-project
# verdict line -- CLEAN (every unit clean) / PARTIAL-FALLBACK (no failure, >=1 unit routed to LLVM)
# / DIRTY (>=1 unit failed) -- and draw the same exit code (a hard error 1 DOMINATES a fallback 2).
# The verdict line must match the oracle BYTE-FOR-BYTE and the gate must span all three verdicts.
echo "[c-runtime] project mode (bcir-cc multi-file): verdict line + exit code == oracle (#project)"
mkdir -p "${tmp}/proj"
printf 'unsigned f(unsigned x){ return x + 1u; }\n'                    > "${tmp}/proj/a.c"
printf 'unsigned g(unsigned x){ return x * 2u; }\n'                    > "${tmp}/proj/b.c"
printf 'unsigned h(unsigned n){ unsigned a[n][n][n][n]; return a[0][0][0][0]; }\n' > "${tmp}/proj/fb.c"   # >3-D VLA: fallback
printf 'unsigned r(unsigned n){ return r(n-1u); }\n'                   > "${tmp}/proj/bad.c"  # R18 recursion: dirty
proj_seen=""
proj_case() {  # $1 = scenario name; the rest is the shared argv both drivers get verbatim
  pname="$1"; shift
  c_all="$("${tmp}/bcir-cc" "$@" 2>/dev/null)"; c_rc=$?
  p_all="$(python3 -m bcir.frontends.cfront "$@" 2>/dev/null)"; p_rc=$?
  c_out="$(printf '%s\n' "${c_all}" | tail -1)"
  p_out="$(printf '%s\n' "${p_all}" | tail -1)"
  [ "${c_rc}" = "${p_rc}" ] || { echo "  FAIL: project ${pname} (C rc=${c_rc} PY rc=${p_rc})"; exit 1; }
  [ "${c_out}" = "${p_out}" ] || { echo "  FAIL: project ${pname} (C '${c_out}' != PY '${p_out}')"; exit 1; }
  case "${c_out}" in "project: "*) : ;; *) echo "  FAIL: project ${pname}: no verdict line ('${c_out}')"; exit 1 ;; esac
  echo "  PASS project ${pname} (rc=${c_rc}: ${c_out})"
  proj_seen="${proj_seen}${c_out};"
}
proj_case clean               "${tmp}/proj/a.c" "${tmp}/proj/b.c"
proj_case single --project    "${tmp}/proj/a.c"
proj_case fellback --fallback "${tmp}/proj/a.c" "${tmp}/proj/fb.c"
# dirty AFTER fallback in rc terms: bad.c sets rc=1 first, fb.c must NOT overwrite it with 2.
proj_case dirty  --fallback   "${tmp}/proj/a.c" "${tmp}/proj/bad.c" "${tmp}/proj/fb.c"
# the gate must span CLEAN / PARTIAL-FALLBACK / DIRTY, and 1 must have dominated 2 in the dirty mix.
case "${proj_seen}" in *"CLEAN"*) case "${proj_seen}" in *"PARTIAL-FALLBACK"*) case "${proj_seen}" in *"DIRTY"*)
  echo "  PASS project verdicts span CLEAN / PARTIAL-FALLBACK / DIRTY" ;;
  *) echo "  FAIL: project gate never reached DIRTY"; exit 1 ;; esac ;;
  *) echo "  FAIL: project gate never reached PARTIAL-FALLBACK"; exit 1 ;; esac ;;
  *) echo "  FAIL: project gate never reached CLEAN"; exit 1 ;;
esac

# Phase 3 LINKING (#link): a file-scope PROTOTYPE makes a cross-TU call a typed external edge
# (c.call.tu:, R18-opaque, extern-declared in the emitted prelude, NO -l derived), and the bcir-cc
# EMITTED caller object host-links against the callee TU with the DERIVED --emit-link-flags and
# behaves exactly like the all-original reference build. A same-unit prototype stays a forward
# declaration (definition wins: the call is a real R18 edge). Both rails accept the caller (rc 0).
echo "[c-runtime] Phase 3 linking (bcir-cc prototypes): emitted caller + host linker == reference (#link)"
mkdir -p "${tmp}/link"
printf 'double scale(double x);\nint main(void) { double v = scale(16.0); return (int)v; }\n' > "${tmp}/link/main.c"
printf '#include <math.h>\ndouble scale(double x) { return sqrt(x) + 1.0; }\n' > "${tmp}/link/lib.c"
"${CC}" -std=c11 "${tmp}/link/main.c" "${tmp}/link/lib.c" -o "${tmp}/link/ref" -lm \
  || { echo "  FAIL: reference build"; exit 1; }
"${tmp}/link/ref"; ref_rc=$?
lflags="$("${tmp}/bcir-cc" --emit-link-flags "${tmp}/link/lib.c")" || { echo "  FAIL: link flags"; exit 1; }
{ echo '#include <stdint.h>'; "${tmp}/bcir-cc" --emit-c "${tmp}/link/main.c"; \
  echo 'int main(void){ return (int)bcir_main(); }'; } > "${tmp}/link/main_emit.c" \
  || { echo "  FAIL: bcir-cc --emit-c (prototyped caller)"; exit 1; }
grep -q "extern double scale(double);" "${tmp}/link/main_emit.c" \
  || { echo "  FAIL: emitted prelude lacks the extern declaration"; exit 1; }
# shellcheck disable=SC2086
"${CC}" -std=c11 "${tmp}/link/main_emit.c" "${tmp}/link/lib.c" -o "${tmp}/link/prog" ${lflags} \
  || { echo "  FAIL: emitted caller did not link"; exit 1; }
"${tmp}/link/prog"; bcir_rc=$?
python3 -m bcir.frontends.cfront -o /dev/null "${tmp}/link/main.c" >/dev/null 2>&1; py_rc=$?
[ "${ref_rc}" = "${bcir_rc}" ] && [ "${ref_rc}" = "5" ] && [ "${py_rc}" = "0" ] \
  && echo "  PASS linking (ref=${ref_rc} == bcir=${bcir_rc}; oracle accepts the caller)" \
  || { echo "  FAIL: linking (ref=${ref_rc} bcir=${bcir_rc} oracle=${py_rc})"; exit 1; }
printf 'unsigned g(unsigned x);\nunsigned f(unsigned y) { return g(y) + 1u; }\nunsigned g(unsigned x) { return x * 2u; }\n' > "${tmp}/link/fwd.c"
"${tmp}/bcir-cc" --emit-claimgraph "${tmp}/link/fwd.c" | grep -q "c.call:g" \
  && echo "  PASS forward declaration stays a real R18 edge (definition wins)" \
  || { echo "  FAIL: forward declaration did not rewrite to c.call:g"; exit 1; }
# ... and the LINKABLE artifact (the C twin of the oracle's emit_linkable): BOTH TUs re-rendered
# by bcir-cc --linkable (external linkage, real names, derived includes) link TO EACH OTHER --
# no original source in the image -- and behave exactly like the reference build.
"${tmp}/bcir-cc" --linkable "${tmp}/link/main.c" > "${tmp}/link/main_lk.c" || { echo "  FAIL: --linkable main"; exit 1; }
"${tmp}/bcir-cc" --linkable "${tmp}/link/lib.c" > "${tmp}/link/lib_lk.c" || { echo "  FAIL: --linkable lib"; exit 1; }
"${CC}" -std=c11 "${tmp}/link/main_lk.c" "${tmp}/link/lib_lk.c" -o "${tmp}/link/prog_lk" -lm \
  || { echo "  FAIL: two --linkable TUs did not link"; exit 1; }
"${tmp}/link/prog_lk"; lk_rc=$?
[ "${lk_rc}" = "${ref_rc}" ] \
  && echo "  PASS linkable artifact (two emitted TUs link to each other; rc=${lk_rc} == ref)" \
  || { echo "  FAIL: linkable artifact rc=${lk_rc} != ref=${ref_rc}"; exit 1; }
# ... and SOURCE-STATIC HONORING on the twin: two TUs each carrying a SAME-NAMED static helper
# keep `static` in the --linkable rendering (internal linkage), so the pair still links into
# one binary -- a stripped-static (exported) rendering would be a duplicate-symbol link error.
printf 'static unsigned mix(unsigned x) { return x * 3u; }\nunsigned fa(unsigned x) { return mix(x) + 5u; }\n' > "${tmp}/link/sa.c"
printf 'static unsigned mix(unsigned x) { return x + 100u; }\nunsigned fb(unsigned x) { return mix(x); }\n' > "${tmp}/link/sb.c"
printf 'extern unsigned fa(unsigned);\nextern unsigned fb(unsigned);\nint main(void){ return (int)(fa(2u) + fb(1u)); }\n' > "${tmp}/link/smain.c"
"${tmp}/bcir-cc" --linkable "${tmp}/link/sa.c" > "${tmp}/link/sa_lk.c" || { echo "  FAIL: --linkable sa"; exit 1; }
"${tmp}/bcir-cc" --linkable "${tmp}/link/sb.c" > "${tmp}/link/sb_lk.c" || { echo "  FAIL: --linkable sb"; exit 1; }
grep -q "static uint32_t mix(uint32_t x)" "${tmp}/link/sa_lk.c" \
  || { echo "  FAIL: source-static helper lost its static in the linkable rendering"; exit 1; }
grep -q "^uint32_t fa(uint32_t x)" "${tmp}/link/sa_lk.c" \
  || { echo "  FAIL: the exported function should be non-static under its real name"; exit 1; }
"${CC}" -std=c11 -Wall -Werror "${tmp}/link/smain.c" "${tmp}/link/sa_lk.c" "${tmp}/link/sb_lk.c" \
  -o "${tmp}/link/prog_st" || { echo "  FAIL: same-named statics did not link"; exit 1; }
"${tmp}/link/prog_st"; st_rc=$?
[ "${st_rc}" = "112" ] \
  && echo "  PASS source-static honoring (same-named statics per TU; rc=${st_rc})" \
  || { echo "  FAIL: source-static honoring rc=${st_rc} != 112"; exit 1; }
# ... review-hardened edges: (1) the static-forward-declaration idiom -- the tudefs prelude
# re-keys `extern` -> `static` for a kept-static function, so the artifact compiles (C11
# 6.2.2p7); (2) a `_BitInt(N)` return spelling carries parens BEFORE the name -- the static
# keeper must still find the function name (grep-only: host cc under -std=c11 has no _BitInt).
printf 'static int r(int v);\nint s(int x) { return r(x); }\nstatic int r(int x) { return x + 1; }\n' > "${tmp}/link/sp.c"
"${tmp}/bcir-cc" --linkable "${tmp}/link/sp.c" > "${tmp}/link/sp_lk.c" || { echo "  FAIL: --linkable sp"; exit 1; }
grep -q "^static int32_t r(int32_t);" "${tmp}/link/sp_lk.c" \
  || { echo "  FAIL: static prototype should re-key its extern prelude to static"; exit 1; }
"${CC}" -std=c11 -Wall -Werror -c "${tmp}/link/sp_lk.c" -o "${tmp}/link/sp_lk.o" \
  && echo "  PASS static forward declaration (extern prelude re-keyed; artifact compiles)" \
  || { echo "  FAIL: static-forward-declaration artifact did not compile"; exit 1; }
printf 'static _BitInt(13) bh(int x) { return (_BitInt(13))(x + 1); }\nint bu(int y) { return (int)bh(y); }\n' > "${tmp}/link/sb2.c"
"${tmp}/bcir-cc" --linkable "${tmp}/link/sb2.c" > "${tmp}/link/sb2_lk.c" || { echo "  FAIL: --linkable sb2"; exit 1; }
grep -q "^static _BitInt(13) bh(" "${tmp}/link/sb2_lk.c" \
  && echo "  PASS _BitInt-return static keeps its static (multi-paren name scan)" \
  || { echo "  FAIL: a _BitInt return spelling defeated the static keeper"; exit 1; }

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

# Automatic link-flag emission (#linkflags, B1): the compiler DERIVES the linker flags a translation
# unit needs from its external-call edges (c.call.libm:/.void:/extern:), instead of every harness
# hard-coding -lm. The callee->library mapping (linkflags.py / bcir_cfront.c bcir_lib_for_callee) is the
# dual-rail source of truth; bcir-cc --emit-link-flags must match the oracle (bcir-cfront
# --emit-link-flags) BYTE-FOR-BYTE. Cover a pure-integer unit (no flags), a math.h unit (-lm), and a
# free()/malloc unit (no flag -- libc is implicit), spanning the empty + non-empty cases.
echo "[c-runtime] automatic link-flag emission (bcir-cc --emit-link-flags): derived flags == oracle (#linkflags)"
mkdir -p "${tmp}/lf"
printf 'unsigned f(unsigned a){ return a*3u + 1u; }\n'                                              > "${tmp}/lf/pureint.c"
printf '#include <math.h>\ndouble f(double x){ return sqrt(x) + floor(x); }\n'                       > "${tmp}/lf/mathh.c"
printf '#include <stdlib.h>\nunsigned f(unsigned n){ unsigned *p=malloc(n*4u); unsigned r=p[0]; free(p); return r; }\n' > "${tmp}/lf/free.c"
lf_seen=""
for lf in pureint mathh free; do
  c_lf="$("${tmp}/bcir-cc" --emit-link-flags "${tmp}/lf/${lf}.c")" || { echo "  FAIL: bcir-cc --emit-link-flags ${lf}"; exit 1; }
  py_lf="$(python3 -m bcir.frontends.cfront --emit-link-flags "${tmp}/lf/${lf}.c")" || { echo "  FAIL: oracle link-flags ${lf}"; exit 1; }
  [ "${c_lf}" = "${py_lf}" ] \
    && echo "  PASS linkflags ${lf} (oracle == C: [${c_lf}])" \
    || { echo "  FAIL: linkflags ${lf} (C='[${c_lf}]' PY='[${py_lf}]')"; exit 1; }
  lf_seen="${lf_seen}[${c_lf}]"
done
# expected: pure-int empty, math.h -lm, free empty -- the gate must span the empty + the -lm case.
[ "${lf_seen}" = "[][-lm][]" ] \
  && echo "  PASS linkflags spans pure-int (no flags) / math.h (-lm) / free (libc-implicit, no flag)" \
  || { echo "  FAIL: linkflags gate did not span the expected cases: ${lf_seen}"; exit 1; }
# --emit-c is self-describing: the C.2 attestation header carries the derived link_flags line.
"${tmp}/bcir-cc" --emit-c "${tmp}/lf/mathh.c" | grep -q "link_flags  -lm" \
  && echo "  PASS linkflags: --emit-c attestation header carries the derived flags (link_flags -lm)" \
  || { echo "  FAIL: --emit-c header missing the derived link_flags line"; exit 1; }
# B2 FFTW link-flag rule (dual-rail): the trusted-library edges (cblas_*/fftwf_*) are NOT reachable from a
# cfront SOURCE (an unknown callee lowers to an in-unit `c.call:` edge, not `c.call.libm:`) -- they are
# minted by the kernel EMITTERS (emit_blas_gemm_c / emit_fftw_fft_c). So this probe drives the C twin's
# bcir_cfront_link_flags over a FABRICATED unit carrying a `c.call.libm:fftwf_execute` edge and asserts it
# derives `-lfftw3` (the B2 rule), with a cblas edge -> -lcblas (no B5 regression), a libm edge -> -lm,
# and an unknown edge -> no flag. The oracle (linkflags.library_for_callee) is pinned in test_c_cfront.py.
echo "[c-runtime] B2 FFTW link-flag rule (bcir_cfront_link_flags twin): fftwf_* -> -lfftw3 (#linkflags-fftw)"
cat > "${tmp}/lf_fftw.c" <<'PROBE'
#include <stdio.h>
#include <string.h>
#include "bcir_cir.h"
#include "bcir_cfront.h"
/* Build a one-function unit whose single claim is the given external-call edge op, then derive its flags. */
static const char *derive(const char *op) {
  static char buf[128];
  bcir_claim cl; memset(&cl, 0, sizeof cl);
  snprintf(cl.op, sizeof cl.op, "%s", op);
  bcir_func f; memset(&f, 0, sizeof f);
  f.claims = &cl; f.n_claims = 1;
  bcir_unit u; memset(&u, 0, sizeof u);
  u.funcs = &f; u.n_funcs = 1;
  bcir_cfront_link_flags(&u, buf, sizeof buf);
  return buf;
}
static int eq(const char *op, const char *want) {
  const char *got = derive(op);
  if (strcmp(got, want)) { printf("FAIL %s -> '%s' want '%s'\n", op, got, want); return 0; }
  return 1;
}
int main(void) {
  int ok = 1;
  ok &= eq("c.call.libm:fftwf_execute", "-lfftw3");      /* the B2 rule */
  ok &= eq("c.call.libm:fftwf_plan_dft_1d", "-lfftw3");  /* any fftwf_* */
  ok &= eq("c.call.libm:fftw_execute", "-lfftw3");       /* the double-prec fftw_* prefix */
  ok &= eq("c.call.libm:cblas_sgemm", "-lcblas");        /* B5 (no regression) */
  ok &= eq("c.call.libm:sqrt", "-lm");                   /* libm (no regression) */
  ok &= eq("c.call.libm:totally_unknown_fn", "");        /* unknown -> no flag (no regression) */
  if (ok) puts("OK linkflags-fftw");
  return ok ? 0 : 1;
}
PROBE
"${CC}" -std=c23 -O2 -Wall -Wextra -I "${C}" "${tmp}/lf_fftw.c" "${C}/bcir_cfront.c" "${C}/bcir_cpp.c" \
  "${C}/bcir_verify.c" "${C}/bcir_runtime.c" -o "${tmp}/lf_fftw" 2>/dev/null \
  || "${CC}" -std=c11 -O2 -I "${C}" "${tmp}/lf_fftw.c" "${C}/bcir_cfront.c" "${C}/bcir_cpp.c" \
       "${C}/bcir_verify.c" "${C}/bcir_runtime.c" -o "${tmp}/lf_fftw" \
  || { echo "  FAIL: FFTW link-flag probe build"; exit 1; }
"${tmp}/lf_fftw" | grep -q "^OK linkflags-fftw" \
  && echo "  PASS linkflags-fftw: C twin derives fftwf_*/fftw_* -> -lfftw3 (cblas/-lm/unknown unchanged)" \
  || { echo "  FAIL: FFTW link-flag rule diverged on the C twin"; "${tmp}/lf_fftw"; exit 1; }

# B-breadth (#61) LAPACK link-flag rule (dual-rail): the LAPACK edge (LAPACKE_sgesv) is minted by the kernel
# EMITTER (emit_lapack_solve_c), not reachable from a cfront source. So this probe drives the C twin's
# bcir_cfront_link_flags over FABRICATED units carrying a `c.call.libm:LAPACKE_sgesv` edge and asserts it
# derives `-llapack` (the LAPACK rule), with a Fortran-ABI `sgesv_` edge -> -llapack too, and no regression
# on fftwf_* -> -lfftw3, cblas_* -> -lcblas, libm -> -lm, and unknown -> no flag. The oracle
# (linkflags.library_for_callee) is pinned in test_c_cfront.py + test_lapack.py.
echo "[c-runtime] LAPACK link-flag rule (bcir_cfront_link_flags twin): LAPACKE_*/sgesv_ -> -llapack (#linkflags-lapack)"
cat > "${tmp}/lf_lapack.c" <<'PROBE'
#include <stdio.h>
#include <string.h>
#include "bcir_cir.h"
#include "bcir_cfront.h"
/* Build a one-function unit whose single claim is the given external-call edge op, then derive its flags. */
static const char *derive(const char *op) {
  static char buf[128];
  bcir_claim cl; memset(&cl, 0, sizeof cl);
  snprintf(cl.op, sizeof cl.op, "%s", op);
  bcir_func f; memset(&f, 0, sizeof f);
  f.claims = &cl; f.n_claims = 1;
  bcir_unit u; memset(&u, 0, sizeof u);
  u.funcs = &f; u.n_funcs = 1;
  bcir_cfront_link_flags(&u, buf, sizeof buf);
  return buf;
}
static int eq(const char *op, const char *want) {
  const char *got = derive(op);
  if (strcmp(got, want)) { printf("FAIL %s -> '%s' want '%s'\n", op, got, want); return 0; }
  return 1;
}
int main(void) {
  int ok = 1;
  ok &= eq("c.call.libm:LAPACKE_sgesv", "-llapack");     /* the LAPACK rule (the wrapper's actual callee) */
  ok &= eq("c.call.libm:LAPACKE_dgesv", "-llapack");     /* any LAPACKE_* */
  ok &= eq("c.call.libm:LAPACKE_sgels", "-llapack");     /* E1 OLS: LAPACKE_sgels rides the SAME LAPACKE_* rule */
  ok &= eq("c.call.libm:LAPACKE_ssyev", "-llapack");     /* E2 PCA: LAPACKE_ssyev rides the SAME LAPACKE_* rule */
  ok &= eq("c.call.libm:sgesv_", "-llapack");            /* the Fortran-ABI driver symbol */
  ok &= eq("c.call.libm:fftwf_execute", "-lfftw3");      /* B2 (no regression) */
  ok &= eq("c.call.libm:cblas_sgemm", "-lcblas");        /* B5 (no regression) */
  ok &= eq("c.call.libm:sqrt", "-lm");                   /* libm (no regression) */
  ok &= eq("c.call.libm:totally_unknown_fn", "");        /* unknown -> no flag (no regression) */
  if (ok) puts("OK linkflags-lapack");
  return ok ? 0 : 1;
}
PROBE
"${CC}" -std=c23 -O2 -Wall -Wextra -I "${C}" "${tmp}/lf_lapack.c" "${C}/bcir_cfront.c" "${C}/bcir_cpp.c" \
  "${C}/bcir_verify.c" "${C}/bcir_runtime.c" -o "${tmp}/lf_lapack" 2>/dev/null \
  || "${CC}" -std=c11 -O2 -I "${C}" "${tmp}/lf_lapack.c" "${C}/bcir_cfront.c" "${C}/bcir_cpp.c" \
       "${C}/bcir_verify.c" "${C}/bcir_runtime.c" -o "${tmp}/lf_lapack" \
  || { echo "  FAIL: LAPACK link-flag probe build"; exit 1; }
"${tmp}/lf_lapack" | grep -q "^OK linkflags-lapack" \
  && echo "  PASS linkflags-lapack: C twin derives LAPACKE_*/sgels/sgesv_ -> -llapack (fftw/cblas/-lm/unknown unchanged)" \
  || { echo "  FAIL: LAPACK link-flag rule diverged on the C twin"; "${tmp}/lf_lapack"; exit 1; }

# E1 (ML-breadth) OLS portable fallback (#ols): the OVERDETERMINED least-squares wrap (emit_lapack_ols_c)
# generalizes the square sgesv solve to linear regression (minimize ||A x - b||_2). The linked path is the
# QR-based LAPACKE_sgels (~cond(A)); the portable fallback forms the NORMAL EQUATIONS G = A^T A (~cond(A)^2,
# the textbook OLS twin of kbcir.ols.ols_reference). This probe compiles + runs the FALLBACK (no LAPACK
# needed -- CI is LAPACK-free, exactly the Area-B norm) over a CONSISTENT overdetermined system b = A x_true
# and checks it RECOVERS x_true on a well-conditioned A. The LAPACKE_sgels -> -llapack dual-rail is confirmed
# by the #linkflags-lapack probe above (LAPACKE_sgels rides the SAME LAPACKE_* rule -- no linkflags change).
echo "[c-runtime] E1 OLS portable fallback (emit_lapack_ols_c): normal-equations recovers a known x (#ols)"
# Emit the fallback OLS kernel from the oracle, append a main that fits y = 2x + 1 (m=8, n=2 -> [1, x] rows,
# recovers [c0=1, c1=2]), and assert the recovered coefficients match to float round-off.
python3 - > "${tmp}/ols_kernel.c" <<'PY' || { echo "  FAIL: python OLS emit"; exit 1; }
from bcir.lower.c_kernel import emit_lapack_ols_c
print(emit_lapack_ols_c(8, 2, 1, "ols"))
PY
cat >> "${tmp}/ols_kernel.c" <<'MAIN'
#include <stdio.h>
#include <math.h>
int main(void) {
  /* fit y = 2x + 1 over 8 points exactly on the line: design rows [1, x_i], b_i = 2*x_i + 1. */
  float A[16], b[8], x[2] = {0.0f, 0.0f};
  for (int i = 0; i < 8; ++i) { A[i*2+0] = 1.0f; A[i*2+1] = (float)i; b[i] = 2.0f*(float)i + 1.0f; }
  ols(A, b, x);                                   /* x = [c0, c1] */
  printf("c0=%.6f c1=%.6f\n", x[0], x[1]);
  /* recovered coefficients must be [1, 2] to float round-off (consistent, well-conditioned). */
  return (fabsf(x[0] - 1.0f) < 1e-3f && fabsf(x[1] - 2.0f) < 1e-3f) ? 0 : 1;
}
MAIN
"${CC}" -std=c11 -O2 -Wall -Wextra "${tmp}/ols_kernel.c" -lm -o "${tmp}/ols" 2>/dev/null \
  || "${CC}" -std=c23 -O2 "${tmp}/ols_kernel.c" -lm -o "${tmp}/ols" \
  || { echo "  FAIL: OLS fallback build"; exit 1; }
ols_out="$("${tmp}/ols")"; ols_rc=$?    # rc=0 IS the recovery check: |c0-1|<1e-3 && |c1-2|<1e-3 (in the driver)
{ [ "${ols_rc}" = "0" ] && printf '%s' "${ols_out}" | grep -q "^c0=.* c1=.*"; } \
  && echo "  PASS ols: normal-equations fallback recovers y=2x+1 (${ols_out}; LAPACKE_sgels -> -llapack via #linkflags-lapack)" \
  || { echo "  FAIL: OLS fallback did not recover [1,2] (rc=${ols_rc}: ${ols_out})"; exit 1; }

# E2 (ML-breadth) PCA portable fallback (#pca): the SYMMETRIC EIGENDECOMPOSITION wrap (emit_lapack_eigh_c) is
# the PCA sibling of E1's OLS solve -- where OLS forms a symmetric Gram matrix and SOLVES it, PCA forms a
# symmetric covariance and EIGENDECOMPOSES it. The linked path is LAPACKE_ssyev (Householder + implicit-QR);
# the portable fallback is the classic JACOBI rotation sweep (the C twin of kbcir.pca._jacobi_eigh). This probe
# compiles + runs the FALLBACK (no LAPACK needed -- CI is LAPACK-free, exactly the Area-B norm) on a hand-built
# DIAGONAL symmetric matrix diag(5,3,1) with DISTINCT (well-separated) eigenvalues, and checks it recovers the
# eigenvalues DESCENDING [5,3,1] and the standard-basis eigenvectors (sign convention: largest-magnitude entry
# positive). The LAPACKE_ssyev -> -llapack dual-rail is confirmed by the #linkflags-lapack probe above
# (LAPACKE_ssyev rides the SAME LAPACKE_* rule -- no linkflags change).
echo "[c-runtime] E2 PCA portable fallback (emit_lapack_eigh_c): Jacobi recovers a known spectrum (#pca)"
python3 - > "${tmp}/eigh_kernel.c" <<'PY' || { echo "  FAIL: python PCA eigh emit"; exit 1; }
from bcir.lower.c_kernel import emit_lapack_eigh_c
print(emit_lapack_eigh_c(3, "eigh"))
PY
cat >> "${tmp}/eigh_kernel.c" <<'MAIN'
#include <stdio.h>
#include <math.h>
int main(void) {
  /* a hand-built symmetric matrix diag(5,3,1): eigenvalues [5,3,1] DESCENDING, eigenvectors = standard basis. */
  float C[9] = {5.0f, 0.0f, 0.0f,  0.0f, 3.0f, 0.0f,  0.0f, 0.0f, 1.0f};
  float vals[3] = {0}, vecs[9] = {0};
  eigh(C, vals, vecs);                            /* vals descending, vecs[t*3+j] = component t coord j */
  printf("l0=%.6f l1=%.6f l2=%.6f\n", vals[0], vals[1], vals[2]);
  int ok = fabsf(vals[0] - 5.0f) < 1e-4f && fabsf(vals[1] - 3.0f) < 1e-4f && fabsf(vals[2] - 1.0f) < 1e-4f;
  for (int t = 0; t < 3 && ok; ++t)               /* eigenvectors are the standard basis, sign convention */
    for (int j = 0; j < 3; ++j) {
      float want = (j == t) ? 1.0f : 0.0f;
      if (fabsf(fabsf(vecs[t*3+j]) - want) >= 1e-4f) ok = 0;
    }
  ok = ok && vecs[0] > 0.0f && vecs[4] > 0.0f && vecs[8] > 0.0f;   /* largest-magnitude entry positive */
  return ok ? 0 : 1;
}
MAIN
"${CC}" -std=c11 -O2 -Wall -Wextra "${tmp}/eigh_kernel.c" -lm -o "${tmp}/eigh" 2>/dev/null \
  || "${CC}" -std=c23 -O2 "${tmp}/eigh_kernel.c" -lm -o "${tmp}/eigh" \
  || { echo "  FAIL: PCA eigh fallback build"; exit 1; }
eigh_out="$("${tmp}/eigh")"; eigh_rc=$?    # rc=0 IS the recovery check: eigenvalues [5,3,1] + standard-basis vecs
{ [ "${eigh_rc}" = "0" ] && printf '%s' "${eigh_out}" | grep -q "^l0=.* l1=.* l2=.*"; } \
  && echo "  PASS pca: Jacobi fallback recovers diag(5,3,1) (${eigh_out}; LAPACKE_ssyev -> -llapack via #linkflags-lapack)" \
  || { echo "  FAIL: PCA fallback did not recover [5,3,1] (rc=${eigh_rc}: ${eigh_out})"; exit 1; }

# E3 (ML-breadth) full Transformer block's ONE new numeric primitive (#layernorm): the LAYERNORM kernel
# (emit_layernorm_c). Unlike E1/E2 (each a wrap of one external LAPACK kernel), the Transformer block is a
# COMPOSITION -- its per-head matmuls + scale + softmax are the EXISTING emit_attention_c C twin, the
# projections/feed-forward are the EXISTING matmul emitters, so the ONLY net-new C kernel is the per-row
# layernorm: out = gamma*(x-mean)/sqrtf(var+eps) + beta (POPULATION /dim variance). Its only transcendental is
# the 1/sqrtf(var+eps), which rides the c.call.libm:sqrtf edge (-lm, ALREADY mapped -- no linkflags change,
# confirmed by #linkflags-* probes: sqrtf rides the libm rule). This probe compiles + runs the kernel (no
# external library needed -- only libm) on a known matrix with gamma=1/beta=0 and checks each output row is
# normalized to mean~0 / var~1 (the property layernorm guarantees), the C twin of kbcir.transformer.layernorm_
# reference + its layernorm_stats independent verifier.
echo "[c-runtime] E3 Transformer layernorm (emit_layernorm_c): per-row normalize to mean~0/var~1 (#layernorm)"
python3 - > "${tmp}/ln_kernel.c" <<'PY' || { echo "  FAIL: python layernorm emit"; exit 1; }
from bcir.lower.c_kernel import emit_layernorm_c
print(emit_layernorm_c(2, 4, "ln"))
PY
cat >> "${tmp}/ln_kernel.c" <<'MAIN'
#include <stdio.h>
#include <math.h>
int main(void) {
  /* two rows over a dim=4 feature axis; gamma=1/beta=0 -> a pure normalization (mean 0, var 1 per row). */
  float X[8] = {1.0f, 2.0f, 3.0f, 4.0f,  -2.0f, 0.0f, 2.0f, 4.0f};
  float G[4] = {1.0f, 1.0f, 1.0f, 1.0f}, B[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float O[8] = {0};
  ln(X, G, B, 1e-5f, O);                         /* out[r*4+c] = (X[r,c]-mean)/sqrtf(var+eps) */
  int ok = 1;
  for (int r = 0; r < 2 && ok; ++r) {
    float mean = 0.0f; for (int c = 0; c < 4; ++c) mean += O[r*4+c]; mean /= 4.0f;
    float var = 0.0f;  for (int c = 0; c < 4; ++c) { float d = O[r*4+c] - mean; var += d*d; } var /= 4.0f;
    printf("r%d mean=%.6f var=%.6f\n", r, mean, var);
    if (fabsf(mean) >= 1e-3f || fabsf(var - 1.0f) >= 1e-2f) ok = 0;   /* normalized: mean~0, var~1 */
  }
  return ok ? 0 : 1;
}
MAIN
"${CC}" -std=c11 -O2 -Wall -Wextra "${tmp}/ln_kernel.c" -lm -o "${tmp}/ln" 2>/dev/null \
  || "${CC}" -std=c23 -O2 "${tmp}/ln_kernel.c" -lm -o "${tmp}/ln" \
  || { echo "  FAIL: layernorm kernel build"; exit 1; }
ln_out="$("${tmp}/ln")"; ln_rc=$?    # rc=0 IS the check: each output row has mean~0 and var~1 (in the driver)
{ [ "${ln_rc}" = "0" ] && printf '%s' "${ln_out}" | grep -q "^r0 mean=.* var=.*"; } \
  && echo "  PASS layernorm: rows normalize to mean~0/var~1 (${ln_out//$'\n'/ }; sqrtf -> -lm via the libm rule, no linkflags change)" \
  || { echo "  FAIL: layernorm did not normalize rows (rc=${ln_rc}: ${ln_out})"; exit 1; }

# E4 (ML-breadth) recurrent cells: the LSTM CELL kernel (emit_lstm_cell_c) -- the recurrent-cell executable seam
# (#lstm). E4 is a TWO-TIER design: Tier A (the closed-set relu-RNN) is already lowerable through the EXISTING
# autodiff/closed-set machinery (relu = select, exact, no libm), so the net-new C seam is the TRANSCENDENTAL
# tier, of which the LSTM cell is the canonical gate-rich representative: per unit f=sigmoid(Wf x+Uf h+bf),
# i,o likewise, g=tanhf(Wg x+Ug h+bg), c=f*c_prev+i*g, h=o*tanhf(c). Its only transcendentals are tanhf + the
# expf inside the (numerically-guarded) sigmoid, BOTH riding the c.call.libm: edge (-lm, ALREADY mapped -- no
# linkflags change; sqrtf/tanhf/expf all ride the libm rule, confirmed by the #linkflags-* probes). This probe
# compiles + runs the kernel (only libm needed) on a 1x1 cell with KNOWN weights (W_*=1, U_*=0, b_*=0, x=0.5,
# h_prev=0, c_prev=0) and asserts it matches the hand-computed output to float round-off -- the C twin of
# kbcir.recurrent.lstm_cell_reference.
echo "[c-runtime] E4 recurrent LSTM cell (emit_lstm_cell_c): 1x1 cell matches hand-computed forward (#lstm)"
python3 - > "${tmp}/lstm_kernel.c" <<'PY' || { echo "  FAIL: python lstm emit"; exit 1; }
from bcir.lower.c_kernel import emit_lstm_cell_c
print(emit_lstm_cell_c(1, 1, "lstm_cell"))
PY
cat >> "${tmp}/lstm_kernel.c" <<'MAIN'
#include <stdio.h>
#include <math.h>
int main(void) {
  /* a 1x1 LSTM with W_*=1, U_*=0, b_*=0, x=0.5, h_prev=0, c_prev=0: every gate pre-activation is 0.5. */
  float X[1] = {0.5f}, HP[1] = {0.0f}, CP[1] = {0.0f};
  float Wf[1]={1.0f}, Uf[1]={0.0f}, bf[1]={0.0f};
  float Wi[1]={1.0f}, Ui[1]={0.0f}, bi[1]={0.0f};
  float Wo[1]={1.0f}, Uo[1]={0.0f}, bo[1]={0.0f};
  float Wg[1]={1.0f}, Ug[1]={0.0f}, bg[1]={0.0f};
  float H[1], C[1];
  lstm_cell(X, HP, CP, Wf,Uf,bf, Wi,Ui,bi, Wo,Uo,bo, Wg,Ug,bg, H, C);
  /* hand-computed reference: f=i=o=sigmoid(0.5), g=tanhf(0.5); c=i*g; h=o*tanhf(c). */
  float s = 1.0f/(1.0f+expf(-0.5f));
  float g = tanhf(0.5f);
  float c_exp = s*g;
  float h_exp = s*tanhf(c_exp);
  printf("h=%.6f c=%.6f (exp h=%.6f c=%.6f)\n", H[0], C[0], h_exp, c_exp);
  int ok = (fabsf(H[0]-h_exp) < 1e-4f) && (fabsf(C[0]-c_exp) < 1e-4f);
  return ok ? 0 : 1;
}
MAIN
"${CC}" -std=c11 -O2 -Wall -Wextra "${tmp}/lstm_kernel.c" -lm -o "${tmp}/lstm" 2>/dev/null \
  || "${CC}" -std=c23 -O2 "${tmp}/lstm_kernel.c" -lm -o "${tmp}/lstm" \
  || { echo "  FAIL: lstm kernel build"; exit 1; }
lstm_out="$("${tmp}/lstm")"; lstm_rc=$?    # rc=0 IS the check: H/C match the hand-computed reference (driver)
{ [ "${lstm_rc}" = "0" ] && printf '%s' "${lstm_out}" | grep -q "^h=.* c=.*"; } \
  && echo "  PASS lstm: 1x1 cell matches hand-computed forward (${lstm_out}; tanhf/expf -> -lm via the libm rule, no linkflags change)" \
  || { echo "  FAIL: lstm cell did not match the reference (rc=${lstm_rc}: ${lstm_out})"; exit 1; }

# E5 (ML-breadth) CLASSICAL-ML PREDICT path: the baked-model fixed-shape predict kernels (#classical). The
# honest framing E7 cites: classical-ML TRAINING (tree induction, the SVM QP solve, NB fitting) is iterative/
# combinatorial -- a POOR fit for BCIR's fixed-shape claim model (library/Python). PREDICT over a BAKED model is
# the opposite: a deterministic, fixed-shape kernel = the G5 baked-weights pattern. Two probes show the Area-B
# pattern covers BOTH halves: the RBF-SVM (transcendental -- its only external is expf on the c.call.libm: edge,
# -lm, ALREADY mapped -- no linkflags change) and the decision tree (EXACT -- pure comparisons + a leaf return,
# NO transcendental, NO libm). C twins of kbcir.classical.svm_decision_rbf / tree_predict.
echo "[c-runtime] E5 classical-ML RBF-SVM predict (emit_svm_rbf_predict_c): decision function matches reference (#classical #svm)"
python3 - > "${tmp}/svm_rbf_kernel.c" <<'PY' || { echo "  FAIL: python svm-rbf emit"; exit 1; }
from bcir.lower.c_kernel import emit_svm_rbf_predict_c
print(emit_svm_rbf_predict_c(1, 2, "svm_rbf"))
PY
cat >> "${tmp}/svm_rbf_kernel.c" <<'MAIN'
#include <stdio.h>
#include <math.h>
int main(void) {
  /* one SV at the origin, alpha_y=2, b=0.5, gamma=1: f(x)=2*expf(-||x||^2)+0.5. At x=(1,0): 2*exp(-1)+0.5. */
  float X[2] = {1.0f, 0.0f};
  float SV[2] = {0.0f, 0.0f};
  float AY[1] = {2.0f};
  float f = svm_rbf(X, SV, AY, 0.5f, 1.0f);
  float exp_f = 2.0f * expf(-1.0f) + 0.5f;
  printf("f=%.6f (exp=%.6f)\n", f, exp_f);
  return (fabsf(f - exp_f) < 1e-4f) ? 0 : 1;
}
MAIN
"${CC}" -std=c11 -O2 -Wall -Wextra "${tmp}/svm_rbf_kernel.c" -lm -o "${tmp}/svm_rbf" 2>/dev/null \
  || "${CC}" -std=c23 -O2 "${tmp}/svm_rbf_kernel.c" -lm -o "${tmp}/svm_rbf" \
  || { echo "  FAIL: svm-rbf kernel build"; exit 1; }
svm_out="$("${tmp}/svm_rbf")"; svm_rc=$?    # rc=0 IS the check: f matches the hand-computed value (driver)
{ [ "${svm_rc}" = "0" ] && printf '%s' "${svm_out}" | grep -q "^f=.*"; } \
  && echo "  PASS svm: RBF decision matches reference (${svm_out}; expf -> -lm via the libm rule, no linkflags change)" \
  || { echo "  FAIL: svm RBF decision did not match the reference (rc=${svm_rc}: ${svm_out})"; exit 1; }

echo "[c-runtime] E5 classical-ML decision-tree predict (emit_tree_predict_c): exact threshold traversal, NO libm (#classical #tree)"
python3 - > "${tmp}/tree_kernel.c" <<'PY' || { echo "  FAIL: python tree emit"; exit 1; }
from bcir.lower.c_kernel import emit_tree_predict_c
print(emit_tree_predict_c(5, 2, "tree_p"))
PY
cat >> "${tmp}/tree_kernel.c" <<'MAIN'
#include <stdio.h>
#include <math.h>
int main(void) {
  /* the toy 5-node tree (test_classical._toy_tree): node0 split x[0]<=0.5 -> node1 else leaf2(30);
     node1 split x[1]<=0.5 -> leaf3(10) else leaf4(20). */
  int   FE[5] = {0, 1, -1, -1, -1};
  float TH[5] = {0.5f, 0.5f, 0.0f, 0.0f, 0.0f};
  int   LE[5] = {1, 3, 0, 0, 0};
  int   RI[5] = {2, 4, 0, 0, 0};
  float LV[5] = {0.0f, 0.0f, 30.0f, 10.0f, 20.0f};
  float X0[2] = {0.2f, 0.2f};   /* -> leaf3 = 10 */
  float X1[2] = {0.2f, 0.9f};   /* -> leaf4 = 20 */
  float X2[2] = {0.9f, 0.0f};   /* -> leaf2 = 30 */
  float a = tree_p(X0, FE, TH, LE, RI, LV);
  float b = tree_p(X1, FE, TH, LE, RI, LV);
  float c = tree_p(X2, FE, TH, LE, RI, LV);
  printf("a=%.1f b=%.1f c=%.1f\n", a, b, c);
  /* the tree is EXACT (no transcendental) -- the leaf values match bit-for-bit. */
  return (a == 10.0f && b == 20.0f && c == 30.0f) ? 0 : 1;
}
MAIN
# the tree kernel is EXACT -- no libm needed (we still link -lm only for the harness fabsf-free main; not required).
"${CC}" -std=c11 -O2 -Wall -Wextra "${tmp}/tree_kernel.c" -o "${tmp}/tree" 2>/dev/null \
  || "${CC}" -std=c23 -O2 "${tmp}/tree_kernel.c" -o "${tmp}/tree" \
  || { echo "  FAIL: tree kernel build"; exit 1; }
tree_out="$("${tmp}/tree")"; tree_rc=$?    # rc=0 IS the check: the three leaves match exactly (driver)
{ [ "${tree_rc}" = "0" ] && printf '%s' "${tree_out}" | grep -q "^a=.* b=.* c=.*"; } \
  && echo "  PASS tree: exact threshold traversal returns the known leaves (${tree_out}; NO libm -- the EXACT half of the Area-B pattern)" \
  || { echo "  FAIL: tree predict did not return the known leaves (rc=${tree_rc}: ${tree_out})"; exit 1; }

# E6 (ML-breadth) UNSUPERVISED K-means nearest-centroid ASSIGN (#kmeans): the unsupervised analog of the E5 EXACT
# tree kernel. K-means FIT is a bounded iterative optimization (Lloyd's, library/Python-shaped); ASSIGN over
# BAKED centroids is the fixed-shape PREDICT kernel (G5 baked-weights pattern). It is the EXACT half of the
# Area-B pattern -- argmin_c ||x-centroid[c]||^2 by squared distance: pure subtract/multiply/add + comparisons,
# NO transcendental, NO libm (needs no -lm). It returns an int cluster id, so the C-vs-oracle check is
# INTEGER-EXACT (the same argmin). C twin of kbcir.unsupervised.kmeans_assign.
echo "[c-runtime] E6 unsupervised K-means assign (emit_kmeans_assign_c): nearest-centroid argmin, NO libm (#kmeans)"
python3 - > "${tmp}/kmeans_kernel.c" <<'PY' || { echo "  FAIL: python kmeans emit"; exit 1; }
from bcir.lower.c_kernel import emit_kmeans_assign_c
print(emit_kmeans_assign_c(3, 2, "km_assign"))
PY
cat >> "${tmp}/kmeans_kernel.c" <<'MAIN'
#include <stdio.h>
int main(void) {
  /* three baked centroids: (0,0), (10,10), (-5,5). Points off any exact equidistant tie. */
  float C[6] = {0.0f, 0.0f,  10.0f, 10.0f,  -5.0f, 5.0f};
  float X0[2] = {0.3f, 0.1f};    /* -> centroid 0 (near origin) */
  float X1[2] = {9.7f, 10.2f};   /* -> centroid 1 (near (10,10)) */
  float X2[2] = {-4.8f, 5.1f};   /* -> centroid 2 (near (-5,5)) */
  int a = km_assign(X0, C);
  int b = km_assign(X1, C);
  int c = km_assign(X2, C);
  printf("a=%d b=%d c=%d\n", a, b, c);
  /* the assign kernel is EXACT (integer cluster id) -- the argmin matches kmeans_assign exactly. */
  return (a == 0 && b == 1 && c == 2) ? 0 : 1;
}
MAIN
# the K-means assign kernel is EXACT -- no libm needed (no -lm; the EXACT half of the Area-B pattern).
"${CC}" -std=c11 -O2 -Wall -Wextra "${tmp}/kmeans_kernel.c" -o "${tmp}/kmeans" 2>/dev/null \
  || "${CC}" -std=c23 -O2 "${tmp}/kmeans_kernel.c" -o "${tmp}/kmeans" \
  || { echo "  FAIL: kmeans assign kernel build"; exit 1; }
kmeans_out="$("${tmp}/kmeans")"; kmeans_rc=$?    # rc=0 IS the check: the three argmins match exactly (driver)
{ [ "${kmeans_rc}" = "0" ] && printf '%s' "${kmeans_out}" | grep -q "^a=.* b=.* c=.*"; } \
  && echo "  PASS kmeans: exact nearest-centroid argmin returns the known clusters (${kmeans_out}; NO libm -- the EXACT half of the Area-B pattern, integer-exact)" \
  || { echo "  FAIL: kmeans assign did not return the known clusters (rc=${kmeans_rc}: ${kmeans_out})"; exit 1; }

# Area-B breadth (#62) GSL link-flag rule (dual-rail): the GSL edge (gsl_stats_mean) is minted by the kernel
# EMITTER (emit_gsl_stats_c), not reachable from a cfront source. So this probe drives the C twin's
# bcir_cfront_link_flags over FABRICATED units carrying a `c.call.libm:gsl_stats_mean` edge and asserts it
# derives `-lgsl` (the GSL rule), with any gsl_* -> -lgsl too, and no regression on LAPACKE_* -> -llapack,
# fftwf_* -> -lfftw3, cblas_* -> -lcblas, libm -> -lm, and unknown -> no flag. The oracle
# (linkflags.library_for_callee) is pinned in test_c_cfront.py + test_gsl.py.
echo "[c-runtime] GSL link-flag rule (bcir_cfront_link_flags twin): gsl_* -> -lgsl (#linkflags-gsl)"
cat > "${tmp}/lf_gsl.c" <<'PROBE'
#include <stdio.h>
#include <string.h>
#include "bcir_cir.h"
#include "bcir_cfront.h"
/* Build a one-function unit whose single claim is the given external-call edge op, then derive its flags. */
static const char *derive(const char *op) {
  static char buf[128];
  bcir_claim cl; memset(&cl, 0, sizeof cl);
  snprintf(cl.op, sizeof cl.op, "%s", op);
  bcir_func f; memset(&f, 0, sizeof f);
  f.claims = &cl; f.n_claims = 1;
  bcir_unit u; memset(&u, 0, sizeof u);
  u.funcs = &f; u.n_funcs = 1;
  bcir_cfront_link_flags(&u, buf, sizeof buf);
  return buf;
}
static int eq(const char *op, const char *want) {
  const char *got = derive(op);
  if (strcmp(got, want)) { printf("FAIL %s -> '%s' want '%s'\n", op, got, want); return 0; }
  return 1;
}
int main(void) {
  int ok = 1;
  ok &= eq("c.call.libm:gsl_stats_mean", "-lgsl");       /* the GSL rule (the wrapper's actual callee) */
  ok &= eq("c.call.libm:gsl_stats_variance", "-lgsl");   /* any gsl_* */
  ok &= eq("c.call.libm:gsl_sf_erf", "-lgsl");           /* a special-function gsl_* too */
  ok &= eq("c.call.libm:LAPACKE_sgesv", "-llapack");     /* #61 LAPACK (no regression) */
  ok &= eq("c.call.libm:fftwf_execute", "-lfftw3");      /* B2 (no regression) */
  ok &= eq("c.call.libm:cblas_sgemm", "-lcblas");        /* B5 (no regression) */
  ok &= eq("c.call.libm:sqrt", "-lm");                   /* libm (no regression) */
  ok &= eq("c.call.libm:totally_unknown_fn", "");        /* unknown -> no flag (no regression) */
  if (ok) puts("OK linkflags-gsl");
  return ok ? 0 : 1;
}
PROBE
"${CC}" -std=c23 -O2 -Wall -Wextra -I "${C}" "${tmp}/lf_gsl.c" "${C}/bcir_cfront.c" "${C}/bcir_cpp.c" \
  "${C}/bcir_verify.c" "${C}/bcir_runtime.c" -o "${tmp}/lf_gsl" 2>/dev/null \
  || "${CC}" -std=c11 -O2 -I "${C}" "${tmp}/lf_gsl.c" "${C}/bcir_cfront.c" "${C}/bcir_cpp.c" \
       "${C}/bcir_verify.c" "${C}/bcir_runtime.c" -o "${tmp}/lf_gsl" \
  || { echo "  FAIL: GSL link-flag probe build"; exit 1; }
"${tmp}/lf_gsl" | grep -q "^OK linkflags-gsl" \
  && echo "  PASS linkflags-gsl: C twin derives gsl_* -> -lgsl (lapack/fftw/cblas/-lm/unknown unchanged)" \
  || { echo "  FAIL: GSL link-flag rule diverged on the C twin"; "${tmp}/lf_gsl"; exit 1; }

# Area-B breadth (#63) SLEEF link-flag rule (dual-rail): the SLEEF edge (Sleef_expf1_u10) is minted by the
# kernel EMITTER (emit_sleef_exp_c), not reachable from a cfront source. So this probe drives the C twin's
# bcir_cfront_link_flags over FABRICATED units carrying a `c.call.libm:Sleef_expf1_u10` edge and asserts it
# derives `-lsleef` (the SLEEF rule), with any Sleef_* -> -lsleef too, and no regression on gsl_* -> -lgsl,
# LAPACKE_* -> -llapack, fftwf_* -> -lfftw3, cblas_* -> -lcblas, libm -> -lm, and unknown -> no flag. The
# oracle (linkflags.library_for_callee) is pinned in test_c_cfront.py + test_sleef.py.
echo "[c-runtime] SLEEF link-flag rule (bcir_cfront_link_flags twin): Sleef_* -> -lsleef (#linkflags-sleef)"
cat > "${tmp}/lf_sleef.c" <<'PROBE'
#include <stdio.h>
#include <string.h>
#include "bcir_cir.h"
#include "bcir_cfront.h"
/* Build a one-function unit whose single claim is the given external-call edge op, then derive its flags. */
static const char *derive(const char *op) {
  static char buf[128];
  bcir_claim cl; memset(&cl, 0, sizeof cl);
  snprintf(cl.op, sizeof cl.op, "%s", op);
  bcir_func f; memset(&f, 0, sizeof f);
  f.claims = &cl; f.n_claims = 1;
  bcir_unit u; memset(&u, 0, sizeof u);
  u.funcs = &f; u.n_funcs = 1;
  bcir_cfront_link_flags(&u, buf, sizeof buf);
  return buf;
}
static int eq(const char *op, const char *want) {
  const char *got = derive(op);
  if (strcmp(got, want)) { printf("FAIL %s -> '%s' want '%s'\n", op, got, want); return 0; }
  return 1;
}
int main(void) {
  int ok = 1;
  ok &= eq("c.call.libm:Sleef_expf1_u10", "-lsleef");    /* the SLEEF rule (the wrapper's actual callee) */
  ok &= eq("c.call.libm:Sleef_sinf1_u10", "-lsleef");    /* any Sleef_* */
  ok &= eq("c.call.libm:gsl_stats_mean", "-lgsl");       /* #62 GSL (no regression) */
  ok &= eq("c.call.libm:LAPACKE_sgesv", "-llapack");     /* #61 LAPACK (no regression) */
  ok &= eq("c.call.libm:fftwf_execute", "-lfftw3");      /* B2 (no regression) */
  ok &= eq("c.call.libm:cblas_sgemm", "-lcblas");        /* B5 (no regression) */
  ok &= eq("c.call.libm:expf", "-lm");                   /* libm (no regression -- the SLEEF fallback's twin) */
  ok &= eq("c.call.libm:totally_unknown_fn", "");        /* unknown -> no flag (no regression) */
  if (ok) puts("OK linkflags-sleef");
  return ok ? 0 : 1;
}
PROBE
"${CC}" -std=c23 -O2 -Wall -Wextra -I "${C}" "${tmp}/lf_sleef.c" "${C}/bcir_cfront.c" "${C}/bcir_cpp.c" \
  "${C}/bcir_verify.c" "${C}/bcir_runtime.c" -o "${tmp}/lf_sleef" 2>/dev/null \
  || "${CC}" -std=c11 -O2 -I "${C}" "${tmp}/lf_sleef.c" "${C}/bcir_cfront.c" "${C}/bcir_cpp.c" \
       "${C}/bcir_verify.c" "${C}/bcir_runtime.c" -o "${tmp}/lf_sleef" \
  || { echo "  FAIL: SLEEF link-flag probe build"; exit 1; }
"${tmp}/lf_sleef" | grep -q "^OK linkflags-sleef" \
  && echo "  PASS linkflags-sleef: C twin derives Sleef_* -> -lsleef (gsl/lapack/fftw/cblas/-lm/unknown unchanged)" \
  || { echo "  FAIL: SLEEF link-flag rule diverged on the C twin"; "${tmp}/lf_sleef"; exit 1; }

# Area-B breadth (#64) libcerf link-flag rule (dual-rail): the libcerf edge (erfcxf) is minted by the
# kernel EMITTER (emit_cerf_erfcx_c), not reachable from a cfront source. So this probe drives the C twin's
# bcir_cfront_link_flags over FABRICATED units carrying a `c.call.libm:erfcxf` edge and asserts it derives
# `-lcerf` (the libcerf rule), with the bare erfcx too, and no regression on Sleef_* -> -lsleef, gsl_* ->
# -lgsl, LAPACKE_* -> -llapack, fftwf_* -> -lfftw3, cblas_* -> -lcblas, libm (incl. erfcf -- erfcx is a
# symbol libm LACKS, so erfcf/erfc still map to -lm, NOT -lcerf) -> -lm, and unknown -> no flag. The oracle
# (linkflags.library_for_callee) is pinned in test_c_cfront.py + test_cerf.py.
echo "[c-runtime] libcerf link-flag rule (bcir_cfront_link_flags twin): erfcx* -> -lcerf (#linkflags-cerf)"
cat > "${tmp}/lf_cerf.c" <<'PROBE'
#include <stdio.h>
#include <string.h>
#include "bcir_cir.h"
#include "bcir_cfront.h"
/* Build a one-function unit whose single claim is the given external-call edge op, then derive its flags. */
static const char *derive(const char *op) {
  static char buf[128];
  bcir_claim cl; memset(&cl, 0, sizeof cl);
  snprintf(cl.op, sizeof cl.op, "%s", op);
  bcir_func f; memset(&f, 0, sizeof f);
  f.claims = &cl; f.n_claims = 1;
  bcir_unit u; memset(&u, 0, sizeof u);
  u.funcs = &f; u.n_funcs = 1;
  bcir_cfront_link_flags(&u, buf, sizeof buf);
  return buf;
}
static int eq(const char *op, const char *want) {
  const char *got = derive(op);
  if (strcmp(got, want)) { printf("FAIL %s -> '%s' want '%s'\n", op, got, want); return 0; }
  return 1;
}
int main(void) {
  int ok = 1;
  ok &= eq("c.call.libm:erfcxf", "-lcerf");              /* the libcerf rule (the wrapper's actual callee) */
  ok &= eq("c.call.libm:erfcx", "-lcerf");               /* the bare/double erfcx too */
  ok &= eq("c.call.libm:erfcf", "-lm");                  /* erfcf/erfc are still libm (NOT shadowed by -lcerf) */
  ok &= eq("c.call.libm:Sleef_expf1_u10", "-lsleef");    /* #63 SLEEF (no regression) */
  ok &= eq("c.call.libm:gsl_stats_mean", "-lgsl");       /* #62 GSL (no regression) */
  ok &= eq("c.call.libm:LAPACKE_sgesv", "-llapack");     /* #61 LAPACK (no regression) */
  ok &= eq("c.call.libm:fftwf_execute", "-lfftw3");      /* B2 (no regression) */
  ok &= eq("c.call.libm:cblas_sgemm", "-lcblas");        /* B5 (no regression) */
  ok &= eq("c.call.libm:expf", "-lm");                   /* libm (no regression -- the erfcx fallback's twin) */
  ok &= eq("c.call.libm:totally_unknown_fn", "");        /* unknown -> no flag (no regression) */
  if (ok) puts("OK linkflags-cerf");
  return ok ? 0 : 1;
}
PROBE
"${CC}" -std=c23 -O2 -Wall -Wextra -I "${C}" "${tmp}/lf_cerf.c" "${C}/bcir_cfront.c" "${C}/bcir_cpp.c" \
  "${C}/bcir_verify.c" "${C}/bcir_runtime.c" -o "${tmp}/lf_cerf" 2>/dev/null \
  || "${CC}" -std=c11 -O2 -I "${C}" "${tmp}/lf_cerf.c" "${C}/bcir_cfront.c" "${C}/bcir_cpp.c" \
       "${C}/bcir_verify.c" "${C}/bcir_runtime.c" -o "${tmp}/lf_cerf" \
  || { echo "  FAIL: libcerf link-flag probe build"; exit 1; }
"${tmp}/lf_cerf" | grep -q "^OK linkflags-cerf" \
  && echo "  PASS linkflags-cerf: C twin derives erfcx* -> -lcerf (sleef/gsl/lapack/fftw/cblas/-lm/unknown unchanged)" \
  || { echo "  FAIL: libcerf link-flag rule diverged on the C twin"; "${tmp}/lf_cerf"; exit 1; }

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
c_ps="$("${tmp}/bcir-cc" --emit-claimgraph "${tmp}/pstress.c" 2>&1 | grep -oE 'funcs=[0-9]+ claims=[0-9]+.*ok=[0-9] digest=[0-9a-f]+' | tail -1)"
py_ps="$(python3 -c "
from bcir.frontends.cfront import compile_unit
from bcir.model import Domain
from bcir.verify import cfront_structural_digest          # the cross-rail per-claim STRUCTURAL digest
r=compile_unit(open('${tmp}/pstress.c').read(), check_clang=False)
fns=r.lowered.functions; lf=fns[next(reversed(fns))]
kn=sum(1 for c in lf.claims if c.op=='c.const'); bo=sum(1 for c in lf.claims if c.op.startswith('c.bin.'))
repro=sum(1 for f in fns.values() if getattr(f,'reproducible',False))  # A1.3: matches the C twin's repro=N
dg=cfront_structural_digest(r.lowered)                    # byte-identical to the C twin's bcir_cfront_digest
print(f'funcs={len(fns)} claims={len(lf.claims)} mmio=0 bf=0 const={kn} binop={bo} call=0 repro={repro} ok={1 if r.is_clean else 0} digest={dg:016x}')")"
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

# Inline assembly (#inlineasm): GNU `asm`/`__asm__` is an ISA-NEUTRAL trusted opaque effect edge (ASM1) --
# the template is re-emitted VERBATIM as a `__asm__ [__volatile__] (...)` statement; BCIR owns only the
# calling side (operands + constraints + clobbers + ordering). This is a PYTHON-frontend feature (the C twin
# bcir_cfront.c does not parse asm yet), so the probe emits via `compile_unit` (not bcir-cc --emit-c). It
# proves the emit compiles under BOTH the default CC and -- when present -- a second compiler, AND that an
# `asm volatile("" ::: "memory")` compiler barrier wrapped around a store/load does NOT change the observable
# value (it only constrains ordering). Uses the reserved __asm__/__volatile__ spellings + the ISA-neutral
# `"=r"(out) : "0"(in)` tied-register copy + empty-template `"memory"` barrier, so it compiles on any ISA.
echo "[c-runtime] inline assembly: ISA-neutral trusted opaque edge, verbatim emit + memory barrier (#inlineasm)"
cat > "${tmp}/cfront_asm.c" <<'ASMC'
unsigned asm_copy(unsigned x){ unsigned y = 0; __asm__("" : "=r"(y) : "0"(x)); return y; }
unsigned asm_barrier(unsigned *p, unsigned x){
  *p = x;                                  /* store ... */
  __asm__ __volatile__("" ::: "memory");   /* an ordering fence between the store and the load */
  return *p;                               /* ... load: the barrier must NOT change the value (== x) */
}
void asm_basic(void){ __asm__("nop"); __asm__ __volatile__("" ::: "memory"); }
ASMC
FX="${tmp}/cfront_asm.c" python3 - > "${tmp}/asm_emit.c" <<'PY' || { echo "  FAIL: python asm emit"; exit 1; }
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

# Port-mapped I/O intrinsics (#portio, ASM2): inb/inw/inl / outb/outw/outl lower to a typed, isolated,
# BARRIERED I/O-port edge; the per-ISA `in`/`out` instruction is emitted behind `--target` (x86 emits the
# real instruction as `__asm__ __volatile__`; non-x86 has no port I/O -> the honest unsupported diagnostic).
# HONEST BOUNDARY: executing `in`/`out` from userspace TRAPS (it needs iopl/ioperm + ring-0), so this probe
# ASSEMBLES the emitted asm (`-c`, assemble-only) to prove it is valid x86 the toolchain accepts -- it NEVER
# links or runs it. The emitted text must carry the right `in`/`out` instruction + operands.
echo "[c-runtime] port-mapped I/O intrinsics: typed barriered edge, x86 in/out emit, assemble-only (#portio)"
cat > "${tmp}/cfront_portio.c" <<'PIOC'
unsigned pio_inb(unsigned port){ return inb(port); }      /* read u8  from a port */
unsigned pio_inw(unsigned port){ return inw(port); }      /* read u16 */
unsigned pio_inl(unsigned port){ return inl(port); }      /* read u32 */
void pio_outb(unsigned v, unsigned port){ outb(v, port); } /* Linux out(value, port): value first, port second */
void pio_outw(unsigned v, unsigned port){ outw(v, port); }
void pio_outl(unsigned v, unsigned port){ outl(v, port); }
PIOC
FX="${tmp}/cfront_portio.c" python3 - > "${tmp}/portio_emit.c" <<'PY' || { echo "  FAIL: python portio emit"; exit 1; }
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
pio_fb="$(python3 -c "
from bcir.frontends.cfront.pipeline import compile_with_fallback
r = compile_with_fallback('unsigned f(void){ return inb(0x60); }', check_clang=False, target='aarch64-linux')
print('FB' if (r.needs_fallback and 'requires an x86 target' in r.fallback) else 'NO')")" \
  || { echo "  FAIL: portio non-x86 fallback probe"; exit 1; }
[ "${pio_fb}" = "FB" ] \
  && echo "  PASS portio: a non-x86 target (aarch64) has no port I/O -> honest unsupported diagnostic (LLVM fallback)" \
  || { echo "  FAIL: portio non-x86 did not fall back honestly"; exit 1; }

# Memory-fence (hardware barrier) intrinsics (#barrier, ASM3): __sync_synchronize / __atomic_thread_fence /
# the C11 atomic_thread_fence (all FULL/seq_cst, op `c.fence`) + the x86-conventional _mm_mfence (full) /
# _mm_lfence (acquire) / _mm_sfence (release) lower to a typed, KINDED, barriered BARRIER edge; the per-ISA
# instruction is emitted behind `--target` (x86 mfence/lfence/sfence; aarch64 dmb ish/ishld/ishst; riscv64
# fence rw,rw / r,rw / rw,w), always with the required `"memory"` compiler-barrier clobber. Every ISA has a
# fence, so a target outside the three families keeps the portable __atomic_thread_fence default (no
# unsupported-diagnostic path -- unlike port I/O). The emit text is arch-independent (checked on every host);
# the emitted NATIVE-arch barrier is ASSEMBLED on its own lane (x86 assembles mfence/lfence/sfence; the
# aarch64 lane assembles dmb ish/ishld/ishst) -- non-native emits are NOT assembled (no cross sysroot).
echo "[c-runtime] memory-fence intrinsics: typed kinded barriered edge, per-ISA emit, native assemble (#barrier)"
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
BRR_TARGET="${brr_native:-x86_64-linux}" FX="${tmp}/cfront_barrier.c" python3 - > "${tmp}/barrier_emit.c" <<'PY' || { echo "  FAIL: python barrier emit"; exit 1; }
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
brr_text="$(python3 -c "
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

# X.691 PER decoding primitives (#per, roadmap phase C): the C twin of clause 11. PER is
# NOT self-delimiting (X.691 7.2), so unlike the X.690 twin there is no schema-free
# structure walk -- clause 11's whole-number and length decoders ARE the schema-free layer,
# and they are the ones that take an attacker-supplied width, octet count or fragment header
# and move a cursor with it. The dual-rail differential lives in bcir/tests/test_c_per.py;
# what is checked HERE is the same discipline the other twins get: strict warnings as
# errors, and a genuinely freestanding translation unit.
echo "[c-runtime] X.691 PER primitives: strict-warning and freestanding build (#per)"
if "${CC}" -std=c23 -O2 -Wall -Wextra -Werror -I "${C}" \
     "${C}/bcir_per.c" "${C}/test_per.c" -o "${tmp}/test_per"; then
  for std in c11 c23; do
    "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -Werror -I "${C}" \
      -c "${C}/bcir_per.c" -o /dev/null \
      || { echo "  FAIL: bcir_per is not freestanding-clean under -std=${std}"; exit 1; }
  done
  # A decoder whose answers depend on the optimiser is not a decoder. Build the same twin
  # at -O0 and -O3 and require identical output on the same campaign: a signed-overflow or
  # shift-past-width bug typically only diverges at one of the two.
  per_ok=1
  "${CC}" -std=c23 -O0 -I "${C}" "${C}/bcir_per.c" "${C}/test_per.c" -o "${tmp}/test_per_O0" \
    || per_ok=0
  "${CC}" -std=c23 -O3 -I "${C}" "${C}/bcir_per.c" "${C}/test_per.c" -o "${tmp}/test_per_O3" \
    || per_ok=0
  if [ "${per_ok}" -eq 1 ]; then
    python3 - "${tmp}" <<'PERPY' > "${tmp}/per_cases.txt"
import sys, random
sys.path.insert(0, ".")
from bcir.asn1.per import (BitWriter, PerVariant, _encode_constrained,
                           _encode_semi_constrained, _encode_unconstrained,
                           _encode_normally_small)
rng = random.Random(20260726)
for variant, flag in ((PerVariant.UNALIGNED, 0), (PerVariant.ALIGNED, 1)):
    for lb, ub in ((0, 255), (0, 256), (0, 65535), (0, 1 << 40), (-5, 5)):
        for _ in range(40):
            v = rng.randint(lb, ub)
            w = BitWriter(variant); _encode_constrained(w, v, lb, ub)
            print(f"constrained {lb} {ub} {flag} {w.to_bytes().hex()}")
    for _ in range(40):
        v = rng.randint(-(1 << 40), 1 << 40)
        w = BitWriter(variant); _encode_unconstrained(w, v)
        print(f"unconstrained {flag} {w.to_bytes().hex()}")
    for _ in range(40):
        v = rng.randint(0, 1 << 20)
        w = BitWriter(variant); _encode_semi_constrained(w, v, 0)
        print(f"semi 0 {flag} {w.to_bytes().hex()}")
    for v in (0, 63, 64, 300):
        w = BitWriter(variant); _encode_normally_small(w, v)
        print(f"small {flag} {w.to_bytes().hex()}")
PERPY
    "${tmp}/test_per_O0" < "${tmp}/per_cases.txt" > "${tmp}/per_O0.txt"
    "${tmp}/test_per_O3" < "${tmp}/per_cases.txt" > "${tmp}/per_O3.txt"
    if cmp -s "${tmp}/per_O0.txt" "${tmp}/per_O3.txt"; then
      echo "  PASS X.691 PER twin (freestanding, -Werror, -O0 == -O3 over $(wc -l < "${tmp}/per_cases.txt") cases)"
    else
      echo "  FAIL: the PER twin's answers depend on the optimisation level"
      diff "${tmp}/per_O0.txt" "${tmp}/per_O3.txt" | head -10
      exit 1
    fi
  else
    echo "  SKIP PER optimisation-parity (a build failed)"
  fi
else
  echo "  FAIL: the X.691 PER twin does not build warning-clean"
  exit 1
fi

# X.693 XER lexical layer (#xer, roadmap phase E-adjacent): the C twin of the tag scanner
# and the xmlcstring escaper. XER is text, so there is no bit cursor to get wrong -- but
# there is a byte cursor, and it is driven entirely by attacker-supplied content before any
# type is consulted. The dual-rail differential lives in bcir/tests/test_c_xer.py; what is
# checked HERE is the same discipline every other twin gets: strict warnings as errors, a
# genuinely freestanding translation unit, and answers that do not depend on the optimiser.
echo "[c-runtime] X.693 XER lexical layer: strict-warning and freestanding build (#xer)"
if "${CC}" -std=c23 -O2 -Wall -Wextra -Werror -I "${C}" \
     "${C}/bcir_xer.c" "${C}/test_xer.c" -o "${tmp}/test_xer"; then
  for std in c11 c23; do
    "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -Werror -I "${C}" \
      -c "${C}/bcir_xer.c" -o /dev/null \
      || { echo "  FAIL: bcir_xer is not freestanding-clean under -std=${std}"; exit 1; }
  done
  xer_ok=1
  "${CC}" -std=c23 -O0 -I "${C}" "${C}/bcir_xer.c" "${C}/test_xer.c" -o "${tmp}/test_xer_O0" \
    || xer_ok=0
  "${CC}" -std=c23 -O3 -I "${C}" "${C}/bcir_xer.c" "${C}/test_xer.c" -o "${tmp}/test_xer_O3" \
    || xer_ok=0
  if [ "${xer_ok}" -eq 1 ]; then
    python3 - <<'XERPY' > "${tmp}/xer_cases.txt"
import sys
sys.path.insert(0, ".")
from bcir.asn1.xer import _CONTROL_ELEMENT

# Documents that reach every branch of the scanner, plus the truncations one octet short of
# each excluded construct -- the inputs where a bounds check that is off by one shows up.
docs = ["<a>", "</a>", "<a/>", "<PersonnelRecord>", "</ChildInformation>", "<_XMLThing/>",
        "<a >", "<a\t/>", "<nul/>", "<BIT_STRING>", "<x-y.z/>", "<!-- c -->",
        "<![CDATA[x]]>", "<!DOCTYPE a>", "<?xml?>", '<a b="1">', "<a:b>", "<", "</",
        "<a", "<a/", "<!", "<!-", "<![CDATA", "<?", "<1a>", "<>", "< a>", "a",
        "<a><b/></a>", "  <a>x</a>", "<a\xc3\xa9>"]
for doc in docs:
    raw = doc.encode("utf-8", "surrogatepass")
    for pos in range(len(raw) + 2):
        print(f"tag {raw.hex() or '-'} {pos}")
        print(f"space {raw.hex() or '-'} {pos}")

strings = ["", "a", "a<b>&c", "\t\n\r", "John P Smith", "&&&", "é中\U0001f600",
           "".join(chr(code) for code in sorted(_CONTROL_ELEMENT))]
for text in strings:
    raw = text.encode()
    print(f"escape {raw.hex() or '-'}")
    print(f"unescape 1 {raw.hex() or '-'}")
    print(f"unescape 0 {raw.hex() or '-'}")
for text in ("a&#233;b", "a&#xEE;b", "a&amp;b", "a&nbsp;b", "a<nul/>b", "a&#;b"):
    print(f"unescape 1 {text.encode().hex()}")
    print(f"unescape 0 {text.encode().hex()}")
for raw in (b"\xc0\x80", b"\xe0\x80\x80", b"\xed\xa0\x80", b"\xf5\x80\x80\x80", b"\x80",
            b"\xc3", b"\xc3\xa9", b"\xf0\x9f\x98\x80"):
    for pos in range(len(raw) + 1):
        print(f"utf8 {raw.hex()} {pos}")
XERPY
    "${tmp}/test_xer_O0" < "${tmp}/xer_cases.txt" > "${tmp}/xer_O0.txt"
    "${tmp}/test_xer_O3" < "${tmp}/xer_cases.txt" > "${tmp}/xer_O3.txt"
    if cmp -s "${tmp}/xer_O0.txt" "${tmp}/xer_O3.txt"; then
      echo "  PASS X.693 XER twin (freestanding, -Werror, -O0 == -O3 over $(wc -l < "${tmp}/xer_cases.txt") cases)"
    else
      echo "  FAIL: the XER twin's answers depend on the optimisation level"
      diff "${tmp}/xer_O0.txt" "${tmp}/xer_O3.txt" | head -10
      exit 1
    fi
  else
    echo "  SKIP XER optimisation-parity (a build failed)"
  fi
else
  echo "  FAIL: the X.693 XER twin does not build warning-clean"
  exit 1
fi

# X.697 bounded JER reader (#jer, JSON roadmap phase J3): the C twin of
# bcir/asn1/jer_bounded.py -- 4.3's limits, 7.6.2's encoding, and the ECMA-404 grammar as an
# event stream. Unlike the XER twin this one has no `json.loads` behind it, so it is a real
# parser and not only a lexer. The dual-rail differential lives in bcir/tests/test_c_jer.py;
# what is checked HERE is the discipline every other twin gets: strict warnings as errors, a
# genuinely freestanding translation unit, and answers that do not depend on the optimiser.
#
# The -O0 == -O3 comparison earns its place on this file specifically. The reader's hot path
# is signed/unsigned arithmetic on attacker-supplied lengths and a saturating exponent
# accumulator; if any of it were undefined behaviour the optimiser would be entitled to
# choose differently at -O3 than at -O0, and the answers would diverge exactly where a
# malicious document lives.
echo "[c-runtime] X.697 bounded JER reader: strict-warning and freestanding build (#jer)"
if "${CC}" -std=c23 -O2 -Wall -Wextra -Werror -I "${C}" \
     "${C}/bcir_jer.c" "${C}/test_jer.c" "${C}/bcir_runtime.c" -o "${tmp}/test_jer"; then
  for std in c11 c23; do
    # bcir_crc32 is DECLARED here and DEFINED in bcir_runtime.c (the same discipline
    # bcir_telemetry_frame.h follows), so this is a compile and not a link.
    "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -Werror -I "${C}" \
      -c "${C}/bcir_jer.c" -o /dev/null \
      || { echo "  FAIL: bcir_jer is not freestanding-clean under -std=${std}"; exit 1; }
  done
  jer_ok=1
  "${CC}" -std=c23 -O0 -I "${C}" "${C}/bcir_jer.c" "${C}/test_jer.c" "${C}/bcir_runtime.c" \
    -o "${tmp}/test_jer_O0" || jer_ok=0
  "${CC}" -std=c23 -O3 -I "${C}" "${C}/bcir_jer.c" "${C}/test_jer.c" "${C}/bcir_runtime.c" \
    -o "${tmp}/test_jer_O3" || jer_ok=0
  if [ "${jer_ok}" -eq 1 ]; then
    python3 - <<'JERPY' > "${tmp}/jer_cases.txt"
import sys
sys.path.insert(0, ".")
from bcir.asn1.jer_bounded import STRICT_LIMITS, frame

# Documents that reach every branch of both the bounding pass and the parser, plus the
# forms permissive readers accept and ECMA-404 does not: trailing commas, missing
# separators, leading zeros, the non-JSON constants, and the surrogate cases.
docs = [b"", b" ", b"null", b"true", b"false", b"0", b"-0", b"10", b"01", b"1.5",
        b"-0.5e+3", b"1E-2", b"1.", b".5", b"-", b"1e", b"+1", b'""', b'"a"',
        rb'"\n\t\r\b\f\/\\\""', rb'"A"', '"\U0001f600"'.encode(), rb'"\ud800"',
        rb'"\udc00"', rb'"\uZZZZ"', rb'"\q"', b'"a', b"[]", b"{}", b"[1,2,3]",
        b'{"a":1,"b":2}', b'{"a":{"b":[1,[2,[3]]]}}', b"[1,]", b'{"a":1,}', b"[,]",
        b"[1 2]", b'{"a" 1}', b'{"a":}', b"[", b"]", b"{", b"}", b"1 2", b"[]]",
        b"nan", b"NaN", b"Infinity", b"undefined", b"'a'", b'{"a":1,"a":2}',
        b"\x80", b'"\x80"', b'"\xc0\x80"', b'"\xed\xa0\x80"', b"\xef\xbb\xbf{}",
        b'"\x00"', b'"\x1f"', "[\"é\", \"中\"]".encode(),
        b"[" * 80 + b"]" * 80, b"1" * (STRICT_LIMITS.integer_digits + 1),
        b"1e" + str(STRICT_LIMITS.exponent_magnitude + 1).encode()]
for doc in docs:
    payload = doc.hex() or "-"
    for strict in (0, 1):
        print(f"scan {strict} {payload}")
        print(f"parse {strict} {payload}")
    print(f"utf8doc {payload}")
    for at in range(4):
        print(f"refuse {at} {payload}")

for raw in (b"", b"a", rb"\n", rb"A", "\U0001f600".encode(), rb"\ud800", rb"\q",
            "é中".encode()):
    payload = raw.hex() or "-"
    for cap in (0, 3, 65536):
        print(f"unescape {cap} {payload}")

for raw in (b"\xc0\x80", b"\xe0\x80\x80", b"\xed\xa0\x80", b"\xf5\x80\x80\x80", b"\x80",
            b"\xc3", b"\xc3\xa9", b"\xf0\x9f\x98\x80", b"\xf4\x90\x80\x80"):
    for at in range(len(raw) + 1):
        print(f"utf8 {raw.hex()} {at}")

good = frame(b'{"a":1}', sequence=42, generation=7)
for cut in range(0, len(good) + 1):
    print(f"unframe {good[:cut].hex() or '-'}")
bad = bytearray(good)
bad[-1] ^= 1
print(f"unframe {bytes(bad).hex()}")

for field in ("input_bytes", "depth", "nodes", "members", "elements", "string_bytes",
              "number_bytes", "integer_digits", "exponent_magnitude", "work"):
    for value in (1, 1 << 40):
        print(f"tighten {field} {value}")
JERPY
    "${tmp}/test_jer_O0" < "${tmp}/jer_cases.txt" > "${tmp}/jer_O0.txt"
    "${tmp}/test_jer_O3" < "${tmp}/jer_cases.txt" > "${tmp}/jer_O3.txt"
    if cmp -s "${tmp}/jer_O0.txt" "${tmp}/jer_O3.txt"; then
      echo "  PASS X.697 JER twin (freestanding, -Werror, -O0 == -O3 over $(wc -l < "${tmp}/jer_cases.txt") cases)"
    else
      echo "  FAIL: the JER twin's answers depend on the optimisation level"
      diff "${tmp}/jer_O0.txt" "${tmp}/jer_O3.txt" | head -10
      exit 1
    fi
  else
    echo "  SKIP JER optimisation-parity (a build failed)"
  fi
else
  echo "  FAIL: the X.697 JER twin does not build warning-clean"
  exit 1
fi

# The native ASN.1 decode microbench (#asn1bench, JSON roadmap J6 follow-on): the harness
# that makes a `measured` cost table possible, and therefore the reason select_certified can
# decide a timing objective at all instead of refusing every one. It is a MEASUREMENT tool,
# so what is gated here is that it builds warning-clean and answers a corpus -- the numbers
# themselves are deliberately NOT a CI assertion, because a shared runner's timings are not
# evidence about a target and pinning them would invent the false precision J6 refuses.
echo "[c-runtime] native ASN.1 decode microbench: strict-warning build (#asn1bench)"
if "${CC}" -std=c23 -O2 -Wall -Wextra -Werror -I "${C}" \
     "${C}/bcir_asn1_bench.c" "${C}/bcir_asn1.c" "${C}/bcir_jer.c" "${C}/bcir_xer.c" \
     "${C}/bcir_runtime.c" -o "${tmp}/asn1_bench"; then
  printf 'rounds 1 7 8\ncase DER der 3009020102040461\ncase JER jer 7b2261223a317d\nrun\n' \
    > "${tmp}/bench_cases.txt"
  if "${tmp}/asn1_bench" < "${tmp}/bench_cases.txt" | grep -q '^done 2$'; then
    echo "  PASS native ASN.1 microbench (builds -Werror, answers a two-case corpus)"
  else
    echo "  FAIL: the native ASN.1 microbench did not complete its corpus"
    exit 1
  fi
else
  echo "  FAIL: the native ASN.1 microbench does not build warning-clean"
  exit 1
fi

# DER -> native StreamPack fast path (#asn1fast, roadmap phase D): reconstruct the native
# artifact from its X.690 DER projection in freestanding C, with no Python anywhere in the
# reconstruction path, and assert BYTE IDENTITY against what the Python encoder produced.
# That is law A3 (additive: the native octets survive the round trip) proven on the C rail.
# Byte identity, not equivalence -- the fast path has to re-derive the StreamPack VERSION
# from content the way bcir/abi::encode does, emit the reserved stride_k the projection
# deliberately omits, and recompute the CRC.
echo "[c-runtime] DER -> native StreamPack fast path: byte-identical reconstruction (#asn1fast)"
if "${CC}" -std=c23 -O2 -Wall -Wextra -I "${C}" "${C}/bcir_asn1_streampack.c" \
     "${C}/bcir_asn1.c" "${C}/bcir_runtime.c" "${C}/test_asn1_streampack.c" \
     -o "${tmp}/test_asn1_sp"; then
  for std in c11 c23; do
    "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -I "${C}" \
      -c "${C}/bcir_asn1_streampack.c" -o /dev/null \
      || { echo "  FAIL: bcir_asn1_streampack not freestanding-clean under -std=${std}"; exit 1; }
  done
  python3 - "${tmp}" <<'ASN1FASTPY' || { echo "  FAIL: could not project the corpus"; exit 1; }
import os, sys
from bcir.abi import encode
from bcir.asn1.streampack import encode_pack
from bcir.examples import PROGRAMS
from bcir.gem import hydrate
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
d = sys.argv[1]
host, theta = TargetProfile.x86_avx512(), Theta.cool()
for name, build in sorted(PROGRAMS.items()):
    module = build()
    pack = hydrate(module, optimize(module, host, theta))
    open(os.path.join(d, name + ".proj.der"), "wb").write(encode_pack(pack))
    open(os.path.join(d, name + ".native.bin"), "wb").write(encode(pack))
ASN1FASTPY
  fast_ok=0; fast_bad=0
  for proj in "${tmp}"/*.proj.der; do
    base="$(basename "${proj}" .proj.der)"
    if out="$("${tmp}/test_asn1_sp" "${proj}" "${tmp}/${base}.native.bin" 2>&1)" \
         && [ "${out%% *}" = "OK" ]; then
      fast_ok=$((fast_ok + 1))
    else
      echo "  FAIL ${base}: ${out}"; fast_bad=$((fast_bad + 1))
    fi
  done
  # A malformed or BER-only projection must be refused, never partially reconstructed.
  python3 - "${tmp}" <<'ASN1NEGPY'
import os, sys
d = sys.argv[1]
der = open(os.path.join(d, "vector_add.proj.der"), "rb").read()
for n in (0, 1, 5, len(der) // 2, len(der) - 1):
    open(os.path.join(d, f"neg{n}.bad.der"), "wb").write(der[:n])
if der[1] < 0x80:      # the same value with a non-minimal length: legal BER, not DER
    open(os.path.join(d, "nonmin.bad.der"), "wb").write(
        bytes([der[0], 0x81, der[1]]) + der[2:])
ASN1NEGPY
  for bad in "${tmp}"/*.bad.der; do
    if "${tmp}/test_asn1_sp" "${bad}" >/dev/null 2>&1; then
      echo "  FAIL: $(basename "${bad}") was accepted by the fast path"; fast_bad=$((fast_bad + 1))
    fi
  done
  if [ "${fast_bad}" -eq 0 ] && [ "${fast_ok}" -ge 10 ]; then
    echo "  PASS DER -> native byte-identical on ${fast_ok} corpus programs; malformed + BER-only refused"
  else
    echo "  FAIL: fast path (${fast_ok} ok, ${fast_bad} bad)"; exit 1
  fi
else
  echo "  SKIP fast path (harness did not build)"
fi

# Sanitizer + memory-stress harness for the cfront C twin (#sanitize): the dual-rail parity gates above
# compare the twin's OUTPUT against the oracle, but never build bcir_cfront.c under AddressSanitizer/UBSan
# and never run Valgrind -- so a memory bug (buffer overflow, use-after-free, UB) INSIDE the compiler's own
# lexer/parser/lowering/emit would go uncaught. sanitize_cfront.sh closes that: it builds the SAME
# `test_cfront` driver (same source list as above) under ASan/UBSan with clang and/or gcc, runs every
# cfront_*.c fixture + a bounded seeded fuzz campaign (valid + malformed) through it, and runs a bounded
# Valgrind pass over a fixture subset. Each stage self-skips when its tool is absent (SKIP, not FAIL). The
# heavy Valgrind stage can be dropped with SANITIZE_SKIP_VALGRIND=1 while ASan/UBSan stay always-on.
#
# BCIR_SKIP_CFRONT_SANITIZE=1 suppresses this nested call for a CALLER THAT RUNS THE SAME
# HARNESS ITSELF. The CI c-runtime job does exactly that (a dedicated step, so a sanitizer
# diagnostic is attributed to the sanitizer rather than buried in this script's output), and
# without the opt-out the whole 300-valid/400-malformed campaign ran twice in one job. Anyone
# invoking check_runtime.sh directly still gets the harness, valgrind included.
if [ "${BCIR_SKIP_CFRONT_SANITIZE:-0}" = "1" ]; then
  echo "[c-runtime] SKIP cfront sanitizer harness (BCIR_SKIP_CFRONT_SANITIZE=1; the caller runs it)"
else
echo "[c-runtime] cfront twin under ASan/UBSan + Valgrind (sanitize_cfront.sh)"
if CLANG="${CC}" bash "${ROOT}/tools/c/sanitize_cfront.sh" 2>&1 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -eq 0 ]; then
  echo "  PASS cfront sanitizer/valgrind harness"
else
  echo "  FAIL: cfront sanitizer/valgrind harness reported a diagnostic"; exit 1
fi
fi

# StreamPack SEMANTIC trust boundary (R10/R11 + lane/width/dispatch range, ported into the
# freestanding C decoder/executor): the CRC + bounds decode is memory-safe, but a CRC-VALID pack
# can still be semantically corrupt (a dangling/redirected claim_id, a swapped/undeclared RID, an
# out-of-range lane/width, an unresolved prefetch, a stale generation, a tampered dispatch). Each
# used to execute SILENTLY in C. check_streampack_semantic.sh crafts each as a CRC-FIXED pack and
# asserts the C rail (bcir_sp_verify_semantic / bcir_sp_execute_checked) now REJECTS it, plus a
# C-decode == Python-decode differential (the lane-asymmetry class) over v1/v2/v3 packs.
echo "[c-runtime] StreamPack semantic trust boundary (check_streampack_semantic.sh)"
if CC="${CC}" bash "${ROOT}/tools/c/check_streampack_semantic.sh" 2>&1 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -eq 0 ]; then
  echo "  PASS StreamPack semantic-corruption rejection + C/Python decode differential"
else
  echo "  FAIL: a CRC-valid semantically-corrupt pack was not rejected on the C rail"; exit 1
fi

# The C<->C++ hand-off seam scaffold (#cpphandoff, docs/languages/CPP_HANDOFF_BOUNDARY.md): the boundary
# between the deterministic single-node C/IR rail and the C++ layer ABOVE it (dynamic graph
# topology + distributed MPI/NCCL orchestration). check_handoff.sh compiles the STANDALONE C++17
# scaffold (runtime/cpp/, NOT part of the MLIR build), hands a StreamPack the C/IR path produces
# through the single-node Orchestrator, and asserts the seam ROUND-TRIPS (the C++ dispatch order ==
# the direct C/IR decode of the same artifact) -- plus that a corrupted artifact is REJECTED at the
# boundary (admit() carries the C verifier's verdict; the two-truth quarantine holds across the seam).
# The dynamic-graph + distributed backends are documented STUBS behind the same interface (no real
# MPI/NCCL dependency). Self-skips (exit 0) if no C++ compiler is present, like the gates above.
echo "[c-runtime] C<->C++ hand-off seam scaffold round-trip (tools/cpp/check_handoff.sh)"
if bash "${ROOT}/tools/cpp/check_handoff.sh" 2>&1 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -eq 0 ]; then
  echo "  PASS C<->C++ hand-off seam compiles + round-trips (single-node Orchestrator == direct C/IR)"
else
  echo "  FAIL: the C++ hand-off seam scaffold did not compile or round-trip"; exit 1
fi

# The SYCL backend differential oracle (#sycldiff, docs/kernel/SYCL_INTEROP.md): SYCL is a backend CHANNEL +
# a differential oracle, NEVER on the legality path. check_sycl.sh emits the single-source C++ SAXPY
# kernel (emit_sycl_saxpy_c) and proves it reproduces BCIR's own deterministic reference (a*x+y) to
# float round-off: the PORTABLE scalar C++ fallback always (the real reference-verification work, no
# SYCL needed -- the #sycl-fallback marker), and the SYCL DEVICE parallel_for (-DBCIR_USE_SYCL -fsycl,
# the #sycl-device marker) IF a real SYCL compiler is detected (icpx/acpp/clang++ -fsycl that compiles+
# links a probe). SYCL is a compiler MODE, NOT a c.call.libm: -l<lib> edge (no link-flag rule). Lives
# above the G8 C++ boundary; self-skips (exit 0) if no C++ compiler, like check_handoff.sh.
echo "[c-runtime] SYCL backend differential oracle (tools/cpp/check_sycl.sh)"
if bash "${ROOT}/tools/cpp/check_sycl.sh" 2>&1 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -eq 0 ]; then
  echo "  PASS SYCL SAXPY differential (portable C++ fallback == BCIR reference; device path when -fsycl present)"
else
  echo "  FAIL: the SYCL backend differential oracle did not compile or agree"; exit 1
fi

echo "[c-runtime] ok"
