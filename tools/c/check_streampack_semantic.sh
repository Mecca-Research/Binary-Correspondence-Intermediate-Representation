#!/usr/bin/env bash
# Adversarial semantic trust-boundary gate for the freestanding StreamPack C rail.
#
# The CRC + bounds-checked decode (bcir_runtime.c) is memory-safe, but a CRC-VALID pack can
# still be SEMANTICALLY corrupt: a dangling/redirected claim_id, a swapped/undeclared RID, an
# out-of-range lane/width, an unresolved prefetch, invalid pipeline metadata, a stale
# generation, a tampered dispatch, or undeclared trailing bytes.
# Such a pack used to execute SILENTLY in C (Python's verify_pack R10/R11 was the only rail).
# This gate crafts each corruption as a CRC-FIXED pack (tools/c/streampack_corrupt.py) and
# asserts the C semantic verifier (bcir_sp_verify_semantic) + the checked executor
# (bcir_sp_execute_checked) now REJECT each with a clean BCIR_ERR_*, and a C-decode ==
# Python-decode differential agrees on every decoded field (the lane-asymmetry class).
#
# StreamPack v4 (S0-2, R11 per resource): the per-resource generation vector is checked
# against a live registry table (`--live rid:map:data ...`) through
# bcir_sp_check_generation_vector + bcir_sp_execute_checked_vector. The stale-vector kinds
# keep the header maxima intact, so the legacy maxima check (bcir_sp_check_generation)
# accepts them -- the witness of the blind spot the vector closes -- while the per-resource
# check refuses them; a vector-less v1-v3 artifact is stale against any registry that
# declares resources, and a vector is stale against an empty one.
#
# Wired into tools/c/check_runtime.sh. Standalone:  bash tools/c/check_streampack_semantic.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C="${ROOT}/runtime/c"
CC="${CC:-$(command -v clang || command -v cc || true)}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [ -z "${CC}" ]; then
  echo "[streampack-sem] no C compiler (clang/cc); skipping." >&2
  exit 0
fi

tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

echo "[streampack-sem] build the semantic harness (C23) + the decode-differential harness"
"${CC}" -std=c23 -O2 -Wall -Wextra -I "${C}" \
  "${C}/bcir_exec.c" "${C}/bcir_runtime.c" "${C}/bcir_verify.c" \
  "${C}/test_streampack_semantic.c" -o "${tmp}/test_sem" \
  || { echo "  FAIL: semantic harness build"; exit 1; }
"${CC}" -std=c23 -O2 -Wall -Wextra -I "${C}" \
  "${C}/bcir_runtime.c" "${C}/test_runtime.c" -o "${tmp}/test_runtime" \
  || { echo "  FAIL: decode harness build"; exit 1; }

# A real pack still validates + executes cleanly (the gate has no false positives).
python3 -c "
from bcir.examples import vector_add
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.gem import hydrate
from bcir.abi import encode
m=vector_add(1024); pack=hydrate(m, optimize(m, TargetProfile.x86_avx512(), Theta.cool()))
open('${tmp}/good.bin','wb').write(encode(pack))
" || { echo "  FAIL: python encode (good pack)"; exit 1; }
good="$("${tmp}/test_sem" "${tmp}/good.bin")"
echo "${good}" | grep -q "^status=0 name=BCIR_OK" \
  && echo "  PASS clean pack accepted (status=BCIR_OK, no false positive)" \
  || { echo "  FAIL: clean pack rejected"; echo "${good}"; exit 1; }
# ... and against its own live registry the v4 per-resource check is clean too.
good_live="$(python3 -c "
from bcir.examples import vector_add
m=vector_add(1024)
print(' '.join(f'{r.rid}:{r.map_gen}:{r.data_gen}' for r in sorted(m.resources.values(), key=lambda r: r.rid)))
")" || { echo "  FAIL: python live table"; exit 1; }
# shellcheck disable=SC2086
good="$("${tmp}/test_sem" "${tmp}/good.bin" --live ${good_live})"
echo "${good}" | grep -q "^vector=BCIR_OK" && echo "${good}" | grep -q "^exec_vector=BCIR_OK" \
  && echo "  PASS clean v4 pack accepted against its live registry (vector=BCIR_OK, no false positive)" \
  || { echo "  FAIL: clean v4 pack rejected against its live registry"; echo "${good}"; exit 1; }

# Map each corruption to the BCIR_ERR_* the C rail must now return.
declare -A WANT=(
  [dangling_claim]=BCIR_ERR_PROVENANCE
  [swapped_rid]=BCIR_ERR_PROVENANCE
  [out_of_range_lane]=BCIR_ERR_LANE
  [out_of_range_width]=BCIR_ERR_WIDTH
  [unresolved_prefetch]=BCIR_ERR_PROVENANCE
  [invalid_pipeline]=BCIR_ERR_PROVENANCE
  [invalid_buffers]=BCIR_ERR_PROVENANCE
  [stale_generation]=BCIR_ERR_STALE
  [tampered_dispatch]=BCIR_ERR_DISPATCH
  [trailing_bytes]=BCIR_ERR_TRAILING
  [reserved_flags]=BCIR_ERR_RESERVED
  [reserved_header]=BCIR_ERR_RESERVED
  [reserved_stride]=BCIR_ERR_RESERVED
  [invalid_utf8]=BCIR_ERR_UTF8
  [duplicate_segment]=BCIR_ERR_PROVENANCE
  [duplicate_prefetch]=BCIR_ERR_PROVENANCE
  [duplicate_trace]=BCIR_ERR_PROVENANCE
)
declare -A WANT_VALIDATE=(
  [reserved_flags]=BCIR_ERR_RESERVED
  [reserved_header]=BCIR_ERR_RESERVED
)
ORDER="dangling_claim swapped_rid out_of_range_lane out_of_range_width unresolved_prefetch invalid_pipeline invalid_buffers stale_generation tampered_dispatch trailing_bytes reserved_flags reserved_header reserved_stride invalid_utf8 duplicate_segment duplicate_prefetch duplicate_trace"
# StreamPack v4 kinds: the semantic verdict (no registry) and the per-resource verdict
# (`--live`). (l) kinds are well-formed and STALE only against the registry; (m) kinds are
# malformed vectors refused before any registry is consulted (the same code on both).
declare -A WANT_V4_SEM=(
  [stale_vector]=BCIR_OK
  [missing_vector_entry]=BCIR_OK
  [undeclared_vector_rid]=BCIR_OK
  [unsorted_vector]=BCIR_ERR_GENERATION
  [duplicate_vector_rid]=BCIR_ERR_GENERATION
  [vector_maxima_mismatch]=BCIR_ERR_GENERATION
  [truncated_vector]=BCIR_ERR_TRUNCATED
  [reserved_n_gens]=BCIR_ERR_RESERVED
)
declare -A WANT_V4_VECTOR=(
  [stale_vector]=BCIR_ERR_STALE
  [missing_vector_entry]=BCIR_ERR_STALE
  [undeclared_vector_rid]=BCIR_ERR_STALE
  [unsorted_vector]=BCIR_ERR_GENERATION
  [duplicate_vector_rid]=BCIR_ERR_GENERATION
  [vector_maxima_mismatch]=BCIR_ERR_GENERATION
  [truncated_vector]=BCIR_ERR_TRUNCATED
  [reserved_n_gens]=BCIR_ERR_RESERVED
)
ORDER_V4="stale_vector missing_vector_entry undeclared_vector_rid unsorted_vector duplicate_vector_rid vector_maxima_mismatch truncated_vector reserved_n_gens"

fail=0
for kind in ${ORDER}; do
  meta="$(python3 "${C}/../../tools/c/streampack_corrupt.py" "${kind}" "${tmp}/${kind}.bin")" \
    || { echo "  FAIL: could not craft ${kind}"; fail=1; continue; }
  mapg="$(echo "${meta}" | awk '{print $3}')"; datag="$(echo "${meta}" | awk '{print $4}')"
  # stale only trips when the C rail is told the live generation; the rest skip the gen check.
  if [ "${kind}" = "stale_generation" ]; then
    out="$("${tmp}/test_sem" "${tmp}/${kind}.bin" "${mapg}" "${datag}")"; rc=$?
  else
    out="$("${tmp}/test_sem" "${tmp}/${kind}.bin")"; rc=$?
  fi
  crc="$(echo "${out}"  | sed -n 's/^crc=//p')"
  name="$(echo "${out}" | sed -n 's/^status=[0-9]* name=//p')"
  exec="$(echo "${out}" | sed -n 's/^exec=//p')"
  # Header-reserved corruptions are rejected by the structural validator itself;
  # record-level corruptions still pass that first gate and fail semantic validation.
  want_validate="${WANT_VALIDATE[$kind]:-BCIR_OK}"
  [ "${crc}" = "${want_validate}" ] || { echo "  FAIL: ${kind} validation want ${want_validate}, got ${crc}"; fail=1; continue; }
  # ... and the semantic verifier + the checked executor MUST both reject it with the right code.
  if [ "${name}" = "${WANT[$kind]}" ] && [ "${exec}" = "${WANT[$kind]}" ] && [ "${rc}" = "1" ]; then
    echo "  PASS ${kind}: CRC-valid but REJECTED (${name}); executor refuses to run it (${exec})"
  else
    echo "  FAIL ${kind}: want ${WANT[$kind]} got verify=${name} exec=${exec} (rc=${rc})"; fail=1
  fi
done
[ "${fail}" = "0" ] || { echo "[streampack-sem] FAIL: a corrupt pack was not rejected"; exit 1; }

echo "[streampack-sem] StreamPack v4: R11 per resource against the live registry (--live)"
for kind in ${ORDER_V4}; do
  meta="$(python3 "${C}/../../tools/c/streampack_corrupt.py" "${kind}" "${tmp}/${kind}.bin")" \
    || { echo "  FAIL: could not craft ${kind}"; fail=1; continue; }
  mapg="$(echo "${meta}" | awk '{print $3}')"; datag="$(echo "${meta}" | awk '{print $4}')"
  live="$(echo "${meta}" | awk '{print $5}' | sed 's/^live=//; s/,/ /g')"
  # shellcheck disable=SC2086
  out="$("${tmp}/test_sem" "${tmp}/${kind}.bin" --live ${live})"; rc=$?
  crc="$(echo "${out}"  | sed -n 's/^crc=//p')"
  name="$(echo "${out}" | sed -n 's/^status=[0-9]* name=//p')"
  vec="$(echo "${out}"  | sed -n 's/^vector=//p')"
  exv="$(echo "${out}"  | sed -n 's/^exec_vector=//p')"
  want_validate="BCIR_OK"; [ "${kind}" = "reserved_n_gens" ] && want_validate="BCIR_ERR_RESERVED"
  [ "${crc}" = "${want_validate}" ] || { echo "  FAIL: ${kind} validation want ${want_validate}, got ${crc}"; fail=1; continue; }
  if [ "${name}" = "${WANT_V4_SEM[$kind]}" ] && [ "${vec}" = "${WANT_V4_VECTOR[$kind]}" ] \
     && [ "${exv}" = "${WANT_V4_VECTOR[$kind]}" ] && [ "${rc}" = "1" ]; then
    echo "  PASS ${kind}: semantic=${name}, per-resource R11 REJECTS (${vec}); vector executor refuses (${exv})"
  else
    echo "  FAIL ${kind}: want sem=${WANT_V4_SEM[$kind]} vector=${WANT_V4_VECTOR[$kind]} got sem=${name} vector=${vec} exec_vector=${exv} (rc=${rc})"; fail=1
  fi
  # The witness of the closed blind spot: the (l) kinds keep the header maxima intact, so
  # the legacy maxima-only check accepts them; only the vector sees the moved resource.
  if [ "${WANT_V4_SEM[$kind]}" = "BCIR_OK" ]; then
    legacy="$("${tmp}/test_sem" "${tmp}/${kind}.bin" "${mapg}" "${datag}" | sed -n 's/^exec=//p')"
    [ "${legacy}" = "BCIR_OK" ] \
      && echo "  PASS ${kind}: invisible to the maxima-only check (legacy exec=${legacy}) -- the vector closes it" \
      || { echo "  FAIL ${kind}: expected the maxima-only check to accept it, got ${legacy}"; fail=1; }
  fi
done
# A vector-less (v1-v3) artifact is stale against any registry that declares resources,
# and a vector is stale against an empty registry; neither is silently executed.
python3 -c "
from tools.c.streampack_corrupt import base_pack, base_pack_v4, LIVE_ARG
from bcir.abi import encode
open('${tmp}/vectorless.bin','wb').write(encode(base_pack()))
open('${tmp}/vector.bin','wb').write(encode(base_pack_v4()))
print(LIVE_ARG.replace(',', ' '))
" > "${tmp}/live.txt" || { echo "  FAIL: python v4 fixtures"; exit 1; }
# shellcheck disable=SC2046
vec="$("${tmp}/test_sem" "${tmp}/vectorless.bin" --live $(cat "${tmp}/live.txt") | sed -n 's/^exec_vector=//p')"
[ "${vec}" = "BCIR_ERR_STALE" ] \
  && echo "  PASS vectorless v1-v3 artifact against a live registry: ${vec} (rehydrate, never execute)" \
  || { echo "  FAIL: vectorless artifact against a live registry got ${vec}"; fail=1; }
vec="$("${tmp}/test_sem" "${tmp}/vector.bin" --live | sed -n 's/^exec_vector=//p')"
[ "${vec}" = "BCIR_ERR_STALE" ] \
  && echo "  PASS v4 vector against an empty registry: ${vec}" \
  || { echo "  FAIL: v4 vector against an empty registry got ${vec}"; fail=1; }
# shellcheck disable=SC2046
vec="$("${tmp}/test_sem" "${tmp}/vector.bin" --live $(cat "${tmp}/live.txt") | sed -n 's/^exec_vector=//p')"
[ "${vec}" = "BCIR_OK" ] \
  && echo "  PASS v4 vector against its own registry: ${vec}" \
  || { echo "  FAIL: v4 vector against its own registry got ${vec}"; fail=1; }
[ "${fail}" = "0" ] || { echo "[streampack-sem] FAIL: a stale v4 pack was not rejected"; exit 1; }

# --- C-decode == Python-decode differential (catches the lane-asymmetry class) ---------------
# Decode the SAME bytes with the C decoder (test_runtime) and the Python decode; assert they
# agree on the decoded fields. A v3 pack exercises dispatch/channel on both rails; a v4
# pack (v3 features + the per-resource generation vector) the vector records.
echo "[streampack-sem] C-decode == Python-decode differential (v1 + v2 + v3 + v4 packs)"
diff_fail=0
for variant in v1 v2 v3 v4; do
  python3 -c "
import sys
from bcir.abi import encode
from bcir.gem.streampack import StreamPack, LaneSegment, Prefetch, TraceNote, Generation
from bcir.model import Lane
v='${variant}'
p=StreamPack(source_plan='plan0', topo_gen=1, map_gen=7, data_gen=19)
if v=='v2': p.pipeline_depth=2
p.prefetches.append(Prefetch('pf0',4,(10,11),'T0','linear', buffers=(2 if v=='v2' else 1)))
disp='pim' if v in ('v3','v4') else 'core'; chan='nvidia_ptx' if v in ('v3','v4') else 'host'
op='reduce.add' if v in ('v3','v4') else 'f32.add'
p.segments.append(LaneSegment(name='seg0', claim_id=1000, phase_id=0, lane=Lane.GGG, width=16,
    opcode=op, reads=(10,11), writes=(12,), prefetch='pf0', dispatch=disp, channel=chan))
p.trace_notes.append(TraceNote(claim_id=1000))
if v=='v4': p.generations=[Generation(10,7,2), Generation(11,1,19), Generation(12,0,0)]
open('${tmp}/${variant}.bin','wb').write(encode(p))
" || { echo "  FAIL: encode ${variant}"; diff_fail=1; continue; }
  cdec="$("${tmp}/test_runtime" "${tmp}/${variant}.bin")" || { echo "  FAIL: C decode ${variant}"; diff_fail=1; continue; }
  pydec="$(python3 -c "
from bcir.abi import decode
p=decode(open('${tmp}/${variant}.bin','rb').read())
s=p.segments[0]
lines=[]
hdr_ver={'v1':1,'v2':2,'v3':3,'v4':4}['${variant}']
lines.append(f'version={hdr_ver}')
lines.append(f'pipeline_depth={p.pipeline_depth}')
lines.append(f'topo_gen={p.topo_gen}')
lines.append(f'map_gen={p.map_gen}')
lines.append(f'data_gen={p.data_gen}')
lines.append(f'n_segments={len(p.segments)}')
lines.append(f'seg0.claim_id={s.claim_id}')
lines.append(f'seg0.phase_id={s.phase_id}')
lines.append(f'seg0.lane={int(s.lane)}')
lines.append(f'seg0.width={s.width}')
lines.append(f'seg0.opcode={s.opcode}')
lines.append(f'seg0.n_reads={len(s.reads)}')
lines.append(f'seg0.read0={s.reads[0]}')
lines.append(f'seg0.write0={s.writes[0]}')
disp={'core':0,'pim':1}[s.dispatch]
lines.append(f'seg0.dispatch={disp}')
lines.append(f'seg0.channel={s.channel}')
lines.append(f'seg0.prefetch={s.prefetch or \"\"}')
lines.append(f'n_gens={len(p.generations)}')
lines.append(f'gens_walked={len(p.generations)}')
if p.generations:
    g=p.generations[0]
    lines.append(f'gen0.rid={g.rid}')
    lines.append(f'gen0.map_gen={g.map_gen}')
    lines.append(f'gen0.data_gen={g.data_gen}')
print(chr(10).join(lines))
")"
  # Compare each Python field line against the C output (C prints a superset incl. 'walked'/'OK').
  mismatch=0
  while IFS= read -r line; do
    [ -z "${line}" ] && continue
    echo "${cdec}" | grep -qxF "${line}" || { echo "    DIFF ${variant}: C decode missing '${line}'"; mismatch=1; }
  done <<< "${pydec}"
  if [ "${mismatch}" = "0" ]; then
    echo "  PASS ${variant}: C decode == Python decode on all fields (lane/dispatch/channel/generations agree)"
  else
    echo "  FAIL ${variant}: C/Python decode differential mismatch"; diff_fail=1
  fi
done
[ "${diff_fail}" = "0" ] || { echo "[streampack-sem] FAIL: decode differential mismatch"; exit 1; }

echo "[streampack-sem] ok"
