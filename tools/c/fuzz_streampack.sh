#!/usr/bin/env bash
# Fuzz every BINARY TRUST BOUNDARY in runtime/c/ under libFuzzer + ASan/UBSan, and run an
# ASan/UBSan smoke on a real Python-encoded pack + byte mutations. Needs clang with
# compiler-rt (libclang-rt-NN-dev). FUZZ_RUNS controls the fuzz iteration count.
#
# The nine harnesses, and whose bytes each one distrusts:
#   StreamPack decoder / executor / encoder  -- an artifact handed to the runtime
#   ETL binary-record decoder                -- device/driver packet fields
#   telemetry-frame decoder                  -- frames a device emits over UART
#   BCIRQ8 model loader                      -- an external model artifact (LangRef 16)
#   BCAB artifact-bundle reader              -- a multi-backend artifact from a peer
#   X.690 BER/DER decoder                    -- an ASN.1 artifact from a foreign peer
#   DER -> native StreamPack fast path       -- a projection a peer hands a driver
#
# Each seeds from real Python-rail output so the campaign starts inside the format rather
# than rediscovering its magic; the BCIRQ8 harness additionally REPAIRS the format's three
# checksum layers, without which a coverage-guided fuzzer only ever proves that the CRC
# check works and never reaches the parser behind it.
#
# WALL TIME. Every campaign is bounded by `-max_total_time` (wall clock), so running them
# one after another made this gate cost the SUM of those bounds -- ~7 minutes for work that
# uses one core. The harnesses are independent processes over independent corpora, so they
# run CONCURRENTLY here, bounded by FUZZ_JOBS (default: the core count). Nothing about the
# campaigns changes -- same FUZZ_RUNS, same per-harness time bound, same seeds -- so the
# coverage is identical and only the scheduling differs. They are started LONGEST-FIRST
# (measured: BCIRQ8 and telemetry-frame saturate the time bound; binrec finishes its runs in
# ~1s), so the long poles own the machine while the short ones drain.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C="${ROOT}/runtime/c"
CLANG="${CLANG:-$(command -v clang || true)}"
RUNS="${FUZZ_RUNS:-200000}"
MAX_TIME="${FUZZ_MAX_TOTAL_TIME:-60}"
JOBS="${FUZZ_JOBS:-$( (command -v nproc >/dev/null && nproc) || echo 4 )}"
if [ -z "${CLANG}" ]; then
  echo "no clang; skipping StreamPack fuzz." >&2
  exit 0
fi
# compiler-rt (the sanitizer/fuzzer runtime) must be present to link. Probe with a
# real fuzzer target (libFuzzer provides its own main, so `int main` would clash).
if ! printf 'int LLVMFuzzerTestOneInput(const unsigned char*d,unsigned long n){(void)d;(void)n;return 0;}' \
     | "${CLANG}" -fsanitize=fuzzer,address -x c - -o /dev/null 2>/dev/null; then
  echo "clang has no compiler-rt (libFuzzer/ASan); skipping (install libclang-rt-NN-dev)." >&2
  exit 0
fi

tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT
SAN="-fsanitize=address,undefined -fno-sanitize-recover=all"
FUZZ_SAN="-fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all"

echo "[fuzz] ASan/UBSan smoke: decode a real pack + byte mutations"
"${CLANG}" -std=c23 -g ${SAN} "${C}/bcir_runtime.c" "${C}/test_runtime.c" -I "${C}" \
  -o "${tmp}/smoke" || { echo "  FAIL: sanitizer build"; exit 1; }
python3 -c "
from bcir.examples import vector_add
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.gem import hydrate
from bcir.abi import encode
m=vector_add(1024); pack=hydrate(m, optimize(m, TargetProfile.x86_avx512(), Theta.cool()))
open('${tmp}/pack.bin','wb').write(encode(pack))
" || { echo "  FAIL: python encode"; exit 1; }
"${tmp}/smoke" "${tmp}/pack.bin" | grep -q "^OK$" || { echo "  FAIL: ASan decode"; exit 1; }
# Truncations + single-byte flips must be rejected cleanly (no ASan/UBSan abort).
python3 - "${tmp}/pack.bin" "${tmp}" <<'PY'
import sys, os
data = open(sys.argv[1],'rb').read(); d = sys.argv[2]
muts = []
for n in range(0, len(data), max(1, len(data)//32)):    # truncations
    muts.append(data[:n])
for i in range(0, len(data), max(1, len(data)//64)):     # single-byte flips
    b = bytearray(data); b[i] ^= 0xFF; muts.append(bytes(b))
for j, m in enumerate(muts):
    open(os.path.join(d, f'm{j}.bin'),'wb').write(m)
print(len(muts))
PY
for f in "${tmp}"/m*.bin; do
  "${tmp}/smoke" "${f}" >/dev/null 2>"${tmp}/err" || true
  if grep -qiE "runtime error|AddressSanitizer|UndefinedBehavior|heap-buffer-overflow" "${tmp}/err"; then
    echo "  FAIL: sanitizer caught UB on a mutated pack ($(basename "${f}")):"; cat "${tmp}/err"; exit 1
  fi
done
echo "  PASS ASan/UBSan smoke (valid decode + mutations rejected cleanly)"

# --- the harness table ------------------------------------------------------------------
# One row per trust boundary: <key> <label> <extra libFuzzer flags> <sources...>. Ordered
# LONGEST-MEASURED-FIRST so the scheduler starts the time-bound-saturating campaigns before
# the ones that exhaust FUZZ_RUNS in seconds.
KEYS=()
declare -A LABEL FLAGS SRCS
add_target() {  # <key> <label> <extra-flags> <sources...>
  local key="$1" label="$2" flags="$3"; shift 3
  KEYS+=("${key}"); LABEL["${key}"]="${label}"; FLAGS["${key}"]="${flags}"; SRCS["${key}"]="$*"
}

# bcir_q8_model.c parses an EXTERNAL model artifact (LangRef 16). The format is sealed by
# a header CRC, a body CRC, and per-tensor CRCs -- right for the format, fatal for a
# fuzzer, since a random mutation dies on a checksum long before it reaches the geometry,
# span-overlap, exponent-range or canonical-inventory checks. The harness therefore drives
# each input twice: RAW (keeps the reject-on-bad-CRC path covered) and CRC-REPAIRED, which
# is what actually explores the parser (measured: 49 -> 228 covered blocks).
add_target q8 "BCIRQ8 model loader" "-max_len=16384" \
  "${C}/fuzz_q8_model.c" "${C}/bcir_q8_model.c"

# bcir_telemetry_frame.c decodes frames a DEVICE emits over a byte transport (UART --
# docs/kernel/TELEMETRY_FRAME_ABI.md), so the count/CRC/resync fields are all hostile.
# Seeded from real Python-encoded frames plus a garbage-prefixed multi-frame stream, so
# the fuzzer starts past the magic instead of rediscovering it.
add_target tf "telemetry frame" "" \
  "${C}/fuzz_telemetry_frame.c" "${C}/bcir_telemetry_frame.c" "${C}/bcir_runtime.c"

# bcir_asn1.c parses BER/DER from a peer. X.690's own structures -- multi-octet tags
# and lengths, indefinite-length constructed encodings closed by end-of-contents
# octets, arbitrary nesting -- are the shapes that historically break hand-written
# parsers, so the harness walks every node and reads every contents octet the decoder
# hands out. Seeded from real StreamPack DER projections plus the standard's own
# worked examples (the constructed/indefinite forms a fuzzer rarely invents).
add_target asn1 "X.690 decoder" "-max_len=8192" \
  "${C}/fuzz_asn1.c" "${C}/bcir_asn1.c"

# X.691 clause 11's decoding primitives. PER is NOT self-delimiting (X.691 7.2), so unlike
# the X.690 target there is no structure to walk without a schema -- what is reachable, and
# what is dangerous, is exactly the layer that takes an attacker-supplied bit width, octet
# count or fragment header and advances a cursor with it. The harness derives the bounds
# from the input so the 11.5 range-selection branches (bit-field / one-octet / two-octet /
# indefinite) are all reachable instead of one being hammered.
add_target per "X.691 PER primitives" "-max_len=4096" \
  "${C}/fuzz_per.c" "${C}/bcir_per.c"

# X.693's lexical layer: the tag scanner and the xmlcstring escaper. This is the half of an
# XER decode that runs BEFORE any type is consulted, on bytes an attacker chose, so it is
# the half where a cursor can be walked out of the buffer. The harness drives the escape and
# unescape pair with an ample buffer, a deliberately undersized one and a NULL measuring
# call, because the measure-then-write path is where an off-by-one hides.
add_target xer "X.693 XER lexical layer" "-max_len=4096" \
  "${C}/fuzz_xer.c" "${C}/bcir_xer.c"

# X.697's bounded reader: the stage of a JER decode that runs BEFORE any type is consulted.
# Three surfaces here have no Python counterpart and so exist nowhere else -- the
# caller-owned container stack, the caller-owned decode scratch, and the event sink -- and
# each is driven with a deliberately undersized buffer as well as an ample one, because the
# measure-then-write path is where an off-by-one hides. The 4.3 limits are derived FROM the
# input rather than fixed: a ceiling that is never reached is a refusal branch never fuzzed,
# and those ceilings are most of this reader's refusal surface.
add_target jer "X.697 bounded JER reader" "-max_len=4096" \
  "${C}/fuzz_jer.c" "${C}/bcir_jer.c" "${C}/bcir_runtime.c"

# X.696's decoder, and the only target here whose PLAN is fuzzed as well as its input.
# X.696 6.2 makes a schema-free walk impossible, so bcir_oer.c is driven by a caller-supplied
# field table as well as by octets -- which doubles the surface: a plan whose declared widths
# and lengths disagree with the document is exactly the shape that walks a cursor out of
# bounds, and it is reachable whenever a descriptor and a document come from different
# places. For a driver reading a manifest, that is always.
add_target oer "X.696 OER decoder" "-max_len=4096" \
  "${C}/fuzz_oer.c" "${C}/bcir_oer.c"

# The StreamPack decoder itself: an artifact handed to the runtime by anyone.
add_target decoder "decoder" "" \
  "${C}/fuzz_streampack.c" "${C}/bcir_runtime.c"

# BCAB is a multi-backend artifact handed to the runtime. Its checksum and digest
# layers are deliberately exercised from a canonical seed as well as arbitrary bytes.
add_target artifact "BCAB artifact bundle" "-max_len=16384" \
  "${C}/fuzz_artifact_bundle.c" "${C}/bcir_artifact_bundle.c" "${C}/bcir_runtime.c"

# bcir_exec.c runs an untrusted pack end to end with fixed caller buffers; a malformed
# pack must return a status and never read/write out of bounds (incl. the NOSPACE path).
add_target exec "executor" "" \
  "${C}/fuzz_exec.c" "${C}/bcir_exec.c" "${C}/bcir_runtime.c"

# bcir_encode.c re-serializes an untrusted pack; a malformed input must return a status
# and the writer must never exceed the output buffer (incl. the small-buffer NOSPACE path).
add_target encode "encoder" "" \
  "${C}/fuzz_encode.c" "${C}/bcir_encode.c" "${C}/bcir_runtime.c"

# bcir_binrec.c is the C twin of bcir/etl/binary.py -- a second binary trust boundary
# (driver/device packet field extraction). An arbitrary descriptor + buffer must never
# read out of bounds; ASan/UBSan would catch it.
add_target binrec "binrec" "-max_len=64" \
  "${C}/fuzz_binrec.c" "${C}/bcir_binrec.c"

# bcir_asn1_streampack.c does not merely READ hostile octets, it WRITES a native artifact
# derived from them. The harness therefore feeds every blessed output straight back into
# the native decoder and the semantic verifier: a fast path that emitted a subtly
# malformed pack would be worse than one that crashed, because the corruption would
# surface far from here. Its output buffer is fixed and small, so NOSPACE stays reachable.
add_target sp_fast "DER->native fast path" "-max_len=8192" \
  "${C}/fuzz_asn1_streampack.c" "${C}/bcir_asn1_streampack.c" "${C}/bcir_asn1.c" \
  "${C}/bcir_runtime.c"

# --- build every harness (independent link jobs, bounded by the same worker cap) -----------
echo "[fuzz] building ${#KEYS[@]} harnesses under libFuzzer + ASan/UBSan (${JOBS} at a time)"
build_fail=0
for key in "${KEYS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "${JOBS}" ]; do wait -n 2>/dev/null || true; done
  # shellcheck disable=SC2086 -- SRCS is a deliberate word-split source list.
  ( "${CLANG}" -std=c23 -g ${FUZZ_SAN} ${SRCS[${key}]} -I "${C}" -o "${tmp}/fuzz_${key}" \
      2>"${tmp}/build_${key}.log" ) &
done
wait || true
for key in "${KEYS[@]}"; do
  if [ ! -x "${tmp}/fuzz_${key}" ]; then
    echo "  FAIL: ${LABEL[${key}]} fuzzer build"; sed 's/^/    /' "${tmp}/build_${key}.log"; build_fail=1
  fi
done
[ "${build_fail}" -eq 0 ] || exit 1

# --- seed each corpus from real Python-rail output ---------------------------------------
# A corpus directory is passed to libFuzzer only when it has seeds; binrec deliberately has
# none (its input is a raw descriptor+buffer, not a framed format).
for key in "${KEYS[@]}"; do mkdir -p "${tmp}/corpus_${key}"; done
cp "${tmp}/pack.bin" "${tmp}/corpus_decoder/"
cp "${tmp}/pack.bin" "${tmp}/corpus_exec/"
cp "${tmp}/pack.bin" "${tmp}/corpus_encode/"
rmdir "${tmp}/corpus_binrec"

if ! python3 - "${tmp}/corpus_artifact" <<'PY'
import sys
from pathlib import Path
from bcir.abi.artifact_bundle import (
    ArtifactBundle, ArtifactFormat, ArtifactKind, ArtifactVariant, encode_bundle,
)
pack = Path(sys.argv[1]).parent.joinpath("pack.bin").read_bytes()
bundle = ArtifactBundle((
    ArtifactVariant(
        "00-root", ArtifactKind.STREAM_PACK, ArtifactFormat.STREAM_PACK,
        pack, channel="host", portable=True,
    ),
), "00-root", "00-root", 7, 3)
Path(sys.argv[1], "minimal.bcab").write_bytes(encode_bundle(bundle))
PY
then
  echo "  FAIL: BCAB seed corpus"
  exit 1
fi

python3 - "${tmp}/corpus_tf" <<'PY' || { echo "  FAIL: telemetry frame seed"; exit 1; }
import os, sys
import bcir.telemetry_frame as tf
d = sys.argv[1]
def mk(n, seq=7, ts=12345):
    recs = [tf.DataDNA(segment_id=f"s{i}", claim_id=1000 + i, cycles=100 * i, bytes=8 * i,
                       misses=i, thermal=40 + i, voltage=-i, utilization=i % 101)
            for i in range(n)]
    return tf.encode_frame(recs, seq=seq, timestamp=ts)
for n in (0, 1, 2, 5, 17, 64):
    open(os.path.join(d, f"frame_{n}.bin"), "wb").write(mk(n))
open(os.path.join(d, "stream2.bin"), "wb").write(mk(2) + mk(3))          # back-to-back frames
open(os.path.join(d, "noise.bin"), "wb").write(                          # the resync path
    b"\xde\xad\xbe\xefBTL" + mk(2) + b"\x00\x01BTLM" + mk(1))
PY

if ! python3 - "${tmp}/corpus_q8" <<'PY'
import sys
from pathlib import Path
from bcir.tests.test_model_weights_io import _write        # the canonical BCIRQ8 writer
for tied in (False, True):
    _write(Path(sys.argv[1]) / ("tied.bcirq8" if tied else "untied.bcirq8"), tied=tied)
PY
then
  echo "  SKIP BCIRQ8 seed corpus (could not build a seed artifact); fuzzing unseeded"
fi

# The fast path takes the same projections as the X.690 harness -- seed both.
python3 - "${tmp}/corpus_asn1" <<'ASN1SEED' || { echo "  FAIL: X.690 seed"; exit 1; }
import os, sys
from bcir.asn1.streampack import encode_pack
from bcir.examples import PROGRAMS
from bcir.gem import hydrate
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
d = sys.argv[1]
h, theta = TargetProfile.x86_avx512(), Theta.cool()
for name, build in sorted(PROGRAMS.items()):
    module = build()
    open(os.path.join(d, name + ".der"), "wb").write(
        encode_pack(hydrate(module, optimize(module, h, theta))))
# X.690's own worked examples: the constructed and indefinite forms coverage needs.
for name, octets in {
    "jones_primitive": "1a054a6f6e6573",
    "jones_indefinite": "3a8004034a6f6e040265730000",
    "bitstring_constructed": "23800303000a3b0305045f291cd00000",
    "oid_2_999_3": "0603883703",
    "sequence_smith": "300a1605536d6974680101ff",
}.items():
    open(os.path.join(d, name + ".bin"), "wb").write(bytes.fromhex(octets))
ASN1SEED
cp "${tmp}"/corpus_asn1/*.der "${tmp}/corpus_sp_fast/" 2>/dev/null || true

# --- run the campaigns, at most ${JOBS} at a time ----------------------------------------
echo "[fuzz] ${#KEYS[@]} campaigns x ${RUNS} runs (<=${MAX_TIME}s each), ${JOBS} at a time"
# Each campaign records its own exit status in a file rather than the caller waiting on a
# specific PID: the slot scheduler below uses `wait -n`, which reaps an ARBITRARY child, so a
# later `wait "${pid}"` would find the status already consumed and report a false pass.
run_campaign() {  # <key>
  local key="$1" corpus="${tmp}/corpus_$1"
  [ -d "${corpus}" ] || corpus=""
  # -artifact_prefix keeps the crashing unit out of the CHECKOUT: libFuzzer writes it to the
  # working directory by default, and seven concurrent campaigns would litter the repo root.
  # The log we tail on failure already carries the unit's Base64, which is what reproduces it.
  # shellcheck disable=SC2086 -- FLAGS/corpus are deliberate word-split libFuzzer arguments.
  "${tmp}/fuzz_${key}" -runs="${RUNS}" -max_total_time="${MAX_TIME}" \
    -artifact_prefix="${tmp}/artifact_${key}_" ${FLAGS[${key}]} \
    ${corpus} >"${tmp}/log_${key}" 2>&1
  echo "$?" >"${tmp}/rc_${key}"
}

for key in "${KEYS[@]}"; do
  # Block until a slot frees. `wait -n` returns as soon as ANY child exits, which is what
  # keeps the long poles running while short campaigns cycle through the remaining slots.
  while [ "$(jobs -rp | wc -l)" -ge "${JOBS}" ]; do wait -n 2>/dev/null || break; done
  run_campaign "${key}" &
done
wait

fail=0
for key in "${KEYS[@]}"; do
  rc="$(cat "${tmp}/rc_${key}" 2>/dev/null || echo missing)"
  if [ "${rc}" = "0" ]; then
    echo "  PASS libFuzzer ${LABEL[${key}]} (${RUNS} runs, no crash)"
  else
    echo "  FAIL: libFuzzer ${LABEL[${key}]} found a crash (exit ${rc})"
    tail -40 "${tmp}/log_${key}" 2>/dev/null; fail=1
  fi
done
[ "${fail}" -eq 0 ] || exit 1
echo "[fuzz] ok"
