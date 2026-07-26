#!/usr/bin/env bash
# Fuzz every BINARY TRUST BOUNDARY in runtime/c/ under libFuzzer + ASan/UBSan, and run an
# ASan/UBSan smoke on a real Python-encoded pack + byte mutations. Needs clang with
# compiler-rt (libclang-rt-NN-dev). FUZZ_RUNS controls the fuzz iteration count.
#
# The six harnesses, and whose bytes each one distrusts:
#   StreamPack decoder / executor / encoder  -- an artifact handed to the runtime
#   ETL binary-record decoder                -- device/driver packet fields
#   telemetry-frame decoder                  -- frames a device emits over UART
#   BCIRQ8 model loader                      -- an external model artifact (LangRef 16)
#   X.690 BER/DER decoder                    -- an ASN.1 artifact from a foreign peer
#
# Each seeds from real Python-rail output so the campaign starts inside the format rather
# than rediscovering its magic; the BCIRQ8 harness additionally REPAIRS the format's three
# checksum layers, without which a coverage-guided fuzzer only ever proves that the CRC
# check works and never reaches the parser behind it.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C="${ROOT}/runtime/c"
CLANG="${CLANG:-$(command -v clang || true)}"
RUNS="${FUZZ_RUNS:-200000}"
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

echo "[fuzz] libFuzzer on the decoder (${RUNS} runs)"
"${CLANG}" -std=c23 -g -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all \
  "${C}/fuzz_streampack.c" "${C}/bcir_runtime.c" -I "${C}" -o "${tmp}/fuzz" \
  || { echo "  FAIL: fuzzer build"; exit 1; }
mkdir -p "${tmp}/corpus" && cp "${tmp}/pack.bin" "${tmp}/corpus/"   # seed from a real pack
if "${tmp}/fuzz" -runs="${RUNS}" -max_total_time=60 "${tmp}/corpus" >"${tmp}/flog" 2>&1; then
  echo "  PASS libFuzzer (${RUNS} runs, no crash)"
else
  echo "  FAIL: libFuzzer found a crash"; tail -40 "${tmp}/flog"; exit 1
fi

echo "[fuzz] libFuzzer on the ETL binary-record decoder (${RUNS} runs)"
# bcir_binrec.c is the C twin of bcir/etl/binary.py -- a second binary trust boundary
# (driver/device packet field extraction). An arbitrary descriptor + buffer must never
# read out of bounds; ASan/UBSan would catch it.
"${CLANG}" -std=c23 -g -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all \
  "${C}/fuzz_binrec.c" "${C}/bcir_binrec.c" -I "${C}" -o "${tmp}/fuzz_binrec" \
  || { echo "  FAIL: binrec fuzzer build"; exit 1; }
if "${tmp}/fuzz_binrec" -runs="${RUNS}" -max_total_time=60 -max_len=64 >"${tmp}/flog2" 2>&1; then
  echo "  PASS libFuzzer binrec (${RUNS} runs, no crash)"
else
  echo "  FAIL: libFuzzer binrec found a crash"; tail -40 "${tmp}/flog2"; exit 1
fi

echo "[fuzz] libFuzzer on the StreamPack executor (${RUNS} runs)"
# bcir_exec.c runs an untrusted pack end to end with fixed caller buffers; a malformed
# pack must return a status and never read/write out of bounds (incl. the NOSPACE path).
"${CLANG}" -std=c23 -g -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all \
  "${C}/fuzz_exec.c" "${C}/bcir_exec.c" "${C}/bcir_runtime.c" -I "${C}" -o "${tmp}/fuzz_exec" \
  || { echo "  FAIL: executor fuzzer build"; exit 1; }
mkdir -p "${tmp}/corpus_exec" && cp "${tmp}/pack.bin" "${tmp}/corpus_exec/"   # seed from a real pack
if "${tmp}/fuzz_exec" -runs="${RUNS}" -max_total_time=60 "${tmp}/corpus_exec" >"${tmp}/flog3" 2>&1; then
  echo "  PASS libFuzzer executor (${RUNS} runs, no crash)"
else
  echo "  FAIL: libFuzzer executor found a crash"; tail -40 "${tmp}/flog3"; exit 1
fi

echo "[fuzz] libFuzzer on the StreamPack encoder (${RUNS} runs)"
# bcir_encode.c re-serializes an untrusted pack; a malformed input must return a status
# and the writer must never exceed the output buffer (incl. the small-buffer NOSPACE path).
"${CLANG}" -std=c23 -g -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all \
  "${C}/fuzz_encode.c" "${C}/bcir_encode.c" "${C}/bcir_runtime.c" -I "${C}" -o "${tmp}/fuzz_encode" \
  || { echo "  FAIL: encoder fuzzer build"; exit 1; }
mkdir -p "${tmp}/corpus_enc" && cp "${tmp}/pack.bin" "${tmp}/corpus_enc/"
if "${tmp}/fuzz_encode" -runs="${RUNS}" -max_total_time=60 "${tmp}/corpus_enc" >"${tmp}/flog4" 2>&1; then
  echo "  PASS libFuzzer encoder (${RUNS} runs, no crash)"
else
  echo "  FAIL: libFuzzer encoder found a crash"; tail -40 "${tmp}/flog4"; exit 1
fi

echo "[fuzz] libFuzzer on the telemetry-frame decoder (${RUNS} runs)"
# bcir_telemetry_frame.c decodes frames a DEVICE emits over a byte transport (UART --
# docs/kernel/TELEMETRY_FRAME_ABI.md), so the count/CRC/resync fields are all hostile.
# Seeded from real Python-encoded frames plus a garbage-prefixed multi-frame stream, so
# the fuzzer starts past the magic instead of rediscovering it.
"${CLANG}" -std=c23 -g -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all \
  "${C}/fuzz_telemetry_frame.c" "${C}/bcir_telemetry_frame.c" "${C}/bcir_runtime.c" \
  -I "${C}" -o "${tmp}/fuzz_tf" || { echo "  FAIL: telemetry-frame fuzzer build"; exit 1; }
mkdir -p "${tmp}/corpus_tf"
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
if "${tmp}/fuzz_tf" -runs="${RUNS}" -max_total_time=60 "${tmp}/corpus_tf" >"${tmp}/flog5" 2>&1; then
  echo "  PASS libFuzzer telemetry frame (${RUNS} runs, no crash)"
else
  echo "  FAIL: libFuzzer telemetry frame found a crash"; tail -40 "${tmp}/flog5"; exit 1
fi

echo "[fuzz] libFuzzer on the BCIRQ8 model loader (${RUNS} runs)"
# bcir_q8_model.c parses an EXTERNAL model artifact (LangRef 16). The format is sealed by
# a header CRC, a body CRC, and per-tensor CRCs -- right for the format, fatal for a
# fuzzer, since a random mutation dies on a checksum long before it reaches the geometry,
# span-overlap, exponent-range or canonical-inventory checks. The harness therefore drives
# each input twice: RAW (keeps the reject-on-bad-CRC path covered) and CRC-REPAIRED, which
# is what actually explores the parser (measured: 49 -> 228 covered blocks).
"${CLANG}" -std=c23 -g -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all \
  "${C}/fuzz_q8_model.c" "${C}/bcir_q8_model.c" -I "${C}" -o "${tmp}/fuzz_q8" \
  || { echo "  FAIL: BCIRQ8 fuzzer build"; exit 1; }
mkdir -p "${tmp}/corpus_q8"
if python3 - "${tmp}/corpus_q8" <<'PY'
import sys
from pathlib import Path
from bcir.tests.test_model_weights_io import _write        # the canonical BCIRQ8 writer
for tied in (False, True):
    _write(Path(sys.argv[1]) / ("tied.bcirq8" if tied else "untied.bcirq8"), tied=tied)
PY
then
  if "${tmp}/fuzz_q8" -runs="${RUNS}" -max_total_time=60 -max_len=16384 "${tmp}/corpus_q8" \
       >"${tmp}/flog6" 2>&1; then
    echo "  PASS libFuzzer BCIRQ8 loader (${RUNS} runs, no crash)"
  else
    echo "  FAIL: libFuzzer BCIRQ8 loader found a crash"; tail -40 "${tmp}/flog6"; exit 1
  fi
else
  echo "  SKIP BCIRQ8 loader fuzz (could not build a seed artifact)"
fi
echo "[fuzz] libFuzzer on the X.690 BER/DER decoder (${RUNS} runs)"
# bcir_asn1.c parses BER/DER from a peer. X.690's own structures -- multi-octet tags
# and lengths, indefinite-length constructed encodings closed by end-of-contents
# octets, arbitrary nesting -- are the shapes that historically break hand-written
# parsers, so the harness walks every node and reads every contents octet the decoder
# hands out. Seeded from real StreamPack DER projections plus the standard's own
# worked examples (the constructed/indefinite forms a fuzzer rarely invents).
"${CLANG}" -std=c23 -g -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all \
  "${C}/fuzz_asn1.c" "${C}/bcir_asn1.c" -I "${C}" -o "${tmp}/fuzz_asn1" \
  || { echo "  FAIL: X.690 fuzzer build"; exit 1; }
mkdir -p "${tmp}/corpus_asn1"
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
if "${tmp}/fuzz_asn1" -runs="${RUNS}" -max_total_time=60 -max_len=8192 \
     "${tmp}/corpus_asn1" >"${tmp}/flog7" 2>&1; then
  echo "  PASS libFuzzer X.690 decoder (${RUNS} runs, no crash)"
else
  echo "  FAIL: libFuzzer X.690 decoder found a crash"; tail -40 "${tmp}/flog7"; exit 1
fi
echo "[fuzz] ok"
