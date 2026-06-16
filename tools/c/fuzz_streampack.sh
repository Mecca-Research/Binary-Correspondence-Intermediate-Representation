#!/usr/bin/env bash
# Fuzz the StreamPack C decoder (a trust boundary) under libFuzzer + ASan/UBSan, and
# run an ASan/UBSan smoke on a real Python-encoded pack + byte mutations. Needs clang
# with compiler-rt (libclang-rt-NN-dev). FUZZ_RUNS controls the fuzz iteration count.
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
echo "[fuzz] ok"
