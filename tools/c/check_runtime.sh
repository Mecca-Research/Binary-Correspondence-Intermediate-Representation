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
echo "[c-runtime] ok"
