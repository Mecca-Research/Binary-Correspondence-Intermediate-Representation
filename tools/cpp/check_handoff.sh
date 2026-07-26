#!/usr/bin/env bash
# Standalone build+run gate for the C<->C++ hand-off seam scaffold (runtime/cpp/).
#
# This is the SEAM between BCIR's deterministic single-node C/IR rail and the C++
# layer ABOVE it (dynamic graph topology + distributed multi-node orchestration).
# The contract is docs/languages/CPP_HANDOFF_BOUNDARY.md; this gate proves the scaffold
# genuinely COMPILES + RUNS and that the seam ROUND-TRIPS (the C++ single-node
# Orchestrator's dispatch order == the direct C/IR decode of the same artifact).
#
# It is STANDALONE: plain C++17, linked only against the existing freestanding C
# runtime (runtime/c/bcir_runtime.c). It is NOT part of the MLIR/LLVM cmake. It
# self-skips (exit 0) if no C++ compiler is present, exactly like check_runtime.sh.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C="${ROOT}/runtime/c"
CPP="${ROOT}/runtime/cpp"
CXX="${CXX:-$(command -v g++ || command -v clang++ || command -v c++ || true)}"
CC="${CC:-$(command -v clang || command -v cc || command -v gcc || true)}"

if [ -z "${CXX}" ]; then
  echo "[cpp-handoff] no C++ compiler (g++/clang++/c++); skipping hand-off check." >&2
  exit 0
fi

tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

echo "[cpp-handoff] compile the standalone C++17 seam scaffold (runtime/cpp/) + link the freestanding C runtime"
# Build: the C++ orchestrator + the round-trip smoke driver + the existing C decoder.
# The C runtime compiles as C (its own std); the C++ as C++17. Try g++/clang++ flags.
build_ok=0
for cxxflags in "-std=c++17 -O2 -Wall -Wextra" "-std=c++17 -O2"; do
  # shellcheck disable=SC2086
  if "${CXX}" ${cxxflags} -I "${C}" \
        "${CPP}/bcir_orchestrator.cpp" "${CPP}/test_orchestrator.cpp" "${C}/bcir_runtime.c" \
        -o "${tmp}/test_orch" 2>"${tmp}/build.err"; then
    build_ok=1; break
  fi
done
if [ "${build_ok}" != "1" ]; then
  echo "  FAIL: C++ seam scaffold did not build"; cat "${tmp}/build.err"; exit 1
fi
echo "  PASS compiles standalone (C++17, links only the freestanding C runtime; not in the MLIR build)"

echo "[cpp-handoff] produce a hand-off artifact via the existing C/IR path (StreamPack) + round-trip it through the seam"
# The C/IR rail PRODUCES the serialized artifact (a real StreamPack the existing
# Python encode path emits -- the same fixture the c-runtime gate uses). The C++
# Orchestrator CONSUMES it; the smoke driver asserts seam order == direct C order.
python3 -c "
from bcir.examples import multi_histogram
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.gem import hydrate
from bcir.abi import encode
m=multi_histogram(); r=optimize(m, TargetProfile.x86_avx512(), Theta.cool())
open('${tmp}/pack.bin','wb').write(encode(hydrate(m, r)))
" || { echo "  FAIL: could not produce a StreamPack artifact"; exit 1; }
[ "$(head -c4 "${tmp}/pack.bin")" = "BSPK" ] || { echo "  FAIL: artifact is not a StreamPack"; exit 1; }

out="$("${tmp}/test_orch" "${tmp}/pack.bin")" || { echo "  FAIL: round-trip smoke run"; echo "${out}"; exit 1; }
case "${out}" in
  OK\ *) echo "  PASS round-trip identity (C++ single-node Orchestrator dispatch == direct C/IR decode: ${out})" ;;
  *) echo "  FAIL: round-trip smoke did not pass: ${out}"; exit 1 ;;
esac

# Second artifact (a different fixture) so the round-trip is not pinned to one shape.
if [ -n "${CC}" ]; then :; fi  # CC unused for the build; kept for symmetry with check_runtime.sh
python3 -c "
from bcir.examples import vector_add
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.gem import hydrate
from bcir.abi import encode
m=vector_add(1024); r=optimize(m, TargetProfile.x86_avx512(), Theta.cool())
open('${tmp}/pack2.bin','wb').write(encode(hydrate(m, r)))
" || { echo "  FAIL: could not produce the second artifact"; exit 1; }
out2="$("${tmp}/test_orch" "${tmp}/pack2.bin")" || { echo "  FAIL: second round-trip run"; echo "${out2}"; exit 1; }
case "${out2}" in
  OK\ *) echo "  PASS round-trip identity on a second fixture (${out2})" ;;
  *) echo "  FAIL: second round-trip: ${out2}"; exit 1 ;;
esac

# A corrupted artifact must be REJECTED at the boundary (admit() delegates to the C
# verifier; the C++ side does not re-derive legality). Flip a header byte so the CRC
# fails -- the seam's admit() must refuse it, proving the two-truth quarantine
# (legality is the C/IR rail's verdict, carried across, not invented in C++).
python3 -c "
b=bytearray(open('${tmp}/pack.bin','rb').read()); b[20]^=0xFF  # corrupt n_segments (breaks CRC)
open('${tmp}/bad.bin','wb').write(b)
"
cat > "${tmp}/reject.cpp" <<'CPP'
#include <cstdio>
#include <cstdint>
#include <vector>
#include "bcir_orchestrator.hpp"
int main(int c, char** v) {
  if (c < 2) return 2;
  std::FILE* f = std::fopen(v[1], "rb"); if (!f) return 2;
  std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
  std::vector<std::uint8_t> b(n > 0 ? (size_t)n : 0);
  if (n > 0 && std::fread(b.data(), 1, b.size(), f) != b.size()) { std::fclose(f); return 2; }
  std::fclose(f);
  bcir::SingleNodeOrchestrator s;
  bcir_status st = s.admit(b.data(), b.size());
  // A corrupted artifact must NOT be admitted (any non-OK status is correct).
  std::printf(st == BCIR_OK ? "ADMITTED\n" : "REJECTED\n");
  return st == BCIR_OK ? 1 : 0;
}
CPP
"${CXX}" -std=c++17 -O2 -I "${C}" -I "${CPP}" "${tmp}/reject.cpp" "${CPP}/bcir_orchestrator.cpp" "${C}/bcir_runtime.c" -o "${tmp}/reject" 2>"${tmp}/reject.err" \
  || { echo "  FAIL: reject harness build"; cat "${tmp}/reject.err"; exit 1; }
rj="$("${tmp}/reject" "${tmp}/bad.bin")"; rrc=$?
{ [ "${rrc}" = "0" ] && [ "${rj}" = "REJECTED" ]; } \
  && echo "  PASS corrupted artifact REJECTED at the boundary (admit() carries the C/IR verdict; the two-truth quarantine holds)" \
  || { echo "  FAIL: corrupted artifact was admitted (got '${rj}', rc=${rrc})"; exit 1; }

echo "[cpp-handoff] BCAB C++ borrowed-view wrapper + deterministic selector"
python3 - "${tmp}/bundle.bcab" <<'PY' || { echo "  FAIL: Python BCAB fixture"; exit 1; }
import struct, sys
from bcir.abi import (ArtifactBundle, ArtifactFormat, ArtifactKind, ArtifactVariant,
                      Endianness, encode, write_bundle)
from bcir.examples import vector_add
from bcir.gem import hydrate
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
m = vector_add(8); pack = hydrate(m, optimize(m, TargetProfile.x86_avx512(), Theta.cool()))
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
write_bundle(sys.argv[1], ArtifactBundle(variants, "00-root", "portable-c", 123, 7))
PY
"${CXX}" -std=c++17 -O2 -Wall -Wextra -Werror -I "${C}" -I "${CPP}" \
  "${CPP}/test_artifact_bundle.cpp" "${C}/bcir_artifact_bundle.c" "${C}/bcir_runtime.c" \
  -o "${tmp}/test_artifact_bundle_cpp" \
  || { echo "  FAIL: BCAB C++ wrapper build"; exit 1; }
cppout="$("${tmp}/test_artifact_bundle_cpp" "${tmp}/bundle.bcab")" \
  || { echo "  FAIL: BCAB C++ wrapper run"; exit 1; }
case "${cppout}" in
  OK\ C++*) echo "  PASS BCAB C++ wrapper validates and selects the native image" ;;
  *) echo "  FAIL: unexpected BCAB C++ output '${cppout}'"; exit 1 ;;
esac

echo "[cpp-handoff] ok"
