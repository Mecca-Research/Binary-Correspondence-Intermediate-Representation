#!/usr/bin/env bash
# Standalone build+run gate for the C<->C++ hand-off seam (runtime/cpp/).
#
# This is the SEAM between BCIR's deterministic single-node C/IR rail and the C++ layer ABOVE it
# (dynamic graph topology + distributed multi-node orchestration). The contract is
# docs/languages/CPP_HANDOFF_BOUNDARY.md; this gate proves the seam genuinely COMPILES + RUNS, that
# it ROUND-TRIPS (the single-node Orchestrator's dispatch order == the direct C/IR decode of the
# same artifact), and that the G16 hand-off holds on the C++ rail: views with an explicit lifetime
# (a view that outlives its owner or its arena is refused, never read), admission against the LIVE
# control plane, the dynamic-graph builder's per-step freeze through the C/IR rail, and the
# manifest-of-shards (every shard runs by itself; the shards reassemble to the whole) -- graded by
# tools/c/check_handoff.py against the Python oracle, and run again under ASan + UBSan.
#
# It is STANDALONE: plain C++17 over the freestanding C units (compiled as C). It is NOT part of
# the MLIR/LLVM cmake. It self-skips (exit 0) if no C++ compiler is present, like check_runtime.sh.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C="${ROOT}/runtime/c"
CPP="${ROOT}/runtime/cpp"
CXX="${CXX:-$(command -v g++ || command -v clang++ || command -v c++ || true)}"
CC="${CC:-$(command -v clang || command -v cc || command -v gcc || true)}"

if [ -z "${CXX}" ] || [ -z "${CC}" ]; then
  echo "[cpp-handoff] no C and C++ compiler pair (cc/clang/gcc + g++/clang++/c++); skipping hand-off check." >&2
  exit 0
fi

tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# The seam's sources: the C units it links (compiled as C) and its C++ units. One list each --
# bcir/tests/test_handoff.py holds them to handoff_fixtures.C_UNITS / CPP_UNITS (the #719 trap).
seam_c=("${C}/bcir_handoff.c" "${C}/bcir_shard_manifest.c" "${C}/bcir_hydrate.c" "${C}/bcir_plan.c" "${C}/bcir_control_plane.c" "${C}/bcir_sha256.c" "${C}/bcir_runtime.c" "${C}/bcir_ring.c" "${C}/bcir_telemetry_envelope.c")
seam_cpp=("${CPP}/bcir_handoff.cpp" "${CPP}/bcir_orchestrator.cpp")

# build_objects <dir> <extra flags...>: the C units as C objects into <dir>
build_objects() {
  local dir="$1"; shift
  mkdir -p "${dir}"
  local src
  for src in "${seam_c[@]}"; do
    # shellcheck disable=SC2068
    "${CC}" -std=c11 -O2 -Wall -Wextra -Werror "$@" -I "${C}" -c "${src}" \
      -o "${dir}/$(basename "${src}" .c).o" 2>>"${tmp}/build.err" || return 1
  done
}

echo "[cpp-handoff] compile the standalone C++17 seam (runtime/cpp/) over the freestanding C units"
build_objects "${tmp}/obj" || { echo "  FAIL: the C units did not build"; cat "${tmp}/build.err"; exit 1; }
"${CXX}" -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror -I "${C}" -I "${CPP}" \
  "${CPP}/test_orchestrator.cpp" "${seam_cpp[@]}" "${tmp}"/obj/*.o -o "${tmp}/test_orch" 2>"${tmp}/build.err" \
  || { echo "  FAIL: C++ seam did not build"; cat "${tmp}/build.err"; exit 1; }
echo "  PASS compiles standalone (C++17 -Wpedantic -Werror; not in the MLIR build)"

echo "[cpp-handoff] hand-off artifacts from the C/IR path (StreamPack) + the live plane's records"
# The C/IR rail PRODUCES the artifact (a real StreamPack hydrated by the Python oracle); the plane
# records -- a lease, the generation switch installing the pack's own registry, and a second switch
# to a moved registry -- are minted by the Python issuer under the fixture root key.
mint() {  # mint <example> <pack.bin> <plane.bin> -- the one minter the tests share
  python3 - "$@" <<'PY'
import sys
from bcir.tests.handoff_fixtures import seam_artifacts
pack, plane = seam_artifacts(sys.argv[1])
open(sys.argv[2], "wb").write(pack)
open(sys.argv[3], "wb").write(plane)
PY
}
mint multi_histogram "${tmp}/pack.bin" "${tmp}/plane.bin" || { echo "  FAIL: could not mint the first artifact"; exit 1; }
mint vector_add "${tmp}/pack2.bin" "${tmp}/plane2.bin" || { echo "  FAIL: could not mint the second artifact"; exit 1; }
[ "$(head -c4 "${tmp}/pack.bin")" = "BSPK" ] || { echo "  FAIL: artifact is not a StreamPack"; exit 1; }

for n in "" 2; do
  out="$("${tmp}/test_orch" "${tmp}/pack${n}.bin" "${tmp}/plane${n}.bin")" \
    || { echo "  FAIL: round-trip smoke run"; echo "${out}"; exit 1; }
  case "${out}" in
    OK\ *) echo "  PASS round-trip identity + the G16 contract on artifact ${n:-1} (${out}): admitted by the live plane, dispatch == direct C walk, shards run by themselves and reassemble, a builder step freezes and runs, a switch makes it stale, a dead view is refused" ;;
    *) echo "  FAIL: round-trip smoke did not pass: ${out}"; exit 1 ;;
  esac
done

# A corrupted artifact must be REFUSED at admission -- the plane's verdict, carried, not derived in
# C++ (the two-truth quarantine): flip a header byte so the CRC fails.
python3 -c "
b=bytearray(open('${tmp}/pack.bin','rb').read()); b[20]^=0xFF
open('${tmp}/bad.bin','wb').write(b)
"
# The same probe must admit the clean pack, or its REJECTED proves nothing (L2).
ok="$("${tmp}/test_orch" --reject "${tmp}/pack.bin" "${tmp}/plane.bin")"; okrc=$?
{ [ "${okrc}" = "1" ] && [ "${ok}" = "ADMITTED" ]; } \
  || { echo "  FAIL: the reject probe did not admit the clean pack (got '${ok}', rc=${okrc})"; exit 1; }
rj="$("${tmp}/test_orch" --reject "${tmp}/bad.bin" "${tmp}/plane.bin")"; rrc=$?
{ [ "${rrc}" = "0" ] && [ "${rj}" = "REJECTED" ]; } \
  && echo "  PASS corrupted artifact REFUSED at admission and never dispatched, the clean one admitted by the same probe (the plane's verdict, carried)" \
  || { echo "  FAIL: corrupted artifact was admitted (got '${rj}', rc=${rrc})"; exit 1; }

echo "[cpp-handoff] G16 on the C++ rail: three-rail traces, lifetime witnesses, shards and re-entry"
"${CXX}" -std=c++17 -O2 -Wall -Wextra -Werror -I "${C}" -I "${CPP}" \
  "${CPP}/test_handoff.cpp" "${seam_cpp[@]}" "${tmp}"/obj/*.o -o "${tmp}/test_handoff_cpp" \
  2>"${tmp}/build.err" || { echo "  FAIL: C++ hand-off harness build"; cat "${tmp}/build.err"; exit 1; }
cpp_measure() {  # <C++ harness> -> the row summary; exits as tools/c/check_handoff.py does
  python3 "${ROOT}/tools/c/check_handoff.py" --cpp-exe "$1" --no-c --tmp "${tmp}" \
    > "${tmp}/cpp_rows.txt" 2>&1
  local status=$?
  tail -n 1 "${tmp}/cpp_rows.txt"
  return "${status}"
}
cpp_rows="$(cpp_measure "${tmp}/test_handoff_cpp")" \
  || { echo "  FAIL: a G16 row is not zero on the C++ rail: ${cpp_rows}"; cat "${tmp}/cpp_rows.txt"; exit 1; }
echo "  PASS C++ seam ${cpp_rows#rows }"
# The C++ gate must be able to fail (L2): a Borrow that never marks its pin returned -- neither its
# arena dropped nor its view cleared (either alone is caught: the table refuses a cleared view) --
# gives it back twice, the second return taking another borrower's pin; the witnesses must see it.
sed '/^HandoffResult Borrow::give_back() {/,/^}/ { s/^  arena_\.reset();$/  \/\/ mutant: the pin is never marked returned/; s/^  view_ = bcir_ho_view{};$/  \/\/ (nor its view cleared)/; }' \
  "${CPP}/bcir_handoff.cpp" > "${tmp}/bcir_handoff_mutant.cpp"
if cmp -s "${CPP}/bcir_handoff.cpp" "${tmp}/bcir_handoff_mutant.cpp"; then
  echo "  FAIL: the double-return fault injection did not apply (the line changed)"; exit 1
fi
"${CXX}" -std=c++17 -O2 -I "${C}" -I "${CPP}" "${CPP}/test_handoff.cpp" "${tmp}/bcir_handoff_mutant.cpp" \
  "${CPP}/bcir_orchestrator.cpp" "${tmp}"/obj/*.o -o "${tmp}/test_handoff_cpp_mutant" 2>"${tmp}/build.err" \
  || { echo "  FAIL: C++ mutant build"; cat "${tmp}/build.err"; exit 1; }
mutant_rows="$(cpp_measure "${tmp}/test_handoff_cpp_mutant")"; mutant_status=$?
if [ "${mutant_status}" -ne 1 ]; then
  echo "  FAIL: a Borrow that returns its pin twice passed the G16 C++ gate (exit ${mutant_status}): ${mutant_rows}"; exit 1
fi
echo "  PASS C++ hand-off gate fires on an injected fault (double return: ${mutant_rows#rows })"

echo "[cpp-handoff] the seam under ASan + UBSan (every lifetime law is refused, never undefined)"
printf 'int main(void){return 0;}\n' > "${tmp}/probe.cpp"
if "${CXX}" -fsanitize=address,undefined "${tmp}/probe.cpp" -o "${tmp}/probe" 2>/dev/null && "${tmp}/probe"; then
  build_objects "${tmp}/obj_asan" -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all \
    || { echo "  FAIL: sanitized C units did not build"; cat "${tmp}/build.err"; exit 1; }
  san=(-std=c++17 -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all -I "${C}" -I "${CPP}")
  "${CXX}" "${san[@]}" "${CPP}/test_orchestrator.cpp" "${seam_cpp[@]}" "${tmp}"/obj_asan/*.o \
    -o "${tmp}/test_orch_asan" 2>"${tmp}/build.err" || { echo "  FAIL: sanitized seam build"; cat "${tmp}/build.err"; exit 1; }
  "${CXX}" "${san[@]}" "${CPP}/test_handoff.cpp" "${seam_cpp[@]}" "${tmp}"/obj_asan/*.o \
    -o "${tmp}/test_handoff_cpp_asan" 2>"${tmp}/build.err" || { echo "  FAIL: sanitized harness build"; cat "${tmp}/build.err"; exit 1; }
  "${tmp}/test_orch_asan" "${tmp}/pack.bin" "${tmp}/plane.bin" > "${tmp}/asan.out" 2>&1 \
    || { echo "  FAIL: the seam under ASan/UBSan"; tail -n 30 "${tmp}/asan.out"; exit 1; }
  asan_rows="$(cpp_measure "${tmp}/test_handoff_cpp_asan")" \
    || { echo "  FAIL: G16 rows under ASan/UBSan: ${asan_rows}"; tail -n 40 "${tmp}/cpp_rows.txt"; exit 1; }
  echo "  PASS the seam, every scenario, the lifetime witnesses and the shard runs under ASan + UBSan (no report)"
else
  echo "  SKIP ASan/UBSan: UNAVAILABLE with ${CXX} on this host (no sanitizer runtime)"
fi

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
  "${CPP}/test_artifact_bundle.cpp" "${C}/bcir_artifact_bundle.c" "${C}/bcir_sha256.c" "${C}/bcir_runtime.c" \
  -o "${tmp}/test_artifact_bundle_cpp" \
  || { echo "  FAIL: BCAB C++ wrapper build"; exit 1; }
cppout="$("${tmp}/test_artifact_bundle_cpp" "${tmp}/bundle.bcab")" \
  || { echo "  FAIL: BCAB C++ wrapper run"; exit 1; }
case "${cppout}" in
  OK\ C++*) echo "  PASS BCAB C++ wrapper validates and selects the native image" ;;
  *) echo "  FAIL: unexpected BCAB C++ output '${cppout}'"; exit 1 ;;
esac

echo "[cpp-handoff] ok"
