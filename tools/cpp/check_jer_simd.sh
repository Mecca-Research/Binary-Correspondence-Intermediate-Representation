#!/usr/bin/env bash
# check_jer_simd.sh -- J5's gate for the hosted SIMD rail (#jersimd).
#
# J5's row: "Optional C++17 structural/UTF-8 scanner behind the C ABI with scalar fallback.
# Same accepted/rejected corpus and trace; statistically significant measured advantage on
# at least two hosts; no unsupported-CPU fault."
#
# This script checks the two clauses that are decidable on ONE machine:
#
#   1. SAME CORPUS AND TRACE. Every tier -- scalar, sse2, avx2, neon, and whatever `auto`
#      resolves to -- must return an identical status AND an identical byte offset to the
#      scalar rail. Identical, not equivalent: 4.2's contract is a stable code plus an
#      offset, and an offset that is off by one still sends someone to the wrong octet.
#   2. NO UNSUPPORTED-CPU FAULT. A tier this CPU does not advertise, or this build did not
#      compile, must degrade to scalar rather than fault or refuse.
#
# The advantage clause is NOT checked here. It asks for two hosts; CI is one runner per lane,
# and 8 says SIMD is admitted "on a declared target" with "non-overlapping intervals". A
# timing threshold on a shared runner would gate on the runner's load, which 8 also refuses
# ("shared CI gates validity and trend evidence, not noisy timing thresholds").
#
# Self-skips (exit 0) when no C++ compiler is present, like the other C++ gates.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C="${ROOT}/runtime/c"
CPP="${ROOT}/runtime/cpp"

CXX="${CXX:-}"
if [ -z "${CXX}" ]; then
  for candidate in clang++ g++ c++; do
    if command -v "${candidate}" >/dev/null 2>&1; then CXX="${candidate}"; break; fi
  done
fi
CC="${CC:-}"
if [ -z "${CC}" ]; then
  for candidate in clang gcc cc; do
    if command -v "${candidate}" >/dev/null 2>&1; then CC="${candidate}"; break; fi
  done
fi
if [ -z "${CXX}" ] || [ -z "${CC}" ]; then
  echo "SKIP: no C++/C compiler pair; the SIMD rail is optional by design"
  exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

# The adapter and the core are compiled by their OWN compilers and then linked. Handing a
# .c file to clang++ is an error under -Werror, and it would be the wrong thing to check
# anyway: the rail ships as a C++ translation unit linked against the C core.
for source in "${CPP}/bcir_jer_simd.cpp" "${CPP}/test_jer_simd.cpp"; do
  "${CXX}" -std=c++17 -O2 -Wall -Wextra -Werror -I "${C}" -I "${CPP}" \
    -c "${source}" -o "${tmp}/$(basename "${source}").o" \
    || { echo "FAIL: $(basename "${source}") is not warning-clean under C++17"; exit 1; }
done
for source in "${C}/bcir_jer.c" "${C}/bcir_runtime.c"; do
  "${CC}" -std=c11 -O2 -Wall -Wextra -Werror -I "${C}" \
    -c "${source}" -o "${tmp}/$(basename "${source}").o" \
    || { echo "FAIL: $(basename "${source}") is not warning-clean"; exit 1; }
done
"${CXX}" "${tmp}"/*.o -o "${tmp}/test_jer_simd" \
  || { echo "FAIL: the SIMD rail does not link against the scalar core"; exit 1; }

echo "tiers: $("${tmp}/test_jer_simd" <<< 'tiers')"

# The corpus. Generated rather than checked in, because the cases that matter are indexed
# families -- a multi-byte sequence straddling EVERY offset in a 32-octet block, and an
# invalid octet at EVERY offset -- and a static file of those is a file nobody re-derives
# when a vector width changes.
python3 - > "${tmp}/cases.txt" <<'SIMDPY'
import random

cases = [b"", b'{"a":1,"b":"hello"}', b'{"k":"' + b"x" * 5000 + b'"}',
         "café".encode(), "日本語".encode(),
         "\U0001f600".encode(), "\U0001f600".encode() * 400,
         b"\xff", bytes(range(0x80, 0x100))]
for pad in range(40):
    head = b"a" * pad
    cases.append(head + "é".encode() + b"b" * 40)
    cases.append(head + "\U0001f600".encode() + b"b" * 40)
    cases.append(head + b"\x80" + b"b" * 40)
    cases.append(head + b"\xc3")
    cases.append(head + b"\xc0\xaf" + b"b" * 10)
    cases.append(head + b"\xed\xa0\x80" + b"b" * 10)
    cases.append(head + b"\xf5\x80\x80\x80" + b"b" * 10)
generator = random.Random(7)
for _ in range(200):
    cases.append(bytes(generator.randrange(256)
                       for _ in range(generator.randrange(0, 200))))

for octets in cases:
    for tier in ("scalar", "sse2", "avx2", "neon", "auto"):
        print("utf8 %s %s" % (tier, octets.hex() or "-"))
SIMDPY

"${tmp}/test_jer_simd" < "${tmp}/cases.txt" > "${tmp}/replies.txt"

python3 - "${tmp}/cases.txt" "${tmp}/replies.txt" <<'CHECKPY'
import sys

tiers = ("scalar", "sse2", "avx2", "neon", "auto")
commands = [line for line in open(sys.argv[1]) if line.startswith("utf8 ")]
replies = [line.rstrip("\n") for line in open(sys.argv[2])]
if len(replies) != len(commands):
    print("FAIL: %d replies for %d commands" % (len(replies), len(commands)))
    raise SystemExit(1)

documents = len(commands) // len(tiers)
divergences = 0
for index in range(documents):
    window = replies[index * len(tiers):(index + 1) * len(tiers)]
    for tier, reply in zip(tiers, window):
        if reply != window[0]:
            print("FAIL: document %d tier %s gave %r, scalar gave %r"
                  % (index, tier, reply, window[0]))
            divergences += 1
if divergences:
    raise SystemExit(1)

# A differential over documents that were all accepted would prove nothing about rejection,
# and one whose rejections all share an offset would not exercise the offset comparison.
verdicts = set(reply.split()[1] for reply in replies)
offsets = set(reply.split()[2] for reply in replies if reply.split()[1] != "0")
if len(verdicts) < 2:
    print("FAIL: every document got the same verdict %r" % verdicts)
    raise SystemExit(1)
if len(offsets) < 20:
    print("FAIL: only %d distinct rejection offsets; the offset check has no teeth"
          % len(offsets))
    raise SystemExit(1)
print("  ok: %d documents x %d tiers agree with the scalar rail on status AND offset"
      % (documents, len(tiers)))
print("  ok: %d distinct rejection offsets exercised" % len(offsets))
CHECKPY

echo "  ok: no unsupported-CPU fault (every tier answered; unavailable tiers degraded)"

# --- every host compiles every tier -----------------------------------------------------------
#
# The differential above can only exercise the tiers THIS CPU has. An x86 developer editing
# the NEON path therefore gets no feedback at all until CI, and an aarch64 one gets none on
# the SSE2/AVX2 path -- which is how a tier rots.
#
# Clang carries every backend, so the other architecture's tier can be COMPILED here even
# though it cannot be run. Compile-only needs no sysroot: the file includes clang's own
# stdint/arm_neon headers and nothing from libc.
#
# "It compiled" is deliberately NOT the check. If `BCIR_SIMD_ARM` were false the file would
# still compile -- to scalar, silently, which is exactly the failure worth catching. So the
# object is disassembled and the tier's OWN instructions must appear in it. `umaxv`/`uminv`
# over a `.16b` register are NEON-only, and they are the horizontal reduction the ASCII-run
# kernel is built on; `pmovmskb`/`pcmpgt` are the x86 equivalent.
cross_target=""
cross_needle=""
case "$(uname -m)" in
  aarch64|arm64) cross_target="x86_64-linux-gnu"; cross_needle="pmovmskb|pcmpgt" ;;
  *)             cross_target="aarch64-linux-gnu"; cross_needle="umaxv|uminv" ;;
esac

if "${CXX}" --target="${cross_target}" -std=c++17 -O2 -Wall -Wextra -Werror -ffreestanding \
     -I "${C}" -I "${CPP}" -c "${CPP}/bcir_jer_simd.cpp" -o "${tmp}/cross.o" 2>"${tmp}/cross.log"
then
  disasm=""
  for tool in llvm-objdump objdump; do
    if command -v "${tool}" >/dev/null 2>&1; then
      disasm="$("${tool}" -d "${tmp}/cross.o" 2>/dev/null || true)"
      [ -n "${disasm}" ] && break
    fi
  done
  if [ -z "${disasm}" ]; then
    echo "  note: ${cross_target} tier compiles, but no disassembler to prove it is not scalar"
  elif printf '%s' "${disasm}" | grep -qE "${cross_needle}"; then
    echo "  ok: the ${cross_target} tier compiles here AND emits its own SIMD instructions"
  else
    echo "FAIL: ${cross_target} compiled without any ${cross_needle} instruction -- the tier"
    echo "      fell back to scalar rather than building, which no run on this host would show"
    exit 1
  fi
else
  # A clang without the other backend, or a gcc that cannot retarget. Skipped rather than
  # failed: the cross build is extra reach, and CI runs the tier natively either way.
  echo "  note: ${CXX} cannot target ${cross_target}; the other tier is unchecked here"
fi

echo "  note: J5's two-host advantage clause is NOT checked here -- see docs 7.3"
