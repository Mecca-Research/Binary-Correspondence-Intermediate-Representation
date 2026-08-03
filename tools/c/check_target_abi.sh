#!/usr/bin/env bash
# check_target_abi.sh -- the freestanding core compiles for targets this host cannot run
# (#targetabi), and no source hand-declares a libc function.
#
# WHY THIS EXISTS, AND THE BUG THAT BOUGHT IT. PR #699 shipped a native bench that did not
# compile under Termux. The counter probe needed a prototype for `syscall`, which glibc hides
# behind a feature macro, so the file declared `syscall` and `ioctl` itself. bionic types
# ioctl's request parameter as `int` where glibc uses `unsigned long`, so the local prototype
# matched one libc and collided with the other:
#
#   error: at most one overload for a given name may lack the 'overloadable' attribute
#     extern int ioctl(int fd, unsigned long request, ...);
#
# Nothing here could catch it. The existing cross-compile gates in #jersimd and #jerindex
# target `aarch64-linux-gnu` -- the right ARCHITECTURE and the wrong LIBC -- so Android's C
# library was exercised only when somebody ran the harness on a phone by hand. This gate
# closes as much of that as is closable from a machine with no bionic sysroot.
#
# WHAT IT CHECKS, AND WHAT IT HONESTLY CANNOT.
#
#   1. THE FREESTANDING CORE, for real, on Android triples. `bcir_jer.c` and its siblings
#      build with -ffreestanding -nostdlib and reach only compiler-provided headers
#      (stdint.h, stddef.h), never libc. That means they can be compiled for
#      aarch64-linux-android WITHOUT a sysroot, which is genuine new coverage: the core is
#      the part that has to run on a device, and until now nothing checked it against
#      Android's ABI at all. 32-bit targets are included because they catch a different class
#      -- anything that assumed a 64-bit size_t or pointer.
#
#   2. NO HAND-DECLARED LIBC FUNCTIONS anywhere in runtime/. This is the actual shape of the
#      #699 bug and the only part of it a machine without bionic can detect. A source that
#      needs a prototype must obtain it from the system header, because only the system
#      header knows what that libc's signature is.
#
#   NOT CHECKED: the HOSTED tools against bionic. They need libc headers, this container has
#   no bionic sysroot, and the network policy denies the NDK (dl.google.com answers 403 to
#   CONNECT). So `bcir_asn1_bench.c` -- the file that actually broke -- is still only compiled
#   against bionic when someone runs it on a phone. Rule 2 covers the mechanism that broke it;
#   it does not cover every way a hosted tool could be non-portable, and this comment exists
#   so nobody reads a green #targetabi as more than it is.
#
# Self-skips a triple clang cannot target, and fails one it can target but cannot build.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C="${ROOT}/runtime/c"

CC="${CC:-}"
if [ -z "${CC}" ]; then
  for candidate in clang gcc cc; do
    if command -v "${candidate}" >/dev/null 2>&1; then CC="${candidate}"; break; fi
  done
fi
if [ -z "${CC}" ]; then
  echo "SKIP: no C compiler"
  exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

# The freestanding core: every source that builds with -ffreestanding -nostdlib.
SOURCES="bcir_jer.c bcir_per.c bcir_oer.c bcir_xer.c bcir_emit.c bcir_asn1.c bcir_runtime.c"

# Android first and deliberately, because Android is the libc this repository ships to and
# does not test. The rest widen the ABI surface rather than the libc surface: armv7a and i686
# are 32-bit, riscv64 is a different register model, wasm32 has 32-bit pointers with 64-bit
# integers. A core that survives all of them is not quietly assuming an x86-64 shape.
TARGETS="aarch64-linux-android armv7a-linux-androideabi i686-linux-gnu riscv64-unknown-elf wasm32-unknown-unknown"

checked=0
for target in ${TARGETS}; do
  # Probe with a trivial file: a triple this clang was not built for is a skip, not a failure.
  printf 'int probe(void){return 0;}\n' > "${tmp}/probe.c"
  if ! "${CC}" --target="${target}" -ffreestanding -nostdlib -c "${tmp}/probe.c" \
       -o "${tmp}/probe.o" 2>/dev/null; then
    echo "  note: ${CC} cannot target ${target}; skipped"
    continue
  fi
  for source in ${SOURCES}; do
    if ! "${CC}" --target="${target}" -ffreestanding -nostdlib -std=c11 \
         -Wall -Wextra -Werror -I "${C}" -c "${C}/${source}" -o "${tmp}/out.o" \
         2>"${tmp}/err.txt"; then
      echo "FAIL: ${source} does not build for ${target}"
      while IFS= read -r line; do printf '      %s\n' "${line}"; done <"${tmp}/err.txt"
      exit 1
    fi
  done
  checked=$((checked + 1))
  echo "  ok: the freestanding core builds for ${target}"
done

if [ "${checked}" -eq 0 ]; then
  echo "  note: no cross target available; the ABI sweep checked nothing"
fi

# Rule 2. A hand-written prototype for a libc function is the #699 bug exactly: it can only
# match one libc's idea of the signature, and the compiler on every other libc is right to
# reject it. `grep -c` is used rather than a pipe into an early-exiting reader, for the reason
# tools/cpp/check_jer_index.sh records.
hits="$(grep -rnE '^[[:space:]]*extern[[:space:]]+[a-zA-Z_].*\(' \
        "${C}"/*.c "${C}"/*.h "${ROOT}"/runtime/cpp/*.cpp "${ROOT}"/runtime/cpp/*.h \
        2>/dev/null | grep -v 'extern "C"' || true)"
if [ -n "${hits}" ]; then
  echo "FAIL: a source declares a function the system header should declare."
  echo "      Only that libc knows its own signatures -- bionic types ioctl's request as"
  echo "      \`int\` and glibc as \`unsigned long\`, and a local prototype matches one and"
  echo "      breaks the other. Include the header instead (see #699)."
  while IFS= read -r line; do printf '      %s\n' "${line}"; done <<<"${hits}"
  exit 1
fi
echo "  ok: no source hand-declares a libc function"

echo "  note: HOSTED tools are still unchecked against bionic -- no sysroot here, and the"
echo "        network policy denies the NDK. That gap is only closed by running on a device."
