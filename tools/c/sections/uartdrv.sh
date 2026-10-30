#!/usr/bin/env bash
# uartdrv: the polled UART driver, linked from bcir-cc's --linkable emit == the oracle's == its source (#uartdrv)
#
# One section of tools/c/check_runtime.sh (blueprint slice U2, docs/kernel/BCIR_UART_DRIVER_BLUEPRINT.md
# section 5): BCIR Make builds its binaries from runtime/manifest.json and runs this script as a task,
# whose verdict the gate shows; the CMake project builds the same binaries and runs it as the
# `c-section-uartdrv` CTest entry, and tools/build/section_parity.py holds the two outputs
# byte-identical. It compiles what bcir-cc and the oracle emit with CC, which the caller names.
#
# Three units make the program: the driver (runtime/c/uart_driver.c, its API runtime/c/uart_driver.h),
# the simulated 16550 (runtime/c/sim16550.c) and the harness (runtime/c/uart_main.c), which prints a
# deterministic transcript. The driver's register window reaches the device through Clang's load/store
# tracing on the driver unit alone (uart_main.c says how), so the linked runs need a CC that has it.
# Under one that has not (GCC: the CMake matrix's gcc cell) they print a SKIP line naming the reason and
# the rest runs, the way barrier.sh and portio.sh report a check their host cannot run; the gate's CC
# is the newest clang on PATH, so the gate runs them. The section holds:
#   - the driver is freestanding C, warning-clean at C11 and C23 under -Wpedantic -Werror;
#   - the manifest's uart_main -- the three units built with the gate's flags and no tracing -- refuses
#     to print a transcript (exit 3), and its script console answers as a 16550A;
#   - the program linked from bcir-cc --linkable's driver prints, byte for byte, the transcript of the
#     program linked from the oracle's --linkable driver, which prints the transcript of the program
#     built from the driver's source, which is the checked-in runtime/c/uart_main.golden (the axi16550
#     placement: PG143's part and programming vector);
#   - a driver laid out for another placement is refused (exit 4, hazard H-A).
# Each rail's --linkable artifact is self-contained (the types, the prototypes and the globals it names,
# and the headers its text needs: CF-LINKEMIT on the oracle, CF-LINKTWIN on bcir-cc), so each builds alone.
#
#   usage: CC=<compiler> uartdrv.sh <bcir-cc> <uart_main>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
FIXTURES="${ROOT}/bcir/frontends/devices/fixtures"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 2 ] || { echo "usage: CC=<compiler> uartdrv.sh <bcir-cc> <uart_main>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc> <uart_main>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
UART_MAIN="$(abs "$2")"; [ -x "${UART_MAIN}" ] || { echo "  FAIL: ${UART_MAIN} is not an executable uart_main"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT
PLACEMENT=axi16550
GOLDEN="${C}/uart_main.golden"
TRACE="-fsanitize-coverage=func,trace-loads,trace-stores"
UNITS=(uart_driver.c uart_driver.h uart_main.c sim16550.c sim16550.h)
# The C23 spelling CC accepts (GCC 13 knows only -std=c2x), as the freestanding section probes it.
c23=c23
printf 'int bcir_c23_probe;\n' | "${CC}" -std=c23 -x c -c -o /dev/null - > /dev/null 2>&1 || c23=c2x

# (1) the driver needs no libc and draws no warning, at either standard
for std in c11 "${c23}"; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -Wpedantic -Werror -I "${C}" \
    -c "${C}/uart_driver.c" -o /dev/null \
    || { echo "  FAIL: uart_driver.c not freestanding-clean under -std=${std}"; exit 1; }
done
echo "  PASS uart_driver.c freestanding-clean (C11 + C23, -Wpedantic -Werror)"

# (2) the trap fails closed: the three units built with no tracing reach the device with no access,
# and the harness says so instead of printing a transcript of a plain array
"${UART_MAIN}" > "${tmp}/plain.out" 2> "${tmp}/plain.err"; plain_rc=$?
if [ "${plain_rc}" = "3" ] && grep -q "the register window is not trapped" "${tmp}/plain.err" \
   && ! grep -q '^-- init' "${tmp}/plain.out"; then
  echo "  PASS the untraced uart_main refuses its transcript (exit 3: the register window is not trapped)"
else
  echo "  FAIL: the untraced uart_main exited ${plain_rc}, not 3 with the untrapped-window refusal"
  head -3 "${tmp}/plain.err" | sed 's/^/    /'
  exit 1
fi
# ... and its script console is the device: the reset LSR, and FCR = 0xC1 answered by IIR[7:6] = 11
printf '%s\n' "config pc16550d 1 0 1 0 1 207 16 1 4 8 14 1 0 1" "r 0x5" "w 0x2 0xc1" "r 0x2" \
  | "${UART_MAIN}" --script > "${tmp}/console.out" 2>&1; console_rc=$?
printf '%s\n' "config pc16550d" "r 0x5 0x60" "r 0x2 0xc1" > "${tmp}/console.want"
if [ "${console_rc}" = "0" ] && cmp -s "${tmp}/console.out" "${tmp}/console.want"; then
  echo "  PASS the sim console: LSR 0x60 at reset, and FCR 0xC1 answered by IIR 0xC1 (a 16550A)"
else
  echo "  FAIL: the sim console (exit ${console_rc}) did not answer as a 16550A at reset"
  sed 's/^/    /' "${tmp}/console.out" | head -5
  exit 1
fi

# The units as they are, beside a placement's generated header: the program's, in units/ (axi16550),
# and the misplaced driver's, in laid-out/ (pc16550d).
stage() {  # <dir> <placement>
  mkdir -p "$1"
  for unit in "${UNITS[@]}"; do cp "${C}/${unit}" "$1/"; done
  cp "${FIXTURES}/uart16550_$2.h" "$1/uart16550_regs.h"
}
# <driver source> <its directory> <name> [flags...]: the driver traced, linked with the harness and the
# device built in ${d}, never traced (their own loads and stores would come back through the shim)
link_driver() {
  local src="$1" dir="$2" name="$3"; shift 3
  "${CC}" -std=c11 -O2 "${TRACE}" "$@" -I "${dir}" -c "${src}" -o "${d}/${name}.o" 2> "${tmp}/build.err" \
    && "${CC}" "${d}/${name}.o" "${d}/uart_main.o" "${d}/sim16550.o" -o "${d}/${name}" 2>> "${tmp}/build.err" \
    || { echo "  FAIL: ${name} did not build"; head -5 "${tmp}/build.err" | sed 's/^/    /'; exit 1; }
}

# (3) the linked runs: they need the load/store tracing, which is Clang's
if printf 'unsigned f(volatile unsigned *p){ *p = 1u; return *p; }\n' \
    | "${CC}" "${TRACE}" -x c -c -o /dev/null - > /dev/null 2>&1; then
  d="${tmp}/units"
  stage "${d}" "${PLACEMENT}"
  for unit in uart_main sim16550; do
    "${CC}" -std=c11 -O2 -I "${d}" -c "${d}/${unit}.c" -o "${d}/${unit}.o" 2> "${tmp}/build.err" \
      || { echo "  FAIL: ${unit}.c did not build"; head -5 "${tmp}/build.err" | sed 's/^/    /'; exit 1; }
  done
  # the driver as the oracle emits it, and as bcir-cc emits it
  (cd "${d}" && "${PYTHON}" -m bcir.frontends.cfront --linkable -o driver_oracle.c uart_driver.c) 2> "${tmp}/oracle.err" \
    || { echo "  FAIL: the oracle's --linkable emit of uart_driver.c"; head -5 "${tmp}/oracle.err" | sed 's/^/    /'; exit 1; }
  "${BCIR_CC}" --linkable "${d}/uart_driver.c" > "${d}/driver_twin.c" 2> "${tmp}/twin.err" \
    || { echo "  FAIL: bcir-cc --linkable uart_driver.c"; head -5 "${tmp}/twin.err" | sed 's/^/    /'; exit 1; }
  link_driver "${d}/uart_driver.c" "${d}" uart_main_source
  link_driver "${d}/driver_oracle.c" "${d}" uart_main_oracle
  link_driver "${d}/driver_twin.c" "${d}" uart_main_twin
  for rail in source oracle twin; do
    "${d}/uart_main_${rail}" > "${d}/${rail}.txt" 2> "${tmp}/run.err"; rc=$?
    [ "${rc}" = "0" ] \
      || { echo "  FAIL: uart_main with the ${rail} driver exited ${rc}"; head -3 "${tmp}/run.err" | sed 's/^/    /'; exit 1; }
  done
  verdict="$(grep '^verdict:' "${d}/source.txt")"
  cmp -s "${d}/twin.txt" "${d}/oracle.txt" \
    && echo "  PASS bcir-cc --linkable's driver, linked with the sim and the harness, prints the oracle-emitted build's transcript" \
    || { echo "  FAIL: the twin-emitted build's transcript is not the oracle-emitted build's"; diff "${d}/oracle.txt" "${d}/twin.txt" | head -8 | sed 's/^/    /'; exit 1; }
  cmp -s "${d}/oracle.txt" "${d}/source.txt" \
    && echo "  PASS the oracle's --linkable driver prints the source build's transcript, byte for byte (the bus trace included)" \
    || { echo "  FAIL: the oracle-emitted build's transcript is not the source build's"; diff "${d}/source.txt" "${d}/oracle.txt" | head -8 | sed 's/^/    /'; exit 1; }
  cmp -s "${d}/source.txt" "${GOLDEN}" \
    && echo "  PASS the transcript is runtime/c/uart_main.golden (${PLACEMENT}); ${verdict}" \
    || { echo "  FAIL: the source build's transcript is not runtime/c/uart_main.golden"; diff "${GOLDEN}" "${d}/source.txt" | head -8 | sed 's/^/    /'; exit 1; }

  # (4) hazard H-A at run time: the driver laid out for the byte-strided pc16550d, on the AXI device
  stage "${tmp}/laid-out" pc16550d
  link_driver "${tmp}/laid-out/uart_driver.c" "${tmp}/laid-out" uart_main_misplaced
  "${d}/uart_main_misplaced" > /dev/null 2> "${tmp}/misplaced.err"; rc=$?
  if [ "${rc}" = "4" ] && grep -q "do not fit placement ${PLACEMENT}" "${tmp}/misplaced.err"; then
    echo "  PASS a driver laid out for pc16550d is refused on the ${PLACEMENT} device (exit 4: hazard H-A)"
  else
    echo "  FAIL: the misplaced driver exited ${rc}, not 4 with the H-A refusal"
    exit 1
  fi
else
  echo "  SKIP uartdrv linked runs: CC has no load/store tracing (${TRACE} is Clang's; run by the gate's clang; the freestanding, untraced-refusal and console checks ran)"
fi
