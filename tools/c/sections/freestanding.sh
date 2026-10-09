#!/usr/bin/env bash
# freestanding: the freestanding core compiled -ffreestanding -nostdlib, C11 + C23
#
# One section of tools/c/check_runtime.sh (BUILD-8): the gate's freestanding compiles, each unit of
# the freestanding core and the claim-graph IR header compiled with no libc at C11 and at C23, moved
# here in the gate's order when BCIR Make took over the gate's builds (docs/BCIR_BUILD_ROADMAP.md).
# BCIR Make runs it as a task and the CMake project as the `c-section-freestanding` CTest entry. It
# takes no binary: it judges the sources under CC, which the caller names. The manifest marks it
# compiler_only, and tools/build/section_parity.py holds its output stable over two runs; M8 of
# tools/build/manifest.py holds the units it compiles to the manifest's freestanding_checks.
#
#   usage: CC=<compiler> freestanding.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
[ $# -eq 0 ] || { echo "usage: CC=<compiler> freestanding.sh" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> freestanding.sh" >&2; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT
# The C23 spelling CC accepts: GCC 13 knows the standard only as -std=c2x, the fallback every
# section that compiles C23 takes (and CMake's spelling there, docs/BCIR_BUILD_ROADMAP.md §6). The
# gate compiled these with clang alone; BCIR Make and CMake run the section under GCC too.
c23=c23
printf 'int bcir_c23_probe;\n' | "${CC}" -std=c23 -x c -c -o /dev/null - > /dev/null 2>&1 || c23=c2x

for std in c11 "${c23}"; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -c "${C}/bcir_runtime.c" -o /dev/null \
    || { echo "  FAIL: runtime not freestanding-clean under -std=${std}"; exit 1; }
done
echo "  PASS freestanding (C11 + C23; ABI static_assert holds)"

for std in c11 "${c23}"; do
  for unit in bcir_artifact_bundle.c bcir_sha256.c bcir_ed25519.c bcir_plan_sign.c; do
    "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -Werror -I "${C}" \
      -c "${C}/${unit}" -o /dev/null \
      || { echo "  FAIL: ${unit} not freestanding-clean under -std=${std}"; exit 1; }
  done
done
echo "  PASS bcir_artifact_bundle.c bcir_sha256.c bcir_ed25519.c bcir_plan_sign.c freestanding-clean (C11 + C23)"

# ETL binary-record decoder: freestanding compile (C11 + C23)
# bcir_binrec.c is the C twin of bcir/etl/binary.py (a second binary trust boundary).
for std in c11 "${c23}"; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -I "${C}" -c "${C}/bcir_binrec.c" -o /dev/null \
    || { echo "  FAIL: bcir_binrec not freestanding-clean under -std=${std}"; exit 1; }
done
echo "  PASS bcir_binrec freestanding (C11 + C23)"

for std in c11 "${c23}"; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -I "${C}" -c "${C}/bcir_exec.c" -o /dev/null \
    || { echo "  FAIL: bcir_exec not freestanding-clean under -std=${std}"; exit 1; }
done
echo "  PASS bcir_exec.c freestanding-clean (C11 + C23)"

for std in c11 "${c23}"; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -I "${C}" -c "${C}/bcir_encode.c" -o /dev/null \
    || { echo "  FAIL: bcir_encode not freestanding-clean under -std=${std}"; exit 1; }
done
echo "  PASS bcir_encode.c freestanding-clean (C11 + C23)"

for std in c11 "${c23}"; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -Werror -I "${C}" \
    -c "${C}/bcir_control_plane.c" -o /dev/null \
    || { echo "  FAIL: control plane not freestanding-clean under -std=${std}"; exit 1; }
done
echo "  PASS bcir_control_plane.c freestanding-clean (C11 + C23)"

for std in c11 "${c23}"; do
  for unit in bcir_ring.c bcir_telemetry_envelope.c; do
    "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -Wconversion -Wpedantic -Werror \
      -I "${C}" -c "${C}/${unit}" -o /dev/null \
      || { echo "  FAIL: ${unit} not freestanding-clean under -std=${std}"; exit 1; }
  done
done
echo "  PASS bcir_ring.c bcir_telemetry_envelope.c freestanding-clean (C11 + C23)"

for std in c11 "${c23}"; do
  for unit in bcir_handoff.c bcir_shard_manifest.c bcir_hydrate.c; do
    "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -Wconversion -Wpedantic -Werror \
      -I "${C}" -c "${C}/${unit}" -o /dev/null \
      || { echo "  FAIL: ${unit} not freestanding-clean under -std=${std}"; exit 1; }
  done
done
echo "  PASS bcir_handoff.c bcir_shard_manifest.c bcir_hydrate.c freestanding-clean (C11 + C23)"

for std in c11 "${c23}"; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -Wconversion -Wpedantic -Werror \
    -I "${C}" -c "${C}/bcir_kplan.c" -o /dev/null \
    || { echo "  FAIL: bcir_kplan.c not freestanding-clean under -std=${std}"; exit 1; }
done
echo "  PASS bcir_kplan.c freestanding-clean (C11 + C23)"

for std in c11 "${c23}"; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -I "${C}" -c "${C}/bcir_telemetry_frame.c" -o /dev/null \
    || { echo "  FAIL: bcir_telemetry_frame not freestanding-clean under -std=${std}"; exit 1; }
done
echo "  PASS bcir_telemetry_frame.c freestanding-clean (C11 + C23)"

printf '#include "bcir_cir.h"\nint probe(void){return (int)sizeof(bcir_claim)+(int)BCIR_OP_LOAD+(int)BCIR_DOM_MMIO;}\n' > "${tmp}/cir_probe.c"
for std in c11 "${c23}"; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -I "${C}" -c "${tmp}/cir_probe.c" -o /dev/null \
    || { echo "  FAIL: bcir_cir.h not freestanding-clean under -std=${std}"; exit 1; }
done
echo "  PASS bcir_cir.h freestanding IR (C11 + C23)"

# bcir_plan.c + bcir_hydrate.c are freestanding (the driver-embeddable planner + StreamPack
# writer that feed the existing bcir_exec.c) -- the loop closes with no Python.
for f in bcir_plan.c bcir_hydrate.c; do
  for std in c11 "${c23}"; do
    "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -I "${C}" -c "${C}/${f}" -o /dev/null \
      || { echo "  FAIL: ${f} not freestanding-clean under -std=${std}"; exit 1; }
  done
done
echo "  PASS bcir_plan.c bcir_hydrate.c freestanding-clean (C11 + C23)"

for std in c11 "${c23}"; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -Werror -I "${C}" \
    -c "${C}/bcir_per.c" -o /dev/null \
    || { echo "  FAIL: bcir_per is not freestanding-clean under -std=${std}"; exit 1; }
done
echo "  PASS bcir_per.c freestanding-clean (C11 + C23)"

for std in c11 "${c23}"; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -Werror -I "${C}" \
    -c "${C}/bcir_xer.c" -o /dev/null \
    || { echo "  FAIL: bcir_xer is not freestanding-clean under -std=${std}"; exit 1; }
done
echo "  PASS bcir_xer.c freestanding-clean (C11 + C23)"

for std in c11 "${c23}"; do
  # bcir_crc32 is DECLARED here and DEFINED in bcir_runtime.c (the same discipline
  # bcir_telemetry_frame.h follows), so this is a compile and not a link.
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -Werror -I "${C}" \
    -c "${C}/bcir_jer.c" -o /dev/null \
    || { echo "  FAIL: bcir_jer is not freestanding-clean under -std=${std}"; exit 1; }
done
echo "  PASS bcir_jer.c freestanding-clean (C11 + C23)"

for std in c11 "${c23}"; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -Werror -I "${C}" \
    -c "${C}/bcir_emit.c" -o /dev/null \
    || { echo "  FAIL: bcir_emit is not freestanding-clean under -std=${std}"; exit 1; }
done
echo "  PASS bcir_emit.c freestanding-clean (C11 + C23)"

for std in c11 c2x; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -Werror -I "${C}" \
    -c "${C}/bcir_per_plan.c" -o /dev/null \
    || { echo "  FAIL: bcir_per_plan is not freestanding-clean under -std=${std}"; exit 1; }
done
echo "  PASS bcir_per_plan.c freestanding-clean (C11 + C2x)"

for std in c11 "${c23}"; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -Werror -I "${C}" \
    -c "${C}/bcir_oer.c" -o /dev/null \
    || { echo "  FAIL: bcir_oer is not freestanding-clean under -std=${std}"; exit 1; }
done
echo "  PASS bcir_oer.c freestanding-clean (C11 + C23)"

for std in c11 "${c23}"; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -I "${C}" \
    -c "${C}/bcir_asn1_streampack.c" -o /dev/null \
    || { echo "  FAIL: bcir_asn1_streampack not freestanding-clean under -std=${std}"; exit 1; }
done
echo "  PASS bcir_asn1_streampack.c freestanding-clean (C11 + C23)"
