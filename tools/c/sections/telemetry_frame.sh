#!/usr/bin/env bash
# telemetry_frame: UART telemetry frame (#telemetry-frame): freestanding compile (C11 + C23) + byte-identical re-encode
#
# One section of tools/c/check_runtime.sh, run against harness binaries built elsewhere: BCIR Make
# builds them from runtime/manifest.json and runs this script as a task, whose verdict the gate
# shows; the CMake project builds them from the same manifest and runs this script as the
# `c-section-telemetry_frame` CTest entry. The body is the gate's section text moved here (BUILD-2,
# docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and tools/build/section_parity.py holds the
# two builds' outputs byte-identical.
#
#   usage: telemetry_frame.sh <test_telemetry_frame>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: $(basename "$0") <test_telemetry_frame>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# bcir_telemetry_frame.c is the C twin of bcir/telemetry_frame.py -- the framed, CRC-sealed,
# resync-able telemetry transport (T2). The producer drains TelemetryRing and frames the 56-byte
# <7q> records; the host decoder reuses RT3. It REUSES bcir_crc32 from bcir_runtime.c (so the C
# and Python (zlib.crc32) CRCs agree). Self-skipping/non-fatal without a C compiler (the section
# is only reached past the CC guard at the top, so a missing CC already exited 0 cleanly above).
# Python-encode a fixed DataDNA batch into one frame; C decode + re-encode; assert byte-identical.
"${PYTHON}" -c "
from bcir.telemetry import DataDNA
from bcir.telemetry_frame import encode_frame
recs=[DataDNA(segment_id='',claim_id=1,cycles=100,bytes=200,misses=5,thermal=40,voltage=10,utilization=30),
      DataDNA(segment_id='',claim_id=2,cycles=999999,bytes=4096,misses=0,thermal=0,voltage=0,utilization=100),
      DataDNA(segment_id='',claim_id=3,cycles=-50,bytes=0,misses=100,thermal=99,voltage=50,utilization=0)]
open('${tmp}/tframe.bin','wb').write(encode_frame(recs, seq=7, timestamp=123456))
" || { echo "  FAIL: python frame encode"; exit 1; }
tfout="$("${HARNESS}" "${tmp}/tframe.bin" "${tmp}/tframe_reenc.bin")" || { echo "  FAIL: C frame decode"; echo "${tfout}"; exit 1; }
echo "${tfout}" | grep -q "^OK " || { echo "  FAIL: C frame decode did not OK"; echo "${tfout}"; exit 1; }
if cmp -s "${tmp}/tframe.bin" "${tmp}/tframe_reenc.bin"; then
  echo "  PASS #telemetry-frame (C bcir_tf decode + re-encode == Python encode_frame, byte-identical; bcir_crc32 == zlib.crc32)"
else
  echo "  FAIL: telemetry-frame bytes differ from the Python encoding"; exit 1
fi
