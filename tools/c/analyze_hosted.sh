#!/usr/bin/env bash
# Hosted C ownership static analysis: clang's analyzer over each hosted unit, one whole-file
# analysis per unit, ANALYZE_JOBS at a time (two by default, AGENTS.md's cap; CI's runners pass
# their core count, since each analysis is single-threaded and bcir_cfront.c alone dominates). A
# unit fails on a nonzero exit OR on any diagnostic at all: `--analyze` reports its findings on
# stderr and still exits 0, so the emptiness check is the part that enforces "no diagnostics".
#
# The one list of the units it analyses, read by every caller: CI's C analysis job (Clang 18), the
# clang 23 analysis cell, and tools/local/check_latest.sh's `analyze` leg.
#
#   usage: CLANG=<clang> [ANALYZE_JOBS=N] analyze_hosted.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLANG="${CLANG:-clang}"
JOBS="${ANALYZE_JOBS:-2}"
[ $# -eq 0 ] || { echo "usage: CLANG=<clang> [ANALYZE_JOBS=N] $(basename "$0")" >&2; exit 2; }
[[ "${JOBS}" =~ ^[1-9][0-9]{0,2}$ ]] || { echo "analyze-hosted: UNUSABLE: ANALYZE_JOBS=${JOBS} is no count of workers" >&2; exit 2; }
command -v "${CLANG}" > /dev/null 2>&1 || { echo "analyze-hosted: UNUSABLE: no ${CLANG}" >&2; exit 2; }
sources="bcir_tensor.c bcir_decoder_train.c bcir_cpp.c bcir_cfront.c bcir_artifact_bundle.c bcir_q8_model.c bcir_q4_kernel.c bcir_ai_kernels.c bcir_ai_microbench.c bcir_llama.c bcir_runtime_channel.c bcir_verify.c bcir_make.c"
workdir="$(mktemp -d)"
trap 'rm -rf "${workdir}"' EXIT
export CLANG ROOT workdir
# shellcheck disable=SC2016  # the unit's script is expanded by the bash xargs starts, per unit
printf '%s\n' ${sources} | xargs -P "${JOBS}" -n 1 bash -c '
  if "${CLANG}" --analyze -std=c11 -I "${ROOT}/runtime/c" "${ROOT}/runtime/c/$1" -o /dev/null \
       2> "${workdir}/$1.log"; then
    [ -s "${workdir}/$1.log" ] && echo failed > "${workdir}/$1.rc"
  else
    echo failed > "${workdir}/$1.rc"
  fi
  exit 0' _
status=0
count=0
for source in ${sources}; do
  count=$((count + 1))
  if [ ! -f "${workdir}/${source}.log" ]; then
    echo "::error::${CLANG} --analyze did not run on runtime/c/${source}" >&2
    status=1
  elif [ -f "${workdir}/${source}.rc" ]; then
    echo "::error::${CLANG} --analyze reported diagnostics in runtime/c/${source}" >&2
    cat "${workdir}/${source}.log" >&2
    status=1
  fi
done
[ "${status}" -eq 0 ] && echo "analyze-hosted: PASS (${count} units, no diagnostics; $("${CLANG}" --version | head -1))"
exit "${status}"
