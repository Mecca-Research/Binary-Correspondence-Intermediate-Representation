#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C="${ROOT}/runtime/c"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

PYTHON="${PYTHON:-}"
if [ -z "${PYTHON}" ]; then
  if command -v python3 >/dev/null 2>&1; then PYTHON=python3; else PYTHON=python; fi
fi

CC_BIN="${CC:-}"
if [ -z "${CC_BIN}" ]; then
  for candidate in clang cc gcc; do
    if command -v "${candidate}" >/dev/null 2>&1; then CC_BIN="${candidate}"; break; fi
  done
fi
if [ -z "${CC_BIN}" ]; then
  echo "memory-discipline: no C compiler available" >&2
  exit 1
fi

model="${tmp}/memory-discipline-toy.bcirq8"
PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON}" - "${model}" <<'PY'
import sys
from pathlib import Path
from bcir.tests.test_model_weights_io import _write

_write(Path(sys.argv[1]))
PY

link_args=()
host_defines=()
case "$(uname -s 2>/dev/null || true)" in
  # UCRT deprecates portable ISO C interfaces (fopen, getenv, gmtime, and the
  # bounded uses of strcpy/strncpy) in favor of Microsoft-only *_s variants.
  # Keep the implementation portable and opt in explicitly for hosted Windows
  # builds; all other diagnostics remain errors.
  MINGW*|MSYS*|CYGWIN*) host_defines=(-D_CRT_SECURE_NO_WARNINGS) ;;
  *) link_args=(-lm) ;;
esac

"${PYTHON}" "${ROOT}/tools/c/check_memory_discipline.py"

strict=(-std=c11 -Wall -Wextra -Wpedantic -Werror "${host_defines[@]}" -I "${C}")
compat_warnings=(-Wno-misleading-indentation)
if ! "${CC_BIN}" --version 2>/dev/null | head -n 1 | grep -qi clang; then
  compat_warnings+=(-Wno-format-truncation)
fi
for unit in bcir_runtime_channel.c bcir_cpp.c bcir_q8_model.c bcir_q4_kernel.c \
            bcir_ai_kernels.c bcir_ai_microbench.c bcir_llama.c bcir_verify.c \
            bcir_cc.c bcir_llama_cli.c bcir_microbench.c; do
  "${CC_BIN}" "${strict[@]}" -c "${C}/${unit}" -o "${tmp}/${unit%.c}.o"
done

sources=(
  "${C}/test_memory_discipline.c"
  "${C}/bcir_runtime_channel.c"
  "${C}/bcir_cfront.c"
  "${C}/bcir_cpp.c"
  "${C}/bcir_verify.c"
  "${C}/bcir_runtime.c"
  "${C}/bcir_q8_model.c"
  "${C}/bcir_decode.c"
  "${C}/bcir_ai_kernels.c"
  "${C}/bcir_llama.c"
)

ai_sources=(
  "${C}/test_ai_kernels.c"
  "${C}/bcir_ai_kernels.c"
  "${C}/bcir_q4_kernel.c"
)

"${CC_BIN}" "${strict[@]}" -O1 "${compat_warnings[@]}" \
  "${sources[@]}" -o "${tmp}/memory-discipline" "${link_args[@]}"
"${tmp}/memory-discipline" "${model}"
"${CC_BIN}" "${strict[@]}" -O2 -ffp-contract=off \
  "${ai_sources[@]}" -o "${tmp}/ai-kernels" "${link_args[@]}"
"${tmp}/ai-kernels"

if [ "${MEMORY_DISCIPLINE_SANITIZE:-0}" = 1 ]; then
  case "$(uname -s 2>/dev/null || true)" in
    Linux*)
      "${CC_BIN}" -std=c11 -O1 -g -fno-omit-frame-pointer \
        -fsanitize=address,undefined "${compat_warnings[@]}" \
        -I "${C}" "${sources[@]}" \
        -o "${tmp}/memory-discipline-sanitize" "${link_args[@]}"
      ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
      UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
        "${tmp}/memory-discipline-sanitize" "${model}"
      "${CC_BIN}" -std=c11 -O1 -g -fno-omit-frame-pointer -ffp-contract=off \
        -fsanitize=address,undefined "${compat_warnings[@]}" -I "${C}" \
        "${ai_sources[@]}" -o "${tmp}/ai-kernels-sanitize" "${link_args[@]}"
      ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
      UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
        "${tmp}/ai-kernels-sanitize"
      ;;
    *)
      echo "memory-discipline: sanitizer stage is Linux-only; strict/fault gates passed"
      ;;
  esac
fi

echo "memory-discipline: all gates passed"
