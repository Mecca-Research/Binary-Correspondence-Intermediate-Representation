#!/usr/bin/env bash

require_cmd() {
  local tool="$1"
  if ! command -v "${tool}" >/dev/null; then
    printf 'error: missing required command: %s\n' "${tool}" >&2
    exit 127
  fi
}

llvm_major_version() {
  local tool="$1"
  "${tool}" --version | sed -n '1s/.*version \([0-9][0-9]*\).*/\1/p'
}

require_llvm_tool() {
  local tool="$1"
  local min_major="${2:-15}"
  if ! command -v "${tool}" >/dev/null; then
    printf 'error: missing required LLVM tool: %s\n' "${tool}" >&2
    printf 'Install LLVM %s or newer; CI validates with the Ubuntu llvm package (LLVM 18.1.3 on ubuntu-24.04).\n' "${min_major}" >&2
    exit 127
  fi
  local major
  major="$(llvm_major_version "${tool}")"
  if [[ -z "${major}" ]]; then
    printf 'error: could not determine %s version from --version output. LLVM %s or newer is required.\n' "${tool}" "${min_major}" >&2
    exit 127
  fi
  if (( major < min_major )); then
    printf 'error: %s reports LLVM major version %s, but LLVM %s or newer is required for opaque-pointer IR.\n' "${tool}" "${major}" "${min_major}" >&2
    exit 127
  fi
}

require_file() {
  local path="$1"
  local hint="$2"
  if [[ ! -f "${path}" ]]; then
    printf 'error: missing required input file: %s\n' "${path}" >&2
    if [[ -n "${hint}" ]]; then
      printf '%s\n' "${hint}" >&2
    fi
    exit 66
  fi
}

require_bcir_as() {
  local tool="tools/bcir-as/bcir-as"
  require_file "${tool}" "Run from the repository root; ${tool} must resolve to the checked-in Python assembler."
  if [[ ! -x "${tool}" ]]; then
    printf 'error: %s is not executable. Restore its executable bit or run: chmod +x tools/bcir-as/bcir-as.py\n' "${tool}" >&2
    exit 126
  fi
  if ! python3 -m py_compile "${tool}"; then
    printf 'error: %s failed Python syntax preflight; validate_phase4.sh cannot generate build/vector_add.generated.ll.\n' "${tool}" >&2
    exit 65
  fi
}
