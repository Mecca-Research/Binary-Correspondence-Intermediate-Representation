#!/usr/bin/env bash
# One section of tools/c/check_runtime.sh as a BCIR Make task (BUILD-7): run the section's script over
# the binaries bcir-make built, with CC and PYTHON the BCIRfile's declared tools -- the runner hands a
# task the tools it `uses` as BCIR_TOOL_<NAME> -- and record the section's verdict, its stdout, its
# stderr and its exit status, as the task's three outputs. The task fails when the section does, with
# the verdict written all the same, so a failure can be read.
#
#   usage: run_section.sh <stdout> <stderr> <status> <section-script> [binary...]
set -uo pipefail
[ $# -ge 4 ] || { echo "usage: run_section.sh <stdout> <stderr> <status> <section-script> [binary...]" >&2; exit 2; }
out=$1; err=$2; code=$3; script=$4; shift 4
export CC="${BCIR_TOOL_CC:?the task uses no cc (BCIR_TOOL_CC)}"
export PYTHON="${BCIR_TOOL_PYTHON:?the task uses no python (BCIR_TOOL_PYTHON)}"
"${BASH}" "${script}" "$@" > "${out}" 2> "${err}"
status=$?
echo "${status}" > "${code}"
exit "${status}"
