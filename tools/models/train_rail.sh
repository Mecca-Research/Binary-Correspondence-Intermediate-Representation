#!/usr/bin/env bash
# The train rail (TRAIN-RAIL; docs/research/BCIR_TRAINING_ACCELERATION_ROADMAP.md sections 3 and 5):
# the IR's build trains the BCIR-native decoder on the corpus recipe -- `training/` beside `docs/`,
# read as data under the one-way corpus law of AGENTS.md section 2, never imported -- through
# tools/models/build_lab_model.py (the corpus standard's preparation, BCIR's BPE, the native C
# trainer compiled with CC, the BCIRQ8 export and its sample), then reads the corpus witness: every
# tracked Markdown file of both trees is in what it trained on (tools/models/train_rail.py).
#
#   usage: train_rail.sh <output-dir> [steps]    steps 0 or absent: the recipe's own count (the full run)
#   env:   CC      the trainer's compiler (NativeDecoder.build reads it; default: clang, then cc)
#          PYTHON  the interpreter (default python3)
#          BCIR_TRAIN_RECIPE  another recipe (default tools/models/configs/bcir-corpus-860k.json)
#          BCIR_TRAIN_THREADS the trainer's threads under the recipe's "host" policy (default: the
#                  CPUs this process may run on; the state is the same at every count)
#
# CTest runs this file as `train-corpus` (label train, BCIR_TRAIN_STEPS steps), BCIR Make as the task
# `train.corpus` through tools/build/run_section.sh (the same number of steps, tools/build/bcirfile.py),
# CI's qualification job before qualifying the model it trains. Exit 0 PASS, 1 FAIL, 2 when it cannot
# run (docs/security/laws.md L1).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-python3}"
RECIPE="${BCIR_TRAIN_RECIPE:-tools/models/configs/bcir-corpus-860k.json}"
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "usage: train_rail.sh <output-dir> [steps]" >&2
  exit 2
fi
out="$1"
steps="${2:-0}"
case "${steps}" in
  ''|*[!0-9]*) echo "train-rail: UNUSABLE: steps must be a non-negative integer, got '${steps}'" >&2; exit 2 ;;
esac
if ! command -v "${PY}" >/dev/null 2>&1; then
  echo "train-rail: UNUSABLE: no interpreter '${PY}' (PYTHON)" >&2
  exit 2
fi
cd "${ROOT}" || exit 2
if [ ! -f "${RECIPE}" ]; then
  echo "train-rail: UNUSABLE: no recipe ${RECIPE} (BCIR_TRAIN_RECIPE)" >&2
  exit 2
fi
args=("${RECIPE}" --output-dir "${out}")
if [ "${steps}" -gt 0 ]; then
  args+=(--steps "${steps}")
fi
echo "train-rail: recipe=${RECIPE} steps=$([ "${steps}" -gt 0 ] && echo "${steps}" || echo recipe) cc=${CC:-default} python=${PY}"
if ! "${PY}" tools/models/build_lab_model.py "${args[@]}"; then
  echo "train-rail: FAIL: the build of ${RECIPE} did not pass"
  exit 1
fi
"${PY}" tools/models/train_rail.py check --recipe "${RECIPE}" --corpus-json "${out}/corpus.json"
rc=$?
if [ "${rc}" -eq 2 ]; then
  echo "train-rail: UNUSABLE: the corpus witness could not be read" >&2
  exit 2
fi
if [ "${rc}" -ne 0 ]; then
  echo "train-rail: FAIL: train.corpus.unselected is not 0 (the model did not see every tracked Markdown file of docs/ and training/)"
  exit 1
fi
card="$("${PY}" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest()[:16])' "${out}/model_card.json")" || exit 2
echo "train-rail: PASS recipe=$(basename "${RECIPE}") steps=$([ "${steps}" -gt 0 ] && echo "${steps}" || echo recipe) card=${card}"
