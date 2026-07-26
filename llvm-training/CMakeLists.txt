cmake_minimum_required(VERSION 3.20)
project(LLVMTrainingQualityGates NONE)

find_program(BASH_EXECUTABLE bash REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(TRAINING_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(TOOLS_DIR "${TRAINING_ROOT}/tools")
set(OPTIONAL_RUNNER "${TOOLS_DIR}/run-if-tools.sh")
set(LLVM_TRAINING_LIT_JOBS "2" CACHE STRING
    "Maximum llvm-lit workers used by the aggregate training gate")
if(NOT LLVM_TRAINING_LIT_JOBS MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "LLVM_TRAINING_LIT_JOBS must be a positive integer")
endif()

# The gates below are INDEPENDENT of one another: llvm-training-check DEPENDS on all of them
# and none of them depends on another, so a parallel `cmake --build -j N` could run them at
# once.  USES_TERMINAL prevents that -- it places a target in the generator's console pool,
# which has depth 1, so every gate is serialized no matter what -j says.  That is the right
# default for a developer watching live output, and the wrong one for CI, where the gate is
# ~16 short bash/python processes whose cost is dominated by process startup (measured on the
# windows-latest runner: 124s serial).  Turning it off lets -j do its job; each gate's output
# is then captured and printed by the generator instead of streaming.
option(LLVM_TRAINING_USES_TERMINAL
       "Give each training gate the console: live output, but forces serial execution" ON)
if(LLVM_TRAINING_USES_TERMINAL)
  set(TRAINING_TERMINAL USES_TERMINAL)
else()
  set(TRAINING_TERMINAL "")
endif()

function(add_training_script_target target script)
  add_custom_target(${target}
    COMMAND "${BASH_EXECUTABLE}" "${TOOLS_DIR}/${script}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
    ${TRAINING_TERMINAL}
    VERBATIM)
endfunction()

function(add_optional_training_target target label script)
  add_custom_target(${target}
    COMMAND "${BASH_EXECUTABLE}" "${OPTIONAL_RUNNER}" "${label}" ${ARGN} --
            "${BASH_EXECUTABLE}" "${TOOLS_DIR}/${script}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
    ${TRAINING_TERMINAL}
    VERBATIM)
endfunction()

add_training_script_target(llvm-training-verify-manifest verify-manifest.sh)
add_custom_target(llvm-training-verify-exercise-manifests
  COMMAND "${Python3_EXECUTABLE}" "${TOOLS_DIR}/verify-exercise-manifests.py"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
  ${TRAINING_TERMINAL}
  VERBATIM)
add_custom_target(llvm-training-autograder-self-test
  COMMAND "${Python3_EXECUTABLE}" "${TOOLS_DIR}/grade-exercises.py" --self-test --format json
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
  ${TRAINING_TERMINAL}
  VERBATIM)
add_custom_target(llvm-training-verify-dataset-export
  COMMAND "${Python3_EXECUTABLE}" "${TOOLS_DIR}/verify-dataset-export.py"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
  ${TRAINING_TERMINAL}
  VERBATIM)
add_training_script_target(llvm-training-verify-csv-schema verify-csv-schema.sh)
add_custom_target(llvm-training-verify-binary-analysis-evidence
  COMMAND "${Python3_EXECUTABLE}" "${TOOLS_DIR}/verify-binary-analysis-evidence.py"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
  ${TRAINING_TERMINAL}
  VERBATIM)
add_custom_target(llvm-training-check-binary-analysis-fixtures
  COMMAND "${Python3_EXECUTABLE}" "${TOOLS_DIR}/generate-binary-analysis-fixtures.py" --check
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
  ${TRAINING_TERMINAL}
  VERBATIM)
add_training_script_target(llvm-training-verify-opaque-pointers verify-opaque-pointers.sh)
add_custom_target(llvm-training-verify-mlir-examples
  COMMAND "${Python3_EXECUTABLE}" "${TOOLS_DIR}/verify-mlir-lowering.py"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
  ${TRAINING_TERMINAL}
  VERBATIM)
add_training_script_target(llvm-training-verify-bcir-mapping verify-bcir-mapping.sh)

add_optional_training_target(llvm-training-verify-examples "LLVM example verifier" verify-examples.sh llvm-as opt)
add_optional_training_target(llvm-training-verify-exercises "LLVM exercise verifier" verify-exercises.sh llvm-as opt)
add_optional_training_target(llvm-training-verify-invalid-fixtures "LLVM invalid-fixture verifier" verify-invalid-fixtures.sh llvm-as opt)
add_optional_training_target(llvm-training-verify-adversarial-fixtures "LLVM adversarial-fixture verifier" verify-adversarial-fixtures.sh llvm-as opt)
add_optional_training_target(llvm-training-smoke-llc "portable llc smoke" smoke-llc.sh llc)
add_optional_training_target(llvm-training-smoke-lli "portable lli smoke" smoke-lli.sh lli)
add_optional_training_target(llvm-training-verify-opt-diff "LLVM opt-diff verifier" verify-opt-diff.sh opt)

# Windows runners can expose Microsoft's unrelated lit.exe.  Accept the generic
# name only when its version banner identifies LLVM lit.
add_custom_target(llvm-training-lit
  COMMAND "${BASH_EXECUTABLE}" -c
          "if command -v llvm-lit >/dev/null 2>&1; then exec llvm-lit -j '${LLVM_TRAINING_LIT_JOBS}' -sv '${TRAINING_ROOT}/tests'; elif command -v lit >/dev/null 2>&1 && lit --version 2>&1 | grep -Eq '^lit( version)? [0-9]'; then exec lit -j '${LLVM_TRAINING_LIT_JOBS}' -sv '${TRAINING_ROOT}/tests'; else echo 'LLVM llvm-lit/lit not present; skipping llvm-training lit tests'; fi"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
  ${TRAINING_TERMINAL}
  VERBATIM)

add_custom_target(llvm-training-check
  DEPENDS
    llvm-training-verify-manifest
    llvm-training-verify-exercise-manifests
    llvm-training-autograder-self-test
    llvm-training-verify-dataset-export
    llvm-training-verify-csv-schema
    llvm-training-verify-binary-analysis-evidence
    llvm-training-check-binary-analysis-fixtures
    llvm-training-verify-opaque-pointers
    llvm-training-verify-mlir-examples
    llvm-training-verify-bcir-mapping
    llvm-training-verify-examples
    llvm-training-verify-exercises
    llvm-training-verify-invalid-fixtures
    llvm-training-verify-adversarial-fixtures
    llvm-training-verify-opt-diff
    llvm-training-lit)
