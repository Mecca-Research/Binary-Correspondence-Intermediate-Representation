cmake_minimum_required(VERSION 3.20)
project(LLVMTrainingQualityGates NONE)

find_program(BASH_EXECUTABLE bash REQUIRED)
set(TRAINING_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(TOOLS_DIR "${TRAINING_ROOT}/tools")
set(OPTIONAL_RUNNER "${TOOLS_DIR}/run-if-tools.sh")

function(add_training_script_target target script)
  add_custom_target(${target}
    COMMAND "${BASH_EXECUTABLE}" "${TOOLS_DIR}/${script}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
    USES_TERMINAL
    VERBATIM)
endfunction()

function(add_optional_training_target target label script)
  add_custom_target(${target}
    COMMAND "${OPTIONAL_RUNNER}" "${label}" ${ARGN} --
            "${BASH_EXECUTABLE}" "${TOOLS_DIR}/${script}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
    USES_TERMINAL
    VERBATIM)
endfunction()

add_training_script_target(llvm-training-verify-manifest verify-manifest.sh)
add_training_script_target(llvm-training-verify-csv-schema verify-csv-schema.sh)
add_training_script_target(llvm-training-verify-opaque-pointers verify-opaque-pointers.sh)
add_training_script_target(llvm-training-verify-mlir-examples verify-mlir-examples.sh)
add_training_script_target(llvm-training-verify-bcir-mapping verify-bcir-mapping.sh)

add_optional_training_target(llvm-training-verify-examples "LLVM example verifier" verify-examples.sh llvm-as opt)
add_optional_training_target(llvm-training-verify-exercises "LLVM exercise verifier" verify-exercises.sh llvm-as opt)
add_optional_training_target(llvm-training-verify-invalid-fixtures "LLVM invalid-fixture verifier" verify-invalid-fixtures.sh llvm-as opt)
add_optional_training_target(llvm-training-verify-adversarial-fixtures "LLVM adversarial-fixture verifier" verify-adversarial-fixtures.sh llvm-as opt)
add_optional_training_target(llvm-training-smoke-llc "portable llc smoke" smoke-llc.sh llc)
add_optional_training_target(llvm-training-smoke-lli "portable lli smoke" smoke-lli.sh lli)
add_optional_training_target(llvm-training-verify-opt-diff "LLVM opt-diff verifier" verify-opt-diff.sh opt)

add_custom_target(llvm-training-lit
  COMMAND "${BASH_EXECUTABLE}" -c
          "if command -v llvm-lit >/dev/null 2>&1; then exec llvm-lit -sv '${TRAINING_ROOT}/tests'; elif command -v lit >/dev/null 2>&1; then exec lit -sv '${TRAINING_ROOT}/tests'; else echo 'llvm-lit/lit not present; skipping llvm-training lit tests'; fi"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
  USES_TERMINAL
  VERBATIM)

add_custom_target(llvm-training-check
  DEPENDS
    llvm-training-verify-manifest
    llvm-training-verify-csv-schema
    llvm-training-verify-opaque-pointers
    llvm-training-verify-mlir-examples
    llvm-training-verify-bcir-mapping
    llvm-training-verify-examples
    llvm-training-verify-exercises
    llvm-training-verify-invalid-fixtures
    llvm-training-verify-adversarial-fixtures
    llvm-training-verify-opt-diff
    llvm-training-lit)
