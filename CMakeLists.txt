cmake_minimum_required(VERSION 3.20)

project(BCIR
  VERSION 0.1.0
  DESCRIPTION "Binary Correspondence Intermediate Representation"
  LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BCIR_BUILD_TESTS "Build BCIR test targets" ON)

# Enable testing before subdirectories so sections (e.g. ir/irdl) can register
# their own CTest cases.
if(BCIR_BUILD_TESTS)
  enable_testing()
endif()

# The BCIR IR itself (surface/core/irdl/mlir/llvm/runtime).
add_subdirectory(ir)

# CLIs that consume the IR libraries.
add_subdirectory(tools)

if(BCIR_BUILD_TESTS)
  add_subdirectory(tests)
endif()

find_program(BASH_EXECUTABLE bash)

function(add_llvm_training_tool_target target_name script_path skip_message)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs REQUIRED_TOOLS)
  cmake_parse_arguments(LLVM_TRAINING_TARGET
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN})

  set(script "${CMAKE_CURRENT_SOURCE_DIR}/${script_path}")

  if(BASH_EXECUTABLE)
    set(tool_check "")
    foreach(tool IN LISTS LLVM_TRAINING_TARGET_REQUIRED_TOOLS)
      string(APPEND tool_check "command -v ${tool} >/dev/null 2>&1 && ")
    endforeach()
    string(REGEX REPLACE " && $" "" tool_check "${tool_check}")

    if(tool_check STREQUAL "")
      add_custom_target(${target_name}
        COMMAND "${script}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Running ${target_name}"
        VERBATIM)
    else()
      add_custom_target(${target_name}
        COMMAND
          ${BASH_EXECUTABLE} -c
            "if ${tool_check}; then exec \"$1\"; else printf '%s\\n' \"$2\"; fi"
            --
            "${script}"
            "${skip_message}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Running ${target_name}"
        VERBATIM)
    endif()
  else()
    add_custom_target(${target_name}
      COMMAND ${CMAKE_COMMAND} -E echo "bash not found; skipping ${target_name}"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      VERBATIM)
  endif()
endfunction()

add_llvm_training_tool_target(
  llvm-training-verify-examples
  llvm-training/tools/verify-examples.sh
  "llvm-as and opt not both present; skipping llvm-training standalone example validation"
  REQUIRED_TOOLS llvm-as opt)

add_llvm_training_tool_target(
  llvm-training-smoke-llc
  llvm-training/tools/smoke-llc.sh
  "llc not present; skipping llvm-training llc smoke test"
  REQUIRED_TOOLS llc)

add_llvm_training_tool_target(
  llvm-training-smoke-lli
  llvm-training/tools/smoke-lli.sh
  "lli not present; skipping llvm-training lli smoke test"
  REQUIRED_TOOLS lli)

add_llvm_training_tool_target(
  llvm-training-verify-exercises
  llvm-training/tools/verify-exercises.sh
  "llvm-as and opt not both present; skipping llvm-training exercise solution validation"
  REQUIRED_TOOLS llvm-as opt)

add_llvm_training_tool_target(
  llvm-training-verify-invalid-fixtures
  llvm-training/tools/verify-invalid-fixtures.sh
  "llvm-as and opt not both present; skipping llvm-training invalid fixture validation"
  REQUIRED_TOOLS llvm-as opt)

add_llvm_training_tool_target(
  llvm-training-verify-opt-diff
  llvm-training/tools/verify-opt-diff.sh
  "opt not present; skipping llvm-training opt-diff golden validation"
  REQUIRED_TOOLS opt)

add_llvm_training_tool_target(
  llvm-training-verify-opaque-pointers
  llvm-training/tools/verify-opaque-pointers.sh
  "skipping llvm-training opaque pointer syntax validation")

add_llvm_training_tool_target(
  llvm-training-verify-manifest
  llvm-training/tools/verify-manifest.sh
  "skipping llvm-training example manifest validation")

add_llvm_training_tool_target(
  llvm-training-verify-csv-schema
  llvm-training/tools/verify-csv-schema.sh
  "skipping llvm-training binary-analysis CSV schema validation")

add_llvm_training_tool_target(
  llvm-training-verify-mlir-examples
  llvm-training/tools/verify-mlir-examples.sh
  "skipping llvm-training MLIR example validation")

add_llvm_training_tool_target(
  llvm-training-verify-bcir-mapping
  llvm-training/tools/verify-bcir-mapping.sh
  "skipping llvm-training BCIR mapping validation")

if(BASH_EXECUTABLE)
  find_program(LLVM_LIT_EXECUTABLE NAMES llvm-lit lit)
  if(LLVM_LIT_EXECUTABLE)
    add_custom_target(llvm-training-lit
      COMMAND ${LLVM_LIT_EXECUTABLE} -sv llvm-training/tests
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Running llvm-training lit tests"
      VERBATIM)
  else()
    add_custom_target(llvm-training-lit
      COMMAND ${CMAKE_COMMAND} -E echo "llvm-lit/lit not found; skipping llvm-training lit tests"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      VERBATIM)
  endif()
else()
  add_custom_target(llvm-training-lit
    COMMAND ${CMAKE_COMMAND} -E echo "bash not found; skipping llvm-training lit tests"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    VERBATIM)
endif()

# Export all public BCIR targets.
install(
  EXPORT BCIRTargets
  NAMESPACE BCIR::
  FILE BCIRTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/BCIR)

configure_package_config_file(
  cmake/BCIRConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/BCIRConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/BCIR)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/BCIRConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/BCIRConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/BCIRConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/BCIR)
