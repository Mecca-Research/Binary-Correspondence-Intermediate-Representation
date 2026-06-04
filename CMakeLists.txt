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

add_subdirectory(dialect)
add_subdirectory(runtime)
add_subdirectory(tools)

if(BCIR_BUILD_TESTS)
  enable_testing()
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
