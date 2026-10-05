# BCIR: the C rails, the C++ seam and (when an MLIR package is found) the MLIR law, as one CMake
# project. docs/BCIR_BUILD_ROADMAP.md is the plan this is slice BUILD-1 of; the source lists live
# in runtime/manifest.json, the policies under cmake/.
#
#   cmake --preset default && cmake --build --preset default && ctest --preset build
#
# BUILD-2 moved the runtime gate's sections into tools/c/sections/, each a CTest entry over the
# binaries built here (`ctest -L section`); the shell gates are CTest entries too (`ctest -L shell`),
# the runtime gate's delegated gates one entry each, all run under the configured compilers, so one
# configure decides which toolchain every rail is judged on.
cmake_minimum_required(VERSION 3.21)

# The repository has one version, pyproject.toml's: the CMake project reads it rather than spell a
# second one (BUILD-3; bcir/tests/test_build_install.py holds bcir.__version__ to it too).
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" _bcir_version_line
     REGEX "^version = \"[0-9]+\\.[0-9]+\\.[0-9]+\"$")
list(LENGTH _bcir_version_line _bcir_version_lines)
if(NOT _bcir_version_lines EQUAL 1)
  message(FATAL_ERROR "BCIR: pyproject.toml must carry exactly one `version = \"X.Y.Z\"` line; found ${_bcir_version_lines}")
endif()
string(REGEX REPLACE "^version = \"([0-9.]+)\"$" "\\1" BCIR_PACKAGE_VERSION "${_bcir_version_line}")

project(BCIR
  VERSION ${BCIR_PACKAGE_VERSION}
  DESCRIPTION "Binary Correspondence Intermediate Representation: C rails, C++ seam, MLIR law"
  LANGUAGES C CXX)
# A version bump in pyproject.toml re-runs the configure, as an edit to a CMakeLists.txt does.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml")

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "BCIR: build out of tree (cmake -S . -B build/cmake, or a preset)")
endif()

option(BCIR_BUILD_HARNESSES "Build the C test harnesses (test_*.c)" ON)
option(BCIR_BUILD_FUZZERS "Build the libFuzzer targets (Clang only)" OFF)
option(BCIR_BUILD_CPP "Build the C++17 orchestration seam (runtime/cpp)" ON)
option(BCIR_WERROR "Warnings are errors, as the gates build" ON)
# A job that installed the ThreadSanitizer runtime owns its absence (docs/security/laws.md L2): ON
# makes a host where the manifest's sanitizer variants cannot be built a configure failure instead
# of a reported skip -- the CMake twin of the runtime gate's BCIR_REQUIRE_TSAN=1.
option(BCIR_REQUIRE_TSAN "Fail the configure when the ThreadSanitizer variants cannot be built here" OFF)
# Likewise for bcir-opt's lit suite: the CI job that installed lit, FileCheck and mlir-opt owns their
# absence, so ON makes a build of bcir-opt without them a configure failure instead of a status line.
option(BCIR_REQUIRE_LIT "Fail the configure when bcir-opt is built here and its lit suite cannot run" OFF)
# OFF by default: the law rail needs a coherent MLIR major built by the compiler that builds it,
# and a package the host happens to carry on PATH must not decide what `cmake --preset default`
# builds. The `mlir` preset turns it ON; AUTO builds it whenever an MLIR package is found.
set(BCIR_BUILD_MLIR "OFF" CACHE STRING "Build the MLIR law (mlir/): ON, OFF, or AUTO when an MLIR package is found")
set_property(CACHE BCIR_BUILD_MLIR PROPERTY STRINGS AUTO ON OFF)
set(BCIR_SANITIZE "" CACHE STRING "Sanitizers for every target (address,undefined / undefined / thread)")
set(BCIR_FUZZ_RUNS 1000 CACHE STRING "libFuzzer iterations per CTest fuzz entry (bounded; > 0)")
if(NOT BCIR_FUZZ_RUNS MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "BCIR_FUZZ_RUNS must be a positive integer, got '${BCIR_FUZZ_RUNS}' (a zero-run fuzz is vacuous)")
endif()

# Release is -O2 with the asserts kept: the gates never define NDEBUG, and a verifier whose
# checks compile out is not the verifier that was tested.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
  set(CMAKE_C_FLAGS_RELEASE "-O2")
  set(CMAKE_CXX_FLAGS_RELEASE "-O2")
endif()

include(GNUInstallDirs)
set(BCIR_C_DIR "${CMAKE_SOURCE_DIR}/runtime/c")
set(BCIR_CPP_DIR "${CMAKE_SOURCE_DIR}/runtime/cpp")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(BCIRCompilerFlags)
include(BCIRDependencies)
include(BCIRManifest)

bcir_manifest_load("${CMAKE_SOURCE_DIR}/runtime/manifest.json")

# Before the subdirectories: a test added below enable_testing()'s directory scope is dropped
# silently, which is how a first configure registered one fuzz entry instead of twenty.
enable_testing()

add_subdirectory(runtime/c)
if(BCIR_BUILD_CPP)
  add_subdirectory(runtime/cpp)
endif()
include(BCIRInstall)

if(BCIR_BUILD_MLIR STREQUAL "ON" OR (BCIR_BUILD_MLIR STREQUAL "AUTO" AND BCIR_HAVE_MLIR))
  if(NOT BCIR_HAVE_MLIR)
    message(FATAL_ERROR "BCIR_BUILD_MLIR=ON but no MLIR package was found; set MLIR_DIR (tools/local/setup_mlir.sh)")
  endif()
  # add_mlir_tool() (LLVM's add_llvm_tool) excludes a tool from ALL unless LLVM_BUILD_TOOLS is on;
  # out of tree nothing defines it, so without this `cmake --build` would silently skip bcir-opt.
  set(LLVM_BUILD_TOOLS ON)
  add_subdirectory(mlir)
  message(STATUS "BCIR: the MLIR law (bcir-opt) is part of this build")
else()
  message(STATUS "BCIR: the MLIR law is not built here (BCIR_BUILD_MLIR=${BCIR_BUILD_MLIR}, MLIR found: ${BCIR_HAVE_MLIR})")
endif()

include(BCIRTests)

message(STATUS "BCIR: ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION} / ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}, ${CMAKE_BUILD_TYPE}, sanitize='${BCIR_SANITIZE}', harnesses=${BCIR_BUILD_HARNESSES}, fuzzers=${BCIR_BUILD_FUZZERS}, cpp=${BCIR_BUILD_CPP}")
