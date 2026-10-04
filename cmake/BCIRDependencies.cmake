# The dependency registry: every external the rails can use, looked for once, recorded once.
#
# Nothing here is required. The rails are dependency-free C and C++ by law; what is listed is the
# optional outer world -- the MLIR/LLVM package the law rail compiles against, the numeric
# libraries the twin's link-flag rules name (FFTW, LAPACK, GSL, SLEEF, libcerf), threads, and the
# Python the oracle and the gates run under. Each is recorded as BCIR_HAVE_<NAME> plus a row in
# ${CMAKE_BINARY_DIR}/bcir-deps.json, the index a harness can read instead of probing on its own
# (docs/security/laws.md L14: one predicate per repeated question), and summarised at configure.
include_guard(GLOBAL)

set(_bcir_deps_rows "")

function(_bcir_dep_record name found detail)
  set(BCIR_HAVE_${name} ${found} PARENT_SCOPE)
  # JSON spells the verdict true/false; CMake's ON/OFF in its place made the index unreadable
  # (found by the cmake-build job's reader, which is now `manifest.py --deps-index`, a CTest entry).
  if(found)
    set(_state "found")
    set(_json_found "true")
  else()
    set(_state "absent")
    set(_json_found "false")
  endif()
  string(REPLACE "\\" "/" detail "${detail}")
  string(REPLACE "\"" "'" detail "${detail}")
  set(_row "    {\"name\": \"${name}\", \"found\": ${_json_found}, \"detail\": \"${detail}\"}")
  set(_bcir_deps_rows "${_bcir_deps_rows};${_row}" PARENT_SCOPE)
  message(STATUS "BCIR dependency ${name}: ${_state}${detail}")
endfunction()

# Python: the oracle, the parity gate and the shell gates all run under it.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  set(BCIR_PYTHON "${Python3_EXECUTABLE}" CACHE FILEPATH "Python interpreter for the oracle and the gates")
  _bcir_dep_record(PYTHON3 ON " (${Python3_VERSION} at ${Python3_EXECUTABLE})")
else()
  set(BCIR_PYTHON "python3" CACHE FILEPATH "Python interpreter for the oracle and the gates")
  _bcir_dep_record(PYTHON3 OFF "")
endif()

# Threads: the C++ seam's orchestrator and the ring tests.
find_package(Threads QUIET)
if(Threads_FOUND)
  _bcir_dep_record(THREADS ON "")
else()
  _bcir_dep_record(THREADS OFF "")
endif()

# MLIR/LLVM: the law rail (mlir/) compiles against a coherent major; AUTO builds it when found.
find_package(MLIR CONFIG QUIET)
if(MLIR_FOUND)
  find_package(LLVM CONFIG QUIET)
  _bcir_dep_record(MLIR ON " (LLVM ${LLVM_PACKAGE_VERSION} at ${MLIR_DIR})")
else()
  _bcir_dep_record(MLIR OFF " (set MLIR_DIR to an MLIRConfig.cmake to build bcir-opt)")
endif()

# The numeric libraries the twin's link-flag rules name (tools/c/check_runtime.sh #linkflags-*):
# their presence decides which E-series fallbacks a host can judge natively.
find_package(PkgConfig QUIET)
macro(_bcir_optional_lib name lib pkg)
  set(_found OFF)
  set(_detail "")
  if(PKG_CONFIG_FOUND AND NOT "${pkg}" STREQUAL "")
    pkg_check_modules(BCIR_PC_${name} QUIET ${pkg})
    if(BCIR_PC_${name}_FOUND)
      set(_found ON)
      set(_detail " (${pkg} ${BCIR_PC_${name}_VERSION})")
    endif()
  endif()
  if(NOT _found)
    find_library(BCIR_LIB_${name} NAMES ${lib})
    if(BCIR_LIB_${name})
      set(_found ON)
      set(_detail " (${BCIR_LIB_${name}})")
    endif()
  endif()
  _bcir_dep_record(${name} ${_found} "${_detail}")
endmacro()
_bcir_optional_lib(FFTW3F fftw3f fftw3f)
_bcir_optional_lib(LAPACKE lapacke lapacke)
_bcir_optional_lib(GSL gsl gsl)
_bcir_optional_lib(SLEEF sleef sleef)
_bcir_optional_lib(CERF cerf libcerf)

# The index: one file a harness reads instead of re-probing.
list(JOIN _bcir_deps_rows ",\n" _bcir_deps_body)
string(REGEX REPLACE "^,\n" "" _bcir_deps_body "${_bcir_deps_body}")
file(WRITE "${CMAKE_BINARY_DIR}/bcir-deps.json"
  "{\n  \"schema\": \"bcir-deps.v1\",\n  \"c_compiler\": \"${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}\",\n  \"cxx_compiler\": \"${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\",\n  \"system\": \"${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}\",\n  \"dependencies\": [\n${_bcir_deps_body}\n  ]\n}\n")
message(STATUS "BCIR dependency index: ${CMAKE_BINARY_DIR}/bcir-deps.json")
