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

# An optional library's row also carries its link flags (ARGN): the flag set the probe linked
# with, empty when it is absent (BUILD-4).
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
  # A detail is one JSON string: a probe's diagnostic can span lines, and a raw control character
  # would make the whole index unreadable.
  string(REGEX REPLACE "[\r\n\t]+" " " detail "${detail}")
  string(REPLACE "\\" "/" detail "${detail}")
  string(REPLACE "\"" "'" detail "${detail}")
  # A semicolon would split the row in the list the rows are kept in, and the JSON join would put a
  # raw line feed inside the string (found by D1 on a probe's "tried -lfftw3f; -lfftw3"); any byte
  # outside printable ASCII is no JSON string either. One rule for every caller (L14).
  string(REPLACE ";" "," detail "${detail}")
  string(REGEX REPLACE "[^ -~]" "?" detail "${detail}")
  set(_link_json "")
  if(_bcir_dep_link_row)
    set(_flags "")
    foreach(_flag IN LISTS ARGN)
      list(APPEND _flags "\"${_flag}\"")
    endforeach()
    list(JOIN _flags ", " _flags)
    set(_link_json ", \"link\": [${_flags}]")
  endif()
  set(_row "    {\"name\": \"${name}\", \"found\": ${_json_found}, \"detail\": \"${detail}\"${_link_json}}")
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

# lit, FileCheck and mlir-opt: bcir-opt's lit suite (mlir/lit.cfg.py, BUILD-5) runs each fixture's
# RUN lines with the FileCheck and mlir-opt of the LLVM the MLIR package belongs to. They are looked
# for beside it (apt's llvm-N-tools and mlir-N-tools, a conda prefix's libexec/llvm), never on PATH,
# where another major can sit. lit is version-free Python: beside that LLVM (apt's
# build/utils/lit/lit.py), else an LLVM lit on PATH (pip's), else the newest apt lit.py. LIT is
# found when all three are; the row says which is missing, and BCIR_REQUIRE_LIT makes that fatal.
if(MLIR_FOUND)
  set(_bcir_llvm_bins "${LLVM_TOOLS_BINARY_DIR}" "${LLVM_TOOLS_BINARY_DIR}/../libexec/llvm"
                      "/usr/lib/llvm-${LLVM_VERSION_MAJOR}/bin")
  find_program(BCIR_FILECHECK NAMES FileCheck PATHS ${_bcir_llvm_bins} NO_DEFAULT_PATH)
  find_program(BCIR_MLIR_OPT NAMES mlir-opt PATHS ${_bcir_llvm_bins} NO_DEFAULT_PATH)
  find_program(BCIR_LIT NAMES lit.py PATHS "${LLVM_TOOLS_BINARY_DIR}/../build/utils/lit"
               "/usr/lib/llvm-${LLVM_VERSION_MAJOR}/build/utils/lit" NO_DEFAULT_PATH)
  if(NOT BCIR_LIT)
    # A Windows runner's PATH can carry Microsoft's unrelated lit.exe: take a PATH lit only when
    # it runs under this Python and its banner is LLVM lit's.
    find_program(_bcir_path_lit NAMES lit llvm-lit)
    if(_bcir_path_lit)
      execute_process(COMMAND "${BCIR_PYTHON}" "${_bcir_path_lit}" --version
                      OUTPUT_VARIABLE _bcir_lit_banner ERROR_VARIABLE _bcir_lit_banner
                      RESULT_VARIABLE _bcir_lit_rc TIMEOUT 60)
      if(_bcir_lit_rc EQUAL 0 AND _bcir_lit_banner MATCHES "^lit( version)? [0-9]")
        set(BCIR_LIT "${_bcir_path_lit}" CACHE FILEPATH "lit, for bcir-opt's lit suite" FORCE)
      endif()
    endif()
  endif()
  if(NOT BCIR_LIT)
    file(GLOB _bcir_lits "/usr/lib/llvm-*/build/utils/lit/lit.py")
    if(_bcir_lits)
      list(SORT _bcir_lits COMPARE NATURAL ORDER DESCENDING)
      list(GET _bcir_lits 0 _bcir_lit)
      set(BCIR_LIT "${_bcir_lit}" CACHE FILEPATH "lit, for bcir-opt's lit suite" FORCE)
    endif()
  endif()
  set(BCIR_LIT_WHY "")
  foreach(_bcir_tool BCIR_LIT BCIR_FILECHECK BCIR_MLIR_OPT)
    if(NOT ${_bcir_tool})
      string(APPEND BCIR_LIT_WHY " no ${_bcir_tool}")
    endif()
  endforeach()
  if(BCIR_LIT_WHY)
    set(BCIR_LIT_WHY "${BCIR_LIT_WHY} for LLVM ${LLVM_VERSION_MAJOR} (apt's llvm-${LLVM_VERSION_MAJOR}-tools and mlir-${LLVM_VERSION_MAJOR}-tools carry all three)")
    _bcir_dep_record(LIT OFF "${BCIR_LIT_WHY}")
  else()
    _bcir_dep_record(LIT ON " (lit ${BCIR_LIT}, FileCheck ${BCIR_FILECHECK}, mlir-opt ${BCIR_MLIR_OPT})")
  endif()
endif()

# ThreadSanitizer: the runtime the ring's sanitizer variants need (runtime/manifest.json). Present
# means a trivial -fsanitize=thread program builds AND runs here, the predicate of
# tools/build/sanitizer.py -- the one tools/c/check_runtime.sh asks before it builds the same
# binaries -- so the gate and this build cannot disagree about the host (laws.md L12, L14). Where
# a job installed the runtime, BCIR_REQUIRE_TSAN turns its absence into a configure failure (L2).
if(BCIR_HAVE_PYTHON3)
  execute_process(
    COMMAND "${BCIR_PYTHON}" "${CMAKE_SOURCE_DIR}/tools/build/sanitizer.py" --cc "${CMAKE_C_COMPILER}" thread
    RESULT_VARIABLE _bcir_tsan_rc OUTPUT_VARIABLE _bcir_tsan_out ERROR_VARIABLE _bcir_tsan_err
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE TIMEOUT 300)
  string(REGEX REPLACE "^sanitizer thread: [A-Z]+: " "" _bcir_tsan_why "${_bcir_tsan_out}")
  if(_bcir_tsan_rc EQUAL 0)
    _bcir_dep_record(TSAN ON " (${_bcir_tsan_why})")
  else()
    _bcir_dep_record(TSAN OFF " (${_bcir_tsan_why}${_bcir_tsan_err})")
  endif()
else()
  _bcir_dep_record(TSAN OFF " (the probe, tools/build/sanitizer.py, needs Python)")
endif()
if(BCIR_REQUIRE_TSAN AND NOT BCIR_HAVE_TSAN)
  message(FATAL_ERROR "BCIR: ThreadSanitizer is required here (BCIR_REQUIRE_TSAN=ON), but a trivial TSan program does not build and run with ${CMAKE_C_COMPILER}")
endif()

# The numeric libraries the twin's link-flag rules name (tools/c/check_runtime.sh #linkflags-*):
# their presence decides which E-series fallbacks a host can judge natively. Asked of the one
# predicate the Python harnesses use, bcir.toolchain's (BUILD-4): a probe that includes the header
# and calls a function, so the link must resolve the symbol, under the configured compiler. Each row
# records the flag set that linked; a harness run with BCIR_DEPS_INDEX at this index reads it
# instead of probing, so the two cannot disagree (docs/security/laws.md L12, L14).
set(_bcir_optional_libraries FFTW3F LAPACKE GSL SLEEF CERF)
set(_bcir_dep_link_row ON)
set(_bcir_probe_json "")
if(BCIR_HAVE_PYTHON3)
  execute_process(
    COMMAND "${BCIR_PYTHON}" -m bcir.toolchain probe-libraries --cc "${CMAKE_C_COMPILER}"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RESULT_VARIABLE _bcir_probe_rc OUTPUT_VARIABLE _bcir_probe_json ERROR_VARIABLE _bcir_probe_err
    OUTPUT_STRIP_TRAILING_WHITESPACE TIMEOUT 600)
  if(NOT _bcir_probe_rc EQUAL 0)
    message(FATAL_ERROR "BCIR: the optional-library probe (python -m bcir.toolchain probe-libraries) failed: ${_bcir_probe_err}")
  endif()
endif()
foreach(_name IN LISTS _bcir_optional_libraries)
  if(NOT _bcir_probe_json)
    _bcir_dep_record(${_name} OFF " (the probe, bcir.toolchain, needs Python)")
    continue()
  endif()
  string(JSON _found ERROR_VARIABLE _err GET "${_bcir_probe_json}" ${_name} found)
  if(_err)
    message(FATAL_ERROR "BCIR: the optional-library probe gave no answer for ${_name}: ${_err}")
  endif()
  string(JSON _detail GET "${_bcir_probe_json}" ${_name} detail)
  string(JSON _nflags LENGTH "${_bcir_probe_json}" ${_name} link)
  set(_flags "")
  if(_nflags GREATER 0)
    math(EXPR _last "${_nflags} - 1")
    foreach(_i RANGE 0 ${_last})
      string(JSON _flag GET "${_bcir_probe_json}" ${_name} link ${_i})
      list(APPEND _flags "${_flag}")
    endforeach()
  endif()
  _bcir_dep_record(${_name} ${_found} " (${_detail})" ${_flags})
endforeach()
set(_bcir_dep_link_row OFF)

# The index: one file a harness reads instead of re-probing.
list(JOIN _bcir_deps_rows ",\n" _bcir_deps_body)
string(REGEX REPLACE "^,\n" "" _bcir_deps_body "${_bcir_deps_body}")
file(WRITE "${CMAKE_BINARY_DIR}/bcir-deps.json"
  "{\n  \"schema\": \"bcir-deps.v1\",\n  \"c_compiler\": \"${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}\",\n  \"cxx_compiler\": \"${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\",\n  \"system\": \"${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}\",\n  \"dependencies\": [\n${_bcir_deps_body}\n  ]\n}\n")
message(STATUS "BCIR dependency index: ${CMAKE_BINARY_DIR}/bcir-deps.json")
