# CTest registration: the gates as tests, with labels for the tiers and the two-worker law.
#
# `ctest -j 2` is the bounded local run; every heavy gate declares PROCESSORS 2 so two of them
# never run at once, as AGENTS.md requires. Labels:
#   build     the manifest reconciliation, the dependency index, the build-parity gate and the
#             section-parity gate (`ctest --preset build`)
#   section   the gate sections that moved into tools/c/sections/, over the harnesses built here
#   python    the oracle's quick tier
#   shell     the shell gates, wrapped as they are until their sections migrate (BUILD-2)
#   c / cpp / fuzz / mlir / docs   what each gate is about
# CC/CXX are handed to the shell gates from the configured compilers, so `cmake --preset clang`
# followed by `ctest` runs them under the same compiler the targets were built with.
include_guard(GLOBAL)

set(_bcir_gate_env "CC=${CMAKE_C_COMPILER}" "CXX=${CMAKE_CXX_COMPILER}" "CLANG=${CMAKE_C_COMPILER}"
    "BCIR_FUZZ_RUNS=${BCIR_FUZZ_RUNS}")

function(bcir_add_shell_gate name script)
  cmake_parse_arguments(_g "" "TIMEOUT" "LABELS;ENV" ${ARGN})
  add_test(NAME ${name} COMMAND ${CMAKE_COMMAND} -E env ${_bcir_gate_env} ${_g_ENV} bash "${CMAKE_SOURCE_DIR}/${script}"
           WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
  if(NOT _g_TIMEOUT)
    set(_g_TIMEOUT 3600)
  endif()
  set_tests_properties(${name} PROPERTIES LABELS "shell;${_g_LABELS}" PROCESSORS 2 TIMEOUT ${_g_TIMEOUT})
endfunction()

# --- build: the manifest and the parity gate (what BUILD-1 proves) ---
add_test(NAME build-manifest
         COMMAND "${BCIR_PYTHON}" "${CMAKE_SOURCE_DIR}/tools/build/manifest.py" --check
         WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
set_tests_properties(build-manifest PROPERTIES LABELS "build" TIMEOUT 120)
add_test(NAME build-deps-index
         COMMAND "${BCIR_PYTHON}" "${CMAKE_SOURCE_DIR}/tools/build/manifest.py" --deps-index "${CMAKE_BINARY_DIR}/bcir-deps.json"
         WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
set_tests_properties(build-deps-index PROPERTIES LABELS "build" TIMEOUT 60)
add_test(NAME build-parity
         COMMAND "${BCIR_PYTHON}" "${CMAKE_SOURCE_DIR}/tools/build/build_parity.py"
                 --bcir-cc "$<TARGET_FILE:bcir-cc>" --cc "${CMAKE_C_COMPILER}"
         WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
set_tests_properties(build-parity PROPERTIES LABELS "build;c" PROCESSORS 2 TIMEOUT 900)

# --- section: the gate sections that moved into tools/c/sections/, over the harnesses built here ---
# Each script is the section's own text (tools/c/check_runtime.sh calls the same file over the
# binaries it compiles), so the entry and the gate judge one thing; build-section-parity holds the
# two builds' outputs byte-identical.
if(BCIR_BUILD_HARNESSES)
  set(_section_binaries "")
  foreach(_sec IN LISTS BCIR_MANIFEST_sections)
    set(_args "")
    foreach(_h IN LISTS BCIR_sections_${_sec}_HARNESSES)
      if(NOT TARGET ${_h})
        message(FATAL_ERROR "BCIR: section ${_sec} names harness ${_h}, which the manifest does not build")
      endif()
      list(APPEND _args "$<TARGET_FILE:${_h}>")
    endforeach()
    add_test(NAME c-section-${_sec}
             COMMAND ${CMAKE_COMMAND} -E env "PYTHON=${BCIR_PYTHON}" bash "${CMAKE_SOURCE_DIR}/${BCIR_sections_${_sec}_SCRIPT}" ${_args}
             WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    set_tests_properties(c-section-${_sec} PROPERTIES LABELS "c;section" TIMEOUT 900)
  endforeach()
  add_test(NAME build-section-parity
           COMMAND "${BCIR_PYTHON}" "${CMAKE_SOURCE_DIR}/tools/build/section_parity.py"
                   --harness-dir "${CMAKE_BINARY_DIR}/harnesses" --cc "${CMAKE_C_COMPILER}"
           WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
  set_tests_properties(build-section-parity PROPERTIES LABELS "build;c" PROCESSORS 2 TIMEOUT 1800)
endif()

# --- python: the oracle's quick tier (bounded, toolchain-hidden) ---
add_test(NAME python-quick
         COMMAND "${BCIR_PYTHON}" -m bcir.tests.run_all --tier quick -j 2
         WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
set_tests_properties(python-quick PROPERTIES LABELS "python" PROCESSORS 2 TIMEOUT 1800)

# --- shell: the gates as they are (BUILD-2 migrates their sections) ---
bcir_add_shell_gate(c-runtime tools/c/check_runtime.sh LABELS "c" ENV "BCIR_SKIP_CFRONT_SANITIZE=1")
bcir_add_shell_gate(c-memory-discipline tools/c/check_memory_discipline.sh LABELS "c" TIMEOUT 900)
bcir_add_shell_gate(cpp-handoff tools/cpp/check_handoff.sh LABELS "cpp" TIMEOUT 900)
if(CMAKE_C_COMPILER_ID MATCHES "Clang")
  bcir_add_shell_gate(cfront-sanitize tools/c/sanitize_cfront.sh LABELS "c" TIMEOUT 1800
                      ENV "SANITIZE_SKIP_VALGRIND=1" "SANITIZE_ENGINES_PARALLEL=1")
  bcir_add_shell_gate(fuzz-streampack tools/c/fuzz_streampack.sh LABELS "fuzz" TIMEOUT 3600
                      ENV "FUZZ_RUNS=${BCIR_FUZZ_RUNS}")
endif()

# --- mlir: when bcir-opt is built here ---
if(TARGET bcir-opt)
  foreach(_pair "mlir-passes;tools/wsl/check_passes.sh" "mlir-ods-examples;tools/wsl/check_ods_examples.sh"
                "mlir-bytecode;tools/wsl/check_bytecode.sh")
    list(GET _pair 0 _name)
    list(GET _pair 1 _script)
    bcir_add_shell_gate(${_name} ${_script} LABELS "mlir" TIMEOUT 1800 ENV "BCIR_OPT=$<TARGET_FILE:bcir-opt>")
  endforeach()
  bcir_add_shell_gate(mlir-irdl-corpus tools/irdl/check_corpus.sh LABELS "mlir" TIMEOUT 600)
endif()

# --- docs governance ---
add_test(NAME docs-status COMMAND "${BCIR_PYTHON}" "${CMAKE_SOURCE_DIR}/tools/docs/gen_status.py" --check
         WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
add_test(NAME docs-links COMMAND "${BCIR_PYTHON}" "${CMAKE_SOURCE_DIR}/tools/docs/check_links.py"
         WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
set_tests_properties(docs-status docs-links PROPERTIES LABELS "docs" TIMEOUT 300)
