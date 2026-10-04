# The C rails from runtime/manifest.json: one static library per memory-class-homogeneous unit
# group, the tools, the harnesses, the fuzz targets and the freestanding object checks.
# Nothing is listed here -- a source belongs to a unit in the manifest or it is not built.

# --- libraries ---
foreach(_lib IN LISTS BCIR_MANIFEST_libraries)
  bcir_manifest_paths(_srcs ${BCIR_libraries_${_lib}_SOURCES})
  add_library(${_lib} STATIC ${_srcs})
  bcir_c_target(${_lib})
  target_include_directories(${_lib} PUBLIC "${BCIR_C_DIR}")
  bcir_manifest_link(${_lib} PUBLIC "${BCIR_libraries_${_lib}_LIBRARIES}" "${BCIR_libraries_${_lib}_LINK}")
  bcir_manifest_options(${_lib} "${BCIR_libraries_${_lib}_OPTIONS}")
  set_target_properties(${_lib} PROPERTIES BCIR_MEMORY_CLASS "${BCIR_libraries_${_lib}_CLASS}")
endforeach()

# --- tools: the executables the rails ship (bcir-cc, the benches, the llama CLI) ---
foreach(_tool IN LISTS BCIR_MANIFEST_tools)
  bcir_manifest_paths(_srcs ${BCIR_tools_${_tool}_SOURCES})
  add_executable(${_tool} ${_srcs})
  bcir_c_target(${_tool})
  target_include_directories(${_tool} PRIVATE "${BCIR_C_DIR}")
  bcir_manifest_link(${_tool} PRIVATE "${BCIR_tools_${_tool}_LIBRARIES}" "${BCIR_tools_${_tool}_LINK}")
  bcir_manifest_options(${_tool} "${BCIR_tools_${_tool}_OPTIONS}")
endforeach()

# --- harnesses: the gates' test executables, built (the manifest's link proof); the gates run
# them with their own argv until BUILD-2 registers each as a CTest entry ---
if(BCIR_BUILD_HARNESSES)
  foreach(_h IN LISTS BCIR_MANIFEST_harnesses)
    bcir_manifest_paths(_srcs ${BCIR_harnesses_${_h}_SOURCES})
    add_executable(${_h} ${_srcs})
    bcir_c_target(${_h})
    target_include_directories(${_h} PRIVATE "${BCIR_C_DIR}")
    bcir_manifest_link(${_h} PRIVATE "${BCIR_harnesses_${_h}_LIBRARIES}" "${BCIR_harnesses_${_h}_LINK}")
    bcir_manifest_options(${_h} "${BCIR_harnesses_${_h}_OPTIONS}")
    set_target_properties(${_h} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/harnesses")
  endforeach()
endif()

# --- fuzz targets: Clang's libFuzzer with ASan+UBSan, the library closure compiled into each
# target so the code under test is instrumented (tools/c/fuzz_streampack.sh's one-command build) ---
if(BCIR_BUILD_FUZZERS)
  if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "BCIR_BUILD_FUZZERS needs Clang (libFuzzer); the configured C compiler is ${CMAKE_C_COMPILER_ID}")
  endif()
  set(_fuzz_flags -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all -g)
  foreach(_f IN LISTS BCIR_MANIFEST_fuzzers)
    bcir_manifest_closure(_closure ${BCIR_fuzzers_${_f}_LIBRARIES})
    bcir_manifest_paths(_srcs ${BCIR_fuzzers_${_f}_SOURCES} ${_closure})
    add_executable(${_f} ${_srcs})
    bcir_c_target(${_f})
    target_include_directories(${_f} PRIVATE "${BCIR_C_DIR}")
    target_compile_options(${_f} PRIVATE ${_fuzz_flags})
    target_link_options(${_f} PRIVATE ${_fuzz_flags})
    bcir_manifest_link(${_f} PRIVATE "" "${_closure_LINK}")
    bcir_manifest_options(${_f} "${_closure_OPTIONS};${BCIR_fuzzers_${_f}_OPTIONS}")
    set_target_properties(${_f} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/fuzzers")
    set(_args -runs=${BCIR_FUZZ_RUNS})
    if(BCIR_fuzzers_${_f}_MAX_LEN)
      list(APPEND _args -max_len=${BCIR_fuzzers_${_f}_MAX_LEN})
    endif()
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/fuzzers/work/${_f}")
    add_test(NAME fuzz-${_f} COMMAND ${_f} ${_args} WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/fuzzers/work/${_f}")
    set_tests_properties(fuzz-${_f} PROPERTIES LABELS "fuzz;c" TIMEOUT 900
                         ENVIRONMENT "ASAN_OPTIONS=detect_leaks=1:halt_on_error=1;UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1")
  endforeach()
endif()

# --- freestanding checks: the core compiled -ffreestanding -nostdlib at C11 and at C23 ---
bcir_manifest_paths(_free ${BCIR_MANIFEST_FREESTANDING})
bcir_freestanding_check(bcir_freestanding_c11 11 ${_free})
bcir_freestanding_check(bcir_freestanding_c23 23 ${_free})
