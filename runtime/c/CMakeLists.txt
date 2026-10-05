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

# --- harnesses: the gates' test executables, built (the manifest's link proof); the migrated
# sections run them as c-section-* entries, as the gate runs its own builds (BUILD-2) ---
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

# --- variants: a harness rebuilt with its whole library closure compiled in, so the change reaches
# the code under test and not only the harness: extra options (the gates' -O0/-O3 sweeps, appended
# after the build's -O2 so the last one wins), one mutated source (the fault-injection witnesses
# a gate section must see fail), another C standard (the two-standard sections) or a sanitizer
# (the ring's ThreadSanitizer leg). A mutant's source is generated at build time by
# tools/build/mutate.py, the applier tools/c/check_runtime.sh uses, from the manifest's one spelling
# of the fault; a mutant is built without the warning policy, as the gate builds it. A variant this
# host cannot build is recorded with its reason (the global property BCIR_UNBUILT_<variant>, and
# BCIR_UNBUILT_VARIANTS for the section-parity gate), so its section is reported as not run here
# rather than failing the configure -- except where BCIR_REQUIRE_TSAN makes that a failure. ---
if(BCIR_BUILD_HARNESSES)
  foreach(_v IN LISTS BCIR_MANIFEST_variants)
    set(_of "${BCIR_variants_${_v}_OF}")
    set(_mfile "${BCIR_variants_${_v}_MUTATION_FILE}")
    set(_san "${BCIR_variants_${_v}_SANITIZER}")
    set(_std 23)
    if(BCIR_variants_${_v}_STANDARD)
      set(_std "${BCIR_variants_${_v}_STANDARD}")
    endif()
    if(NOT DEFINED BCIR_harnesses_${_of}_SOURCES)
      message(FATAL_ERROR "BCIR: variant ${_v} rebuilds '${_of}', which is not a manifest harness")
    endif()
    set(_unbuilt "")
    if(_mfile AND NOT BCIR_HAVE_PYTHON3)
      set(_unbuilt "its fault-injected source needs Python (tools/build/mutate.py)")
    elseif(_san AND NOT BCIR_GNU_LIKE)
      set(_unbuilt "-fsanitize=${_san} needs GCC or Clang")
    elseif(_san STREQUAL "thread" AND NOT BCIR_HAVE_TSAN)
      set(_unbuilt "no working ThreadSanitizer runtime here (the dependency index's TSAN row)")
    elseif(_san STREQUAL "thread" AND BCIR_SANITIZE MATCHES "(^|,)(address|hwaddress|memory|leak)(,|$)")
      # BCIR_SANITIZE applies to every target (the asan/ubsan/tsan presets); ThreadSanitizer joins
      # UndefinedBehaviorSanitizer but no address, memory or leak sanitizer.
      set(_unbuilt "BCIR_SANITIZE=${BCIR_SANITIZE} builds every target with a sanitizer ThreadSanitizer cannot be combined with")
    endif()
    if(_unbuilt)
      if(_san STREQUAL "thread" AND BCIR_REQUIRE_TSAN)
        message(FATAL_ERROR "BCIR: ThreadSanitizer is required here (BCIR_REQUIRE_TSAN=ON), but variant ${_v} cannot be built: ${_unbuilt}")
      endif()
      message(STATUS "BCIR: variant ${_v} is not built: ${_unbuilt}")
      set_property(GLOBAL PROPERTY BCIR_UNBUILT_${_v} "${_unbuilt}")
      set_property(GLOBAL APPEND PROPERTY BCIR_UNBUILT_VARIANTS "${_v}=${_unbuilt}")
      continue()
    endif()
    bcir_manifest_closure(_closure ${BCIR_harnesses_${_of}_LIBRARIES})
    set(_srcs "")
    set(_mutated OFF)
    foreach(_s IN LISTS BCIR_harnesses_${_of}_SOURCES _closure)
      if(_mfile AND _s STREQUAL _mfile)
        set(_mutant "${CMAKE_BINARY_DIR}/mutants/${_v}/${_s}")
        add_custom_command(OUTPUT "${_mutant}"
          COMMAND "${BCIR_PYTHON}" "${CMAKE_SOURCE_DIR}/tools/build/mutate.py" --variant "${_v}" --out "${_mutant}"
          DEPENDS "${BCIR_C_DIR}/${_s}" "${CMAKE_SOURCE_DIR}/tools/build/mutate.py" "${CMAKE_SOURCE_DIR}/runtime/manifest.json"
          COMMENT "BCIR: fault-injecting ${_s} for ${_v}"
          VERBATIM)
        list(APPEND _srcs "${_mutant}")
        set(_mutated ON)
      else()
        list(APPEND _srcs "${BCIR_C_DIR}/${_s}")
      endif()
    endforeach()
    if(_mfile AND NOT _mutated)
      message(FATAL_ERROR "BCIR: variant ${_v} mutates ${_mfile}, which ${_of}'s closure does not compile")
    endif()
    add_executable(${_v} ${_srcs})
    set_target_properties(${_v} PROPERTIES C_STANDARD ${_std} C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
                          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/harnesses")
    if(NOT _mfile)
      target_compile_options(${_v} PRIVATE ${BCIR_C_WARNINGS})
    endif()
    target_include_directories(${_v} PRIVATE "${BCIR_C_DIR}")
    set(_link ${_closure_LINK} ${BCIR_harnesses_${_of}_LINK})
    if(_link)
      list(REMOVE_DUPLICATES _link)
    endif()
    bcir_manifest_link(${_v} PRIVATE "" "${_link}")
    set(_opts ${_closure_OPTIONS} ${BCIR_harnesses_${_of}_OPTIONS})
    bcir_manifest_options(${_v} "${_opts}")
    bcir_manifest_options(${_v} "${BCIR_variants_${_v}_OPTIONS}")
    if(_san)
      target_compile_options(${_v} PRIVATE -fsanitize=${_san})
      target_link_options(${_v} PRIVATE -fsanitize=${_san})
    endif()
  endforeach()
endif()

# --- kernels: C the Python oracle emits (bcir.lower.c_kernel) with a driver appended, the E-series
# programs. tools/build/emit_kernel.py -- the writer tools/c/check_runtime.sh runs -- writes each
# unit at build time with a depfile naming the oracle modules the emitter imported, so an edit to any
# of them re-emits the kernel, and only such an edit does. The unit is built in C11, as the gate
# builds it. A host without Python records each kernel as not built, with the reason, and its section
# as not run here. ---
if(BCIR_BUILD_HARNESSES)
  foreach(_k IN LISTS BCIR_MANIFEST_kernels)
    if(NOT BCIR_HAVE_PYTHON3)
      set(_why "the Python oracle emits its source (tools/build/emit_kernel.py)")
      message(STATUS "BCIR: kernel ${_k} is not built: ${_why}")
      set_property(GLOBAL PROPERTY BCIR_UNBUILT_${_k} "${_why}")
      set_property(GLOBAL APPEND PROPERTY BCIR_UNBUILT_VARIANTS "${_k}=${_why}")
      continue()
    endif()
    set(_unit "${CMAKE_BINARY_DIR}/kernels/${_k}.c")
    add_custom_command(OUTPUT "${_unit}"
      COMMAND "${BCIR_PYTHON}" "${CMAKE_SOURCE_DIR}/tools/build/emit_kernel.py" "${_k}"
              -o "${_unit}" --depfile "${_unit}.d"
      DEPENDS "${CMAKE_SOURCE_DIR}/tools/build/emit_kernel.py" "${CMAKE_SOURCE_DIR}/runtime/manifest.json"
              "${BCIR_C_DIR}/kernels/${BCIR_kernels_${_k}_MAIN}"
      DEPFILE "${_unit}.d"
      COMMENT "BCIR: emitting kernel ${_k}"
      VERBATIM)
    add_executable(${_k} "${_unit}")
    set_target_properties(${_k} PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
                          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/harnesses")
    target_compile_options(${_k} PRIVATE ${BCIR_C_WARNINGS})
    bcir_manifest_link(${_k} PRIVATE "" "${BCIR_kernels_${_k}_LINK}")
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
