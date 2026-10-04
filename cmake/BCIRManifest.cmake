# The source manifest reader: runtime/manifest.json is the ONE list of what the C and C++ rails
# are made of -- libraries by memory class, tools, harnesses, fuzz targets, the C++ seam and the
# freestanding checks -- and this reads it into variables the CMakeLists under runtime/ turn into
# targets.
#
# Why a manifest and not CMake lists: three Python harnesses and two shell gates used to carry
# their own copies of these source sets (the #719 trap: a gate and a harness that link different
# sources), and `bcir/tests/test_build_manifest.py` now holds every one of them to this file.
# CMake reads the JSON itself (string(JSON ...), CMake >= 3.19) so a configure needs no Python.
#
# After bcir_manifest_load(path):
#   BCIR_MANIFEST_<kind>            lists of unit names, for each kind in BCIR_MANIFEST_KINDS
#   BCIR_<kind>_<name>_SOURCES      the unit's sources (basenames under runtime/c, or runtime/cpp
#                                   for the seam_* kinds)
#   BCIR_<kind>_<name>_LIBRARIES    the manifest libraries it links
#   BCIR_<kind>_<name>_LINK         system libraries it links (m, pthread)
#   BCIR_<kind>_<name>_OPTIONS      GNU-style compile options the gates pass for that unit
#   BCIR_<kind>_<name>_CLASS        libraries only: freestanding_core / hosted_tool / driver_adapter
#   BCIR_<kind>_<name>_MAX_LEN      fuzzers only: libFuzzer's -max_len, or empty
#   BCIR_MANIFEST_FREESTANDING      the sources compiled -ffreestanding -nostdlib at C11 and C23
include_guard(GLOBAL)

set(BCIR_MANIFEST_KINDS libraries tools harnesses fuzzers seam_libraries seam_tests)

# A JSON array under `key` of `json` as a CMake list in `outvar` (empty when the key is absent).
function(_bcir_json_list json key outvar)
  string(JSON _n ERROR_VARIABLE _err LENGTH "${json}" ${key})
  if(_err OR _n EQUAL 0)
    set(${outvar} "" PARENT_SCOPE)
    return()
  endif()
  set(_items "")
  math(EXPR _last "${_n} - 1")
  foreach(_i RANGE 0 ${_last})
    string(JSON _item GET "${json}" ${key} ${_i})
    list(APPEND _items "${_item}")
  endforeach()
  set(${outvar} "${_items}" PARENT_SCOPE)
endfunction()

# A JSON string under `key` of `json` in `outvar` (empty when absent).
function(_bcir_json_string json key outvar)
  string(JSON _value ERROR_VARIABLE _err GET "${json}" ${key})
  if(_err)
    set(${outvar} "" PARENT_SCOPE)
  else()
    set(${outvar} "${_value}" PARENT_SCOPE)
  endif()
endfunction()

function(bcir_manifest_load path)
  file(READ "${path}" _json)
  string(JSON _schema ERROR_VARIABLE _err GET "${_json}" schema)
  if(_err OR NOT _schema STREQUAL "bcir-build-manifest.v1")
    message(FATAL_ERROR "BCIR: ${path} is not a bcir-build-manifest.v1 (${_err})")
  endif()
  set(_total 0)
  foreach(_kind IN LISTS BCIR_MANIFEST_KINDS)
    string(JSON _n ERROR_VARIABLE _err LENGTH "${_json}" ${_kind})
    if(_err)
      message(FATAL_ERROR "BCIR: ${path} has no '${_kind}' object (${_err})")
    endif()
    set(_names "")
    if(_n GREATER 0)
      math(EXPR _last "${_n} - 1")
      foreach(_i RANGE 0 ${_last})
        string(JSON _name MEMBER "${_json}" ${_kind} ${_i})
        list(APPEND _names "${_name}")
        string(JSON _unit GET "${_json}" ${_kind} ${_name})
        _bcir_json_list("${_unit}" sources _sources)
        _bcir_json_list("${_unit}" libraries _libs)
        _bcir_json_list("${_unit}" link _link)
        _bcir_json_list("${_unit}" options _options)
        _bcir_json_string("${_unit}" class _class)
        _bcir_json_string("${_unit}" max_len _max_len)
        if(NOT _sources)
          message(FATAL_ERROR "BCIR: manifest unit ${_kind}/${_name} lists no sources")
        endif()
        set(BCIR_${_kind}_${_name}_SOURCES "${_sources}" PARENT_SCOPE)
        set(BCIR_${_kind}_${_name}_LIBRARIES "${_libs}" PARENT_SCOPE)
        set(BCIR_${_kind}_${_name}_LINK "${_link}" PARENT_SCOPE)
        set(BCIR_${_kind}_${_name}_OPTIONS "${_options}" PARENT_SCOPE)
        set(BCIR_${_kind}_${_name}_CLASS "${_class}" PARENT_SCOPE)
        set(BCIR_${_kind}_${_name}_MAX_LEN "${_max_len}" PARENT_SCOPE)
      endforeach()
    endif()
    set(BCIR_MANIFEST_${_kind} "${_names}" PARENT_SCOPE)
    math(EXPR _total "${_total} + ${_n}")
  endforeach()
  _bcir_json_list("${_json}" freestanding_checks _free)
  if(NOT _free)
    message(FATAL_ERROR "BCIR: ${path} lists no freestanding_checks (the freestanding core is unproven)")
  endif()
  set(BCIR_MANIFEST_FREESTANDING "${_free}" PARENT_SCOPE)
  list(LENGTH _free _nfree)
  message(STATUS "BCIR manifest: ${_total} units, ${_nfree} freestanding checks (${path})")
endfunction()

# The manifest's basenames as paths under runtime/c (bcir_manifest_paths) or runtime/cpp
# (bcir_manifest_cpp_paths).
function(bcir_manifest_paths outvar)
  set(_paths "")
  foreach(_s IN LISTS ARGN)
    list(APPEND _paths "${BCIR_C_DIR}/${_s}")
  endforeach()
  set(${outvar} "${_paths}" PARENT_SCOPE)
endfunction()

function(bcir_manifest_cpp_paths outvar)
  set(_paths "")
  foreach(_s IN LISTS ARGN)
    list(APPEND _paths "${BCIR_CPP_DIR}/${_s}")
  endforeach()
  set(${outvar} "${_paths}" PARENT_SCOPE)
endfunction()

# The transitive closure of manifest libraries: every source, system library and option the
# named libraries and their dependencies carry, each once, dependents first. A fuzz target
# compiles this closure into itself (the gates' one-command build) so libFuzzer's coverage and
# the sanitizers instrument the code under test, not only the harness.
function(bcir_manifest_closure outvar)
  set(_todo ${ARGN})
  set(_seen "")
  set(_srcs "")
  set(_link "")
  set(_opts "")
  while(_todo)
    list(POP_FRONT _todo _lib)
    if(_lib IN_LIST _seen)
      continue()
    endif()
    if(NOT DEFINED BCIR_libraries_${_lib}_SOURCES)
      message(FATAL_ERROR "BCIR: manifest names an unknown library '${_lib}'")
    endif()
    list(APPEND _seen "${_lib}")
    list(APPEND _srcs ${BCIR_libraries_${_lib}_SOURCES})
    list(APPEND _link ${BCIR_libraries_${_lib}_LINK})
    list(APPEND _opts ${BCIR_libraries_${_lib}_OPTIONS})
    list(APPEND _todo ${BCIR_libraries_${_lib}_LIBRARIES})
  endwhile()
  list(REMOVE_DUPLICATES _link)
  list(REMOVE_DUPLICATES _opts)
  set(${outvar} "${_srcs}" PARENT_SCOPE)
  set(${outvar}_LINK "${_link}" PARENT_SCOPE)
  set(${outvar}_OPTIONS "${_opts}" PARENT_SCOPE)
endfunction()

# Link a target against manifest libraries and the manifest's system-library names.
function(bcir_manifest_link target scope libraries link)
  foreach(_lib IN LISTS libraries)
    target_link_libraries(${target} ${scope} ${_lib})
  endforeach()
  foreach(_sys IN LISTS link)
    if(_sys STREQUAL "pthread")
      if(TARGET Threads::Threads)
        target_link_libraries(${target} ${scope} Threads::Threads)
      elseif(NOT MSVC)
        target_link_libraries(${target} ${scope} pthread)
      endif()
    elseif(_sys STREQUAL "m")
      if(NOT MSVC)
        target_link_libraries(${target} ${scope} m)
      endif()
    else()
      message(FATAL_ERROR "BCIR: manifest link entry '${_sys}' on ${target} is not one of: m, pthread")
    endif()
  endforeach()
endfunction()

# The manifest's per-unit compile options: GNU spellings, applied on GCC and Clang; an entry
# prefixed `gcc:` or `clang:` applies to that compiler only (a warning family one compiler lacks
# is an unknown-option error under -Werror on the other). MSVC takes none of them.
function(bcir_manifest_options target options)
  if(NOT options OR NOT BCIR_GNU_LIKE)
    return()
  endif()
  set(_apply "")
  foreach(_opt IN LISTS options)
    if(_opt MATCHES "^gcc:(.*)$")
      if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        list(APPEND _apply "${CMAKE_MATCH_1}")
      endif()
    elseif(_opt MATCHES "^clang:(.*)$")
      if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        list(APPEND _apply "${CMAKE_MATCH_1}")
      endif()
    else()
      list(APPEND _apply "${_opt}")
    endif()
  endforeach()
  if(_apply)
    target_compile_options(${target} PRIVATE ${_apply})
  endif()
endfunction()
