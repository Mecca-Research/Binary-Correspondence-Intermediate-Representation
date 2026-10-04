# BCIR compiler-flag policy: one place that spells what the shell gates spelled 260 times.
#
# The C rails are built the way `tools/c/check_runtime.sh` builds them -- ISO C23 (CMake maps it to
# `-std=c2x` on a compiler that predates the final spelling, as the gate's `for std in c23 c2x`
# loop did), `-Wall -Wextra`, `-Werror` unless BCIR_WERROR is off, and `-O2` with the asserts kept
# (the root sets the Release flags; no NDEBUG). The C++ seam is C++17 with `-Wpedantic`, as
# `tools/cpp/check_handoff.sh` builds it. Freestanding sources are additionally compiled as
# objects with `-ffreestanding -nostdlib` at C11 and at C23 -- the gate's "freestanding compile
# (C11 + C23)" sections -- so a hosted-only dependency in a freestanding unit fails the build.
#
# Platform policy: GCC and Clang on Linux (x86-64 and aarch64, CI's matrix) and macOS take the
# flags above; MSVC takes /W4 (/WX) and the latest C standard its CMake knows, and is not
# claimed (docs/BCIR_BUILD_ROADMAP.md names what each platform is held to). No -march is ever
# set: the rails are portable C, and the measured rigs (tools/silicon) choose their own flags.
include_guard(GLOBAL)

set(BCIR_GNU_LIKE OFF)
if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
  set(BCIR_GNU_LIKE ON)
endif()

if(BCIR_GNU_LIKE)
  set(BCIR_C_WARNINGS -Wall -Wextra)
  set(BCIR_CXX_WARNINGS -Wall -Wextra -Wpedantic)
  if(BCIR_WERROR)
    list(APPEND BCIR_C_WARNINGS -Werror)
    list(APPEND BCIR_CXX_WARNINGS -Werror)
  endif()
elseif(MSVC)
  set(BCIR_C_WARNINGS /W4)
  set(BCIR_CXX_WARNINGS /W4 /permissive-)
  if(BCIR_WERROR)
    list(APPEND BCIR_C_WARNINGS /WX)
    list(APPEND BCIR_CXX_WARNINGS /WX)
  endif()
  message(WARNING "BCIR: MSVC is configured on a best-effort basis; the C rails are claimed on GCC and Clang only")
else()
  set(BCIR_C_WARNINGS "")
  set(BCIR_CXX_WARNINGS "")
  message(WARNING "BCIR: unknown C compiler '${CMAKE_C_COMPILER_ID}'; building without the gates' warning policy")
endif()

# Sanitizers for every target (presets asan / ubsan / tsan): the gates' `-fsanitize=... -g` with the
# frame pointer kept so reports name their frames.
if(BCIR_SANITIZE)
  if(BCIR_GNU_LIKE)
    add_compile_options(-fsanitize=${BCIR_SANITIZE} -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=${BCIR_SANITIZE})
  elseif(MSVC AND BCIR_SANITIZE STREQUAL "address")
    add_compile_options(/fsanitize=address)
  else()
    message(FATAL_ERROR "BCIR_SANITIZE=${BCIR_SANITIZE} is not available with ${CMAKE_C_COMPILER_ID}")
  endif()
endif()

# The C standard and warnings every rail target takes (libraries, tools, harnesses, fuzzers).
function(bcir_c_target target)
  set_target_properties(${target} PROPERTIES C_STANDARD 23 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
  target_compile_options(${target} PRIVATE ${BCIR_C_WARNINGS})
endfunction()

# The C++ seam's standard and warnings.
function(bcir_cxx_target target)
  set_target_properties(${target} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
  target_compile_options(${target} PRIVATE ${BCIR_CXX_WARNINGS})
endfunction()

# A freestanding object check: the sources compiled with -ffreestanding -nostdlib at C standard
# `std`, never linked. Part of ALL, so `cmake --build` is the proof. GCC/Clang only. Compiled
# without optimization, as the gates' sections do: the proof is that the unit needs nothing
# hosted, not what the optimizer makes of it (GCC's -O2 flow analysis adds maybe-uninitialized
# verdicts under -ffreestanding that the hosted -O2 build of the same unit does not).
function(bcir_freestanding_check name std)
  if(NOT BCIR_GNU_LIKE)
    message(STATUS "BCIR: freestanding check '${name}' needs GCC or Clang; skipped on ${CMAKE_C_COMPILER_ID}")
    return()
  endif()
  add_library(${name} OBJECT ${ARGN})
  set_target_properties(${name} PROPERTIES C_STANDARD ${std} C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
  target_compile_options(${name} PRIVATE -ffreestanding -nostdlib -O0 ${BCIR_C_WARNINGS})
  target_include_directories(${name} PRIVATE "${BCIR_C_DIR}")
endfunction()
