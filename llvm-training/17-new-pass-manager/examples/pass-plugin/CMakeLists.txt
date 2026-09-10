# Out-of-tree LLVM pass plugin.
#
# This is a *standalone* CMake project: it is deliberately not added by the
# training corpus's own CMakeLists.txt, because building it requires LLVM
# development headers (the `llvm-dev` package, or a source/installed LLVM tree),
# which the core corpus gates do not assume.
#
#   cmake -S llvm-training/17-new-pass-manager/examples/pass-plugin \
#         -B build/bcir-pass-plugin \
#         -DLLVM_DIR="$(llvm-config --cmakedir)"
#   cmake --build build/bcir-pass-plugin
#
# The driver script llvm-training/tools/build-pass-plugin.sh runs those two
# commands, then exercises the plugin through `opt`, and skips cleanly when the
# headers are absent.

cmake_minimum_required(VERSION 3.20)
project(BCIRPassPlugin LANGUAGES CXX)

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Building BCIR pass plugin against LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "LLVM include dirs: ${LLVM_INCLUDE_DIRS}")

add_library(BCIRRegisterBinding MODULE BCIRRegisterBinding.cpp)

# MODULE libraries get no "lib" prefix on some generators; keep the name the
# driver script and the chapter both expect.
set_target_properties(BCIRRegisterBinding PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)

target_include_directories(BCIRRegisterBinding SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(BCIRRegisterBinding PRIVATE ${LLVM_DEFINITIONS})

# LLVM is usually built without RTTI. A plugin compiled with RTTI against an
# -fno-rtti LLVM links, loads, and then fails in ways that look unrelated, so
# match the host build instead of guessing.
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(BCIRRegisterBinding PRIVATE -fno-rtti)
endif()

# The plugin resolves LLVM symbols from the `opt` process that dlopen()s it, so
# it links against nothing on ELF platforms. macOS needs to be told that
# undefined symbols are expected.
if(APPLE)
  target_link_options(BCIRRegisterBinding PRIVATE -Wl,-undefined,dynamic_lookup)
endif()
