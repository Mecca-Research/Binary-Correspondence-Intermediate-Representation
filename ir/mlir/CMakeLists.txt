# ir/mlir — compiled MLIR BCIR dialect + conversion to the LLVM dialect.
#
# Unlike ir/irdl (pure IR, no compilation), this section IS compilation: ODS/
# TableGen definitions, generated C++ op classes, a registered verifier, and
# dialect-conversion patterns down to the LLVM dialect. It requires MLIR/LLVM
# development libraries, so it is OFF by default and never affects the default
# build or CI.

option(BCIR_ENABLE_MLIR
  "Build the compiled MLIR BCIR dialect and conversion passes (requires MLIR/LLVM dev libraries)"
  OFF)

if(BCIR_ENABLE_MLIR)
  find_package(MLIR REQUIRED CONFIG)
  message(STATUS
    "BCIR: building compiled MLIR dialect against MLIR ${MLIR_PACKAGE_VERSION}")

  # TODO(bcir-mlir): scaffold the dialect + conversion targets.
  #   list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
  #   include(TableGen); include(AddLLVM); include(AddMLIR)
  #   add_subdirectory(lib/Dialect)      # ODS + generated op classes + verifier
  #   add_subdirectory(lib/Conversion)   # BCIR -> LLVM dialect patterns
  #   add_subdirectory(test)             # lit/FileCheck tests
else()
  message(STATUS
    "BCIR: compiled MLIR dialect disabled (configure with -DBCIR_ENABLE_MLIR=ON to build)")
endif()
