# ir/irdl — pure-IR projection of the BCIR dialect (no compilation).
#
# IRDL (MLIR's IR Definition Language) describes the BCIR dialect declaratively
# as IR that `mlir-opt --irdl-file=` loads at runtime. There is no TableGen/ODS,
# no C++ op classes, and no lowering here. The only build artifact is a verified
# round-trip: when `mlir-opt` is available, we parse bcir.irdl.mlir back out.

option(BCIR_ENABLE_MLIR_TOOL_TESTS
  "Register mlir-opt-based IRDL round-trip tests when mlir-opt is found" ON)

if(BCIR_ENABLE_MLIR_TOOL_TESTS)
  find_program(MLIR_OPT_EXECUTABLE NAMES mlir-opt)
  if(MLIR_OPT_EXECUTABLE AND BCIR_BUILD_TESTS)
    add_test(
      NAME bcir-irdl-parse-roundtrip
      COMMAND ${MLIR_OPT_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/bcir.irdl.mlir
              -o ${CMAKE_CURRENT_BINARY_DIR}/bcir.irdl.roundtrip.mlir)
    message(STATUS
      "BCIR: registered IRDL round-trip test (mlir-opt: ${MLIR_OPT_EXECUTABLE})")
  else()
    message(STATUS
      "BCIR: mlir-opt not found; skipping IRDL round-trip test (pure-IR, optional)")
  endif()
endif()
