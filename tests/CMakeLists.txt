add_executable(bcir-tests
  src/component_smoke_test.cpp)

add_executable(bcir-extended-tests
  src/extended_component_test.cpp)

target_compile_features(bcir-tests PRIVATE cxx_std_17)
target_compile_features(bcir-extended-tests PRIVATE cxx_std_17)

target_compile_definitions(bcir-extended-tests PRIVATE BCIR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

target_link_libraries(bcir-tests
  PRIVATE
    bcir-dialect
    bcir-lowering
    gem-runtime)

target_link_libraries(bcir-extended-tests
  PRIVATE
    bcir-dialect
    bcir-lowering
    gem-runtime)

add_test(NAME bcir-component-smoke COMMAND bcir-tests)
add_test(NAME bcir-parser-verifier-unit COMMAND bcir-extended-tests parser_verifier)
add_test(NAME bcir-macro-golden COMMAND bcir-extended-tests macro_golden)
add_test(NAME bcir-rop-roundtrip COMMAND bcir-extended-tests roundtrip)
add_test(NAME bcir-gem-exec COMMAND bcir-extended-tests gem_exec)
add_test(NAME bcir-reference-examples COMMAND bcir-extended-tests reference_examples)
add_test(NAME bcir-runtime-regression COMMAND bcir-extended-tests runtime_regression)
