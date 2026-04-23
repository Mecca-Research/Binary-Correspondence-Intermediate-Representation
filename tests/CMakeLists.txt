add_executable(bcir-tests
  src/component_smoke_test.cpp)

target_compile_features(bcir-tests PRIVATE cxx_std_17)
target_link_libraries(bcir-tests
  PRIVATE
    bcir-dialect
    bcir-lowering
    gem-runtime)

add_test(NAME bcir-component-smoke COMMAND bcir-tests)
