add_executable(bcir-tools
  src/bcir_tools_main.cpp)
add_executable(BCIR::bcir-tools ALIAS bcir-tools)

target_compile_features(bcir-tools PRIVATE cxx_std_17)
target_link_libraries(bcir-tools
  PRIVATE
    bcir-dialect
    bcir-lowering
    gem-runtime)

install(TARGETS bcir-tools
  EXPORT BCIRTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
