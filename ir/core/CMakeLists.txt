# bcir-core: canonical typed BCIR graph model + surface->core graph builder.
add_library(bcir-core
  src/core_builder.cpp)

add_library(BCIR::bcir-core ALIAS bcir-core)

target_include_directories(bcir-core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_features(bcir-core PUBLIC cxx_std_17)

# The core builder consumes the surface AST (BlockNode/Operation).
target_link_libraries(bcir-core PUBLIC bcir-surface)

install(TARGETS bcir-core
  EXPORT BCIRTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/bcir/bcir_ir.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/bcir/core_builder.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/bcir)
