# bcir-surface: BCIR surface text frontend (tokenizer, parser, ROP/MAP verifier).
add_library(bcir-surface
  src/surface.cpp)

add_library(BCIR::bcir-surface ALIAS bcir-surface)

target_include_directories(bcir-surface
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_features(bcir-surface PUBLIC cxx_std_17)

set_target_properties(bcir-surface PROPERTIES
  OUTPUT_NAME bcir-surface)

install(TARGETS bcir-surface
  EXPORT BCIRTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/bcir/surface.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/bcir)
