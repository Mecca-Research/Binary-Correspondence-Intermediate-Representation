add_library(bcir-dialect
  src/dialect.cpp)

add_library(BCIR::bcir-dialect ALIAS bcir-dialect)

target_include_directories(bcir-dialect
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_features(bcir-dialect PUBLIC cxx_std_17)

set_target_properties(bcir-dialect PROPERTIES
  OUTPUT_NAME bcir-dialect)

install(TARGETS bcir-dialect
  EXPORT BCIRTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(FILES
  ${PROJECT_SOURCE_DIR}/include/bcir/dialect.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/bcir)
