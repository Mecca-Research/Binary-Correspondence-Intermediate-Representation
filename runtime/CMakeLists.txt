add_library(bcir-lowering
  src/lowering.cpp)
add_library(BCIR::bcir-lowering ALIAS bcir-lowering)

target_include_directories(bcir-lowering
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(bcir-lowering PUBLIC cxx_std_17)

add_library(gem-runtime
  src/runtime.cpp)
add_library(BCIR::gem-runtime ALIAS gem-runtime)

target_include_directories(gem-runtime
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(gem-runtime PUBLIC cxx_std_17)

# Lowering uses runtime interfaces to schedule executable graphs.
target_link_libraries(bcir-lowering PUBLIC gem-runtime)

install(TARGETS bcir-lowering gem-runtime
  EXPORT BCIRTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(FILES
  ${PROJECT_SOURCE_DIR}/include/bcir/lowering.hpp
  ${PROJECT_SOURCE_DIR}/include/bcir/runtime.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/bcir)
