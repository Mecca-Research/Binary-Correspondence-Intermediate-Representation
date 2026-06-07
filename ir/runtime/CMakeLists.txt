# gem-runtime: GEM execution engine (threaded phase execution, deterministic order).
add_library(gem-runtime
  src/runtime.cpp)

add_library(BCIR::gem-runtime ALIAS gem-runtime)

find_package(Threads REQUIRED)

target_include_directories(gem-runtime
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_features(gem-runtime PUBLIC cxx_std_17)
target_link_libraries(gem-runtime PUBLIC Threads::Threads)

install(TARGETS gem-runtime
  EXPORT BCIRTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/bcir/runtime.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/bcir)
