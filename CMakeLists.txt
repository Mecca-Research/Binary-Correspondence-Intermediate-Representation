cmake_minimum_required(VERSION 3.20)

project(BCIR
  VERSION 0.1.0
  DESCRIPTION "Binary Correspondence Intermediate Representation"
  LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BCIR_BUILD_TESTS "Build BCIR test targets" ON)

add_subdirectory(dialect)
add_subdirectory(runtime)
add_subdirectory(tools)

if(BCIR_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# Export all public BCIR targets.
install(
  EXPORT BCIRTargets
  NAMESPACE BCIR::
  FILE BCIRTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/BCIR)

configure_package_config_file(
  cmake/BCIRConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/BCIRConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/BCIR)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/BCIRConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/BCIRConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/BCIRConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/BCIR)
