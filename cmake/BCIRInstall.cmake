# Install and export (BUILD-3, docs/BCIR_BUILD_ROADMAP.md): the manifest's libraries and tools, the
# C++ seam's libraries when they are built, the public headers with a version header, and a CMake
# package an out-of-tree project finds with find_package(BCIR).
#
#   <prefix>/lib/libbcir_*.a              the manifest's libraries (static, as the build makes them)
#   <prefix>/bin/<tool>                   the manifest's tools
#   <prefix>/include/bcir/bcir_*.h(pp)    every public header, and bcir_version.h
#   <prefix>/lib/cmake/BCIR/              BCIRConfig.cmake, BCIRConfigVersion.cmake, BCIRTargets*.cmake
#
# The public headers are every runtime/c/bcir_*.h (each library unit's own, and the header-only
# contracts such as bcir_x86_interrupt.h) and the seam's bcir_*.h/.hpp; the cfront fixtures' headers
# (cfront_*.h, uart_regs.h, ...) are no part of it. Nothing here names a unit: the lists come from
# the manifest. tools/build/install_consumer.py is the gate -- it installs a configured build into a
# scratch prefix and builds tools/build/consumer against it, knowing nothing of the source tree.
include_guard(GLOBAL)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(BCIR_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/BCIR")
set(BCIR_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/bcir")

configure_file("${CMAKE_SOURCE_DIR}/cmake/bcir_version.h.in" "${CMAKE_BINARY_DIR}/include/bcir_version.h" @ONLY)

set(BCIR_EXPORTED_LIBRARIES ${BCIR_MANIFEST_libraries})
if(BCIR_BUILD_CPP)
  list(APPEND BCIR_EXPORTED_LIBRARIES ${BCIR_MANIFEST_seam_libraries})
endif()
set(BCIR_EXPORTED_TOOLS ${BCIR_MANIFEST_tools})

install(TARGETS ${BCIR_EXPORTED_LIBRARIES} ${BCIR_EXPORTED_TOOLS}
  EXPORT BCIRTargets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

file(GLOB _bcir_public_headers CONFIGURE_DEPENDS "${BCIR_C_DIR}/bcir_*.h")
if(BCIR_BUILD_CPP)
  file(GLOB _bcir_seam_headers CONFIGURE_DEPENDS "${BCIR_CPP_DIR}/bcir_*.h" "${BCIR_CPP_DIR}/bcir_*.hpp")
  list(APPEND _bcir_public_headers ${_bcir_seam_headers})
endif()
install(FILES ${_bcir_public_headers} "${CMAKE_BINARY_DIR}/include/bcir_version.h"
  DESTINATION "${BCIR_INSTALL_INCLUDEDIR}")

install(EXPORT BCIRTargets NAMESPACE BCIR:: DESTINATION "${BCIR_INSTALL_CMAKEDIR}")

# A library whose manifest entry links pthread publishes Threads::Threads (bcir_manifest_link), so
# the package must find Threads before its targets are read.
set(BCIR_EXPORTS_THREADS OFF)
if(TARGET Threads::Threads)
  foreach(_lib IN LISTS BCIR_EXPORTED_LIBRARIES)
    get_target_property(_links ${_lib} INTERFACE_LINK_LIBRARIES)
    if(_links MATCHES "Threads::Threads")
      set(BCIR_EXPORTS_THREADS ON)
    endif()
  endforeach()
endif()
configure_package_config_file("${CMAKE_SOURCE_DIR}/cmake/BCIRConfig.cmake.in"
  "${CMAKE_BINARY_DIR}/BCIRConfig.cmake"
  INSTALL_DESTINATION "${BCIR_INSTALL_CMAKEDIR}"
  PATH_VARS BCIR_INSTALL_INCLUDEDIR)
# 0.x: a consumer asking for 0.2 takes any 0.2.z and nothing else.
write_basic_package_version_file("${CMAKE_BINARY_DIR}/BCIRConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMinorVersion)
install(FILES "${CMAKE_BINARY_DIR}/BCIRConfig.cmake" "${CMAKE_BINARY_DIR}/BCIRConfigVersion.cmake"
  DESTINATION "${BCIR_INSTALL_CMAKEDIR}")
message(STATUS "BCIR: installs ${CMAKE_INSTALL_PREFIX}: package BCIR ${PROJECT_VERSION}, "
               "libraries ${BCIR_EXPORTED_LIBRARIES}; tools ${BCIR_EXPORTED_TOOLS}")
