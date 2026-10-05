# The C++17 orchestration seam from runtime/manifest.json (seam_libraries / seam_tests): the C++
# units as C++, linked over the C libraries built as C -- the rule tools/cpp/check_handoff.sh
# states (g++ silently compiles a .c handed to it as C++; clang++ refuses under -Werror).
foreach(_lib IN LISTS BCIR_MANIFEST_seam_libraries)
  bcir_manifest_cpp_paths(_srcs ${BCIR_seam_libraries_${_lib}_SOURCES})
  add_library(${_lib} STATIC ${_srcs})
  bcir_cxx_target(${_lib})
  target_include_directories(${_lib} PUBLIC "$<BUILD_INTERFACE:${BCIR_C_DIR}>" "$<BUILD_INTERFACE:${BCIR_CPP_DIR}>"
                             "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/bcir>")
  bcir_manifest_link(${_lib} PUBLIC "${BCIR_seam_libraries_${_lib}_LIBRARIES}" "${BCIR_seam_libraries_${_lib}_LINK}")
  bcir_manifest_options(${_lib} "${BCIR_seam_libraries_${_lib}_OPTIONS}")
endforeach()

foreach(_t IN LISTS BCIR_MANIFEST_seam_tests)
  bcir_manifest_cpp_paths(_srcs ${BCIR_seam_tests_${_t}_SOURCES})
  add_executable(${_t} ${_srcs})
  bcir_cxx_target(${_t})
  target_include_directories(${_t} PRIVATE "${BCIR_C_DIR}" "${BCIR_CPP_DIR}")
  bcir_manifest_link(${_t} PRIVATE "${BCIR_seam_tests_${_t}_LIBRARIES}" "${BCIR_seam_tests_${_t}_LINK}")
  bcir_manifest_options(${_t} "${BCIR_seam_tests_${_t}_OPTIONS}")
  set_target_properties(${_t} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/harnesses")
endforeach()
