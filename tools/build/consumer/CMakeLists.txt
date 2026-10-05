# The out-of-tree consumer of an installed BCIR (BUILD-3, docs/BCIR_BUILD_ROADMAP.md). It knows nothing
# of the source tree: tools/build/install_consumer.py installs a configured build into a scratch
# prefix, copies this file and the harness source beside each other in a scratch directory, and
# configures this project with CMAKE_PREFIX_PATH at the prefix and the build's own compilers.
#
#   -DBCIR_CONSUMER_VERSION=X.Y.Z          the version find_package asks for (the package's own)
#   -DBCIR_CONSUMER_LIBRARIES=a;b;...      every library the manifest lists: each must be imported
#   -DBCIR_CONSUMER_TOOLS=a;b;...          every tool the manifest lists: each must be imported
#   -DBCIR_CONSUMER_HEADERS=N              how many public headers the install must carry
#   -DBCIR_CONSUMER_HARNESS=test_runtime   the harness built here, from <harness>.c in this directory
#   -DBCIR_CONSUMER_HARNESS_LIBRARIES=...  the manifest libraries that harness links
cmake_minimum_required(VERSION 3.21)
project(bcir_consumer LANGUAGES C CXX)

find_package(BCIR ${BCIR_CONSUMER_VERSION} REQUIRED CONFIG)
if(NOT BCIR_VERSION VERSION_EQUAL BCIR_CONSUMER_VERSION)
  message(FATAL_ERROR "consumer: found BCIR ${BCIR_VERSION}, asked for ${BCIR_CONSUMER_VERSION}")
endif()

# Every library and tool the manifest lists is an imported target, and the package says so too.
foreach(_kind LIBRARIES TOOLS)
  foreach(_name IN LISTS BCIR_CONSUMER_${_kind})
    if(NOT TARGET BCIR::${_name})
      message(FATAL_ERROR "consumer: the installed BCIR exports no BCIR::${_name}")
    endif()
    if(NOT _name IN_LIST BCIR_${_kind})
      message(FATAL_ERROR "consumer: BCIR_${_kind} omits ${_name}")
    endif()
  endforeach()
endforeach()

# Every public header compiles on its own, from the installed include directory alone: a C header
# as C23, a C++ header (.hpp) as C++17 -Wpedantic, each in a translation unit of its own.
file(GLOB _headers "${BCIR_INCLUDE_DIRS}/*.h" "${BCIR_INCLUDE_DIRS}/*.hpp")
list(LENGTH _headers _nheaders)
if(NOT _nheaders EQUAL BCIR_CONSUMER_HEADERS)
  message(FATAL_ERROR "consumer: the install carries ${_nheaders} public headers, the tree ${BCIR_CONSUMER_HEADERS}")
endif()
set(_c_checks "")
set(_cxx_checks "")
foreach(_h IN LISTS _headers)
  get_filename_component(_name "${_h}" NAME)
  string(MAKE_C_IDENTIFIER "${_name}" _id)
  if(_name MATCHES "\\.hpp$")
    set(_tu "${CMAKE_BINARY_DIR}/headers/${_id}.cpp")
    list(APPEND _cxx_checks "${_tu}")
  else()
    set(_tu "${CMAKE_BINARY_DIR}/headers/${_id}.c")
    list(APPEND _c_checks "${_tu}")
  endif()
  file(WRITE "${_tu}" "#include \"${_name}\"\nint bcir_consumer_${_id};\n")
endforeach()
add_library(bcir_consumer_c_headers OBJECT ${_c_checks})
set_target_properties(bcir_consumer_c_headers PROPERTIES C_STANDARD 23 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
target_include_directories(bcir_consumer_c_headers PRIVATE "${BCIR_INCLUDE_DIRS}")
target_compile_options(bcir_consumer_c_headers PRIVATE -Wall -Wextra -Werror)
if(_cxx_checks)
  add_library(bcir_consumer_cxx_headers OBJECT ${_cxx_checks})
  set_target_properties(bcir_consumer_cxx_headers PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
  target_include_directories(bcir_consumer_cxx_headers PRIVATE "${BCIR_INCLUDE_DIRS}")
  target_compile_options(bcir_consumer_cxx_headers PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

# The version header agrees with the package: a mismatch fails to compile.
string(REPLACE "." ";" _v "${BCIR_VERSION}")
list(GET _v 0 _major)
list(GET _v 1 _minor)
list(GET _v 2 _patch)
file(WRITE "${CMAKE_BINARY_DIR}/version_check.c"
  "#include \"bcir_version.h\"\n"
  "_Static_assert(BCIR_VERSION_MAJOR == ${_major} && BCIR_VERSION_MINOR == ${_minor} && BCIR_VERSION_PATCH == ${_patch}, \"bcir_version.h is not the package's version\");\n"
  "int bcir_consumer_version_check;\n")
add_library(bcir_consumer_version OBJECT "${CMAKE_BINARY_DIR}/version_check.c")
set_target_properties(bcir_consumer_version PROPERTIES C_STANDARD 23 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
target_include_directories(bcir_consumer_version PRIVATE "${BCIR_INCLUDE_DIRS}")

# The harness, linked against the imported libraries alone.
add_executable(${BCIR_CONSUMER_HARNESS} "${BCIR_CONSUMER_HARNESS}.c")
set_target_properties(${BCIR_CONSUMER_HARNESS} PROPERTIES C_STANDARD 23 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
target_compile_options(${BCIR_CONSUMER_HARNESS} PRIVATE -Wall -Wextra -Werror)
foreach(_lib IN LISTS BCIR_CONSUMER_HARNESS_LIBRARIES)
  target_link_libraries(${BCIR_CONSUMER_HARNESS} PRIVATE BCIR::${_lib})
endforeach()
