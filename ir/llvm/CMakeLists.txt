# bcir-llvm: legal LLVM IR emission, ROP<->LLVM lowering tables, and ABI substrate.
#
# NOTE: this is the textual/reference LLVM backend. The hand-written *.ll seed
# that previously lived under runtime/llvm/ has been removed; this section is the
# forward path and is expected to grow an LLVM-API emitter. See README.md.
add_library(bcir-llvm
  src/lowering.cpp
  src/bcir_llvm_ir.cpp
  src/llvm_emit.cpp)

add_library(BCIR::bcir-llvm ALIAS bcir-llvm)

target_include_directories(bcir-llvm
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_features(bcir-llvm PUBLIC cxx_std_17)

# Lowering consumes the core graph + surface AST and schedules via the runtime.
target_link_libraries(bcir-llvm PUBLIC bcir-core gem-runtime)

install(TARGETS bcir-llvm
  EXPORT BCIRTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/bcir/lowering.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/bcir/bcir_llvm_ir.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/bcir/llvm_emit.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/bcir)
