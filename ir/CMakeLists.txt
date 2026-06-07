# BCIR — Binary Correspondence Intermediate Representation
#
# This directory holds the IR itself, split into sections by pipeline role:
#
#   surface/  text frontend: tokenizer + parser + ROP/MAP verifier (C++)
#   core/     canonical typed graph model + surface->core builder (C++)
#   irdl/     pure-IR declarative dialect projection (no compilation)
#   mlir/     compiled MLIR dialect + conversion patterns (opt-in)
#   llvm/     legal LLVM IR emission + ABI substrate (C++)
#   runtime/  GEM execution engine (C++)
#
# See docs/BCIR_Repo_Structure.md for the ownership map and build matrix.

# Order matters: dependents are listed after their dependencies.
add_subdirectory(surface)
add_subdirectory(core)
add_subdirectory(runtime)
add_subdirectory(llvm)

# Declarative / compiled MLIR sections. These never break the default build:
# irdl tests skip when mlir-opt is absent; the mlir dialect is OFF by default.
add_subdirectory(irdl)
add_subdirectory(mlir)
