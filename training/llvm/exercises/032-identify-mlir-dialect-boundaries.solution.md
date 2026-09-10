# Solution 032: MLIR dialect boundary identification

`bcir.vertex_attr` and `bcir.runtime.call` are domain-specific BCIR operations;
they carry graph and runtime semantics that must be lowered or converted by a
BCIR-aware pass. `func.func` and `return` define the function boundary. The
`arith.constant` and `arith.cmpi` operations are generic arithmetic operations.
`cf.cond_br` is a control-flow dialect operation that can be structurally lowered
to LLVM branches after block arguments and types are converted.

Before lowering to the LLVM dialect, `!bcir.graph` must be converted to an ABI
representation such as a pointer to a runtime graph descriptor, vertex attribute
access must become explicit loads or runtime calls, and the runtime call must
have a stable symbol and converted function type. Generic `arith` and `cf`
operations can then be lowered through standard conversions once their operand
and result types are legal for the LLVM dialect.
