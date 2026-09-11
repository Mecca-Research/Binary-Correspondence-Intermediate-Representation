// A rewrite pattern written as IR data rather than as C++.
//
// 03-rewritepattern-and-conversiontarget.md teaches the C++ RewritePattern. This is the
// same idea -- "match arith.addi %x, 0 and replace it with %x" -- expressed in the `pdl`
// dialect, where the pattern is a value the compiler can read rather than code it must
// link. It is the pattern-level counterpart of the IRDL projection in
// 22-bcir-approach/04-dialect-as-data.md, which does the same for the dialect DEFINITION.
//
//   mlir-opt pdl-pattern-as-data.mlir --convert-pdl-to-pdl-interp
//
// compiles it into an explicit matcher: a CFG of checks in the `pdl_interp` dialect.

pdl.pattern @add_zero : benefit(1) {
  %type = pdl.type : i32

  // the constant 0 this pattern is looking for
  %zero_attr = pdl.attribute = 0 : i32
  %zero_op = pdl.operation "arith.constant" {"value" = %zero_attr} -> (%type : !pdl.type)
  %zero = pdl.result 0 of %zero_op

  // ... added to anything
  %lhs = pdl.operand : %type
  %add = pdl.operation "arith.addi"(%lhs, %zero : !pdl.value, !pdl.value) -> (%type : !pdl.type)

  pdl.rewrite %add {
    pdl.replace %add with (%lhs : !pdl.value)
  }
}
