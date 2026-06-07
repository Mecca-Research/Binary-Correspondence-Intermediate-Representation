//===- BCIRInterfaces.td - K_BCIR op interface ---------------*- tablegen -*-===//
//
// KBCIRCostOpInterface is the MLIR home of K_BCIR: each cost-bearing op exposes a
// cost vector under Theta and a legality predicate under a target H. The C++
// bodies (cost types + method impls) land with the realization pass; the
// interface is declared here so the dialect compiles standalone.
//
//===----------------------------------------------------------------------===//
#ifndef BCIR_INTERFACES
#define BCIR_INTERFACES

include "mlir/IR/OpBase.td"

def KBCIRCostOpInterface : OpInterface<"KBCIRCostOpInterface"> {
  let description = [{
    Ops that participate in K_BCIR realization expose their cost contribution and
    legality. The oracle in `bcir/` is the executable reference (docs/PARITY.md).
  }];
  let cppNamespace = "::bcir";
  let methods = [
    InterfaceMethod<
      /*desc=*/"Return this op's 12-d K_BCIR cost vector under runtime state Theta.",
      /*retTy=*/"::llvm::SmallVector<int64_t, 12>",
      /*methodName=*/"getCostVector",
      /*args=*/(ins "::mlir::Attribute":$theta)>,
    InterfaceMethod<
      /*desc=*/"Return true if this op is legal on the given target capability H.",
      /*retTy=*/"bool",
      /*methodName=*/"isLegalOn",
      /*args=*/(ins "::mlir::Attribute":$target)>,
  ];
}

#endif // BCIR_INTERFACES
