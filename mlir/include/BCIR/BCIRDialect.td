//===- BCIRDialect.td - BCIR dialect root --------------------*- tablegen -*-===//
//
// Root dialect + common BCIR_Op / BCIR_Type / BCIR_Attr helper classes. Authored
// spec (v0.1); validate with tools/wsl/tblgen_check.sh once an MLIR toolchain is
// available. See ../../docs/BCIR_LANGREF.md.
//
//===----------------------------------------------------------------------===//
#ifndef BCIR_DIALECT
#define BCIR_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def BCIR_Dialect : Dialect {
  let name = "bcir";
  let cppNamespace = "::bcir";
  let summary = "Binary Correspondence Intermediate Representation dialect";
  let description = [{
    BCIR is a registry-first, claim-first, phase-aware, lane-typed, cost-governed
    IR for lowering semantic claims into GEM StreamPacks and backend target IR.
  }];
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

class BCIR_Op<string mnemonic, list<Trait> traits = []>
    : Op<BCIR_Dialect, mnemonic, traits>;

class BCIR_Type<string name, string mnemonic, list<Trait> traits = []>
    : TypeDef<BCIR_Dialect, name, traits> {
  let mnemonic = mnemonic;
}

class BCIR_Attr<string name, string mnemonic, list<Trait> traits = []>
    : AttrDef<BCIR_Dialect, name, traits> {
  let mnemonic = mnemonic;
}

#endif // BCIR_DIALECT
