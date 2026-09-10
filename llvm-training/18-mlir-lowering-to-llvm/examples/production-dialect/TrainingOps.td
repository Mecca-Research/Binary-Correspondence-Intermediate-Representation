//===- TrainingOps.td - a minimal complete ODS dialect ------*- tablegen -*-===//
//
// The smallest dialect that still exercises every ODS mechanism a production
// dialect needs: a dialect declaration, a custom type, an enum attribute, a
// shared argument bundle, traits, a hand-written verifier, custom assembly
// format, and a terminator.
//
// The production reference is ../../../../mlir/include/BCIR/*.td.
//
//===----------------------------------------------------------------------===//
#ifndef TRAINING_OPS
#define TRAINING_OPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def Training_Dialect : Dialect {
  let name = "training";
  let cppNamespace = "::mlir::training";
  let summary = "A minimal dialect used by the llvm-training corpus.";
  let description = [{
    Models one idea: a value carried on a lane of a fixed width, scaled by an
    integer factor, and eventually handed to a runtime. Small enough to read in
    one sitting; complete enough to build, register, and test.
  }];

  // Needed whenever the dialect declares its own types or attributes.
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

class Training_Op<string mnemonic, list<Trait> traits = []>
    : Op<Training_Dialect, mnemonic, traits>;

class Training_Type<string name, string typeMnemonic>
    : TypeDef<Training_Dialect, name> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Types and attributes
//===----------------------------------------------------------------------===//

// !training.lane<width = 4>
def Training_LaneType : Training_Type<"Lane", "lane"> {
  let summary = "A lane of a fixed power-of-two width.";
  let parameters = (ins "unsigned":$width);
  let assemblyFormat = "`<` `width` `=` $width `>`";
}

def Training_RoundNearest : I32EnumAttrCase<"Nearest", 0, "nearest">;
def Training_RoundZero    : I32EnumAttrCase<"Zero",    1, "zero">;

def Training_RoundingAttr
    : I32EnumAttr<"Rounding", "rounding mode",
                  [Training_RoundNearest, Training_RoundZero]> {
  let cppNamespace = "::mlir::training";
}

// One bundle, declared once, shared by every op that carries the contract.
// Repeating these across op definitions is how the fifth one ends up missing a
// field -- see chapter 08.
defvar Training_ContractAttrs = (ins
  DefaultValuedAttr<I32Attr, "1">:$factor,
  // New fields are optional or defaulted so that existing IR keeps parsing,
  // verifying, and hashing unchanged (the non-disturbance rule).
  OptionalAttr<Training_RoundingAttr>:$rounding
);

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

def Training_ScaleOp : Training_Op<"scale", [Pure, SameOperandsAndResultType]> {
  let summary = "Scale a value by a compile-time integer factor.";

  let arguments = !con((ins AnyInteger:$input), Training_ContractAttrs);
  let results = (outs AnyInteger:$result);

  // `Pure` and `SameOperandsAndResultType` already cover purity and typing;
  // only the domain rule needs hand-written code.
  let hasVerifier = 1;

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def Training_LaneCastOp : Training_Op<"lane_cast", [Pure]> {
  let summary = "Reinterpret an integer as a value carried on a lane.";

  let arguments = (ins AnyInteger:$input);
  let results = (outs Training_LaneType:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($result)";
}

def Training_EmitOp : Training_Op<"emit"> {
  let summary = "Hand a lane value to the runtime.";
  let description = [{
    Side-effecting and deliberately not `Pure`: the whole point of the op is the
    effect. Marking it pure would let the optimizer delete every call whose
    result is unused, which is all of them.
  }];

  let arguments = (ins Training_LaneType:$value);

  let assemblyFormat = "$value attr-dict `:` type($value)";
}

#endif // TRAINING_OPS
