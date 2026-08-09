//===- BCIREcnOps.td - X.692 ECN encoding definitions ----*- tablegen -*-===//
//
// The law-rail form of an X.692 encoding definition module. Mirrors
// bcir/asn1/ecn_syntax.py's `EcnModule` and the property groups of
// bcir/asn1/ecn_user.py: a module holds encoding classes assigned from built-in ones, one
// concatenation structure, and the encoding objects that realize those classes.
//
// WHAT THIS BUYS OVER THE ORACLE ALONE. Same argument as BCIRAsn1Ops.td, and it lands
// harder here. The Python rail rejects a bad ECN specification when something is encoded
// through it, and an ECN specification's whole point is that it is written once and applied
// to many types -- so a fault that only fires on the right value can sit in a module for a
// long time. Law R25 rejects these at parse time, because every one of them is a statement
// about the SPECIFICATION rather than about any value passing through it:
//
//   9.5.2   at most one encoding object per class in a set
//   21.11.5 the last three RangeConditions take a Comparison and a comparator; the rest
//           take neither
//   22.2.2.2 ALIGNED TO ANY requires a START-POINTER
//   22.8.2.2 USING is present exactly when the determination is not `not-needed`
//   22.8.2.3 ENCODER-TRANSFORMS only with `field-to-be-set`
//   22.8.2.5 DECODER-TRANSFORMS only with `field-to-be-used`
//   22.12.2.3 no bit reversal when the unit is one bit (it is the identity)
//   21.14.5  the half-unit reversals need an even unit
//   23.7.2.4 at most one of IF, IF-ALL and ELSE
//   23.7.2.7 `subtract:lower-bound` only under a condition guaranteeing a lower bound
//   22.1.2.8 REPLACE STRUCTURE forbids INSERT AT HEAD and requires ENCODED BY
//
// WHAT IS DELIBERATELY NOT MODELLED. The bit-level *values* -- patterns, widths, transform
// operands -- are in the oracle and not here, for the same reason the ASN.1 ops carry a
// type's shape and not its values: a law rail states what is checkable about a
// specification, and "this pattern is 0101" is not a proposition that can be false. What IS
// here is every property whose combination with another property can be wrong.
//
// Rec. ITU-T X.692 (02/2021) | ISO/IEC 8825-3:2021.
//
//===----------------------------------------------------------------------===//
#ifndef BCIR_ECN_OPS
#define BCIR_ECN_OPS

// Explicit rather than relying on another ops file having pulled it in first:
// every enum attribute below is defined there, and TableGen include guards make
// this idempotent.
include "BCIR/BCIRAttrs.td"
include "BCIR/BCIRDialect.td"
include "mlir/IR/SymbolInterfaces.td"

def BCIR_EcnModuleOp : BCIR_Op<"ecn.module", [Symbol, SymbolTable, NoTerminator]> {
  let summary = "ECN encoding definition module: classes, one structure, and the objects";
  let description = [{
    The container an X.692 `ENCODING-DEFINITIONS` block maps to (clause 14). Its body holds
    `bcir.ecn.class`, `bcir.ecn.structure` and `bcir.ecn.object` operations.

    `applies_to` names the ASN.1 type this module's object set is applied to, when there is
    one. It is OPTIONAL because clause 14 separates an encoding definition module from the
    encoding link module that applies it: a module is meaningful on its own, and an EDM that
    named its application point would be claiming a binding the notation puts elsewhere.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       OptionalAttr<FlatSymbolRefAttr>:$applies_to);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
}

def BCIR_EcnClassOp : BCIR_Op<"ecn.class", [Symbol]> {
  let summary = "An encoding class assigned from a built-in one (clause 11)";
  let description = [{
    `#Version ::= #INT` is an ordinary thing to write, and it is what makes two integer
    fields of different widths expressible: 9.5.2 permits an encoding object SET at most one
    object per encoding class, so two fields that encode differently need two classes.

    `base` names the built-in class -- `#INT`, `#BOOL`, `#PAD`, `#CONCATENATION`,
    `#CONDITIONAL-INT`, `#TRANSFORM`, `#OUTER` -- or another class this module assigns. R25
    resolves the chain and rejects a cycle, since a class that is its own base names no
    category and no object could realize it.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$base);
  let assemblyFormat = "$sym_name attr-dict-with-keyword";
}

def BCIR_EcnParameterizedOp : BCIR_Op<"ecn.parameterized", [Symbol]> {
  let summary = "An Annex C parameterized assignment and its dummy parameters";
  let description = [{
    Annex C rewrites X.683 for ECN, and C.2 gives three parameterized assignments rather than
    X.683's own set: a class, an object and an object set. ECN assigns those three and leaves
    types to ASN.1, which is what `kind` records.

    The dummies are held as three parallel lists rather than as nested operations. C.1's
    `ParameterList` is positional -- an actual parameter is matched to a dummy by position,
    not by name -- so the position IS the identity, and a nested operation would invite it to
    be reordered by a pass that had no reason to know better.

    `governors` carries C.1's `Governor` for each dummy, or the empty string where the kind
    takes none. That pairing is the rule R25 checks: C.1 admits an encoding-class dummy with
    no governor, requires an `EncodingClassFieldType` for a value, a value set or an ordered
    value list, requires a `REFERENCE` for an identifier, and requires a
    `DefinedOrBuiltinEncodingClass` for an object, an ordered object list or an object set.
    A dummy governed by the wrong thing names a parameter no actual could satisfy, and that is
    decidable from the operation alone.

    The BODY is deliberately absent. X.683 9.7 makes instantiation a substitution, so a body
    read against dummies would have to invent an encoding for each one and any encoding it
    invented would be wrong for some instantiation -- the same reason
    `ecn_syntax.EcnModule.parameterized` holds C.2's assignments uninstantiated.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$kind,
                       StrArrayAttr:$dummies, StrArrayAttr:$parameter_kinds,
                       StrArrayAttr:$governors);
  let assemblyFormat = "$sym_name attr-dict-with-keyword";
}

def BCIR_EcnStructureOp : BCIR_Op<"ecn.structure", [Symbol, NoTerminator]> {
  let summary = "A 16.5 ConcatenationStructure: named fields in textual order";
  let description = [{
    The structure whose TEXTUAL ORDER is the transmission order. 22.10.1.1 gives a
    concatenation object only `{textual, tag, random}` and no property naming a component
    order, and 22.10.3.1 reads `textual` from "the ASN.1 type specification **or the ECN
    structure definition**" -- so this operation is where a wire order that differs from the
    abstract type's actually lives.

    The body holds `bcir.ecn.field` operations. R25 rejects a repeated field name: 16.3.1's
    `NamedField` identifies a field within its structure, and two fields with one name make
    every REFERENCE to that name ambiguous.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$encoding_class);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
}

def BCIR_EcnFieldOp : BCIR_Op<"ecn.field"> {
  let summary = "One field of an encoding structure, with its class";
  let description = [{
    `auxiliary` marks a field that carries no abstract value -- 22.1.2.6's category, which
    21.3.4's `field-to-be-set`, 22.3's start pointer and 22.8's `UNUSED BITS` all write into.
    In X.692 the classification comes from the encoding link module, since a structure field
    with no ASN.1 component behind it is auxiliary; recording it here is the same stated
    deviation the oracle makes, and for the same reason: this rail has no ELM.

    R25 requires every field a REFERENCE names to be `auxiliary` when the determination sets
    it. 21.3.4 sets a field; 21.3.5's `field-to-be-used` reads one "whose value may be set
    from the abstract syntax", which is the opposite -- so the two determinations want
    opposite answers here, and a law that checked only one of them would pass the wrong
    specifications.
  }];
  let arguments = (ins StrAttr:$name,
                       FlatSymbolRefAttr:$encoding_class,
                       UnitAttr:$auxiliary);
  let assemblyFormat = "attr-dict";
}

def BCIR_EcnObjectOp : BCIR_Op<"ecn.object", [Symbol, NoTerminator]> {
  let summary = "An encoding object: the property groups clause 23 gives its class";
  let description = [{
    One encoding object, with the clause 22 property groups whose *combination* a law can
    check. The body holds `bcir.ecn.condition` operations for 23.7's `IF`/`IF-ALL`.

    Pre-alignment (22.2): `align_unit` is the `Unit`, and `align_any` is 22.2.2.1's `ANY`
    against the defaulted `NEXT`. 22.2.2.2 makes `ANY` require a start pointer, which R25
    checks -- nothing else could tell a decoder how many bits the encoder chose.

    Encoding space (21.2, 22.4): `space_size` in `space_unit`s. `space_determination` and
    `space_reference` are 21.3's determinant; R25 requires the reference exactly when a
    determination is stated, since 21.3.4 and 21.3.5 both "require the specification of a
    REFERENCE".

    Start pointer (22.3): `start_pointer` names the field carrying this element's offset.

    Value padding (22.8): `unused_determination` and `unused_reference`, with the two
    transform flags. R25 enforces 22.8.2.2, 22.8.2.3 and 22.8.2.5, which are the three ways
    to write a transform list that would never run.

    Bit reversal (22.12): `reversal` over `space_unit`s. R25 enforces 22.12.2.3 and 21.14.5.

    Replacement (22.1): `replace` with `replacement_structure`, `replacement_object` and
    `head_end`. R25 enforces 22.1.2.8 and 23.3.2.2's "if REPLACE is set, then no other
    encoding property group shall be set", which every category restates.

    Identification handle (22.9): `exhibited_handle` names it, `handle_positions` is 22.9.1.6's
    set of bit positions, and `handle_value_kind` is which 21.16 alternative states the value
    set. The VALUE is not here, for the same reason no pattern is: "this handle admits 4..7"
    is not a proposition a law can falsify. What R25 checks is 22.9.1.6 (the positions are a
    SET), 22.9.2.1 (all handles of one name agree about them) and 22.9.2.3 (their objects
    agree about pre-alignment) -- three statements about a whole module that no single-object
    check could reach, which is exactly the kind of fault this rail exists for.

    Optionality (22.5) and alternative (22.6) determination, each with its reference, its two
    transform flags and its HANDLE marker. R25 enforces 22.5.2.2/.3/.6/.8 and
    22.6.2.2/.3/.5/.7 -- eight rules of one shape, each a way to write a property that would
    never be read.

    Concatenation (22.10): `concatenation_order`. 22.10.2.1 makes `random` require an
    identification handle on every component, which is a statement about the structure rather
    than about this object, so R25 checks the marker's presence and the oracle checks the sets.

    Contained type (22.11): `contents_encoding` marks the group set -- 22.11.1.5 makes that a
    fact about whether the keyword was written, which is why it is a UnitAttr and not the
    emptiness of an object set. `contents_completed_by` and `contents_override` are the two
    optional tails. The object SETS are not here, for the same reason no pattern is: "this set
    contains an object for #Foo" is a fact about another module, and 22.11.2's table is about
    which set applies, not what is in it. What R25 checks is the pair -- 22.11.1.2 brackets
    both tails inside `CONTENTS-ENCODING`, so either one without it is a property that can
    never be read.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       FlatSymbolRefAttr:$encoding_class,
                       // 22.2
                       OptionalAttr<I64Attr>:$align_unit,
                       OptionalAttr<BCIR_EcnPaddingAttr>:$align_padding,
                       UnitAttr:$align_any,
                       // 22.3
                       OptionalAttr<StrAttr>:$start_pointer,
                       // 21.2 / 22.4
                       OptionalAttr<I64Attr>:$space_size,
                       OptionalAttr<I64Attr>:$space_unit,
                       OptionalAttr<BCIR_EcnSpaceDeterminationAttr>:$space_determination,
                       OptionalAttr<StrAttr>:$space_reference,
                       // 22.8
                       OptionalAttr<BCIR_EcnUnusedBitsAttr>:$unused_determination,
                       OptionalAttr<StrAttr>:$unused_reference,
                       UnitAttr:$unused_encoder_transforms,
                       UnitAttr:$unused_decoder_transforms,
                       // 22.12
                       OptionalAttr<BCIR_EcnReversalAttr>:$reversal,
                       // 22.1
                       OptionalAttr<BCIR_EcnReplaceAttr>:$replace,
                       OptionalAttr<FlatSymbolRefAttr>:$replacement_structure,
                       OptionalAttr<FlatSymbolRefAttr>:$replacement_object,
                       OptionalAttr<FlatSymbolRefAttr>:$head_end,
                       // 22.9
                       OptionalAttr<StrAttr>:$exhibited_handle,
                       OptionalAttr<DenseI64ArrayAttr>:$handle_positions,
                       OptionalAttr<BCIR_EcnHandleValueKindAttr>:$handle_value_kind,
                       // 22.5
                       OptionalAttr<BCIR_EcnOptionalityDeterminationAttr>
                           :$optionality_determination,
                       OptionalAttr<StrAttr>:$optionality_reference,
                       UnitAttr:$optionality_encoder_transforms,
                       UnitAttr:$optionality_decoder_transforms,
                       UnitAttr:$optionality_handle_set,
                       // 22.6
                       OptionalAttr<BCIR_EcnAlternativeDeterminationAttr>
                           :$alternative_determination,
                       OptionalAttr<StrAttr>:$alternative_reference,
                       UnitAttr:$alternative_encoder_transforms,
                       UnitAttr:$alternative_decoder_transforms,
                       UnitAttr:$alternative_handle_set,
                       OptionalAttr<BCIR_EcnComponentOrderAttr>:$alternative_ordering,
                       // 22.10
                       OptionalAttr<BCIR_EcnComponentOrderAttr>:$concatenation_order,
                       // 22.11
                       UnitAttr:$contents_encoding,
                       UnitAttr:$contents_completed_by,
                       UnitAttr:$contents_override,
                       // 23.7 / 24.3.9
                       UnitAttr:$subtracts_lower_bound,
                       // 23.7.2.2's ELSE, distinct from "no IF was written" only in intent.
                       UnitAttr:$unconditional);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
}

def BCIR_EcnConditionOp : BCIR_Op<"ecn.condition"> {
  let summary = "One 21.11 RangeCondition guarding a #CONDITIONAL-INT object";
  let description = [{
    21.11.5 splits this type in two, and the split is what R25 checks: the last three values
    "shall be provided" with a `Comparison` and an integer comparator, and "if the other
    enumeration values are used, these shall not be provided". Both directions are faults --
    a comparison on a shape condition tests nothing, and a shape on `test-range` has nothing
    to compare against.

    Several conditions in one object's body are 23.7.1's `IF-ALL`, which 23.7.2.2 reads as a
    conjunction across corresponding positions of its lists.
  }];
  let arguments = (ins BCIR_EcnRangeConditionAttr:$condition,
                       OptionalAttr<BCIR_EcnComparisonAttr>:$comparison,
                       OptionalAttr<I64Attr>:$comparator);
  let assemblyFormat = "attr-dict";
}

#endif // BCIR_ECN_OPS
