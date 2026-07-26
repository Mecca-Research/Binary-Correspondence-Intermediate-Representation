//===- BCIRAsn1Ops.td - ASN.1 / X.690 type + encoding ops -*- tablegen -*-===//
//
// The law-rail form of the ASN.1 interoperability profile (LangRef §17,
// docs/BCIR_ASN1_X690_ABI.md). Mirrors bcir/asn1/schema.py: a module holds named
// types, a type is either a universal primitive or a constructor, and a SEQUENCE/SET
// is a list of components carrying their X.680 optionality.
//
// WHAT THIS BUYS OVER THE ORACLE ALONE. The Python rail can only reject a bad schema
// when something is encoded through it. Here the schema is IR, so `-bcir-verify`'s
// law R24 rejects it at parse time: a SET whose components share a tag is
// undecodable no matter what values it later carries, and X.680 §24.4/§25.3 say so
// statically. The same applies to reserved universal tags, DEFAULT on an OPTIONAL
// component, and any operation that declares it emits CER.
//
// Rec. ITU-T X.690 (02/2021) | ISO/IEC 8825-1:2021, over the tag assignments of
// Rec. ITU-T X.680 (02/2021) | ISO/IEC 8824-1:2021.
//
//===----------------------------------------------------------------------===//
#ifndef BCIR_ASN1_OPS
#define BCIR_ASN1_OPS

include "BCIR/BCIRDialect.td"
include "mlir/IR/SymbolInterfaces.td"

def BCIR_Asn1ModuleOp : BCIR_Op<"asn1.module", [Symbol, SymbolTable, NoTerminator]> {
  let summary = "ASN.1 module: a named collection of types with its object identifier";
  let description = [{
    The container an X.680 `DEFINITIONS` block maps to. `oid` is the module's object
    identifier as its component arcs (X.690 §8.19.4 packs the first two on the wire;
    the IR keeps them separate because that is the abstract value). `rules` is the
    encoding rules the module's operations emit -- R24 requires `der`.

    `default_tagging` records the module's `IMPLICIT TAGS` / `EXPLICIT TAGS` header, so
    a component that does not state its own tagging mode inherits it rather than
    silently defaulting.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       DenseI64ArrayAttr:$oid,
                       BCIR_Asn1RulesAttr:$rules,
                       BCIR_Asn1TaggingAttr:$default_tagging);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
}

def BCIR_Asn1TypeOp : BCIR_Op<"asn1.type", [Symbol, NoTerminator]> {
  let summary = "A named ASN.1 type: a universal primitive or a constructor";
  let description = [{
    `kind` names the constructor: `primitive`, `sequence`, `sequence_of`, `set`,
    `set_of`, or `choice`. A `primitive` carries the X.680 Table 1 universal tag number
    in `universal`; a `sequence_of`/`set_of` names its element type in `element`; a
    `sequence`/`set`/`choice` owns `bcir.asn1.component` operations in its body.

    `universal` is required for `primitive` and meaningless otherwise -- R24 enforces
    both directions, so a `sequence` cannot smuggle a universal tag past the verifier.

    `constraint_low`/`constraint_high` and `size_low`/`size_high` are the type's
    *effective* X.680 clause 51 constraint bounds (X.696 §8.2.7/§8.2.8) — the ones an
    encoding is chosen from, already reduced from whatever set arithmetic the module
    wrote. They are optional because an unconstrained type has none, and an EXTENSIBLE
    constraint deliberately reports none (X.696 §8.2.2 g): a field sized from today's
    bounds could not carry a later version's values. R24 rejects a pair whose low exceeds
    its high, which is an empty value set — a static fault in the same family as two
    components sharing a tag, since no value of the type could ever be encoded.

    The body is `AnyRegion` because it is EMPTY for the kinds that own no components,
    and an empty body has zero blocks rather than one. The type is a `Symbol` (the
    module refers to it) but deliberately NOT a `SymbolTable`: an ASN.1 component is
    identified by its position and name within its type, never referenced from
    elsewhere, so modelling it as a symbol would claim a property X.680 does not give
    it -- and would force the exactly-one-block rule onto every primitive.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       StrAttr:$kind,
                       OptionalAttr<I64Attr>:$universal,
                       OptionalAttr<FlatSymbolRefAttr>:$element,
                       OptionalAttr<I64Attr>:$constraint_low,
                       OptionalAttr<I64Attr>:$constraint_high,
                       OptionalAttr<I64Attr>:$size_low,
                       OptionalAttr<I64Attr>:$size_high);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
}

def BCIR_Asn1ComponentOp : BCIR_Op<"asn1.component"> {
  let summary = "One SEQUENCE/SET/CHOICE component with its X.680 optionality";
  let description = [{
    `type` names the component's type. An optional `tag` gives it a context-specific
    tag; `tagging` selects X.690 §8.14.3 (explicit) or §8.14.4 (implicit) for it.

    `optional` and `has_default` mirror the X.680 keywords. They are mutually
    exclusive -- X.680 §25.5 gives OPTIONAL and DEFAULT distinct meanings and a
    component cannot have both, since a DEFAULT already makes the component omissible.
    `default_value` is the DEFAULT rendered as text; DER omits any component encoded at
    it (X.690 §11.5), which is why the value must be present in the IR for the encoder
    to compare against.
  }];
  let arguments = (ins StrAttr:$name,
                       FlatSymbolRefAttr:$type,
                       OptionalAttr<I64Attr>:$tag,
                       OptionalAttr<BCIR_Asn1TaggingAttr>:$tagging,
                       UnitAttr:$optional,
                       UnitAttr:$has_default,
                       OptionalAttr<StrAttr>:$default_value);
  let assemblyFormat = "attr-dict";
}

def BCIR_Asn1EncodeOp : BCIR_Op<"asn1.encode", [Symbol]> {
  let summary = "Encode a value of a named ASN.1 type under stated encoding rules";
  let description = [{
    The law-rail counterpart of `bcir.asn1.encode_der`. `rules` must be `der`: BCIR
    digests and replays what it encodes, and BER/CER leave the octets to the sender's
    choice. R24 rejects anything else at verification rather than at emit time.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       FlatSymbolRefAttr:$type,
                       BCIR_Asn1RulesAttr:$rules,
                       OptionalAttr<FlatSymbolRefAttr>:$source);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_Asn1DecodeOp : BCIR_Op<"asn1.decode", [Symbol]> {
  let summary = "Decode octets as a named ASN.1 type under stated encoding rules";
  let description = [{
    `rules` states what the decoder ACCEPTS, and here `ber` is legal and expected: the
    interoperability half of the profile is that a peer built against a BER toolkit can
    still talk to BCIR. `strict_der` marks a decode that additionally refuses every
    BER-only form -- the posture a trust boundary takes when it stores or digests what
    it receives, since otherwise a peer picks the digest by picking a spelling.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       FlatSymbolRefAttr:$type,
                       BCIR_Asn1RulesAttr:$rules,
                       UnitAttr:$strict_der);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_Asn1ProjectionOp : BCIR_Op<"asn1.projection", [Symbol]> {
  let summary = "An ADDITIVE ASN.1 projection of a native BCIR artifact";
  let description = [{
    Records that a native BCIR wire format has a second transfer syntax, and that the
    native octets are unchanged by the round trip. `native` names the artifact
    (`streampack`), `type` the ASN.1 type it projects to.

    `additive` is a unit attribute and R24 requires it: a projection that REPLACED a
    frozen format would invalidate every digest and provenance manifest taken over the
    native bytes, so the IR refuses to express one. The law that makes this safe is A3
    in docs/BCIR_ASN1_X690_ABI.md -- native octets survive the round trip.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       StrAttr:$native,
                       FlatSymbolRefAttr:$type,
                       UnitAttr:$additive);
  let assemblyFormat = "$sym_name attr-dict";
}

#endif // BCIR_ASN1_OPS
