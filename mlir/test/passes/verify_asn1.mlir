// RUN: bcir-opt -bcir-verify -verify-diagnostics -split-input-file %s
//
// R24 (ASN.1 / X.690 encoding-rule legality) over the bcir.asn1.* schema ops. Oracle twins:
// bcir/asn1/schema.py (the type model, OPTIONAL/DEFAULT) and bcir/asn1/der.py (the clause 10 +
// 11 restrictions). LangRef §17, docs/BCIR_ASN1_X690_ABI.md.
//
// What is checked HERE and not on the oracle rail: these are the faults decidable from the TYPE
// alone, before any value exists. A SET whose components share a tag is undecodable for every
// value it could ever hold, and X.680 says so about the type -- so the law belongs where the
// type is written down, not where a value happens to be encoded.
//
// Every op below parses and passes its own ODS verifier; only the LAW is violated.

// (1 -- POSITIVE) the BCIR-StreamPack module: DER, a well-formed private-enterprise OID, a
// primitive, a SEQUENCE OF, and a SEQUENCE whose component tags are distinct.
bcir.asn1.module @BCIR_StreamPack attributes {
  oid = array<i64: 1, 3, 6, 1, 4, 1, 62596, 1>,
  rules = #bcir.asn1_rules<der>,
  default_tagging = #bcir.asn1_tagging<implicit>
} {
  bcir.asn1.type @Utf8 attributes { kind = "primitive", universal = 12 : i64 } { }
  bcir.asn1.type @Int  attributes { kind = "primitive", universal = 2 : i64 } { }
  bcir.asn1.type @Ints attributes { kind = "sequence_of", element = @Int } { }
  bcir.asn1.type @TraceNote attributes { kind = "sequence" } {
    bcir.asn1.component { name = "claimId",   type = @Int, tag = 0 : i64,
                                     tagging = #bcir.asn1_tagging<implicit> }
    bcir.asn1.component { name = "srcHash",   type = @Int, tag = 1 : i64, has_default,
                                     default_value = "0" }
    bcir.asn1.component { name = "traceHash", type = @Int, tag = 2 : i64, has_default,
                                     default_value = "0" }
    bcir.asn1.component { name = "label",     type = @Utf8, tag = 3 : i64, optional }
  }
  bcir.asn1.encode @emit { type = @TraceNote, rules = #bcir.asn1_rules<der> }
  bcir.asn1.decode @accept_peer { type = @TraceNote, rules = #bcir.asn1_rules<ber> }
  bcir.asn1.decode @accept_trusted { type = @TraceNote, rules = #bcir.asn1_rules<der>,
                                     strict_der }
  bcir.asn1.projection @pack { native = "streampack", type = @TraceNote, additive }
}

// -----
// (2 -- NEGATIVE) CER is namable but never emittable: X.690 9.1 makes the indefinite length
// form mandatory for constructed encodings, so a CER artifact cannot be byte-stable.
// expected-error @+1 {{R24: ASN.1 module cer_module declares encoding rules cer; BCIR emits DER only (X.690 clause 10 + 11)}}
bcir.asn1.module @cer_module attributes {
  oid = array<i64: 1, 3, 6, 1, 4, 1, 62596, 2>,
  rules = #bcir.asn1_rules<cer>,
  default_tagging = #bcir.asn1_tagging<implicit>
} {
  bcir.asn1.type @Int attributes { kind = "primitive", universal = 2 : i64 } { }
}

// -----
// (3 -- NEGATIVE) X.690 8.19.4 NOTE: only three values are allocated from the root node.
// expected-error @+1 {{R24: ASN.1 module bad_root object identifier root arc 3 is not 0, 1 or 2 (X.690 8.19.4)}}
bcir.asn1.module @bad_root attributes {
  oid = array<i64: 3, 1>,
  rules = #bcir.asn1_rules<der>,
  default_tagging = #bcir.asn1_tagging<implicit>
} {
  bcir.asn1.type @Int attributes { kind = "primitive", universal = 2 : i64 } { }
}

// -----
// (4 -- NEGATIVE) under root arcs 0 and 1 the second component is 0..39 -- that bound is what
// makes the X*40 + Y packing of 8.19.4 invertible.
// expected-error @+1 {{R24: ASN.1 module bad_second object identifier second arc 40 must be 0..39 under root arc 1 (X.690 8.19.4)}}
bcir.asn1.module @bad_second attributes {
  oid = array<i64: 1, 40, 7>,
  rules = #bcir.asn1_rules<der>,
  default_tagging = #bcir.asn1_tagging<implicit>
} {
  bcir.asn1.type @Int attributes { kind = "primitive", universal = 2 : i64 } { }
}

// -----
// (5 -- NEGATIVE) X.680 Table 1 reserves universal 15 for future editions; a conforming sender
// never emits it, so accepting it would silently admit garbage.
bcir.asn1.module @reserved_tag attributes {
  oid = array<i64: 1, 3, 6, 1, 4, 1, 62596, 3>,
  rules = #bcir.asn1_rules<der>,
  default_tagging = #bcir.asn1_tagging<implicit>
} {
  // expected-error @+1 {{R24: ASN.1 type Reserved names reserved universal tag number 15 (X.680 Table 1: 0, 15 and 37+ are reserved)}}
  bcir.asn1.type @Reserved attributes { kind = "primitive", universal = 15 : i64 } { }
}

// -----
// (6 -- NEGATIVE) THE law: X.680 24.4/25.3/29.3 require component tags to be distinct. This
// type is undecodable as written, for every value it could ever hold.
bcir.asn1.module @dup_tags attributes {
  oid = array<i64: 1, 3, 6, 1, 4, 1, 62596, 4>,
  rules = #bcir.asn1_rules<der>,
  default_tagging = #bcir.asn1_tagging<implicit>
} {
  bcir.asn1.type @Int attributes { kind = "primitive", universal = 2 : i64 } { }
  bcir.asn1.type @Clash attributes { kind = "sequence" } {
    bcir.asn1.component { name = "a", type = @Int, tag = 0 : i64 }
    // expected-error @+1 {{R24: ASN.1 type Clash components a and b share tag [0] (X.680 24.4/25.3/29.3: component tags shall be distinct)}}
    bcir.asn1.component { name = "b", type = @Int, tag = 0 : i64 }
  }
}

// -----
// (7 -- NEGATIVE) X.680 25.5: OPTIONAL and DEFAULT are alternatives. A DEFAULT already makes
// the component omissible, so carrying both is a contradiction.
bcir.asn1.module @both_opt_default attributes {
  oid = array<i64: 1, 3, 6, 1, 4, 1, 62596, 5>,
  rules = #bcir.asn1_rules<der>,
  default_tagging = #bcir.asn1_tagging<implicit>
} {
  bcir.asn1.type @Int attributes { kind = "primitive", universal = 2 : i64 } { }
  bcir.asn1.type @Contradiction attributes { kind = "sequence" } {
    // expected-error @+1 {{R24: ASN.1 component a is both OPTIONAL and DEFAULT (X.680 25.5)}}
    bcir.asn1.component { name = "a", type = @Int, tag = 0 : i64, optional, has_default,
                             default_value = "0" }
  }
}

// -----
// (8 -- NEGATIVE) X.690 11.5 requires a DER encoder to omit a component equal to its DEFAULT;
// it cannot do that without the value, so the type is unencodable under DER.
bcir.asn1.module @default_without_value attributes {
  oid = array<i64: 1, 3, 6, 1, 4, 1, 62596, 6>,
  rules = #bcir.asn1_rules<der>,
  default_tagging = #bcir.asn1_tagging<implicit>
} {
  bcir.asn1.type @Int attributes { kind = "primitive", universal = 2 : i64 } { }
  bcir.asn1.type @Unencodable attributes { kind = "sequence" } {
    // expected-error @+1 {{R24: ASN.1 component a declares DEFAULT but carries no value; X.690 11.5 requires the encoder to compare against it}}
    bcir.asn1.component { name = "a", type = @Int, tag = 0 : i64, has_default }
  }
}

// -----
// (9 -- NEGATIVE) a SET is order-free on the wire (X.690 8.11.2), so its components are told
// apart by tag alone -- an untagged one among tagged siblings is ambiguous in a way the
// SEQUENCE case, which still has position, is not.
bcir.asn1.module @mixed_set attributes {
  oid = array<i64: 1, 3, 6, 1, 4, 1, 62596, 7>,
  rules = #bcir.asn1_rules<der>,
  default_tagging = #bcir.asn1_tagging<implicit>
} {
  bcir.asn1.type @Int attributes { kind = "primitive", universal = 2 : i64 } { }
  // expected-error @+1 {{R24: ASN.1 SET Ambiguous mixes tagged and untagged components; a set is order-free on the wire (X.690 8.11.2) so every component needs a distinct tag}}
  bcir.asn1.type @Ambiguous attributes { kind = "set" } {
    bcir.asn1.component { name = "a", type = @Int, tag = 0 : i64 }
    bcir.asn1.component { name = "b", type = @Int }
  }
}

// -----
// (10 -- NEGATIVE) a primitive without a universal tag number is unencodable.
bcir.asn1.module @primitive_no_tag attributes {
  oid = array<i64: 1, 3, 6, 1, 4, 1, 62596, 8>,
  rules = #bcir.asn1_rules<der>,
  default_tagging = #bcir.asn1_tagging<implicit>
} {
  // expected-error @+1 {{R24: ASN.1 type Naked is primitive but names no universal tag number}}
  bcir.asn1.type @Naked attributes { kind = "primitive" } { }
}

// -----
// (11 -- NEGATIVE) a SEQUENCE OF without an element type names no element at all.
bcir.asn1.module @of_no_element attributes {
  oid = array<i64: 1, 3, 6, 1, 4, 1, 62596, 9>,
  rules = #bcir.asn1_rules<der>,
  default_tagging = #bcir.asn1_tagging<implicit>
} {
  // expected-error @+1 {{R24: ASN.1 type Empty has kind 'sequence_of' but names no element type}}
  bcir.asn1.type @Empty attributes { kind = "sequence_of" } { }
}

// -----
// (12 -- NEGATIVE) emitting BER would let a peer choose the octets, so a digest over them
// would mean nothing.
bcir.asn1.module @encode_ber attributes {
  oid = array<i64: 1, 3, 6, 1, 4, 1, 62596, 10>,
  rules = #bcir.asn1_rules<der>,
  default_tagging = #bcir.asn1_tagging<implicit>
} {
  bcir.asn1.type @Int attributes { kind = "primitive", universal = 2 : i64 } { }
  // expected-error @+1 {{R24: ASN.1 encode loose declares encoding rules ber; BCIR emits DER only (X.690 clause 10 + 11)}}
  bcir.asn1.encode @loose { type = @Int, rules = #bcir.asn1_rules<ber> }
}

// -----
// (13 -- NEGATIVE) strict_der and "accepts BER" are a direct contradiction.
bcir.asn1.module @strict_ber attributes {
  oid = array<i64: 1, 3, 6, 1, 4, 1, 62596, 11>,
  rules = #bcir.asn1_rules<der>,
  default_tagging = #bcir.asn1_tagging<implicit>
} {
  bcir.asn1.type @Int attributes { kind = "primitive", universal = 2 : i64 } { }
  // expected-error @+1 {{R24: ASN.1 decode confused is marked strict_der but declares it accepts BER}}
  bcir.asn1.decode @confused { type = @Int, rules = #bcir.asn1_rules<ber>, strict_der }
}

// -----
// (14 -- NEGATIVE) the additive invariant: a projection that REPLACED a frozen wire format
// would invalidate every digest and provenance manifest taken over the native octets, so the
// IR refuses to express one.
bcir.asn1.module @replacing_projection attributes {
  oid = array<i64: 1, 3, 6, 1, 4, 1, 62596, 12>,
  rules = #bcir.asn1_rules<der>,
  default_tagging = #bcir.asn1_tagging<implicit>
} {
  bcir.asn1.type @Int attributes { kind = "primitive", universal = 2 : i64 } { }
  // expected-error @+1 {{R24: ASN.1 projection replace of streampack is not marked additive; an ASN.1 projection is a SECOND transfer syntax, never a replacement for a frozen wire format}}
  bcir.asn1.projection @replace { native = "streampack", type = @Int }
}
