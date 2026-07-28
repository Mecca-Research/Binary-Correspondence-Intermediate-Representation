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
// expected-error @+1 {{R24: ASN.1 module cer_module declares encoding rules cer (X.690), which is not canonical; BCIR emits only a transfer syntax whose octets are a function of the abstract value, because it digests what it emits}}
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
  // expected-error @+1 {{R24: ASN.1 encode loose declares encoding rules ber (X.690), which is not canonical; BCIR emits only a transfer syntax whose octets are a function of the abstract value}}
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
  // expected-error @+1 {{R24: ASN.1 decode confused is marked strict_der but declares it accepts ber, which is not a canonical transfer syntax}}
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

// -----
// (15 -- NEGATIVE) an X.680 clause 51 constraint whose bounds cross permits NO value, so
// every use of the type is dead. The bounds carried in the IR are already EFFECTIVE
// (X.696 8.2.7), which is why an extensible constraint -- reporting no bounds at all --
// can never trip this.
bcir.asn1.module @empty_constraint attributes {
  oid = array<i64: 1, 3, 6, 1, 4, 1, 62596, 13>,
  rules = #bcir.asn1_rules<der>,
  default_tagging = #bcir.asn1_tagging<implicit>
} {
  // expected-error @+1 {{R24: ASN.1 type Impossible has an empty value constraint (10..1); no value of the type can be encoded (X.680 49)}}
  bcir.asn1.type @Impossible attributes { kind = "primitive", universal = 2 : i64,
                                          constraint_low = 10 : i64,
                                          constraint_high = 1 : i64 } { }
}

// -----
// (16 -- NEGATIVE) the same fault on a SIZE, which bounds a LENGTH rather than a value.
bcir.asn1.module @empty_size attributes {
  oid = array<i64: 1, 3, 6, 1, 4, 1, 62596, 14>,
  rules = #bcir.asn1_rules<der>,
  default_tagging = #bcir.asn1_tagging<implicit>
} {
  bcir.asn1.type @Int attributes { kind = "primitive", universal = 2 : i64 } { }
  // expected-error @+1 {{R24: ASN.1 type NoLength has an empty SIZE constraint (5..2); no value of the type can be encoded (X.680 51.5)}}
  bcir.asn1.type @NoLength attributes { kind = "sequence_of", element = @Int,
                                        size_low = 5 : i64, size_high = 2 : i64 } { }
}

// -----
// (17 -- NEGATIVE) a negative SIZE lower bound. A length is a count of octets or
// occurrences and cannot be below zero, so this is unsatisfiable however the upper
// bound is written.
bcir.asn1.module @negative_size attributes {
  oid = array<i64: 1, 3, 6, 1, 4, 1, 62596, 15>,
  rules = #bcir.asn1_rules<der>,
  default_tagging = #bcir.asn1_tagging<implicit>
} {
  bcir.asn1.type @Int attributes { kind = "primitive", universal = 2 : i64 } { }
  // expected-error @+1 {{R24: ASN.1 type Negative has a negative SIZE lower bound (-1); a length cannot be negative (X.680 51.5)}}
  bcir.asn1.type @Negative attributes { kind = "sequence_of", element = @Int,
                                        size_low = -1 : i64, size_high = 4 : i64 } { }
}

// -----
// (18 -- POSITIVE) an EXTENSIBLE constraint reports NO effective bounds (X.696 8.2.2 g),
// so a type whose root bounds would look odd is still legal -- and, crucially, is encoded
// as though unbounded, because a field sized from today's root could not carry a later
// version's values.
bcir.asn1.module @extensible_ok attributes {
  oid = array<i64: 1, 3, 6, 1, 4, 1, 62596, 16>,
  rules = #bcir.asn1_rules<der>,
  default_tagging = #bcir.asn1_tagging<implicit>
} {
  bcir.asn1.type @Bounded attributes { kind = "primitive", universal = 2 : i64,
                                       constraint_low = 0 : i64,
                                       constraint_high = 255 : i64 } { }
}

// -----
// (19 -- POSITIVE) Every canonical transfer syntax the repository speaks is emittable.
// The generalized R24 law is CANONICALITY, not "der": DER is the X.690 member of the set,
// and CANONICAL-PER (both alignments), COER, CXER and BCIR's canonical JER profile are the
// others. `bcir/asn1/selection.py` measures exactly these five, and
// test_asn1_law_parity.py pins the two lists against each other so neither can grow a
// member the other does not know about.
bcir.asn1.module @canonical_per attributes {
  oid = array<i64: 1, 3, 6, 1, 4, 1, 62596, 17>,
  rules = #bcir.asn1_rules<canonical_per_unaligned>,
  default_tagging = #bcir.asn1_tagging<implicit>
} {
  bcir.asn1.type @Int attributes { kind = "primitive", universal = 2 : i64 } { }
  bcir.asn1.encode @emit_unaligned { type = @Int,
                                     rules = #bcir.asn1_rules<canonical_per_unaligned> }
  bcir.asn1.encode @emit_aligned { type = @Int,
                                   rules = #bcir.asn1_rules<canonical_per_aligned> }
  bcir.asn1.encode @emit_coer { type = @Int, rules = #bcir.asn1_rules<coer> }
  bcir.asn1.encode @emit_cxer { type = @Int, rules = #bcir.asn1_rules<cxer> }
  bcir.asn1.encode @emit_jer { type = @Int, rules = #bcir.asn1_rules<bcir_canonical_jer> }
  // The permissive half: a decoder may accept anything a peer can write, in any family.
  bcir.asn1.decode @take_basic_per { type = @Int,
                                     rules = #bcir.asn1_rules<basic_per_unaligned> }
  bcir.asn1.decode @take_jer { type = @Int, rules = #bcir.asn1_rules<jer> }
  bcir.asn1.decode @take_oer { type = @Int, rules = #bcir.asn1_rules<oer> }
  bcir.asn1.decode @take_xer { type = @Int, rules = #bcir.asn1_rules<xer> }
  // ...and may declare it refuses non-canonical octets, in the family-neutral spelling.
  bcir.asn1.decode @take_strict_jer { type = @Int,
                                      rules = #bcir.asn1_rules<bcir_canonical_jer>,
                                      strict_canonical }
}

// -----
// (20) A non-canonical PER encode fails for the SAME reason `ber` does. The old law named
// DER, so this document would have been rejected with a message about X.690 that has
// nothing to do with X.691; the generalized law names the property and the family.
bcir.asn1.module @basic_per_emit attributes {
  oid = array<i64: 1, 3, 6, 1, 4, 1, 62596, 18>,
  rules = #bcir.asn1_rules<der>,
  default_tagging = #bcir.asn1_tagging<implicit>
} {
  bcir.asn1.type @Int attributes { kind = "primitive", universal = 2 : i64 } { }
  // expected-error @+1 {{R24: ASN.1 encode loose_per declares encoding rules basic_per_aligned (X.691), which is not canonical; BCIR emits only a transfer syntax whose octets are a function of the abstract value}}
  bcir.asn1.encode @loose_per { type = @Int, rules = #bcir.asn1_rules<basic_per_aligned> }
}

// -----
// (21) The hole the old law left open. `strict_der` with `cer` passed verification,
// because the test was `rules == ber` rather than "is the declared syntax canonical".
// CER is exactly as un-byte-stable as BER -- X.690 9.1 makes the indefinite length form
// mandatory for constructed encodings -- so a decoder claiming to refuse non-canonical
// octets while declaring it accepts CER is contradicting itself just as plainly.
bcir.asn1.module @strict_cer attributes {
  oid = array<i64: 1, 3, 6, 1, 4, 1, 62596, 19>,
  rules = #bcir.asn1_rules<der>,
  default_tagging = #bcir.asn1_tagging<implicit>
} {
  bcir.asn1.type @Int attributes { kind = "primitive", universal = 2 : i64 } { }
  // expected-error @+1 {{R24: ASN.1 decode confused_cer is marked strict_der but declares it accepts cer, which is not a canonical transfer syntax}}
  bcir.asn1.decode @confused_cer { type = @Int, rules = #bcir.asn1_rules<cer>, strict_der }
}

// -----
// (22) The second hole: `strict_der` on a decode outside X.690. "Strict DER" is a
// category error about a JER decoder, not a stricter setting -- there is no DER for it to
// be strict about. R24 points at `strict_canonical` rather than silently accepting.
bcir.asn1.module @strict_der_on_jer attributes {
  oid = array<i64: 1, 3, 6, 1, 4, 1, 62596, 20>,
  rules = #bcir.asn1_rules<der>,
  default_tagging = #bcir.asn1_tagging<implicit>
} {
  bcir.asn1.type @Int attributes { kind = "primitive", universal = 2 : i64 } { }
  // expected-error @+1 {{R24: ASN.1 decode wrong_family is marked strict_der but declares the X.697 syntax bcir_canonical_jer; strict_der names X.690's own canonical form, so use strict_canonical for another family}}
  bcir.asn1.decode @wrong_family { type = @Int,
                                   rules = #bcir.asn1_rules<bcir_canonical_jer>,
                                   strict_der }
}

// -----
// (23 -- POSITIVE) A transcode: one schema, one abstract value, two transfer syntaxes.
// This is the law-rail form of what `bcir-asn1c --transcode` does and what the selection
// harness measures. Reading a peer's BER and emitting DER is the ordinary trust-boundary
// shape; the JER-to-COER pair is the build-plane shape the JSON roadmap is for.
bcir.asn1.module @transcodes attributes {
  oid = array<i64: 1, 3, 6, 1, 4, 1, 62596, 21>,
  rules = #bcir.asn1_rules<der>,
  default_tagging = #bcir.asn1_tagging<implicit>
} {
  bcir.asn1.type @Int attributes { kind = "primitive", universal = 2 : i64 } { }
  bcir.asn1.transcode @ber_in_der_out { type = @Int, from = #bcir.asn1_rules<ber>,
                                        to = #bcir.asn1_rules<der> }
  bcir.asn1.transcode @jer_in_coer_out { type = @Int, from = #bcir.asn1_rules<jer>,
                                         to = #bcir.asn1_rules<coer> }
  // preserve_value needs a canonical SOURCE too, and this one has it.
  bcir.asn1.transcode @replayable { type = @Int,
                                    from = #bcir.asn1_rules<bcir_canonical_jer>,
                                    to = #bcir.asn1_rules<der>, preserve_value }
}

// -----
// (24) A transcode EMITS its target, so the target falls under the same canonicality law
// as an encode.
bcir.asn1.module @transcode_loose_target attributes {
  oid = array<i64: 1, 3, 6, 1, 4, 1, 62596, 22>,
  rules = #bcir.asn1_rules<der>,
  default_tagging = #bcir.asn1_tagging<implicit>
} {
  bcir.asn1.type @Int attributes { kind = "primitive", universal = 2 : i64 } { }
  // expected-error @+1 {{R24: ASN.1 transcode to_ber targets ber (X.690), which is not canonical; a transcode EMITS its target}}
  bcir.asn1.transcode @to_ber { type = @Int, from = #bcir.asn1_rules<jer>,
                                to = #bcir.asn1_rules<ber> }
}

// -----
// (25) Transcoding a syntax to itself is not a transcode. It reads as one in a pass
// pipeline and does nothing, which is the kind of no-op that hides a wrong attribute
// instead of announcing it.
bcir.asn1.module @transcode_identity attributes {
  oid = array<i64: 1, 3, 6, 1, 4, 1, 62596, 23>,
  rules = #bcir.asn1_rules<der>,
  default_tagging = #bcir.asn1_tagging<implicit>
} {
  bcir.asn1.type @Int attributes { kind = "primitive", universal = 2 : i64 } { }
  // expected-error @+1 {{R24: ASN.1 transcode nothing has the same source and target syntax der}}
  bcir.asn1.transcode @nothing { type = @Int, from = #bcir.asn1_rules<der>,
                                 to = #bcir.asn1_rules<der> }
}

// -----
// (26) `preserve_value` asserts REPLAYABILITY: the same source octets always give the
// same target octets. A non-canonical source admits several encodings of one value, so
// the sender has a choice the replay cannot reproduce, and the claim is unsupportable
// however canonical the target is.
bcir.asn1.module @transcode_loose_source attributes {
  oid = array<i64: 1, 3, 6, 1, 4, 1, 62596, 24>,
  rules = #bcir.asn1_rules<der>,
  default_tagging = #bcir.asn1_tagging<implicit>
} {
  bcir.asn1.type @Int attributes { kind = "primitive", universal = 2 : i64 } { }
  // expected-error @+1 {{R24: ASN.1 transcode unreplayable claims preserve_value but reads jer, which admits more than one encoding of a value; a value-preserving transcode must read a canonical syntax or it cannot be replayed}}
  bcir.asn1.transcode @unreplayable { type = @Int, from = #bcir.asn1_rules<jer>,
                                      to = #bcir.asn1_rules<der>, preserve_value }
}
