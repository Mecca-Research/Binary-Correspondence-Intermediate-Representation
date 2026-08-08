// RUN: bcir-opt -bcir-verify -verify-diagnostics -split-input-file %s
//
// R25 (X.692 ECN encoding-definition legality) over the bcir.ecn.* ops. Oracle twins:
// bcir/asn1/ecn_user.py (the property groups and their restrictions) and
// bcir/asn1/ecn_syntax.py (the defined syntax that sets them).
//
// What is checked HERE and not on the oracle rail: these are the faults decidable from the
// SPECIFICATION alone, before any value exists. That distinction is sharper for ECN than for
// X.690, because an encoding definition module is written once and applied to many types --
// a fault that only fires on the right value can sit in one for a long time without anything
// noticing.
//
// Every op below parses and passes its own ODS verifier; only the LAW is violated.

// (1 -- POSITIVE) the frame header of bcir/asn1/BCIR-FrameHeader.ecn: a class per field, one
// concatenation structure whose textual order is the wire order, and one object per class.
bcir.ecn.module @FrameEncodings {
  bcir.ecn.class @Scaled_length attributes { base = "#INT" }
  bcir.ecn.class @Version       attributes { base = "#INT" }
  bcir.ecn.class @Active_low    attributes { base = "#BOOL" }
  bcir.ecn.class @Reserved      attributes { base = "#PAD" }
  bcir.ecn.class @Frame_header  attributes { base = "#CONCATENATION" }

  bcir.ecn.structure @FrameHeader_structure attributes { encoding_class = @Frame_header } {
    bcir.ecn.field { name = "payloadOctets", encoding_class = @Scaled_length }
    bcir.ecn.field { name = "version",       encoding_class = @Version }
    bcir.ecn.field { name = "urgent",        encoding_class = @Active_low }
    bcir.ecn.field { name = "reserved",      encoding_class = @Reserved }
  }

  bcir.ecn.object @lengthField attributes {
    encoding_class = @Scaled_length, space_size = 4 : i64, space_unit = 1 : i64
  } { }
  bcir.ecn.object @versionField attributes {
    encoding_class = @Version, space_size = 3 : i64, space_unit = 1 : i64
  } { }
  bcir.ecn.object @urgentFlag attributes {
    encoding_class = @Active_low, space_size = 1 : i64, space_unit = 1 : i64
  } { }
  bcir.ecn.object @reservedBits attributes {
    encoding_class = @Reserved, space_size = 2 : i64, space_unit = 1 : i64
  } { }
}

// -----

// (2 -- POSITIVE) the determinant groups, each written the way its clause requires: a length
// field set by the encoder, a start pointer, unused bits with the transform list its
// determination admits, and a bit reversal over an even unit.
bcir.ecn.module @Determinants {
  bcir.ecn.class @Len   attributes { base = "#INT" }
  bcir.ecn.class @Body  attributes { base = "#INT" }
  bcir.ecn.class @Frame attributes { base = "#CONCATENATION" }

  bcir.ecn.structure @S attributes { encoding_class = @Frame } {
    bcir.ecn.field { name = "len",  encoding_class = @Len, auxiliary }
    bcir.ecn.field { name = "body", encoding_class = @Body }
  }

  bcir.ecn.object @lenField attributes {
    encoding_class = @Len, space_size = 8 : i64, space_unit = 1 : i64
  } { }
  bcir.ecn.object @bodyField attributes {
    encoding_class = @Body,
    space_size = 2 : i64, space_unit = 8 : i64,
    space_determination = #bcir.ecn_space_determination<field_to_be_set>,
    space_reference = "len",
    start_pointer = "len",
    unused_determination = #bcir.ecn_unused_bits<field_to_be_set>,
    unused_reference = "len",
    unused_encoder_transforms,
    reversal = #bcir.ecn_reversal<reverse_half_units>
  } { }
}

// -----

// (3) 9.5.2: at most one encoding object per class in a set.
bcir.ecn.module @TwoObjects {
  bcir.ecn.class @Version attributes { base = "#INT" }
  bcir.ecn.object @first attributes {
    encoding_class = @Version, space_size = 3 : i64, space_unit = 1 : i64
  } { }
  // expected-error@+1 {{R25: ECN object second and first both realize Version}}
  bcir.ecn.object @second attributes {
    encoding_class = @Version, space_size = 4 : i64, space_unit = 1 : i64
  } { }
}

// -----

// (4) A class assignment chain that loops names no encoding category. The expectation
// matches on the tail rather than the head because the name that trips first depends on the
// map's iteration order, and a fixture that pinned it would be testing the container.
// expected-error@below {{is circular}}
bcir.ecn.module @CircularClass {
  bcir.ecn.class @A attributes { base = "B" }
  bcir.ecn.class @B attributes { base = "A" }
}

// -----

// (5) 16.3.1: a structure field name identifies one field, and every clause 22 REFERENCE
// names one.
bcir.ecn.module @DuplicateField {
  bcir.ecn.class @Int   attributes { base = "#INT" }
  bcir.ecn.class @Frame attributes { base = "#CONCATENATION" }
  bcir.ecn.structure @S attributes { encoding_class = @Frame } {
    bcir.ecn.field { name = "x", encoding_class = @Int }
    // expected-error@+1 {{R25: the ECN structure S names the field x twice}}
    bcir.ecn.field { name = "x", encoding_class = @Int }
  }
}

// -----

// (6) 22.2.2.2: ALIGNED TO ANY requires a START-POINTER, because the number of inserted bits
// is the encoder's choice and only the start pointer records it.
bcir.ecn.module @AnyWithoutPointer {
  bcir.ecn.class @Int attributes { base = "#INT" }
  // expected-error@+1 {{R25: ECN object f specifies ALIGNED TO ANY without a START-POINTER}}
  bcir.ecn.object @f attributes {
    encoding_class = @Int, align_unit = 8 : i64, align_any,
    space_size = 8 : i64, space_unit = 1 : i64
  } { }
}

// -----

// (7) 21.3.4/21.3.5: a determination with no USING reference names no field at all.
bcir.ecn.module @DeterminationWithoutReference {
  bcir.ecn.class @Int attributes { base = "#INT" }
  // expected-error@+1 {{R25: ECN object f states an encoding-space determination}}
  bcir.ecn.object @f attributes {
    encoding_class = @Int, space_size = 8 : i64, space_unit = 1 : i64,
    space_determination = #bcir.ecn_space_determination<field_to_be_set>
  } { }
}

// -----

// (8) 22.8.2.2, in the direction that is easy to miss: `not-needed` with a USING reference
// is as wrong as a determination without one.
bcir.ecn.module @NotNeededWithReference {
  bcir.ecn.class @Int attributes { base = "#INT" }
  // expected-error@+1 {{R25: ECN object f sets UNUSED BITS DETERMINED BY not_needed}}
  bcir.ecn.object @f attributes {
    encoding_class = @Int, space_size = 8 : i64, space_unit = 1 : i64,
    unused_determination = #bcir.ecn_unused_bits<not_needed>,
    unused_reference = "len"
  } { }
}

// -----

// (9) 22.8.2.5: DECODER-TRANSFORMS on `field-to-be-set` would never run, which reads as
// though they did.
bcir.ecn.module @WrongTransformList {
  bcir.ecn.class @Int attributes { base = "#INT" }
  // expected-error@+1 {{R25: ECN object f gives UNUSED BITS DECODER-TRANSFORMS}}
  bcir.ecn.object @f attributes {
    encoding_class = @Int, space_size = 8 : i64, space_unit = 1 : i64,
    unused_determination = #bcir.ecn_unused_bits<field_to_be_set>,
    unused_reference = "len",
    unused_decoder_transforms
  } { }
}

// -----

// (10) 22.12.2.3: reversing a one-bit unit is the identity, so asking for it has
// misunderstood something.
bcir.ecn.module @ReversalOverOneBit {
  bcir.ecn.class @Int attributes { base = "#INT" }
  // expected-error@+1 {{R25: ECN object f sets BIT-REVERSAL over a 1-bit unit}}
  bcir.ecn.object @f attributes {
    encoding_class = @Int, space_size = 8 : i64, space_unit = 1 : i64,
    reversal = #bcir.ecn_reversal<reverse_bits_in_units>
  } { }
}

// -----

// (11) 21.14.5: an odd unit has no half.
bcir.ecn.module @HalfOfAnOddUnit {
  bcir.ecn.class @Int attributes { base = "#INT" }
  // expected-error@+1 {{R25: ECN object f sets reverse_half_units over an odd 5-bit unit}}
  bcir.ecn.object @f attributes {
    encoding_class = @Int, space_size = 1 : i64, space_unit = 5 : i64,
    reversal = #bcir.ecn_reversal<reverse_half_units>
  } { }
}

// -----

// (12) 22.1.2.8: REPLACE STRUCTURE forbids INSERT AT HEAD.
bcir.ecn.module @StructureWithHeadEnd {
  bcir.ecn.class @Frame attributes { base = "#CONCATENATION" }
  // expected-error@+1 {{R25: ECN object f has both REPLACE STRUCTURE and INSERT AT HEAD}}
  bcir.ecn.object @f attributes {
    encoding_class = @Frame,
    replace = #bcir.ecn_replace<structure>,
    replacement_structure = @Repl,
    replacement_object = @ReplObj,
    head_end = @Head
  } { }
}

// -----

// (13) 23.3.2.2 / 23.7.2.5 / 23.12.2.3, all the same sentence: if REPLACE is set, no other
// encoding property group shall be.
bcir.ecn.module @ReplaceAndMore {
  bcir.ecn.class @Frame attributes { base = "#CONCATENATION" }
  // expected-error@+1 {{R25: ECN object f sets REPLACE and another encoding property group}}
  bcir.ecn.object @f attributes {
    encoding_class = @Frame,
    replace = #bcir.ecn_replace<all_components>,
    replacement_structure = @Repl,
    space_size = 8 : i64, space_unit = 1 : i64
  } { }
}

// -----

// (14) 23.7.2.4: at most one of IF, IF-ALL and ELSE.
bcir.ecn.module @ConditionAndElse {
  bcir.ecn.class @CInt attributes { base = "#CONDITIONAL-INT" }
  // expected-error@+1 {{R25: ECN object f has both a condition and ELSE}}
  bcir.ecn.object @f attributes {
    encoding_class = @CInt, space_size = 8 : i64, space_unit = 1 : i64, unconditional
  } {
    bcir.ecn.condition {
      condition = #bcir.ecn_range_condition<bounded_without_negatives>
    }
  }
}

// -----

// (15) 23.7.2.7: `subtract:lower-bound` under a condition that does not guarantee a lower
// bound subtracts a bound that may not exist.
bcir.ecn.module @SubtractWithoutBound {
  bcir.ecn.class @CInt attributes { base = "#CONDITIONAL-INT" }
  // expected-error@+1 {{R25: ECN object f applies the INT-TO-INT transform}}
  bcir.ecn.object @f attributes {
    encoding_class = @CInt, space_size = 8 : i64, space_unit = 1 : i64,
    subtracts_lower_bound
  } {
    bcir.ecn.condition {
      condition = #bcir.ecn_range_condition<unbounded_or_no_lower_bound>
    }
  }
}

// -----

// (16) 21.11.5, the direction where a comparison is REQUIRED: `test-range` has nothing to
// compare against without one.
bcir.ecn.module @RangeWithoutComparison {
  bcir.ecn.class @CInt attributes { base = "#CONDITIONAL-INT" }
  bcir.ecn.object @f attributes {
    encoding_class = @CInt, space_size = 8 : i64, space_unit = 1 : i64
  } {
    // expected-error@+1 {{R25: the ECN range condition test_range requires a Comparison}}
    bcir.ecn.condition { condition = #bcir.ecn_range_condition<test_range> }
  }
}

// -----

// (17) 21.11.5, the other direction: a comparison on a bound SHAPE tests nothing. This is
// the half a one-sided law would pass.
bcir.ecn.module @ShapeWithComparison {
  bcir.ecn.class @CInt attributes { base = "#CONDITIONAL-INT" }
  bcir.ecn.object @f attributes {
    encoding_class = @CInt, space_size = 8 : i64, space_unit = 1 : i64
  } {
    // expected-error@+1 {{does not admit a Comparison or a comparator}}
    bcir.ecn.condition {
      condition = #bcir.ecn_range_condition<bounded_without_negatives>,
      comparison = #bcir.ecn_comparison<greater_than>,
      comparator = 0 : i64
    }
  }
}

// -----

// (18 -- POSITIVE) the constructor categories, with the handle machinery each of them reads.
// Two alternatives exhibiting one handle at one set of positions, an optional component whose
// presence is a field the encoder sets, and a concatenation ordered `random` -- which is legal
// exactly because a handle is exhibited (22.10.2.1).
bcir.ecn.module @Constructors {
  bcir.ecn.class @Kind    attributes { base = "#INT" }
  bcir.ecn.class @Label   attributes { base = "#TAG" }
  bcir.ecn.class @Label2  attributes { base = "Label" }
  bcir.ecn.class @AltA    attributes { base = "#INT" }
  bcir.ecn.class @AltB    attributes { base = "#INT" }
  bcir.ecn.class @Choice  attributes { base = "#ALTERNATIVES" }
  bcir.ecn.class @Maybe   attributes { base = "#OPTIONAL" }
  bcir.ecn.class @Whole   attributes { base = "#CONCATENATION" }

  bcir.ecn.structure @S attributes { encoding_class = @Whole } {
    bcir.ecn.field { name = "present", encoding_class = @Kind, auxiliary }
    bcir.ecn.field { name = "body",    encoding_class = @Choice }
    bcir.ecn.field { name = "extra",   encoding_class = @Maybe }
  }

  // 22.9.1.9's accepting half, through a clause 11 assignment chain: @Label2 is assigned
  // from @Label which is assigned from #TAG, and 21.16.5 makes `tag:any` legal for all three.
  bcir.ecn.object @label attributes {
    encoding_class = @Label2, space_size = 8 : i64, space_unit = 1 : i64,
    exhibited_handle = "tagnum", handle_positions = array<i64: 0, 1, 2, 3>,
    handle_value_kind = #bcir.ecn_handle_value_kind<tag>
  } { }
  bcir.ecn.object @altA attributes {
    encoding_class = @AltA, space_size = 8 : i64, space_unit = 1 : i64,
    exhibited_handle = "kind", handle_positions = array<i64: 0, 1>,
    handle_value_kind = #bcir.ecn_handle_value_kind<range>
  } { }
  bcir.ecn.object @altB attributes {
    encoding_class = @AltB, space_size = 8 : i64, space_unit = 1 : i64,
    exhibited_handle = "kind", handle_positions = array<i64: 1, 0>,
    handle_value_kind = #bcir.ecn_handle_value_kind<range>
  } { }
  bcir.ecn.object @choice attributes {
    encoding_class = @Choice,
    alternative_determination = #bcir.ecn_alternative_determination<handle>,
    alternative_handle_set,
    alternative_ordering = #bcir.ecn_component_order<textual>
  } { }
  bcir.ecn.object @maybe attributes {
    encoding_class = @Maybe,
    optionality_determination = #bcir.ecn_optionality_determination<field_to_be_set>,
    optionality_reference = "present",
    optionality_encoder_transforms
  } { }
  bcir.ecn.object @whole attributes {
    encoding_class = @Whole,
    concatenation_order = #bcir.ecn_component_order<random>,
    exhibited_handle = "kind", handle_positions = array<i64: 0, 1>,
    handle_value_kind = #bcir.ecn_handle_value_kind<ranges>
  } { }
}

// -----

// (19) 22.9.2.1: "all identification handles with the same name shall specify the same set of
// bit positions". Order does not matter -- 22.9.1.6 makes the list a SET, ordered from zero
// upwards by encoders and decoders alike -- so this trips on the members, not the spelling.
bcir.ecn.module @HandlePositionsDisagree {
  bcir.ecn.class @A attributes { base = "#INT" }
  bcir.ecn.class @B attributes { base = "#INT" }
  bcir.ecn.object @a attributes {
    encoding_class = @A, space_size = 8 : i64, space_unit = 1 : i64,
    exhibited_handle = "kind", handle_positions = array<i64: 0, 1>,
    handle_value_kind = #bcir.ecn_handle_value_kind<number>
  } { }
  // expected-error@+1 {{R25: ECN object b gives the identification handle kind different bit positions}}
  bcir.ecn.object @b attributes {
    encoding_class = @B, space_size = 8 : i64, space_unit = 1 : i64,
    exhibited_handle = "kind", handle_positions = array<i64: 0, 2>,
    handle_value_kind = #bcir.ecn_handle_value_kind<number>
  } { }
}

// -----

// (20) 22.9.2.3, whose NOTE gives the reason: a decoder "needs to move to the alignment
// position before looking for the handle", and cannot if two objects disagree about where
// that is.
bcir.ecn.module @HandleAlignmentDisagrees {
  bcir.ecn.class @A attributes { base = "#INT" }
  bcir.ecn.class @B attributes { base = "#INT" }
  bcir.ecn.object @a attributes {
    encoding_class = @A, space_size = 8 : i64, space_unit = 1 : i64, align_unit = 8 : i64,
    exhibited_handle = "kind", handle_positions = array<i64: 0>,
    handle_value_kind = #bcir.ecn_handle_value_kind<number>
  } { }
  // expected-error@+1 {{X.692 22.9.2.3 requires one pre-alignment unit per handle}}
  bcir.ecn.object @b attributes {
    encoding_class = @B, space_size = 8 : i64, space_unit = 1 : i64, align_unit = 4 : i64,
    exhibited_handle = "kind", handle_positions = array<i64: 0>,
    handle_value_kind = #bcir.ecn_handle_value_kind<number>
  } { }
}

// -----

// (21) 22.9.1.6: the positions in AT are "a set of integer values", so a repeat puts one bit
// twice into the conceptual handle field.
bcir.ecn.module @HandleRepeatsAPosition {
  bcir.ecn.class @A attributes { base = "#INT" }
  // expected-error@+1 {{X.692 22.9.1.6 calls the list in AT "a set of integer values"}}
  bcir.ecn.object @a attributes {
    encoding_class = @A, space_size = 8 : i64, space_unit = 1 : i64,
    exhibited_handle = "kind", handle_positions = array<i64: 2, 0, 2>,
    handle_value_kind = #bcir.ecn_handle_value_kind<number>
  } { }
}

// -----

// (22) 22.9.1.9: `tag:any` takes its value from a tag number, so only a #TAG object has one.
bcir.ecn.module @TagAnyOnANonTagClass {
  bcir.ecn.class @A attributes { base = "#INT" }
  // expected-error@+1 {{X.692 22.9.1.9 admits it only for an encoding object of the #TAG class}}
  bcir.ecn.object @a attributes {
    encoding_class = @A, space_size = 8 : i64, space_unit = 1 : i64,
    exhibited_handle = "kind", handle_positions = array<i64: 0, 1>,
    handle_value_kind = #bcir.ecn_handle_value_kind<tag>
  } { }
}

// -----

// (23) 22.5.2.3: USING says which field carries the presence information, and `handle` reads
// bits that are already there. The two cannot both be how absence is detected.
bcir.ecn.module @PresenceHandleWithUsing {
  bcir.ecn.class @M attributes { base = "#OPTIONAL" }
  // expected-error@+1 {{X.692 22.5.2.3 forbids USING for `handle` and `pointer`}}
  bcir.ecn.object @m attributes {
    encoding_class = @M,
    optionality_determination = #bcir.ecn_optionality_determination<handle>,
    optionality_reference = "p"
  } { }
}

// -----

// (24) 22.5.2.6: a transform list on the wrong determination never runs, which reads as
// though it did.
bcir.ecn.module @PresenceTransformsThatNeverRun {
  bcir.ecn.class @M attributes { base = "#OPTIONAL" }
  // expected-error@+1 {{X.692 22.5.2.6 admits them only for `field-to-be-set`}}
  bcir.ecn.object @m attributes {
    encoding_class = @M,
    optionality_determination = #bcir.ecn_optionality_determination<field_to_be_used>,
    optionality_reference = "p",
    optionality_encoder_transforms
  } { }
}

// -----

// (25) 22.5.2.4: 21.5.9 reads a start pointer's zero as absence, so `pointer` without one
// leaves nothing to distinguish absent from present.
bcir.ecn.module @PointerPresenceWithoutAPointer {
  bcir.ecn.class @M attributes { base = "#OPTIONAL" }
  // expected-error@+1 {{X.692 22.5.2.4 requires one in the same encoding object}}
  bcir.ecn.object @m attributes {
    encoding_class = @M,
    optionality_determination = #bcir.ecn_optionality_determination<pointer>
  } { }
}

// -----

// (26) 22.6.2.2, the alternatives twin of 22.5.2.2.
bcir.ecn.module @AlternativeHandleWithoutHandle {
  bcir.ecn.class @C attributes { base = "#ALTERNATIVES" }
  // expected-error@+1 {{X.692 22.6.2.2 admits HANDLE only for `handle`}}
  bcir.ecn.object @c attributes {
    encoding_class = @C,
    alternative_determination = #bcir.ecn_alternative_determination<field_to_be_set>,
    alternative_reference = "k",
    alternative_handle_set
  } { }
}

// -----

// (27) 22.6.1.1 declares the alternatives' ordering as ENUMERATED {textual, tag} -- two
// values where 22.10.1.1 has three. One enum holds both, so this is a law rather than a
// parse failure, and that is the honest place for it: the two clauses genuinely share three
// meanings and differ only in which are admissible.
bcir.ecn.module @RandomAlternatives {
  bcir.ecn.class @C attributes { base = "#ALTERNATIVES" }
  // expected-error@+1 {{X.692 22.6.1.1 declares that property as ENUMERATED}}
  bcir.ecn.object @c attributes {
    encoding_class = @C,
    alternative_determination = #bcir.ecn_alternative_determination<field_to_be_set>,
    alternative_reference = "k",
    alternative_ordering = #bcir.ecn_component_order<random>
  } { }
}

// -----

// (28) 22.10.2.1: an encoder free to reorder is decodable only if each component announces
// which one it is.
bcir.ecn.module @RandomOrderWithoutAHandle {
  bcir.ecn.class @W attributes { base = "#CONCATENATION" }
  // expected-error@+1 {{X.692 22.10.2.1 requires the encoding objects applied to all}}
  bcir.ecn.object @w attributes {
    encoding_class = @W,
    concatenation_order = #bcir.ecn_component_order<random>
  } { }
}

// -----

// (29 -- POSITIVE) containment in both directions. 22.11's CONTENTS-ENCODING says which rules
// encode a contained type; 21.3.6's `container` determination says whose end bounds an element.
// They are different relationships and an object may carry either.
bcir.ecn.module @Containment {
  bcir.ecn.class @Wrapper attributes { base = "#OCTETS" }
  bcir.ecn.class @Tail    attributes { base = "#INT" }
  bcir.ecn.class @Whole   attributes { base = "#CONCATENATION" }

  bcir.ecn.structure @S attributes { encoding_class = @Whole } {
    bcir.ecn.field { name = "wrapper", encoding_class = @Wrapper }
    bcir.ecn.field { name = "tail",    encoding_class = @Tail }
  }

  bcir.ecn.object @wrapper attributes {
    encoding_class = @Wrapper, space_size = -2 : i64, space_unit = 8 : i64,
    space_determination = #bcir.ecn_space_determination<field_to_be_set>,
    space_reference = "len",
    contents_encoding, contents_completed_by, contents_override
  } { }
  // 21.3.6's other direction: this element's end is its container's, so it carries no
  // determinant of its own beyond naming the container.
  bcir.ecn.object @tail attributes {
    encoding_class = @Tail, space_size = -2 : i64, space_unit = 1 : i64,
    space_determination = #bcir.ecn_space_determination<container>,
    space_reference = "wrapper"
  } { }
}

// -----

// (30) 22.11.1.2 brackets COMPLETED BY inside CONTENTS-ENCODING, and 22.11.1.5 makes the group
// set only when that keyword is used. A tail without its head is a property nothing reads.
bcir.ecn.module @CompletedByWithoutContents {
  bcir.ecn.class @W attributes { base = "#OCTETS" }
  // expected-error@+1 {{X.692 22.11.1.2 brackets both inside it}}
  bcir.ecn.object @w attributes {
    encoding_class = @W, space_size = 8 : i64, space_unit = 1 : i64,
    contents_completed_by
  } { }
}

// -----

// (31) The same rule for OVERRIDE, and this is the half that matters: a reader would take a
// bare OVERRIDE for a statement about the ASN.1 ENCODED BY, when 22.11.2.1 gives that
// constraint outright precedence whenever CONTENTS-ENCODING is unset.
bcir.ecn.module @OverrideWithoutContents {
  bcir.ecn.class @W attributes { base = "#OCTETS" }
  // expected-error@+1 {{X.692 22.11.1.2 brackets both inside it}}
  bcir.ecn.object @w attributes {
    encoding_class = @W, space_size = 8 : i64, space_unit = 1 : i64,
    contents_override
  } { }
}

// -----

// (32) 22.4.2.3/22.4.2.4: a container's end is a position, not a value carried through a
// field, so there is nothing for a transform list to convert.
bcir.ecn.module @ContainerWithTransforms {
  bcir.ecn.class @T attributes { base = "#INT" }
  // expected-error@+1 {{X.692 22.4.2.3/22.4.2.4 admit them only for `field-to-be-set`}}
  bcir.ecn.object @t attributes {
    encoding_class = @T, space_size = -2 : i64, space_unit = 1 : i64,
    space_determination = #bcir.ecn_space_determination<container>,
    space_reference = "box",
    unused_determination = #bcir.ecn_unused_bits<field_to_be_set>,
    unused_reference = "u",
    unused_encoder_transforms
  } { }
}
