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
