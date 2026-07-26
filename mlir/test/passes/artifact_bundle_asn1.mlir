// RUN: bcir-opt -bcir-verify %s | FileCheck %s
//
// The native BCAB directory and the ASN.1 transfer syntax are independent,
// versioned contracts. R24 sees this as an additive projection, never a
// replacement for the native bytes.

bcir.asn1.module @BCIR_ArtifactBundle attributes {
  oid = array<i64: 1, 3, 6, 1, 4, 1, 62596, 2>,
  rules = #bcir.asn1_rules<der>,
  default_tagging = #bcir.asn1_tagging<implicit>
} {
  bcir.asn1.type @U8 attributes { kind = "primitive", universal = 2 : i64,
    constraint_low = 0 : i64, constraint_high = 255 : i64 } { }
  bcir.asn1.type @U32 attributes { kind = "primitive", universal = 2 : i64,
    constraint_low = 0 : i64, constraint_high = 4294967295 : i64 } { }
  bcir.asn1.type @U64 attributes { kind = "primitive", universal = 2 : i64 } { }
  bcir.asn1.type @I32 attributes { kind = "primitive", universal = 2 : i64,
    constraint_low = -2147483648 : i64, constraint_high = 2147483647 : i64 } { }
  bcir.asn1.type @Enum attributes { kind = "primitive", universal = 10 : i64 } { }
  bcir.asn1.type @Utf8 attributes { kind = "primitive", universal = 12 : i64 } { }
  bcir.asn1.type @Octets attributes { kind = "primitive", universal = 4 : i64 } { }
  bcir.asn1.type @FeatureList attributes {
    kind = "sequence_of", element = @Utf8
  } { }
  bcir.asn1.type @ArtifactVariant attributes { kind = "sequence" } {
    bcir.asn1.component { name = "variantId", type = @Utf8, tag = 0 : i64,
      tagging = #bcir.asn1_tagging<implicit> }
    bcir.asn1.component { name = "kind", type = @Enum, tag = 1 : i64,
      tagging = #bcir.asn1_tagging<implicit> }
    bcir.asn1.component { name = "format", type = @Enum, tag = 2 : i64,
      tagging = #bcir.asn1_tagging<implicit> }
    bcir.asn1.component { name = "payload", type = @Octets, tag = 3 : i64,
      tagging = #bcir.asn1_tagging<implicit> }
    bcir.asn1.component { name = "triple", type = @Utf8, tag = 4 : i64,
      tagging = #bcir.asn1_tagging<implicit> }
    bcir.asn1.component { name = "architecture", type = @Utf8, tag = 5 : i64,
      tagging = #bcir.asn1_tagging<implicit> }
    bcir.asn1.component { name = "osAbi", type = @Utf8, tag = 6 : i64,
      tagging = #bcir.asn1_tagging<implicit> }
    bcir.asn1.component { name = "channel", type = @Utf8, tag = 7 : i64,
      tagging = #bcir.asn1_tagging<implicit> }
    bcir.asn1.component { name = "entrySymbol", type = @Utf8, tag = 8 : i64,
      tagging = #bcir.asn1_tagging<implicit> }
    bcir.asn1.component { name = "requiredFeatures", type = @FeatureList, tag = 9 : i64,
      tagging = #bcir.asn1_tagging<implicit> }
    bcir.asn1.component { name = "prohibitedFeatures", type = @FeatureList,
      tag = 10 : i64, tagging = #bcir.asn1_tagging<implicit> }
    bcir.asn1.component { name = "endianness", type = @Enum, tag = 11 : i64,
      tagging = #bcir.asn1_tagging<implicit> }
    bcir.asn1.component { name = "pointerBits", type = @U8, tag = 12 : i64,
      tagging = #bcir.asn1_tagging<implicit> }
    bcir.asn1.component { name = "machine", type = @U32, tag = 13 : i64,
      tagging = #bcir.asn1_tagging<implicit> }
    bcir.asn1.component { name = "priority", type = @I32, tag = 14 : i64,
      tagging = #bcir.asn1_tagging<implicit> }
    bcir.asn1.component { name = "provenanceDigest", type = @U64, tag = 15 : i64,
      tagging = #bcir.asn1_tagging<implicit> }
    bcir.asn1.component { name = "targetManifest", type = @Octets, tag = 16 : i64,
      tagging = #bcir.asn1_tagging<implicit> }
    bcir.asn1.component { name = "calibrationGen", type = @U64, tag = 17 : i64,
      tagging = #bcir.asn1_tagging<implicit> }
    bcir.asn1.component { name = "flags", type = @U8, tag = 18 : i64,
      tagging = #bcir.asn1_tagging<implicit> }
  }
  bcir.asn1.type @ArtifactVariants attributes {
    kind = "sequence_of", element = @ArtifactVariant
  } { }
  bcir.asn1.type @ArtifactBundle attributes { kind = "sequence" } {
    bcir.asn1.component { name = "version", type = @U8, tag = 0 : i64,
      tagging = #bcir.asn1_tagging<implicit> }
    bcir.asn1.component { name = "rootVariant", type = @Utf8, tag = 1 : i64,
      tagging = #bcir.asn1_tagging<implicit>, optional }
    bcir.asn1.component { name = "defaultVariant", type = @Utf8, tag = 2 : i64,
      tagging = #bcir.asn1_tagging<implicit>, optional }
    bcir.asn1.component { name = "provenanceDigest", type = @U64, tag = 3 : i64,
      tagging = #bcir.asn1_tagging<implicit> }
    bcir.asn1.component { name = "generation", type = @U64, tag = 4 : i64,
      tagging = #bcir.asn1_tagging<implicit> }
    bcir.asn1.component { name = "variants", type = @ArtifactVariants, tag = 5 : i64,
      tagging = #bcir.asn1_tagging<implicit> }
  }
  bcir.asn1.encode @emit_bundle_der {
    type = @ArtifactBundle, rules = #bcir.asn1_rules<der>
  }
  bcir.asn1.decode @accept_bundle_ber {
    type = @ArtifactBundle, rules = #bcir.asn1_rules<ber>
  }
  bcir.asn1.projection @artifact_bundle_projection {
    native = "artifact_bundle", type = @ArtifactBundle, additive
  }
}

// CHECK: bcir.asn1.module @BCIR_ArtifactBundle
// CHECK: oid = array<i64: 1, 3, 6, 1, 4, 1, 62596, 2>
// CHECK: bcir.asn1.type @ArtifactBundle
// CHECK: bcir.asn1.encode @emit_bundle_der
// CHECK: bcir.asn1.decode @accept_bundle_ber
// CHECK: bcir.asn1.projection @artifact_bundle_projection
// CHECK-SAME: additive
// CHECK-SAME: native = "artifact_bundle"
