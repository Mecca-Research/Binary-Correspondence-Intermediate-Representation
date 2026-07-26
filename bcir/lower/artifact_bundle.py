"""Lower validated BCAB directory and selection metadata to the BCIR MLIR dialect."""

from __future__ import annotations

import json
import re

from ..abi.artifact_bundle import (
    ArtifactBundle,
    CompatibilityEnvelope,
    BundleError,
    compatibility_sha256,
    encode_bundle,
    inspect_bundle,
    select_variant,
)


_SYMBOL = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$.]*$")


def _string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _symbol(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise BundleError("MLIR symbol names must be nonempty strings")
    return "@" + value if _SYMBOL.fullmatch(value) else "@" + _string(value)


def _array(values) -> str:
    return "[" + ", ".join(_string(value) for value in values) + "]"


def bundle_to_mlir(bundle: ArtifactBundle, *, symbol_name: str = "artifact_bundle") -> str:
    """Emit parsed, verifier-ready ``bcir.artifact.*`` directory operations."""
    data = encode_bundle(bundle)
    info = inspect_bundle(data)
    payload_spans = {span.index: span for span in info.spans if span.kind == "payload"}
    lines = [
        f"bcir.artifact.bundle {_symbol(symbol_name)} attributes {{",
        f"  version = 1 : i32, root_variant = {_string(bundle.root_variant_id)},",
        f"  default_variant = {_string(bundle.default_variant_id)},",
        f"  provenance_digest = {_string(f'{bundle.provenance_digest:016x}')},",
        f"  generation = {bundle.generation} : i64, wire_bytes = {len(data)} : i64,",
        f"  artifact_sha256 = {_string(info.artifact_sha256)},",
        f"  body_crc32 = {info.body_crc32} : i64, header_crc32 = {info.header_crc32} : i64",
        "} {",
    ]
    for index, variant in enumerate(bundle.variants):
        span = payload_spans[index]
        lines.extend([
            f"  bcir.artifact.variant {_symbol(variant.variant_id)} {{",
            f"    kind = {_string(variant.kind.name.lower())}, "
            f"format = {_string(variant.format.name.lower())},",
            f"    triple = {_string(variant.triple)}, architecture = {_string(variant.architecture)},",
            f"    os_abi = {_string(variant.os_abi)}, channel = {_string(variant.channel)},",
            f"    entry_symbol = {_string(variant.entry_symbol)}, "
            f"endianness = {_string(variant.endianness.name.lower())},",
            f"    pointer_bits = {variant.pointer_bits} : i32, machine = {variant.e_machine} : i64,",
            f"    priority = {variant.priority} : i32, flags = {variant.flags} : i32,",
            f"    provenance_digest = {_string(f'{variant.provenance_digest:016x}')},",
            f"    required_features = {_array(variant.required_features)},",
            f"    prohibited_features = {_array(variant.prohibited_features)},",
            f"    payload_offset = {span.offset} : i64, payload_size = {span.length} : i64,",
            f"    payload_crc32 = {variant.payload_crc32} : i64,",
            f"    payload_sha256 = {_string(variant.payload_sha256)},",
            f"    target_manifest_sha256 = {_string(variant.target_manifest_sha256)},",
            f"    cal_gen = {variant.cal_gen} : i64",
            "  }",
        ])
    lines.append("}")
    return "\n".join(lines) + "\n"


def selection_to_mlir(bundle: ArtifactBundle, envelope: CompatibilityEnvelope, *,
                      bundle_symbol: str = "artifact_bundle",
                      selection_symbol: str = "artifact_selection",
                      classification: str = "exact") -> str:
    """Emit the deterministic selector decision as a bundle-companion verifier op."""
    if classification not in ("exact", "quantized", "approximate"):
        raise BundleError("classification must be exact, quantized, or approximate")
    selected = select_variant(bundle, envelope)
    return (
        f"bcir.artifact.selection {_symbol(selection_symbol)} {{ "
        f"bundle = {_symbol(bundle_symbol)}, variant = {_string(selected.variant_id)}, "
        f"classification = {_string(classification)}, "
        f"envelope_sha256 = {_string(compatibility_sha256(envelope))}, "
        f"generation = {bundle.generation} : i64 }}\n"
    )


def asn1_contract_to_mlir(*, module_symbol: str = "BCIR_ArtifactBundle") -> str:
    """Emit the R24-visible additive ASN.1 contract for native BCAB.

    Payload bytes do not appear in this schema declaration.  The existing
    ``bcir.artifact.*`` operations carry a concrete native directory; this module
    states that the same abstract value also has DER/BER transfer syntax.
    """
    symbol = _symbol(module_symbol)
    component_rows = (
        ("variantId", "Utf8", 0, False),
        ("kind", "Enum", 1, False),
        ("format", "Enum", 2, False),
        ("payload", "Octets", 3, False),
        ("triple", "Utf8", 4, False),
        ("architecture", "Utf8", 5, False),
        ("osAbi", "Utf8", 6, False),
        ("channel", "Utf8", 7, False),
        ("entrySymbol", "Utf8", 8, False),
        ("requiredFeatures", "FeatureList", 9, False),
        ("prohibitedFeatures", "FeatureList", 10, False),
        ("endianness", "Enum", 11, False),
        ("pointerBits", "U8", 12, False),
        ("machine", "U32", 13, False),
        ("priority", "I32", 14, False),
        ("provenanceDigest", "U64", 15, False),
        ("targetManifest", "Octets", 16, False),
        ("calibrationGen", "U64", 17, False),
        ("flags", "U8", 18, False),
    )
    lines = [
        f"bcir.asn1.module {symbol} attributes {{",
        "  oid = array<i64: 1, 3, 6, 1, 4, 1, 62596, 2>,",
        "  rules = #bcir.asn1_rules<der>,",
        "  default_tagging = #bcir.asn1_tagging<implicit>",
        "} {",
        '  bcir.asn1.type @U8 attributes { kind = "primitive", universal = 2 : i64, '
        "constraint_low = 0 : i64, constraint_high = 255 : i64 } { }",
        '  bcir.asn1.type @U32 attributes { kind = "primitive", universal = 2 : i64, '
        "constraint_low = 0 : i64, constraint_high = 4294967295 : i64 } { }",
        '  bcir.asn1.type @U64 attributes { kind = "primitive", universal = 2 : i64 } { }',
        '  bcir.asn1.type @I32 attributes { kind = "primitive", universal = 2 : i64, '
        "constraint_low = -2147483648 : i64, "
        "constraint_high = 2147483647 : i64 } { }",
        '  bcir.asn1.type @Enum attributes { kind = "primitive", universal = 10 : i64 } { }',
        '  bcir.asn1.type @Utf8 attributes { kind = "primitive", universal = 12 : i64 } { }',
        '  bcir.asn1.type @Octets attributes { kind = "primitive", universal = 4 : i64 } { }',
        '  bcir.asn1.type @FeatureList attributes { kind = "sequence_of", '
        "element = @Utf8 } { }",
        '  bcir.asn1.type @ArtifactVariant attributes { kind = "sequence" } {',
    ]
    for name, type_name, tag, optional in component_rows:
        suffix = ", optional" if optional else ""
        lines.append(
            "    bcir.asn1.component { "
            f"name = {_string(name)}, type = @{type_name}, tag = {tag} : i64, "
            f"tagging = #bcir.asn1_tagging<implicit>{suffix} }}"
        )
    lines.extend([
        "  }",
        '  bcir.asn1.type @ArtifactVariants attributes { kind = "sequence_of", '
        "element = @ArtifactVariant } { }",
        '  bcir.asn1.type @ArtifactBundle attributes { kind = "sequence" } {',
        '    bcir.asn1.component { name = "version", type = @U8, tag = 0 : i64, '
        "tagging = #bcir.asn1_tagging<implicit> }",
        '    bcir.asn1.component { name = "rootVariant", type = @Utf8, tag = 1 : i64, '
        "tagging = #bcir.asn1_tagging<implicit>, optional }",
        '    bcir.asn1.component { name = "defaultVariant", type = @Utf8, tag = 2 : i64, '
        "tagging = #bcir.asn1_tagging<implicit>, optional }",
        '    bcir.asn1.component { name = "provenanceDigest", type = @U64, tag = 3 : i64, '
        "tagging = #bcir.asn1_tagging<implicit> }",
        '    bcir.asn1.component { name = "generation", type = @U64, tag = 4 : i64, '
        "tagging = #bcir.asn1_tagging<implicit> }",
        '    bcir.asn1.component { name = "variants", type = @ArtifactVariants, '
        "tag = 5 : i64, tagging = #bcir.asn1_tagging<implicit> }",
        "  }",
        "  bcir.asn1.encode @emit_bundle_der { type = @ArtifactBundle, "
        "rules = #bcir.asn1_rules<der> }",
        "  bcir.asn1.decode @accept_bundle_ber { type = @ArtifactBundle, "
        "rules = #bcir.asn1_rules<ber> }",
        '  bcir.asn1.projection @artifact_bundle_projection { native = "artifact_bundle", '
        "type = @ArtifactBundle, additive }",
        "}",
    ])
    return "\n".join(lines) + "\n"


__all__ = ["asn1_contract_to_mlir", "bundle_to_mlir", "selection_to_mlir"]
