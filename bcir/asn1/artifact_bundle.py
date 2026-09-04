"""ASN.1 projection of the BCIR Artifact Bundle.

The native BCAB v1 wire remains frozen.  This module adds a second, schema-visible
transfer syntax for the same abstract bundle:

* DER out, BER in (X.690);
* CANONICAL-OER out, BASIC-OER in (X.696).

The projection carries every semantic directory field and every payload, but not
derived BCAB offsets, padding, CRCs, or SHA-256 values.  Reconstructing the native
wire recomputes those fields canonically.  Consequently, a valid native bundle
survives either projection byte-for-byte:

    der_to_native(native_to_der(native)) == native
    oer_to_native(native_to_oer(native), canonical=True) == native
"""

from __future__ import annotations

from ..abi.artifact_bundle import (
    ENTRY_SIZE,
    HEADER_SIZE,
    MAX_BUNDLE_BYTES,
    MAX_ENTRIES,
    ArtifactBundle,
    ArtifactFormat,
    ArtifactKind,
    ArtifactVariant,
    BundleError,
    Endianness,
    encode_bundle,
    inspect_bundle,
)
from .codec import Strictness
from .constraints import ValueRange
from .schema import Component, Module, Primitive, Sequence, SequenceOf
from .tags import Universal


BCIR_ARC: tuple[int, ...] = (1, 3, 6, 1, 4, 1, 62596)
ARTIFACT_BUNDLE_MODULE_OID: tuple[int, ...] = (*BCIR_ARC, 2)
PROJECTION_VERSION = 1

# A native payload can occupy almost the whole 1-GiB BCAB limit.  The projection
# adds bounded metadata and ASN.1 framing per entry.  Refuse larger input before
# schema decoding so an attacker cannot use this convenience API as an unbounded
# ingestion path.
MAX_PROJECTION_BYTES = MAX_BUNDLE_BYTES + MAX_ENTRIES * 2048 + 4096

_FLAG_R12_ATTESTED = 1 << 0
_FLAG_EXECUTABLE = 1 << 1
_FLAG_PORTABLE = 1 << 2
_FLAG_DEBUG = 1 << 3
_FLAG_MASK = _FLAG_R12_ATTESTED | _FLAG_EXECUTABLE | _FLAG_PORTABLE | _FLAG_DEBUG

_BUNDLE_KEYS = frozenset(
    {
        "version",
        "rootVariant",
        "defaultVariant",
        "provenanceDigest",
        "generation",
        "variants",
    }
)
_BUNDLE_REQUIRED_KEYS = frozenset(
    {
        "version",
        "provenanceDigest",
        "generation",
        "variants",
    }
)
_VARIANT_KEYS = frozenset(
    {
        "variantId",
        "kind",
        "format",
        "payload",
        "triple",
        "architecture",
        "osAbi",
        "channel",
        "entrySymbol",
        "requiredFeatures",
        "prohibitedFeatures",
        "endianness",
        "pointerBits",
        "machine",
        "priority",
        "provenanceDigest",
        "targetManifest",
        "calibrationGen",
        "flags",
    }
)

_U8 = Primitive(Universal.INTEGER, "U8", ValueRange(0, (1 << 8) - 1))
_U32 = Primitive(Universal.INTEGER, "U32", ValueRange(0, (1 << 32) - 1))
_U64 = Primitive(Universal.INTEGER, "U64", ValueRange(0, (1 << 64) - 1))
_I32 = Primitive(
    Universal.INTEGER,
    "I32",
    ValueRange(-(1 << 31), (1 << 31) - 1),
)
# One ENUMERATED per named type, each carrying its own enumeration, rather than a single
# anonymous `ENUMERATED` shared by all three. DER and OER encode the enumeration VALUE
# (X.690 §8.4, X.696 §11), so a bare primitive was sufficient for them and the difference
# was invisible; X.691 §14.1 encodes the enumeration INDEX, which cannot be derived without
# the root list. Sharing one object would additionally have given ArtifactKind and
# Endianness the same PER width, which they do not have (5 bits against 2).
_KIND_ENUM = Primitive(
    Universal.ENUMERATED,
    "ArtifactKind",
    enumeration=tuple((k.name.lower(), int(k)) for k in ArtifactKind),
)
_FORMAT_ENUM = Primitive(
    Universal.ENUMERATED,
    "ArtifactFormat",
    enumeration=tuple((f.name.lower(), int(f)) for f in ArtifactFormat),
)
_ENDIAN_ENUM = Primitive(
    Universal.ENUMERATED,
    "Endianness",
    enumeration=tuple((e.name.lower(), int(e)) for e in Endianness),
)
_UTF8 = Primitive(Universal.UTF8_STRING, "UTF8String")
_OCTETS = Primitive(Universal.OCTET_STRING, "OCTET STRING")
_FEATURES = SequenceOf(_UTF8, "FeatureList")

ARTIFACT_VARIANT = Sequence(
    (
        Component("variantId", _UTF8, tag=0),
        Component("kind", _KIND_ENUM, tag=1),
        Component("format", _FORMAT_ENUM, tag=2),
        Component("payload", _OCTETS, tag=3),
        Component("triple", _UTF8, tag=4),
        Component("architecture", _UTF8, tag=5),
        Component("osAbi", _UTF8, tag=6),
        Component("channel", _UTF8, tag=7),
        Component("entrySymbol", _UTF8, tag=8),
        Component("requiredFeatures", _FEATURES, tag=9),
        Component("prohibitedFeatures", _FEATURES, tag=10),
        Component("endianness", _ENDIAN_ENUM, tag=11),
        Component("pointerBits", _U8, tag=12),
        Component("machine", _U32, tag=13),
        Component("priority", _I32, tag=14),
        Component("provenanceDigest", _U64, tag=15),
        Component("targetManifest", _OCTETS, tag=16),
        Component("calibrationGen", _U64, tag=17),
        Component("flags", _U8, tag=18),
    ),
    name="ArtifactVariant",
)

ARTIFACT_BUNDLE = Sequence(
    (
        Component("version", _U8, tag=0),
        Component("rootVariant", _UTF8, tag=1, optional=True),
        Component("defaultVariant", _UTF8, tag=2, optional=True),
        Component("provenanceDigest", _U64, tag=3),
        Component("generation", _U64, tag=4),
        Component(
            "variants",
            SequenceOf(ARTIFACT_VARIANT, "SEQUENCE OF ArtifactVariant"),
            tag=5,
        ),
    ),
    name="ArtifactBundle",
)

MODULE = Module(
    "BCIR-ArtifactBundle",
    ARTIFACT_BUNDLE_MODULE_OID,
    {
        "U8": _U8,
        "U32": _U32,
        "U64": _U64,
        "I32": _I32,
        "ArtifactKind": _KIND_ENUM,
        "ArtifactFormat": _FORMAT_ENUM,
        "Endianness": _ENDIAN_ENUM,
        "FeatureList": _FEATURES,
        "ArtifactVariant": ARTIFACT_VARIANT,
        "ArtifactBundle": ARTIFACT_BUNDLE,
    },
)


def _bounded_octets(data: bytes, label: str) -> bytes:
    if not isinstance(data, bytes):
        raise BundleError(f"{label} must be immutable bytes")
    if not 1 <= len(data) <= MAX_PROJECTION_BYTES:
        raise BundleError(f"{label} size is outside [1, {MAX_PROJECTION_BYTES}] bytes")
    return data


def _integer(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BundleError(f"ASN.1 {field} must be an integer")
    return value


def _string(value, field: str) -> str:
    if not isinstance(value, str):
        raise BundleError(f"ASN.1 {field} must be a string")
    return value


def _strings(value, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise BundleError(f"ASN.1 {field} must be a string sequence")
    return tuple(value)


def _require_keys(
    value: dict,
    allowed: frozenset[str],
    required: frozenset[str],
    label: str,
) -> None:
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise BundleError(f"ASN.1 {label} has unknown fields {sorted(unknown)}")
    if missing:
        raise BundleError(f"ASN.1 {label} is missing fields {sorted(missing)}")


def _require_native_geometry(bundle: ArtifactBundle) -> None:
    """Check native BCAB's aggregate bound without materializing a second payload copy."""
    cursor = (HEADER_SIZE + len(bundle.variants) * ENTRY_SIZE + 7) & ~7
    for variant in bundle.variants:
        cursor = (cursor + len(variant.payload) + 7) & ~7
        if cursor > MAX_BUNDLE_BYTES:
            raise BundleError(f"bundle exceeds the {MAX_BUNDLE_BYTES}-byte native wire limit")


def bundle_to_value(bundle: ArtifactBundle) -> dict:
    """Return the ASN.1 abstract value for a validated ``ArtifactBundle``."""
    if not isinstance(bundle, ArtifactBundle):
        raise BundleError("bundle_to_value expects an ArtifactBundle")
    _require_native_geometry(bundle)
    value = {
        "version": PROJECTION_VERSION,
        "provenanceDigest": bundle.provenance_digest,
        "generation": bundle.generation,
        "variants": [],
    }
    if bundle.root_variant_id:
        value["rootVariant"] = bundle.root_variant_id
    if bundle.default_variant_id:
        value["defaultVariant"] = bundle.default_variant_id
    for variant in bundle.variants:
        value["variants"].append(
            {
                "variantId": variant.variant_id,
                "kind": int(variant.kind),
                "format": int(variant.format),
                "payload": variant.payload,
                "triple": variant.triple,
                "architecture": variant.architecture,
                "osAbi": variant.os_abi,
                "channel": variant.channel,
                "entrySymbol": variant.entry_symbol,
                "requiredFeatures": list(variant.required_features),
                "prohibitedFeatures": list(variant.prohibited_features),
                "endianness": int(variant.endianness),
                "pointerBits": variant.pointer_bits,
                "machine": variant.e_machine,
                "priority": variant.priority,
                "provenanceDigest": variant.provenance_digest,
                "targetManifest": (
                    bytes.fromhex(variant.target_manifest_sha256)
                    if variant.target_manifest_sha256
                    else b""
                ),
                "calibrationGen": variant.cal_gen,
                "flags": variant.flags,
            }
        )
    return value


def value_to_bundle(value: dict) -> ArtifactBundle:
    """Recover and fully validate a native bundle object from an ASN.1 value."""
    if not isinstance(value, dict):
        raise BundleError("ASN.1 ArtifactBundle value must be a mapping")
    _require_keys(value, _BUNDLE_KEYS, _BUNDLE_REQUIRED_KEYS, "ArtifactBundle")
    version = _integer(value.get("version"), "version")
    if version != PROJECTION_VERSION:
        raise BundleError(f"unsupported ASN.1 ArtifactBundle projection version {version}")
    raw_variants = value.get("variants")
    if not isinstance(raw_variants, list) or not 1 <= len(raw_variants) <= MAX_ENTRIES:
        raise BundleError(f"ASN.1 variants must contain 1..{MAX_ENTRIES} entries")

    variants: list[ArtifactVariant] = []
    for index, raw in enumerate(raw_variants):
        if not isinstance(raw, dict):
            raise BundleError(f"ASN.1 variant {index} must be a mapping")
        _require_keys(raw, _VARIANT_KEYS, _VARIANT_KEYS, f"variant {index}")
        manifest = raw.get("targetManifest")
        payload = raw.get("payload")
        if not isinstance(manifest, bytes) or len(manifest) not in (0, 32):
            raise BundleError(f"ASN.1 variant {index} targetManifest must contain 0 or 32 octets")
        if not isinstance(payload, bytes):
            raise BundleError(f"ASN.1 variant {index} payload must be octets")
        flags = _integer(raw.get("flags"), f"variant {index} flags")
        if flags < 0 or flags & ~_FLAG_MASK:
            raise BundleError(f"ASN.1 variant {index} has unknown flag bits")
        try:
            variant = ArtifactVariant(
                _string(raw.get("variantId"), f"variant {index} variantId"),
                ArtifactKind(_integer(raw.get("kind"), f"variant {index} kind")),
                ArtifactFormat(_integer(raw.get("format"), f"variant {index} format")),
                payload,
                _string(raw.get("triple"), f"variant {index} triple"),
                _string(raw.get("architecture"), f"variant {index} architecture"),
                _string(raw.get("osAbi"), f"variant {index} osAbi"),
                _string(raw.get("channel"), f"variant {index} channel"),
                _string(raw.get("entrySymbol"), f"variant {index} entrySymbol"),
                _strings(
                    raw.get("requiredFeatures"),
                    f"variant {index} requiredFeatures",
                ),
                _strings(
                    raw.get("prohibitedFeatures"),
                    f"variant {index} prohibitedFeatures",
                ),
                Endianness(_integer(raw.get("endianness"), f"variant {index} endianness")),
                _integer(raw.get("pointerBits"), f"variant {index} pointerBits"),
                _integer(raw.get("machine"), f"variant {index} machine"),
                _integer(raw.get("priority"), f"variant {index} priority"),
                _integer(
                    raw.get("provenanceDigest"),
                    f"variant {index} provenanceDigest",
                ),
                manifest.hex(),
                _integer(raw.get("calibrationGen"), f"variant {index} calibrationGen"),
                bool(flags & _FLAG_R12_ATTESTED),
                bool(flags & _FLAG_EXECUTABLE),
                bool(flags & _FLAG_PORTABLE),
                bool(flags & _FLAG_DEBUG),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, BundleError):
                raise
            raise BundleError(f"ASN.1 variant {index} is invalid: {exc}") from exc
        variants.append(variant)

    root = value.get("rootVariant", "")
    default = value.get("defaultVariant", "")
    bundle = ArtifactBundle(
        tuple(variants),
        _string(root, "rootVariant"),
        _string(default, "defaultVariant"),
        _integer(value.get("provenanceDigest"), "provenanceDigest"),
        _integer(value.get("generation"), "generation"),
    )
    # Check native header/directory geometry and the aggregate 1-GiB limit before a
    # caller can treat the decoded projection as an admissible native artifact.
    _require_native_geometry(bundle)
    return bundle


def encode_bundle_der(bundle: ArtifactBundle) -> bytes:
    """Emit the canonical DER projection of ``bundle``."""
    return MODULE.encode("ArtifactBundle", bundle_to_value(bundle))


def decode_bundle_der(
    data: bytes,
    *,
    strictness: Strictness = Strictness.DER,
) -> ArtifactBundle:
    """Decode the projection, requiring DER unless BER is explicitly requested."""
    return value_to_bundle(
        MODULE.decode(
            "ArtifactBundle",
            _bounded_octets(data, "ASN.1 artifact-bundle projection"),
            strictness=strictness,
        )
    )


def encode_bundle_oer(bundle: ArtifactBundle) -> bytes:
    """Emit the CANONICAL-OER projection of ``bundle``."""
    from .oer import OerRules, encode_oer

    return encode_oer(
        ARTIFACT_BUNDLE,
        bundle_to_value(bundle),
        rules=OerRules.CANONICAL,
    )


def decode_bundle_oer(data: bytes, *, canonical: bool = False) -> ArtifactBundle:
    """Decode BASIC-OER, or require CANONICAL-OER when ``canonical`` is true."""
    from .oer import OerRules, decode_oer

    rules = OerRules.CANONICAL if canonical else OerRules.BASIC
    bounded = _bounded_octets(data, "OER artifact-bundle projection")
    value = decode_oer(ARTIFACT_BUNDLE, bounded, rules=rules)
    if canonical:
        # The generic decoder validates type-specific COER choices (for example the
        # BOOLEAN spelling).  Re-encoding additionally catches every non-minimal
        # length/number spelling and ordering choice, so "canonical" is an exact
        # trust-boundary guarantee rather than a best-effort mode.
        from .oer import encode_oer

        if encode_oer(ARTIFACT_BUNDLE, value, rules=OerRules.CANONICAL) != bounded:
            from .tags import Asn1Error

            raise Asn1Error("artifact-bundle input is not CANONICAL-OER")
    return value_to_bundle(value)


def encode_bundle_per(bundle: ArtifactBundle, *, aligned: bool = False) -> bytes:
    """Emit the CANONICAL-PER projection of ``bundle``.

    UNALIGNED is the default because the projection is dominated by bounded integers and
    a compact wire is the whole reason to reach for PER; ``aligned=True`` selects the
    variant that trades size for octet-aligned reads.  The two do not interwork (X.691
    §7.8), so the caller's choice has to travel with the bytes.
    """
    from .per import PerRules, PerVariant, encode_per

    variant = PerVariant.ALIGNED if aligned else PerVariant.UNALIGNED
    return encode_per(
        ARTIFACT_BUNDLE,
        bundle_to_value(bundle),
        variant=variant,
        rules=PerRules.CANONICAL,
    )


def decode_bundle_per(data: bytes, *, aligned: bool = False) -> ArtifactBundle:
    """Decode a PER projection of a bundle."""
    from .per import PerRules, PerVariant, decode_per

    variant = PerVariant.ALIGNED if aligned else PerVariant.UNALIGNED
    bounded = _bounded_octets(data, "PER artifact-bundle projection")
    value = decode_per(bounded, ARTIFACT_BUNDLE, variant=variant, rules=PerRules.CANONICAL)
    return value_to_bundle(value)


def native_to_per(data: bytes, *, aligned: bool = False) -> bytes:
    """Validate native BCAB bytes and project the abstract bundle to CANONICAL-PER."""
    return encode_bundle_per(inspect_bundle(data).bundle, aligned=aligned)


def per_to_native(data: bytes, *, aligned: bool = False) -> bytes:
    """Decode a PER projection and reconstruct canonical native BCAB bytes."""
    return encode_bundle(decode_bundle_per(data, aligned=aligned))


def native_to_der(data: bytes) -> bytes:
    """Validate native BCAB bytes and project the abstract bundle to DER."""
    return encode_bundle_der(inspect_bundle(data).bundle)


def der_to_native(
    data: bytes,
    *,
    strictness: Strictness = Strictness.DER,
) -> bytes:
    """Decode a DER/BER projection and reconstruct canonical native BCAB bytes."""
    return encode_bundle(decode_bundle_der(data, strictness=strictness))


def native_to_oer(data: bytes) -> bytes:
    """Validate native BCAB bytes and project the abstract bundle to COER."""
    return encode_bundle_oer(inspect_bundle(data).bundle)


def oer_to_native(data: bytes, *, canonical: bool = False) -> bytes:
    """Decode an OER projection and reconstruct canonical native BCAB bytes."""
    return encode_bundle(decode_bundle_oer(data, canonical=canonical))


__all__ = [
    "ARTIFACT_BUNDLE",
    "ARTIFACT_BUNDLE_MODULE_OID",
    "ARTIFACT_VARIANT",
    "BCIR_ARC",
    "MAX_PROJECTION_BYTES",
    "MODULE",
    "PROJECTION_VERSION",
    "bundle_to_value",
    "decode_bundle_der",
    "decode_bundle_oer",
    "decode_bundle_per",
    "der_to_native",
    "encode_bundle_der",
    "encode_bundle_oer",
    "encode_bundle_per",
    "native_to_der",
    "native_to_oer",
    "native_to_per",
    "oer_to_native",
    "per_to_native",
    "value_to_bundle",
]
