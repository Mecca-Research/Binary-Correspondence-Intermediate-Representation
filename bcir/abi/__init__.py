"""The frozen BCIR StreamPack binary ABI (the portable artifact).

The StreamPack is BCIR's WASM-analog: a self-contained, portable, hot executable
artifact. `streampack_abi` defines a **versioned, frozen wire format** (v1) plus a
reference encoder/decoder. The same bytes are consumed by the Python oracle, the
(forthcoming) C runtime (`runtime/c/bcir_streampack.h`), and any embedder.
"""

from .streampack_abi import (
    ABI_MAGIC,
    ABI_VERSION,
    AbiError,
    StreamPackInspection,
    WireSpan,
    decode,
    encode,
    inspect_stream_pack,
)
from .artifact_bundle import (
    ENTRY_SIZE as ARTIFACT_ENTRY_SIZE,
    HEADER_SIZE as ARTIFACT_HEADER_SIZE,
    MAGIC as ARTIFACT_MAGIC,
    MAX_BUNDLE_BYTES,
    VERSION as ARTIFACT_VERSION,
    ArtifactBundle,
    ArtifactBundleInspection,
    ArtifactFormat,
    ArtifactKind,
    ArtifactVariant,
    BundleError,
    CompatibilityEnvelope,
    Endianness,
    WireSpan as ArtifactWireSpan,
    decode_bundle,
    encode_bundle,
    compatibility_sha256,
    host_envelope,
    inspect_bundle,
    is_compatible,
    read_bundle,
    select_variant,
    write_bundle,
)

__all__ = [
    "ABI_MAGIC",
    "ABI_VERSION",
    "AbiError",
    "StreamPackInspection",
    "WireSpan",
    "decode",
    "encode",
    "inspect_stream_pack",
    "ARTIFACT_MAGIC",
    "ARTIFACT_VERSION",
    "ARTIFACT_HEADER_SIZE",
    "ARTIFACT_ENTRY_SIZE",
    "MAX_BUNDLE_BYTES",
    "ArtifactBundle",
    "ArtifactBundleInspection",
    "ArtifactFormat",
    "ArtifactKind",
    "ArtifactVariant",
    "ArtifactWireSpan",
    "BundleError",
    "CompatibilityEnvelope",
    "Endianness",
    "decode_bundle",
    "encode_bundle",
    "compatibility_sha256",
    "host_envelope",
    "inspect_bundle",
    "is_compatible",
    "read_bundle",
    "select_variant",
    "write_bundle",
]
