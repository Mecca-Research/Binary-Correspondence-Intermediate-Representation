"""The frozen BCIR StreamPack binary ABI (the portable artifact).

The StreamPack is BCIR's WASM-analog: a self-contained, portable, hot executable
artifact. `streampack_abi` defines a **versioned, frozen wire format** (v1) plus a
reference encoder/decoder. The same bytes are consumed by the Python oracle, the
C runtime (`runtime/c/bcir_streampack.h`), and any embedder.

`execution_plan_abi` is the plan the pack was derived from, as bytes (ExecutionPlanV1,
G11): the same conventions, its own magic, a C twin (`runtime/c/bcir_execution_plan.h`).
`control_abi` is the control plane as bytes (ControlRecordV1, G14): fixed-width records a
resident plane decides by their bytes, with a C twin (`runtime/c/bcir_control_plane.h`).
`telemetry_envelope` is the identity-carrying telemetry record (TelemetryEnvelopeV0, G15) the
live ring carries, with a C twin (`runtime/c/bcir_telemetry_envelope.h`). It is VERSION ZERO:
experimental, with no compatibility promise until the driver roadmap's traces freeze it.
`shard_manifest` is the manifest-of-shards (BSHM, G16): a pack too large to ship as one travels as
runnable shards and a frame named by digest, reassembling to the whole's bytes. Version zero too.
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
from .execution_plan_abi import (
    PLAN_HEADER_SIZE,
    PLAN_MAGIC,
    PLAN_VERSION,
    decode_plan,
    encode_plan,
    validate_plan,
)
from .control_abi import (
    CONTROL_MAGIC,
    CONTROL_VERSION,
    ControlError,
    check_control_mac,
    decode_control,
    encode_control,
    issue_control,
    sign_control,
    validate_control,
)
from .telemetry_envelope import (
    ENVELOPE_MAGIC,
    ENVELOPE_SIZES,
    ENVELOPE_VERSION,
    TelemetryEnvelope,
    TelemetryError,
    datadna_of,
    decode_envelope,
    encode_envelope,
    validate_envelope,
)
from .shard_manifest import (
    MANIFEST_MAGIC,
    MANIFEST_VERSION,
    ShardError,
    ShardManifest,
    Split,
    decode_manifest,
    encode_manifest,
    reassemble,
    split,
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
    "PLAN_HEADER_SIZE",
    "PLAN_MAGIC",
    "PLAN_VERSION",
    "decode_plan",
    "encode_plan",
    "validate_plan",
    "CONTROL_MAGIC",
    "CONTROL_VERSION",
    "ControlError",
    "check_control_mac",
    "decode_control",
    "encode_control",
    "issue_control",
    "sign_control",
    "validate_control",
    "ENVELOPE_MAGIC",
    "ENVELOPE_SIZES",
    "ENVELOPE_VERSION",
    "TelemetryEnvelope",
    "TelemetryError",
    "datadna_of",
    "decode_envelope",
    "encode_envelope",
    "validate_envelope",
    "MANIFEST_MAGIC",
    "MANIFEST_VERSION",
    "ShardError",
    "ShardManifest",
    "Split",
    "decode_manifest",
    "encode_manifest",
    "reassemble",
    "split",
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
