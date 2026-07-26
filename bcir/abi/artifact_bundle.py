"""BCIR Artifact Bundle (BCAB) v1 -- strict multi-backend artifact container.

BCAB does not replace ELF, COFF, Mach-O, WASM, JVM class files, SPIR-V, or
StreamPack.  It is the deterministic directory and compatibility contract that
binds those standard payloads to BCIR's R12/R13 provenance and target envelope.

The wire is deliberately simple enough for a freestanding, allocation-free C
reader: one 128-byte header, ``n`` fixed 448-byte directory entries, zero
padding to eight-byte boundaries, then payloads in canonical variant-id order.
Every payload has CRC-32 and SHA-256; the body has CRC-32; the complete artifact
has an embedded SHA-256 computed with the digest and header-CRC fields zeroed.
Unknown versions, flags, kinds, formats, reserved bytes, gaps, overlaps, and
non-canonical metadata are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
import os
import platform
import re
import struct
import tempfile
import zlib


MAGIC = b"BCAB"
VERSION = 1
HEADER_SIZE = 128
ENTRY_SIZE = 448
MAX_ENTRIES = 1024
MAX_BUNDLE_BYTES = 1 << 30
NO_INDEX = 0xFFFFFFFF

# magic, version, header size, flags, count, entry size, root, default,
# directory offset/size, payload offset, file size, provenance, generation,
# body CRC, header CRC, embedded SHA-256, reserved.
_HEADER = struct.Struct("<4sHHIIIIIQQQQQQII32s12s")
_ENTRY = struct.Struct(
    "<HHBBHIiQQQQII32s32s48s48s24s24s24s32s64s64s"
)
assert _HEADER.size == HEADER_SIZE
assert _ENTRY.size == ENTRY_SIZE

_HEADER_CRC_OFFSET = 80
_BUNDLE_SHA_END = 116

_FLAG_R12_ATTESTED = 1 << 0
_FLAG_EXECUTABLE = 1 << 1
_FLAG_PORTABLE = 1 << 2
_FLAG_DEBUG = 1 << 3
_ENTRY_FLAG_MASK = (
    _FLAG_R12_ATTESTED | _FLAG_EXECUTABLE | _FLAG_PORTABLE | _FLAG_DEBUG
)

_FIELD_SIZES = {
    "variant_id": 48,
    "triple": 48,
    "architecture": 24,
    "os_abi": 24,
    "channel": 24,
    "entry_symbol": 32,
    "required_features": 64,
    "prohibited_features": 64,
}
_FEATURE_RE = re.compile(r"^[A-Za-z0-9_.+:-]+$")
_ZERO_SHA = bytes(32)
_ZERO_INTEGRITY_FIELDS = bytes(36)


class BundleError(ValueError):
    """A malformed or incompatible BCAB artifact."""


class ArtifactKind(IntEnum):
    STREAM_PACK = 1
    ELF_OBJECT = 2
    ELF_SHARED = 3
    COFF_OBJECT = 4
    MACHO_OBJECT = 5
    ARCHIVE = 6
    WASM = 7
    LLVM_BITCODE = 8
    LLVM_IR = 9
    PTX = 10
    CUBIN = 11
    SPIRV = 12
    JVM_CLASS = 13
    CIL = 14
    C_SOURCE = 15
    CPP_SOURCE = 16
    SYCL_SOURCE = 17
    ASSEMBLY = 18
    ELF_EXECUTABLE = 19
    PE_EXECUTABLE = 20
    PE_SHARED = 21
    MACHO_EXECUTABLE = 22
    MACHO_SHARED = 23
    RAW_BINARY = 24


class ArtifactFormat(IntEnum):
    NONE = 0
    STREAM_PACK = 1
    ELF = 2
    COFF = 3
    MACHO = 4
    ARCHIVE = 5
    WASM = 6
    LLVM_BITCODE = 7
    TEXT = 8
    SPIRV = 9
    JVM_CLASS = 10
    PE = 11
    RAW = 12


class Endianness(IntEnum):
    NEUTRAL = 0
    LITTLE = 1
    BIG = 2


_KIND_FORMATS = {
    ArtifactKind.STREAM_PACK: frozenset((ArtifactFormat.STREAM_PACK,)),
    ArtifactKind.ELF_OBJECT: frozenset((ArtifactFormat.ELF,)),
    ArtifactKind.ELF_SHARED: frozenset((ArtifactFormat.ELF,)),
    ArtifactKind.COFF_OBJECT: frozenset((ArtifactFormat.COFF,)),
    ArtifactKind.MACHO_OBJECT: frozenset((ArtifactFormat.MACHO,)),
    ArtifactKind.ARCHIVE: frozenset((ArtifactFormat.ARCHIVE,)),
    ArtifactKind.WASM: frozenset((ArtifactFormat.WASM,)),
    ArtifactKind.LLVM_BITCODE: frozenset((ArtifactFormat.LLVM_BITCODE,)),
    ArtifactKind.LLVM_IR: frozenset((ArtifactFormat.TEXT,)),
    ArtifactKind.PTX: frozenset((ArtifactFormat.TEXT,)),
    ArtifactKind.CUBIN: frozenset((ArtifactFormat.ELF,)),
    ArtifactKind.SPIRV: frozenset((ArtifactFormat.SPIRV,)),
    ArtifactKind.JVM_CLASS: frozenset((ArtifactFormat.JVM_CLASS,)),
    ArtifactKind.CIL: frozenset((ArtifactFormat.TEXT,)),
    ArtifactKind.C_SOURCE: frozenset((ArtifactFormat.TEXT,)),
    ArtifactKind.CPP_SOURCE: frozenset((ArtifactFormat.TEXT,)),
    ArtifactKind.SYCL_SOURCE: frozenset((ArtifactFormat.TEXT,)),
    ArtifactKind.ASSEMBLY: frozenset((ArtifactFormat.TEXT,)),
    ArtifactKind.ELF_EXECUTABLE: frozenset((ArtifactFormat.ELF,)),
    ArtifactKind.PE_EXECUTABLE: frozenset((ArtifactFormat.PE,)),
    ArtifactKind.PE_SHARED: frozenset((ArtifactFormat.PE,)),
    ArtifactKind.MACHO_EXECUTABLE: frozenset((ArtifactFormat.MACHO,)),
    ArtifactKind.MACHO_SHARED: frozenset((ArtifactFormat.MACHO,)),
    ArtifactKind.RAW_BINARY: frozenset((ArtifactFormat.RAW,)),
}
_NATIVE_FORMATS = frozenset((
    ArtifactFormat.ELF,
    ArtifactFormat.COFF,
    ArtifactFormat.MACHO,
    ArtifactFormat.PE,
))
_NAMED_EXECUTABLE_KINDS = frozenset((
    ArtifactKind.ELF_EXECUTABLE,
    ArtifactKind.PE_EXECUTABLE,
    ArtifactKind.MACHO_EXECUTABLE,
))


def _u(value, bits: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < (1 << bits):
        raise BundleError(f"{field} must be an unsigned {bits}-bit integer")
    return value


def _i32(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not -(1 << 31) <= value < (1 << 31):
        raise BundleError(f"{field} must be a signed 32-bit integer")
    return value


def _ascii(value: str, field: str, *, required: bool = False) -> str:
    if not isinstance(value, str) or (required and not value):
        raise BundleError(f"{field} must be {'a nonempty ' if required else ''}string")
    if value != value.strip() or any(ord(ch) < 0x20 or ord(ch) > 0x7E for ch in value):
        raise BundleError(f"{field} must be canonical printable ASCII without edge whitespace")
    limit = _FIELD_SIZES[field] - 1
    if len(value.encode("ascii")) > limit:
        raise BundleError(f"{field} exceeds its {limit}-byte wire limit")
    return value


def _features(value, field: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list, frozenset, set)):
        raise BundleError(f"{field} must be a feature sequence")
    out = tuple(value)
    if any(not isinstance(item, str) or not _FEATURE_RE.fullmatch(item) for item in out):
        raise BundleError(f"{field} contains a malformed feature name")
    canonical = tuple(sorted(set(out)))
    if tuple(out) != canonical:
        raise BundleError(f"{field} must be sorted and duplicate-free")
    joined = ",".join(canonical)
    _ascii(joined, field)
    return canonical


def _sha(value: str, field: str, *, optional: bool = True) -> str:
    if optional and value == "":
        return value
    if (not isinstance(value, str) or len(value) != 64 or value != value.lower()
            or any(ch not in "0123456789abcdef" for ch in value)):
        raise BundleError(f"{field} must be a lowercase SHA-256 digest")
    if optional and value == "0" * 64:
        raise BundleError(f"{field} cannot use the all-zero absent-value sentinel")
    return value


def _fixed(value: str, field: str) -> bytes:
    raw = value.encode("ascii")
    size = _FIELD_SIZES[field]
    if len(raw) >= size:
        raise BundleError(f"{field} exceeds its fixed wire field")
    return raw + bytes(size - len(raw))


def _unfixed(raw: bytes, field: str) -> str:
    try:
        end = raw.index(0)
    except ValueError as exc:
        raise BundleError(f"{field} is missing its NUL terminator") from exc
    if any(raw[end:]):
        raise BundleError(f"{field} has nonzero bytes after its terminator")
    try:
        value = raw[:end].decode("ascii")
    except UnicodeDecodeError as exc:
        raise BundleError(f"{field} is not ASCII") from exc
    return _ascii(value, field, required=(field == "variant_id"))


def _align8(value: int) -> int:
    return (value + 7) & ~7


def _text_payload(payload: bytes, label: str) -> str:
    if b"\x00" in payload:
        raise BundleError(f"{label} text contains NUL")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BundleError(f"{label} is not UTF-8") from exc


def _elf_identity(payload: bytes) -> tuple[Endianness, int, int]:
    if len(payload) < 20 or payload[:4] != b"\x7fELF":
        raise BundleError("ELF payload has no valid ELF identification")
    cls, data = payload[4], payload[5]
    if cls not in (1, 2) or data not in (1, 2):
        raise BundleError("ELF payload has an unsupported class or byte order")
    endian = Endianness.LITTLE if data == 1 else Endianness.BIG
    machine = struct.unpack_from("<H" if data == 1 else ">H", payload, 18)[0]
    return endian, 32 if cls == 1 else 64, machine


def _macho_identity(payload: bytes) -> tuple[Endianness, int, int, int]:
    if len(payload) < 28:
        raise BundleError("Mach-O payload is shorter than its fixed header")
    identities = {
        b"\xce\xfa\xed\xfe": (Endianness.LITTLE, 32, "<"),
        b"\xcf\xfa\xed\xfe": (Endianness.LITTLE, 64, "<"),
        b"\xfe\xed\xfa\xce": (Endianness.BIG, 32, ">"),
        b"\xfe\xed\xfa\xcf": (Endianness.BIG, 64, ">"),
    }
    try:
        endian, bits, prefix = identities[payload[:4]]
    except KeyError as exc:
        raise BundleError("Mach-O payload has no supported object magic") from exc
    cpu_type = struct.unpack_from(prefix + "I", payload, 4)[0]
    file_type = struct.unpack_from(prefix + "I", payload, 12)[0]
    return endian, bits, cpu_type, file_type


def _pe_identity(payload: bytes) -> tuple[int, int, bool]:
    if len(payload) < 0x40 or payload[:2] != b"MZ":
        raise BundleError("PE payload has no DOS header")
    pe_offset = struct.unpack_from("<I", payload, 0x3C)[0]
    if pe_offset > len(payload) - 24 or payload[pe_offset:pe_offset + 4] != b"PE\x00\x00":
        raise BundleError("PE payload has no bounded PE/COFF header")
    machine = struct.unpack_from("<H", payload, pe_offset + 4)[0]
    optional_size = struct.unpack_from("<H", payload, pe_offset + 20)[0]
    characteristics = struct.unpack_from("<H", payload, pe_offset + 22)[0]
    if optional_size < 2 or pe_offset + 24 + optional_size > len(payload):
        raise BundleError("PE optional header is truncated")
    optional_magic = struct.unpack_from("<H", payload, pe_offset + 24)[0]
    if optional_magic not in (0x10B, 0x20B):
        raise BundleError("PE optional header has an unknown class")
    return machine, 32 if optional_magic == 0x10B else 64, bool(characteristics & 0x2000)


def _validate_payload(variant: "ArtifactVariant") -> None:
    payload = variant.payload
    kind = variant.kind
    if not payload:
        raise BundleError(f"variant {variant.variant_id!r} has an empty payload")
    if kind == ArtifactKind.STREAM_PACK:
        from .streampack_abi import decode
        try:
            pack = decode(payload)
            segment_ids = [segment.claim_id for segment in pack.segments]
            trace_ids = [trace.claim_id for trace in pack.trace_notes]
            prefetch_names = [prefetch.name for prefetch in pack.prefetches]
            if len(segment_ids) != len(set(segment_ids)):
                raise BundleError("duplicate segment claim_id")
            if len(trace_ids) != len(set(trace_ids)):
                raise BundleError("duplicate trace claim_id")
            if len(prefetch_names) != len(set(prefetch_names)):
                raise BundleError("duplicate prefetch name")
            traces = set(trace_ids)
            prefetch_targets = {
                prefetch.name: set(prefetch.targets) for prefetch in pack.prefetches
            }
            for segment in pack.segments:
                if segment.claim_id not in traces:
                    raise BundleError(
                        f"segment {segment.name!r} has no matching trace note"
                    )
                if segment.prefetch is None:
                    continue
                if segment.prefetch not in prefetch_targets:
                    raise BundleError(
                        f"segment {segment.name!r} names an unknown prefetch"
                    )
                if (segment.reads
                        and not set(segment.reads) & prefetch_targets[segment.prefetch]):
                    raise BundleError(
                        f"segment {segment.name!r} prefetch covers no read resource"
                    )
        except Exception as exc:
            raise BundleError(f"variant {variant.variant_id!r} is not a valid StreamPack: {exc}") from exc
    elif variant.format == ArtifactFormat.ELF:
        endian, bits, machine = _elf_identity(payload)
        if variant.endianness != endian or variant.pointer_bits != bits:
            raise BundleError(f"variant {variant.variant_id!r} ELF class/endianness disagrees with metadata")
        if variant.e_machine != machine:
            raise BundleError(f"variant {variant.variant_id!r} ELF e_machine {machine} != metadata {variant.e_machine}")
        elf_type = struct.unpack_from("<H" if endian == Endianness.LITTLE else ">H", payload, 16)[0]
        if kind == ArtifactKind.ELF_OBJECT and elf_type != 1:
            raise BundleError(f"variant {variant.variant_id!r} is not an ELF relocatable object")
        if kind == ArtifactKind.ELF_SHARED and elf_type != 3:
            raise BundleError(f"variant {variant.variant_id!r} is not an ELF shared object")
        if kind == ArtifactKind.ELF_EXECUTABLE and elf_type not in (2, 3):
            raise BundleError(f"variant {variant.variant_id!r} is not an ELF executable/PIE")
    elif kind == ArtifactKind.COFF_OBJECT:
        if len(payload) < 20:
            raise BundleError(f"variant {variant.variant_id!r} is too short for a COFF object")
        machine = struct.unpack_from("<H", payload, 0)[0]
        if variant.endianness != Endianness.LITTLE:
            raise BundleError(f"variant {variant.variant_id!r} COFF metadata must be little-endian")
        if variant.e_machine != machine:
            raise BundleError(
                f"variant {variant.variant_id!r} COFF machine {machine} != metadata "
                f"{variant.e_machine}"
            )
    elif kind == ArtifactKind.MACHO_OBJECT:
        endian, bits, cpu_type, file_type = _macho_identity(payload)
        if variant.endianness != endian or variant.pointer_bits != bits:
            raise BundleError(
                f"variant {variant.variant_id!r} Mach-O class/endianness disagrees with metadata"
            )
        if variant.e_machine != cpu_type:
            raise BundleError(
                f"variant {variant.variant_id!r} Mach-O CPU type {cpu_type} != metadata "
                f"{variant.e_machine}"
            )
        if file_type != 1:
            raise BundleError(f"variant {variant.variant_id!r} is not a Mach-O object")
    elif kind in (ArtifactKind.MACHO_EXECUTABLE, ArtifactKind.MACHO_SHARED):
        endian, bits, cpu_type, file_type = _macho_identity(payload)
        expected = 2 if kind == ArtifactKind.MACHO_EXECUTABLE else 6
        if variant.endianness != endian or variant.pointer_bits != bits:
            raise BundleError(f"variant {variant.variant_id!r} Mach-O metadata disagrees with its header")
        if variant.e_machine != cpu_type:
            raise BundleError(f"variant {variant.variant_id!r} Mach-O CPU type disagrees with metadata")
        if file_type != expected:
            raise BundleError(f"variant {variant.variant_id!r} has the wrong Mach-O file type")
    elif kind in (ArtifactKind.PE_EXECUTABLE, ArtifactKind.PE_SHARED):
        machine, bits, is_dll = _pe_identity(payload)
        if variant.endianness != Endianness.LITTLE or variant.pointer_bits != bits:
            raise BundleError(f"variant {variant.variant_id!r} PE class/endianness disagrees with metadata")
        if variant.e_machine != machine:
            raise BundleError(f"variant {variant.variant_id!r} PE machine disagrees with metadata")
        if is_dll != (kind == ArtifactKind.PE_SHARED):
            raise BundleError(f"variant {variant.variant_id!r} PE DLL flag disagrees with its kind")
    elif kind == ArtifactKind.ARCHIVE:
        if not payload.startswith(b"!<arch>\n"):
            raise BundleError(f"variant {variant.variant_id!r} is not an ar archive")
    elif kind == ArtifactKind.WASM:
        if len(payload) < 8 or payload[:8] != b"\x00asm\x01\x00\x00\x00":
            raise BundleError(f"variant {variant.variant_id!r} is not a WASM v1 module")
    elif kind == ArtifactKind.LLVM_BITCODE:
        if payload[:4] not in (b"BC\xc0\xde", b"\xde\xc0\x17\x0b"):
            raise BundleError(f"variant {variant.variant_id!r} is not LLVM bitcode")
    elif kind == ArtifactKind.PTX:
        text = _text_payload(payload, "PTX payload")
        if ".version" not in text or ".target" not in text:
            raise BundleError(f"variant {variant.variant_id!r} is not recognizable PTX")
    elif kind == ArtifactKind.SPIRV:
        if len(payload) < 20 or len(payload) % 4 or payload[:4] not in (
                b"\x03\x02\x23\x07", b"\x07\x23\x02\x03"):
            raise BundleError(f"variant {variant.variant_id!r} is not a SPIR-V module")
    elif kind == ArtifactKind.JVM_CLASS:
        if len(payload) < 10 or payload[:4] != b"\xca\xfe\xba\xbe":
            raise BundleError(f"variant {variant.variant_id!r} is not a JVM class file")
    elif variant.format == ArtifactFormat.TEXT:
        _text_payload(payload, f"variant {variant.variant_id!r}")


@dataclass(frozen=True)
class ArtifactVariant:
    variant_id: str
    kind: ArtifactKind
    format: ArtifactFormat
    payload: bytes
    triple: str = ""
    architecture: str = ""
    os_abi: str = ""
    channel: str = ""
    entry_symbol: str = ""
    required_features: tuple[str, ...] = ()
    prohibited_features: tuple[str, ...] = ()
    endianness: Endianness = Endianness.NEUTRAL
    pointer_bits: int = 0
    e_machine: int = 0
    priority: int = 0
    provenance_digest: int = 0
    target_manifest_sha256: str = ""
    cal_gen: int = 0
    r12_attested: bool = False
    executable: bool = False
    portable: bool = False
    debug: bool = False

    def __post_init__(self) -> None:
        _ascii(self.variant_id, "variant_id", required=True)
        for field in ("triple", "architecture", "os_abi", "channel", "entry_symbol"):
            _ascii(getattr(self, field), field)
        try:
            kind = ArtifactKind(self.kind)
            fmt = ArtifactFormat(self.format)
            endian = Endianness(self.endianness)
        except (TypeError, ValueError) as exc:
            raise BundleError("variant has an unknown kind, format, or endianness") from exc
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "format", fmt)
        object.__setattr__(self, "endianness", endian)
        if fmt not in _KIND_FORMATS[kind]:
            raise BundleError(f"{kind.name} cannot use {fmt.name}")
        if not isinstance(self.payload, bytes):
            raise BundleError("payload must be immutable bytes")
        req = _features(self.required_features, "required_features")
        pro = _features(self.prohibited_features, "prohibited_features")
        if set(req) & set(pro):
            raise BundleError("required_features and prohibited_features overlap")
        object.__setattr__(self, "required_features", req)
        object.__setattr__(self, "prohibited_features", pro)
        if self.pointer_bits not in (0, 32, 64):
            raise BundleError("pointer_bits must be 0, 32, or 64")
        _u(self.e_machine, 32, "e_machine")
        _i32(self.priority, "priority")
        _u(self.provenance_digest, 64, "provenance_digest")
        _u(self.cal_gen, 64, "cal_gen")
        _sha(self.target_manifest_sha256, "target_manifest_sha256")
        for field in ("r12_attested", "executable", "portable", "debug"):
            if not isinstance(getattr(self, field), bool):
                raise BundleError(f"{field} must be boolean")
        if fmt in _NATIVE_FORMATS:
            if (self.pointer_bits == 0 or self.endianness == Endianness.NEUTRAL
                    or self.e_machine == 0):
                raise BundleError(
                    "native object variants require pointer_bits, endianness, and machine"
                )
        if kind in _NAMED_EXECUTABLE_KINDS and not self.executable:
            raise BundleError(f"{kind.name} variants must set the executable flag")
        _validate_payload(self)

    @property
    def payload_sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    @property
    def payload_crc32(self) -> int:
        return zlib.crc32(self.payload) & 0xFFFFFFFF

    @property
    def flags(self) -> int:
        return ((_FLAG_R12_ATTESTED if self.r12_attested else 0)
                | (_FLAG_EXECUTABLE if self.executable else 0)
                | (_FLAG_PORTABLE if self.portable else 0)
                | (_FLAG_DEBUG if self.debug else 0))


@dataclass(frozen=True)
class ArtifactBundle:
    variants: tuple[ArtifactVariant, ...]
    root_variant_id: str = ""
    default_variant_id: str = ""
    provenance_digest: int = 0
    generation: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.variants, tuple) or not 1 <= len(self.variants) <= MAX_ENTRIES:
            raise BundleError(f"variants must be a tuple with 1..{MAX_ENTRIES} entries")
        if any(not isinstance(v, ArtifactVariant) for v in self.variants):
            raise BundleError("every variant must be an ArtifactVariant")
        ids = [v.variant_id for v in self.variants]
        if len(ids) != len(set(ids)):
            raise BundleError("variant IDs must be unique")
        if ids != sorted(ids):
            raise BundleError("variants must be sorted by variant_id")
        _u(self.provenance_digest, 64, "bundle provenance_digest")
        _u(self.generation, 64, "bundle generation")
        for field in ("root_variant_id", "default_variant_id"):
            value = getattr(self, field)
            if value:
                _ascii(value, "variant_id", required=True)
                if value not in ids:
                    raise BundleError(f"{field} does not name a bundle variant")
        if self.root_variant_id:
            root = self.variant(self.root_variant_id)
            if root.kind != ArtifactKind.STREAM_PACK:
                raise BundleError("root_variant_id must name a StreamPack variant")
        if self.default_variant_id:
            default = self.variant(self.default_variant_id)
            if default.executable and not default.r12_attested:
                raise BundleError("an executable default variant must carry R12 attestation")

    def variant(self, variant_id: str) -> ArtifactVariant:
        for variant in self.variants:
            if variant.variant_id == variant_id:
                return variant
        raise KeyError(variant_id)

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(encode_bundle(self)).hexdigest()


@dataclass(frozen=True)
class CompatibilityEnvelope:
    triple: str = ""
    architecture: str = ""
    os_abi: str = ""
    channel: str = ""
    features: frozenset[str] = frozenset()
    accepted_kinds: frozenset[ArtifactKind] = frozenset()
    accepted_formats: frozenset[ArtifactFormat] = frozenset()
    endianness: Endianness = Endianness.NEUTRAL
    pointer_bits: int = 0
    e_machine: int = 0
    target_manifest_sha256: str = ""
    cal_gen: int | None = None
    require_r12: bool = True
    allow_debug: bool = False

    def __post_init__(self) -> None:
        for field in ("triple", "architecture", "os_abi", "channel"):
            _ascii(getattr(self, field), field)
        if not isinstance(self.features, (tuple, list, frozenset, set)):
            raise BundleError("compatibility features must be a feature sequence")
        if any(not isinstance(feature, str) for feature in self.features):
            raise BundleError("compatibility features contain a malformed feature name")
        feats = frozenset(_features(tuple(sorted(self.features)), "required_features"))
        object.__setattr__(self, "features", feats)
        try:
            kinds = frozenset(ArtifactKind(v) for v in self.accepted_kinds)
            formats = frozenset(ArtifactFormat(v) for v in self.accepted_formats)
            endian = Endianness(self.endianness)
        except (TypeError, ValueError) as exc:
            raise BundleError("compatibility envelope has an unknown enum") from exc
        object.__setattr__(self, "accepted_kinds", kinds)
        object.__setattr__(self, "accepted_formats", formats)
        object.__setattr__(self, "endianness", endian)
        if self.pointer_bits not in (0, 32, 64):
            raise BundleError("compatibility pointer_bits must be 0, 32, or 64")
        _u(self.e_machine, 32, "compatibility e_machine")
        _sha(self.target_manifest_sha256, "compatibility target_manifest_sha256")
        if self.cal_gen is not None:
            _u(self.cal_gen, 64, "compatibility cal_gen")
        if not isinstance(self.require_r12, bool) or not isinstance(self.allow_debug, bool):
            raise BundleError("compatibility policy fields must be boolean")


@dataclass(frozen=True)
class WireSpan:
    kind: str
    index: int | None
    name: str
    offset: int
    length: int

    @property
    def end(self) -> int:
        return self.offset + self.length


@dataclass(frozen=True)
class ArtifactBundleInspection:
    bundle: ArtifactBundle
    artifact_sha256: str
    embedded_sha256: str
    body_crc32: int
    header_crc32: int
    spans: tuple[WireSpan, ...]
    length: int


def _pack_entry(variant: ArtifactVariant, payload_offset: int) -> bytes:
    manifest = (bytes.fromhex(variant.target_manifest_sha256)
                if variant.target_manifest_sha256 else _ZERO_SHA)
    return _ENTRY.pack(
        int(variant.kind), int(variant.format), int(variant.endianness),
        variant.pointer_bits, variant.flags, variant.e_machine, variant.priority,
        payload_offset, len(variant.payload), variant.provenance_digest, variant.cal_gen,
        variant.payload_crc32, 0, bytes.fromhex(variant.payload_sha256), manifest,
        _fixed(variant.variant_id, "variant_id"), _fixed(variant.triple, "triple"),
        _fixed(variant.architecture, "architecture"), _fixed(variant.os_abi, "os_abi"),
        _fixed(variant.channel, "channel"), _fixed(variant.entry_symbol, "entry_symbol"),
        _fixed(",".join(variant.required_features), "required_features"),
        _fixed(",".join(variant.prohibited_features), "prohibited_features"),
    )


def _embedded_digest(data: bytes) -> bytes:
    """Hash BCAB with header-CRC and embedded-digest fields logically zeroed."""
    view = memoryview(data)
    digest = hashlib.sha256()
    digest.update(view[:_HEADER_CRC_OFFSET])
    digest.update(_ZERO_INTEGRITY_FIELDS)
    digest.update(view[_BUNDLE_SHA_END:])
    return digest.digest()


def encode_bundle(bundle: ArtifactBundle) -> bytes:
    if not isinstance(bundle, ArtifactBundle):
        raise BundleError("encode_bundle expects an ArtifactBundle")
    variants = bundle.variants
    directory_offset = HEADER_SIZE
    directory_size = len(variants) * ENTRY_SIZE
    payload_offset = _align8(directory_offset + directory_size)
    projected_size = payload_offset
    for variant in variants:
        projected_size = _align8(projected_size + len(variant.payload))
        if projected_size > MAX_BUNDLE_BYTES:
            raise BundleError(f"bundle exceeds the {MAX_BUNDLE_BYTES}-byte wire limit")
    cursor = payload_offset
    entries: list[bytes] = []
    payload_area = bytearray()
    for variant in variants:
        absolute = cursor
        entries.append(_pack_entry(variant, absolute))
        gap = absolute - (payload_offset + len(payload_area))
        payload_area += bytes(gap)
        payload_area += variant.payload
        cursor = _align8(absolute + len(variant.payload))
    payload_area += bytes(cursor - (payload_offset + len(payload_area)))
    directory = b"".join(entries)
    body = directory + bytes(payload_offset - directory_offset - len(directory)) + bytes(payload_area)
    file_size = HEADER_SIZE + len(body)
    if file_size != projected_size:
        raise BundleError("internal bundle size calculation mismatch")
    ids = [v.variant_id for v in variants]
    root = ids.index(bundle.root_variant_id) if bundle.root_variant_id else NO_INDEX
    default = ids.index(bundle.default_variant_id) if bundle.default_variant_id else NO_INDEX
    body_crc = zlib.crc32(body) & 0xFFFFFFFF
    base = _HEADER.pack(
        MAGIC, VERSION, HEADER_SIZE, 0, len(variants), ENTRY_SIZE, root, default,
        directory_offset, directory_size, payload_offset, file_size,
        bundle.provenance_digest, bundle.generation, body_crc, 0, _ZERO_SHA, bytes(12),
    )
    digest = hashlib.sha256()
    digest.update(base)
    digest.update(body)
    embedded = digest.digest()
    with_sha = _HEADER.pack(
        MAGIC, VERSION, HEADER_SIZE, 0, len(variants), ENTRY_SIZE, root, default,
        directory_offset, directory_size, payload_offset, file_size,
        bundle.provenance_digest, bundle.generation, body_crc, 0, embedded, bytes(12),
    )
    header_crc = zlib.crc32(with_sha) & 0xFFFFFFFF
    header = bytearray(with_sha)
    struct.pack_into("<I", header, _HEADER_CRC_OFFSET, header_crc)
    return bytes(header) + body


def _feature_csv(value: str, field: str) -> tuple[str, ...]:
    if not value:
        return ()
    return _features(tuple(value.split(",")), field)


def inspect_bundle(data: bytes) -> ArtifactBundleInspection:
    if not isinstance(data, bytes):
        raise BundleError("bundle input must be bytes")
    if not HEADER_SIZE <= len(data) <= MAX_BUNDLE_BYTES:
        raise BundleError("bundle size is outside the supported range")
    values = _HEADER.unpack(data[:HEADER_SIZE])
    (magic, version, header_size, flags, count, entry_size, root, default,
     directory_offset, directory_size, payload_offset, file_size,
     provenance, generation, body_crc, header_crc, embedded, reserved) = values
    if magic != MAGIC:
        raise BundleError(f"bad bundle magic {magic!r}")
    if version != VERSION:
        raise BundleError(f"unsupported bundle version {version}")
    if header_size != HEADER_SIZE or entry_size != ENTRY_SIZE:
        raise BundleError("bundle header or directory entry size is incompatible")
    if flags or any(reserved):
        raise BundleError("bundle header reserved fields must be zero")
    if not 1 <= count <= MAX_ENTRIES:
        raise BundleError("bundle entry count is outside the supported range")
    if file_size != len(data):
        raise BundleError("bundle file_size does not match the input length")
    if directory_offset != HEADER_SIZE or directory_size != count * ENTRY_SIZE:
        raise BundleError("bundle directory geometry is not canonical")
    if payload_offset != _align8(directory_offset + directory_size) or payload_offset > len(data):
        raise BundleError("bundle payload offset is not canonical")
    for name, index in (("root", root), ("default", default)):
        if index != NO_INDEX and index >= count:
            raise BundleError(f"bundle {name} index is out of range")
    header_copy = bytearray(data[:HEADER_SIZE])
    struct.pack_into("<I", header_copy, _HEADER_CRC_OFFSET, 0)
    if zlib.crc32(header_copy) & 0xFFFFFFFF != header_crc:
        raise BundleError("bundle header CRC mismatch")
    if zlib.crc32(data[HEADER_SIZE:]) & 0xFFFFFFFF != body_crc:
        raise BundleError("bundle body CRC mismatch")
    actual_embedded = _embedded_digest(data)
    if actual_embedded != embedded:
        raise BundleError("bundle embedded SHA-256 mismatch")
    directory_end = directory_offset + directory_size
    if any(data[directory_end:payload_offset]):
        raise BundleError("bundle directory padding must be zero")

    variants: list[ArtifactVariant] = []
    spans = [WireSpan("header", None, "header", 0, HEADER_SIZE)]
    previous_id = ""
    previous_end = payload_offset
    payload_spans: list[WireSpan] = []
    padding_spans: list[WireSpan] = []
    for index in range(count):
        offset = directory_offset + index * ENTRY_SIZE
        raw = _ENTRY.unpack(data[offset:offset + ENTRY_SIZE])
        (kind_raw, format_raw, endian_raw, pointer_bits, entry_flags, e_machine,
         priority, poff, psize, pdigest, cal_gen, pcrc, entry_reserved,
         psha, msha, vid_raw, triple_raw, arch_raw, os_raw, channel_raw,
         symbol_raw, req_raw, pro_raw) = raw
        if entry_reserved or entry_flags & ~_ENTRY_FLAG_MASK:
            raise BundleError(f"directory entry {index} has unknown flags or reserved data")
        try:
            kind = ArtifactKind(kind_raw)
            fmt = ArtifactFormat(format_raw)
            endian = Endianness(endian_raw)
        except ValueError as exc:
            raise BundleError(f"directory entry {index} has an unknown enum") from exc
        vid = _unfixed(vid_raw, "variant_id")
        if previous_id and vid <= previous_id:
            raise BundleError("bundle directory is not strictly sorted by variant_id")
        previous_id = vid
        triple = _unfixed(triple_raw, "triple")
        arch = _unfixed(arch_raw, "architecture")
        os_abi = _unfixed(os_raw, "os_abi")
        channel = _unfixed(channel_raw, "channel")
        symbol = _unfixed(symbol_raw, "entry_symbol")
        req = _feature_csv(_unfixed(req_raw, "required_features"), "required_features")
        pro = _feature_csv(_unfixed(pro_raw, "prohibited_features"), "prohibited_features")
        if poff != _align8(previous_end) or poff % 8 or psize == 0 or poff + psize > len(data):
            raise BundleError(f"directory entry {index} has non-canonical payload geometry")
        if any(data[previous_end:poff]):
            raise BundleError("bundle payload alignment padding must be zero")
        if poff > previous_end:
            padding_spans.append(
                WireSpan("padding", None, f"before:{vid}", previous_end, poff - previous_end)
            )
        payload = data[poff:poff + psize]
        previous_end = poff + psize
        if zlib.crc32(payload) & 0xFFFFFFFF != pcrc:
            raise BundleError(f"variant {vid!r} payload CRC mismatch")
        if hashlib.sha256(payload).digest() != psha:
            raise BundleError(f"variant {vid!r} payload SHA-256 mismatch")
        manifest = "" if msha == _ZERO_SHA else msha.hex()
        variant = ArtifactVariant(
            vid, kind, fmt, payload, triple, arch, os_abi, channel, symbol,
            req, pro, endian, pointer_bits, e_machine, priority, pdigest,
            manifest, cal_gen,
            bool(entry_flags & _FLAG_R12_ATTESTED),
            bool(entry_flags & _FLAG_EXECUTABLE),
            bool(entry_flags & _FLAG_PORTABLE),
            bool(entry_flags & _FLAG_DEBUG),
        )
        variants.append(variant)
        spans.append(WireSpan("directory", index, vid, offset, ENTRY_SIZE))
        payload_spans.append(WireSpan("payload", index, vid, poff, psize))
    final_end = _align8(previous_end)
    if final_end != len(data) or any(data[previous_end:final_end]):
        raise BundleError("bundle has trailing bytes or nonzero final padding")
    root_id = variants[root].variant_id if root != NO_INDEX else ""
    default_id = variants[default].variant_id if default != NO_INDEX else ""
    bundle = ArtifactBundle(tuple(variants), root_id, default_id, provenance, generation)
    spans.extend(payload_spans)
    spans.extend(padding_spans)
    if final_end > previous_end:
        spans.append(WireSpan("padding", None, "final", previous_end, final_end - previous_end))
    return ArtifactBundleInspection(
        bundle, hashlib.sha256(data).hexdigest(), embedded.hex(), body_crc,
        header_crc, tuple(spans), len(data))


def decode_bundle(data: bytes) -> ArtifactBundle:
    return inspect_bundle(data).bundle


def write_bundle(path: str | os.PathLike[str], bundle: ArtifactBundle) -> None:
    data = encode_bundle(bundle)
    target = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(target)) or "."
    fd, temporary = tempfile.mkstemp(prefix=os.path.basename(target) + ".", suffix=".tmp",
                                     dir=directory)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_bundle(path: str | os.PathLike[str], *, max_bytes: int = MAX_BUNDLE_BYTES) -> ArtifactBundle:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not HEADER_SIZE <= max_bytes <= MAX_BUNDLE_BYTES:
        raise BundleError(f"max_bytes must be in [{HEADER_SIZE}, {MAX_BUNDLE_BYTES}]")
    target = os.fspath(path)
    try:
        with open(target, "rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            if size > max_bytes:
                raise BundleError(f"bundle exceeds max_bytes={max_bytes}")
            data = stream.read(max_bytes + 1)
    except OSError as exc:
        raise BundleError(f"cannot read bundle {target!r}: {exc}") from exc
    if len(data) > max_bytes or len(data) != size:
        raise BundleError("bundle changed, grew, or was truncated while reading")
    return decode_bundle(data)


def is_compatible(variant: ArtifactVariant, envelope: CompatibilityEnvelope) -> bool:
    if envelope.accepted_kinds and variant.kind not in envelope.accepted_kinds:
        return False
    if envelope.accepted_formats and variant.format not in envelope.accepted_formats:
        return False
    for field in ("triple", "architecture", "os_abi", "channel"):
        required = getattr(variant, field)
        available = getattr(envelope, field)
        if required and required != available:
            return False
    if variant.endianness != Endianness.NEUTRAL and variant.endianness != envelope.endianness:
        return False
    if variant.pointer_bits and variant.pointer_bits != envelope.pointer_bits:
        return False
    if variant.e_machine and variant.e_machine != envelope.e_machine:
        return False
    if not set(variant.required_features).issubset(envelope.features):
        return False
    if set(variant.prohibited_features) & envelope.features:
        return False
    if variant.target_manifest_sha256:
        if variant.target_manifest_sha256 != envelope.target_manifest_sha256:
            return False
    if variant.cal_gen and variant.cal_gen != envelope.cal_gen:
        return False
    if envelope.require_r12 and variant.executable and not variant.r12_attested:
        return False
    if variant.debug and not envelope.allow_debug:
        return False
    return True


def _specificity(variant: ArtifactVariant) -> int:
    fields = (variant.triple, variant.architecture, variant.os_abi, variant.channel,
              variant.entry_symbol, variant.target_manifest_sha256)
    return (sum(bool(v) for v in fields) + bool(variant.pointer_bits)
            + bool(variant.e_machine) + (variant.endianness != Endianness.NEUTRAL)
            + bool(variant.cal_gen))


def select_variant(bundle: ArtifactBundle, envelope: CompatibilityEnvelope | None = None,
                   *, variant_id: str = "") -> ArtifactVariant:
    if not isinstance(bundle, ArtifactBundle):
        raise BundleError("select_variant expects an ArtifactBundle")
    if variant_id:
        _ascii(variant_id, "variant_id", required=True)
        try:
            selected = bundle.variant(variant_id)
        except KeyError as exc:
            raise BundleError(f"bundle has no variant {variant_id!r}") from exc
        if envelope is not None and not is_compatible(selected, envelope):
            raise BundleError(f"variant {variant_id!r} is incompatible with the requested envelope")
        return selected
    if envelope is None:
        if not bundle.default_variant_id:
            raise BundleError("bundle has no default variant")
        return bundle.variant(bundle.default_variant_id)
    compatible = [v for v in bundle.variants if is_compatible(v, envelope)]
    if not compatible:
        raise BundleError("bundle has no compatible artifact variant")
    compatible.sort(key=lambda v: (-v.priority, -_specificity(v),
                                   -len(v.required_features), v.variant_id))
    return compatible[0]


def host_envelope(*, features=(), channel: str = "host", require_r12: bool = True) -> CompatibilityEnvelope:
    machine = platform.machine().lower()
    system = platform.system().lower()
    bits = struct.calcsize("P") * 8
    if machine in ("amd64", "x86_64"):
        arch = "x86_64"
    elif machine in ("arm64", "aarch64"):
        arch = "aarch64"
    elif machine.startswith("riscv"):
        arch = "riscv64"
    elif machine in ("i386", "i686", "x86"):
        arch = "i386"
    else:
        arch = machine
    if system == "windows":
        triple, os_abi = f"{arch}-pc-windows-msvc", "windows-msvc"
        native_formats = (ArtifactFormat.COFF, ArtifactFormat.PE)
        native_machine = {"i386": 0x014C, "x86_64": 0x8664, "aarch64": 0xAA64}
    elif system == "darwin":
        triple, os_abi = f"{arch}-apple-darwin", "darwin"
        native_formats = (ArtifactFormat.MACHO,)
        native_machine = {
            "i386": 7, "x86_64": 0x01000007,
            "aarch64": 0x0100000C,
        }
    else:
        triple, os_abi = f"{arch}-unknown-linux-gnu", "linux-gnu"
        native_formats = (ArtifactFormat.ELF,)
        native_machine = {"i386": 3, "arm": 40, "x86_64": 62,
                          "aarch64": 183, "riscv64": 243}
    return CompatibilityEnvelope(
        triple=triple, architecture=arch, os_abi=os_abi, channel=channel,
        features=frozenset(features),
        accepted_formats=frozenset((*native_formats, ArtifactFormat.WASM,
                                    ArtifactFormat.TEXT, ArtifactFormat.STREAM_PACK,
                                    ArtifactFormat.LLVM_BITCODE, ArtifactFormat.JVM_CLASS,
                                    ArtifactFormat.SPIRV)),
        endianness=(Endianness.LITTLE if struct.pack("=I", 1)[0] == 1 else Endianness.BIG),
        pointer_bits=bits, e_machine=native_machine.get(arch, 0),
        require_r12=require_r12,
    )


def compatibility_sha256(envelope: CompatibilityEnvelope) -> str:
    """Content address for a canonical selector envelope (not a legality proof)."""
    if not isinstance(envelope, CompatibilityEnvelope):
        raise BundleError("compatibility_sha256 expects a CompatibilityEnvelope")
    record = {
        "accepted_formats": sorted(int(value) for value in envelope.accepted_formats),
        "accepted_kinds": sorted(int(value) for value in envelope.accepted_kinds),
        "allow_debug": envelope.allow_debug,
        "architecture": envelope.architecture,
        "cal_gen": envelope.cal_gen,
        "channel": envelope.channel,
        "e_machine": envelope.e_machine,
        "endianness": int(envelope.endianness),
        "features": sorted(envelope.features),
        "os_abi": envelope.os_abi,
        "pointer_bits": envelope.pointer_bits,
        "require_r12": envelope.require_r12,
        "target_manifest_sha256": envelope.target_manifest_sha256,
        "triple": envelope.triple,
    }
    wire = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(wire).hexdigest()


__all__ = [
    "MAGIC", "VERSION", "HEADER_SIZE", "ENTRY_SIZE", "MAX_ENTRIES",
    "MAX_BUNDLE_BYTES", "NO_INDEX", "BundleError", "ArtifactKind",
    "ArtifactFormat", "Endianness", "ArtifactVariant", "ArtifactBundle",
    "CompatibilityEnvelope", "WireSpan", "ArtifactBundleInspection",
    "encode_bundle", "decode_bundle", "inspect_bundle", "write_bundle",
    "read_bundle", "is_compatible", "select_variant", "host_envelope",
    "compatibility_sha256",
]
