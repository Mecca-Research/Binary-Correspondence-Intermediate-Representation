"""ControlRecordV1 binary ABI (frozen v1) -- reference codec.

The control plane as bytes (G14, staged plan S3-A). Layout (little-endian; every field
fixed-width; the normative spec is docs/kernel/BCIR_CONTROL_PLANE_ABI.md and the C view
runtime/c/bcir_control_plane.h):

    Header (64 bytes, one cache line; every u64 at an 8-aligned offset):
      magic[4]="BCTL"  version:u16  flags:u16
      kind:u8 @8  scope:u8 @9  reason:u8 @10  reserved0:u8 @11  body_len:u32 @12
      generation:u32 @16  expect:u32 @20  capability:u64 @24  boundary:u64 @32
      sequence:u64 @40  lease:u64 @48  subject:u64 @56
    Body (fixed per kind):
      1 lease       lease_id:u64 granted:u64 issued_epoch:u64 expiry_epoch:u64 holder:u64  (40)
      2 generation  map_gen:u32 data_gen:u32 topo_gen:u32 reserved:u32 registry_digest[32] (48)
      3 quiesce     drain_deadline:u64                                                    (8)
      4 activate    artifact_sha256[32] previous_sha256[32]                              (64)
      5 rollback    restore_sha256[32] rollback_token[32]                                (64)
      6 cancel      first_sequence:u64 last_sequence:u64                                 (16)
    Trailer:
      mac[32]    HMAC-SHA256(key, header || body): the root key for a lease grant (lease 0),
                 otherwise lease_key(root, lease)
      crc32:u32  CRC-32 of every preceding byte

The wire laws are one predicate, `validate_control`, applied by `encode_control` AND
`decode_control` and by the C twin (`bcir_ctl_verify`) identically, in the specification's
order so the two rails name the same first violation. Every refusal is a `ControlError`
(an `AbiError`) whose `status` is the C twin's `bcir_status` name for the same bytes.
The format is frozen at v1 and evolves append-only (`control_version` returns the lowest
carrying version -- v1 for every record today).
"""

from __future__ import annotations

import hashlib
import hmac
import struct
import zlib

from ..gem.control import (
    BODY_TYPES,
    CAP_GRANTABLE,
    CAPABILITY,
    CONTROL_KINDS,
    CONTROL_SCOPES,
    REASONS,
    SWITCH_KINDS,
    ZERO32,
    Activate,
    Cancel,
    ControlRecord,
    GenerationSwitch,
    LeaseGrant,
    Quiesce,
    Rollback,
)
from .streampack_abi import AbiError

CONTROL_MAGIC = b"BCTL"
CONTROL_VERSION = 1
CONTROL_VERSION_MAX = 1
CONTROL_HEADER_SIZE = 64
CONTROL_MAC_SIZE = 32
CONTROL_TRAILER_SIZE = CONTROL_MAC_SIZE + 4
#: The declared bound on any record (the largest v1 record is 164 bytes).
CONTROL_RECORD_MAX_BYTES = 192
#: The body length of each kind, per version (append-only: a later version may grow a tail).
CONTROL_BODY_BYTES = {
    1: {
        "lease": 40,
        "generation": 48,
        "quiesce": 8,
        "activate": 64,
        "rollback": 64,
        "cancel": 16,
    }
}

# magic version flags kind scope reason reserved0 body_len generation expect
# capability boundary sequence lease subject -> exactly the 64-byte line.
_HEADER = struct.Struct("<4sHHBBBBIIIQQQQQ")
assert _HEADER.size == CONTROL_HEADER_SIZE
_BODY = {
    "lease": struct.Struct("<QQQQQ"),
    "generation": struct.Struct("<IIII32s"),
    "quiesce": struct.Struct("<Q"),
    "activate": struct.Struct("<32s32s"),
    "rollback": struct.Struct("<32s32s"),
    "cancel": struct.Struct("<QQ"),
}
assert all(_BODY[k].size == n for k, n in CONTROL_BODY_BYTES[1].items())
_U32_MAX = (1 << 32) - 1


class ControlError(AbiError):
    """A control record the wire laws refuse. `status` is the C twin's `bcir_status` name for
    the same bytes (`BCIR_ERR_TRUNCATED`, `_MAGIC`, `_VERSION`, `_RESERVED`, `_CONTROL`,
    `_TRAILING`, `_CRC`, `_MAC`)."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(f"{status}: {message}")
        self.status = status


def _uint(name: str, value, bits: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < (1 << bits):
        raise ControlError(
            "BCIR_ERR_CONTROL", f"{name} must be an unsigned {bits}-bit integer, got {value!r}"
        )
    return value


def _digest(name: str, value, *, nonzero: bool = True) -> bytes:
    if not isinstance(value, (bytes, bytearray)) or len(value) != 32:
        raise ControlError("BCIR_ERR_CONTROL", f"{name} must be exactly 32 bytes")
    if nonzero and not any(value):
        raise ControlError("BCIR_ERR_CONTROL", f"{name} must not be all zero")
    return bytes(value)


def validate_control(record: ControlRecord) -> None:
    """The field laws (the specification's laws 9-11), shared by `encode_control` and
    `decode_control`. The framing laws (1-8) belong to the bytes and are checked by the
    decoder before this runs; the decoder runs these three parts itself, in order, so the
    generation body's reserved word is judged at its place among the body laws."""
    _header_laws(record)
    _body_laws(record)
    _mac_law(record)


def _header_laws(record: ControlRecord) -> None:
    if record.kind not in CONTROL_KINDS:
        raise ControlError("BCIR_ERR_CONTROL", f"unknown kind {record.kind!r}")
    if not isinstance(record.body, BODY_TYPES[record.kind]):
        raise ControlError(
            "BCIR_ERR_CONTROL",
            f"a {record.kind} record needs a {BODY_TYPES[record.kind].__name__} body",
        )
    # law 9: header fields
    if record.scope not in CONTROL_SCOPES:
        raise ControlError("BCIR_ERR_CONTROL", f"unknown scope {record.scope!r}")
    if record.reason not in REASONS[record.kind]:
        raise ControlError(
            "BCIR_ERR_CONTROL",
            f"reason {record.reason!r} is not one of {record.kind}'s {REASONS[record.kind]}",
        )
    generation = _uint("generation", record.generation, 32)
    expect = _uint("expect", record.expect, 32)
    _uint("boundary", record.boundary, 64)
    sequence = _uint("sequence", record.sequence, 64)
    lease = _uint("lease", record.lease, 64)
    _uint("subject", record.subject, 64)
    if sequence == 0:
        raise ControlError("BCIR_ERR_CONTROL", "sequence must be at least 1")
    if (lease == 0) != (record.kind == "lease"):
        raise ControlError(
            "BCIR_ERR_CONTROL", "lease must be 0 exactly on a lease grant (and only there)"
        )
    if record.kind in SWITCH_KINDS:
        if expect >= _U32_MAX or generation != expect + 1:
            raise ControlError(
                "BCIR_ERR_CONTROL",
                f"a switch asserts expect + 1 without wrapping, got {expect} -> {generation}",
            )
    elif generation != expect:
        raise ControlError(
            "BCIR_ERR_CONTROL", f"a {record.kind} record asserts the generation it witnessed"
        )


def _body_laws(record: ControlRecord) -> None:
    """Law 10, in field order (the generation body's reserved word, which the abstract value
    does not carry, is judged by the decoder immediately before this)."""
    body = record.body
    if record.kind == "lease":
        if _uint("lease_id", body.lease_id, 64) == 0:
            raise ControlError("BCIR_ERR_CONTROL", "a lease grant names lease 0")
        granted = _uint("granted", body.granted, 64)
        if granted == 0 or granted & ~CAP_GRANTABLE:
            raise ControlError(
                "BCIR_ERR_CONTROL",
                f"granted 0x{granted:x} must be a nonempty subset of 0x{CAP_GRANTABLE:x} "
                "(no bits outside v1, no delegation)",
            )
        issued = _uint("issued_epoch", body.issued_epoch, 64)
        if _uint("expiry_epoch", body.expiry_epoch, 64) <= issued:
            raise ControlError("BCIR_ERR_CONTROL", "a lease's window is empty or reversed")
        if _uint("holder", body.holder, 64) == 0:
            raise ControlError("BCIR_ERR_CONTROL", "a lease names no holder")
    elif record.kind == "generation":
        _uint("map_gen", body.map_gen, 32)
        _uint("data_gen", body.data_gen, 32)
        _uint("topo_gen", body.topo_gen, 32)
        _digest("registry_digest", body.registry_digest)
    elif record.kind == "quiesce":
        if _uint("drain_deadline", body.drain_deadline, 64) < record.boundary:
            raise ControlError("BCIR_ERR_CONTROL", "a drain deadline before the record's boundary")
    elif record.kind == "activate":
        artifact = _digest("artifact_sha256", body.artifact_sha256)
        if artifact == _digest("previous_sha256", body.previous_sha256, nonzero=False):
            raise ControlError("BCIR_ERR_CONTROL", "an activation must change the artifact")
    elif record.kind == "rollback":
        _digest("restore_sha256", body.restore_sha256)
        _digest("rollback_token", body.rollback_token)
    else:  # cancel
        first = _uint("first_sequence", body.first_sequence, 64)
        last = _uint("last_sequence", body.last_sequence, 64)
        sequence = record.sequence
        if not 1 <= first <= last < sequence:
            raise ControlError(
                "BCIR_ERR_CONTROL",
                f"a cancel range [{first}, {last}] must be nonempty, from 1, and before "
                f"its own sequence {sequence}",
            )


def _mac_law(record: ControlRecord) -> None:
    """Law 11: the MAC is present (whether it is the right one needs the key)."""
    if record.mac:
        if not isinstance(record.mac, (bytes, bytearray)) or len(record.mac) != CONTROL_MAC_SIZE:
            raise ControlError("BCIR_ERR_MAC", "a MAC is exactly 32 bytes")
        if not any(record.mac):
            raise ControlError("BCIR_ERR_MAC", "an all-zero MAC")


def control_version(record: ControlRecord) -> int:
    """The lowest wire version that carries `record` (v1 for every record today)."""
    return CONTROL_VERSION


def _body_bytes(record: ControlRecord) -> bytes:
    b = record.body
    fmt = _BODY[record.kind]
    if record.kind == "lease":
        return fmt.pack(b.lease_id, b.granted, b.issued_epoch, b.expiry_epoch, b.holder)
    if record.kind == "generation":
        return fmt.pack(b.map_gen, b.data_gen, b.topo_gen, 0, bytes(b.registry_digest))
    if record.kind == "quiesce":
        return fmt.pack(b.drain_deadline)
    if record.kind == "activate":
        return fmt.pack(bytes(b.artifact_sha256), bytes(b.previous_sha256))
    if record.kind == "rollback":
        return fmt.pack(bytes(b.restore_sha256), bytes(b.rollback_token))
    return fmt.pack(b.first_sequence, b.last_sequence)


def _signed(record: ControlRecord) -> bytes:
    version = control_version(record)
    header = _HEADER.pack(
        CONTROL_MAGIC,
        version,
        0,
        CONTROL_KINDS.index(record.kind) + 1,
        CONTROL_SCOPES.index(record.scope),
        REASONS[record.kind].index(record.reason),
        0,
        CONTROL_BODY_BYTES[version][record.kind],
        record.generation,
        record.expect,
        CAPABILITY[record.kind],
        record.boundary,
        record.sequence,
        record.lease,
        record.subject,
    )
    return header + _body_bytes(record)


def control_mac(key: bytes, message: bytes) -> bytes:
    """HMAC-SHA256 -- the C twin's `bcir_hmac_sha256`."""
    return hmac.new(bytes(key), bytes(message), hashlib.sha256).digest()


def sign_control(record: ControlRecord, key: bytes) -> ControlRecord:
    """The record with its MAC under `key` (the root key for a lease grant, the issuer's lease
    key otherwise). Refuses a record the field laws reject."""
    from dataclasses import replace

    validate_control(replace(record, mac=b""))
    return replace(record, mac=control_mac(key, _signed(record)))


def encode_control(record: ControlRecord) -> bytes:
    """Serialize a signed record (header, body, MAC, CRC); refuses one the wire laws reject."""
    if not record.mac:
        raise ControlError("BCIR_ERR_MAC", "an unsigned record (sign_control first)")
    validate_control(record)
    body = _signed(record) + bytes(record.mac)
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


def issue_control(record: ControlRecord, key: bytes) -> bytes:
    """Sign and encode in one step: what an issuer sends."""
    return encode_control(sign_control(record, key))


def signed_part(data: bytes) -> bytes:
    """The MAC's input (header || body) of an already-framed record."""
    return bytes(data[: len(data) - CONTROL_TRAILER_SIZE])


def check_control_mac(data: bytes, key: bytes) -> bool:
    """Whether a framed record's MAC is the one `key` produces (constant time). The keyless
    framing is not re-checked here: decode first."""
    data = bytes(data)
    if len(data) < CONTROL_HEADER_SIZE + CONTROL_TRAILER_SIZE:
        return False
    mac = data[-CONTROL_TRAILER_SIZE:-4]
    return hmac.compare_digest(control_mac(key, signed_part(data)), mac)


def decode_control(data: bytes) -> ControlRecord:
    """Parse a v1 record (the framing laws 1-8, then `validate_control`'s field laws 9-11).
    The MAC is carried, not checked: that needs the key (`check_control_mac`)."""
    data = bytes(data)
    if len(data) < CONTROL_HEADER_SIZE + CONTROL_TRAILER_SIZE:
        raise ControlError("BCIR_ERR_TRUNCATED", "buffer too small for a control record")
    (
        magic,
        version,
        flags,
        kind_code,
        scope_code,
        reason_code,
        reserved0,
        body_len,
        generation,
        expect,
        capability,
        boundary,
        sequence,
        lease,
        subject,
    ) = _HEADER.unpack(data[:CONTROL_HEADER_SIZE])
    if magic != CONTROL_MAGIC:
        raise ControlError("BCIR_ERR_MAGIC", f"bad magic {magic!r} (expected {CONTROL_MAGIC!r})")
    if not CONTROL_VERSION <= version <= CONTROL_VERSION_MAX:
        raise ControlError(
            "BCIR_ERR_VERSION",
            f"unsupported control record version {version} "
            f"(this reader handles v{CONTROL_VERSION}..v{CONTROL_VERSION_MAX})",
        )
    if flags or reserved0:
        raise ControlError("BCIR_ERR_RESERVED", "reserved control header bytes must be zero")
    if not 1 <= kind_code <= len(CONTROL_KINDS):
        raise ControlError("BCIR_ERR_CONTROL", f"unknown kind code {kind_code}")
    kind = CONTROL_KINDS[kind_code - 1]
    if body_len != CONTROL_BODY_BYTES[version][kind]:
        raise ControlError(
            "BCIR_ERR_CONTROL",
            f"a {kind} body is {CONTROL_BODY_BYTES[version][kind]} bytes, header says {body_len}",
        )
    total = CONTROL_HEADER_SIZE + body_len + CONTROL_TRAILER_SIZE
    if len(data) < total:
        raise ControlError("BCIR_ERR_TRUNCATED", f"a {kind} record is {total} bytes")
    if len(data) > total:
        raise ControlError(
            "BCIR_ERR_TRAILING", f"{len(data) - total} byte(s) after a {total}-byte {kind} record"
        )
    if (zlib.crc32(data[:-4]) & 0xFFFFFFFF) != struct.unpack("<I", data[-4:])[0]:
        raise ControlError("BCIR_ERR_CRC", "CRC mismatch (corrupt control record)")
    if scope_code >= len(CONTROL_SCOPES):
        raise ControlError("BCIR_ERR_CONTROL", f"unknown scope code {scope_code}")
    if reason_code >= len(REASONS[kind]):
        raise ControlError("BCIR_ERR_CONTROL", f"reason code {reason_code} is not one of {kind}'s")
    if capability != CAPABILITY[kind]:
        raise ControlError(
            "BCIR_ERR_CONTROL",
            f"a {kind} record exercises capability 0x{CAPABILITY[kind]:x}, not 0x{capability:x}",
        )
    raw = data[CONTROL_HEADER_SIZE : CONTROL_HEADER_SIZE + body_len]
    fields = _BODY[kind].unpack(raw)
    if kind == "lease":
        body = LeaseGrant(*fields)
    elif kind == "generation":
        body = GenerationSwitch(fields[0], fields[1], fields[2], fields[4])
    elif kind == "quiesce":
        body = Quiesce(*fields)
    elif kind == "activate":
        body = Activate(*fields)
    elif kind == "rollback":
        body = Rollback(*fields)
    else:
        body = Cancel(*fields)
    record = ControlRecord(
        kind=kind,
        scope=CONTROL_SCOPES[scope_code],
        subject=subject,
        generation=generation,
        expect=expect,
        boundary=boundary,
        sequence=sequence,
        lease=lease,
        body=body,
        reason=REASONS[kind][reason_code],
        mac=data[-CONTROL_TRAILER_SIZE:-4],
    )
    _header_laws(record)
    if kind == "generation" and fields[3]:
        raise ControlError("BCIR_ERR_RESERVED", "the generation body's reserved word is nonzero")
    _body_laws(record)
    _mac_law(record)
    return record


__all__ = [
    "CONTROL_BODY_BYTES",
    "CONTROL_HEADER_SIZE",
    "CONTROL_MAC_SIZE",
    "CONTROL_MAGIC",
    "CONTROL_RECORD_MAX_BYTES",
    "CONTROL_TRAILER_SIZE",
    "CONTROL_VERSION",
    "CONTROL_VERSION_MAX",
    "ControlError",
    "check_control_mac",
    "control_mac",
    "control_version",
    "decode_control",
    "encode_control",
    "issue_control",
    "sign_control",
    "signed_part",
    "validate_control",
    "ZERO32",
]
