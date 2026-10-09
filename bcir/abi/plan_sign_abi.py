"""PlanStatementV1 and TrustStoreV1 -- the plan bound by SHA-256 and signed (GEM+ G21).

The 2026-10-06 audit (item 6) found the ExecutionPlan carrying a CRC-32, naming its module and
target by 63-bit FNV-1a R13 hashes, and its authority resting on a shared-key HMAC over the
control record -- "not a signature scheme", in the control plane's own words. This module
is the signature scheme: a **detached statement** that binds a plan's bytes, by SHA-256, to
the scope, the module, the pack and the certificate it was planned under, signed with Ed25519
(`bcir.abi.ed25519`, RFC 8032), and a **trust store** that says which keys may sign, when, and
which have been revoked.

Why detached, and not an ExecutionPlan v4 record: the plan's versions are cumulative -- a v4
plan would be a v3 plan, and v3's laws require a nonzero source/spec binding that a plan that
moves nothing cannot have. The statement binds the plan's exact bytes instead, so it signs every
plan version as it is, and the frozen plan ABI does not move.

Wire format (little-endian; docs/kernel/BCIR_PLAN_SIGNATURE_ABI.md is the prose spec, the C
twin is runtime/c/bcir_plan_sign.h):

    PlanStatementV1 (272 bytes)
      magic[4]="BPSG"  version:u16=1  flags:u16=0  signed_at:u64 (seconds since the epoch)
      key_id[32]       SHA-256 of the signer's Ed25519 public key
      plan[32]         SHA-256 of the ExecutionPlan bytes          (never zero)
      scope[32]        SHA-256 of the ExecutionScopeV1             (zero: not bound)
      module[32]       SHA-256 of the scope's program component P  (zero: not bound)
      pack[32]         SHA-256 of the StreamPack bytes             (zero: not bound)
      certificate[32]  SHA-256 of the certificate's JSON           (zero: not bound)
      signature[64]    Ed25519 over bytes 0..207

    TrustStoreV1 (8 + 56 * n bytes)
      magic[4]="BTRS"  version:u16=1  count:u16
      count x { public_key[32]  not_before:u64  not_after:u64  flags:u32 (bit 0 revoked)
                reserved:u32=0 }        -- public keys strictly ascending, so a store has
                                           one spelling and a key appears once

The verdict is total over untrusted bytes: `verify_statement` returns the one law a statement
breaks (a `PlanSignError` names it) or the trusted key that signed it. A key signs only inside
its window [not_before, not_after); a revoked key signs nothing, whatever date a statement
claims (a compromised key can write any date); a statement dated after `now` is refused.
Rotation is two keys whose windows meet: the store accepts each inside its own window.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field, replace

from . import ed25519

STATEMENT_MAGIC = b"BPSG"
STATEMENT_VERSION = 1
STATEMENT_SIZE = 272
SIGNED_SIZE = 208
STORE_MAGIC = b"BTRS"
STORE_VERSION = 1
STORE_ENTRY_SIZE = 56
KEY_REVOKED = 1
KEY_FLAGS = KEY_REVOKED
_ZERO = bytes(32)

_HEADER = struct.Struct("<4sHHQ")  # magic, version, flags, signed_at
_ENTRY = struct.Struct("<32sQQII")  # public key, not_before, not_after, flags, reserved


class PlanSignError(ValueError):
    """A statement or a trust store that breaks a law of the wire format or of trust. `code`
    names the law, as the C twin's verdict does (`bcir_ps_verdict_name`): statement, store,
    plan, pack, unknown-key, revoked, future, window or signature."""

    def __init__(self, message: str, code: str = "statement"):
        super().__init__(message)
        self.code = code


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def key_id(public_key: bytes) -> bytes:
    """A key's identity: the SHA-256 of its 32-byte Ed25519 public key."""
    return sha256(public_key)


@dataclass(frozen=True)
class PlanStatement:
    signed_at: int
    key_id: bytes
    plan: bytes
    scope: bytes = _ZERO
    module: bytes = _ZERO
    pack: bytes = _ZERO
    certificate: bytes = _ZERO
    signature: bytes = bytes(64)

    def signed_bytes(self) -> bytes:
        return (
            _HEADER.pack(STATEMENT_MAGIC, STATEMENT_VERSION, 0, self.signed_at)
            + self.key_id
            + self.plan
            + self.scope
            + self.module
            + self.pack
            + self.certificate
        )

    def encode(self) -> bytes:
        _check_statement(self)
        return self.signed_bytes() + self.signature


@dataclass(frozen=True)
class TrustedKey:
    public_key: bytes
    not_before: int
    not_after: int
    revoked: bool = False

    @property
    def key_id(self) -> bytes:
        return key_id(self.public_key)


@dataclass(frozen=True)
class TrustStore:
    keys: tuple[TrustedKey, ...] = field(default_factory=tuple)

    def encode(self) -> bytes:
        _check_store(self)
        out = bytearray(struct.pack("<4sHH", STORE_MAGIC, STORE_VERSION, len(self.keys)))
        for key in self.keys:
            out += _ENTRY.pack(
                key.public_key, key.not_before, key.not_after, KEY_REVOKED if key.revoked else 0, 0
            )
        return bytes(out)

    def find(self, wanted: bytes) -> TrustedKey | None:
        for key in self.keys:
            if key.key_id == wanted:
                return key
        return None


def _u64(name: str, value, code: str = "statement") -> int:
    if type(value) is not int or not 0 <= value < 1 << 64:
        raise PlanSignError(f"{name} must be a u64, got {value!r}", code)
    return value


def _digest_field(name: str, value) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise PlanSignError(f"{name} must be a 32-byte SHA-256 digest")
    return value


def _check_statement(st: PlanStatement) -> None:
    _u64("signed_at", st.signed_at)
    for name in ("key_id", "plan", "scope", "module", "pack", "certificate"):
        _digest_field(name, getattr(st, name))
    if st.plan == _ZERO:
        raise PlanSignError("a statement binds a plan: its plan digest is never zero")
    if not isinstance(st.signature, bytes) or len(st.signature) != 64:
        raise PlanSignError("the signature is 64 bytes")


def _check_store(store: TrustStore) -> None:
    if len(store.keys) > 0xFFFF:
        raise PlanSignError("a trust store holds at most 65535 keys", "store")
    previous = None
    for key in store.keys:
        if not isinstance(key.public_key, bytes) or len(key.public_key) != 32:
            raise PlanSignError("a trusted public key is 32 bytes", "store")
        if not ed25519.usable_key(key.public_key):
            raise PlanSignError(
                "a trusted key is the canonical encoding of a curve point not of small order",
                "store",
            )
        if previous is not None and key.public_key <= previous:
            raise PlanSignError(
                "trusted keys are strictly ascending (one spelling, no duplicate)", "store"
            )
        previous = key.public_key
        _u64("not_before", key.not_before, "store")
        _u64("not_after", key.not_after, "store")
        if key.not_after <= key.not_before:
            raise PlanSignError("a key's window [not_before, not_after) is empty", "store")


def decode_statement(data: bytes) -> PlanStatement:
    if len(data) != STATEMENT_SIZE:
        raise PlanSignError(f"a PlanStatementV1 is {STATEMENT_SIZE} bytes, got {len(data)}")
    magic, version, flags, signed_at = _HEADER.unpack_from(data, 0)
    if magic != STATEMENT_MAGIC:
        raise PlanSignError(f"bad magic {magic!r}")
    if version != STATEMENT_VERSION:
        raise PlanSignError(f"unsupported PlanStatement version {version}")
    if flags:
        raise PlanSignError(f"reserved PlanStatement flags must be zero, got 0x{flags:04x}")
    o = _HEADER.size
    parts = [data[o + 32 * i : o + 32 * (i + 1)] for i in range(6)]
    st = PlanStatement(signed_at, *parts, signature=data[SIGNED_SIZE:STATEMENT_SIZE])
    _check_statement(st)
    return st


def decode_store(data: bytes) -> TrustStore:
    if len(data) < 8:
        raise PlanSignError("a TrustStoreV1 has an 8-byte header", "store")
    magic, version, count = struct.unpack_from("<4sHH", data, 0)
    if magic != STORE_MAGIC:
        raise PlanSignError(f"bad magic {magic!r}", "store")
    if version != STORE_VERSION:
        raise PlanSignError(f"unsupported TrustStore version {version}", "store")
    if len(data) != 8 + STORE_ENTRY_SIZE * count:
        raise PlanSignError(
            f"{count} keys need {8 + STORE_ENTRY_SIZE * count} bytes, got {len(data)}", "store"
        )
    keys = []
    for i in range(count):
        public, not_before, not_after, flags, reserved = _ENTRY.unpack_from(
            data, 8 + STORE_ENTRY_SIZE * i
        )
        if flags & ~KEY_FLAGS or reserved:
            raise PlanSignError(
                "undefined trust-store key flags or a nonzero reserved word", "store"
            )
        keys.append(TrustedKey(public, not_before, not_after, bool(flags & KEY_REVOKED)))
    store = TrustStore(tuple(keys))
    _check_store(store)
    return store


def sign_plan(
    plan_bytes: bytes,
    secret: bytes,
    signed_at: int,
    *,
    scope=None,
    pack_bytes: bytes | None = None,
    certificate=None,
) -> PlanStatement:
    """The signed statement over `plan_bytes` -- and, when given, the `ExecutionScope` it
    was planned under (its digest and its program component's), the pack's bytes and the
    certificate (anything with a `digest` that is a SHA-256 hex string)."""
    public = ed25519.public_key(secret)
    unsigned = PlanStatement(
        signed_at=_u64("signed_at", signed_at),
        key_id=key_id(public),
        plan=sha256(plan_bytes),
        scope=bytes.fromhex(scope.digest()) if scope is not None else _ZERO,
        module=bytes.fromhex(scope.component_digest("P")) if scope is not None else _ZERO,
        pack=sha256(pack_bytes) if pack_bytes is not None else _ZERO,
        certificate=bytes.fromhex(_hex_digest(certificate)) if certificate is not None else _ZERO,
    )
    _check_statement(unsigned)
    return replace(unsigned, signature=ed25519.sign(secret, unsigned.signed_bytes()))


def _hex_digest(obj) -> str:
    """A certificate's SHA-256, hex: its own `digest` when it has one, else the digest of its
    canonical JSON (`to_dict()`, sorted keys, no whitespace) -- never a repr."""
    digest = getattr(obj, "digest", None)
    if digest is not None:
        return digest() if callable(digest) else digest
    body = json.dumps(obj.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def verify_statement(
    statement: bytes,
    store: TrustStore,
    plan_bytes: bytes,
    now: int,
    *,
    pack_bytes: bytes | None = None,
) -> TrustedKey:
    """The trusted key that signed `statement` over exactly `plan_bytes` (and `pack_bytes`,
    when the statement binds a pack and the caller holds one), or a `PlanSignError` naming the
    first law it breaks: the store's own laws, the wire format, the plan's digest, the pack's, an
    unknown key, a revoked key, a statement dated after `now` or outside the key's window, the
    signature.

    The store is held to the laws `decode_store` holds its bytes to, whoever built it, and before
    anything else, as the C twin's `bcir_ps_verify` validates its store first: a store that
    trusts a small-order key accepts a forgery of every statement that names it."""
    _check_store(store)
    st = decode_statement(statement)
    if sha256(plan_bytes) != st.plan:
        raise PlanSignError("the statement does not bind these plan bytes", "plan")
    if pack_bytes is not None and st.pack != _ZERO and sha256(pack_bytes) != st.pack:
        raise PlanSignError("the statement does not bind this pack", "pack")
    key = store.find(st.key_id)
    if key is None:
        raise PlanSignError(
            "the statement is signed by a key the trust store does not hold", "unknown-key"
        )
    if key.revoked:
        raise PlanSignError("the signing key is revoked", "revoked")
    if st.signed_at > now:
        raise PlanSignError("the statement is dated after the time it is verified at", "future")
    if not key.not_before <= st.signed_at < key.not_after:
        raise PlanSignError("the statement is dated outside the signing key's window", "window")
    if not ed25519.verify(key.public_key, statement[:SIGNED_SIZE], st.signature):
        raise PlanSignError("the signature does not verify", "signature")
    return key


__all__ = [
    "KEY_REVOKED",
    "PlanSignError",
    "PlanStatement",
    "STATEMENT_SIZE",
    "TrustStore",
    "TrustedKey",
    "decode_statement",
    "decode_store",
    "key_id",
    "sign_plan",
    "verify_statement",
]
