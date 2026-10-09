"""Fixtures of G21 (plan signatures), shared by the tests, the C differential and the gate section.

Keys are derived from fixed seeds, so every statement is deterministic; the store holds a
rotation (A, then B, whose windows meet at 2000) and a revoked key R. `cases()` is every
forgery the verifier must refuse -- each with the one verdict code both rails must name -- and
the three statements it must accept (A in its window, B after the rotation, and A's
statement checked without the pack in hand).
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import replace

from bcir.abi import ed25519
from bcir.abi.plan_sign_abi import (
    PlanStatement,
    TrustedKey,
    TrustStore,
    sign_plan,
)


def secret(label: str) -> bytes:
    return hashlib.sha256(f"bcir plan-sign fixture {label}".encode()).digest()


A, B, R, C = (secret(x) for x in ("A", "B", "R", "C"))
WINDOWS = {"A": (1000, 2000), "B": (2000, 3000), "R": (0, 1 << 40)}


def store(revoked: bool = True) -> TrustStore:
    keys = [
        TrustedKey(ed25519.public_key(A), *WINDOWS["A"]),
        TrustedKey(ed25519.public_key(B), *WINDOWS["B"]),
        TrustedKey(ed25519.public_key(R), *WINDOWS["R"], revoked=revoked),
    ]
    return TrustStore(tuple(sorted(keys, key=lambda k: k.public_key)))


def material():
    """(plan bytes, pack bytes, scope, certificate): G19's explicit plan of the first six-job
    instance the dispatch gets wrong, the pack hydrated from its realization, the scope it was
    planned under and the exact rail's certificate of its placement."""
    from bcir.abi import encode, encode_plan
    from bcir.gem.exact import certify_schedule
    from bcir.gem.execution_plan import exact_plan
    from bcir.gem.streampack import hydrate
    from bcir.kbcir.scope import scope_for
    from bcir.tests.exact_fixtures import six_job_module, six_job_realization, six_job_target

    module, target = six_job_module(), six_job_target(2)
    result = six_job_realization((1, 3, 3, 3, 4, 4))
    plan, _search = exact_plan(module, result, target)
    return (
        encode_plan(plan),
        encode(hydrate(module, result, "plan0")),
        scope_for(module, target),
        certify_schedule(module, result, target),
    )


def genuine() -> PlanStatement:
    """A's statement over `material()` inside A's window, every digest bound: the statement the
    store accepts, and the decoder campaign's seed."""
    plan, pack, scope, cert = material()
    return sign_plan(plan, A, 1500, scope=scope, pack_bytes=pack, certificate=cert)


def _flip(blob: bytes, offset: int, mask: int = 1) -> bytes:
    b = bytearray(blob)
    b[offset] ^= mask
    return bytes(b)


def _resigned(st: PlanStatement, key: bytes, **changes) -> bytes:
    """A statement with `changes`, re-signed by `key` -- a forgery the signature cannot see."""
    unsigned = replace(st, **changes)
    return replace(unsigned, signature=ed25519.sign(key, unsigned.signed_bytes())).encode()


def cases():
    """(name, statement, store, plan, pack, now, expected code) -- "ok" for the two accepted."""
    plan, pack, scope, cert = material()
    good = store().encode()
    st_a = sign_plan(plan, A, 1500, scope=scope, pack_bytes=pack, certificate=cert)
    st_b = sign_plan(plan, B, 2500, scope=scope, pack_bytes=pack, certificate=cert)
    a = st_a.encode()
    out = [
        ("ok-A", a, good, plan, pack, 1600, "ok"),
        ("ok-B-after-rotation", st_b.encode(), good, plan, pack, 2600, "ok"),
        ("ok-no-pack-in-hand", a, good, plan, None, 1600, "ok"),
        # trust
        ("A-after-its-window", sign_plan(plan, A, 2500).encode(), good, plan, None, 2600, "window"),
        (
            "B-before-its-window",
            sign_plan(plan, B, 1500).encode(),
            good,
            plan,
            None,
            1600,
            "window",
        ),
        ("dated-after-now", a, good, plan, pack, 1400, "future"),
        ("revoked-key", sign_plan(plan, R, 1500).encode(), good, plan, None, 1600, "revoked"),
        ("unknown-key", sign_plan(plan, C, 1500).encode(), good, plan, None, 1600, "unknown-key"),
        # binding
        ("another-plan", a, good, _flip(plan, 70), pack, 1600, "plan"),
        ("another-pack", a, good, plan, _flip(pack, 40), 1600, "pack"),
        # the signed fields, each changed after signing
        ("scope-changed", _flip(a, 80), good, plan, pack, 1600, "signature"),
        ("module-changed", _flip(a, 112), good, plan, pack, 1600, "signature"),
        ("certificate-changed", _flip(a, 176), good, plan, pack, 1600, "signature"),
        ("signed-at-changed", _flip(a, 8, 2), good, plan, pack, 1600, "signature"),
        ("R-changed", _flip(a, 208), good, plan, pack, 1600, "signature"),
        ("S-not-below-L", _flip(a, 271, 0x10), good, plan, pack, 1600, "signature"),
        # re-signed by another trusted key, still claiming A
        ("key-id-of-A-signed-by-B", _resigned(st_a, B), good, plan, pack, 1600, "signature"),
        # the statement's wire
        ("statement-truncated", a[:-1], good, plan, pack, 1600, "statement"),
        ("statement-magic", _flip(a, 0), good, plan, pack, 1600, "statement"),
        ("statement-version", _flip(a, 4, 2), good, plan, pack, 1600, "statement"),
        ("statement-flags", _flip(a, 6), good, plan, pack, 1600, "statement"),
        ("statement-zero-plan", a[:48] + bytes(32) + a[80:], good, plan, pack, 1600, "statement"),
    ]
    # the store's wire
    pk_a = ed25519.public_key(A)
    entry = lambda pk, nb, na, fl=0, rs=0: struct.pack("<32sQQII", pk, nb, na, fl, rs)
    head = lambda n: struct.pack("<4sHH", b"BTRS", 1, n)
    identity = bytes([1]) + bytes(31)
    stores = {
        "store-truncated": good[:-1],
        "store-magic": _flip(good, 0),
        "store-version": _flip(good, 4, 2),
        "store-unsorted": head(2) + good[8 + 56 : 8 + 112] + good[8 : 8 + 56],
        "store-duplicate": head(2) + good[8 : 8 + 56] * 2,
        "store-empty-window": head(1) + entry(pk_a, 5, 5),
        "store-undefined-flag": head(1) + entry(pk_a, 1000, 2000, 2),
        "store-reserved": head(1) + entry(pk_a, 1000, 2000, 0, 1),
        "store-small-order-key": head(1) + entry(identity, 0, 1 << 40),
        "store-off-curve-key": head(1) + entry(bytes([2]) + bytes(31), 0, 1 << 40),
    }
    for name, blob in stores.items():
        out.append((name, a, blob, plan, pack, 1600, "store"))
    return out


def identity_forgery() -> tuple[bytes, bytes, bytes]:
    """(public key, message, signature): the forgery a small-order key admits -- the identity
    as the key and R = [S]B for an arbitrary S verify every message under RFC 8032's group
    equation, which is why the trust store refuses such keys."""
    s = (12345).to_bytes(32, "little")
    r = ed25519._compress(ed25519._mul(12345, ed25519._G))
    return bytes([1]) + bytes(31), b"any plan at all", r + s
