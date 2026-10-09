"""Ed25519 (RFC 8032 section 5.1), dependency-free: the oracle half of G21's plan signatures.

The C twin is `runtime/c/bcir_ed25519.c`; both rails are held to RFC 8032's section 7.1 test
vectors and to each other (`bcir/tests/test_plan_sign.py`). This rail signs and verifies;
it is a reference, not a hardened signer -- Python integers are not constant-time, so a key
that must not leak through timing is held by the C twin or a hardware signer, never here.

Verification is the strict RFC 8032 section 5.1.7 check, made total over untrusted bytes: a
public key or an R that does not decode to a curve point, and an S that is not below the group
order L, are refused rather than reduced (the malleability the RFC's encoding rules close), and
the group equation is checked cofactorless, [S]B = R + [k]A, exactly as the C twin checks it.
"""

from __future__ import annotations

import hashlib

_P = 2**255 - 19
_L = 2**252 + 27742317777372353535851937790883648493
_D = (-121665 * pow(121666, _P - 2, _P)) % _P
_SQRT_M1 = pow(2, (_P - 1) // 4, _P)

#: Extended homogeneous coordinates (X, Y, Z, T) with x = X/Z, y = Y/Z, xy = T/Z.
_Point = tuple[int, int, int, int]


def _add(p: _Point, q: _Point) -> _Point:
    a = (p[1] - p[0]) * (q[1] - q[0]) % _P
    b = (p[1] + p[0]) * (q[1] + q[0]) % _P
    c = 2 * p[3] * q[3] * _D % _P
    d = 2 * p[2] * q[2] % _P
    e, f, g, h = b - a, d - c, d + c, b + a
    return (e * f % _P, g * h % _P, f * g % _P, e * h % _P)


def _mul(s: int, p: _Point) -> _Point:
    q: _Point = (0, 1, 1, 0)  # the neutral element
    while s > 0:
        if s & 1:
            q = _add(q, p)
        p = _add(p, p)
        s >>= 1
    return q


def _equal(p: _Point, q: _Point) -> bool:
    return (p[0] * q[2] - q[0] * p[2]) % _P == 0 and (p[1] * q[2] - q[1] * p[2]) % _P == 0


def _recover_x(y: int, sign: int) -> int | None:
    if y >= _P:
        return None
    x2 = (y * y - 1) * pow(_D * y * y + 1, _P - 2, _P) % _P
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P != 0:
        x = x * _SQRT_M1 % _P
    if (x * x - x2) % _P != 0:
        return None
    if (x & 1) != sign:
        x = _P - x
    return x


_GY = 4 * pow(5, _P - 2, _P) % _P
_GX = _recover_x(_GY, 0)
assert _GX is not None
_G: _Point = (_GX, _GY, 1, _GX * _GY % _P)


def _compress(p: _Point) -> bytes:
    zinv = pow(p[2], _P - 2, _P)
    x, y = p[0] * zinv % _P, p[1] * zinv % _P
    return int.to_bytes(y | ((x & 1) << 255), 32, "little")


def _decompress(s: bytes) -> _Point | None:
    if len(s) != 32:
        return None
    y = int.from_bytes(s, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y, sign)
    if x is None:
        return None
    return (x, y, 1, x * y % _P)


def _sha512_int(*parts: bytes) -> int:
    return int.from_bytes(hashlib.sha512(b"".join(parts)).digest(), "little")


def _expand(secret: bytes) -> tuple[int, bytes]:
    if len(secret) != 32:
        raise ValueError("an Ed25519 secret key is 32 bytes")
    h = hashlib.sha512(secret).digest()
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a, h[32:]


def public_key(secret: bytes) -> bytes:
    """The 32-byte public key of a 32-byte secret key."""
    a, _prefix = _expand(secret)
    return _compress(_mul(a, _G))


def sign(secret: bytes, message: bytes) -> bytes:
    """The deterministic 64-byte Ed25519 signature of `message` (RFC 8032 section 5.1.6)."""
    a, prefix = _expand(secret)
    pk = _compress(_mul(a, _G))
    r = _sha512_int(prefix, message) % _L
    rs = _compress(_mul(r, _G))
    k = _sha512_int(rs, pk, message) % _L
    s = (r + k * a) % _L
    return rs + int.to_bytes(s, 32, "little")


def verify(public: bytes, message: bytes, signature: bytes) -> bool:
    """Whether `signature` is `public`'s signature of `message`; False on any malformed input
    (a key or R off the curve, an S not below L, a wrong length) -- never an exception."""
    if len(public) != 32 or len(signature) != 64:
        return False
    a = _decompress(public)
    r = _decompress(signature[:32])
    if a is None or r is None:
        return False
    s = int.from_bytes(signature[32:], "little")
    if s >= _L:
        return False
    k = _sha512_int(signature[:32], public, message) % _L
    return _equal(_mul(s, _G), _add(r, _mul(k, a)))


def usable_key(public: bytes) -> bool:
    """Whether `public` may be trusted to sign: the canonical encoding of a curve point that
    is not of small order. A small-order key (the identity among them) makes [k]A vanish from
    the group equation after a cofactor's worth of doubling -- for the identity, R = [S]B
    "verifies" for every message, so anyone can sign as it. RFC 8032 does not forbid such keys;
    a trust store must (`plan_sign_abi`)."""
    if len(public) != 32:
        return False
    a = _decompress(public)
    if a is None:
        return False
    return not _equal(_mul(8, a), (0, 1, 1, 0))


__all__ = ["public_key", "sign", "usable_key", "verify"]
