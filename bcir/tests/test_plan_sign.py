"""G21 PLAN-SIGN: the plan bound by SHA-256 and signed (the oracle half).

The 2026-10-06 audit (item 6) found a plan carrying a CRC-32, naming its module and target by
63-bit FNV-1a hashes, and authorised only by a shared-key HMAC over the control record. Every
test here was RED on the parent (`bcir.abi.ed25519` and `bcir.abi.plan_sign_abi` did not
exist) and states the property, not the repair. `test_c_plan_sign.py` holds the C twin to the
same vectors and the same verdict on every case.
"""

from __future__ import annotations

import dataclasses
import hashlib

from bcir.abi import ed25519
from bcir.abi.plan_sign_abi import (
    SIGNED_SIZE,
    STATEMENT_SIZE,
    PlanSignError,
    PlanStatement,
    TrustedKey,
    TrustStore,
    decode_statement,
    decode_store,
    key_id,
    sha256,
    sign_plan,
    verify_statement,
)
from bcir.tests import plan_sign_fixtures as fx

#: RFC 8032 section 7.1: TEST 1, 2, 3 and TEST SHA(abc) -- (secret, public, message, signature).
RFC8032 = (
    (
        "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
        "",
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
    ),
    (
        "4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
        "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
        "72",
        "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00",
    ),
    (
        "c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
        "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
        "af82",
        "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a",
    ),
    (
        "833fe62409237b9d62ec77587520911e9a759cec1d19755b7da901b96dca3d42",
        "ec172b93ad5e563bf4932c70e1245034c35467ef2efd4d64ebf819683467e2bf",
        hashlib.sha512(b"abc").hexdigest(),
        "dc2a4459e7369633a52b1bf277839a00201009a3efbf3ecb69bea2186c26b58909351fc9ac90b3ecfdfbc7c66431e0303dca179c138ac17ad9bef1177331a704",
    ),
)


def _verdict(statement, store_bytes, plan, pack, now) -> str:
    try:
        verify_statement(statement, decode_store(store_bytes), plan, now, pack_bytes=pack)
    except PlanSignError as exc:
        return exc.code
    return "ok"


def test_ed25519_reproduces_the_rfc_8032_vectors():
    for sk, pk, msg, sig in RFC8032:
        sk, pk, msg, sig = map(bytes.fromhex, (sk, pk, msg, sig))
        assert ed25519.public_key(sk) == pk
        assert ed25519.sign(sk, msg) == sig
        assert ed25519.verify(pk, msg, sig)
        assert not ed25519.verify(pk, msg + b"\x00", sig)


def test_verification_is_strict_and_total_over_untrusted_bytes():
    """RFC 8032 section 5.1.7, with no exception escaping: an S not below L (the malleable
    second spelling of a valid signature), a key or R that is no curve point, a non-canonical
    key, and wrong lengths are each refused."""
    sk, pk, msg, sig = map(bytes.fromhex, RFC8032[1])
    L = 2**252 + 27742317777372353535851937790883648493
    s = int.from_bytes(sig[32:], "little")
    malleated = sig[:32] + (s + L).to_bytes(32, "little")
    assert not ed25519.verify(pk, msg, malleated)
    off_curve = bytes([2]) + bytes(31)
    assert not ed25519.verify(off_curve, msg, sig)
    assert not ed25519.verify(pk, msg, off_curve + sig[32:])
    p = 2**255 - 19
    one_plus_p = (1 + p).to_bytes(32, "little")  # the identity, spelled y + p
    assert not ed25519.verify(one_plus_p, msg, sig) and not ed25519.usable_key(one_plus_p)
    for short in (pk[:31], b""):
        assert not ed25519.verify(short, msg, sig)
    assert not ed25519.verify(pk, msg, sig[:63])


def test_a_small_order_key_would_let_anyone_sign_so_no_store_trusts_one():
    """The identity is a valid RFC 8032 public key, and under it R = [S]B verifies every
    message for every S: anyone can sign. The trust store refuses it and every small-order key
    on both rails."""
    pk, msg, forged = fx.identity_forgery()
    assert ed25519.verify(pk, msg, forged), "the hole the store law closes"
    assert not ed25519.usable_key(pk)
    assert ed25519.usable_key(ed25519.public_key(fx.A))
    # A store built in memory, not decoded, is held to the same law by the verifier: on the
    # parent of this check it accepted a statement forged under the identity key.
    plan = b"a plan nobody signed"
    unsigned = PlanStatement(signed_at=1500, key_id=key_id(pk), plan=sha256(plan))
    s = (12345).to_bytes(32, "little")
    r = ed25519._compress(ed25519._mul(12345, ed25519._G))
    statement = dataclasses.replace(unsigned, signature=r + s).encode()
    assert ed25519.verify(pk, statement[:SIGNED_SIZE], r + s)
    built = TrustStore((TrustedKey(pk, 1000, 2000),))
    try:
        verify_statement(statement, built, plan, 1500)
    except PlanSignError as exc:
        assert exc.code == "store", exc.code
    else:  # pragma: no cover - a regression
        raise AssertionError("a store trusting the identity key accepted a forged statement")


def test_every_case_draws_its_verdict():
    """Three accepted statements (A in its window, B after the rotation, A's without the pack
    in hand) and every forgery:
    a key outside its window, a statement dated after `now`, a revoked key, an unknown key,
    another plan, another pack, each signed field changed after signing, a statement re-signed
    by another trusted key while claiming A, and every malformed statement and store."""
    seen = {}
    for name, statement, store_bytes, plan, pack, now, want in fx.cases():
        got = _verdict(statement, store_bytes, plan, pack, now)
        assert got == want, (name, got, want)
        seen[want] = seen.get(want, 0) + 1
    assert set(seen) == {
        "ok",
        "window",
        "future",
        "revoked",
        "unknown-key",
        "plan",
        "pack",
        "signature",
        "statement",
        "store",
    }, seen


def test_the_statement_binds_the_scope_the_module_the_pack_and_the_certificate():
    plan, pack, scope, cert = fx.material()
    st = sign_plan(plan, fx.A, 1500, scope=scope, pack_bytes=pack, certificate=cert)
    blob = st.encode()
    assert len(blob) == STATEMENT_SIZE and decode_statement(blob) == st
    assert st.plan == hashlib.sha256(plan).digest()
    assert st.pack == hashlib.sha256(pack).digest()
    assert st.scope.hex() == scope.digest() and st.module.hex() == scope.component_digest("P")
    assert st.certificate != bytes(32)
    # an unbound statement (only the plan) still verifies, and binds no pack
    bare = sign_plan(plan, fx.A, 1500)
    assert bare.scope == bare.module == bare.pack == bare.certificate == bytes(32)
    assert _verdict(bare.encode(), fx.store().encode(), plan, b"any pack", 1600) == "ok"


def test_a_store_has_one_spelling_and_round_trips():
    store = fx.store()
    blob = store.encode()
    assert decode_store(blob) == store
    assert decode_store(blob).encode() == blob
    keys = [k.public_key for k in store.keys]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)
