"""The BCIR-ExecutionPlan ASN.1 module and its DER/OER/JER projections (G11, S1-C).

The projection follows the StreamPack precedent: it is *additive* -- the native ExecutionPlanV1
wire format stays frozen and byte-for-byte unchanged, and DER (and OER, and JER) are further
transfer syntaxes for the same abstract value. The same laws are gated:

    faithful  : decode_plan_der(encode_plan_der(p)) == p
    canonical : encode_plan_der(p) is DER, and encoding it again returns identical octets
    additive  : encode_plan(decode_plan_der(encode_plan_der(decode_plan(b)))) == b
"""

from __future__ import annotations

import os

from bcir.abi import decode_plan, encode_plan
from bcir.asn1 import Asn1Error
from bcir.asn1.artifact_bundle import ARTIFACT_BUNDLE_MODULE_OID
from bcir.asn1.der import is_der
from bcir.asn1.execution_plan import (
    EXECUTION_PLAN_MODULE_OID,
    MODULE,
    PROJECTION_VERSION,
    decode_plan_der,
    decode_plan_jer,
    decode_plan_oer,
    encode_plan_der,
    encode_plan_jer,
    encode_plan_oer,
    plan_to_value,
    value_to_plan,
)
from bcir.asn1.streampack import STREAMPACK_MODULE_OID
from bcir.asn1.tlv import decode_one
from bcir.examples import PROGRAMS
from bcir.gem.execution_plan import Lifetime, MovementEdge, plan_from_realization
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.kbcir.realize import optimize
from bcir.tests.plan_fixtures import audit_fixture

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ASN1 = os.path.join(_ROOT, "bcir", "asn1", "BCIR-ExecutionPlan.asn1")


def _plans():
    """One plan per corpus program and mode -- the real artifacts -- plus a plan carrying
    every record family (lifetimes, movement edges, a signed cost, the tail)."""
    h, theta = TargetProfile.x86_avx512(), Theta.cool()
    for name, build in sorted(PROGRAMS.items()):
        module = build()
        result = optimize(module, h, theta)
        for mode in ("eft", "tokens"):
            yield f"{name}/{mode}", plan_from_realization(module, result, h, mode)
    module, target, _theta, result = audit_fixture()
    plan = plan_from_realization(module, result, target, "tokens", plan="full")
    plan.lifetimes = [Lifetime(1, "ram", 0, 4096, 64, 0, 2), Lifetime(2, "hbm", 4096, 64, 64, 1, 1)]
    plan.moves = [
        MovementEdge(1, "ram", "hbm", 0, 4096, "pcie", "staged", "flush", 1, 2, 5, 9),
        MovementEdge(2, "ram", "hbm", 0, 64),
    ]
    yield "full", plan


def test_projection_round_trips_every_corpus_plan():
    """faithful: the abstract value survives DER exactly."""
    count = 0
    for name, plan in _plans():
        assert decode_plan_der(encode_plan_der(plan)) == plan, name
        count += 1
    assert count >= 25, f"corpus degenerated to {count} plan(s)"


def test_projection_is_der_and_idempotent():
    """canonical: a plan has exactly one DER spelling."""
    for name, plan in _plans():
        raw = encode_plan_der(plan)
        assert is_der(decode_one(raw)), name
        assert encode_plan_der(decode_plan_der(raw)) == raw, name


def test_native_bytes_survive_the_projection():
    """additive: the native ExecutionPlanV1 octets come back byte for byte."""
    for name, plan in _plans():
        native = encode_plan(plan)
        assert encode_plan(decode_plan_der(encode_plan_der(decode_plan(native)))) == native, name


def test_oer_and_jer_are_realizations_over_the_same_type_model():
    for name, plan in _plans():
        oer = encode_plan_oer(plan)
        assert decode_plan_oer(oer, canonical=True) == plan, name
        assert encode_plan_oer(decode_plan_oer(oer)) == oer, name
        jer = encode_plan_jer(plan)
        assert decode_plan_jer(jer) == plan, name
        assert encode_plan(decode_plan_jer(jer)) == encode_plan(plan), name


def test_the_compiled_module_encodes_byte_identically_to_the_hand_built_one():
    """The X.680 source is the schema: the front-end's model reproduces the hand-built DER."""
    from bcir.frontends.asn1 import compile_module

    compiled = compile_module(
        open(_ASN1, encoding="utf-8").read(), "BCIR-ExecutionPlan.asn1"
    ).module
    assert compiled.oid == MODULE.oid
    for name, plan in _plans():
        value = plan_to_value(plan)
        hand, parsed = (
            MODULE.encode("ExecutionPlan", value),
            compiled.encode("ExecutionPlan", value),
        )
        assert hand == parsed, f"{name}: compiled module produced different DER"
        assert compiled.decode("ExecutionPlan", parsed) == MODULE.decode("ExecutionPlan", hand)


def test_the_module_has_its_own_arc_and_version():
    assert EXECUTION_PLAN_MODULE_OID == (1, 3, 6, 1, 4, 1, 62596, 3)
    assert len({EXECUTION_PLAN_MODULE_OID, STREAMPACK_MODULE_OID, ARTIFACT_BUNDLE_MODULE_OID}) == 3
    module, target, _theta, result = audit_fixture()
    plan = plan_from_realization(module, result, target, "eft")
    value = plan_to_value(plan)
    assert value["version"] == PROJECTION_VERSION == 1
    without = dict(value)
    del without["version"]
    assert value_to_plan(without) == plan
    assert (
        value_to_plan(MODULE.decode("ExecutionPlan", MODULE.encode("ExecutionPlan", without)))
        == plan
    )


def test_an_unlisted_enumeration_value_is_refused():
    module, target, _theta, result = audit_fixture()
    plan = plan_from_realization(module, result, target, "eft")
    value = plan_to_value(plan)
    for field, bad in (("mode", 7),):
        broken = dict(value)
        broken[field] = bad
        try:
            value_to_plan(broken)
        except Asn1Error:
            continue
        raise AssertionError(f"{field}={bad} was accepted")
    step = dict(value["steps"][0])
    for field, bad in (("lane", 9),):
        step[field] = bad
        try:
            value_to_plan({**value, "steps": [step]})
        except (Asn1Error, ValueError):
            continue
        raise AssertionError(f"step {field}={bad} was accepted")
    plan.moves = [MovementEdge(1, "ram", "hbm", 0, 8, kind="direct")]
    broken = plan_to_value(plan)
    broken["moves"][0]["kind"] = 9
    try:
        value_to_plan(broken)
        raise AssertionError("move kind 9 was accepted")
    except Asn1Error:
        pass
