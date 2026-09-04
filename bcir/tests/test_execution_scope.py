"""`S = digest(P, H, W, Theta, A, B, O, M, U, G)` and the class it licenses.

GEM+ slice G0. A certificate says "optimal", "bounded" or "measured best"; none of those
words mean anything without saying **over what**, and `ExecutionScope` is that. Two tests
here matter more than the rest, because they are the two failures the scope exists to stop.

**The collisions.** The 2026-08-12 audit found two inputs that change a plan without changing
its digest: scaling one memory tier's factors moved a score from 51,200 to 1,574,912, and
declaring two claims in the other order moved it from 3,840 to 4,352. Both survived because
`hash_target` omits the memory hierarchy and `hash_module` sorts claims by id. The scope
covers both, and `diff` names which component moved.

**The over-claim.** A class is not a label a caller picks. TMSAO-1 and TMSAO-2 are refused
outright when the scope leaves `P`, `H`, `A` or `O` undeclared, because an optimality claim
over a model with no declared objective is a claim about a model nobody wrote down. That
refusal is the whole reason the ladder is worth having: without it every certificate would
be TMSAO-1 by assertion.

Why the scope is a NEW object rather than a wider `hash_target`: those hashes are one half of
R13's cross-rail check, recomputed field for field in `BCIRVerifyPass.cpp`, and the tiers have
no ODS attribute to recompute from. Widening them is a dialect change that must land on both
rails in one commit. The scope contains the target hash as one field, so the cross-rail check
stays intact and the gap still closes.
"""

from __future__ import annotations

import json
from dataclasses import replace

from bcir.kbcir import TARGETS
from bcir.kbcir.cost import MemoryHierarchy, Theta, Tier
from bcir.kbcir.scope import (
    COMPONENTS,
    SCOPE_VERSION,
    UNDECLARED,
    ExecutionScope,
    certificate_class_allowed,
    gap,
    scope_for,
)
from bcir.kbcir.weights import PERF
from bcir.model import Claim, Lane, Module, Opcode, Phase, Resource, StrideClass

_HOST = TARGETS["x86_avx2"]


def _chain(order: str) -> Module:
    """Two dependent claims; `order` decides which is DECLARED first.

    The audit's fixture: these plan to 3,840 and 4,352 and shared a module hash.
    """
    module = Module(name="chain")
    for rid in (1, 2, 3):
        module.add_resource(Resource(rid=rid, shape=(256,)))
    first = Claim(
        id=10,
        opcode=Opcode.ADD,
        lane=Lane.U,
        stride_class=StrideClass.UNIT,
        count=256,
        rd=(1,),
        wr=(2,),
        op="vector.add",
    )
    second = Claim(
        id=20,
        opcode=Opcode.MUL,
        lane=Lane.U,
        stride_class=StrideClass.UNIT,
        count=256,
        rd=(2,),
        wr=(3,),
        op="vector.mul",
    )
    module.add_phase(
        Phase(phase_id=0, claims=[first, second] if order == "ab" else [second, first])
    )
    return module


def _slower_dram(target):
    return replace(
        target,
        mem=MemoryHierarchy(
            tuple(
                Tier(t.name, t.latency_cyc, t.bw_factor * 32, t.lat_factor * 32, t.capacity)
                if t.name == "DRAM"
                else t
                for t in target.mem.tiers
            )
        ),
    )


def _full_scope(module=None):
    """A scope with every optimality-relevant component declared."""
    return scope_for(
        module if module is not None else _chain("ab"),
        _HOST,
        Theta.cool(),
        PERF,
        objective={"relation": "scalarized", "weights": "PERF"},
        admitted={"candidates": "ALL_CANDIDATES"},
        budget={"caps": []},
        workload={"shape": "static"},
        measurement={"protocol": "median-of-5"},
        uncertainty={"model": "none"},
        generations={"cal_gen": 0},
    )


# --- the two collisions --------------------------------------------------------------------


def test_the_memory_hierarchy_is_inside_the_scope() -> None:
    """Scaling one tier moved a score thirty-fold under an unchanged digest."""
    base = scope_for(_chain("ab"), _HOST, Theta.cool(), PERF)
    altered = scope_for(_chain("ab"), _slower_dram(_HOST), Theta.cool(), PERF)

    assert base.digest() != altered.digest()
    assert base.diff(altered) == ("H",), "the diff must name the component that moved"

    # And it is genuinely the tiers doing it, not some other target field moving with them.
    assert base.component("H")["memory_hierarchy"] != altered.component("H")["memory_hierarchy"]
    assert base.component("H")["target_hash"] == altered.component("H")["target_hash"], (
        "hash_target still collides -- which is exactly why the scope carries the tiers "
        "itself rather than relying on it"
    )


def test_the_declared_claim_order_is_inside_the_scope() -> None:
    """`hash_module` sorts claims by id, so the two orders shared a digest."""
    forward = scope_for(_chain("ab"), _HOST, Theta.cool(), PERF)
    reverse = scope_for(_chain("ba"), _HOST, Theta.cool(), PERF)

    assert forward.digest() != reverse.digest()
    assert forward.diff(reverse) == ("P",)
    assert forward.component("P")["module_hash"] == reverse.component("P")["module_hash"], (
        "the cross-rail module hash is unchanged by design; the scope adds the order on top"
    )


def test_an_unchanged_program_produces_an_unchanged_scope() -> None:
    """The other direction. A digest that changes when nothing did is equally useless."""
    one = _full_scope()
    two = _full_scope()
    assert one.digest() == two.digest()
    assert one.diff(two) == ()

    # A note is provenance for a human and must not be able to move an identity.
    assert replace(one, note="a run I did on Tuesday").digest() == one.digest()


# --- canonical serialization ------------------------------------------------------------------


def test_equal_scopes_serialize_to_equal_bytes_whatever_the_insertion_order() -> None:
    """The `value_digest` defect from the same audit, pre-empted here.

    A Python dict's `repr` and its default JSON rendering both follow insertion order, and a
    certificate that followed it produced two digests for one value.
    """
    forward = ExecutionScope(W={"shape": "static", "concurrency": 4, "horizon": 100})
    reverse = ExecutionScope(W={"horizon": 100, "concurrency": 4, "shape": "static"})
    assert forward.to_canonical_json() == reverse.to_canonical_json()
    assert forward.digest() == reverse.digest()

    # Sets are unordered as values, so they must be ordered as bytes.
    assert ExecutionScope(A={1, 2, 3}).digest() == ExecutionScope(A={3, 2, 1}).digest()


def test_a_float_may_not_enter_a_scope() -> None:
    """Identity must not depend on binary rounding.

    A caller with a measured quantity has to say what precision it meant -- a rational, a
    fixed-point integer, or a string -- rather than let the last bit of an IEEE double decide
    whether two certificates are about the same model.
    """
    try:
        ExecutionScope(M={"warmup_seconds": 0.1}).digest()
    except TypeError as exc:
        assert "float" in str(exc)
    else:
        raise AssertionError("a float was accepted into a scope")

    # The stated alternatives all work.
    assert ExecutionScope(M={"warmup_ms": 100}).digest()
    assert ExecutionScope(M={"warmup_seconds": "1/10"}).digest()


def test_the_serialization_is_versioned_and_complete() -> None:
    """A silent format change would make two incomparable scopes look equal."""
    body = json.loads(_full_scope().to_canonical_json())
    assert body["version"] == SCOPE_VERSION
    assert set(body["components"]) == {name for name, _ in COMPONENTS}
    assert len(COMPONENTS) == 10

    # A version bump is a different identity even over identical components.
    bumped = replace(_full_scope(), version="ExecutionScopeV2")
    assert bumped.digest() != _full_scope().digest()
    assert bumped.diff(_full_scope()) == ("version",)


def test_undeclared_is_distinct_from_declared_and_empty() -> None:
    """ "Nobody said" and "said, and it is empty" are different statements about a model."""
    silent = ExecutionScope()
    empty = ExecutionScope(W={})
    assert silent.digest() != empty.digest()
    assert "W" in silent.undeclared()
    assert "W" not in empty.undeclared()
    assert len(silent.undeclared()) == 10


# --- the class ladder ---------------------------------------------------------------------


def test_the_class_is_earned_by_evidence_not_asserted() -> None:
    """Each rung needs what it claims, and nothing promotes without it."""
    scope = _full_scope()
    incumbent = {"incumbent": 100}
    assert certificate_class_allowed(scope, incumbent)[0] == "TMSAO-4"
    assert certificate_class_allowed(scope, {**incumbent, "lower_bound": 80})[0] == "TMSAO-2"
    assert (
        certificate_class_allowed(
            scope, {**incumbent, "search_coverage": 0.9, "prediction_interval": (95, 105)}
        )[0]
        == "TMSAO-3"
    )
    assert (
        certificate_class_allowed(
            scope,
            {
                **incumbent,
                "lower_bound": 80,
                "candidate_census": 12,
                "proof": "branch-and-bound",
                "census_complete": True,
            },
        )[0]
        == "TMSAO-1"
    )

    # A census that is not complete is not a proof of optimality, however large.
    assert (
        certificate_class_allowed(
            scope,
            {
                **incumbent,
                "lower_bound": 80,
                "candidate_census": 12,
                "proof": "branch-and-bound",
                "census_complete": False,
            },
        )[0]
        == "TMSAO-2"
    )


def test_an_optimality_claim_needs_a_declared_model() -> None:
    """The refusal that makes the ladder mean something.

    Evidence alone is not enough: "optimal" needs an objective to be optimal under, and a
    candidate census is a census of the admitted set. A scope missing either cannot support
    TMSAO-1 or TMSAO-2 no matter what evidence accompanies it.
    """
    strong = {
        "incumbent": 100,
        "lower_bound": 80,
        "candidate_census": 12,
        "proof": "branch-and-bound",
        "census_complete": True,
    }
    for missing in ("O", "A"):
        partial = replace(_full_scope(), **{missing: UNDECLARED})
        name, reason = certificate_class_allowed(partial, strong)
        assert name == "TMSAO-4", f"{missing} undeclared still produced {name}"
        assert missing in reason and "nobody wrote down" in reason

    # TMSAO-3 is about MEASURED evidence, so it is still available -- a measurement over an
    # undeclared objective is a real statement, it just is not an optimality one.
    partial = replace(_full_scope(), O=UNDECLARED)
    assert (
        certificate_class_allowed(
            partial, {"incumbent": 100, "search_coverage": 0.9, "prediction_interval": (95, 105)}
        )[0]
        == "TMSAO-3"
    )


def test_the_reason_is_always_actionable() -> None:
    """ "Why only TMSAO-4" is the question a reader has; the answer is a work item."""
    name, reason = certificate_class_allowed(_full_scope(), {"incumbent": 100})
    assert name == "TMSAO-4"
    assert "lower bound" in reason and "gap is only a gap when its size is known" in reason


def test_the_gap_is_reported_both_ways_and_refuses_a_contradiction() -> None:
    """`(U - L)` and `(U - L) / max(|U|, eps)`, as the proposal specifies."""
    assert gap(100, 80) == {"incumbent": 100, "lower_bound": 80, "absolute": 20, "relative": 0.2}
    assert gap(100, 100)["relative"] == 0.0, "on the bound is a zero gap, not an error"

    try:
        gap(80, 100)
    except ValueError as exc:
        assert "exceeds incumbent" in str(exc)
    else:
        raise AssertionError("a bound above the incumbent was reported as a negative gap")


def test_a_scope_built_from_nothing_still_has_a_stable_identity() -> None:
    """The degenerate case has to work: an empty scope is a legitimate starting point."""
    assert ExecutionScope().digest() == ExecutionScope().digest()
    assert scope_for().undeclared() == tuple(name for name, _ in COMPONENTS)
