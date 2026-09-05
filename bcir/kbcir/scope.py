"""`ExecutionScopeV1` — the ten-component identity a TMSAO certificate is *about*.

    S = digest(P, H, W, Theta, A, B, O, M, U, G)

A certificate says "this plan is optimal / bounded / measured best". None of those words mean
anything without saying **over what**, and that is what `S` is: the complete declared model
the claim ranges over. Two certificates may be compared only when their scopes are equal, and
a certificate whose scope omits a plan-affecting input is a claim about a model nobody wrote
down.

WHY THIS IS A NEW OBJECT RATHER THAN (ONLY) A WIDER `hash_target`.

The obvious repair for the gap the 2026-08-12 audit found — `hash_target` omits the memory
hierarchy, so scaling one tier's factors moved a plan's score thirty-fold with the digest
unchanged — is to add the tiers to `hash_target`. Doing that ALONE was the wrong move, and
it took a failed attempt to see why.

`hash_module` and `hash_target` are not general-purpose identity functions. They are one half
of R13's **cross-rail** check: `BCIRVerifyPass.cpp` recomputes both field for field from the
IR, and the two must agree exactly. Widening them is therefore a change to the MLIR dialect,
the C++ walk, and the pinned constants in `verify_provenance.mlir` — all of which must land in
one commit or the rails silently disagree about a content address, which is worse than the gap.

So the two jobs were separated: G0 landed `ExecutionScopeV1` as the *complete* identity
certificates bind to, and S0-D (S0-1 of the staged plan) then widened the two cross-rail
hashes on both rails in one commit — `hash_target` folds the memory hierarchy
(`target.capability`'s `mem_tier_names` / `mem_tier_values`, absent = the default hierarchy)
and `hash_module` folds
the claims in DECLARED order. The scope keeps naming the tiers and the order as components of
its own: a hash says *that* two scopes differ, `diff` says *which* component moved, and the
scope is what a certificate class is judged against.

THE COMPONENTS, and the honest state of each. `absent` is a first-class value here. A scope
that silently omits `W` is indistinguishable from one where the workload happened to be
empty, and the difference decides whether a TMSAO-3 claim is available at all — so an
undeclared component is recorded as undeclared, digested as such, and reported by
`undeclared()` so a certificate can refuse to over-claim.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

#: Bump when the serialization changes shape. A scope digest is only comparable within a
#: version, and a silent format change would make two incomparable scopes look equal --
#: exactly the failure this object exists to prevent.
SCOPE_VERSION = "ExecutionScopeV1"

#: What a component holds when nothing has been declared for it. Distinct from an empty
#: mapping, which means "declared, and empty".
UNDECLARED = "<undeclared>"

#: The ten components, in the order the proposal names them, with what each one carries.
COMPONENTS: tuple[tuple[str, str], ...] = (
    ("P", "program, input contract, BCIR laws, semantics, precision, admitted approximation"),
    ("H", "hardware topology, instruction/capability set, memory banks, links, capacities"),
    ("W", "workload shapes, input distribution, concurrency, service level, horizon"),
    ("Theta", "firmware, microcode, driver, OS, clocks, thermal, contention, wear"),
    ("A", "admitted transformations, libraries, kernels, schedules, candidate boundary"),
    ("B", "capacity, security, reliability, temperature, power, policy constraints"),
    ("O", "objective relation: lexicographic, Pareto, robust, constrained, scalarized"),
    ("M", "measurement protocol, warm-up, sampling, counters, outlier policy, environment"),
    ("U", "uncertainty model and confidence/prediction coverage"),
    ("G", "generations of hardware profile, calibration, firmware, driver, model, artifacts"),
)
_NAMES = tuple(name for name, _ in COMPONENTS)


def _canonical(value) -> object:
    """A value's canonical form, so equal scopes serialize to equal bytes.

    Mappings are emitted in sorted key order and sets as sorted sequences, because Python's
    iteration order is an artifact of insertion and hashing, not of the value. That is not a
    hypothetical concern here: the same audit found a certificate's `value_digest` following
    dict insertion order, so `{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` -- byte-identical once
    encoded -- produced two different digests and two certificates for one value.

    Floats are rejected rather than rounded. A scope is an identity, and a value whose
    equality depends on the last bit of an IEEE double is not one; a caller with a measured
    quantity should declare it as a rational or a fixed-point integer so the scope says what
    precision it meant.
    """
    if isinstance(value, bool) or isinstance(value, int) or isinstance(value, str):
        return value
    if value is None:
        return None
    if isinstance(value, float):
        raise TypeError(
            "a scope component may not hold a float: identity must not depend on binary "
            "rounding. Declare a rational, a fixed-point integer, or a string."
        )
    if isinstance(value, (bytes, bytearray)):
        return {"__bytes__": bytes(value).hex()}
    if isinstance(value, dict):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (set, frozenset)):
        return {"__set__": [_canonical(item) for item in sorted(value, key=repr)]}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    raise TypeError(
        f"a scope component may not hold {type(value).__name__}; a scope is serialized, so "
        f"every part of it has to have a written form"
    )


@dataclass(frozen=True)
class ExecutionScope:
    """The ten components and the digest over them."""

    P: object = UNDECLARED
    H: object = UNDECLARED
    W: object = UNDECLARED
    Theta: object = UNDECLARED
    A: object = UNDECLARED
    B: object = UNDECLARED
    O: object = UNDECLARED
    M: object = UNDECLARED
    U: object = UNDECLARED
    G: object = UNDECLARED
    version: str = SCOPE_VERSION
    #: Free-form provenance for a human. Deliberately NOT digested: a note must never be able
    #: to change an identity, or two runs of the same plan stop comparing.
    note: str = field(default="", compare=False)

    def component(self, name: str) -> object:
        if name not in _NAMES:
            raise KeyError(f"{name!r} is not one of {_NAMES}")
        return getattr(self, name)

    def undeclared(self) -> tuple[str, ...]:
        """The components nothing was declared for.

        A certificate consults this before choosing its class: an optimality statement over a
        model with an undeclared workload or objective is a statement about an unwritten
        model, and `certificate_class_allowed` uses exactly this to refuse one.
        """
        return tuple(name for name in _NAMES if self.component(name) == UNDECLARED)

    def to_canonical_json(self) -> str:
        body = {
            "version": self.version,
            "components": {name: _canonical(self.component(name)) for name in _NAMES},
        }
        return json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def digest(self) -> str:
        """`S`. SHA-256 over the canonical serialization."""
        return hashlib.sha256(self.to_canonical_json().encode("utf-8")).hexdigest()

    def component_digest(self, name: str) -> str:
        """One component's digest, so `diff` can say WHICH input moved.

        Domain-separated by the component's name, so two components that happen to hold the
        same value do not produce the same digest and a diff cannot confuse them.
        """
        body = json.dumps(
            {"version": self.version, "component": name, "value": _canonical(self.component(name))},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def diff(self, other: "ExecutionScope") -> tuple[str, ...]:
        """Which components differ. The debugging view the manifest's `diff` gives today."""
        if self.version != other.version:
            return ("version",)
        return tuple(
            name for name in _NAMES if self.component_digest(name) != other.component_digest(name)
        )


def scope_for(
    module=None,
    target=None,
    theta=None,
    policy=None,
    *,
    workload=None,
    admitted=None,
    budget=None,
    objective=None,
    measurement=None,
    uncertainty=None,
    generations=None,
    note: str = "",
) -> ExecutionScope:
    """Build a scope from the objects BCIR already has, declaring only what is supplied.

    `H` carries the MEMORY HIERARCHY, which is the component `hash_target` omits and the
    reason a plan's score could move thirty-fold under an unchanged digest. It also carries
    `hash_target` itself, so R13's cross-rail check remains inside the scope rather than
    being replaced by it: the scope is strictly stronger, and the two never disagree about
    the part they share.
    """
    from .provenance import hash_module, hash_policy, hash_target, hash_theta

    program = UNDECLARED
    if module is not None:
        program = {
            "module_hash": hash_module(module),
            "name": module.name,
            # DECLARED order. Two claims declared `a, b` and `b, a` plan to different scores;
            # `hash_module` folds the order too since S0-D, and the scope names it as a
            # component so `diff` can say that P moved because of the order.
            "claim_order": [
                [phase.phase_id, [claim.id for claim in phase.claims]] for phase in module.phases
            ],
            "laws": "R1-R25",
        }

    hardware = UNDECLARED
    if target is not None:
        tiers = [
            [tier.name, tier.latency_cyc, tier.bw_factor, tier.lat_factor, tier.capacity]
            for tier in getattr(getattr(target, "mem", None), "tiers", ())
        ]
        hardware = {
            "target_hash": hash_target(target),
            "name": target.name,
            "triple": target.triple,
            "lane_widths": sorted(target.lane_widths),
            "affinity_domains": target.affinity_domains,
            # Named as a component so `diff` can say that H moved because of the tiers;
            # `hash_target` folds them too since S0-D (`target.capability` `mem_tier_names` /
            # `mem_tier_values`).
            "memory_hierarchy": tiers,
            "cal_gen": int(getattr(target, "cal_gen", 0)),
        }

    runtime = UNDECLARED if theta is None else {"theta_hash": hash_theta(theta)}
    admitted_set = UNDECLARED
    if policy is not None or admitted is not None:
        admitted_set = {
            "policy_hash": hash_policy(policy) if policy is not None else None,
            "admitted": admitted if admitted is not None else UNDECLARED,
        }

    return ExecutionScope(
        P=program,
        H=hardware,
        W=workload if workload is not None else UNDECLARED,
        Theta=runtime,
        A=admitted_set,
        B=budget if budget is not None else UNDECLARED,
        O=objective if objective is not None else UNDECLARED,
        M=measurement if measurement is not None else UNDECLARED,
        U=uncertainty if uncertainty is not None else UNDECLARED,
        G=generations if generations is not None else UNDECLARED,
        note=note,
    )


# --- the certificate classes ---------------------------------------------------------------

#: What each class is permitted to say, and what has to exist before it may say it. Ordered
#: strongest first; `certificate_class_allowed` walks it in this order.
CLASS_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "TMSAO-1",
        "exact optimum over the declared finite model",
        ("candidate_census", "proof", "lower_bound", "incumbent"),
    ),
    ("TMSAO-2", "best bounded result, with an explicit gap", ("lower_bound", "incumbent")),
    (
        "TMSAO-3",
        "best measured admitted realization",
        ("prediction_interval", "search_coverage", "incumbent"),
    ),
    (
        "TMSAO-4",
        "heuristic incumbent; legality and reproducibility only, no optimality",
        ("incumbent",),
    ),
)

#: Components whose absence makes an OPTIMALITY claim meaningless, whatever evidence exists.
#: `O` because "optimal" needs an objective relation to be optimal under; `A` because a
#: candidate census is a census of the admitted set; `H` and `P` because they are what is
#: being optimized and what it runs on. `W`, `M` and `U` are not here: a scope may legitimately
#: be optimal over a model that declares no workload, and TMSAO-3 checks its own evidence.
OPTIMALITY_COMPONENTS = ("P", "H", "A", "O")


def certificate_class_allowed(scope: ExecutionScope, evidence: dict) -> tuple[str, str]:
    """The strongest class this scope and evidence support, and why not a stronger one.

    Returns `(class, reason)`. The reason is always populated, including for the strongest
    class, because "why is this only TMSAO-4" is the question a reader actually has and
    answering it is what turns the ladder into a work list.
    """
    missing_scope = [name for name in OPTIMALITY_COMPONENTS if scope.component(name) == UNDECLARED]

    for name, statement, required in CLASS_REQUIREMENTS:
        if name == "TMSAO-4":
            break  # the floor: its reason is always the diagnosis below
        if any(not evidence.get(key) for key in required):
            continue
        if name in ("TMSAO-1", "TMSAO-2") and missing_scope:
            continue
        if name == "TMSAO-1" and not evidence.get("census_complete"):
            continue
        return name, statement

    # Landing on the floor, the useful answer is not "this is a heuristic incumbent" -- the
    # caller can see that. It is what would have to exist for the next rung up, because that
    # is a work item. Reported most-actionable first: a missing scope component blocks an
    # optimality claim no matter how much evidence is gathered, so it outranks a missing bound.
    if not evidence.get("incumbent"):
        return "TMSAO-4", ("no incumbent was recorded, so there is nothing to certify")
    if missing_scope:
        return "TMSAO-4", (
            f"an optimality claim needs a declared {', '.join(missing_scope)}; without it "
            f"the claim ranges over a model nobody wrote down"
        )
    if not evidence.get("lower_bound"):
        return "TMSAO-4", (
            "no lower bound was computed, so the distance to optimal is unknown -- a gap is "
            "only a gap when its size is known"
        )
    return "TMSAO-4", (
        "the evidence present supports no stronger statement; see CLASS_REQUIREMENTS for "
        "what each rung needs"
    )


def gap(incumbent: int | float, lower_bound: int | float, *, epsilon: float = 1e-9) -> dict:
    """The absolute and relative gap a TMSAO-2 statement reports.

    `(U - L) / max(|U|, epsilon)`, as the proposal specifies. A lower bound above the
    incumbent is a contradiction rather than a negative gap -- one of the two is wrong, and
    reporting a tidy negative number would hide which.
    """
    if lower_bound > incumbent:
        raise ValueError(
            f"lower bound {lower_bound} exceeds incumbent {incumbent}: a bound above the "
            f"best known result means the bound or the incumbent is wrong, not that the gap "
            f"is negative"
        )
    absolute = incumbent - lower_bound
    return {
        "incumbent": incumbent,
        "lower_bound": lower_bound,
        "absolute": absolute,
        "relative": absolute / max(abs(incumbent), epsilon),
    }


__all__ = [
    "CLASS_REQUIREMENTS",
    "COMPONENTS",
    "ExecutionScope",
    "OPTIMALITY_COMPONENTS",
    "SCOPE_VERSION",
    "UNDECLARED",
    "certificate_class_allowed",
    "gap",
    "scope_for",
]
