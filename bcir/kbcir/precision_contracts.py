"""Opt-in numerical contracts and conservative graph-precision propagation.

MPAT supplies backward requirements; AEDACI supplies *qualified* fallback candidates.
Smart Numbers' learned proposals remain outside the numerical proof boundary.

This is a pure Python reference planner, not a new runtime number type or a
hardware lowering pass. Inputs are exact integers/Fractions; floats are rejected
at the contract boundary to avoid silently rounding the *error budget*.

Proof scope: a finite DAG with declared, globally valid nonnegative absolute
input-error gains (Lipschitz bounds), absolute local-error bounds in ONE common
unit, and trusted evidence for each option. The planner never interprets
measured cross-entropy, ULPs at different grids, or probabilistic SR statements
as deterministic absolute-error proofs.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Sequence


class PrecisionContractError(ValueError):
    """A precision claim cannot be validated in this planner's proof domain."""


def _rational(value: int | Fraction, label: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise PrecisionContractError(f"{label} must be an exact int or Fraction")
    return Fraction(value)


@dataclass(frozen=True)
class ErrorBound:
    """Absolute bound in an explicitly named unit; never an implicit format ULP."""

    amount: Fraction
    unit: str = "real"

    def __post_init__(self) -> None:
        value = _rational(self.amount, "error amount")
        if value < 0 or not isinstance(self.unit, str) or not self.unit.strip():
            raise PrecisionContractError("error bound must be nonnegative with a named unit")
        object.__setattr__(self, "amount", value)


def quantization_error_bound_real(
    scale_exponent: int, *, rounding: str = "nearest", range_proved: bool = False
) -> ErrorBound:
    """No-clipping absolute bound for q*2**e with nearest or toward-zero rounding.

    'nearest' covers both common nearest tie rules; its bound is half a step.
    This is NOT an error bound if a code saturates, if inputs are nonfinite, or
    if conversion/scale arithmetic introduces extra unmodeled rounding.
    An explicit external range proof is required before constructing it.
    """
    if type(scale_exponent) is not int or not -300 <= scale_exponent <= 300:
        raise PrecisionContractError("scale exponent must be an integer in [-300, 300]")
    if rounding not in ("nearest", "truncate"):
        raise PrecisionContractError("unsupported deterministic quantizer rounding")
    if range_proved is not True:
        raise PrecisionContractError("quantization bound requires a no-clipping range proof")
    step = Fraction(2) ** scale_exponent
    return ErrorBound(step / 2 if rounding == "nearest" else step)


def quantization_error_in_q8_ticks(
    scale_exponent: int, *, rounding: str = "nearest", range_proved: bool = False
) -> int:
    """Conservative integer Q8-tick ceiling; never confuse grid ULP with Q8 ULP."""
    bound = quantization_error_bound_real(
        scale_exponent, rounding=rounding, range_proved=range_proved
    )
    ticks = bound.amount * 256
    return -(-ticks.numerator // ticks.denominator)


def signed_dot_accumulator_bits(bits_a: int, bits_b: int, count: int) -> int:
    """Sufficient signed width for a dot of symmetric signed quantizer codes.

    Codes are restricted to +/- (2**(bits-1)-1); products and their sum must
    be carried without overflow. This does not bound errors in the operands.
    """
    if any(type(x) is not int for x in (bits_a, bits_b, count)):
        raise PrecisionContractError("lane widths and count must be integers")
    if not (2 <= bits_a <= 4096 and 2 <= bits_b <= 4096 and count >= 1):
        raise PrecisionContractError("invalid signed code width or accumulation count")
    maximum = count * ((1 << (bits_a - 1)) - 1) * ((1 << (bits_b - 1)) - 1)
    return maximum.bit_length() + 1


_MX_FORMATS = {"MXFP8", "MXFP6", "MXFP4", "MXINT8"}


@dataclass(frozen=True)
class NumericalRepresentation:
    """Storage, compute, accumulator and rounding are separate P1/P2 contracts."""

    storage: str
    compute: str
    accumulator: str
    rounding: str = "nearest"
    overflow: str = "reject"
    block_size: int | None = None
    scale_format: str | None = None
    stochastic_seed: int | None = None
    stochastic_counter: str | None = None

    def __post_init__(self) -> None:
        if any(
            not isinstance(x, str) or not x for x in (self.storage, self.compute, self.accumulator)
        ):
            raise PrecisionContractError("numerical formats must be named")
        if self.rounding not in ("nearest", "truncate", "stochastic"):
            raise PrecisionContractError("unsupported rounding mode")
        if self.overflow not in ("reject", "saturate", "wrap", "ieee"):
            raise PrecisionContractError("unsupported overflow policy")
        if self.storage in _MX_FORMATS or self.compute in _MX_FORMATS:
            if self.block_size != 32 or self.scale_format != "E8M0":
                raise PrecisionContractError(
                    "OCP MX v1 concrete formats require 32-element blocks and E8M0 scale"
                )
        if self.rounding == "stochastic":
            if (
                type(self.stochastic_seed) is not int
                or not isinstance(self.stochastic_counter, str)
                or not self.stochastic_counter.strip()
            ):
                raise PrecisionContractError(
                    "stochastic rounding requires seed and stable operation counter identity"
                )
        elif self.stochastic_seed is not None or self.stochastic_counter is not None:
            raise PrecisionContractError("PRNG identity only applies to stochastic rounding")


@dataclass(frozen=True)
class PrecisionOption:
    """Candidate implementation with evidence, local error and estimated cost.

    'exact'/'bounded' candidates require explicit verifier evidence and a
    deterministic local error bound. 'empirical' and 'predicted' are recorded
    but never promoted to a mathematical proof by the planner.
    """

    name: str
    representation: NumericalRepresentation
    local_error: ErrorBound | None
    cost: int
    assurance: str = "predicted"
    evidence_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise PrecisionContractError("option requires a name")
        if type(self.cost) is not int or self.cost < 0:
            raise PrecisionContractError("cost must be a nonnegative integer")
        if not isinstance(self.representation, NumericalRepresentation):
            raise PrecisionContractError("option needs an explicit numerical representation")
        if self.local_error is not None and not isinstance(self.local_error, ErrorBound):
            raise PrecisionContractError("local error needs an ErrorBound")
        if not isinstance(self.evidence_id, str):
            raise PrecisionContractError("evidence ID must be a string")
        if self.assurance not in ("exact", "bounded", "empirical", "predicted"):
            raise PrecisionContractError("invalid assurance classification")
        if self.assurance in ("exact", "bounded"):
            if self.local_error is None or not self.evidence_id.strip():
                raise PrecisionContractError("certified options need a bound and evidence ID")
            if self.assurance == "exact" and self.local_error.amount != 0:
                raise PrecisionContractError("exact option cannot declare a nonzero local error")

    @property
    def certifiable(self) -> bool:
        return (
            self.assurance in ("exact", "bounded")
            and self.local_error is not None
            and self.representation.rounding != "stochastic"
        )


@dataclass(frozen=True)
class PrecisionNode:
    """Topologically ordered DAG node with conservative absolute-error gains."""

    name: str
    inputs: tuple[str, ...]
    gains: tuple[Fraction, ...]
    options: tuple[PrecisionOption, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise PrecisionContractError("node name is required")
        if len(self.inputs) != len(self.gains) or not self.options:
            raise PrecisionContractError("node needs aligned inputs/gains and options")
        if len(set(opt.name for opt in self.options)) != len(self.options):
            raise PrecisionContractError("option names must be unique per node")
        gains = tuple(_rational(v, "gain") for v in self.gains)
        if any(x < 0 for x in gains):
            raise PrecisionContractError("error gains cannot be negative")
        object.__setattr__(self, "gains", gains)


@dataclass(frozen=True)
class PrecisionPlan:
    """Conservative, feasible (not necessarily globally minimum-cost) selection."""

    selected: dict[str, PrecisionOption]
    local_budgets: dict[str, ErrorBound | None]
    output_bounds: dict[str, ErrorBound]
    total_cost: int


def propagate_precision(
    nodes: Sequence[PrecisionNode],
    output_tolerances: Mapping[str, ErrorBound],
) -> PrecisionPlan:
    """MPAT-inspired backward budget propagation with a forward proof check.

    For each requested output, sum ALL dependency paths, including shared DAG
    fanout, to obtain a conservative amplification per source. Share each
    output's tolerance equally among contributing nodes; select the cheapest
    analytically qualified candidate within each local budget. The final
    forward bound is independently recomputed before accepting a plan.

    This is intentionally conservative. It makes no claim of a globally optimal
    precision assignment, and rejects heterogeneous units and unsupported
    stochastic/probabilistic error claims.
    """
    if not nodes or not output_tolerances:
        raise PrecisionContractError("graph and output tolerances must be nonempty")
    by_name: dict[str, PrecisionNode] = {}
    for node in nodes:
        if not isinstance(node, PrecisionNode) or node.name in by_name:
            raise PrecisionContractError("duplicate or invalid precision node")
        for upstream in node.inputs:
            if upstream not in by_name:
                raise PrecisionContractError("nodes must be in DAG topological order")
        by_name[node.name] = node
    if any(not isinstance(bound, ErrorBound) for bound in output_tolerances.values()):
        raise PrecisionContractError("output tolerances must be explicit ErrorBound values")
    if any(out not in by_name for out in output_tolerances):
        raise PrecisionContractError("output name not present in graph")
    units = {bound.unit for bound in output_tolerances.values()}
    if len(units) != 1:
        raise PrecisionContractError("this planner needs one common absolute-error unit")
    unit = next(iter(units))
    budgets: dict[str, Fraction | None] = {name: None for name in by_name}

    for output, tolerance in output_tolerances.items():
        adjoint = {name: Fraction(0) for name in by_name}
        adjoint[output] = Fraction(1)
        for node in reversed(nodes):
            for upstream, gain in zip(node.inputs, node.gains):
                adjoint[upstream] += adjoint[node.name] * gain
        contributing = [name for name, amp in adjoint.items() if amp > 0]
        for name in contributing:
            cap = tolerance.amount / (len(contributing) * adjoint[name])
            previous = budgets[name]
            budgets[name] = cap if previous is None else min(previous, cap)

    selected: dict[str, PrecisionOption] = {}
    local_budgets: dict[str, ErrorBound | None] = {}
    for node in nodes:
        cap = budgets[node.name]
        local_budgets[node.name] = None if cap is None else ErrorBound(cap, unit)
        eligible = [
            option
            for option in node.options
            if option.certifiable
            and option.local_error is not None
            and option.local_error.unit == unit
            and (cap is None or option.local_error.amount <= cap)
        ]
        if not eligible:
            raise PrecisionContractError(f"no qualified precision option for {node.name!r}")
        selected[node.name] = min(
            eligible, key=lambda option: (option.cost, option.local_error.amount, option.name)
        )

    forward: dict[str, Fraction] = {}
    for node in nodes:
        option = selected[node.name]
        assert option.local_error is not None
        forward[node.name] = option.local_error.amount + sum(
            gain * forward[upstream] for upstream, gain in zip(node.inputs, node.gains)
        )
    outputs = {name: ErrorBound(forward[name], unit) for name in output_tolerances}
    if any(outputs[name].amount > bound.amount for name, bound in output_tolerances.items()):
        raise PrecisionContractError("forward error certificate exceeds output tolerance")
    return PrecisionPlan(
        selected=selected,
        local_budgets=local_budgets,
        output_bounds=outputs,
        total_cost=sum(option.cost for option in selected.values()),
    )
