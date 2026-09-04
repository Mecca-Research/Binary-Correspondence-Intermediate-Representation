"""bcir.kbcir.mapping -- Modular Mapping Functions: a map must preserve where the
objective matters (support), and equivalent conversion paths must agree.

A lowering, a calibration apply, a representation change -- each is a *mapping
function* `f` between cost-bearing representations. MOPC imposes two laws on any
such `f`, and both are checkable strengthenings of laws already in the repo:

  * **Objective-support preservation (an R12 refinement).**  `Supp(J)` is the set
    of cost dimensions on which the objective `J` is nonzero -- *where the
    objective matters*. A legal map must carry that support forward:

        f(Supp(J))  ⊆  Supp(J')        (J' = the mapped objective)

    A lowering may sharpen, rescale, or fuse a cost, but it may not silently
    **drop** a dimension that mattered (lose the thermal / security / accuracy /
    verification term) -- unless it carries an explicit *discharge* (the same
    escape R12 already grants bounds/hazard/precision via `bcir.trace`). This
    strengthens the lowering law R12 (lane geometry, bounds, hazard, precision)
    with a cost-support obligation: the objective's footprint is an invariant of
    legal lowering.

  * **Commutativity / path independence (a parity law).**  If two conversion
    paths reach the same target -- a direct map `Φ` and a two-step `Ψ` then `Λ` --
    they must agree:

        Λ ∘ Ψ  =  Φ

    This is the PARITY/manifest discipline *generalized* to any representation
    rail. Oracle↔MLIR parity, manifest replay (`reproduces`), JSON round-trips,
    and any future rail are all instances of one commuting-square law: the result
    cannot depend on which legal path produced it.

Deterministic and dependency-light: support is read off the 12-d `CostVector`;
commutativity is checked by running both paths and comparing. Plan-time / offline
work; nothing here runs on the L0 hot path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .cost import DIMS, CostVector


# --- objective support ------------------------------------------------------------


def support(cv: CostVector) -> frozenset:
    """Supp(J): the set of cost-dimension names where the objective is nonzero --
    where the objective matters."""
    return frozenset(name for name, c in zip(DIMS, cv.v) if c != 0)


def identity_dimmap() -> dict:
    """The default mapping on dimensions: a lowering keeps the cost taxonomy, so
    each source dimension maps to the same target dimension."""
    return {name: name for name in DIMS}


@dataclass(frozen=True)
class MappingFunction:
    """A modular mapping function `f` between cost-bearing representations,
    described by where it sends each cost dimension (`dim_map`) and which source
    dimensions it explicitly *discharges* (legally drops, with a trace note)."""

    name: str
    dim_map: dict = field(default_factory=identity_dimmap)
    discharged: frozenset = frozenset()

    def image(self, dims) -> frozenset:
        """f applied to a set of source dimensions (those it does not discharge)."""
        return frozenset(
            self.dim_map[d] for d in dims if d not in self.discharged and d in self.dim_map
        )

    def preserves_support(self, source: CostVector, target: CostVector) -> bool:
        """f(Supp(J)) ⊆ Supp(J'): every mattering source dim maps to a mattering
        target dim, modulo explicit discharges."""
        return self.image(support(source)) <= support(target)

    def dropped(self, source: CostVector, target: CostVector) -> frozenset:
        """The source dimensions whose objective support f silently dropped (not
        discharged, and absent from the target's support) -- the R12 offenders."""
        tgt = support(target)
        return frozenset(
            d
            for d in support(source)
            if d not in self.discharged and self.dim_map.get(d) not in tgt
        )


# --- commutativity (path independence) -------------------------------------------


@dataclass(frozen=True)
class CommutingSquare:
    """Two conversion paths that must agree: a direct map `phi` and a two-step
    `lam ∘ psi`. The law is `lam(psi(x)) == phi(x)` for every input -- a parity
    law for any representation rail (the manifest/PARITY discipline generalized)."""

    name: str
    phi: object  # the direct path  x -> z
    psi: object  # the first leg    x -> y
    lam: object  # the second leg   y -> z

    def commutes_on(self, x, eq=None) -> bool:
        eq = eq or (lambda a, b: a == b)
        return eq(self.lam(self.psi(x)), self.phi(x))

    def mismatches(self, inputs, eq=None) -> list:
        """The inputs on which the two paths disagree (the parity offenders)."""
        return [x for x in inputs if not self.commutes_on(x, eq)]
