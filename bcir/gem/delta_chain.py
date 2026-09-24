"""GEM+ G18 (S4-B): the K_BCIR -> StreamPack chain, advanced by declared deltas.

The chain is plan -> pack -> bytes -> verdict:

    result = realize.optimize(module, h, theta, policy)
    pack   = hydrate_pipelined(module, result, plan, depth)
    data   = streampack_abi.encode(pack)
    diags  = verify(module) + verify_plan(module, result, h, theta=theta, policy=policy)
                            + verify_pack(module, pack)

`full_chain` is that, from scratch -- the reference. `DeltaChain` holds the three incremental
states (`kbcir.delta.IncrementalPlan`, `gem.delta_pack.PackState`, `verify.delta.VerifyState`)
and advances them by a `Delta`; every step's `Link` equals `full_chain` of the module the delta
declares (`planner.delta.parity`, `pack.delta.identity`, `verify.delta.identity`), while the
interpreted work is proportional to the delta's dependency cone, not to the module
(`kbcir-streampack.delta`, `kbcir-streampack.delta.calls`).

A delta the chain does not admit raises `DeltaError` before anything moves. A pack the wire
cannot carry raises what `encode` raises, as the full chain does; the plan has moved by then,
and the next delta re-emits the pack in full and re-verifies everything that changed since the
last verdict.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..abi.streampack_abi import encode
from ..kbcir.cost import HProfile, Theta
from ..kbcir.delta import Delta, IncrementalPlan
from ..kbcir.realize import RealizationResult, optimize
from ..kbcir.weights import PERF, Policy
from ..model import Module
from .delta_pack import PackState
from .streampack import StreamPack, hydrate_pipelined

__all__ = ["DeltaChain", "Link", "full_chain"]


@dataclass(frozen=True)
class Link:
    """One state of the chain: the module, its plan, its pack, the pack's bytes and the
    verdict over the three."""

    module: Module
    result: RealizationResult
    pack: StreamPack
    data: bytes
    diagnostics: list


def full_chain(
    module: Module,
    h: HProfile,
    theta: Theta,
    policy: Policy = PERF,
    plan: str = "plan0",
    depth: int = 2,
) -> Link:
    """The chain from scratch (the reference every `DeltaChain` step is held to)."""
    from ..verify import verify, verify_pack, verify_plan

    result = optimize(module, h, theta, policy)
    pack = hydrate_pipelined(module, result, plan, depth)
    data = encode(pack)
    diags = (
        verify(module)
        + verify_plan(module, result, h, theta=theta, policy=policy)
        + verify_pack(module, pack)
    )
    return Link(module, result, pack, data, diags)


class DeltaChain:
    """The chain's state for one module under (h, theta, policy), advanced by deltas."""

    __slots__ = ("h", "theta", "policy", "plan", "depth", "link", "_plan", "_pack", "_verify")

    @classmethod
    def build(
        cls,
        module: Module,
        h: HProfile,
        theta: Theta,
        policy: Policy = PERF,
        plan: str = "plan0",
        depth: int = 2,
    ) -> "DeltaChain":
        """Run the chain once, keeping the state a delta re-derives from."""
        from ..verify.delta import VerifyState

        self = cls.__new__(cls)
        self.h, self.theta, self.policy, self.plan, self.depth = h, theta, policy, plan, depth
        self._plan = IncrementalPlan.build(module, h, theta, policy)
        self._pack = PackState.build(module, self._plan.result, plan, depth)
        self._verify = VerifyState.build(
            module, self._plan.result, self._pack.pack, h, theta, policy
        )
        self.link = Link(
            module,
            self._plan.result,
            self._pack.pack,
            self._pack.data,
            self._verify.diagnostics,
        )
        return self

    @property
    def module(self) -> Module:
        """The module the last delta declared -- also when its pack was refused."""
        return self._plan.module

    @property
    def result(self) -> RealizationResult:
        """The plan of `module` -- also when its pack was refused (the plan moves first)."""
        return self._plan.result

    @property
    def rebuilds(self) -> int:
        """How many times the verdict was re-derived from scratch (a delta outside v0's shape)."""
        return self._verify.rebuilds

    def apply(self, delta: Delta) -> Link:
        """Advance the chain by `delta` and return the new link."""
        plan = self._plan
        result = plan.apply(delta)
        module = plan.module
        pack, data = self._pack.apply(
            module,
            result,
            plan.claims,
            plan.changed,
            plan.edited,
            {r.rid for r in delta.resources},
        )
        diags = self._verify.apply(module, result, pack)
        self.link = Link(module, result, pack, data, diags)
        return self.link
