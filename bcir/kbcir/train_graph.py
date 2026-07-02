"""D1 (ML/AI roadmap §8.2), first slice: TRAINING AS A PLANNED GRAPH, not a Python loop.

The M3 loop (`kbcir/training.py`) drives real convergence but is invisible to the planner: no
cost, no budget, no replay. This module promotes ONE TRAINING STEP to first-class claims --

    forward (gem.matmul) -> activation (gem.activation:<act>) -> per-example loss
    (gem.loss:<loss>) -> reduce (reduce.loss_mean) -> backward (gem.autodiff) ->
    update (gem.opt_step:<opt>)

-- each stage its own phase (the RAW chain IS the phase DAG), so a training step is verified by
the R-laws, PRICED by the tropical optimizer, composed over steps/epochs via `kbcir.compose`
(a run is a `Seq` of steps -- epochs become phases, mini-batches planned streams), capped by an
RCSP budget (a run is budget-FEASIBLE or NOT before it executes), and R13-replayable (the module
hash is deterministic). Execution stays the M3 loop -- this is the PLANNING/verification shadow
of the same step, per the prototype-then-port discipline; wiring the GEM executor to run the
planned stream is the next D1 increment. Cost-side module: imports no verifier (two-truth)."""

from __future__ import annotations

from dataclasses import dataclass

from ..model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from .compose import CompositeResult, Leaf, Region, Seq, plan_composite
from .cost import HProfile, Theta
from .rcsp import Budget
from .weights import PERF, Policy

# RIDs of the step's resources (fixed layout -- the step is a closed universe).
_X, _W, _Z, _ACT, _Y, _LOSSV, _LOSS, _GRAD = range(1, 9)


@dataclass(frozen=True)
class TrainStepSpec:
    """The shape of one supervised training step (the M3 logistic-regression step by default)."""

    n_features: int = 4
    batch: int = 8
    activation: str = "sigmoid"
    loss: str = "bce"
    optimizer: str = "sgd"

    @property
    def n_params(self) -> int:
        return self.n_features + 1                   # weights + bias, the M3 readout shape


def _resources(spec: TrainStepSpec) -> dict[int, Resource]:
    b, nf, np_ = spec.batch, spec.n_features, spec.n_params
    return {r.rid: r for r in (
        Resource(rid=_X, domain=Domain.RAM, shape=(b, nf), name="X"),
        Resource(rid=_W, domain=Domain.RAM, shape=(np_,), name="W"),
        Resource(rid=_Z, domain=Domain.RAM, shape=(b,), name="Z"),
        Resource(rid=_ACT, domain=Domain.RAM, shape=(b,), name="ACT"),
        Resource(rid=_Y, domain=Domain.RAM, shape=(b,), name="Y"),
        Resource(rid=_LOSSV, domain=Domain.RAM, shape=(b,), name="LOSSV"),
        Resource(rid=_LOSS, domain=Domain.RAM, shape=(1,), name="LOSS"),
        Resource(rid=_GRAD, domain=Domain.RAM, shape=(np_,), name="GRAD"),
    )}


def _step_claims(spec: TrainStepSpec, base_id: int = 0) -> tuple[tuple[Claim, ...], ...]:
    """The six stages of one step, one claim tuple per stage (= per phase / compose Leaf)."""
    b, np_ = spec.batch, spec.n_params
    c = base_id
    stages = (
        # forward reads ALL of W per output element (a structured tile walk): the affine R7
        # extent model does not describe it, so the claim is `assumed_safe` -- the tensor-level
        # validation is the gem shape law (R22), not the per-index affine proof.
        Claim(id=c + 1, opcode=Opcode.T_MACC, lane=Lane.T, stride_class=StrideClass.TILE,
              count=b, rd=(_X, _W), wr=(_Z,), op="gem.matmul", domain=Domain.RAM,
              bounds="assumed_safe"),
        Claim(id=c + 2, opcode=Opcode.MUL, lane=Lane.U, stride_class=StrideClass.UNIT,
              count=b, rd=(_Z,), wr=(_ACT,), op=f"gem.activation:{spec.activation}",
              domain=Domain.RAM),
        Claim(id=c + 3, opcode=Opcode.MUL, lane=Lane.U, stride_class=StrideClass.UNIT,
              count=b, rd=(_ACT, _Y), wr=(_LOSSV,), op=f"gem.loss:{spec.loss}",
              domain=Domain.RAM),
        Claim(id=c + 4, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
              count=b, rd=(_LOSSV,), wr=(_LOSS,), op="reduce.loss_mean", domain=Domain.RAM),
        Claim(id=c + 5, opcode=Opcode.T_MACC, lane=Lane.T, stride_class=StrideClass.TILE,
              count=np_, rd=(_X, _ACT, _Y), wr=(_GRAD,), op="gem.autodiff", domain=Domain.RAM),
        Claim(id=c + 6, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
              count=np_, rd=(_GRAD, _W), wr=(_W,), op=f"gem.opt_step:{spec.optimizer}",
              domain=Domain.RAM),
    )
    return tuple((s,) for s in stages)


N_STAGES = 6


def train_step_module(spec: TrainStepSpec = TrainStepSpec()) -> Module:
    """One training step as a verifiable, plannable Module: six chained phases (the RAW order
    forward -> activation -> loss -> reduce -> backward -> update)."""
    m = Module(name=f"train_step_{spec.activation}_{spec.loss}_{spec.optimizer}")
    for r in _resources(spec).values():
        m.add_resource(r)
    prev: tuple[int, ...] = ()
    for pid, stage in enumerate(_step_claims(spec)):
        m.add_phase(Phase(phase_id=pid, deps=prev, claims=list(stage)))
        prev = (pid,)
    return m


def train_run_region(spec: TrainStepSpec, steps: int) -> Region:
    """A whole run as a compose region: a Seq of `steps` identical training steps (mini-batches /
    epochs are steps; the caller picks the granularity). Each step is itself a Seq of the six
    stage Leaves, so the composed cost is the series sum the planner can budget."""
    step = Seq(tuple(Leaf(stage) for stage in _step_claims(spec)))
    return Seq(tuple(step for _ in range(max(1, steps))))


def plan_train_run(spec: TrainStepSpec, steps: int, h: HProfile, theta: Theta,
                   policy: Policy = PERF, *, budget: Budget = None) -> CompositeResult:
    """Price a whole training run compositionally -- and, with an RCSP `budget`, decide its
    FEASIBILITY before it ever executes (`rcsp.Infeasible` when a stage cannot fit the caps)."""
    return plan_composite(train_run_region(spec, steps), {}, _resources(spec), h, theta,
                          policy, budget=budget)


def steps_for(n_examples: int, batch_size: int, epochs: int) -> int:
    """The M3 loop's step count for a dataset/epoch shape -- the structural bridge between the
    planned run (`train_run_region(spec, steps_for(...))`) and what `training.train` executes."""
    per_epoch = max(1, (n_examples + batch_size - 1) // batch_size)
    return per_epoch * max(1, epochs)
