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


# --- D1 step 2: the plan BECOMES the execution path -------------------------------------------
#
# `hydrate_train_step` lowers the selected realization into a StreamPack (R10 provenance,
# R11 generation tags); `train_planned` then drives REAL numeric training where the GEM
# executor dispatches each stage kernel in the planned claim order -- the M3 numerics ride
# the claim graph instead of a Python loop, and every epoch commits a ProvenanceManifest
# (the R13 flight-recorder entry: same plan inputs, the epoch as the in-force artifact tag).

from .provenance import ProvenanceManifest, build_manifest      # noqa: E402
from ..gem.execute import ExecResult, execute                   # noqa: E402
from ..gem.streampack import StreamPack, hydrate                # noqa: E402
from .realize import RealizationResult, optimize                # noqa: E402
from .recurrent import sigmoid                                  # noqa: E402


def hydrate_train_step(spec: TrainStepSpec, h: HProfile, theta: Theta,
                       policy: Policy = PERF) -> tuple[StreamPack, RealizationResult]:
    """Plan one training step and lower the selected realization into a StreamPack."""
    m = train_step_module(spec)
    result = optimize(m, h, theta, policy)
    return hydrate(m, result, plan=m.name), result


def _step_kernels(spec: TrainStepSpec, st: dict) -> dict[int, "object"]:
    """The REAL numeric stage kernels, keyed by the claim ids of `train_step_module` --
    the M3 logistic step (sigmoid + BCE, exact closed-form gradient: the M1 seed pattern;
    sigmoid rides the closed-form/libm edge exactly as in E4 Tier-B). Each kernel reads and
    writes the shared state `st` (X, y, w, z, act, lossv, loss, grad, lr) -- the resources
    of the claim graph, materialized."""
    nf, b = spec.n_features, spec.batch

    def forward():                                   # claim 1: z = X @ w + bias
        for i in range(b):
            st["z"][i] = sum(st["X"][i][j] * st["w"][j] for j in range(nf)) + st["w"][nf]

    def activation():                                # claim 2: act = sigmoid(z)
        for i in range(b):
            st["act"][i] = sigmoid(st["z"][i])

    def loss_vec():                                  # claim 3: per-example BCE
        eps = 1e-12
        import math
        for i in range(b):
            a = min(max(st["act"][i], eps), 1.0 - eps)
            yv = st["y"][i]
            st["lossv"][i] = -(yv * math.log(a) + (1.0 - yv) * math.log(1.0 - a))

    def reduce_mean():                               # claim 4: loss = mean(lossv)
        st["loss"][0] = sum(st["lossv"]) / b

    def backward():                                  # claim 5: exact BCE+sigmoid gradient
        for j in range(nf):
            st["grad"][j] = sum((st["act"][i] - st["y"][i]) * st["X"][i][j]
                                for i in range(b)) / b
        st["grad"][nf] = sum(st["act"][i] - st["y"][i] for i in range(b)) / b

    def update():                                    # claim 6: sgd step
        for j in range(nf + 1):
            st["w"][j] -= st["lr"] * st["grad"][j]

    return {1: forward, 2: activation, 3: loss_vec, 4: reduce_mean, 5: backward, 6: update}


@dataclass
class PlannedTrainRun:
    """A GEM-executed training run: the trained weights, the per-epoch mean loss curve, the
    per-epoch R13 manifests (each replayable), the hydrated pack, and the dispatch witness."""

    weights: list
    losses: list
    manifests: tuple
    pack: StreamPack
    exec_orders: list                                # per executed step: the claim dispatch order


def train_planned(spec: TrainStepSpec, X: list, y: list, w0: list, *, epochs: int,
                  lr: float, h: HProfile, theta: Theta,
                  policy: Policy = PERF) -> PlannedTrainRun:
    """REAL training on the PLANNED path: the GEM executor dispatches the six stage kernels
    in the claim graph's phase order for every step (deterministic batches, no shuffle), and
    each epoch commits a ProvenanceManifest whose artifact tags pin the epoch + the pack
    generations -- manifest equality across runs => identical training trajectory."""
    m = train_step_module(spec)
    pack, _result = hydrate_train_step(spec, h, theta, policy)
    b = spec.batch
    n = len(X)
    st = {"X": [[0.0] * spec.n_features for _ in range(b)], "y": [0.0] * b,
          "w": list(w0), "z": [0.0] * b, "act": [0.0] * b, "lossv": [0.0] * b,
          "loss": [0.0], "grad": [0.0] * spec.n_params, "lr": lr}
    kernels = _step_kernels(spec, st)
    losses: list = []
    manifests: list = []
    exec_orders: list = []
    for e in range(epochs):
        epoch_losses = []
        for lo in range(0, n - b + 1, b):            # full batches only, deterministic order
            for i in range(b):
                st["X"][i] = list(X[lo + i])
                st["y"][i] = float(y[lo + i])
            r: ExecResult = execute(m, kernels)
            exec_orders.append(list(r.order))
            epoch_losses.append(st["loss"][0])
        losses.append(sum(epoch_losses) / max(1, len(epoch_losses)))
        manifests.append(build_manifest(
            m, h, theta, policy,
            artifacts=(("epoch", e), ("topo_gen", pack.topo_gen),
                       ("map_gen", pack.map_gen), ("data_gen", pack.data_gen))))
    return PlannedTrainRun(weights=list(st["w"]), losses=losses, manifests=tuple(manifests),
                           pack=pack, exec_orders=exec_orders)
