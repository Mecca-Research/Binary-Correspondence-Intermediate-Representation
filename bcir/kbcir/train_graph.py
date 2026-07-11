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
hash is deterministic). The execution paths below dispatch the selected activation, loss, gradient,
and stateful optimizer through that planned graph. Cost-side module: imports no verifier (two-truth)."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ..model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from .compose import CompositeResult, Leaf, Region, Seq, plan_composite
from .cost import HProfile, Theta
from .rcsp import Budget
from .weights import PERF, Policy

# RIDs of the step's resources (fixed layout -- the step is a closed universe).  Optimizer
# state lives outside the stream RID bands, so the same state resources can be shared by the
# ordinary, multi-step, and streamed graphs without colliding with per-stream tensors.
_X, _W, _Z, _ACT, _Y, _LOSSV, _LOSS, _GRAD = range(1, 9)
_OPT_STATE1, _OPT_STATE2, _OPT_STEP = 90_001, 90_002, 90_003

_ACTIVATIONS = ("relu", "sigmoid", "tanh", "gelu")
_LOSSES = ("bce", "mse")
_OPTIMIZERS = ("sgd", "momentum", "rmsprop", "adam")


@dataclass(frozen=True)
class TrainStepSpec:
    """The shape of one supervised training step (the M3 logistic-regression step by default)."""

    n_features: int = 4
    batch: int = 8
    activation: str = "sigmoid"
    loss: str = "bce"
    optimizer: str = "sgd"

    def __post_init__(self) -> None:
        if not isinstance(self.n_features, int) or self.n_features < 1:
            raise ValueError(f"n_features must be a positive integer; got {self.n_features!r}")
        if not isinstance(self.batch, int) or self.batch < 1:
            raise ValueError(f"batch must be a positive integer; got {self.batch!r}")
        if self.activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {_ACTIVATIONS}; got {self.activation!r}")
        if self.loss not in _LOSSES:
            raise ValueError(f"loss must be one of {_LOSSES}; got {self.loss!r}")
        if self.loss == "bce" and self.activation != "sigmoid":
            raise ValueError("bce requires activation='sigmoid' (the fused BCE-with-logits gradient)")
        if self.optimizer not in _OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {_OPTIMIZERS}; got {self.optimizer!r}")

    @property
    def n_params(self) -> int:
        return self.n_features + 1                   # weights + bias, the M3 readout shape


def _optimizer_state_rids(spec: TrainStepSpec) -> tuple[int, ...]:
    """Claim-visible state read and written by the selected optimizer."""
    if spec.optimizer in ("momentum", "rmsprop"):
        return (_OPT_STATE1,)
    if spec.optimizer == "adam":
        return (_OPT_STATE1, _OPT_STATE2, _OPT_STEP)
    return ()


def _optimizer_resources(spec: TrainStepSpec) -> tuple[Resource, ...]:
    np_ = spec.n_params
    if spec.optimizer == "momentum":
        return (Resource(rid=_OPT_STATE1, domain=Domain.RAM, shape=(np_,), name="VELOCITY"),)
    if spec.optimizer == "rmsprop":
        return (Resource(rid=_OPT_STATE1, domain=Domain.RAM, shape=(np_,), name="SQ_AVG"),)
    if spec.optimizer == "adam":
        return (Resource(rid=_OPT_STATE1, domain=Domain.RAM, shape=(np_,), name="ADAM_M"),
                Resource(rid=_OPT_STATE2, domain=Domain.RAM, shape=(np_,), name="ADAM_V"),
                # The update claim is vector-width ``n_params``; model the scalar step as
                # a broadcast lane so the ordinary extent law remains exact.
                Resource(rid=_OPT_STEP, domain=Domain.RAM, shape=(np_,), name="ADAM_STEP"))
    return ()


def _resources(spec: TrainStepSpec) -> dict[int, Resource]:
    b, nf, np_ = spec.batch, spec.n_features, spec.n_params
    resources = (
        Resource(rid=_X, domain=Domain.RAM, shape=(b, nf), name="X"),
        Resource(rid=_W, domain=Domain.RAM, shape=(np_,), name="W"),
        Resource(rid=_Z, domain=Domain.RAM, shape=(b,), name="Z"),
        Resource(rid=_ACT, domain=Domain.RAM, shape=(b,), name="ACT"),
        Resource(rid=_Y, domain=Domain.RAM, shape=(b,), name="Y"),
        Resource(rid=_LOSSV, domain=Domain.RAM, shape=(b,), name="LOSSV"),
        Resource(rid=_LOSS, domain=Domain.RAM, shape=(1,), name="LOSS"),
        Resource(rid=_GRAD, domain=Domain.RAM, shape=(np_,), name="GRAD"),
    ) + _optimizer_resources(spec)
    return {r.rid: r for r in resources}


def _step_claims(spec: TrainStepSpec, base_id: int = 0) -> tuple[tuple[Claim, ...], ...]:
    """The six stages of one step, one claim tuple per stage (= per phase / compose Leaf)."""
    b, np_ = spec.batch, spec.n_params
    c = base_id
    opt_state = _optimizer_state_rids(spec)
    backward_rd = (_X, _ACT, _Y) if spec.loss == "bce" else (_X, _Z, _ACT, _Y)
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
              count=np_, rd=backward_rd, wr=(_GRAD,), op="gem.autodiff", domain=Domain.RAM),
        Claim(id=c + 6, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
              count=np_, rd=(_GRAD, _W, *opt_state), wr=(_W, *opt_state),
              op=f"gem.opt_step:{spec.optimizer}",
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
from ..lower.optimizers import (adam_step, momentum_step,       # noqa: E402
                                rmsprop_step, sgd_step)


def hydrate_train_step(spec: TrainStepSpec, h: HProfile, theta: Theta,
                       policy: Policy = PERF) -> tuple[StreamPack, RealizationResult]:
    """Plan one training step and lower the selected realization into a StreamPack."""
    m = train_step_module(spec)
    result = optimize(m, h, theta, policy)
    return hydrate(m, result, plan=m.name), result


def _activation_and_derivative(kind: str, z: float) -> tuple[float, float]:
    """One supported scalar activation and its derivative with respect to ``z``.

    GELU uses the same tanh approximation as ``activation.gelu_reference``.  ReLU takes
    the conventional zero subgradient at the kink.
    """
    if kind == "relu":
        return (z if z > 0.0 else 0.0, 1.0 if z > 0.0 else 0.0)
    if kind == "sigmoid":
        value = sigmoid(z)
        return value, value * (1.0 - value)
    if kind == "tanh":
        value = math.tanh(z)
        return value, 1.0 - value * value
    if kind == "gelu":
        k = math.sqrt(2.0 / math.pi)
        inner = k * (z + 0.044715 * z * z * z)
        t = math.tanh(inner)
        value = 0.5 * z * (1.0 + t)
        deriv = (0.5 * (1.0 + t)
                 + 0.5 * z * (1.0 - t * t) * k * (1.0 + 3.0 * 0.044715 * z * z))
        return value, deriv
    raise ValueError(f"unsupported activation {kind!r}")


def _loss_and_dz(loss: str, activation: str, z: float, prediction: float,
                 target: float) -> tuple[float, float]:
    """One example's loss and derivative with respect to the affine logit ``z``."""
    if loss == "bce":
        # TrainStepSpec admits only sigmoid+BCE.  Preserve the historic monitored-value
        # convention (clamped probabilities) and use the exact fused gradient.
        eps = 1e-12
        a = min(max(prediction, eps), 1.0 - eps)
        value = -(target * math.log(a) + (1.0 - target) * math.log(1.0 - a))
        return value, prediction - target
    if loss == "mse":
        err = prediction - target
        _value, deriv = _activation_and_derivative(activation, z)
        return err * err, 2.0 * err * deriv
    raise ValueError(f"unsupported loss {loss!r}")


@dataclass
class _OptimizerRuntime:
    """Mutable numeric twin of the optimizer resources carried by the claim graph."""

    name: str
    n_params: int
    state1: list | None = None
    state2: list | None = None
    step_index: int = 0

    def __post_init__(self) -> None:
        if self.name not in _OPTIMIZERS:
            raise ValueError(f"unknown optimizer {self.name!r}")
        if self.n_params < 1:
            raise ValueError("optimizer needs at least one parameter")
        if self.name in ("momentum", "rmsprop", "adam") and self.state1 is None:
            self.state1 = [0.0] * self.n_params
        if self.name == "adam" and self.state2 is None:
            self.state2 = [0.0] * self.n_params

    def update(self, params: list, grads: list, lr: float) -> None:
        """Advance once using the established M2 reference rule, preserving ``params`` identity."""
        if self.name == "sgd":
            new_params = sgd_step(params, grads, lr)
        elif self.name == "momentum":
            new_params, self.state1 = momentum_step(params, grads, self.state1, lr)
        elif self.name == "rmsprop":
            new_params, self.state1 = rmsprop_step(params, grads, self.state1, lr)
        else:
            new_params, self.state1, self.state2, self.step_index = adam_step(
                params, grads, self.state1, self.state2, self.step_index, lr)
        params[:] = new_params


def _step_kernels(spec: TrainStepSpec, st: dict) -> dict[int, "object"]:
    """The REAL numeric stage kernels keyed by `train_step_module` claim ids.

    The selected activation/loss/optimizer are the computation, not just provenance text.
    ``st["optimizer"]`` persists its claim-visible state across every dispatched batch.
    """
    nf, b = spec.n_features, spec.batch

    def forward():                                   # claim 1: z = X @ w + bias
        for i in range(b):
            st["z"][i] = sum(st["X"][i][j] * st["w"][j] for j in range(nf)) + st["w"][nf]

    def activation():                                # claim 2: selected activation
        for i in range(b):
            st["act"][i] = _activation_and_derivative(spec.activation, st["z"][i])[0]

    def loss_vec():                                  # claim 3: selected per-example loss
        for i in range(b):
            st["lossv"][i] = _loss_and_dz(
                spec.loss, spec.activation, st["z"][i], st["act"][i], st["y"][i])[0]

    def reduce_mean():                               # claim 4: loss = mean(lossv)
        st["loss"][0] = sum(st["lossv"]) / b

    def backward():                                  # claim 5: d(loss)/dz through selected activation
        dz = [_loss_and_dz(spec.loss, spec.activation, st["z"][i], st["act"][i],
                           st["y"][i])[1] for i in range(b)]
        for j in range(nf):
            st["grad"][j] = sum(dz[i] * st["X"][i][j] for i in range(b)) / b
        st["grad"][nf] = sum(dz) / b

    def update():                                    # claim 6: selected stateful optimizer
        st["optimizer"].update(st["w"], st["grad"], st["lr"])

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
    weights = list(w0)
    optimizer = _OptimizerRuntime(spec.optimizer, spec.n_params)

    def make_context(batch_spec: TrainStepSpec, module: Module) -> tuple[Module, dict, dict]:
        bs = batch_spec.batch
        state = {"X": [[0.0] * spec.n_features for _ in range(bs)], "y": [0.0] * bs,
                 "w": weights, "z": [0.0] * bs, "act": [0.0] * bs,
                 "lossv": [0.0] * bs, "loss": [0.0],
                 "grad": [0.0] * spec.n_params, "lr": lr, "optimizer": optimizer}
        return module, state, _step_kernels(batch_spec, state)

    contexts = {b: make_context(spec, m)}
    remainder = n % b
    if remainder:
        tail_spec = replace(spec, batch=remainder)
        tail_module = train_step_module(tail_spec)
        # The tail is planned and hydrated under its true shape before any numeric mutation.
        hydrate_train_step(tail_spec, h, theta, policy)
        contexts[remainder] = make_context(tail_spec, tail_module)
    losses: list = []
    manifests: list = []
    exec_orders: list = []
    for e in range(epochs):
        epoch_loss_sum = 0.0
        for lo in range(0, n, b):                    # deterministic full batches + one planned tail
            active = min(b, n - lo)
            batch_module, st, kernels = contexts[active]
            for i in range(active):
                st["X"][i] = list(X[lo + i])
                st["y"][i] = float(y[lo + i])
            r: ExecResult = execute(batch_module, kernels)
            exec_orders.append(list(r.order))
            epoch_loss_sum += st["loss"][0] * active
        losses.append(epoch_loss_sum / max(1, n))
        manifests.append(build_manifest(
            m, h, theta, policy,
            artifacts=(("epoch", e), ("topo_gen", pack.topo_gen),
                       ("map_gen", pack.map_gen), ("data_gen", pack.data_gen))))
    return PlannedTrainRun(weights=list(weights), losses=losses, manifests=tuple(manifests),
                           pack=pack, exec_orders=exec_orders)


# --- D1 step 5: overlap/EFT scheduling of the stage streams (software pipelining) --------------

from ..gem.schedule import GemSchedule, durations_from, execute_tokens, schedule_eft  # noqa: E402


def train_run_module(spec: TrainStepSpec, steps: int) -> Module:
    """The MULTI-STEP training run as ONE module: steps x six stage claims (fresh ids/phases
    per step) over the SAME resources, so the token DAG (`gem.async_plan`) carries the TRUE
    dependencies -- step i+1's forward awaits step i's weight update (RAW on W), while step
    i's METRICS TAIL (per-example loss + reduce) sits off the weight-critical path and may
    overlap step i's backward and the next step's forward. Software pipelining falls out of
    the dependency structure; no scheduler special case."""
    if steps < 1:
        raise ValueError(f"train_run_module needs steps >= 1; got {steps}")
    m = Module(name=f"train_run_{steps}x_{spec.activation}_{spec.loss}_{spec.optimizer}")
    for r in _resources(spec).values():
        m.add_resource(r)
    prev: tuple[int, ...] = ()
    pid = 0
    for s in range(steps):
        for stage in _step_claims(spec, base_id=s * 10):
            m.add_phase(Phase(phase_id=pid, deps=prev, claims=list(stage)))
            prev = (pid,)
            pid += 1
    return m


@dataclass(frozen=True)
class PipelineCertificate:
    """The D1 step-5 witness: the priced makespans of the SAME planned run under the three
    disciplines -- serial (no overlap: the sum of every stage's duration), phase-barriered
    EFT (`schedule_eft`), and token-DAG pipelined (`execute_tokens`). The pipelined makespan
    can only improve on the barriers (its degenerate case), and the barriers on serial."""

    steps: int
    serial: int
    barriered: int
    pipelined: int

    @property
    def overlap_win(self) -> int:
        return self.barriered - self.pipelined         # >= 0: what the token DAG buys

    @property
    def admitted(self) -> bool:
        return 0 < self.pipelined <= self.barriered <= self.serial


def schedule_train_run(spec: TrainStepSpec, steps: int, h: HProfile, theta: Theta,
                       policy: Policy = PERF) -> tuple[PipelineCertificate, GemSchedule]:
    """Price + place a multi-step training run: optimize the run module (per-claim step
    costs -> durations), schedule it phase-barriered AND token-pipelined, and certify the
    overlap win. Returns (certificate, the pipelined schedule)."""
    m = train_run_module(spec, steps)
    result = optimize(m, h, theta, policy)
    dur = durations_from(result)
    serial = sum(dur.values())
    barriered = schedule_eft(m, dur, target=h)
    pipelined = execute_tokens(m, dur, target=h)
    cert = PipelineCertificate(steps=steps, serial=serial, barriered=barriered.makespan,
                               pipelined=pipelined.makespan)
    return cert, pipelined


# --- D1 step 6: mini-batch STREAMS within a step (concurrent micro-batches) --------------------
#
# The autodiff closure is the enabler: the gradient DAG has a fixed vocabulary, so a streamed
# step hydrates to a StreamPack like any program. One step's batch splits into `streams` equal
# micro-batches; each stream runs forward -> activation -> loss -> reduce -> backward over its
# OWN resources (disjoint RIDs, so the token DAG overlaps the streams with zero scheduler
# change), all streams READ the same W (read-read never conflicts), the per-stream mean
# gradients combine (`reduce.grad_mean` -- awaits every stream), and exactly ONE weight update
# runs (the single-update law: mean-of-equal-split-means == the full-batch gradient, the
# numeric gate in test_train_graph). RID bands: stream s owns s*10 + {1..8} minus the shared
# W (rid 2); claim ids stream s stage k = s*10 + k; combine/update sit at streams*10 + 1/2.


def _stream_claims(spec: TrainStepSpec, s: int, mb: int) -> tuple[tuple[Claim, ...], ...]:
    """Stream s's five stages (forward .. backward) over its own RID band, micro-batch mb."""
    np_, c, o = spec.n_params, s * 10, s * 10
    x, z, act, y, lossv, loss, grad = o + _X, o + _Z, o + _ACT, o + _Y, o + _LOSSV, o + _LOSS, o + _GRAD
    backward_rd = (x, act, y) if spec.loss == "bce" else (x, z, act, y)
    stages = (
        Claim(id=c + 1, opcode=Opcode.T_MACC, lane=Lane.T, stride_class=StrideClass.TILE,
              count=mb, rd=(x, _W), wr=(z,), op="gem.matmul", domain=Domain.RAM,
              bounds="assumed_safe"),
        Claim(id=c + 2, opcode=Opcode.MUL, lane=Lane.U, stride_class=StrideClass.UNIT,
              count=mb, rd=(z,), wr=(act,), op=f"gem.activation:{spec.activation}",
              domain=Domain.RAM),
        Claim(id=c + 3, opcode=Opcode.MUL, lane=Lane.U, stride_class=StrideClass.UNIT,
              count=mb, rd=(act, y), wr=(lossv,), op=f"gem.loss:{spec.loss}",
              domain=Domain.RAM),
        Claim(id=c + 4, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
              count=mb, rd=(lossv,), wr=(loss,), op="reduce.loss_mean", domain=Domain.RAM),
        # backward reads the WHOLE micro-batch per output parameter (a structured tile
        # walk, like the forward): with mb < n_params the affine R7 extent model cannot
        # describe it, so the claim is `assumed_safe` -- the tensor-level validation is the
        # gem shape law (R22), the same posture as the forward matmul above.
        Claim(id=c + 5, opcode=Opcode.T_MACC, lane=Lane.T, stride_class=StrideClass.TILE,
              count=spec.n_params, rd=backward_rd, wr=(grad,), op="gem.autodiff",
              domain=Domain.RAM, bounds="assumed_safe"),
    )
    del np_
    return tuple((st,) for st in stages)


def train_stream_module(spec: TrainStepSpec, streams: int) -> Module:
    """ONE training step with `streams` concurrent micro-batch streams: per-stream stage
    chains over disjoint RID bands + the shared read-only W, a gradient-combine claim that
    awaits every stream, and the SINGLE weight update. The phase DAG states the true
    structure (parallel chains -> combine -> update); the token DAG discovers the same
    overlap from the hazards alone."""
    if streams < 1:
        raise ValueError(f"train_stream_module needs streams >= 1; got {streams}")
    if spec.batch % streams:
        raise ValueError(f"batch {spec.batch} not divisible by streams {streams} "
                         f"(equal micro-batches, or the gradient mean is not the batch mean)")
    mb = spec.batch // streams
    nf, np_ = spec.n_features, spec.n_params
    m = Module(name=f"train_stream_{streams}x_{spec.activation}_{spec.loss}_{spec.optimizer}")
    m.add_resource(Resource(rid=_W, domain=Domain.RAM, shape=(np_,), name="W"))
    for r in _optimizer_resources(spec):
        m.add_resource(r)
    gradc = streams * 10 + _GRAD                       # the combined gradient's RID
    for s in range(streams):
        o = s * 10
        for r in (Resource(rid=o + _X, domain=Domain.RAM, shape=(mb, nf), name=f"X{s}"),
                  Resource(rid=o + _Z, domain=Domain.RAM, shape=(mb,), name=f"Z{s}"),
                  Resource(rid=o + _ACT, domain=Domain.RAM, shape=(mb,), name=f"ACT{s}"),
                  Resource(rid=o + _Y, domain=Domain.RAM, shape=(mb,), name=f"Y{s}"),
                  Resource(rid=o + _LOSSV, domain=Domain.RAM, shape=(mb,), name=f"LOSSV{s}"),
                  Resource(rid=o + _LOSS, domain=Domain.RAM, shape=(1,), name=f"LOSS{s}"),
                  Resource(rid=o + _GRAD, domain=Domain.RAM, shape=(np_,), name=f"GRAD{s}")):
            m.add_resource(r)
    m.add_resource(Resource(rid=gradc, domain=Domain.RAM, shape=(np_,), name="GRADC"))
    last_pids = []
    pid = 0
    for s in range(streams):                           # parallel per-stream chains
        prev: tuple[int, ...] = ()
        for stage in _stream_claims(spec, s, mb):
            m.add_phase(Phase(phase_id=pid, deps=prev, claims=list(stage)))
            prev = (pid,)
            pid += 1
        last_pids.append(pid - 1)
    combine = Claim(id=streams * 10 + 1, opcode=Opcode.ADD, lane=Lane.U,
                    stride_class=StrideClass.UNIT, count=np_,
                    rd=tuple(s * 10 + _GRAD for s in range(streams)), wr=(gradc,),
                    op="reduce.grad_mean", domain=Domain.RAM)
    m.add_phase(Phase(phase_id=pid, deps=tuple(last_pids), claims=[combine]))
    opt_state = _optimizer_state_rids(spec)
    update = Claim(id=streams * 10 + 2, opcode=Opcode.ADD, lane=Lane.U,
                   stride_class=StrideClass.UNIT, count=np_, rd=(gradc, _W, *opt_state),
                   wr=(_W, *opt_state), op=f"gem.opt_step:{spec.optimizer}",
                   domain=Domain.RAM)
    m.add_phase(Phase(phase_id=pid + 1, deps=(pid,), claims=[update]))
    return m


@dataclass(frozen=True)
class StreamCertificate:
    """The D1 step-6 witness (the PipelineCertificate recipe over the STREAMED step): the
    priced makespans of the same streamed step under serial / phase-barriered EFT /
    token-DAG disciplines. The overlap win is what running the micro-batch streams
    concurrently buys over the barriered schedule."""

    streams: int
    serial: int
    barriered: int
    pipelined: int

    @property
    def overlap_win(self) -> int:
        return self.barriered - self.pipelined         # >= 0: what stream concurrency buys

    @property
    def admitted(self) -> bool:
        return 0 < self.pipelined <= self.barriered <= self.serial


def schedule_stream_step(spec: TrainStepSpec, streams: int, h: HProfile, theta: Theta,
                         policy: Policy = PERF) -> tuple[StreamCertificate, GemSchedule]:
    """Price + place one streamed step: optimize the stream module (per-claim costs ->
    durations), schedule it barriered AND token-pipelined, certify the concurrency win.
    Returns (certificate, the pipelined schedule)."""
    m = train_stream_module(spec, streams)
    result = optimize(m, h, theta, policy)
    dur = durations_from(result)
    serial = sum(dur.values())
    barriered = schedule_eft(m, dur, target=h)
    pipelined = execute_tokens(m, dur, target=h)
    cert = StreamCertificate(streams=streams, serial=serial, barriered=barriered.makespan,
                             pipelined=pipelined.makespan)
    return cert, pipelined


# --- D1 step 7: the STREAMED step EXECUTES (oracle now; the C rail twins it) --------------------


def _stream_kernels(spec: TrainStepSpec, streams: int, st: dict) -> dict[int, "object"]:
    """The streamed step's numeric kernels, keyed by `train_stream_module`'s claim ids
    (stream s stage k = s*10 + k; combine = streams*10 + 1; update = streams*10 + 2).
    Per-stream state lives in lists indexed by stream; W and the combined gradient are
    shared -- the resources of the streamed claim graph, materialized (the step-2 pattern)."""
    nf, mb = spec.n_features, spec.batch // streams

    def forward(s):                                  # s*10+1: z_s = X_s @ w + bias
        def k():
            for i in range(mb):
                st["z"][s][i] = (sum(st["X"][s][i][j] * st["w"][j] for j in range(nf))
                                 + st["w"][nf])
        return k

    def activation(s):                               # s*10+2: selected activation
        def k():
            for i in range(mb):
                st["act"][s][i] = _activation_and_derivative(
                    spec.activation, st["z"][s][i])[0]
        return k

    def loss_vec(s):                                 # s*10+3: selected per-example loss
        def k():
            for i in range(mb):
                st["lossv"][s][i] = _loss_and_dz(
                    spec.loss, spec.activation, st["z"][s][i], st["act"][s][i],
                    st["y"][s][i])[0]
        return k

    def reduce_mean(s):                              # s*10+4: loss_s = mean(lossv_s)
        def k():
            st["loss"][s] = sum(st["lossv"][s]) / mb
        return k

    def backward(s):                                 # s*10+5: selected loss/activation mean gradient
        def k():
            dz = [_loss_and_dz(spec.loss, spec.activation, st["z"][s][i],
                               st["act"][s][i], st["y"][s][i])[1] for i in range(mb)]
            for j in range(nf):
                st["grad"][s][j] = sum(dz[i] * st["X"][s][i][j]
                                       for i in range(mb)) / mb
            st["grad"][s][nf] = sum(dz) / mb
        return k

    def combine():                                   # streams*10+1: mean of the stream means
        for j in range(nf + 1):
            st["gradc"][j] = sum(st["grad"][s][j] for s in range(streams)) / streams

    def update():                                    # streams*10+2: the SINGLE selected optimizer step
        st["optimizer"].update(st["w"], st["gradc"], st["lr"])

    kernels: dict = {}
    for s in range(streams):
        kernels[s * 10 + 1] = forward(s)
        kernels[s * 10 + 2] = activation(s)
        kernels[s * 10 + 3] = loss_vec(s)
        kernels[s * 10 + 4] = reduce_mean(s)
        kernels[s * 10 + 5] = backward(s)
    kernels[streams * 10 + 1] = combine
    kernels[streams * 10 + 2] = update
    return kernels


def train_streamed(spec: TrainStepSpec, streams: int, X: list, y: list, w0: list, *,
                   epochs: int, lr: float, h: HProfile, theta: Theta,
                   policy: Policy = PERF) -> PlannedTrainRun:
    """REAL training on the STREAMED planned path (D1 step 7): every batch splits into
    `streams` equal micro-batches (stream s takes rows [s*mb, (s+1)*mb) -- the split the C
    twin reproduces), the GEM executor dispatches the per-stream stage kernels + the
    gradient combine + the SINGLE weight update in the streamed claim graph's phase order,
    and each epoch commits an R13 manifest tagged with the stream count. The trajectory
    matches `train_planned` to float round-off (mean-of-equal-split-means == the batch
    mean; only the summation ORDER differs). A divisible dataset tail gets its own reduced
    planned module; a ragged tail refuses before optimizer state can advance."""
    m = train_stream_module(spec, streams)           # validates streams/batch divisibility
    b = spec.batch
    n = len(X)
    remainder = n % b
    if remainder and remainder % streams:
        raise ValueError(f"dataset remainder {remainder} not divisible by streams {streams}; "
                         "equal micro-batches are required (no training was executed)")
    result = optimize(m, h, theta, policy)
    pack = hydrate(m, result, plan=m.name)
    weights = list(w0)
    optimizer = _OptimizerRuntime(spec.optimizer, spec.n_params)

    def make_context(batch_spec: TrainStepSpec, module: Module) -> tuple[Module, dict, dict]:
        mb = batch_spec.batch // streams
        state = {"X": [[[0.0] * spec.n_features for _ in range(mb)] for _ in range(streams)],
                 "y": [[0.0] * mb for _ in range(streams)],
                 "z": [[0.0] * mb for _ in range(streams)],
                 "act": [[0.0] * mb for _ in range(streams)],
                 "lossv": [[0.0] * mb for _ in range(streams)],
                 "loss": [0.0] * streams,
                 "grad": [[0.0] * spec.n_params for _ in range(streams)],
                 "gradc": [0.0] * spec.n_params, "w": weights, "lr": lr,
                 "optimizer": optimizer}
        return module, state, _stream_kernels(batch_spec, streams, state)

    contexts = {b: make_context(spec, m)}
    if remainder:
        tail_spec = replace(spec, batch=remainder)
        tail_module = train_stream_module(tail_spec, streams)
        tail_result = optimize(tail_module, h, theta, policy)
        hydrate(tail_module, tail_result, plan=tail_module.name)
        contexts[remainder] = make_context(tail_spec, tail_module)
    losses: list = []
    manifests: list = []
    exec_orders: list = []
    for e in range(epochs):
        epoch_loss_sum = 0.0
        for lo in range(0, n, b):                    # deterministic full batches + divisible tail
            active = min(b, n - lo)
            batch_module, st, kernels = contexts[active]
            mb = active // streams
            for s in range(streams):
                for i in range(mb):
                    st["X"][s][i] = list(X[lo + s * mb + i])
                    st["y"][s][i] = float(y[lo + s * mb + i])
            r: ExecResult = execute(batch_module, kernels)
            exec_orders.append(list(r.order))
            batch_loss = sum(st["loss"]) / streams          # == the full-batch mean loss
            epoch_loss_sum += batch_loss * active
        losses.append(epoch_loss_sum / max(1, n))
        manifests.append(build_manifest(
            m, h, theta, policy,
            artifacts=(("epoch", e), ("streams", streams), ("topo_gen", pack.topo_gen),
                       ("map_gen", pack.map_gen), ("data_gen", pack.data_gen))))
    return PlannedTrainRun(weights=list(weights), losses=losses, manifests=tuple(manifests),
                           pack=pack, exec_orders=exec_orders)


# --- D1 step 8: the stream COUNT is a plan decision ---------------------------------------------


@dataclass(frozen=True)
class StreamPlan:
    """The D1.8 witness: the swept (streams -> pipelined makespan) frontier and the
    CHOSEN count -- the optimizer picks the stream count from the cost model; the
    caller stops passing a knob. Ties break to FEWER streams (when concurrency buys
    nothing, the simpler plan wins)."""

    streams: int
    swept: tuple                       # ((streams, pipelined_makespan), ...) sweep order
    certificate: StreamCertificate     # the chosen count's full D1.6 certificate

    @property
    def admitted(self) -> bool:
        return (self.certificate.admitted
                and all(self.certificate.pipelined <= mk for _, mk in self.swept))


def plan_stream_count(spec: TrainStepSpec, h: HProfile, theta: Theta,
                      policy: Policy = PERF, *, max_streams: int | None = None) -> StreamPlan:
    """D1 step 8: sweep every divisor of the batch (equal micro-batches -- the
    single-update law's precondition) up to `max_streams`, price each streamed step
    through the D1.6 machinery, and CHOOSE the argmin pipelined makespan. Strict `<`
    keeps the first minimum, so ties break to fewer streams."""
    cap = max_streams if max_streams is not None else spec.batch
    divisors = [s for s in range(1, spec.batch + 1) if spec.batch % s == 0 and s <= cap]
    if not divisors:
        raise ValueError(f"plan_stream_count: no admissible stream count <= {cap}")
    swept: list = []
    best: tuple = ()
    for s in divisors:
        cert, _ = schedule_stream_step(spec, s, h, theta, policy)
        swept.append((s, cert.pipelined))
        if not best or cert.pipelined < best[1].pipelined:
            best = (s, cert)
    return StreamPlan(streams=best[0], swept=tuple(swept), certificate=best[1])
