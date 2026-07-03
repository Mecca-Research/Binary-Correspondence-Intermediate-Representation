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
              count=spec.n_params, rd=(x, act, y), wr=(grad,), op="gem.autodiff",
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
    update = Claim(id=streams * 10 + 2, opcode=Opcode.ADD, lane=Lane.U,
                   stride_class=StrideClass.UNIT, count=np_, rd=(gradc, _W), wr=(_W,),
                   op=f"gem.opt_step:{spec.optimizer}", domain=Domain.RAM)
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
