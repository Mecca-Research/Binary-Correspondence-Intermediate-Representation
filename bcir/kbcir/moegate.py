"""The learned Mixture-of-Experts gate: a GNN over the claim graph (the ensemble).

`classify(Theta)` is a hand-coded router: workload class -> portfolio expert. This
module replaces it with a *learned* router -- the literal "ensemble of specialized
compilers," where the experts are the (already-certified) portfolio policies and a
small Graph Neural Network over the claim graph chooses which one to deploy:

  * **GNN encoder** -- one message-passing layer over the claim graph (nodes =
    claims; edges = operand-sharing / hazard conflict / same phase). Node features
    are the claim's geometry (lane, stride, count, domain, hazard); a node mixes
    its own and its neighbours' features (a_i = x_i + mean_j x_j), a hardtanh layer
    `h_i = hardtanh(W a_i + b)` embeds it, and a mean readout pools the graph.
  * **Routing head** -- the graph embedding is concatenated with the Theta
    features and a softmax head `softmax(U[g;theta] + c)` produces a distribution
    over experts; argmax routes. This is a Mixture-of-Experts *gate*: a learned
    selection among frozen experts, exactly the design of sparse-MoE transformers.
  * **Trained on the regret ledger** -- the label for each (module, Theta) episode
    is the hindsight-best expert (least regret under the neutral judge metric,
    `kbcir.regret`). Exact softmax-cross-entropy SGD (hand-rolled, no autograd, no
    third-party deps): the head and the GNN layer are learned end-to-end (the
    inputs/graph are fixed features, so gradients flow only into W/b/U/c).

DEPLOYMENT DISCIPLINE (LangRef Sec. 13): the gate is the *safe* learning regime --
it only *selects among already-certified experts*, never emits a new cost table or
policy. Training is L2/L3 offline (floats); the gate then **freezes** to Q8
integers (`FrozenGate`, exact integer forward with a hardtanh clamp -- no floats,
deterministic across hosts), deploys only behind the **replay gate**
(`gate_replay_gate`: counterfactual no-regression vs the incumbent `classify`
router over logged episodes), and is witnessed by **R13** (`bcir.kbcir.moe_gate`).
The propose-verify discipline: the network proposes a route, the verifier disposes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model import Claim, Domain, Lane, Module, StrideClass
from .cost import Theta
from .portfolio import PolicyPortfolio, ReplayCertificate
from .weights import PERF, Policy

Q8 = 256
_THETA_FEATS = 5         # thermal, power, mem_pressure, contention, voltage (each /100)
_NODE_FEATS = 15


# --- claim-graph feature extraction (deterministic) ------------------------------

def node_features(c: Claim) -> list[float]:
    """The fixed geometry features of a claim node (length _NODE_FEATS)."""
    import math
    lane = [c.lane == L for L in (Lane.U, Lane.UX, Lane.T, Lane.GGG, Lane.A)]
    stride = [c.stride_class == S for S in (StrideClass.UNIT, StrideClass.STRIDED,
              StrideClass.CACHELINE, StrideClass.TILE, StrideClass.RANDOM)]
    return [1.0,
            *[float(x) for x in lane],
            *[float(x) for x in stride],
            math.log2(max(1, c.count)) / 16.0,
            float(c.domain == Domain.HBM),
            float(c.domain == Domain.RAM),
            float(c.hazard in ("atomic", "barriered"))]


def theta_features(theta: Theta) -> list[float]:
    return [theta.thermal / 100.0, theta.power / 100.0, theta.mem_pressure / 100.0,
            theta.contention / 100.0, theta.voltage / 100.0]


def _claims(module: Module) -> list[Claim]:
    return [c for ph in module.phases for c in ph.claims]


def _aggregated(module: Module) -> list[list[float]]:
    """Per-node a_i = x_i + mean_{j in N(i)} x_j over the claim graph (edges =
    shared RID / hazard conflict / same phase)."""
    claims = _claims(module)
    phase_of = {c.id: ph.phase_id for ph in module.phases for c in ph.claims}
    feats = [node_features(c) for c in claims]
    n = len(claims)
    out: list[list[float]] = []
    for i, ci in enumerate(claims):
        ri = set(ci.rd) | set(ci.wr)
        nbrs = []
        for j, cj in enumerate(claims):
            if i == j:
                continue
            rj = set(cj.rd) | set(cj.wr)
            if (ri & rj) or phase_of[ci.id] == phase_of[cj.id]:
                nbrs.append(j)
        a = list(feats[i])
        if nbrs:
            for k in range(_NODE_FEATS):
                a[k] += sum(feats[j][k] for j in nbrs) / len(nbrs)
        out.append(a)
    return out


# --- math helpers (pure Python, no deps) -----------------------------------------

def _hardtanh(x: float) -> float:
    return -1.0 if x < -1.0 else (1.0 if x > 1.0 else x)


def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    ex = [pow(2.718281828459045, v - m) for v in logits]
    s = sum(ex) or 1.0
    return [e / s for e in ex]


def _softmax_t(logits: list[float], temperature: float) -> list[float]:
    """Temperature-scaled softmax: high T -> flat (explore all paths), T -> 0 ->
    one-hot (exploit). The fuzzy/continuous routing knob."""
    t = max(1e-9, float(temperature))
    return _softmax([v / t for v in logits])


def harden(distribution: list[float], threshold: float = 0.7) -> tuple[bool, "int | None"]:
    """Collapse a fuzzy routing distribution to a single expert once one path
    dominates (max weight >= `threshold`) -- the end of exploration. Returns
    (hardened, expert_index): if no path is confident enough, (False, None) and the
    caller keeps blending. Deterministic (argmax, ties -> lowest index)."""
    bi, best = 0, distribution[0]
    for k in range(1, len(distribution)):
        if distribution[k] > best:
            bi, best = k, distribution[k]
    return (best >= threshold, bi if best >= threshold else None)


def _lcg(seed: int):
    state = seed & 0xFFFFFFFFFFFFFFFF
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        yield ((state >> 33) / float(1 << 31) - 1.0) * 0.1   # in [-0.1, 0.1]


# --- the trained gate ------------------------------------------------------------

@dataclass
class GNNGate:
    """A trained GNN routing head over the claim graph (float; the training form)."""

    experts: list                      # list[Policy], expert k = experts[k]
    W: list                            # H x F
    b: list                            # H
    U: list                            # K x (H + T)
    c: list                            # K
    hidden: int

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def _embed(self, module: Module) -> list[float]:
        agg = _aggregated(module)
        H = self.hidden
        acc = [0.0] * H
        for a in agg:
            for h in range(H):
                pre = self.b[h] + sum(self.W[h][k] * a[k] for k in range(_NODE_FEATS))
                acc[h] += _hardtanh(pre)
        n = max(1, len(agg))
        return [v / n for v in acc]

    def distribution(self, module: Module, theta: Theta) -> list[float]:
        z = self._embed(module) + theta_features(theta)
        logits = [self.c[k] + sum(self.U[k][j] * z[j] for j in range(len(z)))
                  for k in range(self.num_experts)]
        return _softmax(logits)

    def route(self, module: Module, theta: Theta) -> int:
        p = self.distribution(module, theta)
        best, bi = p[0], 0
        for k in range(1, len(p)):
            if p[k] > best:
                best, bi = p[k], k
        return bi

    def route_policy(self, module: Module, theta: Theta) -> Policy:
        return self.experts[self.route(module, theta)]

    # --- fuzzy / continuous routing (exploration before hardening) ---------------

    def route_fuzzy(self, module: Module, theta: Theta, temperature: float = 1.0) -> list[float]:
        """Continuous edge weights over the experts at a given temperature -- data
        flows *partially* through every path so gradients can be gathered from all
        optimizations at once. Anneal `temperature` toward 0 to harden the choice."""
        z = self._embed(module) + theta_features(theta)
        logits = [self.c[k] + sum(self.U[k][j] * z[j] for j in range(len(z)))
                  for k in range(self.num_experts)]
        return _softmax_t(logits, temperature)

    def blend(self, module: Module, theta: Theta,
              temperature: float = 1.0) -> list[tuple[Policy, float]]:
        """The fuzzy route as (expert, weight) pairs -- the partial flows."""
        return list(zip(self.experts, self.route_fuzzy(module, theta, temperature)))


def _new_gate(experts: list, hidden: int, seed: int) -> GNNGate:
    rng = _lcg(seed)
    F, T, K = _NODE_FEATS, _THETA_FEATS, len(experts)
    W = [[next(rng) for _ in range(F)] for _ in range(hidden)]
    b = [0.0] * hidden
    U = [[next(rng) for _ in range(hidden + T)] for _ in range(K)]
    c = [0.0] * K
    return GNNGate(experts=list(experts), W=W, b=b, U=U, c=c, hidden=hidden)


def train_gate(samples: list, experts: list, hidden: int = 8, epochs: int = 300,
               lr: float = 0.2, seed: int = 0x9E3779B97F4A7C15) -> GNNGate:
    """Exact softmax-cross-entropy SGD. `samples` = [(module, theta, expert_idx)].
    Deterministic: seeded init, fixed sample order, fixed lr/epochs."""
    gate = _new_gate(experts, hidden, seed)
    H, T, K = hidden, _THETA_FEATS, len(experts)
    for _ in range(epochs):
        for module, theta, label in samples:
            agg = _aggregated(module)
            # forward, caching per-node pre-activations for the backward pass
            hsum = [0.0] * H
            pre_cache = []
            for a in agg:
                pre = [gate.b[h] + sum(gate.W[h][k] * a[k] for k in range(_NODE_FEATS))
                       for h in range(H)]
                pre_cache.append(pre)
                for h in range(H):
                    hsum[h] += _hardtanh(pre[h])
            n = max(1, len(agg))
            g = [v / n for v in hsum]
            z = g + theta_features(theta)
            logits = [gate.c[k] + sum(gate.U[k][j] * z[j] for j in range(H + T))
                      for k in range(K)]
            p = _softmax(logits)
            dlogits = [p[k] - (1.0 if k == label else 0.0) for k in range(K)]
            # head grads + dz
            dz = [0.0] * (H + T)
            for k in range(K):
                gk = dlogits[k]
                gate.c[k] -= lr * gk
                for j in range(H + T):
                    dz[j] += gate.U[k][j] * gk
                    gate.U[k][j] -= lr * gk * z[j]
            # back through the readout + hardtanh into W, b
            dg = dz[:H]
            for i, a in enumerate(agg):
                for h in range(H):
                    if -1.0 < pre_cache[i][h] < 1.0:        # hardtanh gradient
                        dpre = (dg[h] / n)
                        gate.b[h] -= lr * dpre
                        for k in range(_NODE_FEATS):
                            gate.W[h][k] -= lr * dpre * a[k]
    return gate


# --- the frozen gate (Q8 integer forward; the certified-rail artifact) -----------

@dataclass(frozen=True)
class FrozenGate:
    """A gate frozen to Q8 integers: deterministic integer routing, no floats."""

    experts: list
    Wq: list
    bq: list
    Uq: list
    cq: list
    hidden: int
    gate_gen: int = 1

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    @property
    def fingerprint(self) -> int:
        """A stable provenance hash of the frozen parameters (witnessed by R13)."""
        flat = [v for row in self.Wq for v in row] + list(self.bq) + \
               [v for row in self.Uq for v in row] + list(self.cq)
        h = 1469598103934665603
        for v in flat:
            h = ((h ^ (v & 0xFFFFFFFF)) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return h

    def logits(self, module: Module, theta: Theta) -> list[int]:
        """Integer routing logits over the experts (no floats; the shared forward)."""
        H, T = self.hidden, _THETA_FEATS
        agg = _aggregated(module)
        xq = [[round(v * Q8) for v in a] for a in agg]
        hsum = [0] * H
        for a in xq:
            for h in range(H):
                pre = self.bq[h] + (sum(self.Wq[h][k] * a[k]
                                        for k in range(_NODE_FEATS)) >> 8)
                hsum[h] += -Q8 if pre < -Q8 else (Q8 if pre > Q8 else pre)
        n = max(1, len(agg))
        g = [s // n for s in hsum]
        zq = g + [round(t * Q8) for t in theta_features(theta)]
        return [self.cq[k] + (sum(self.Uq[k][j] * zq[j] for j in range(H + T)) >> 8)
                for k in range(self.num_experts)]

    def route(self, module: Module, theta: Theta) -> int:
        logits = self.logits(module, theta)
        best, bi = logits[0], 0
        for k in range(1, len(logits)):
            if logits[k] > best:
                best, bi = logits[k], k
        return bi

    def distribution(self, module: Module, theta: Theta, temperature: int = 1) -> list[int]:
        """Q8 fuzzy weights over experts (sum == 256), deterministic integer -- the
        frozen gate's continuous edge directions, no floats. A monotone normalization
        of the logits (higher logit => more flow); `temperature` >= 1 flattens it
        (divides the logit spread). `argmax(distribution) == route`."""
        lg = self.logits(module, theta)
        lo = min(lg)
        t = max(1, int(temperature))
        shifted = [(v - lo) // t + 1 for v in lg]      # strictly positive, monotone
        tot = sum(shifted) or 1
        w = [(s * Q8) // tot for s in shifted]
        bi = max(range(len(w)), key=lambda k: (w[k], -k))
        w[bi] += Q8 - sum(w)                            # exact: weights sum to Q8
        return w

    def route_policy(self, module: Module, theta: Theta) -> Policy:
        return self.experts[self.route(module, theta)]


def freeze(gate: GNNGate, gate_gen: int = 1) -> FrozenGate:
    q = lambda m: [[round(v * Q8) for v in row] for row in m]
    return FrozenGate(experts=list(gate.experts),
                      Wq=q(gate.W), bq=[round(v * Q8) for v in gate.b],
                      Uq=q(gate.U), cq=[round(v * Q8) for v in gate.c],
                      hidden=gate.hidden, gate_gen=max(1, gate_gen))


# --- training labels from the regret ledger --------------------------------------

def ledger_labels(module: Module, h, thetas: list, portfolio: PolicyPortfolio,
                  judge: Policy = PERF) -> list:
    """Per-episode label = the hindsight-best expert (least scheduled cost under
    the neutral judge). The training signal IS the regret ledger."""
    from ..gem.overlap import price_scheduled
    from .realize import optimize

    experts = [e.policy for _, e in sorted(portfolio.entries.items()) if e.certified]
    samples = []
    for theta in thetas:
        best_i, best_m = 0, None
        for i, pol in enumerate(experts):
            m = price_scheduled(module, optimize(module, h, theta, pol), h, theta, judge).makespan
            if best_m is None or m < best_m:
                best_i, best_m = i, m
        samples.append((module, theta, best_i))
    return samples, experts


# --- the replay gate for routers (deploys the gate behind a no-regression cert) --

def gate_replay_gate(module: Module, h, thetas: list, gate, portfolio: PolicyPortfolio,
                     judge: Policy = PERF) -> ReplayCertificate:
    """Counterfactual no-regression check of the learned router vs the incumbent
    `classify` router, judged under the neutral metric M(pi, Theta). Admit iff the
    learned gate never prices worse than `portfolio.select` over the episodes."""
    from ..gem.overlap import price_scheduled
    from .realize import optimize

    regressions = 0
    for theta in thetas:
        learned = gate.route_policy(module, theta)
        incumbent = portfolio.select(theta)
        m_l = price_scheduled(module, optimize(module, h, theta, learned), h, theta, judge).makespan
        m_i = price_scheduled(module, optimize(module, h, theta, incumbent), h, theta, judge).makespan
        if m_l > m_i:
            regressions += 1
    return ReplayCertificate(candidate="moe_gate", incumbent="classify",
                             episodes=len(thetas), regressions=regressions)
