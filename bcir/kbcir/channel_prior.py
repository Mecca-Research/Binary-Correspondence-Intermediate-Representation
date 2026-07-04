"""D3 slice 3 (ML/AI roadmap §8.2): CHANNEL-CHOICE priors + per-shape-class tables.

`plan_calling_side` / `channels.orchestrate` price EVERY suitable channel of the tower with a
full `optimize()` run per channel (the exhaustive ground truth). This module adds the L1 layer
on top, the `tile_prior` discipline function-for-function:

  * a PER-SHAPE-CLASS TABLE (`shape_class` -- the log2-bucket key tile_prior minted for exactly
    this purpose) mapping a class to its exhaustively-verified winning channel: a table hit
    answers the channel choice with ZERO per-channel optimize() runs -- the certified reduction;
  * a Q8-frozen logistic prior over CHEAP features (profile constants only, no pricing call)
    that orders the pricing on a table MISS -- an anytime warm-start; the miss still prices
    every suitable channel, so the returned choice is ALWAYS the exhaustive argmin (exactness
    is never delegated to the learned layer);
  * a certificate (`ChannelPriorCertificate`, the AccelCertificate pattern): guided choice ==
    exhaustive argmin over held-out shapes, mismatches MUST be 0, the pricing reduction is the
    witness -- a poisoned table is CAUGHT, not trusted;
  * a persisted envelope (kind `bcir.channel_prior`, the 0.4b decision-record pattern) tied to
    the TOWER: every (channel name, cal_gen) pair is recorded, and a load under a different
    tower or a recalibrated channel REFUSES loudly (a stale table's zero-cost answer is an
    unearned claim -- the tile-prior staleness law, verbatim).

Cost-side module: imports no verifier (two-truth); opt-in -- `plan_calling_side` itself is
untouched (non-disturbance, the tile-prior precedent)."""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..channels import channel_suits
from .calling_side import _channel_cost, _gemm_claim
from .cost import Theta
from .tile_prior import shape_class
from .weights import PERF, Policy

_Q8 = 256


def channel_features(M: int, N: int, K: int, ch) -> list[float]:
    """8 CHEAP per-(shape, channel) features -- shape log2 buckets + the channel profile's
    published constants. NO optimize()/cost_of call (the whole point is to rank without
    pricing); clamped like tile_features."""
    p = ch.profile
    return [
        min(4.0, shape_class(M, N, K)[0] / 8.0),
        min(4.0, shape_class(M, N, K)[1] / 8.0),
        min(4.0, shape_class(M, N, K)[2] / 8.0),
        min(2.0, p.mem_channels / 32.0),               # bandwidth-stream capacity
        min(2.0, p.gather_penalty / 32.0),             # random-access cost (lower is better)
        min(2.0, max(p.lane_widths) / 16.0),           # widest vector lane
        1.0 if ch.kind == "cpu" else 0.0,              # the host tie-break axis
        1.0,                                           # bias
    ]


class LearnedChannelPrior:
    """A logistic ranker over `channel_features` (the tile_prior/train_ranker recipe)."""

    N_FEATURES = 8

    def __init__(self) -> None:
        self.w = [0.0] * self.N_FEATURES

    def score(self, feats: list[float]) -> float:
        z = sum(w * f for w, f in zip(self.w, feats))
        z = max(-30.0, min(30.0, z))
        import math
        return 1.0 / (1.0 + math.exp(-z))

    def freeze(self) -> "FrozenChannelPrior":
        return FrozenChannelPrior(wq=tuple(round(v * _Q8) for v in self.w))


@dataclass(frozen=True)
class FrozenChannelPrior:
    """The Q8 integer table (the L1 artifact): deterministic, no floats in the order path."""

    wq: tuple

    def order(self, cands: list, feats: list[list[float]]) -> list:
        """Candidates sorted by descending integer score; index tie-break (deterministic)."""
        scored = []
        for i, c in enumerate(cands):
            z = sum(self.wq[k] * round(feats[i][k] * _Q8) for k in range(len(self.wq))) >> 8
            scored.append((-z, i, c))
        return [c for _, _, c in sorted(scored, key=lambda t: (t[0], t[1]))]


def prior_channel_samples(shapes: list, channels: list, theta: Theta,
                          policy: Policy = PERF) -> list:
    """(features, label) pairs from the EXACT per-channel pricing: label 1 iff the channel is
    the exhaustive argmin under the calling-side key (cost, host-tie, name) for that shape."""
    samples: list = []
    for (M, N, K) in shapes:
        claim = _gemm_claim(M, N, K)
        suit = [c for c in channels if channel_suits(claim, c)]
        if not suit:
            continue
        priced = [(c, _channel_cost(M, N, K, c, theta, policy)) for c in suit]
        best, _ = min(priced, key=lambda cc: (cc[1], 0 if cc[0].kind == "cpu" else 1, cc[0].name))
        for c in suit:
            samples.append((channel_features(M, N, K, c), 1.0 if c.name == best.name else 0.0))
    return samples


def train_channel_prior(samples: list, *, epochs: int = 200,
                        lr: float = 0.5) -> LearnedChannelPrior:
    """Deterministic logistic SGD (the train_ranker recipe, verbatim from tile_prior)."""
    prior = LearnedChannelPrior()
    for _ in range(epochs):
        for feats, label in samples:
            p = prior.score(feats)
            g = p - label
            for k in range(prior.N_FEATURES):
                prior.w[k] -= lr * g * feats[k]
    return prior


def build_channel_table(shapes: list, channels: list, theta: Theta,
                        policy: Policy = PERF) -> dict:
    """The per-shape-class table: class -> the exhaustively-priced winning channel NAME.
    Ground truth by construction (every entry is an exhaustive argmin under the same key
    plan_calling_side uses); generalization to unseen shapes IN a trained class is what the
    certificate measures (mismatches MUST be 0)."""
    table: dict = {}
    for (M, N, K) in shapes:
        claim = _gemm_claim(M, N, K)
        suit = [c for c in channels if channel_suits(claim, c)]
        if not suit:
            continue
        priced = [(c, _channel_cost(M, N, K, c, theta, policy)) for c in suit]
        best, _ = min(priced, key=lambda cc: (cc[1], 0 if cc[0].kind == "cpu" else 1, cc[0].name))
        table[shape_class(M, N, K)] = best.name
    return table


def guided_plan_channel(M: int, N: int, K: int, channels: list, theta: Theta,
                        policy: Policy = PERF, *, table: dict | None = None,
                        prior: FrozenChannelPrior | None = None):
    """The guided channel choice: a table HIT answers with ZERO pricings (the class's verified
    winner, provided it still suits); a MISS prices every suitable channel -- in prior order
    when a prior is given (anytime warm-start) -- and returns the exhaustive argmin. Returns
    (channel, n_priced). Worst case (no table, no prior): exactly plan_calling_side's
    exhaustive pricing."""
    claim = _gemm_claim(M, N, K)
    suit = [c for c in channels if channel_suits(claim, c)]
    if not suit:
        raise ValueError(f"no suitable channel in the tower for a {M}x{N}x{K} gemm")
    if table:
        hit = table.get(shape_class(M, N, K))
        if hit is not None:
            for c in suit:
                if c.name == hit:
                    return c, 0                        # the certified zero-pricing answer
    cands = suit
    if prior is not None:
        feats = [channel_features(M, N, K, c) for c in suit]
        cands = prior.order(suit, feats)
    priced = [(c, _channel_cost(M, N, K, c, theta, policy)) for c in cands]
    best, _ = min(priced, key=lambda cc: (cc[1], 0 if cc[0].kind == "cpu" else 1, cc[0].name))
    return best, len(priced)


@dataclass(frozen=True)
class ChannelPriorCertificate:
    """The safety witness (the AccelCertificate pattern): over held-out shapes the guided
    choice must equal the exhaustive argmin -- `mismatches` MUST be 0 -- and the reduction
    in per-channel optimize() pricings is the earned win."""

    checked: int
    mismatches: int
    priced_guided: int
    priced_exhaustive: int

    @property
    def admitted(self) -> bool:
        return self.checked >= 1 and self.mismatches == 0

    @property
    def reduction(self) -> float:
        return 1.0 - self.priced_guided / max(1, self.priced_exhaustive)


def channel_prior_certificate(shapes: list, channels: list, theta: Theta,
                              policy: Policy = PERF, *, table: dict,
                              prior: FrozenChannelPrior | None = None) -> ChannelPriorCertificate:
    """Certify the guided path against the exhaustive path per held-out shape."""
    checked = mismatches = pg = pe = 0
    for (M, N, K) in shapes:
        guided, n = guided_plan_channel(M, N, K, channels, theta, policy,
                                        table=table, prior=prior)
        exact, ne = guided_plan_channel(M, N, K, channels, theta, policy)   # no table: exhaustive
        checked += 1
        pg += n
        pe += ne
        if guided.name != exact.name:
            mismatches += 1
    return ChannelPriorCertificate(checked=checked, mismatches=mismatches,
                                   priced_guided=pg, priced_exhaustive=pe)


# --- the persisted envelope (D3: "per-shape-class tables persisted alongside cal_gen") --------

PRIOR_KIND = "bcir.channel_prior"
PRIOR_SCHEMA = 1


def _tower_id(channels: list) -> list:
    """The tower identity the envelope ties to: sorted (name, cal_gen) pairs -- a renamed,
    re-towered, or RECALIBRATED channel changes it, and the load refuses."""
    return sorted([c.name, int(c.profile.cal_gen)] for c in channels)


def save_channel_prior(path: str, prior: FrozenChannelPrior, table: dict,
                       channels: list) -> None:
    """Persist prior + table + the tower identity (sorted-keys JSON, the 0.4b pattern)."""
    doc = {"kind": PRIOR_KIND, "schema": PRIOR_SCHEMA, "tower": _tower_id(channels),
           "wq": list(prior.wq),
           "table": {",".join(str(v) for v in k): name for k, name in sorted(table.items())}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)


def load_channel_prior(path: str, expect_channels: list):
    """Load (prior, table) -- REFUSING loudly (ValueError) on: a wrong document kind, a NEWER
    schema, or a tower mismatch (different channels, or any channel recalibrated to a new
    cal_gen -- the STALE case). A stale table's zero-pricing answer is an unearned claim."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("kind") != PRIOR_KIND:
        raise ValueError(f"not a {PRIOR_KIND} document (kind={doc.get('kind')!r})")
    if int(doc.get("schema", 0)) > PRIOR_SCHEMA:
        raise ValueError(f"channel-prior schema v{doc['schema']} is newer than this build's "
                         f"v{PRIOR_SCHEMA}; upgrade BCIR to load this prior")
    live = _tower_id(expect_channels)
    saved = [list(p) for p in doc.get("tower", [])]
    if sorted(n for n, _ in saved) != sorted(n for n, _ in live):
        raise ValueError(f"channel prior was trained for tower "
                         f"{[n for n, _ in saved]}, not {[n for n, _ in live]} -- retrain")
    for (sn, sg), (ln, lg) in zip(sorted(saved), sorted(live)):
        if sg != lg:
            raise ValueError(f"channel prior is STALE: trained under {sn}@cal_gen {sg}, "
                             f"the live tower has {ln}@cal_gen {lg} -- retrain")
    prior = FrozenChannelPrior(wq=tuple(int(v) for v in doc["wq"]))
    table = {tuple(int(v) for v in k.split(",")): name
             for k, name in doc.get("table", {}).items()}
    return prior, table
