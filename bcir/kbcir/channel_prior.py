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
import os
import tempfile
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

    def __post_init__(self) -> None:
        if (len(self.wq) != LearnedChannelPrior.N_FEATURES or
                any(isinstance(v, bool) or not isinstance(v, int) or abs(v) > (1 << 31) - 1
                    for v in self.wq)):
            raise ValueError("FrozenChannelPrior requires exactly 8 signed 32-bit weights")

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
_MAX_PRIOR_BYTES = 1 << 20
_MAX_TABLE_ENTRIES = 1 << 16


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _prior_document(doc: object) -> tuple[list, tuple[int, ...], dict]:
    if not isinstance(doc, dict):
        raise ValueError("channel prior must be a JSON object")
    fields = {"kind", "schema", "tower", "wq", "table"}
    if set(doc) != fields:
        raise ValueError(f"channel prior fields must be exactly {sorted(fields)}")
    if doc["kind"] != PRIOR_KIND:
        raise ValueError(f"not a {PRIOR_KIND} document (kind={doc['kind']!r})")
    schema = doc["schema"]
    if isinstance(schema, bool) or not isinstance(schema, int):
        raise ValueError("channel-prior schema must be an integer")
    if schema > PRIOR_SCHEMA:
        raise ValueError(f"channel-prior schema v{schema} is newer than this build's "
                         f"v{PRIOR_SCHEMA}; upgrade BCIR to load this prior")
    if schema != PRIOR_SCHEMA:
        raise ValueError(f"unsupported channel-prior schema v{schema}")

    raw_tower = doc["tower"]
    if not isinstance(raw_tower, list) or not raw_tower or len(raw_tower) > 256:
        raise ValueError("channel-prior tower must contain 1..256 entries")
    tower = []
    for i, pair in enumerate(raw_tower):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"channel-prior tower[{i}] must be [name, cal_gen]")
        name, generation = pair
        if (not isinstance(name, str) or not name or len(name) > 128 or
                any(ord(ch) < 0x20 for ch in name)):
            raise ValueError(f"channel-prior tower[{i}] has an invalid name")
        if (isinstance(generation, bool) or not isinstance(generation, int) or
                generation < 0 or generation > (1 << 63) - 1):
            raise ValueError(f"channel-prior tower[{i}] has an invalid cal_gen")
        tower.append([name, generation])
    if len({name for name, _ in tower}) != len(tower):
        raise ValueError("channel-prior tower contains duplicate channel names")
    if tower != sorted(tower):
        raise ValueError("channel-prior tower must be in canonical sorted order")

    raw_weights = doc["wq"]
    if not isinstance(raw_weights, list):
        raise ValueError("channel-prior wq must be an array")
    prior = FrozenChannelPrior(tuple(raw_weights))

    raw_table = doc["table"]
    if not isinstance(raw_table, dict) or len(raw_table) > _MAX_TABLE_ENTRIES:
        raise ValueError(f"channel-prior table must have at most {_MAX_TABLE_ENTRIES} entries")
    names = {name for name, _ in tower}
    table = {}
    for raw_key, name in raw_table.items():
        if not isinstance(raw_key, str) or not isinstance(name, str) or name not in names:
            raise ValueError("channel-prior table values must name a channel in the saved tower")
        parts = raw_key.split(",")
        if len(parts) != 3:
            raise ValueError(f"invalid channel-prior shape key {raw_key!r}")
        try:
            key = tuple(int(part) for part in parts)
        except ValueError as exc:
            raise ValueError(f"invalid channel-prior shape key {raw_key!r}") from exc
        if any(v < 1 or v > 64 for v in key) or raw_key != ",".join(str(v) for v in key):
            raise ValueError(f"non-canonical channel-prior shape key {raw_key!r}")
        table[key] = name
    return tower, prior.wq, table


def _tower_id(channels: list) -> list:
    """The tower identity the envelope ties to: sorted (name, cal_gen) pairs -- a renamed,
    re-towered, or RECALIBRATED channel changes it, and the load refuses."""
    return sorted([c.name, int(c.profile.cal_gen)] for c in channels)


def save_channel_prior(path: str, prior: FrozenChannelPrior, table: dict,
                       channels: list) -> None:
    """Persist a validated prior atomically; never leave a partial installable artifact."""
    doc = {"kind": PRIOR_KIND, "schema": PRIOR_SCHEMA, "tower": _tower_id(channels),
           "wq": list(prior.wq),
           "table": {",".join(str(v) for v in k): name for k, name in sorted(table.items())}}
    _prior_document(doc)
    payload = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, temporary = tempfile.mkstemp(prefix=".bcir-channel-prior-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_channel_prior(path: str, expect_channels: list):
    """Load (prior, table) -- REFUSING loudly (ValueError) on: a wrong document kind, a NEWER
    schema, or a tower mismatch (different channels, or any channel recalibrated to a new
    cal_gen -- the STALE case). A stale table's zero-pricing answer is an unearned claim."""
    try:
        with open(path, "rb") as f:
            raw = f.read(_MAX_PRIOR_BYTES + 1)
        if len(raw) > _MAX_PRIOR_BYTES:
            raise ValueError(f"channel prior exceeds {_MAX_PRIOR_BYTES} bytes")
        doc = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, RecursionError) as exc:
        raise ValueError(f"invalid channel prior {path!r}: {exc}") from exc
    saved, weights, table = _prior_document(doc)
    live = _tower_id(expect_channels)
    if len({name for name, _ in live}) != len(live):
        raise ValueError("live channel tower contains duplicate names")
    if sorted(n for n, _ in saved) != sorted(n for n, _ in live):
        raise ValueError(f"channel prior was trained for tower "
                         f"{[n for n, _ in saved]}, not {[n for n, _ in live]} -- retrain")
    for (sn, sg), (ln, lg) in zip(sorted(saved), sorted(live)):
        if sg != lg:
            raise ValueError(f"channel prior is STALE: trained under {sn}@cal_gen {sg}, "
                             f"the live tower has {ln}@cal_gen {lg} -- retrain")
    prior = FrozenChannelPrior(wq=weights)
    return prior, table


# --- D3.4: the table wired into orchestrate (opt-in) + the tower certificate --------------------


def orchestrate_guided(module, channels: list, theta: Theta, policy: Policy = PERF, *,
                       table: dict, dims: dict):
    """D3.4: the per-shape-class table wired into the TOWER pass -- opt-in, so
    `channels.orchestrate` itself stays untouched (the module's own non-disturbance
    precedent). `dims` maps a gemm claim id to its (M, N, K) -- dimensions the module's
    author already knows. Every gemm claim whose class HITS the table pins its verified
    winner into a REDUCED tower; channels that suit the module's OTHER claims always
    stay (their placement remains the exhaustive greedy pass); ANY miss falls back to
    the full tower (exactness is never delegated to the learned layer). Returns
    (plan, n_channels_priced): the win is fewer whole-module optimize() runs -- the
    dominant cost of `orchestrate` -- and the certificate below is the safety witness
    (a poisoned table CHANGES PLACEMENTS and is caught, mismatches must be 0)."""
    from ..channels import orchestrate
    from .realize import _flatten
    keep: set = set()
    full = False
    for _, claim in _flatten(module):
        if claim.id in dims:
            M, N, K = dims[claim.id]
            hit = table.get(shape_class(M, N, K))
            suits = {ch.name for ch in channels if channel_suits(claim, ch)}
            if hit is not None and hit in suits:
                keep.add(hit)                          # the certified winner, pinned
            else:
                full = True                            # a miss prices the whole tower
        else:
            keep.update(ch.name for ch in channels if channel_suits(claim, ch))
    tower = channels if full else [ch for ch in channels if ch.name in keep]
    if not tower:
        tower = channels
    return orchestrate(module, tower, theta, policy), len(tower)


@dataclass(frozen=True)
class OrchestratePriorCertificate:
    """The tower-pass safety witness (the ChannelPriorCertificate recipe one level up):
    over fixture modules, guided placements must equal the full-tower placements CLAIM
    FOR CLAIM (`mismatches` MUST be 0); the reduction in whole-module optimize() runs
    is the earned win."""

    checked: int
    mismatches: int
    priced_guided: int
    priced_exhaustive: int

    @property
    def admitted(self) -> bool:
        return self.checked >= 1 and self.mismatches == 0

    @property
    def reduction(self) -> float:
        if self.priced_exhaustive == 0:
            return 0.0
        return 1.0 - self.priced_guided / self.priced_exhaustive


def orchestrate_prior_certificate(fixtures: list, channels: list, theta: Theta,
                                  policy: Policy = PERF, *,
                                  table: dict) -> OrchestratePriorCertificate:
    """Certify the wiring over `fixtures` (a list of (module, dims) pairs): run the
    guided pass and the exhaustive pass, compare every placement's chosen channel."""
    from ..channels import orchestrate
    checked = mismatches = pg = pe = 0
    for module, dims in fixtures:
        guided, priced = orchestrate_guided(module, channels, theta, policy,
                                            table=table, dims=dims)
        exact = orchestrate(module, channels, theta, policy)
        pg += priced
        pe += len(channels)
        for a, b in zip(guided.placements, exact.placements):
            checked += 1
            if (a.claim_id, a.channel) != (b.claim_id, b.channel):
                mismatches += 1
    return OrchestratePriorCertificate(checked=checked, mismatches=mismatches,
                                       priced_guided=pg, priced_exhaustive=pe)
