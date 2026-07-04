"""Rung 6 of the open-weight ladder (ML/AI roadmap §7.4), FIRST SLICE: `generate()` as a
PLANNED, PROOF-CARRYING artifact -- the D5 statement ("decode scheduling is a K_BCIR
problem; prefill/decode split ≈ phase structure") made concrete.

One generation =

  * the SESSION MODULE (`decode_session_module`): a claim graph whose phase structure IS
    the prefill/decode split -- one batched prefill claim over the whole prompt, then one
    decode claim per generated token, serialized by their true hazards (each step reads
    AND writes the KV store and the token tape -- the RAW chain is the autoregression);
    R-law verified, K_BCIR priced, StreamPack-hydratable like any program;
  * the SPLIT CERTIFICATE (`SessionCertificate`, the PipelineCertificate recipe): the
    priced cost of the batched prefill vs the same prompt pushed token-by-token through
    decode steps -- the prefill/decode split is ADMITTED only when batching genuinely
    wins under the cost model (measured-then-pinned in the tests);
  * the EXECUTION (`generate`): the rung-3 KV-cache decoder drives real token emission
    (the ids are `decode_with_kv_cache`'s ids BIT-FOR-BIT -- the session is the cached
    path with a flight recorder strapped on), emitting one `DataDNA` frame per token
    through a `Broker` (ring + optional `DurableLog` -- the log replays), a
    `gem.kv_cache`-shaped record per step (the MLIR op's attrs, live), and ONE R13
    `ProvenanceManifest` per generation whose artifacts pin the prompt digest, the
    emitted ids digest, and the final cache position -- `provenance.replay` reproduces
    the plan.

Deferred to later rung-6 slices (recorded, not hidden): streaming/chunked emission,
schema-constrained tool-call output, batched multi-session serving (rung 7's continuous
batching). Cost-side module: imports no verifier (two-truth); the tests verify."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ...kbcir.cost import HProfile, Theta
from ...kbcir.provenance import ProvenanceManifest, build_manifest
from ...kbcir.realize import optimize
from ...kbcir.weights import PERF, Policy
from ...model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from ...telemetry import Broker, DataDNA, DurableLog, TelemetryRing
from .decode import DecoderSpec, DecoderWeights, KVCache, _argmax, _head_logits

# RIDs of the session's resources (a closed universe, the train_graph convention).
_TOK, _WTS, _KV, _LOGITS = 1, 2, 3, 4


def _session_resources(spec: DecoderSpec, capacity: int) -> tuple:
    return (Resource(rid=_TOK, domain=Domain.RAM, shape=(capacity,), name="TOK"),
            Resource(rid=_WTS, domain=Domain.RAM, shape=(max(1, spec.n_layers),), name="WTS"),
            Resource(rid=_KV, domain=Domain.RAM,
                     shape=(capacity, max(1, spec.n_layers * spec.kv_dim)), name="KV"),
            Resource(rid=_LOGITS, domain=Domain.RAM, shape=(spec.vocab_size,), name="LOGITS"))


def _prefill_claim(spec: DecoderSpec, rows: int, cid: int) -> Claim:
    """The BATCHED prefill: one tile claim advancing `rows` prompt tokens through every
    layer (reads the token tape + weights, fills the KV store). A structured tile walk --
    `assumed_safe`, the train_graph posture; the tensor law is R22 on the gem rail."""
    return Claim(id=cid, opcode=Opcode.T_MACC, lane=Lane.T, stride_class=StrideClass.TILE,
                 count=max(1, rows * spec.n_layers), rd=(_TOK, _WTS), wr=(_KV,),
                 op="gem.prefill", domain=Domain.RAM, bounds="assumed_safe")


def _decode_claim(spec: DecoderSpec, cid: int) -> Claim:
    """One generated token: reads the tape + weights + the WHOLE cache so far, appends
    its KV rows, writes the logits, appends the sampled id to the tape -- reading AND
    writing TOK/KV is what serializes the autoregression (the true hazard chain)."""
    return Claim(id=cid, opcode=Opcode.T_MACC, lane=Lane.T, stride_class=StrideClass.TILE,
                 count=max(1, spec.n_layers), rd=(_TOK, _WTS, _KV), wr=(_KV, _LOGITS, _TOK),
                 op="gem.decode_step", domain=Domain.RAM, bounds="assumed_safe")


def decode_session_module(spec: DecoderSpec, prompt_len: int, max_new: int, *,
                          batched_prefill: bool = True) -> Module:
    """The generation session as a Module: phase 0 = the prefill (ONE batched claim, or
    `prompt_len` sequential decode-shaped claims when `batched_prefill=False` -- the
    certificate's comparison arm), then one decode phase per generated token, each
    dependent on its predecessor (the autoregressive chain, stated twice: phase deps AND
    the TOK/KV read-write hazards)."""
    if prompt_len < 1 or max_new < 0:
        raise ValueError("decode_session_module needs prompt_len >= 1 and max_new >= 0")
    capacity = prompt_len + max_new
    m = Module(name=f"decode_session_{prompt_len}p_{max_new}n")
    for r in _session_resources(spec, capacity):
        m.add_resource(r)
    pid = 0
    prev: tuple = ()
    if batched_prefill:
        m.add_phase(Phase(phase_id=pid, deps=prev, claims=[_prefill_claim(spec, prompt_len, 1)]))
        prev = (pid,)
        pid += 1
    else:                                              # the sequential arm: token-by-token
        for i in range(prompt_len):
            m.add_phase(Phase(phase_id=pid, deps=prev, claims=[_decode_claim(spec, 1 + i)]))
            prev = (pid,)
            pid += 1
    for t in range(max_new):
        m.add_phase(Phase(phase_id=pid, deps=prev, claims=[_decode_claim(spec, 100 + t)]))
        prev = (pid,)
        pid += 1
    return m


@dataclass(frozen=True)
class SessionCertificate:
    """The prefill/decode-split witness: the K_BCIR cost of the BATCHED prefill vs the
    same prompt pushed token-by-token through decode steps. The split is admitted only
    when batching genuinely wins (or ties) under the cost model."""

    prompt_len: int
    max_new: int
    prefill_batched: int
    prefill_sequential: int

    @property
    def split_win(self) -> int:
        return self.prefill_sequential - self.prefill_batched

    @property
    def admitted(self) -> bool:
        return 0 < self.prefill_batched <= self.prefill_sequential


def certify_session(spec: DecoderSpec, prompt_len: int, max_new: int, h: HProfile,
                    theta: Theta, policy: Policy = PERF) -> SessionCertificate:
    """Price both prefill arms (the decode tail is identical in both modules) and
    certify the split."""
    def prefill_cost(batched: bool) -> int:
        m = decode_session_module(spec, prompt_len, max_new, batched_prefill=batched)
        result = optimize(m, h, theta, policy)
        return sum(s.cost for s in result.steps if s.claim_id < 100)
    return SessionCertificate(prompt_len=prompt_len, max_new=max_new,
                              prefill_batched=prefill_cost(True),
                              prefill_sequential=prefill_cost(False))


@dataclass
class GenerationResult:
    """One generation's proof-carrying record: the emitted ids, the per-token DataDNA
    frames, the live `gem.kv_cache`-shaped record (the MLIR op's attrs at the final
    position), the R13 manifest, and the split certificate."""

    ids: list
    frames: list                      # one DataDNA per emitted token
    kv_record: dict                   # {n_layers, n_kv_heads, d_k, capacity, pos, dtype}
    manifest: ProvenanceManifest
    certificate: SessionCertificate


def generate(prompt_ids: list, spec: DecoderSpec, w: DecoderWeights, max_new: int, *,
             h: HProfile, theta: Theta, policy: Policy = PERF, eos_id: int | None = None,
             log_path: str | None = None) -> GenerationResult:
    """The rung-6 entry: greedy generation over the rung-3 KV-cache decoder (the emitted
    ids are `decode_with_kv_cache`'s BIT-FOR-BIT), with the flight recorder on: one
    DataDNA frame per token through a Broker (ring + optional DurableLog at `log_path`),
    the live kv_cache record, the split certificate, and ONE R13 manifest whose artifacts
    pin the prompt digest, the ids digest, and the final cache position."""
    if not prompt_ids or max_new < 0:
        raise ValueError("generate needs a non-empty prompt and max_new >= 0")
    broker = Broker()
    ring = broker.subscribe(TelemetryRing(capacity=max(16, 2 * (max_new + 1))))
    if log_path is not None:
        broker.subscribe(DurableLog(log_path))
    cache = KVCache(spec)
    hrow: list = []
    for tid in prompt_ids:                             # the prefill (phase 0, executed)
        hrow = cache._step_row(w.embedding.row(int(tid)), spec, w)
    capacity = len(prompt_ids) + max_new
    frames: list = []
    out: list = []
    for t in range(max_new):
        logits = _head_logits(hrow, w)
        nxt = _argmax(logits)
        out.append(nxt)
        frame = DataDNA(
            segment_id=f"serve:{spec.n_layers}L:{spec.d_model}d",
            claim_id=100 + t,                          # the decode claim's id band
            cycles=cache.pos,                          # positions attended this step
            bytes=spec.d_model * 8,                    # one f64 hidden row emitted
            utilization=min(100, 100 * (cache.pos + 1) // max(1, capacity)),
            provenance=f"tok:{nxt}")
        broker.emit(frame)
        frames.append(frame)
        if eos_id is not None and nxt == eos_id:
            break
        hrow = cache._step_row(w.embedding.row(nxt), spec, w)
    kv_record = {"n_layers": spec.n_layers, "n_kv_heads": spec.kv_heads, "d_k": spec.d_k,
                 "capacity": capacity, "pos": cache.pos, "dtype": "f32"}
    cert = certify_session(spec, len(prompt_ids), max_new, h, theta, policy)
    m = decode_session_module(spec, len(prompt_ids), max_new)
    manifest = build_manifest(
        m, h, theta, policy,
        artifacts=(("prompt_sha", int(hashlib.sha256(
                        ",".join(str(i) for i in prompt_ids).encode()).hexdigest()[:12], 16)),
                   ("ids_sha", int(hashlib.sha256(
                        ",".join(str(i) for i in out).encode()).hexdigest()[:12], 16)),
                   ("tokens", len(out)), ("kv_pos", cache.pos)))
    return GenerationResult(ids=out, frames=frames, kv_record=kv_record,
                            manifest=manifest, certificate=cert)
