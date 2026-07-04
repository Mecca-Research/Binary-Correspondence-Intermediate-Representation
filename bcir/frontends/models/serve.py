"""Rung 6 of the open-weight ladder (ML/AI roadmap §7.4): `generate()` as a PLANNED,
PROOF-CARRYING artifact -- the D5 statement ("decode scheduling is a K_BCIR problem;
prefill/decode split ≈ phase structure") made concrete.

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

SECOND SLICE (this file too): STREAMING emission -- `generate_stream` yields one
"token" `StreamEvent` per emitted id the moment it is minted (its DataDNA frame is
already through the Broker when the event surfaces), then ONE terminal "done" event
carrying the complete `GenerationResult`; `generate()` just drains the stream, so every
slice-1 pin (bit-for-bit ids, eos-not-fed-back, the manifest artifacts) holds for the
stream BY CONSTRUCTION -- one code path. And SCHEMA-CONSTRAINED output: a `TokenDFA`
(per-state allowed token sets + edges) masks the argmax -- the model's logits still
rank the candidates, the schema merely forbids off-schema ids; an empty allowed set is
a DEADLOCK and refuses loudly (the engine never silently emits outside the schema); a
constrained generation carries the `("constrained", 1)` manifest artifact.

Deferred to rung 7 (recorded, not hidden): batched multi-session serving / continuous
batching -- opened by `paged_kv`. Cost-side module: imports no verifier (two-truth);
the tests verify."""

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


# --- schema-constrained emission (rung-6 slice 2) ---------------------------------------------

@dataclass(frozen=True)
class TokenDFA:
    """A token-level output schema as a DFA: `allowed[s]` is the tuple of token ids the
    schema admits in state `s`; `edges[s]` maps each admitted token to its successor
    state. Constraining is a MASK, not a sampler change -- the model's logits still rank
    the candidates; the DFA merely forbids off-schema ids. An empty allowed set is a
    DEADLOCK: the engine refuses loudly rather than emit outside the schema."""

    allowed: tuple                    # allowed[s]: tuple of admissible token ids
    edges: tuple                      # edges[s]: {token id -> successor state}
    start: int = 0


def check_token_dfa(dfa: TokenDFA, vocab_size: int) -> list:
    """Well-formedness (the registry posture: validate, then trust): every allowed token
    is in vocab AND has an edge; every edge is over an allowed token AND lands in a real
    state; the start state exists. Messages, not exceptions -- callers refuse."""
    msgs: list = []
    n = len(dfa.allowed)
    if len(dfa.edges) != n:
        return [f"TokenDFA has {n} allowed sets but {len(dfa.edges)} edge maps"]
    if not 0 <= dfa.start < max(1, n):
        msgs.append(f"TokenDFA start state {dfa.start} is not in [0, {n})")
    for s in range(n):
        for tok in dfa.allowed[s]:
            if not 0 <= tok < vocab_size:
                msgs.append(f"TokenDFA state {s} allows token {tok} outside the "
                            f"vocab [0, {vocab_size})")
            if tok not in dfa.edges[s]:
                msgs.append(f"TokenDFA state {s} allows token {tok} but has no edge for it")
        for tok, nxt in dfa.edges[s].items():
            if tok not in dfa.allowed[s]:
                msgs.append(f"TokenDFA state {s} has an edge over token {tok} it does "
                            "not allow")
            if not 0 <= nxt < n:
                msgs.append(f"TokenDFA state {s} edge on token {tok} targets state "
                            f"{nxt} which does not exist")
    return msgs


def _masked_argmax(logits: list, allowed: tuple) -> int:
    """`_argmax` under the schema mask: the best ADMITTED id. Walking ids ascending with
    a strict `>` keeps the first maximum, so ties break to the lowest id -- the same
    determinism law as the unmasked pick."""
    best = -1
    for v in sorted(allowed):
        if best < 0 or logits[v] > logits[best]:
            best = v
    return best


# --- streaming emission (rung-6 slice 2) -------------------------------------------------------

@dataclass(frozen=True)
class StreamEvent:
    """One streamed emission. kind "token": `token` + `frame` surface the moment the id
    is minted (the frame is already through the Broker; the id is NOT yet fed back).
    kind "done": `result` carries the complete GenerationResult -- the terminal event IS
    the batch answer, which is why `generate()` is just the stream drained."""

    kind: str                         # "token" | "done"
    index: int                        # 0-based emission index; the token count on "done"
    token: int = -1
    frame: DataDNA | None = None
    result: GenerationResult | None = None


def generate_stream(prompt_ids: list, spec: DecoderSpec, w: DecoderWeights, max_new: int,
                    *, h: HProfile, theta: Theta, policy: Policy = PERF,
                    eos_id: int | None = None, log_path: str | None = None,
                    schema: TokenDFA | None = None):
    """The rung-6 slice-2 entry: the same generation as `generate`, yielded token-by-
    token -- one "token" StreamEvent per emission, then ONE terminal "done" event with
    the full GenerationResult. Argument errors and a malformed `schema` refuse HERE (at
    the call), not at first iteration; only the deadlock refusal is inherently mid-walk.
    With `schema`, the pick is `_masked_argmax` over the current DFA state's allowed set
    and the manifest gains the ("constrained", 1) artifact."""
    if not prompt_ids or max_new < 0:
        raise ValueError("generate needs a non-empty prompt and max_new >= 0")
    if schema is not None:
        bad = check_token_dfa(schema, spec.vocab_size)
        if bad:
            raise ValueError("TokenDFA rejected: " + "; ".join(bad))
    return _stream(list(prompt_ids), spec, w, max_new, h, theta, policy, eos_id,
                   log_path, schema)


def _stream(prompt_ids: list, spec: DecoderSpec, w: DecoderWeights, max_new: int,
            h: HProfile, theta: Theta, policy: Policy, eos_id: int | None,
            log_path: str | None, schema: TokenDFA | None):
    """The generator body behind `generate_stream` (split so validation is eager)."""
    broker = Broker()
    ring = broker.subscribe(TelemetryRing(capacity=max(16, 2 * (max_new + 1))))
    if log_path is not None:
        broker.subscribe(DurableLog(log_path))
    cache = KVCache(spec)
    hrow: list = []
    for tid in prompt_ids:                             # the prefill (phase 0, executed)
        hrow = cache._step_row(w.embedding.row(int(tid)), spec, w)
    capacity = len(prompt_ids) + max_new
    state = schema.start if schema is not None else 0
    frames: list = []
    out: list = []
    for t in range(max_new):
        logits = _head_logits(hrow, w)
        if schema is None:
            nxt = _argmax(logits)
        else:
            allowed = schema.allowed[state]
            if not allowed:
                raise ValueError(
                    f"TokenDFA deadlock: state {state} admits no token at step {t} -- "
                    "constrained generation refuses rather than emit off-schema")
            nxt = _masked_argmax(logits, allowed)
            state = schema.edges[state][nxt]
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
        yield StreamEvent(kind="token", index=t, token=nxt, frame=frame)
        if eos_id is not None and nxt == eos_id:
            break
        hrow = cache._step_row(w.embedding.row(nxt), spec, w)
    kv_record = {"n_layers": spec.n_layers, "n_kv_heads": spec.kv_heads, "d_k": spec.d_k,
                 "capacity": capacity, "pos": cache.pos, "dtype": "f32"}
    cert = certify_session(spec, len(prompt_ids), max_new, h, theta, policy)
    m = decode_session_module(spec, len(prompt_ids), max_new)
    artifacts = (("prompt_sha", int(hashlib.sha256(
                      ",".join(str(i) for i in prompt_ids).encode()).hexdigest()[:12], 16)),
                 ("ids_sha", int(hashlib.sha256(
                      ",".join(str(i) for i in out).encode()).hexdigest()[:12], 16)),
                 ("tokens", len(out)), ("kv_pos", cache.pos))
    if schema is not None:
        artifacts += (("constrained", 1),)
    manifest = build_manifest(m, h, theta, policy, artifacts=artifacts)
    result = GenerationResult(ids=out, frames=frames, kv_record=kv_record,
                              manifest=manifest, certificate=cert)
    yield StreamEvent(kind="done", index=len(out), result=result)


def generate(prompt_ids: list, spec: DecoderSpec, w: DecoderWeights, max_new: int, *,
             h: HProfile, theta: Theta, policy: Policy = PERF, eos_id: int | None = None,
             log_path: str | None = None,
             schema: TokenDFA | None = None) -> GenerationResult:
    """The rung-6 batch entry: greedy generation over the rung-3 KV-cache decoder (the
    emitted ids are `decode_with_kv_cache`'s BIT-FOR-BIT), with the flight recorder on:
    one DataDNA frame per token through a Broker (ring + optional DurableLog at
    `log_path`), the live kv_cache record, the split certificate, and ONE R13 manifest
    whose artifacts pin the prompt digest, the ids digest, and the final cache position.
    Since slice 2 this is `generate_stream` DRAINED -- one code path, so the stream and
    the batch cannot drift."""
    result: GenerationResult | None = None
    for ev in generate_stream(prompt_ids, spec, w, max_new, h=h, theta=theta,
                              policy=policy, eos_id=eos_id, log_path=log_path,
                              schema=schema):
        if ev.kind == "done":
            result = ev.result
    assert result is not None                          # the stream always terminates in "done"
    return result
