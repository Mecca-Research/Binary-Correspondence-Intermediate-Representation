"""Rung 7 OPENER (ML/AI roadmap §7.4): PAGED KV -- the `gem.kv_cache` capacity law made
LIVE, and continuous batching revealed as wave scheduling.

Three moves, each riding something already proven:

  * `PagedKV`: the KV store behind a PAGE TABLE whose pages are registry RESOURCES
    (rid band 7000+p) with live `data_gen`s -- every write to position `pos` bumps the
    generation of page `pos // page_size`, so R11 (generation validity) sees KV traffic
    exactly as it sees any other registry mutation: a StreamPack hydrated over the pages
    goes STALE the moment another token lands (pinned in the tests). Every page is
    allocated as a wave-11 `StridedView` checked against a `DeviceManifest` bank --
    D-R4 live at the serving layer: no stride vector, no memory; a fragmenting page
    size (15 against a 16-native bank) refuses at CONSTRUCTION, not at runtime. And the
    MLIR op's paging law (`pos <= capacity`) runs here, live: advancing past capacity
    raises the verifier's own words ("pos N exceeds capacity C ... the paging lie").
    The NUMERICS ride the proven rung-3 `KVCache` unchanged -- `generate_paged`'s ids
    are `decode_with_kv_cache`'s BIT-FOR-BIT by construction, because paging is a
    REGISTRY story, not a math story.

  * `batched_sessions_module`: N decode sessions in ONE module -- per-session TOK/KV/
    LOGITS on disjoint RID bands, all sessions reading the single shared WTS (read-read
    never conflicts: the `train_stream_module` recipe), each session its own prefill ->
    decode phase chain with NO cross-session deps. The token DAG discovers the overlap
    from the hazards alone.

  * `BatchCertificate` (`certify_batch`): serial / phase-barriered / token-pipelined
    makespans of the same batch -- the overlap win is what "continuous batching" buys,
    and it falls out of the EXISTING scheduler with zero new machinery: continuous
    batching IS wave scheduling over the merged claim graph (measured-then-pinned).

RUNG-7 SLICE 2 (Part VII A3, this file too): the page-claim WIRING --
`paged_session_module` now threads the page RIDs through the claims themselves (the
prefill claim writes the pages it fills; decode step t reads every page up to its
position and writes the page owning its row), so the token DAG sees page-level
hazards. That is what makes the other two moves fall out: EVICTION is a scheduled
claim (`PagedKV.evict` is the registry act -- map_gen bumps, the view frees, a live
session refuses because full-context attention still reads every page;
`schedule_eviction` appends the same act as a `kv.evict:<p>` claim), and ADMISSION is
appending phases (`admit_session` joins a new session to a LIVE batched module --
claim-identical to having built it upfront, pinned by hash_module equality).

Deferred to later rung-7 slices (recorded, not hidden): windowed-attention eviction
(evicting pages a numerics-visible attention window no longer reads), page REUSE
across sessions (the freed view re-entering an allocator). Cost-side module: imports
no verifier (two-truth)."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ...gem.schedule import GemSchedule, durations_from, execute_tokens, schedule_eft
from ...kbcir.cost import HProfile, Theta
from ...kbcir.device_manifest import DeviceManifest, StridedView, check_strided_view
from ...kbcir.realize import optimize
from ...kbcir.weights import PERF, Policy
from ...model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from .decode import (
    _MAX_REFERENCE_CONTEXT,
    DecoderSpec,
    DecoderWeights,
    KVCache,
    _argmax,
    _head_logits,
    _validate_decode_request,
    check_decoder_weights,
)
from .serve import _validate_session_shape, decode_session_module

_PAGE_RID = 7000  # the page band: page p is Resource rid 7000 + p
_WTS = 2  # the shared weights RID (decode_session's convention)
_MAX_U32 = (1 << 32) - 1
_MAX_U64 = (1 << 64) - 1
_MAX_PAGES = 1 << 16
_MAX_BATCH_SESSIONS = 4096
_MAX_BATCH_CLAIMS = 1 << 16
_SESSION_CLAIM_STRIDE = 1 << 20


class PagedKV:
    """The paged KV store: proven `KVCache` numerics behind a page table of registry
    Resources. Construction IS allocation: each page is a `StridedView` against the
    manifest bank and refuses (D-R4) before any token flows; `step_row` enforces the
    `gem.kv_cache` capacity law live and bumps the owning page's `data_gen` -- the
    registry generation IS the paging state."""

    def __init__(
        self,
        spec: DecoderSpec,
        capacity: int,
        page_size: int,
        man: DeviceManifest,
        bank: str,
        *,
        base_rid: int = _PAGE_RID,
    ) -> None:
        if not isinstance(spec, DecoderSpec):
            raise ValueError("PagedKV spec is not a DecoderSpec")
        if type(capacity) is not int or not 1 <= capacity <= _MAX_REFERENCE_CONTEXT:
            raise ValueError(
                f"PagedKV capacity must be an integer in [1, {_MAX_REFERENCE_CONTEXT}]"
            )
        if type(page_size) is not int or not 1 <= page_size <= capacity:
            raise ValueError("PagedKV page_size must be an integer in [1, capacity]")
        if not isinstance(man, DeviceManifest):
            raise ValueError("PagedKV manifest is not a DeviceManifest")
        if not isinstance(bank, str) or not bank:
            raise ValueError("PagedKV bank must be a non-empty string")
        if type(base_rid) is not int or not 5 <= base_rid <= _MAX_U32:
            raise ValueError("PagedKV base_rid must be an integer in [5, 2^32-1]")
        b = man.bank(bank)  # KeyError on a ghost bank (a veto)
        # one position = one K row + one V row per layer per kv head, f32.
        row_elems = 2 * max(1, spec.n_layers * spec.kv_dim)
        n_pages = -(-capacity // page_size)
        if n_pages > _MAX_PAGES:
            raise ValueError(f"PagedKV needs {n_pages} pages; limit is {_MAX_PAGES}")
        if base_rid > _MAX_U32 - (n_pages - 1):
            raise ValueError("PagedKV page RID range overflows uint32")
        page_bytes = page_size * row_elems * 4
        views: list = []
        pages: list = []
        for p in range(n_pages):
            view = StridedView(
                bank=bank,
                offset_bytes=p * page_bytes,
                shape=(page_size, row_elems),
                strides=(row_elems, 1),
                elem_bytes=4,
            )
            bad = check_strided_view(view, man)
            if bad:  # D-R4, live: no view, no memory
                raise ValueError(f"PagedKV page {p} refused: " + "; ".join(bad))
            views.append(view)
            pages.append(
                Resource(
                    rid=base_rid + p,
                    domain=b.domain,
                    shape=(page_size, row_elems),
                    name=f"KVPAGE{p}",
                )
            )
        # Commit only after every page has passed the manifest checks.
        self.spec, self.capacity, self.page_size = spec, capacity, page_size
        self.row_elems, self.n_pages = row_elems, n_pages
        self.views, self.pages = views, pages
        self.cache = KVCache(spec)

    @property
    def pos(self) -> int:
        return self.cache.pos

    def page_of(self, pos: int) -> int:
        if type(pos) is not int or not 0 <= pos < self.capacity:
            raise ValueError(f"kv_cache: position must be an integer in [0, {self.capacity})")
        return pos // self.page_size

    def step_row(self, x_row: list, spec: DecoderSpec, w: DecoderWeights) -> list:
        """Advance one token through the proven cache, THEN account for it: the write
        lands in page `pos // page_size`, whose data_gen bumps (R11's currency). The
        capacity law refuses FIRST, in the MLIR verifier's own words."""
        if spec != self.spec:
            raise ValueError("kv_cache: step spec does not match the cache spec")
        if not isinstance(x_row, (list, tuple)) or len(x_row) != self.spec.d_model:
            raise ValueError(f"kv_cache: row must have {self.spec.d_model} values")
        for value in x_row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("kv_cache: row values must be finite numbers")
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError("kv_cache: row values must be finite numbers")
        bad = check_decoder_weights(self.spec, w)
        if bad:
            raise ValueError("kv_cache: decoder weights rejected: " + "; ".join(bad))
        if self.cache.pos >= self.capacity:
            raise ValueError(
                f"kv_cache: pos {self.cache.pos + 1} exceeds capacity {self.capacity} "
                "(an over-full cache is the paging lie)"
            )
        p = self.page_of(self.cache.pos)
        if self.views[p] is None:
            raise ValueError(
                f"kv_cache: page {p} was evicted; writing through a freed "
                "view is the use-after-free shape"
            )
        out = self.cache._step_row(x_row, spec, w)
        self.pages[p] = replace(self.pages[p], data_gen=self.pages[p].data_gen + 1)
        return out

    def evict(self, p: int):
        """A3: eviction as a REGISTRY act. Refused while the session is live -- rung-3
        attention is full-context, so every page is still read (evicting one would be a
        numerics lie); legal once pos == capacity. The page's map_gen bumps (the remap
        generation -- R11's other currency, so hydrated packs go stale as 'rehydrate:
        repack') and the StridedView frees for reuse. Returns the freed view."""
        if type(p) is not int or not 0 <= p < self.n_pages:
            raise ValueError(f"kv.evict: no page {p} (0..{self.n_pages - 1})")
        if self.cache.pos < self.capacity:
            raise ValueError(
                f"kv.evict: page {p} refused at pos {self.cache.pos} < capacity "
                f"{self.capacity} -- full-context attention still reads every page; "
                "eviction before completion is a numerics lie"
            )
        if self.views[p] is None:
            raise ValueError(f"kv.evict: page {p} already evicted (the double-free shape)")
        view, self.views[p] = self.views[p], None
        self.pages[p] = replace(self.pages[p], map_gen=self.pages[p].map_gen + 1)
        return view

    def evict_claim(self, p: int, cid: int) -> Claim:
        """The SAME act as a scheduled claim (`kv.evict:<p>` reads and writes the page
        -- invalidation is a write), so eviction takes its place in the token DAG like
        any other work."""
        if type(p) is not int or not 0 <= p < self.n_pages:
            raise ValueError(f"kv.evict: no page {p!r} (0..{self.n_pages - 1})")
        if type(cid) is not int or not 1 <= cid <= _MAX_U64:
            raise ValueError("kv.evict: claim id must be an integer in [1, 2^64-1]")
        rid = self.pages[p].rid
        return Claim(
            id=cid,
            opcode=Opcode.STORE,
            lane=Lane.U,
            stride_class=StrideClass.SCALAR,
            count=1,
            rd=(rid,),
            wr=(rid,),
            op=f"kv.evict:{p}",
            domain=self.pages[p].domain,
        )


def generate_paged(
    prompt_ids: list,
    spec: DecoderSpec,
    w: DecoderWeights,
    max_new: int,
    *,
    man: DeviceManifest,
    bank: str,
    page_size: int,
    eos_id: int | None = None,
) -> tuple:
    """Greedy decode over the PAGED store, capacity = the exact session budget
    (prompt + max_new). The ids are `decode_with_kv_cache`'s BIT-FOR-BIT -- the same
    `KVCache` runs underneath; the pages only account. Returns (ids, kv) so callers can
    read the page table (generations, views, final pos)."""
    prompt = _validate_decode_request(prompt_ids, spec, w, max_new, eos_id)
    kv = PagedKV(spec, len(prompt) + max_new, page_size, man, bank)
    hrow: list = []
    for tid in prompt:  # the prefill fills pages in order
        hrow = kv.step_row(w.embedding.row(tid), spec, w)
    out: list = []
    for _ in range(max_new):
        nxt = _argmax(_head_logits(hrow, w))
        out.append(nxt)
        if eos_id is not None and nxt == eos_id:  # emitted, never fed back (the
            break  # serve.py eos law)
        hrow = kv.step_row(w.embedding.row(nxt), spec, w)
    return out, kv


def paged_session_module(spec: DecoderSpec, prompt_len: int, max_new: int, kv: PagedKV) -> Module:
    """The decode-session module JOINED with the live page resources at their CURRENT
    generations -- and WIRED (A3): the prefill claim writes the pages it fills; decode
    step t reads every page up to its position (full-context attention touches the
    whole cache) and writes the page owning row prompt_len + t. The token DAG now sees
    page-level hazards, which is what makes eviction schedulable and admission
    appendable. A pack hydrated from this module is R11-stale the moment any page
    takes another write (data_gen) or is evicted (map_gen)."""
    _validate_session_shape(prompt_len, max_new)
    if not isinstance(kv, PagedKV) or kv.spec != spec:
        raise ValueError("paged session cache/spec mismatch")
    required = prompt_len + max_new
    if required > kv.capacity:
        raise ValueError(f"paged session needs {required} rows but cache capacity is {kv.capacity}")
    if len(kv.pages) != kv.n_pages or len(kv.views) != kv.n_pages or kv.n_pages < 1:
        raise ValueError("paged session cache has a malformed page table")
    base = kv.pages[0].rid if isinstance(kv.pages[0], Resource) else -1
    for p, resource in enumerate(kv.pages):
        if not isinstance(resource, Resource) or resource.rid != base + p:
            raise ValueError("paged session page resources are not a contiguous RID band")
    m = decode_session_module(spec, prompt_len, max_new)
    if any(r.rid in m.resources for r in kv.pages):
        raise ValueError("paged session page RID collides with a session resource")
    for r in kv.pages:
        m.add_resource(r)

    def page_of(pos: int) -> int:
        return kv.page_of(pos)

    t = 0
    for ph in m.phases:
        for c in ph.claims:
            if c.op == "gem.prefill":
                c.wr = tuple(c.wr) + tuple(base + i for i in range(page_of(prompt_len - 1) + 1))
            elif c.op == "gem.decode_step":
                pos = prompt_len + t
                t += 1
                c.rd = tuple(c.rd) + tuple(base + i for i in range(page_of(pos) + 1))
                c.wr = tuple(c.wr) + (base + page_of(pos),)
    return m


def schedule_eviction(m: Module, kv: PagedKV, p: int, *, cid: int = 0) -> Module:
    """A3: eviction takes its place in the SCHEDULE -- one appended phase, dependent on
    the module's last phase, carrying `kv.evict:<p>`. (The registry act itself is
    `PagedKV.evict`; this is its planned shadow.)"""
    if not isinstance(m, Module):
        raise ValueError("kv.evict schedule target is not a Module")
    existing_cids = [c.id for ph in m.phases for c in ph.claims]
    if len(set(existing_cids)) != len(existing_cids):
        raise ValueError("kv.evict refuses a module with duplicate claim ids")
    phase_ids = [ph.phase_id for ph in m.phases]
    if len(set(phase_ids)) != len(phase_ids):
        raise ValueError("kv.evict refuses a module with duplicate phase ids")
    if cid == 0:
        cid = max(existing_cids, default=0) + 1
    claim = kv.evict_claim(p, cid)
    if cid in set(existing_cids):
        raise ValueError(f"kv.evict claim id {cid} already exists")
    if claim.rd[0] not in m.resources:
        raise ValueError(f"kv.evict page RID {claim.rd[0]} is not in the module")
    last = m.phases[-1].phase_id if m.phases else -1
    phase_id = max(phase_ids, default=-1) + 1
    if phase_id > _MAX_U32:
        raise ValueError("kv.evict phase id overflows uint32")
    m.add_phase(Phase(phase_id=phase_id, deps=(last,) if m.phases else (), claims=[claim]))
    return m


def _session_parts(
    spec: DecoderSpec, s: int, prompt_len: int, max_new: int, phase_start: int
) -> tuple[tuple[Resource, ...], tuple[Phase, ...]]:
    """One session's resources + prefill->decode chain, appended to `m` (phase ids
    continue from the module's tail -- which is why building upfront and admitting
    mid-flight produce the IDENTICAL graph)."""
    _validate_session_shape(prompt_len, max_new)
    if type(s) is not int or not 0 <= s < _MAX_BATCH_SESSIONS:
        raise ValueError(f"session index must be in [0, {_MAX_BATCH_SESSIONS})")
    o, base = s * 10, _SESSION_CLAIM_STRIDE * s
    cap = prompt_len + max_new
    tok, kvr, logi = o + 1, o + 3, o + 4
    resources = (
        Resource(rid=tok, domain=Domain.RAM, shape=(cap,), name=f"TOK{s}"),
        Resource(
            rid=kvr,
            domain=Domain.RAM,
            shape=(cap, max(1, spec.n_layers * spec.kv_dim)),
            name=f"KV{s}",
        ),
        Resource(rid=logi, domain=Domain.RAM, shape=(spec.vocab_size,), name=f"LOGITS{s}"),
    )
    phases: list[Phase] = []
    pid = phase_start
    prev: tuple = ()
    prefill = Claim(
        id=base + 1,
        opcode=Opcode.T_MACC,
        lane=Lane.T,
        stride_class=StrideClass.TILE,
        count=max(1, prompt_len * spec.n_layers),
        rd=(tok, _WTS),
        wr=(kvr,),
        op="gem.prefill",
        domain=Domain.RAM,
        bounds="assumed_safe",
    )
    phases.append(Phase(phase_id=pid, deps=prev, claims=[prefill]))
    prev = (pid,)
    pid += 1
    for t in range(max_new):
        dec = Claim(
            id=base + 100 + t,
            opcode=Opcode.T_MACC,
            lane=Lane.T,
            stride_class=StrideClass.TILE,
            count=max(1, spec.n_layers),
            rd=(tok, _WTS, kvr),
            wr=(kvr, logi, tok),
            op="gem.decode_step",
            domain=Domain.RAM,
            bounds="assumed_safe",
        )
        phases.append(Phase(phase_id=pid, deps=prev, claims=[dec]))
        prev = (pid,)
        pid += 1
    return resources, tuple(phases)


def _add_session(m: Module, spec: DecoderSpec, s: int, prompt_len: int, max_new: int) -> None:
    """Preflight one session completely, then append it atomically to the graph."""
    resources, phases = _session_parts(spec, s, prompt_len, max_new, len(m.phases))
    new_rids = {r.rid for r in resources}
    if new_rids & set(m.resources):
        raise ValueError("session resource RID collides with the live module")
    phase_ids = [ph.phase_id for ph in m.phases]
    if len(set(phase_ids)) != len(phase_ids) or set(phase_ids) != set(range(len(phase_ids))):
        raise ValueError("session admission requires contiguous unique phase ids")
    old_cids = {c.id for ph in m.phases for c in ph.claims}
    if sum(len(ph.claims) for ph in m.phases) != len(old_cids):
        raise ValueError("session admission refuses duplicate live claim ids")
    new_cids = {c.id for ph in phases for c in ph.claims}
    if old_cids & new_cids or len(new_cids) != sum(len(ph.claims) for ph in phases):
        raise ValueError("session claim-id band collides with the live module")
    for resource in resources:
        m.add_resource(resource)
    for phase in phases:
        m.add_phase(phase)


def _session_count(m: Module, spec: DecoderSpec) -> int:
    """Validate the complete resident batch graph and return its session count.

    This is a trust boundary for admission and certification: deriving a count from the
    resource table alone let a caller attach an arbitrary or empty claim graph to a
    valid-looking registry and obtain a meaningless certificate.
    """
    if not isinstance(m, Module):
        raise ValueError("batch is not a Module")
    if not isinstance(spec, DecoderSpec):
        raise ValueError("batch spec is not a DecoderSpec")
    if len(m.resources) < 4 or (len(m.resources) - 1) % 3:
        raise ValueError("batch registry is not WTS plus three resources per session")
    count = (len(m.resources) - 1) // 3
    if not 1 <= count <= _MAX_BATCH_SESSIONS:
        raise ValueError("batch session count is outside the supported range")
    expected = {_WTS}
    wts = m.resources.get(_WTS)
    if (
        wts is None
        or wts.name != "WTS"
        or wts.domain != Domain.RAM
        or wts.shape != (max(1, spec.n_layers),)
    ):
        raise ValueError("batch WTS resource does not match the decoder spec")
    expected_wts = Resource(rid=_WTS, domain=Domain.RAM, shape=(max(1, spec.n_layers),), name="WTS")
    if (
        not isinstance(wts.map_gen, int)
        or isinstance(wts.map_gen, bool)
        or not isinstance(wts.data_gen, int)
        or isinstance(wts.data_gen, bool)
        or not 0 <= wts.map_gen <= _MAX_U64
        or not 0 <= wts.data_gen <= _MAX_U64
        or replace(wts, map_gen=0, data_gen=0) != expected_wts
    ):
        raise ValueError("batch WTS resource metadata is malformed")
    resident: list[tuple[Resource, Resource, Resource]] = []
    for s in range(count):
        o = s * 10
        ids = (o + 1, o + 3, o + 4)
        expected.update(ids)
        tok, kvr, logi = (m.resources.get(rid) for rid in ids)
        if any(r is None for r in (tok, kvr, logi)):
            raise ValueError(f"batch session {s} resource band is incomplete")
        if tok.name != f"TOK{s}" or kvr.name != f"KV{s}" or logi.name != f"LOGITS{s}":
            raise ValueError(f"batch session {s} resource names are malformed")
        if tok.domain != Domain.RAM or kvr.domain != Domain.RAM or logi.domain != Domain.RAM:
            raise ValueError(f"batch session {s} resources must be RAM")
        if (
            len(tok.shape) != 1
            or type(tok.shape[0]) is not int
            or tok.shape[0] < 1
            or kvr.shape != (tok.shape[0], max(1, spec.n_layers * spec.kv_dim))
            or logi.shape != (spec.vocab_size,)
        ):
            raise ValueError(f"batch session {s} resource shapes are malformed")
        resident.append((tok, kvr, logi))
    if set(m.resources) != expected:
        raise ValueError("batch registry contains an unexpected resource")
    if any(
        not isinstance(ph, Phase) or len(ph.claims) != 1 or not isinstance(ph.claims[0], Claim)
        for ph in m.phases
    ):
        raise ValueError("batch phases must each contain exactly one claim")
    phase_ids = [ph.phase_id for ph in m.phases]
    if len(set(phase_ids)) != len(phase_ids) or set(phase_ids) != set(range(len(phase_ids))):
        raise ValueError("batch phase ids are not contiguous and unique")
    cids = [ph.claims[0].id for ph in m.phases]
    if (
        len(cids) > _MAX_BATCH_CLAIMS
        or any(type(cid) is not int or not 1 <= cid <= _MAX_U64 for cid in cids)
        or len(set(cids)) != len(cids)
    ):
        raise ValueError("batch claim inventory is oversized or ambiguous")
    cursor = 0
    for s, actual_resources in enumerate(resident):
        base = _SESSION_CLAIM_STRIDE * s
        band = [cid for cid in cids if base <= cid < base + _SESSION_CLAIM_STRIDE]
        if not band or base + 1 not in band:
            raise ValueError(f"batch session {s} has no canonical prefill claim")
        max_new = len(band) - 1
        expected_cids = {base + 1, *(base + 100 + t for t in range(max_new))}
        if set(band) != expected_cids:
            raise ValueError(f"batch session {s} claim-id inventory is malformed")
        capacity = actual_resources[0].shape[0]
        prompt_len = capacity - max_new
        _validate_session_shape(prompt_len, max_new)
        expected_resources, expected_phases = _session_parts(spec, s, prompt_len, max_new, cursor)
        for actual, expected_resource in zip(actual_resources, expected_resources):
            if (
                not isinstance(actual.map_gen, int)
                or isinstance(actual.map_gen, bool)
                or not isinstance(actual.data_gen, int)
                or isinstance(actual.data_gen, bool)
                or not 0 <= actual.map_gen <= _MAX_U64
                or not 0 <= actual.data_gen <= _MAX_U64
                or replace(actual, map_gen=0, data_gen=0) != expected_resource
            ):
                raise ValueError(f"batch session {s} resource metadata is malformed")
        actual_phases = tuple(m.phases[cursor : cursor + len(expected_phases)])
        if actual_phases != expected_phases:
            raise ValueError(f"batch session {s} phase/claim graph is malformed")
        cursor += len(expected_phases)
    if cursor != len(m.phases):
        raise ValueError("batch contains claims outside every resident session")
    if (
        m.name != f"batched_sessions_{count}x"
        or m.cacheline != 64
        or m.align != 64
        or m.target != "registry-first"
    ):
        raise ValueError("batch module metadata is malformed")
    return count


def batched_sessions_module(spec: DecoderSpec, sessions: tuple) -> Module:
    """N decode sessions as ONE claim graph: session s owns TOK/KV/LOGITS on the rid
    band s*10 + {1, 3, 4} (disjoint by construction), every session reads the single
    shared WTS (rid 2 -- read-read never conflicts), and each session is its own
    prefill -> decode chain with no cross-session phase deps. Claim ids: session s
    prefill = stride*s + 1, decode t = stride*s + 100 + t (session 0 matches
    `decode_session_module`'s bands and fixed-width bands cannot overlap)."""
    if not isinstance(spec, DecoderSpec):
        raise ValueError("batch spec is not a DecoderSpec")
    if not isinstance(sessions, (tuple, list)) or not sessions:
        raise ValueError("batched_sessions_module needs at least one session")
    if len(sessions) > _MAX_BATCH_SESSIONS:
        raise ValueError(f"batch has too many sessions; limit is {_MAX_BATCH_SESSIONS}")
    normalized: list[tuple[int, int]] = []
    claims = 0
    for s, item in enumerate(sessions):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(f"session {s} must be a (prompt_len, max_new) pair")
        prompt_len, max_new = item
        _validate_session_shape(prompt_len, max_new)
        claims += 1 + max_new
        if claims > _MAX_BATCH_CLAIMS:
            raise ValueError(f"batch exceeds {_MAX_BATCH_CLAIMS} claims")
        normalized.append((prompt_len, max_new))
    m = Module(name=f"batched_sessions_{len(sessions)}x")
    m.add_resource(
        Resource(rid=_WTS, domain=Domain.RAM, shape=(max(1, spec.n_layers),), name="WTS")
    )
    for s, (prompt_len, max_new) in enumerate(normalized):
        _add_session(m, spec, s, prompt_len, max_new)
    return m


def admit_session(m: Module, spec: DecoderSpec, prompt_len: int, max_new: int) -> Module:
    """A3: mid-flight admission IS appending phases. Session s joins a LIVE batched
    module as a fresh rid band + its own prefill->decode chain; no existing phase or
    claim changes, and the result is CLAIM-IDENTICAL to having built the batch upfront
    (pinned by hash_module equality in the tests) -- so the scheduler's overlap story
    is unchanged by WHEN a session arrived."""
    if not isinstance(spec, DecoderSpec):
        raise ValueError("admitted session spec is not a DecoderSpec")
    s = _session_count(m, spec)
    _validate_session_shape(prompt_len, max_new)
    existing_claims = sum(len(ph.claims) for ph in m.phases)
    if s >= _MAX_BATCH_SESSIONS or existing_claims + 1 + max_new > _MAX_BATCH_CLAIMS:
        raise ValueError("admitted session exceeds the bounded batch inventory")
    _add_session(m, spec, s, prompt_len, max_new)
    m.name = f"batched_sessions_{s + 1}x"
    return m


@dataclass(frozen=True)
class BatchCertificate:
    """The rung-7 witness (the StreamCertificate recipe over SESSIONS): the priced
    makespans of the same batch under serial / phase-barriered EFT / token-DAG
    disciplines. The overlap win is what continuous batching buys, and it comes from
    the EXISTING scheduler discovering that disjoint sessions overlap -- continuous
    batching is wave scheduling, not new machinery."""

    sessions: int
    serial: int
    barriered: int
    pipelined: int

    @property
    def overlap_win(self) -> int:
        return self.barriered - self.pipelined  # >= 0: what batching buys

    @property
    def admitted(self) -> bool:
        return 0 < self.pipelined <= self.barriered <= self.serial


def certify_batch(
    spec: DecoderSpec,
    sessions: tuple,
    h: HProfile,
    theta: Theta,
    policy: Policy = PERF,
    *,
    module: Module | None = None,
) -> tuple:
    """Price + place one batch of sessions: optimize the merged module (per-claim costs
    -> durations), schedule it barriered AND token-pipelined, certify the win. Returns
    (certificate, the pipelined schedule). Pass `module` to certify a LIVE (e.g.
    mid-flight-admitted) module instead of rebuilding from `sessions`."""
    m = module if module is not None else batched_sessions_module(spec, sessions)
    count = _session_count(m, spec)
    if module is not None and sessions and len(sessions) != count:
        raise ValueError("certificate session list does not match the supplied module")
    result = optimize(m, h, theta, policy)
    dur = durations_from(result)
    barriered = schedule_eft(m, dur, target=h)
    pipelined: GemSchedule = execute_tokens(m, dur, target=h)
    cert = BatchCertificate(
        sessions=count,
        serial=sum(dur.values()),
        barriered=barriered.makespan,
        pipelined=pipelined.makespan,
    )
    return cert, pipelined
