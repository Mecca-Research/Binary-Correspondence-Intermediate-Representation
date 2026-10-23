"""PX-1 (TRACE): the native training step's launches, as the step program states them.

`training_step_program` (native_program.py) states the step as values and kernel calls -- the
vocabulary GEM+'s bounds read. What reaches the kernels is that program under the decisions a
trainer takes, and nothing else checked it until now: which calls share one pool job (a linear's
two VJPs, NDT-SCHED-0; the linears reading one input, NDT-SCHED-1), the entry each GEMM takes
under the state's contract (v1, fused, blocked, bf16, a provider's), the panels the SwiGLU block
writes and reads instead of packs (BF16-PANELS), the slot a recomputed layer's cache lives in and
the forward the backward re-runs (REMAT), the scans and fills a step's facts leave out (NDT-SCAN,
NDT-ZERO, FACTS-RUN), and where in its arena every operand sits. `run_launches` states all of it
for one `bcir_decoder_run_accum` call -- `steps` updates of `accumulation` micro-batches each --
from the program's buffers and the trainer's decisions alone: its contract, recompute set and
workspace, the panels' threshold (`bcir_decoder_workspace_panels`, the C's own figure, as
`memory_plan` reads it), its arenas' alignment and whether a provider replaces the GEMM or SiLU.

The C rail is held to it, call for call. A build defining BCIR_DT_TRACE_PROBE records each launch
the step issues -- every call that reaches the tensor unit, a provider or the pool, in order -- as
`RECORD_WORDS` words (the layout below, the probe's): `read_trace` decodes them, and
`launch_mismatches(planned, traced)` lists every difference -- a launch extra or missing, calls
grouped otherwise, another entry, shape, scalar or operand place. The one launch the plan cannot
state ahead is the update's preflight, which runs only when the first pass's maxima fail to prove
every candidate in range (`adamw_bounded`, a fact of the values): it is planned `optional`, and a
trace may hold it there or not. `train.plan.executed.mismatch` counts the configurations whose
trace differs from their plan (`bcir/tests/native_decoder_fixtures.py`).

PX-2 (PACK) runs the plan. `step_program` is the same walk with the host's records in place -- a
CHECK where the step tests what its kernels reported (the forward's verdict, a refused product, the
micro-batch's loss) and an UPDATE where it updates -- and with the decisions a planner may take
otherwise than the C step does: calls run alone instead of grouped (`group`) or paired (`pair`), the
SwiGLU block without its panels (`panels`), every scan made whatever the run's facts know (`facts`),
and any recompute set whose cache fits the state's arena. `program_words` spells it as the words
`bcir_decoder_run_program` runs, and `program_launches` is what its run issues -- the kernel records,
each UPDATE as the update's pass, preflight and commit -- so `run_launches` is `step_program`'s
projection, and a program's trace is held to the program as PX-1 holds the step's to its plan
(`train.plan.program.mismatch`).

A record: kind, launch (the calls of one job share it), flags, eight dimensions, then ten operands
as (space, byte offset, bytes) -- spaces 1-4 the parameter arenas (weights, gradients, first and
second moments), 5 activations, 6 scratch, 7 the workspace, 8 outside the state (the
caller's tokens and targets, offset 0); an absent operand is (0, 0, 0). A GEMM call's flags are its
entry (0 v1, 1 fused, 2 blocked, 3 bf16 -- each a call alone; 4 a pair's call, 5 a group's, 6 a
bf16 group's; 7 a provider's) | role << 4 (0 alone, 1 a pair's first or a chain's, 2 the pair's
second or an independent call) | index << 8 | contract << 16 (15 none) | A a panel << 24 | B a
panel << 25.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, replace

from ...frontends.models.decode import DecoderSpec
from .native_program import ARENAS, recompute_layers, training_step_program

#: Record kinds, as the probe numbers them.
SCAN, ZERO, EMBED, RMS, GEMM, ROPE, ATTN, ATTN_VJP, ADD, SILU_MUL = range(1, 11)
SILU_MUL_PANELS, SWIGLU_VJP, SWIGLU_VJP_PANELS, RMS_VJP, LOSS, SCATTER = range(11, 17)
PASS, PREFLIGHT, COMMIT, MUL, PROVIDER_SILU = range(17, 22)
#: The host's records of a program (PX-2): a check, the update -- and a check's three tests.
CHECK, UPDATE = 22, 23
CHECK_FORWARD, CHECK_RAN, CHECK_LOSS = 1, 2, 4
KIND_NAMES = {
    SCAN: "scan", ZERO: "zero", EMBED: "embed", RMS: "rms", GEMM: "gemm", ROPE: "rope",
    ATTN: "attention", ATTN_VJP: "attention.vjp", ADD: "add", SILU_MUL: "silu_mul",
    SILU_MUL_PANELS: "silu_mul.panels", SWIGLU_VJP: "swiglu.vjp",
    SWIGLU_VJP_PANELS: "swiglu.vjp.panels", RMS_VJP: "rms.vjp", LOSS: "loss", SCATTER: "scatter",
    PASS: "update.pass", PREFLIGHT: "update.preflight", COMMIT: "update.commit", MUL: "mul",
    PROVIDER_SILU: "provider.silu",
}  # fmt: skip
#: Every kind a program may hold or a trace record: the trace's and the host's.
PROGRAM_KIND_NAMES = KIND_NAMES | {CHECK: "check", UPDATE: "update"}
#: Operand spaces: the six arenas in the state's order, then the workspace and the outside.
SPACES = {name: i + 1 for i, name in enumerate(ARENAS)} | {"workspace": 7, "external": 8}
#: A record's words: kind, launch, flags, 8 dimensions, 10 operands of 3 words, the job's instance
#: and workers (PX-3b: `WORD_INSTANCE`, `WORD_WORKERS`), 5 spare.
RECORD_WORDS = 48
DIMS, REFS = 8, 10
#: GEMM entries (the flags' low nibble).
E_V1, E_FUSED, E_KBLOCK, E_BF16, E_PAIR, E_GROUP, E_BF16_GROUP, E_PROVIDER = range(8)

# the state's contract bits (bcir_decoder_train.h) and the tensor unit's flags (bcir_tensor.h)
_FUSED, _EXP, _KBLOCK, _REDUCE, _BF16, _ATTN, _ATTNBF16 = 1, 2, 4, 8, 16, 32, 64
_ATTENTION_EXP, _ATTENTION_FUSED, _ATTENTION_LOWER = 1, 2, 16
_RMSB_ADD, _RMSB_FIRST = 1, 2
# loss_backward_with's facts (KNOWN, DEFER, COMMITTED) and the update's chunk rule
_KNOWN, _DEFER, _COMMITTED = 1, 2, 4
_PASS_CHUNK = _PASS_TASKS = 1024
_GROUP = 4  # BCIR_TENSOR_MM_GROUP

_NONE = (0, 0, 0)


def _bits(value: float) -> int:
    """A float's binary32 bits (the probe records alpha, beta and eps as the C holds them)."""
    return struct.unpack("<I", struct.pack("<f", value))[0]


_ONE = _bits(1.0)


@dataclass(frozen=True)
class Record:
    """One call the step issues: its kind, flags, dimensions and operands (space, offset, bytes)."""

    kind: int
    flags: int
    dims: tuple[int, ...]
    refs: tuple[tuple[int, int, int], ...]
    instance: int = 0  # PX-3b: word 41, the job's instance (0 the widest its family admits)
    workers: int = 0  # PX-3b: word 42, the job's workers (0 all the pool's threads)

    def describe(self) -> str:
        operands = " ".join(
            f"{_SPACE_NAMES.get(s, s)}+{o}/{b}" for s, o, b in self.refs if (s, o, b) != _NONE
        )
        name = PROGRAM_KIND_NAMES.get(self.kind, self.kind)
        how = (f" instance={self.instance}" if self.instance else "") + (
            f" workers={self.workers}" if self.workers else ""
        )
        return f"{name} flags={self.flags:#x} dims={list(self.dims)} {operands}{how}"


_SPACE_NAMES = {v: k for k, v in SPACES.items()}


@dataclass(frozen=True)
class Launch:
    """One job the step issues: its calls (a GEMM job may hold several), and whether the plan
    allows its absence (the update's preflight: a fact of the values, not of the plan)."""

    records: tuple[Record, ...]
    optional: bool = False


def _record(kind: int, flags: int = 0, dims=(), refs=()) -> Record:
    dims = tuple(int(x) for x in dims) + (0,) * (DIMS - len(dims))
    refs = tuple(refs) + (_NONE,) * (REFS - len(refs))
    return Record(kind, flags, dims, refs)


def reduce_chunk(count: int) -> int:
    """bcir_decoder_reduce_chunk: 1024 doubled until the chunks number at most 1024."""
    chunk = _PASS_CHUNK
    while -(-count // chunk) > _PASS_TASKS:
        chunk *= 2
    return chunk


def _layout(program) -> dict[str, tuple[int, int, int]]:
    """Every buffer of the program: (space, byte offset, bytes), its arena's spans in order."""
    at = dict.fromkeys(ARENAS, 0)
    out = {}
    for buf in program.buffers:
        size = 4 * buf.elements
        out[buf.name] = (SPACES[buf.arena], at[buf.arena], size)
        at[buf.arena] += size
    return out


def run_launches(
    spec: DecoderSpec,
    batch: int,
    time: int,
    *,
    contract: int,
    recompute: int = 0,
    workspace_bytes: int = 0,
    panels_need: int = 0,
    alignment: dict | None = None,
    provider_mm: bool = False,
    provider_silu: bool = False,
    steps: int = 1,
    accumulation: int = 1,
    threads: int = 1,
    instances: dict | None = None,
) -> list[Launch]:
    """The launches one `bcir_decoder_run_accum(state, ..., steps, accumulation, ...)` call issues
    under the given decisions (module docstring): the launches of the program that takes the C
    step's own. `contract` is the state's bits, `alignment` each space's base address modulo 64
    (the panels need 64-byte slots; absent: aligned); `threads` and `instances` the host's
    (PX-3b: what each launch is traced as running on, `program_launches`)."""
    return program_launches(
        step_program(
            spec,
            batch,
            time,
            contract=contract,
            recompute=recompute,
            workspace_bytes=workspace_bytes,
            panels_need=panels_need,
            alignment=alignment,
            provider_mm=provider_mm,
            provider_silu=provider_silu,
            steps=steps,
            accumulation=accumulation,
        ),  # fmt: skip
        threads=threads,
        instances=instances,
    )


#: PX-3 (CHOOSE): the program's choice sites -- each layer's sites (`L{layer}.{kind}`) and the
#: head's VJP (`head.vjp`) -- and each site's alternatives, every one training the same bits
#: (NDT-SCHED-0/1, BF16-PANELS): `group` one pool job of the linears that read one input (and of
#: their VJPs), `pair` one job of a linear's two VJPs, `calls` each call a job of its own; `panels`
#: the SwiGLU block's bf16 panels where the program takes them at all, `plain` the block without.
SITE_KINDS = {
    "qkv.fwd": ("group", "calls"),
    "gateup.fwd": ("group", "calls"),
    "panels": ("panels", "plain"),
    "down.vjp": ("pair", "calls"),
    "gateup.vjp": ("group", "pair", "calls"),
    "o.vjp": ("pair", "calls"),
    "qkv.vjp": ("group", "pair", "calls"),
}
HEAD_SITE, HEAD_ALTERNATIVES = "head.vjp", ("pair", "calls")


#: PX-3b (DISPATCH): how a job runs, the other half of the planner's choices -- the instance of a
#: GEMM or an attention (word 41: 0 the widest its family admits, else one bit) and the job's workers
#: (word 42: 0 all the pool's threads, else 1 .. its threads), alike in every record of the job,
#: each the same bits at every value (bcir_decoder_train.h). A DISPATCH KIND names the jobs a choice
#: applies to: a GEMM job by the linear it runs -- each layer's q/k/v, o, gate/up and down
#: projections forward (`*.fwd`) and their VJPs (`*.vjp`), the head's -- and every other job by its
#: kernel (`KIND_NAMES`); an alternative is `i<bit>` (that instance) or `t<k>` (k workers).
WORD_INSTANCE, WORD_WORKERS = 41, 42
GEMM_DISPATCH_KINDS = ("qkv.fwd", "o.fwd", "gateup.fwd", "down.fwd", "head.fwd", "head.vjp",
                       "down.vjp", "gateup.vjp", "o.vjp", "qkv.vjp")  # fmt: skip
#: the kernels a program holds (no trace-only record, no CHECK), each a dispatch kind of its own
KERNEL_DISPATCH_KINDS = tuple(
    PROGRAM_KIND_NAMES[k] for k in sorted(PROGRAM_KIND_NAMES)
    if k not in (GEMM, PASS, PREFLIGHT, COMMIT, CHECK)
)  # fmt: skip
DISPATCH_KINDS = GEMM_DISPATCH_KINDS + KERNEL_DISPATCH_KINDS
#: the kinds whose jobs may name an instance (a provider's GEMM is not BCIR's to name)
INSTANCE_KINDS = frozenset(GEMM_DISPATCH_KINDS) | {"attention", "attention.vjp"}
#: an instance's bits, as bcir_tensor.h numbers them
INSTANCE_BITS = {1: "portable", 2: "avx2", 4: "avx512", 8: "amx"}


def _widest(mask: int) -> int:
    return next(bit for bit in (8, 4, 2, 1) if mask & bit)


def dispatch_alternatives(*, threads: int, instances: dict, provider_mm: bool = False,
                          **_decisions) -> dict[str, tuple[str, ...]]:  # fmt: skip
    """Every dispatch kind and the alternatives this host admits for it other than the default:
    an instance kind each instance its family admits but the widest (`instances`: the families'
    masks, as `NativeDecoder.program_decisions` reads them), every kind each worker count below the
    pool's `threads`. A provider's GEMMs name no instance. Other decisions are accepted and
    ignored, so a trainer's whole decisions can be passed."""
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    out = {}
    for kind in DISPATCH_KINDS:
        alts = []
        if kind in INSTANCE_KINDS and not (provider_mm and kind in GEMM_DISPATCH_KINDS):
            family = "gemm" if kind in GEMM_DISPATCH_KINDS else "attention"
            mask = instances[family]
            alts += [f"i{bit}" for bit in (1, 2, 4, 8) if mask & bit and bit != _widest(mask)]
        alts += [f"t{k}" for k in range(1, threads)]
        out[kind] = tuple(alts)
    return out


def parse_dispatch(kind: str, alternative) -> tuple[str, int]:
    """("instance" | "workers", value) for a dispatch alternative's spelling -- `i<bit>` an
    instance (1, 2, 4 or 8, of a kind that names one), `t<k>` k workers (k >= 1, no leading zero,
    ASCII digits) -- refusing any other."""
    if type(alternative) is not str or len(alternative) < 2 or alternative[0] not in "it":
        raise ValueError(f"{kind}: {alternative!r} is not i<bit> or t<workers>")
    digits = alternative[1:]
    if not digits.isascii() or not digits.isdigit() or (digits[0] == "0"):
        raise ValueError(f"{kind}: {alternative!r} is not i<bit> or t<workers>")
    value = int(digits)
    if alternative[0] == "i":
        if value not in INSTANCE_BITS:
            raise ValueError(f"{kind}: {alternative!r} names no instance (1, 2, 4 or 8)")
        if kind not in INSTANCE_KINDS:
            raise ValueError(f"{kind}: names no instance (only a GEMM or an attention does)")
        return "instance", value
    return "workers", value


def choice_sites(spec: DecoderSpec) -> dict[str, tuple[str, ...]]:
    """Every choice site of `spec`'s step program and its alternatives: the layers' in layer order,
    then the head's."""
    out = {f"L{layer}.{kind}": alternatives for layer in range(spec.n_layers)
           for kind, alternatives in SITE_KINDS.items()}  # fmt: skip
    out[HEAD_SITE] = HEAD_ALTERNATIVES
    return out


def default_sites(spec: DecoderSpec, *, group: bool = True, pair: bool = True,
                  panels: bool = True) -> dict[str, str]:  # fmt: skip
    """The alternative every site takes under `step_program`'s booleans: `group` the linears'
    sites grouped, else their VJPs paired as `pair` says; `pair` a linear's two VJPs one job;
    `panels` the SwiGLU block's panels."""

    def pick(alternatives):
        if "panels" in alternatives:
            return "panels" if panels else "plain"
        if group and "group" in alternatives:
            return "group"
        return "pair" if pair and "pair" in alternatives else "calls"

    return {site: pick(alternatives) for site, alternatives in choice_sites(spec).items()}


def step_program(
    spec: DecoderSpec,
    batch: int,
    time: int,
    *,
    contract: int,
    recompute: int = 0,
    workspace_bytes: int = 0,
    panels_need: int = 0,
    alignment: dict | None = None,
    provider_mm: bool = False,
    provider_silu: bool = False,
    steps: int = 1,
    accumulation: int = 1,
    group: bool = True,
    pair: bool = True,
    panels: bool = True,
    facts: bool = True,
    sites: dict | None = None,
    labels: list | None = None,
    threads: int = 1,
    instances: dict | None = None,
    dispatch: dict | None = None,
    dispatch_labels: list | None = None,
) -> list[Launch]:
    """The program of one `bcir_decoder_run_program(state, ..., steps, accumulation, ...)` call:
    `run_launches`' walk with the host's CHECK and UPDATE records in place, under the given
    decisions -- the state's (`run_launches`' arguments, the state the program runs on) and the
    planner's own: `group` False runs the linears that read one input as calls of their own (and
    their VJPs as pairs), `pair` False a linear's two VJPs as two calls, `panels` False the SwiGLU
    block without its panels, `facts` False makes every scan the run's facts would leave out; and
    `recompute` is the program's own set, whose cache must fit the state's arena. Every decision
    trains the same bits (NDT-SCHED-0/1, BF16-PANELS, FACTS-RUN, REMAT).

    PX-3: `sites` takes any choice site's alternative (`choice_sites`) in place of the booleans'
    (`default_sites`) -- a site or alternative the program does not have is refused -- and a list
    passed as `labels` receives each launch's site, None for a launch no choice site holds: the
    site's GEMMs, and the SwiGLU block's kernels for its `panels` site.

    PX-3b: `dispatch` maps a dispatch kind (`DISPATCH_KINDS`) to an alternative -- `i<bit>` the
    instance its jobs run, `t<k>` the workers they run on -- into every record of every job of the
    kind (words 41 and 42); a kind or alternative outside the grammar is refused, and so, given the
    trainer's `threads` and `instances` (`NativeDecoder.program_decisions`), one this host does not
    admit; a list passed as `dispatch_labels` receives each launch's dispatch kind."""
    if type(steps) is not int or steps < 1 or type(accumulation) is not int or accumulation < 1:
        raise ValueError("steps and accumulation must be positive integers")
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    chosen_dispatch: dict[str, tuple[str, int]] = {}
    if dispatch is not None:
        if not isinstance(dispatch, dict):
            raise ValueError("dispatch maps a dispatch kind to an alternative")
        for kind, alternative in dispatch.items():
            if type(kind) is not str or kind not in DISPATCH_KINDS:
                raise ValueError(f"no dispatch kind {kind!r}")
            what, value = parse_dispatch(kind, alternative)
            if what == "instance":
                if provider_mm and kind in GEMM_DISPATCH_KINDS:
                    raise ValueError(f"{kind}: a provider's GEMM names no instance")
                family = "gemm" if kind in GEMM_DISPATCH_KINDS else "attention"
                if instances is not None and not instances[family] & value:
                    raise ValueError(
                        f"{kind}: the host's {family} family admits no instance {value}"
                    )
            elif value > threads:
                raise ValueError(f"{kind}: {value} workers, past the pool's {threads} threads")
            chosen_dispatch[kind] = (what, value)
    if dispatch_labels is not None and (not isinstance(dispatch_labels, list) or dispatch_labels):
        raise ValueError("dispatch_labels is an empty list the launches' kinds are appended to")
    if any(type(flag) is not bool for flag in (group, pair, panels, facts)):
        raise ValueError("group, pair, panels and facts are booleans")
    known = choice_sites(spec)
    chosen = default_sites(spec, group=group, pair=pair, panels=panels)
    if sites is not None:
        if not isinstance(sites, dict):
            raise ValueError("sites maps a choice site to one of its alternatives")
        for site, alternative in sites.items():
            if type(site) is not str or site not in known or alternative not in known[site]:
                raise ValueError(f"no choice site {site!r} with the alternative {alternative!r}")
            chosen[site] = alternative
    if labels is not None and (not isinstance(labels, list) or labels):
        raise ValueError("labels is an empty list the program's sites are appended to")
    current: list = [None]
    gemm_kind: list = [None]  # the dispatch kind of the GEMMs emitted from here on (PX-3b)
    program = training_step_program(spec, batch, time, recompute)
    lay = _layout(program)
    redone = set(recompute_layers(spec, recompute))
    top = max(redone, default=None)
    align = {SPACES[name]: 0 for name in (*ARENAS, "workspace")} | dict(alignment or {})
    n, d, L = batch * time, spec.d_model, spec.n_layers
    h, kh = spec.n_heads, spec.kv_heads
    hd = d // h
    k, f, v = kh * hd, spec.d_ff, spec.vocab_size
    nd, nf = n * d, n * f
    params = sum(buf.elements for buf in program.buffers if buf.arena == "weights")
    eps = _bits(spec.rms_norm_eps)
    gemm_contract = (
        3 if contract & _BF16 else 2 if contract & _KBLOCK else 1 if contract & _FUSED else 0
    )
    v2 = 1 if contract & _EXP else 0
    attention_contract = (_ATTENTION_EXP if contract & _EXP else 0) | (
        _ATTENTION_FUSED if contract & _ATTN else 0
    )
    ws = (SPACES["workspace"], 0, workspace_bytes) if gemm_contract == 3 else _NONE
    # ATT-CONTIG: the binary32 attention is lent the state's workspace under any contract, where it
    # has bytes -- the entry copies a head's rows into it from its need on, and runs them in place below
    ws_attention = (SPACES["workspace"], 0, workspace_bytes) if workspace_bytes else _NONE
    out: list[Launch] = []

    def ref(name: str, offset: int = 0, size: int | None = None):
        space, at, whole = lay[name]
        return (space, at + offset, whole - offset if size is None else size)

    def ext(size: int):
        return (SPACES["external"], 0, size)

    def emit(*records: Record, optional: bool = False) -> None:
        if records[0].kind == GEMM:
            kind = gemm_kind[0]
            if kind is None:
                raise AssertionError("a GEMM emitted outside a dispatch kind")
        else:
            kind = PROGRAM_KIND_NAMES[records[0].kind]
        how = chosen_dispatch.get(kind)
        if how is not None:
            records = tuple(replace(r, **{how[0]: how[1]}) for r in records)
        out.append(Launch(tuple(records), optional))
        if labels is not None:
            labels.append(current[0])
        if dispatch_labels is not None:
            dispatch_labels.append(None if records[0].kind == CHECK else kind)

    def as_gemms(kind: str) -> None:
        """The dispatch kind of the GEMMs emitted from here on."""
        gemm_kind[0] = kind

    def within(site: str | None) -> None:
        """The site the launches emitted from here on belong to (None: none)."""
        current[0] = site

    def aligned(r) -> bool:
        return (align[r[0]] + r[1]) % 64 == 0

    def gemm_flags(entry, role, index, contract_field, panel_a=False, panel_b=False) -> int:
        return (entry | role << 4 | index << 8 | contract_field << 16 | int(panel_a) << 24
                | int(panel_b) << 25)  # fmt: skip

    def call(entry, role, index, contract_field, c, pre=(None, None), workspace=_NONE) -> Record:
        """`c` = (ta, tb, m, n, k, beta, a, b, out) with operand refs; `pre` the panels' refs."""
        ta, tb, m, nn, kk, beta, a, b, y = c
        pa, pb = pre
        refs = (
            (pa[0], pa[1], m * kk * 2) if pa else (a[0], a[1], m * kk * 4) if a else _NONE,
            (pb[0], pb[1], kk * nn * 2) if pb else (b[0], b[1], kk * nn * 4) if b else _NONE,
            (y[0], y[1], m * nn * 4),
            workspace,
        )
        flags = gemm_flags(entry, role, index, contract_field, pa is not None, pb is not None)
        return _record(GEMM, flags, (ta, tb, m, nn, kk, _ONE, _bits(beta)), refs)

    def gemm(c) -> None:
        if provider_mm:
            emit(call(E_PROVIDER, 0, 0, 15, c))
        elif gemm_contract == 3:
            emit(call(E_BF16, 0, 0, 3, c, workspace=ws))
        else:
            emit(call(gemm_contract, 0, 0, gemm_contract, c))

    def gemm_job(chain, rest, pre=None) -> None:
        """gemm_group_with: one job, the chain then the rest (`pre` per call, bf16 only)."""
        calls = list(chain) + list(rest)
        pre = pre or [(None, None)] * len(calls)
        if gemm_contract == 3:
            emit(*(call(E_BF16_GROUP, 1 if q < len(chain) else 2, q, 3, c, pre[q], ws)
                   for q, c in enumerate(calls)))  # fmt: skip
        else:
            emit(*(call(E_GROUP, 1 if q < len(chain) else 2, q, gemm_contract, c)
                   for q, c in enumerate(calls)))  # fmt: skip

    def linear(rows, inp, outp, x, w, y) -> None:
        gemm((0, 1, rows, outp, inp, 0.0, x, w, y))

    def linear_backward(rows, inp, outp, x, w, dy, dx, dw, beta, wbeta, site) -> None:
        cx = (0, 0, rows, inp, outp, beta, dy, w, dx)
        cw = (1, 0, outp, inp, rows, wbeta, dy, x, dw)
        if provider_mm or chosen[site] == "calls":
            gemm(cx)
            gemm(cw)
        elif gemm_contract == 3:
            gemm_job([cx], [cw])
        else:
            emit(call(E_PAIR, 1, 0, gemm_contract, cx), call(E_PAIR, 2, 1, gemm_contract, cw))

    def linears(rows, inp, x, outs, ws_, ys, site) -> None:
        if provider_mm or chosen[site] != "group" or len(outs) > _GROUP:
            for o, w, y in zip(outs, ws_, ys):
                linear(rows, inp, o, x, w, y)
            return
        gemm_job([], [(0, 1, rows, o, inp, 0.0, x, w, y) for o, w, y in zip(outs, ws_, ys)])

    def linears_backward(rows, inp, x, outs, ws_, dys, dx, dws, beta, wbeta, site) -> None:
        if provider_mm or chosen[site] != "group" or len(outs) > _GROUP:
            for q, (o, w, dy, dw) in enumerate(zip(outs, ws_, dys, dws)):
                linear_backward(rows, inp, o, x, w, dy, dx, dw, 1.0 if q else beta, wbeta, site)
            return
        cx = [(0, 0, rows, inp, o, 1.0 if q else beta, dy, w, dx)
              for q, (o, w, dy) in enumerate(zip(outs, ws_, dys))]  # fmt: skip
        cw = [(1, 0, o, inp, rows, wbeta, dy, x, dw) for o, dy, dw in zip(outs, dys, dws)]
        gemm_job(cx, cw)

    def scan(r) -> None:
        emit(_record(SCAN, 0, (r[2] // 4,), (r,)))

    def rms(x, g, y) -> None:
        emit(_record(RMS, 0, (n, d, eps), (x, g, y)))

    def rope(x, heads, inverse) -> None:
        emit(_record(ROPE, 0, (batch, time, heads, hd, inverse), (x,)))

    def add(x, y) -> None:
        emit(_record(ADD, 0, (nd,), (x, y)))

    def silu_mul(x, u, y) -> None:
        if provider_silu:
            emit(_record(PROVIDER_SILU, 0, (nf,), (x, y)))
            emit(_record(MUL, 0, (nf,), (u, y)))
            for r in (x, u, y):
                scan(r)
        else:
            emit(_record(SILU_MUL, v2, (nf,), (x, u, y)))

    def use_panels(a, b, site) -> bool:
        """panels(): the bf16 contract, BCIR's GEMM and SiLU, whole 32-tiles, 64-byte slots, and
        the workspace at the panel groups' need -- where the program takes the panels at all."""
        return (chosen[site] == "panels" and gemm_contract == 3 and not provider_mm
                and not provider_silu and n % 32 == 0
                and d % 32 == 0 and f % 32 == 0 and aligned(a) and aligned(b)
                and workspace_bytes >= panels_need)  # fmt: skip

    def attention(q, kk, vv, p, y) -> None:
        dims = (batch, time, h, kh, hd)
        if contract & _ATTNBF16:
            emit(_record(ATTN, attention_contract | 1 << 8, dims, (q, kk, vv, p, y, _NONE, _NONE,
                                                                   _NONE, ws)))  # fmt: skip
        else:  # P-LOWER: the binary32 backward reads no probability past a row's diagonal
            flags = attention_contract | _ATTENTION_LOWER
            emit(_record(ATTN, flags, dims, (q, kk, vv, p, y, _NONE, _NONE, _NONE,
                                             ws_attention)))  # fmt: skip

    def attention_backward(q, kk, vv, p, dy, dq, dk, dv) -> None:
        dims = (batch, time, h, kh, hd)
        if contract & _ATTNBF16:
            emit(_record(ATTN_VJP, 1 << 8, dims, (q, kk, vv, p, dy, dq, dk, dv, ws)))
        else:
            emit(_record(ATTN_VJP, attention_contract, dims, (q, kk, vv, p, dy, dq, dk, dv,
                                                              ws_attention)))  # fmt: skip

    def rms_backward(x, g, dy, dx, dg, inv, how) -> None:
        emit(_record(RMS_VJP, how, (n, d, eps), (x, g, dy, dx, dg, inv)))

    def slot(layer: int) -> str:
        return "R" if layer in redone else f"L{layer}"

    def layer_forward(layer: int, full: bool) -> None:
        c = lambda name: ref(f"{slot(layer)}.{name}")  # noqa: E731
        p = lambda name: ref(f"weights:L{layer}.{name}")  # noqa: E731
        x, y = ref(f"x{layer}"), ref(f"x{layer + 1}")
        rms(x, p("norm1"), c("n1"))
        within(f"L{layer}.qkv.fwd")
        as_gemms("qkv.fwd")
        linears(n, d, c("n1"), (d, k, k), (p("q"), p("k"), p("v")), (c("q"), c("k"), c("v")),
                f"L{layer}.qkv.fwd")  # fmt: skip
        within(None)
        rope(c("q"), h, 0)
        rope(c("k"), kh, 0)
        attention(c("q"), c("k"), c("v"), c("p"), c("ctx"))
        as_gemms("o.fwd")
        linear(n, d, d, c("ctx"), p("o"), c("x2"))
        add(x, c("x2"))
        rms(c("x2"), p("norm2"), c("n2"))
        within(f"L{layer}.gateup.fwd")
        as_gemms("gateup.fwd")
        linears(n, d, c("n2"), (f, f), (p("gate"), p("up")), (c("gate"), c("up")),
                f"L{layer}.gateup.fwd")  # fmt: skip
        within(f"L{layer}.panels")
        as_gemms("down.fwd")
        hidden = c("hidden")
        if use_panels(hidden, hidden, f"L{layer}.panels"):
            a, b = (
                ref(f"{slot(layer)}.hidden", 0, nf * 2),
                ref(f"{slot(layer)}.hidden", nf * 2, nf * 2),
            )
            emit(_record(SILU_MUL_PANELS, v2, (n, f), (c("gate"), c("up"), a, b)))
            if full:
                gemm_job([], [(0, 1, n, d, f, 0.0, None, p("down"), y)], [(a, None)])
        else:
            silu_mul(c("gate"), c("up"), hidden)
            if full:
                linear(n, f, d, hidden, p("down"), y)
        within(None)
        if full:
            add(c("x2"), y)

    head = "embed" if spec.tied_embeddings else "head"

    def forward() -> None:
        emit(_record(EMBED, 0, (n, d), (ext(4 * n), ref("weights:embed"), ref("x0"))))
        for layer in range(L):
            layer_forward(layer, True)
        rms(ref(f"x{L}"), ref("weights:final_norm"), ref("norm"))
        as_gemms("head.fwd")
        linear(n, d, v, ref("norm"), ref(f"weights:{head}"), ref("logits"))
        scan(ref("logits"))
        emit(_record(CHECK, CHECK_FORWARD))

    def loss_backward(accumulate: bool, how: int) -> None:
        if not how & (_KNOWN | _COMMITTED):
            scan(ref("weights:embed", 0, 4 * params))
        if not how & _KNOWN and accumulate:
            scan(ref("gradients:embed", 0, 4 * params))
        forward()
        first = not accumulate and not provider_mm
        wb, gf = (0.0, _RMSB_FIRST) if first else (1.0, 0)
        if not accumulate and not first:
            emit(_record(ZERO, 0, (params,), (ref("gradients:embed", 0, 4 * params),)))
        if not accumulate and first and not spec.tied_embeddings:
            emit(_record(ZERO, 0, (v * d,), (ref("gradients:embed"),)))
        sc = lambda name, off=0, size=None: ref(name, off, size)  # noqa: E731
        cur, dn, dq, dk, dv = sc("cur"), sc("dn"), sc("dq"), sc("dk"), sc("dv")
        dc, dg, du, dh, dl = sc("dc"), sc("dg"), sc("du"), sc("dh"), sc("dl")
        emit(_record(LOSS, 0, (n, v), (ref("logits"), ext(4 * n), dl, sc("cur", 0, 8 * n))))
        within(HEAD_SITE)
        as_gemms("head.vjp")
        linear_backward(n, d, v, ref("norm"), ref(f"weights:{head}"), dl, dn,
                        ref(f"gradients:{head}"), 0.0, wb, HEAD_SITE)  # fmt: skip
        within(None)
        inv = sc("dq", 0, 4 * n)
        rms_backward(ref(f"x{L}"), ref("weights:final_norm"), dn, cur,
                     ref("gradients:final_norm"), inv, gf)  # fmt: skip
        for layer in reversed(range(L)):
            c = lambda name, layer=layer: ref(f"{slot(layer)}.{name}")  # noqa: E731
            p = lambda name, layer=layer: ref(f"weights:L{layer}.{name}")  # noqa: E731
            q = lambda name, layer=layer: ref(f"gradients:L{layer}.{name}")  # noqa: E731
            if layer in redone and layer != top:
                layer_forward(layer, False)
            hidden = c("hidden")
            site = f"L{layer}.down.vjp"
            within(site)
            as_gemms("down.vjp")
            if use_panels(hidden, hidden, f"L{layer}.panels"):
                cx = (0, 0, n, f, d, 0.0, cur, p("down"), dh)
                cw = (1, 0, d, f, n, wb, cur, None, q("down"))
                panel = (None, ref(f"{slot(layer)}.hidden", nf * 2, nf * 2))
                if chosen[site] != "calls":
                    gemm_job([cx], [cw], [(None, None), panel])
                else:  # the two VJPs as two jobs: the panel's call a bf16 group of its own
                    gemm(cx)
                    gemm_job([], [cw], [panel])
            else:
                linear_backward(n, f, d, hidden, p("down"), cur, dh, q("down"), 0.0, wb, site)
            site = f"L{layer}.gateup.vjp"
            as_gemms("gateup.vjp")
            if use_panels(dg, du, f"L{layer}.panels"):
                within(f"L{layer}.panels")
                pg = (ref("dg", 0, nf * 2), ref("dg", nf * 2, nf * 2))
                pu = (ref("du", 0, nf * 2), ref("du", nf * 2, nf * 2))
                emit(_record(SWIGLU_VJP_PANELS, v2, (n, f),
                             (c("gate"), c("up"), dh, pg[0], pg[1], pu[0], pu[1])))  # fmt: skip
                cx = [(0, 0, n, d, f, 0.0, None, p("gate"), dn),
                      (0, 0, n, d, f, 1.0, None, p("up"), dn)]  # fmt: skip
                cw = [(1, 0, f, d, n, wb, None, c("n2"), q("gate")),
                      (1, 0, f, d, n, wb, None, c("n2"), q("up"))]  # fmt: skip
                pre = [(pg[0], None), (pu[0], None), (pg[1], None), (pu[1], None)]
                within(site)
                if chosen[site] == "group":
                    gemm_job(cx, cw, pre)
                else:  # each projection's VJPs a job of their own, as linear_backward's are
                    for j in range(2):
                        if chosen[site] == "pair":
                            gemm_job([cx[j]], [cw[j]], [pre[j], pre[2 + j]])
                        else:
                            gemm_job([], [cx[j]], [pre[j]])
                            gemm_job([], [cw[j]], [pre[2 + j]])
            else:
                within(f"L{layer}.panels")
                emit(_record(SWIGLU_VJP, v2, (nf,), (c("gate"), c("up"), dh, dg, du)))
                within(site)
                linears_backward(n, d, c("n2"), (f, f), (p("gate"), p("up")), (dg, du), dn,
                                 (q("gate"), q("up")), 0.0, wb, site)  # fmt: skip
            within(None)
            rms_backward(c("x2"), p("norm2"), dn, cur, q("norm2"), inv, _RMSB_ADD | gf)
            within(f"L{layer}.o.vjp")
            as_gemms("o.vjp")
            linear_backward(n, d, d, c("ctx"), p("o"), cur, dc, q("o"), 0.0, wb, f"L{layer}.o.vjp")
            within(None)
            attention_backward(c("q"), c("k"), c("v"), c("p"), dc, dq, dk, dv)
            rope(dq, h, 1)
            rope(dk, kh, 1)
            within(f"L{layer}.qkv.vjp")
            as_gemms("qkv.vjp")
            linears_backward(n, d, c("n1"), (d, k, k), (p("q"), p("k"), p("v")), (dq, dk, dv), dn,
                             (q("q"), q("k"), q("v")), 0.0, wb, f"L{layer}.qkv.vjp")  # fmt: skip
            within(None)
            rms_backward(ref(f"x{layer}"), p("norm1"), dn, cur, q("norm1"), inv, _RMSB_ADD | gf)
        emit(_record(SCATTER, 0, (n, d), (ext(4 * n), cur, ref("gradients:embed"))))
        emit(_record(CHECK, CHECK_RAN))
        if not how & _DEFER:
            scan(ref("gradients:embed", 0, 4 * params))
        emit(_record(CHECK, CHECK_LOSS))

    def update() -> None:
        # the first pass, the preflight and the commit read the gradients, then the weights and
        # the two moments, each whole
        quad = tuple(ref(f"{arena}:embed", 0, 4 * params)
                     for arena in ("gradients", "weights", "moment1", "moment2"))  # fmt: skip
        emit(_record(UPDATE, 1 if contract & _REDUCE else 0, (params, reduce_chunk(params)), quad))

    for step in range(steps):
        for micro in range(accumulation):
            how = (_KNOWN if micro else _COMMITTED if step else 0) | (
                _DEFER if micro + 1 == accumulation else 0
            )
            if not facts:  # every scan, whatever the run knows
                how = 0
            loss_backward(micro != 0, how)
        update()
    return out


def _ran(r: Record, threads: int, instances: dict | None) -> Record:
    """PX-3b: the record as the probe traces it -- the instance its job RAN (an entry's return:
    the family's widest for a 0, given the host's `instances`; a provider's GEMM and every other
    kernel none) and the workers its job ran on (a 0 the pool's `threads`)."""
    instance = r.instance
    if not instance and instances is not None:
        if r.kind == GEMM and r.flags & 15 != E_PROVIDER:
            instance = _widest(instances["gemm"])
        elif r.kind in (ATTN, ATTN_VJP):
            instance = _widest(instances["attention"])
    return replace(r, instance=instance, workers=min(r.workers, threads) or threads)


def program_launches(program: list[Launch], *, threads: int = 1,
                     instances: dict | None = None) -> list[Launch]:  # fmt: skip
    """The launches a program's run issues, as the probe traces them: its kernel jobs as they
    stand, each UPDATE as the update's pass, its preflight (optional: a fact of the values) and its
    commit; a CHECK issues none. PX-3b: each record with the instance and workers its job ran on
    (`_ran`) for a trainer of `threads` threads and the instance families `instances` --
    `NativeDecoder.program_decisions`' -- the defaults unresolved without `instances`."""
    out = []
    for launch in program:
        record = launch.records[0]
        if record.kind == CHECK:
            continue
        if record.kind == UPDATE:
            commit = (record.dims[0],) + (0,) * (DIMS - 1)
            update = _ran(record, threads, instances)
            out.append(Launch((replace(update, kind=PASS),)))
            out.append(Launch((replace(update, kind=PREFLIGHT, flags=0),), optional=True))
            out.append(Launch((replace(update, kind=COMMIT, flags=0, dims=commit),)))
        else:
            out.append(Launch(tuple(_ran(r, threads, instances) for r in launch.records),
                              launch.optional))  # fmt: skip
    return out


def program_words(program: list[Launch]) -> list[int]:
    """A program as the `RECORD_WORDS`-word records `bcir_decoder_run_program` runs: its jobs
    numbered from 1 in order, each record its kind, its job's number, flags, dimensions, operands,
    its job's instance and workers (`WORD_INSTANCE`, `WORD_WORKERS`: PX-3b) and five words of
    zero."""
    words: list[int] = []
    spare = RECORD_WORDS - 3 - DIMS - 3 * REFS - 2  # after the two dispatch words (PX-3b)
    for number, launch in enumerate(program, 1):
        for r in launch.records:
            words += [r.kind, number, r.flags, *r.dims, *(x for ref in r.refs for x in ref)]
            words += [r.instance, r.workers] + [0] * spare
    return words


def read_trace(words, count: int) -> list[tuple[int, Record]]:
    """The probe's `count` records from `words` (a flat sequence of 64-bit words) as (launch,
    record) pairs, in issue order."""
    out = []
    for i in range(count):
        w = words[i * RECORD_WORDS : (i + 1) * RECORD_WORDS]
        refs = tuple(tuple(w[11 + 3 * s : 14 + 3 * s]) for s in range(REFS))
        out.append((w[1], Record(w[0], w[2], tuple(w[3 : 3 + DIMS]), refs, w[WORD_INSTANCE],
                                 w[WORD_WORKERS])))  # fmt: skip
    return out


def launch_mismatches(planned: list[Launch], traced: list[tuple[int, Record]]) -> list[str]:
    """Every difference between the launches a plan states and the records a trace holds: a launch
    the trace lacks or adds, calls grouped otherwise, a record that differs. An optional launch
    may be absent; present, it must match. Empty when the C issued exactly the plan."""
    jobs: list[tuple[Record, ...]] = []
    previous = None
    for launch, record in traced:
        if jobs and launch == previous:
            jobs[-1] = jobs[-1] + (record,)
        else:
            jobs.append((record,))
        previous = launch
    problems: list[str] = []
    i = 0
    for index, launch in enumerate(planned):
        if i < len(jobs) and jobs[i] == launch.records:
            i += 1
            continue
        if launch.optional:
            continue
        got = jobs[i] if i < len(jobs) else ()
        problems.append(
            f"launch {index} ({KIND_NAMES.get(launch.records[0].kind)}): planned "
            f"[{'; '.join(r.describe() for r in launch.records)}], traced "
            f"[{'; '.join(r.describe() for r in got) or 'nothing'}]"
        )
        if not got:
            continue
        i += 1
    for extra in jobs[i:]:
        problems.append(f"traced beyond the plan: [{'; '.join(r.describe() for r in extra)}]")
    return problems


__all__ = [
    "DISPATCH_KINDS",
    "GEMM_DISPATCH_KINDS",
    "HEAD_ALTERNATIVES",
    "HEAD_SITE",
    "INSTANCE_BITS",
    "INSTANCE_KINDS",
    "KERNEL_DISPATCH_KINDS",
    "KIND_NAMES",
    "PROGRAM_KIND_NAMES",
    "Launch",
    "RECORD_WORDS",
    "Record",
    "SITE_KINDS",
    "SPACES",
    "choice_sites",
    "default_sites",
    "dispatch_alternatives",
    "launch_mismatches",
    "parse_dispatch",
    "program_launches",
    "program_words",
    "read_trace",
    "reduce_chunk",
    "run_launches",
    "step_program",
    "WORD_INSTANCE",
    "WORD_WORKERS",
]
