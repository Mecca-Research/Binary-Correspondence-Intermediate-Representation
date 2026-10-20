"""PX-3 (CHOOSE): the step program's per-site choices taken from measured candidates.

`native_launch.step_program` states a program for a state and the planner's choices, and PX-3 makes
those choices per SITE: each layer's q/k/v and gate/up projections (one pool job, or a call each),
its SwiGLU block's bf16 panels, the VJPs of its down, o, gate/up and q/k/v projections (one job, a
job a projection, or a call each), and the head's VJP (`native_launch.choice_sites`). Every
alternative trains the same bits (NDT-SCHED-0/1, BF16-PANELS); what differs is the time, and a time
is a fact of the host -- so the choice is MEASURED, never modelled:

* a library built with the job clock (`NativeDecoder.build(..., probe="time")`, BCIR_DT_TIME_PROBE)
  records the nanoseconds every job of a program run takes (`timed_run`), and `measure` runs the
  trainer's default program and, for each kind of site, the program with every site of that kind
  switched to an alternative, in rotated rounds on the one trainer -- they train the same bits, so
  they take turns on one state -- reading the switched sites' own launches in place
  (`native_launch.step_program(labels=...)`; for the panels kind, every launch the panels change);
* `estimate` turns the paired rounds into the alternative's time over the default's: the geometric
  mean and its 95% Student-t interval, as Q8 integers (256 = the same time);
* the result is a `ChoiceTable`: a frozen, generation-tagged integer table naming the shape it was
  measured at (the decoder, the call, the trainer's threads, recompute set, workspace and contract)
  and the host -- the one form a measurement may take into BCIR's planning (LangRef §13's learning
  law, `kbcir.microbench`'s tables): it ranks, it never decides legality, and the planning never
  measures;
* `choose` takes, for each kind, the alternative whose whole interval lies below 256 and whose mean
  is the lowest -- an alternative not resolved faster is never taken -- and every other site keeps
  its default; no table is the defaults (non-disturbance: the program is the one the booleans
  state, word for word).

The chosen program is then planned and verified like any other: `native_pack.training_pack(...,
sites=...)` states it as a module whose claims carry each call's role, K_BCIR plans it, the plan's
laws (R9 among them) verify it, and the C lowers it to the program it states. The measured prices
inform the choice and do not enter the plan's own costs: carrying them into K_BCIR's candidates on
its three rails (a BKPI table beside the target's constants) is the open half of the ladder's row.
"""

from __future__ import annotations

import ctypes as ct
import json
import math
import os
import platform
import statistics
from array import array
from dataclasses import dataclass

from ...frontends.models.decode import DecoderSpec
from .native_launch import (
    HEAD_ALTERNATIVES,
    HEAD_SITE,
    RECORD_WORDS,
    SITE_KINDS,
    choice_sites,
    default_sites,
    program_words,
    step_program,
)

__all__ = [
    "CHOICE_KINDS",
    "ChoiceEntry",
    "ChoiceTable",
    "Q8",
    "SCOPES",
    "choose",
    "estimate",
    "host_attestation",
    "kind_of",
    "measure",
    "scope",
    "timed_run",
    "trainer_shape",
]

#: 1.0 in Q8: an alternative as fast as the default.
Q8 = 256
#: Every kind of site and its alternatives, the layers' kinds then the head's.
CHOICE_KINDS = dict(SITE_KINDS) | {HEAD_SITE: HEAD_ALTERNATIVES}
#: The launches an alternative of a kind changes, by the kind of site their labels name: the
#: panels' change the down and gate/up VJPs too (the panels the GEMMs read instead of packing).
SCOPES = {"panels": ("panels", "down.vjp", "gateup.vjp")}
#: Student-t's two-sided 97.5% quantiles for the degrees of freedom a measurement takes.
_T975 = {
    4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
    29: 2.045, 30: 2.042, 31: 2.040, 35: 2.030, 39: 2.023, 47: 2.012, 63: 1.998,
}  # fmt: skip


def kind_of(site: str) -> str:
    """The kind of a site: `L3.qkv.vjp` -> `qkv.vjp`, the head's its own."""
    if site == HEAD_SITE:
        return site
    if type(site) is not str:
        raise ValueError(f"not a choice site: {site!r}")
    layer, _, kind = site.partition(".")
    digits = layer[1:]
    if (layer[:1] != "L" or not digits or not digits.isascii() or not digits.isdigit()
            or (digits != "0" and digits[0] == "0") or kind not in SITE_KINDS):  # fmt: skip
        raise ValueError(f"not a choice site: {site!r}")
    return kind


def scope(kind: str) -> tuple[str, ...]:
    """The kinds of site whose launches an alternative of `kind` changes."""
    if kind not in CHOICE_KINDS:
        raise ValueError(f"not a kind of choice site: {kind!r}")
    return SCOPES.get(kind, (kind,))


def _t975(df: int) -> float:
    """The quantile at `df` (four or more: `estimate`'s rounds), a df between the table's rows
    taken at the row below -- the larger quantile, the wider interval."""
    return _T975[max(k for k in _T975 if k <= df)]


def estimate(default_ns, alternative_ns) -> tuple[int, int, int]:
    """The alternative's time over the default's from paired rounds (round i of each measured in
    one rotated round): the geometric mean of the ratios and its 95% Student-t interval, each as
    Q8 (rounded half up). Refuses unpaired, short or non-positive measurements."""
    default_ns, alternative_ns = list(default_ns), list(alternative_ns)
    if len(default_ns) != len(alternative_ns) or len(default_ns) < 5:
        raise ValueError("estimate needs at least five paired rounds")
    if any(type(v) is not int or v <= 0 for v in (*default_ns, *alternative_ns)):
        raise ValueError("a round's time is a positive integer of nanoseconds")
    logs = [math.log(a / d) for d, a in zip(default_ns, alternative_ns)]
    mean = statistics.fmean(logs)
    half = _t975(len(logs) - 1) * statistics.stdev(logs) / math.sqrt(len(logs))

    def q8(x: float) -> int:
        return int(math.floor(Q8 * math.exp(x) + 0.5))

    return q8(mean), q8(mean - half), q8(mean + half)


@dataclass(frozen=True)
class ChoiceEntry:
    """One alternative of one kind of site, measured against the kind's default: its time over the
    default's (Q8), the interval's ends, and the paired rounds behind them."""

    kind: str
    alternative: str
    q8: int
    low_q8: int
    high_q8: int
    rounds: int


#: A table's shape: the decoder's dimensions, the call's batch, time and steps, the trainer's
#: threads -- each a positive integer -- its recompute set and workspace bytes (integers from 0),
#: and its contract by name: everything that fixes the program a choice applies to and the times
#: it was measured at.
_SHAPE_POSITIVE = ("vocab", "d_model", "n_heads", "n_layers", "d_ff", "kv_heads", "batch", "time",
                   "threads", "steps")  # fmt: skip
_SHAPE_NATURAL = ("recompute", "workspace_bytes")
_SHAPE_KEYS = frozenset((*_SHAPE_POSITIVE, *_SHAPE_NATURAL, "contract"))
_SPEC_KEYS = ("vocab", "d_model", "n_heads", "n_layers", "d_ff", "kv_heads")


def _checked_shape(shape) -> dict:
    if (not isinstance(shape, dict) or set(shape) != _SHAPE_KEYS
            or type(shape["contract"]) is not str):  # fmt: skip
        raise ValueError(f"a choice table's shape names {sorted(_SHAPE_KEYS)}")
    if any(type(shape[k]) is not int or shape[k] < 1 for k in _SHAPE_POSITIVE) or any(
        type(shape[k]) is not int or shape[k] < 0 for k in _SHAPE_NATURAL
    ):
        raise ValueError("a choice table's shape is positive dimensions, a recompute set, "
                         "workspace bytes and a contract")  # fmt: skip
    return dict(shape)


def _spec_shape(spec: DecoderSpec) -> tuple[int, ...]:
    return (spec.vocab_size, spec.d_model, spec.n_heads, spec.n_layers, spec.d_ff, spec.kv_heads)


def trainer_shape(trainer, *, steps: int = 1) -> dict:
    """The shape a table measured on `trainer` names, for calls of `steps` steps."""
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    shape = dict(zip(_SPEC_KEYS, _spec_shape(trainer.spec)))
    shape.update(batch=trainer.batch_size, time=trainer.context_length, threads=trainer.threads,
                 steps=steps, recompute=trainer.recompute,
                 workspace_bytes=trainer.workspace_bytes, contract=trainer.contract)  # fmt: skip
    return _checked_shape(shape)


@dataclass(frozen=True)
class ChoiceTable:
    """A frozen, generation-tagged measurement of the choice sites' alternatives (module
    docstring): the generation (>= 1), the shape it was measured at (`trainer_shape`), the host's
    attestation (as the measurement found it: informational, never a verdict), and one entry per
    measured alternative, held in one order (by kind, then alternative). Its JSON form is its wire
    form; `from_json` refuses any other spelling."""

    generation: int
    shape: dict
    host: dict
    entries: tuple[ChoiceEntry, ...]

    def __post_init__(self):
        if type(self.generation) is not int or self.generation < 1:
            raise ValueError("a choice table's generation is a positive integer")
        shape = _checked_shape(self.shape)
        if not isinstance(self.host, dict) or any(
            type(k) is not str or type(v) not in (str, int) for k, v in self.host.items()
        ):
            raise ValueError("a choice table's host is strings and integers by name")
        if not isinstance(self.entries, (tuple, list)):
            raise ValueError("a choice table's entries are a sequence of ChoiceEntry values")
        entries, seen = tuple(self.entries), set()
        for e in entries:
            if not isinstance(e, ChoiceEntry):
                raise ValueError("a choice table holds ChoiceEntry values")
            if type(e.kind) is not str or type(e.alternative) is not str:
                raise ValueError("a choice entry names its kind and alternative")
            alternatives = CHOICE_KINDS.get(e.kind)
            if alternatives is None or e.alternative not in alternatives:
                raise ValueError(f"no kind {e.kind!r} with the alternative {e.alternative!r}")
            if (e.kind, e.alternative) in seen:
                raise ValueError(f"{e.kind} {e.alternative} measured twice")
            seen.add((e.kind, e.alternative))
            if any(type(v) is not int for v in (e.q8, e.low_q8, e.high_q8, e.rounds)):
                raise ValueError("a choice entry is integers")
            if not 0 < e.low_q8 <= e.q8 <= e.high_q8 or e.rounds < 5:
                raise ValueError(f"{e.kind} {e.alternative}: not an interval of five rounds")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "host", dict(self.host))
        object.__setattr__(self, "entries",
                           tuple(sorted(entries, key=lambda e: (e.kind, e.alternative))))  # fmt: skip

    def to_json(self) -> str:
        return json.dumps(
            {"generation": self.generation, "shape": self.shape, "host": self.host,
             "entries": [[e.kind, e.alternative, e.q8, e.low_q8, e.high_q8, e.rounds]
                         for e in self.entries]},
            sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        )  # fmt: skip

    @staticmethod
    def from_json(text: str) -> ChoiceTable:
        from ..._artifact_json import strict_json_loads

        doc = strict_json_loads(text, "choice table")
        if not isinstance(doc, dict) or set(doc) != {"generation", "shape", "host", "entries"}:
            raise ValueError("a choice table is {generation, shape, host, entries}")
        if not isinstance(doc["entries"], list) or not isinstance(doc["shape"], dict):
            raise ValueError("a choice table's entries are a list and its shape an object")
        entries = []
        for row in doc["entries"]:
            if not isinstance(row, list) or len(row) != 6:
                raise ValueError("a choice entry is [kind, alternative, q8, low, high, rounds]")
            entries.append(ChoiceEntry(*row))
        table = ChoiceTable(doc["generation"], doc["shape"], doc["host"], tuple(entries))
        if table.to_json() != text:
            raise ValueError("a choice table has one spelling: its canonical JSON")
        return table


def choose(table: ChoiceTable | None, spec: DecoderSpec, shape: dict) -> dict[str, str]:
    """Every site's alternative for a program of `spec` at `shape` (`trainer_shape`): for each
    kind, the measured alternative whose whole interval lies below 256 and whose mean is the
    lowest (the narrower interval on a tie, then the table's order), at every site of that kind;
    every other site the planner's default (`default_sites`, the program a table is measured
    against). `table` None is the defaults. A table measured at another shape is refused: its times
    are not this program's."""
    shape = _checked_shape(shape)
    if tuple(shape[k] for k in _SPEC_KEYS) != _spec_shape(spec):
        raise ValueError("the shape names another decoder than spec")
    sites = default_sites(spec)
    if table is None:
        return sites
    if not isinstance(table, ChoiceTable):
        raise ValueError("choose takes a ChoiceTable or None")
    if table.shape != shape:
        raise ValueError(f"the table was measured at {table.shape}, not at {shape}")
    best: dict[str, ChoiceEntry] = {}
    for e in table.entries:
        if e.high_q8 < Q8 and (
            e.kind not in best or (e.q8, e.high_q8) < (best[e.kind].q8, best[e.kind].high_q8)
        ):
            best[e.kind] = e
    for site in sites:
        e = best.get(kind_of(site))
        if e is not None:
            sites[site] = e.alternative
    return sites


def host_attestation() -> dict:
    """What the measuring host says of itself -- informational, never a verdict: its machine,
    system and logical CPUs and, where the system names it, the CPU's model."""
    out: dict = {"machine": platform.machine() or "unknown", "system": platform.system() or "unknown",
                 "cpus": os.cpu_count() or 0}  # fmt: skip
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("model name"):
                    out["cpu"] = line.split(":", 1)[1].strip()[:120]
                    break
    except OSError:
        pass
    return out


def timed_run(trainer, words, tokens, targets, *, steps: int = 1, **run) -> tuple[tuple, list[int]]:
    """`trainer.run_program(words, tokens, targets, steps=steps, **run)` on a library built with
    the job clock (`NativeDecoder.build(..., probe="time")`): the run's events, and the nanoseconds
    each job took at its first record's index (every other index 0). A library without the clock
    is refused; so is a run whose clock dropped a job (none can: it is lent a word a record)."""
    lib = trainer.lib
    begin = getattr(lib, "bcir_decoder_time_probe_begin", None)
    end = getattr(lib, "bcir_decoder_time_probe_end", None)
    if begin is None or end is None:
        raise ValueError('this native decoder library has no job clock: build it with probe="time"')
    begin.argtypes, begin.restype = [ct.POINTER(ct.c_uint64), ct.c_size_t], None
    end.argtypes, end.restype = [], ct.c_size_t
    if len(words) % RECORD_WORDS:
        raise ValueError(f"a program is records of {RECORD_WORDS} unsigned 64-bit words")
    records = len(words) // RECORD_WORDS
    ns = (ct.c_uint64 * max(1, records))()
    begin(ns, records)
    try:
        events = trainer.run_program(words, tokens, targets, steps=steps, **run)
    finally:
        dropped = end()
    if dropped:
        raise RuntimeError(f"the job clock dropped {dropped} job(s)")
    return events, list(ns)[:records]


def measure(trainer, *, steps: int = 1, rounds: int = 16, generation: int = 1, kinds=None,
            host: dict | None = None, raw: list | None = None) -> ChoiceTable:  # fmt: skip
    """Measures the choice sites' alternatives on `trainer`, a trainer of a library built with the
    job clock: for each kind (`CHOICE_KINDS`, or `kinds`) and each of its alternatives that states
    a program other than the trainer's default one, the program with every site of that kind
    switched to it; one untimed run of each, then `rounds` rounds, each running the default and
    every alternative once -- calls of `steps` steps on one token stream, the order rotated round
    by round -- on the one trainer: they train the same bits, so they take turns on one state. In
    each run it reads the nanoseconds of the launches the kind's scope labels (`scope`), and each
    alternative's entry is `estimate` of its rounds paired with the default's. `raw`, an empty
    list, receives each round's readings. Returns the table at `generation`, its shape the
    trainer's (`trainer_shape`), its host `host` or `host_attestation()`. A trainer with a
    provider is refused: a provider's calls are not BCIR's to choose."""
    if type(rounds) is not int or rounds < 5:
        raise ValueError("a measurement takes at least five rounds")
    if raw is not None and (not isinstance(raw, list) or raw):
        raise ValueError("raw is an empty list each round's readings are appended to")
    kinds = tuple(CHOICE_KINDS) if kinds is None else tuple(kinds)
    if not kinds or len(set(kinds)) != len(kinds) or any(k not in CHOICE_KINDS for k in kinds):
        raise ValueError(f"kinds names kinds of choice site once each: {sorted(CHOICE_KINDS)}")
    spec, batch, time = trainer.spec, trainer.batch_size, trainer.context_length
    shape = trainer_shape(trainer, steps=steps)
    decisions = trainer.program_decisions(steps=steps)
    if decisions["provider_mm"] or decisions["provider_silu"]:
        raise ValueError("a provider's calls are not BCIR's to choose: measure without one")
    defaults = default_sites(spec)
    arms: list[tuple[str, str | None, list, list]] = []
    labels: list = []
    base = step_program(spec, batch, time, **decisions, labels=labels)
    arms.append(("default", None, base, labels))
    for kind in kinds:
        for alternative in CHOICE_KINDS[kind]:
            switched = {site: alternative for site in defaults if kind_of(site) == kind}
            if all(defaults[site] == alternative for site in switched):
                continue  # the default's own alternative
            labels = []
            program = step_program(spec, batch, time, **decisions, sites=switched, labels=labels)
            if program != base:  # else the contract or shape leaves the site one way here
                arms.append((kind, alternative, program, labels))
    words = [array("Q", program_words(program)) for _, _, program, _ in arms]
    starts = []
    for _, _, program, _ in arms:
        at, firsts = 0, []
        for launch in program:
            firsts.append(at)
            at += len(launch.records)
        starts.append(firsts)
    n = batch * time * steps
    tokens = [(7 * i + 3) % spec.vocab_size for i in range(n)]
    targets = [(5 * i + 1) % spec.vocab_size for i in range(n)]
    measured = sorted({kind for kind, _, _, _ in arms[1:]})

    def site_ns(arm: int, ns: list[int], kind: str) -> int:
        names = scope(kind)
        return sum(ns[at] for at, label in zip(starts[arm], arms[arm][3])
                   if label is not None and kind_of(label) in names)  # fmt: skip

    for arm in range(len(arms)):  # the first touch of every page each program reaches
        timed_run(trainer, words[arm], tokens, targets, steps=steps)
    base_ns: dict[str, list[int]] = {kind: [] for kind in measured}
    arm_ns: list[list[int]] = [[] for _ in arms]
    names = ["default" if alt is None else f"{kind}={alt}" for kind, alt, _, _ in arms]
    for r in range(rounds):
        shift = r % len(arms)
        order = list(range(shift, len(arms))) + list(range(shift))
        totals, sites_read = {}, {}
        for arm in order:
            _, ns = timed_run(trainer, words[arm], tokens, targets, steps=steps)
            totals[names[arm]] = sum(ns)
            if arm == 0:
                for kind in measured:
                    base_ns[kind].append(site_ns(0, ns, kind))
                    sites_read[f"default@{kind}"] = base_ns[kind][-1]
            else:
                arm_ns[arm].append(site_ns(arm, ns, arms[arm][0]))
                sites_read[names[arm]] = arm_ns[arm][-1]
        if raw is not None:
            raw.append({"round": r, "order": [names[a] for a in order], "total_ns": totals,
                        "site_ns": sites_read})  # fmt: skip
    entries = []
    for arm in range(1, len(arms)):
        kind, alternative = arms[arm][0], arms[arm][1]
        q8, low, high = estimate(base_ns[kind], arm_ns[arm])
        entries.append(ChoiceEntry(kind, alternative, q8, low, high, rounds))
    return ChoiceTable(generation, shape, host_attestation() if host is None else host,
                       tuple(entries))  # fmt: skip
