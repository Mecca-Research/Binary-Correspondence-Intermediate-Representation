"""Before/after performance audit of one change: the same host, both trees, one table.

Every slice of the GEM+ completion ladder ends with this audit (roadmap §0.4). It answers the
three questions the per-slice analysis protocol (§0.2) asks, on evidence a reviewer can rerun:

  * **did anything the frozen harness grades get worse?** `tools/perf/gemplus_baseline.py
    --compare` runs in each tree, and the two tables are diffed row by row under the harness's
    own kinds: an `exact` row is graded with zero tolerance (it is deterministic, so any movement
    is the change), a `ratio` row inside the 25% band is NO-CHANGE, and a `wall` row is never
    graded -- it is reported as the before/after ratio, INDICATIVE, exactly as the harness
    reports it.
  * **did the outputs stay the same?** `bcir.performance_audit` runs in each tree and its
    timing-free `correctness_sha256` and per-case `result_sha256` must match, and every hot-path
    row below compares a digest of its output between the trees. An output that moved is a
    finding unless the slice declared it (`--expect-change`).
  * **what did it cost or buy on the hot paths?** Each row builds its fixture outside the timed
    interval and times the call in a child process that imports `bcir` from that tree. The
    trees alternate round by round (before first, then after first), so drift on a busy host
    lands on both sides. The call count (cProfile, builtins included, one entry per code
    object, both trees counted by this checkout's counter) is exact for one interpreter, so it is
    graded like an `exact` row. The time is INDICATIVE.

**A regression is not landable until it is explained** (§0.2). `--explain ROW=reason` records
the explanation in the report and lets the audit pass. Without it, a REGRESSION, an unexpected
output change or a metric that disappeared fails the audit. Every exit is a verdict (laws.md L1):

    0  PASS         nothing regressed, nothing changed undeclared, nothing went missing
    1  FAIL         a regression, an undeclared output change, or a removed metric
    2  UNAVAILABLE  a tree could not be measured (bad ref, a child that died); the report says which

The JSON and Markdown reports are written on every path, including the failing ones.

Usage::

    python tools/perf/ab_audit.py --before origin/main                    # after = this checkout
    python tools/perf/ab_audit.py --before <ref|dir> --after <dir> --group sp,sched \\
        --rounds 3 --samples 5 --json out.json --markdown out.md
    python tools/perf/ab_audit.py --before HEAD~1 --no-harness --no-audit --rows plan@4

The child mode (`--child`) is the measuring half and is internal.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import enum
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

#: The harness's grading bands (tools/perf/gemplus_baseline.py `_DEFAULT_NOISE`) for the two
#: graded kinds. An A/B on one interpreter makes an `exact` row bit-identical unless the change
#: moved it, so this audit grades `exact` with zero tolerance; the harness's 2% band is for
#: comparing against a baseline frozen on another host.
RATIO_BAND = 0.25

PASS, FAIL, UNAVAILABLE = 0, 1, 2


# --- the canonical digest of a row's output -----------------------------------------------------


def _canon(value):
    """A JSON-able projection of a row's output, stable across runs and trees.

    Dataclasses, enums, containers and slotted objects are walked structurally, so two trees
    that compute the same value agree byte for byte. The last fallback is `repr` with heap
    addresses removed; a row should return canonical data rather than lean on it."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"__bytes__": hashlib.sha256(value).hexdigest(), "len": len(value)}
    if isinstance(value, enum.Enum):
        return {"__enum__": type(value).__name__, "name": value.name}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _canon(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _canon(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canon(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canon(v) for v in value), key=json.dumps)
    slots = [
        s
        for klass in type(value).__mro__
        for s in getattr(klass, "__slots__", ())
        if s != "__weakref__"
    ]
    if slots:
        return {
            "__class__": type(value).__name__,
            **{k: _canon(getattr(value, k)) for k in sorted(slots) if hasattr(value, k)},
        }
    if hasattr(value, "__dict__"):
        return {
            "__class__": type(value).__name__,
            **{k: _canon(v) for k, v in sorted(vars(value).items()) if not k.startswith("_")},
        }
    import re

    return {"__repr__": re.sub(r" at 0x[0-9a-f]+", "", repr(value))}


def digest(value) -> str:
    """The first 16 hex digits of the SHA-256 of a row's canonical output."""
    text = json.dumps(_canon(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# --- the hot-path rows (each returns the zero-argument call to time) -----------------------------
#
# A row builds its fixture outside the timed interval and returns the call. It imports from the
# tree under measurement only, so a row naming a module the before-tree lacks is reported as NEW
# rather than measured against nothing.


def _sp_chain(scale):
    from bcir.abi.streampack_abi import encode
    from bcir.gem.streampack import hydrate_pipelined
    from bcir.kbcir.realize import optimize
    from bcir.performance_audit import kbcir_streampack_fixture

    module, target, theta = kbcir_streampack_fixture(scale)
    result = optimize(module, target, theta)
    pack = hydrate_pipelined(module, result, plan="tmsao", depth=2)
    return module, target, theta, result, pack, encode(pack)


def row_plan(scale):
    from bcir.kbcir.realize import optimize
    from bcir.performance_audit import kbcir_streampack_fixture

    module, target, theta = kbcir_streampack_fixture(scale)
    return lambda: optimize(module, target, theta)


def row_hydrate(scale):
    from bcir.gem.streampack import hydrate_pipelined

    module, _target, _theta, result, _pack, _wire = _sp_chain(scale)
    return lambda: hydrate_pipelined(module, result, plan="tmsao", depth=2)


def row_encode(scale):
    from bcir.abi.streampack_abi import encode

    pack = _sp_chain(scale)[4]
    return lambda: encode(pack)


def row_decode(scale):
    from bcir.abi.streampack_abi import decode

    wire = _sp_chain(scale)[5]
    return lambda: decode(wire)


def row_verify_module(scale):
    from bcir.verify import verify

    module = _sp_chain(scale)[0]
    return lambda: verify(module)


def row_verify_plan(scale):
    from bcir.kbcir.weights import PERF
    from bcir.verify import verify_plan

    module, target, theta, result, _pack, _wire = _sp_chain(scale)
    return lambda: verify_plan(module, result, target, theta=theta, policy=PERF)


def row_verify_pack(scale):
    from bcir.verify import verify_pack

    module, _target, _theta, _result, pack, _wire = _sp_chain(scale)
    return lambda: verify_pack(module, pack)


def row_chain(scale):
    from bcir.gem.streampack import hydrate_pipelined
    from bcir.kbcir.realize import optimize
    from bcir.kbcir.weights import PERF
    from bcir.performance_audit import kbcir_streampack_fixture
    from bcir.verify import verify, verify_pack, verify_plan

    module, target, theta = kbcir_streampack_fixture(scale)

    def run():
        result = optimize(module, target, theta)
        pack = hydrate_pipelined(module, result, plan="tmsao", depth=2)
        verdicts = (
            verify(module)
            + verify_plan(module, result, target, theta=theta, policy=PERF)
            + verify_pack(module, pack)
        )
        return verdicts, result.score, len(pack.segments)

    return run


def row_sched_waves(scale):
    from bcir.gem.concurrency import schedule_concurrent
    from bcir.performance_audit import scheduler_fixture

    module, target, _durations = scheduler_fixture(scale)
    return lambda: schedule_concurrent(module, target)


def row_sched_tokens(scale):
    from bcir.gem.async_tokens import async_plan
    from bcir.performance_audit import scheduler_fixture

    module = scheduler_fixture(scale)[0]
    return lambda: async_plan(module)


def row_sched_eft(scale):
    from bcir.gem.schedule import schedule_eft
    from bcir.performance_audit import scheduler_fixture

    module, target, durations = scheduler_fixture(scale)
    return lambda: schedule_eft(module, durations, target)


def row_dag_exec(scale):
    from bcir.gem.execute import execute
    from bcir.performance_audit import phase_dag_module

    module = phase_dag_module(scale)
    return lambda: execute(module)


def row_dag_verify(scale):
    from bcir.performance_audit import phase_dag_module
    from bcir.verify import verify

    module = phase_dag_module(scale)
    return lambda: verify(module)


def row_audit_case(name, scale):
    from bcir.performance_audit import _case_definitions

    for _group, case, _items, function in _case_definitions(scale):
        if case == name:
            return lambda: function(scale)
    raise KeyError(f"the performance audit has no case {name!r}")


def row_selftest(_scale):
    """A row that needs no fixture: what the unit test drives the child through."""
    return lambda: sum(i * i for i in range(2000))


AUDIT_CASES = (
    "iterative-phase-dag",
    "mixed-wave-token-eft",
    "kbcir-streampack",
    "static-lifetime-planner",
    "bounded-overwrite-ring",
    "q8-q4-blocks",
    "kmeans-knn-scaler-embedding",
    "tiled-matmul",
    "ols-pca",
    "transformer-block",
    "lstm-gru",
    "autodiff-adam",
    "bounded-mcts",
)

ROWS = {
    "plan@4": lambda: row_plan(4),
    "plan@8": lambda: row_plan(8),
    "hydrate@4": lambda: row_hydrate(4),
    "hydrate@8": lambda: row_hydrate(8),
    "encode@4": lambda: row_encode(4),
    "decode@4": lambda: row_decode(4),
    "decode@8": lambda: row_decode(8),
    "verify_module@4": lambda: row_verify_module(4),
    "verify_plan@4": lambda: row_verify_plan(4),
    "verify_pack@4": lambda: row_verify_pack(4),
    "chain@4": lambda: row_chain(4),
    "sched_waves@4": lambda: row_sched_waves(4),
    "sched_tokens@4": lambda: row_sched_tokens(4),
    "sched_eft@4": lambda: row_sched_eft(4),
    "dag_exec@4": lambda: row_dag_exec(4),
    "dag_verify@4": lambda: row_dag_verify(4),
    "selftest": lambda: row_selftest(0),
}
for _case in AUDIT_CASES:
    ROWS[f"audit:{_case}@4"] = (lambda c: lambda: row_audit_case(c, 4))(_case)

GROUPS = {
    "sp": [
        "plan@4",
        "hydrate@4",
        "encode@4",
        "decode@4",
        "verify_module@4",
        "verify_plan@4",
        "verify_pack@4",
        "chain@4",
    ],
    "sp8": ["plan@8", "hydrate@8", "decode@8"],
    "sched": ["sched_waves@4", "sched_tokens@4", "sched_eft@4", "dag_exec@4", "dag_verify@4"],
    "audit": [f"audit:{c}@4" for c in AUDIT_CASES],
}


# --- the child: measure rows in one tree ---------------------------------------------------------


def _call_counter():
    """The one call counter (`bcir/tests/call_counts.py`), loaded by path from THIS checkout under
    a private name: importing it as `bcir.tests.call_counts` would bind `bcir` to this checkout
    before the child imports the measured tree's, and counting each tree with its own counter
    would compare two instruments rather than two trees."""
    module = sys.modules.get("_ab_audit_call_counts")
    if module is None:
        import importlib.util

        path = os.path.join(_ROOT, "bcir", "tests", "call_counts.py")
        spec = importlib.util.spec_from_file_location("_ab_audit_call_counts", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules["_ab_audit_call_counts"] = module
    return module


def measure_row(name: str, samples: int) -> dict:
    """Time one row: one warm run, then `samples` timed runs from a collected heap, one profiled
    run for the call count. A row whose output changes between runs is refused, because its
    digest would then compare nothing. The count is the one counter's (`_call_counter`): one
    entry per code object, never `pstats`' total, which merges two functions that share a label
    -- every dataclass `__init__` -- into one row and keeps one of their counts, which one
    depending on where the code objects were allocated."""
    import gc

    run = ROWS[name]()
    out = digest(run())
    times = []
    for _ in range(samples):
        gc.collect()
        t0 = time.perf_counter_ns()
        result = run()
        times.append(time.perf_counter_ns() - t0)
        if digest(result) != out:
            raise RuntimeError(f"{name}: the output changed between runs of one tree")
    calls, _value = _call_counter().profiled(run)
    return {
        "row": name,
        "status": "measured",
        "times_ns": times,
        "calls": calls,
        "digest": out,
    }


def child_main(tree: str, rows: list[str], samples: int, out_path: str) -> int:
    """Measure `rows` with `bcir` imported from `tree`. A row the tree cannot build (a module
    it does not have) is recorded as unavailable, never as a failure of the whole child."""
    sys.path.insert(0, tree)
    os.chdir(tree)
    records = []
    for name in rows:
        if name not in ROWS:
            records.append({"row": name, "status": "unknown-row"})
            continue
        try:
            records.append(measure_row(name, samples))
        except (ImportError, AttributeError, KeyError) as exc:
            records.append(
                {"row": name, "status": "unavailable", "why": f"{type(exc).__name__}: {exc}"}
            )
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"tree": tree, "rows": records}, fh)
    return 0


# --- grading -------------------------------------------------------------------------------------


@dataclasses.dataclass
class Finding:
    """One graded line of the report."""

    section: str  # harness | audit | rows
    key: str
    kind: str  # exact | ratio | wall | digest | calls | time
    before: object
    after: object
    verdict: (
        str  # SAME, GAIN, NO-CHANGE, REGRESSION, INDICATIVE, NEW, REMOVED, CHANGED, UNAVAILABLE
    )
    note: str = ""

    def failing(self, explained: dict[str, str]) -> bool:
        if self.verdict in ("REGRESSION", "REMOVED", "CHANGED"):
            return self.key not in explained
        return False


def _improved(before: dict, after: dict):
    """Which way a harness row moved, read from the harness's own `improvement` (positive is
    better against the frozen baseline): +1 better, -1 worse, 0 equal, None unknowable."""
    b, a = before.get("improvement"), after.get("improvement")
    if b is None or a is None:
        return None
    if a > b:
        return 1
    if a < b:
        return -1
    return 0


def grade_harness(before_rows: list[dict], after_rows: list[dict]) -> list[Finding]:
    """Diff two `gemplus_baseline.py --compare --json` row lists of one host."""
    before = {r["key"]: r for r in before_rows}
    after = {r["key"]: r for r in after_rows}
    out: list[Finding] = []
    for key in sorted(set(before) | set(after)):
        b, a = before.get(key), after.get(key)
        if b is None:
            out.append(
                Finding(
                    "harness",
                    key,
                    a.get("kind", "?"),
                    None,
                    a.get("measured"),
                    "NEW",
                    f"harness verdict {a.get('verdict')}",
                )
            )
            continue
        if a is None:
            out.append(
                Finding(
                    "harness",
                    key,
                    b.get("kind", "?"),
                    b.get("measured"),
                    None,
                    "REMOVED",
                    "a metric the before-tree grades is gone",
                )
            )
            continue
        kind, mb, ma = a.get("kind", b.get("kind")), b.get("measured"), a.get("measured")
        if mb is None or ma is None:
            verdict = "SAME" if mb == ma else "UNAVAILABLE"
            out.append(Finding("harness", key, kind, mb, ma, verdict, "not measured in one tree"))
            continue
        if kind == "wall":
            ratio = (mb / ma) if ma else None
            note = (
                f"{ratio:.2f}x faster"
                if ratio and ratio >= 1
                else (f"{1 / ratio:.2f}x slower" if ratio else "")
            )
            out.append(Finding("harness", key, kind, mb, ma, "INDICATIVE", note))
            continue
        direction = _improved(b, a)
        if mb == ma or direction == 0:
            out.append(Finding("harness", key, kind, mb, ma, "SAME"))
            continue
        if direction is None:
            out.append(
                Finding("harness", key, kind, mb, ma, "CHANGED", "moved with no direction to grade")
            )
            continue
        if kind == "ratio" and mb and abs(ma - mb) / abs(mb) <= RATIO_BAND:
            out.append(
                Finding("harness", key, kind, mb, ma, "NO-CHANGE", "inside the 25% ratio band")
            )
            continue
        out.append(Finding("harness", key, kind, mb, ma, "GAIN" if direction > 0 else "REGRESSION"))
    return out


def grade_audit(before: dict, after: dict) -> list[Finding]:
    """Diff two `bcir.performance_audit` reports: digests exact, times indicative."""
    out: list[Finding] = []
    cb, ca = before.get("correctness_sha256"), after.get("correctness_sha256")
    out.append(
        Finding("audit", "correctness_sha256", "digest", cb, ca, "SAME" if cb == ca else "CHANGED")
    )
    sb = {s["name"]: s for s in before.get("samples", [])}
    sa = {s["name"]: s for s in after.get("samples", [])}
    for name in sorted(set(sb) | set(sa)):
        b, a = sb.get(name), sa.get(name)
        if b is None or a is None:
            out.append(
                Finding(
                    "audit",
                    name,
                    "digest",
                    b and b.get("result_sha256"),
                    a and a.get("result_sha256"),
                    "NEW" if b is None else "REMOVED",
                )
            )
            continue
        same = b.get("result_sha256") == a.get("result_sha256")
        mb, ma = b.get("median_ns"), a.get("median_ns")
        ratio = (mb / ma) if (mb and ma) else None
        note = f"time {ratio:.2f}x (before/after, indicative)" if ratio else ""
        out.append(
            Finding(
                "audit",
                name,
                "digest",
                b.get("result_sha256", "")[:16],
                a.get("result_sha256", "")[:16],
                "SAME" if same else "CHANGED",
                note,
            )
        )
    return out


def grade_rows(before: list[dict], after: list[dict]) -> list[Finding]:
    """Diff the hot-path rows: digest and calls exact, time indicative."""
    bmap = {r["row"]: r for r in before}
    amap = {r["row"]: r for r in after}
    out: list[Finding] = []
    for name in list(dict.fromkeys([*bmap, *amap])):
        b, a = bmap.get(name), amap.get(name)
        b_ok = b is not None and b.get("status") == "measured"
        a_ok = a is not None and a.get("status") == "measured"
        unstable = [
            t
            for t, r in (("before", b), ("after", a))
            if r and r.get("status") == "nondeterministic"
        ]
        if unstable:
            why = "; ".join(
                f"{t} tree: {r.get('why', '')}"
                for t, r in (("before", b), ("after", a))
                if r and r.get("status") == "nondeterministic"
            )
            out.append(
                Finding(
                    "rows",
                    name,
                    "digest",
                    None,
                    None,
                    "CHANGED",
                    f"nondeterministic ({why}): its digest compares nothing",
                )
            )
            continue
        if not a_ok and not b_ok:
            out.append(
                Finding(
                    "rows",
                    name,
                    "time",
                    None,
                    None,
                    "UNAVAILABLE",
                    (a or b or {}).get("why", (a or b or {}).get("status", "")),
                )
            )
            continue
        if not b_ok:
            out.append(
                Finding(
                    "rows",
                    name,
                    "time",
                    None,
                    _ms(a),
                    "NEW",
                    "the before-tree cannot build this row",
                )
            )
            continue
        if not a_ok:
            out.append(
                Finding("rows", name, "time", _ms(b), None, "REMOVED", (a or {}).get("why", ""))
            )
            continue
        out.append(
            Finding(
                "rows",
                name,
                "digest",
                b["digest"],
                a["digest"],
                "SAME" if b["digest"] == a["digest"] else "CHANGED",
            )
        )
        cb, ca = b["calls"], a["calls"]
        out.append(
            Finding(
                "rows",
                name,
                "calls",
                cb,
                ca,
                "SAME" if cb == ca else ("GAIN" if ca < cb else "REGRESSION"),
            )
        )
        mb, ma = _ms(b), _ms(a)
        note = (
            f"{mb / ma:.2f}x faster"
            if ma and mb >= ma
            else (f"{ma / mb:.2f}x slower" if mb else "")
        )
        out.append(Finding("rows", name, "time", round(mb, 3), round(ma, 3), "INDICATIVE", note))
    return out


def _ms(record: dict | None):
    if not record or not record.get("times_ns"):
        return None
    return statistics.median(record["times_ns"]) / 1e6


def merge_rounds(rounds: list[list[dict]]) -> list[dict]:
    """Pool one tree's samples across interleaved rounds. A call count or digest that differs
    between rounds is a nondeterministic row, recorded as such rather than averaged."""
    merged: dict[str, dict] = {}
    for records in rounds:
        for r in records:
            name = r["row"]
            if name not in merged:
                merged[name] = dict(r, times_ns=list(r.get("times_ns", [])))
                continue
            m = merged[name]
            if r.get("status") != "measured" or m.get("status") != "measured":
                continue
            if r["calls"] != m["calls"] or r["digest"] != m["digest"]:
                m["status"] = "nondeterministic"
                m["why"] = (
                    f"call count {m['calls']} vs {r['calls']}"
                    if r["calls"] != m["calls"]
                    else f"digest {m['digest']} vs {r['digest']}"
                ) + " between rounds"
                continue
            m["times_ns"].extend(r["times_ns"])
    return list(merged.values())


def verdict_of(findings: list[Finding], explained: dict[str, str], unavailable: list[str]) -> int:
    if unavailable:
        return UNAVAILABLE
    return FAIL if any(f.failing(explained) for f in findings) else PASS


# --- the parent: two trees, one report -----------------------------------------------------------


@contextlib.contextmanager
def materialized(ref_or_dir: str, label: str, keep: list[str]):
    """A tree to measure: an existing directory as-is, else a detached worktree of the git ref,
    removed again on exit."""
    if os.path.isdir(ref_or_dir):
        yield os.path.abspath(ref_or_dir)
        return
    tmp = tempfile.mkdtemp(prefix=f"bcir-ab-{label}-")
    os.rmdir(tmp)
    proc = subprocess.run(
        ["git", "-C", _ROOT, "worktree", "add", "--detach", tmp, ref_or_dir],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise LookupError(f"{label}: cannot check out {ref_or_dir!r}: {proc.stderr.strip()[-300:]}")
    keep.append(tmp)
    try:
        yield tmp
    finally:
        subprocess.run(
            ["git", "-C", _ROOT, "worktree", "remove", "--force", tmp], capture_output=True
        )
        shutil.rmtree(tmp, ignore_errors=True)


def _run(cmd: list[str], cwd: str, timeout: int) -> subprocess.CompletedProcess:
    # One hash seed for every measuring process, so set and str-keyed dict order -- and with
    # them a call count -- cannot differ between two processes that run the same code.
    env = dict(os.environ, PYTHONHASHSEED="0")
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)


def run_harness(tree: str, scratch: str, label: str, timeout: int, group: str = "") -> list[dict]:
    path = os.path.join(scratch, f"harness-{label}.json")
    cmd = [sys.executable, os.path.join("tools", "perf", "gemplus_baseline.py"), "--compare"]
    if group:
        cmd += ["--group", group]
    proc = _run(cmd + ["--json", path], tree, timeout)
    if not os.path.exists(path):
        raise RuntimeError(
            f"{label}: the harness wrote no table (exit {proc.returncode}): {proc.stderr[-400:]}"
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["rows"]


def confirm_ratios(graded, before_rows, after_rows, before, after, scratch, args):
    """A `ratio` row that left its band earns confirming measurements before it is graded.

    A timed ratio cancels the machine's speed, not its load: one sample per tree on a busy host
    can leave the 25% band with no change in the code (bcir-cicd: a ratio row earns a confirming
    re-run, and a second failure is real). Each regressed ratio row's harness group is measured
    again `--ratio-confirm` times per tree, alternating which tree goes first, and the row is
    graded on the median of every measurement it has."""
    suspects = [f for f in graded if f.kind == "ratio" and f.verdict == "REGRESSION"]
    if not suspects or args.ratio_confirm < 1:
        return graded
    group_of = {r["key"]: r.get("group", "") for r in after_rows}
    samples = {f.key: ([f.before], [f.after]) for f in suspects}
    groups = sorted({group_of.get(f.key, "") for f in suspects} - {""})
    for i in range(args.ratio_confirm):
        for group in groups:
            order = (("before", before, 0), ("after", after, 1))
            for label, tree, side in order if i % 2 else order[::-1]:
                rows = run_harness(
                    tree, scratch, f"confirm-{label}-{group}-{i}", args.timeout, group
                )
                for row in rows:
                    if row["key"] in samples and row.get("measured") is not None:
                        samples[row["key"]][side].append(row["measured"])
    out = []
    for f in graded:
        if f not in suspects:
            out.append(f)
            continue
        b_med = statistics.median(samples[f.key][0])
        a_med = statistics.median(samples[f.key][1])
        # The first grading called before -> after worse, which fixes the row's direction.
        lower_is_better = f.after > f.before
        worse = a_med > b_med if lower_is_better else a_med < b_med
        outside = bool(b_med) and abs(a_med - b_med) / abs(b_med) > RATIO_BAND
        verdict = "REGRESSION" if outside and worse else ("GAIN" if outside else "NO-CHANGE")
        n = len(samples[f.key][0])
        note = f"median of {n} measurements per tree, confirmed after the first left the band"
        out.append(Finding("harness", f.key, f.kind, b_med, a_med, verdict, note))
    return out


def run_audit(tree: str, scratch: str, label: str, repeats: int, timeout: int) -> dict:
    path = os.path.join(scratch, f"audit-{label}.json")
    proc = _run(
        [
            sys.executable,
            "-m",
            "bcir.performance_audit",
            "--repeats",
            str(repeats),
            "--output",
            path,
        ],
        tree,
        timeout,
    )
    if proc.returncode != 0 or not os.path.exists(path):
        raise RuntimeError(
            f"{label}: the performance audit failed (exit {proc.returncode}): {proc.stderr[-400:]}"
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def run_child(
    tree: str, rows: list[str], samples: int, scratch: str, tag: str, timeout: int
) -> list[dict]:
    path = os.path.join(scratch, f"rows-{tag}.json")
    proc = _run(
        [
            sys.executable,
            os.path.abspath(__file__),
            "--child",
            "--tree",
            tree,
            "--rows",
            ",".join(rows),
            "--samples",
            str(samples),
            "--json",
            path,
        ],
        tree,
        timeout,
    )
    if proc.returncode != 0 or not os.path.exists(path):
        raise RuntimeError(
            f"{tag}: the measuring child died (exit {proc.returncode}): {proc.stderr[-400:]}"
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["rows"]


def render_markdown(report: dict) -> str:
    lines = [
        f"### Before/after audit -- {report['verdict']}",
        "",
        f"before `{report['before']}` · after `{report['after']}` · host `{report['host']}` · "
        f"{report['rounds']} interleaved rounds × {report['samples']} samples",
        "",
    ]
    for section, title in (
        ("harness", "Frozen harness (one host, both trees)"),
        ("audit", "Performance audit (correctness digests)"),
        ("rows", "Hot paths (interleaved A/B)"),
    ):
        rows = [f for f in report["findings"] if f["section"] == section]
        if not rows:
            continue
        if section == "harness":
            moved = [f for f in rows if f["verdict"] not in ("SAME",)]
            lines += [f"**{title}:** {len(rows)} rows, {len(rows) - len(moved)} unchanged.", ""]
            rows = moved
            if not rows:
                continue
        else:
            lines += [f"**{title}**", ""]
        lines += ["| key | kind | before | after | verdict | note |", "|---|---|---:|---:|---|---|"]
        for f in rows:
            explained = report["explained"].get(f["key"])
            note = f["note"] + (f" — explained: {explained}" if explained else "")
            lines.append(
                f"| `{f['key']}` | {f['kind']} | {_fmt(f['before'])} | {_fmt(f['after'])} | "
                f"{f['verdict']} | {note} |"
            )
        lines.append("")
    if report["unavailable"]:
        lines += ["**Unavailable:**", *[f"- {u}" for u in report["unavailable"]], ""]
    return "\n".join(lines)


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.4g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def parent_main(args) -> int:
    explained = {}
    for item in args.explain:
        key, sep, why = item.partition("=")
        if not sep or not why.strip():
            print(f"ab_audit: --explain wants ROW=reason, got {item!r}", file=sys.stderr)
            return UNAVAILABLE
        explained[key] = why.strip()
    for key in args.expect_change:
        explained.setdefault(key, "declared output change")
    rows = [r for r in args.rows.split(",") if r]
    for group in [g for g in args.group.split(",") if g]:
        if group not in GROUPS:
            print(
                f"ab_audit: unknown group {group!r}; the groups are {sorted(GROUPS)}",
                file=sys.stderr,
            )
            return UNAVAILABLE
        rows += GROUPS[group]
    rows = list(dict.fromkeys(rows))
    unknown = [r for r in rows if r not in ROWS]
    if unknown:
        print(f"ab_audit: unknown rows {unknown}", file=sys.stderr)
        return UNAVAILABLE
    findings: list[Finding] = []
    unavailable: list[str] = []
    scratch = tempfile.mkdtemp(prefix="bcir-ab-")
    keep: list[str] = []
    try:
        with (
            materialized(args.before, "before", keep) as before,
            materialized(args.after, "after", keep) as after,
        ):
            if not args.no_harness:
                try:
                    before_rows = run_harness(before, scratch, "before", args.timeout)
                    after_rows = run_harness(after, scratch, "after", args.timeout)
                    graded = grade_harness(before_rows, after_rows)
                    findings += confirm_ratios(
                        graded, before_rows, after_rows, before, after, scratch, args
                    )
                except (RuntimeError, subprocess.TimeoutExpired, OSError, ValueError) as exc:
                    unavailable.append(f"harness: {exc}")
            if not args.no_audit:
                try:
                    findings += grade_audit(
                        run_audit(before, scratch, "before", args.audit_repeats, args.timeout),
                        run_audit(after, scratch, "after", args.audit_repeats, args.timeout),
                    )
                except (RuntimeError, subprocess.TimeoutExpired, OSError, ValueError) as exc:
                    unavailable.append(f"audit: {exc}")
            if rows:
                b_rounds, a_rounds = [], []
                try:
                    for i in range(args.rounds):
                        order = (("before", before, b_rounds), ("after", after, a_rounds))
                        for label, tree, sink in order if i % 2 == 0 else order[::-1]:
                            sink.append(
                                run_child(
                                    tree, rows, args.samples, scratch, f"{label}-{i}", args.timeout
                                )
                            )
                    graded = grade_rows(merge_rounds(b_rounds), merge_rounds(a_rounds))
                    findings += graded
                    unavailable += [
                        f"row {f.key}: {f.note}" for f in graded if f.verdict == "UNAVAILABLE"
                    ]
                except (RuntimeError, subprocess.TimeoutExpired, OSError, ValueError) as exc:
                    unavailable.append(f"rows: {exc}")
    except LookupError as exc:
        unavailable.append(str(exc))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    code = verdict_of(findings, explained, unavailable)
    report = {
        "schema": "bcir.ab_audit.v1",
        "verdict": {PASS: "PASS", FAIL: "FAIL", UNAVAILABLE: "UNAVAILABLE"}[code],
        "before": args.before,
        "after": args.after,
        "host": f"{sys.platform} python {sys.version.split()[0]}",
        "rounds": args.rounds,
        "samples": args.samples,
        "explained": explained,
        "unavailable": unavailable,
        "findings": [dataclasses.asdict(f) for f in findings],
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
    text = render_markdown(report)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--before", help="git ref or directory of the tree before the change")
    ap.add_argument(
        "--after",
        default=_ROOT,
        help="git ref or directory after the change (default: this checkout)",
    )
    ap.add_argument("--rows", default="", help="comma-separated hot-path rows")
    ap.add_argument(
        "--group", default="", help=f"comma-separated row groups: {', '.join(sorted(GROUPS))}"
    )
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--audit-repeats", type=int, default=3)
    ap.add_argument(
        "--ratio-confirm",
        type=int,
        default=2,
        help="confirming measurements per tree for a ratio row that leaves its band",
    )
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per measuring process")
    ap.add_argument("--no-harness", action="store_true")
    ap.add_argument("--no-audit", action="store_true")
    ap.add_argument(
        "--expect-change",
        action="append",
        default=[],
        metavar="KEY",
        help="an output the change is declared to move (repeatable)",
    )
    ap.add_argument(
        "--explain",
        action="append",
        default=[],
        metavar="KEY=REASON",
        help="accept a regression with its recorded explanation (repeatable)",
    )
    ap.add_argument("--json", default="")
    ap.add_argument("--markdown", default="")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--tree", default="", help=argparse.SUPPRESS)
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return UNAVAILABLE if exc.code else PASS
    if args.rounds < 1 or args.samples < 1 or args.audit_repeats < 1 or args.timeout < 1:
        print(
            "ab_audit: --rounds, --samples, --audit-repeats and --timeout must be positive",
            file=sys.stderr,
        )
        return UNAVAILABLE
    if args.ratio_confirm < 0:
        print("ab_audit: --ratio-confirm must not be negative", file=sys.stderr)
        return UNAVAILABLE
    if args.child:
        if not args.tree or not args.json:
            print("ab_audit: --child needs --tree and --json", file=sys.stderr)
            return UNAVAILABLE
        return child_main(
            args.tree, [r for r in args.rows.split(",") if r], args.samples, args.json
        )
    if not args.before:
        print("ab_audit: --before is required", file=sys.stderr)
        return UNAVAILABLE
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
