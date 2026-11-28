"""PORT-5: the hosted C++ solver library's first solver, `gem.exact`, at parity with the oracle.

N5 of the native port audit (`docs/research/BCIR_NATIVE_PORT_AUDIT_2026-11-09.md`): the GEM+ solver portfolio
leaves Python as one C++23 library behind a C ABI façade. `runtime/cpp/bcir_exact.cpp` is its foundation -- the
façade (`bcir_exact.h`: records in, a placement and its certificate out, a status for every refusal, nothing
thrown), the memory (one arena per phase over the caller's `bcir_host_allocator`) and the error convention
(`bcir::expected` inside, `bcir_exact_status` across) -- and its first solver: `exact_schedule`'s branch-and-bound,
expansion for expansion.

What holds it here:

- **Parity.** Each problem of a generated corpus -- the six-job fixture under every duration vector, random modules
  optimized for two targets and three domain counts, the quality corpus -- goes to the oracle and to `test_exact
  FILE` under five budgets, from 0 to past closing, and the two print one text: every phase's heuristic, incumbent,
  bound, stop, expansions, root bounds and placement, and the module's totals.
- **The plan is checked, not trusted.** Every placement the C++ returns is checked against its phase here, by code
  that shares nothing with the solver: each claim once, on a stream it may run on, for its duration, after its
  predecessors, no two at once on a stream, its span the incumbent; and the certificate's order, `L <= U <= H`, with
  `L == U` exactly when it says optimal.
- **The façade's contracts** run in `test_exact` itself: every refusal, the budget's stops, determinism, and every
  allocation of a run failed in turn.

Skips cleanly where no C++23 compiler is visible (the quick tier hides the host's) or outside a source checkout.
"""

from __future__ import annotations

import os
import random
import subprocess
import tempfile
from dataclasses import replace

from bcir.gem.concurrency import _topo_phase_ids
from bcir.gem.exact import _PhaseSearch, exact_schedule
from bcir.gem.schedule import _PhaseDispatch, _streams, durations_from, phase_hazards
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import Theta
from bcir.kbcir.weights import ENERGY, PERF

from .exact_fixtures import quality_corpus, six_job_corpus, six_job_module, six_job_target
from .handoff_fixtures import cxx_compiler
from .test_exact_schedule import random_module

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_C = os.path.join(_ROOT, "runtime", "c")
_CPP = os.path.join(_ROOT, "runtime", "cpp")
# the harness and the library it links: seam_tests/test_exact's closure in runtime/manifest.json
_SOURCES = ("test_exact.cpp", "bcir_exact.cpp")
_BUDGETS = (200_000, 0, 1, 3, 17)
_CXX = cxx_compiler()  # None in the quick tier, which hides the host's compilers


def _build(tmp: str) -> str | None:
    """`test_exact` built as the gate builds it, or None without a C++ compiler or a source checkout."""
    from .run_all import _is_source_checkout

    if _CXX is None or not _is_source_checkout():
        return None
    exe = os.path.join(tmp, "test_exact" + (".exe" if os.name == "nt" else ""))
    argv = [
        _CXX,
        "-std=c++23",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Werror",
        "-I",
        _C,
        "-I",
        _CPP,
    ]
    argv += [os.path.join(_CPP, s) for s in _SOURCES] + ["-o", exe]
    try:
        built = subprocess.run(argv, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AssertionError(f"test_exact did not build: {exc}") from exc
    assert built.returncode == 0, f"test_exact did not build:\n{built.stderr[-3000:]}"
    return exe


def _searches(module, durations, target):
    """Each phase's `_PhaseSearch` and its heuristic placement, in the order `exact_schedule` takes them."""
    domains, knee = _streams(target)
    hazards = phase_hazards(module)
    pmap = module.phase_map()
    for pid in _topo_phase_ids(module):
        claims = sorted(pmap[pid].claims, key=lambda c: c.id)
        search = _PhaseSearch(
            _PhaseDispatch(claims, hazards[pid], domains, knee, True), durations, knee
        )
        yield pid, domains, knee, search


def tool_input(module, durations, target, budget) -> str:
    lines = [f"budget {budget}"]
    for pid, domains, knee, search in _searches(module, durations, target):
        h, hslots = search.heuristic()
        lines.append(f"phase {pid} {domains} {knee} {h}")
        for cid in search.ids:
            preds = ",".join(map(str, search.preds[cid])) or "-"
            streams = ",".join(map(str, search.eligible[cid])) or "-"
            lines.append(f"claim {cid} {search.dur[cid]} {preds} {streams}")
        lines += [f"hslot {s.claim_id} {s.domain} {s.start} {s.finish}" for s in hslots]
    return "\n".join(lines) + "\n"


def expected(module, durations, target, budget) -> str:
    """The oracle's `exact_schedule`, as `test_exact FILE` prints its own."""
    r = exact_schedule(module, durations, target, budget=budget)
    out = []
    for ph in r.phases:
        bounds = ",".join(str(b.value) for b in ph.bounds)
        slots = ",".join(f"{s.claim_id}:{s.domain}:{s.start}:{s.finish}" for s in ph.slots)
        out.append(
            f"phase {ph.phase_id} heuristic {ph.heuristic} incumbent {ph.incumbent} lower {ph.lower_bound} "
            f"{ph.stop_reason} expansions {ph.expansions} bounds {bounds} slots {slots}"
        )
    out.append(
        f"total heuristic {r.heuristic} incumbent {r.incumbent} lower {r.lower_bound} {r.stop_reason} "
        f"expansions {r.expansions}"
    )
    return "\n".join(out) + "\n"


def corpus():
    """(module, durations, target) problems: the six-job fixture, random modules, the quality corpus."""
    cases = []
    six = six_job_module()
    for durs in six_job_corpus()[::4]:
        for dom in (2, 3):
            cases.append((six, {k + 1: v for k, v in enumerate(durs)}, six_job_target(dom)))
    rng = random.Random(5)
    for seed in range(240):
        module = random_module(seed)
        for name in ("x86_avx2", "nvidia_ptx"):
            target = replace(TARGETS[name], affinity_domains=rng.choice([1, 2, 3]))
            durations = durations_from(
                optimize(module, target, Theta.cool(), rng.choice([PERF, ENERGY]))
            )
            if durations:
                cases.append((module, durations, target))
    # targets whose bandwidth knee is below their domains: the band bound in play
    for seed in range(240, 300):
        module = random_module(seed)
        for target in (TARGETS["nvidia_ptx"], replace(TARGETS["x86_avx2"], affinity_domains=8)):
            durations = durations_from(optimize(module, target, Theta.cool(), PERF))
            if durations:
                cases.append((module, durations, target))
    for _name, module in quality_corpus(12):
        target = TARGETS["x86_avx2"]
        durations = durations_from(optimize(module, target, Theta.cool(), PERF))
        if durations:
            cases.append((module, durations, target))
    return cases


def _drive(exe: str, text: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "problems")
        with open(path, "w", encoding="ascii") as f:
            f.write(text)
        try:
            ran = subprocess.run([exe, path], capture_output=True, text=True, timeout=900)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AssertionError(f"test_exact did not run: {exc}") from exc
    assert ran.returncode == 0, ran.stdout[-2000:] + ran.stderr[-2000:]
    return ran.stdout


def _problems(text: str) -> list[str]:
    """`test_exact FILE`'s output, one text per problem (each ends with its `total` line, or is an `ERR`)."""
    out, cur = [], []
    for line in text.splitlines(keepends=True):
        cur.append(line)
        if line.startswith(("total ", "ERR ")):
            out.append("".join(cur))
            cur = []
    assert not cur, f"a problem's output has no total: {cur!r}"
    return out


def _is_schedule(search, domains, incumbent, slots) -> str | None:
    """Why `slots` is not a schedule of `search`'s phase spanning `incumbent`, or None."""
    if sorted(c for c, *_ in slots) != sorted(search.ids):
        return "not each claim once"
    at = {c: (s, st, f) for c, s, st, f in slots}
    for c, (s, st, f) in at.items():
        stream = domains if s == -1 else s
        if stream not in search.eligible[c]:
            return f"claim {c} on stream {s}, not one it may run on"
        if st < 0 or f - st != search.dur[c]:
            return f"claim {c} placed for {f - st}, not its duration {search.dur[c]}"
        for q in search.preds[c]:
            if at[q][2] > st:
                return f"claim {c} starts before its predecessor {q} ends"
    for c, (s, st, f) in at.items():
        for o, (s2, st2, f2) in at.items():
            if o != c and s2 == s and f > st and f2 > st2 and st < f2 and st2 < f:
                return f"claims {c} and {o} at once on stream {s}"
    if max((f for _c, _s, _st, f in slots), default=0) != incumbent:
        return "the placement's span is not the incumbent"
    return None


def test_the_facades_contracts_hold():
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return
        try:
            ran = subprocess.run([exe], capture_output=True, text=True, timeout=900)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AssertionError(f"test_exact did not run: {exc}") from exc
        assert ran.returncode == 0 and ran.stdout.strip() == "test_exact: OK", (
            ran.stdout + ran.stderr
        )


def test_the_cpp_scheduler_answers_as_the_oracle_does_and_its_plans_check():
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return
        cases = corpus()
        runs = [(case, budget) for case in cases for budget in _BUDGETS]
        got = _problems(_drive(exe, "".join(tool_input(*case, budget) for case, budget in runs)))
        assert len(got) == len(runs)
        differ, stops, searched = [], set(), 0
        for ((module, durations, target), budget), text in zip(runs, got):
            want = expected(module, durations, target, budget)
            if text != want:
                differ.append((budget, want, text))
                continue
            searches = {
                pid: (domains, s) for pid, domains, _knee, s in _searches(module, durations, target)
            }
            for line in text.splitlines()[:-1]:
                f = line.split()
                pid, h, u, lower, stop, expansions = (
                    int(f[1]),
                    int(f[3]),
                    int(f[5]),
                    int(f[7]),
                    f[8],
                    int(f[10]),
                )
                stops.add(stop)
                searched += expansions > 0
                assert lower <= u <= h and (stop == "optimal") == (lower == u), line
                slots = (
                    [tuple(map(int, s.split(":"))) for s in f[14].split(",")] if len(f) > 14 else []
                )
                domains, search = searches[pid]
                why = _is_schedule(search, domains, u, slots)
                assert why is None, f"{why}: {line}"
        assert not differ, f"{len(differ)} of {len(runs)} differ; first: {differ[0]}"
        # the corpus is not vacuous: searches ran, some closed, some stopped on the budget
        assert stops == {"optimal", "budget"} and searched > 100 and len(runs) > 3000, (
            stops,
            searched,
            len(runs),
        )


def test_a_refusal_is_a_status_and_a_malformed_file_the_tools():
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return
        cyclic = "budget 9\nphase 1 2 2 6\nclaim 1 3 2 0,1\nclaim 2 3 1 0,1\nhslot 1 0 0 3\nhslot 2 0 3 6\n"
        not_a_schedule = "budget 9\nphase 1 2 2 5\nclaim 1 3 - 0,1\nclaim 2 3 - 0,1\nhslot 1 0 0 3\nhslot 2 0 2 5\n"
        assert _drive(exe, cyclic) == "ERR 3\n"
        assert _drive(exe, not_a_schedule) == "ERR 1\n"
        with open(os.path.join(tmp, "bad"), "w", encoding="ascii") as f:
            f.write("budget 1\nphase 1 2 2 x\n")
        ran = subprocess.run(
            [exe, os.path.join(tmp, "bad")], capture_output=True, text=True, timeout=60
        )
        assert ran.returncode == 2 and ran.stdout.startswith("TOOL "), ran.stdout
