"""PORT-6: the hosted C++ solver library's second solver, the exact finite-domain solver (`kbcir.csp`), at parity.

N5 of the native port audit (`docs/research/BCIR_NATIVE_PORT_AUDIT_2026-11-09.md`, CXX3): `runtime/cpp/bcir_csp.cpp`
behind `bcir_csp.h` is `csp.solve`'s branch and bound node for node -- the propagators in their order to their
fixpoint, the incumbent's cut, the branching rule, the stack, the budget -- over the arena and the error convention
PORT-5 laid down (`bcir_solver.hpp`).

What holds it here:

- **Parity.** Every model of the corpus the oracle's own tests use -- `random_model` (linear rows, cumulatives and
  rectangles over small domains) and the joint schedule x memory regions (`joint_model(random_region(...))`) -- goes
  to the oracle and to `test_csp FILE` under six budgets, from 0 to past closing, without and with an incumbent, and
  the two print one text: the stop, the values, the objective, the bound and the nodes expanded.
- **The answer is checked, not trusted.** Every assignment the C++ returns satisfies the model by the oracle's own
  definitions (`csp.satisfied`), and where the search closed its objective is brute force's.
- **The façade's contracts** run in `test_csp` itself: every refusal, the budget's stop and bound, an incumbent
  seeding the search and one refused, term merging, determinism, and every allocation of a solve refused in turn.

Skips cleanly where no C++23 compiler is visible (the quick tier hides the host's) or outside a source checkout.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

from bcir.kbcir.csp import Cumulative, Linear, Model, brute_force, satisfied, solve
from bcir.kbcir.joint import joint_model
from bcir.tests.handoff_fixtures import cxx_compiler
from bcir.tests.joint_fixtures import CSP_MODELS, JOINT_REGIONS, random_model, random_region

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_C = os.path.join(_ROOT, "runtime", "c")
_CPP = os.path.join(_ROOT, "runtime", "cpp")
# the harness and the library it links: seam_tests/test_csp's closure in runtime/manifest.json
_SOURCES = ("test_csp.cpp", "bcir_csp.cpp", "bcir_exact.cpp")
_BUDGETS = (200_000, 0, 1, 3, 17, 60)
_FLAGS = ("-std=c++23", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Werror")
_CXX = cxx_compiler()  # None in the quick tier, which hides the host's compilers
_OPS = {"<=": "le", ">=": "ge", "==": "eq"}


def _build(tmp: str) -> str | None:
    """`test_csp` built as the gate builds it, or None without a C++ compiler or a source checkout."""
    from bcir.tests.run_all import _is_source_checkout

    if _CXX is None or not _is_source_checkout():
        return None
    exe = os.path.join(tmp, "test_csp" + (".exe" if os.name == "nt" else ""))
    argv = [_CXX, *_FLAGS, "-I", _C, "-I", _CPP]
    argv += [os.path.join(_CPP, s) for s in _SOURCES] + ["-o", exe]
    try:
        built = subprocess.run(argv, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AssertionError(f"test_csp did not build: {exc}") from exc
    assert built.returncode == 0, f"test_csp did not build:\n{built.stderr[-3000:]}"
    return exe


def _terms(terms) -> str:
    return ",".join(f"{c}:{v}" for c, v in terms) or "-"


def model_text(model: Model, budget: int, incumbent=None) -> str:
    """`model` in `test_csp FILE`'s text, solved under `budget` (and `incumbent`)."""
    lines = ["problem"]
    lines += [f"var {lo} {hi} {p}" for lo, hi, p in zip(model.lo, model.hi, model.priority)]
    for con in model.constraints:
        if isinstance(con, Linear):
            lines.append(f"linear {_OPS[con.op]} {con.rhs} {_terms(con.terms)}")
        elif isinstance(con, Cumulative):
            lines.append(
                f"cumulative {con.capacity} {','.join(f'{s}:{d}' for s, d in con.tasks) or '-'}"
            )
        else:
            lines.append("rects " + (";".join(":".join(map(str, r)) for r in con.rects) or "-"))
    lines.append(f"objective {model.offset} {_terms(model.objective)}")
    if incumbent is not None:
        lines.append("incumbent " + ",".join(map(str, incumbent)))
    lines.append(f"budget {budget}")
    return "\n".join(lines) + "\n"


def expected(model: Model, budget: int, incumbent=None) -> str:
    """The oracle's `solve`, as `test_csp FILE` prints its own."""
    r = solve(model, budget=budget, incumbent=incumbent)
    values = ",".join(map(str, r.values)) if r.values is not None else "-"
    objective = r.objective if r.objective is not None else "-"
    lower = r.lower_bound if r.lower_bound is not None else "-"
    return f"stop {r.stop_reason} values {values} objective {objective} lower {lower} nodes {r.nodes}\n"


def corpus():
    """(model, incumbent) problems: the oracle tests' random models and joint regions, each bare and, where a cheap
    search finds one, seeded with an assignment that is not yet the optimum."""
    models = [random_model(s) for s in range(CSP_MODELS)]
    models += [joint_model(random_region(s))[0] for s in range(JOINT_REGIONS)]
    cases = []
    for model in models:
        cases.append((model, None))
        seed = solve(model, budget=4)
        if seed.values is not None:
            cases.append((model, seed.values))
    return cases


def _drive(exe: str, text: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "problems")
        with open(path, "w", encoding="ascii") as f:
            f.write(text)
        try:
            ran = subprocess.run([exe, path], capture_output=True, text=True, timeout=1800)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AssertionError(f"test_csp did not run: {exc}") from exc
    assert ran.returncode == 0, ran.stdout[-2000:] + ran.stderr[-2000:]
    return ran.stdout


def parity(exe: str) -> tuple[list, dict]:
    """Every corpus problem under every budget on both rails: (the differences, what the corpus reached)."""
    runs = [(m, inc, b) for m, inc in corpus() for b in _BUDGETS]
    got = _drive(exe, "".join(model_text(m, b, inc) for m, inc, b in runs)).splitlines(
        keepends=True
    )
    assert len(got) == len(runs), (len(got), len(runs))
    differ, seen = [], {"optimal": 0, "infeasible": 0, "budget": 0, "seeded": 0, "runs": len(runs)}
    for (model, inc, budget), line in zip(runs, got):
        want = expected(model, budget, inc)
        if line != want:
            differ.append((budget, inc is not None, want, line))
            continue
        f = line.split()
        seen[f[1]] += 1
        seen["seeded"] += inc is not None
        if f[3] != "-":
            values = tuple(int(x) for x in f[3].split(","))
            assert satisfied(model, values), (
                line
            )  # checked by the oracle's definitions, not the solver's
        if f[1] == "optimal" and budget == _BUDGETS[0] and inc is None:
            size = 1
            for lo, hi in zip(model.lo, model.hi):
                size *= hi - lo + 1
            if size <= 200_000:
                assert brute_force(model)[0] == int(f[5]), line
    return differ, seen


def test_the_facades_contracts_hold():
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return
        try:
            ran = subprocess.run([exe], capture_output=True, text=True, timeout=900)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AssertionError(f"test_csp did not run: {exc}") from exc
        assert ran.returncode == 0 and ran.stdout.strip() == "test_csp: OK", ran.stdout + ran.stderr


def test_the_cpp_solver_answers_as_the_oracle_does_and_its_answers_check():
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return
        differ, seen = parity(exe)
        assert not differ, f"{len(differ)} of {seen['runs']} differ; first: {differ[0]}"
        # the corpus is not vacuous: every stop is reached, seeded searches among them
        assert all(seen[k] > 20 for k in ("optimal", "infeasible", "budget", "seeded")), seen


def test_a_malformed_problem_is_the_tools_and_a_refusal_a_status():
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return
        refused = "problem\nvar 3 2 0\nobjective 0 -\nbudget 5\n"
        wrong = "problem\nvar 0 3 0\nlinear ge 2 1:0\nobjective 0 1:0\nincumbent 1\nbudget 5\n"
        assert _drive(exe, refused) == "ERR 1\n"
        assert _drive(exe, wrong) == "ERR 4\n"
        with open(os.path.join(tmp, "bad"), "w", encoding="ascii") as f:
            f.write("problem\nvar 0 x 0\n")
        ran = subprocess.run(
            [exe, os.path.join(tmp, "bad")], capture_output=True, text=True, timeout=60
        )
        assert ran.returncode == 2 and ran.stdout.startswith("TOOL "), ran.stdout
