"""N14 in C: budgeted planning, `kbcir.rcsp.optimize_constrained` in the C planner, plan for plan.

`bcir_kp_plan_budgeted` (`runtime/c/bcir_kplan.c`) runs the oracle's label DP over the planner's own offer: per
realization the non-dominated (score, tracked resources) labels, an extension past a cap cut, a dominated label
pruned with ties keeping the first, the plan the last column's least label. Held here, over the planner corpus
(`planner_fixtures.corpus_cases`: every example program and generated module, under every target, Theta and
policy), each under budgets drawn from the unconstrained plan's own resources -- one, two and three capped
dimensions, from at the plan's own figure (feasible, maybe binding) down to zero (infeasible):

- **The plan** is `optimize_constrained`'s BKPR, octet for octet, or both refuse it as infeasible at the same claim.
- **The degenerate case**: with no cap the budgeted planner's plan is `bcir_kp_plan`'s (and the oracle's
  `optimize`'s) byte for byte.
- **Feasibility**: every plan both rails admit fits its caps (`rcsp.feasible`).
"""

from __future__ import annotations

import random
import struct
import tempfile

from bcir.abi import planner_abi as pa
from bcir.kbcir.realize import optimize
from bcir.kbcir import rcsp
from bcir.kbcir.cost import DIMS
from bcir.kbcir.rcsp import (
    Budget,
    Infeasible,
    feasible,
    optimize_constrained,
    pareto_plans,
    plan_resources,
)
from bcir.tests import planner_fixtures as pf

_DIM_SETS = ((5,), (6,), (1,), (0,), (5, 6), (1, 5), (0, 1, 6), (2, 3, 7))
_FRACTIONS = (None, 1.0, 0.95, 0.8, 0.5, 0.0)  # None: no cap at all


def _budgets(r: random.Random, case) -> list[tuple]:
    """Budgets for one case, as (dim, cap) tuples: the unconstrained plan's resources scaled, and a point of the
    oracle's own Pareto front over the same dimensions -- feasible, and binding when it is not the cheapest."""
    res = plan_resources(optimize(case.module, case.h, case.theta, case.policy), case.theta).v
    out = [()]
    for dims in r.sample(_DIM_SETS, 3):
        f = r.choice(_FRACTIONS[1:])
        out.append(tuple((d, max(0, int(res[d] * f))) for d in dims))
    dims = r.choice(_DIM_SETS[4:7])
    front = pareto_plans(case.module, case.h, case.theta, case.policy, tuple(DIMS[d] for d in dims))
    point = plan_resources(r.choice(front[1:] or front), case.theta).v
    out.append(tuple((d, max(0, point[d] - r.choice((0, 0, 1)))) for d in dims))
    return out


def _oracle(case, caps) -> tuple[str, int, int, bytes]:
    """(status, the infeasible claim, the labels the DP kept, the BKPR). The labels kept are every slot's
    survivors, counted off `rcsp._insert`'s own slot lists -- what the C planner must hold, no more."""
    slots: dict = {}
    insert = rcsp._insert

    def counting(labels, new):
        insert(labels, new)
        slots[id(labels)] = labels

    rcsp._insert = counting
    try:
        result = optimize_constrained(case.module, case.h, case.theta, case.policy, Budget(caps))
    except Infeasible as exc:
        kept = sum(len(v) for v in slots.values())
        return "BCIR_ERR_INFEASIBLE", int(str(exc).split()[1].rstrip(":")), kept, b""
    finally:
        rcsp._insert = insert
    kept = sum(len(v) for v in slots.values())
    if caps:
        assert feasible(result, case.theta, Budget(caps)), caps
    try:
        return "BCIR_OK", 0xFFFFFFFF, kept, pa.encode_realization(result)
    except pa.PlannerAbiError as exc:
        return exc.status, 0xFFFFFFFF, kept, b""


def requests(seed: int = 59, count: int = 500):
    """(the batch's records, the oracle's answers): a sample of the planner corpus, each case under its budgets."""
    r = random.Random(seed)
    cases = pf.corpus_cases()
    picked = r.sample(cases, count)
    records, want = [], []
    for case in picked:
        bkpi = pa.encode_input(case.module, case.h, case.theta, case.policy)
        for caps in _budgets(r, case):
            records.append((bkpi, caps))
            want.append((case.label, caps, _oracle(case, caps)))
    return records, want


def c_batch_budget(exe: str, tmp: str, records) -> list[tuple[str, int, int, bytes]]:
    import os
    import subprocess

    src, dst = os.path.join(tmp, "budget.in"), os.path.join(tmp, "budget.out")
    with open(src, "wb") as f:
        for bkpi, caps in records:
            f.write(struct.pack("<I", len(bkpi)) + bkpi + struct.pack("<I", len(caps)))
            for d, cap in caps:
                f.write(struct.pack("<IQ", d, cap))
    run = subprocess.run(
        [exe, "--batch-budget", src, dst], capture_output=True, text=True, timeout=900
    )
    if run.returncode != 0:
        raise AssertionError(f"the budget batch failed ({run.returncode}): {run.stderr[-500:]}")
    with open(dst, "rb") as f:
        blob = f.read()
    out, pos = [], 0
    for _ in records:
        status, infeasible, kept, n = struct.unpack_from("<IIII", blob, pos)
        pos += 16
        out.append((pf.status_name(status), infeasible, kept, blob[pos : pos + n]))
        pos += n
    assert pos == len(blob), "the batch output does not match the records sent"
    return out


def check_driver(exe: str) -> dict:
    """The budgeted C planner == the oracle over every (case, budget); what the corpus reached. The one comparison
    the test and the gate's `rcsp` section run."""
    records, want = requests()
    with tempfile.TemporaryDirectory() as tmp:
        got = c_batch_budget(exe, tmp, records)
        plain = pf.c_batch(exe, tmp, [bkpi for (bkpi, caps) in records if not caps])
    seen = {"runs": len(want), "planned": 0, "infeasible": 0, "binding": 0, "unbounded": 0}
    unconstrained = iter(plain)
    for (label, caps, w), g in zip(want, got):
        assert g == w, (label, caps, w[:3], g[:3])
        if not caps:  # the degenerate case: bcir_kp_plan's plan, byte for byte
            assert next(unconstrained) == (w[0], w[3]), label
            seen["unbounded"] += 1
        if w[0] == "BCIR_OK":
            seen["planned"] += 1
        elif w[0] == "BCIR_ERR_INFEASIBLE":
            seen["infeasible"] += 1
    # a binding budget: a feasible plan other than the unconstrained one
    by_case: dict = {}
    for label, caps, w in want:
        by_case.setdefault(label, []).append((caps, w))
    for runs in by_case.values():
        free = next(w for caps, w in runs if not caps)
        seen["binding"] += sum(
            1 for caps, w in runs if caps and w[0] == "BCIR_OK" and w[3] != free[3]
        )
    seen["labels"] = sum(w[2] for (_label, _caps, w) in want)
    return seen


def test_the_c_planner_plans_under_a_budget_as_the_oracle_does():
    with tempfile.TemporaryDirectory() as tmp:
        exe = pf.build_harness(tmp)
        if exe is None:
            return
        seen = check_driver(exe)
        # not vacuous: plans under budgets that bind, infeasible budgets, and the degenerate case
        assert seen["planned"] > 1000 and seen["infeasible"] > 100 and seen["binding"] > 100, seen
        assert seen["unbounded"] == 500, seen


def test_the_budgeted_planner_holds_its_api_laws():
    """`test_kplan --api-budget`: no cap is `bcir_kp_plan`'s plan byte for byte (and every dimension capped at
    2**64 - 1, over misaligned label storage); thirteen caps, a dimension past cost.DIMS, falling or repeated
    dimensions refused as BCIR_ERR_PLANNER; no label storage, storage short of the plan's columns and no out_len
    as BCIR_ERR_NOSPACE; compute capped at 0 as BCIR_ERR_INFEASIBLE with the claim named -- each refusal zeroing
    the output."""
    import os
    import subprocess

    with tempfile.TemporaryDirectory() as tmp:
        exe = pf.build_harness(tmp)
        if exe is None:
            return
        seed = os.path.join(tmp, "seed.bkpi")
        with open(seed, "wb") as f:
            f.write(pf._seed_input()[0])
        run = subprocess.run(
            [exe, "--api-budget", seed], capture_output=True, text=True, timeout=120
        )
        assert run.returncode == 0 and run.stdout.startswith("API OK "), run.stdout + run.stderr
