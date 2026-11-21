"""PORT-7's question, measured: the C planner's full re-plan against the oracle's incremental planner,
one-claim deltas of the audit fixture, per scale."""

import gc, json, os, statistics, subprocess, sys, tempfile, time
from bcir.abi.planner_abi import encode_input
from bcir.kbcir.delta import Delta, IncrementalPlan
from bcir.kbcir.realize import optimize
from bcir.kbcir.weights import PERF
from bcir.tests.delta_fixtures import audit_delta, declared
from bcir.tests.planner_fixtures import audit_fixture

exe = sys.argv[1]
rows = []
for scale in (2, 4, 6, 8):
    module, h, theta = audit_fixture(scale)
    n = sum(len(p.claims) for p in module.phases)
    plan = IncrementalPlan.build(module, h, theta, PERF)
    inc, full = [], []
    cur = module
    for k in range(7):
        claims, resources = audit_delta(cur, k)
        new = declared(cur, claims, resources)
        gc.collect()
        t0 = time.perf_counter()
        optimize(new, h, theta, PERF)
        full.append(time.perf_counter() - t0)
        gc.collect()
        t0 = time.perf_counter()
        plan.apply(Delta(claims, resources))
        inc.append(time.perf_counter() - t0)
        cur = new
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "in.bkpi")
        open(p, "wb").write(encode_input(cur, h, theta, PERF))
        reps = max(1, 2000 // max(1, n // 512))
        out = subprocess.run(
            [exe, "--bench", p, str(reps), "9"], capture_output=True, text=True, check=True
        ).stdout
        c_ns = int(out.split()[1])
    rows.append(
        {
            "scale": scale,
            "claims": n,
            "oracle_full_ms": round(statistics.median(full) * 1e3, 3),
            "oracle_incremental_ms": round(statistics.median(inc) * 1e3, 3),
            "c_full_ms": round(c_ns / 1e6, 4),
        }
    )
    print(json.dumps(rows[-1]), flush=True)
