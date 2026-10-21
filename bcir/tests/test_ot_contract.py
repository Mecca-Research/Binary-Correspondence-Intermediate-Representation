"""OT/BG (2026-10-21): the backend-optimality experiments' measurement contract and its gates.

`tools/perf/ot_contract.py` transcribes PR #821's proposal -- sixteen optimality experiments (OT-01
to OT-16) and eight entry gates (BG0 to BG7) -- into a registry, the record every experiment
writes (BG0: route, semantic and target IDs, an independent reference, raw records, memory
scopes, unsupported and fallback labels) and the gates as requirements over the evidence;
`tools/perf/ot_run.py` runs the experiments a host can run and writes the others' records as
unavailable or blocked, each with its reason.

The witnesses hold, without a compiler: the registry's sixteen and eight, each experiment's routes,
gaps, needs and gates drawn from the contract's own vocabularies, every experiment feeding BG0;
the record's one shape and every rule's refusal by its own message; the gates' evaluation --
nothing PASS without the evidence that meets every requirement, an unavailable capability or
record UNAVAILABLE and never a pass, a gate never ahead of the gate it requires, a milestone met
only by a report the repository holds, a record the contract refuses never evidence -- and the
host's capabilities, a declared one admitted only from the vocabulary. With one: a smoke run of the runner writes all sixteen records under the contract,
the experiments this host runs measured, OT-01's every element within its binary32 bound.
"""

import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.perf import ot_contract as oc  # noqa: E402

#: The host's C compiler, or None: the runner's one capability beyond Python.
_CC = __import__("bcir.toolchain", fromlist=["host_c_compiler"]).host_c_compiler()


def _expect_refused(label, record, reason):
    try:
        oc.validate_record(record)
    except ValueError as exc:
        assert reason in str(exc), (label, str(exc))
        return
    raise AssertionError(f"{label} admitted")


def _measured(experiment="OT-04", **over):
    record = {
        "experiment": experiment, "status": "measured", "routes": ["B-DIRECT", "B-PLAN"],
        "semantic": "native.decoder.step", "reference": "bcir.default.program",
        "target": {"cpu": "test cpu", "threads": 1}, "versions": {"bcir": "abc"},
        "results": [{"one_state": True}], "raw": [{"ms": [1.0]}], "memory_scopes": {"x": 4},
        "labels": {"unsupported": [], "fallback": []}, "gaps": ["integration"],
    }  # fmt: skip
    record.update(over)
    return record


def test_the_registry_is_the_proposals_sixteen_and_eight():
    """OT-01 to OT-16 in order, each drawing its routes, gaps and needs from the contract's
    vocabularies and feeding BG0 and gates of the eight; BG0 to BG7 in order, every requirement of
    a known kind, BG0 requiring every experiment's record, every gate a gate requires defined
    before it, and every route of section 3 compared by some experiment."""
    assert [e.id for e in oc.EXPERIMENTS] == [f"OT-{i:02d}" for i in range(1, 17)]
    assert [g.id for g in oc.GATES] == [f"BG{i}" for i in range(8)]
    gates = {g.id for g in oc.GATES}
    for e in oc.EXPERIMENTS:
        assert e.routes and set(e.routes) <= set(oc.ROUTES), e.id
        assert e.gaps and set(e.gaps) <= set(oc.GAPS), e.id
        assert e.needs and set(e.needs) <= set(oc.CAPABILITIES), e.id
        assert "BG0" in e.feeds and set(e.feeds) <= gates, e.id
        assert e.section in ("5.1", "5.2", "5.3") and e.title, e.id
    assert [e.section for e in oc.EXPERIMENTS] == ["5.1"] * 6 + ["5.2"] * 5 + ["5.3"] * 5
    defined = set()
    for g in oc.GATES:
        for r in g.requirements:
            assert r.kind in oc.REQUIREMENT_KINDS and r.why, (g.id, r)
            if r.kind == "gate":
                assert r.key in defined, (g.id, r.key)  # evaluated in order
            if r.kind in ("record", "any-record"):
                assert r.key in {e.id for e in oc.EXPERIMENTS}, (g.id, r.key)
            if r.kind == "capability":
                assert r.key in oc.CAPABILITIES, (g.id, r.key)
            if r.kind == "milestone":
                assert r.key in oc.MILESTONES, (g.id, r.key)
        defined.add(g.id)
    bg0 = next(g for g in oc.GATES if g.id == "BG0")
    assert sorted(r.key for r in bg0.requirements) == [e.id for e in oc.EXPERIMENTS]
    assert {r for e in oc.EXPERIMENTS for r in e.routes} == set(oc.ROUTES)
    # an experiment's record feeds a gate exactly where the gate requires it measured
    for g in oc.GATES:
        for r in g.requirements:
            if r.kind == "record":
                assert g.id in next(e for e in oc.EXPERIMENTS if e.id == r.key).feeds, (g.id, r)


def test_a_record_has_one_shape_and_refuses_the_rest():
    """A measured record names its routes (each its experiment's), a semantic ID and a reference
    (lower-case ASCII), its target and versions, non-empty results and raw records, its memory
    scopes in bytes, its two label lists and the gaps it reports (each its experiment's), and no
    reason; an unavailable one a reason and the capabilities it lacks (each one its experiment
    needs); a blocked one a reason alone -- every other shape refused by its own rule."""
    assert oc.validate_record(_measured()) == _measured()
    unavailable = {"experiment": "OT-09", "status": "unavailable", "missing": ["gpu"],
                   "reason": "no GPU"}  # fmt: skip
    blocked = {"experiment": "OT-15", "status": "blocked", "reason": "no adapter"}
    oc.validate_record(unavailable)
    oc.validate_record(blocked)
    cases = [
        ("not an object", [], "is a JSON object"),
        ("an empty object", {}, "is a JSON object"),
        ("an unknown key", _measured(speedup=2.0), "has no key 'speedup'"),
        ("an experiment of none", _measured(experiment="OT-17"), "one of the sixteen"),
        ("a status of none", _measured(status="passed"), "has a status of"),
        ("no routes", _measured(routes=[]), "names routes of OT-04's"),
        ("another experiment's route", _measured(routes=["T-DSL"]), "names routes of OT-04's"),
        ("a route twice", _measured(routes=["B-PLAN", "B-PLAN"]), "names routes of OT-04's"),
        ("a semantic ID of capitals", _measured(semantic="Step"), "its semantic as a lower-case"),
        ("a non-ASCII reference", _measured(reference="étape"), "its reference as a lower"),
        ("a target without threads", _measured(target={"cpu": "x"}), "its target's cpu and"),
        ("no threads", _measured(target={"cpu": "x", "threads": 0}), "its target's cpu and"),
        ("a version not a string", _measured(versions={"bcir": 1}), "versions it ran as"),
        ("no versions", _measured(versions={}), "versions it ran as"),
        ("no results", _measured(results=[]), "carries its results"),
        ("no raw records", _measured(raw=[]), "carries its raw"),
        ("a negative scope", _measured(memory_scopes={"x": -1}), "each memory scope in bytes"),
        ("a scope in floats", _measured(memory_scopes={"x": 1.5}), "each memory scope in bytes"),
        ("one label list", _measured(labels={"unsupported": []}), "labels what was unsupported"),
        ("a label not a string", _measured(labels={"unsupported": [1], "fallback": []}),
         "labels what was unsupported"),
        ("no gaps", _measured(gaps=[]), "reports gaps of OT-04's"),
        ("another experiment's gap", _measured(gaps=["compilation"]), "reports gaps of OT-04's"),
        ("a measured record's reason", _measured(reason="x"), "measured carries no reason"),
        ("a measured record's missing", _measured(missing=["gpu"]), "measured carries no reason"),
        ("unavailable without a reason", {**unavailable, "reason": " "}, "unavailable names its"),
        ("unavailable without missing", {k: v for k, v in unavailable.items() if k != "missing"},
         "names the capabilities it lacks"),
        ("unavailable missing what it needs not", {**unavailable, "missing": ["qemu"]},
         "names the capabilities it lacks"),
        ("unavailable with results", {**unavailable, "results": [1]}, "carries no results"),
        ("blocked with missing", {**blocked, "missing": ["cpu"]}, "names a prerequisite"),
        ("blocked with raw records", {**blocked, "raw": [1]}, "carries no results"),
    ]  # fmt: skip
    for key in ("routes", "semantic", "target", "versions", "reference", "results", "raw",
                "memory_scopes", "labels", "gaps"):  # fmt: skip
        record = _measured()
        del record[key]
        cases.append((f"measured without {key}", record, f"measured names its {key}"))
    for label, record, reason in cases:
        _expect_refused(label, record, reason)


def _all_records(**over):
    """A record of every experiment: measured where `over` names one, else unavailable."""
    out = []
    for e in oc.EXPERIMENTS:
        if e.id in over:
            out.append(over[e.id])
        else:
            out.append({"experiment": e.id, "status": "unavailable", "missing": [e.needs[-1]],
                        "reason": "absent here"})  # fmt: skip
    return out


_ROWS = dict.fromkeys(("train.plan.executed.mismatch", "train.plan.program.mismatch",
                       "train.plan.pack.mismatch", "train.plan.choose.mismatch"), 0)  # fmt: skip


def test_a_gate_passes_only_on_the_evidence_that_meets_it():
    """No records: BG0 OPEN, naming every experiment. Every experiment written (most unavailable):
    BG0 PASS -- the contract holds every record -- but BG1 UNAVAILABLE while OT-03 or OT-04 is
    unavailable, OPEN when they are measured but a row is not at its bound or missing, and PASS
    with them and the rows at zero; BG2 UNAVAILABLE without a GPU, and UNAVAILABLE still with one
    declared while its experiments' records are unavailable; a gate requiring an unavailable gate is
    UNAVAILABLE, never PASS; a driver-ladder milestone met only by a report the repository holds,
    named by a plain relative path (no `..`, no absolute or drive path, a file), and a milestone no
    gate names refused; a record the contract refuses, or two of one experiment, is no evidence:
    the evaluation refuses it."""
    caps = {"cpu", "numpy"}
    states = oc.evaluate_gates([], {}, capabilities=caps)
    assert states["BG0"][0] == "OPEN" and len(states["BG0"][1]) == 16
    assert all(states[g][0] != "PASS" for g in states)
    records = _all_records()
    states = oc.evaluate_gates(records, _ROWS, capabilities=caps)
    assert states["BG0"] == ("PASS", [])
    assert states["BG1"][0] == "UNAVAILABLE"  # OT-03 and OT-04 unavailable here
    measured = _all_records(**{"OT-03": _measured("OT-03"), "OT-04": _measured("OT-04")})
    assert oc.evaluate_gates(measured, {}, capabilities=caps)["BG1"][0] == "OPEN"
    off = dict(_ROWS, **{"train.plan.pack.mismatch": 3})
    state, unmet = oc.evaluate_gates(measured, off, capabilities=caps)["BG1"]
    assert state == "OPEN" and unmet == ["row train.plan.pack.mismatch: PX-2b: the step from its "
                                         "pack, bit for bit"]  # fmt: skip
    assert oc.evaluate_gates(measured, _ROWS, capabilities=caps)["BG1"] == ("PASS", [])
    states = oc.evaluate_gates(measured, _ROWS, capabilities=caps)
    assert states["BG2"][0] == "UNAVAILABLE"
    assert (
        oc.evaluate_gates(measured, _ROWS, capabilities=caps | {"gpu"})["BG2"][0] == "UNAVAILABLE"
    )
    assert states["BG7"][0] == "UNAVAILABLE"  # it requires BG2
    for g in ("BG3", "BG4", "BG5", "BG6"):
        assert states[g][0] in ("OPEN", "UNAVAILABLE"), (g, states[g])
    # a milestone is met only by a report the repository holds, named by a plain relative path
    assert states["BG3"] == ("OPEN", ["milestone UART-D2: a D0-D2 UART package beside the "
                                      "RuntimeChannel loopback, with telemetry and bounded "
                                      "lifecycle tests"])  # fmt: skip
    held = "docs/STATUS.md"  # a file the repository holds: the gate reads presence, not content
    assert (ROOT / held).is_file()
    assert oc.evaluate_gates(measured, _ROWS, capabilities=caps,
                             milestones={"UART-D2": held})["BG3"] == ("PASS", [])  # fmt: skip
    for report in (
        "docs/NO_SUCH_REPORT.md",
        "/etc/hostname",
        "../" + held,
        "docs/../" + held,
        "./" + held,
        "docs\\STATUS.md",
        "C:/x",
        "",
        None,
        7,
        "docs",
    ):
        state = oc.evaluate_gates(measured, _ROWS, capabilities=caps,
                                  milestones={"UART-D2": report})["BG3"][0]  # fmt: skip
        assert state == "OPEN", (report, state)
    try:
        oc.evaluate_gates(measured, _ROWS, capabilities=caps, milestones={"M99": held})
    except ValueError as exc:
        assert "M99" in str(exc)
    else:
        raise AssertionError("an unnamed milestone admitted")
    bad = copy.deepcopy(measured)
    bad[3]["routes"] = ["T-DSL"]
    for label, recs in (("a refused record", bad), ("two records of one", measured + measured[:1])):
        try:
            oc.evaluate_gates(recs, _ROWS, capabilities=caps)
        except ValueError:
            continue
        raise AssertionError(f"{label} evaluated")


def _measured_for(e) -> dict:
    """A measured record of experiment `e`: its first route and its first gap."""
    return _measured(e.id, routes=[e.routes[0]], gaps=[e.gaps[0]])


def test_every_gate_can_pass_and_none_outruns_the_gate_it_requires():
    """Complete evidence -- every experiment measured, the rows at their bounds, every
    capability, every milestone attested by a report the repository holds -- passes all eight
    gates: no gate is unpassable by construction. Take away one capability (the GPU) and BG2 is
    UNAVAILABLE, and BG7, which requires BG2 and nothing else that is unmet, UNAVAILABLE too:
    a gate is never PASS while a gate it requires is not, though every other requirement holds."""
    held = "docs/STATUS.md"
    full = [_measured_for(e) for e in oc.EXPERIMENTS]
    caps = set(oc.CAPABILITIES)
    milestones = dict.fromkeys(oc.MILESTONES, held)
    states = oc.evaluate_gates(full, _ROWS, capabilities=caps, milestones=milestones)
    assert all(states[g.id] == ("PASS", []) for g in oc.GATES), states
    states = oc.evaluate_gates(full, _ROWS, capabilities=caps - {"gpu"}, milestones=milestones)
    assert states["BG2"] == ("UNAVAILABLE", ["capability gpu: one real target with a resident "
                                             "provider"]), states["BG2"]  # fmt: skip
    assert states["BG7"] == ("UNAVAILABLE", ["gate BG2: BG2 passed"]), states["BG7"]
    assert all(states[g][0] == "PASS" for g in ("BG0", "BG1", "BG3", "BG4", "BG5", "BG6"))


def test_the_hosts_capabilities_are_probed_or_declared_from_the_vocabulary():
    """The CPU is always a capability; one the operator declares (`BCIR_OT_CAPABILITIES`) is added
    from the vocabulary and a name outside it refused -- a GPU or a device rig is never inferred."""
    saved = os.environ.pop("BCIR_OT_CAPABILITIES", None)
    try:
        caps = oc.host_capabilities()
        assert "cpu" in caps and caps <= set(oc.CAPABILITIES)
        assert "gpu" not in caps and "physical-device" not in caps
        os.environ["BCIR_OT_CAPABILITIES"] = "gpu, physical-device"
        assert {"gpu", "physical-device"} <= oc.host_capabilities()
        os.environ["BCIR_OT_CAPABILITIES"] = "gpu,warp-drive"
        try:
            oc.host_capabilities()
        except ValueError as exc:
            assert "warp-drive" in str(exc)
        else:
            raise AssertionError("an undeclared capability admitted")
    finally:
        os.environ.pop("BCIR_OT_CAPABILITIES", None)
        if saved is not None:
            os.environ["BCIR_OT_CAPABILITIES"] = saved


def test_the_cli_lists_the_registry_and_refuses_a_malformed_records_file():
    """`--list` prints the sixteen and the eight; `--gates` over a file the contract refuses is
    INVALID (exit 2), never a gate verdict."""
    tool = str(ROOT / "tools" / "perf" / "ot_contract.py")
    listed = subprocess.run([sys.executable, tool, "--list"], capture_output=True, text=True,
                            timeout=300)  # fmt: skip
    assert listed.returncode == 0
    assert sum(line.startswith("OT-") for line in listed.stdout.splitlines()) == 16
    assert sum(line.startswith("BG") for line in listed.stdout.splitlines()) == 8
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "records.json"
        for text in (
            json.dumps({"records": [{"experiment": "OT-01", "status": "won"}]}),
            '{"records": [], "records": []}',  # a key twice: refused, never its last value
            json.dumps([]),  # not an object holding records
        ):
            path.write_text(text)
            bad = subprocess.run([sys.executable, tool, "--gates", str(path)],
                                 capture_output=True, text=True, timeout=300)  # fmt: skip
            assert bad.returncode == 2 and "INVALID" in bad.stdout, (text, bad.stdout)


def test_ot01s_checks_hold_a_product_to_its_bound_and_fire_past_it():
    """OT-01's two checks are the experiment's verdict, so each must fire: the float64 product
    rounded once to binary32 is within the GEMM bound (any summation order: gamma_k |A| |B| and the
    output's rounding), one element moved by a thousandth of its magnitude is past it, as is a
    product whose operands were transposed; an elementwise result within its units of roundoff
    passes and one moved by a thousandth fails. Without NumPy there is no reference: the witness
    returns, as the runner records OT-01 unavailable."""
    try:
        import numpy as np
    except ImportError:
        return
    from tools.perf import ot_run

    rng = np.random.default_rng(1)
    a = rng.standard_normal((5, 33))
    b = rng.standard_normal((33, 7))
    exact = (a @ b).astype(np.float32)
    assert ot_run.gemm_check("ok", exact, a, b)["max_error_over_bound"] <= 1.0
    moved = exact.copy()
    moved[2, 3] += np.float32(1e-3) * max(abs(moved[2, 3]), np.float32(1.0))
    assert ot_run.gemm_check("moved", moved, a, b)["max_error_over_bound"] > 1.0
    wrong = (a @ rng.permutation(b)).astype(np.float32)  # another product altogether
    assert ot_run.gemm_check("wrong", wrong, a, b)["max_error_over_bound"] > 1.0
    x = rng.standard_normal(64)
    y = (x * 1.5).astype(np.float32)
    assert (
        ot_run.elementwise_check("ok", y, x * 1.5, np.abs(x * 1.5))["max_error_over_bound"] <= 1.0
    )
    y[10] += np.float32(1e-3) * max(abs(y[10]), np.float32(1.0))
    assert (
        ot_run.elementwise_check("moved", y, x * 1.5, np.abs(x * 1.5))["max_error_over_bound"] > 1.0
    )


def test_a_smoke_run_writes_every_record_under_the_contract():
    """`tools/perf/ot_run.py --smoke`: sixteen records, each validated, the ones this host runs
    measured -- OT-01 (with NumPy), OT-03, OT-04, OT-06 and OT-14 -- every OT-01 element within
    its binary32 bound and the zero-size product a no-op, OT-03's direct, program and pack steps
    one state, OT-04's intervention changing the launches and keeping the state; and the gates
    stated, BG0 PASS."""
    if _CC is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "ot.json"
        run = subprocess.run([sys.executable, str(ROOT / "tools" / "perf" / "ot_run.py"),
                              str(out), "--smoke"], capture_output=True, text=True,
                             timeout=1800)  # fmt: skip
        assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
        doc = json.loads(out.read_text())
    records = {r["experiment"]: r for r in doc["records"]}
    assert sorted(records) == [e.id for e in oc.EXPERIMENTS]
    for r in records.values():
        oc.validate_record(r)
    measured = {k for k, r in records.items() if r["status"] == "measured"}
    want = {"OT-03", "OT-04", "OT-06", "OT-14"} | ({"OT-01"} if "numpy" in doc["capabilities"]
                                                  else set())  # fmt: skip
    assert measured == want, measured
    if "OT-01" in records and records["OT-01"]["status"] == "measured":
        assert all(r["within_bound"] for r in records["OT-01"]["results"]), records["OT-01"]
    assert all(r["one_state"] for r in records["OT-03"]["results"])
    (intervention,) = records["OT-04"]["results"]
    assert intervention["one_state"]
    assert intervention["launches"]["switched"] > intervention["launches"]["default"]
    assert records["OT-14"]["results"][0]["dropped"] == 0
    assert doc["gates"]["BG0"]["state"] == "PASS"
    assert all(g["state"] != "PASS" for k, g in doc["gates"].items() if k in ("BG2", "BG7"))
