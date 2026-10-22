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


def test_the_runner_refuses_an_unusable_input_before_any_experiment_runs():
    """`ot_run.py` reads every file it is named -- the GEM+ rows, a Triton-CPU session, the
    training frontier -- before it builds or runs anything: one it cannot use (empty, not JSON, a
    key twice, no rows, not the report) is refused with exit status 2 and nothing written, never a
    traceback after the run's measurements. A rows report keeps the rows that measured a number."""
    from bcir.tests import native_decoder_fixtures as nf
    from tools.perf import ot_run

    def ran(*_args, **_kwargs):
        raise AssertionError("an experiment's library was built before the inputs were read")

    report = {"rows": [{"key": "a", "measured": 0.0}, {"key": "b", "measured": None},
                       {"key": "c", "measured": True}, {"key": "d", "measured": 3}]}  # fmt: skip
    with tempfile.TemporaryDirectory() as tmp:
        path, out = Path(tmp) / "input.json", Path(tmp) / "ot.json"
        path.write_text(json.dumps(report))
        assert ot_run._rows(path) == {"a": 0.0, "d": 3}
        saved = nf.build_library
        nf.build_library = ran
        try:
            for flag, text in (
                ("--rows", ""),
                ("--rows", "{"),
                ("--rows", '{"rows": [{"key": "a", "measured": 1}], "rows": []}'),
                ("--rows", json.dumps({"rows": []})),
                ("--rows", json.dumps([report])),
                ("--triton", json.dumps({"protocol": "v1"})),
                ("--frontier", None),
            ):  # fmt: skip
                if text is not None:
                    path.write_text(text)
                named = str(path) if text is not None else str(Path(tmp) / "absent")
                assert ot_run.main([str(out), flag, named]) == 2, (flag, text)
                assert not out.exists(), (flag, text)
        finally:
            nf.build_library = saved


def test_a_pytorch_routes_record_names_its_pytorch_and_its_threads():
    """A measured record names the software that ran -- with PyTorch a capability, the PyTorch the
    P-* routes ran, each version a plain string (`torch.__version__` is a str subclass, which the
    contract's versions refuse) -- and its target names the threads it ran at: the PyTorch routes'
    runners the host's, OT-10 its runs', every other experiment one."""
    from tools.perf import ot_run, ot_torch

    class Version(str):
        pass

    saved = ot_torch.torch_versions
    ot_torch.torch_versions = lambda: {"torch": Version("2.13.0+cpu"), "torch_threads": "4"}
    try:
        with_torch, without = ot_run._versions({"cpu", "torch"}), ot_run._versions({"cpu"})
    finally:
        ot_torch.torch_versions = saved
    assert type(with_torch["torch"]) is str and with_torch["torch"] == "2.13.0+cpu", with_torch
    assert "torch" not in without and without["bcir"] == with_torch["bcir"], without
    frontier = {"results": [{"recipe": {"steps": 300}}, {"route": "B-DIRECT", "threads": 4},
                            {"route": "P-EAGER", "threads": 2}]}  # fmt: skip
    ran_at = ot_run._ran_at(4, frontier)
    assert ran_at == {"OT-02": 4, "OT-05": 4, "OT-11": 4, "OT-10": 4}, ran_at
    assert "OT-10" not in ot_run._ran_at(4, None)
    measured = {"experiment": "OT-11", "status": "measured"}
    for experiment, threads in (("OT-11", 4), ("OT-10", 4), ("OT-03", 1)):
        stamped = ot_run._stamp({**measured, "experiment": experiment}, ran_at, with_torch)
        assert stamped["target"]["threads"] == threads and stamped["versions"] is with_torch
    blocked = {"experiment": "OT-15", "status": "blocked", "reason": "no adapter"}
    assert ot_run._stamp(blocked, ran_at, with_torch) is blocked


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


def test_an_experiment_without_a_runner_is_recorded_on_every_host():
    """The runner writes a record for every experiment on every host: one this run does not
    measure is unavailable where the host lacks a capability it needs, and blocked -- by the
    prerequisite `BLOCKED` names, or by its runner not yet in this tree -- where the host has them
    all (a host with PyTorch, Triton or a declared GPU), never a failed run; every such record is
    one the contract admits."""
    from tools.perf import ot_run

    measured_here = {"OT-01", "OT-03", "OT-04", "OT-06", "OT-14", *ot_run.TORCH_RUNNERS}
    every = set(oc.CAPABILITIES)
    for e in oc.EXPERIMENTS:  # a PyTorch route's runner where PyTorch imports; unavailable apart
        if e.id in ot_run.TORCH_RUNNERS:
            poor = oc.validate_record(ot_run.unrun(e, {"cpu", "numpy"}))
            assert poor["status"] == "unavailable" and poor["missing"] == ["torch"], (e.id, poor)
    for e in oc.EXPERIMENTS:
        if e.id in measured_here:
            continue
        rich = oc.validate_record(ot_run.unrun(e, every))
        assert rich["status"] == "blocked" and rich["reason"].strip(), (e.id, rich)
        if e.id not in ot_run.BLOCKED:
            assert "no runner" in rich["reason"], (e.id, rich)
        poor = oc.validate_record(ot_run.unrun(e, {"cpu"}))
        lacking = [c for c in e.needs if c != "cpu"]
        if lacking:
            assert poor["status"] == "unavailable" and poor["missing"] == lacking, (e.id, poor)
        else:
            assert poor == rich, (e.id, poor)
    # OT-10 has its runner (ot10): on a host with every capability what it lacks is its data,
    # which --frontier names -- never "no runner"
    ot10 = next(e for e in oc.EXPERIMENTS if e.id == "OT-10")
    frontier = oc.validate_record(ot_run.unrun(ot10, every))
    assert frontier["status"] == "blocked" and "--frontier" in frontier["reason"], frontier
    assert "no runner" not in frontier["reason"], frontier


def test_a_smoke_run_writes_every_record_under_the_contract():
    """`tools/perf/ot_run.py --smoke`: sixteen records, each validated, the ones this host runs
    measured -- OT-01 (with NumPy), OT-03, OT-04, OT-06 and OT-14 -- every OT-01 element within
    its binary32 bound and the zero-size product a no-op, OT-03's direct, program and pack steps
    one state, OT-04's intervention changing the launches and keeping the state, OT-04 and OT-14
    each timed beside an A/A control; and the gates stated, BG0 PASS."""
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
    if "torch" in doc["capabilities"]:
        from tools.perf import ot_run

        want |= set(ot_run.TORCH_RUNNERS)
        assert records["OT-01"]["routes"] == ["B-DIRECT", "P-EAGER"], records["OT-01"]["routes"]
    assert measured == want, measured
    if "OT-01" in records and records["OT-01"]["status"] == "measured":
        assert all(r["within_bound"] for r in records["OT-01"]["results"]), records["OT-01"]
    assert all(r["one_state"] for r in records["OT-03"]["results"])
    (intervention,) = records["OT-04"]["results"]
    assert intervention["one_state"]
    assert intervention["launches"]["switched"] > intervention["launches"]["default"]
    assert records["OT-14"]["results"][0]["dropped"] == 0
    assert records["OT-14"]["results"][0]["one_state"]
    for timed in (intervention, records["OT-14"]["results"][0]):  # each effect beside its A/A
        control = timed["aa_control"]
        assert control["geometric_mean"] > 0 and len(control["ci95"]) == 2, timed
    assert doc["gates"]["BG0"]["state"] == "PASS"
    assert all(g["state"] != "PASS" for k, g in doc["gates"].items() if k in ("BG2", "BG7"))


def _session(**over):
    """A Triton-CPU pilot session as tools/perf/triton_cpu_compare.py writes one, over this tree."""
    import hashlib

    tensor = hashlib.sha256((ROOT / "runtime" / "c" / "bcir_tensor.c").read_bytes()).hexdigest()
    # the harness's two shapes of case: one BCIR implementation (an add, an RMSNorm) or several
    # against the one Triton kernel (the GEMM's fused and k-blocked contracts; SwiGLU's exponentials)
    add = {"kind": "add", "shape": [4096], "threads": 1,
           "correctness": {"bcir": {"pass": True}, "triton": {"pass": True}},
           "median_ns": {"bcir": 1000.0, "triton": 2000.0},
           "ratios": {"bcir": {"bcir_over_triton": 0.5, "paired_bootstrap_95pct": [0.4, 0.6]}}}  # fmt: skip
    gemm = {"kind": "gemm", "shape": [128, 128, 128], "threads": 2,
            "correctness": {"bcir_fused": {"pass": True}, "bcir_kblock": {"pass": True},
                            "triton": {"pass": True}},
            "median_ns": {"bcir_fused": 1500.0, "bcir_kblock": 900.0, "triton": 2000.0},
            "ratios": {"bcir_fused": {"bcir_over_triton": 0.75, "paired_bootstrap_95pct": [0.7, 0.8]},
                       "bcir_kblock": {"bcir_over_triton": 0.45, "paired_bootstrap_95pct": [0.4, 0.5]}}}  # fmt: skip
    doc = {"protocol": "bcir-triton-cpu-pilot-v2", "status": "PASS", "cases": [add, gemm],
           "tensor_c_sha256": tensor, "harness_sha256": "h", "triton_package": "3.8.0+cpu"}  # fmt: skip
    doc.update(over)
    return doc


def test_a_triton_session_is_evidence_about_this_tree_alone():
    """OT-05's T-DSL rows come from the Triton-CPU pilot (its own interpreter): a session is
    admitted under its protocol, passed, every case correct on both sides with its paired ratio,
    and its BCIR side this tree's tensor unit -- a session measured over another tree's
    bcir_tensor.c, a failed one, one with an incorrect case or of another protocol is refused by its
    own reason, as is a file spelling a key twice."""
    from tools.perf import ot_run

    def admitted(doc):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.json"
            path.write_text(doc if isinstance(doc, str) else json.dumps(doc), encoding="utf-8")
            return ot_run._triton_evidence(path)

    rows = admitted(_session())["summary"]["cases"]
    assert rows == [
        {"kind": "add", "shape": [4096], "threads": 1, "implementation": "bcir",
         "median_ns": {"bcir": 1000.0, "triton": 2000.0}, "bcir_over_triton": 0.5,
         "ci95": [0.4, 0.6]},
        {"kind": "gemm", "shape": [128, 128, 128], "threads": 2, "implementation": "bcir_fused",
         "median_ns": {"bcir_fused": 1500.0, "triton": 2000.0}, "bcir_over_triton": 0.75,
         "ci95": [0.7, 0.8]},
        {"kind": "gemm", "shape": [128, 128, 128], "threads": 2, "implementation": "bcir_kblock",
         "median_ns": {"bcir_kblock": 900.0, "triton": 2000.0}, "bcir_over_triton": 0.45,
         "ci95": [0.4, 0.5]},
    ], rows  # fmt: skip

    def changed(edit):
        doc = copy.deepcopy(_session())
        edit(doc["cases"][1])
        return doc

    def untimed(case):
        case["ratios"]["bcir_kblock"]["bcir_over_triton"] = None

    for label, doc, reason in (
        ("another tree", _session(tensor_c_sha256="0" * 64), "this tree's bcir_tensor.c"),
        ("a failed session", _session(status="FAIL"), "passed"),
        ("no cases", _session(cases=[]), "passed"),
        ("an incorrect Triton side",
         changed(lambda c: c["correctness"]["triton"].update({"pass": False})),
         "correct on both sides"),
        ("one of two BCIR sides incorrect",
         changed(lambda c: c["correctness"]["bcir_kblock"].update({"pass": False})),
         "correct on both sides: gemm [128, 128, 128] (bcir_kblock)"),
        ("a BCIR side with no correctness", changed(lambda c: c["correctness"].pop("bcir_fused")),
         "(bcir_fused)"),
        ("a ratio not BCIR's", changed(lambda c: c["ratios"].update(triton=c["ratios"]["bcir_fused"])),
         "(triton)"),
        ("an untimed side", changed(untimed), "(bcir_kblock)"),
        ("a side with no median", changed(lambda c: c["median_ns"].pop("bcir_kblock")), "(bcir_kblock)"),
        ("no implementation timed", changed(lambda c: c.update(ratios={})), "times one of BCIR's"),
        ("another protocol", _session(protocol="v1"), "names protocol"),
        ("a key twice", '{"protocol": "a", "protocol": "bcir-triton-cpu-pilot-v2"}', "twice"),
    ):  # fmt: skip
        try:
            admitted(doc)
        except ValueError as exc:
            assert reason in str(exc), (label, str(exc))
        else:
            raise AssertionError(f"{label} admitted")


def test_the_pytorch_routes_records_keep_every_route_and_failure():
    """OT-02, OT-05 and OT-11 as the runner records them, from measurements shaped as
    tools/perf/ot_torch.py returns them (no PyTorch needed): each record one the contract admits,
    its routes the ones that ran -- a route that failed recorded among the unsupported with its
    failure, never dropped and never named among the routes -- and OT-05's T-DSL rows only where a
    Triton session was named."""
    from tools.perf import ot_run, ot_torch

    saved = (ot_torch.capture_cost, ot_torch.portfolio, ot_torch.lifecycle)
    ms = {"B-DIRECT/v1": 2.0, "B-DIRECT/fused": 1.5, "B-DIRECT/kblock": 1.0,
          "B-DIRECT/kblock-AA": 1.0, "P-EAGER": 3.0}  # fmt: skip
    phase = {"ms": ms, "kblock_over": {}, "raw_ms": {k: [v] for k, v in ms.items()}}
    step = {"first_s": 2.0, "warm_s": 0.5, "frames": 2, "graph_breaks": 1,
            "break_reasons": ["Data-dependent branching\n  Explanation: ..."]}  # fmt: skip
    try:
        ot_torch.capture_cost = lambda threads, smoke: {
            "decoder": {}, "batch": 2, "time": 32, "threads": threads,
            "routes": {"P-FX/module": dict(step), "P-IND/loss": {"failed": "exit 1: boom"},
                       "B-PLAN": {"step_program_ms": 4.0}}}  # fmt: skip
        ot_torch.portfolio = lambda lib, shapes, threads, rounds: {
            "rows": [{"shape": {"m": 4, "k": 33, "n": 65}, "threads": 1, "forward": phase,
                      "forward_backward": phase}],
            "failures": ["P-IND at 4x33x65, 1 thread(s): RuntimeError: no compiler"]}  # fmt: skip
        ot_torch.lifecycle = lambda threads, smoke: {
            "decoder": {}, "batch": 2, "time": 32, "second_time": 16, "threads": threads,
            "cold_static": dict(step), "persistent_static": {"failed": "timeout"},
            "cold_dynamic": dict(step), "B-PLAN": {"plans_ms": [4.0, 3.0]}}  # fmt: skip
        target = {"target": {"cpu": "test cpu", "threads": 1}, "versions": {"bcir": "abc"}}
        two = oc.validate_record({**ot_run.ot02(1, True), **target})
        assert two["routes"] == ["P-FX", "B-PLAN"], two["routes"]
        assert any("P-IND/loss: exit 1: boom" in u for u in two["labels"]["unsupported"])
        fx = next(r for r in two["results"] if r.get("route") == "P-FX/module")
        assert fx["capture_s"] == 1.5 and fx["break_reasons"] == ["Data-dependent branching"]
        five = oc.validate_record({**ot_run.ot05("lib", 4, True, 4), **target})
        assert five["routes"] == ["P-EAGER", "B-DIRECT"], five["routes"]
        assert any("no compiler" in u for u in five["labels"]["unsupported"])
        assert any(u.startswith("T-DSL") for u in five["labels"]["unsupported"])
        session = ot_run._triton_evidence  # the T-DSL rows from a named session alone
        evidence = {"summary": {"cases": []}, "source": {"protocol": ot_run.TRITON_PROTOCOL}}
        with_t = oc.validate_record({**ot_run.ot05("lib", 4, True, 4, evidence), **target})
        assert "T-DSL" in with_t["routes"] and session is ot_run._triton_evidence
        eleven = oc.validate_record({**ot_run.ot11(1, True), **target})
        assert eleven["routes"] == ["P-IND", "B-PLAN"], eleven["routes"]
        assert any("persistent_static: timeout" in u for u in eleven["labels"]["unsupported"])
    finally:
        ot_torch.capture_cost, ot_torch.portfolio, ot_torch.lifecycle = saved


def test_triton_is_a_capability_only_with_a_usable_backend():
    """A Triton package is a capability only where it names an active backend: a GPU build on a
    host without its device imports and runs no kernel, so it is not one; a build whose driver
    resolves is."""
    import importlib.machinery
    import types

    saved = sys.modules.get("triton")

    class Driver:
        def __init__(self, usable):
            self.usable = usable

        @property
        def active(self):
            if not self.usable:
                raise RuntimeError("0 active drivers")
            return object()

    try:
        for usable in (False, True):
            fake = types.ModuleType("triton")
            fake.__spec__ = importlib.machinery.ModuleSpec("triton", None)
            fake.runtime = types.SimpleNamespace(driver=Driver(usable))
            sys.modules["triton"] = fake
            assert oc._triton_usable() is usable, usable
    finally:
        if saved is None:
            sys.modules.pop("triton", None)
        else:
            sys.modules["triton"] = saved


def test_the_frontier_is_read_from_one_recipe_alone():
    """OT-10 reads train_comparison.py's runs over one prepared recipe: each backend its route
    (bcir B-DIRECT, torch-eager P-EAGER, torch-compiled P-IND), each run's curve, held-out
    cross-entropy, step time and peak RSS in a record the contract admits, and the routes it has no
    run of labelled; runs of different step counts, two runs of one route, and a directory without
    runs are refused -- a frontier across two recipes is not one."""
    from tools.perf import ot_run

    def run(backend, steps=3):
        return {"task": "lm", "backend": backend, "contract": "v1", "threads": 4,
                "losses": [5.0] * steps, "evals": [{"step": 0, "train_seconds": 0.0,
                                                    "cross_entropy": 5.5}],
                "final": {"validation": {"cross_entropy": 4.0, "accuracy": 0.2},
                          "test": {"cross_entropy": 4.1, "accuracy": 0.2}},
                "median_step_ms": 10.0, "tokens_per_second": 100.0, "train_seconds": 1.0,
                "rss_mb": {"peak": 300.0}, "state_sha256": "s"}  # fmt: skip

    def frontier(runs):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            (data / "runs").mkdir()
            (data / "prepare.json").write_text(json.dumps({"weights_sha256": "w", "batch": 8,
                                                           "context": 128, "parameters": 9}))  # fmt: skip
            for name, r in runs.items():
                (data / "runs" / f"{name}.json").write_text(json.dumps(r))
            return ot_run.ot10(data)

    record = frontier({"lm-bcir-t4": run("bcir"), "lm-torch-eager-t4": run("torch-eager")})
    record = oc.validate_record({**record, "target": {"cpu": "c", "threads": 4},
                                 "versions": {"bcir": "abc"}})  # fmt: skip
    assert record["routes"] == ["P-EAGER", "B-DIRECT"], record["routes"]
    assert any(u.startswith("P-IND") for u in record["labels"]["unsupported"])
    for label, runs, reason in (
        ("two step counts", {"lm-bcir": run("bcir"), "lm-torch-eager": run("torch-eager", 4)},
         "different step counts"),
        ("one route twice", {"lm-bcir": run("bcir"), "lm-bcir-t4": run("bcir")}, "two B-DIRECT"),
        ("no runs", {}, "no train_comparison.py runs"),
    ):  # fmt: skip
        try:
            frontier(runs)
        except ValueError as exc:
            assert reason in str(exc), (label, str(exc))
        else:
            raise AssertionError(f"{label} admitted")


def test_ot01s_verdict_is_every_routes():
    """OT-01 where PyTorch imports: P-EAGER is among its routes and the record's verdict is every
    route's -- a P-EAGER tensor past its bound makes the final within_bound false whatever
    B-DIRECT's, and one within it leaves the verdict B-DIRECT's. P-EAGER's measurement stands in
    here (no PyTorch needed); B-DIRECT runs on the plain library."""
    if _CC is None:
        return
    try:
        import numpy  # noqa: F401
    except ImportError:
        return
    import ctypes

    from bcir.tests import native_decoder_fixtures as nf
    from tools.perf import ot_run, ot_torch

    saved = ot_torch.p_eager_block

    def stand_in(worst):
        row = {"route": "P-EAGER", "shape": {"m": 4, "k": 33, "n": 65},
               "within_bound": worst <= 1.0, "max_error_over_bound": worst}  # fmt: skip
        return lambda shapes: ([row], [{"route": "P-EAGER", "checks": []}], worst)

    try:
        with tempfile.TemporaryDirectory() as tmp:
            lib = nf.build_library(tmp)
            if lib is None:
                return
            for worst, verdict in ((2.0, False), (0.5, True)):
                ot_torch.p_eager_block = stand_in(worst)
                rec = ot_run.ot01(ctypes.CDLL(str(lib)), ot_run.OT01_SMOKE, with_torch=True)
                assert rec["routes"] == ["B-DIRECT", "P-EAGER"], rec["routes"]
                final = rec["results"][-1]
                assert final["within_bound"] is verdict, (worst, final)
                assert final["p_eager_worst_error_over_bound"] == worst
                assert not any("P-EAGER" in u for u in rec["labels"]["unsupported"])
    finally:
        ot_torch.p_eager_block = saved
