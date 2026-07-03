"""Proof-carrying optimization records: explain / replay / reduce (bcir-explain/replay/reduce).

Roadmap (BCIR_MASTER_ROADMAP.md §5.3 / §6): replayable decision records + rewrite
certificates over R13's provenance foundation. `explain` builds a `DecisionRecord` (the
provenance digest + the per-claim candidates-weighed-and-chosen rationale + the bundle
rewrite certificates); `replay` reproduces it bit-for-bit from the same inputs or reports
exactly what diverged; `reduce` minimizes a module to a witness. The record round-trips
through JSON, and the CLI exposes all three.
"""

from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import Theta
from bcir.kbcir.proof import DecisionRecord, explain, explain_text, reduce, replay
from bcir.examples import fused_chain, matmul_tiled, multi_histogram, vector_add
from bcir.verify import verify

AVX = TARGETS["x86_avx512"]
SVE = TARGETS["arm64_sve"]
COOL = Theta.cool()
HOT = Theta.hot()


def test_explain_records_the_per_claim_rationale():
    rec = explain(vector_add(), AVX, COOL, target_name="x86_avx512")
    assert rec.module_name == "vec_add" and rec.target == "x86_avx512"
    assert rec.theta == "cool" and rec.policy == "latency"
    assert rec.total_score == 7808
    (d,) = rec.decisions
    assert d.claim_id == 1000 and d.chosen == "vec16" and d.width == 16 and d.score == 7808
    cands = dict(d.candidates)
    # the optimizer weighed all three widths; vec16 (7808) beat vec8 (9472) and scalar.
    assert cands["vec16"] == 7808 and cands["vec8"] == 9472 and cands["vec16"] < cands["vec8"]


def test_record_round_trips_through_json():
    rec = explain(matmul_tiled(), AVX, COOL, target_name="x86_avx512", joint=True)
    assert DecisionRecord.from_json(rec.to_json()) == rec


def test_records_are_versioned_self_describing_envelopes():
    """0.4b (the stable certificate schema): a serialized record carries its kind + schema
    version, so a decoder can upgrade -- or refuse -- by version alone."""
    import json
    from bcir.kbcir.proof import RECORD_KIND, SCHEMA_VERSION
    d = json.loads(explain(vector_add(), AVX, COOL, target_name="x86_avx512").to_json())
    assert d["kind"] == RECORD_KIND == "bcir.decision_record"
    assert d["schema"] == SCHEMA_VERSION == 2
    assert d["record"]["module_name"] == "vec_add"       # the payload sits under the envelope


def test_a_legacy_v1_record_upgrades_and_replays():
    """The decode/UPGRADE path: a bare pre-envelope (v1) document -- exactly what an older
    build wrote -- decodes through the upgrade chain to the identical record and still
    replays bit-for-bit. Old certificates never rot."""
    import json
    rec = explain(vector_add(), AVX, COOL, target_name="x86_avx512")
    v1_text = json.dumps(json.loads(rec.to_json())["record"], indent=2, sort_keys=True)
    assert "schema" not in json.loads(v1_text)           # genuinely the old bare shape
    upgraded = DecisionRecord.from_json(v1_text)
    assert upgraded == rec
    assert replay(upgraded, vector_add(), AVX, COOL).reproduced


def test_an_unknown_schema_fails_loudly_never_misreads():
    import json
    rec = json.loads(explain(vector_add(), AVX, COOL, target_name="x86_avx512").to_json())
    rec["schema"] = 99                                   # a future revision this build lacks
    try:
        DecisionRecord.from_json(json.dumps(rec))
        raise AssertionError("a newer schema must be refused")
    except ValueError as e:
        assert "newer" in str(e)
    rec["schema"] = 2
    rec["kind"] = "not.a.record"                         # the wrong document kind
    try:
        DecisionRecord.from_json(json.dumps(rec))
        raise AssertionError("a wrong kind must be refused")
    except ValueError as e:
        assert "bcir.decision_record" in str(e)


def test_replay_reproduces_from_the_same_inputs():
    rec = explain(vector_add(), AVX, COOL, target_name="x86_avx512")
    rr = replay(rec, vector_add(), AVX, COOL)
    assert rr.reproduced and rr.mismatches == ()
    assert bool(rr) is True


def test_replay_detects_a_changed_commit():
    rec = explain(vector_add(), AVX, COOL, target_name="x86_avx512")
    # different target -> a different commit: the provenance digest gate catches it.
    rr = replay(rec, vector_add(), SVE, COOL)
    assert not rr.reproduced
    assert any("digest" in m for m in rr.mismatches)
    # different Theta likewise diverges.
    assert not replay(rec, vector_add(), AVX, HOT).reproduced


def test_joint_explain_carries_bundle_rewrite_certificates():
    rec = explain(matmul_tiled(), AVX, COOL, target_name="x86_avx512", joint=True)
    assert rec.certificates and all(c.kind == "bundle" and c.gain > 0 for c in rec.certificates)
    # the same certificates must replay identically.
    assert replay(rec, matmul_tiled(), AVX, COOL, joint=True).reproduced
    # ... and a non-joint replay of a joint record diverges on the certificates.
    assert not replay(rec, matmul_tiled(), AVX, COOL, joint=False).reproduced


def test_explain_text_is_human_readable():
    txt = explain_text(explain(vector_add(), AVX, COOL, target_name="x86_avx512"))
    assert "vec_add" in txt and "provenance digest" in txt and "claim 1000" in txt
    assert "vec16" in txt and "weighed:" in txt


def test_reduce_minimizes_to_a_legal_witness():
    # fused_chain has 2 claims; reduce to the minimal still-plannable module.
    red = reduce(fused_chain(), lambda m: any(ph.claims for ph in m.phases))
    assert sum(len(ph.claims) for ph in red.phases) >= 1
    assert verify(red) == []                       # the witness stays legal
    assert optimize(red, AVX, COOL).score > 0      # ... and plannable

    # multi_histogram reduces too, preserving the predicate (a gather claim present).
    def has_gather(m):
        return any(c.op == "histogram.scatter" for ph in m.phases for c in ph.claims)
    red2 = reduce(multi_histogram(), has_gather)
    assert has_gather(red2) and verify(red2) == []


def test_reduce_rejects_a_predicate_that_does_not_hold():
    try:
        reduce(vector_add(), lambda m: False)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_mlir_replay_recheck_is_in_sync():
    """Pins the constants -bcir-replay rechecks in mlir/test/passes/replay.mlir: the law rail
    diffs a fresh plan's per-claim (chosen width, edge score) + total against the declared
    kbcir.explain_* record. The faithful record (w16 / score 7808) reproduces; a tampered edge
    score (9999) is flagged with the exact (w16/7808) != (w16/9999) divergence."""
    from dataclasses import replace
    from bcir.kbcir.proof import ClaimDecision
    rec = explain(vector_add(), AVX, COOL, target_name="x86_avx512")
    assert rec.total_score == 7808
    (d,) = rec.decisions
    assert d.width == 16 and d.score == 7808              # the record -bcir-explain emits
    assert replay(rec, vector_add(), AVX, COOL).reproduced  # faithful -> reproduced (replay_reproduced=true)
    # Tamper the claim's recorded edge score: replay flags exactly that field (replay_mismatches).
    bad = replace(rec, decisions=(replace(d, score=9999),))
    rr = replay(bad, vector_add(), AVX, COOL)
    assert not rr.reproduced
    assert any("9999" in m for m in rr.mismatches)


def test_the_external_replay_cli_contract():
    """0.4b's second half: the CLI contract a third party re-checks records through --
    exit 0 reproduced / 3 diverged / 4 undecodable-or-mismatched, one machine-readable
    `replay-verdict:` line first, stable across builds (the record self-describes via the
    schema envelope). Driven as a real subprocess, the way a third party would."""
    import json
    import os
    import subprocess
    import sys
    import tempfile
    rec = explain(vector_add(), AVX, COOL, target_name="x86_avx512")
    env = {**os.environ, "PYTHONPATH": os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))}

    def cli(record_text, program="vector_add"):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rec.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write(record_text)
            r = subprocess.run([sys.executable, "-m", "bcir.run", program,
                                "--target", "x86_avx512", "--replay", path],
                               capture_output=True, text=True, env=env)
        verdicts = [ln for ln in r.stdout.splitlines() if ln.startswith("replay-verdict:")]
        assert len(verdicts) == 1, r.stdout + r.stderr
        return r.returncode, verdicts[0]

    rc, line = cli(rec.to_json())
    assert rc == 0 and "reproduced" in line, (rc, line)
    tampered = json.loads(rec.to_json())
    tampered["record"]["total_score"] = 1                   # a lying score must DIVERGE
    rc, line = cli(json.dumps(tampered))
    assert rc == 3 and "diverged" in line, (rc, line)
    rc, line = cli("{not json")
    assert rc == 4 and "undecodable" in line, (rc, line)
    rc, line = cli(rec.to_json(), program="matmul_tiled")   # wrong module: usage error,
    assert rc == 4 and "undecodable" in line, (rc, line)    # never a bogus divergence
