"""The two-truth separation (MOPC): classical truth v for the laws, graded
confidence w for everything else, quarantined apart (R13).

The discipline: a graded proposition may inform but never become a legality
verdict. These checks pin the wrapper, the sanctioned collapse, and the verifier
guard that no confidence-weighted value reaches the R-laws uncollapsed.
"""

from bcir.examples import vector_add
from bcir.kbcir import TARGETS
from bcir.kbcir.cost import Theta
from bcir.kbcir.regret import BoundaryVerdict
from bcir.kbcir.softdp import softselect
from bcir.kbcir.twotruth import (
    Decision,
    Graded,
    GradedTruthLeak,
    assert_classical,
    decide,
    from_regret,
    from_soft,
    g_and,
    g_not,
    g_or,
    is_classical,
)
from bcir.verify import verify_quarantine

AVX = TARGETS["x86_avx512"]
COOL = Theta.cool()


def _laws(diags):
    return {d.law for d in diags}


# --- the graded proposition (v, w) -----------------------------------------------

def test_graded_carries_a_value_and_a_confidence():
    g = Graded("vec16", 700, source="softdp")
    assert g.value == "vec16" and g.confidence_milli == 700 and g.confidence == 0.7
    assert not g.is_certain
    assert Graded("x", 1000).is_certain


def test_confidence_must_be_in_range():
    for bad in (-1, 1001):
        try:
            Graded("x", bad)
            assert False, "expected ValueError"
        except ValueError:
            pass


# --- the quarantine: graded is not classical -------------------------------------

def test_graded_is_not_classical_but_a_collapsed_value_is():
    g = Graded("vec16", 700)
    assert not is_classical(g)
    d = decide(g, 600)
    assert is_classical(d) and is_classical(d.value)      # the value crossed, classical
    assert is_classical(42) and is_classical("legal")


def test_decision_is_the_sanctioned_recorded_collapse():
    g = Graded("vec16", 700, source="softdp")
    assert decide(g, 600).admitted                        # confidence cleared threshold
    assert not decide(g, 800).admitted                    # did not clear -> not admitted
    d = decide(g, 600)
    assert d.value == "vec16" and d.confidence_milli == 700 and d.threshold_milli == 600


def test_assert_classical_is_the_runtime_guard():
    assert_classical("legal", 42, decide(Graded("v", 900), 500).value)   # no raise
    try:
        assert_classical(Graded("v", 900))
        assert False, "expected GradedTruthLeak"
    except GradedTruthLeak:
        pass


# --- the graded algebra (dynamic truth tables, on the graded side) ---------------

def test_graded_connectives_combine_confidence():
    a, b = Graded(True, 900), Graded(True, 800)
    assert g_and(a, b).confidence_milli == 720          # product t-norm 0.9*0.8
    assert g_or(a, b).confidence_milli == 980           # 0.9 + 0.8 - 0.72
    assert g_not(a).confidence_milli == 100             # 1 - 0.9
    assert g_and(a, b).value is True and g_not(a).value is False


def test_the_graded_algebra_never_yields_classical_truth():
    # Everything the graded truth-table algebra produces is still graded: it can
    # never *become* a verdict, only inform one.
    a, b = Graded(True, 900), Graded(False, 400)
    for g in (g_and(a, b), g_or(a, b), g_not(a)):
        assert not is_classical(g)


# --- R13: the verifier guard -----------------------------------------------------

def test_raw_graded_into_a_verdict_is_R13():
    assert "R13" in _laws(verify_quarantine(verdict_inputs=[Graded("legal", 700)]))


def test_a_collapsed_value_passes_the_quarantine():
    d = decide(Graded("vec16", 900, source="softdp"), 500)
    assert verify_quarantine(verdict_inputs=[d.value, 7, "legal"], decisions=[d]) == []


def test_malformed_collapse_is_R13():
    bad = Decision(value=1, confidence_milli=1500, threshold_milli=500)
    assert "R13" in _laws(verify_quarantine(decisions=[bad]))


def test_a_graded_verdict_is_R13():
    # A Diagnostic must be classical; a confidence-bearing "verdict" is a category
    # error the verifier must never emit.
    assert "R13" in _laws(verify_quarantine(diagnostics=[Graded("legal", 700)]))


def test_faithful_separation_satisfies_R13():
    assert verify_quarantine(verdict_inputs=["legal", 1, True]) == []


# --- the bridge to the real graded organs (softdp / regret) ----------------------

def test_softdp_posterior_wraps_as_a_graded_proposition():
    soft = softselect(vector_add(1024), AVX, COOL, temperature=1.0)
    cid = next(iter(soft.map_by_claim))
    g = from_soft(soft, cid)
    assert g.source == "softdp" and not is_classical(g)
    # routed raw into a verdict, it is quarantined; only its collapse may cross.
    assert "R13" in _laws(verify_quarantine(verdict_inputs=[g]))
    assert verify_quarantine(verdict_inputs=[decide(g, 500).value]) == []


def test_regret_verdict_wraps_with_an_evidence_confidence():
    v = BoundaryVerdict(rule="nominal", verdict="retune", regret_rate=0.3,
                        episodes=50, data_fit_nats=4.0, complexity_nats=1.0)
    g = from_regret(v)
    assert g.value == "retune" and g.source == "regret"
    assert g.confidence_milli == 600          # margin 3 / sum 5 -> 0.6
    assert not is_classical(g)
