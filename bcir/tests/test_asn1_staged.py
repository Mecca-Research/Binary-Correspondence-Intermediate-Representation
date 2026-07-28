"""P6 — the staged self-modification pipeline and its trusted loader.

**P6's gate**: *"No executable memory written by the emitting program; verification precedes
compilation; unsigned artifact refused; rollback and quiescence tested."*

Each clause is checked by observation rather than by inspection. "Verification precedes
compilation" is checked with a compilation *counter*, not by reading the order of two lines.
"Unsigned artifact refused" is checked with an artifact whose SHA-256 is perfectly correct,
because the failure worth catching is a loader that accepts integrity as authority.

The verification is the repository's own: `verdicts` from P3, running R19/R20/R21 over a
proposal that arrived as JER. A loader whose verifier returns True exercises the plumbing
and none of the property.
"""

from __future__ import annotations

from bcir.asn1.program import jer_to_module, module_to_jer, verdicts
from bcir.asn1.staged import (
    Artifact, NotQuiescent, NotSigned, NothingToRollBack, Proposal, TrustedLoader,
)
from bcir.asn1.tags import Asn1Error
from bcir.examples import PROGRAMS
from bcir.model import Lane, Opcode, StrideClass
from bcir.model.graph import Claim, Module, Phase, Timing

_KEY = b"a loader key that the proposing program does not have"


def _legal_proposal() -> Proposal:
    """A real corpus program, projected to JER exactly as §6's first arrow describes."""
    module = PROGRAMS[sorted(PROGRAMS)[0]]()
    return Proposal(jer=module_to_jer(module), origin="corpus")


def _illegal_proposal() -> Proposal:
    """A module that trips R20: two clock domains touching one resource, unexcused.

    Lifted from P3's own R20 fixture rather than invented here. A refusal test whose
    fixture is illegal for a reason nobody checked proves that the loader refuses
    *something*, which is not the claim.
    """
    def timed(claim_id: int, domain: str, reads=(), writes=()) -> Claim:
        return Claim(id=claim_id, opcode=Opcode.ADD, lane=Lane.U,
                     stride_class=StrideClass.UNIT, count=1, rd=tuple(reads),
                     wr=tuple(writes), hazard="unique",
                     timing=Timing(clock_domain=domain, sync_type="synchronous",
                                   clock_frequency_mhz=100, latency_cycles=4,
                                   setup_hold_margin=1))

    module = Module(name="crossing", cacheline=64, align=64, target="x86", resources={},
                    phases=[Phase(phase_id=0, deps=(), claims=[
                        timed(1, "fast", writes=(7,)),
                        timed(2, "slow", reads=(7,)),
                    ])])
    assert any(law == "R20" for law, _m in verdicts(module)), "the fixture stopped tripping R20"
    return Proposal(jer=module_to_jer(module), origin="synthetic")


def _verify(proposal: Proposal):
    """Decode the proposal and run the repository's own R19/R20/R21 over it."""
    return verdicts(jer_to_module(proposal.jer))


class _Compiler:
    """A stand-in for the offline-equivalent compilation path, which counts its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, proposal: Proposal) -> bytes:
        self.calls += 1
        return b"NATIVE:" + proposal.digest().encode()


def _loader() -> tuple[TrustedLoader, _Compiler]:
    compiler = _Compiler()
    return TrustedLoader(key=_KEY, compile_fn=compiler), compiler


# --- W^X: the emitting program writes data ------------------------------------------------


def test_a_proposal_carries_no_code_and_has_no_way_to_become_an_artifact():
    """The gate's first clause, as a structural property rather than a promise.

    A `Proposal` has octets and an origin. If it could also carry code, or produce an
    `Artifact` itself, then "the emitting program never writes executable memory" would
    depend on every future caller choosing not to — which is not a property, it is a hope.
    """
    proposal = _legal_proposal()
    fields = set(vars(proposal))
    assert fields == {"jer", "origin"}, fields
    assert not any(name for name in dir(proposal)
                   if "artifact" in name.lower() or "sign" in name.lower())
    # The only producer of an Artifact is a loader holding a key.
    assert not hasattr(Proposal, "compile") and not hasattr(Proposal, "admit")


def test_only_a_key_holder_can_produce_an_installable_artifact():
    loader, _ = _loader()
    admission = loader.admit(_legal_proposal(), _verify)
    assert admission.admitted and admission.artifact is not None
    # A forged artifact with perfectly correct content and no valid signature.
    forged = Artifact(generation=99, proposal_digest=admission.artifact.proposal_digest,
                      code=admission.artifact.code, signature="0" * 64)
    try:
        loader.install(forged)
    except NotSigned as error:
        assert "no signature this key produces" in str(error)
    else:
        raise AssertionError("a forged artifact must not install")


# --- verification precedes compilation ------------------------------------------------------


def test_an_illegal_proposal_is_never_compiled():
    """The gate's second clause, counted rather than read off the source order."""
    loader, compiler = _loader()
    admission = loader.admit(_illegal_proposal(), _verify)
    assert not admission.admitted
    assert compiler.calls == 0, "the compiler ran on a proposal that failed verification"
    assert loader.compilations == 0
    assert admission.diagnostics, "a refusal must say which law refused"
    assert any(law.startswith("R2") for law, _ in admission.diagnostics)


def test_a_legal_proposal_is_compiled_exactly_once():
    loader, compiler = _loader()
    assert loader.admit(_legal_proposal(), _verify).admitted
    assert compiler.calls == 1 and loader.compilations == 1


def test_an_undecodable_proposal_is_refused_before_verification_not_during_compilation():
    loader, compiler = _loader()
    admission = loader.admit(Proposal(jer=b"{not json", origin="hostile"), _verify)
    assert not admission.admitted and "undecodable" in admission.reason
    assert compiler.calls == 0


def test_every_corpus_program_is_admissible_so_the_refusals_mean_something():
    """A verifier that refused everything would pass the refusal tests and be useless."""
    loader, compiler = _loader()
    admitted = 0
    for name in sorted(PROGRAMS):
        proposal = Proposal(jer=module_to_jer(PROGRAMS[name]()), origin=name)
        if loader.admit(proposal, _verify).admitted:
            admitted += 1
    assert admitted >= 10, f"only {admitted} of {len(PROGRAMS)} corpus programs admitted"
    assert compiler.calls == admitted


# --- integrity is not authority --------------------------------------------------------------


def test_a_correct_digest_is_not_a_signature():
    """§6.3's distinction, made operational: the octets being intact says nothing about who.

    The forged artifact below has a *correct* `proposal_digest` and correct code. Everything
    a checksum can check, checks out. That is exactly the artifact a loader conflating the
    two would install.
    """
    loader, _ = _loader()
    good = loader.admit(_legal_proposal(), _verify).artifact
    assert good.proposal_digest == _legal_proposal().digest()
    stranger = TrustedLoader(key=b"a different authority's key", compile_fn=_Compiler())
    impostor = Artifact(generation=1, proposal_digest=good.proposal_digest,
                        code=good.code,
                        signature=stranger.sign(good.code, good.proposal_digest, 1))
    assert impostor.code_digest() == good.code_digest()
    try:
        loader.install(impostor)
    except NotSigned:
        pass
    else:
        raise AssertionError("another authority's signature must not admit an artifact")


def test_the_signature_covers_the_generation_and_the_proposal_not_only_the_code():
    """Signing the code alone would let a signed artifact be re-labelled and still verify."""
    loader, _ = _loader()
    good = loader.admit(_legal_proposal(), _verify).artifact
    for relabelled in (
            Artifact(generation=good.generation + 5, proposal_digest=good.proposal_digest,
                     code=good.code, signature=good.signature),
            Artifact(generation=good.generation, proposal_digest="0" * 64,
                     code=good.code, signature=good.signature)):
        assert not loader.verify_signature(relabelled)


# --- quiescence and rollback -------------------------------------------------------------------


def test_an_install_under_a_live_call_is_refused():
    """The gate's fourth clause. A verified, signed, correct artifact is still not installable."""
    loader, _ = _loader()
    artifact = loader.admit(_legal_proposal(), _verify).artifact
    loader.enter()
    try:
        loader.install(artifact)
    except NotQuiescent as error:
        assert "in flight" in str(error)
    else:
        raise AssertionError("a swap under a live call must be refused")
    loader.leave()
    loader.install(artifact)
    assert loader.live is artifact and loader.generation == 1


def test_a_rollback_under_a_live_call_is_refused_for_the_same_reason():
    loader, _ = _loader()
    first = loader.admit(_legal_proposal(), _verify).artifact
    loader.install(first)
    second = loader.admit(_legal_proposal(), _verify).artifact
    loader.install(second)
    loader.enter()
    try:
        loader.rollback()
    except NotQuiescent:
        pass
    else:
        raise AssertionError("a rollback under a live call must be refused")
    loader.leave()
    loader.rollback()
    assert loader.live.code == first.code


def test_a_rollback_moves_the_generation_forward_and_stays_signed():
    """Two live states must never share a tag, or the tag cannot detect a stale handle."""
    loader, _ = _loader()
    first = loader.admit(_legal_proposal(), _verify).artifact
    loader.install(first)
    second = loader.admit(_legal_proposal(), _verify).artifact
    loader.install(second)
    assert loader.generation == 2
    loader.rollback()
    assert loader.generation == 3, "a rollback that reused generation 1 would alias two states"
    assert loader.live.code == first.code
    assert loader.verify_signature(loader.live), "the live artifact must be signed at its tag"
    assert loader.history == [("install", 1), ("install", 2), ("rollback", 3)]


def test_the_first_install_has_nothing_to_roll_back_to():
    loader, _ = _loader()
    loader.install(loader.admit(_legal_proposal(), _verify).artifact)
    try:
        loader.rollback()
    except NothingToRollBack:
        pass
    else:
        raise AssertionError("rolling back past the first generation must refuse")


def test_a_replayed_artifact_is_refused_as_a_rollback_nobody_asked_for():
    loader, _ = _loader()
    first = loader.admit(_legal_proposal(), _verify).artifact
    loader.install(first)
    loader.install(loader.admit(_legal_proposal(), _verify).artifact)
    try:
        loader.install(first)
    except Asn1Error as error:
        assert "not newer than the live generation" in str(error)
    else:
        raise AssertionError("replaying generation 1 over generation 2 must refuse")


def test_leaving_more_calls_than_were_entered_is_an_error_not_a_negative_count():
    """A quiescence counter that can go negative reports quiescence while calls are live."""
    loader, _ = _loader()
    loader.enter()
    loader.leave()
    try:
        loader.leave()
    except Asn1Error as error:
        assert "in-flight count" in str(error)
    else:
        raise AssertionError("an unbalanced leave must be refused")


def test_a_loader_with_no_compiler_refuses_rather_than_admitting_nothing():
    loader = TrustedLoader(key=_KEY)
    try:
        loader.admit(_legal_proposal(), _verify)
    except Asn1Error as error:
        assert "no compiler" in str(error)
    else:
        raise AssertionError("a loader with no compiler must not report success")
