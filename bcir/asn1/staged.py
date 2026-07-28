"""P6 — §6's staged self-modification pipeline, and the trusted loader it turns on.

P6's gate: *"No executable memory written by the emitting program; verification precedes
compilation; unsigned artifact refused; rollback and quiescence tested."*

§6's own summary is the design: *"a program can safely propose new logic at runtime. It
cannot safely admit it without a trusted loader, and no schema changes that."* The pipeline::

    running program
      -> emits a candidate subtree as canonical JER      (data, not code)
      -> schema validation + R1-R24 verification         (legality, not cost)
      -> K_BCIR selection over the candidate set         (cost, gated by legality)
      -> compilation to a native artifact                (offline-equivalent path)
      -> signing + trusted-loader admission              (authority)
      -> generation-tagged swap, quiescence, rollback    (liveness)

Four separations do the work, and each is a *type* here rather than a convention, because a
discipline that lives only in review comments is one refactor from being gone:

- **W^X.** A `Proposal` carries octets and an origin. There is no path from a proposal to an
  `Artifact` except through `TrustedLoader.admit`, and the loader is the only holder of the
  key. The emitting program cannot make an artifact by being careful; it cannot make one at
  all.
- **Verification precedes compilation.** `admit` verifies first and *counts* compilations, so
  "a subtree that fails R1–R24 is never compiled" is checked by observation rather than by
  reading the order of two statements.
- **Integrity is not authority.** A digest says the octets are the ones that were verified;
  anybody can compute it. A signature says an authority admitted them. The JSON roadmap §6.3
  already insists on the distinction — *"BCAB CRCs and SHA-256 fields detect corruption and
  bind content; they are not signatures"* — and self-modification is where conflating them
  would be fatal.
- **Liveness is not correctness.** A verified, signed, correct artifact installed while calls
  are in flight is still a bug. Quiescence and generation tags are the driver roadmap's
  existing requirements, and nothing about self-modification makes them cheaper.

**The verification is real.** `verdicts` from P3 runs the repository's own R19/R20/R21 over
the proposed module — not a stub returning True. A loader whose "verification" is a
placeholder tests the plumbing and none of the property.

**A stated limitation.** The signature here is HMAC-SHA256, which is symmetric: it proves
possession of the loader's key, not the identity of a signer. That is enough to separate
authority from integrity — the point P6's gate is about — and it is *not* enough for a real
deployment, where the proposer and the admitting authority are different principals and the
key cannot live on both sides. Saying so is cheaper than a test suite that passes and a
threat model that does not.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field

from .tags import Asn1Error


class NotVerified(Asn1Error):
    """A proposal failed R1–R24 and is therefore never compiled."""


class NotSigned(Asn1Error):
    """An artifact carries no signature, or one this loader's key does not produce."""


class NotQuiescent(Asn1Error):
    """A swap was attempted with calls still in flight.

    Its own class because the response is to wait, not to retry harder and not to fix the
    artifact — which is what a generic error would leave a caller guessing.
    """


class NothingToRollBack(Asn1Error):
    """Rollback was asked for with no previous generation to return to."""


@dataclass(frozen=True)
class Proposal:
    """What a running program is allowed to produce: **data**.

    There is deliberately no `code` field and no method that produces one. The emitting
    program's whole output is octets and a claim about where they came from.
    """

    jer: bytes
    origin: str

    def digest(self) -> str:
        return hashlib.sha256(self.jer).hexdigest()


@dataclass(frozen=True)
class Artifact:
    """A compiled, signed, generation-tagged unit. Only a loader can make one.

    `digest` binds the artifact to the octets that were verified, so an artifact cannot be
    re-pointed at a different proposal after the fact. `signature` is separate and keyed —
    see the module docstring on why the two are not one field.
    """

    generation: int
    proposal_digest: str
    code: bytes
    signature: str

    def code_digest(self) -> str:
        return hashlib.sha256(self.code).hexdigest()


@dataclass(frozen=True)
class Admission:
    """The record of one admission decision, refusals included.

    Returned rather than logged: a caller that has to parse a log to find out whether its
    proposal was admitted will eventually stop checking.
    """

    admitted: bool
    generation: int
    reason: str = ""
    diagnostics: tuple[tuple[str, str], ...] = ()
    artifact: Artifact | None = None


@dataclass
class TrustedLoader:
    """The authority. Holds the key, counts compilations, and owns the live generation."""

    key: bytes
    #: Compiles a verified module into a native artifact. Injected so a test can observe
    #: *whether* it ran, which is how "verification precedes compilation" is checked.
    compile_fn: object = None
    generation: int = 0
    live: Artifact | None = None
    previous: Artifact | None = None
    in_flight: int = 0
    compilations: int = 0
    history: list[tuple[str, int]] = field(default_factory=list)

    # --- authority ---------------------------------------------------------------------

    def sign(self, artifact_code: bytes, proposal_digest: str, generation: int) -> str:
        """HMAC over everything that identifies the artifact, not over the code alone.

        Signing only the code would let a signed artifact be re-labelled with another
        proposal's digest or another generation — the signature would still verify and it
        would be attesting to something nobody admitted.
        """
        message = b"|".join((str(generation).encode(), proposal_digest.encode(),
                             artifact_code))
        return hmac.new(self.key, message, hashlib.sha256).hexdigest()

    def verify_signature(self, artifact: Artifact) -> bool:
        expected = self.sign(artifact.code, artifact.proposal_digest, artifact.generation)
        # Constant-time: a signature check that leaks its answer through timing is a check
        # an attacker completes one octet at a time.
        return hmac.compare_digest(expected, artifact.signature)

    # --- admission ---------------------------------------------------------------------

    def admit(self, proposal: Proposal, verify) -> Admission:
        """Verify, then compile, then sign. In that order, and observably so.

        `verify` returns the diagnostics for the proposed module — P3's `verdicts` over
        R19/R20/R21 in the tests, which is the repository's real answer rather than a stub.
        A non-empty result means the proposal is illegal and **`compile_fn` is never
        called**, which is the half of the gate that is easy to claim and easy to get wrong.
        """
        try:
            diagnostics = tuple(verify(proposal))
        except Asn1Error as error:
            # A proposal that will not even decode is refused here, not at compile time.
            return Admission(False, self.generation, f"undecodable proposal: {error}")
        if diagnostics:
            return Admission(False, self.generation,
                             "verification failed; nothing was compiled",
                             diagnostics=diagnostics)
        if self.compile_fn is None:
            raise Asn1Error("a loader with no compiler cannot admit anything")
        self.compilations += 1
        code = self.compile_fn(proposal)
        generation = self.generation + 1
        digest = proposal.digest()
        return Admission(True, generation, artifact=Artifact(
            generation=generation, proposal_digest=digest, code=code,
            signature=self.sign(code, digest, generation)))

    # --- liveness ----------------------------------------------------------------------

    def enter(self) -> None:
        """A call begins. Quiescence is the absence of these, counted rather than assumed."""
        self.in_flight += 1

    def leave(self) -> None:
        if self.in_flight == 0:
            raise Asn1Error("more calls left than entered; the in-flight count is wrong")
        self.in_flight -= 1

    def install(self, artifact: Artifact) -> None:
        """Admit a signed artifact into service. Refuses unsigned, stale, or busy.

        The signature is checked **here** and not only at admission, because an artifact can
        travel — persisted, shipped, replayed — between the two, and a loader that trusted
        its own earlier decision would be trusting whatever wrote the file since.
        """
        if not self.verify_signature(artifact):
            raise NotSigned(
                f"generation {artifact.generation} carries no signature this key produces; "
                f"a matching SHA-256 would prove the octets are intact and says nothing "
                f"about who admitted them")
        if artifact.generation <= self.generation:
            raise Asn1Error(
                f"generation {artifact.generation} is not newer than the live generation "
                f"{self.generation}; a replayed artifact is a rollback nobody asked for")
        if self.in_flight:
            raise NotQuiescent(
                f"{self.in_flight} call(s) still in flight; a swap now would move the "
                f"ground under a running call, which no amount of verification prevents")
        self.previous, self.live = self.live, artifact
        self.generation = artifact.generation
        self.history.append(("install", artifact.generation))

    def rollback(self) -> None:
        """Return to the previous generation. Also generation-tagged, and also quiesced.

        The generation moves FORWARD on a rollback. Reusing the old number would make two
        different live states share a tag, and a tag that does not identify a state cannot
        be used to decide whether a caller is holding a stale handle.
        """
        if self.previous is None:
            raise NothingToRollBack(
                "there is no previous generation; the first install has nothing behind it")
        if self.in_flight:
            raise NotQuiescent(
                f"{self.in_flight} call(s) still in flight; rolling back under a live call "
                f"has the same hazard as installing under one")
        restored = self.previous
        self.generation += 1
        # Re-signed at its new generation, so the live artifact's signature always covers
        # the generation it is actually live at.
        self.live = Artifact(
            generation=self.generation, proposal_digest=restored.proposal_digest,
            code=restored.code,
            signature=self.sign(restored.code, restored.proposal_digest, self.generation))
        self.previous = None
        self.history.append(("rollback", self.generation))


__all__ = [
    "Admission", "Artifact", "NotQuiescent", "NotSigned", "NotVerified",
    "NothingToRollBack", "Proposal", "TrustedLoader",
]
