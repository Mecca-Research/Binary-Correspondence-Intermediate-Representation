"""X.692 §17.5: the `EncodeStructure`, which is how an encoding object encodes a construction.

Clause 17's other object forms say how one class becomes bits. `EncodeStructure` is the one
that says **which object encodes which field**, component by component::

    { ENCODE STRUCTURE { <identifier> <object-or-USE-SET>, ... [STRUCTURED WITH ...] }
      [WITH <object set>] }

That distinction is why this module exists at all. `ecn_user.ReplacementStructure` has always
needed an `auxiliary` spec per field and a `determinant` for the instantiated one, and §22.1
never supplies them — §22.1.3.5 only says the values "shall be set according to the
specification in the **replacement structure encoding object**", which is this production. The
repository recorded the missing piece as §22.1.2.6 until D.3.2.3's worked replacement was read;
§22.1.2.6 *classifies* the auxiliary fields and never says how they are encoded, so it could
not have been the answer.

**`CombinedEncodings` — the trailing `WITH <object set>` — is required by three different
clauses for three different reasons**, and an implementation that checks one will accept
specifications the other two forbid:

* **§17.5.3** — there is no `STRUCTURED WITH`, so nothing else encodes the constructor itself.
  Its NOTE gives the reason in five words: "a complete encoding has to be produced".
* **§17.5.6** — some `EncodingOrUseSet` is `USE-SET`, which *means* "apply the
  `CombinedEncodings`", so the phrase is a dangling reference without them. §17.5.13 and
  §17.5.14 repeat the requirement for the two nested positions.
* **§17.5.10** — some component has no `ComponentEncoding` at all, and the set is what has to
  "provide a complete encoding of that component".

They are checked as three, with three messages, because they are three different repairs: add
a `STRUCTURED WITH`, drop a `USE-SET`, or write the missing component in.

Two rules are **biconditionals**, the same shape §22.1.2.5 uses for `INSERT AT HEAD`:
§17.5.9's optional-component spec is used "if and only if the component is optional", and
§17.5.11's identifier is omitted "if and only if" the governor is a repetition class with no
identifier on its element. Both directions of both are faults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .tags import Asn1Error

#: §17.5.1's `EncodingOrUseSet ::= EncodingObject | USE-SET`. The keyword is not a name, so it
#: is this sentinel rather than the string `"USE-SET"` — a module is free to define an encoding
#: object called `USE-SET`, and the two must not collide.
USE_SET = "\x00USE-SET"


class GovernorCategory(Enum):
    """§17.5.2's admissible governors.

    "`EncodeStructure` can be used to define an encoding only if the governing encoding class
    de-references to a construction defined using an encoding constructor in the alternatives,
    concatenation, or repetition categories, or to a construction defined using one of these
    categories preceded by a class in the tag category."

    So the tag class is a *prefix*, not a fourth category — which is why it is a separate flag
    on `EncodeStructure` rather than a member here.
    """

    ALTERNATIVES = "alternatives"
    CONCATENATION = "concatenation"
    REPETITION = "repetition"


@dataclass(frozen=True)
class ComponentEncoding:
    """§17.5.7's `ComponentEncoding`, in either of its two forms.

    §17.5.10's `NonOptionalComponentEncodingSpec` and `OptionalComponentEncodingSpec` differ in
    exactly one thing — the trailing `OPTIONAL-ENCODING OptionalEncoding` — and §17.5.9 makes
    which one is legal a property of the *component*, not a choice: it "shall be used if and
    only if the component is optional".

    `tag` is §17.5.10's `TagEncoding`, the `[ ... ]` before the element's own encoding.
    §17.5.12 says the pair "shall provide a complete encoding for the component (including any
    class in the tag category that is prefixed to the element, but **excluding** any class in
    the optionality category that follows the element)" — the exclusion is what makes
    `optional_encoding` a third slot rather than something `element` could cover.
    """

    #: §17.5.11's identifier. Empty is legal only for an unnamed repetition element.
    identifier: str = ""
    #: The element's own encoding: an object reference, or `USE_SET`.
    element: str = USE_SET
    #: §17.5.10's `TagEncoding`, or `None` when the component has no tag class prefixed.
    tag: str | None = None
    #: §17.5.14's `OptionalEncoding`, present exactly when the component is optional.
    optional_encoding: str | None = None
    #: C.4 actual parameters supplied to `element`, as D.3.2.3's
    #: `if-component-present-encoding{<determinant>}` supplies them.
    actuals: tuple = ()

    def uses_set(self) -> bool:
        """Whether any position in this component says `USE-SET` (§17.5.13, §17.5.14)."""
        return USE_SET in (self.element, self.tag, self.optional_encoding)


@dataclass(frozen=True)
class EncodeStructure:
    """§17.5.1's production, with §17.5.2 to §17.5.14's restrictions checked on construction.

    `components` is §17.5.7's `ComponentEncodingList`, which may be **empty** — `"," *` is
    zero-or-more, unlike C.1's `"," +`. An empty list is not a degenerate case: §17.5.10 then
    requires `CombinedEncodings` to encode every component, which is a complete and useful
    specification meaning "encode all of this with that object set".

    `structure_encoding` is §17.5.1's `STRUCTURED WITH [TagEncoding] EncodingOrUseSet`, which
    encodes the constructor itself rather than any component.
    """

    #: §17.5.2's governing constructor category.
    governor: GovernorCategory
    #: The governing constructor's components, in textual order. `()` for a repetition class
    #: whose element has no identifier.
    component_names: tuple = ()
    #: Which of `component_names` are optional (§17.5.9's condition).
    optional_names: frozenset = frozenset()
    components: tuple = ()
    #: §17.5.1's `StructureEncoding`, or `None`.
    structure_encoding: str | None = None
    #: §17.5.3's `CombinedEncodings` — the trailing `WITH <object set>`.
    combined: str = ""
    #: §17.5.2's "preceded by a class in the tag category".
    tagged: bool = False
    #: §17.5.11's escape hatch: a repetition governor whose element carries no identifier.
    unnamed_element: bool = False

    def __post_init__(self) -> None:
        self._check_components()
        self._check_combined_is_present()
        self._check_optionality()

    # --- §17.5.8 and §17.5.11: which components may appear, and in what order --------------

    def _check_components(self) -> None:
        if self.unnamed_element and self.governor is not GovernorCategory.REPETITION:
            raise Asn1Error(
                f"ECN: §17.5.11 — an identifier is omitted if and only if the governing "
                f"encoding constructor is a class in the repetition category with no "
                f"identifier on the repeated element; this governor is "
                f"{self.governor.value}")
        seen: list[str] = []
        for component in self.components:
            if not component.identifier:
                if not self.unnamed_element:
                    raise Asn1Error(
                        "ECN: §17.5.11 — the identifier shall be omitted if and only if the "
                        "governing encoding constructor is a repetition class with no "
                        "identifier on the repeated element, and this one has identifiers")
                continue
            if self.unnamed_element:
                raise Asn1Error(
                    f"ECN: §17.5.11 — {component.identifier!r} names a component of a "
                    f"repetition whose element has no identifier; the biconditional runs both "
                    f"ways, so a name here is as much a fault as a missing one there")
            if component.identifier not in self.component_names:
                raise Asn1Error(
                    f"ECN: §17.5.11 — the identifier shall be that of a component of the "
                    f"governing encoding constructor; {component.identifier!r} is not one of "
                    f"{list(self.component_names)}")
            if component.identifier in seen:
                raise Asn1Error(
                    f"ECN: §17.5.8 — there shall be at most one ComponentEncoding for each "
                    f"component; {component.identifier!r} has two")
            seen.append(component.identifier)
        # §17.5.8's second sentence: "The `ComponentEncoding`s shall be in the same textual
        # order." A SUBSET is allowed (§17.5.10 covers the rest), so this is a subsequence
        # test rather than a prefix or an equality one — the weaker check is the correct one.
        order = [name for name in self.component_names if name in set(seen)]
        if seen != order:
            raise Asn1Error(
                f"ECN: §17.5.8 — the ComponentEncodings shall be in the same textual order as "
                f"the components; {seen} against {order}")

    # --- the three independent reasons CombinedEncodings must be present -------------------

    def _check_combined_is_present(self) -> None:
        if self.structure_encoding is None and not self.combined:
            raise Asn1Error(
                "ECN: §17.5.3 — if the StructureEncoding is absent, the CombinedEncodings "
                "shall be present; its NOTE gives the reason, that \"a complete encoding has "
                "to be produced\"")
        if not self.combined:
            using = [component.identifier or "<element>" for component in self.components
                     if component.uses_set()]
            if using:
                raise Asn1Error(
                    f"ECN: §17.5.6 — USE-SET means the encoding \"is obtained by applying the "
                    f"CombinedEncodings, which shall be present\", and there are none; "
                    f"{using} say USE-SET")
            if self.structure_encoding == USE_SET:
                raise Asn1Error(
                    "ECN: §17.5.6 — the StructureEncoding says USE-SET and there are no "
                    "CombinedEncodings for it to apply")
            missing = self.missing_components()
            if missing:
                raise Asn1Error(
                    f"ECN: §17.5.10 — a component with no ComponentEncoding is encoded by the "
                    f"CombinedEncodings, which \"shall be present\" and provide \"a complete "
                    f"encoding of that component\"; {list(missing)} have none")

    def missing_components(self) -> tuple:
        """§17.5.10's components: named by the constructor, absent from the list."""
        if self.unnamed_element:
            return ()
        written = {component.identifier for component in self.components}
        return tuple(name for name in self.component_names if name not in written)

    # --- §17.5.9's biconditional ------------------------------------------------------------

    def _check_optionality(self) -> None:
        for component in self.components:
            if not component.identifier:
                continue
            optional = component.identifier in self.optional_names
            written = component.optional_encoding is not None
            if optional and not written:
                raise Asn1Error(
                    f"ECN: §17.5.9 — the OptionalComponentEncodingSpec shall be used if and "
                    f"only if the component is optional, and {component.identifier!r} is "
                    f"optional with no OPTIONAL-ENCODING")
            if written and not optional:
                raise Asn1Error(
                    f"ECN: §17.5.9 — {component.identifier!r} carries an OPTIONAL-ENCODING and "
                    f"is not optional; §17.5.14 makes that clause encode \"the class in the "
                    f"optionality category of the component\", and there is none")

    # --- what the replacement machinery actually wants out of this -------------------------

    def encoding_for(self, component_name: str) -> str:
        """The object encoding one component, resolving §17.5.6's `USE-SET` to the set.

        This is the function §22.1.3.5's "set according to the specification in the replacement
        structure encoding object" names, and the reason this module is what `REPLACE` was
        waiting on: it turns a replacement structure's field list into one object per field,
        which is exactly `ecn_user.ReplacementStructure`'s `auxiliary` mapping.
        """
        for component in self.components:
            if component.identifier != component_name:
                continue
            return self.combined if component.element == USE_SET else component.element
        if component_name in self.component_names:
            # §17.5.10 already guaranteed the set is present and complete for this component.
            return self.combined
        raise Asn1Error(
            f"ECN: {component_name!r} is not a component of the governing encoding constructor")

    def replacement_actions_allowed(self) -> bool:
        """§17.5.4: with a non-empty `ComponentEncodingList`, the object applied to the
        governing constructor "shall not specify any replacement actions".

        Reported rather than refused, because the object it constrains is named here and
        *defined* elsewhere — the fault belongs to whoever pairs the two, and §22.1's group is
        what would have to be inspected to see it.
        """
        return not self.components


__all__ = [
    "USE_SET", "ComponentEncoding", "EncodeStructure", "GovernorCategory",
]
