"""X.692 clauses 12 and 13: the encoding link module, and how encodings reach a class.

**The ELM is clause 12, not clause 14.** Clause 12's own NOTE draws the line, and it is worth
quoting because this repository got it wrong twice before reading it: "There are two top-level
productions in ECN, the `ELMDefinition` specified in this clause and the `EDMDefinition`
specified in clause 14." An EDM *defines* encoding classes, objects and structures; the ELM
*applies* them to ASN.1 types. §12.1.9 says so in one line — "the sole function of an ELM is to
apply encodings".

WHY THIS MODULE RETIRES TWO STATED DEVIATIONS. Until now `ecn_syntax.py` accepted an
`AUXILIARY` keyword on an `#INT` object and a `BOUNDS` clause on an `#INT` selector, both
marked in their own comments as deviations rather than notation X.692 gives. Both existed for
the same reason: the facts they carry live in the *link* between an ASN.1 type and an encoding
structure, and this rail had no link — it encoded against a value dictionary.

* **Auxiliary** is §22.1.2.6's category: fields "that are not part of the encoding class
  parameter". §19.3.1 says the same thing from clause 19's side — an encoding structure "has
  fields corresponding to the components of the type, **but also has added fields for
  determinants**". Given the type and the structure, which fields are auxiliary is a
  *computation*, not a declaration, and `LinkedStructure.auxiliary_fields` is that computation.
* **Bounds** are the type's, not the object's. §21.11.3 tests "the bounds on the integer values
  associated with an encoding class in the integer category", and §23.7.2.6's NOTE insists the
  condition is tested "on the bounds of the original value". `LinkedStructure.bounds_for` reads
  them off the ASN.1 component's own constraint.

Both remain *accepted* at the surface, because an ECN specification written against this rail
may still have no ASN.1 type in hand — but they are now the fallback rather than the only
source, and a link supersedes them.

§13.2.10 IS AN ALGORITHM, AND IT IS SHORT. Everything about "which encoding object encodes
this class" reduces to §13.2.10.1: if the combined set has an object of the same class, apply
it; otherwise de-reference the class and recurse; and §13.2.10.8, "Otherwise the ECN
specification is in error". `resolve` is that, and the de-referencing is what makes clause 11's
`#Version ::= #INT` mean anything — an object for `#INT` encodes a `#Version` when no object
names `#Version` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ecn_props import IntegerBounds
from .tags import Asn1Error


@dataclass(frozen=True)
class EncodingApplication:
    """§12.2.1's `ENCODE <class>,+ WITH <primary> [COMPLETED BY <secondary>]`.

    §12.2.2: it "defines the encoding of the ASN.1 types corresponding to the
    `SimpleDefinedEncodingClass`es which shall be generated encoding structures".

    §12.2.3 is the sentence that keeps an ELM from having action at a distance, and it is
    easy to miss: "Encodings applied to a generated encoding structure corresponding to an
    ASN.1 type … are linked **solely to the use of that type as application messages**. They
    have no implications on the encoding of that type when referenced by other types or when
    exported … and imported into a different ASN.1 module." So an application names top-level
    messages, and encoding `Foo` here says nothing about a `Foo` nested inside `Bar`.
    """

    #: §12.2.1's `SimpleDefinedEncodingClass ","+` — one or more, never zero.
    classes: tuple = ()
    #: §13.1.3's `PrimaryEncodings`: class name -> encoding object.
    primary: dict = field(default_factory=dict)
    #: §13.1.3's `COMPLETED BY SecondaryEncodings`. `None` is §13.2.2's "no CompletionClause".
    secondary: "dict | None" = None

    def __post_init__(self) -> None:
        if not self.classes:
            raise Asn1Error(
                "ECN: §12.2.1 gives `ENCODE` one or more encoding classes; an application "
                "that names none applies encodings to nothing"
            )
        if len(set(self.classes)) != len(self.classes):
            raise Asn1Error(
                f'ECN: §12.2.5 — an ELM "shall not apply encodings more than once to the '
                f'same ASN.1 type"; {sorted(self.classes)} repeats one within a single '
                f"application"
            )

    def combined(self) -> dict:
        """§13.2.2 and §13.2.3's combined encoding object set.

        §13.2.3 b) states the merge exactly: every object in the secondary set "is added to
        the combined encoding object set **if (and only if) there is no encoding object
        already in the combined encoding object set that has the same encoding class**". So
        the primary wins and `COMPLETED BY` fills gaps — the same left-biased rule §9.23.2
        gives and §22.11.1.4 defers to, which is why `COMPLETED BY PER-BASIC-UNALIGNED` is
        safe to write under a handful of specialized objects.
        """
        out = dict(self.primary)
        if self.secondary:
            for cls, obj in self.secondary.items():
                out.setdefault(cls, obj)
        return out


@dataclass(frozen=True)
class ElmModule:
    """§12.1.1's `ELMDefinition`: `<name> LINK-DEFINITIONS ::= BEGIN <body> END`.

    §12.1.2 is the constraint that makes an ELM a *singular* thing: "In any given application
    of ECN, there shall be precisely one ELM which determines the encoding of all the messages
    used in that application." One module cannot check that — it is a property of a whole
    application — so what is enforced here is §12.2.5's within-ELM half.

    §12.1.7's import rule is stronger than ASN.1's and is enforced when `imports` is stated:
    "All reference names used in the `ELMModuleBody` shall be imported into the ELM." Its NOTE
    explains the difference: in ASN.1 "external references can be used for types and values
    that have not been imported", and in an ELM they cannot, because "the purpose of external
    references is solely to resolve ambiguities between imported names and built-in names".
    """

    name: str = ""
    applications: tuple = ()
    #: §12.1.6's `Imports`. Empty means "not stated", which switches §12.1.7's check off
    #: rather than asserting that nothing is imported — a module with no imports clause and a
    #: module importing nothing are different statements, and only the second is checkable.
    imports: tuple = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise Asn1Error("ECN: §12.1.3 gives an ELM a ModuleIdentifier")
        if not self.applications:
            raise Asn1Error(
                'ECN: §12.1.9 — the `EncodingApplicationList` "is required to contain at '
                "least one `EncodingApplication`, as the sole function of an ELM is to apply "
                'encodings"'
            )
        seen: dict = {}
        for index, application in enumerate(self.applications):
            for cls in application.classes:
                if cls in seen:
                    raise Asn1Error(
                        f'ECN: §12.2.5 — an ELM "shall not apply encodings more than once to '
                        f'the same ASN.1 type"; {cls!r} is applied by applications '
                        f"{seen[cls]} and {index}"
                    )
                seen[cls] = index
        if self.imports:
            allowed = set(self.imports)
            for application in self.applications:
                for cls in application.classes:
                    if cls not in allowed:
                        raise Asn1Error(
                            f'ECN: §12.1.7 — "All reference names used in the ELMModuleBody '
                            f'shall be imported into the ELM"; {cls!r} is not among '
                            f'{sorted(allowed)}. Its NOTE calls this "a stronger requirement '
                            f'than that imposed for ASN.1 modules"'
                        )

    def set_for(self, class_name: str) -> dict:
        """The combined encoding object set this ELM applies to a top-level class."""
        for application in self.applications:
            if class_name in application.classes:
                return application.combined()
        raise Asn1Error(
            f"ECN: this ELM applies no encodings to {class_name!r}; §12.2.2 makes an "
            f"`EncodingApplication` the only thing that binds a class to an object set"
        )


def resolve(objects: dict, class_name: str, assignments: "dict | None" = None):
    """§13.2.10.1, §13.2.10.7 and §13.2.10.8: which object encodes the class at this point.

    The algorithm in full, and it is three lines of clause:

    * §13.2.10.1 — "If the combined encoding object set contains an encoding object of the
      same class … as the current application point, then that encoding object is applied."
    * §13.2.10.1 a) / §13.2.10.7 — otherwise, if the class "is a reference to another encoding
      class … it is de-referenced, and the procedures of 13.2.10 are recursively applied".
    * §13.2.10.8 — "Otherwise the ECN specification is in error."

    The de-referencing is what makes clause 11's `#Version ::= #INT` mean something: an object
    written for `#INT` encodes a `#Version` when no object names `#Version` directly, and that
    is exactly how one object set covers a module full of assigned classes. §9.5.2's
    one-object-per-class rule is what makes the first step unambiguous.

    `assignments` maps a class name to the class it is assigned from. A cycle is refused rather
    than followed, for the reason R25 gives on the law rail: a class that is its own base names
    no category, so no object could realize it.
    """
    assignments = assignments or {}
    current = class_name
    seen: list = []
    while True:
        if current in objects:
            return objects[current]
        if current in seen:
            raise Asn1Error(
                f"ECN: the class assignment chain from {class_name!r} is circular "
                f"({' -> '.join(seen + [current])}); a class that is its own base names no "
                f"encoding category, so no object could realize it"
            )
        seen.append(current)
        if current not in assignments:
            # §13.2.10.8. Named as the specification error it is, with the chain that was
            # tried, because "no encoding object" and "no such class" are different faults and
            # a bare failure would not say which.
            tried = " -> ".join(seen)
            raise Asn1Error(
                f"ECN: §13.2.10.8 — the ECN specification is in error: no encoding object in "
                f"the combined set has the class {class_name!r}, and de-referencing reached "
                f"{current!r} which no assignment defines (tried {tried})"
            )
        current = assignments[current]


@dataclass(frozen=True)
class LinkedStructure:
    """The join of an ASN.1 type and an encoding structure — what the two deviations needed.

    §9.24.2 names what this is: "The combined encoding object set is applied to a generated
    encoding structure, and it is the encodings defined for the abstract values of this
    encoding structure that encode the abstract values of the ASN.1 type." So there are two
    field lists — the type's components and the structure's fields — and every fact this class
    computes comes from comparing them.
    """

    #: The ASN.1 type this structure encodes. A `Sequence`, in the case that matters.
    asn1_type: object = None
    #: The encoding structure's field names, in its own textual order (§16.5).
    fields: tuple = ()

    def component_names(self) -> tuple:
        """The ASN.1 type's own component names, which the structure's fields are matched to."""
        components = getattr(self.asn1_type, "components", None)
        if components is None:
            raise Asn1Error(
                f"ECN: a linked structure compares an encoding structure's fields with the "
                f"ASN.1 type's components; {type(self.asn1_type).__name__} has none"
            )
        return tuple(component.name for component in components)

    def auxiliary_fields(self) -> tuple:
        """§22.1.2.6's auxiliary fields, computed rather than declared.

        §22.1.2.6 defines them as fields "that are not part of the encoding class parameter",
        and §19.3.1 gives the same set from clause 19's side: a structure "has fields
        corresponding to the components of the type, **but also has added fields for
        determinants**". A field with no component behind it is one of those.

        This is what retires the `AUXILIARY` keyword `ecn_syntax.py` accepts as a stated
        deviation. The keyword stays — an ECN specification written against this rail may have
        no ASN.1 type in hand — but where there *is* a link, the answer is derived and the
        declaration is redundant rather than authoritative.
        """
        components = set(self.component_names())
        return tuple(name for name in self.fields if name not in components)

    def missing_fields(self) -> tuple:
        """Components the structure has no field for — §9.24.2's mapping with a hole in it."""
        present = set(self.fields)
        return tuple(name for name in self.component_names() if name not in present)

    def bounds_for(self, component_name: str) -> IntegerBounds:
        """§21.11's bounds, read off the ASN.1 component rather than declared on the object.

        §21.11.3 tests "the bounds on the integer values associated with an encoding class in
        the integer category", and §23.7.2.6's NOTE is emphatic that the condition is tested
        "on the bounds of the original value, and is not affected by these transforms". Both
        say the bounds belong to the *type*.

        This is what retires the `BOUNDS` clause. An absent constraint gives
        `IntegerBounds()` — no bounds — which §21.11.4 a)'s `unbounded-or-no-lower-bound`
        is precisely the predicate for, so "unconstrained" is an answer here and not a gap.
        """
        for component in getattr(self.asn1_type, "components", ()):
            if component.name != component_name:
                continue
            constraint = getattr(component.type, "constraint", None)
            if constraint is None:
                return IntegerBounds()
            bounds = getattr(constraint, "value_bounds", None)
            if bounds is None:
                return IntegerBounds()
            low, high = bounds()
            return IntegerBounds(low, high)
        raise Asn1Error(
            f"ECN: {component_name!r} is not a component of this ASN.1 type "
            f"({', '.join(self.component_names())})"
        )

    def check(self) -> None:
        """Every component reachable, and every extra field accounted for as auxiliary.

        §9.24.2's mapping has to be total in one direction: a component with no field would be
        an abstract value with nowhere to go. The other direction is *expected* to be partial —
        that is what auxiliary fields are — so only the first is a fault.
        """
        missing = self.missing_fields()
        if missing:
            raise Asn1Error(
                f"ECN: the encoding structure has no field for the component"
                f"{'s' if len(missing) > 1 else ''} {', '.join(repr(n) for n in missing)}; "
                f"§9.24.2 makes the structure's encodings the encodings of the type's abstract "
                f"values, so a component with no field has nowhere to be encoded"
            )


__all__ = ["ElmModule", "EncodingApplication", "LinkedStructure", "resolve"]
