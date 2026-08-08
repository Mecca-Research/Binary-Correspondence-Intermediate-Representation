"""X.692 Annex C: X.683's parameterization, as ECN modifies it.

Annex C "forms an integral part of this Recommendation", and it does not merely *reference*
X.683 — it **rewrites** productions from it. The rewrite that matters most is two characters
wide::

    X.683 8.3   ParameterList ::= "{"  Parameter "," + "}"
    X.692 C.1   ParameterList ::= "{<" Parameter "," + ">}"

Same for the actual list (C.4). An implementation that reaches for the X.683 parser it
already has will accept `#Length-prefixed{#D}` and reject `#Length-prefixed{<#D>}`, which is
the only spelling ECN admits — and it will do so while citing X.683 correctly. This
repository made exactly that mistake in prose: `ecn_syntax.py` described the shape it could
not read as `` `#Length-prefixed{#D} ::= ...` ``, in ASN.1's braces. Corrected here and there.

The delimiters are not cosmetic. `{` already opens an encoding object body, a structure's
field list and a `Size`; `{<` is unambiguous at one token of lookahead, which is what lets
§17.1.3's NOTE 2 hold — "the syntax of the governed notation has been designed so that a
parser can find the end of it without knowledge of the governor".

**What this module is not.** It is the parameterization *model*: what a dummy may stand for,
what governor each kind requires, which actual fits, and what §22.1 demands of the structures
and objects a `REPLACE` names. Reading `{< >}` out of module text is the surface layer's job
and lives in [`ecn_syntax.py`](ecn_syntax.py); the replacement *semantics* — instantiating a
structure around a field, hoisting head-end insertions — were already built in
[`ecn_user.py`](ecn_user.py). This is the layer between them, and it is the layer §22.1.2.2
and §22.1.2.4 are written in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .tags import Asn1Error

# --- C.1: what a dummy reference may stand for ---------------------------------------------


class GovernorKind(Enum):
    """C.1's `Governor`, whose five alternatives are the shapes a `ParamGovernor` may take.

    ECN adds three of these to X.683's list — `REFERENCE`, `#ENCODINGS` and
    `EncodingClassFieldType` — because ECN parameterizes over things ASN.1 has no dummy for.
    `TYPE` is X.683's own and is kept for a reason worth stating: C.1's production admits it,
    while C.1's a)-d) list of what a dummy may stand for never assigns it to any kind. It is
    therefore *writable and unreachable* through that list, and modelling it is how that stays
    visible rather than being quietly dropped as "not in the list".
    """

    #: A type extracted from an encoding class — governs values, value sets and value lists.
    ENCODING_CLASS_FIELD_TYPE = "EncodingClassFieldType"
    #: The literal keyword `REFERENCE`, which governs an `identifier` dummy.
    REFERENCE = "REFERENCE"
    #: An encoding class, which governs encoding objects and ordered object lists.
    DEFINED_OR_BUILTIN_ENCODING_CLASS = "DefinedOrBuiltinEncodingClass"
    #: The literal keyword `#ENCODINGS`, which governs an encoding object set.
    ENCODINGS = "#ENCODINGS"
    #: X.683's own `Type`, admitted by C.1's production and named by none of its rules.
    TYPE = "Type"


class ParameterKind(Enum):
    """What a `DummyReference` stands for — C.1's list of five, expanded to seven names.

    C.1 groups them: "an ASN.1 value, value set, or fixed-type ordered value list" share one
    governor, and "an encoding object, or an ordered encoding object list" share another. They
    are split apart here because C.4's correspondence rules do **not** group them — a), b) and
    c) name three different actual-parameter alternatives for the three value kinds, and e)
    and g) name two for the two object kinds. Grouping by governor and grouping by actual are
    different partitions of the same five sentences, and only one of them can be the enum.

    (C.1's list is lettered `a) a) b) c) d)` in the published text — five items, four letters,
    the first repeated. Noted rather than relied upon: the count of *rules* is what this enum
    encodes, and that is unambiguous.)
    """

    ENCODING_CLASS = "encoding-class"
    VALUE = "value"
    VALUE_SET = "value-set"
    ORDERED_VALUE_LIST = "ordered-value-list"
    IDENTIFIER = "identifier"
    ENCODING_OBJECT = "encoding-object"
    ORDERED_ENCODING_OBJECT_LIST = "ordered-encoding-object-list"
    ENCODING_OBJECT_SET = "encoding-object-set"


#: C.1's correspondence: the governor each dummy kind requires. `None` means the clause says
#: "there shall be no `ParamGovernor`" — which only the encoding-class kind gets.
_REQUIRED_GOVERNOR: dict = {
    ParameterKind.ENCODING_CLASS: None,
    ParameterKind.VALUE: GovernorKind.ENCODING_CLASS_FIELD_TYPE,
    ParameterKind.VALUE_SET: GovernorKind.ENCODING_CLASS_FIELD_TYPE,
    ParameterKind.ORDERED_VALUE_LIST: GovernorKind.ENCODING_CLASS_FIELD_TYPE,
    ParameterKind.IDENTIFIER: GovernorKind.REFERENCE,
    ParameterKind.ENCODING_OBJECT: GovernorKind.DEFINED_OR_BUILTIN_ENCODING_CLASS,
    ParameterKind.ORDERED_ENCODING_OBJECT_LIST:
        GovernorKind.DEFINED_OR_BUILTIN_ENCODING_CLASS,
    ParameterKind.ENCODING_OBJECT_SET: GovernorKind.ENCODINGS,
}


@dataclass(frozen=True)
class Parameter:
    """C.1's `Parameter`: a `DummyReference` with the `ParamGovernor` its kind requires.

    The governor is not decoration. C.1 makes it *mandatory* for four of the five kinds and
    *forbidden* for the fifth, so "governor present" is a fact about the dummy's kind rather
    than a style choice — which means a parameter list can be checked without knowing anything
    about the assignment it heads.

    `governed_by` carries the name when the governor is a class or an extracted type
    (`#Frame`, `#INT.&length`); `REFERENCE` and `#ENCODINGS` are keywords with nothing to name,
    so it stays empty for those and a name supplied anyway is refused rather than ignored.
    """

    dummy: str
    kind: ParameterKind
    governor: GovernorKind | None = None
    governed_by: str = ""

    def __post_init__(self) -> None:
        if not self.dummy:
            raise Asn1Error("ECN: C.1 — a Parameter needs a DummyReference")
        required = _REQUIRED_GOVERNOR[self.kind]
        if required is None and self.governor is not None:
            raise Asn1Error(
                f"ECN: C.1 a) — a dummy standing for an encoding class shall have no "
                f"ParamGovernor; {self.dummy} declares {self.governor.value}")
        if required is not None and self.governor is not required:
            got = "none" if self.governor is None else self.governor.value
            raise Asn1Error(
                f"ECN: C.1 — a dummy standing for {self.kind.value} shall be governed by "
                f"{required.value}; {self.dummy} is governed by {got}")
        names = self.governor in (GovernorKind.ENCODING_CLASS_FIELD_TYPE,
                                  GovernorKind.DEFINED_OR_BUILTIN_ENCODING_CLASS,
                                  GovernorKind.TYPE)
        if names and not self.governed_by:
            raise Asn1Error(
                f"ECN: C.1 — {self.governor.value} is a governor that names something, and "
                f"{self.dummy} names nothing")
        if not names and self.governed_by:
            governor = "none" if self.governor is None else self.governor.value
            raise Asn1Error(
                f"ECN: C.1 — {governor} is a keyword governor with nothing to name, and "
                f"{self.dummy} names {self.governed_by!r}")


@dataclass(frozen=True)
class ParameterList:
    """C.1's `ParameterList ::= "{<" Parameter "," + ">}"`.

    `"," +` is X.680 §13.13's one-or-more separated list, so an **empty** parameter list is
    not a `ParameterList` at all. That is not pedantry: C.3 gives `Reference "{<" ">}"` as a
    way to *reference* a parameterized definition with no actuals, and if `{<>}` were also a
    legal parameter list the two would be the same token sequence meaning opposite things.

    **DummyGovernors are refused**, which is C.1's own NOTE: "DummyGovernor's are not allowed
    in ECN". X.683 lets one dummy govern another; ECN does not, so a governor naming a sibling
    dummy is an error here even though the identical text is legal ASN.1.
    """

    parameters: tuple[Parameter, ...] = ()

    def __post_init__(self) -> None:
        if not self.parameters:
            raise Asn1Error(
                "ECN: C.1 — a ParameterList is `\"{<\" Parameter \",\" + \">}\"`, which is "
                "one or more; `{<>}` is C.3's empty ACTUAL list and means the opposite")
        seen: set[str] = set()
        for parameter in self.parameters:
            if parameter.dummy in seen:
                raise Asn1Error(
                    f"ECN: C.1 — {parameter.dummy} appears twice in one ParameterList")
            seen.add(parameter.dummy)
        for parameter in self.parameters:
            if parameter.governed_by and parameter.governed_by in seen:
                raise Asn1Error(
                    f"ECN: C.1's NOTE — \"DummyGovernors are not allowed in ECN\", and "
                    f"{parameter.dummy} is governed by the sibling dummy "
                    f"{parameter.governed_by}")

    def __len__(self) -> int:
        return len(self.parameters)

    def kinds(self) -> tuple[ParameterKind, ...]:
        return tuple(parameter.kind for parameter in self.parameters)

    def names(self) -> tuple[str, ...]:
        return tuple(parameter.dummy for parameter in self.parameters)

    def render(self) -> str:
        """The list as ECN writes it — `{< ... >}`, never `{ ... }`."""
        return "{<" + ", ".join(self.names()) + ">}"


# --- C.4: the actual parameter list --------------------------------------------------------


class ActualKind(Enum):
    """C.4's `ActualParameter` alternatives — ten of them, for eight correspondence rules.

    The arithmetic is the interesting part. C.4's a)-h) assign an alternative to each dummy
    kind, and h) covers references with three spellings at once: "the `identifier`, `STRUCTURE`
    or `OUTER` alternative shall be used". `ComponentIdList` appears in the production and in
    **no** rule — which reads like an orphan until §17.5.15 supplies the missing sentence: a
    `REFERENCE` actual "can either be supplied as a dummy parameter of the encoding object that
    is being defined, or it can be supplied as a `ComponentIdList`". So a reference actual has
    four spellings, C.4 h) lists three, and the fourth is written down a clause away.
    """

    VALUE = "Value"
    VALUE_SET = "ValueSet"
    ORDERED_VALUE_LIST = "OrderedValueList"
    ENCODING_CLASS = "DefinedOrBuiltinEncodingClass"
    ENCODING_OBJECT = "EncodingObject"
    ENCODING_OBJECT_SET = "EncodingObjectSet"
    ORDERED_ENCODING_OBJECT_LIST = "OrderedEncodingObjectList"
    #: §15.3.1's `identifier "." +` — a dot-separated path to a component.
    COMPONENT_ID_LIST = "ComponentIdList"
    #: A bare `identifier`, which is C.4 h)'s first spelling of a reference.
    IDENTIFIER = "identifier"
    #: §17.5.15's keyword, admitted "only when the actual parameter is used as specified in
    #: 17.5.15".
    STRUCTURE = "STRUCTURE"
    #: The container of the entire encoding — the same `#OUTER` clause 25 gives a class for.
    OUTER = "OUTER"


#: C.4 a)-h): which actual alternatives fit each dummy kind. A reference dummy takes four,
#: every other kind takes exactly one.
_ACCEPTED_ACTUALS: dict = {
    ParameterKind.VALUE: (ActualKind.VALUE,),
    ParameterKind.VALUE_SET: (ActualKind.VALUE_SET,),
    ParameterKind.ORDERED_VALUE_LIST: (ActualKind.ORDERED_VALUE_LIST,),
    ParameterKind.ENCODING_CLASS: (ActualKind.ENCODING_CLASS,),
    ParameterKind.ENCODING_OBJECT: (ActualKind.ENCODING_OBJECT,),
    ParameterKind.ENCODING_OBJECT_SET: (ActualKind.ENCODING_OBJECT_SET,),
    ParameterKind.ORDERED_ENCODING_OBJECT_LIST: (ActualKind.ORDERED_ENCODING_OBJECT_LIST,),
    ParameterKind.IDENTIFIER: (ActualKind.IDENTIFIER, ActualKind.COMPONENT_ID_LIST,
                               ActualKind.STRUCTURE, ActualKind.OUTER),
}


@dataclass(frozen=True)
class ActualParameter:
    """One entry of C.4's `ActualParameterList`.

    `text` is what was written — a class name, an object name, a dotted `ComponentIdList`, or
    nothing at all for `STRUCTURE` and `OUTER`, which are keywords that denote themselves.
    """

    kind: ActualKind
    text: str = ""

    def __post_init__(self) -> None:
        keyword = self.kind in (ActualKind.STRUCTURE, ActualKind.OUTER)
        if keyword and self.text:
            raise Asn1Error(
                f"ECN: C.4 — {self.kind.value} is a keyword actual and denotes itself; "
                f"{self.text!r} adds nothing it could mean")
        if not keyword and not self.text:
            raise Asn1Error(f"ECN: C.4 — a {self.kind.value} actual parameter needs a value")
        if self.kind is ActualKind.COMPONENT_ID_LIST:
            # §15.3.1: `ComponentIdList ::= identifier "." +`.
            parts = self.text.split(".")
            if not all(parts):
                raise Asn1Error(
                    f"ECN: §15.3.1 — a ComponentIdList is `identifier \".\" +`; "
                    f"{self.text!r} has an empty component")

    def components(self) -> tuple[str, ...]:
        """The `ComponentIdList`'s path, outermost first."""
        if self.kind is not ActualKind.COMPONENT_ID_LIST:
            raise Asn1Error(
                f"ECN: {self.kind.value} is not a ComponentIdList and has no component path")
        return tuple(self.text.split("."))


@dataclass(frozen=True)
class ActualParameterList:
    """C.4's `ActualParameterList ::= "{<" ActualParameter "," + ">}"`.

    Empty is admitted here and refused in `ParameterList`, and C.3 is why: it modifies X.683's
    `ParameterizedReference` to `Reference | Reference "{<" ">}"`, so `Foo{<>}` is a legal way
    to write a reference. The two productions share their delimiters and disagree about zero.
    """

    actuals: tuple[ActualParameter, ...] = ()

    def __len__(self) -> int:
        return len(self.actuals)

    def render(self) -> str:
        return "{<" + ", ".join(a.text or a.kind.value for a in self.actuals) + ">}"


def check_actuals(parameters: ParameterList, actuals: ActualParameterList,
                  *, what: str = "the reference") -> None:
    """X.683 §9.6 and C.4's a)-h): the actuals fit the dummies, in number and in kind.

    Two different faults, named as two, because a specification with the right count and the
    wrong kinds is a different mistake from one that miscounted — and the second is the one a
    reader can fix from the message alone.
    """
    if len(actuals) != len(parameters):
        raise Asn1Error(
            f"ECN: X.683 9.6 — {what} supplies {len(actuals)} actual parameter(s) to a "
            f"definition with {len(parameters)} dummy parameter(s)")
    for parameter, actual in zip(parameters.parameters, actuals.actuals):
        accepted = _ACCEPTED_ACTUALS[parameter.kind]
        if actual.kind not in accepted:
            names = " or ".join(kind.value for kind in accepted)
            raise Asn1Error(
                f"ECN: C.4 — the dummy {parameter.dummy} stands for {parameter.kind.value}, "
                f"so its actual shall use the {names} alternative; {actual.kind.value} was "
                f"supplied")


# --- C.2: the three parameterized assignments ----------------------------------------------


class AssignmentKind(Enum):
    """C.2's `ParameterizedAssignment` alternatives.

    Three, not X.683's own set: ECN has no parameterized *type* assignment, because ECN
    assigns encoding classes, objects and object sets and leaves types to ASN.1.
    """

    ENCODING_CLASS = "ParameterizedEncodingClassAssignment"
    ENCODING_OBJECT = "ParameterizedEncodingObjectAssignment"
    ENCODING_OBJECT_SET = "ParameterizedEncodingObjectSetAssignment"


@dataclass(frozen=True)
class ParameterizedAssignment:
    """C.2's parameterized assignment, with §8.4's ECN-modified scope rule enforced.

    The scope rule is the whole reason this class knows about its governor. X.683 §8.4 scopes
    a dummy to "that part of the `ParameterizedAssignment` which follows the `::=`"; C.2
    modifies it so that for an object assignment "the scope extends to the
    `DefinedOrBuiltinEncodingClass` which **precedes** the `::=`". C.2's NOTE gives the
    motivating shape::

        new-component-encoding {<#Any-class>} #New-component {<#Any-class>} ::= { ... }

    where the dummy is used as an actual parameter of the governor. Under X.683's unmodified
    scope that line does not parse, so the modification is load-bearing rather than editorial
    — and it is only granted to the object form, which is why using a dummy in an object
    *set*'s governor is refused here.
    """

    name: str
    kind: AssignmentKind
    parameters: ParameterList
    #: For an object assignment, the `DefinedOrBuiltinEncodingClass` before the `::=`.
    governor: str = ""
    #: Actuals the governor supplies, when it is itself a parameterized class reference.
    governor_actuals: ActualParameterList | None = None
    #: Whatever the assignment assigns. Kept unlowered: instantiation is substitution, and
    #: there is nothing to resolve until a reference supplies actuals.
    body: object = None

    def __post_init__(self) -> None:
        if self.kind is AssignmentKind.ENCODING_OBJECT:
            if not self.governor:
                raise Asn1Error(
                    f"ECN: C.2 — a ParameterizedEncodingObjectAssignment carries a "
                    f"DefinedOrBuiltinEncodingClass before its `::=`; {self.name} has none")
        elif self.governor:
            raise Asn1Error(
                f"ECN: C.2 — only a ParameterizedEncodingObjectAssignment names a governor "
                f"before its `::=`; {self.name} is a {self.kind.value}")
        if self.governor_actuals is None:
            return
        dummies = set(self.parameters.names())
        used = {actual.text for actual in self.governor_actuals.actuals if actual.text}
        borrowed = sorted(used & dummies)
        if borrowed and self.kind is not AssignmentKind.ENCODING_OBJECT:
            raise Asn1Error(
                f"ECN: C.2's 8.4 extends a dummy's scope across the `::=` only for a "
                f"ParameterizedEncodingObjectAssignment; {self.name} is a {self.kind.value} "
                f"and its governor uses {borrowed}")

    def instantiate(self, actuals: ActualParameterList) -> dict:
        """X.683 §9.7: bind each dummy to its actual, having checked C.4's correspondence.

        Returns the bindings rather than a substituted body. Substituting into an ECN body
        needs the body's own vocabulary — a structure's fields, an object's property groups —
        and doing it here would mean this module knowing every shape in the two clauses above
        it. The caller has that vocabulary; what it does not have is C.4's table.
        """
        check_actuals(self.parameters, actuals, what=f"the reference to {self.name}")
        return dict(zip(self.parameters.names(), actuals.actuals))


@dataclass(frozen=True)
class ParameterizedReference:
    """C.3's `ParameterizedReference ::= Reference | Reference "{<" ">}"`.

    Both alternatives name a definition without supplying anything, and C.3 admits the second
    *explicitly* — an empty actual list is a spelling of a reference, not a zero-arity
    instantiation. `Foo` and `Foo{<>}` therefore denote the same thing, which `is_bare` keeps
    visible rather than normalizing away: the two differ in the source text and a canonical
    serialization has to choose one.
    """

    name: str
    actuals: ActualParameterList | None = None

    @property
    def is_bare(self) -> bool:
        """Whether this is `Reference` rather than `Reference {< ... >}`."""
        return self.actuals is None

    def render(self) -> str:
        return self.name if self.actuals is None else self.name + self.actuals.render()


# --- §22.1.2: what a replacement's parameterization has to look like -----------------------


@dataclass(frozen=True)
class ReplacementParameterization:
    """§22.1.2.2 to §22.1.2.8: the parameterization a `REPLACE ... WITH ... ENCODED BY` needs.

    This is the missing half of `ecn_user.ReplacementStructure`. That class knows how to
    instantiate a replacement around a field; these are the rules about the *definitions* it
    instantiates, and they are rules about parameter lists rather than about bits — which is
    why they wait for this module.

    Four of them are worth reading twice.

    **§22.1.2.2** — the `WITH` structure has "a single encoding class parameter". Exactly one,
    and of exactly that kind, so a replacement structure parameterized over a value or an
    object set is refused even though C.1 would admit such a dummy anywhere else.

    **§22.1.2.4** — the `ENCODED BY` object "shall be defined in a parameterized encoding
    object assignment in which the governor is the corresponding `WITH` parameterized encoding
    structure, **instantiated with `#D`**". The governor is not the structure; it is the
    structure applied to the object's own dummy. That is precisely the shape C.2's §8.4
    modification exists to permit, so §22.1.2.4 is unwritable without it — the two clauses are
    a matched pair, and this class is where that is checked.

    **§22.1.2.5** — the object may carry two further dummies, and the second is a
    biconditional: a `REFERENCE` parameter "shall be present **if and only if** `INSERT AT
    HEAD` is specified". Both directions fail: a head-end insertion with no reference dummy has
    no way to reach the inserted structure (§22.1.2.7 makes the `ENCODED BY` object set its
    fields "through its `REFERENCE` parameter"), and a reference dummy with no head-end
    insertion would be instantiated with nothing.

    **§22.1.2.2 and §22.1.2.4 both close with the same sentence** — "when they are specified
    in the above defined syntax, only the ... name shall be given. They shall not have any
    parameter list in this use of the names." So the definitions are parameterized and the
    *uses inside `REPLACE`* are bare. `bare_use` is that check, and it is the one a
    copy-from-the-definition mistake trips.
    """

    #: The `WITH` structure's own parameter list.
    structure: ParameterList
    #: The `ENCODED BY` object's, or `None` when there is no `ENCODED BY`.
    encoded_by: ParameterList | None = None
    #: Actuals the `ENCODED BY` assignment's governor supplies — §22.1.2.4's "instantiated
    #: with #D".
    governor_actuals: ActualParameterList | None = None
    #: Whether the group has an `INSERT AT HEAD` clause (§22.1.1.10).
    insert_at_head: bool = False
    #: §22.1.2.7's head-end structure parameter list. It shall be absent.
    head_end: ParameterList | None = None

    def __post_init__(self) -> None:
        self._check_structure()
        if self.head_end is not None:
            raise Asn1Error(
                "ECN: §22.1.2.7 — the INSERT AT HEAD encoding structures shall not have "
                "dummy parameters")
        if self.encoded_by is None:
            if self.insert_at_head:
                raise Asn1Error(
                    "ECN: §22.1.2.7 — the head-end structure's fields are set by the ENCODED "
                    "BY object through its REFERENCE parameter, and there is no ENCODED BY")
            return
        self._check_encoded_by()

    def _check_structure(self) -> None:
        kinds = self.structure.kinds()
        if kinds != (ParameterKind.ENCODING_CLASS,):
            written = ", ".join(kind.value for kind in kinds)
            raise Asn1Error(
                f"ECN: §22.1.2.2 — the WITH replacement structures shall be parameterized "
                f"encoding structures with a single encoding class parameter; this one is "
                f"parameterized over {written}")

    def _check_encoded_by(self) -> None:
        kinds = list(self.encoded_by.kinds())
        if not kinds or kinds[0] is not ParameterKind.ENCODING_CLASS:
            raise Asn1Error(
                "ECN: §22.1.2.4 — the ENCODED BY objects shall have a dummy parameter (#D) "
                "that is an encoding class")
        extra = kinds[1:]
        object_sets = extra.count(ParameterKind.ENCODING_OBJECT_SET)
        references = extra.count(ParameterKind.IDENTIFIER)
        if object_sets > 1:
            raise Asn1Error(
                "ECN: §22.1.2.5 — an ENCODED BY object may have another (but only one) dummy "
                f"parameter that is an encoding object set; {object_sets} were declared")
        if references > 1:
            raise Asn1Error(
                "ECN: §22.1.2.5 — an ENCODED BY object may have another (but only one) dummy "
                f"parameter that is a REFERENCE parameter; {references} were declared")
        unexpected = [kind.value for kind in extra
                      if kind not in (ParameterKind.ENCODING_OBJECT_SET,
                                      ParameterKind.IDENTIFIER)]
        if unexpected:
            raise Asn1Error(
                f"ECN: §22.1.2.5 lists the only further dummies an ENCODED BY object may "
                f"have — one encoding object set and one REFERENCE; {unexpected} is neither")
        if references and not self.insert_at_head:
            raise Asn1Error(
                "ECN: §22.1.2.5 — the REFERENCE parameter shall be present if and only if "
                "INSERT AT HEAD is specified, and it is not")
        if self.insert_at_head and not references:
            raise Asn1Error(
                "ECN: §22.1.2.5 — the REFERENCE parameter shall be present if and only if "
                "INSERT AT HEAD is specified, and it is")
        self._check_governor()

    def _check_governor(self) -> None:
        if self.governor_actuals is None:
            raise Asn1Error(
                "ECN: §22.1.2.4 — the ENCODED BY object's governor is the corresponding WITH "
                "structure instantiated with #D; this assignment instantiates it with nothing")
        dummy = self.encoded_by.names()[0]
        supplied = tuple(actual.text for actual in self.governor_actuals.actuals)
        if supplied != (dummy,):
            written = ", ".join(supplied) or "nothing"
            raise Asn1Error(
                f"ECN: §22.1.2.4 — the governor shall be the WITH structure instantiated "
                f"with {dummy}, the object's own encoding class dummy; it is instantiated "
                f"with {written}")

    def takes_object_set(self) -> bool:
        """§22.1.2.5's optional object-set dummy, whose actual is "the current combined
        encoding object set" — the very set `ecn_link.EncodingApplication.combined` builds."""
        if self.encoded_by is None:
            return False
        return ParameterKind.ENCODING_OBJECT_SET in self.encoded_by.kinds()[1:]


def bare_use(reference: ParameterizedReference, *, clause: str) -> str:
    """§22.1.2.2 and §22.1.2.4's shared closing sentence: inside `REPLACE`, the name is bare.

    "When they are specified in the above defined syntax, only the ... name shall be given.
    They shall not have any parameter list in this use of the names." So the same structure is
    written `#Length-prefixed{<#D>}` where it is *defined* and `#Length-prefixed` where it is
    *used*, and the natural mistake — copying the definition's spelling — is what this refuses.

    Note that C.3's `Foo{<>}` is refused too. It is a legal `ParameterizedReference` elsewhere
    and it is still "a parameter list in this use of the name".
    """
    if reference.actuals is not None:
        raise Asn1Error(
            f"ECN: {clause} — only the name shall be given here, with no parameter list; "
            f"{reference.render()} has one")
    return reference.name


# --- §17.5.17: finding the component a ComponentIdList names -------------------------------


def resolve_component(structure: dict, path: tuple[str, ...]) -> tuple[str, ...]:
    """§17.5.16 to §17.5.18: resolve a `ComponentIdList` against a nested structure.

    **§17.5.17 searches breadth-first, and that is the reading to get right.** The chosen
    identifier is "determined by the first match in a scan (in textual order) of the
    outer-level identifiers, then by a scan (in textual order) of the second level
    identifiers, then by a scan (in textual order) of the third-level identifiers, and so on".

    The obvious implementation — walk the structure recursively, return the first `name` seen —
    is depth-first, and the two disagree exactly when an inner component of an early field
    shares a name with an outer component of a later one. Depth-first picks the deep one;
    §17.5.17 picks the shallow one. Both produce a valid reference to a real field, so nothing
    fails: the encoding just points at a different component than the specification meant.

    `structure` maps a field name to either a leaf (anything that is not a `dict`) or a nested
    structure. The return is the resolved path, which differs from `path` only in that each
    step after the first is looked up inside the previous one (§17.5.18), while the first is
    found "at some level of nesting" (§17.5.16).
    """
    if not path:
        raise Asn1Error("ECN: §15.3.1 — a ComponentIdList has at least one identifier")
    first = _breadth_first(structure, path[0])
    if first is None:
        raise Asn1Error(
            f"ECN: §17.5.16 — the first identifier of a ComponentIdList shall be that of a "
            f"textually present NamedType of the de-referenced governor; {path[0]!r} is not "
            f"present at any level of nesting")
    resolved = list(first)
    cursor = structure
    for step in first:
        cursor = cursor[step]
    for step in path[1:]:
        # §17.5.18: every later identifier is looked up *inside* what the previous one named,
        # not searched for again. Only the first identifier gets §17.5.16's nesting search.
        if not isinstance(cursor, dict) or step not in cursor:
            raise Asn1Error(
                f"ECN: §17.5.18 — {step!r} shall be an identifier in a NamedType of the "
                f"structure identified by {'.'.join(resolved)}, and it is not")
        cursor = cursor[step]
        resolved.append(step)
    return tuple(resolved)


def _breadth_first(structure: dict, wanted: str) -> tuple[str, ...] | None:
    """§17.5.17's scan: every outer-level name, then every second-level name, and so on."""
    level: list[tuple[tuple[str, ...], dict]] = [((), structure)]
    while level:
        following: list[tuple[tuple[str, ...], dict]] = []
        for prefix, node in level:
            for name, child in node.items():
                if name == wanted:
                    return prefix + (name,)
                if isinstance(child, dict):
                    following.append((prefix + (name,), child))
        level = following
    return None


__all__ = [
    "ActualKind", "ActualParameter", "ActualParameterList", "AssignmentKind", "GovernorKind",
    "Parameter", "ParameterKind", "ParameterList", "ParameterizedAssignment",
    "ParameterizedReference", "ReplacementParameterization", "bare_use", "check_actuals",
    "resolve_component",
]
