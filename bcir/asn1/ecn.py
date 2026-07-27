"""Encoding Control Notation — Rec. ITU-T X.692 (02/2021) | ISO/IEC 8825-3:2021.

ECN is the odd one out in the suite: every other Recommendation *is* an encoding rule,
while ECN is a **notation for defining encoding rules**. Encoding classes are declared,
encoding objects realize them, and encoding object sets are applied to ASN.1 types to
determine their wire form. That is why the roadmap calls it the destination — an encoding
object set *is* a plan, and applying one *is* realization.

This module is **part one of two**, and the split is not arbitrary. §9.5.3 and §18.2 give
ECN seven **built-in** encoding object sets (the BER and PER variants), and §18.2.3 says
each is "a complete set of encoding objects which can be applied to any encoding structure
… to specify the corresponding BER or PER encodings". That half is required no matter what
— it is what lets an ECN specification name an encoding this repo already implements. The
*user-defined* half (the defined syntax of clauses 20–23, `#TRANSFORM`, `#OUTER`) is the
part the roadmap's own reduction gate may cut, and that gate is decided by measurement over
{DER, CANONICAL-PER-ALIGNED, CANONICAL-PER-UNALIGNED, COER, **CJER**} — a set that cannot
be measured yet, because JER is not built. So this module builds what both branches of the
gate require, and leaves the gated half to be built when the gate can actually be decided.

THE MODEL, in the order clause 9 introduces it:

* **Encoding class** (§9.2) — "an implicit property of all ASN.1 types … the set of all
  possible encoding specifications for that type". Names begin with `#`.
* **Encoding object** (§9.4) — a specific definition of encoding rules for one class.
* **Encoding object set** (§9.5) — objects grouped under the governor `#ENCODINGS`, and the
  thing an ELM applies to a type.
* **EDM / ELM** (§9.1.1) — Encoding Definition Modules define rules; a **single** Encoding
  Link Module applies them.

TWO STATIC LAWS, both of exactly the R24 kind — statements a verifier can refuse to express
rather than rules a runtime has to check:

* **§9.5.2** — "any set can contain only one encoding object of a given encoding class …
  Thus there is no ambiguity when an encoding object set is applied to a type". §18.1.7
  restates it and adds that a set may not hold an encoding-procedure class other than
  `#OUTER`. `EncodingObjectSet` refuses to be constructed otherwise, so an ambiguous set is
  not a set this module can hold.
* **§12.2.5** — "An ELM shall not apply encodings more than once to the same ASN.1 type."

The subtlety in §9.5.2 is what "given encoding class" means. §9.6.2 is explicit that if a
new class is *created from* an existing one, objects of **both** may appear in one set — so
the rule is keyed on class **identity**, not on category or on the primitive it derives
from. A set holding objects for `#SEQUENCE` and for `#My-Sequence ::= #SEQUENCE` is legal
and unambiguous; two objects for `#SEQUENCE` are not.

WHERE THE BUILT-IN SETS COME FROM. §18.2.2's NOTE is candid that X.690 and X.691 "were
written before this ECN Recommendation … and do not use the encoding object terminology.
They define, for example, the way an ASN.1 INTEGER or BOOLEAN type is to be encoded. This
should be interpreted as the definition of an encoding object of class #INTEGER or class
#BOOLEAN." So the built-in sets are not reimplemented here — they *name* `der.py` and
`per.py`, which is the whole point: ECN's contribution is the naming and the algebra, and
the octets stay the ones the existing rails already produce and already test against each
standard's own Annex A.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .tags import Asn1Error


class Category(Enum):
    """§9.6.6 — the categories an encoding class can be in.

    The category is not decoration: §9.6.5 makes it the thing that survives a class
    assignment, and §16/§18 use it to decide what may be written where.
    """

    ALTERNATIVES = "alternatives"
    CONCATENATION = "concatenation"
    REPETITION = "repetition"
    OPTIONALITY = "optionality"
    TAG = "tag"
    BOOLEAN = "boolean"
    BITSTRING = "bitstring"
    CHARACTERSTRING = "characterstring"
    INTEGER = "integer"
    NULL = "null"
    OBJECTIDENTIFIER = "objectidentifier"
    OCTETSTRING = "octetstring"
    OPENTYPE = "opentype"
    PAD = "pad"
    REAL = "real"
    TIME = "time"
    ENCODING_STRUCTURE = "encoding structure"
    #: Not a §9.6.6 category. The four classes of §9.6.7's encoding procedure group are
    #: "not directly related to ASN.1 constructs, and … cannot be assigned new names",
    #: so they need a category that `derive` can refuse on.
    ENCODING_PROCEDURE = "encoding procedure"


class CategoryGroup(Enum):
    """§9.6.7 — the three groups of categories.

    Note what is NOT here: the optionality and tag categories belong to no group. §9.6.4
    lists them alongside the groups rather than inside one ("classes in the bit-field group
    … combined using classes in the encoding constructor group …, together with classes in
    the optionality … and tag … categories"), and inventing a group for them would put them
    somewhere the standard deliberately does not.
    """

    BIT_FIELD = "bit-field"
    ENCODING_CONSTRUCTOR = "encoding constructor"
    ENCODING_PROCEDURE = "encoding procedure"


#: §9.6.7 — the bit-field group is every category that "correspond[s] to actual fields in an
#: encoding", plus every class in the encoding structure category.
_BIT_FIELD_CATEGORIES = frozenset({
    Category.BOOLEAN, Category.BITSTRING, Category.CHARACTERSTRING, Category.INTEGER,
    Category.NULL, Category.OBJECTIDENTIFIER, Category.OCTETSTRING, Category.OPENTYPE,
    Category.PAD, Category.REAL, Category.TIME, Category.ENCODING_STRUCTURE,
})
_CONSTRUCTOR_CATEGORIES = frozenset({
    Category.ALTERNATIVES, Category.CONCATENATION, Category.REPETITION,
})


def category_group(category: Category) -> CategoryGroup | None:
    """§9.6.7 — the group a category is in, or None for optionality and tag."""
    if category in _BIT_FIELD_CATEGORIES:
        return CategoryGroup.BIT_FIELD
    if category in _CONSTRUCTOR_CATEGORIES:
        return CategoryGroup.ENCODING_CONSTRUCTOR
    if category is Category.ENCODING_PROCEDURE:
        return CategoryGroup.ENCODING_PROCEDURE
    return None


@dataclass(frozen=True)
class EncodingClass:
    """§9.2 — "the set of all possible encoding specifications" for a type.

    `derived_from` records the §9.6.1 chain. It is what makes §9.6.2 expressible: a set may
    hold an object for `#SEQUENCE` *and* one for a class derived from it, because they are
    different classes even though they do the same structural job.
    """

    name: str
    category: Category
    derived_from: "EncodingClass | None" = None

    def __post_init__(self) -> None:
        # §9.2.1 and §9.3.1: a class reference name begins with "#".
        if not self.name.startswith("#"):
            raise Asn1Error(
                f"ECN: an encoding class reference begins with \"#\" (9.2.1); got "
                f"{self.name!r}")

    @property
    def group(self) -> CategoryGroup | None:
        return category_group(self.category)

    def primitive(self) -> "EncodingClass":
        """§9.6.3 — the primitive class at the root of the derivation chain."""
        cls = self
        while cls.derived_from is not None:
            cls = cls.derived_from
        return cls

    def derives_from(self, other: "EncodingClass") -> bool:
        cls: EncodingClass | None = self
        while cls is not None:
            if cls == other:
                return True
            cls = cls.derived_from
        return False

    def derive(self, name: str) -> "EncodingClass":
        """§9.6.1/§9.6.5 — `#My-Sequence ::= #SEQUENCE` keeps the category.

        §9.6.7 forbids this for the encoding procedure group, whose classes "cannot be
        assigned new names": they define procedures rather than structure, so a second name
        for one would name nothing new.
        """
        if self.category is Category.ENCODING_PROCEDURE:
            raise Asn1Error(
                f"ECN: {self.name} is in the encoding procedure group of categories and "
                f"cannot be assigned a new name (9.6.7)")
        return EncodingClass(name, self.category, self)


def _primitive(name: str, category: Category) -> EncodingClass:
    return EncodingClass(name, category)


# --- §9.6.3 the primitive classes, and §11.2 Table 2's derived built-ins -----------------

#: §9.6.3: "All built-in encoding classes are derived from one of a small number of
#: primitive encoding classes." These are those classes.
INT = _primitive("#INT", Category.INTEGER)
BOOL = _primitive("#BOOL", Category.BOOLEAN)
NUL = _primitive("#NUL", Category.NULL)
CHARS = _primitive("#CHARS", Category.CHARACTERSTRING)
OCTETS = _primitive("#OCTETS", Category.OCTETSTRING)
BITS = _primitive("#BITS", Category.BITSTRING)
REAL = _primitive("#REAL", Category.REAL)
OBJECT_IDENTIFIER = _primitive("#OBJECT-IDENTIFIER", Category.OBJECTIDENTIFIER)
OPEN_TYPE = _primitive("#OPEN-TYPE", Category.OPENTYPE)
TIME = _primitive("#TIME", Category.TIME)
PAD = _primitive("#PAD", Category.PAD)
CONCATENATION = _primitive("#CONCATENATION", Category.CONCATENATION)
ALTERNATIVES = _primitive("#ALTERNATIVES", Category.ALTERNATIVES)
REPETITION = _primitive("#REPETITION", Category.REPETITION)
OPTIONAL = _primitive("#OPTIONAL", Category.OPTIONALITY)
TAG = _primitive("#TAG", Category.TAG)

#: §9.6.7 — the encoding procedure group. Named here so a set can refuse them (§18.1.7)
#: even though their *definitions* belong to part two.
OUTER = _primitive("#OUTER", Category.ENCODING_PROCEDURE)
TRANSFORM = _primitive("#TRANSFORM", Category.ENCODING_PROCEDURE)
CONDITIONAL_INT = _primitive("#CONDITIONAL-INT", Category.ENCODING_PROCEDURE)
CONDITIONAL_REPETITION = _primitive("#CONDITIONAL-REPETITION",
                                    Category.ENCODING_PROCEDURE)

#: §11.2 Table 2 — the class each piece of ASN.1 notation becomes in an implicitly
#: generated encoding structure, and the primitive it derives from. The three entries
#: Table 2 marks "Defined using #SEQUENCE" (CHARACTER STRING, EMBEDDED PDV, EXTERNAL) are
#: derived from #CONCATENATION here, because that is what #SEQUENCE itself derives from and
#: the table is describing the same construction one step less directly.
_TABLE_2: tuple[tuple[str, str, EncodingClass], ...] = (
    ("BIT STRING", "#BIT-STRING", BITS),
    ("BOOLEAN", "#BOOLEAN", BOOL),
    ("CHARACTER STRING", "#CHARACTER-STRING", CONCATENATION),
    ("CHOICE", "#CHOICE", ALTERNATIVES),
    ("EMBEDDED PDV", "#EMBEDDED-PDV", CONCATENATION),
    ("ENUMERATED", "#ENUMERATED", INT),
    ("EXTERNAL", "#EXTERNAL", CONCATENATION),
    ("INTEGER", "#INTEGER", INT),
    ("NULL", "#NULL", NUL),
    ("OBJECT IDENTIFIER", "#OBJECT-IDENTIFIER", OBJECT_IDENTIFIER),
    ("OCTET STRING", "#OCTET-STRING", OCTETS),
    ("open type notation", "#OPEN-TYPE", OPEN_TYPE),
    ("OPTIONAL", "#OPTIONAL", OPTIONAL),
    ("REAL", "#REAL", REAL),
    ("RELATIVE-OID", "#RELATIVE-OID", OBJECT_IDENTIFIER),
    ("SEQUENCE", "#SEQUENCE", CONCATENATION),
    ("SEQUENCE OF", "#SEQUENCE-OF", REPETITION),
    ("SET", "#SET", CONCATENATION),
    ("SET OF", "#SET-OF", REPETITION),
    ("TIME", "#TIME", TIME),
    ("DATE", "#DATE", TIME),
    ("TIME-OF-DAY", "#TIME-OF-DAY", TIME),
    ("DATE-TIME", "#DATE-TIME", TIME),
    ("DURATION", "#DURATION", TIME),
    ("GeneralizedTime", "#GeneralizedTime", CHARS),
    ("UTCTime", "#UTCTime", CHARS),
    ("ObjectDescriptor", "#ObjectDescriptor", CHARS),
    ("BMPString", "#BMPString", CHARS),
    ("GeneralString", "#GeneralString", CHARS),
    ("GraphicString", "#GraphicString", CHARS),
    ("IA5String", "#IA5String", CHARS),
    ("NumericString", "#NumericString", CHARS),
    ("PrintableString", "#PrintableString", CHARS),
    ("TeletexString", "#TeletexString", CHARS),
    ("UniversalString", "#UniversalString", CHARS),
    ("UTF8String", "#UTF8String", CHARS),
    ("VideotexString", "#VideotexString", CHARS),
    ("VisibleString", "#VisibleString", CHARS),
    ("Textually present tag notation", "#TAG", TAG),
)


def _build_builtins() -> dict[str, EncodingClass]:
    classes = {cls.name: cls for cls in (
        INT, BOOL, NUL, CHARS, OCTETS, BITS, REAL, OBJECT_IDENTIFIER, OPEN_TYPE, TIME,
        PAD, CONCATENATION, ALTERNATIVES, REPETITION, OPTIONAL, TAG,
        OUTER, TRANSFORM, CONDITIONAL_INT, CONDITIONAL_REPETITION)}
    for _notation, name, primitive in _TABLE_2:
        # A Table 2 row whose class IS its primitive (#TIME, #OPEN-TYPE, #TAG,
        # #OBJECT-IDENTIFIER, #OPTIONAL) adds no new class -- the table is naming the
        # primitive itself, not deriving from it.
        if name == primitive.name:
            continue
        classes[name] = EncodingClass(name, primitive.category, primitive)
    return classes


#: Every class ECN has before a user writes anything: the primitives of §9.6.3 plus the
#: implicitly generated classes of §11.2 Table 2.
BUILTIN_CLASSES: dict[str, EncodingClass] = _build_builtins()

#: §11.2 Table 2 column 1 -> column 2, so an ASN.1 type can name its encoding class.
CLASS_FOR_NOTATION: dict[str, EncodingClass] = {
    notation: BUILTIN_CLASSES[name] for notation, name, _primitive in _TABLE_2
}


# --- §9.4 encoding objects, §9.5 encoding object sets ------------------------------------

@dataclass(frozen=True)
class EncodingObject:
    """§9.4.1 — "the specific definition of encoding rules for a given encoding class"."""

    encoding_class: EncodingClass
    name: str = ""
    #: What realizes this object. For a built-in set this names an existing rail (a
    #: `Strictness`, a `(PerRules, PerVariant)` pair); §18.2.2's NOTE is the licence to
    #: read X.690 and X.691 as definitions of encoding objects rather than reimplement them.
    realization: object | None = None


@dataclass(frozen=True)
class EncodingObjectSet:
    """§9.5 — objects governed by `#ENCODINGS`, and what an ELM applies to a type.

    The constructor is where §9.5.2 and §18.1.7 live. Both are static laws, so an
    ill-formed set is refused at construction rather than diagnosed at application time —
    the same posture R24 takes, and the reason §9.5.2 can say "Thus there is no ambiguity
    when an encoding object set is applied to a type" as a *fact* rather than a hope.
    """

    objects: tuple[EncodingObject, ...] = ()
    name: str = ""

    def __post_init__(self) -> None:
        seen: dict[EncodingClass, EncodingObject] = {}
        for obj in self.objects:
            cls = obj.encoding_class
            if cls in seen:
                # Keyed on class IDENTITY, never on category: §9.6.2 makes an object for
                # `#SEQUENCE` and one for `#My-Sequence ::= #SEQUENCE` legal in one set.
                raise Asn1Error(
                    f"ECN: an encoding object set holds at most one object per encoding "
                    f"class (9.5.2, 18.1.7); {cls.name} appears twice")
            if (cls.category is Category.ENCODING_PROCEDURE and cls != OUTER):
                raise Asn1Error(
                    f"ECN: an encoding object set may not hold {cls.name}, which is in "
                    f"the encoding procedure group of categories; 18.1.7 admits only "
                    f"#OUTER")
            seen[cls] = obj

    def object_for(self, cls: EncodingClass) -> EncodingObject | None:
        """The object that encodes `cls`, with §18.2.3's "appropriate de-referencing".

        An exact match wins; failing that the derivation chain of §9.6.1 is walked, which
        is what lets a set carrying only `#SEQUENCE` still encode a user's
        `#My-Sequence ::= #SEQUENCE`. §18.2.3's NOTE 2 states the precedence directly: an
        object added for the specific class "will take precedence over any encoding which
        could be obtained by de-referencing".
        """
        for obj in self.objects:
            if obj.encoding_class == cls:
                return obj
        target: EncodingClass | None = cls.derived_from
        while target is not None:
            for obj in self.objects:
                if obj.encoding_class == target:
                    return obj
            target = target.derived_from
        return None

    def union(self, other: "EncodingObjectSet", *, name: str = "") -> "EncodingObjectSet":
        """§18.1.5's `UnionMark`. The §9.5.2 law is re-checked, not assumed."""
        return EncodingObjectSet(self.objects + other.objects, name)


# --- §18.2 the built-in encoding object sets ---------------------------------------------

class BuiltinEncodingObjectSet(Enum):
    """§18.2.1's `BuiltinEncodingObjectSetReference` — the seven names ECN reserves."""

    PER_BASIC_ALIGNED = "PER-BASIC-ALIGNED"
    PER_BASIC_UNALIGNED = "PER-BASIC-UNALIGNED"
    PER_CANONICAL_ALIGNED = "PER-CANONICAL-ALIGNED"
    PER_CANONICAL_UNALIGNED = "PER-CANONICAL-UNALIGNED"
    BER = "BER"
    CER = "CER"
    DER = "DER"


#: §18.2.2 Table 4 — the object identifier each built-in set names.
#:
#: Table 4 as printed has a defect in one row: `PER-CANONICAL-UNALIGNED` is given as
#: `{joint-iso-itu-t(2) packed-encoding(3) canonical(1) unaligned(1)}`, with the `asn1(1)`
#: arc missing, which would make it a four-arc OID under a different parent while its three
#: siblings have five arcs. X.691 §33.2 is the defining clause and gives
#: `{joint-iso-itu-t asn1(1) packed-encoding(3) canonical(1) unaligned(1)}`. The defining
#: clause wins, and `test_asn1_ecn.py` pins every value here against the constants the
#: X.690 and X.691 rails already carry, so a transcription slip cannot survive.
BUILTIN_SET_OID: dict[BuiltinEncodingObjectSet, tuple[int, ...]] = {
    BuiltinEncodingObjectSet.PER_BASIC_ALIGNED: (2, 1, 3, 0, 0),
    BuiltinEncodingObjectSet.PER_BASIC_UNALIGNED: (2, 1, 3, 0, 1),
    BuiltinEncodingObjectSet.PER_CANONICAL_ALIGNED: (2, 1, 3, 1, 0),
    BuiltinEncodingObjectSet.PER_CANONICAL_UNALIGNED: (2, 1, 3, 1, 1),
    BuiltinEncodingObjectSet.BER: (2, 1, 1),
    BuiltinEncodingObjectSet.CER: (2, 1, 2, 0),
    BuiltinEncodingObjectSet.DER: (2, 1, 2, 1),
}

#: §18.2.4 — the classes every built-in set carries an *identical* object for. These are
#: the "basic building blocks of encodings" of §18.2.5, and §18.2.5.1-§18.2.5.4 define them
#: all in terms of PER-BASIC-UNALIGNED regardless of which set they appear in.
SHARED_CLASSES: tuple[EncodingClass, ...] = (
    INT, BOOL, NUL, CHARS, OCTETS, BITS, CONCATENATION)

#: §18.2.4, stated as a prohibition rather than an omission: the built-in sets "do not
#: contain encoding objects for #ALTERNATIVES, #REPETITION, and #PAD". A set that did would
#: be claiming BER or PER defines something they do not.
ABSENT_FROM_BUILTIN_SETS: tuple[EncodingClass, ...] = (ALTERNATIVES, REPETITION, PAD)


def _realization(which: BuiltinEncodingObjectSet):
    """What rail a built-in set names, in the vocabulary that rail already uses."""
    from .codec import Strictness
    from .per import PerRules, PerVariant

    per = {
        BuiltinEncodingObjectSet.PER_BASIC_ALIGNED: (PerRules.BASIC, PerVariant.ALIGNED),
        BuiltinEncodingObjectSet.PER_BASIC_UNALIGNED:
            (PerRules.BASIC, PerVariant.UNALIGNED),
        BuiltinEncodingObjectSet.PER_CANONICAL_ALIGNED:
            (PerRules.CANONICAL, PerVariant.ALIGNED),
        BuiltinEncodingObjectSet.PER_CANONICAL_UNALIGNED:
            (PerRules.CANONICAL, PerVariant.UNALIGNED),
    }
    if which in per:
        return per[which]
    if which is BuiltinEncodingObjectSet.BER:
        return Strictness.BER
    if which is BuiltinEncodingObjectSet.DER:
        return Strictness.DER
    # §18.2.1 names CER, and this rail deliberately does not implement it: X.690 §9.1
    # makes the indefinite form mandatory for constructed encodings, which is
    # irreconcilable with a frozen, digested artifact. The set still EXISTS here, because
    # refusing to name it would misreport what ECN offers -- it simply has no realization,
    # and applying it says so.
    return None


def builtin_object_set(which: BuiltinEncodingObjectSet) -> EncodingObjectSet:
    """§18.2.3 — "a complete set of encoding objects which can be applied to any encoding
    structure … to specify the corresponding BER or PER encodings".

    Complete, and no more than complete: §18.2.4's three excluded classes are excluded
    here too, so `object_for(ALTERNATIVES)` returns None rather than a fiction.
    """
    realization = _realization(which)
    objects = [
        EncodingObject(cls, f"{which.value}.{cls.name}", realization)
        for cls in BUILTIN_CLASSES.values()
        if cls not in ABSENT_FROM_BUILTIN_SETS
        and cls.category is not Category.ENCODING_PROCEDURE
    ]
    return EncodingObjectSet(tuple(objects), which.value)


def shared_object_constraints(cls: EncodingClass, kind) -> None:
    """§18.2.5.1-§18.2.5.4 — the design errors the shared objects are specified with.

    Every one of these is a *specification* error rather than a runtime failure: §18.2.5.1
    says "It is an ECN design error if the #INT does not have both a lower and an upper
    bound when this encoding object is applied", which is a statement about the ECN
    specification, not about any particular value. Checking it at application time is the
    earliest point this rail can, and refusing beats emitting an encoding whose width the
    standard never defined.
    """
    from .constraints import effective_size_constraint, effective_value_constraint

    primitive = cls.primitive()
    constraint = getattr(kind, "constraint", None)
    if primitive == INT:
        low, high = effective_value_constraint(constraint)
        if low is None or high is None:
            raise Asn1Error(
                f"ECN: the shared #INT encoding object is a PER-BASIC-UNALIGNED "
                f"#INTEGER encoding \"provided it is bounded\"; {cls.name} has no "
                f"{'lower' if low is None else 'upper'} bound (18.2.5.1)")
    elif primitive in (CHARS, OCTETS, BITS):
        low, high = effective_size_constraint(constraint)
        if low is None or high is None or low != high:
            raise Asn1Error(
                f"ECN: the shared {primitive.name} encoding object applies \"provided "
                f"they are a single size\"; {cls.name} has no effective size constraint "
                f"restricting it to one size (18.2.5.3)")
    elif primitive == CONCATENATION:
        components = getattr(kind, "components", ())
        loose = [c.name for c in components if c.optional or c.has_default]
        if loose:
            raise Asn1Error(
                f"ECN: the shared #CONCATENATION encoding object is a "
                f"PER-BASIC-UNALIGNED #SEQUENCE \"with no optional components\"; "
                f"{cls.name} has {sorted(loose)} (18.2.5.4)")


# --- §9.1.1, §12, §14: the modules -------------------------------------------------------

@dataclass
class EncodingDefinitionModule:
    """§14 — an EDM. Defines encoding classes, objects and object sets; applies nothing."""

    name: str
    classes: dict[str, EncodingClass] = field(default_factory=dict)
    objects: dict[str, EncodingObject] = field(default_factory=dict)
    object_sets: dict[str, EncodingObjectSet] = field(default_factory=dict)

    def assign_class(self, name: str, base: EncodingClass) -> EncodingClass:
        """§16.1.1's `EncodingClassAssignment`, with §9.3.3's name collision rule."""
        if name in self.classes or name in BUILTIN_CLASSES:
            raise Asn1Error(
                f"ECN: {name} is already an encoding class in this module or a built-in "
                f"(9.3.3)")
        self.classes[name] = base.derive(name)
        return self.classes[name]

    def lookup(self, name: str) -> EncodingClass:
        cls = self.classes.get(name) or BUILTIN_CLASSES.get(name)
        if cls is None:
            raise Asn1Error(f"ECN: {name} is not an encoding class in {self.name}")
        return cls


@dataclass(frozen=True)
class EncodingApplication:
    """§12.2.1's `EncodingApplication` — `ENCODE <classes> WITH <encodings>`."""

    classes: tuple[str, ...]
    encodings: EncodingObjectSet

    def __post_init__(self) -> None:
        if not self.classes:
            raise Asn1Error("ECN: an ENCODE names at least one encoding class (12.2.1)")


@dataclass
class EncodingLinkModule:
    """§12 — the ELM. There is exactly one per application (§12.1.2), and its sole
    function is to apply encodings (§12.1.9).

    §12.2.5 is enforced on construction: "An ELM shall not apply encodings more than once
    to the same ASN.1 type." Two applications naming one type would leave the type with two
    encodings and no rule for choosing, which is the same ambiguity §9.5.2 removes one level
    down.
    """

    name: str
    applications: tuple[EncodingApplication, ...] = ()

    def __post_init__(self) -> None:
        if not self.applications:
            raise Asn1Error(
                f"ECN: the ELM {self.name} has no EncodingApplication, and \"the sole "
                f"function of an ELM is to apply encodings\" (12.1.9)")
        seen: set[str] = set()
        for application in self.applications:
            for name in application.classes:
                if name in seen:
                    raise Asn1Error(
                        f"ECN: the ELM {self.name} applies encodings to {name} more than "
                        f"once (12.2.5)")
                seen.add(name)

    def encodings_for(self, class_name: str) -> EncodingObjectSet | None:
        for application in self.applications:
            if class_name in application.classes:
                return application.encodings
        return None


def encode_with(encodings: EncodingObjectSet, cls: EncodingClass, kind, value) -> bytes:
    """Apply an encoding object set to a value of `kind` (§13).

    This is the point of the whole notation: an object set names an encoding, and applying
    it produces octets. What it does NOT do is produce them itself — §18.2.2's NOTE reads
    X.690 and X.691 as definitions of encoding objects, so a built-in set dispatches to the
    rail that already implements it and the octets are the ones those rails' own Annex A
    tests already pin.
    """
    from .codec import Strictness
    from .per import PerRules, PerVariant, encode_per
    from .schema import Module

    obj = encodings.object_for(cls)
    if obj is None:
        raise Asn1Error(
            f"ECN: {encodings.name or 'the encoding object set'} holds no encoding object "
            f"for {cls.name} (9.5.1)")
    if obj.realization is None:
        raise Asn1Error(
            f"ECN: {encodings.name} names an encoding this rail does not implement; the "
            f"set exists because 18.2.1 reserves the name, not because the octets do")
    if cls in SHARED_CLASSES or cls.primitive() in SHARED_CLASSES:
        shared_object_constraints(cls, kind)
    if isinstance(obj.realization, tuple):
        rules, variant = obj.realization
        assert isinstance(rules, PerRules) and isinstance(variant, PerVariant)
        return encode_per(kind, value, variant=variant, rules=rules)
    assert isinstance(obj.realization, Strictness)
    module = Module("<ecn>", (), {"T": kind})
    return module.encode("T", value)


__all__ = [
    "ABSENT_FROM_BUILTIN_SETS", "BUILTIN_CLASSES", "BUILTIN_SET_OID",
    "BuiltinEncodingObjectSet", "CLASS_FOR_NOTATION", "Category", "CategoryGroup",
    "EncodingApplication", "EncodingClass", "EncodingDefinitionModule",
    "EncodingLinkModule", "EncodingObject", "EncodingObjectSet", "SHARED_CLASSES",
    "builtin_object_set", "category_group", "encode_with", "shared_object_constraints",
]
