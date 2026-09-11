"""Retrieval as a planned query: legality first, then a price on BCIR's twelve axes.

`search_chunks.py` used to execute one plan. The backend was a flag, the row set was
always every row, and the text of every chunk was materialized whether or not the
ranking wanted it. Nothing chose anything, so nothing could explain why.

This module makes the choice explicit, in the shape BCIR already uses for a
different question. `bcir/asn1/selection.py` picks an ASN.1 encoding rule by
enumerating candidates, discarding the ones that are not *legal* for the value at
hand, and minimizing over a named cost axis -- with `Objective.WIRE_SIZE` spelled
`"memory"` and `Objective.ENCODE_LATENCY` spelled `"compute.encode"`, because the
objectives are the cost vector's own axes. Choosing between the exact Q15 scan, the
C kernel and the lossy Q8 view is the same shape of decision, over the same twelve
axes, one of which is `accuracy`.

**Legality is decided before cost, and never from a measurement.** The four rules
below are structural properties of a plan, checked the way the verifier's R-laws
are checked: a plan that fails one is not a more expensive plan, it is not a plan.
In particular no amount of speed makes the Q8 view legal for a request that asked
for the exact ranking -- that is `accuracy`, and it is a legality question first
and a priced axis second.

**The prices are modeled, and say so.** The constants in `CostModel` are calibrated
from wall-clock measurements on one host; they are here to *order* plans, not to
predict a duration. Nothing in this module gates on them, no claim of optimality is
made from them (TMSAO-4: heuristic, no claim), and a plan's chosen-ness is always
explainable as "this axis, this number, these candidates".

Declared scope of the predicate language: a conjunction of `column op value` terms
with `op` one of `eq`, `ne`, `in`, `prefix`, the four comparisons `<`, `<=`, `>`,
`>=` on a measurement column, and the two presence tests. No disjunction across
columns, no arithmetic, no user expressions. Answering the next soundness question
by adding an expression evaluator is how a retrieval tool becomes an interpreter
nobody asked for; the boundary is stated here so the answer can point at it.
`BETWEEN` is not an operator because it does not need to be -- two comparisons on
one column intersect to the same interval.

**Two declared divergences from SQL, both about rows that carry no value.**

A comparison is *existential*: it admits a row that has a value satisfying it. A row
with no measurement satisfies neither `char_count >= 500` nor `char_count < 500`, so
the two do not partition the table. That is SQL's rule, and it is the one this
layout makes easiest to lose, because the null sentinel is an ordinary negative
integer to the packed column.

`!=` is *complement*, which is not SQL's rule: `language != c` admits a row with no
language, where SQL would return UNKNOWN and drop it. Retrieval wants the complement
far more often than it wants three-valued logic, and the surprising half of SQL's
answer is one term away: `language!=c` with `language=?` is exactly `<>`. Both
readings are gated in `verify_database.py` so neither can drift into the other.
"""

from __future__ import annotations

import itertools
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bcir.kbcir.cost import DIMS, CostVector  # noqa: E402  (path set above)

import catalog as catalog_module  # noqa: E402


class PlanError(ValueError):
    """A request that cannot be turned into a plan. Never a silently narrowed one."""


# --------------------------------------------------------------------------
# Objectives -- the cost vector's own axis names, as in bcir/asn1/selection.py
# --------------------------------------------------------------------------


class Objective(Enum):
    """What the caller is short of. The value names the axis it minimizes.

    `EXACTNESS` is not "no objective": it is the named objective whose answer is the
    definition -- the exact Q15 ranking, whatever it costs. It is the degenerate case
    this rail pins, the way `Objective.NONE` pins DER in the ASN.1 selector.
    """

    EXACTNESS = "accuracy"
    LATENCY = "compute"
    FOOTPRINT = "memory"
    STARTUP = "compile"


#: Per-objective weight vectors. One axis carries the objective's weight; the rest
#: carry 1, so a tie on the named axis is broken by total work rather than by
#: declaration order alone.
def _weights(primary: str) -> tuple[int, ...]:
    return tuple(1024 if name == primary else 1 for name in DIMS)


# --------------------------------------------------------------------------
# Predicates
# --------------------------------------------------------------------------

OPERATORS = ("eq", "ne", "in", "prefix", "lt", "le", "gt", "ge", "isnull", "notnull")

#: The operators that compare a measurement against an integer.
RANGE_OPERATORS = ("lt", "le", "gt", "ge")

#: The operators that take no value at all.
NULLARY_OPERATORS = ("isnull", "notnull")

#: How `--where` spells each operator. The parse takes the *leftmost* spelling and,
#: at that position, the longest -- not the first entry of this table that occurs
#: anywhere in the term. Scanning in table order reads `title=a!=b` as `!=` with the
#: column `title=a`, which is a confusing error where `title = "a!=b"` is the answer.
_SPELLINGS = (
    ("!=", "ne"),
    (">=", "ge"),
    ("<=", "le"),
    ("^=", "prefix"),
    (">", "gt"),
    ("<", "lt"),
    ("=", "eq"),
)

#: How each range operator prints, so an explained plan reads as the SQL it means.
_RANGE_SQL = {"lt": "<", "le": "<=", "gt": ">", "ge": ">="}

#: The reserved right-hand side meaning "a value, any value": `column=?` is SQL's
#: IS NOT NULL and `column!=?` is IS NULL. Reserved on every column rather than only
#: on the measurement columns where nothing could collide with it, so the grammar
#: answers the same question the same way everywhere (`docs/security/laws.md` L12).
#: The cost is stated rather than hidden: a column holding the literal string "?"
#: cannot be matched by equality.
PRESENCE = "?"

#: A value may be wrapped in double quotes so it can contain characters the grammar
#: would otherwise read as syntax -- above all the comma that separates a set. Inside
#: quotes `\"` is a quote and `\\` is a backslash, and those are the only two escapes;
#: a backslash before anything else is refused rather than silently dropped.
#:
#: This is PostgREST's rule, adopted for the reason PostgREST states it: without it a
#: value containing a comma does not fail, it silently becomes two values. On this
#: corpus that is not hypothetical -- 103 titles and 393 heading trails contain a
#: comma, and `title=Hello, World` asked about two titles that do not exist rather
#: than the one that does.
#:
#: A value is quoted *whole* or not at all: a quote is syntax only as the first
#: character of a value, the matching close quote must be the value's last character,
#: and a quote anywhere inside an unquoted value is refused. The alternative -- quotes
#: significant wherever they appear, as a shell reads them -- means `title=say "hi"`
#: silently asks about `say hi`, which is the same silent misreading the comma rule
#: exists to remove, just spelled with a different character.
QUOTE = '"'
ESCAPE = "\\"

#: The one character a value may never contain, quoted or not.
#:
#: The catalog reserves NUL for itself twice over: `catalog.NULL_KEY` is a NUL
#: followed by `null`, under which the postings list the rows carrying no value for
#: an indexed column, and `catalog.pair_key` joins two column names with a NUL. Both
#: choices are sound precisely because no corpus value contains one -- but until this
#: predicate existed, nothing enforced that on the *input* side, and the sentinel was
#: reachable by typing it: `language=\0null` resolved to exactly the rows that
#: `language!=?` resolves to, and its negation to exactly `language=?`.
#:
#: That is a second spelling of a question the grammar already spells once, which is
#: how a reserved implementation value becomes part of the domain it was supposed to
#: sit outside (`docs/security/laws.md` L20). The grammar already reserves `?` for
#: presence and says so; this reserves the character the storage layer had quietly
#: been relying on, and says so in the same place.
#:
#: Refusing it is total rather than a heuristic: a NUL cannot appear in a value the
#: corpus holds -- no chunk record in this corpus contains one, and a column whose
#: values could would have broken the postings file long before it reached a query.
RESERVED_CHARACTER = "\0"

#: A measurement is compared against an integer in ASCII digits and nothing else.
#: `int()` would also accept `1_000`, `+5`, Unicode digits and surrounding
#: whitespace -- a larger language than the one documented, admitted by the host
#: rather than by this module. Same rule as the wire-format rails: enforce the
#: grammar before the conversion.
_INTEGER = re.compile(r"-?[0-9]+", re.ASCII)


def require_integer(column: str, op: str, text: str) -> int:
    """Read a bound, or refuse it. The one place a comparison's right side is decided.

    Construction and evaluation both ask here, so a term cannot be accepted when it
    is built and rejected when it is read, or the reverse (`docs/security/laws.md`
    L14). `--having` asks too, for the operators `parse_predicate` does not validate.
    """
    if not _INTEGER.fullmatch(text):
        raise PlanError(
            f"{column} {_RANGE_SQL.get(op, op)} {text!r}: a measurement is compared "
            "against an integer, in ASCII digits with an optional leading '-'"
        )
    return int(text)


@dataclass(frozen=True)
class Predicate:
    """One `column op value` term. Total, closed, and comparable.

    `values` is always a tuple, so every operator shares one shape and a caller never
    has to ask which arity it is holding. A presence test holds the empty tuple: it
    compares against nothing, and inventing a placeholder value for it would put a
    string in the one place nothing should read one.
    """

    column: str
    op: str
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.op not in OPERATORS:
            raise PlanError(f"unknown operator {self.op!r}; the language is {', '.join(OPERATORS)}")
        if self.op in NULLARY_OPERATORS:
            if self.values:
                raise PlanError(f"operator {self.op!r} takes no value, got {len(self.values)}")
            return
        if not self.values:
            raise PlanError(f"predicate on {self.column!r} has no value")
        if self.op in ("eq", "ne", "prefix", *RANGE_OPERATORS) and len(self.values) != 1:
            raise PlanError(f"operator {self.op!r} takes one value, got {len(self.values)}")
        if self.op in RANGE_OPERATORS:
            # Refuse a malformed bound at the prompt rather than partway through a
            # scan: the caller can still fix the term they typed, and a plan priced
            # from a comparison that cannot be made was never a plan.
            require_integer(self.column, self.op, self.values[0])

    @property
    def bound(self) -> int:
        """The integer a comparison compares against, or a refusal."""
        return require_integer(self.column, self.op, self.values[0])

    def __str__(self) -> str:
        """The term as `EXPLAIN` prints it, spelled so it can be pasted back.

        Every value goes through `spell_value`, so a value carrying a comma, a quote
        or the presence sentinel prints as the term that reads back to this predicate
        rather than as one that reads back to a different one. An ordinary value is
        untouched, so the common line is unchanged.
        """
        if self.op == "isnull":
            return f"{self.column} IS NULL"
        if self.op == "notnull":
            return f"{self.column} IS NOT NULL"
        if self.op == "in":
            return f"{self.column} in ({', '.join(spell_value(v) for v in self.values)})"
        spelling = {"eq": "=", "ne": "!=", "prefix": "^="}.get(self.op) or _RANGE_SQL[self.op]
        return f"{self.column} {spelling} {spell_value(self.values[0])}"


def interval(predicate: Predicate) -> tuple[int | None, int | None]:
    """A comparison as one closed integer interval `[low, high]`; None is unbounded.

    Four operators collapse to one shape because the column is integral -- `> n` is
    `>= n + 1` exactly, with nothing between them. So the catalog gets one range
    entry point instead of four, and two comparisons on the same column intersect by
    taking the tighter end of each, which is how `BETWEEN` is spelled here
    (`docs/security/laws.md` L14).
    """
    value = predicate.bound
    if predicate.op == "ge":
        return value, None
    if predicate.op == "gt":
        return value + 1, None
    if predicate.op == "le":
        return None, value
    if predicate.op == "lt":
        return None, value - 1
    raise PlanError(f"{predicate.op!r} is not a comparison; the comparisons are {_RANGE_SQL}")


def parse_predicate(text: str) -> Predicate:
    """Parse one `--where` term.

    `subject=llvm`            -> eq
    `subject!=llvm`           -> ne
    `source_path^=training/`  -> prefix
    `subject=llvm,data`       -> in, because a comma-separated right side is a set
    `title="Hello, World"`    -> eq, because a quoted comma is a character
    `char_count>=500`         -> ge, and `<=`, `>`, `<` likewise
    `language=?`              -> IS NOT NULL; `language!=?` is IS NULL

    A term with no operator is an error rather than a guess. Guessing here would
    silently widen a query the caller meant to narrow.

    **One value grammar, every operator, and the arity checked after the split.** The
    right side is read by `split_values` whatever the operator is. Reading a comma as
    a separator under `=` and as an ordinary character under `!=` -- which is what
    this function used to do -- makes `title=X` and `title!=X` stop being complements
    for exactly the values that contain a comma, which is the one thing this module's
    header promises they are (`docs/security/laws.md` L12). The cost is stated rather
    than hidden: `title!=Hello, World` is now refused, and spelled
    `title!="Hello, World"`.

    **The set is decided by the count, not by the comma.** `subject=llvm,` used to
    read as a one-element IN and `subject=llvm` as an equality, two spellings of one
    predicate differing in what `EXPLAIN` printed. One value is `=` and several are
    IN, on every input.
    """
    split = _split_term(text)
    if split is None:
        raise PlanError(
            f"predicate {text!r} names no operator; write column=value, column!=value, "
            f"column^=prefix, column>=500 (or <=, >, <), column={PRESENCE} for "
            f"'has a value', or column=one,two for a set"
        )
    column, op, raw = split
    if not column:
        raise PlanError(f"predicate {text!r} has no column")
    pieces = split_values(raw)
    if not pieces:
        raise PlanError(f"predicate {text!r} has no value")
    if any(value == PRESENCE and not was_quoted for value, was_quoted in pieces):
        if len(pieces) != 1:
            raise PlanError(
                f"predicate {text!r} puts the reserved value {PRESENCE!r} in a set; "
                f"{PRESENCE} asks whether a value is present at all and cannot be one "
                f"of several -- for the character itself write {QUOTE}{PRESENCE}{QUOTE}"
            )
        if op not in ("eq", "ne"):
            raise PlanError(
                f"predicate {text!r} asks whether {column!r} carries a value, which only "
                f"'=' and '!=' spell; write {column}={PRESENCE} or {column}!={PRESENCE}"
            )
        return Predicate(column, "notnull" if op == "eq" else "isnull", ())
    values = tuple(value for value, _ in pieces)
    if len(values) == 1:
        return Predicate(column, op, values)
    if op != "eq":
        raise PlanError(
            f"predicate {text!r} gives {len(values)} values to an operator that takes "
            f"one; only '=' spells a set, and a value that itself contains a comma is "
            f"written {QUOTE}like,this{QUOTE}"
        )
    return Predicate(column, "in", values)


def _read_quoted(raw: str, index: int) -> tuple[str, int]:
    """One quoted value starting at `raw[index] == QUOTE`; the text and where it ended.

    Only `\\"` and `\\\\` are escapes. A backslash before any other character is refused
    rather than read as that character, so a backslash has one spelling inside quotes
    instead of two (`\\x` and `x` would otherwise both mean `x`).
    """
    out: list[str] = []
    index += 1
    while index < len(raw):
        character = raw[index]
        if character == ESCAPE:
            if index + 1 >= len(raw):
                raise PlanError(
                    f"predicate value {raw!r} ends in a dangling {ESCAPE!r}; inside "
                    f"quotes a backslash escapes the character after it"
                )
            following = raw[index + 1]
            if following not in (QUOTE, ESCAPE):
                raise PlanError(
                    f"predicate value {raw!r} spells {ESCAPE + following!r}, which is "
                    f"not an escape; inside quotes only {ESCAPE + QUOTE!r} and "
                    f"{ESCAPE + ESCAPE!r} are"
                )
            out.append(following)
            index += 2
            continue
        if character == QUOTE:
            return "".join(out), index + 1
        out.append(character)
        index += 1
    raise PlanError(
        f"predicate value {raw!r} opens a quote it never closes; a value that contains "
        f"a comma must be quoted, and a quote must be closed"
    )


def split_values(raw: str) -> tuple[tuple[str, bool], ...]:
    """Split a right-hand side into (value, was_quoted) pairs, honouring quotes.

    A comma inside quotes does not separate. Whether a piece was quoted travels with
    it because the callers need it: an *unquoted* `?` is the presence sentinel and a
    *quoted* one is the literal character, and an unquoted empty piece is dropped (as
    `a,,b` always has been) where a quoted one is a deliberate empty value.

    Every way of writing something this function cannot read back unchanged is a
    refusal, never a guess: an unterminated quote, a quote inside an unquoted value,
    text after a closing quote, an unknown escape. Guessing turns a typo into a
    different query that still returns rows, which is the failure mode this whole
    function exists to remove.
    """
    pieces: list[tuple[str, bool]] = []
    index, length = 0, len(raw)
    while True:
        while index < length and raw[index] == " ":
            index += 1
        if index < length and raw[index] == QUOTE:
            value, index = _read_quoted(raw, index)
            quoted = True
            while index < length and raw[index] == " ":
                index += 1
            if index < length and raw[index] != ",":
                raise PlanError(
                    f"predicate value {raw!r} carries text after a closing quote; a "
                    f"value is quoted whole or not at all"
                )
        else:
            start = index
            while index < length and raw[index] != ",":
                if raw[index] == QUOTE:
                    raise PlanError(
                        f"predicate value {raw!r} puts a quote inside an unquoted "
                        f"value; a value is quoted whole or not at all, so write "
                        f"{spell_value(raw)} to ask about the text itself"
                    )
                index += 1
            value, quoted = raw[start:index].strip(), False
        if RESERVED_CHARACTER in value:
            raise PlanError(
                f"predicate value {value!r} contains the reserved character "
                f"{RESERVED_CHARACTER!r}, which the catalog uses for the key that "
                "marks a row as carrying no value; ask for those rows with "
                f"{PRESENCE!r} instead ('column!={PRESENCE}' is IS NULL)"
            )
        pieces.append((value, quoted))
        if index >= length:
            break
        index += 1  # the comma
    return tuple(piece for piece in pieces if piece[1] or piece[0])


def spell_value(value: str) -> str:
    """How a value is written so that `split_values` reads it back unchanged.

    `EXPLAIN` prints predicates, and a printed predicate a caller cannot paste back is
    a worse answer than no printed predicate. Quoting is applied only where it is
    needed, so the ordinary case still reads as plain text.

    A backslash needs no quoting: outside quotes it is an ordinary character, and the
    escapes exist only inside them.
    """
    needs = value == "" or value != value.strip() or value == PRESENCE
    needs = needs or any(character in value for character in (",", QUOTE))
    if not needs:
        return value
    escaped = value.replace(ESCAPE, ESCAPE + ESCAPE).replace(QUOTE, ESCAPE + QUOTE)
    return f"{QUOTE}{escaped}{QUOTE}"


def _split_term(text: str) -> tuple[str, str, str] | None:
    """The leftmost operator in a term, and there the longest spelling of it.

    Maximal munch at the leftmost position, rather than the first table entry found
    anywhere: `char_count>=500` has to read as `>=` and not as `=` with the column
    `char_count>`, and `title=a>=b` has to read as `=` on `title` and not as `>=` on
    the column `title=a`. One rule settles both.
    """
    best: tuple[int, str, str] | None = None
    for spelling, op in _SPELLINGS:
        index = text.find(spelling)
        if index <= 0:
            continue
        if best is None or index < best[0] or (index == best[0] and len(spelling) > len(best[1])):
            best = (index, spelling, op)
    if best is None:
        return None
    index, spelling, op = best
    return text[:index].strip(), op, text[index + len(spelling) :].strip()


# --------------------------------------------------------------------------
# Selection -- which rows, and how many the catalog says that is
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Selection:
    """The rows a predicate conjunction admits, plus what the catalog knew in advance.

    `rows` is None when every row is admitted -- the unfiltered case, kept distinct
    from "a filter that happens to match everything" so a plan can price the mask it
    would have to build.

    `estimated` is what the statistics said before any postings were read, and
    `estimate_exact` says whether that number was a count or a bound. Both are
    recorded because the estimate is what the *planner* used, and a plan that was
    chosen on a bound should be explainable as having been chosen on a bound.
    """

    rows: tuple[int, ...] | None
    estimated: int
    estimate_exact: bool
    predicates: tuple[Predicate, ...] = ()

    @property
    def admitted(self) -> int:
        return self.estimated if self.rows is None else len(self.rows)


def estimate(catalog, predicate: Predicate) -> tuple[int, bool]:
    """How many rows a single predicate admits, and whether that count is exact."""
    if predicate.op == "prefix":
        if predicate.column != catalog_module.PATH_COLUMN:
            raise PlanError(
                f"prefix is only defined on {catalog_module.PATH_COLUMN}, not {predicate.column!r}"
            )
        return catalog.count_prefix(predicate.values[0])
    if predicate.op == "eq":
        count = catalog.count_equals(predicate.column, predicate.values[0])
        if count is None:
            raise PlanError(_unindexed(catalog, predicate.column))
        return count, True
    if predicate.op == "in":
        # Over the *distinct* values. Each row carries one value per indexed column,
        # so the postings lists of distinct values are disjoint and their sizes add;
        # a value written twice would otherwise add its rows twice and report a count
        # larger than the table (`subject=llvm,llvm` said 4320 of 2215 rows, and said
        # it was exact). `dict.fromkeys` rather than `set` so the order a caller wrote
        # is the order the sum walks, and the estimate does not depend on hash order.
        total = 0
        for value in dict.fromkeys(predicate.values):
            count = catalog.count_equals(predicate.column, value)
            if count is None:
                raise PlanError(_unindexed(catalog, predicate.column))
            total += count
        return total, True
    if predicate.op in RANGE_OPERATORS:
        low, high = interval(predicate)
        # Exact, from the sorted index. This was first written to price from the zone
        # map instead, on the principle that estimating must not cost an artifact
        # read. Measurement retired that: the zone map calls a 25%-selective predicate
        # 98% selective, because an open interval keeps every block holding one large
        # value. An estimate 16x over the truth is worse than the read it saves, and
        # the predicate being priced is about to read that index anyway.
        with _as_plan_error():
            return catalog.count_in_range(predicate.column, low, high), True
    if predicate.op in NULLARY_OPERATORS:
        with _as_plan_error():
            return len(_presence_rows(catalog, predicate)), True
    if predicate.op == "ne":
        count = catalog.count_equals(predicate.column, predicate.values[0])
        if count is None:
            raise PlanError(_unindexed(catalog, predicate.column))
        # Complement, not SQL's three-valued `<>`: a row carrying no value is not
        # equal to this one, so it is admitted. Declared in the module docstring and
        # gated, so the two readings cannot quietly swap.
        return catalog.rows_total - count, True
    raise PlanError(f"operator {predicate.op!r} has no selectivity rule")


@contextmanager
def _as_plan_error():
    """Turn the catalog's refusals into this module's, so a caller catches one type.

    A `KeyError` escaping here reaches `search_chunks` as a traceback rather than as
    a usage verdict, which is the same exit either way to a human and a different one
    to a script (`docs/security/laws.md` L1).
    """
    try:
        yield
    except (KeyError, catalog_module.CatalogError) as exc:
        raise PlanError(str(exc).strip('"')) from exc


def _presence_rows(catalog, predicate: Predicate) -> list[int]:
    """The rows a presence test admits. One lookup, so `estimate` and `_rows_for` agree."""
    if predicate.op == "isnull":
        return catalog.rows_absent(predicate.column)
    return catalog.rows_present(predicate.column)


def _unindexed(catalog, column: str) -> str:
    indexed = ", ".join(catalog.indexed_columns()) + f", {catalog_module.PATH_COLUMN}"
    return f"column {column!r} is not indexed; the catalog indexes {indexed}"


#: The most pair lookups a joint estimate may make before it declines and lets the
#: bound stand. An indexed column is low cardinality by role, so the real product here
#: is 68; the cap is a bound on the *resource*, placed where the work is committed
#: rather than asserted about the data (`docs/security/laws.md` L3). Without it a
#: column that grew to a million distinct values would put a million-entry cross
#: product in the estimator, which exists to be cheaper than the answer.
MAX_JOINT_CELLS = 4096


def value_keys(catalog, predicate: Predicate) -> tuple[str, frozenset[str]] | None:
    """The index keys one predicate admits on an indexed column, or None.

    Every operator that is a statement about *which values* a column holds folds into
    the same shape -- a set of keys -- and once folded there is no polarity left to
    reason about. `!=` is the complement, `IS NULL` is the null key alone, `IS NOT
    NULL` is everything else, `IN` is the set as written. Writing it this way rather
    than as an inclusion-exclusion formula per operator pair is the difference between
    one rule and sixteen (`docs/security/laws.md` L14).

    A key is passed back out as a key, not as a value: `index_key` is the identity on
    strings, so a key read from the statistics is also the argument that looks it up
    again. Nothing converts in either direction, so nothing can convert differently.
    """
    column = predicate.column
    if column not in catalog.indexed_columns():
        return None
    every = frozenset(catalog.distinct(column))
    if predicate.op == "eq":
        return column, every & {catalog_module.index_key(predicate.values[0])}
    if predicate.op == "in":
        return column, every & {catalog_module.index_key(v) for v in predicate.values}
    if predicate.op == "ne":
        return column, every - {catalog_module.index_key(predicate.values[0])}
    if predicate.op == "isnull":
        return column, every & {catalog_module.NULL_KEY}
    if predicate.op == "notnull":
        return column, every - {catalog_module.NULL_KEY}
    return None


def _count_pairwise(catalog, left, left_keys, right, right_keys) -> int | None:
    """Exact rows admitted on two indexed columns at once, or None if unavailable."""
    if not catalog.has_pairs(left, right):
        return None
    if len(left_keys) * len(right_keys) > MAX_JOINT_CELLS:
        return None
    return sum(
        catalog.count_pair(left, a, right, b) for a in sorted(left_keys) for b in sorted(right_keys)
    )


def joint(catalog, predicates) -> tuple[int, bool] | None:
    """Rows admitted by the value-set terms of a conjunction, and whether that is all of it.

    The catalog stores the full joint distribution over every pair of indexed columns,
    so a conjunction naming at most two of them is a *count* and not a bound -- and an
    empty pair, which is most of them, counts zero instead of "at most the smaller
    marginal". Terms this cannot fold (a path prefix, a comparison on a measurement)
    are left out, and the second element of the answer says so: what comes back is
    then still an upper bound on the whole conjunction, just a much tighter one.

    Three or more indexed columns have no stored statistic, and building one from the
    pairs would mean assuming independence -- the assumption this whole mechanism
    exists to stop making. What is taken instead is the *tightest pair*, which assumes
    nothing: each pair counts a strictly larger set than the conjunction does, so the
    smallest of them is an upper bound on it, and a far tighter one than any marginal.

    Returns None only when no term folds at all.
    """
    admitted: dict[str, frozenset[str]] = {}
    whole = True
    for predicate in predicates:
        folded = value_keys(catalog, predicate)
        if folded is None:
            whole = False
            continue
        column, keys = folded
        admitted[column] = keys if column not in admitted else admitted[column] & keys
    if not admitted:
        return None
    with _as_plan_error():
        if len(admitted) == 1:
            ((column, keys),) = admitted.items()
            return sum(catalog.count_equals(column, key) for key in sorted(keys)), whole
        bounds = []
        for left, right in itertools.combinations(sorted(admitted), 2):
            counted = _count_pairwise(catalog, left, admitted[left], right, admitted[right])
            if counted is not None:
                bounds.append(counted)
        if not bounds:
            return None
        # Two columns and a pair table: the smallest (only) bound is the count itself.
        # More than two: it is the tightest pair, and the conjunction is not covered.
        return min(bounds), whole and len(admitted) == 2


def select(catalog, predicates) -> Selection:
    """Resolve a conjunction of predicates to the exact rows it admits.

    The estimate is computed first, from statistics alone, and the *smallest*
    predicate is evaluated first so the intersection shrinks fastest -- the only
    ordering decision here, and it is made from exact counts rather than a guess.

    Evaluation is exact even where the estimate was a bound: a prefix the index did
    not count still resolves by walking distinct paths, which is over paths, not over
    rows. Pricing may use a bound; answering never does.

    What the catalog does *not* hold is a joint statistic, so a conjunction is priced
    at the tightest marginal and reported as the bound it is. Holding the exact pair
    counts is a real option -- they are small and the corpus is static -- and it is
    written up as a candidate rather than assumed here; what is not an option is
    calling the bound a count.
    """
    predicates = tuple(predicates)
    if not predicates:
        return Selection(None, catalog.rows_total, True, ())

    estimates = []
    for predicate in predicates:
        count, exact = estimate(catalog, predicate)
        estimates.append((count, exact, predicate))
    # Three sources, in order of what each can claim.
    #
    # The tightest *marginal* bounds the conjunction from above, because every estimate
    # in this module is an upper bound. It is not a count of the conjunction, and
    # `all(exact)` -- which this used to say -- does not make it one: that asks whether
    # each term was counted exactly, where the claim being made is about their
    # intersection. Two exactly-counted terms still give a bound, and on this corpus
    # 83% of indexed value pairs were reported `[exact]` over a number that was wrong,
    # `kind=code AND language=asm` saying 1 row where there are none
    # (`docs/security/laws.md` L1: the label is part of the verdict).
    #
    # The joint statistics answer exactly for the terms they cover. When they cover
    # every term, that is the count and the bound is retired. When they cover some, it
    # is a tighter bound than any marginal.
    #
    # And an upper bound of zero is a count of zero whatever the terms were, which is
    # worth keeping exact because it is the case a planner acts on.
    estimated = min(count for count, _, _ in estimates)
    estimate_exact = len(estimates) == 1 and estimates[0][1]
    covered = joint(catalog, predicates)
    if covered is not None:
        count, whole = covered
        if whole:
            estimated, estimate_exact = count, True
        elif count < estimated:
            estimated = count
    if estimated == 0:
        estimate_exact = True

    order = sorted(range(len(estimates)), key=lambda i: (estimates[i][0], i))
    admitted: set[int] | None = None
    for index in order:
        _, _, predicate = estimates[index]
        rows = _rows_for(catalog, predicate)
        admitted = rows if admitted is None else (admitted & rows)
        if not admitted:
            break
    resolved = tuple(sorted(admitted or ()))
    return Selection(resolved, estimated, estimate_exact, predicates)


def _rows_for(catalog, predicate: Predicate) -> set[int]:
    if predicate.op == "prefix":
        return set(catalog.rows_with_prefix(predicate.values[0]))
    if predicate.op == "eq":
        rows = catalog.rows_equal(predicate.column, predicate.values[0])
        if rows is None:
            raise PlanError(_unindexed(catalog, predicate.column))
        return set(rows)
    if predicate.op == "in":
        found: set[int] = set()
        for value in predicate.values:
            rows = catalog.rows_equal(predicate.column, value)
            if rows is None:
                raise PlanError(_unindexed(catalog, predicate.column))
            found |= set(rows)
        return found
    if predicate.op in RANGE_OPERATORS:
        low, high = interval(predicate)
        # Two binary searches over the sorted index, taking the window unsorted: this
        # goes straight into a set, and `select` sorts the intersection once at the
        # end, so ordering the window here would be an ordering nobody reads.
        with _as_plan_error():
            return set(catalog.seek_range(predicate.column, low, high)[0])
    if predicate.op in NULLARY_OPERATORS:
        with _as_plan_error():
            return set(_presence_rows(catalog, predicate))
    if predicate.op == "ne":
        rows = catalog.rows_equal(predicate.column, predicate.values[0])
        if rows is None:
            raise PlanError(_unindexed(catalog, predicate.column))
        return set(range(catalog.rows_total)) - set(rows)
    raise PlanError(f"operator {predicate.op!r} has no evaluation rule")


# --------------------------------------------------------------------------
# The cost model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CostModel:
    """Modeled unit costs, in nanoseconds of work on the calibration host.

    Every field was measured, and the measurement is named beside it. They are here
    to *order* plans; nothing gates on them, and no optimality is claimed from them.
    A host where they are wrong chooses a slower plan, never a wrong answer -- which
    is why legality is decided without them.
    """

    #: One thousand Q15 multiply-accumulates, per backend. Measured as the full-corpus
    #: scan (2,012 x 512) divided by 1,007 thousand-MAC units.
    ns_per_kmac_reference: int = 53_700
    ns_per_kmac_native: int = 884
    ns_per_kmac_q8: int = 4_727

    #: Materializing chunk text. `full` is one parse of every record (the plan that
    #: reads the whole table); `seek` is one recorded span; `open` is per file touched.
    ns_per_row_full_scan: int = 22_400
    ns_per_row_seek: int = 8_000
    ns_per_file_open: int = 20_000

    #: Building a derived column over the whole set, per row -- what the reference
    #: backend pays for ``||p||^2`` when the set does not store that column.
    ns_per_row_derived: int = 31_000

    #: ...and per row when it does. Reading `squares.u32` is a digest of 8,860 bytes
    #: and one `array.frombytes`, measured at 0.20 ms over 2,215 rows. The two
    #: constants differ by 344x, which is the whole reason the column is stored -- and
    #: the reason a plan that charged the first for a set that pays the second would be
    #: mispricing the reference backend by 68.5 ms of work nobody does.
    ns_per_row_derived_stored: int = 90

    #: Producing the kernel: a compiler invocation, or the stamped cache path.
    ns_compile_kernel: int = 281_900_000
    ns_load_cached_kernel: int = 1_900_000

    #: Quantizing the corpus matrix into BCIRQ8, per row. No q8 artifact is persisted,
    #: so this is paid per process by any plan that uses that view.
    #:
    #: This field was 4_000 and is the one constant here that was ever badly wrong.
    #: Re-measured over 2,215 rows it is 150,152 ns/row -- 37.5x the old value -- split
    #: 45.21 ms flattening the f32 column into a list and 287.38 ms inside
    #: `bcir_ai_quantize_q8_f64`, 332.59 ms of setup that the plan was charging 8.9 ms
    #: for. The error was reachable and it changed a decision: on a host carrying the
    #: q8 kernel but not the q15 one -- a partial build, which is a supported state --
    #: `choose(LATENCY)` preferred `q8 /seek` over `reference /seek`, and the plan it
    #: picked runs 530 ms against the rejected plan's 190 ms (median of 11, IQR 44 and
    #: 10; an earlier reading of 833 ms was one noisy run, IQR 312, and is not the
    #: number). A cost model is allowed to
    #: be approximate; it is not allowed to be wrong by a factor that reorders the
    #: plans, because then the planner is choosing confidently in the wrong direction.
    #: `training/plans/baseline-v1.json` now pins that availability case so the
    #: ordering cannot drift back without S18 saying so.
    ns_per_row_q8_quantize: int = 150_000

    #: Bytes are counted for the `memory` axis directly; this scales them into the
    #: same integer space as the nanosecond fields so one weight vector spans both.
    memory_units_per_byte: int = 1

    #: What a lossy ranking costs on the `accuracy` axis, per requested result. Zero
    #: for the exact backends by construction -- they are the definition.
    accuracy_penalty_per_result: int = 1_000_000

    #: What a two-implementation differential costs on `verification`: it is the
    #: second scan, plus the comparison, and it buys the bit-equality claim.
    verification_units_per_row: int = 1_000


DEFAULT_MODEL = CostModel()

#: Where the defaults came from, so a reader can re-derive or challenge them.
CALIBRATION = {
    "method": "wall clock, warm, median of 3-25, one query against lexical-hash-v1",
    "corpus": "2,215 rows x 512 dims (re-measured; first calibrated at 2,012 rows)",
    "checked": (
        "every field re-derived from a measurement: kmac_native 0.99x, "
        "row_derived 1.01x, kmac_reference 0.96x, kmac_q8 0.85x of the stored "
        "value -- and row_q8_quantize 37.5x, which was corrected"
    ),
    "class": "wall -- indicative, never gating (docs/PERFORMANCE_AUDIT.md)",
    "note": "these order plans; they do not predict a duration and nothing gates on them",
}


# --------------------------------------------------------------------------
# Plans
# --------------------------------------------------------------------------

BACKENDS = ("reference", "native", "q8", "both")

#: Which backends compute the exact Q15 ranking that is this rail's definition.
EXACT_BACKENDS = frozenset({"reference", "native", "both"})

#: Which backends need a compiled kernel.
COMPILED_BACKENDS = frozenset({"native", "q8", "both"})


@dataclass(frozen=True)
class Plan:
    """One way to answer the request, with its verdict and its price."""

    backend: str
    selection: Selection
    top_k: int
    materialize: str  # "seek" | "full" | "none"
    kernel_cached: bool
    legal: bool
    refusals: tuple[str, ...] = ()
    cost: CostVector = field(default_factory=CostVector.zero)

    @property
    def scanned(self) -> int:
        return self.selection.admitted

    def label(self) -> str:
        where = "" if self.selection.rows is None else f" +where({self.scanned})"
        return f"{self.backend}{where} /{self.materialize}"


# -- legality ---------------------------------------------------------------
#
# Four rules, checked before any cost is computed, each a structural property of
# the plan rather than a measurement of it. The names are the vocabulary an
# explanation uses, so a refusal is always attributable to one rule.

LEGALITY_RULES = ("exactness", "availability", "coverage", "materialization")


def legality(
    plan_backend: str,
    selection: Selection,
    *,
    require_exact: bool,
    available_backends: frozenset,
    rows_total: int,
    materialize: str,
    want_text: bool,
) -> tuple[str, ...]:
    """Every rule this plan breaks, in rule order. Empty means legal."""
    refusals = []
    if require_exact and plan_backend not in EXACT_BACKENDS:
        refusals.append(
            f"exactness: {plan_backend} is a different arithmetic, and the request "
            "asked for the exact Q15 ranking"
        )
    if plan_backend not in available_backends:
        refusals.append(f"availability: the {plan_backend} backend is not reachable on this host")
    if selection.rows is not None:
        if any(not 0 <= row < rows_total for row in selection.rows):
            refusals.append("coverage: the selection names a row outside the set")
    if want_text and materialize == "none":
        refusals.append("materialization: the request wants text and this plan reads none")
    if not want_text and materialize == "full":
        refusals.append(
            "materialization: the request wants no text and this plan reads every row's"
        )
    return tuple(refusals)


# -- pricing ----------------------------------------------------------------


def price(
    backend: str,
    selection: Selection,
    *,
    top_k: int,
    dim: int,
    rows_total: int,
    materialize: str,
    kernel_cached: bool,
    files_touched: int,
    model: CostModel = DEFAULT_MODEL,
    derived_cached: bool = False,
) -> CostVector:
    """The twelve-axis cost of running this plan once, in modeled units.

    Only the axes a single-host in-process scan actually spends on are non-zero.
    `fabric`, `sync`, `thermal`, `power`, `reliability`, `security` and `contention`
    are zero here and are left zero rather than invented: an axis with a fabricated
    number is worse than an axis with none, because it will be minimized against.
    """
    scanned = selection.admitted
    # `bcir_ai_q8_rows_dot_f64` is a rows-dot, not a top-k: it takes no eligibility
    # mask, so a q8 plan scores every row whatever the predicate admitted and filters
    # afterwards. Pricing it as though the predicate narrowed its scan would make the
    # planner prefer it for exactly the queries where it helps least.
    scan_rows = rows_total if backend == "q8" else scanned
    kmacs = max(0, (scan_rows * dim + 1023) // 1024)

    per_kmac = {
        "reference": model.ns_per_kmac_reference,
        "native": model.ns_per_kmac_native,
        "q8": model.ns_per_kmac_q8,
        "both": model.ns_per_kmac_reference + model.ns_per_kmac_native,
    }[backend]
    compute = kmacs * per_kmac

    # A pure-Python scan reads its rows through a derived column; the C backends
    # hand the packed codes to the kernel and build nothing. Whether that column is
    # *built* or *read* is the difference between 31,000 ns a row and 90, so it is
    # asked rather than assumed -- the same shape as `kernel_cached` above, and for
    # the same reason: a price that ignores an artifact the caller already has is a
    # price for work that will not happen.
    if backend in ("reference", "both"):
        compute += rows_total * (
            model.ns_per_row_derived_stored if derived_cached else model.ns_per_row_derived
        )
    if backend == "q8":
        compute += rows_total * model.ns_per_row_q8_quantize

    if materialize == "full":
        compute += rows_total * model.ns_per_row_full_scan
        memory_bytes = rows_total * 1024
    elif materialize == "seek":
        compute += top_k * model.ns_per_row_seek + files_touched * model.ns_per_file_open
        memory_bytes = top_k * 1024
    else:
        memory_bytes = 0

    memory_bytes += scan_rows * dim * 2  # the codes the scan actually reads
    if selection.rows is not None:
        memory_bytes += rows_total  # the eligibility mask, one byte per row

    compile_units = (
        (model.ns_load_cached_kernel if kernel_cached else model.ns_compile_kernel)
        if backend in COMPILED_BACKENDS
        else 0
    )
    accuracy = 0 if backend in EXACT_BACKENDS else top_k * model.accuracy_penalty_per_result
    verification = scanned * model.verification_units_per_row if backend == "both" else 0

    return CostVector.of(
        compute=compute,
        memory=memory_bytes * model.memory_units_per_byte,
        compile=compile_units,
        accuracy=accuracy,
        verification=verification,
    )


# -- enumeration and choice -------------------------------------------------


def candidates(
    catalog,
    selection: Selection,
    *,
    top_k: int,
    dim: int,
    want_text: bool,
    require_exact: bool,
    available_backends: frozenset,
    kernel_cached: bool,
    files_touched: int,
    model: CostModel = DEFAULT_MODEL,
    derived_cached: bool = False,
) -> list[Plan]:
    """Every plan considered, legal or not, in declaration order.

    Illegal plans are kept rather than dropped, so an explanation can say *why* a
    plan the reader expected is not the one that ran. A planner that silently omits
    its rejects cannot be argued with.
    """
    plans = []
    materializations = ("seek", "full") if want_text else ("none",)
    for backend in BACKENDS:
        for materialize in materializations:
            refusals = legality(
                backend,
                selection,
                require_exact=require_exact,
                available_backends=available_backends,
                rows_total=catalog.rows_total,
                materialize=materialize,
                want_text=want_text,
            )
            plans.append(
                Plan(
                    backend=backend,
                    selection=selection,
                    top_k=top_k,
                    materialize=materialize,
                    kernel_cached=kernel_cached,
                    legal=not refusals,
                    refusals=refusals,
                    cost=price(
                        backend,
                        selection,
                        top_k=top_k,
                        dim=dim,
                        rows_total=catalog.rows_total,
                        materialize=materialize,
                        kernel_cached=kernel_cached,
                        files_touched=files_touched,
                        model=model,
                        derived_cached=derived_cached,
                    ),
                )
            )
    return plans


def choose(plans, objective: Objective) -> Plan:
    """The cheapest legal plan on the objective's axis. Legality first, always.

    Ties break by the scalarized cost and then by declaration order, so the same
    request against the same catalog picks the same plan on every host -- a planner
    whose choice wobbles cannot be gated.
    """
    legal = [plan for plan in plans if plan.legal]
    if not legal:
        raise PlanError(
            "no legal plan for this request:\n"
            + "\n".join(f"  {plan.label()}: {'; '.join(plan.refusals)}" for plan in plans)
        )
    weights = _weights(objective.value)
    order = {id(plan): index for index, plan in enumerate(plans)}
    return min(legal, key=lambda plan: (plan.cost.dot(weights), order[id(plan)]))


# -- explanation ------------------------------------------------------------


def explain(
    plans, chosen: Plan, objective: Objective, *, selection: Selection, requested: bool = False
) -> str:
    """EXPLAIN: what was considered, what it would cost, and why this one won.

    `requested` says the plan was named by a flag rather than picked by `choose`, and
    it changes the verdict word because the two are different facts. `--backend`
    defaults to `reference`, so the common case is a forced plan -- and marking it
    CHOSEN put that word on a plan that won nothing, beside a legal plan costing a
    hundredth as much and marked merely "legal". The natural reading of that table is
    that the planner preferred the expensive one, which is the opposite of true.
    A forced plan is marked REQUESTED, and the plan `choose` would have returned is
    named underneath, so the cost of overriding the planner is visible rather than
    inferred.
    """
    lines = [
        f"EXPLAIN  objective={objective.name.lower()} (minimize {objective.value})",
    ]
    if selection.predicates:
        terms = " AND ".join(str(predicate) for predicate in selection.predicates)
        exactness = "exact" if selection.estimate_exact else "bound"
        lines.append(
            f"  WHERE {terms}\n"
            f"    catalog estimated {selection.estimated} row(s) [{exactness}], "
            f"resolved {selection.admitted}"
        )
    else:
        lines.append(f"  WHERE (none) -- every one of {selection.estimated} row(s)")
    widest = max(len(plan.label()) for plan in plans)
    weights = _weights(objective.value)
    lines.append(f"  {'plan'.ljust(widest)}  {'scalar':>14}  {objective.value:>14}  verdict")
    for plan in plans:
        scalar = plan.cost.dot(weights)
        axis = plan.cost.as_dict()[objective.value]
        if not plan.legal:
            verdict = "REFUSED " + plan.refusals[0].split(":", 1)[0]
        elif plan is chosen:
            verdict = "REQUESTED" if requested else "CHOSEN"
        else:
            verdict = "legal"
        lines.append(f"  {plan.label().ljust(widest)}  {scalar:14d}  {axis:14d}  {verdict}")
    for plan in plans:
        if not plan.legal:
            for refusal in plan.refusals:
                lines.append(f"    {plan.label()}: {refusal}")
    if requested:
        # What the override cost, named rather than left for the reader to work out.
        try:
            would = choose(plans, objective)
        except PlanError:
            would = None
        if would is not None and would is not chosen:
            weights = _weights(objective.value)
            mine, theirs = chosen.cost.dot(weights), would.cost.dot(weights)
            ratio = f"{mine / theirs:.1f}x" if theirs else "more"
            lines.append(
                f"  requested by --backend; {objective.name.lower()} would have chosen "
                f"{would.label()} ({ratio} cheaper on this objective)"
            )
        elif would is not None:
            lines.append("  requested by --backend, and it is what this objective would choose")
    nonzero = {name: value for name, value in chosen.cost.as_dict().items() if value}
    lines.append(
        ("  requested" if requested else "  chosen")
        + " cost vector: "
        + ", ".join(f"{name}={value}" for name, value in sorted(nonzero.items()))
    )
    lines.append(f"  ({CALIBRATION['class']}; {CALIBRATION['note']})")
    return "\n".join(lines)
