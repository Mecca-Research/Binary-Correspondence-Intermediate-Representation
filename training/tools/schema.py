"""The table, declared once: what a column is called, what it holds, what it is for.

Before this module the answer was spread across three places that agreed by habit.
`training/schema/chunk-v1.json` declared eighteen fields and their JSON types.
`catalog.py` named six of them in three Python tuples -- `INDEXED_COLUMNS`,
`PATH_COLUMN`, `NUMERIC_COLUMNS` -- and said nothing about their types. The chunk
rows themselves carried whatever `build_chunks.py` put there. Nothing reconciled any
pair of the three, so renaming a field in the JSON Schema, or adding an index for a
column that does not exist, was a green run (`docs/security/laws.md` L15: a mirror
list will drift).

This module is the Python rail. `training/schema/chunk-v1.json` remains the JSON rail
-- it is what an external consumer validates against, and it is not generated from
here. The two are *reconciled by the gate*, out of their own sources, rather than one
being mirrored into the other: `check_schema` in `verify_database.py` reads the JSON
file and this table and refuses any disagreement about which fields exist, which are
required, which are nullable, what type each holds, and what values each admits.

**What a column declares, and why each part earns its place.**

`kind` is the role the database plays for this column, and it is the thing the three
tuples used to encode positionally:

    key         the row's identity -- `chunk_id`
    indexed     an inverted index by whole value
    path        the one column indexed by every directory prefix
    numeric     a packed integer column, gathered rather than scanned
    text        carried in the chunk file, projectable, not indexed
    structural  provenance and schema bookkeeping; not a queryable column

`type` is the JSON type, `required` is whether the row must carry the key at all, and
`nullable` is whether the value may be `null` once it does. Those two are genuinely
different questions and a row can answer them differently: `language` is optional
*and* nullable, `char_count` is required and not nullable.

`domain` and `minimum` are the constraints. They exist because "this is an integer"
was the only thing the build ever checked, and the interesting failures are not type
failures. A `kind` of `"prose "` with a trailing space types fine, indexes fine, and
silently creates a fifth kind that no query written against the documented four will
ever match. A `char_count` of `0` types fine and makes every average wrong. Both are
now refusals that name the row (`docs/security/laws.md` L1).

**Absent and invalid must not share a spelling.** `cell` returns `None` only where a
row genuinely carries no value, and raises for everything else. That distinction is
the whole point: a null is an answer, and a malformed row is not.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

#: The JSON Schema this table is reconciled against. Repository-relative, and read by
#: the gate rather than by this module -- nothing here touches the filesystem, so the
#: engine keeps working from an installed package as happily as from a checkout.
JSON_SCHEMA = "training/schema/chunk-v1.json"

#: The `schema` field every row must carry. Named here and not in `catalog.py` so the
#: row contract and the column table are declared in one file.
ROW_SCHEMA = "bcir-training/chunk/v1"

#: The roles a column may play. A role is not a hint: `ROLES` is enumerated so that a
#: column declared with a role nothing implements fails the gate rather than being
#: quietly treated as `structural`.
ROLES = ("key", "indexed", "path", "numeric", "text", "structural")


class SchemaError(Exception):
    """A row, or a request, that the declared table does not admit."""


@dataclass(frozen=True)
class Column:
    """One declared column.

    Frozen because this is a declaration. A caller that could rewrite the domain of
    `kind` at run time would be able to make a violating row legal by editing the rule
    it violates, which is not a check.
    """

    name: str
    type: str
    role: str
    required: bool = True
    nullable: bool = False
    domain: tuple[str, ...] | None = None
    minimum: int | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise SchemaError(f"column {self.name!r} has role {self.role!r}; the roles are {ROLES}")
        if self.role == "numeric" and self.type != "integer":
            raise SchemaError(f"column {self.name!r} is numeric but holds {self.type!r}")
        if self.domain is not None and self.type != "string":
            raise SchemaError(f"column {self.name!r} has a domain but holds {self.type!r}")
        if self.minimum is not None and self.type != "integer":
            raise SchemaError(f"column {self.name!r} has a minimum but holds {self.type!r}")


#: The table. Order is the order `chunk-v1.json` declares its properties in, so the
#: two read side by side.
COLUMNS: tuple[Column, ...] = (
    Column(
        "schema",
        "string",
        "structural",
        domain=(ROW_SCHEMA,),
        note="the row's own contract, checked before anything reads the row",
    ),
    Column(
        "chunk_id",
        "string",
        "key",
        note="sha256 over the identity tuple; the row's name everywhere else",
    ),
    Column("corpus", "string", "structural", domain=("training",)),
    Column(
        "subject",
        "string",
        "indexed",
        note="the training/<subject>/ folder; eight values, so a whole-value index",
    ),
    Column(
        "source_path",
        "string",
        "path",
        note="indexed by every directory prefix rather than by whole value",
    ),
    Column("source_sha256", "string", "structural"),
    Column("span", "object", "structural"),
    Column("heading_trail", "array", "text"),
    Column("title", "string", "text"),
    Column(
        "kind",
        "string",
        "indexed",
        domain=("prose", "code", "mixed", "table"),
        note="four values, enumerated: a fifth would index cleanly and match nothing",
    ),
    Column(
        "language",
        "string",
        "indexed",
        required=False,
        nullable=True,
        note="fence language for code and mixed chunks; absent is the common case",
    ),
    Column("text", "string", "text"),
    Column("char_count", "integer", "numeric", minimum=1),
    Column("token_estimate", "integer", "numeric", minimum=1),
    Column("embedding", "array", "structural", nullable=True),
    Column("embedding_spec", "object", "structural"),
    Column("provenance", "object", "structural"),
    Column("verified_by", "array", "structural"),
)


class Table:
    """A declared set of columns, and every question that can be asked of one.

    A table is a *parameter* of a build rather than a global, because the alternative
    is worse than it looks. The chunk contract says every row carries a `char_count`
    of at least 1, which is true and worth enforcing -- and it also means no row this
    corpus can hold ever leaves a measurement absent. The packed column still has to
    spell absence, and the comparison semantics still have to answer for it, so those
    have to be exercised against a corpus whose table admits it. The choice is between
    a second table and a check that stops short of the contract; a second table is a
    declaration, and a weakened check is a hole (`docs/security/laws.md` L2).
    """

    def __init__(self, columns: tuple[Column, ...]) -> None:
        self.columns = tuple(columns)
        self.by_name = {column.name: column for column in self.columns}
        if len(self.by_name) != len(self.columns):
            raise SchemaError("a column table declares a name twice")
        self.indexed = tuple(c.name for c in self.columns if c.role == "indexed")
        self.numeric = tuple(c.name for c in self.columns if c.role == "numeric")
        self.text = tuple(c.name for c in self.columns if c.role == "text")
        self.key = next(c.name for c in self.columns if c.role == "key")
        self.path = next(c.name for c in self.columns if c.role == "path")
        self.filterable = self.indexed + (self.path,) + self.numeric
        self.projectable = (self.key,) + self.filterable + self.text

    def column(self, name: str) -> Column:
        """One column by name, or a refusal that names every column there is."""
        try:
            return self.by_name[name]
        except KeyError:
            raise SchemaError(
                f"column {name!r} is not in the table; the columns are "
                f"{', '.join(c.name for c in self.columns)}"
            ) from None

    def check_row(self, row: dict) -> None:
        """Every declared column of one row, plus a refusal for any field not declared.

        Both halves matter and only together. Checking the declared columns catches a
        field that went bad; refusing an undeclared one catches a field that was
        renamed, which otherwise reads as the old one going absent *and* a new one
        nobody indexes appearing -- two findings for one cause, in the one case where
        the cause is obvious (`docs/security/laws.md` L15).
        """
        if not isinstance(row, dict):
            raise SchemaError(f"a chunk row is an object, not {_type_of(row)}")
        for spec in self.columns:
            check_value(row, spec)
        extra = sorted(set(row) - set(self.by_name))
        if extra:
            raise SchemaError(
                f"chunk {_row_name(row)} carries "
                f"{', '.join(repr(name) for name in extra)}, which the table does not "
                f"declare"
            )

    def cell(self, row: dict, name: str):
        """One column's value, typed, or `None` where the row carries no value.

        The one reader every other reader goes through, so that "absent" means the
        same thing to the packed column, to a projection and to a presence test.
        Anything the declaration does not admit raises here rather than reading as
        absent -- a schema error that arrives downstream as one more null is
        indistinguishable from a chunk that really has no value, and the two need
        different answers.
        """
        spec = self.column(name)
        check_value(row, spec)
        value = row.get(name)
        return None if value is None else value

    def relaxed(self, **overrides) -> "Table":
        """This table with the named columns replaced, for a corpus with its own rules.

        Used to declare a *test* table beside the shipped one, so a state the chunk
        contract forbids can still be built and queried where the format has to answer
        for it. Every column named must already exist: a relaxation is a change to a
        declared column, never a way to introduce an undeclared one.
        """
        replaced = []
        for spec in self.columns:
            change = overrides.get(spec.name)
            replaced.append(dataclasses.replace(spec, **change) if change else spec)
        unknown = sorted(set(overrides) - set(self.by_name))
        if unknown:
            raise SchemaError(f"relaxed() names {', '.join(unknown)}, which this table has not")
        return Table(tuple(replaced))


#: The shipped table -- what `training/schema/chunk-v1.json` declares, in Python.
TABLE = Table(COLUMNS)

#: Derived views. `catalog.py` reads these rather than carrying its own tuples, so
#: adding a column to the table is the whole of adding it to the database.
INDEXED_COLUMNS = TABLE.indexed
NUMERIC_COLUMNS = TABLE.numeric
TEXT_COLUMNS = TABLE.text
KEY_COLUMN = TABLE.key
PATH_COLUMN = TABLE.path

#: Every column a predicate may name. The path column is filterable by prefix and the
#: indexed columns by value, so both are here; `text` columns are projectable but not
#: filterable, and `structural` columns are neither.
FILTERABLE = TABLE.filterable

#: Every column a projection may name -- everything a chunk row actually carries as a
#: scalar a caller can read back, which is the filterable set plus the text columns
#: and the key.
PROJECTABLE = TABLE.projectable


def column(name: str) -> Column:
    """One column of the shipped table by name, or a refusal listing them all.

    The refusal lists the table rather than saying "unknown column", because the
    caller who typed `char_cnt` needs to see `char_count`, and a tool that knows the
    answer and does not print it has chosen to be less useful than it is.
    """
    return TABLE.column(name)


def require_filterable(name: str) -> Column:
    """The column a predicate may name, or a refusal saying which ones it may."""
    found = column(name)
    if found.name not in FILTERABLE:
        raise SchemaError(
            f"column {name!r} is {found.role} and cannot be filtered on; the "
            f"filterable columns are {', '.join(FILTERABLE)}"
        )
    return found


def _type_of(value) -> str:
    """The declared type name for a Python value. `bool` is not `integer`, on purpose.

    `True` is an `int` in Python and would pack into a measurement column as a 1. The
    table says `char_count` holds an integer, and a boolean is not one; saying so here
    rather than at each call site is what keeps the answer the same everywhere.
    """
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return type(value).__name__


def _row_name(row: dict) -> str:
    """How a refusal names the offending row. Its chunk_id when it has one."""
    found = row.get(KEY_COLUMN)
    return repr(found) if isinstance(found, str) else "<no chunk_id>"


def check_value(row: dict, spec: Column) -> None:
    """One column of one row against its declaration, or a refusal naming both.

    Four questions, in the order that makes each refusal say the most useful thing:
    is the key there at all, is `null` allowed if it is, is the type right, and is the
    value inside its declared domain or above its declared minimum.
    """
    if spec.name not in row:
        if spec.required:
            raise SchemaError(
                f"chunk {_row_name(row)} has no {spec.name!r}; the table declares it required"
            )
        return
    value = row[spec.name]
    if value is None:
        if not (spec.nullable or not spec.required):
            raise SchemaError(
                f"chunk {_row_name(row)} has {spec.name}=null; the table declares it not nullable"
            )
        return
    actual = _type_of(value)
    if actual != spec.type:
        raise SchemaError(
            f"chunk {_row_name(row)} has {spec.name}={value!r}, which is {actual} and "
            f"not {spec.type}"
        )
    if spec.role == "key" and value == "":
        raise SchemaError(
            f"a chunk row has {spec.name}=''; the key is what every other artifact "
            f"names the row by, and an empty one names nothing"
        )
    if spec.domain is not None and value not in spec.domain:
        raise SchemaError(
            f"chunk {_row_name(row)} has {spec.name}={value!r}, which is outside the "
            f"declared domain {{{', '.join(repr(v) for v in spec.domain)}}}"
        )
    if spec.minimum is not None and value < spec.minimum:
        raise SchemaError(
            f"chunk {_row_name(row)} has {spec.name}={value!r}, below the declared "
            f"minimum of {spec.minimum}"
        )


def check_row(row: dict) -> None:
    """Every declared column of one row, against the shipped table."""
    TABLE.check_row(row)


def cell(row: dict, name: str):
    """One column's value from the shipped table, or None where the row has none."""
    return TABLE.cell(row, name)
