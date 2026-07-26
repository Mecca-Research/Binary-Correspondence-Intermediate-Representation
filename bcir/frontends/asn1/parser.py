"""Recursive-descent parser for the X.680 (02/2021) notation subset BCIR consumes.

Production names in the comments are X.680's own, so this file can be read against the
standard's grammar rather than against a paraphrase of it.

WHAT IS PARSED. Clause 13 module structure (identifier, tag default, extensibility,
imports/exports), clause 16-31 types: the built-in types, SEQUENCE / SET / CHOICE /
SEQUENCE OF / SET OF, tagged types, and the clause 17 values needed for DEFAULT.

WHAT IS REFUSED, LOUDLY. X.681 information object classes, X.682 constraint expressions
beyond the shapes below, and X.683 parameterization raise `Asn1SyntaxError` naming the
Recommendation that defines them. A front-end that silently skipped them would produce
a type model that disagrees with the module -- which is the one outcome a schema
compiler must never have. Constraints are PARSED AND DISCARDED rather than refused
(§49: a constraint restricts the value set, it does not change the tag or the
structure, so discarding one cannot change a DER encoding) -- with the exception noted
at `_constraint`, where a constraint carries an encoding consequence.
"""

from __future__ import annotations

from . import ast
from .lexer import Asn1SyntaxError, Token, tokenize

#: X.680 Table 1 built-in types that need no extra notation to parse.
_SIMPLE_BUILTINS = {
    "BOOLEAN", "NULL", "REAL", "UTF8String", "NumericString", "PrintableString",
    "TeletexString", "T61String", "VideotexString", "IA5String", "GraphicString",
    "VisibleString", "ISO646String", "GeneralString", "UniversalString", "BMPString",
    "ObjectDescriptor", "UTCTime", "GeneralizedTime", "DATE", "TIME-OF-DAY",
    "DATE-TIME", "DURATION", "EXTERNAL", "TIME", "OID-IRI", "RELATIVE-OID-IRI",
    "RELATIVE-OID",
}

#: Constructs from the companion Recommendations, refused by name rather than by a
#: generic "unexpected token" so the failure tells the reader which phase would add it.
_OUT_OF_SCOPE = {
    "CLASS": "X.681 information object classes",
    "INSTANCE": "X.681 instance-of types",
    "TYPE-IDENTIFIER": "X.681 the TYPE-IDENTIFIER class",
    "ABSTRACT-SYNTAX": "X.681 the ABSTRACT-SYNTAX class",
    "ENCODING-CONTROL": "X.692 encoding control sections",
}


class Parser:
    def __init__(self, text: str, source: str = "<asn1>"):
        self.tokens = tokenize(text, source)
        self.index = 0
        self.source = source

    # --- token plumbing ---------------------------------------------------------------

    @property
    def current(self) -> Token:
        return self.tokens[self.index]

    def peek(self, offset: int = 0) -> Token:
        return self.tokens[min(self.index + offset, len(self.tokens) - 1)]

    def error(self, message: str, token: Token | None = None) -> Asn1SyntaxError:
        tok = token or self.current
        return Asn1SyntaxError(message, tok.line, tok.column, self.source)

    def at(self, kind: str, text: str | None = None, offset: int = 0) -> bool:
        tok = self.peek(offset)
        return tok.kind == kind and (text is None or tok.text == text)

    def at_punct(self, text: str, offset: int = 0) -> bool:
        return self.at("punct", text, offset)

    def at_word(self, text: str, offset: int = 0) -> bool:
        return self.at("reserved", text, offset)

    def take(self) -> Token:
        tok = self.current
        if tok.kind != "end":
            self.index += 1
        return tok

    def accept(self, kind: str, text: str | None = None) -> Token | None:
        if self.at(kind, text):
            return self.take()
        return None

    def expect(self, kind: str, text: str | None = None) -> Token:
        if not self.at(kind, text):
            want = f"{text!r}" if text else kind
            raise self.error(f"expected {want}, found {self.current.text!r}")
        return self.take()

    def expect_punct(self, text: str) -> Token:
        return self.expect("punct", text)

    # --- clause 13: module definition -------------------------------------------------

    def parse_module(self) -> ast.ModuleNode:
        name = self.expect("typereference").text
        oid = self.parse_oid_value() if self.at_punct("{") else None
        self.expect("reserved", "DEFINITIONS")

        tag_default = "EXPLICIT"
        for keyword in ("EXPLICIT", "IMPLICIT", "AUTOMATIC"):
            if self.at_word(keyword):
                self.take()
                self.expect("reserved", "TAGS")
                tag_default = keyword
                break

        extensibility = False
        if self.at_word("EXTENSIBILITY"):
            self.take()
            self.expect("reserved", "IMPLIED")
            extensibility = True

        self.expect_punct("::=")
        self.expect("reserved", "BEGIN")

        exports, imports = self.parse_module_body_header()
        module = ast.ModuleNode(name=name, oid=oid, tag_default=tag_default,
                                extensibility_implied=extensibility,
                                imports=imports, exports=exports)
        while not self.at_word("END"):
            if self.current.kind == "end":
                raise self.error("module is missing its END (X.680 13.1)")
            module.assignments.append(self.parse_assignment())
        self.expect("reserved", "END")
        return module

    def parse_module_body_header(self):
        exports: tuple[str, ...] | None = None
        imports: list[ast.SymbolsFromModule] = []
        if self.at_word("EXPORTS"):
            self.take()
            if self.at_word("ALL"):
                self.take()
            else:
                exports = tuple(self.parse_symbol_list())
            self.expect_punct(";")
        if self.at_word("IMPORTS"):
            self.take()
            while not self.at_punct(";"):
                symbols = tuple(self.parse_symbol_list())
                self.expect("reserved", "FROM")
                module_name = self.expect("typereference").text
                oid = self.parse_oid_value() if self.at_punct("{") else None
                imports.append(ast.SymbolsFromModule(symbols, module_name, oid))
            self.expect_punct(";")
        return exports, tuple(imports)

    def parse_symbol_list(self) -> list[str]:
        symbols: list[str] = []
        while self.current.kind in ("typereference", "identifier"):
            symbols.append(self.take().text)
            # A symbol may be followed by `{}` marking a parameterized reference (X.683).
            if self.at_punct("{") and self.at_punct("}", 1):
                self.take(); self.take()
            if not self.accept("punct", ","):
                break
        return symbols

    def parse_assignment(self):
        tok = self.current
        if self.at_word("ENCODING-CONTROL"):
            raise self.error(f"{_OUT_OF_SCOPE['ENCODING-CONTROL']} are not supported "
                             "(roadmap phase G)")
        if tok.kind == "typereference":
            name = self.take().text
            if self.at_punct("{"):
                raise self.error(
                    f"parameterized type {name!r} needs X.683 parameterization "
                    "(roadmap phase F)")
            self.expect_punct("::=")
            if self.at_word("CLASS"):
                raise self.error(f"{_OUT_OF_SCOPE['CLASS']} are not supported "
                                 "(roadmap phase F)")
            return ast.TypeAssignment(name, self.parse_type())
        if tok.kind == "identifier":
            name = self.take().text
            value_type = self.parse_type()
            self.expect_punct("::=")
            return ast.ValueAssignment(name, value_type, self.parse_value())
        raise self.error(f"expected an assignment, found {tok.text!r}")

    # --- clauses 16-31: types ---------------------------------------------------------

    def parse_type(self):
        node = self.parse_untagged_type()
        # §49: trailing constraints restrict the value set; they do not change the tag
        # or the structure, so they are consumed and discarded.
        while self.at_punct("(") or self.at_word("SIZE") or self.at_word("WITH"):
            self._constraint()
        return node

    def parse_untagged_type(self):
        if self.at_punct("["):
            return self.parse_tagged_type()
        tok = self.current

        if tok.kind == "reserved":
            if tok.text in _OUT_OF_SCOPE:
                raise self.error(f"{_OUT_OF_SCOPE[tok.text]} are not supported")
            if tok.text in _SIMPLE_BUILTINS:
                self.take()
                return ast.Builtin(tok.text)
            if tok.text == "INTEGER":
                self.take()
                return ast.Builtin("INTEGER", self.parse_named_numbers())
            if tok.text == "ENUMERATED":
                self.take()
                named, extensible = self.parse_enumerations()
                return ast.Builtin("ENUMERATED", named, extensible)
            if tok.text == "BIT":
                self.take()
                self.expect("reserved", "STRING")
                return ast.Builtin("BIT STRING", self.parse_named_numbers())
            if tok.text == "OCTET":
                self.take()
                self.expect("reserved", "STRING")
                return ast.Builtin("OCTET STRING")
            if tok.text == "OBJECT":
                self.take()
                self.expect("reserved", "IDENTIFIER")
                return ast.Builtin("OBJECT IDENTIFIER")
            if tok.text == "CHARACTER":
                self.take()
                self.expect("reserved", "STRING")
                return ast.Builtin("CHARACTER STRING")
            if tok.text == "EMBEDDED":
                self.take()
                self.expect("reserved", "PDV")
                return ast.Builtin("EMBEDDED PDV")
            if tok.text == "SEQUENCE":
                return self.parse_sequence()
            if tok.text == "SET":
                return self.parse_set()
            if tok.text == "CHOICE":
                return self.parse_choice()
            if tok.text == "ANY":                          # pragma: no cover - see below
                raise self.error("ANY / ANY DEFINED BY is the pre-1994 open type; the "
                                 "modern spelling needs X.681 (roadmap phase F)")
            raise self.error(f"{tok.text!r} is a reserved word, not a type")

        if tok.kind == "typereference":
            # ANY / ANY DEFINED BY is X.680:1988 notation, WITHDRAWN from the standard
            # (the 2021 edition has no `ANY`). It lexes as an ordinary typereference, so
            # without this it would be reported as an undefined type or as a stray token
            # -- neither of which tells the reader that the module needs an open type and
            # therefore X.681. RFC 5280's AlgorithmIdentifier.parameters is exactly this.
            if tok.text == "ANY":
                raise self.error(
                    "ANY / ANY DEFINED BY is withdrawn X.680:1988 notation; the modern "
                    "spelling is an X.681 open type (e.g. ALGORITHM.&Type), which this "
                    "front-end does not implement (roadmap phase F)")
            # `Module.Type` -- an external type reference (§14.1).
            if self.at_punct(".", 1) and self.peek(2).kind == "typereference":
                module_name = self.take().text
                self.take()
                return ast.TypeRef(self.take().text, module_name)
            name = self.take().text
            if self.at_punct("{"):
                raise self.error(f"parameterized reference {name!r} needs X.683 "
                                 "parameterization (roadmap phase F)")
            return ast.TypeRef(name)

        raise self.error(f"expected a type, found {tok.text!r}")

    def parse_tagged_type(self):
        """§31: `[ EncodingReference : Class ClassNumber ] (IMPLICIT|EXPLICIT)? Type`."""
        self.expect_punct("[")
        tag_class = ""
        for keyword in ("UNIVERSAL", "APPLICATION", "PRIVATE"):
            if self.at_word(keyword):
                tag_class = self.take().text
                break
        number = int(self.expect("number").text)
        self.expect_punct("]")
        mode = None
        if self.at_word("IMPLICIT") or self.at_word("EXPLICIT"):
            mode = self.take().text
        return ast.Tagged(tag_class, number, self.parse_type(), mode)

    def parse_sequence(self):
        self.expect("reserved", "SEQUENCE")
        if self.at_word("OF") or self.at_punct("(") or self.at_word("SIZE"):
            return self._sequence_of()
        return ast.SequenceType(self.parse_component_list())

    def _sequence_of(self):
        while self.at_punct("(") or self.at_word("SIZE"):
            self._constraint()
        self.expect("reserved", "OF")
        name, element = self.parse_maybe_named_type()
        return ast.SequenceOfType(element, name)

    def parse_set(self):
        self.expect("reserved", "SET")
        if self.at_word("OF") or self.at_punct("(") or self.at_word("SIZE"):
            while self.at_punct("(") or self.at_word("SIZE"):
                self._constraint()
            self.expect("reserved", "OF")
            name, element = self.parse_maybe_named_type()
            return ast.SetOfType(element, name)
        return ast.SetType(self.parse_component_list())

    def parse_choice(self):
        self.expect("reserved", "CHOICE")
        return ast.ChoiceType(self.parse_component_list(choice=True))

    def parse_maybe_named_type(self):
        """`Type` or `identifier Type` (§25.1) -- the identifier is documentation."""
        if (self.current.kind == "identifier"
                and not self.at_punct(",", 1) and not self.at_punct("}", 1)):
            return self.take().text, self.parse_type()
        return None, self.parse_type()

    def parse_component_list(self, choice: bool = False) -> tuple[object, ...]:
        self.expect_punct("{")
        items: list[object] = []
        if self.accept("punct", "}"):
            return ()
        while True:
            if self.at_punct("..."):
                self.take()
                items.append(ast.ExtensionMarker())
                # An extension addition group `[[ ... ]]` may follow (§25.1).
                if self.at_punct("[["):
                    self._skip_balanced("[[", "]]")
            elif self.at_word("COMPONENTS"):
                raise self.error("COMPONENTS OF requires component-list inlining, "
                                 "which this front-end does not implement (X.680 25.1)")
            else:
                items.append(self.parse_component(choice))
            if not self.accept("punct", ","):
                break
        self.expect_punct("}")
        return tuple(items)

    def parse_component(self, choice: bool) -> ast.ComponentNode:
        name = self.expect("identifier").text
        node = self.parse_type()
        optional, default, has_default = False, None, False
        if not choice:
            if self.at_word("OPTIONAL"):
                self.take()
                optional = True
            elif self.at_word("DEFAULT"):
                self.take()
                default, has_default = self.parse_value(), True
        return ast.ComponentNode(name, node, optional, default, has_default)

    def parse_named_numbers(self) -> tuple[ast.NamedNumber, ...]:
        """§19.1/§22.1: an optional `{ name(number), ... }` list."""
        if not self.at_punct("{"):
            return ()
        self.take()
        out: list[ast.NamedNumber] = []
        if self.accept("punct", "}"):
            return ()
        while True:
            item = self.expect("identifier").text
            self.expect_punct("(")
            out.append(ast.NamedNumber(item, self.parse_signed_number()))
            self.expect_punct(")")
            if not self.accept("punct", ","):
                break
        self.expect_punct("}")
        return tuple(out)

    def parse_enumerations(self):
        """§20.1: enumeration items may be bare identifiers, and `...` may appear."""
        self.expect_punct("{")
        out: list[ast.NamedNumber] = []
        extensible = False
        next_implicit = 0
        while True:
            if self.at_punct("..."):
                self.take()
                extensible = True
            else:
                item = self.expect("identifier").text
                if self.accept("punct", "("):
                    number = self.parse_signed_number()
                    self.expect_punct(")")
                else:
                    # §20.1: an item without a number takes the next unused value.
                    number = next_implicit
                out.append(ast.NamedNumber(item, number))
                next_implicit = max((n.number for n in out), default=-1) + 1
            if not self.accept("punct", ","):
                break
        self.expect_punct("}")
        return tuple(out), extensible

    def parse_signed_number(self) -> int:
        negative = bool(self.accept("punct", "-"))
        value = int(self.expect("number").text)
        return -value if negative else value

    # --- clause 17: values ------------------------------------------------------------

    def parse_value(self):
        tok = self.current
        if tok.kind == "number" or self.at_punct("-"):
            return ast.IntValue(self.parse_signed_number())
        if tok.kind == "cstring":
            return ast.StrValue(self.take().text)
        if tok.kind == "bstring":
            self.take()
            data = int(tok.text, 2).to_bytes((len(tok.text) + 7) // 8, "big") \
                if tok.text else b""
            return ast.BitsValue(data, len(tok.text))
        if tok.kind == "hstring":
            self.take()
            padded = tok.text + "0" * (len(tok.text) % 2)
            return ast.BitsValue(bytes.fromhex(padded), len(tok.text) * 4)
        if tok.kind == "reserved":
            if tok.text in ("TRUE", "FALSE"):
                self.take()
                return ast.BoolValue(tok.text == "TRUE")
            if tok.text == "NULL":
                self.take()
                return ast.NullValue()
        if tok.kind == "identifier":
            # A bare identifier is an ENUMERATED item or a valuereference; which one is
            # a question about the TYPE, so it stays unresolved until lowering.
            return ast.RefValue(self.take().text)
        if self.at_punct("{"):
            return self.parse_braced_value()
        raise self.error(f"expected a value, found {tok.text!r}")

    def parse_braced_value(self) -> ast.BracedValue:
        """`{ ... }` — recorded in EVERY reading the text admits, not resolved here.

        `{ 1 }` is a valid SEQUENCE OF INTEGER value and a valid one-arc OBJECT
        IDENTIFIER value, and X.680 gives the parser nothing to choose between them --
        the governing type does that, in `lower.value`. So both readings are carried
        and `None` marks a reading the text rules out (a comma makes it not an OID; a
        `name(number)` arc makes it not a value list).
        """
        start = self.index
        self.expect_punct("{")
        items = self._try_value_list(start)
        arcs = self._try_arc_list(start)
        if items is None and arcs is None:
            raise self.error("brace content is neither a value list nor an object "
                             "identifier", self.tokens[start])
        # Leave the cursor after the closing brace regardless of which reading parsed.
        self.index = start
        self._skip_balanced("{", "}")
        return ast.BracedValue(items, arcs)

    def _try_value_list(self, start: int):
        """Re-parse the brace body as `{ v, v, ... }`; None if it cannot be one."""
        self.index = start
        try:
            self.expect_punct("{")
            if self.accept("punct", "}"):
                return ()
            items: list[object] = []
            while True:
                items.append(self.parse_value())
                if not self.accept("punct", ","):
                    break
            self.expect_punct("}")
            return tuple(items)
        except Asn1SyntaxError:
            return None

    def _try_arc_list(self, start: int):
        """Re-parse the brace body as an OID arc list; None if it cannot be one."""
        self.index = start
        try:
            return self.parse_oid_value().arcs
        except Asn1SyntaxError:
            return None

    def parse_oid_value(self) -> ast.OidValue:
        self.expect_punct("{")
        arcs: list[ast.OidArc] = []
        while not self.at_punct("}"):
            tok = self.take()
            if tok.kind == "number":
                arcs.append(ast.OidArc(None, int(tok.text)))
            elif tok.kind == "identifier":
                if self.accept("punct", "("):
                    number = int(self.expect("number").text)
                    self.expect_punct(")")
                    arcs.append(ast.OidArc(tok.text, number))
                else:
                    arcs.append(ast.OidArc(tok.text, None))
            else:
                raise self.error(
                    f"expected an object identifier arc, found {tok.text!r}", tok)
        self.expect_punct("}")
        return ast.OidValue(tuple(arcs))

    # --- clause 49: constraints (consumed, not modelled) --------------------------------

    def _constraint(self) -> None:
        """Consume a constraint.

        Discarding is sound for DER: X.680 §49 defines a constraint as a restriction on
        the VALUE SET, and X.690 encodes a value the same way whether or not a
        constraint admitted it. The one constraint with an encoding consequence is
        CONTAINING/ENCODED BY (§36), which is refused rather than dropped because it
        changes what the contents octets ARE.
        """
        if self.at_word("SIZE") or self.at_word("WITH"):
            self.take()
            if self.at_word("COMPONENT") or self.at_word("COMPONENTS"):
                self.take()
        if not self.at_punct("("):
            return
        depth, start = 0, self.current
        while True:
            if self.current.kind == "end":
                raise self.error("unterminated constraint", start)
            if self.at_word("CONTAINING") or self.at_word("ENCODED"):
                raise self.error(
                    "a CONTAINING / ENCODED BY constraint changes the contents octets "
                    "(X.680 36), so it cannot be discarded like a value-set constraint")
            if self.at_punct("("):
                depth += 1
            elif self.at_punct(")"):
                depth -= 1
                if depth == 0:
                    self.take()
                    return
            self.take()

    def _skip_balanced(self, opener: str, closer: str) -> None:
        depth = 0
        while self.current.kind != "end":
            if self.at_punct(opener):
                depth += 1
            elif self.at_punct(closer):
                depth -= 1
                if depth == 0:
                    self.take()
                    return
            self.take()
        raise self.error(f"unterminated {opener}")


def parse_module(text: str, source: str = "<asn1>") -> ast.ModuleNode:
    """Parse exactly one ModuleDefinition; trailing text is an error, not ignored."""
    parser = Parser(text, source)
    module = parser.parse_module()
    if parser.current.kind != "end":
        raise parser.error(f"unexpected {parser.current.text!r} after END")
    return module


def parse_modules(text: str, source: str = "<asn1>") -> list[ast.ModuleNode]:
    """Parse a file holding one or more modules (X.680 §13.1 ModuleDefinition*)."""
    parser = Parser(text, source)
    out: list[ast.ModuleNode] = []
    while parser.current.kind != "end":
        out.append(parser.parse_module())
    return out


__all__ = ["Parser", "parse_module", "parse_modules"]
