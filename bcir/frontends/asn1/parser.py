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

from bcir.asn1 import constraints

from . import ast
from .lexer import Asn1SyntaxError, Token, tokenize

#: Distinguishes "this endpoint is MIN/MAX" (None) from "this form is not
#: represented" -- conflating them would silently turn an unparsed constraint
#: into an unbounded one, which is the same value set but a different intent.
_UNREPRESENTABLE = object()

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
        #: Class definitions seen so far, by name. X.681 §11.4 makes reading an object body
        #: depend on whether its class declared WITH SYNTAX, so the class has to be in hand
        #: BEFORE its objects are parsed -- which it is, since a module defines a class
        #: before the objects of that class. An object whose class has not been seen falls
        #: back to the §7 capitalisation rule rather than failing.
        self.classes_seen: dict[str, ast.ClassAssignment] = {}

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
        # X.681 §11.1/§12.1: an information object or object set is `<name> CLASSNAME ::=
        # { ... }` -- TWO references before the assignment, where a type assignment has
        # one. That two-token lookahead is the whole discriminator.
        if (tok.kind in ("typereference", "identifier")
                and self.peek(1).kind == "typereference"
                and self.at_punct("::=", 2)):
            return self.parse_object_or_set()
        # X.683 §8.2: a parameterized assignment is an ordinary one with a ParameterList
        # after the initial reference. It is spotted here, before the X.681 two-token
        # lookahead below, because `Set {P} CLASS ::= {...}` puts the list BETWEEN the two
        # references and would otherwise not look like an object set assignment at all.
        if (tok.kind in ("typereference", "identifier") and self.at_punct("{", 1)):
            save = self.index
            name = self.take().text
            params, governors = self._parameter_list()
            if params is not None:
                body = self.parse_assignment_body(name)
                return ast.ParameterizedAssignment(name, params, body, governors)
            self.index = save
        if tok.kind == "typereference":
            name = self.take().text
            self.expect_punct("::=")
            if self.at_word("CLASS"):
                assignment = self.parse_class(name)
                self.classes_seen[assignment.name] = assignment
                return assignment
            # `Set CLASS ::= { obj | obj }` -- an object SET assignment (X.681 §12.1).
            # It is told from a type assignment by the class name standing where a type
            # would: `Algorithms ALGORITHM ::= {...}` has already consumed `Algorithms`.
            return ast.TypeAssignment(name, self.parse_type())
        if tok.kind == "identifier":
            name = self.take().text
            value_type = self.parse_type()
            self.expect_punct("::=")
            return ast.ValueAssignment(name, value_type, self.parse_value())
        raise self.error(f"expected an assignment, found {tok.text!r}")

    def parse_assignment_body(self, name: str):
        """Parse whatever follows an assignment's (already consumed) reference name."""
        if self.at("typereference") and self.at_punct("::=", 1):
            return self.parse_object_or_set(name)
        if self.at_punct("::="):
            self.take()
            if self.at_word("CLASS"):
                assignment = self.parse_class(name)
                self.classes_seen[assignment.name] = assignment
                return assignment
            return ast.TypeAssignment(name, self.parse_type())
        value_type = self.parse_type()
        self.expect_punct("::=")
        return ast.ValueAssignment(name, value_type, self.parse_value())

    def _parameter_list(self):
        """X.683 §8.3 `{ [Governor ":"] DummyReference , ... }`, or (None, ()).

        Returns (None, ()) without consuming when the braces are NOT a parameter list --
        `Name {1 2 3}` is a value assignment's braced value and `Name {a(1)}` an
        enumeration, so the shape has to be checked rather than assumed from the brace.
        """
        save = self.index
        if not self.at_punct("{"):
            return None, ()
        self.take()
        params: list[str] = []
        governors: list[object] = []
        while not self.at_punct("}"):
            governor = None
            mark = self.index
            if not (self.at_punct(",", 1) or self.at_punct("}", 1)):
                # A Governor precedes the ":" (§8.3). Parse it as a type and require the
                # colon; anything else means these braces were never a parameter list.
                try:
                    governor = self.parse_type()
                except Asn1SyntaxError:
                    self.index = save
                    return None, ()
                if not self.at_punct(":"):
                    self.index = save
                    return None, ()
                self.take()
            if self.current.kind not in ("typereference", "identifier", "fieldreference"):
                self.index = save
                return None, ()
            params.append(self.take().text)
            governors.append(governor)
            if not self.accept("punct", ","):
                break
        if not self.at_punct("}") or not params:
            self.index = save
            return None, ()
        self.take()
        # A ParameterList is followed by the rest of the assignment; if what follows cannot
        # begin one, the braces were something else entirely.
        if not (self.at_punct("::=") or self.at("typereference")
                or self.at("reserved") or self.at("identifier")):
            self.index = save
            return None, ()
        _ = mark
        return tuple(params), tuple(governors)

    # --- X.681: information object classes, objects, and object sets ------------------

    def parse_class(self, name: str) -> ast.ClassAssignment:
        """X.681 §9.1 `CLASS { &field ..., ... }`, plus the WITH SYNTAX clause.

        WITH SYNTAX defines a *user-friendly notation* for writing objects of the class
        (§10). It changes how an object is spelled, never what an encoding looks like -- but
        it is not discardable: §11.4 makes an object of a class that HAS a WithSyntaxSpec use
        DefinedSyntax, so `{"A" 1 INTEGER}` can only be read back into field settings by
        knowing the syntax list's field order. The literal words between the fields are
        decoration and are dropped; the `&field` order is kept.
        """
        self.expect("reserved", "CLASS")
        self.expect_punct("{")
        fields: list[ast.ClassField] = []
        while not self.at_punct("}"):
            token = self.expect("fieldreference")
            field_name = token.text
            if len(field_name) < 2:
                raise self.error("a field reference needs a name after '&'", token)
            is_type_field = field_name[1].isupper()
            declared = None
            if not is_type_field and not (self.at_punct(",") or self.at_punct("}")
                                          or self.at_word("OPTIONAL")
                                          or self.at_word("UNIQUE")):
                declared = self.parse_type()
            unique = bool(self.accept("reserved", "UNIQUE"))
            optional = bool(self.accept("reserved", "OPTIONAL"))
            if self.at_word("DEFAULT"):
                self.take()
                self.parse_value() if not self.at_punct("{") else self.parse_value()
            fields.append(ast.ClassField(field_name, is_type_field, declared, optional,
                                        unique))
            if not self.accept("punct", ","):
                break
        self.expect_punct("}")
        syntax: list[str] = []
        if self.at_word("WITH"):
            self.take()
            self.expect("reserved", "SYNTAX")
            start = self.index
            self._skip_balanced("{", "}")
            # Everything between the braces except the braces themselves. `[` and `]` mark
            # an OPTIONAL group (§10.5); they are kept so the object reader can tell an
            # omitted group from a mismatch.
            syntax = [self.tokens[i].text for i in range(start + 1, self.index - 1)]
        return ast.ClassAssignment(name, tuple(fields), tuple(syntax))

    def parse_object_or_set(self, name: str | None = None):
        """X.681 §11.1 an information object, §12.1 an object set.

        Both are recorded rather than interpreted: their content selects WHICH type an
        open type contains, which is X.682's table-constraint machinery. Recording the
        names keeps the module's shape honest -- a later phase can resolve them without
        the parser having quietly dropped them.
        """
        name = self.take().text if name is None else name
        object_class = self.expect("typereference").text
        self.expect_punct("::=")
        start = self.index
        if not self.at_punct("{"):
            self.take()
            return ast.ObjectAssignment(name, object_class,
                                        self._raw_span(start, self.index))
        # A capitalised name denotes an object SET (§12.1); a lower-case one an object.
        is_set = name[0].isupper()
        cls = self.classes_seen.get(object_class)
        try:
            if is_set:
                elements, extensible = self._parse_object_set_body(cls)
            else:
                settings = self._parse_object_body(cls)
        except Asn1SyntaxError:
            # An object or set this front-end cannot read yet must not fail the module:
            # X.681 §11.3 admits ObjectFromObject and ParameterizedObject forms that need
            # X.683. Fall back to recording the raw span, which is what the printer and the
            # round-trip law consume -- the associated table simply gains no row from it,
            # and a table constraint naming it will say so rather than resolve wrongly.
            self.index = start
            self._skip_balanced("{", "}")
            raw = self._raw_span(start, self.index)
            members = tuple(
                self.tokens[i].text for i in range(start, self.index)
                if self.tokens[i].kind in ("typereference", "identifier"))
            if is_set:
                return ast.ObjectSetAssignment(name, object_class, members, raw)
            return ast.ObjectAssignment(name, object_class, raw)
        raw = self._raw_span(start, self.index)
        members = tuple(
            self.tokens[i].text for i in range(start, self.index)
            if self.tokens[i].kind in ("typereference", "identifier"))
        if is_set:
            return ast.ObjectSetAssignment(name, object_class, members, raw,
                                           tuple(elements), extensible)
        return ast.ObjectAssignment(name, object_class, raw, tuple(settings))

    def _parse_object_body(self, cls) -> list:
        """X.681 §11.5 DefaultSyntax `{&field setting, ...}` / §11.6 DefinedSyntax.

        Which one applies is decided by the CLASS, not by looking at the tokens: §11.4 says
        DefaultSyntax iff the class has no WithSyntaxSpec. A class with WITH SYNTAX spells
        its objects positionally (`{"A" 1 INTEGER}`), so the field names come from the
        syntax list rather than from the object body.
        """
        self.expect_punct("{")
        settings: list[ast.FieldSetting] = []
        if cls is not None and cls.with_syntax:
            for token in cls.with_syntax:
                if self.at_punct("}"):
                    break                                   # an omitted OPTIONAL group
                if token in ("[", "]"):
                    continue                                # §10.5 optional-group brackets
                if token.startswith("&"):
                    settings.append(ast.FieldSetting(
                        token, self._parse_setting(cls, token)))
                elif self.current.text == token:
                    self.take()                             # a literal word of the syntax
                else:
                    # The object does not follow its class's DefinedSyntax. Raising sends
                    # `parse_object_or_set` down the raw-span fallback rather than producing
                    # a half-read object that would contribute a wrong row to the table.
                    raise self.error(
                        f"object does not match the WITH SYNTAX of its class: expected "
                        f"{token!r} (X.681 11.4/11.6)")
                self.accept("punct", ",")
        else:
            while not self.at_punct("}"):
                field_name = self.expect("fieldreference").text
                settings.append(ast.FieldSetting(
                    field_name, self._parse_setting(cls, field_name)))
                if not self.accept("punct", ","):
                    break
        self.expect_punct("}")
        return settings

    def _parse_setting(self, cls, field_name: str):
        """§11.7: a Setting is a TYPE for a type field and a VALUE for a value field.

        The distinction cannot be made from the tokens alone -- `INTEGER` is a type and `1`
        is a value, but a value field's setting may itself be a defined value whose name
        looks like a type reference. The class definition is the authority, and the leading
        capital of the field name is the fallback X.681 §7 gives when the class is unknown.
        """
        is_type_field = None
        if cls is not None:
            for field in cls.fields:
                if field.name == field_name:
                    is_type_field = field.is_type_field
                    break
        if is_type_field is None:
            is_type_field = len(field_name) > 1 and field_name[1].isupper()
        return self.parse_type() if is_type_field else self.parse_value()

    def _parse_object_set_body(self, cls):
        """X.681 §12.3 `{ obj | obj, ... }`.

        The separator is `|` (union) but real modules also use `,`; both are accepted since
        §12.3 NOTE 1 makes the set the union of its elements either way.
        """
        self.expect_punct("{")
        elements: list[object] = []
        extensible = False
        while not self.at_punct("}"):
            if self.at_punct("..."):
                self.take()
                extensible = True                           # §12.3
            elif self.at_punct("{"):
                elements.append(tuple(self._parse_object_body(cls)))
            elif self.at("typereference") or self.at("identifier"):
                elements.append(self.take().text)           # a defined object / object set
            else:
                raise self.error("an object set element must be an object, a reference "
                                 "or '...' (X.681 12.3)")
            if not (self.accept("punct", "|") or self.accept("punct", ",")):
                break
        self.expect_punct("}")
        return elements, extensible

    def _parse_table_constraint(self):
        """X.682 §10.3/§10.7, when the next tokens are `({ObjectSet} [{@...}])`.

        Returns None when there is no table constraint, leaving the ordinary constraint path
        (and its subtype constraints) untouched -- an ObjectClassFieldType may carry either.
        """
        if not (self.at_punct("(") and self.at_punct("{", 1)):
            return None
        save = self.index
        self.take()                                        # "("
        self.take()                                        # "{"
        if not self.at("typereference"):
            self.index = save                              # a braced VALUE, not an ObjectSet
            return None
        object_set = self.take().text
        if not self.at_punct("}"):
            self.index = save
            return None
        self.take()                                        # "}"
        ats: list[str] = []
        if self.at_punct("{"):                             # §10.7 ComponentRelationConstraint
            self.take()
            while not self.at_punct("}"):
                if not self.at_punct("@"):
                    self.index = save
                    return None
                self.take()
                # §10.10: the dots after `@` count enclosing levels, so they are kept
                # verbatim -- `@.x` and `@...x` mean different parents.
                text = "@"
                while self.at_punct("."):
                    self.take()
                    text += "."
                parts = [self.expect("identifier").text]
                while self.at_punct("."):
                    self.take()
                    parts.append(self.expect("identifier").text)
                ats.append(text + ".".join(parts))
                if not self.accept("punct", ","):
                    break
            self.expect_punct("}")
        if not self.at_punct(")"):
            self.index = save
            return None
        self.take()                                        # ")"
        return ast.TableConstraintNode(object_set, tuple(ats))

    def _raw_span(self, start: int, stop: int) -> str:
        """The tokens in [start, stop) as normalized text.

        Normalized rather than verbatim because the round-trip law compares ASTs: what has
        to survive is the token sequence, not the author's whitespace. Re-lexing this
        string yields the same tokens, so the law holds without the parser having to keep
        byte offsets.
        """
        out: list[str] = []
        for index in range(start, stop):
            tok = self.tokens[index]
            out.append(f'"{tok.text}"' if tok.kind == "cstring" else tok.text)
        return " ".join(out)

    # --- clauses 16-31: types ---------------------------------------------------------

    def parse_type(self):
        node = self.parse_untagged_type()
        # §49: a trailing constraint restricts the value set. It changes no tag and no
        # structure -- so DER never sees it -- but OER and PER choose the encoding from
        # it, so it is attached to the node rather than dropped. A serial application of
        # constraints (§49.9) intersects: each one further restricts the last.
        collected = []
        while self.at_punct("(") or self.at_word("SIZE") or self.at_word("WITH") \
                or self.at_word("FROM"):
            built = self._constraint()
            if built is not None:
                collected.append(built)
        if collected:
            node = ast.Constrained(node, tuple(collected))
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
                named, extensible, additions = self.parse_enumerations()
                return ast.Builtin("ENUMERATED", named, extensible, additions)
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
            raise self.error(f"{tok.text!r} is a reserved word, not a type")

        if tok.kind == "typereference":
            # ANY / ANY DEFINED BY is withdrawn X.680:1988 notation -- the 2021 edition
            # dropped both `ANY` and `DEFINED` from the reserved words of Table 3, which
            # is why they arrive here as ordinary typereferences and are matched on TEXT.
            # RFC 5280's 1988 module still uses the spelling, and it means exactly what
            # X.681 §14 calls an open type, so accepting it is not accepting a dialect:
            # both spellings lower to the same OpenType.
            if tok.text == "ANY":
                self.take()
                governed = None
                if self.at("typereference", "DEFINED"):
                    self.take()
                    self.expect("reserved", "BY")
                    governed = self.expect("identifier").text
                return ast.OpenTypeNode(governed_by=governed)
            # `CLASS.&Field` -- an information object class field reference. A TYPE
            # field (`&Type`, capitalised after the ampersand) is an OPEN TYPE; a VALUE
            # field (`&id`) has whatever type the class declared for it, which the
            # lowering resolves from the class definition.
            if self.at_punct(".", 1) and self.peek(2).kind == "fieldreference":
                class_name = self.take().text
                self.take()
                field = self.take().text
                # X.682 §10.1: a TABLE CONSTRAINT may only be applied to an
                # ObjectClassFieldType, so this is the one place it can appear. It is parsed
                # here rather than by the general constraint path because `({Set}{@a})` is
                # not an ElementSetSpec -- the general path would read `{Set}` as a braced
                # value and lose the AtNotation entirely.
                table = self._parse_table_constraint()
                return ast.OpenTypeNode(object_class=class_name, field=field, table=table)
            # `Module.Type` -- an external type reference (§14.1).
            if self.at_punct(".", 1) and self.peek(2).kind == "typereference":
                module_name = self.take().text
                self.take()
                return ast.TypeRef(self.take().text, module_name)
            name = self.take().text
            if self.at_punct("{"):                         # X.683 §9.2 ParameterizedType
                return ast.ParameterizedRef(name, self._actual_parameter_list())
            return ast.TypeRef(name)

        raise self.error(f"expected a type, found {tok.text!r}")

    def _actual_parameter_list(self) -> tuple:
        """X.683 §9.5 `{ ActualParameter , ... }`.

        An ActualParameter may be a Type, a Value, an ObjectSet or a class reference (§9.5).
        A bare reference is kept as its NAME rather than being forced into a type node: at
        parse time there is nothing to say whether `Supported` is a type or an object set,
        and §9.6 makes that the instantiation's business, not the parser's.
        """
        self.expect_punct("{")
        actuals: list[object] = []
        while not self.at_punct("}"):
            if (self.current.kind in ("typereference", "identifier")
                    and (self.at_punct(",", 1) or self.at_punct("}", 1))):
                actuals.append(self.take().text)
            elif self.at_punct("{"):
                start = self.index
                self._skip_balanced("{", "}")
                actuals.append(self._raw_span(start, self.index))
            else:
                actuals.append(self.parse_type())
            if not self.accept("punct", ","):
                break
        self.expect_punct("}")
        return tuple(actuals)

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
        # `SEQUENCE SIZE (1..64) OF T` -- the constraint bounds the OCCURRENCE COUNT
        # (§51.5.2), so it belongs to the sequence-of type and not to its element.
        collected = []
        while self.at_punct("(") or self.at_word("SIZE"):
            built = self._constraint()
            if built is not None:
                collected.append(built)
        self.expect("reserved", "OF")
        name, element = self.parse_maybe_named_type()
        node = ast.SequenceOfType(element, name)
        return ast.Constrained(node, tuple(collected)) if collected else node

    def parse_set(self):
        self.expect("reserved", "SET")
        if self.at_word("OF") or self.at_punct("(") or self.at_word("SIZE"):
            collected = []
            while self.at_punct("(") or self.at_word("SIZE"):
                built = self._constraint()
                if built is not None:
                    collected.append(built)
            self.expect("reserved", "OF")
            name, element = self.parse_maybe_named_type()
            node = ast.SetOfType(element, name)
            return ast.Constrained(node, tuple(collected)) if collected else node
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
            elif self.at_punct("[["):
                items.append(self.parse_extension_group(choice))
            elif self.at_word("COMPONENTS"):
                raise self.error("COMPONENTS OF requires component-list inlining, "
                                 "which this front-end does not implement (X.680 25.1)")
            else:
                items.append(self.parse_component(choice))
            if not self.accept("punct", ","):
                break
        self.expect_punct("}")
        return tuple(items)

    def parse_extension_group(self, choice: bool) -> ast.ExtensionGroup:
        """`[[ a INTEGER, b BOOLEAN OPTIONAL ]]` (X.680 §25.1).

        A version number may prefix the members (`[[ 2: a INTEGER ]]`); it identifies the
        version that added the group and has no effect on the encoding, so it is consumed
        and dropped rather than modelled.
        """
        self.expect_punct("[[")
        if self.at("number") and self.at_punct(":", 1):
            self.take(); self.take()
        members: list[object] = []
        while not self.at_punct("]]"):
            members.append(self.parse_component(choice))
            if not self.accept("punct", ","):
                break
        self.expect_punct("]]")
        return ast.ExtensionGroup(tuple(members))

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
        """§20.1: enumeration items may be bare identifiers, and `...` may appear.

        Returns `(root, extensible, additions)`. The split is kept because X.691 §14 gives an
        item after the marker a different encoding from one before it; flattening the two made
        an extension item indistinguishable from a root value downstream.
        """
        self.expect_punct("{")
        out: list[ast.NamedNumber] = []
        additions: list[ast.NamedNumber] = []
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
                    # §20.1: an item without a number takes the next unused value, counted
                    # across the root AND the additions -- the marker does not restart it.
                    number = next_implicit
                (additions if extensible else out).append(ast.NamedNumber(item, number))
                next_implicit = max((n.number for n in out + additions), default=-1) + 1
            if not self.accept("punct", ","):
                break
        self.expect_punct("}")
        return tuple(out), extensible, tuple(additions)

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

    def _constraint(self):
        """Parse a constraint into the model (X.680 clauses 49–51), or None if unusable.

        Constraints used to be consumed and thrown away, which was sound for DER — §49
        defines a constraint as a restriction on the VALUE SET, and X.690 encodes a value
        the same way regardless. It is NOT sound for OER or PER, which choose the encoding
        FROM the constraint, so they are now built.

        A form this model does not represent (an inner type constraint, a pattern, a table
        constraint) is consumed and reported as None rather than refused: it still cannot
        change a DER encoding, and for OER an unrepresented constraint simply leaves the
        type unconstrained — which is the SAFE direction, because the length-prefixed form
        can carry every value the narrower form could.

        Two X.682 forms are modelled rather than skipped: §11's CONTAINING / ENCODED BY,
        which says what the contents octets ARE and so cannot be dropped, and §9's
        CONSTRAINED BY, which is recorded because §9 NOTE 1 calls it a form of comment and
        deleting an author's stated intent is not the same as ignoring it.
        """
        if self.at_punct("(") and (self.at_word("CONTAINING", 1)
                                   or self.at_word("ENCODED", 1)):
            return self._contents_constraint()
        if self.at_punct("(") and self.at_word("CONSTRAINED", 1):
            return self._user_defined_constraint()
        is_size = is_alphabet = False
        if self.at_word("SIZE"):
            self.take()
            is_size = True
        elif self.at_word("FROM"):
            self.take()
            is_alphabet = True
        elif self.at_word("WITH"):
            self.take()
            if self.at_word("COMPONENT") or self.at_word("COMPONENTS"):
                self.take()
        if not self.at_punct("("):
            return None
        start = self.index
        try:
            inner = self._element_set_specs()
        except Asn1SyntaxError:
            inner = None
            self.index = start
            self._consume_constraint()
        if inner is None:
            return None
        if is_size:
            return constraints.Size(inner)
        if is_alphabet:
            return constraints.PermittedAlphabet(inner)
        return inner

    def _contents_constraint(self):
        """X.682 §11.1: `( CONTAINING Type [ENCODED BY Value] )` or `( ENCODED BY Value )`."""
        self.expect_punct("(")
        contained = None
        encoded_by = None
        if self.at_word("CONTAINING"):
            self.take()
            contained = self.parse_type()
        if self.at_word("ENCODED"):
            self.take()
            self.expect("reserved", "BY")
            encoded_by = self.parse_value()               # §11.2: an OBJECT IDENTIFIER
        self.expect_punct(")")
        return ast.ContentsConstraintNode(contained, encoded_by)

    def _user_defined_constraint(self):
        """X.682 §9.1: `( CONSTRAINED BY { ... } )`, recorded verbatim."""
        self.expect_punct("(")
        self.expect("reserved", "CONSTRAINED") if self.at("reserved", "CONSTRAINED") \
            else self.take()
        if self.at_word("BY"):
            self.take()
        start = self.index
        if self.at_punct("{"):
            self._skip_balanced("{", "}")
        raw = self._raw_span(start, self.index)
        self.expect_punct(")")
        return ast.UserDefinedConstraintNode(raw)

    def _consume_constraint(self) -> None:
        """Skip a constraint this model cannot represent, refusing only §36's."""
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

    def _element_set_specs(self):
        """§49.4 `RootElementSetSpec [ "," "..." [ "," AdditionalElementSetSpec ] ]`."""
        self.expect_punct("(")
        if self.at_word("CONTAINING") or self.at_word("ENCODED"):
            raise self.error(
                "a CONTAINING / ENCODED BY constraint changes the contents octets "
                "(X.680 36), so it cannot be discarded like a value-set constraint")
        root = self._unions()
        extensible = False
        if self.accept("punct", ","):
            self.expect_punct("...")
            extensible = True
            if self.accept("punct", ","):
                self._unions()          # the additional set: not OER-visible either
        self.expect_punct(")")
        if root is None:
            return None
        return constraints.Extensible(root) if extensible else root

    def _unions(self):
        """§49.6 `A | B`, spelled `|` or `UNION`."""
        parts = [self._intersections()]
        while self.at_punct("|") or self.at_word("UNION"):
            self.take()
            parts.append(self._intersections())
        if any(part is None for part in parts):
            return None
        return parts[0] if len(parts) == 1 else constraints.Union(tuple(parts))

    def _intersections(self):
        """§49.7 `A ^ B`, spelled `^` or `INTERSECTION`; §49.8 `EXCEPT`."""
        parts = [self._constraint_element()]
        while self.at_punct("^") or self.at_word("INTERSECTION"):
            self.take()
            parts.append(self._constraint_element())
        if self.at_word("EXCEPT"):
            # §49.8 removes values. The encoder must not narrow on the strength of an
            # EXCEPT: the remaining set is a subset, so the bounds it would compute could
            # exclude a value the parent still permits. Report unrepresentable.
            self.take()
            self._constraint_element()
            return None
        if any(part is None for part in parts):
            return None
        return parts[0] if len(parts) == 1 else constraints.Intersection(tuple(parts))

    def _constraint_element(self):
        """§51 SubtypeElements, restricted to the forms this model represents."""
        if self.at_punct("("):                     # a parenthesised element set
            return self._element_set_specs()
        if self.at_word("SIZE"):
            self.take()
            inner = self._element_set_specs()
            return None if inner is None else constraints.Size(inner)
        if self.at_word("FROM"):
            self.take()
            inner = self._element_set_specs()
            return None if inner is None else constraints.PermittedAlphabet(inner)
        if self.at_word("ALL") or self.at_word("WITH") or self.at_word("PATTERN") \
                or self.at_word("SETTINGS") or self.at_word("INCLUDES"):
            self._skip_element()
            return None
        low = self._endpoint_value(lower=True)
        if low is _UNREPRESENTABLE:
            self._skip_element()
            return None
        lower_open = bool(self.accept("punct", "<"))
        if not self.at_punct(".."):
            if lower_open:                         # `v <` with no range is not a form
                return None
            return constraints.SingleValue(low)
        self.take()
        upper_open = bool(self.accept("punct", "<"))
        high = self._endpoint_value(lower=False)
        if high is _UNREPRESENTABLE:
            return None
        return constraints.ValueRange(low, high, lower_open, upper_open)

    def _endpoint_value(self, lower: bool):
        """§51.4.4 `Value | MIN` / `Value | MAX`; None is the unbounded endpoint."""
        if self.at_word("MIN") if lower else self.at_word("MAX"):
            self.take()
            return None
        tok = self.current
        if tok.kind == "number" or self.at_punct("-"):
            return self.parse_signed_number()
        if tok.kind == "cstring":
            self.take()
            return tok.text                        # a permitted-alphabet endpoint
        if tok.kind == "reserved" and tok.text in ("TRUE", "FALSE"):
            self.take()
            return tok.text == "TRUE"
        return _UNREPRESENTABLE

    def _skip_element(self) -> None:
        """Consume one element of a set spec whose form this model does not represent."""
        depth = 0
        while self.current.kind != "end":
            if self.at_punct("("):
                depth += 1
            elif self.at_punct(")"):
                if depth == 0:
                    return
                depth -= 1
            elif depth == 0 and (self.at_punct("|") or self.at_punct("^")
                                 or self.at_punct(",")):
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
