"""P5 — the sparse text surface, as a lossless *view* of the canonical node table.

P5's gate: *"`surface -> canonical -> surface` identity; formatting confined to a side table;
two presentations of one program hash identically."*

§4.2 of the note states the discipline this file has to obey, and the wording matters:

- The **canonical JER is the artifact**. It is what is hashed, signed, stored and compiled.
- Every surface is a **lossless projection** of it, and the round trip must be the identity
  *on the canonical side* — not on the text. Two spellings of one program are supposed to
  converge; demanding the text survive unchanged would make the text authoritative, which is
  the failure §4.2 is written to prevent.
- **Formatting, naming and layout live in a side table**, never in the canonical form, so two
  programs differing only in presentation are byte-identical and hash identically.

So this is *"a typing shortcut with a printer, not a language"*: no macros, no syntax
extensions, no semantics of its own. `parse_surface` returns a `Graph` **and** a separate
`Presentation`, and nothing in the `Presentation` can reach the `Graph`. That separation is
the gate's second clause expressed as a type rather than as a promise, and the tests check it
by parsing two differently-presented spellings and comparing canonical bytes.

**The surface is flat, like the table.** Printing walks the node table in order and parsing
appends to a list; neither recurses. A thousand-node chain therefore costs a thousand lines
and no stack — the same property P1 bought by choosing integer-index edges over nesting, now
inherited by the text rather than re-argued for it.

Syntax, in full::

    (graph
      (roots #0)                          ; entry points, by index or alias
      (program "demo" @entry              ; kind, label, and a presentation-only alias
        :align "64"                       ; an attribute: name and value
        (-> phase #1))                    ; an edge: label and target
      (phase "0"
        (-> claim #2))
      (claim "7" :op "add"))

A node's index is its **position** among the node forms, so the text needs no index
declarations that could disagree with the table. `@alias`, `;` comments and indentation are
the whole of the presentation layer.
"""

from __future__ import annotations

from dataclasses import dataclass

from .graph import Edge, Graph, Node, graph_to_jer, jer_to_graph
from .tags import Asn1Error

#: Characters a bare symbol may carry after the first. A bare symbol never starts with `-`,
#: which is what keeps `->` unambiguous without a lookahead rule.
_BARE_INNER = "_.-"


def _is_bare(text: str) -> bool:
    if not text or not (text[0].isalpha() or text[0] == "_"):
        return False
    return all(ch.isalnum() or ch in _BARE_INNER for ch in text)


_ESCAPES = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\r": "\\r", "\t": "\\t"}
_UNESCAPES = {"\\": "\\", '"': '"', "n": "\n", "r": "\r", "t": "\t"}


def _quote(text: str) -> str:
    out = ['"']
    for ch in text:
        if ch in _ESCAPES:
            out.append(_ESCAPES[ch])
        elif ch < " " or ch == "\x7f":
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _symbol(text: str) -> str:
    """A name printed bare when it can be, quoted when it cannot.

    Which one the printer picks is a *presentation* decision — both spellings parse to the
    same name — so this is one more thing the canonical form is indifferent to.
    """
    return text if _is_bare(text) else _quote(text)


# --- the side table ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Presentation:
    """Formatting, naming and layout — everything the canonical form must not carry.

    Every field here is *discardable*: `print_surface(graph)` with no presentation produces
    a different text from `print_surface(graph, rich)`, and both parse to the same `Graph`.
    That is the property §4.2 asks for, and keeping the record separate from `Graph` is what
    makes it checkable rather than merely intended.
    """

    #: index -> human alias, so an edge can read `@loop` instead of `#7`.
    aliases: tuple[tuple[int, str], ...] = ()
    #: index -> a comment printed after that node's opening line.
    comments: tuple[tuple[int, str], ...] = ()
    #: Comment lines before the first node.
    preamble: tuple[str, ...] = ()
    #: Spaces per nesting level.
    indent: int = 2

    def alias_of(self, index: int) -> str | None:
        for at, name in self.aliases:
            if at == index:
                return name
        return None

    def comment_of(self, index: int) -> str | None:
        for at, text in self.comments:
            if at == index:
                return text
        return None


# --- the reader -------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Token:
    kind: str  # "(" ")" "sym" "str" "attr" "alias" "index" "arrow"
    value: str
    line: int


def _tokenize(text: str) -> tuple[list[_Token], list[tuple[int, str]]]:
    """Tokens, plus every comment paired with the token index it followed.

    Comments are collected here rather than discarded so the parser can attach them to the
    node they belong to. A reader that dropped them would make `Presentation` unable to
    round-trip a commented program, and the side table would then be lossy in the one
    direction a human notices.
    """
    tokens: list[_Token] = []
    comments: list[tuple[int, str]] = []
    at, line = 0, 1
    while at < len(text):
        ch = text[at]
        if ch == "\n":
            line += 1
            at += 1
        elif ch.isspace():
            at += 1
        elif ch == ";":
            end = text.find("\n", at)
            end = len(text) if end < 0 else end
            comments.append((len(tokens), text[at + 1 : end].strip()))
            at = end
        elif ch in "()":
            tokens.append(_Token(ch, ch, line))
            at += 1
        elif ch == '"':
            value, at = _read_string(text, at, line)
            tokens.append(_Token("str", value, line))
        elif text.startswith("->", at):
            tokens.append(_Token("arrow", "->", line))
            at += 2
        elif ch in "@#:":
            at += 1
            quoted = at < len(text) and text[at] == '"'
            if quoted:
                value, at = _read_string(text, at, line)
            else:
                start = at
                while at < len(text) and not text[at].isspace() and text[at] not in "();":
                    at += 1
                value = text[start:at]
            # An EMPTY name is legal in the canonical form — `Attribute.name` is a UTF8String
            # with no size constraint — so `:""` has to be readable or the surface would be
            # unable to spell a document the schema admits, which is exactly the lossiness
            # P5's gate forbids. A bare `:` with nothing after it is still a typo.
            if not value and not quoted:
                raise Asn1Error(f"line {line}: {ch!r} with no name after it")
            kind = {"@": "alias", "#": "index", ":": "attr"}[ch]
            tokens.append(_Token(kind, value, line))
        else:
            start = at
            while at < len(text) and not text[at].isspace() and text[at] not in '();"':
                at += 1
            tokens.append(_Token("sym", text[start:at], line))
    return tokens, comments


def _read_string(text: str, at: int, line: int) -> tuple[str, int]:
    at += 1
    out: list[str] = []
    while True:
        if at >= len(text):
            raise Asn1Error(f"line {line}: a quoted string is never closed")
        ch = text[at]
        if ch == '"':
            return "".join(out), at + 1
        if ch != "\\":
            out.append(ch)
            at += 1
            continue
        at += 1
        if at >= len(text):
            raise Asn1Error(f"line {line}: a string ends in a backslash")
        code = text[at]
        if code in _UNESCAPES:
            out.append(_UNESCAPES[code])
            at += 1
        elif code == "u":
            digits = text[at + 1 : at + 5]
            if len(digits) != 4 or any(d not in "0123456789abcdefABCDEF" for d in digits):
                raise Asn1Error(f"line {line}: \\u needs four hex digits, got {digits!r}")
            out.append(chr(int(digits, 16)))
            at += 5
        else:
            # Refused rather than passed through: an unknown escape that silently became
            # its own second character would make two spellings of one string, and the
            # canonical form is supposed to have exactly one.
            raise Asn1Error(f"line {line}: unknown escape \\{code}")


def parse_surface(text: str) -> tuple[Graph, Presentation]:
    """Read the surface into the canonical graph and a separate presentation record."""
    tokens, comments = _tokenize(text)
    if not tokens:
        raise Asn1Error("an empty surface is not a graph; write `(graph)` for an empty one")
    pos = 0

    def expect(kind: str, what: str) -> _Token:
        nonlocal pos
        if pos >= len(tokens):
            raise Asn1Error(f"the surface ends where {what} was expected")
        token = tokens[pos]
        if token.kind != kind:
            raise Asn1Error(f"line {token.line}: expected {what}, found {token.value!r}")
        pos += 1
        return token

    expect("(", "`(`")
    head = expect("sym", "`graph`")
    if head.value != "graph":
        raise Asn1Error(f"line {head.line}: the top-level form is `graph`, not {head.value!r}")

    # Nodes are collected with their edge targets still as TEXT (`#3` or `@loop`) and
    # resolved once the whole form has been read. A cyclic graph makes forward references
    # the ordinary case, not the exception, so resolving as we go would be the wrong shape.
    pending: list[tuple[str, str, tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]]
    pending = []
    aliases: dict[int, str] = {}
    node_comments: dict[int, str] = {}
    root_refs: list[str] = []
    seen_roots = False
    # Token index of each node's opening paren, so comments can be attached by position.
    node_starts: list[int] = []

    while pos < len(tokens) and tokens[pos].kind != ")":
        start = pos
        expect("(", "a node form or `(roots ...)`")
        head = tokens[pos] if pos < len(tokens) else None
        # `roots` is a keyword only where a node cannot appear, and that is decided by the
        # NEXT token rather than by position: a node's second token is always its quoted
        # label, and the roots form's arguments are always `#index` or `@alias`. Making it
        # positional instead — "`roots` is a keyword until the first node" — leaves a node
        # whose kind is literally `roots` with no spelling at all, which is the lossiness
        # P5's gate exists to forbid. A quoted `("roots" "0")` is always a node.
        follows = tokens[pos + 1] if pos + 1 < len(tokens) else None
        if (
            head is not None
            and head.kind == "sym"
            and head.value == "roots"
            and (follows is None or follows.kind in (")", "index", "alias"))
        ):
            if seen_roots:
                raise Asn1Error(f"line {head.line}: `roots` appears twice")
            seen_roots = True
            pos += 1
            while pos < len(tokens) and tokens[pos].kind in ("index", "alias"):
                root_refs.append(_ref(tokens[pos]))
                pos += 1
            expect(")", "`)` closing `roots`")
            continue
        index = len(pending)
        node_starts.append(start)
        kind_token = tokens[pos] if pos < len(tokens) else None
        if kind_token is None or kind_token.kind not in ("sym", "str"):
            raise Asn1Error(f"line {head.line if head else 0}: a node begins with its kind")
        pos += 1
        label = expect("str", "the node's label, in quotes").value
        attributes: list[tuple[str, str]] = []
        edges: list[tuple[str, str]] = []
        while pos < len(tokens) and tokens[pos].kind != ")":
            token = tokens[pos]
            if token.kind == "alias":
                if index in aliases:
                    raise Asn1Error(f"line {token.line}: node {index} already has an alias")
                if token.value in aliases.values():
                    raise Asn1Error(f"line {token.line}: alias @{token.value} is used twice")
                aliases[index] = token.value
                pos += 1
            elif token.kind == "attr":
                pos += 1
                attributes.append(
                    (token.value, expect("str", "the attribute's value, in quotes").value)
                )
            elif token.kind == "(":
                pos += 1
                expect("arrow", "`->` opening an edge")
                label_token = tokens[pos] if pos < len(tokens) else None
                if label_token is None or label_token.kind not in ("sym", "str"):
                    raise Asn1Error(f"line {token.line}: an edge needs a label")
                pos += 1
                target = tokens[pos] if pos < len(tokens) else None
                if target is None or target.kind not in ("index", "alias"):
                    raise Asn1Error(f"line {token.line}: an edge targets #index or @alias")
                pos += 1
                edges.append((label_token.value, _ref(target)))
                expect(")", "`)` closing an edge")
            else:
                raise Asn1Error(f"line {token.line}: {token.value!r} is not a node clause")
        expect(")", "`)` closing a node")
        pending.append((kind_token.value, label, tuple(attributes), tuple(edges)))
    expect(")", "`)` closing `graph`")
    if pos != len(tokens):
        raise Asn1Error(f"line {tokens[pos].line}: text after the closing `)`")

    by_alias = {name: at for at, name in aliases.items()}

    def target(ref: str) -> int:
        """An index or an alias, resolved. Range is deliberately NOT checked here.

        `Edge.target` is an INTEGER in the schema and `resolve` reports an out-of-range one
        as an `EdgeFault` — a value, not a fault. A surface that refused it would be stricter
        than the canonical form, leaving a decodable document with no spelling; the reader's
        job is to carry what the schema admits and let the enrichment pass judge it.
        """
        if ref.startswith("@"):
            name = ref[1:]
            if name not in by_alias:
                raise Asn1Error(f"@{name} is referenced but never defined")
            return by_alias[name]
        try:
            return int(ref)
        except ValueError:
            raise Asn1Error(f"#{ref} is not a node index") from None

    resolved = tuple(
        Node(
            kind=kind,
            label=label,
            attributes=attributes,
            edges=tuple(Edge(edge_label, target(ref)) for edge_label, ref in edges),
        )
        for kind, label, attributes, edges in pending
    )

    for at, text_ in comments:
        index = _node_at(node_starts, at)
        if index is None:
            continue
        node_comments.setdefault(index, text_)
    preamble = tuple(t for at, t in comments if _node_at(node_starts, at) is None)

    graph = Graph(nodes=resolved, roots=tuple(target(ref) for ref in root_refs))
    presentation = Presentation(
        aliases=tuple(sorted(aliases.items())),
        comments=tuple(sorted(node_comments.items())),
        preamble=preamble,
    )
    return graph, presentation


def _ref(token: _Token) -> str:
    return f"@{token.value}" if token.kind == "alias" else token.value


def _node_at(starts: list[int], token_index: int) -> int | None:
    """Which node form a comment at `token_index` sits inside, or None for the preamble."""
    found = None
    for index, start in enumerate(starts):
        if start < token_index:
            found = index
        else:
            break
    return found


# --- the printer ------------------------------------------------------------------------------


def print_surface(graph: Graph, presentation: Presentation | None = None) -> str:
    """Render the canonical graph as surface text. Iterative — the table is flat."""
    view = presentation or Presentation()
    pad = " " * max(0, view.indent)
    lines = [f"; {comment}" for comment in view.preamble]

    def ref(index: int) -> str:
        alias = view.alias_of(index)
        return f"@{alias}" if alias else f"#{index}"

    body: list[str] = []
    if graph.roots:
        body.append(f"{pad}(roots {' '.join(ref(r) for r in graph.roots)})")
    for index, node in enumerate(graph.nodes):
        head = f"{pad}({_symbol(node.kind)} {_quote(node.label)}"
        alias = view.alias_of(index)
        if alias:
            head += f" @{alias}"
        comment = view.comment_of(index)
        if comment:
            head += f"  ; {comment}"
        clauses = [f"{pad * 2}:{_symbol(name)} {_quote(value)}" for name, value in node.attributes]
        clauses += [
            f"{pad * 2}(-> {_symbol(edge.label)} {ref(edge.target)})" for edge in node.edges
        ]
        if clauses:
            body.append("\n".join([head, *clauses[:-1], clauses[-1] + ")"]))
        elif comment:
            # The comment already ended the line, so the closing paren needs its own.
            body.append(f"{head}\n{pad})")
        else:
            body.append(head + ")")
    if not body:
        return "\n".join([*lines, "(graph)"]) + "\n"
    return "\n".join([*lines, "(graph", *body]) + ")\n"


# --- the ends of the projection ----------------------------------------------------------------


def surface_to_graph(text: str) -> Graph:
    """The canonical graph alone — the presentation deliberately discarded."""
    return parse_surface(text)[0]


def graph_to_surface(graph: Graph, presentation: Presentation | None = None) -> str:
    return print_surface(graph, presentation)


def surface_to_jer(text: str, **kwargs) -> bytes:
    return graph_to_jer(surface_to_graph(text), **kwargs)


def jer_to_surface(data: bytes, presentation: Presentation | None = None, **kwargs) -> str:
    return print_surface(jer_to_graph(data, **kwargs), presentation)


__all__ = [
    "Presentation",
    "graph_to_surface",
    "jer_to_surface",
    "parse_surface",
    "print_surface",
    "surface_to_graph",
    "surface_to_jer",
]
