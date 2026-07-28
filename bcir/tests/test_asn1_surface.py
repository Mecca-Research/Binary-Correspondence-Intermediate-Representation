"""P5 — the sparse text surface as a lossless view of the canonical node table.

**P5's gate**: *"`surface -> canonical -> surface` identity; formatting confined to a side
table; two presentations of one program hash identically."*

The first clause is the one worth reading twice. §4.2 says the round trip must be the
identity *on the canonical side*, not on the text — two spellings of one program are supposed
to converge, and demanding the text survive byte-for-byte would make the text authoritative,
which is the failure the whole section is written to prevent. So the law under test is
`parse(print(parse(s))) == parse(s)`, and the tests that matter most are the ones where the
two spellings differ wildly and the canonical bytes do not.

The second and third clauses are checked structurally rather than by inspection: `Presentation`
is a separate return value, and the tests parse deliberately divergent spellings and compare
`graph_to_jer` bytes and `content_address` digests.

Two defects found by building this against the real corpus rather than against fixtures are
pinned below, each in its own test: an empty attribute name, and a node whose kind collides
with the `roots` keyword. Both had the same shape — the surface was *stricter* than the
canonical form, so a decodable document had no spelling — which is the precise lossiness the
gate forbids.
"""

from __future__ import annotations

import ast
import os
import sys

from bcir.asn1.dialect import parse_mlir
from bcir.asn1.graph import (
    DIALECT_NODE_CLASS, Edge, Graph, Node, content_address, dialect_to_graph,
    graph_to_dialect, graph_to_jer, resolve,
)
from bcir.asn1.program import graph_to_module, module_to_graph, verdicts
from bcir.asn1.surface import (
    Presentation, jer_to_surface, parse_surface, print_surface, surface_to_graph,
    surface_to_jer,
)
from bcir.asn1.tags import Asn1Error
from bcir.examples import PROGRAMS

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_FIXTURE = os.path.join(_ROOT, "mlir", "test", "passes", "verify_asn1.mlir")


def _dialect_corpus():
    return parse_mlir(open(_FIXTURE, encoding="utf-8").read())


def _round_trips(graph: Graph) -> bool:
    """The gate's first clause for one graph: identity on the canonical side."""
    back, _ = parse_surface(print_surface(graph))
    return back == graph and graph_to_jer(back) == graph_to_jer(graph)


# --- the gate, over real corpora --------------------------------------------------------------


def test_every_corpus_program_round_trips_through_the_surface():
    """Twelve real programs, not hand-written fixtures — the node tables P3 actually emits."""
    count = 0
    for name, build in sorted(PROGRAMS.items()):
        assert _round_trips(module_to_graph(build())), name
        count += 1
    assert count >= 10, f"the corpus collapsed to {count} programs"


def test_the_dialect_corpus_round_trips_through_the_surface():
    """P2's graphs too, so the surface is a view of the node TABLE and not of one producer."""
    corpus = _dialect_corpus()
    for module in corpus:
        graph = dialect_to_graph(module)
        back, _ = parse_surface(print_surface(graph))
        assert back == graph, module.name
        assert graph_to_dialect(back) == module, module.name
    assert len(corpus) >= 20, f"the dialect corpus collapsed to {len(corpus)}"


def test_the_verdicts_survive_the_text_surface():
    """The strongest end-to-end available: R19/R20/R21 answers after a trip through text.

    P3 proved the verdicts survive the JER projection. This proves the *text* adds nothing
    and loses nothing on top of it — which is what "the canonical form is authoritative and
    the syntax is a view" has to mean operationally.
    """
    for name, build in sorted(PROGRAMS.items()):
        module = build()
        rebuilt = graph_to_module(surface_to_graph(print_surface(module_to_graph(module))))
        assert rebuilt == module, name
        assert verdicts(rebuilt) == verdicts(module), name


# --- the gate's second and third clauses --------------------------------------------------------


_TERSE = '(graph (roots #0) (program "demo" (-> phase #1)) (phase "0"))'
_RICH = '''; the entry point
(graph
  (roots @entry)
  (program "demo" @entry      ; the module
      (-> phase @loop))
  (phase "0" @loop))
'''
_QUOTED = '(graph(roots #0)("program" "demo"(-> "phase" #1))("phase" "0"))'


def test_two_presentations_of_one_program_hash_identically():
    """The gate's third clause, spelled three ways: terse, aliased-and-commented, quoted."""
    graphs = [surface_to_graph(text) for text in (_TERSE, _RICH, _QUOTED)]
    assert len({graph_to_jer(g) for g in graphs}) == 1, "the canonical bytes diverged"
    assert len({content_address(g, 0) for g in graphs}) == 1, "the digests diverged"
    assert graphs[0] == graphs[1] == graphs[2]


def test_the_graph_is_independent_of_the_presentation():
    """The gate's second clause. Aliases, comments and indentation reach nothing canonical."""
    graph, rich = parse_surface(_RICH)
    assert rich.aliases and rich.comments and rich.preamble, "the fixture lost its formatting"
    # Printed with the presentation, and printed with none at all: two different texts.
    with_view = print_surface(graph, rich)
    without = print_surface(graph)
    assert with_view != without
    assert surface_to_graph(with_view) == surface_to_graph(without) == graph
    assert surface_to_jer(with_view) == surface_to_jer(without)
    # Indentation is presentation too, and changing it must not reach the canonical form.
    wide = print_surface(graph, Presentation(indent=8))
    assert wide != without and surface_to_graph(wide) == graph


def test_the_presentation_survives_its_own_round_trip():
    """The side table is lossless in its own right, or a commented program loses its comments."""
    graph, view = parse_surface(_RICH)
    again, view_again = parse_surface(print_surface(graph, view))
    assert again == graph
    assert view_again == view


def test_the_canonical_side_is_the_identity_for_every_spelling():
    """`parse(print(parse(s))) == parse(s)` — the gate's first clause, stated exactly."""
    for text in (_TERSE, _RICH, _QUOTED):
        once, view = parse_surface(text)
        twice, _ = parse_surface(print_surface(once, view))
        assert twice == once, text


# --- the two defects this build found -----------------------------------------------------------


def test_an_empty_attribute_name_has_a_spelling():
    """`Attribute.name` is an unconstrained UTF8String, so `""` is a legal name.

    The first reader refused `:""` as "a colon with no name after it", which made a
    decodable document unspellable. The rule is now that an empty name is legal when it was
    *quoted* and a typo when it was bare — the distinction the canonical form actually draws.
    """
    graph = Graph((Node(kind="k", label="x", attributes=(("", "v"), ("a", ""))),))
    assert _round_trips(graph)
    assert surface_to_graph('(graph (k "x" :"" "v"))').nodes[0].attributes == (("", "v"),)
    try:
        parse_surface('(graph (k "x" : "v"))')
    except Asn1Error as error:
        assert "no name after it" in str(error)
    else:
        raise AssertionError("a bare `:` is still a typo and must be refused")


def test_a_node_whose_kind_is_the_roots_keyword_has_a_spelling():
    """`roots` was a keyword *by position* — legal until the first node — and that lost a node.

    A node of kind `roots` printed as `(roots "0")` and then failed to parse: the reader
    treated it as the roots form and demanded `#index` arguments. The keyword is now decided
    by the following TOKEN — a node's second token is always its quoted label — so the two
    forms are distinguishable wherever they appear.
    """
    for graph in (Graph((Node(kind="roots", label="0"),)),
                  Graph((Node(kind="roots", label="0"),), roots=(0,)),
                  Graph((Node(kind="roots", label="", edges=(Edge("roots", 0),)),)),
                  Graph((Node(kind="graph", label="0"),))):
        assert _round_trips(graph), graph.nodes[0].kind
    # And the roots form is no longer confined to the head of the table.
    assert surface_to_graph('(graph (a "x") (roots #0))').roots == (0,)
    assert surface_to_graph('(graph (roots #0) (a "x"))').roots == (0,)
    # A quoted kind is always a node, never the keyword.
    assert surface_to_graph('(graph ("roots" "0"))').nodes[0].kind == "roots"


# --- what the surface must be able to spell ------------------------------------------------------


def test_hostile_labels_and_values_survive():
    """Quotes, newlines, control characters and astral-plane text are all legal UTF8String."""
    for label in ('', 'he said "hi"', "a\nb\tc\\d", "\x00\x1f\x7f", "\U0001f600 emoji",
                  "; not a comment", "(paren) #0 @alias -> :attr"):
        assert _round_trips(Graph((Node(kind="k", label=label),))), repr(label)
        assert _round_trips(Graph((Node(kind="k", label="x",
                                        attributes=(("n", label),)),))), repr(label)
        assert _round_trips(Graph((Node(kind=label or "k", label="x"),))), repr(label)


def test_a_cycle_has_a_spelling():
    """The reason P1 chose a node table at all — so the surface must not reintroduce nesting."""
    cycle = Graph((Node(kind="a", label="0", edges=(Edge("next", 1),)),
                   Node(kind="b", label="1", edges=(Edge("next", 0),))), roots=(0,))
    assert _round_trips(cycle)
    # A forward reference by alias resolves after the whole form is read, so a cycle written
    # "backwards" costs nothing.
    text = '(graph (roots @a) (a "0" @a (-> next @b)) (b "1" @b (-> next @a)))'
    assert surface_to_graph(text) == cycle


def test_the_empty_and_rootless_graphs_both_have_spellings():
    """A graph with no roots is a library, and refusing it would halve what P1 can describe."""
    assert _round_trips(Graph())
    assert surface_to_graph(print_surface(Graph())) == Graph()
    assert _round_trips(Graph((Node(kind="k", label="x"),)))
    assert _round_trips(Graph((Node(kind="k", label="x"),), roots=(0, 0)))


def test_duplicate_attribute_names_survive():
    """`attributes` is a SEQUENCE OF, not a map: two rows with one name is a legal document."""
    graph = Graph((Node(kind="k", label="x", attributes=(("a", "1"), ("a", "2"))),))
    assert _round_trips(graph)
    assert surface_to_graph(print_surface(graph)).nodes[0].attributes == (("a", "1"),
                                                                          ("a", "2"))


def test_a_dangling_edge_is_carried_and_reported_rather_than_refused():
    """The reader is not stricter than the schema; `resolve` is where judgement lives.

    `Edge.target` is an INTEGER, and P1 already decided an out-of-range one is an
    `EdgeFault` — a value. A surface that refused it would leave a decodable document with
    no spelling, which is the same defect as the two above wearing a safety label.
    """
    graph = surface_to_graph('(graph (module "m" (-> type #9)))')
    assert graph.nodes[0].edges == (Edge("type", 9),)
    assert _round_trips(graph)
    faults = resolve(graph, DIALECT_NODE_CLASS).edge_faults
    assert len(faults) == 1 and "outside the node table" in faults[0].reason


# --- refusals ------------------------------------------------------------------------------------


def test_the_reader_refuses_what_it_cannot_read_unambiguously():
    for text, fragment in (
            ('(graph (program "d" (-> phase @nope)))', "never defined"),
            ('(graph (a "x" @n) (b "y" @n))', "used twice"),
            ('(graph (a "x" @n @m))', "already has an alias"),
            ('(graph (a "x")) trailing', "text after the closing"),
            ('(module (a "x"))', "top-level form is `graph`"),
            ('(graph (a x))', "label, in quotes"),
            ('(graph (a "x" :k))', "value, in quotes"),
            ('(graph (a "x" (-> lab)))', "#index or @alias"),
            ('(graph (a "x\\q"))', "unknown escape"),
            ('(graph (a "x\\u00"))', "four hex digits"),
            ('(graph (a "x))', "never closed"),
            ('(graph (a "x")', "closing `graph`"),
            ('', "an empty surface"),
            ('(graph (roots #0) (roots #0))', "appears twice"),
    ):
        try:
            parse_surface(text)
        except Asn1Error as error:
            assert fragment in str(error), f"{text!r} gave {error}"
        else:
            raise AssertionError(f"{text!r} should have been refused")


def test_an_unknown_escape_is_refused_rather_than_passed_through():
    """Two spellings of one string would mean the canonical form has two encodings."""
    try:
        parse_surface('(graph (k "a\\qb"))')
    except Asn1Error as error:
        assert "unknown escape" in str(error)
    else:
        raise AssertionError("\\q must not silently become 'q'")


# --- structural claims ----------------------------------------------------------------------------


def test_neither_direction_of_the_surface_recurses():
    """A thousand-node chain, printed and parsed under a recursion limit of eighty.

    P1 bought cycle- and depth-tolerance by choosing integer-index edges over nesting. A
    surface that recursed per edge would hand that back at the front door, and the failure
    would only appear on the programs big enough to matter.
    """
    size = 1000
    chain = Graph(tuple(Node(kind="link", label=str(i),
                             edges=(Edge("next", i + 1),) if i + 1 < size else ())
                        for i in range(size)), roots=(0,))
    previous = sys.getrecursionlimit()
    sys.setrecursionlimit(80)
    try:
        rebuilt, _ = parse_surface(print_surface(chain))
    finally:
        sys.setrecursionlimit(previous)
    assert rebuilt == chain
    assert graph_to_jer(rebuilt) == graph_to_jer(chain)


def test_the_surface_knows_nothing_but_the_node_table():
    """*"A typing shortcut with a printer, not a language"* — checked, not promised.

    The moment the surface can reach the verifier, the dialect or a program projection, it
    has semantics of its own and the canonical form has stopped being authoritative. The
    cheapest way to keep that true is to make it a structural property of the module.
    """
    source = open(os.path.join(_ROOT, "bcir", "asn1", "surface.py"), encoding="utf-8").read()
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add("." * node.level + (node.module or ""))
    # Checked over the IMPORT GRAPH rather than over the text: a word search finds the same
    # word in a sentence explaining why it is absent, which is a test that fails for being
    # well documented.
    assert imported <= {"__future__", "dataclasses", ".graph", ".tags"}, sorted(imported)


def test_the_jer_ends_of_the_projection_agree_with_the_graph_ends():
    """`surface_to_jer` and `jer_to_surface` are conveniences, not a second implementation."""
    for name, build in sorted(PROGRAMS.items()):
        graph = module_to_graph(build())
        text = print_surface(graph)
        assert surface_to_jer(text) == graph_to_jer(graph), name
        assert jer_to_surface(graph_to_jer(graph)) == text, name
