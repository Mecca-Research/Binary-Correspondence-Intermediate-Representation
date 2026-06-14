"""The enriched-operad memory interface: labels + content-addressed indexes +
trace over the memory modules (the higher intelligence layer, R13).

A memory module's fixpoint is lifted into a labeled, indexed, traceable operad.
Indexes are content-addressed (identical operations share one -- CSE / the
liked-pair identity), composition preserves labels and indexes, and the whole
layer is quarantined out of the deterministic spine.
"""

from bcir.kbcir import memory as M
from bcir.kbcir.egraph import Add, C, Mul, V
from bcir.kbcir.operad import (
    EnrichedOp,
    EnrichedOperad,
    enrich_memory,
    f_index,
    f_label,
)
from bcir.verify import verify_enriched


def _laws(diags):
    return {d.law for d in diags}


# --- content-addressed indexing: identical operations are one liked pair ----------

def test_identical_operations_share_a_content_addressed_index():
    op = EnrichedOperad()
    a1 = op.add("var", label=("MEMORY", "var", "a"))
    a2 = op.add("var", label=("MEMORY", "var", "a"))
    assert a1 == a2 and len(op.ops) == 1               # CSE: a = a, one entry
    b = op.add("var", label=("MEMORY", "var", "b"))
    assert b != a1                                       # distinct content, distinct index


def test_the_index_is_the_fnv_of_name_label_children():
    op = EnrichedOperad()
    idx = op.add("var", label=("MEMORY", "var", "a"))
    assert idx == f_index("var", ("MEMORY", "var", "a"), ())


# --- composition preserves labels and indexes (γ_L, f_L, f_I) --------------------

def test_composition_preserves_labels():
    op = EnrichedOperad()
    b = op.add("var", label=("MEMORY", "var", "b"))
    c = op.add("var", label=("MEMORY", "var", "c"))
    s = op.compose("add", children=(b, c), label=("MEMORY", "op", "add"))
    # label is a systematic function of the operator + its children's heads.
    assert op.get(s).label == f_label(("MEMORY", "op", "add"),
                                      (op.get(b).label, op.get(c).label))


def test_composition_is_deterministic():
    op = EnrichedOperad()
    b = op.add("var", label=("MEMORY", "var", "b"))
    c = op.add("var", label=("MEMORY", "var", "c"))
    assert op.compose("add", (b, c)) == op.compose("add", (b, c))   # same index


# --- trace: reverse mapping back to the sources ----------------------------------

def test_trace_recovers_all_constituent_operations():
    op = EnrichedOperad()
    a = op.add("var", label=("MEMORY", "var", "a"))
    b = op.add("var", label=("MEMORY", "var", "b"))
    c = op.add("var", label=("MEMORY", "var", "c"))
    bc = op.compose("add", (b, c), label=("MEMORY", "op", "add"))
    root = op.compose("mul", (a, bc), label=("MEMORY", "op", "mul"))
    names = [o.name for o in op.trace(root)]
    assert names[0] == "mul" and set(names) == {"mul", "add", "var"}
    assert op.sources(root) == (a, bc)


# --- 2-morphisms: rewrites as transformation tracking ----------------------------

def test_a_rewrite_is_recorded_as_a_two_cell():
    op = EnrichedOperad()
    x = op.add("var", label=("MEMORY", "var", "x"))
    z = op.add("const", label=("MEMORY", "const", "0"))
    cell = op.record_rewrite(x, z, "identity")
    assert cell.src == x and cell.dst == z and cell.rule == ("REWRITE", "identity")
    assert op.cells == [cell]


# --- R13: enriched-operad integrity ----------------------------------------------

def test_a_consistent_operad_satisfies_R13():
    op = EnrichedOperad()
    b = op.add("var", label=("MEMORY", "var", "b"))
    c = op.add("var", label=("MEMORY", "var", "c"))
    root = op.compose("add", (b, c), label=("MEMORY", "op", "add"))
    assert verify_enriched(op, root) == []


def test_a_dangling_source_is_R13():
    op = EnrichedOperad()
    # an operation whose child index does not resolve (a broken SourceMap).
    op.ops[123] = EnrichedOp(name="mul", arity=1, label=("MEMORY", "op", "mul"),
                             index=123, children=(999,))
    assert "R13" in _laws(verify_enriched(op))


def test_a_tampered_index_is_R13():
    op = EnrichedOperad()
    # index does not equal f_index(content): content addressing is violated.
    op.ops[7] = EnrichedOp(name="mul", arity=0, label=("MEMORY", "op", "mul"),
                           index=7, children=())
    assert "R13" in _laws(verify_enriched(op))


def test_an_empty_label_under_active_labeling_is_R13():
    op = EnrichedOperad()
    idx = op.add("var", label=("MEMORY", "var", "x"))
    op.ops[idx] = EnrichedOp(name="var", arity=0, label=(), index=idx, children=())
    assert "R13" in _laws(verify_enriched(op))


# --- conditional activation (the cost/benefit toggle) ----------------------------

def test_labeling_and_indexing_can_be_disabled():
    off = EnrichedOperad(enable_labeling=False, enable_indexing=False)
    i1 = off.add("var")
    i2 = off.add("var")
    assert off.get(i1).label == () and i1 != i2          # no labels; ordinal keys
    assert verify_enriched(off) == []                    # integrity holds with metadata off


# --- the bridge: lift a frozen memory module into an enriched operad --------------

def test_enrich_memory_lifts_the_fixpoint_into_a_labeled_indexed_tree():
    mm = M.freeze(Add(Mul(V("a"), V("b")), Mul(V("a"), V("c"))), generation=1)
    operad, root = enrich_memory(mm)
    assert verify_enriched(operad, root) == []
    assert operad.get(root).label == ("MEMORY", "op", "mul")     # a*(b+c)
    # the shared operand 'a' is one content-addressed e-class (CSE preserved).
    a_nodes = [i for i, o in operad.ops.items()
               if o.label == ("MEMORY", "var", "a")]
    assert len(a_nodes) == 1
    # trace recovers every source of the frozen memory.
    labels = {o.label for o in operad.trace(root)}
    assert ("MEMORY", "var", "b") in labels and ("MEMORY", "op", "add") in labels


def test_enriching_is_deterministic_across_runs():
    mm = M.freeze(Add(Mul(V("a"), V("b")), Mul(V("a"), V("c"))), generation=1)
    (o1, r1), (o2, r2) = enrich_memory(mm), enrich_memory(mm)
    assert r1 == r2 and o1.source_map() == o2.source_map()        # content-addressed
