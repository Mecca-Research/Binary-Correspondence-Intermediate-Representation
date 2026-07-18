"""B3 dual-rail parity (Phase 2a): the MLIR law rail (bcir.gem.autodiff) carries the autodiff oracle's
CLOSED-SET forward DAG (bcir/kbcir/autodiff.py) as a first-class serialized law object. Slice 5 proved the
differentiable DAG is a CLOSED SET over the fixed vocabulary {const,var,neg,add,sub,mul,div,dot,select}
(autodiff.CLOSED_SET / _ARITY / differentiable_ops): differentiating any expression in that set stays in the
same set. That closure is exactly what lets the law rail carry the forward DAG with a fixed-vocabulary verifier
(closed vocabulary + per-op arity + strictly-earlier DAG index bounds).

NO lowering and NO cost pass in this phase (Phase 2a is the op + verifier + round-trip; the lowering/cost pass
is the next slice). So -- unlike the sibling *_law_parity tests, which cross-check a recomputed cost against the
oracle's cost_of -- this is a ROUND-TRIP parity: serialize a REAL Tape DAG to the gem.autodiff IR and assert the
law rail VERIFIES + round-trips it byte-identically.

Two arms (mirrors test_matmul_law_parity's structure):
  * a PURE-PYTHON arm (always runs): the serializer emits a well-formed closed-set DAG for a real Tape -- every
    opcode is in CLOSED_SET, the per-node arity mirrors _ARITY (dot variadic), operands reference strictly
    earlier nodes, leaves carry a payload, non-leaves the -1 sentinel. This is the contract the MLIR verifier
    mirrors, asserted against autodiff.Tape directly so it tracks the oracle.
  * the ULTIMATE cross-check (self-skips without the built bcir-opt): drives the real `bcir-opt` over the
    serialized gem.autodiff IR and asserts the closed-set verifier ACCEPTS a faithful Tape DAG and the op
    round-trips (parse -> print -> parse) unchanged; and that a corrupted DAG (a foreign opcode) is REJECTED.
"""

from __future__ import annotations

import os
import subprocess

from bcir.kbcir.autodiff import CLOSED_SET, Tape, _ARITY

# The fixed opcode int codes the law op + verifier use (mirror autodiff._ARITY's order). The vocabulary is the
# CLOSED SET; this map is the serialization of it.
_CODE = {"const": 0, "var": 1, "neg": 2, "add": 3, "sub": 4, "mul": 5,
         "div": 6, "dot": 7, "select": 8, "exp": 9, "log": 10, "sqrt": 11,
         "tanh": 12, "sin": 13, "cos": 14}


def _build_f_xy_plus_x():
    """f(x,y) = x*y + x -- the canonical small DAG (two var leaves, mul, add, with x SHARED). Returns
    (tape, output, input_names)."""
    t = Tape()
    x = t.var("x")
    y = t.var("y")
    out = t.add(t.mul(x, y), x)
    return t, out, ["x", "y"]


def _build_rich():
    """A rich DAG exercising EVERY closed-set op kind: const, var, neg, sub, div, add, the variadic dot, and
    select. Returns (tape, output, input_names)."""
    t = Tape()
    a, b, c = t.var("a"), t.var("b"), t.var("c")
    na = t.neg(a)
    ab = t.sub(a, b)
    k = t.const(3)
    d = t.div(ab, k)
    dp = t.dot((a, b), (b, c))
    sel = t.select(c, d, dp)
    positive = t.add(t.mul(a, a), t.const(2.0))
    trans = t.add(t.exp(t.neg(positive)), t.log(positive))
    trans = t.add(trans, t.sqrt(positive))
    trans = t.add(trans, t.tanh(b))
    trans = t.add(trans, t.sin(a))
    trans = t.add(trans, t.cos(c))
    out = t.add(t.add(sel, na), trans)
    return t, out, ["a", "b", "c"]


def _serialize(tape: Tape, output: int, input_names: list[str]) -> dict:
    """Serialize a Tape's forward DAG to the parallel per-node arrays the gem.autodiff law op carries. The Tape
    interns operands before their users, so node i's operands are all < i (already topological/acyclic)."""
    slot = {n: i for i, n in enumerate(input_names)}
    opcodes, arities, arg_base, args, consts = [], [], [], [], []
    base = 0
    for nid in range(tape.node_count()):
        n = tape.node(nid)
        opcodes.append(_CODE[n.op])
        ar = len(n.args)
        arities.append(ar)
        arg_base.append(base)
        args.extend(n.args)
        base += ar
        if n.op == "const":
            consts.append(int(n.const))      # the const value as an integer payload (the structural law)
        elif n.op == "var":
            consts.append(slot[n.const])     # the input slot
        else:
            consts.append(-1)                # a non-leaf op carries no payload -> the sentinel
    return {
        "n_inputs": len(input_names), "opcodes": opcodes, "arities": arities,
        "arg_base": arg_base, "args": args, "consts": consts, "output": output,
    }


def _to_mlir(name: str, s: dict) -> str:
    """Emit the bcir.gem.autodiff carrier-op MLIR for a serialized DAG."""
    def arr(a):
        return "array<i64: " + ", ".join(str(x) for x in a) + ">"
    return (
        "bcir.module @m {\n"
        f'  bcir.gem.autodiff @{name} {{ n_inputs = {s["n_inputs"]} : i64, '
        f'opcodes = {arr(s["opcodes"])}, arities = {arr(s["arities"])}, '
        f'arg_base = {arr(s["arg_base"])}, args = {arr(s["args"])}, '
        f'consts = {arr(s["consts"])}, output = {s["output"]} : i64 }}\n'
        "}\n"
    )


# --- the pure-Python parity arm (always runs) -----------------------------------------------------

def test_serializer_emits_a_well_formed_closed_set_dag():
    """The serializer turns a real Tape into the exact closed-set well-formedness the MLIR verifier enforces:
    every opcode is in CLOSED_SET, each node's arity mirrors autodiff._ARITY (dot variadic 2k), operands
    reference strictly earlier nodes, a var's slot is a real input, and leaves carry a payload / non-leaves the
    -1 sentinel. Asserted against autodiff.Tape directly so it tracks the oracle vocabulary."""
    code_to_op = {v: k for k, v in _CODE.items()}
    assert frozenset(_CODE) == CLOSED_SET, "the serialized vocabulary must be exactly autodiff.CLOSED_SET"
    for build in (_build_f_xy_plus_x, _build_rich):
        tape, output, names = build()
        s = _serialize(tape, output, names)
        n = len(s["opcodes"])
        assert len(s["arities"]) == n == len(s["arg_base"]) == len(s["consts"]), "parallel arrays must be equal length"
        assert sum(s["arities"]) == len(s["args"]), "the flat args must hold exactly sum(arities) operands"
        assert 0 <= s["output"] < n, "output must be a valid node index"
        for i in range(n):
            op = code_to_op[s["opcodes"][i]]
            assert op in CLOSED_SET, f"node {i} opcode {op} not in the closed set"
            ar = s["arities"][i]
            if op == "dot":
                assert ar >= 2 and ar % 2 == 0, f"node {i} dot arity must be an even 2k, got {ar}"
            else:
                assert ar == _ARITY[op], f"node {i} ({op}) arity {ar} != _ARITY {_ARITY[op]}"
            base = s["arg_base"][i]
            for a in range(ar):
                operand = s["args"][base + a]
                assert 0 <= operand < i, f"node {i} operand {operand} must reference a strictly earlier node"
            payload = s["consts"][i]
            if op == "var":
                assert 0 <= payload < s["n_inputs"], f"node {i} var slot {payload} out of [0, n_inputs)"
            elif op != "const":
                assert payload == -1, f"node {i} ({op}) non-leaf must carry the -1 sentinel, got {payload}"


# --- the ultimate cross-check: the real bcir-opt law rail (self-skips without the built bcir-opt) -

def test_law_rail_verifies_and_round_trips_a_faithful_autodiff_dag():
    """Drive the real `bcir-opt` over the serialized gem.autodiff IR and assert the closed-set verifier ACCEPTS
    a faithful Tape DAG and the op round-trips (parse -> print -> parse) byte-identically -- the Phase-2a
    primary cross-check (no cost/lowering pass yet)."""
    bo = _find_bcir_opt()
    if not bo:
        return  # the MLIR toolchain is not built in this environment (the oracle-only CI job)
    for build in (_build_f_xy_plus_x, _build_rich):
        tape, output, names = build()
        ir = _to_mlir("ad", _serialize(tape, output, names))
        first = subprocess.run([bo], input=ir, capture_output=True, text=True)
        assert first.returncode == 0, f"the law rail rejected a faithful autodiff DAG:\n{first.stderr}"
        assert "bcir.gem.autodiff @ad" in first.stdout
        # round-trip: re-parse/print the printed form -> identical (the op is stable through the printer).
        second = subprocess.run([bo], input=first.stdout, capture_output=True, text=True)
        assert second.returncode == 0, f"the law rail rejected its own printed form:\n{second.stderr}"
        assert second.stdout == first.stdout, "gem.autodiff must round-trip byte-identically"


def test_law_rail_rejects_a_foreign_opcode():
    """The headline closed-set gate, end-to-end on the law rail: a serialized DAG with a FOREIGN opcode (a code
    outside the closed vocabulary) is REJECTED by the gem.autodiff verifier -- the law the closure proof
    underwrites. Self-skips without bcir-opt."""
    bo = _find_bcir_opt()
    if not bo:
        return
    tape, output, names = _build_f_xy_plus_x()
    s = _serialize(tape, output, names)
    s["opcodes"] = list(s["opcodes"])
    s["opcodes"][2] = 15                        # foreign: closed vocabulary ends at code 14
    proc = subprocess.run([bo], input=_to_mlir("bad", s), capture_output=True, text=True)
    assert proc.returncode != 0, "the law rail must reject a foreign opcode"
    assert "is not in the closed set" in proc.stderr, \
        f"the rejection must cite the closed-set law, got:\n{proc.stderr}"


def _find_bcir_opt():
    env = os.environ.get("BCIR_OPT")
    if env and os.path.exists(env):
        return env
    root = os.path.join(os.path.dirname(__file__), "..", "..", "build", "mlir-build")
    for dirpath, _dirs, files in os.walk(os.path.normpath(root)) if os.path.isdir(root) else []:
        if "bcir-opt" in files:
            return os.path.join(dirpath, "bcir-opt")
    return None
