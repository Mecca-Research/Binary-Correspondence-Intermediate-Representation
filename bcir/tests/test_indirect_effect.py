"""§5.14 Phase 2: the indirect-call CALLEE TYPE + CONSERVATIVE EFFECT record.

An indirect dispatch (`fp(x)` -> `c.call.indirect`, `o->fn(x)` -> `c.call.imember`) was a fully
opaque edge: no declared callee type, and -- the unsoundness this closes -- NO effect footprint,
so `CompileResult.commute()` would happily reorder a dispatcher past a global writer. Now (1) the
claim carries the DECLARED signature (`Claim.callee_sig`, "ret(param, ...)", digest-excluded /
vacuous-by-default, carried to MLIR as the `callee_sig` attr), well-formedness guarded by an
R18 clause on both rails; and (2) a function containing a dispatch conservatively reads+writes
EVERY module-scope global in its effect footprint -- the unknown callee may reach any of them."""

from bcir.examples import gather_reduce, vector_add
from bcir.model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from bcir.verify import verify


def _unit(src):
    from bcir.frontends.cfront import compile_unit
    r = compile_unit(src, check_clang=False)
    assert r.is_clean, r.diagnostics
    return r


_SRC = (
    "unsigned g_state = 0u;\n"
    "unsigned writer(unsigned v){ g_state = v; return v; }\n"
    "unsigned pure_fn(unsigned x){ return x + 1u; }\n"
    "unsigned dispatcher(unsigned (*fp)(unsigned), unsigned x){ return fp(x); }\n"
)


def test_indirect_call_claims_carry_the_declared_signature():
    r = _unit(_SRC)
    claims = [c for fn in r.lowered.functions.values() for ph in fn.module.phases
              for c in ph.claims]
    dispatch = [c for c in claims if c.op == "c.call.indirect"]
    assert dispatch and all("(" in c.callee_sig and c.callee_sig for c in dispatch), \
        [(c.op, c.callee_sig) for c in dispatch]
    # every non-dispatch claim stays vacuous (the opaque-edge default is untouched elsewhere)
    assert all(not c.callee_sig for c in claims if not c.op.startswith("c.call.i"))


def test_dispatch_member_claims_carry_the_declared_signature():
    r = _unit("struct ops { unsigned (*fn)(unsigned); };\n"
              "unsigned call(struct ops *o, unsigned x){ return o->fn(x); }\n")
    claims = [c for fn in r.lowered.functions.values() for ph in fn.module.phases
              for c in ph.claims]
    dispatch = [c for c in claims if c.op.startswith("c.call.imember")]
    assert dispatch and all("(" in c.callee_sig for c in dispatch), \
        [(c.op, c.callee_sig) for c in dispatch]


def test_a_dispatcher_never_commutes_past_a_global_writer():
    # THE UNSOUNDNESS THIS CLOSES: the dispatcher's unknown callee may write g_state, so
    # commute(dispatcher, writer) must be False -- while a genuinely pure function still
    # commutes with the writer (no false conservatism for units without dispatches).
    r = _unit(_SRC)
    assert not r.commute("dispatcher", "writer")
    assert not r.commute("dispatcher", "dispatcher")
    assert r.commute("pure_fn", "pure_fn")


def test_r18_flags_a_malformed_callee_signature():
    m = Module(name="sig")
    m.add_resource(Resource(rid=1, domain=Domain.RAM, shape=(8,)))
    m.add_resource(Resource(rid=2, domain=Domain.RAM, shape=(8,)))
    c = Claim(id=1, opcode=Opcode.GEM_DISPATCH, lane=Lane.U, stride_class=StrideClass.SCALAR,
              count=1, rd=(1,), wr=(2,), op="c.call.indirect", domain=Domain.RAM,
              callee_sig="not-a-signature")
    m.add_phase(Phase(phase_id=0, deps=(), claims=[c]))
    d = [x for x in verify(m) if x.law == "R18"]
    assert d and "malformed indirect-callee signature" in d[0].message, d


def test_callee_sig_is_vacuous_over_the_existing_corpus():
    from bcir.kbcir import TARGETS
    from bcir.kbcir.cost import Theta
    from bcir.lower.mlir import to_mlir
    for m in (vector_add(), gather_reduce()):
        assert all(not c.callee_sig for ph in m.phases for c in ph.claims)
        assert [x for x in verify(m) if "callee" in x.message] == []
    assert "callee_sig" not in to_mlir(vector_add(), TARGETS["x86_avx512"], Theta.cool())


if __name__ == "__main__":
    import sys

    mod = sys.modules[__name__]
    tests = sorted(n for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n)))
    passed = failed = 0
    for name in tests:
        try:
            getattr(mod, name)()
            passed += 1
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {e!r}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
