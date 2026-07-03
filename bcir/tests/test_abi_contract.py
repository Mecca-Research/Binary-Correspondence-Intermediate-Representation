"""§5.14 Phase 2, the LAST area: the per-function CALL-ABI CONTRACT (R12 on the call ABI).

Every compile now records the layout facts the emitter materialized each function's frame from
(`CompileResult.abi_contracts`: target + per-parameter (name, size, align) + the return slot) and
runs the R12 law over them (`abi_diagnostics`, advisory): the record must agree with BOTH the
laid-out CTypes and the target's data model -- a pointer-kind parameter is pointer_size, a `long`
parameter is long_size. A tampered contract is caught; cross-target layouts (LLP64/ILP32) record
their own facts. The MLIR rail carries the same record (`bcir.abi_contract`) with the matching
R12 clause (`verify_abi_contract.mlir`)."""

from bcir.frontends.cfront.abi import TARGETS, abi_contract_for, verify_abi_contract


def _unit(src, target=None):
    from bcir.frontends.cfront import compile_unit
    r = compile_unit(src, check_clang=False, target=target)
    assert r.is_clean, r.diagnostics
    return r


_SRC = "unsigned f(unsigned *p, unsigned long n, unsigned x){ return x + (unsigned)n; }\n"


def test_every_compile_records_a_clean_contract():
    r = _unit(_SRC)
    assert r.abi_diagnostics == [], r.abi_diagnostics
    c = r.abi_contracts["f"]
    assert c.target == "x86_64-linux" and c.fn == "f"
    names = [p[0] for p in c.params]
    assert names == ["p", "n", "x"]
    sizes = dict((p[0], p[1]) for p in c.params)
    assert sizes["p"] == 8 and sizes["n"] == 8 and sizes["x"] == 4     # LP64 facts
    assert c.ret == (4, 4)


def test_cross_target_contracts_record_the_data_model():
    # LLP64: long is 4; ILP32: pointers are 4 -- the axes the frontend actually lays out.
    win = _unit(_SRC, target="x86_64-windows").abi_contracts["f"]
    assert dict((p[0], p[1]) for p in win.params) == {"p": 8, "n": 4, "x": 4}
    i386 = _unit(_SRC, target="i386-linux").abi_contracts["f"]
    assert dict((p[0], p[1]) for p in i386.params) == {"p": 4, "n": 4, "x": 4}


def test_a_tampered_contract_is_caught_by_the_law():
    from dataclasses import replace
    r = _unit(_SRC)
    lf = r.lowered.functions["f"]
    abi = TARGETS["x86_64-linux"]
    good = abi_contract_for(lf, abi)
    assert verify_abi_contract(good, lf, abi) == []
    # lie about the pointer parameter's size: both the laid-out check and the data-model check fire.
    bad = replace(good, params=(("p", 4, 8),) + good.params[1:])
    msgs = verify_abi_contract(bad, lf, abi)
    assert msgs and any("pointer size" in m for m in msgs), msgs
    # lie about the target: caught before any per-parameter check.
    assert verify_abi_contract(replace(good, target="i386-linux"), lf, abi)
    # lie about the return slot.
    assert verify_abi_contract(replace(good, ret=(8, 8)), lf, abi)


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
