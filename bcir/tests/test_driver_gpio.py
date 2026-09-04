"""Phase 3 breadth (real fixtures): a CMSIS-style GPIO driver through the full pipeline.

`cmsis_gpio.h` carries the header shapes an actual vendor ships that the UART fixture does
not -- the `__IO` volatile qualifier macro, RESERVED padding arrays, a WRITE-ONLY
hardware-atomic set/reset register (BSRR), a 2-bit-per-pin mode field, and the RCC
clock-gate-first ordering. Pins: the driver compiles CLEAN on the oracle; every volatile
register access lowers to the MMIO domain (the Phase-2 volatile law engages on real header
shapes, not just synthetic fixtures); the BSRR path never read-modify-writes the output
register (the atomicity idiom is visible in the claim graph: stores without a paired ODR
load); and every function records its R12 call-ABI contract."""

import os

from bcir.frontends.cfront import compile_unit
from bcir.model import Domain

_RUNTIME_C = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "runtime", "c"))


def _compiled():
    with open(os.path.join(_RUNTIME_C, "cfront_driver_gpio.c"), encoding="utf-8") as f:
        src = f.read()
    r = compile_unit(src, check_clang=False, search_paths=[_RUNTIME_C])
    assert r.is_clean, r.diagnostics
    return r


def test_cmsis_driver_compiles_clean_with_mmio_domains():
    r = _compiled()
    assert set(r.lowered.functions) == {"gpio_mode_set", "gpio_write_pin", "gpio_wait_pin"}
    for name, lf in r.lowered.functions.items():
        mmio = [c for c in lf.claims if c.domain == Domain.MMIO]
        assert mmio, f"{name}: no MMIO claim -- the __IO accesses did not reach the MMIO domain"


def test_bsrr_writes_are_stores_never_rmw():
    """The hardware-atomicity idiom: gpio_write_pin drives BSRR with pure STORES -- no load
    of the output register anywhere in the function (a RMW on ODR would race pin owners)."""
    lf = _compiled().lowered.functions["gpio_write_pin"]
    stores = [c for c in lf.claims if c.op == "c.store" and c.domain == Domain.MMIO]
    loads = [c for c in lf.claims if c.op.startswith("c.load") and c.domain == Domain.MMIO]
    assert len(stores) == 2 and not loads, (stores, loads)


def test_every_function_records_its_abi_contract():
    r = _compiled()
    assert set(r.abi_contracts) == set(r.lowered.functions)
    assert r.abi_diagnostics == []
    c = r.abi_contracts["gpio_mode_set"]
    assert len(c.params) == 5 and c.params[0][1] == 8  # the rcc pointer is LP64-wide


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
