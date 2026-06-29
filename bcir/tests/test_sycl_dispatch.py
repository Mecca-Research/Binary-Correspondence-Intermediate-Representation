"""S2: the SYCL channel's RESIDENT DISPATCH / EXECUTION path -- the channel EXECUTES routed work.

Where test_sycl_channel proves SYCL is a priced+routed channel and a STANDALONE differential oracle (emit
+ diff a SAXPY kernel), this proves the channel now has a resident DISPATCHER: a module ``orchestrate``
places onto a tower including ``sycl_spirv`` can be RUN end-to-end via ``gem.execute``, with the sycl-placed
claim dispatched through the emitted SYCL kernel (the PORTABLE C++ fallback on CI, the ``-fsycl`` device
path when a SYCL toolchain is present), round-trip-verified against ``saxpy_reference``.

HONEST DEPTH (same discipline as test_sycl_channel): there is NO SYCL toolchain / GPU on CI. The end-to-end
``execute()`` round-trip via the portable fallback is REAL and runs here; the ``-fsycl`` device submission +
a real SPIR-V backend in ``llc`` are GATED and self-skip. The kernels/store WIRING is asserted built either
way (the seam is real even when the compile/run self-skips). The dispatcher is HOST-SIDE, above the G8
boundary, and NEVER on the legality path (two-truth)."""

import os

from bcir.channel_plugin import load_manifest, register_from_manifest
from bcir.channels import orchestrate
from bcir.frontends.cfront.linkflags import library_for_callee
from bcir.gem.execute import execute
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.kbcir.sycl_saxpy import saxpy_reference
from bcir.lower.sycl_dispatch import (
    ChannelDispatcher, DispatchUnavailable, SyclDispatcher, build_execute_kernels,
    seed_store, sycl_cxx, try_emit_spirv,
)
from bcir.model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
import bcir.verify as bverify

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MANIFEST = os.path.join(_REPO, "channels", "sycl.channel.json")

# fixed SAXPY data (the same shape test_sycl_channel uses), a=2.0.
_A = 2.0
_X = [1.0, 2.0, 3.0, 0.5, -1.0, 4.0, -2.5, 0.0]
_Y = [10.0, 20.0, 30.0, -0.5, 1.0, -4.0, 2.5, 7.0]


def _sycl_channel():
    return register_from_manifest(load_manifest(_MANIFEST))


def _saxpy_module(n=8, a=_A):
    """A small data_parallel SAXPY-shaped module: out = a*x + y over a dense float[n]. `a` rides imm[0]
    (the dispatcher's scalar convention); x/y are the two read rids, out the write rid."""
    m = Module(name="saxpy_dispatch")
    m.add_resource(Resource(rid=71, domain=Domain.RAM, shape=(n,), name="x"))
    m.add_resource(Resource(rid=72, domain=Domain.RAM, shape=(n,), name="y"))
    m.add_resource(Resource(rid=73, domain=Domain.RAM, shape=(n,), name="out"))
    claim = Claim(id=7100, opcode=Opcode.MUL, lane=Lane.U, stride_class=StrideClass.STRIDED, count=n,
                  rd=(71, 72), wr=(73,), op="call.saxpy", domain=Domain.RAM, imm=(a,))
    m.add_phase(Phase(phase_id=0, deps=(), claims=[claim]))
    return m


# --- (1) the headline: end-to-end round-trip -- the channel EXECUTES a routed claim --------------------

def test_end_to_end_routed_saxpy_executes_and_matches_reference():
    sycl = _sycl_channel()
    n = len(_X)
    m = _saxpy_module(n, _A)

    # orchestrate onto a tower that INCLUDES sycl_spirv and where it wins (the GPU/data_parallel tower);
    # confirm the SAXPY claim lands on sycl_spirv (the routed placement we then EXECUTE).
    plan = orchestrate(m, [sycl], Theta.cool())
    assert plan.placements[0].channel == "sycl_spirv", plan.placements[0].channel
    assert plan.placements[0].claim_id == 7100

    # seed the store with the live-in inputs (x, y); build the execute() kernels (the WIRING is asserted
    # built even if the compile/run self-skips -- the seam is real either way).
    store = seed_store(m, {71: list(_X), 72: list(_Y)})
    disp = SyclDispatcher()
    kernels = build_execute_kernels(m, plan, store, {"sycl_spirv": disp})
    assert 7100 in kernels and callable(kernels[7100])           # the seam is built

    try:
        res = execute(m, kernels)                                # run the module end-to-end
    except DispatchUnavailable:
        return                                                   # no C++ compiler -> self-skip the run
    finally:
        disp.close()

    assert res.executed == 1 and res.order == [7100]
    got = store[73]                                              # the sycl-dispatched output
    ref = saxpy_reference(_A, _X, _Y)
    assert len(got) == len(ref)
    for g, r in zip(got, ref):                                   # executed == reference to float round-off
        assert abs(g - r) < 1e-4 * (1.0 + abs(r)), (g, r)
    assert disp.mode in ("fallback", "sycl-device")             # honest: a real path executed


# --- (2) the dispatcher direct: run_saxpy matches the reference + mode is honest -----------------------

def test_dispatcher_run_saxpy_matches_reference():
    disp = SyclDispatcher()
    try:
        got = disp.run_saxpy(_A, _X, _Y)
    except DispatchUnavailable:
        return                                                   # no C++ compiler -> self-skip
    finally:
        disp.close()
    ref = saxpy_reference(_A, _X, _Y)
    for g, r in zip(got, ref):
        assert abs(g - r) < 1e-4 * (1.0 + abs(r)), (g, r)
    # mode reports the path that ran: "fallback" on CI (no SYCL toolchain), "sycl-device" if one is present.
    assert disp.mode in ("fallback", "sycl-device")
    if sycl_cxx() is None:
        assert disp.mode == "fallback"                           # the expected CI mode


def test_dispatcher_is_a_channel_dispatcher_and_validates_inputs():
    disp = SyclDispatcher()
    assert isinstance(disp, ChannelDispatcher)
    # input validation is PURE and happens before any compile (no toolchain needed): empty / mismatched
    # lengths raise ValueError regardless of whether a C++ compiler is present.
    for bad in (([], []), ([1.0, 2.0], [1.0])):
        try:
            disp.run_saxpy(_A, *bad)
            assert False, "expected ValueError on a bad SAXPY shape"
        except ValueError:
            pass
    disp.close()


# --- (3) the device path is GATED: runs + agrees only if a real SYCL compiler is present ---------------

def test_device_path_agrees_when_a_sycl_compiler_is_present():
    sy = sycl_cxx()
    if not sy:
        return        # no SYCL toolchain here -> the -fsycl device path self-skips (the EXPECTED CI case)
    disp = SyclDispatcher(prefer_device=True)
    got = disp.run_saxpy(_A, _X, _Y)
    disp.close()
    assert disp.mode == "sycl-device"                            # the device path actually ran
    ref = saxpy_reference(_A, _X, _Y)
    for g, r in zip(got, ref):
        assert abs(g - r) < 1e-2 * (1.0 + abs(r)), (g, r)        # the SYCL device computes the same SAXPY


# --- (4) the SPIR-V emission attempt: reachable, never crashes -----------------------------------------

def test_try_emit_spirv_never_crashes_and_returns_a_note():
    # the channel's SPIR-V codegen identity is reachable via the spirv64 target. Stock llc has no SPIR-V
    # backend (the gated case), so this is expected to return (False, "no SPIR-V backend: ...") on CI --
    # but it must NEVER raise (mirrors test_codegen.test_spirv_is_graceful_when_absent).
    m = _saxpy_module()
    result = optimize(m, TargetProfile.x86_avx512(), Theta.cool())
    ok, note = try_emit_spirv(m, result)
    assert isinstance(ok, bool)
    assert isinstance(note, str) and note                        # a sensible, non-empty note
    if not ok:
        assert "no SPIR-V backend" in note or "llc not found" in note, note


# --- (5) two-truth: the dispatch path is DATA only, never a legality verdict ---------------------------

def test_dispatch_is_off_the_legality_path():
    # The dispatcher EXECUTES (produces data); it must NEVER render or alter an R-law legality verdict.
    # The module verifies clean BEFORE any dispatch, and STILL verifies clean after the end-to-end run --
    # the dispatch produced no Diagnostic and did not touch the verifier (it is a graded backend above G8).
    m = _saxpy_module()
    assert bverify.verify(m) == []                               # clean before dispatch
    plan = orchestrate(m, [_sycl_channel()], Theta.cool())
    store = seed_store(m, {71: list(_X), 72: list(_Y)})
    disp = SyclDispatcher()
    kernels = build_execute_kernels(m, plan, store, {"sycl_spirv": disp})
    try:
        execute(m, kernels)
    except DispatchUnavailable:
        pass                                                     # the run self-skips; the verdict check still holds
    finally:
        disp.close()
    assert bverify.verify(m) == []                               # STILL clean after dispatch (no verdict change)
    # try_emit_spirv likewise returns data/notes only -- it produces no Diagnostic.
    ok, note = try_emit_spirv(m, optimize(m, TargetProfile.x86_avx512(), Theta.cool()))
    assert isinstance(note, str)


# --- (6) the architectural boundary: SYCL is still NOT a link-flag edge ---------------------------------

def test_sycl_dispatch_is_not_a_link_flag_edge():
    # defends the boundary (mirrors test_sycl_channel): the emitted kernel name resolves to None (SYCL is a
    # compiler MODE -fsycl, not a c.call.libm: -l<lib> edge); the dispatcher added NO link rule.
    assert library_for_callee("bcir_saxpy") is None
    assert library_for_callee("sycl_saxpy_kernel") is None


# --- (7) the kernels/store wiring is built even without a dispatcher (non-sycl reference kernel) -------

def test_build_kernels_wires_every_placed_claim():
    sycl = _sycl_channel()
    m = _saxpy_module()
    plan = orchestrate(m, [sycl], Theta.cool())
    store = seed_store(m, {71: list(_X), 72: list(_Y)})
    # NO dispatcher registered -> the sycl-placed claim gets the minimal reference kernel (saxpy_reference),
    # so execute() still produces data for every claim (the seam is built regardless of a real backend).
    kernels = build_execute_kernels(m, plan, store, {})
    assert 7100 in kernels
    res = execute(m, kernels)
    assert res.executed == 1
    ref = saxpy_reference(_A, _X, _Y)
    assert store[73] == ref                                      # the reference kernel ran (pure data)


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")
    sys.exit(0)
