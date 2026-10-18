"""Dependency-free native planning, kernels, learning and checkpoint gates.

Framework differentials run separately in the hosted-model CI job. The quick
runner hides compilers; pure layout/configuration contracts still execute.
"""

from __future__ import annotations

import json
import math
import mmap
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

from bcir.frontends.models.decode import DecoderSpec
from bcir.hosted.models.native import (
    NativeAdamW,
    NativeDecoder,
    parameter_layout,
    _checkpoint_digest,
)


#: The compiler the NDT-GEM witnesses build with (`bcir.toolchain.host_c_compiler`), or None: the
#: capability their fault tables (tools/testing/faults/ndt-gem.json, ndt-perf.json and
#: ndt-clang.json) require.
_CC = __import__("bcir.toolchain", fromlist=["host_c_compiler"]).host_c_compiler()
#: Whether Python exposes MADV_HUGEPAGE (Linux): where the trainer's state shares one mapping
#: advised for transparent huge pages (NDT-PAGES), the capability `ndt-pages.json` requires.
_MADV_HUGEPAGE = hasattr(mmap, "MADV_HUGEPAGE")
#: The fault tables the sweeps inject (tools/testing/faults); the thorough tests that inject a
#: fault of their own read it from here, so the two cannot drift apart.
_FAULTS = Path(__file__).resolve().parents[2] / "tools" / "testing" / "faults"


def _spec():
    return DecoderSpec(16, 8, 2, 2, 16, activation="silu_gate", n_kv_heads=1)


def _reject(fn):
    try:
        fn()
    except (ValueError, OverflowError):
        return
    raise AssertionError("invalid input accepted")


def test_native_parameter_layout_and_optimizer_contracts():
    layout = parameter_layout(_spec())
    assert sum(row[3] for row in layout) == 1320
    assert "lm_head.weight" not in {row[0] for row in layout}
    assert len(layout) == 20
    for settings in (
        {"lr": math.inf},
        {"epsilon": 0},
        {"beta1": 1},
        {"beta2": -1},
        {"grad_clip": -1},
        {"lr": True},
        {"weight_decay": math.nan},
    ):
        _reject(lambda settings=settings: NativeAdamW(**settings))
    NativeAdamW(lr=0, grad_clip=0, beta1=0, beta2=0)


def test_native_decoder_learning_resume_memory_limits_and_corruption():
    cc = shutil.which("clang") or shutil.which("cc")
    if not cc:
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
        root = Path(temporary)
        lib = NativeDecoder.build(root, cc=cc)
        trainer = NativeDecoder(lib, _spec())
        _reject(lambda: NativeDecoder(lib, _spec(), max_memory_bytes=100))
        _reject(lambda: NativeDecoder(lib, _spec(), context_length=2**32))
        _reject(lambda: trainer.backward([1, 2, 3, 16], [2, 3, 4, 5]))
        inputs, targets = [1, 2, 3, 4], [2, 3, 4, 5]
        opt = NativeAdamW(lr=0.01, weight_decay=0.01)
        events = trainer.run(inputs * 40, targets * 40, steps=40, optimizer=opt)
        assert events[-1]["loss"] < events[0]["loss"] * 0.5
        path = root / "checkpoint.bcirdt"
        trainer.save(path)
        other = NativeDecoder.resume(lib, path)
        assert other.state.step == 40
        assert trainer.run(inputs * 2, targets * 2, steps=2, optimizer=opt) == other.run(
            inputs * 2, targets * 2, steps=2, optimizer=opt
        )
        for name in ("weights", "moment1", "moment2"):
            assert bytes(trainer.buffers[name]) == bytes(other.buffers[name])
        before = bytes(trainer.buffers["weights"])
        trainer.buffers["gradients"][-1] = math.nan
        _reject(lambda: trainer.update(opt))
        assert bytes(trainer.buffers["weights"]) == before
        original = path.read_bytes()
        changed = bytearray(original)
        changed[-1] ^= 0x80
        path.write_bytes(changed)
        _reject(lambda: NativeDecoder.resume(lib, path))
        size = struct.unpack("<I", original[8:12])[0]
        meta = json.loads(original[12 : 12 + size])
        altered = dict(meta)
        altered["decoder"] = dict(meta["decoder"], rope_base=1234.0)
        header = json.dumps(altered).encode()
        path.write_bytes(
            original[:8] + struct.pack("<I", len(header)) + header + original[12 + size :]
        )
        _reject(lambda: NativeDecoder.resume(lib, path))
        header = original[12 : 12 + size].decode().rstrip("}") + ',"step":0}'
        encoded = header.encode()
        path.write_bytes(
            original[:8] + struct.pack("<I", len(encoded)) + encoded + original[12 + size :]
        )
        _reject(lambda: NativeDecoder.resume(lib, path))
        payload = bytearray(original[12 + size :])
        payload[:4] = struct.pack("<f", math.nan)
        meta["sha256"] = _checkpoint_digest(meta, payload)
        header = json.dumps(meta).encode()
        path.write_bytes(original[:8] + struct.pack("<I", len(header)) + header + payload)
        _reject(lambda: NativeDecoder.resume(lib, path))
        path.write_bytes(original + b"x")
        _reject(lambda: NativeDecoder.resume(lib, path))
        staged = trainer.state_dict()
        staged["model.norm.weight"][0] = math.inf
        _reject(lambda: trainer.load_state_dict(staged))
        assert bytes(trainer.buffers["weights"]) == before


def test_native_c_harness_c11_and_c23():
    cc = shutil.which("clang") or shutil.which("cc")
    if not cc:
        return
    root = Path(__file__).resolve().parents[2]
    c = root / "runtime" / "c"
    from bcir.toolchain import host_link_args

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
        # and C23 once more with the writers' verdict bytes bounded at two tasks, so every scan
        # entry takes the scan after its job (NDT-SCAN-W), and once capped at AVX2, so a host whose
        # widest variant is AVX-512 runs the update's AVX2 pairs and commit too (NDT-UPDATE); every
        # build with the commit's shared division exposed to the harness's check of it
        builds = (
            ("c11", [], ""),
            ("c2x", [], ""),
            ("c2x", ["-DBCIR_TENSOR_SCAN_TASKS=2"], "-bounded"),
            ("c2x", ["-DBCIR_TENSOR_MAX_VARIANT=2"], "-avx2"),
        )
        for standard, options, tag in builds:
            exe = Path(temporary) / f"gate-{standard}{tag}"
            cmd = [
                cc,
                f"-std={standard}",
                "-DBCIR_DT_DIVISION_PROBE",
                "-DBCIR_DT_WORKSPACE_PROBE",
                "-DBCIR_DT_ATTENTION_PROBE",
                "-DBCIR_DT_PANELS_PROBE",
                *options,
                "-O2",
                "-Wall",
                "-Wextra",
                "-Wpedantic",
                "-Werror",
                "-ffp-contract=off",
                "-I",
                str(c),
                str(c / "test_decoder_train.c"),
                str(c / "bcir_tensor.c"),
                str(c / "bcir_decoder_train.c"),
                "-o",
                str(exe),
            ]
            subprocess.run(
                cmd + host_link_args(["-lm"]), check=True, capture_output=True, timeout=120
            )
            result = subprocess.run(
                [str(exe)], check=True, capture_output=True, text=True, timeout=60
            )
            # update_pairs, division_probe (NDT-UPDATE); the bf16 contract, its trainer and the
            # workspace probe (NDT-BF16); the fused attention contract and its trainer (ATTN-FMA);
            # the bf16 panels' writers and grouped GEMMs, and their trainer (BF16-PANELS)
            assert result.stdout.count("PASS") == 24
        # Once more under UBSan, float-cast-overflow included: every path of the harness -- the
        # trainer's NaN and overflow refusals among them -- is defined behaviour (the exponential's
        # clamp let a NaN reach an integer conversion until NDT-SCAN's sanitized run found it). The
        # first compiler whose UBSan runtime builds and runs a probe here; with none, this leg is a
        # skip, as the C harness itself is without a compiler.
        sanitize = ["-fsanitize=undefined,float-cast-overflow", "-fno-sanitize-recover=all"]
        probe = Path(temporary) / "probe.c"
        probe.write_text("int main(void) { return 0; }\n", encoding="utf-8")
        for engine in (shutil.which("clang"), shutil.which("gcc")):
            if not engine:
                continue
            probe_exe = Path(temporary) / "probe"
            built = subprocess.run(
                [engine, *sanitize, str(probe), "-o", str(probe_exe)],
                capture_output=True,
                timeout=120,
            )
            if built.returncode or subprocess.run([str(probe_exe)], timeout=60).returncode:
                continue
            exe = Path(temporary) / "gate-ubsan"
            cmd = [engine, "-std=c11", "-O1", "-g", "-ffp-contract=off", *sanitize]
            cmd += [
                "-DBCIR_DT_DIVISION_PROBE",
                "-DBCIR_DT_WORKSPACE_PROBE",
                "-DBCIR_DT_ATTENTION_PROBE",
                "-DBCIR_DT_PANELS_PROBE",
            ]
            cmd += ["-I", str(c)]
            cmd += [str(c / u) for u in ("test_decoder_train.c", "bcir_tensor.c")]
            cmd += [str(c / "bcir_decoder_train.c"), "-o", str(exe)]
            subprocess.run(
                cmd + host_link_args(["-lm"]), check=True, capture_output=True, timeout=300
            )
            result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300)
            assert result.returncode == 0 and "runtime error" not in result.stderr, result.stderr[
                -2000:
            ]
            assert result.stdout.count("PASS") == 24
            break


def test_native_provider_c_abi_keeps_frameworks_off_execution_path():
    cc = shutil.which("g++") or shutil.which("clang++")
    if not cc:
        return
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
        src = '#include "bcir_decoder_train.h"\nstatic_assert(sizeof(float)==4);\nint f(){return BCIR_DT_OK;}\n'
        subprocess.run(
            [
                cc,
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Wpedantic",
                "-Werror",
                "-I",
                str(root / "runtime" / "c"),
                "-x",
                "c++",
                "-c",
                "-o",
                str(Path(temporary) / "boundary.o"),
                "-",
            ],
            input=src,
            text=True,
            capture_output=True,
            check=True,
            timeout=60,
        )


# --- NDT-GEM: the step held to its GEM+ program (bcir/tests/native_decoder_fixtures.py) -----


def _ndt_library(tmp, sources=None):
    from bcir.tests import native_decoder_fixtures as nf

    path = nf.build_library(tmp, sources=sources)
    return None if path is None else __import__("ctypes").CDLL(str(path))


def test_the_compiled_plan_and_gemm_are_the_programs():
    """Both exact rows at zero: the C plan's arenas are the program's over the corpus, and
    bcir_tensor_mm is its binary32 contract bit for bit on all 20 cases."""
    from bcir.tests import native_decoder_fixtures as nf

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        lib = _ndt_library(tmp)
        if lib is None:
            return  # no C compiler: the native rows are a skip, never a pass
        assert nf.plan_mismatch(lib) == 0
        assert nf.mm_mismatch(lib) == 0
        assert nf.mm_fused_mismatch(lib) == 0
        assert nf.mm_kblock_mismatch(lib) == 0
        assert nf.mm_bf16_mismatch(lib) == 0 and 1 in nf.mm_bf16_variants(lib)
        assert nf.provider_unsound(lib._name) in (None, 0)  # None: no OpenBLAS wheel, a skip
        swept, mismatch, ulp = nf.exp_sweep(lib)
        assert swept > 8_000_000 and mismatch == 0 and ulp <= 1, (swept, mismatch, ulp)
        assert len(nf.MM_CASES) == 20 and len(nf.MM_KBLOCK_CASES) == 40
        assert len(nf.MM_BF16_CASES) == 28
        assert len(nf.PLAN_CORPUS) == 6
        assert nf.kblock_split(1024) == [448, 288, 288] and nf.kblock_split(0) == []


def test_the_rows_fire_on_wrong_kernels_and_plans():
    """Wrong copies of the C units, built from copies of their sources (never by editing the
    tree): the packed B^T tile read without its depth offset, the register block accumulated in
    descending order or fused into one rounding, A read untransposed whatever ta says, the
    blocked kernel's generic and AVX-512 instances each wrong in their own code (beta's fold
    into the first K block among them, NDT-BETA); the plan's workspace or cache count off by
    one span. Each turns its row red."""
    from bcir.tests import native_decoder_fixtures as nf

    if nf.compiler() is None:
        return
    tensor, train = nf.source("bcir_tensor.c"), nf.source("bcir_decoder_train.c")
    mutants = (
        ("mm", "bcir_tensor.c", tensor,
         "const float *src=x->b+(jb+j)*x->ldb+lb;", "const float *src=x->b+(jb+j)*x->ldb;"),
        ("mm", "bcir_tensor.c", tensor,
         "for (l=0;l<nl;l++) {\n    const float *bl=bt+l*nr;",
         "for (l=nl;l-->0;) {\n    const float *bl=bt+l*nr;"),
        ("mm", "bcir_tensor.c", tensor,
         "acc[r][j]+=av[r]*bl[j];", "acc[r][j]=fmaf(av[r],bl[j],acc[r][j]);"),
        ("mm", "bcir_tensor.c", tensor,
         "x->a[x->ta ? (lb+l)*x->lda+i+r : (i+r)*x->lda+lb+l]", "x->a[(i+r)*x->lda+lb+l]"),
        ("mm_fused", "bcir_tensor.c", tensor,
         "acc[r][j]=fmaf(av[r],bl[j],acc[r][j]);", "acc[r][j]+=av[r]*bl[j];"),
        ("mm_fused", "bcir_tensor.c", tensor,
         "mm_variant_any(mm_widest_fused(),MM_FUSED,", "mm_variant_any(mm_widest_fused(),MM_V1,"),
        ("mm_kblock", "bcir_tensor.c", tensor,
         "return rem>=2*q ? q : rem>q ? (rem/2+u-1)/u*u : rem;", "return rem>q ? q : rem;"),
        ("mm_kblock", "bcir_tensor.c", tensor,
         "  for (r=0;r<mr;r++) for (j=0;j<nr;j++) acc[r][j]=0.0f;\n",
         "  for (r=0;r<mr;r++) for (j=0;j<nr;j++) acc[r][j]=j<nj ? x->c[(i+r)*x->ldc+jb+j] : 0.0f;\n"),
        ("mm_kblock", "bcir_tensor.c", tensor,
         "    else for (j=0;j<nr;j++) if (j<nj) ci[j]=ci[j]+x->alpha*acc[r][j];\n",
         "    else for (j=0;j<nr;j++) if (j<nj) ci[j]=fmaf(x->alpha,acc[r][j],ci[j]);\n"),
        # NDT-BETA: the generic instance scales c at every K block, not only the first
        ("mm_kblock", "bcir_tensor.c", tensor,
         "  first=x->fold && lb==0;\n", "  first=x->fold;\n"),
        # NDT-BLOCK: the AVX-512 instance is its own code, so its own wrong copies -- the last
        # single row left at its beta-scaled value, a transposed tile's ragged columns stale
        # below the whole blocks, a ragged block's partial added with one fused rounding
        ("mm_kblock@4", "bcir_tensor.c", tensor,
         "      for (;i<r->r1;i++) kb512_block(1,x,bt,i,jb,nj,lb,nl);\n",
         "      for (;i+1<r->r1;i++) kb512_block(1,x,bt,i,jb,nj,lb,nl);\n"),
        ("mm_kblock@4", "bcir_tensor.c", tensor,
         "      for (l=j<jw ? lw : 0;l<nl;l++) bt[l*48+j]=column[l];\n",
         "      for (l=lw;l<nl;l++) bt[l*48+j]=column[l];\n"),
        ("mm_kblock@4", "bcir_tensor.c", tensor,
         "      _mm512_mask_storeu_ps(ci,m0,_mm512_add_ps(kb512_c(ci,m0&keep,scale,beta),_mm512_mul_ps(alpha,acc0[r])));\n",
         "      _mm512_mask_storeu_ps(ci,m0,_mm512_fmadd_ps(alpha,acc0[r],kb512_c(ci,m0&keep,scale,beta)));\n"),
        # NDT-BETA: the AVX-512 instance's first K block reads c under beta 0
        ("mm_kblock@4", "bcir_tensor.c", tensor,
         "  keep=(__mmask16)(x->fold && lb==0 && x->beta==0.0f ? 0u : 0xffffu);\n",
         "  keep=(__mmask16)0xffffu;\n"),
        # NDT-BF16: the portable instance's lanes swapped at the block's end, its pack's pairs
        # interleaved odd first, and the region's columns one block short at a ragged edge
        ("mm_bf16", "bcir_tensor.c", tensor,
         "        acc=bf16_ftz(bf16_ftz(acc)+bf16_ftz(e+o));\n",
         "        acc=bf16_ftz(bf16_ftz(acc)+bf16_ftz(e)+o);\n"),
        ("mm_bf16", "bcir_tensor.c", tensor,
         "      t[p*32+2*jj+e]=j<x->n && d<x->k ?", "      t[p*32+2*jj+(1-e)]=j<x->n && d<x->k ?"),
        ("mm_bf16", "bcir_tensor.c", tensor,
         "  r1=r0+BF16_ROWS<rend ? r0+BF16_ROWS : rend; c1=c0+c->cols<cend ? c0+c->cols : cend;",
         "  r1=r0+BF16_ROWS<rend ? r0+BF16_ROWS : rend; c1=c0+c->cols<cend ? c0+c->cols : cend-1;"),
        # the AMX instance's tiles out of order: the admission withdraws it where the host has it
        ("bf16_admission", "bcir_tensor.c", tensor,
         "      _tile_dpbf16ps(0,4,6); _tile_dpbf16ps(1,4,7); _tile_dpbf16ps(2,5,6); _tile_dpbf16ps(3,5,7);",
         "      _tile_dpbf16ps(0,4,7); _tile_dpbf16ps(1,4,6); _tile_dpbf16ps(2,5,6); _tile_dpbf16ps(3,5,7);"),
        ("exp_ulp", "bcir_tensor.c", tensor,
         "p=1.0f/5040.0f; p=p*r+1.0f/720.0f; p=p*r+1.0f/120.0f;", "p=1.0f/120.0f;"),
        ("exp_ulp", "bcir_tensor.c", tensor,
         "#define BCIR_EXP_LN2_LO 1.428606765330187e-06f", "#define BCIR_EXP_LN2_LO 0.0f"),
        ("exp_mismatch", "bcir_tensor.c", tensor,
         "static void exp_avx2(size_t n, const float *x, float *y) { exp_body(n,x,y); }",
         "static void exp_avx2(size_t n, const float *x, float *y) { size_t i; for (i=0;i<n;i++) y[i]=expf(x[i]); }"),
        ("plan", "bcir_decoder_train.c", train,
         "!term(&p.scratch,5,nd)", "!term(&p.scratch,6,nd)"),
        ("plan", "bcir_decoder_train.c", train,
         "!term(&la,3,nf)", "!term(&la,4,nf)"),
    )  # fmt: skip
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for i, (row, unit, text, old, new) in enumerate(mutants):
            assert text.count(old) == 1, old
            lib = _ndt_library(Path(tmp) / str(i), {unit: text.replace(old, new)})
            if row == "exp_ulp":
                fired = nf.exp_sweep(lib)[2] > 1
            elif row == "exp_mismatch":
                fired = (
                    nf.exp_sweep(lib)[1]
                    if nf.exp_sweep(lib)[1]
                    else int(
                        __import__("ctypes").CDLL(str(lib._name)).bcir_tensor_mm_variants() == 1
                    )
                )  # a host without an AVX2 instance cannot see this one: not a pass, a skip
            elif row == "bf16_admission":
                amx = 8 in nf.mm_bf16_variants(
                    __import__("ctypes").CDLL(str(_ndt_library(Path(tmp) / "clean")._name))
                )
                fired = 8 not in nf.mm_bf16_variants(lib) if amx else 1
                # a host without AMX cannot see this one: not a pass, a skip
            elif row.startswith("mm_kblock@"):
                need = int(row.partition("@")[2])
                fired = (
                    nf.mm_kblock_mismatch(lib) if need in nf.mm_variants(lib, kblock=True) else 1
                )  # a host without that instance cannot see this one: not a pass, a skip
            else:
                fired = {"mm": nf.mm_mismatch, "mm_fused": nf.mm_fused_mismatch,
                         "mm_kblock": nf.mm_kblock_mismatch, "mm_bf16": nf.mm_bf16_mismatch,
                         "plan": nf.plan_mismatch}[row](lib)  # fmt: skip
            assert fired > 0, f"train.{row}.mismatch cannot see {new!r}"


def test_the_training_state_is_identical_at_every_thread_count():
    """NDT-THREADS: three steps of the threads bench on a trainer of one, two, three and four
    threads leave one state, bit for bit (the row at zero), under v1 and under the fused contract
    (whose state is its own), and under the reduce contracts (NDT-REDUCE: the one reduction the
    pool reaches is chunked by the count alone, so the state -- and the losses and norms the
    steps report, which the digest folds in -- is the one-thread run's there too);
    a trainer resumed on more threads continues the same state, with the checkpoint's contract; a
    thread count outside [1, 4096] or an unknown contract is refused before any pool."""
    from bcir.hosted.models.native import NativeAdamW, NativeDecoder
    from bcir.tests import native_decoder_fixtures as nf

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = nf.build_library(tmp)
        if path is None:
            return  # no C compiler: the row is a skip, never a pass
        states = nf.thread_states(path)
        assert set(states) == set(nf.THREAD_COUNTS) and len(set(states.values())) == 1, states
        assert nf.threads_mismatch(path) == 0
        fused = nf.thread_states(path, contract="fused")
        assert len(set(fused.values())) == 1 and fused[1] != states[1], fused
        digests = {states[1], fused[1]}
        for name in ("exp", "fused+exp"):
            under = nf.thread_states(path, contract=name)
            assert len(set(under.values())) == 1 and under[1] not in digests, (name, under)
            digests.add(under[1])
        # NDT-BF16: the bf16 contract's state is its own, one at every thread count
        under = nf.thread_states(path, contract="bf16")
        assert len(set(under.values())) == 1 and under[1] not in digests, under
        # ATTN-FMA: the fused attention's state is its own, one at every thread count
        for name in ("attn", "kblock+exp+attn"):
            under = nf.thread_states(path, contract=name)
            assert len(set(under.values())) == 1 and under[1] not in digests, (name, under)
            digests.add(under[1])
        # the reduce bit changes the norm's last bits, which the float rounding of every update
        # absorbs over three steps of this bench: one state at every thread count is the claim
        for name in ("reduce", "kblock+exp+reduce", "bf16+exp+reduce", "bf16+exp+reduce+attn"):
            under = nf.thread_states(path, contract=name)
            assert set(under) == set(nf.THREAD_COUNTS) and len(set(under.values())) == 1, under
        spec, batch, time_ = nf.THREADS_BENCH
        one = NativeDecoder(path, spec, batch_size=batch, context_length=time_)
        assert one.threads == 1 and one.lib.bcir_tensor_pool_threads(one.state.pool) == 1
        x = [(3 * i + 1) % spec.vocab_size for i in range(batch * time_ * 2)]
        y = [(3 * i + 2) % spec.vocab_size for i in range(batch * time_ * 2)]
        one.run(x, y, steps=2, optimizer=NativeAdamW(lr=2e-3))
        one.save(Path(tmp) / "two-steps.bcirdt")
        four = NativeDecoder.resume(path, Path(tmp) / "two-steps.bcirdt", threads=4)
        assert four.threads == 4 and four.lib.bcir_tensor_pool_threads(four.state.pool) == 4
        assert (one.contract, four.contract, four.state.contract) == ("v1", "v1", 0)
        assert one.run(x, y, steps=2, optimizer=NativeAdamW(lr=2e-3)) == four.run(
            x, y, steps=2, optimizer=NativeAdamW(lr=2e-3)
        )
        assert nf.state_digest(one) == nf.state_digest(four)
        four.close()
        four.close()  # idempotent
        assert four.state.pool is None
        one.close()
        for bad in (0, -1, 4097, 2.0, True):
            try:
                NativeDecoder(path, spec, batch_size=batch, context_length=time_, threads=bad)
            except ValueError:
                continue
            raise AssertionError(f"threads={bad!r} admitted")
        for bad in ("v2", 1, None):
            try:
                NativeDecoder(path, spec, batch_size=batch, context_length=time_, contract=bad)
            except ValueError:
                continue
            raise AssertionError(f"contract={bad!r} admitted")
        two = NativeDecoder(path, spec, batch_size=batch, context_length=time_, contract="fused")
        assert two.state.contract == 1
        two.run(x, y, steps=2, optimizer=NativeAdamW(lr=2e-3))
        two.save(Path(tmp) / "fused.bcirdt")
        back = NativeDecoder.resume(path, Path(tmp) / "fused.bcirdt", threads=2)
        assert (back.contract, back.state.contract) == ("fused", 1)
        assert nf.state_digest(back) == nf.state_digest(two)
        two.close()
        back.close()


def test_the_bf16_contract_lends_its_workspace_and_admits_no_provider():
    """NDT-BF16: a trainer under a bf16 contract lends its GEMMs the workspace the C plan asks
    (bcir_decoder_workspace_bytes), counted in the memory budget and none under the other
    contracts; a CBLAS provider is refused under it (an sgemm computes in binary32) unless
    installed for measurement; its checkpoint resumes it, contract and workspace, as one state."""
    from bcir.hosted.models.native import CONTRACTS, NativeAdamW, NativeDecoder
    from bcir.tests import native_decoder_fixtures as nf

    assert {"bf16", "bf16+exp", "bf16+reduce", "bf16+exp+reduce"} <= set(CONTRACTS)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = nf.build_library(tmp)
        if path is None:
            return  # no C compiler: a skip, never a pass
        spec, batch, time_ = nf.THREADS_BENCH
        plain = NativeDecoder(path, spec, batch_size=batch, context_length=time_, contract="kblock")
        bf16 = NativeDecoder(path, spec, batch_size=batch, context_length=time_, threads=2,
                             contract="bf16+exp")  # fmt: skip
        assert plain.workspace_bytes == 0 and plain.state.workspace is None
        assert bf16.workspace_bytes > 0 and bf16.state.workspace_bytes == bf16.workspace_bytes
        assert bf16.memory_bytes == plain.memory_bytes + bf16.workspace_bytes
        assert bf16.state.contract == 18
        try:
            NativeDecoder(path, spec, batch_size=batch, context_length=time_, contract="bf16",
                          max_memory_bytes=plain.memory_bytes)  # fmt: skip
        except ValueError as e:
            assert "budget" in str(e)
        else:
            raise AssertionError("the workspace escaped the budget")
        try:
            bf16.use_cblas_function(None)
        except ValueError as e:
            assert "not admitted under the bf16+exp contract" in str(e)
        else:
            raise AssertionError("a CBLAS provider admitted under bf16")
        assert bf16.state.provider.mm is None or not bf16.state.provider.ctx
        x = [(3 * i + 1) % spec.vocab_size for i in range(batch * time_ * 2)]
        y = [(3 * i + 2) % spec.vocab_size for i in range(batch * time_ * 2)]
        bf16.run(x, y, steps=2, optimizer=NativeAdamW(lr=2e-3))
        bf16.save(Path(tmp) / "bf16.bcirdt")
        back = NativeDecoder.resume(path, Path(tmp) / "bf16.bcirdt", threads=4)
        assert (back.contract, back.state.contract) == ("bf16+exp", 18)
        assert back.workspace_bytes == bf16.workspace_bytes
        assert nf.state_digest(back) == nf.state_digest(bf16)
        assert bf16.run(x, y, steps=2, optimizer=NativeAdamW(lr=2e-3)) == back.run(
            x, y, steps=2, optimizer=NativeAdamW(lr=2e-3)
        )
        assert nf.state_digest(back) == nf.state_digest(bf16)
        for trainer in (plain, bf16, back):
            trainer.close()


def test_the_bf16_panels_keep_every_state():
    """BF16-PANELS: under a bf16 contract the trainer writes the SwiGLU block's outputs -- the
    hidden activation, the gate and up adjoints -- as the bf16 GEMMs' panels where their GEMMs
    are whole 32-tiles and their slots 64-byte aligned, and the GEMMs read them pre-packed. Three
    steps of a bench that admits them leave the state a build without panels
    (BCIR_DT_NO_PANELS) leaves, bit for bit, under each bf16 contract, at one thread and at four,
    on the huge-page placement and on ctypes arrays. The probe says which panel paths ran, so the
    agreement is not vacuous: all three on the huge-page mapping (Linux), none in the build
    without them, none under a binary32 GEMM contract."""
    import ctypes as ct
    import mmap

    from bcir.tests import native_decoder_fixtures as nf

    unit = nf.source("bcir_decoder_train.c")
    probe = "#define BCIR_DT_PANELS_PROBE 1\n"
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        built = {
            "panels": nf.build_library(
                Path(tmp) / "panels", sources={"bcir_decoder_train.c": probe + unit}
            )
        }
        if built["panels"] is None:
            return  # no C compiler: a skip, never a pass
        built["none"] = nf.build_library(
            Path(tmp) / "none",
            sources={"bcir_decoder_train.c": probe + "#define BCIR_DT_NO_PANELS 1\n" + unit},
        )
        # the path NativeDecoder loads (resolved), so the probe is the trainers' own library's
        libs = {name: ct.CDLL(str(Path(path).resolve())) for name, path in built.items()}
        for lib in libs.values():
            lib.bcir_decoder_panels_probe.restype = ct.c_uint
            lib.bcir_decoder_panels_probe.argtypes = []
        for contract in ("bf16", "bf16+exp", "bf16+exp+reduce+attn"):
            for placement in (True, False):
                states, seen = {}, {}
                for name, path in built.items():
                    libs[name].bcir_decoder_panels_probe()
                    states[name] = nf.thread_states(path, nf.PANELS_BENCH, counts=(1, 4),
                                                    contract=contract, huge_pages=placement)  # fmt: skip
                    seen[name] = libs[name].bcir_decoder_panels_probe()
                assert states["panels"] == states["none"], (contract, placement, states)
                assert len(set(states["none"].values())) == 1, (contract, placement, states)
                assert seen["none"] == 0, (contract, placement, seen)
                # the forward's path and the down projection's go together (one slot, one predicate)
                assert seen["panels"] in (0, 3, 4, 7), (contract, placement, seen)
                if placement and hasattr(mmap, "MADV_HUGEPAGE"):
                    assert seen["panels"] == 7, (contract, seen)
        libs["panels"].bcir_decoder_panels_probe()
        nf.thread_states(built["panels"], nf.PANELS_BENCH, counts=(1,), contract="kblock+exp")
        assert libs["panels"].bcir_decoder_panels_probe() == 0


def _vma(address):
    """The /proc/self/smaps entry (Linux) whose range holds `address`: its bounds, its
    permissions and its VmFlags."""
    entry = current = None
    for line in Path("/proc/self/smaps").read_text().splitlines():
        head = line.split()
        if not head:
            continue
        if head[0].endswith(":"):
            if current is not None and head[0] == "VmFlags:":
                current["flags"] = tuple(head[1:])
            continue
        lo, hi = (int(x, 16) for x in head[0].split("-"))
        current = {"lo": lo, "hi": hi, "perms": head[1], "flags": ()}
        current = current if lo <= address < hi else None
        entry = current or entry
    return entry


def test_the_state_lives_on_huge_pages_where_the_host_has_them():
    """NDT-PAGES: where Python exposes MADV_HUGEPAGE (Linux) the trainer's six state buffers
    share one private anonymous mapping, advised for transparent huge pages -- the advice taken
    wherever the kernel has them -- with the first buffer on a 2 MB boundary and each next 64
    bytes past the last one's end, in the plan's order; three steps of the threads bench leave
    the state the ctypes arrays hold (`huge_pages=False`), bit for bit, at one and at four
    threads, so placement never reaches a value. A resumed trainer takes the placement it is
    given; a placement that is not a bool is refused before any pool."""
    import ctypes as ct

    from bcir.hosted.models.native import NativeAdamW, NativeDecoder
    from bcir.tests import native_decoder_fixtures as nf

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = nf.build_library(tmp)
        if path is None:
            return  # no C compiler: the row is a skip, never a pass
        mapped = nf.thread_states(path, counts=(1, 4))
        arrays = nf.thread_states(path, counts=(1, 4), huge_pages=False)
        assert mapped == arrays and len(set(mapped.values())) == 1, (mapped, arrays)
        spec, batch, time_ = nf.THREADS_BENCH
        one = NativeDecoder(path, spec, batch_size=batch, context_length=time_)
        plain = NativeDecoder(path, spec, batch_size=batch, context_length=time_, huge_pages=False)
        assert plain.huge_pages is False
        names = ("weights", "gradients", "moment1", "moment2", "activations", "scratch")
        assert tuple(one.buffers) == tuple(plain.buffers) == names
        if _MADV_HUGEPAGE:
            if Path("/sys/kernel/mm/transparent_hugepage").is_dir():
                assert one.huge_pages is True  # a kernel with THP takes the advice
            spans = [(ct.addressof(one.buffers[n]), ct.sizeof(one.buffers[n])) for n in names]
            assert spans[0][0] % (2 << 20) == 0, hex(spans[0][0])
            for (start, size), (after, _) in zip(spans, spans[1:]):
                assert after == (start + size + 63) & ~63, (hex(start), size, hex(after))
            lo, hi = spans[0][0], spans[-1][0] + spans[-1][1]
            vma = _vma(lo)
            assert vma is not None and vma["lo"] <= lo and hi <= vma["hi"], (vma, lo, hi)
            assert vma["perms"][3] == "p", vma  # private: shared anonymous memory is shmem
            assert ("hg" in vma["flags"]) is one.huge_pages, vma
        else:
            assert one.huge_pages is False
        x = [(3 * i + 1) % spec.vocab_size for i in range(batch * time_ * 2)]
        y = [(3 * i + 2) % spec.vocab_size for i in range(batch * time_ * 2)]
        one.run(x, y, steps=2, optimizer=NativeAdamW(lr=2e-3))
        one.save(Path(tmp) / "pages.bcirdt")
        back = NativeDecoder.resume(path, Path(tmp) / "pages.bcirdt", huge_pages=False)
        again = NativeDecoder.resume(path, Path(tmp) / "pages.bcirdt")
        assert back.huge_pages is False and again.huge_pages is one.huge_pages
        assert nf.state_digest(back) == nf.state_digest(again) == nf.state_digest(one)
        for trainer in (one, plain, back, again):
            trainer.close()
        for bad in (1, 0, None, "yes"):
            try:
                NativeDecoder(path, spec, batch_size=batch, context_length=time_, huge_pages=bad)
            except ValueError:
                continue
            raise AssertionError(f"huge_pages={bad!r} admitted")


def test_a_provider_is_admitted_under_the_blocked_contract_only_where_it_is_the_contract():
    """`use_cblas` under "kblock" runs the admission differential over the trainer's own GEMM
    shapes at beta 0 and 1: BCIR's blocked kernel behind a CBLAS callback is admitted; the v1
    kernel is refused on every case and the fused kernel on the beta-1 cases (one block, but
    the chain from c), each leaving the state on BCIR's kernels; the differential refuses an
    empty shape set. Where the scipy_openblas32 wheel is importable, OpenBLAS is refused on
    the threads bench's small products (its small-matrix path keeps the whole chain), and on
    the comparison decoder's shapes the verdict agrees with the differential -- admitted where
    the wheel's target computes the contract, refused with the count where it does not: the
    admission is per host and per shape set, never a claim that a wheel is admitted. Under v1
    a provider is installed with no admission."""
    import ctypes as ct

    from bcir.hosted.models.native import _CBLAS, NativeDecoder, provider_mismatch
    from bcir.tests import native_decoder_fixtures as nf

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = nf.build_library(tmp)
        if path is None:
            return
        lib = ct.CDLL(str(path))
        spec, batch, time_ = nf.THREADS_BENCH
        fp = ct.POINTER(ct.c_float)
        sig = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_size_t, ct.c_size_t, ct.c_size_t,
               ct.c_float, fp, fp, ct.c_float, fp]  # fmt: skip

        def callback(name):
            fn = getattr(lib, name)
            fn.argtypes, fn.restype = sig, None
            return _CBLAS(
                lambda order, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc: fn(
                    None, ta == 112, tb == 112, m, n, k, alpha, a, b, beta, c
                )
            )

        verdicts = {}
        for name in ("bcir_tensor_mm_kblock", "bcir_tensor_mm", "bcir_tensor_mm_fused"):
            trainer = NativeDecoder(path, spec, batch_size=batch, context_length=time_,
                                    contract="kblock")  # fmt: skip
            try:
                trainer.use_cblas_function(callback(name), name=name)
                verdicts[name] = (trainer.provider_admission["mismatch"], trainer.backend)
            except ValueError as refusal:
                verdicts[name] = (str(refusal), trainer.backend)
                assert trainer.provider_admission is None and not trainer.state.provider.ctx
            finally:
                trainer.close()
        cases = 30  # the threads bench's fifteen distinct GEMM shapes at beta 0 and 1
        assert verdicts["bcir_tensor_mm_kblock"] == (0, "cblas:bcir_tensor_mm_kblock")
        assert verdicts["bcir_tensor_mm"][1] == "portable-c"
        assert f"{cases} of {cases} GEMM cases differ" in verdicts["bcir_tensor_mm"][0]
        assert f"{cases // 2} of {cases} GEMM cases differ" in verdicts["bcir_tensor_mm_fused"][0]
        try:
            provider_mismatch(lib, None, ())
        except ValueError:
            pass
        else:
            raise AssertionError("an empty differential admitted a provider")
        trainer = NativeDecoder(path, spec, batch_size=batch, context_length=time_)
        trainer.use_cblas_function(callback("bcir_tensor_mm"), name="v1")
        assert trainer.backend == "cblas:v1" and trainer.provider_admission is None
        trainer.close()
        openblas = nf.openblas_library()
        if openblas is None:
            return  # the wheel is an experiment dependency, never BCIR's: a skip
        trainer = NativeDecoder(path, spec, batch_size=batch, context_length=time_,
                                contract="kblock")  # fmt: skip
        try:
            trainer.use_cblas(openblas, symbol="scipy_cblas_sgemm")
        except ValueError as refusal:
            assert "not admitted" in str(refusal) and trainer.backend == "portable-c"
        else:
            raise AssertionError("OpenBLAS admitted on products its small-matrix path computes")
        finally:
            trainer.close()
        # the comparison decoder's shapes: admitted where the wheel computes the contract (the
        # SkylakeX target), refused with the differential's count where it does not (PR #817's
        # AMD host, 14 of 30): the verdict is sound on both, and that is the claim
        assert nf.provider_unsound(path) == 0


def test_the_threads_row_fires_on_a_partition_that_depends_on_the_thread_count():
    """Wrong copies of the tensor unit whose wrong partition reaches the state only on a pool
    of more than one thread (one thread runs the serial kernels): the GEMM's row tasks that
    stop before the ragged tail, and a runner whose claims stop one task short of every job
    (NDT-POOL: the threads claim their tasks from one counter). Each turns
    the row red (the C harness holds the partitions to the serial kernels at every count,
    and the TSan section holds their memory discipline) -- with a different state, or, since
    NDT-SCAN put the finiteness scans on the pool, with a refusal the serial trainer does not
    make (a dropped scan task's report is never written), which the row records as a state of
    its own rather than a traceback. The two copies are the sweep's own
    faults, `ndt-threads.json` NT1 and NT2, read from the table: one anchor for the sweep and
    for this test, so a kernel that moves (NDT-SCHED-0 moved the partition) cannot leave one
    of them pointing at a line that is no longer there."""
    from bcir.tests import native_decoder_fixtures as nf

    if nf.compiler() is None:
        return
    tensor = nf.source("bcir_tensor.c")
    table = json.loads(_FAULTS.joinpath("ndt-threads.json").read_text(encoding="utf-8"))
    faults = {f["label"].split(" ", 1)[0]: f for f in table["faults"]}
    mutants = [(faults[k]["old"], faults[k]["new"]) for k in ("NT1", "NT2")]
    assert all(
        faults[k]["path"] == "runtime/c/bcir_tensor.c"
        and faults[k]["expects"] == "test_the_training_state_is_identical_at_every_thread_count"
        for k in ("NT1", "NT2")
    )
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for i, (old, new) in enumerate(mutants):
            assert tensor.count(old) == 1, old
            path = nf.build_library(
                Path(tmp) / str(i), sources={"bcir_tensor.c": tensor.replace(old, new)}
            )
            states = nf.thread_states(path, refusals=True)
            # one thread runs the serial kernels: the state, never a refusal
            assert not states[1].startswith("refused: "), states[1]
            assert states[1] != states[2], f"train.state.threads.mismatch cannot see {new!r}"
            assert nf.threads_mismatch(path) > 0
        # a library that refuses at every count leaves no serial state to hold the others to: the
        # row counts that once per contract, where the four refusals would otherwise agree
        train = nf.source("bcir_decoder_train.c")
        scan = "!finite_span(j->x+i0,j->n-i0<j->chunk ? j->n-i0 : j->chunk);\n"
        assert train.count(scan) == 1, scan
        refusing = train.replace(scan, "1|" + scan)  # every chunk reports a non-finite element
        path = nf.build_library(Path(tmp) / "refusing", sources={"bcir_decoder_train.c": refusing})
        states = nf.thread_states(path, refusals=True)
        assert all(state.startswith("refused: ") for state in states.values()), states
        assert nf.threads_mismatch(path) == 2


def test_the_clipping_norm_keeps_its_contract():
    """NDT-REDUCE: the norm every update reports is its contract's, bit for bit, against the
    oracle's emulation from the gradients the backward left -- v1's index-order chain, and the
    reduce contract's chunk chains added in chunk order (the row at zero); the C chunk rule and
    the Python one agree on the counts around the first doubling and on the record's decoder; the
    two contracts' norms are one value to a part in 10^12 and not one double at every step (the
    threads bench's 135,864 parameters are 133 chunks). The same differential over one chunk of
    728 parameters: on a vector host the update's first pass takes the threads bench's whole pairs
    of chunks together (NDT-UPDATE), and its chunk loop then sums only the short last chunk, whose
    order rounds away in the total; one chunk's norm is that loop's chain alone."""
    from bcir.hosted.models.native import NativeAdamW, NativeDecoder
    from bcir.tests import native_decoder_fixtures as nf

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = nf.build_library(tmp)
        if path is None:
            return  # no C compiler: the row is a skip, never a pass
        assert nf.norm_mismatch(path) == 0
        spec, batch, time_ = nf.ONE_CHUNK_BENCH
        trainer = NativeDecoder(path, spec, batch_size=batch, context_length=time_,
                                contract="reduce")  # fmt: skip
        try:  # one reduce chunk: the norm is the first pass's chunk loop alone
            assert trainer.state.plan.parameters == 728
            assert nf.reduce_chunk(trainer.state.plan.parameters) >= 728
        finally:
            trainer.close()
        assert nf.norm_mismatch(path, bench=nf.ONE_CHUNK_BENCH) == 0
        assert [nf.reduce_chunk(c) for c in nf.REDUCE_CHUNK_COUNTS] == [
            1024,
            1024,
            1024,
            1024,
            1024,
            2048,
            16384,
            2048 * 1024,
        ]
        spec, batch, time_ = nf.THREADS_BENCH
        n = batch * time_
        x = [(3 * i + 1) % spec.vocab_size for i in range(n * 3)]
        y = [(3 * i + 2) % spec.vocab_size for i in range(n * 3)]
        norms = {}
        for name in ("v1", "reduce"):
            trainer = NativeDecoder(path, spec, batch_size=batch, context_length=time_,
                                    contract=name)  # fmt: skip
            try:
                assert trainer.state.plan.parameters == 135_864
                assert nf.reduce_chunk(trainer.state.plan.parameters) == 1024
                norms[name] = [e["grad_norm"] for e in trainer.run(x, y, steps=3,
                                                                   optimizer=NativeAdamW(lr=1e-3))]  # fmt: skip
            finally:
                trainer.close()
        assert all(abs(a - b) <= 1e-12 * b for a, b in zip(norms["reduce"], norms["v1"]))
        assert norms["reduce"] != norms["v1"], norms


def test_the_step_floor_is_measured_beside_the_step():
    """`train.step.ms`'s floor is G26's compute term at the best rate the same library reaches on
    the step's own GEMM shapes, in the same run: positive and below the step on a small bench."""
    from bcir.tests import native_decoder_fixtures as nf

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = nf.build_library(tmp)
        if path is None:
            return
        small = (DecoderSpec(64, 32, 4, 1, 64, activation="silu_gate", n_kv_heads=2), 2, 16)
        rows = nf.measure_step(path, small)
        assert 0 < rows["train.step.ms.floor"] < rows["train.step.ms"], rows


def test_the_harness_rows_are_the_programs_numbers():
    """`train.memory.transient`'s frozen baseline is the bench program's own ratio, and every
    row of the harness's `train` group is one the fixtures measure -- the four NDT-GEM rows, and
    TRAIN-RAIL's corpus row -- so a change to the program, the C plan or the corpus recipe
    cannot leave the harness quoting a stale number."""
    import sys

    from bcir.tests import native_decoder_fixtures as nf

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.perf.gemplus_baseline import METRICS

    rows = {m.key: m for m in METRICS if m.group == "train"}
    assert set(rows) == {"train.plan.mismatch", "train.mm.mismatch", "train.memory.transient",
                         "train.step.ms", "train.corpus.unselected",
                         "train.state.threads.mismatch", "train.mm.fused.mismatch",
                         "train.exp.mismatch", "train.exp.ulp.max",
                         "train.mm.kblock.mismatch", "train.mm.provider.unsound",
                         "train.mm.provider.differences", "train.norm.reduce.mismatch",
                         "train.mm.bf16.mismatch"}  # fmt: skip
    bf16 = rows["train.mm.bf16.mismatch"]
    assert bf16.kind == "exact" and bf16.bound == 0 and bf16.baseline == 0
    assert (
        bf16.what.startswith(f"of {len(nf.MM_BF16_CASES)} GEMM cases") and not bf16.host_dependent
    )
    kblock, unsound = rows["train.mm.kblock.mismatch"], rows["train.mm.provider.unsound"]
    differences = rows["train.mm.provider.differences"]
    assert kblock.kind == "exact" and kblock.bound == 0 and kblock.baseline == 0
    assert kblock.what.startswith(f"of {len(nf.MM_KBLOCK_CASES)} GEMM cases")
    # the guard's row is graded on every host; the wheel's count is the host's (PR817-P0)
    assert unsound.kind == "exact" and unsound.bound == 0 and unsound.baseline == 0
    assert unsound.what.startswith("whether the admission verdict") and not unsound.host_dependent
    assert differences.kind == "exact" and differences.bound == 0 and differences.baseline == 0
    assert differences.what.startswith("of 30 GEMM cases") and differences.host_dependent
    for key, baseline in (("train.exp.mismatch", 0), ("train.exp.ulp.max", 1)):
        assert rows[key].kind == "exact" and rows[key].bound == baseline == rows[key].baseline
    fused = rows["train.mm.fused.mismatch"]
    assert fused.kind == "exact" and fused.bound == 0 and fused.baseline == 0
    assert fused.what.startswith(f"of {len(nf.MM_CASES)} GEMM cases")
    threads = rows["train.state.threads.mismatch"]
    assert threads.kind == "exact" and threads.bound == 0 and threads.baseline == 0
    assert threads.what.startswith(f"thread counts among {nf.THREAD_COUNTS[1:]}")
    reduce = rows["train.norm.reduce.mismatch"]
    assert reduce.kind == "exact" and reduce.bound == 0 and reduce.baseline == 0
    assert reduce.what.startswith("of 6 clipping norms and 8 chunk counts")
    corpus = rows["train.corpus.unselected"]
    assert corpus.kind == "exact" and corpus.bound == 0 and corpus.baseline == 328
    unselected = nf.corpus_unselected()
    assert unselected is None or unselected == 0, unselected  # None outside a source checkout
    ratio, held, bound, incumbent = nf.transient_ratio()
    assert rows["train.memory.transient"].baseline == ratio == held / bound
    assert (held, bound, incumbent) == (999_424, 753_664, 753_664)
    assert rows["train.mm.mismatch"].what.startswith(f"of {len(nf.MM_CASES)} GEMM cases")
    pairs = len(nf.PLAN_CORPUS) * 2  # the parent stated activations and scratch only in C
    assert rows["train.plan.mismatch"].baseline == pairs
