"""PX-2b (2026-10-20): the pack lowering's parts against the number of steps a pack carries -- what
`bcir_decoder_lower_pack` costs on the 9.84M comparison decoder (B=8, T=128) for packs of K steps,
and the verifiers it runs, each alone:

  sp_validate          the StreamPack's header, magic, version and CRC (`bcir_sp_validate`)
  sp_verify_semantic   its laws (`bcir_sp_verify_semantic`): R10 among them, linear since QUAL-LIN
  ep_validate          the plan's header and CRC (`bcir_ep_validate`)
  ep_verify            the plan's laws (`bcir_ep_verify`): its step walk's claim check (linear on
                       GEM's ascending plans, PX-2b) and the alias law (each lifetime against every
                       earlier one, no heap: quadratic in the lifetimes)
  ep_check_pack        the plan's laws and the pack's binding to it (`bcir_ep_check_pack`)
  lower_pack           the whole lowering (`NativeDecoder.lower_pack`: the sizes, the one walk of
                       the plan's laws that hands the lowering its lifetimes, the pack's walks and
                       GEM's dispatch twice)

each the median of REPS calls (15 by default), with the pack's claims, values and bytes.

  python3 lower.py OUT.json [--steps 1,2,5,10,20] [--reps 15] [--contract kblock+exp+reduce+attn]
      [--cc gcc]
"""

import argparse
import ctypes as ct
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", type=Path)
    p.add_argument("--steps", default="1,2,5,10,20")
    p.add_argument("--reps", type=int, default=15)
    p.add_argument("--contract", default="kblock+exp+reduce+attn")
    p.add_argument("--cc", default="gcc")
    a = p.parse_args()
    from bcir.frontends.models.decode import DecoderSpec
    from bcir.hosted.models import native as nat
    from bcir.hosted.models.native import NativeDecoder
    from bcir.hosted.models.native_launch import SPACES, step_program
    from bcir.hosted.models.native_pack import pack_artifacts, program_pack

    lib = NativeDecoder.build(a.out.parent / (a.out.stem + "-build"), cc=a.cc, pack=True)
    spec = DecoderSpec(260, 384, 8, 6, 1024, activation="silu_gate", n_kv_heads=4,
                       tied_embeddings=True)  # fmt: skip
    nd = NativeDecoder(lib, spec, batch_size=8, context_length=128, threads=1, contract=a.contract)
    L = nd.lib
    header = ct.create_string_buffer(4096)
    for name in ("bcir_sp_validate", "bcir_ep_validate"):
        getattr(L, name).argtypes = [ct.c_char_p, ct.c_size_t, ct.c_void_p]
    L.bcir_sp_verify_semantic.argtypes = [ct.c_char_p, ct.c_size_t, ct.c_uint32, ct.c_uint32]
    L.bcir_ep_verify.argtypes = [ct.c_char_p, ct.c_size_t]
    L.bcir_ep_check_pack.argtypes = [ct.c_char_p, ct.c_size_t, ct.c_char_p, ct.c_size_t]

    def median_ms(fn, *args) -> float:
        times = []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            fn(*args)  # each verdict checked once before the timing (lower_pack raises a refusal)
            times.append(time.perf_counter() - t0)
        return 1000 * statistics.median(times)

    rows = []
    for k in (int(x) for x in a.steps.split(",")):
        program = step_program(
            spec, 8, 128, contract=nat._CONTRACTS[a.contract], recompute=nd.recompute,
            workspace_bytes=nd.workspace_bytes, panels_need=nd.plan_memory.workspace_panels,
            alignment={SPACES[n]: ct.addressof(b) % 64 for n, b in nd.buffers.items()},
            steps=k, accumulation=1)  # fmt: skip
        tp = program_pack(spec, 8, 128, program, recompute=nd.recompute)
        pack, plan, _, _ = pack_artifacts(tp)
        for fn, args in ((L.bcir_sp_validate, (pack, len(pack), header)),
                         (L.bcir_sp_verify_semantic, (pack, len(pack), 0xFFFFFFFF, 0xFFFFFFFF)),
                         (L.bcir_ep_validate, (plan, len(plan), header)),
                         (L.bcir_ep_verify, (plan, len(plan))),
                         (L.bcir_ep_check_pack, (plan, len(plan), pack, len(pack)))):  # fmt: skip
            assert fn(*args) == 0, fn
        row = {"steps": k, "claims": tp.claims, "values": len(tp.placements),
               "pack_bytes": len(pack), "plan_bytes": len(plan),
               "ms": {"sp_validate": median_ms(L.bcir_sp_validate, pack, len(pack), header),
                      "sp_verify_semantic": median_ms(L.bcir_sp_verify_semantic, pack, len(pack),
                                                      0xFFFFFFFF, 0xFFFFFFFF),
                      "ep_validate": median_ms(L.bcir_ep_validate, plan, len(plan), header),
                      "ep_verify": median_ms(L.bcir_ep_verify, plan, len(plan)),
                      "ep_check_pack": median_ms(L.bcir_ep_check_pack, plan, len(plan), pack,
                                                 len(pack)),
                      "lower_pack": median_ms(nd.lower_pack, pack, plan)}}  # fmt: skip
        rows.append(row)
        print(
            json.dumps({key: row[key] for key in ("steps", "claims", "values")}),
            json.dumps({key: round(v, 3) for key, v in row["ms"].items()}),
            flush=True,
        )
    nd.close()
    out = {"cpu": next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo")
                       if line.startswith("model name")),
           "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                    capture_output=True, text=True).stdout.strip(),
           "worktree_diff_sha256_16": hashlib.sha256(subprocess.run(
               ["git", "-C", str(REPO), "diff", "HEAD", "--", "runtime/c", "bcir"],
               capture_output=True).stdout).hexdigest()[:16],
           "contract": a.contract, "reps": a.reps,
           "compiler": subprocess.run([a.cc, "--version"], capture_output=True,
                                      text=True).stdout.splitlines()[0],
           "rows": rows}  # fmt: skip
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
