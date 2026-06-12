"""BCIR oracle CLI.

    python -m bcir.run vector_add --target nvidia_ptx --theta cool
    python -m bcir.run vector_add --target x86_avx512 --run      # compile+run via clang
    python -m bcir.run vector_add --emit-llvm

Prints the K_BCIR plan (per-claim selected realization + score) for a program on
a given target under a given runtime state Theta.
"""

from __future__ import annotations

import argparse
import sys

from .examples import PROGRAMS
from .gem import hydrate
from .kbcir import TARGETS, optimize
from .kbcir.cost import Theta
from .kbcir.weights import POLICIES, PERF
from .verify import verify

_THETAS = {"cool": Theta.cool(), "hot": Theta.hot(), "mem_bound": Theta.mem_bound()}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bcir.run", description="BCIR K_BCIR oracle")
    p.add_argument("program", nargs="?", default="vector_add", choices=sorted(PROGRAMS))
    p.add_argument("--target", default="x86_avx512", choices=sorted(TARGETS))
    p.add_argument("--theta", default="cool", choices=sorted(_THETAS))
    p.add_argument("--policy", default="latency", choices=sorted(POLICIES))
    p.add_argument("--emit-llvm", action="store_true", help="print the lowered LLVM IR")
    p.add_argument("--run", action="store_true", help="compile+run the lowering via clang (AOT)")
    p.add_argument("--jit", action="store_true", help="JIT-run the lowering via lli (in-process)")
    p.add_argument("--wasm", action="store_true", help="compile to WASM and run via node (self-checking)")
    p.add_argument("--schedule", action="store_true", help="print the CT2 concurrent wave schedule")
    p.add_argument("--budget", metavar="DIM=CAP[,DIM=CAP...]",
                   help="constrained (RCSP) selection, e.g. thermal=700,power=700")
    p.add_argument("--overlap", action="store_true",
                   help="price the plan under wave overlap: M(pi,Theta) makespan vs serial")
    p.add_argument("--codegen", metavar="TARGET",
                   help="per-target codegen via llc (aarch64|riscv64|nvptx64|bpf|x86_64|c|all)")
    args = p.parse_args(argv)

    module = PROGRAMS[args.program]()
    h = TARGETS[args.target]
    theta = _THETAS[args.theta]
    policy = POLICIES.get(args.policy, PERF)

    diags = verify(module)
    if diags:
        print(f"[verify] {len(diags)} diagnostic(s):")
        for d in diags:
            print(f"  {d.law}: {d.message}")

    print(f"program={args.program} target={h.name} theta={args.theta} policy={policy.name}")
    if args.budget:
        from .kbcir import Budget, Infeasible, optimize_constrained
        caps = {k: int(v) for k, v in (kv.split("=", 1) for kv in args.budget.split(","))}
        try:
            result = optimize_constrained(module, h, theta, policy, Budget.of(**caps))
        except Infeasible as exc:
            print(f"[rcsp] infeasible: {exc}")
            return 1
        print(f"[rcsp] budget {args.budget}")
    else:
        result = optimize(module, h, theta, policy)
    print(f"K_BCIR score = {result.score}")
    for step in result.steps:
        c = step.candidate
        print(f"  claim {step.claim_id} phase {step.phase_id}: "
              f"lane={c.lane.name} width={c.width} realization={c.name} cost={step.cost}")

    pack = hydrate(module, result)
    print(f"StreamPack: {len(pack.segments)} segment(s), "
          f"map_gen={pack.map_gen} data_gen={pack.data_gen} provenance_ok={pack.provenance_ok()}")

    if args.schedule:
        from .gem import schedule_concurrent
        sc = schedule_concurrent(module, h)
        print(f"[ct2] concurrent: {len(sc.waves)} wave(s) max_parallelism={sc.max_parallelism()} "
              f"ggg_tail={sc.ggg_tail} contention={sc.contention} affinity={sc.affinity}")

    if args.overlap:
        from .gem import price_scheduled
        sp = price_scheduled(module, result, h, theta, policy)
        print(f"[overlap] M(pi,Theta): makespan={sp.makespan} serial={sp.serial} "
              f"gain={sp.overlap_gain}")

    if args.emit_llvm or args.run or args.jit or args.wasm:
        from .lower import compile_and_run, emit_kernel_ll, jit_run, run_wasm_node
        try:
            if args.emit_llvm:
                print("\n; ---- lowered LLVM IR ----")
                print(emit_kernel_ll(module, result, fn_name="bcir_kernel"))
            if args.run:
                ok, out = compile_and_run(module, result, fn_name="bcir_kernel")
                print(f"[run] clang build+run {'OK' if ok else 'FAILED'}: {out.strip()}")
                if not ok:
                    return 1
            if args.jit:
                ok, out = jit_run(module, result, fn_name="bcir_kernel")
                print(f"[jit] lli run {'OK' if ok else 'FAILED'}: {out.strip()}")
                if not ok:
                    return 1
            if args.wasm:
                ok, out = run_wasm_node(module, result, fn_name="bcir_kernel")
                print(f"[wasm] node run {'OK' if ok else 'FAILED'}: {out.strip()}")
                if not ok:
                    return 1
        except NotImplementedError as exc:
            print(f"[lower] {exc}")

    if args.codegen:
        from .codegen import codegen, codegen_all, codegen_c
        try:
            if args.codegen == "all":
                items = codegen_all(module, result).items()
            elif args.codegen == "c":
                items = [("c", codegen_c(module, result))]
            else:
                items = [(args.codegen, codegen(module, result, args.codegen))]
            for name, r in items:
                size = len(r.artifact) if isinstance(r.artifact, (bytes, str)) and r.artifact else 0
                print(f"[codegen] {name}: {'OK' if r.ok else 'FAILED'} ({r.message[:60]}{' %dB' % size if size else ''})")
        except NotImplementedError as exc:
            print(f"[codegen] {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
