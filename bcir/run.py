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
    p.add_argument("--eft", action="store_true",
                   help="duration-aware (HEFT-lite) schedule: LPT + EFT + locality + knee")
    p.add_argument("--tokens", action="store_true",
                   help="token-DAG execution schedule (pipelined phases, no barriers)")
    p.add_argument("--tables", metavar="FILE",
                   help="apply a frozen calibration table (JSON; see bcir.kbcir.microbench)")
    p.add_argument("--regret", action="store_true",
                   help="print the L3 regret ledger over the standard Theta episodes")
    p.add_argument("--moe", action="store_true",
                   help="train the learned MoE gate (GNN over the claim graph) on the "
                        "regret ledger and show its route vs the classify router")
    p.add_argument("--accel", action="store_true",
                   help="propose-verify search accelerator: greedy vs worst candidate "
                        "ordering (same optimum, fewer expansions)")
    p.add_argument("--manifest", action="store_true",
                   help="print the provenance manifest (the commit hash of this plan) "
                        "and confirm it replays")
    p.add_argument("--egraph", action="store_true",
                   help="building-blocks engine: how many shared blocks the claims "
                        "compose to after CSE (liked-pair memory)")
    p.add_argument("--soft-temp", metavar="T", type=float, default=None,
                   help="soft (log-sum-exp) plan distribution at temperature T "
                        "(T=0 reproduces the tropical optimizer exactly)")
    p.add_argument("--codegen", metavar="TARGET",
                   help="per-target codegen via llc (aarch64|riscv64|nvptx64|bpf|x86_64|c|all)")
    p.add_argument("--emit-c", action="store_true",
                   help="print the portable C23 kernel for the selected realization")
    p.add_argument("--run-c", action="store_true",
                   help="compile+run the C23 kernel self-check (clang -std=c23 / gcc -std=c2x)")
    p.add_argument("--calibrate", action="store_true",
                   help="close the calibration loop on real hardware: microbench this "
                        "host, freeze the Q8 table, fold --theta as telemetry, replan, "
                        "and certify the measured win")
    p.add_argument("--native", action="store_true",
                   help="with --calibrate, use the bare-metal C microbench (real cache "
                        "latency) instead of the conservative interpreter harness")
    p.add_argument("--bench", action="store_true",
                   help="measured-evidence rail: time BCIR's selected realization vs the "
                        "scalar baseline (compile + run); reports the measured speedup")
    p.add_argument("--bench-gather", action="store_true",
                   help="measured gather-avoidance: time BCIR's direct realization vs the "
                        "gather form it avoids (random indices); reports the gather penalty")
    p.add_argument("--bench-reduce", action="store_true",
                   help="measured gather/blocked reduction (gather_reduce): time BCIR's "
                        "selected blocked realization vs the gather form (same sum)")
    p.add_argument("--bench-strided", action="store_true",
                   help="measured strided gather-avoidance (saxpy_strided): time BCIR's "
                        "direct strided realization vs the gather form (same result)")
    args = p.parse_args(argv)

    module = PROGRAMS[args.program]()
    h = TARGETS[args.target]
    theta = _THETAS[args.theta]
    policy = POLICIES.get(args.policy, PERF)

    if args.tables:
        from .kbcir.microbench import load_table
        table = load_table(args.tables)
        h = table.apply(h)
        print(f"[tables] applied {args.tables} (cal_gen={h.cal_gen}, "
              f"gather_penalty={h.gather_penalty}, base_overhead={h.base_overhead})")

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

    if args.soft_temp is not None:
        from .kbcir import softselect
        s = softselect(module, h, theta, policy, args.soft_temp)
        print(f"[softdp] T={s.temperature} free_energy={s.free_energy:.2f} "
              f"hard_score={s.hard_score} gap={s.gap:.2f}")
        for cid in sorted(s.marginals):
            dist = ", ".join(f"{name}={p:.3f}" for name, p in
                             sorted(s.marginals[cid].items(), key=lambda kv: -kv[1]))
            print(f"  claim {cid} marginals: {dist}  (MAP={s.map_by_claim[cid]})")

    if args.moe:
        from .kbcir import (PolicyPortfolio, freeze, gate_replay_gate,
                            ledger_labels, train_gate)
        thetas = list(_THETAS.values())
        pf = PolicyPortfolio.default()
        samples, experts = ledger_labels(module, h, thetas, pf)
        gate = freeze(train_gate(samples, experts, epochs=300))
        cert = gate_replay_gate(module, h, thetas, gate, pf)
        print(f"[moe] gate routes among {len(experts)} experts "
              f"(fingerprint={gate.fingerprint}); replay cert: "
              f"{cert.regressions}/{cert.episodes} regressions, "
              f"admitted={cert.admitted}")
        for name, th in _THETAS.items():
            learned = gate.route_policy(module, th).name
            classic = pf.select(th).name
            flag = "" if learned == classic else "  <- diverges"
            print(f"  theta={name:<9} learned={learned:<11} classify={classic}{flag}")

    if args.egraph:
        from .kbcir import shared_blocks
        nclaims = sum(len(ph.claims) for ph in module.phases)
        print(f"[egraph] {nclaims} claim(s) compose to {shared_blocks(module)} "
              f"building block(s) after CSE (shared liked pairs)")

    if args.manifest:
        from .kbcir import build_manifest, reproduces
        man = build_manifest(module, h, theta, policy)
        print(f"[manifest] digest={man.digest} score={man.score} "
              f"widths={dict(man.widths)} reproduces={reproduces(man, module, h, theta, policy)}")

    if args.emit_c or args.run_c:
        from .lower import compile_and_run_c, emit_kernel_c
        try:
            if args.emit_c:
                print("\n/* ---- portable C23 kernel ---- */")
                print(emit_kernel_c(module, result, fn_name="bcir_kernel"))
            if args.run_c:
                ok, out = compile_and_run_c(module, result, fn_name="bcir_kernel")
                print(f"[run-c] C23 build+run {'OK' if ok else 'FAILED'}: {out.strip()}")
                if not ok:
                    return 1
        except NotImplementedError as exc:
            print(f"[c] {exc}")

    if args.calibrate:
        from .kbcir import measure_and_close
        from .kbcir.calibrate import EwmaCalibrator
        from .telemetry import DataDNA
        # fold the chosen Theta in as telemetry from the live machine (vs the
        # shipped cool default), so a hot machine shows the replan win.
        events = ([DataDNA(segment_id="cli", claim_id=0, thermal=theta.thermal,
                           utilization=theta.utilization, voltage=theta.voltage,
                           misses=theta.mem_pressure)]
                  if (theta.thermal or theta.voltage or theta.mem_pressure) else ())
        cert, table = measure_and_close(module, h, events=events, native=args.native,
                                        calibrator=EwmaCalibrator(alpha=200))
        print(f"[calibrate{'/native' if args.native else ''}] {table.provenance}")
        print(f"  cal_gen={table.cal_gen} gather_penalty={table.gather_penalty} "
              f"base_overhead={table.base_overhead} measured_thermal={cert.measured_thermal}")
        print(f"  seeded={dict(cert.seeded_widths)} -> calibrated={dict(cert.calibrated_widths)} "
              f"replanned={cert.replanned} win={cert.win} admissible={cert.admissible}")

    if args.bench:
        from .bench import bench_available, compare
        if not bench_available():
            print("[bench] no C compiler; skipping")
        else:
            for opt in ("-O1", "-O2", "-O3"):
                c = compare(args.program, target=args.target, theta=args.theta,
                            policy=args.policy, opt=opt)
                print(f"[bench] {opt}: bcir(w{c.bcir.width})={c.bcir.ns_per_call}ns "
                      f"scalar={c.baseline.ns_per_call}ns "
                      f"speedup={c.speedup_milli / 1000:.2f}x bcir_wins={c.bcir_wins}")

    if args.bench_gather:
        from .bench import bench_available, compare_gather
        if not bench_available():
            print("[bench-gather] no C compiler; skipping")
        else:
            try:
                for opt in ("-O2", "-O3"):
                    c = compare_gather(args.program, target=args.target, theta=args.theta,
                                       policy=args.policy, opt=opt, shuffle=True)
                    print(f"[bench-gather] {opt}: direct(w{c.direct.width})="
                          f"{c.direct.ns_per_call}ns gather={c.gather.ns_per_call}ns "
                          f"avoided_penalty={c.speedup_milli / 1000:.2f}x wins={c.avoidance_wins}")
            except NotImplementedError as exc:
                print(f"[bench-gather] {exc}")

    if args.bench_reduce:
        from .bench import bench_available, compare_reduce
        if not bench_available():
            print("[bench-reduce] no C compiler; skipping")
        else:
            try:
                for opt in ("-O2", "-O3"):
                    c = compare_reduce(args.program if args.program == "gather_reduce"
                                       else "gather_reduce", target=args.target,
                                       theta=args.theta, policy=args.policy, opt=opt)
                    print(f"[bench-reduce] {opt}: selected={c.selected} correct={c.correct} "
                          f"blocked={c.blocked.ns_per_call}ns gather={c.gather.ns_per_call}ns "
                          f"penalty={c.speedup_milli / 1000:.2f}x wins={c.avoidance_wins}")
            except NotImplementedError as exc:
                print(f"[bench-reduce] {exc}")

    if args.bench_strided:
        from .bench import bench_available, compare_strided
        if not bench_available():
            print("[bench-strided] no C compiler; skipping")
        else:
            try:
                for opt in ("-O2", "-O3"):
                    c = compare_strided(args.program if args.program == "saxpy_strided"
                                        else "saxpy_strided", target=args.target,
                                        theta=args.theta, policy=args.policy, opt=opt)
                    print(f"[bench-strided] {opt}: selected={c.selected} correct={c.correct} "
                          f"strided={c.blocked.ns_per_call}ns gather={c.gather.ns_per_call}ns "
                          f"penalty={c.speedup_milli / 1000:.2f}x wins={c.avoidance_wins}")
            except NotImplementedError as exc:
                print(f"[bench-strided] {exc}")

    if args.accel:
        from .kbcir import greedy_order, optimize_ordered, worst_order
        gr, gs = optimize_ordered(module, h, theta, policy, order=greedy_order)
        wr, ws = optimize_ordered(module, h, theta, policy, order=worst_order)
        print(f"[accel] same optimum: greedy={gr.score} worst={wr.score} "
              f"(exact); greedy expansions={gs.expansions} "
              f"(first plan {gs.first_complete_cost}) vs "
              f"worst expansions={ws.expansions} (first plan {ws.first_complete_cost})")

    if args.regret:
        from .kbcir import PolicyPortfolio, boundary_report, ledger_from_episodes
        ledger = ledger_from_episodes(module, h, list(_THETAS.values()),
                                      PolicyPortfolio.default())
        print(ledger.dashboard())
        for v in boundary_report(ledger):
            print(f"[boundary] {v.rule}: {v.verdict} "
                  f"(rate={v.regret_rate}, episodes={v.episodes})")

    if args.eft or args.tokens:
        from .gem import durations_from, execute_tokens, schedule_eft
        d = durations_from(result)
        for flag, fn in (("eft", schedule_eft), ("tokens", execute_tokens)):
            if getattr(args, flag):
                s = fn(module, d, h)
                print(f"[{flag}] makespan={s.makespan} knee={s.knee} "
                      f"affinity={s.affinity}")

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
