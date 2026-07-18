# `bcir/` — the executable conformance oracle

A dependency-free, deterministic Python realization of BCIR + K_BCIR + GEM. It is
the runnable reference for the MLIR law under [`../mlir/`](../mlir) (see
[`../docs/PARITY.md`](../docs/PARITY.md)). The dependency-free quick tier needs only
Python; compiler, LLVM/MLIR, model, and architecture paths are capability-gated and do
not become supported merely because they skip on a host.

## Layout

```
model/      BCIR-0..2 semantic model: lanes, opcodes, resources, claims, phases
kbcir/      K_BCIR (BCIR-3): cost vectors, target profiles + memory hierarchy,
            min-plus semiring, Theta-driven weights, the optimizer, the CT4 calibrator,
            the e-graph engine + memory-module fixpoint (a = Lim(Res(U))), provenance,
            the two-truth quarantine (MOPC) + modular mapping functions (support/parity),
            the enriched-operad memory interface (labels + content-addressed indexes + trace)
gem/        GEM (BCIR-4): StreamPack hydration, deterministic phase executor, and
            CT2 concurrent wave scheduling + GGG decoupling + affinity
etl/        M5 Event Transduction: events, FSM transducer, parser, binary decoder
frontends/  CT3 ROP/MAP/C front-ends plus model manifest/tokenizer/decode and
            payload-free model assessment/plan lowering
lower/      BCIR-5: single-claim elementwise LLVM AOT/JIT subset / portable C23 kernels
abi/        versioned StreamPack and artifact byte contracts
telemetry.py CT4 "data DNA" schema + local sinks (null/list/file; no live remote transport)
verify/     runnable LangRef verifier laws R1-R23
silicon.py  real-signal probes: PMU + RAPL energy + on-die thermal + cpufreq (honest)
kbcir/fuzz.py  fuzz the trust boundaries (StreamPack/ROP/MAP/ETL/JSON/MLIR), gen-seeded
api.py      the embeddable library facade: plan -> KernelArtifact (C + ABI header + R12 attestation)
bench.py    the measured-evidence rail: time the selected realization vs the baseline
examples.py the goal-graph corpus (vector_add, saxpy_strided, histogram_gather,
            gather_reduce, fused_chain, scan_chain, tiled_matmul) + the widened
            CORPUS: matmul_tiled (real blocked matmul), scan, multi_histogram
run.py      the CLI (--target/--theta/--policy/--run/--emit-c/--emit-mlir/--calibrate/--bench)
kbcir/differential.py  generated, adversarial Python<->MLIR parity (gen_module +
            independent law_select + check_module + shrink + run_campaign)
lower/mlir.py  to_mlir: emit GEM-pipeline BCIR-MLIR from any oracle plan (the bridge)
tests/      dependency-free runner; generated static inventory in ../docs/STATUS.md
```

## Run it

```bash
# K_BCIR plan for a program on a target under a runtime state Theta:
python -m bcir.run vector_add --target x86_avx512 --theta cool
python -m bcir.run vector_add --target nvidia_ptx           # GPU warp -> vec32
python -m bcir.run vector_add --target x86_avx512 --theta hot  # replans vec16 -> vec8

# Lower + compile + run the supported single-claim elementwise kernel via clang:
python -m bcir.run vector_add --target x86_avx512 --run

# Emit the GEM-pipeline MLIR for the plan (the law rail recomputes the score):
python -m bcir.run multi_histogram --target nvidia_ptx --emit-mlir

# Generated, adversarial Python<->MLIR parity (a proof, not curated pins):
python -m bcir.kbcir.differential -n 5000        # campaign across the six targets + verifier diff
python -m bcir.kbcir.differential --emit-corpus  # (re)freeze mlir/test/passes/gem_corpus.mlir
python -m bcir.kbcir.fuzz -n 4000                # fuzz the trust boundaries (gen-seeded)

# Close the calibration loop on real silicon (PMU + RAPL + thermal; honest in a sandbox):
python -m bcir.run vector_add --silicon

# Tests (no pytest required; also works under `python -m pytest bcir/tests`):
python -m bcir.tests.run_all --tier quick -j 2
```

## The central equation, made runnable

```
K_BCIR(G | H, Theta) = min_pi  sum_i  T_i (X) f_i(pi)  =  min_pi C_H(pi, Theta)
```

`optimize()` builds a layered realization DAG (one column of legal candidate
lowerings per claim), couples each base cost `T_i` by the path-context factor
`f_i(pi)`, scalarizes under `w(H, Theta, phase, policy)`, and runs a tropical
(min,+) shortest path to select `pi*`. Everything is integer/Q-fixed for
determinism — the worked example (`vector_add`, AVX-512, cool Theta) scores
exactly **7808** and selects `vec16`, matching `mlir/examples/full_vec_add_ct1.mlir`.
