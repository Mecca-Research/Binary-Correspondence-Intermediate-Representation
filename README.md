# Binary-Correspondence-Intermediate-Representation

An IR-first realization of **Kolmogorov Binary Correspondence Intermediate Representation optimization (K_BCIR) + Graph Execution Model (GEM)**: a registry-first, phase-ordered,
lane-typed, **cost-governed** correspondence IR that selects legal physical
realization paths via the tropical (min,+) optimizer

```
K_BCIR(G | H, Θ) = min_{π ∈ Legal(G,H)}  M(π, Θ)
                   subject to  R(π, Θ) ⪯ B(H, Θ)

M(π,Θ) = makespan of π under the wave/token DAG     — (max,+) over parallel, (min,+)⊗ over series
R(π,Θ) = additive resource vector (energy, traffic, …) — Σ as today
B(H,Θ) = live budgets (thermal cap, power cap, bandwidth)
```

> **Two separate things live in this repo.** `bcir/` + `mlir/` **are** the BCIR
> IR. `llvm-training/` is an LLVM/MLIR training corpus for agents and is **not**
> part of the IR. See [`AGENTS.md`](AGENTS.md) and
> [`docs/BCIR_Repo_Structure.md`](docs/BCIR_Repo_Structure.md).

## Top-level layout

```
.
├── bcir/                the executable conformance oracle (Python, runnable today)
│   ├── model/           BCIR-0..2 semantic model (lanes, opcodes, resources, claims, phases)
│   ├── kbcir/           K_BCIR (BCIR-3): cost algebra, target profiles, min-plus optimizer, ML calibrator
│   ├── gem/             GEM (BCIR-4): StreamPack hydration, deterministic + concurrent execution
│   ├── etl/             M5 event transduction (events, FSM, parser, binary decoder)
│   ├── frontends/       ROP (declarative) + MAP (macro-assembly) front-ends
│   ├── lower/           BCIR-5: legal LLVM IR run AOT (clang) or JIT (lli)
│   ├── telemetry.py     "data DNA" schema + sinks (null/list/file/Kafka)
│   └── verify/          runnable subset of verifier laws R1–R12
├── mlir/                the IR law: TableGen/ODS dialect family + compiled bcir-opt + IRDL projection
│   ├── include/BCIR/    *.td (enums/types/attrs/ops) + *.h
│   ├── lib/, tools/     BCIRDialect.cpp + bcir-opt.cpp (the compiled dialect)
│   ├── irdl/            pure-data IRDL projection for stock mlir-opt (portability rail)
│   └── examples/, test/ pretty ODS corpus (bcir-opt) + generic IRDL corpus (mlir-opt)
├── tools/               validation scripts (tblgen / IRDL round-trip / build + check bcir-opt)
├── docs/                LangRef, Blueprint, PARITY, and the repo-structure decision
└── llvm-training/       SEPARATE agent-readable LLVM/MLIR curriculum (not the IR)
```

## Quickstart

```bash
# The runnable oracle (no third-party deps): plan, schedule, lower, run.
python -m bcir.run vector_add --target x86_avx512 --theta cool      # vec16, score 7808
python -m bcir.run vector_add --target nvidia_ptx                    # GPU warp -> vec32
python -m bcir.run vector_add --target x86_avx512 --schedule --jit   # CT2 schedule + CT5 JIT (lli)
python -m bcir.run vector_add --target x86_avx512 --wasm             # compile to WASM + run via node
python -m bcir.run vector_add --budget thermal=700 --overlap         # RCSP rail: vec8 @ 9472 + M(pi,Theta)
python -m bcir.run vector_add --soft-temp 3000                       # temperature dial: soft plan distribution (T=0 == tropical)
python -m bcir.run vector_add --moe                                  # learned MoE gate (GNN) route vs classify, replay-gated
python -m bcir.run vector_add --accel                                # propose-verify search accelerator (same optimum, fewer expansions)
python -m bcir.run vector_add --manifest                             # provenance manifest (the commit hash of the plan) + replay check
python -m bcir.run vector_add --egraph                               # building-blocks engine: claims -> shared blocks after CSE
python -m bcir.run multi_histogram --target nvidia_ptx --emit-mlir   # emit the GEM-pipeline MLIR for the plan (the law rail)
python -m bcir.kbcir.differential -n 5000                            # generated, adversarial Python<->MLIR parity (proof, not pins)
python -m bcir.kbcir.microbench --target x86_avx512 --out cal.json   # measure -> freeze a Q8 cost table
python -m bcir.kbcir.microbench --target x86_avx512 --bayes          # Bayesian posterior + conformal +/- delta
python -m bcir.run vector_add --tables bcir/kbcir/tables/x86_64_reference.json  # apply frozen table
python -m bcir.tests.run_all                                         # the conformance test suite

# The compiled MLIR dialect (needs libmlir-NN-dev + llvm-NN-dev):
bash tools/wsl/build_mlir.sh            # build bcir-opt (LangRef M3)
bash tools/wsl/check_ods_examples.sh    # parse/verify the pretty ODS corpus via bcir-opt
bash tools/irdl/check_corpus.sh         # round-trip the IRDL projection on stock mlir-opt
```

## Where the law lives

- [`docs/BCIR_LANGREF.md`](docs/BCIR_LANGREF.md) — the normative language reference (levels, laws R1–R12, the equation).
- [`docs/BCIR_BLUEPRINT.md`](docs/BCIR_BLUEPRINT.md) — the target-open / heterogeneous build program (CT1–CT5).
- [`docs/PARITY.md`](docs/PARITY.md) — the Python (`bcir/`) ↔ MLIR (`mlir/`) lockstep contract.
- [`docs/BCIR_Repo_Structure.md`](docs/BCIR_Repo_Structure.md) — how the repo is organized and why.

The Python package is the **executable conformance oracle**; the MLIR dialect is
the **law** it must agree with. LLVM/Clang are backends, not the conceptual center.
