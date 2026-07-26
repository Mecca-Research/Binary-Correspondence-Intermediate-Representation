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
│   ├── frontends/       ROP/MAP/C front-ends + model ingest/header-only assessment
│   ├── hosted/          opt-in model training/alignment + hardware-policy references
│   ├── lower/           BCIR-5: single-claim elementwise LLVM AOT/JIT subset + broader C lowering
│   ├── telemetry.py     "data DNA" schema + local sinks (null/list/file)
│   └── verify/          runnable reference of verifier laws R1–R23
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
python -m bcir.tests.run_all --tier quick -j 2                       # bounded local tier; static inventory in docs/STATUS.md
python -m bcir.performance_audit --repeats 3                         # strict result digests + informative timings

# Pinned real-model gate (downloads three checksum-pinned files; --offline uses the cache):
python tools/models/run_real_model_gate.py --output-dir build/model-gate

# Payload-free model capacity/placement report + verified execution StreamPack:
bcir-model-assess MODEL_DIR --hardware hardware.json --workload workload.json \
  --report-out build/model-plan/report.json --plan-out build/model-plan/plan.json \
  --pack-out build/model-plan/plan.bspk

# Optional pinned-CPU micro gate: telemetry/topology policy → PUCT → verified plan/addresses.
# Requires the model-lab extra; uses generated simulated episodes and performs no GPU/large training.
python tools/models/run_hardware_rl_gate.py --output-dir build/hardware-rl-gate

# Optional one-thread sequence-interface/adaptation/growth confirmation; generated fixtures only.
python tools/models/test_sequence_interfaces.py --output-dir build/sequence-interface-gate

# The compiled MLIR dialect (needs libmlir-NN-dev + llvm-NN-dev):
bash tools/wsl/build_mlir.sh            # build bcir-opt (LangRef M3)
bash tools/wsl/check_ods_examples.sh    # parse/verify the pretty ODS corpus via bcir-opt
bash tools/irdl/check_corpus.sh         # round-trip the IRDL projection on stock mlir-opt
```

## Where the law lives

- [`docs/BCIR_LANGREF.md`](docs/BCIR_LANGREF.md) — the normative language reference (levels, laws R1–R23, the equation, and BCIRQ8 v1 artifact ABI).
- [`docs/BCIR_MASTER_ROADMAP.md`](docs/BCIR_MASTER_ROADMAP.md) — cross-program dependency order, promotion gates, stop conditions, and release policy; history and counts live elsewhere.
- [`docs/kernel/BCIR_DRIVER_KERNEL_ROADMAP.md`](docs/kernel/BCIR_DRIVER_KERNEL_ROADMAP.md) — the canonical proof-carrying driver, BCIR-Linux, UAPI, native-kernel, telemetry, and IPC sequence.
- [`docs/kernel/BCIR_HAM_MEMORY_FABRIC.md`](docs/kernel/BCIR_HAM_MEMORY_FABRIC.md) — the verified semantic-memory/context-shard control plane and the explicit GDS/P2PDMA/CXL/NVMe firmware/kernel boundary.
- [`docs/machine-learning/BCIR_ML_AI_INTEGRATION_ROADMAP.md`](docs/machine-learning/BCIR_ML_AI_INTEGRATION_ROADMAP.md) — the detailed ML/model closure program, from low-bit/scheduling/AD gaps through production serving and data organs.
- [`docs/BCIR_ASN1_X690_ABI.md`](docs/BCIR_ASN1_X690_ABI.md) — ASN.1 / X.690 binary format compatibility: the DER-out/BER-in stance, the `BCIR-StreamPack` ASN.1 module, and the additive DER projection of the frozen StreamPack ABI.
- [`docs/PARITY.md`](docs/PARITY.md) — the Python↔MLIR law contract and Python↔C artifact/runtime parity ledger.
- [`docs/STATUS.md`](docs/STATUS.md) — the generated static inventory of tests, ODS ops, passes, runtime files, and verifier-law fixture tags; it does not claim those gates were executed.
- [`docs/PERFORMANCE_AUDIT.md`](docs/PERFORMANCE_AUDIT.md) — the bounded TMSAO methodology, fixed bottlenecks, local evidence, and hardware-gated limits.
- [`docs/languages/C_MEMORY_DISCIPLINE.md`](docs/languages/C_MEMORY_DISCIPLINE.md) — the enforced freestanding/hosted/driver memory classes, allocator and context contracts, direct driver ABI, and driver-gated IPC sequence.
- [`docs/machine-learning/THIRD_PARTY_MODELS.md`](docs/machine-learning/THIRD_PARTY_MODELS.md) — pinned model provenance, license, and non-redistribution boundary.
- [`docs/BCIR_Repo_Structure.md`](docs/BCIR_Repo_Structure.md) — current code and documentation ownership, contract placement, and validation entry points.
- [`docs/DEVELOPMENT_HISTORY.md`](docs/DEVELOPMENT_HISTORY.md) — how it was built: the development method, the PR-era arc, and the condensed changelog.
- [`docs/languages/CFRONT_GUIDE.md`](docs/languages/CFRONT_GUIDE.md) — the `bcir-cfront` C-frontend user guide: the CLI, diagnostics, the target ABI matrix, the fallback contract, and the supported subset + limits. The frontend lowers a wide C surface **dual-rail** (oracle + the `bcir_cfront.c` twin) — including the full array-compound-literal surface (1-D + multi-dim, scalar + aggregate-element), computed goto (`goto *p` / `&&L`), and function-pointer local variables — all Clang-equivalence + fuzzer gated (`_Decimal32/64/128` is **blocked**: Clang 18 cannot compile it).

The Python package is the **executable conformance oracle**; the MLIR dialect is
the **law** it must agree with. LLVM/Clang are backends, not the conceptual center.

## License

This project is licensed under the **BCIR Non-Commercial License, Version 1.0**
(`LicenseRef-BCIR-NC-1.0`) — see [`LICENSE`](LICENSE) for the full terms.

The code is open and free for **open-source use, development, modification,
free redistribution, and private use**. **Commercial use, commercial
distribution, and patent use are not permitted** without explicit prior
written permission from Mecca-Research; contact the maintainers through this
repository to request a commercial license. No trademark rights are granted,
and the software is provided without warranty or liability.
