# OpenAI + BCIR Integration Research and Proposal Versions

This document records a first-pass integration study for connecting OpenAI developer products to the Binary Correspondence Intermediate Representation (BCIR) as both an LLVM/MLIR agent-training repository and a compiler/ML architecture. It deliberately separates the repository's two roles: `llvm-training/` is a training corpus for agents, while `bcir/` and `mlir/` are the IR oracle/law system.

## 1. Repository capability map

### 1.1 Agent training corpus: `llvm-training/`

`llvm-training/` is best treated as a curated curriculum, not as part of the BCIR runtime. Its value for agents is that it teaches LLVM/MLIR mental models, verifier discipline, target lowering, ORC/JIT runtime patterns, operand bundles, attributes, poison/undef/freeze, and backend concepts in small checked examples. This makes it useful as a retrieval corpus for code-review agents, onboarding agents, and automated lesson-generation workflows, but it must not become a dependency of IR semantics.

Recommended agent uses:

- **Retrieval and tutoring:** expose chapters and checked examples through an MCP resource server or a vector/file-search index.
- **Evaluation:** turn chapter exercises and verify commands into regression evals for LLVM-literate agents.
- **Repair loops:** let an agent propose edits to examples, then gate them with the existing verify commands.
- **Boundary enforcement:** every agent prompt should state that changes to `llvm-training/` are independent from the canonical IR unless a user explicitly asks to curate the training corpus.

### 1.2 BCIR executable oracle: `bcir/`

The Python rail is the runnable conformance oracle. It is dependency-free by design and implements the practical BCIR stack today: semantic goal graphs, K_BCIR optimization, GEM execution, event transduction, ROP/MAP/C front ends, telemetry, silicon probes, C/LLVM/JIT lowering, verifier laws, fuzzing, and ML-guided organs. The existing `bcir/README.md` already frames the core equation as deterministic min-plus selection over legal candidate realizations.

OpenAI integration should therefore treat `bcir/` as the **tool-execution and experiment rail**, not as an opaque document corpus. The best integrations are tool calls that run deterministic BCIR commands, collect artifacts, and summarize results back to users.

Candidate tool surface:

- `plan_program(program, target, theta, policy)` → runs K_BCIR planning and returns chosen realization, score, cost vector, and provenance.
- `emit_mlir(program, target, theta)` → emits GEM-pipeline MLIR for law-rail validation.
- `run_kernel(program, target)` → compiles/runs selected kernels where local tools are available.
- `run_oracle_tests(selector)` → runs `python -m bcir.tests.run_all` or selected tests.
- `run_differential_campaign(n, seed)` → invokes Python↔MLIR differential parity when `bcir-opt` is present.
- `summarize_telemetry(run_id)` → reads telemetry frames and converts them into plan/replan explanations.

### 1.3 MLIR/C++ law rail: `mlir/`

The MLIR tree is the compiled dialect law: TableGen/ODS ops, custom verifiers, conversion passes, compiled `bcir-opt`, and IRDL projection. It should be exposed to agents as the **legality and parity rail**. Agents may propose changes to the oracle, but any semantic change needs a law-rail story: ODS op/type/attr changes, verifier changes, pass updates, and parity tests.

Candidate tool surface:

- `validate_ods_examples()` → runs pretty corpus through `bcir-opt`.
- `validate_irdl_corpus()` → checks the IRDL projection on stock `mlir-opt`.
- `explain_bcir_opt(input_mlir, passes)` → runs selected passes and returns normalized IR plus diagnostics.
- `law_diff(oracle_artifact, mlir_artifact)` → compares oracle decisions against law-rail recomputation.

### 1.4 Compiler/ML architecture layers

BCIR already has several AI-relevant layers:

1. **Classical legality:** R-laws and verifier passes define binary validity; no learned model should be able to override this.
2. **Deterministic optimization:** K_BCIR uses integer/Q-fixed cost selection, target profiles, resource ledgers, and min-plus/RCSP selection.
3. **GEM execution:** StreamPack hydration, deterministic wave scheduling, overlap/concurrency, DVFS/CIM, and async token modeling.
4. **Telemetry and calibration:** telemetry frames, silicon probes, microbenchmarks, regret, bayesian calibration, throttle/replan gates.
5. **ML-guided organs:** e-graphs, memory/operad, MoE gates, soft dynamic programming, portfolio thresholds, quantization, losses, autodiff, transformer/recurrent/classical modules.
6. **Lowering and deployment:** C23 kernels, LLVM AOT/JIT, Wasm, SYCL dispatch, and specialist lowerings.

The integration opportunity is not to make ChatGPT part of the hot path. It is to make OpenAI models into **L2/L3 meta-agents** that propose experiments, write patches, inspect traces, summarize telemetry, synthesize evals, and recommend frozen Q8 policy updates that BCIR's own gates must accept before promotion.

## 2. Current OpenAI developer capability research

This section uses current official OpenAI developer documentation as of July 1, 2026.

### 2.1 Responses API

The Responses API is the right base layer when BCIR needs one model call plus application-owned tool logic. It supports tool-using workflows and can be paired with file search, MCP, code execution/sandboxing patterns, and structured outputs. For BCIR, Responses should be the simple integration path for CLI assistants, repository Q&A, single-run experiment explanation, and deterministic wrapper tools.

### 2.2 Agents SDK

The Agents SDK is appropriate when BCIR owns orchestration, approvals, state, and multi-agent decomposition. OpenAI's guide describes agents as applications that plan, call tools, collaborate across specialists, and keep enough state for multi-step work; it recommends the Responses API for simpler one-call-plus-tools logic and the Agents SDK when the application owns orchestration and state. The SDK runner loops over model calls, tool calls, handoffs, and final answers, and supports handoffs, agents-as-tools, sessions/conversation strategies, streaming, guardrails, and tracing.

BCIR fit:

- **Manager agent:** owns a user's high-level goal, delegates to specialized agents.
- **Oracle agent:** calls `bcir/` tools and explains deterministic plan choices.
- **Law agent:** calls `bcir-opt`, ODS/IRDL validation, and parity scripts.
- **Training-corpus agent:** retrieves `llvm-training/` lessons and generates evals.
- **Telemetry agent:** summarizes run traces, regret ledgers, and replan opportunities.
- **Patch agent:** writes code/docs changes, then runs gates.

The official Agents SDK observability docs state that tracing can capture model calls, tool calls, handoffs, guardrails, and custom spans. BCIR can map those traces into its own telemetry vocabulary, but OpenAI traces must remain external evidence, not verifier truth.

Sources: OpenAI Agents SDK guide (<https://developers.openai.com/api/docs/guides/agents>), running agents guide (<https://developers.openai.com/api/docs/guides/agents/running-agents>), orchestration guide (<https://developers.openai.com/api/docs/guides/agents/orchestration>), and observability guide (<https://developers.openai.com/api/docs/guides/agents/integrations-observability>).

### 2.3 Apps SDK and ChatGPT apps

The Apps SDK is the path for putting a BCIR interface inside ChatGPT. Official docs describe an Apps SDK app as an MCP server plus optional web-component UI. MCP is the backbone: the server advertises tools with JSON Schema contracts, ChatGPT calls those tools, and structured results can render inline UI components. UI components run inside an iframe and communicate with the host through the MCP Apps bridge.

BCIR fit:

- **BCIR Workbench app:** a ChatGPT app for selecting a corpus program, target, theta, and policy; running planning/lowering tools; visualizing cost vectors, selected paths, verifier laws, and telemetry.
- **Law/Oracle parity dashboard:** a component showing oracle output beside MLIR law recomputation and failing diagnostics.
- **LLVM training tutor:** a component for lessons, examples, exercises, and verify status.

Recommended design rule: expose small, auditable tools with structured outputs. Do not expose arbitrary shell by default. High-risk actions such as running local compilers, touching hardware probes, writing files, or opening network endpoints should require explicit user approval in the hosting application.

Sources: Apps SDK overview (<https://developers.openai.com/apps-sdk>), MCP server concept (<https://developers.openai.com/apps-sdk/concepts/mcp-server>), build MCP server guide (<https://developers.openai.com/apps-sdk/build/mcp-server>), ChatGPT UI guide (<https://developers.openai.com/apps-sdk/build/chatgpt-ui>), MCP Apps compatibility guide (<https://developers.openai.com/apps-sdk/mcp-apps-in-chatgpt>), and tool-design guide (<https://developers.openai.com/apps-sdk/plan/tools>).

### 2.4 Codex and agent improvement loops

OpenAI's cookbook material includes an agent improvement loop using traces, evals, feedback, and Codex-ready handoffs. This maps closely to BCIR's own telemetry/regret/calibration loop: traces and feedback can create evals; eval failures become patch handoffs; patches run deterministic repo gates; accepted changes produce a new frozen generation/provenance artifact.

BCIR should keep the loops separate:

- **OpenAI loop:** improves prompts, tool schemas, app UX, routing, and patch suggestions.
- **BCIR loop:** accepts only deterministic artifacts that pass parity, verifier, performance, and provenance gates.

Sources: OpenAI agent improvement loop cookbook (<https://developers.openai.com/cookbook/examples/agents_sdk/agent_improvement_loop>) and Codex with Agents SDK guide (<https://developers.openai.com/codex/guides/agents-sdk>).

## 3. How deep ChatGPT can integrate into BCIR

### 3.1 Integration depth that is realistic now

1. **Documentation and code intelligence:** ChatGPT can read, retrieve, explain, and cross-reference BCIR docs/code through file search or MCP resources.
2. **Tool orchestration:** ChatGPT can call BCIR tools through Responses/Agents/MCP: plan, lower, validate, benchmark, fuzz, summarize.
3. **Agentic workflows:** ChatGPT can coordinate oracle/law/training/telemetry specialists and maintain run state through Agents SDK sessions or application storage.
4. **Interactive UI:** ChatGPT apps can host a BCIR workbench UI for cost vectors, target matrices, MLIR diffs, telemetry plots, and training exercises.
5. **Patch generation:** Codex-style agents can propose code/docs/tests and run deterministic gates.
6. **Eval generation:** OpenAI models can generate synthetic tasks, expected-behavior rubrics, and prompt/tool eval suites.
7. **Meta-learning loops:** OpenAI models can inspect BCIR traces and recommend candidate policy changes, features, or experiments for BCIR to validate.

### 3.2 Integration depth that must remain gated

- **Verifier legality:** ChatGPT cannot decide BCIR legality. It can only propose evidence or patches for the classical verifier/law to check.
- **Hot-path optimization:** ChatGPT calls are nondeterministic, remote, high-latency, and not reproducible enough for L0/L1 hot paths. Any learned signal must be frozen into deterministic tables or code before promotion.
- **Autonomous hardware probing:** PMU/RAPL/thermal/cpufreq and device dispatch should be allowlisted, logged, and user-controlled.
- **Training data provenance:** GPT-generated data must be tagged as synthetic/model-generated and should not silently mix with measured silicon data or authoritative LLVM examples.
- **Security:** MCP tools must validate schemas, cap resource use, sandbox untrusted inputs, and avoid arbitrary shell/file access unless the environment is explicitly trusted.

### 3.3 What can and cannot change about ChatGPT itself

OpenAI API integration can change **application behavior** around the model: system/developer instructions, tool availability, retrieval context, structured output schemas, agent routing, memory/session state, UI, eval-driven prompt revisions, and fine-tuned/distilled companion models where product support permits.

It cannot directly change the hosted ChatGPT model's base weights. The base model remains frozen from the integrator's perspective. BCIR can still build new AI systems around GPT outputs by using GPT as:

- a data generator with provenance tags,
- a teacher/critic for synthetic labels,
- an experiment planner,
- a code-editing assistant,
- a trace summarizer,
- a meta-policy proposer,
- a benchmark/eval author.

The safe bootstrapping pattern is **GPT proposes, BCIR disposes**: GPT produces candidate code, datasets, hypotheses, tool traces, or policies; BCIR's deterministic verifier, parity gates, tests, and measured telemetry decide whether anything enters a frozen generation.

### 3.4 BCIR as an instruction-processing exoskeleton

BCIR cannot rewrite GPT's internal transformer, tokenizer, attention weights, or native instruction hierarchy. What BCIR *can* change is the **effective instruction-processing system around GPT**: the instructions GPT sees, the tools it can call, the retrieval substrate it reasons over, the schemas it must emit, the agent graph that routes subtasks, and the memory/session state that is re-injected into future calls. That is a bootstrap change in capability even when the hosted GPT weights are frozen.

The deepest useful pattern is a **BCIR instruction compiler**:

1. **Parse user intent into a typed goal graph.** Convert natural-language work requests into a BCIR-style task graph with resources, hazards, expected artifacts, verification gates, and provenance slots.
2. **Optimize the instruction plan.** Use K_BCIR-like selection to choose retrieval packs, tool availability, model/agent routing, context budget, schema strictness, and evaluation gates under latency/cost/risk budgets.
3. **Hydrate execution context.** Assemble a StreamPack-like prompt bundle: immutable system/developer rules, task-local evidence, retrieved code/docs, previous run telemetry, allowed tools, and structured output contracts.
4. **Execute with typed tools.** Let GPT call only the selected MCP/Responses/Agents tools, with BCIR recording tool traces, outputs, failures, environment, and hashes.
5. **Verify and replay.** Treat model output as a candidate artifact. Run BCIR gates, tests, law/oracle parity, schema validation, and replay checks before anything is trusted.
6. **Freeze learned improvements.** Convert repeated successful patterns into deterministic prompts, tool schemas, routing policies, retrieval indexes, evals, or BCIR-native model weights/tables.

This does not make GPT itself more powerful in the weights sense. It makes the **closed-loop system** more powerful: GPT spends fewer tokens rediscovering repo facts, gets better tools at the right time, emits machine-checkable artifacts, and learns through external memory/evals rather than hidden weight updates.

### 3.5 Mapping BCIR infrastructure to GPT capability surfaces

| GPT capability surface | BCIR infrastructure that can enhance it | How far it can go | Hard boundary |
|---|---|---|---|
| Instruction processing | A task-graph compiler plus K_BCIR-style routing over instruction packs, examples, constraints, and gates | Strongly improves reliability, decomposition, and domain discipline for BCIR/LLVM tasks | Does not change GPT's internal policy, tokenizer, or weights |
| Tool availability | MCP/Responses tools wrapping `bcir/`, `bcir-opt`, telemetry, tests, retrieval, and patch workflows | Makes GPT operationally capable of planning, lowering, validating, benchmarking, and repairing | Tools remain external calls; they need auth, sandboxing, quotas, and user approval for risky actions |
| Retrieval context | `llvm-training/` lessons, `docs/`, ODS/IRDL, code slices, telemetry records, and provenance manifests | Gives GPT fresh, repo-specific, run-specific context beyond pretraining | Context is bounded and must be selected; retrieved text can still be misused without schema/eval gates |
| Structured outputs | BCIR result types, verifier diagnostics, plan manifests, JSON Schema, MLIR snippets, provenance records | Turns GPT output into parseable artifacts that BCIR can validate, replay, diff, and promote | Schema conformance is not semantic truth; BCIR must still verify legality and behavior |
| Agent routing | Agents SDK manager plus oracle/law/training/telemetry/patch specialists | Enables division of labor and persistent workflows across planning, execution, and repair | Routing is application state, not model cognition; bad routing can amplify errors |
| Memory/session state | BCIR telemetry, regret ledgers, eval outcomes, patch history, prompt/tool versions, and provenance DAGs | Gives GPT a durable external memory that improves future calls and supports audit/replay | Memory must be curated, scoped, and expired; it must not become an unverifiable authority |

OpenAI's model-optimization guidance explicitly treats prompt engineering, evals, and fine-tuning as a feedback flywheel, and notes that LLM output is non-deterministic and must be measured continuously. BCIR's advantage is that it already has the deterministic half of that flywheel: verifier laws, parity gates, replay, telemetry, regret, and provenance. OpenAI's Structured Outputs support is especially important here because BCIR can define the JSON/typed artifacts that GPT must produce before the oracle/law rails inspect them.

Sources: OpenAI model optimization guide (<https://developers.openai.com/api/docs/guides/model-optimization>), tool-use guide (<https://developers.openai.com/api/docs/guides/tools>), and Structured Outputs guide (<https://developers.openai.com/api/docs/guides/structured-outputs>).

### 3.6 Fine-tuning and distillation into BCIR-native models

There are three distinct targets; conflating them leads to overclaiming.

1. **Fine-tune an OpenAI-hosted model, where available.** This can improve format discipline, task style, and domain response patterns, but it is product-surface-limited and does not expose or merge weights into BCIR. Current OpenAI docs present model optimization as an eval/prompt/fine-tuning workflow and note that some related workflows are being moved into legacy documentation, so BCIR should keep hosted fine-tuning as an optional product-surface path rather than the keystone architecture.
2. **Distill GPT behavior into BCIR-owned models.** This is the deepest BCIR-controlled path. Use GPT as teacher/critic to generate traces, plans, labels, rationales, negative examples, and repair demonstrations; then train BCIR-native models in `bcir/kbcir/` and lower/freeze them through BCIR's own deterministic gates. Candidate students include routers, rankers, retrieval selectors, diagnostic classifiers, prompt-pack selectors, cost-model priors, and small code/action policies.
3. **Train independent BCIR neural networks and agents seeded by GPT.** This is broader than distillation: GPT supplies curricula, synthetic counterexamples, eval cases, and candidate architectures, but the final model is trained and validated as a BCIR artifact. The strongest students are not general GPT replacements; they are compiler/ML specialists that are small, auditable, quantizable, and promotable into L1/L2 only after replay and parity gates.

The practical ceiling for BCIR-owned distillation is high for **narrow compiler-agent skills** and low for **full general-language replacement**. BCIR can plausibly train compact models for retrieval ranking, pass selection, diagnostic repair classification, prompt/tool routing, workload generation, and cost-prior estimation. Recreating GPT-scale general instruction following inside BCIR would require enormous data/compute and would still lack the hosted model's broad pretraining unless BCIR imports or trains comparable foundation models.

A safe BCIR distillation pipeline is:

1. Collect GPT traces under explicit provenance: prompt pack, retrieved context, tool calls, outputs, evaluator decisions, and user feedback.
2. Convert traces into BCIR datasets: `(state, action, outcome, verifier result, regret, cost, provenance)` records.
3. Separate synthetic, human, measured, and verifier-derived labels.
4. Train candidate students in Python research tier (`L2/L3`) using existing training, losses, optimizers, quantization, recurrent/transformer/classical modules where appropriate.
5. Evaluate against held-out BCIR tasks, adversarial counterexamples, and law/oracle parity.
6. Freeze deployable pieces into Q8 tables, small C kernels, or deterministic routing policies.
7. Promote only behind R-law, two-truth, replay, and provenance gates.

### 3.7 Final answer to the boundary question

The integration is **not limited** to GPT seeding BCIR neural networks, models, and agents. It includes a much larger application-level capability layer: instruction compilation, context selection, tool orchestration, structured artifact generation, multi-agent routing, trace memory, eval loops, and UI integration.

But the integration **is limited** in one crucial way: BCIR cannot mutate the hosted GPT model's base weights or make GPT's hidden reasoning deterministic. Therefore BCIR should treat GPT as a powerful but external stochastic component. The deepest trustworthy architecture is:

```text
GPT as stochastic teacher / planner / tool user
        -> BCIR typed traces, datasets, evals, tool outputs
        -> BCIR deterministic verification + measured telemetry
        -> BCIR-owned frozen prompts, schemas, retrieval packs, routing policies, Q8 tables, and small specialist models
        -> optional future GPT calls with improved external instruction-processing exoskeleton
```

In short: BCIR can substantially change GPT's **effective capabilities** by changing everything around the call. It can distill repeated GPT successes into BCIR-owned models and deterministic policies. It cannot directly change GPT's base model; any trusted long-term capability must eventually become a BCIR artifact that is replayable, measurable, and gated.

### 3.8 GPT L2/L3 meta-agents as cloud teachers for BCIR model training

The next step beyond "GPT helps write code" is a **cloud-teacher training loop**: GPT L2/L3 meta-agents generate tasks, data, labels, critiques, curriculum steps, and repair traces through API calls; BCIR turns those calls into typed training episodes; BCIR trains its own neural networks, routers, optimizers, and endpoint models on the resulting corpus; deterministic gates decide what survives.

This is a real training architecture, but it is not magic gradient access to GPT. The API call is an **inference-time teacher/sample generator**, not a differentiable layer inside BCIR's optimizer. The learnable weights that change are the BCIR-owned student models, prompt/tool policies, retrieval rankers, Q8 tables, or hosted fine-tuned models where product support permits.

A concrete session ABI should look like this:

```text
TrainingSession(gen_id, objective, budget, policy, target_model)
  -> Episode(seed, prompt_pack, retrieved_context, tool_manifest, expected_schema)
  -> GPT teacher calls: propose / solve / critique / adversarial_mutate / grade
  -> BCIR validation: schema -> verifier -> oracle/law parity -> tests -> telemetry
  -> Dataset record: state, action, answer, critique, label, confidence, costs, failures, provenance
  -> Student update: train/eval/freeze candidate BCIR model or policy
  -> Promotion gate: held-out evals + replay + regret + provenance + two-truth quarantine
```

The generated dataset should be multi-channel rather than a flat text dump:

| Channel | What GPT meta-agents generate | BCIR consumer | Gate before use |
|---|---|---|---|
| Task synthesis | New LLVM/MLIR/BCIR exercises, malformed IR, hardware scenarios, edge cases | Training corpus evals, repair agents, verifier tests | Deduplicate, license/provenance tag, run verifier/toolchain where possible |
| Teacher solutions | Candidate MLIR, C kernels, plans, explanations, expected outputs | Student code/action models and retrieval answerers | Compile/run/compare; reject unverifiable answers |
| Critic labels | Failure diagnosis, risk tags, routing recommendations, fix hints | Diagnostic classifiers, agent routers, prompt-pack selectors | Agreement with tests, human review for high-impact labels |
| Adversarial mutations | Counterexamples, fuzz seeds, prompt-injection cases, malicious tool inputs | Red-team evals and guardrails | Sandbox; never promote directly to trusted corpus |
| Policy traces | Which tools, context packs, schemas, and specialists worked | K_BCIR-style routing/ranking policies | Replay on held-out traces and cost/regret gates |
| Numeric/model data | Toy supervised datasets, synthetic traces, curriculum schedules | `bcir.kbcir.training`, losses, optimizers, transformer/recurrent/classical modules | Separate synthetic from measured data; require eval improvement |

This is where BCIR's existing ML substrate matters. The repository already has deterministic datasets/mini-batches, losses, optimizers, supervised training loops, quantization, classical models, recurrent models, transformer blocks, autodiff kernels, and C lowerings. GPT meta-agents can fill the missing *experience stream*: they can manufacture and label workloads, propose curriculum orderings, create adversarial examples, and explain failures. BCIR then trains and freezes the students.

#### Incremental API-call training sessions

A new kind of incremental training loop is possible:

1. **Seed.** Start with a small BCIR model or routing policy and a baseline eval set.
2. **Generate.** Ask GPT meta-agents for batches of tasks, solutions, critiques, and adversarial variants under strict schemas.
3. **Validate.** Run BCIR tools, tests, law/oracle parity, and human review for high-impact categories.
4. **Train.** Update a BCIR student model on accepted records. For neural students this is ordinary gradient training; for policies it may be bandit/regret optimization; for prompt/tool packs it may be eval-driven selection.
5. **Evaluate.** Re-run held-out evals and macro-evals; measure cost, latency, correctness, tool use, and replay stability.
6. **Freeze.** If the candidate beats the prior under the promotion rule, freeze weights/tables/prompts/schema versions with a provenance digest.
7. **Bootstrap.** Use the improved student to filter, route, or critique the next generation of GPT calls, reducing cost and increasing sample quality.

The novel part is not that GPT is being trained incrementally. The novelty is **frontier-inference bootstrapping**: repeated GPT calls create a growing, validated experience buffer; BCIR students learn from that buffer; improved BCIR students make the next teacher-call campaign cheaper, more targeted, and more rigorous. That is a credible path to new BCIR-native AI capabilities.

#### How far this can go

High-confidence targets:

- BCIR-specific coding/repair agents trained on verified patches and failing traces.
- Retrieval rankers that select the best `docs/`, `bcir/`, `mlir/`, and `llvm-training/` context packs.
- Tool routers that choose oracle vs law vs training-corpus vs telemetry specialists.
- Diagnostic models that classify verifier failures, C-front fallback causes, parity drift, and environment limitations.
- Cost-prior and policy models that propose candidate K_BCIR search orders while exact search/verifier gates preserve correctness.
- Synthetic workload generators for fuzzing, parity tests, and curriculum learning.
- Small endpoint models: classifiers, routers, summarizers, kernel-shape advisors, and compiler-action policies.

Lower-confidence or research-only targets:

- Large general-purpose BCIR foundation models trained mainly from GPT outputs. Synthetic-only corpora can inherit teacher blind spots, collapse diversity, and become expensive quickly.
- Autonomous self-improving systems without human review. They may optimize eval loopholes rather than real capability.
- Using GPT labels as legality. That violates the two-truth quarantine; only BCIR verifier/law/tool evidence can decide legality.

OpenAI's model-optimization guide frames evals, prompt engineering, and fine-tuning as a continuous feedback loop, and the agent-improvement cookbook shows traces plus human/model feedback becoming reusable evals and Codex-ready harness changes. BCIR can generalize that pattern from "improve the agent harness" to "produce a typed training stream for BCIR-owned students," as long as every record carries provenance and every promotion is replay-gated.

Sources: OpenAI model optimization guide (<https://developers.openai.com/api/docs/guides/model-optimization>), agent improvement loop cookbook (<https://developers.openai.com/cookbook/examples/agents_sdk/agent_improvement_loop>), and macro evals for agentic systems (<https://developers.openai.com/cookbook/examples/partners/macro_evals_for_agentic_systems/macro_evals_for_agentic_systems>).

### 3.9 Open-weight model ingestion: GLM, Gemma, Qwen, and BCIR readiness

Open weights change the integration problem from "GPT as a remote teacher" to "the model is an artifact BCIR may own, inspect, quantize, place, and serve." BCIR is conceptually well suited to this because its core job is to turn a semantic computation into a legal, costed, target-aware realization with telemetry and replay. The gap is that BCIR currently has **ML primitives and small-model training/inference**, not a full LLM runtime capable of directly loading trillion-parameter-scale checkpoint formats.

#### Model-family fit

| Open-weight family | Fit for BCIR now | Why | Main difficulty |
|---|---|---|---|
| GLM-5.2-class heavy models | Research / cluster-scale target | Strong open-weight coding/agent model; useful as a local teacher or high-end endpoint if the deployment stack already exists | Very large memory/KV-cache, likely tensor/expert parallelism, long-context attention, production scheduler, tokenizer/checkpoint compatibility, safety and license review |
| Gemma 4-class models | Best practical first target | Google describes Gemma as open weights for responsible commercial use; Gemma 4 is explicitly positioned for advanced reasoning/agentic workflows and optimized deployment across hardware classes | Need exact tokenizer, weight-layout importer, attention/rope/norm kernels, quantization and eval harness |
| Qwen open-weight models | Practical first/second target, especially coder/agent variants | Qwen releases provide widely used open-weight coding/reasoning models and deployment recipes; smaller dense/MoE variants are realistic for local or hosted BCIR endpoints | Architecture variants, chat templates, tokenizer edge cases, MoE/expert routing for larger variants, license/version matrix |

The practical recommendation is: **start with a smaller Gemma/Qwen dense instruct model**, prove the checkpoint → BCIR manifest → quantized inference → telemetry → eval loop, then add larger Qwen/Gemma variants, then treat GLM-5.2-class models as a scale-out target once BCIR has sharding, KV-cache management, and production serving.

Sources: GLM-5.2 announcement (<https://z.ai/blog/glm-5.2>), GLM-5 repository (<https://github.com/zai-org/GLM-5>), Gemma 4 model overview (<https://ai.google.dev/gemma/docs/core>), Google DeepMind Gemma 4 page (<https://deepmind.google/models/gemma/gemma-4/>), Gemma open-weight library (<https://github.com/google-deepmind/gemma>), Qwen3 announcement (<https://qwenlm.github.io/blog/qwen3/>), Qwen3.5 announcement (<https://qwen.ai/blog?id=qwen3.5>), and Qwen3.6 repository (<https://github.com/QwenLM/Qwen3.6>).

#### What BCIR already has

BCIR already has many of the pieces required to become an open-weight model substrate:

- **Tensor/math primitives:** matmul, activation, softmax, attention, transformer block references, layernorm, recurrent models, classical models, quantization, losses, optimizers, and autodiff.
- **Training and fine-tuning scaffolding:** deterministic datasets, mini-batches, train/validation splits, supervised training loops, metrics, early stopping, and optimizer state.
- **Lowering paths:** C kernels, LLVM/JIT/AOT hooks, SYCL dispatch, Wasm, specialist lowerings, and target/channel descriptions.
- **Optimization and placement:** K_BCIR cost vectors, target profiles, RCSP, telemetry, calibration, regret, portfolio routing, and provenance manifests.
- **Safety and correctness gates:** R-laws, two-truth quarantine, parity discipline, fuzzing, replay, C/LLVM equivalence checks, telemetry security, and docs/training separation.

These are enough for **small BCIR-native endpoint models** and for **pieces of LLM inference**. They are not yet enough for direct drop-in loading of a modern open-weight chat model.

#### What is missing to plug in open weights

| Missing layer | What must be built | Why it matters |
|---|---|---|
| Checkpoint importer | Load `safetensors`/GGUF/HF shard layouts; map tensor names/shapes/dtypes to a BCIR `ModelManifest`; validate hashes/licenses | BCIR needs a trustworthy bridge from external weights into content-addressed artifacts |
| Tokenizer and chat-template rail | BPE/SentencePiece tokenizer compatibility, special tokens, tool-call tokens, chat templates, detokenization tests | An LLM endpoint is wrong if tokenization or prompt formatting drifts from the model contract |
| LLM graph dialect | First-class ops for embedding, RMSNorm/LayerNorm variants, RoPE/ALiBi, grouped-query attention, sliding/window attention, MoE routing, KV-cache read/write, logits head, sampling | Current transformer code is an oracle composition, not a complete modern decoder-only LLM dialect |
| KV-cache and serving runtime | Paged KV cache, prefill/decode split, continuous batching, speculative decoding hooks, streaming tokens, cancellation, multi-session state | Production endpoints are dominated by decode scheduling and KV memory, not one-shot matmul alone |
| Quantization formats | Weight-only int4/int8, activation quantization, per-channel/per-group scales, GGUF/AWQ/GPTQ/FP8-style adapters, accuracy-law extensions | Open models are practical only when quantized and accuracy-bounded |
| Parallel placement | Tensor parallel, pipeline parallel, expert parallel, CPU/GPU/NPU offload, multi-device channel cost model | GLM-5.2-class models require scale-out; even smaller models benefit from heterogeneous placement |
| Kernel library | Fused QKV, attention kernels, RoPE kernels, RMSNorm, gated MLP/SwiGLU/GELU, dequantized GEMM, MoE dispatch, logits/sampling kernels | Existing matmul/attention references need production kernels and law parity |
| Endpoint API | OpenAI-compatible `/v1/chat/completions` or Responses-like adapter, streaming, tool-calling schema, structured outputs, auth/quota/rate limits | Makes BCIR-owned models usable by existing agent tooling |
| Eval and safety harness | Per-model eval packs, jailbreak/prompt-injection tests, license/safety metadata, red-team corpora, hallucination/faithfulness checks | Open weights remove provider-side guardrails; BCIR must own the deployment safety envelope |

#### A staged implementation path

1. **Manifest-only ingestion.** Create `ModelManifest` records for a small Gemma/Qwen model: architecture, license, tokenizer ref, weight shards, hashes, dtype, parameter count, context length, and required kernels.
2. **Tokenizer parity.** Add tokenizer round-trip tests and chat-template fixtures before touching weights.
3. **Reference decode.** Implement a slow, dependency-light Python reference for one small dense decoder layer using existing matmul/activation/attention pieces plus missing RMSNorm/RoPE/KV-cache primitives.
4. **Quantized inference artifact.** Import a tiny or small model subset, quantize, run deterministic prompt fixtures, and record accuracy/perplexity drift.
5. **C/MLIR law rail.** Add ODS ops and C++/MLIR verification for LLM-specific ops; keep Python oracle and law rail in parity.
6. **Serving endpoint.** Build a BCIR endpoint wrapper with streaming decode, schema-constrained tool-call output, telemetry frames, and replay manifests.
7. **Scale-out.** Add continuous batching, paged KV, multi-device placement, and expert/tensor parallelism for larger Qwen/Gemma and eventually GLM-class models.
8. **Fine-tune/adapt.** Add LoRA/QLoRA-style adapters as first-class artifacts before full-parameter training; freeze adapters with the same provenance and eval gates as kernels.

#### Bottom line

BCIR is **well suited architecturally** for open weights because it already thinks in terms of typed graphs, lowering, costed placement, telemetry, quantization, parity, and provenance. BCIR is **not yet a plug-and-play LLM inference engine**. The fastest credible path is not GLM-5.2 first; it is a small Gemma/Qwen dense model, imported through a manifest/tokenizer/KV-cache path, then lowered into BCIR kernels and exposed as a guarded endpoint. Once that is stable, heavier models become an engineering problem of sharding, KV memory, kernel performance, and safety operations rather than a conceptual mismatch.

## 4. Proposed architecture

```text
ChatGPT / API client / Codex
        |
        v
OpenAI Responses API or Agents SDK
        |
        +-- Manager agent
        |     +-- BCIR oracle tools      -> bcir/ CLI + Python APIs
        |     +-- MLIR law tools         -> bcir-opt + ODS/IRDL scripts
        |     +-- Training corpus tools  -> llvm-training retrieval/evals
        |     +-- Telemetry tools        -> telemetry frames + traces
        |     +-- Patch tools            -> git workspace + tests
        |
        v
MCP server for ChatGPT Apps
        |
        +-- Structured tool contracts
        +-- UI resources/widgets
        +-- Auth, approvals, audit log
        |
        v
BCIR deterministic gates
        +-- R-laws / verifier
        +-- Python↔MLIR parity
        +-- C/LLVM equivalence
        +-- tests/fuzz/differential campaigns
        +-- provenance generation
```

### Tool-contract principles

- Return structured JSON plus concise natural-language summaries.
- Include provenance: command, repo commit, target, theta, seed, environment, and artifact hashes.
- Keep tools narrow and typed; prefer `plan_program` over `run_shell`.
- Separate read-only tools from mutating tools.
- Treat MLIR availability, compilers, and hardware probes as capabilities discovered at runtime.

## 5. Proposal versions

### Version 0: Research-only repository intelligence

Scope:

- Build a read-only BCIR/LLVM-training retrieval index or MCP resource server.
- Add canned prompts and evals for repository Q&A.
- No tool execution except file retrieval.

Value:

- Fastest path to a useful ChatGPT/Codex companion.
- Low security risk.
- Teaches agents the separation between training corpus and IR.

Limitations:

- Cannot validate claims by running BCIR.
- No telemetry or parity feedback.

### Version 1: Deterministic BCIR tool server

Scope:

- Add an MCP/HTTP tool server wrapping read-only and bounded deterministic commands: planning, MLIR emission, selected tests, status summaries.
- Add structured outputs and provenance.
- Add a simple ChatGPT app UI for plan visualization.

Value:

- ChatGPT can operate BCIR as a conformance oracle.
- Users get interactive compiler explanations inside ChatGPT.

Required gates:

- JSON Schema validation.
- Timeouts and resource limits.
- No arbitrary shell.

### Version 2: Multi-agent parity and repair loop

Scope:

- Agents SDK manager with oracle, law, training, telemetry, and patch specialists.
- Trace every run.
- Generate evals from failures.
- Use Codex-style patch handoffs for docs/tests/code.

Value:

- Closes the loop from question → run → diagnosis → patch → gate.
- Aligns OpenAI traces/evals with BCIR's own parity and provenance discipline.

Required gates:

- Human approval before mutating repository or running expensive campaigns.
- Mandatory test/parity command recording.
- PR summaries that cite changed files and commands.

### Version 3: BCIR-guided AI model development loop

Scope:

- Use GPT to propose workloads, features, candidate policies, synthetic labels, and experiment plans.
- Run incremental cloud-teacher training sessions: GPT meta-agents generate typed episodes; BCIR validates them; BCIR-owned students train on accepted records; promotion freezes deterministic artifacts.
- Feed outputs into BCIR training/calibration organs as explicitly tagged synthetic evidence.
- Promote only deterministic frozen artifacts after replay, regret, parity, and provenance gates.

Value:

- Uses GPT calls to bootstrap new model-development datasets and policy searches without pretending GPT is in the hot path.
- Enables frontier-inference bootstrapping: current GPT capabilities manufacture a validated experience buffer that trains narrower, cheaper, BCIR-owned models and agents.
- Lets BCIR become a compiler substrate for AI model training/inference experiments.

Required gates:

- Synthetic-vs-measured data separation.
- Two-truth quarantine: learned confidence never becomes legality.
- Frozen Q8/code artifacts for L1 or below.
- Held-out evals and macro-evals to prevent self-generated data from overfitting the teacher's blind spots.

### Version 4: ChatGPT-hosted BCIR Workbench

Scope:

- A full Apps SDK UI with target selection, cost-vector visualization, MLIR diffs, law diagnostics, telemetry charts, and training lessons.
- Optional user-authenticated workspace operations.

Value:

- Makes BCIR explorable by researchers and compiler engineers from ChatGPT.
- Provides a natural front end for teaching, demos, and guided development.

Required gates:

- Auth and deployment hardening.
- Sandbox compilers and hardware tools.
- Clear user approval for writes and expensive runs.

### Version 5: BCIR-trained endpoint models

Scope:

- Serve BCIR-owned student models as endpoints for narrow compiler/AI tasks: retrieval ranking, diagnostic classification, tool routing, workload synthesis, and kernel-shape advice.
- Add an open-weight endpoint track: start with small Gemma/Qwen models, then scale toward larger Qwen/Gemma and GLM-class deployments after checkpoint import, tokenizer parity, KV-cache, quantization, and serving runtime exist.
- Use GPT meta-agents as cloud teachers during training, then route easy/high-confidence production calls to the cheaper BCIR endpoint and reserve GPT or heavier open models for hard cases, critique, and curriculum generation.
- Export endpoint artifacts with model card, dataset/model provenance, eval suite, quantization record, latency/cost envelope, tokenizer/chat-template hash, and rollback plan.

Value:

- Turns API-call bootstrapping into deployable BCIR-owned AI services rather than a permanent dependency on every GPT call.
- Creates a ladder from GPT-assisted data generation to auditable BCIR inference endpoints.

Required gates:

- Shadow-mode deployment before live routing.
- Confidence/uncertainty thresholds that escalate to GPT or human review.
- Drift monitoring and periodic replay against frozen evals.
- Clear separation of endpoint predictions from BCIR legality verdicts.

## 6. Recommended next implementation steps

1. Create a narrow `bcir.tools` Python facade for structured operations already supported by the CLI.
2. Add JSON-serializable result types that include provenance and artifact hashes.
3. Build a local MCP server with read-only tools first: `repo_summary`, `plan_program`, `emit_mlir`, `training_lookup`.
4. Add a small eval set from `llvm-training/` exercises and BCIR parity examples.
5. Add mutating patch workflows only after read-only tools are stable.
6. Design the ChatGPT Apps UI around structured artifacts, not free-form logs.
7. Keep all OpenAI-derived recommendations outside BCIR legality until they are frozen, replayed, and accepted by deterministic gates.
8. Define a `TrainingSession`/`Episode` schema for GPT-generated data with provenance, teacher model ID, prompt pack hash, validation result, and promotion status.
9. Start with narrow students (retrieval ranker, diagnostic classifier, tool router) before attempting larger neural endpoint models.
10. For open weights, build manifest-only ingestion and tokenizer parity before implementing weight loading or decode kernels.
11. Use a small Gemma/Qwen dense model as the first open-weight target; defer GLM-5.2-class deployments until BCIR has paged KV cache, quantized kernels, and multi-device placement.

## 7. Core conclusion

ChatGPT can integrate deeply with BCIR as an orchestrator, researcher, teacher, patch author, trace critic, UI host, cloud-teacher data generator, and meta-learning proposer. The frontier model can bootstrap new BCIR neural networks and endpoint models by producing validated training episodes, but it should not be integrated as a verifier, hot-path planner, unmediated source of truth, or hidden gradient oracle. Open-weight models add a second path: BCIR can eventually own the weights and endpoint, but only after manifest ingestion, tokenizer parity, LLM dialect/KV-cache support, quantized kernels, serving runtime, and safety/eval gates exist. The correct architecture is a layered one: OpenAI APIs and open weights expand exploration and synthesize experience; BCIR's oracle/law/parity/training system decides what becomes a frozen, replayable machine artifact.
