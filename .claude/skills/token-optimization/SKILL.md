---
name: token-optimization
description: Token-frugal operating protocol for working in this repo. Use at the START of any session and whenever context is growing fast — before exploring the repo, launching subagents, reading large files, or producing long outputs. Triggers - "optimize tokens", "token budget", "context is huge", "hit the limit", session start on this repo.
---

# Token Optimization Protocol (BCIR repo)

Goal: minimize tokens per unit of completed work with **no loss of correctness** —
`min Σ H(Tᵢ) subject to I(T_opt) = I(T_original)` (compress representation, never meaning).
Treat context like the K_BCIR cost model treats memory bandwidth: a priced, budgeted
resource with a legality floor (correctness) that is never traded away.

## 1. Digest-first navigation (persistence / knowledge regeneration)

- **Read `.claude/context/BCIR_DIGEST.md` FIRST** — it is the compressed knowledge base
  (repo map, subsystem summaries, counts, doc inventory, key invariants). It replaces
  ~500k tokens of repo exploration with ~4k.
- Only read source files the digest cannot answer. Regenerate/extend the digest at
  session end if you learned something durable:
  `python .claude/skills/token-optimization/scripts/build_digest.py --check`
- Never re-derive facts already in the digest, `docs/STATUS.md` (generated counts), or
  the current conversation.

## 2. Retrieval discipline (grep-before-read, headroom-style routing)

- **Grep → targeted Read → full Read**, in that order. Use `Grep` with `head_limit`,
  `Read` with `offset`/`limit`. Never read a >500-line file end-to-end to answer a
  point question.
- Route by content type: structured data → extract fields with one `python3 -c`;
  logs/test output → tail/grep the failure lines only; big command output → redirect
  to the scratchpad and print a ≤10-line summary (reversible compression: the full
  artifact stays on disk, retrievable on demand — cite the path instead of pasting).
- Batch independent shell operations into ONE Bash call; batch independent tool calls
  into one message.

## 3. Fan-out budget (agents are the fuel-burners)

- Default is **inline work, zero subagents**. A subagent costs its whole transcript.
- Spawn an agent only when the task genuinely exceeds one context (e.g. sweeping
  hundreds of files) AND the user has opted in. Cap fleets; prefer 1–3 agents with
  tight, structured-output prompts over 10 broad ones. Never duplicate an agent's
  search yourself.
- Concurrency here is ~2 (4 cores): a 10-agent fan-out runs 5× serial — slow AND
  expensive. Size fleets to the machine.

## 4. Write/edit discipline

- Edit surgically (`Edit`, or one `python3` heredoc for bulk transforms). Never
  rewrite a whole large file to change a section; never Read a file back after
  Write/Edit (the tool errors if it failed).
- For bulk doc restructuring, compute the transform in one scripted pass instead of
  many round-trips.

## 5. Context hygiene (sliding window + cache alignment)

- Keep a running ≤10-line worklist (TaskList) instead of re-narrating state.
- Long waits: don't poll. Background tasks notify on completion; if you must wait,
  block once with a long timeout. (Cache TTL is ~5 min — either stay under it or
  commit to one long sleep; don't pay the miss repeatedly.)
- Emit checkpoint notes into the scratchpad before context grows near compaction, so
  a summarized session can resume from the note, not from re-exploration.
- Final replies: lead with the outcome; supporting detail only where it changes what
  the reader does next. No re-pasting of file contents the user can open.

## 6. Alias table (smart-number encoding for intra-session notes)

Use these in scratchpad notes/plans (NOT in user-facing docs): oracle=`bcir/` Python
rail · law=`mlir/` rail · twin=`runtime/c/bcir_cfront.c` · SP=StreamPack · MR=master
roadmap · MLAI=ML/AI roadmap · DKR=driver/kernel roadmap · GT=ground truth
(STATUS.md) · R-laws=R1–R25 · DH=DEVELOPMENT_HISTORY.md.

## 7. Optional heavy machinery

For multi-agent or proxy-level compression, the headroom library
(github.com/headroomlabs-ai/headroom) offers tool-output compression (SmartCrusher/
CodeCompressor), reversible retrieval (`headroom_retrieve`), and KV-cache prefix
alignment — integrate as an MCP server or proxy only if the user asks; this protocol
captures the same wins behaviorally at zero dependency cost.
