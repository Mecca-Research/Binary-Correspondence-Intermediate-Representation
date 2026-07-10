# BCIR Machine-Code / HAL / ABI / ISA Audit — the wave-14 research pass (2026-07-04)

**Question set** (verbatim intent): does BCIR need a dedicated HAL backend or does the IR
already cover it; should HAL functions migrate into the BCIR ABI; how mature is the ABI
toward POSIX/Linux backward compatibility with unique BCIR signatures; what exists for hex
convert/dump/load, assembly listing, mnemonics, peek/poke, hierarchical access memory,
semantic swap, carry flags, Registry-Oriented Programming, macro assembly, jumping/
branching, displacement; how complete are the assembler/linker/object-code toolchains
against the MLIR dialect and GEM; how much rides Clang vs native BCIR; how complete are
disassembly, integer registers, data movement, operand combinations, addressing modes, and
the ISA backend overall.

**Research inputs**: the Microchip MCU16 HAL spec (peripheral drivers + hardware-access
facade + two-level BSP macro mapping), the LinuxCNC/EMC HAL Handbook (components / typed
pins / signals / threads; `halcmd` `setp`/`getp`; halmeter/halscope), two ISA texts
(Hamacher ch. 2: memory/addressing/endianness, instruction classes, addressing modes,
assembly, stacks/subroutines; an ISA-design survey: RISC/CISC, encodings, SIMD, compiler
interaction), and an x86 opcode-structure reference (prefix/opcode/ModR/M/SIB/
displacement/immediate). Repo evidence: a three-rail audit (toolchain, machine model,
HAL/ABI) with `file:line` anchors throughout.

---

## 0. Executive verdicts

| # | Question | Verdict |
|---|---|---|
| V1 | **Dedicated HAL backend?** | **No new backend.** The registry/channel/manifest stack **is** the HAL's schema + facade layers *by construction* (compile-time). What's missing is not a backend but three specific HAL parts: **resident peripheral drivers** (runtime), a **BSP-style name-binding table**, and the **operator tools** (`halcmd`-class peek/poke). See §4, gaps MC2/MC8. |
| V2 | **HAL functions → BCIR ABI?** | **Yes — as a versioned hook vtable.** `RuntimeChannel` today carries only telemetry-source descriptors (`bcir/channels.py:47-56`); the classic HAL verbs (open/claim, map, submit, sync, event delivery) belong there as a **frozen, append-only ABI** exactly like the StreamPack wire format. The laws stay compile-time (two-truth); the hooks are the runtime half. Gap MC8. |
| V3 | **ABI maturity / POSIX-Linux compat?** | **Five frozen artifacts exist** (§5): StreamPack v1–v3 (BSPK), the `bcir_exec` calling convention, the R12 `TargetABI` matrix (Linux LP64 ×3 + Windows LLP64 + ILP32), the Q8 tier tables, and the telemetry frame (BTLM). POSIX coupling today is **read-only** (two syscalls + sysfs/procfs + one gated DVFS write). Full backward compat still needs the device-fd lifecycle, BAR/UIO mmap, ioctl submission, eventfd/poll delivery, and errno bridging. The "unique BCIR signatures" already exist: magics + digests + generation tags on every artifact. |
| V4 | **Hex/dump/loader, listing, mnemonics, disassembly?** | **Loader: done, dual-rail** (`bcir_sp_validate` freestanding C + Python `decode`). **Hex dump/convert, assembly listing, and the disassembler: missing entirely** — the highest-leverage, lowest-cost gap in the audit (MC1). |
| V5 | **Peek/poke?** | **Half-built.** The *scope* half exists (TelemetryRing / DataDNA / DurableLog ≈ halscope; the shared-mmap ring is the zero-copy channel). The *interactive* half (`setp`/`getp` on live registry state, with `data_gen` bumps making R11 refuse stale packs) is missing — and the refusal law it needs **already exists** (MC2). |
| V6 | **Assembler/linker/object code?** | **Source-level: done** (bcir-cc, `--linkable`, derived link flags, real ELF via resident clang/llc — eBPF/x86-64/aarch64, `e_machine`-verified). **Native isel: deferred by the standing gate** (correct; G1–G4 unmet). **Pack-level linking (multi-pack compose, RID relocation, symbol section): missing** (MC7). |
| V7 | **Clang reliance?** | Every path from IR to *machine code or execution* rides clang/llc/lli/wasm-ld **by design** (the resident-compiler path). Everything else — all text emission (LLVM IR, C23, MLIR dialect, stack-VM mnemonics), the StreamPack codec, the loader/validator/executor, bcir-cc — is native BCIR (pure Python + freestanding C). This is the gate's intended split; the audit found no accidental clang dependence. |
| V8 | **Registers / flags / branches / addressing?** | **Registers: none, by design** (RID-addressed registry). **Flags/carry: none** — comparisons produce *values* (registry-first: conditions are data, not hidden state); what's genuinely missing is carry-as-data for wide arithmetic (MC4). **Branches: structured emit-only tree, erased before planning** — the planner sees straight-line phased claims (a real, documented modeling boundary; MC5 names what a CFG-aware plan would add). **Addressing: affine only** (`offset`/`stride_k`/`count`/stride-class); indexed access is the GGG gather lane, not a mode — SIB-style compound modes are *intentionally* absent. |
| V9 | **HAM / semantic swap / hotness?** | **HAM is a cost knob only** (realize + R3 + MLIR attr; never consumed by allocator or executor). **`Resource.priority` ("CXL semantic hotness") is inert** — read only by the provenance digest. **No swap/eviction exists** outside the KV-page evict. MC6 makes all three real *as planned claims* (the A2/D-R3 machinery now makes that cheap). |

---

## 1. The machine model as built (the "BCIR ISA")

The instruction set is a **variadic, RID-addressed, phase-ordered dataflow claim graph** —
deliberately not a register machine:

- **18 opcodes** (`bcir/model/opcodes.py:8-27`): NOP, LOAD, STORE, ADD, SUB, MUL,
  ATOMIC_ADD/SUB/XOR, CMPXCHG, BARRIER, PHASE_ENTER/LEAVE, GGG_LOAD/STORE (gather/
  scatter), T_MACC (tile matmul-accumulate), GEM_DISPATCH, PROV_NOTE. No DIV/MOD/shift/
  compare/branch *opcodes* — the C frontend multiplexes those onto ADD/SUB/MUL with the
  semantics in the op **string** (`c.bin.lt`, `c.bin.shl`, … — `cfront/lower.py:44-52`).
  The op string is the real mnemonic space; the opcode is the cost/hazard class.
- **Operands are unbounded tuples**: `rd`/`wr`/`imm` (`model/graph.py:83-85`) — 2-read/
  1-write elementwise through many-read reductions and immediate-carrying loads. The
  operand-combination question dissolves: any combination is expressible; legality is
  policed by R-laws, not encoding slots.
- **Addressing is affine-per-resource**: `offset` (displacement), `stride_k` (stride),
  `count` (extent), `stride_class` (SCALAR/UNIT/STRIDED/CACHELINE/TILE/RANDOM), lowering
  to StreamPack `Block(base, count, strides)` (`gem/streampack.py:101`). x86's
  base+index×scale+disp compound mode has no analog **on purpose** — data-dependent
  indexing is the GGG lane (priced, always-legal, minimized), and D-R4's `StridedView`
  is the multi-dimensional generalization at the allocation seam.
- **No condition flags anywhere.** `if (a < b)` lowers to a `c.bin.lt` claim producing a
  *value* RID that feeds `IfNode.cond` (`cfront/lower.py:405,2280-2282`). The one "carry"
  in the model is `precision="compensated"` (residual-carry MAC — numerics, not a flag).
  Registry-first answer, stated as a design position: **conditions are data**. The gap
  that remains is *carry-as-data* for multi-word arithmetic (MC4).
- **Branch/jump truth**: control flow lives in an **emit-only structured body tree**
  (`IfNode/WhileNode/GotoNode/LabelNode/ComputedGotoNode/SwitchNode`,
  `cfront/lower.py:403-448`) that `_flatten_block` **erases before planning**
  (`lower.py:490-502`) — the planner/verifier/executor see one straight-line phase list;
  a `WhileNode` folds to a per-iteration cost region with a static `bound`
  (`lower.py:2426-2448`). Displacements/targets are label *strings* and value RIDs, never
  numeric offsets. This is the biggest honest boundary in the machine model (MC5).
- **The executor holds no value state** (`gem/execute.py:28-33,58`): it is a
  deterministic dispatch engine (topo phases → stable claim order → caller kernels), so
  there is nothing to single-step and no register file to inspect — peek/poke must be
  built on the **registry**, not on the executor (MC2).
- **Data representations**: elem_bytes per resource, C23 `_BitInt(N)` and Q8/Q-fixed
  lanes in the C kernels, safetensors F64/F32/F16/BF16 ingestion, endianness fixed
  little-endian on every wire artifact.

## 2. The toolchain as built

| Stage | State | Anchor |
|---|---|---|
| Text emission — LLVM IR / C23 / MLIR dialect / stack-VM mnemonics | **Mixed scope**: LLVM IR is a single-claim elementwise subset; MLIR AOT preparation is partial; C and stack emitters have their separately documented subsets | `lower/llvm.py`, `lower/c_kernel.py`, `lower/mlir.py`, `lower/stackify.py` |
| Object emission | **Done via resident compiler**: `codegen` (llc → ELF/PTX), `codegen_object_c` (clang → ELF, `e_machine`-verified for eBPF 247 / x86-64 62 / aarch64 183) | `codegen/codegen.py:36-186` |
| Execution | Single-claim elementwise AOT (clang), JIT (`lli`), and WASM (clang+wasm-ld+node); JVM/CIL bounded stack subsets are execution-validated in tests | `lower/jit.py`, `lower/wasm.py` |
| Binary artifact (the BCIR "object code") | **StreamPack, frozen v1 + append-only v2/v3**, dual-rail codec, CRC + R10/R11 semantic trust boundary, adversarial corruptor harness | `abi/streampack_abi.py`, `runtime/c/bcir_runtime.h:46-107`, `tools/c/streampack_corrupt.py` |
| Loader | **Done, dual-rail, freestanding** (no-libc C validate/walk/execute) | `bcir_runtime.h:46-90`, `bcir_exec.c:33` |
| Compiler driver | **bcir-cc** (Python-free: cpp → cfront → plan → hydrate → `--emit-pack`), `--linkable`, `--emit-link-flags` | `runtime/c/bcir_cc.c:126-341` |
| Assembler (text → machine code) | **Missing / correctly gated** — native isel deferred (G1–G4 unmet); stack encoders stop at mnemonics | `BCIR_NATIVE_OBJECT_GATE.md` |
| **Disassembler / listing (pack → text)** | **Missing entirely** — no StreamPack lister, no annotated hex dump; `llvm-objdump` used only as `-f` format validation | `codegen.py:66-70` |
| Pack-level linker | **Missing** — no multi-pack compose, no RID relocation, no symbol section (packs carry integer RIDs + TraceNotes only) | `streampack_abi.py:180-183` |
| Naming traps (recorded) | `runtime/c/bcir_decode.c` is transformer math, not the pack decoder; `bcir_exec.c` dispatches caller kernels, it does not run machine code | — |

**Clang-vs-native split** (V7): the boundary is exactly the gate's design — *plan,
verify, encode, load, schedule* are native; *instruction selection and machine encoding*
are the resident backend's. Nothing in the audit argues for moving that boundary; the
missing tools (MC1/MC7) are all on the **native** side of it.

## 3. What the research docs add to the frame

- **Microchip-style HAL** (drivers + facade + BSP): BCIR has the *schema* analog of
  drivers (blueprints/regmaps as compiled fixtures), a *read-only* facade
  (silicon probes + signal registry under veto-not-steer), and **no BSP** — no two-level
  "application name → peripheral method" indirection. In BCIR terms a BSP is a **binding
  table: logical name → (RID, claim recipe)** — a registry object, diffable and
  digestable like everything else (MC3/MC8).
- **LinuxCNC HAL** (components/pins/signals/threads + `halcmd`): maps 1:1 —
  component ≈ claim/kernel, pin ≈ RID port, signal ≈ shared RID binding, typed pins ≈
  Domain/elem contracts, thread+period ≈ phase + §5.11 Timing, `addf` order ≈ claims-in-
  phase order, **`setp`/`getp` ≈ the missing governed poke/peek** (MC2), halmeter/
  halscope ≈ TelemetryRing/DataDNA (already built), dynamic re-netting while running ≈
  R11 rehydration (`map_gen` bump) — the law for safe live reconfiguration **already
  exists**; only the tool is missing.
- **ISA texts + x86 encoding**: the classic checklist (registers, addressing modes,
  condition codes, branching, displacement, assembler/loader) is exactly the list §1
  answers — mostly "by design, differently"; the x86 ModR/M/SIB/disp machinery is the
  thing D-R4/GGG deliberately replace with stride vectors and priced gathers.

## 4. HAL/ABI as built

Compile-time HAL (all built, all law-governed): channel registry + capability routing
(dual-rail C twin), DeviceManifest banks/distance/StridedView (D-R1..D-R4), silicon
probes (perf_event_open + ioctl on the counter fd, sysfs/procfs, RAPL, thermal —
read-only; one gated DVFS write), signal registry with declared-unavailable gap
providers, event phases + DMA descriptor rings as claims (A1/A2).

Runtime HAL (the missing half): `RuntimeChannel` has **no open/alloc/map/submit/sync/
event hooks** (`channels.py:47-56`); modeled channels have no resident driver; DMA/IRQ
exist as claims, not as device I/O; no `/dev` access anywhere; no BAR mmap (the only mmap
is the telemetry ring). This is not drift — it is the documented "resident drivers ⏭"
line in `HETEROGENEOUS_CHANNELS.md` — but it is now the binding constraint for the deep
driver phase.

## 5. The ABI ledger (what is frozen today)

1. **StreamPack** BSPK v1 (frozen) / v2 pipeline / v3 dispatch+channel — dual-rail,
   CRC'd, generation-tagged, semantically verified post-CRC.
2. **`bcir_exec` calling convention** — caller-owned memory, `bcir_exec_fn(item, ctx)`
   kernel callback, generation-checked variant (`bcir_exec.h:31-79`).
3. **R12 `TargetABI` matrix** — x86_64/aarch64/riscv64-linux (LP64), x86_64-windows
   (LLP64), i386-linux (ILP32); layout-only by design; MLIR twin op + pass check.
4. **Q8 tier tables** — frozen LE blobs + `#embed` header, drift-gated.
5. **Telemetry frame** BTLM v1 — 56-byte DataDNA records, CRC, resync-by-magic.

No runtime version *negotiation* exists (frozen bytes + reject-newer is the policy) —
adequate for artifacts; the MC8 hook vtable needs the same append-only discipline.

---

## 6. The gap register — the MC-track (ranked)

| # | Gap | Shape of the fix | Cost | Blocks |
|---|---|---|---|---|
| **MC1** | **Disassembler + hex dump + listing** | `bcir dis`: StreamPack → annotated listing (header fields, per-segment mnemonic line: op string, lane/width, rd/wr RIDs *resolved to names via TraceNotes/module*, blocks, prefetches, CRC state); `bcir hexdump`: offset-annotated hex of every record; round-trip pinned against the codec. One normative mnemonic table shared with ROP v2 (D-R6 discipline: the table IS the ISA language, both directions). Pure native, both rails eventually (Python first). | S | Debugging everything downstream; the driver phase's first tool |
| **MC2** | **Peek/poke (`halcmd` class)** | Registry inspector + governed mutator: `show res/claims/phases` (peek = read-only), `setp rid value` (poke = write + `data_gen` bump → R11 makes every hydrated pack refuse until rehydrated — **the law already exists**, only the tool is missing); halscope half already built (TelemetryRing/DurableLog). Wire into `bcir/run.py` as a REPL subcommand. | S | Driver bring-up ergonomics; live-tuning workflows |
| **MC3** | **ROP v2 — the registry assembly language** | Grow `frontends/rop.py` to the full claim surface: named constants, `imm`, `offset`/`stride_k`, hazard/bounds/verify contracts, phase `deps`, `event` phases, **macros** (MAP's unkept promise), includes, and the **BSP binding-table section** (logical name → RID). MAP folds in or retires. This is "Macro Assembly Programming" done registry-first. | M | Hand-written driver fixtures; MC1's symbol resolution |
| **MC4** | **Carry-as-data + typed predicates** | Two registry-first moves: (a) promote comparison results from op-string suffixes to a typed predicate contract (the value already flows; type it so R-laws can see it); (b) wide/checked arithmetic emitting **carry as a second write RID** (`ADD` with `wr=(sum, carry)`) — the multi-word `_BitInt` path and checked-arith C lowering both want it. No status register is ever introduced; both are claims writing data. | M | Wide-integer codegen honesty; crypto/checksum kernels |
| **MC5** | **CFG-aware planning (named boundary, not yet a build)** | Today the planner prices loops as one region with a static bound and never sees branches. Recording the options: (i) keep the boundary (dataflow stays the plan currency; branchy code is the emitter's job) — the standing position; (ii) conditional phase deps (a phase guarded by a predicate RID) — the smallest true extension, needed only when a driver blueprint demands *planned* divergent paths. Decide when a fixture forces it, not before. | L | Nothing today; revisit at the first branchy driver plan |
| **MC6** | **HAM + semantic swap made real** | (a) Allocator consumes `Resource.priority` (today inert) as the tie-break/pinning input; (b) **swap as planned claims**: an eviction pass that mints `mem.move.*` (D-R3-priced, A2-descriptor-backed) when a hotter resource wants a full tier — the machinery all exists post-wave-13, it has never been composed; (c) HAM into placement (an access="ham" resource prefers the tier whose latency the log-model assumed). All measured-then-pinned like every cost feature. | M | Honest memory-strategy story; CXL-tier work later |
| **MC7** | **Pack-level linking + symbol section** | Multi-pack compose: RID-band relocation (renumber with provenance), claim-id rebasing, a v4 append-only **symbol section** (name ↔ RID/claim map — today only TraceNotes tie ids to hashes). Enables multi-TU StreamPack programs and gives MC1's disassembler real names without the source module. | M | Multi-module deployment; whole-program packs |
| **MC8** | **RuntimeChannel v2 — the HAL hook vtable (HAL→ABI migration)** | The V2 verdict made concrete: an append-only, versioned hook table on `RuntimeChannel` — `open/claim(device)`, `map(StridedView)→host view`, `submit(pack/segment)`, `sync/fence`, `event(source)→delivery` — with the POSIX backing (fd lifecycle, UIO/VFIO mmap, ioctl submit, eventfd/poll delivery, errno→`bcir_status` bridge) as the first implementation. Laws stay compile-time; hooks are runtime; probe results may **veto, never steer** (D-R1 holds at runtime too). Rig-gated pieces stay gated. | L | The deep-driver phase's runtime half; real DMA/IRQ |
| **MC9** | **POSIX/Linux compat completion** | The remaining backward-compat items under MC8's umbrella: real UAPI header fixtures (today "uapi" exists only in docs — the named fixtures are CMSIS-style `uart_regs.h`/`cmsis_gpio.h`), `/dev` lifecycle, errno propagation, and the Linux-ABI conformance pins (LP64 matrix already frozen in R12). | M | Linux-resident deployments |

**Reading order into the deep-driver phase**: MC1 → MC2 → MC3 (the tools the driver
bring-up will use daily, all cheap and native), then MC8 in design alongside the driver
analysis (its hook list should be *derived from* the first real driver's needs, exactly
as Part VII derived A1/A2 from the datasheets), with MC4/MC6/MC7 as independent build
tracks and MC5 parked until a fixture forces it.

---

## 7. Standing positions this audit confirms (no change needed)

- **No register file, no flags, no branch opcodes** — registry-first dataflow is the
  design, not an omission; conditions and carries are data (MC4 completes the carry
  half without introducing hidden state).
- **Native isel stays deferred** — the gate (G1–G4/S1–S4) survived this audit untouched;
  every missing tool lands on the native side of the resident-compiler boundary.
- **Two-truth + veto-not-steer extend to the runtime HAL**: MC8's hooks observe and
  refuse; they never alter legality or steer plans.
- **The one-table law (D-R6) governs mnemonics**: assembler (ROP v2) and disassembler
  (MC1) must consume the same normative op table — never two spellings.
