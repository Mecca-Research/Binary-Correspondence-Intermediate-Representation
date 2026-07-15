# BCIR Machine-Code / HAL / ABI / ISA Audit — verified gap register (2026-07-15)

> Status in this document is derived from repository implementation and tests, not from roadmap
> prose. The original wave-14 analysis is retained, but MC1/MC2 and the x86 asm-edge statuses are
> updated to the current tree, MC10–MC14 capture backend contracts omitted by the first audit,
> and MC15 records the measurement/trace feedback ABI required by drivers and kernels.

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
| V1 | **Dedicated HAL backend?** | **No new backend.** The registry/channel/manifest stack **is** the HAL's schema + facade layers *by construction* (compile-time). What's missing is not a backend but three specific HAL parts: **resident peripheral drivers** (runtime), a **BSP-style name-binding table**, and a **live hardware binding** for the landed in-process operator-tool baseline. See §4, gaps MC2/MC3/MC8. |
| V2 | **HAL functions → BCIR ABI?** | **Yes — as a versioned hook vtable.** The direct C ABI is now `bcir_runtime_channel.h`: open/claim, offset-based map, submit, sync/cancel, event, and close with append-only size/version checks. The laws stay compile-time (two-truth); a real hardware binding and any optional transport remain MC8 work. |
| V3 | **ABI maturity / POSIX-Linux compat?** | **Six bounded contracts exist** (§5): StreamPack v1–v3 (BSPK), the `bcir_exec` calling convention, the R12 `TargetABI` matrix, Q8 tier tables, frozen BTLM v1, and the direct RuntimeChannel v1 hook ABI. BTLM is only a single-producer DataDNA codec; it is not a source/session/clock-aware driver ABI. POSIX coupling today is **read-only** (two syscalls + sysfs/procfs + one gated DVFS write). Full compatibility still needs device-fd lifecycle, BAR/UIO mmap, ioctl submission, eventfd/poll delivery, errno bridging, and MC15's driver telemetry contract. |
| V4 | **Hex/dump/loader, listing, mnemonics, disassembly?** | **Loader: done, dual-rail. MC1 Python baseline: landed.** `bcir-pack dis` and `bcir-pack hexdump` validate the whole artifact before displaying exact record spans generated from the codec's own writers. Remaining: a freestanding C twin, source-module symbol/name resolution, and device-command ISA disassemblers. |
| V5 | **Peek/poke?** | **MC2 in-process baseline: landed.** `bcir-registry` provides deterministic show/getp/setp; each poke advances the resource's u32 `data_gen`, so R11 refuses old packs. It is not a live RuntimeChannel or hardware transport; binding, permissions, register width/side effects, and device lifecycle remain open. |
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
| **Disassembler / listing (pack → text)** | **Python baseline landed** — validated StreamPack listing and record-delimited hex dump; exact spans share the normative codec writers. C twin and device-ISA disassembly remain open | `abi/streampack_tool.py`, `tests/test_streampack_tool.py` |
| Pack-level linker | **Missing** — no multi-pack compose, no RID relocation, no symbol section (packs carry integer RIDs + TraceNotes only) | `streampack_abi.py:180-183` |
| Naming traps (recorded) | `runtime/c/bcir_decode.c` is transformer math, not the pack decoder; `bcir_exec.c` dispatches caller kernels, it does not run machine code | — |

**Clang-vs-native split** (V7): the boundary is exactly the gate's design — *plan,
verify, encode, load, schedule* are native; *instruction selection and machine encoding*
are the resident backend's. Nothing in the audit argues for moving that boundary; the
remaining tools (MC1's C/device twins and MC7) are on the **native** side of it.

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
  phase order, **`setp`/`getp` ≈ the governed in-process poke/peek baseline** (MC2), halmeter/
  halscope ≈ TelemetryRing/DataDNA (already built), dynamic re-netting while running ≈
  R11 rehydration (`map_gen` bump) — the law for safe live reconfiguration **already
  exists**; only the tool is missing.
- **Telemetry standards** add a distinct measurement ABI, now tracked as MC15. The Python
  registry has stable IDs and explicit units/kind/temporality; BTLM has strict dual-rail
  framing and sequence evidence; the shared ring supports bounded quiescent snapshots.
  Missing are the generated C signal table, source/session/clock/loss envelope, live SPSC
  ring, transport parity, and claim/PC correlation needed by drivers and kernel replay.
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

Runtime HAL: the allocation-free, append-only direct hook ABI now exists in
`runtime/c/bcir_runtime_channel.{h,c}` with open/claim, offset-based map, submit,
sync/cancel, event, and close plus a bounded resident loopback implementation. Modeled
channels still have no hardware-resident driver; DMA/IRQ remain claims rather than
device I/O, and there is still no `/dev` or BAR-mmap implementation. The first hardware
driver must prove the direct lifetime and teardown contract before a Linux transport is
added.

## 5. The ABI ledger (what is frozen today)

1. **StreamPack** BSPK v1 (frozen) / v2 pipeline / v3 dispatch+channel — dual-rail,
   CRC'd, generation-tagged, semantically verified post-CRC, and exact-consuming (no
   undeclared trailing body bytes).
2. **`bcir_exec` calling convention** — caller-owned memory, `bcir_exec_fn(item, ctx)`
   kernel callback, generation-checked variant (`bcir_exec.h:31-79`).
3. **R12 `TargetABI` matrix** — x86_64/aarch64/riscv64-linux (LP64), x86_64-windows
   (LLP64), i386-linux (ILP32); layout-only by design; MLIR twin op + pass check.
4. **Q8 tier tables** — frozen LE blobs + `#embed` header, drift-gated.
5. **Telemetry frame** BTLM v1 — strict 56-byte DataDNA record batches, reserved-flags
   refusal, CRC, exact single-frame decode, resync-by-magic, and u32 continuity evidence.
   Frozen scope: one externally separated producer stream with an opaque clock; not the
   future driver envelope.
6. **RuntimeChannel v1 direct hook ABI** — allocation-free append-only hook table with
   `abi_version`/`struct_size`, fixed-width values, generation-tagged handles, byte-offset
   maps, and loopback lifecycle coverage. It is an in-process contract, not Linux IPC/UAPI.

Runtime hook compatibility uses `abi_version` plus `struct_size`: v1 rejects an
incompatible major and gates append-only tail fields by size. Artifact formats retain
their frozen-bytes/reject-newer policy.

---

## 6. The gap register — the MC-track (code-backed)

| # | State | Gap / delivered slice | Remaining acceptance boundary |
|---|---|---|---|
| **MC1** | **Partial landed** | Python `bcir-pack dis` and `bcir-pack hexdump`; validation precedes display, spans share codec writers, and CRC-valid trailing bytes are refused on Python/C semantic rails | Add a freestanding C twin, source-module/symbol-section resolution, and per-device command-ISA decoder/disassembler round trips |
| **MC2** | **Partial landed** | In-process `bcir-registry` show/getp/setp; array indexes are explicit, failed validation preserves state, generation overflow is refused, and a poke advances `data_gen` exactly once | Bind permissions, widths, side effects, RuntimeChannel, telemetry audit, device reset/death, and real register access without bypassing R11 |
| **MC3** | **Open** | ROP v2 registry assembly: constants, immediates, offsets/strides, full contracts, deps/events, macros/includes, and BSP logical-name binding | One parser/pretty-printer grammar, hygienic expansion bounds, canonical output, and MC1 shared names |
| **MC4** | **Open** | Carry-as-data and typed predicate contracts | Oracle/MLIR/C twins, checked/wide arithmetic differential, no hidden status register |
| **MC5** | **Deferred** | CFG-aware planning boundary | Build only when a driver fixture needs planned divergence; otherwise retain structured emit-time control and straight-line phased planning |
| **MC6** | **Mostly open** | HAM placement and semantic swap; `Resource.priority` is currently provenance-only and HAM is cost-only | Allocator consumption, D-R3/A2-priced eviction claims, capacity/liveness laws, replay and measured tier data |
| **MC7** | **Open** | StreamPack-level linking, RID/claim relocation, and symbol section | Append-only wire version, collision/overflow checks, deterministic compose, provenance-preserving unlink/list tests |
| **MC8** | **Baseline landed** | Direct append-only RuntimeChannel hook ABI and loopback | First in-process hardware driver must prove ownership, cancellation, saturation, event loss, teardown, reset, and restart before transport work |
| **MC9** | **Open** | Linux/POSIX adapter and UAPI | Version-zero device-fd/ioctl/mmap/poll/errno rail, direct/adapter trace parity, then UART+virtio-blk evidence before v1 freeze |
| **MC10** | **Deferred by native-backend gate** | Target machine description, legalization, ISel, register allocation/spilling, scheduling, hazard tables, encoding, and relaxation | LLVM supplies this for current CPU objects. A native target needs independent semantics, encode/decode identity, and measured G1/G2 justification before implementation |
| **MC11** | **Resident-only** | Native object/archive/link contract: sections, symbols, relocations, COMDAT/TLS where required, deterministic archives, and system-linker interop | Current objects are emitted by LLVM/Clang and `e_machine` checked; BCIR-native emission and relocation corpus are absent. This is distinct from MC7 |
| **MC12** | **Entry slice only** | ABI calling conventions, frames, prologue/epilogue, stack maps, CFI and unwind | The long-mode entry masks interrupts before switching stacks; the ordinary x86 interrupt edge has a fixed 176-byte frame and refuses the five paranoid/IST vectors (#DB/NMI/#DF/#MC/#VC). Reset/mode transition, SMAP/CET/IBT/CR3/PTI/speculation policy, general SysV/Windows/AArch64/RISC-V ABI lowering, `.eh_frame`, exception and signal interoperability are absent |
| **MC13** | **Open** | DWARF/debug/profiling metadata and source-to-claim/PC correlation | Line/type/location tables, symbolization, debugger tests, deterministic paths, and telemetry correlation |
| **MC14** | **Partial resident evidence** | Binary loader/trust plus differential ISA validation | Add bounds/W^X/signature/feature/errata policy, malformed-object corpus, independent assembler/disassembler comparison, simulator and hardware parity for each native/device ISA |
| **MC15** | **Oracle/codec slice landed** | Stable Python signal IDs 1–15 with exact units and explicit metric semantics; strict Python/C BTLM parity; frame continuity counters; long-wrap quiescent ring parsing; Prometheus-text/OTLP/Redfish serialization shapes | Generate a fixed-width C signal table and ID ranges; add source/session/generation/clock/loss driver envelope; build a live SPSC ring with publish/backpressure/death semantics; add claim/PC correlation, hardware providers/transports, and direct/Linux/native trace parity |

The expanded list follows the component boundaries in LLVM's
[code-generator design](https://llvm.org/docs/CodeGenerator.html), the
[ELF generic ABI](https://gabi.xinuos.com/elf/), and the
[DWARF 5 standard](https://dwarfstd.org/dwarf5std.html). Those standards are resident dependencies
today, not claims that BCIR has reimplemented them.

**Reading order into the deep-driver phase:** use the landed MC1/MC2 baselines during MC3 and the
first direct UART driver; prove MC8's hardware lifecycle next; let that evidence shape MC9. MC4/MC6/
MC7 are independent compiler tracks, MC5 stays fixture-gated, MC10–MC14 remain resident-toolchain
contracts unless the native-backend gate opens, and MC15 must reach a version-zero Python/C ABI
before the first D2 driver while remaining unfrozen until UART and virtio-blk evidence exists.

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
