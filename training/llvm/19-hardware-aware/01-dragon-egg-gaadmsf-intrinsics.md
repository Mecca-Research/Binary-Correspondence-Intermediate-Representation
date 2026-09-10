# Dragon Egg and GAADMSF Intrinsic Boundaries

## A practical Dragon Egg operation taxonomy

Use this taxonomy during lowering reviews. It classifies who must observe an
operation; it does not add instruction classes to LLVM itself.

| Class | Meaning | First-choice lowering |
|---|---|---|
| **Portable compute** | Ordinary arithmetic, comparisons, address formation, or control flow | Generic LLVM instructions |
| **Advisory policy** | Affinity, reuse, placement, confidence, or preferred scheduling | Metadata or an explicit profile record |
| **Runtime-owned action** | Queue submission, device protocol, firmware service, or dynamically selected implementation | Narrow runtime call |
| **Selection-visible operation** | A hardware-shaped action that instruction selection/legalization must recognize | Registered intrinsic followed by target pseudo/instruction |
| **Machine-constrained operation** | Register-bank, fixed-register, hazard, packet, or post-RA expansion requirements | Backend and MIR customization |

A Dragon Egg operation may move between runtime-owned and selection-visible
forms as a target matures. Keep the ABI fallback stable so a generic JIT or AOT
pipeline can still execute the operation.

## GAADMSF operation categories

GAADMSF graph-aware operations commonly include:

- graph-indexed loads, stores, copies, gathers, and scatters;
- reductions and accumulator updates;
- programming pulses that configure a hardware execution context;
- flow execution that launches or advances a prepared graph fragment;
- scheduling, locality, and hierarchical-memory advice.

Lower graph addressing to explicit GEPs and loops when it is portable. Keep a
programming pulse or flow execution as a call/intrinsic when scalar expansion
would erase sequencing, immediate modes, or backend-visible hazards.

## Intrinsic versus runtime call versus metadata

Ask these questions in order:

1. **Can ordinary IR express the complete behavior?** Use it.
2. **Can a library or runtime perform the behavior without special instruction
   selection?** Use a call with explicit state and status.
3. **Must codegen see one operation?** Define and register an intrinsic, then add
   legalization/selection and a fallback.
4. **Is the information only a preference?** Attach metadata and define what
   happens when it is ignored.
5. **Does the choice depend on physical registers or final machine layout?**
   Consume any hints in a backend/MIR pass rather than pretending SSA names are
   physical resources.

The examples declare `llvm.bcir.*` design shapes to teach the boundary. Such a
name is production-ready only after LLVM intrinsic registration and backend
support; otherwise a pre-codegen pass must rewrite it to a runtime ABI.

## Memory effects and optimization

A real intrinsic definition should declare memory effects accurately. A
programming pulse that only changes inaccessible device state is different from
a flow that reads and writes buffers. Overstating effects blocks optimization;
understating them permits illegal motion or deletion. Metadata cannot repair an
incorrect memory-effects contract.

## Related material

- [`../bcir-mapping/07-gaadmsf-operations.md`](../bcir-mapping/07-gaadmsf-operations.md)
- [`../bcir-mapping/08-dragon-egg-operations.md`](../bcir-mapping/08-dragon-egg-operations.md)
- [`../12-backend-jit/06-custom-bcir-intrinsics.md`](../12-backend-jit/06-custom-bcir-intrinsics.md)
- [`../reference/intrinsics-quickref.md`](../reference/intrinsics-quickref.md)
