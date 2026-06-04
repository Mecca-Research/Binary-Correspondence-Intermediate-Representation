# Solution 035: Missing symbol relocation diagnosis

The IR can verify successfully with only a declaration; verification proves the
call is type-correct, not that a runtime address exists. The unresolved symbol is
a link or JIT materialization problem. Supply the definition by linking a runtime
object/library, registering an absolute symbol with the JIT, adding the process
symbol generator for host symbols, or compiling the runtime module into the same
JIT session.

Check that the symbol spelling, target mangling, calling convention, visibility,
and data layout match the generated object. Also confirm that the runtime object
is added to the correct JITDylib or link unit before the kernel is looked up. If
lowering renamed the function or emitted a private/internal definition in another
module, the relocation will still fail even though the IR looked valid.
