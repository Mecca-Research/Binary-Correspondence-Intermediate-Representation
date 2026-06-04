# Solution 036: ORC layer failure identification

Because parsing, verification, IR transforms, and object generation ran, the
first suspect is symbol resolution or the JITDylib/link layer rather than the IR
front end. Check whether the object defines `kernel` with the expected linkage
and mangled name, whether it was added to the same `JITDylib` being queried, and
whether lookup uses the platform mangler.

A compile-layer failure usually reports target-machine or object-emission errors.
An object/link-layer failure usually reports relocation, section, or dependency
materialization errors. A symbol-resolution failure can look like a clean compile
followed by a missing definition. Add dumps of exported object symbols, ORC
materialization-unit names, JITDylib search order, and the exact mangled lookup
name.
