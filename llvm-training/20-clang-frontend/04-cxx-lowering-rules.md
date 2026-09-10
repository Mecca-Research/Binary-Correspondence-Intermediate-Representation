# Exact C++ Lowering Rules

Claims here are asserted against
[`examples/cxx-object-model.cpp`](examples/cxx-object-model.cpp) by
[`../tools/verify-frontend-lowering.py`](../tools/verify-frontend-lowering.py).
Output is `clang++ 18.1.3`, target `x86_64-unknown-linux-gnu`, Itanium C++ ABI.

## Names: one C++ identifier, several symbols

```llvm
@_Z10total_areaRK5Shape     ; int total_area(const Shape&)
@_Z5scalei                  ; int    scale(int)
@_Z5scaled                  ; double scale(double)
@_Z5twiceIiET_S0_           ; int twice<int>(int)
@_ZN5GuardC1Ev              ; Guard::Guard()
@_ZN5GuardD1Ev              ; Guard::~Guard()
```

Reading Itanium mangling well enough to navigate:

| Fragment | Meaning |
| --- | --- |
| `_Z` | mangled C++ name follows |
| `N ... E` | nested name (namespace/class scope) |
| `10total_area` | length-prefixed identifier |
| `RK5Shape` | reference to const `Shape` |
| `i`, `d`, `v`, `c` | `int`, `double`, `void`, `char` |
| `I ... E` | template arguments |
| `C1`/`C2` | complete/base object constructor |
| `D0`/`D1`/`D2` | deleting/complete/base object destructor |
| `S0_` | back-reference to a previously mangled component |

`llvm-cxxfilt` demangles; `c++filt` is the GNU equivalent. Both are worth having
in a debugging loop — a mangled name in a link error is a *signature* mismatch
report, and reading it tells you which parameter changed.

Two consequences: overloads are unrelated symbols (so an ABI break can be a
silent link success against a stale library), and `extern "C"` opts a function
out of mangling entirely, which is why exactly one overload of a name can be
`extern "C"`.

## Virtual dispatch: a load, an index, an indirect call

```llvm
define dso_local noundef i32 @_Z10total_areaRK5Shape(ptr noundef nonnull align 8
                                                     dereferenceable(8) %s) {
entry:
  %0      = load ptr, ptr %s.addr, align 8
  %vtable = load ptr, ptr %0, align 8              ; the vptr is at offset 0
  %vfn    = getelementptr inbounds ptr, ptr %vtable, i64 2
  %1      = load ptr, ptr %vfn, align 8
  %call   = call noundef i32 %1(ptr noundef nonnull align 8 dereferenceable(8) %0)
  ret i32 %call
}
```

Four facts:

1. **The vptr lives at offset 0** of the most-derived subobject (Itanium ABI).
2. **The slot index is fixed at compile time.** `i64 2` here — slot 0 and 1 are
   the two destructors in this class's layout, so `area()` is slot 2. The
   indices come from the class's vtable layout, which
   `clang++ -Xclang -fdump-vtable-layouts` prints.
3. **`this` is an ordinary first argument.** Nothing about it is special in IR.
4. **The call is indirect**, so it is opaque to most interprocedural analysis
   unless devirtualization proves the dynamic type.

Devirtualization is an *optimization*, not a lowering rule: `-O0` always emits
the indirect form even when the type is obvious. `final` on the class or the
method, and `-fstrict-vtable-pointers`, are what let the optimizer close it.

## Constructors, destructors, and temporaries

The AST shape from [`02-ast-and-sema.md`](02-ast-and-sema.md) becomes calls:

```cpp
int guarded() { return Guard().n; }
```

emits a `Guard::Guard()` call, the member load, then `Guard::~Guard()` — placed
at the end of the full expression, because that is where `ExprWithCleanups`
said it goes. Destructor placement is a frontend decision, and it is a
*semantic* one: moving it is a language-conformance change, not an
optimization.

| Situation | What the frontend emits |
| --- | --- |
| Automatic object leaving scope | destructor call on every exit path, including EH |
| Temporary in a full expression | destructor at the end of that expression |
| Object with a lifetime-extending reference binding | destructor at the reference's scope end |
| Multiple objects in a scope | destructors in reverse construction order |
| Partially constructed array/aggregate throwing | destructors only for the constructed prefix |

Under exceptions, all of that is duplicated onto cleanup paths — which is why
the same source produces far more blocks with `-fexceptions` than with
`-fno-exceptions`. See [`../16-exception-handling/README.md`](../16-exception-handling/README.md).

Note `C1`/`C2` and `D1`/`D2`: the *complete object* and *base object* variants
exist because a base subobject's constructor must skip virtual-base
initialization the most-derived constructor already did. Seeing two nearly
identical symbols is normal, not duplication.

## Templates: instantiations, `linkonce_odr`, and comdats

```llvm
$_Z5twiceIiET_S0_ = comdat any
define linkonce_odr dso_local noundef i32 @_Z5twiceIiET_S0_(i32 noundef %v) comdat {
```

- A template that is never instantiated emits **nothing**.
- Each instantiation emits one symbol, `linkonce_odr`, in a comdat group, so
  the linker keeps exactly one copy across translation units.
- `linkonce_odr` also tells the optimizer the definition may be replaced by an
  identical one — so it may inline it but may not assume its address is unique.
- `extern template` suppresses implicit instantiation; explicit instantiation
  emits a strong definition instead.

The practical consequence for build debugging: an ODR violation (two different
definitions with the same mangled name) is *not* diagnosed. The linker keeps
one arbitrarily and the program's behaviour depends on which.

## References, and what they are not

A `T&` parameter lowers to a `ptr`, usually with `nonnull` and
`dereferenceable(N)`:

```llvm
@_Z11square_areaRK6Square(ptr noundef nonnull align 8 dereferenceable(12) %s)
```

There is no reference type in IR. `dereferenceable(12)` is the frontend telling
the optimizer that 12 bytes are readable — a promise derived from the language
rule that a reference binds to a valid object. A tool that synthesizes a null
"reference" produces IR whose attributes are lies, and the optimizer will use
them.

## Other C++ constructs, briefly

| Construct | IR result |
| --- | --- |
| Non-virtual member function | ordinary function with a leading `this` pointer |
| Static member function | ordinary function, no `this` |
| `new` / `delete` | `call ptr @_Znwm(i64)` / `@_ZdlPv`, plus the constructor/destructor |
| Global with a dynamic initializer | an entry in `@llvm.global_ctors` |
| Function-local `static` | guard variable plus `__cxa_guard_acquire`/`release` |
| `dynamic_cast` | `call ptr @__dynamic_cast(...)` |
| `typeid` | a reference to a `@_ZTI*` type-info object |
| Virtual inheritance | vtable entries holding virtual-base offsets; `this` adjustment |
| Lambda | a closure class, its `operator()`, and possibly a conversion to a function pointer |
| Coroutine | `llvm.coro.*` intrinsics; see [`../13-advanced-ir/03-special-types-and-tokens.md`](../13-advanced-ir/03-special-types-and-tokens.md) |
| `[[no_unique_address]]` | changes record layout; check `-fdump-record-layouts` |

The function-local `static` case is a common surprise: a one-line declaration
becomes a guard variable, two runtime calls, and a branch, because the language
requires thread-safe initialization exactly once.

## Pitfalls

- **Assuming a virtual call is devirtualized.** At `-O0` it never is.
- **Assuming vtable slot indices are stable.** Adding a virtual function
  renumbers slots — an ABI break invisible in the header diff.
- **Comparing mangled names across ABIs.** MSVC mangling is entirely different;
  cross-ABI symbol comparison is meaningless.
- **Expecting one symbol per template.** One per instantiation, per TU, merged
  by comdat.
- **Ignoring the destructor's exception path.** A cleanup exists on the unwind
  edge too, and losing it in an IR transform leaks or double-frees.
- **Treating `dereferenceable` as a check.** It is an assertion, and the
  optimizer will speculate loads on the strength of it.
- **Reasoning about `-fno-exceptions` code from `-fexceptions` IR.** The block
  structure is materially different.

## BCIR notes

- The C++ rail in this repository is deliberately a **narrow orchestration
  seam** above a C ABI, not a place where object-model complexity is allowed to
  spread. The lowering rules above are why: virtual dispatch, RTTI, EH cleanups,
  and comdat merging each introduce a mechanism whose costs and legality are
  hard to price in a cost-governed IR. See
  [`../../docs/languages/CPP_HANDOFF_BOUNDARY.md`](../../docs/languages/CPP_HANDOFF_BOUNDARY.md).
- When a BCIR artifact crosses that seam, it crosses as data over a C ABI —
  handles and offsets — precisely so that neither side depends on the other's
  object layout or symbol mangling.
- `linkonce_odr` and comdat merging interact badly with content-addressed
  artifact identity: two "identical" instantiations that differ in an
  optimization flag are one symbol to the linker and two artifacts to a digest.
  Keep artifact identity on the BCIR side, never on the C++ symbol.

## See also

- [`03-c-lowering-rules.md`](03-c-lowering-rules.md) — the C rules this builds on
- [`../16-exception-handling/README.md`](../16-exception-handling/README.md) — cleanup and unwinding IR
- [`../13-advanced-ir/04-attributes.md`](../13-advanced-ir/04-attributes.md) — `nonnull`, `dereferenceable`, and friends
