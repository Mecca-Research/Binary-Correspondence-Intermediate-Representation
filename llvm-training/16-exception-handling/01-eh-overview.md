# EH Overview: Unwind Edges, Personalities, and Pads

LLVM exception handling is not a single universal source-language model. It is a
low-level representation that keeps enough structure for optimization and target
code generation while respecting the platform ABI.

## Core concepts

| Concept | Meaning in IR | Why it matters |
| --- | --- | --- |
| Throwing call | A call-like operation that may leave through an exceptional edge | Use `invoke` when the exceptional successor is represented in the function CFG. |
| Normal edge | The successor taken when the call returns normally | The `invoke` result is only available along this edge. |
| Unwind edge | The successor taken when propagation enters this function's EH machinery | The destination must begin with the EH pad shape required by the EH family. |
| Personality function | Function referenced by `personality ptr @...` on the definition | It tells the runtime/codegen which EH ABI rules and clause encodings apply. |
| EH pad | A block-start instruction or terminator that begins an exception region | Pads have strict verifier placement rules and often produce special values. |

## Two common families

### Itanium-style landing pads

Many Unix-like C++ targets use the Itanium ABI family in LLVM IR. The common
shape is:

```llvm
%r = invoke i32 @may_throw()
        to label %normal unwind label %lpad

lpad:
  %exn = landingpad { ptr, i32 }
           cleanup
           catch ptr null
  ; inspect or clean up, then maybe resume
```

The `landingpad` result is an exception package. A cleanup-only path can pass it
to `resume` to continue unwinding.

### Windows EH funclets

Windows EH uses token-valued funclet pads. A `catchswitch` dispatches to handler
blocks, `catchpad` starts a catch handler, and calls inside that handler usually
carry a `"funclet"(token %pad)` operand bundle.

```llvm
catch.dispatch:
  %cs = catchswitch within none [label %catch] unwind to caller

catch:
  %cp = catchpad within %cs [ptr null, i32 0, ptr null]
  call void @handle() [ "funclet"(token %cp) ]
  catchret from %cp to label %done
```

## Instruction roles at a glance

| Instruction | Family | Role |
| --- | --- | --- |
| `invoke` | Both | Call-like terminator with normal and unwind successors. |
| `landingpad` | Itanium | First non-`phi` instruction in an unwind destination; describes catches, filters, and cleanups. |
| `resume` | Itanium | Terminator that continues propagating the landingpad exception package. |
| `catchswitch` | WinEH | EH pad terminator that chooses one or more catch handlers and yields a parent token. |
| `catchpad` | WinEH | Begins a catch funclet and yields the token used by calls and `catchret`. |
| `cleanuppad` | WinEH | Begins a cleanup funclet and yields the token used by calls and `cleanupret`. |
| `catchret` | WinEH | Leaves a catch funclet on the normal path. |
| `cleanupret` | WinEH | Leaves a cleanup funclet, either unwinding onward or returning unwind control to the caller. |

## Review checklist

- Does every EH function that needs a personality have a `personality ptr @...`
  clause on the function definition?
- Does each `invoke` unwind destination start with the correct pad kind?
- Is the `invoke` result used only where the normal edge dominates the use?
- Are pad tokens preserved when cloning or moving calls inside WinEH funclets?
- Are `phi` predecessor lists updated after changing normal or unwind edges?
