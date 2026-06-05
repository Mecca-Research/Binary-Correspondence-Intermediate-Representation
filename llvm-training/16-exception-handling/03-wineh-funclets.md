# Windows EH Funclets: `catchswitch`, `catchpad`, and `cleanuppad`

Windows EH is represented as explicit funclets. A funclet is a token-owned region
that code generation can outline or arrange according to the Windows unwinding
ABI.

## `catchswitch`

`catchswitch` is both an EH pad and a terminator. It dispatches to one or more
catch handler labels and either unwinds to another EH pad or to the caller.

```llvm
%cs = catchswitch within none [label %catch] unwind to caller
```

Rules to remember:

- It produces a `token` consumed by child `catchpad` instructions.
- It must be the only non-`phi` instruction in its block.
- Its `within` operand is `none` for a top-level dispatch or a parent pad token
  for nested EH.

## `catchpad` and `catchret`

A catch handler starts with `catchpad` and leaves normally with `catchret`:

```llvm
catch:
  %cp = catchpad within %cs [ptr null, i32 0, ptr null]
  call void @handle() [ "funclet"(token %cp) ]
  catchret from %cp to label %done
```

The bracket arguments are interpreted by the personality. The important CFG fact
for IR review is that `%cp` is the active funclet token. `catchret` exits that
same token and transfers control to a normal successor.

## `cleanuppad` and `cleanupret`

A cleanup funclet starts with `cleanuppad` and exits with `cleanupret`:

```llvm
cleanup:
  %cleanup = cleanuppad within none []
  call void @destroy() [ "funclet"(token %cleanup) ]
  cleanupret from %cleanup unwind to caller
```

`cleanupret` can unwind to another EH pad or `unwind to caller`. It is the WinEH
counterpart to cleanup work followed by continued propagation.

## Operand bundle interaction: `"funclet"`

Because [`13-advanced-ir/07-operand-bundles.md`](../13-advanced-ir/07-operand-bundles.md)
exists, use it as the operand-bundle background for this chapter. The key EH
interaction is:

```llvm
call void @handle() [ "funclet"(token %cp) ]
```

The bundle marks the call as belonging to the active funclet token. This matters
when a pass clones, sinks, hoists, or replaces calls: dropping the bundle can make
the call appear outside its EH region, and moving the call to another funclet
requires updating the token to the destination funclet's pad token.

## Review checklist

- Does each handler block begin with the correct pad instruction?
- Do `catchret` and `cleanupret` name the same token produced by the active pad?
- Do calls inside funclets preserve `"funclet"(token %pad)` bundles?
- Does nested EH use the correct parent token in `within` operands?
