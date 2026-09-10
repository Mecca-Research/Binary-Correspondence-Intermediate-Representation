# Cleanups and Resuming Exception Propagation

Cleanup paths run side effects such as destructors, unlocks, or temporary object
release. In LLVM IR, the way a cleanup leaves depends on the EH family.

## Itanium cleanup plus `resume`

A landing pad with a `cleanup` clause can run local cleanup code and then resume
propagation:

```llvm
lpad:
  %lp = landingpad { ptr, i32 }
        cleanup
  call void @destroy()
  resume { ptr, i32 } %lp
```

The `resume` value is the landingpad exception package. This preserves the
exception identity and selector data for outer handlers.

## WinEH cleanup plus `cleanupret`

WinEH cleanups use a token-valued `cleanuppad`:

```llvm
cleanup:
  %cp = cleanuppad within none []
  call void @destroy() [ "funclet"(token %cp) ]
  cleanupret from %cp unwind to caller
```

`cleanupret` names the cleanup token and states where unwinding continues. If the
cleanup may call code that itself can unwind and that unwind is represented, use
an `invoke` inside the funclet and include the active `"funclet"` operand bundle.

## Safe transformation rules

- Keep cleanup side effects on every path that previously executed them.
- Preserve `landingpad` packages passed to `resume`.
- Preserve funclet tokens and `"funclet"` operand bundles on calls inside WinEH
  pads.
- Do not merge cleanups from different EH families; the verifier and backend
  expect ABI-specific shapes.
- When simplifying CFGs, remember that EH pads have stricter placement rules than
  ordinary blocks.

## Example-driven reading

- [`examples/cleanup-resume.ll`](examples/cleanup-resume.ll) shows an
  Itanium-style cleanup that resumes propagation.
- [`examples/catchswitch-funclet.ll`](examples/catchswitch-funclet.ll) shows the
  token and operand-bundle pattern used by a WinEH catch funclet.
