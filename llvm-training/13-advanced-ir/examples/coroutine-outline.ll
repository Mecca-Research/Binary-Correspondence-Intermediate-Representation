target triple = "x86_64-unknown-linux-gnu"

; Minimal presplit coroutine outline.
; This is a teaching sketch for token flow, not a complete frontend output.
; Coroutine passes such as coro-early/coro-split/coro-cleanup normally lower it
; before backend code generation.

source_filename = "coroutine-outline.ll"

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare i8 @llvm.coro.suspend(token, i1)

; llvm.coro.end is intentionally not declared in this version-neutral sketch.
; Its intrinsic signature is release-sensitive (including an i1-to-void return
; change in LLVM 22), so a standalone known-good fixture cannot spell one call
; that verifies across every supported LLVM major. Frontend-emitted IR should use
; the declaration for the LLVM version that will consume it.

define ptr @coroutine_outline(i1 %finish_immediately) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %hdl = call noalias ptr @llvm.coro.begin(token %id, ptr null)
  br i1 %finish_immediately, label %cleanup, label %yield

yield:
  %state = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %state, label %suspended [
    i8 0, label %resume
    i8 1, label %cleanup
  ]

resume:
  br label %cleanup

cleanup:
  br label %suspended

suspended:
  ret ptr %hdl
}
