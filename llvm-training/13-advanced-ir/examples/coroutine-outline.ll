target triple = "x86_64-unknown-linux-gnu"

; Minimal presplit coroutine outline.
; This is a teaching sketch for token flow, not a complete frontend output.
; Coroutine passes such as coro-early/coro-split/coro-cleanup normally lower it
; before backend code generation.

source_filename = "coroutine-outline.ll"

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare i8 @llvm.coro.suspend(token, i1)
declare i1 @llvm.coro.end(ptr, i1, token)

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
  %done = call i1 @llvm.coro.end(ptr %hdl, i1 false, token none)
  br label %suspended

suspended:
  ret ptr %hdl
}
