; Calls and returns: direct, indirect, variadic, ABI-attributed, multi-value,
; and a guaranteed tail call.
;
; Companion chapter: training/llvm/05-control-flow/05-call-and-ret.md
;
;   llvm-as training/llvm/05-control-flow/examples/call-and-ret.ll -o /dev/null
;   opt -passes=verify -S training/llvm/05-control-flow/examples/call-and-ret.ll -o /dev/null
;
; Assembly-only: the exported functions take caller-provided arguments, so this
; module is not part of the lli smoke subset.

@.fmt = private unnamed_addr constant [8 x i8] c"n = %d\0A\00"

declare i32 @printf(ptr, ...)
declare void @fill(ptr sret({ i64, i64 }), i32 signext)
declare void @consume_by_value(ptr byval({ i32, i32 }) align 4)

; --- direct call, ordinary return -------------------------------------------

define i32 @add(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b
  ret i32 %sum
}

define i32 @direct(i32 %a, i32 %b) {
entry:
  %sum = call i32 @add(i32 %a, i32 %b)
  ret i32 %sum
}

; --- indirect call: under opaque pointers the type lives at the call site ----

define i32 @indirect(ptr %callee, i32 %x) {
entry:
  %r = call i32 %callee(i32 %x)
  ret i32 %r
}

; --- variadic call: the full function type is mandatory ---------------------

define void @trace(i32 %n) {
entry:
  %ignored = call i32 (ptr, ...) @printf(ptr @.fmt, i32 %n)
  ret void
}

; --- sret and byval: attributes must appear at the call site too ------------

define void @abi_boundary(i32 signext %seed) {
entry:
  %out = alloca { i64, i64 }, align 8
  call void @fill(ptr sret({ i64, i64 }) %out, i32 signext %seed)
  %pair = alloca { i32, i32 }, align 4
  store { i32, i32 } { i32 1, i32 2 }, ptr %pair, align 4
  call void @consume_by_value(ptr byval({ i32, i32 }) align 4 %pair)
  ret void
}

; --- multiple results as a literal struct return ----------------------------

define { i32, i1 } @divmod_checked(i32 %a, i32 %b) {
entry:
  %ok = icmp ne i32 %b, 0
  br i1 %ok, label %divide, label %bail

divide:
  %q = sdiv i32 %a, %b
  br label %join

bail:
  br label %join

join:
  %value = phi i32 [ %q, %divide ], [ 0, %bail ]
  %p0 = insertvalue { i32, i1 } poison, i32 %value, 0
  %p1 = insertvalue { i32, i1 } %p0, i1 %ok, 1
  ret { i32, i1 } %p1
}

define i32 @use_multi_result(i32 %a, i32 %b) {
entry:
  %pair = call { i32, i1 } @divmod_checked(i32 %a, i32 %b)
  %quotient = extractvalue { i32, i1 } %pair, 0
  %valid = extractvalue { i32, i1 } %pair, 1
  %result = select i1 %valid, i32 %quotient, i32 -1
  ret i32 %result
}

; --- musttail: matching signature and convention, immediately followed by ret

define internal i32 @worker(i32 %n) {
entry:
  %doubled = shl i32 %n, 1
  ret i32 %doubled
}

define i32 @trampoline(i32 %n) {
entry:
  %r = musttail call i32 @worker(i32 %n)
  ret i32 %r
}

; --- notail: suppress the optimization on a call the runtime must see -------

define i32 @keep_frame(i32 %n) {
entry:
  %r = notail call i32 @worker(i32 %n)
  ret i32 %r
}
