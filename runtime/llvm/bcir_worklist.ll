source_filename = "bcir_worklist.ll"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = ""

%bcir.claim = type { i64, [4 x i32], [4 x i32], i64, [2 x i64] }

declare void @bcir.gem.execute_claim(ptr, ptr, ptr)

define void @bcir.gem.execute_worklist(ptr %ctx, ptr %claims, i64 %claim_count, ptr %registry_table) {
entry:
  br label %loop

loop:
  %i = phi i64 [0, %entry], [%next, %body]
  %done = icmp uge i64 %i, %claim_count
  br i1 %done, label %exit, label %body

body:
  %claim = getelementptr inbounds %bcir.claim, ptr %claims, i64 %i
  call void @bcir.gem.execute_claim(ptr %ctx, ptr %claim, ptr %registry_table)
  %next = add i64 %i, 1
  br label %loop

exit:
  ret void
}
