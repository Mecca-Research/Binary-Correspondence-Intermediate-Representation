source_filename = "bcir_examples.ll"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = ""

%bcir.execctx = type { ptr, i32, i32, i32, i64, ptr }

declare void @bcir.op.phase.enter(ptr, i32, i32)
declare void @bcir.op.phase.leave(ptr, i32, i32)
declare void @bcir.op.barrier(ptr, i32)
declare <8 x i32> @bcir.op.load.v8i32(ptr, i64)
declare void @bcir.op.store.v8i32(ptr, i64, <8 x i32>)

define void @bcir.example.ux_vec_add(ptr %ctx, ptr %a, ptr %b, ptr %c) {
entry:
  call void @bcir.op.phase.enter(ptr %ctx, i32 0, i32 0)
  %va = call <8 x i32> @bcir.op.load.v8i32(ptr %a, i64 0)
  %vb = call <8 x i32> @bcir.op.load.v8i32(ptr %b, i64 0)
  call void @bcir.op.phase.leave(ptr %ctx, i32 0, i32 0)

  call void @bcir.op.phase.enter(ptr %ctx, i32 0, i32 1)
  %vc = add <8 x i32> %va, %vb
  call void @bcir.op.phase.leave(ptr %ctx, i32 0, i32 1)

  call void @bcir.op.phase.enter(ptr %ctx, i32 0, i32 2)
  call void @bcir.op.store.v8i32(ptr %c, i64 0, <8 x i32> %vc)
  call void @bcir.op.barrier(ptr %ctx, i32 2)
  call void @bcir.op.phase.leave(ptr %ctx, i32 0, i32 2)
  ret void
}
