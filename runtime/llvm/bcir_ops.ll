source_filename = "bcir_ops.ll"
target triple = "unknown-unknown-unknown"
target datalayout = ""

%bcir.execctx = type { ptr, i32, i32, i32, i64, ptr }

define void @bcir.op.phase.enter(ptr %ctx, i32 %epoch, i32 %phase) {
entry:
  %epoch_p = getelementptr inbounds %bcir.execctx, ptr %ctx, i32 0, i32 2
  %phase_p = getelementptr inbounds %bcir.execctx, ptr %ctx, i32 0, i32 1
  store i32 %epoch, ptr %epoch_p, align 4
  store i32 %phase, ptr %phase_p, align 4
  ret void
}

define void @bcir.op.phase.leave(ptr %ctx, i32 %epoch, i32 %phase) {
entry:
  ret void
}

define void @bcir.op.barrier(ptr %ctx, i32 %scope) {
entry:
  %scope_p = getelementptr inbounds %bcir.execctx, ptr %ctx, i32 0, i32 3
  store i32 %scope, ptr %scope_p, align 4
  fence seq_cst
  ret void
}

define void @bcir.op.prov.note(ptr %ctx, i64 %claim_id, i64 %src_hash, i64 %trace_hash, i32 %opcode, i32 %lane, i32 %epoch, i32 %phase) {
entry:
  ret void
}

define i32 @bcir.op.load.i32(ptr %base, i64 %offset) alwaysinline {
entry:
  %p = getelementptr i8, ptr %base, i64 %offset
  %v = load i32, ptr %p, align 4
  ret i32 %v
}

define void @bcir.op.store.i32(ptr %base, i64 %offset, i32 %value) alwaysinline {
entry:
  %p = getelementptr i8, ptr %base, i64 %offset
  store i32 %value, ptr %p, align 4
  ret void
}

define <8 x i32> @bcir.op.load.v8i32(ptr %base, i64 %offset) alwaysinline {
entry:
  %p = getelementptr i8, ptr %base, i64 %offset
  %v = load <8 x i32>, ptr %p, align 32
  ret <8 x i32> %v
}

define void @bcir.op.store.v8i32(ptr %base, i64 %offset, <8 x i32> %value) alwaysinline {
entry:
  %p = getelementptr i8, ptr %base, i64 %offset
  store <8 x i32> %value, ptr %p, align 32
  ret void
}

define i32 @bcir.op.atomic.add.i32(ptr %base, i64 %offset, i32 %delta) alwaysinline {
entry:
  %p = getelementptr i8, ptr %base, i64 %offset
  %old = atomicrmw add ptr %p, i32 %delta seq_cst, align 4
  ret i32 %old
}

define i32 @bcir.op.atomic.sub.i32(ptr %base, i64 %offset, i32 %delta) alwaysinline {
entry:
  %p = getelementptr i8, ptr %base, i64 %offset
  %old = atomicrmw sub ptr %p, i32 %delta seq_cst, align 4
  ret i32 %old
}

define i32 @bcir.op.atomic.xor.i32(ptr %base, i64 %offset, i32 %delta) alwaysinline {
entry:
  %p = getelementptr i8, ptr %base, i64 %offset
  %old = atomicrmw xor ptr %p, i32 %delta seq_cst, align 4
  ret i32 %old
}

define { i32, i1 } @bcir.op.cmpxchg.i32(ptr %base, i64 %offset, i32 %expected, i32 %desired) alwaysinline {
entry:
  %p = getelementptr i8, ptr %base, i64 %offset
  %pair = cmpxchg ptr %p, i32 %expected, i32 %desired seq_cst monotonic, align 4
  ret { i32, i1 } %pair
}

define <8 x i32> @bcir.op.ux.add.v8i32(<8 x i32> %a, <8 x i32> %b) alwaysinline {
entry:
  %r = add <8 x i32> %a, %b
  ret <8 x i32> %r
}

define <8 x i32> @bcir.op.ux.shuffle.v8i32(<8 x i32> %a, <8 x i32> %b) alwaysinline {
entry:
  %r = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  ret <8 x i32> %r
}

define <8 x i32> @bcir.op.ggg.load.v8i32(ptr %base, ptr %indices, i64 %count) {
entry:
  ret <8 x i32> zeroinitializer
}

define void @bcir.op.ggg.store.v8i32(ptr %base, ptr %indices, <8 x i32> %value, i64 %count) {
entry:
  ret void
}

define void @bcir.op.t.macc.f32(ptr %a, ptr %b, ptr %c, i64 %m, i64 %n, i64 %k) {
entry:
  ret void
}
