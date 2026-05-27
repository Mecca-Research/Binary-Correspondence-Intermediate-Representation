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

define <8 x i32> @bcir.op.ggg.load.v8i32.ref(ptr %base, ptr %indices) {
entry:
  %i0p = getelementptr i32, ptr %indices, i64 0
  %i0 = load i32, ptr %i0p, align 4
  %p0 = getelementptr i32, ptr %base, i32 %i0
  %v0 = load i32, ptr %p0, align 4
  %r0 = insertelement <8 x i32> zeroinitializer, i32 %v0, i32 0

  %i1p = getelementptr i32, ptr %indices, i64 1
  %i1 = load i32, ptr %i1p, align 4
  %p1 = getelementptr i32, ptr %base, i32 %i1
  %v1 = load i32, ptr %p1, align 4
  %r1 = insertelement <8 x i32> %r0, i32 %v1, i32 1

  %i2p = getelementptr i32, ptr %indices, i64 2
  %i2 = load i32, ptr %i2p, align 4
  %p2 = getelementptr i32, ptr %base, i32 %i2
  %v2 = load i32, ptr %p2, align 4
  %r2 = insertelement <8 x i32> %r1, i32 %v2, i32 2

  %i3p = getelementptr i32, ptr %indices, i64 3
  %i3 = load i32, ptr %i3p, align 4
  %p3 = getelementptr i32, ptr %base, i32 %i3
  %v3 = load i32, ptr %p3, align 4
  %r3 = insertelement <8 x i32> %r2, i32 %v3, i32 3

  %i4p = getelementptr i32, ptr %indices, i64 4
  %i4 = load i32, ptr %i4p, align 4
  %p4 = getelementptr i32, ptr %base, i32 %i4
  %v4 = load i32, ptr %p4, align 4
  %r4 = insertelement <8 x i32> %r3, i32 %v4, i32 4

  %i5p = getelementptr i32, ptr %indices, i64 5
  %i5 = load i32, ptr %i5p, align 4
  %p5 = getelementptr i32, ptr %base, i32 %i5
  %v5 = load i32, ptr %p5, align 4
  %r5 = insertelement <8 x i32> %r4, i32 %v5, i32 5

  %i6p = getelementptr i32, ptr %indices, i64 6
  %i6 = load i32, ptr %i6p, align 4
  %p6 = getelementptr i32, ptr %base, i32 %i6
  %v6 = load i32, ptr %p6, align 4
  %r6 = insertelement <8 x i32> %r5, i32 %v6, i32 6

  %i7p = getelementptr i32, ptr %indices, i64 7
  %i7 = load i32, ptr %i7p, align 4
  %p7 = getelementptr i32, ptr %base, i32 %i7
  %v7 = load i32, ptr %p7, align 4
  %r7 = insertelement <8 x i32> %r6, i32 %v7, i32 7
  ret <8 x i32> %r7
}

define void @bcir.op.ggg.store.v8i32.ref(ptr %base, ptr %indices, <8 x i32> %value) {
entry:
  %i0 = load i32, ptr %indices, align 4
  %v0 = extractelement <8 x i32> %value, i32 0
  %p0 = getelementptr i32, ptr %base, i32 %i0
  store i32 %v0, ptr %p0, align 4

  %i1p = getelementptr i32, ptr %indices, i64 1
  %i1 = load i32, ptr %i1p, align 4
  %v1 = extractelement <8 x i32> %value, i32 1
  %p1 = getelementptr i32, ptr %base, i32 %i1
  store i32 %v1, ptr %p1, align 4

  %i2p = getelementptr i32, ptr %indices, i64 2
  %i2 = load i32, ptr %i2p, align 4
  %v2 = extractelement <8 x i32> %value, i32 2
  %p2 = getelementptr i32, ptr %base, i32 %i2
  store i32 %v2, ptr %p2, align 4

  %i3p = getelementptr i32, ptr %indices, i64 3
  %i3 = load i32, ptr %i3p, align 4
  %v3 = extractelement <8 x i32> %value, i32 3
  %p3 = getelementptr i32, ptr %base, i32 %i3
  store i32 %v3, ptr %p3, align 4

  %i4p = getelementptr i32, ptr %indices, i64 4
  %i4 = load i32, ptr %i4p, align 4
  %v4 = extractelement <8 x i32> %value, i32 4
  %p4 = getelementptr i32, ptr %base, i32 %i4
  store i32 %v4, ptr %p4, align 4

  %i5p = getelementptr i32, ptr %indices, i64 5
  %i5 = load i32, ptr %i5p, align 4
  %v5 = extractelement <8 x i32> %value, i32 5
  %p5 = getelementptr i32, ptr %base, i32 %i5
  store i32 %v5, ptr %p5, align 4

  %i6p = getelementptr i32, ptr %indices, i64 6
  %i6 = load i32, ptr %i6p, align 4
  %v6 = extractelement <8 x i32> %value, i32 6
  %p6 = getelementptr i32, ptr %base, i32 %i6
  store i32 %v6, ptr %p6, align 4

  %i7p = getelementptr i32, ptr %indices, i64 7
  %i7 = load i32, ptr %i7p, align 4
  %v7 = extractelement <8 x i32> %value, i32 7
  %p7 = getelementptr i32, ptr %base, i32 %i7
  store i32 %v7, ptr %p7, align 4
  ret void
}
