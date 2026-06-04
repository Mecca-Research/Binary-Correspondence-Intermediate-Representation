; Before vectorization: scalar loop with a guard around each load.
source_filename = "predicated-load-before.c"

define void @predicated_load_before(ptr %dst, ptr %src, i32 %n, i32 %limit) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %latch ]
  %in.range = icmp slt i32 %i, %limit
  br i1 %in.range, label %load, label %latch

load:
  %p = getelementptr inbounds i32, ptr %src, i32 %i
  %v = load i32, ptr %p, align 4
  %q = getelementptr inbounds i32, ptr %dst, i32 %i
  store i32 %v, ptr %q, align 4
  br label %latch

latch:
  %next = add nuw i32 %i, 1
  %done = icmp eq i32 %next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
