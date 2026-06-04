; Run:
;   opt -disable-output -debug-pass-manager -passes='default<O2>' o2-pipeline-inspection.ll
;   opt -S -passes='default<O2>' o2-pipeline-inspection.ll -o /tmp/o2-pipeline-inspection.after.ll
;
; This compact module contains promotable stack state, redundant arithmetic,
; a foldable branch, and a small loop, so default<O2> has visible work for many
; pass-manager debugging commands.

target triple = "x86_64-unknown-linux-gnu"
source_filename = "o2-pipeline-inspection.ll"

define i32 @pipeline_probe(ptr nocapture readonly %a, i64 %n, i32 %seed) {
entry:
  %slot = alloca i32, align 4
  %seed2 = add i32 %seed, 0
  store i32 %seed2, ptr %slot, align 4
  %has.work = icmp ugt i64 %n, 0
  br i1 %has.work, label %loop, label %no.work

no.work:
  %from.slot = load i32, ptr %slot, align 4
  br label %exit

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %acc = phi i32 [ %seed2, %entry ], [ %acc.next, %loop ]
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v0 = load i32, ptr %p, align 4
  %v1 = load i32, ptr %p, align 4
  %pair = add i32 %v0, %v1
  %acc.next = add i32 %acc, %pair
  %next = add nuw i64 %i, 1
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %loop

exit:
  %loop.result = phi i32 [ %from.slot, %no.work ], [ %acc.next, %loop ]
  %always = icmp eq i32 7, 7
  br i1 %always, label %ret, label %dead

ret:
  ret i32 %loop.result

dead:
  ret i32 0
}
