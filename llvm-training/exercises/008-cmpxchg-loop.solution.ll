; Exercise 008: cmpxchg increment loop.
; Verification: llvm-as -disable-output llvm-training/exercises/008-cmpxchg-loop.solution.ll

source_filename = "008-cmpxchg-loop.solution.ll"

define i32 @atomic_increment(ptr %addr) {
entry:
  %initial = load atomic i32, ptr %addr monotonic, align 4
  br label %loop

loop:
  %expected = phi i32 [ %initial, %entry ], [ %observed, %loop ]
  %desired = add i32 %expected, 1
  %result = cmpxchg ptr %addr, i32 %expected, i32 %desired acq_rel monotonic, align 4
  %observed = extractvalue { i32, i1 } %result, 0
  %success = extractvalue { i32, i1 } %result, 1
  br i1 %success, label %done, label %loop

done:
  ret i32 %desired
}
