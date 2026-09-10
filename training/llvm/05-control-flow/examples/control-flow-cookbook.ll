; control-flow-cookbook.ll — unconditional branches, conditional branches,
; switch dispatch, and loop phis.
; Assemble/check with: llvm-as control-flow-cookbook.ll -o /dev/null

source_filename = "control-flow-cookbook.ll"

define i32 @control_flow_cookbook(i32 %x, i32 %n) {
entry:
  ; Unconditional branch: entry always jumps to the first real block.
  br label %classify

classify:
  ; Conditional branch: negative inputs skip the switch and loop.
  %is_negative = icmp slt i32 %x, 0
  br i1 %is_negative, label %negative, label %dispatch

negative:
  br label %done

dispatch:
  ; Switch branch: special cases for 0 and 1, default falls into a loop.
  switch i32 %x, label %loop.preheader [
    i32 0, label %zero
    i32 1, label %one
  ]

zero:
  br label %done

one:
  br label %done

loop.preheader:
  br label %loop

loop:
  ; Loop phis merge the initial values with values from the backedge.
  %i = phi i32 [ 0, %loop.preheader ], [ %next, %loop ]
  %acc = phi i32 [ 0, %loop.preheader ], [ %sum, %loop ]
  %sum = add i32 %acc, %i
  %next = add i32 %i, 1
  %keep_going = icmp slt i32 %next, %n
  br i1 %keep_going, label %loop, label %done

done:
  %result = phi i32 [ -1, %negative ], [ 0, %zero ], [ 1, %one ], [ %sum, %loop ]
  ret i32 %result
}
