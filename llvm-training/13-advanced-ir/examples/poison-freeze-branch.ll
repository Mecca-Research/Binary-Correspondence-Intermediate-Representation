; Freeze makes a poison-producing comparison safe to branch on.
source_filename = "poison-freeze-branch.c"

define i32 @freeze_before_branch(i32 %x) {
entry:
  %maybe.poison = add nsw i32 %x, 2147483647
  %stable = freeze i32 %maybe.poison
  %is.zero = icmp eq i32 %stable, 0
  br i1 %is.zero, label %zero, label %nonzero

zero:
  ret i32 0

nonzero:
  ret i32 1
}
