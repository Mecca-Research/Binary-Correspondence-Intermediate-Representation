; indirectbr can jump only to labels listed in its destination set.
source_filename = "indirectbr-table.c"

define i32 @jump_by_address(i1 %choose) {
entry:
  %target = select i1 %choose, ptr blockaddress(@jump_by_address, %left), ptr blockaddress(@jump_by_address, %right)
  indirectbr ptr %target, [label %left, label %right]

left:
  ret i32 1

right:
  ret i32 2
}
