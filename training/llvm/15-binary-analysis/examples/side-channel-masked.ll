; Branchless pair for comparison with side-channel-branchy.ll.
source_filename = "side-channel-masked.c"

define i32 @masked_lookup(ptr %table, i32 %secret) {
entry:
  %p0 = getelementptr inbounds i32, ptr %table, i32 0
  %p1 = getelementptr inbounds i32, ptr %table, i32 1
  %v0 = load i32, ptr %p0, align 4
  %v1 = load i32, ptr %p1, align 4
  %bit = and i32 %secret, 1
  %is.odd = icmp ne i32 %bit, 0
  %result = select i1 %is.odd, i32 %v1, i32 %v0
  ret i32 %result
}
