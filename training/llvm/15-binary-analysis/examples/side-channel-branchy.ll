; Secret-dependent branch: useful as a review target for side-channel analysis.
source_filename = "side-channel-branchy.c"

define i32 @branchy_lookup(ptr %table, i32 %secret) {
entry:
  %bit = and i32 %secret, 1
  %take = icmp ne i32 %bit, 0
  br i1 %take, label %odd, label %even

even:
  %p0 = getelementptr inbounds i32, ptr %table, i32 0
  %v0 = load i32, ptr %p0, align 4
  ret i32 %v0

odd:
  %p1 = getelementptr inbounds i32, ptr %table, i32 1
  %v1 = load i32, ptr %p1, align 4
  ret i32 %v1
}
