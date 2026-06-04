; Function, parameter, and call-site attributes refine optimizer assumptions.
source_filename = "attributes-callsite.c"

declare void @write_i32(ptr nocapture writeonly, i32) nounwind

define void @attribute_demo(ptr noalias nonnull align 4 %p, i32 %v) local_unnamed_addr nounwind {
entry:
  call void @write_i32(ptr nonnull align 4 %p, i32 %v) nounwind
  ret void
}
