; Grammar focus: instruction metadata attachment and metadata node syntax.
source_filename = "metadata-attachments.ll"

define i32 @metadata_attachment(ptr %p) {
entry:
  %v = load i32, ptr %p, align 4, !annotation !0
  ret i32 %v
}

!0 = !{!"syntax-only tag"}
