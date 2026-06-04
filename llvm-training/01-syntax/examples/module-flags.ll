; Named metadata can attach module-wide flags.
source_filename = "module-flags.c"

!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 2}

define i32 @module_flag_demo() {
entry:
  ret i32 4
}
