; Run:
;   opt -passes='print<memoryssa>' memoryssa-alias-shape.ll -disable-output
;   opt -S -passes=gvn memoryssa-alias-shape.ll -o -
;
; The store through %maybe_alias may clobber %slot, while %neighbor has a
; precise GEP shape. MemorySSA organizes the memory def/use chain; alias
; analysis decides which definitions may overlap the final load.

source_filename = "memoryssa-alias-shape.ll"
target triple = "x86_64-unknown-linux-gnu"

define i32 @memoryssa_alias_shape(ptr %base, ptr %maybe_alias, i1 %choose) {
entry:
  %slot = getelementptr inbounds i32, ptr %base, i64 0
  %neighbor = getelementptr inbounds i32, ptr %base, i64 1
  store i32 7, ptr %slot, align 4, !bcir.diag !0
  br i1 %choose, label %may_alias_path, label %neighbor_path

may_alias_path:
  store i32 11, ptr %maybe_alias, align 4, !bcir.diag !1
  br label %join

neighbor_path:
  store i32 13, ptr %neighbor, align 4, !bcir.diag !2
  br label %join

join:
  %observed = load i32, ptr %slot, align 4, !bcir.diag !3
  ret i32 %observed
}

!0 = !{!"bcir.node", i32 10, !"initialize slot"}
!1 = !{!"bcir.node", i32 11, !"unknown alias write"}
!2 = !{!"bcir.node", i32 12, !"neighbor write"}
!3 = !{!"bcir.node", i32 13, !"reload slot"}
