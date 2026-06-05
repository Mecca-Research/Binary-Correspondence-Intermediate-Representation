; Vtable-like type metadata and CFI-style checked dispatch.
;
; Assemble with:
;   llvm-as llvm-training/06-metadata/examples/type-metadata-cfi.ll -o /tmp/type-metadata-cfi.bc

%Object = type { ptr, i32 }

@vtable.Widget = internal constant { ptr } { ptr @widget_do }, !type !0
@good_object = internal global %Object { ptr @vtable.Widget, i32 7 }, align 8

; A deliberately untagged table. Loading this as an object's vtable pointer makes
; the type test fail before the indirect call is reached.
@vtable.NotAWidget = internal constant { ptr } { ptr @not_widget_do }
@bad_object = internal global %Object { ptr @vtable.NotAWidget, i32 9 }, align 8

declare i1 @llvm.type.test(ptr, metadata) nounwind readnone
declare void @llvm.trap() cold noreturn nounwind

define internal i32 @widget_do(ptr %self) {
entry:
  %value.ptr = getelementptr inbounds %Object, ptr %self, i32 0, i32 1
  %value = load i32, ptr %value.ptr, align 4
  ret i32 %value
}

define internal i32 @not_widget_do(ptr %self) {
entry:
  ret i32 -1
}

define i32 @dispatch_widget(ptr %obj) {
entry:
  %vptr.slot = getelementptr inbounds %Object, ptr %obj, i32 0, i32 0
  %vptr = load ptr, ptr %vptr.slot, align 8
  %ok = call i1 @llvm.type.test(ptr %vptr, metadata !"typeid.Widget")
  br i1 %ok, label %checked, label %trap

checked:
  %fn.slot = getelementptr inbounds { ptr }, ptr %vptr, i32 0, i32 0
  %fn = load ptr, ptr %fn.slot, align 8
  %result = call i32 %fn(ptr %obj)
  ret i32 %result

trap:
  call void @llvm.trap()
  unreachable
}

define i32 @demo_good() {
entry:
  %result = call i32 @dispatch_widget(ptr @good_object)
  ret i32 %result
}

!0 = !{i64 0, !"typeid.Widget"}
