; Minimal Windows EH catch funclet with a funclet operand bundle.
target triple = "x86_64-pc-windows-msvc"

declare void @may_throw()
declare void @handle()
declare i32 @__CxxFrameHandler3(...)

define void @catchswitch_funclet_demo() personality ptr @__CxxFrameHandler3 {
entry:
  invoke void @may_throw()
          to label %done unwind label %catch.dispatch

catch.dispatch:
  %cs = catchswitch within none [label %catch] unwind to caller

catch:
  %cp = catchpad within %cs [ptr null, i32 0, ptr null]
  call void @handle() [ "funclet"(token %cp) ]
  catchret from %cp to label %done

done:
  ret void
}
