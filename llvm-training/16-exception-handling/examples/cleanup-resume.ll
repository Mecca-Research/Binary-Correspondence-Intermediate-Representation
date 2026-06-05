; Cleanup-only landing pad that keeps unwinding with resume.
target triple = "x86_64-unknown-linux-gnu"

declare void @may_throw()
declare void @destroy()
declare i32 @__gxx_personality_v0(...)

define void @cleanup_resume_demo() personality ptr @__gxx_personality_v0 {
entry:
  invoke void @may_throw()
          to label %done unwind label %lpad

done:
  ret void

lpad:
  %lp = landingpad { ptr, i32 }
          cleanup
  call void @destroy()
  resume { ptr, i32 } %lp
}
