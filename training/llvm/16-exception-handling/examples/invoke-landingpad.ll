; Minimal Itanium-style exceptional edge.
target triple = "x86_64-unknown-linux-gnu"

declare void @may_throw()
declare i32 @__gxx_personality_v0(...)

define i32 @invoke_landingpad_demo() personality ptr @__gxx_personality_v0 {
entry:
  invoke void @may_throw()
          to label %ok unwind label %lpad

ok:
  ret i32 0

lpad:
  %lp = landingpad { ptr, i32 }
          cleanup
          catch ptr null
  ret i32 1
}
