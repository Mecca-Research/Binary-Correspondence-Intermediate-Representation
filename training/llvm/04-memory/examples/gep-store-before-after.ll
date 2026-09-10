; GEP computes an address; store writes through the computed pointer.
source_filename = "gep-store-before-after.c"

%Point = type { i32, i32 }

define void @set_point_y(ptr %p, i32 %y) {
entry:
  %y.addr = getelementptr inbounds %Point, ptr %p, i32 0, i32 1
  store i32 %y, ptr %y.addr, align 4
  ret void
}

define i32 @read_point_y(ptr %p) {
entry:
  %y.addr = getelementptr inbounds %Point, ptr %p, i32 0, i32 1
  %y = load i32, ptr %y.addr, align 4
  ret i32 %y
}
