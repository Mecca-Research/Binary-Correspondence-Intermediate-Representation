; Named structs make recursive and shared layouts readable.
source_filename = "named-structs.c"

%Pair = type { i32, i32 }
%Node = type { i32, ptr }

define i32 @sum_pair(ptr %p) {
entry:
  %a.ptr = getelementptr inbounds %Pair, ptr %p, i32 0, i32 0
  %b.ptr = getelementptr inbounds %Pair, ptr %p, i32 0, i32 1
  %a = load i32, ptr %a.ptr, align 4
  %b = load i32, ptr %b.ptr, align 4
  %sum = add i32 %a, %b
  ret i32 %sum
}
