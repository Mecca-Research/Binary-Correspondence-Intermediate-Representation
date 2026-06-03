; Exercise 005: Index a struct field with getelementptr.
; Verification: llvm-as -disable-output llvm-training/exercises/005-struct-gep.solution.ll

source_filename = "005-struct-gep.solution.ll"

%Pair = type { i32, i64 }

define i64 @get_second(ptr %pair) {
entry:
  %second_ptr = getelementptr inbounds %Pair, ptr %pair, i32 0, i32 1
  %second = load i64, ptr %second_ptr
  ret i64 %second
}
