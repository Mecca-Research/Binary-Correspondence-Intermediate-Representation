; Exercise 006: GEP into an array of structs.
; Verification: llvm-as -disable-output llvm-training/exercises/006-array-of-structs-gep.solution.ll

source_filename = "006-array-of-structs-gep.solution.ll"

%Entry = type { i32, i64 }

define i64 @load_entry_value(ptr %base, i64 %index) {
entry:
  %field_ptr = getelementptr inbounds %Entry, ptr %base, i64 %index, i32 1
  %value = load i64, ptr %field_ptr, align 8
  ret i64 %value
}
