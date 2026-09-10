source_filename = "030-lower-register-binding.solution.ll"

define i64 @lower_register_binding(ptr %binding_table, i32 %logical_id) {
entry:
  %idx = zext i32 %logical_id to i64
  %slot = getelementptr ptr, ptr %binding_table, i64 %idx
  %payload = load ptr, ptr %slot, align 8
  %value = load i64, ptr %payload, align 8
  ret i64 %value
}
