; memory-cookbook.ll — alloca, load, store, globals, and address spaces.
; Assemble/check with: llvm-as memory-cookbook.ll -o /dev/null

source_filename = "memory-cookbook.ll"

@global_counter = global i32 0, align 4
@global_limit = constant i32 100, align 4

; Non-zero address spaces are target-defined, but the IR syntax is portable.
@device_counter = addrspace(1) global i32 7, align 4

define i32 @stack_slot_demo(i32 %input) {
entry:
  %slot = alloca i32, align 4
  store i32 %input, ptr %slot, align 4
  %loaded = load i32, ptr %slot, align 4
  %result = add i32 %loaded, 1
  ret i32 %result
}

define i32 @global_load_store_demo(i32 %delta) {
entry:
  %old = load i32, ptr @global_counter, align 4
  %new = add i32 %old, %delta
  store i32 %new, ptr @global_counter, align 4
  ret i32 %new
}

define i32 @read_global_constant_demo() {
entry:
  %limit = load i32, ptr @global_limit, align 4
  ret i32 %limit
}

define i32 @address_space_load_demo() {
entry:
  %value = load i32, ptr addrspace(1) @device_counter, align 4
  ret i32 %value
}

define void @address_space_store_demo(i32 %value) {
entry:
  store i32 %value, ptr addrspace(1) @device_counter, align 4
  ret void
}
