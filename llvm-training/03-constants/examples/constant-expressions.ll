; Constant expressions can compute addresses for global initializers.
source_filename = "constant-expressions.c"

@bytes = constant [4 x i8] c"abc\00", align 1
@byte1 = global ptr getelementptr inbounds ([4 x i8], ptr @bytes, i32 0, i32 1), align 8

define i8 @load_constant_expr_pointer() {
entry:
  %p = load ptr, ptr @byte1, align 8
  %ch = load i8, ptr %p, align 1
  ret i8 %ch
}
