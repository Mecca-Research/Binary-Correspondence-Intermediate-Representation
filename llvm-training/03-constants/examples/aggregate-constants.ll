; Aggregate constants initialize arrays and structs at module scope.
source_filename = "aggregate-constants.c"

%Header = type { i32, [4 x i8] }

@table = constant [3 x i32] [i32 10, i32 20, i32 30], align 4
@header = constant %Header { i32 3, [4 x i8] c"BCIR" }, align 4

define i32 @read_table_1() {
entry:
  %p = getelementptr inbounds [3 x i32], ptr @table, i32 0, i32 1
  %v = load i32, ptr %p, align 4
  ret i32 %v
}
