; Final LLVM IR snapshot for the MLIR BCIR lowering walkthrough.
; Verifies with: llvm-as bcir-final.ll -o /dev/null
source_filename = "bcir-final.ll"
target triple = "x86_64-unknown-linux-gnu"

%bcir.edge = type { i64, i64, i32 }

define i32 @bcir_edge_weight(ptr readonly %edges, i64 %edge_count, i64 %edge_index) {
entry:
  %in_bounds = icmp ult i64 %edge_index, %edge_count
  br i1 %in_bounds, label %load, label %missing

load:
  %edge_ptr = getelementptr inbounds %bcir.edge, ptr %edges, i64 %edge_index
  %weight_ptr = getelementptr inbounds %bcir.edge, ptr %edge_ptr, i32 0, i32 2
  %weight = load i32, ptr %weight_ptr, align 4
  ret i32 %weight

missing:
  ret i32 -1
}

define i32 @main() {
entry:
  %edges = alloca [2 x %bcir.edge], align 8
  %edge0 = getelementptr inbounds [2 x %bcir.edge], ptr %edges, i32 0, i32 0
  %edge0_src = getelementptr inbounds %bcir.edge, ptr %edge0, i32 0, i32 0
  store i64 7, ptr %edge0_src, align 8
  %edge0_dst = getelementptr inbounds %bcir.edge, ptr %edge0, i32 0, i32 1
  store i64 9, ptr %edge0_dst, align 8
  %edge0_weight = getelementptr inbounds %bcir.edge, ptr %edge0, i32 0, i32 2
  store i32 42, ptr %edge0_weight, align 4
  %result = call i32 @bcir_edge_weight(ptr %edge0, i64 1, i64 0)
  ret i32 %result
}
