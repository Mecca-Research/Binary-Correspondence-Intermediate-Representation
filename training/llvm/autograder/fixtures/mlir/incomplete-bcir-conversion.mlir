// EXPECTED-FAILURE: the stock conversion pipeline lowers func/arith but has no
// BCIR conversion pattern, so bcir.unlowered must remain and fail the registry's
// illegal-operation elimination check.
module {
  func.func @incomplete_conversion(%arg0: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %sum = arith.addi %arg0, %c1 : i32
    %result = "bcir.unlowered"(%sum) : (i32) -> i32
    return %result : i32
  }
}
