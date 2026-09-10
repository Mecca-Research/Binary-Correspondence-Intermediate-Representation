// Illustrative source-level BCIR register prelock and HAM hint lowering input.
// The `bcir` dialect is unregistered in this repository; validate syntax with
// mlir-opt --allow-unregistered-dialect.

module attributes {bcir.example = "register-prelock-ham"} {
  "bcir.graph"() ({
  ^entry:
    %reg = "bcir.register_prelock"() {
      bank = "gpr",
      logical = "r7",
      required = true,
      claim_id = 1204 : i64
    } : () -> !bcir.register<bank = "gpr">

    %ptr = "bcir.lookup_resource"(%reg) {
      table = "thread_local_resource_table"
    } : (!bcir.register<bank = "gpr">) -> !bcir.ptr<mutable>

    %hint = "bcir.ham_hint"(%ptr) {
      policy = "prefetch",
      locality = 3 : i32,
      rw = "read",
      claim_id = 1204 : i64
    } : (!bcir.ptr<mutable>) -> !bcir.ham_hint

    "bcir.gaadmsf.transfer"(%ptr, %hint) {
      bytes = 64 : i64,
      mode = "read-mostly",
      diagnostic = "prelocked register feeds HAM prefetch"
    } : (!bcir.ptr<mutable>, !bcir.ham_hint) -> ()
    "bcir.yield"() : () -> ()
  }) : () -> ()
}
