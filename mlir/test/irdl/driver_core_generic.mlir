// RUN: mlir-opt --irdl-file=%S/../../irdl/bcir.irdl.mlir %s | FileCheck %s
//
// G32: the driver-subset core in generic syntax, round-tripped through the IRDL projection on
// stock mlir-opt. The lowered accessors take a resolved integer address of any width; the
// register, MSR and descriptor operations the exact widths ODS declares; port I/O its two
// positional forms (`in` port -> value, `out` value, port). Attributes the
// compiled dialect spells as `#bcir...` enums ride here as plain strings: the projection is
// structural, and stock mlir-opt has no BCIR attribute parser.

func.func @driver(%addr: i64, %addr32: i32, %v32: i32, %v64: i64, %sel: i16, %port: i16) {
  %r = "bcir.volatile_load"(%addr) : (i64) -> i32
  "bcir.volatile_store"(%v32, %addr) : (i32, i64) -> ()
  %old = "bcir.atomic_rmw"(%addr, %v32) {kind = "add", ordering = "seq_cst"} : (i64, i32) -> i32
  %cas = "bcir.atomic_cas"(%addr32, %v32, %r) {ordering = "acq_rel", weak} : (i32, i32, i32) -> i32
  %cr = "bcir.creg_read"() {reg = "cr3"} : () -> i64
  "bcir.creg_write"(%cr) {reg = "cr3"} : (i64) -> ()
  %msr = "bcir.msr_read"(%v32) : (i32) -> i64
  "bcir.msr_write"(%v32, %msr) : (i32, i64) -> ()
  "bcir.descriptor_load"(%v64) {table = "gdt"} : (i64) -> ()
  "bcir.segment_reload"(%sel, %sel) : (i16, i16) -> ()
  "bcir.task_register_load"(%sel) : (i16) -> ()
  %in = "bcir.portio"(%port) {direction = "in", width = 8 : i32} : (i16) -> i8
  "bcir.portio"(%in, %port) {direction = "out", width = 8 : i32} : (i8, i16) -> ()
  return
}
"bcir.entry"() {sym_name = "_start", stack_top = @stack_top, c_entry = @kmain} : () -> ()
"bcir.interrupt_trampoline"() {sym_name = "isr14", vector = 14 : i32, handler = @page_fault,
  swapgs_on_user = true} : () -> ()
"bcir.abi_contract"() {sym_name = "f", target = "x86_64-linux", pointer_size = 8 : i32,
  long_size = 8 : i32, param_sizes = array<i64: 8, 4>, param_aligns = array<i64: 8, 4>,
  return_size = 4 : i32} : () -> ()
"bcir.device_manifest"() {sym_name = "nic0", device = "virtio-net", banks = "ram,mmio",
  domains = "RAM,MMIO", capacities = array<i64: 4096, 256>} : () -> ()

// CHECK: "bcir.volatile_load"
// CHECK: "bcir.volatile_store"
// CHECK: "bcir.atomic_rmw"
// CHECK: "bcir.atomic_cas"
// CHECK: "bcir.creg_read"
// CHECK: "bcir.creg_write"
// CHECK: "bcir.msr_read"
// CHECK: "bcir.msr_write"
// CHECK: "bcir.descriptor_load"
// CHECK: "bcir.segment_reload"
// CHECK: "bcir.task_register_load"
// CHECK: "bcir.portio"
// CHECK: "bcir.entry"
// CHECK: "bcir.interrupt_trampoline"
// CHECK: "bcir.abi_contract"
// CHECK: "bcir.device_manifest"
