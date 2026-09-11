// Where the host stops and the device starts, made structural.
//
//   mlir-opt gpu-launch-outlining.mlir --gpu-kernel-outlining
//
// gpu.launch is a REGION: device code written inline in a host function, freely reading
// host SSA values that are in scope. That is convenient and it is a lie about the machine
// -- nothing is in scope across that boundary. Outlining is where the lie is paid for:
// every captured value becomes a kernel argument, and the launch becomes a
// gpu.launch_func naming a kernel that now lives in its own gpu.module.
//
// This runs on any host. No device, no vendor toolkit, no driver: the split is a
// structural fact about the IR, and only --convert-gpu-to-nvvm or -rocdl afterwards needs
// to know whose hardware it is.

func.func @saxpy(%a: f32, %x: memref<256xf32>, %y: memref<256xf32>) {
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%bxs = %c256, %bys = %c1, %bzs = %c1) {
    %xv = memref.load %x[%tx] : memref<256xf32>
    %yv = memref.load %y[%tx] : memref<256xf32>
    %m = arith.mulf %a, %xv : f32
    %s = arith.addf %m, %yv : f32
    memref.store %s, %y[%tx] : memref<256xf32>
    gpu.terminator
  }
  return
}
