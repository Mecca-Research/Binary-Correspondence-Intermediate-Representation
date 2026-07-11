"""WASM deployment target (via the LLVM path) -- one portable artifact, many backends.

The same K_BCIR-selected kernel that the AOT/JIT paths run is compiled to a
WebAssembly module with `clang --target=wasm32` (+ wasm-ld), validated, and -- when
node is present -- instantiated and executed to self-check. This proves the
target-open thesis on a second, fully-portable backend.
"""

from __future__ import annotations

import os
import struct
import subprocess
import tempfile
from shutil import which

from ..model import Module
from ..kbcir.realize import RealizationResult
from ..toolchain import resolve_llvm_tools
from .llvm import emit_kernel_ll


def is_valid_wasm(data: bytes) -> bool:
    """True if `data` begins with the WebAssembly magic + version 1 header."""
    return len(data) >= 8 and data[:4] == b"\x00asm" and data[4:8] == b"\x01\x00\x00\x00"


def _tool(*names: str) -> str | None:
    for n in names:
        p = which(n)
        if p:
            return p
    return None


def compile_to_wasm(
    module: Module,
    result: RealizationResult,
    fn_name: str = "bcir_kernel",
    workdir: str | None = None,
) -> tuple[bool, bytes | None, str]:
    """Compile the lowered kernel to a .wasm module. Returns (ok, bytes, message)."""
    llvm = resolve_llvm_tools("clang", "wasm-ld", pipeline="WASM")
    if not llvm.ok:
        return False, None, llvm.message
    clang, wasm_ld = (llvm.paths[name] for name in ("clang", "wasm-ld"))

    created = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="bcir-wasm-")
    try:
        ll = os.path.join(workdir, "kernel.ll")
        wasm = os.path.join(workdir, "kernel.wasm")
        with open(ll, "w") as f:
            f.write(emit_kernel_ll(module, result, fn_name))
        cmd = [clang, "--target=wasm32", f"-fuse-ld={wasm_ld}", "-nostdlib", "-O2",
               "-Wl,--no-entry", f"-Wl,--export={fn_name}", "-Wl,--export-memory",
               ll, "-o", wasm]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return False, None, "wasm compile failed:\n" + r.stdout + r.stderr
        with open(wasm, "rb") as f:
            data = f.read()
        if not is_valid_wasm(data):
            return False, data, "produced file is not a valid wasm module"
        return True, data, "ok"
    finally:
        if created:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)


# A JS harness that instantiates the module, lays out A/B/C in linear memory above a
# safe base, runs the kernel, and self-checks C == A + B (A[i]=i, B[i]=2i).
_HARNESS_JS = r"""
const fs = require('fs');
const path = process.argv[2];
const n = parseInt(process.argv[3], 10);
const fn = process.argv[4];
const bytes = fs.readFileSync(path);
WebAssembly.instantiate(bytes, {}).then(({ instance }) => {
  const mem = instance.exports.memory;
  const base = 1 << 20;                 // above clang's stack/data region
  const need = base + 3 * n * 4;
  while (mem.buffer.byteLength < need) mem.grow(16);
  const f32 = new Float32Array(mem.buffer);
  const iA = base / 4, iB = iA + n, iC = iB + n;
  for (let i = 0; i < n; i++) { f32[iA + i] = i; f32[iB + i] = 2 * i; f32[iC + i] = -1; }
  instance.exports[fn](iA * 4, iB * 4, iC * 4, BigInt(n));
  for (let i = 0; i < n; i++) {
    const want = f32[iA + i] + f32[iB + i];
    if (f32[iC + i] !== want) { console.log(`FAIL at ${i}: ${f32[iC + i]} != ${want}`); process.exit(1); }
  }
  console.log(`OK wasm ${fn} n=${n}`);
}).catch((e) => { console.log('ERROR ' + e); process.exit(2); });
"""


def run_wasm_node(
    module: Module,
    result: RealizationResult,
    fn_name: str = "bcir_kernel",
    n: int | None = None,
) -> tuple[bool, str]:
    """Compile to wasm and execute via node, self-checking the result. (ok, output)."""
    node = _tool("node")
    if node is None:
        return False, "node not found for WASM execution"
    if n is None:
        # the single elementwise claim's count
        n = next((c.count for ph in module.phases for c in ph.claims
                  if len(c.rd) == 2 and len(c.wr) == 1), 1024)

    workdir = tempfile.mkdtemp(prefix="bcir-wasm-run-")
    try:
        ok, data, msg = compile_to_wasm(module, result, fn_name, workdir=workdir)
        if not ok:
            return False, msg
        wasm = os.path.join(workdir, "kernel.wasm")
        with open(wasm, "wb") as f:
            f.write(data)
        js = os.path.join(workdir, "harness.js")
        with open(js, "w") as f:
            f.write(_HARNESS_JS)
        run = subprocess.run([node, js, wasm, str(n), fn_name], capture_output=True, text=True)
        ok = run.returncode == 0 and "OK" in run.stdout
        return ok, run.stdout + run.stderr
    finally:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)
