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
from .alias_facts import kernel_facts
from .llvm import _FOP, emit_kernel_ll, find_elementwise


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
        with open(ll, "w", newline="\n") as f:
            f.write(emit_kernel_ll(module, result, fn_name))
        cmd = [
            clang,
            "--target=wasm32",
            f"-fuse-ld={wasm_ld}",
            "-nostdlib",
            "-O2",
            "-Wl,--no-entry",
            f"-Wl,--export={fn_name}",
            "-Wl,--export-memory",
            ll,
            "-o",
            wasm,
        ]
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
const [path, nArg, fn, bindArg, op] = process.argv.slice(2);
const n = parseInt(nArg, 10);
// One buffer per declared resource: bind[p] is the buffer operand p (A, B, C) is bound to.
const bind = bindArg.split(',').map(Number);
const nbuf = Math.max(...bind) + 1;
const CANARY = 64;
const total = n + CANARY;
const apply = { '+': (a, b) => Math.fround(a + b), '-': (a, b) => Math.fround(a - b),
                '*': (a, b) => Math.fround(a * b) }[op];
const bytes = fs.readFileSync(path);
WebAssembly.instantiate(bytes, {}).then(({ instance }) => {
  const mem = instance.exports.memory;
  const base = 1 << 20;                 // above clang's stack/data region
  const need = base + nbuf * total * 4;
  while (mem.buffer.byteLength < need) mem.grow(16);
  const f32 = new Float32Array(mem.buffer);
  const at = (k) => base / 4 + k * total;
  const pattern = [(i) => i, (i) => 2 * i, (i) => -7777777];
  const first = [];                     // each buffer's pattern: the first operand bound to it
  for (let p = 0; p < 3; p++) if (first[bind[p]] === undefined) first[bind[p]] = p;
  for (let k = 0; k < nbuf; k++) for (let i = 0; i < total; i++) f32[at(k) + i] = pattern[first[k]](i);
  const snap = [];
  for (let k = 0; k < nbuf; k++) snap.push(f32.slice(at(k), at(k) + total));
  instance.exports[fn](at(bind[0]) * 4, at(bind[1]) * 4, at(bind[2]) * 4, BigInt(n));
  // The written buffer holds A op B below n and its snapshot from n on (an unmasked tail);
  // every other buffer holds its snapshot everywhere (a write through a read pointer).
  for (let k = 0; k < nbuf; k++) for (let i = 0; i < total; i++) {
    const want = (k === bind[2] && i < n) ? apply(snap[bind[0]][i], snap[bind[1]][i]) : snap[k][i];
    if (f32[at(k) + i] !== want) {
      console.log(`FAIL buffer ${k} at ${i}: ${f32[at(k) + i]} != ${want}`);
      process.exit(1);
    }
  }
  console.log(`OK wasm ${fn} n=${n}`);
}).catch((e) => { console.log('ERROR ' + e); process.exit(2); });
"""


def harness_binding(module: Module, result: RealizationResult) -> tuple[int, int, int]:
    """The buffer each operand (A, B, C) is bound to in the node self-check: one buffer per
    declared resource (S5-A), so an in-place claim runs in place."""
    claim, _ = find_elementwise(module, result)
    facts = kernel_facts(module, claim, "f32")
    return tuple(facts.resources.index(rid) for rid in facts.rids)


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
    claim, _ = find_elementwise(module, result)
    if n is None:
        n = max(1, claim.count)  # the single elementwise claim's count
    binding = ",".join(str(k) for k in harness_binding(module, result))
    op = _FOP[claim.opcode][1]

    workdir = tempfile.mkdtemp(prefix="bcir-wasm-run-")
    try:
        ok, data, msg = compile_to_wasm(module, result, fn_name, workdir=workdir)
        if not ok:
            return False, msg
        wasm = os.path.join(workdir, "kernel.wasm")
        with open(wasm, "wb") as f:
            f.write(data)
        js = os.path.join(workdir, "harness.js")
        with open(js, "w", newline="\n") as f:
            f.write(_HARNESS_JS)
        run = subprocess.run(
            [node, js, wasm, str(n), fn_name, binding, op], capture_output=True, text=True
        )
        ok = run.returncode == 0 and "OK" in run.stdout
        return ok, run.stdout + run.stderr
    finally:
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)
