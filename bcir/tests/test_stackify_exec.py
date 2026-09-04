"""Execution-validation for the stackify bytecode targets (H5: native-codegen-outside-LLVM honesty).

The sibling ``test_stackify.py`` checks the WASM/JVM/CIL encoders for byte-SHAPE only (exact mnemonic-string
equality on one hand-pinned expression). That is not honest as *execution* evidence: a stack-op renderer that
emits the wrong opcode for some other expression would pass. This module closes that gap two ways:

  1. A CROSS-ENCODER SEMANTIC DIFFERENTIAL (always runs, pure Python): over a fuzzed corpus of random
     expression trees, render each via ``to_jvm`` / ``to_cil`` / ``to_wasm`` and INTERPRET the emitted
     mnemonics under each target's stack-machine semantics; all three must reproduce the direct float
     evaluation of the expression. This validates the encoders' *meaning* across inputs, not one pinned string.

  2. A REAL-JVM EXECUTION check (gated on ``java``): assemble the actual ``to_jvm()`` mnemonics into a real
     ``.class`` file (a small, self-contained constant-pool + Code-attribute writer for the bounded float
     subset stackify emits) and RUN it on the in-container JVM, comparing stdout to the oracle. The assembler
     only knows the CORRECT float mnemonics, so a bad mnemonic from ``to_jvm`` fails to assemble -- the JVM's
     own verifier + execution semantics are the ground truth, not another copy of our own code.

  3. A REAL-CLR EXECUTION check (gated on ``ilasm`` + ``mono``): wrap the actual ``to_cil()`` mnemonics in a
     minimal ``.il`` compilation unit (a static ``float32 run()`` + an ``.entrypoint`` printing it), assemble
     with the real ``ilasm``, and RUN it under mono, comparing stdout to the same StackOp oracle -- the exact
     mirror of the JVM harness. The CLR's own assembler + execution semantics are the ground truth. Without a
     CLR toolchain this is a clean documented skip (the differential above still validates the semantics).

The WASM clang->wasm-ld->node path is already execution-validated in ``test_wasm.py`` (a distinct,
resident-compiler path); the stackify ``to_wasm`` TEXT encoder is covered by the differential here.
"""

import random
import struct
from shutil import which

from bcir.lower.stackify import (
    BinOp,
    Const,
    Local,
    assign,
    stackify,
    to_cil,
    to_jvm,
    to_wasm,
)


# === reference semantics: a direct float evaluator + a StackOp stack-machine =============================


def _eval_expr(e, vals: dict) -> float:
    """Direct float evaluation of an Expr tree (the oracle)."""
    if isinstance(e, Const):
        return float(e.value)
    if isinstance(e, Local):
        return vals[e.index]
    a, b = _eval_expr(e.lhs, vals), _eval_expr(e.rhs, vals)
    if e.op == "add":
        return a + b
    if e.op == "sub":
        return a - b
    if e.op == "mul":
        return a * b
    return a / b


def _interp_stackops(ops, n_locals: int = 16) -> float:
    """Run a StackOp sequence on a Python stack machine (locals start at 0.0). The independent oracle for the
    real-JVM run: it interprets the StackOps, NOT the rendered mnemonic strings."""
    st, loc = [], [0.0] * n_locals
    for o in ops:
        if o.kind == "const":
            st.append(float(o.arg))
        elif o.kind == "local_get":
            st.append(loc[o.arg])
        elif o.kind == "local_set":
            loc[o.arg] = st.pop()
        elif o.kind in ("add", "sub", "mul", "div"):
            b, a = st.pop(), st.pop()
            if o.kind == "add":
                st.append(a + b)
            elif o.kind == "sub":
                st.append(a - b)
            elif o.kind == "mul":
                st.append(a * b)
            else:
                st.append(a / b)
        else:
            raise ValueError(f"bad StackOp {o.kind!r}")
    return st[-1]


def _interp_mnemonics(lines, get_p, set_p, const_p, binops) -> float:
    """Generic stack-machine interpreter over a target's emitted mnemonic STRINGS (validates the encoder)."""
    st, loc = [], {}
    for ln in lines:
        if ln.startswith(get_p):
            st.append(loc[int(ln[len(get_p) :])])
        elif ln.startswith(set_p):
            loc[int(ln[len(set_p) :])] = st.pop()
        elif ln.startswith(const_p):
            st.append(float(ln[len(const_p) :].rstrip("f")))
        elif ln in binops:
            b, a = st.pop(), st.pop()
            st.append(binops[ln](a, b))
        else:
            raise ValueError(f"unrecognized mnemonic {ln!r}")
    return st[-1]


_ADD, _SUB, _MUL, _DIV = (
    (lambda a, b: a + b),
    (lambda a, b: a - b),
    (lambda a, b: a * b),
    (lambda a, b: a / b),
)


def _interp_jvm(lines):
    return _interp_mnemonics(
        lines, "fload ", "fstore ", "ldc ", {"fadd": _ADD, "fsub": _SUB, "fmul": _MUL, "fdiv": _DIV}
    )


def _interp_cil(lines):
    return _interp_mnemonics(
        lines, "ldloc.", "stloc.", "ldc.r4 ", {"add": _ADD, "sub": _SUB, "mul": _MUL, "div": _DIV}
    )


def _interp_wasm(lines):
    return _interp_mnemonics(
        lines,
        "local.get ",
        "local.set ",
        "f32.const ",
        {"f32.add": _ADD, "f32.sub": _SUB, "f32.mul": _MUL, "f32.div": _DIV},
    )


# === a minimal JVM .class assembler for the bounded stackify float subset ================================


class _ConstPool:
    """A growable, deduplicating JVM constant pool (1-indexed, as the class-file format requires)."""

    def __init__(self):
        self.entries = []  # raw bytes per entry
        self._idx = {}  # dedupe key -> 1-based index

    def _add(self, key, raw):
        if key in self._idx:
            return self._idx[key]
        self.entries.append(raw)
        self._idx[key] = len(self.entries)
        return self._idx[key]

    def utf8(self, s):
        b = s.encode("utf-8")
        return self._add(("u", s), bytes([1]) + struct.pack(">H", len(b)) + b)

    def float_(self, f):  # CONSTANT_Float (tag 4): a 32-bit IEEE-754 value
        raw = struct.pack(">f", f)
        return self._add(("f", raw), bytes([4]) + raw)

    def class_(self, name):
        return self._add(("c", name), bytes([7]) + struct.pack(">H", self.utf8(name)))

    def nat(self, name, desc):
        return self._add(
            ("nt", name, desc), bytes([12]) + struct.pack(">HH", self.utf8(name), self.utf8(desc))
        )

    def fieldref(self, cls, name, desc):
        return self._add(
            ("fr", cls, name, desc),
            bytes([9]) + struct.pack(">HH", self.class_(cls), self.nat(name, desc)),
        )

    def methodref(self, cls, name, desc):
        return self._add(
            ("mr", cls, name, desc),
            bytes([10]) + struct.pack(">HH", self.class_(cls), self.nat(name, desc)),
        )

    def to_bytes(self):
        return struct.pack(">H", len(self.entries) + 1) + b"".join(self.entries)


_JVM_NOARG = {"fadd": 0x62, "fsub": 0x66, "fmul": 0x6A, "fdiv": 0x6E}


def _assemble_run_code(cp, jvm_mnemonics):
    """Assemble the to_jvm() float mnemonics into Code bytes for `static float run()` (ending in freturn).
    Only the CORRECT float mnemonics are recognized -- an unexpected mnemonic raises (so a bad to_jvm() emit
    is caught here rather than silently executed)."""
    code, max_local = bytearray(), 0
    for m in jvm_mnemonics:
        if m.startswith("fload "):
            n = int(m[6:])
            code += bytes([0x17, n])  # fload n
            max_local = max(max_local, n + 1)
        elif m.startswith("fstore "):
            n = int(m[7:])
            code += bytes([0x38, n])  # fstore n
            max_local = max(max_local, n + 1)
        elif m.startswith("ldc ") and m.endswith("f"):
            fi = cp.float_(float(m[4:-1]))
            if fi > 0xFF:
                raise ValueError(
                    "constant pool too large for 1-byte ldc (expand to ldc_w if this ever fires)"
                )
            code += bytes([0x12, fi])  # ldc #fi
        elif m in _JVM_NOARG:
            code += bytes([_JVM_NOARG[m]])
        else:
            raise ValueError(f"unsupported JVM mnemonic for assembly: {m!r}")
    code += bytes([0xAE])  # freturn
    return bytes(code), max(max_local, 1)


def _method(code_attr_idx, access, name_idx, desc_idx, code, max_stack, max_locals):
    body = struct.pack(">HHI", max_stack, max_locals, len(code)) + code + struct.pack(">HH", 0, 0)
    attr = struct.pack(">HI", code_attr_idx, len(body)) + body
    return struct.pack(">HHHH", access, name_idx, desc_idx, 1) + attr


def build_class(class_name, jvm_program):
    """Build a runnable `.class` (major 52 / Java 8) with `static float run()` = the assembled to_jvm() program
    and `static void main` that prints run(). Straight-line float code => no StackMapTable needed."""
    cp = _ConstPool()
    run_code, run_locals = _assemble_run_code(cp, jvm_program)  # interns the float constants first
    this_c = cp.class_(class_name)
    obj_c = cp.class_("java/lang/Object")
    code_attr = cp.utf8("Code")
    run_n, run_d = cp.utf8("run"), cp.utf8("()F")
    main_n, main_d = cp.utf8("main"), cp.utf8("([Ljava/lang/String;)V")
    run_mref = cp.methodref(class_name, "run", "()F")
    sys_out = cp.fieldref("java/lang/System", "out", "Ljava/io/PrintStream;")
    println = cp.methodref("java/io/PrintStream", "println", "(F)V")
    main_code = (
        bytes([0xB2])
        + struct.pack(">H", sys_out)  # getstatic System.out
        + bytes([0xB8])
        + struct.pack(">H", run_mref)  # invokestatic run()F
        + bytes([0xB6])
        + struct.pack(">H", println)  # invokevirtual println(F)V
        + bytes([0xB1])
    )  # return
    methods = _method(code_attr, 0x0009, run_n, run_d, run_code, 8, run_locals) + _method(
        code_attr, 0x0009, main_n, main_d, main_code, 2, 1
    )
    out = struct.pack(">IHH", 0xCAFEBABE, 0, 52)  # magic, minor, major(=52, Java 8)
    out += cp.to_bytes()
    out += struct.pack(
        ">HHHHHH", 0x0021, this_c, obj_c, 0, 0, 2
    )  # access, this, super, ifaces, fields, methods
    out += methods
    out += struct.pack(">H", 0)  # class attributes_count
    return out


def _run_on_jvm(jvm_program):
    import os
    import subprocess
    import tempfile

    cls = "BcirStackTest"
    d = tempfile.mkdtemp()
    with open(os.path.join(d, cls + ".class"), "wb") as fh:
        fh.write(build_class(cls, jvm_program))
    p = subprocess.run(["java", "-cp", d, cls], capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        raise RuntimeError(f"java exited {p.returncode}: {p.stderr.strip()}")
    return float(p.stdout.strip())


# === a fuzz corpus of expression trees ===================================================================


def _rand_expr(rng, nloc, depth):
    if depth <= 0 or rng.random() < 0.4:
        return (
            Local(rng.randrange(nloc))
            if rng.random() < 0.5
            else Const(round(rng.uniform(-3.0, 3.0), 3))
        )
    op = rng.choice(["add", "sub", "mul"])  # div handled in a dedicated case (avoid /0 in the fuzz)
    return BinOp(op, _rand_expr(rng, nloc, depth - 1), _rand_expr(rng, nloc, depth - 1))


def _program_for(expr, vals):
    """Self-contained stack program: set each local to its value, then evaluate `expr` (result left on top)."""
    prog = []
    for i in sorted(vals):
        prog += assign(i, Const(vals[i]))
    return prog + stackify(expr)


# === the tests ==========================================================================================


def test_three_text_encoders_execute_consistently():
    # The cross-encoder SEMANTIC differential: over a fuzz corpus, all three rendered mnemonic streams interpret
    # to the same value as the direct expression evaluation -- far stronger than the single pinned-string check.
    rng = random.Random(0x57AC)
    for _ in range(60):
        nloc = rng.randint(1, 3)
        vals = {i: round(rng.uniform(-5.0, 5.0), 3) for i in range(nloc)}
        expr = _rand_expr(rng, nloc, depth=3)
        oracle = _eval_expr(expr, vals)
        prog = _program_for(expr, vals)
        assert abs(_interp_jvm(to_jvm(prog)) - oracle) < 1e-6, ("jvm", expr, vals)
        assert abs(_interp_cil(to_cil(prog)) - oracle) < 1e-6, ("cil", expr, vals)
        assert abs(_interp_wasm(to_wasm(prog)) - oracle) < 1e-6, ("wasm", expr, vals)


def test_text_encoders_handle_division():
    # division is exercised explicitly (a nonzero divisor) across all three encoders.
    prog = (
        assign(0, Const(6.0)) + assign(1, Const(4.0)) + stackify(BinOp("div", Local(0), Local(1)))
    )
    assert abs(_interp_jvm(to_jvm(prog)) - 1.5) < 1e-9
    assert abs(_interp_cil(to_cil(prog)) - 1.5) < 1e-9
    assert abs(_interp_wasm(to_wasm(prog)) - 1.5) < 1e-9


def test_jvm_stackify_assembles_and_executes_on_real_jvm():
    # H5 honesty capstone: assemble the actual to_jvm() output into a real .class and RUN it on the JVM.
    if not which("java"):
        return  # no JVM toolchain -> clean skip (the differential above still validated the encoder semantics)
    ops = (
        assign(0, Const(2.0))
        + assign(1, Const(3.0))
        + assign(2, Const(4.0))
        + stackify(BinOp("mul", BinOp("add", Local(0), Local(1)), Local(2)))
    )  # (2+3)*4 = 20
    oracle = _interp_stackops(ops)
    assert abs(oracle - 20.0) < 1e-9
    got = _run_on_jvm(to_jvm(ops))
    assert abs(got - oracle) < 1e-4, (got, oracle)


def test_jvm_real_execution_over_a_small_fuzz_corpus():
    # the real JVM agrees with the oracle across several distinct expressions (not just the one headline case).
    if not which("java"):
        return  # documented skip without a JVM
    rng = random.Random(0x1A2B)
    for _ in range(6):
        nloc = rng.randint(1, 3)
        vals = {i: round(rng.uniform(-4.0, 4.0), 2) for i in range(nloc)}
        expr = _rand_expr(rng, nloc, depth=2)
        prog = _program_for(expr, vals)
        oracle = _interp_stackops(prog)
        got = _run_on_jvm(to_jvm(prog))
        assert abs(got - oracle) < 1e-3, (got, oracle, expr, vals)


def _build_il(cil_program, class_name="BcirStackTest"):
    """Wrap the actual to_cil() mnemonics in a minimal runnable .il unit: static float32 run() = the program,
    plus an .entrypoint Main that prints run() via the invariant culture (so the float parses back)."""
    n_locals = 1 + max(
        (int(ln.rsplit(".", 1)[1]) for ln in cil_program if ln.startswith(("ldloc.", "stloc."))),
        default=0,
    )
    body = "\n".join(f"    {ln}" for ln in cil_program)
    locals_init = ", ".join(["float32"] * n_locals)
    return f"""
.assembly extern mscorlib {{}}
.assembly {class_name} {{}}
.method static float32 run() cil managed
{{
    .maxstack 8
    .locals init ({locals_init})
{body}
    ret
}}
.method static void Main() cil managed
{{
    .entrypoint
    .maxstack 2
    call float32 run()
    call void [mscorlib]System.Console::WriteLine(float32)
    ret
}}
"""


def _run_on_cil(cil_program):
    """Assemble the actual to_cil() mnemonics via the real ilasm and RUN the .exe under mono. The CLR's own
    assembler + verifier + execution are the ground truth (a bad to_cil() mnemonic fails to assemble)."""
    import os
    import subprocess
    import tempfile

    d = tempfile.mkdtemp()
    il = os.path.join(d, "BcirStackTest.il")
    exe = os.path.join(d, "BcirStackTest.exe")
    with open(il, "w", encoding="utf-8") as fh:
        fh.write(_build_il(cil_program))
    p = subprocess.run(
        ["ilasm", "/exe", f"/output:{exe}", il], capture_output=True, text=True, timeout=60
    )
    if p.returncode != 0 or not os.path.exists(exe):
        raise RuntimeError(f"ilasm exited {p.returncode}: {p.stdout.strip()} {p.stderr.strip()}")
    p = subprocess.run(["mono", exe], capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        raise RuntimeError(f"mono exited {p.returncode}: {p.stderr.strip()}")
    return float(p.stdout.strip().replace(",", "."))  # tolerate a non-invariant decimal separator


def test_cil_stackify_assembles_and_executes_on_real_clr():
    # H5 honesty capstone, CIL half: assemble the actual to_cil() output via the real ilasm and RUN it on the
    # CLR (mono), mirroring the JVM check. Without a CLR toolchain this is a clean documented skip -- the
    # cross-encoder differential above still validates the encoder semantics.
    if not (which("ilasm") and which("mono")):
        return  # honest documented skip -- no CLR toolchain in this environment
    ops = (
        assign(0, Const(2.0))
        + assign(1, Const(3.0))
        + assign(2, Const(4.0))
        + stackify(BinOp("mul", BinOp("add", Local(0), Local(1)), Local(2)))
    )  # (2+3)*4 = 20
    oracle = _interp_stackops(ops)
    assert abs(oracle - 20.0) < 1e-9
    got = _run_on_cil(to_cil(ops))
    assert abs(got - oracle) < 1e-4, (got, oracle)


def test_cil_real_execution_over_a_small_fuzz_corpus():
    # the real CLR agrees with the StackOp oracle across several distinct expressions (not just the headline).
    if not (which("ilasm") and which("mono")):
        return  # documented skip without a CLR toolchain
    rng = random.Random(0x2B3C)
    for _ in range(6):
        nloc = rng.randint(1, 3)
        vals = {i: round(rng.uniform(-4.0, 4.0), 2) for i in range(nloc)}
        expr = _rand_expr(rng, nloc, depth=2)
        prog = _program_for(expr, vals)
        oracle = _interp_stackops(prog)
        got = _run_on_cil(to_cil(prog))
        assert abs(got - oracle) < 1e-3, (got, oracle, expr, vals)


if __name__ == "__main__":
    import sys

    mod = sys.modules[__name__]
    tests = sorted(n for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n)))
    passed = failed = 0
    for name in tests:
        try:
            getattr(mod, name)()
            passed += 1
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {e!r}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
