"""Minimal JVM class-file assembler for BCIR's bounded float stack subset.

This promotes the execution-proven test assembler into a reusable backend seam.
It deliberately supports only the mnemonics emitted by ``stackify.to_jvm``;
unsupported instructions fail instead of producing unverifiable bytecode.
"""

from __future__ import annotations

import re
import struct


_CLASS_NAME = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
_NOARG = {"fadd": 0x62, "fsub": 0x66, "fmul": 0x6A, "fdiv": 0x6E}
_MAX_PROGRAM_INSTRUCTIONS = 16_384


class JVMClassError(ValueError):
    """The bounded JVM program cannot be represented safely."""


class _ConstantPool:
    def __init__(self) -> None:
        self.entries: list[bytes] = []
        self.indices: dict[tuple, int] = {}

    def _add(self, key: tuple, raw: bytes) -> int:
        if key in self.indices:
            return self.indices[key]
        if len(self.entries) >= 0xFFFE:
            raise JVMClassError("JVM constant pool is full")
        self.entries.append(raw)
        index = len(self.entries)
        self.indices[key] = index
        return index

    def utf8(self, value: str) -> int:
        raw = value.encode("utf-8")
        if len(raw) > 0xFFFF:
            raise JVMClassError("JVM UTF-8 constant is too large")
        return self._add(("utf8", value), b"\x01" + struct.pack(">H", len(raw)) + raw)

    def float(self, value: float) -> int:
        try:
            raw = struct.pack(">f", value)
        except (OverflowError, struct.error) as exc:
            raise JVMClassError(f"float constant is not representable: {value!r}") from exc
        return self._add(("float", raw), b"\x04" + raw)

    def class_(self, name: str) -> int:
        return self._add(("class", name), b"\x07" + struct.pack(">H", self.utf8(name)))

    def name_and_type(self, name: str, descriptor: str) -> int:
        return self._add(
            ("nat", name, descriptor),
            b"\x0c" + struct.pack(">HH", self.utf8(name), self.utf8(descriptor)),
        )

    def fieldref(self, owner: str, name: str, descriptor: str) -> int:
        return self._add(
            ("field", owner, name, descriptor),
            b"\x09" + struct.pack(
                ">HH", self.class_(owner), self.name_and_type(name, descriptor)
            ),
        )

    def methodref(self, owner: str, name: str, descriptor: str) -> int:
        return self._add(
            ("method", owner, name, descriptor),
            b"\x0a" + struct.pack(
                ">HH", self.class_(owner), self.name_and_type(name, descriptor)
            ),
        )

    def encode(self) -> bytes:
        return struct.pack(">H", len(self.entries) + 1) + b"".join(self.entries)


def _assemble_run(pool: _ConstantPool, mnemonics: tuple[str, ...]) -> tuple[bytes, int, int]:
    code = bytearray()
    max_local = 1
    depth = 0
    max_depth = 0
    for mnemonic in mnemonics:
        if not isinstance(mnemonic, str):
            raise JVMClassError("JVM mnemonics must be strings")
        if mnemonic.startswith("fload "):
            try:
                local = int(mnemonic[6:])
            except ValueError as exc:
                raise JVMClassError(f"malformed JVM mnemonic {mnemonic!r}") from exc
            if not 0 <= local <= 0xFF:
                raise JVMClassError("bounded JVM assembler supports local indices 0..255")
            code += bytes((0x17, local))
            max_local = max(max_local, local + 1)
            depth += 1
        elif mnemonic.startswith("fstore "):
            try:
                local = int(mnemonic[7:])
            except ValueError as exc:
                raise JVMClassError(f"malformed JVM mnemonic {mnemonic!r}") from exc
            if not 0 <= local <= 0xFF or depth < 1:
                raise JVMClassError("invalid JVM local store or stack underflow")
            code += bytes((0x38, local))
            max_local = max(max_local, local + 1)
            depth -= 1
        elif mnemonic.startswith("ldc ") and mnemonic.endswith("f"):
            try:
                value = float(mnemonic[4:-1])
            except ValueError as exc:
                raise JVMClassError(f"malformed JVM float constant {mnemonic!r}") from exc
            index = pool.float(value)
            if index > 0xFF:
                raise JVMClassError("bounded JVM assembler requires one-byte ldc indices")
            code += bytes((0x12, index))
            depth += 1
        elif mnemonic in _NOARG:
            if depth < 2:
                raise JVMClassError(f"stack underflow at {mnemonic!r}")
            code.append(_NOARG[mnemonic])
            depth -= 1
        else:
            raise JVMClassError(f"unsupported JVM mnemonic {mnemonic!r}")
        max_depth = max(max_depth, depth)
        if len(code) > 0xFFFF:
            raise JVMClassError("bounded JVM method exceeds 65535 code bytes")
    if depth != 1:
        raise JVMClassError(f"JVM run program must leave exactly one value, found {depth}")
    code.append(0xAE)  # freturn
    if len(code) > 0xFFFF:
        raise JVMClassError("bounded JVM method exceeds 65535 code bytes")
    return bytes(code), max_local, max_depth


def _method(code_index: int, access: int, name: int, descriptor: int,
            code: bytes, max_stack: int, max_locals: int) -> bytes:
    body = struct.pack(">HHI", max_stack, max_locals, len(code))
    body += code + struct.pack(">HH", 0, 0)
    attribute = struct.pack(">HI", code_index, len(body)) + body
    return struct.pack(">HHHH", access, name, descriptor, 1) + attribute


def build_jvm_class(class_name: str, jvm_program) -> bytes:
    """Build a deterministic Java 8 class with ``static float run()`` and main."""
    if not isinstance(class_name, str) or not _CLASS_NAME.fullmatch(class_name):
        raise JVMClassError("class_name must be an unqualified JVM identifier")
    try:
        iterator = iter(jvm_program)
    except TypeError as exc:
        raise JVMClassError("jvm_program must be iterable") from exc
    program = []
    for index, mnemonic in enumerate(iterator):
        if index >= _MAX_PROGRAM_INSTRUCTIONS:
            raise JVMClassError(
                f"JVM program exceeds {_MAX_PROGRAM_INSTRUCTIONS} instructions"
            )
        program.append(mnemonic)
    pool = _ConstantPool()
    run_code, run_locals, run_stack = _assemble_run(pool, tuple(program))
    this_class = pool.class_(class_name)
    object_class = pool.class_("java/lang/Object")
    code_name = pool.utf8("Code")
    run_name, run_descriptor = pool.utf8("run"), pool.utf8("()F")
    main_name = pool.utf8("main")
    main_descriptor = pool.utf8("([Ljava/lang/String;)V")
    run_ref = pool.methodref(class_name, "run", "()F")
    system_out = pool.fieldref("java/lang/System", "out", "Ljava/io/PrintStream;")
    println = pool.methodref("java/io/PrintStream", "println", "(F)V")
    main_code = (
        b"\xb2" + struct.pack(">H", system_out)
        + b"\xb8" + struct.pack(">H", run_ref)
        + b"\xb6" + struct.pack(">H", println)
        + b"\xb1"
    )
    methods = _method(
        code_name, 0x0009, run_name, run_descriptor, run_code,
        max(run_stack, 1), run_locals,
    ) + _method(code_name, 0x0009, main_name, main_descriptor, main_code, 2, 1)
    output = struct.pack(">IHH", 0xCAFEBABE, 0, 52)
    output += pool.encode()
    output += struct.pack(">HHHHHH", 0x0021, this_class, object_class, 0, 0, 2)
    output += methods + struct.pack(">H", 0)
    return output


__all__ = ["JVMClassError", "build_jvm_class"]
