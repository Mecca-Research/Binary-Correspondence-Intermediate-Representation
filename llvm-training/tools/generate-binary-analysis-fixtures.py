#!/usr/bin/env python3
"""Build tiny Chapter 15 objects and normalize deterministic static evidence."""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[2]
CHAPTER = REPO_ROOT / "llvm-training/15-binary-analysis"
MANIFEST_PATH = CHAPTER / "evidence-manifest.json"
GENERATED_HEADER = [
    "fixture", "evidence_kind", "scope", "name", "value", "unit", "classification"
]


class EvidenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class Section:
    index: int
    name: str
    section_type: int
    flags: int
    offset: int
    size: int
    link: int
    entry_size: int


@dataclass(frozen=True)
class Symbol:
    name: str
    value: int
    size: int
    info: int
    section_index: int


@dataclass(frozen=True)
class Instruction:
    offset: int
    size: int
    instruction_class: str
    target: int | None = None
    conditional_branch: bool = False
    return_instruction: bool = False
    call_instruction: bool = False


def repository_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise EvidenceError(f"path escapes repository: {value}") from exc
    return path


def load_manifest() -> dict[str, Any]:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot load {MANIFEST_PATH.relative_to(REPO_ROOT)}: {exc}") from exc
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("evidence"), list):
        raise EvidenceError("evidence manifest must have schema_version 1 and an evidence array")
    return manifest


def find_clang() -> str | None:
    override = os.environ.get("CLANG")
    if override:
        return shutil.which(override) or (override if Path(override).is_file() else None)
    for name in ("clang", "clang-20", "clang-19", "clang-18", "clang-17", "clang-16", "clang-15", "clang-14"):
        found = shutil.which(name)
        if found:
            return found
    return None


def command_version(command: str) -> str:
    result = subprocess.run(
        [command, "--version"], check=True, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout.splitlines()[0].strip()


def c_string(data: bytes, offset: int) -> str:
    end = data.find(b"\0", offset)
    if end < 0:
        raise EvidenceError("unterminated ELF string table entry")
    return data[offset:end].decode("utf-8")


def parse_elf(path: Path) -> tuple[bytes, list[Section], list[Symbol]]:
    data = path.read_bytes()
    if len(data) < 64 or data[:4] != b"\x7fELF":
        raise EvidenceError(f"{path}: expected an ELF object")
    if data[4] != 2 or data[5] != 1:
        raise EvidenceError(f"{path}: only ELF64 little-endian objects are supported")
    if struct.unpack_from("<H", data, 16)[0] != 1:
        raise EvidenceError(f"{path}: expected a relocatable ELF object")
    machine = struct.unpack_from("<H", data, 18)[0]
    if machine != 62:
        raise EvidenceError(f"{path}: expected x86-64 machine 62, found {machine}")

    section_offset = struct.unpack_from("<Q", data, 40)[0]
    section_entry_size, section_count, names_index = struct.unpack_from("<HHH", data, 58)
    if section_entry_size != 64 or section_count == 0 or names_index >= section_count:
        raise EvidenceError(f"{path}: malformed section table")

    raw_sections = [
        struct.unpack_from("<IIQQQQIIQQ", data, section_offset + index * section_entry_size)
        for index in range(section_count)
    ]
    names_header = raw_sections[names_index]
    names = data[names_header[4]:names_header[4] + names_header[5]]
    sections = [
        Section(index, c_string(names, raw[0]), raw[1], raw[2], raw[4], raw[5], raw[6], raw[9])
        for index, raw in enumerate(raw_sections)
    ]

    symbols: list[Symbol] = []
    for section in sections:
        if section.section_type != 2:  # SHT_SYMTAB
            continue
        if section.link >= len(sections) or section.entry_size != 24:
            raise EvidenceError(f"{path}: malformed symbol table")
        strings_section = sections[section.link]
        strings = data[strings_section.offset:strings_section.offset + strings_section.size]
        for offset in range(section.offset, section.offset + section.size, section.entry_size):
            name_offset, info, _other, section_index, value, size = struct.unpack_from("<IBBHQQ", data, offset)
            symbols.append(Symbol(c_string(strings, name_offset), value, size, info, section_index))
    return data, sections, symbols


def decode_x86_64(code: bytes, base: int) -> list[Instruction]:
    instructions: list[Instruction] = []
    cursor = 0
    while cursor < len(code):
        opcode = code[cursor]
        absolute = base + cursor
        if opcode in (0x89, 0x8B):
            instruction = Instruction(absolute, 2, "data-transfer")
        elif opcode in (0x01, 0x31):
            instruction = Instruction(absolute, 2, "integer-arithmetic")
        elif opcode == 0x8D and cursor + 2 < len(code):
            instruction = Instruction(absolute, 3, "integer-arithmetic")
        elif opcode == 0x85:
            instruction = Instruction(absolute, 2, "compare-test")
        elif opcode in range(0x70, 0x80) and cursor + 1 < len(code):
            displacement = struct.unpack_from("<b", code, cursor + 1)[0]
            instruction = Instruction(
                absolute, 2, "conditional-branch", absolute + 2 + displacement,
                conditional_branch=True,
            )
        elif opcode == 0x0F and cursor + 5 < len(code) and code[cursor + 1] in range(0x80, 0x90):
            displacement = struct.unpack_from("<i", code, cursor + 2)[0]
            instruction = Instruction(
                absolute, 6, "conditional-branch", absolute + 6 + displacement,
                conditional_branch=True,
            )
        elif opcode == 0xE8 and cursor + 4 < len(code):
            displacement = struct.unpack_from("<i", code, cursor + 1)[0]
            instruction = Instruction(
                absolute, 5, "direct-call", absolute + 5 + displacement,
                call_instruction=True,
            )
        elif opcode == 0xC3:
            instruction = Instruction(absolute, 1, "return", return_instruction=True)
        elif opcode == 0x90:
            instruction = Instruction(absolute, 1, "padding")
        else:
            remaining = code[cursor:].hex(" ")
            raise EvidenceError(f"unsupported x86-64 bytes at .text+0x{absolute:x}: {remaining}")
        if cursor + instruction.size > len(code):
            raise EvidenceError("truncated x86-64 instruction")
        instructions.append(instruction)
        cursor += instruction.size
    return instructions


def collect_rows(entry: dict[str, Any], object_path: Path) -> list[list[str]]:
    data, sections, symbols = parse_elf(object_path)
    text_sections = [section for section in sections if section.name == ".text"]
    if len(text_sections) != 1:
        raise EvidenceError(f"{entry['id']}: expected exactly one .text section")
    text = text_sections[0]
    functions = sorted(
        (
            symbol for symbol in symbols
            if symbol.name and symbol.section_index == text.index and symbol.info & 0x0F == 2 and symbol.size > 0
        ),
        key=lambda symbol: (symbol.value, symbol.name),
    )
    if not functions:
        raise EvidenceError(f"{entry['id']}: no function symbols found")

    rows: list[list[str]] = []
    classification = entry["classification"]
    fixture_id = entry["id"]
    by_address = {symbol.value: symbol.name for symbol in functions}
    aggregate_classes: Counter[str] = Counter()

    for function in functions:
        rows.append([fixture_id, "symbol", "artifact", function.name, "1", "present", classification])
        start = text.offset + function.value
        instructions = decode_x86_64(data[start:start + function.size], function.value)
        aggregate_classes.update(instruction.instruction_class for instruction in instructions)

        leaders = {function.value}
        function_end = function.value + function.size
        for instruction in instructions:
            if instruction.conditional_branch:
                if instruction.target is not None and function.value <= instruction.target < function_end:
                    leaders.add(instruction.target)
                fallthrough = instruction.offset + instruction.size
                if fallthrough < function_end:
                    leaders.add(fallthrough)
            if instruction.call_instruction:
                target_name = by_address.get(instruction.target if instruction.target is not None else -1)
                if target_name is None:
                    raise EvidenceError(
                        f"{fixture_id}: unresolved direct call target 0x{(instruction.target if instruction.target is not None else 0):x}"
                    )
                rows.append([
                    fixture_id, "call-edge", function.name,
                    f"{function.name}->{target_name}", "1", "edge", classification,
                ])
        rows.append([
            fixture_id, "basic-block-count", function.name, "basic_blocks",
            str(len(leaders)), "blocks", classification,
        ])

    for name in sorted(aggregate_classes):
        rows.append([
            fixture_id, "instruction-class-count", "artifact", name,
            str(aggregate_classes[name]), "instructions", classification,
        ])

    section_type_names = {1: "PROGBITS", 8: "NOBITS"}
    for section in sections:
        if section.name not in {".text", ".data", ".bss"}:
            continue
        flags = "".join(letter for bit, letter in ((2, "A"), (1, "W"), (4, "X")) if section.flags & bit) or "none"
        descriptor = f"{section.name}|{section_type_names.get(section.section_type, str(section.section_type))}|{flags}"
        rows.append([
            fixture_id, "object-section", "artifact", descriptor,
            str(section.size), "bytes", classification,
        ])
    return sorted(rows, key=lambda row: tuple(row[1:]))


def render_csv(rows: list[list[str]]) -> str:
    import io
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(GENERATED_HEADER)
    writer.writerows(rows)
    return stream.getvalue()


def render_provenance(entry: dict[str, Any], clang: str) -> str:
    payload = {
        "schema_version": 1,
        "fixture_id": entry["id"],
        "classification": entry["classification"],
        "source_fixture": entry["source_fixture"],
        "checked_in_csv": entry["checked_in_csv"],
        "target": {
            "triple": entry["target_triple"],
            "object_format": "ELF64",
            "architecture": "x86_64",
        },
        "build": {
            "command": entry["build_command"],
            "optimization_flags": entry["optimization_flags"],
            "expected_artifact_type": entry["expected_artifact_type"],
        },
        "collection_command": entry["collection_command"],
        "toolchain": {
            "clang": command_version(clang),
            "python": platform.python_version(),
        },
        "note": "Tool versions describe the last regeneration environment; --check compares deterministic CSV evidence and stable provenance fields, not version strings.",
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def stable_provenance(text: str) -> dict[str, Any]:
    data = json.loads(text)
    data.pop("toolchain", None)
    return data


def compare_or_write(path: Path, generated: str, check: bool, provenance: bool = False) -> bool:
    if not check:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(generated, encoding="utf-8")
        print(f"[write] {path.relative_to(REPO_ROOT)}")
        return True
    try:
        current = path.read_text(encoding="utf-8")
    except OSError:
        print(f"error: missing generated file: {path.relative_to(REPO_ROOT)}", file=sys.stderr)
        return False
    equal = stable_provenance(current) == stable_provenance(generated) if provenance else current == generated
    if equal:
        print(f"[check] {path.relative_to(REPO_ROOT)} ... ok")
        return True
    print(f"error: deterministic evidence drift: {path.relative_to(REPO_ROOT)}", file=sys.stderr)
    if not provenance:
        diff = difflib.unified_diff(
            current.splitlines(), generated.splitlines(),
            fromfile=str(path.relative_to(REPO_ROOT)), tofile="regenerated", lineterm="",
        )
        print("\n".join(diff), file=sys.stderr)
    return False


def generated_entries(manifest: dict[str, Any], fixture: str | None) -> list[dict[str, Any]]:
    entries = [entry for entry in manifest["evidence"] if entry.get("classification") == "deterministic"]
    if fixture is not None:
        entries = [entry for entry in entries if entry.get("id") == fixture]
        if not entries:
            raise EvidenceError(f"unknown deterministic fixture id: {fixture}")
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="regenerate in a temporary directory and fail on drift")
    mode.add_argument("--write", action="store_true", help="write checked-in CSV and provenance files (default)")
    parser.add_argument("--fixture", help="generate/check one deterministic fixture id")
    parser.add_argument("--require-tools", action="store_true", help="fail instead of skipping when clang is unavailable")
    args = parser.parse_args()

    try:
        manifest = load_manifest()
        entries = generated_entries(manifest, args.fixture)
        clang = find_clang()
        if clang is None:
            message = "clang is unavailable; deterministic Chapter 15 fixture generation skipped"
            if args.require_tools:
                raise EvidenceError(message)
            print(f"warning: {message}", file=sys.stderr)
            return 0

        success = True
        with tempfile.TemporaryDirectory(prefix="bcir-binary-evidence-") as temporary:
            temp = Path(temporary)
            for entry in entries:
                source = repository_path(entry["source_fixture"])
                artifact = temp / f"{entry['id']}.o"
                command = [
                    clang, f"--target={entry['target_triple']}",
                    "-Wno-unused-command-line-argument", *entry["optimization_flags"],
                    "-c", str(source), "-o", str(artifact),
                ]
                print("[build] " + " ".join(command))
                subprocess.run(command, check=True)
                csv_text = render_csv(collect_rows(entry, artifact))
                provenance_text = render_provenance(entry, clang)
                success &= compare_or_write(repository_path(entry["checked_in_csv"]), csv_text, args.check)
                success &= compare_or_write(
                    repository_path(entry["provenance_path"]), provenance_text, args.check, provenance=True
                )
        return 0 if success else 1
    except (EvidenceError, OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
