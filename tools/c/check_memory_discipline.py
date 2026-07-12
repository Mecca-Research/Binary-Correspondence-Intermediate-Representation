#!/usr/bin/env python3
"""Static enforcement for runtime/c ownership and allocation boundaries."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
C_DIR = ROOT / "runtime" / "c"
MANIFEST = C_DIR / "MEMORY_CLASSIFICATION.txt"


def without_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def fail(errors: list[str], path: Path, message: str) -> None:
    errors.append(f"{path.relative_to(ROOT)}: {message}")


def read_manifest(errors: list[str]) -> dict[str, str]:
    entries: dict[str, str] = {}
    valid = {"freestanding_core", "hosted_tool", "driver_adapter"}
    for line_number, raw in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2 or parts[0] not in valid:
            errors.append(f"{MANIFEST.relative_to(ROOT)}:{line_number}: invalid entry")
            continue
        category, name = parts
        if name in entries:
            errors.append(f"{MANIFEST.relative_to(ROOT)}:{line_number}: duplicate {name}")
        entries[name] = category
    return entries


def public_value_struct(text: str, name: str) -> str:
    match = re.search(rf"typedef struct {re.escape(name)}\s*\{{(.*?)\}}\s*{re.escape(name)}\s*;", text, re.S)
    return match.group(1) if match else ""


def main() -> int:
    errors: list[str] = []
    entries = read_manifest(errors)
    production = {path.name for path in C_DIR.glob("bcir_*.c")}
    missing = sorted(production - entries.keys())
    stale = sorted(entries.keys() - production)
    for name in missing:
        fail(errors, MANIFEST, f"unclassified production unit {name}")
    for name in stale:
        fail(errors, MANIFEST, f"classification names missing file {name}")

    heap_call = re.compile(r"(?<![-\w\"])(?:malloc|calloc|realloc|free)\s*\(")
    hosted_api = re.compile(r'#\s*include\s*[<"](?:stdio|stdlib|string)\.h[>"]|\bFILE\b|\bfopen\s*\(')
    for name, category in entries.items():
        path = C_DIR / name
        if not path.exists():
            continue
        code = without_comments(path.read_text(encoding="utf-8"))
        if category == "freestanding_core":
            if heap_call.search(code):
                fail(errors, path, "freestanding core calls a heap allocator")
            if hosted_api.search(code):
                fail(errors, path, "freestanding core imports or calls a hosted libc API")
            if "bcir_host_alloc.h" in code:
                fail(errors, path, "freestanding core includes the hosted allocator")
        if category == "driver_adapter" and heap_call.search(code):
            fail(errors, path, "driver adapter must be allocation-free")
        if category == "hosted_tool" and heap_call.search(code):
            fail(errors, path, "hosted tool bypasses bcir_host_allocator")

    # Reentrant context APIs may retain a documented process-static compatibility
    # context, but parser/preprocessor scratch must never reappear as hidden
    # function- or file-static mutable storage.
    allowed_compatibility_statics = {
        "static bcir_cpp_context context;",
        "static bcir_cfront_context context;",
        "static int initialized;",
    }
    local_static = re.compile(
        r"^\s*static\s+(?!const\b)(?:[A-Za-z_]\w*(?:\s+|\s*\*\s*))+"
        r"[A-Za-z_]\w*(?:\s*\[[^\]]*\])?\s*(?:=[^;]*)?;\s*$",
        re.M,
    )
    for name in ("bcir_cpp.c", "bcir_cfront.c"):
        path = C_DIR / name
        code = without_comments(path.read_text(encoding="utf-8"))
        for match in local_static.finditer(code):
            declaration = match.group(0).strip()
            if declaration not in allowed_compatibility_statics:
                fail(errors, path, f"hidden mutable static state: {declaration}")

    channel_header = (C_DIR / "bcir_runtime_channel.h").read_text(encoding="utf-8")
    for struct_name in (
        "bcir_channel_handle", "bcir_channel_mapping", "bcir_channel_submission",
        "bcir_channel_event",
    ):
        body = public_value_struct(channel_header, struct_name)
        if not body:
            fail(errors, C_DIR / "bcir_runtime_channel.h", f"missing value type {struct_name}")
        elif "*" in without_comments(body):
            fail(errors, C_DIR / "bcir_runtime_channel.h", f"{struct_name} exposes a raw pointer")
    for token in ("BCIR_CHANNEL_RING_BACKPRESSURE", "BCIR_CHANNEL_RING_OVERWRITE_OLDEST"):
        if token not in channel_header:
            fail(errors, C_DIR / "bcir_runtime_channel.h", f"missing explicit ring policy {token}")

    driver_code = without_comments((C_DIR / "bcir_runtime_channel.c").read_text(encoding="utf-8"))
    deferred_ipc = re.compile(r"\b(?:memfd_create|eventfd|epoll_|io_uring|shm_open|msgget|mq_open)\b|<sys/socket\.h>")
    if deferred_ipc.search(driver_code):
        fail(errors, C_DIR / "bcir_runtime_channel.c", "Linux IPC entered before the direct driver ABI stabilized")

    ownership_headers = {
        "bcir_cpp.h": ("borrowed", "NON-THREAD-SAFE"),
        "bcir_cfront.h": ("owns", "Idempotent"),
        "bcir_q8_model.h": ("owned", "borrowed"),
        "bcir_llama.h": ("borrowed", "released"),
    }
    for name, markers in ownership_headers.items():
        text = (C_DIR / name).read_text(encoding="utf-8")
        for marker in markers:
            if marker.lower() not in text.lower():
                fail(errors, C_DIR / name, f"missing ownership marker '{marker}'")

    if errors:
        print("memory-discipline static check failed:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    print(f"memory-discipline static check: {len(entries)} production units classified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
