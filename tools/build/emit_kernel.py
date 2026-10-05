#!/usr/bin/env python3
"""Write one of the manifest's `kernels`: C the Python oracle emits, with its driver appended.

The E-series sections of tools/c/check_runtime.sh judge kernels bcir.lower.c_kernel emits by compiling
each with a driver `main` and running it. runtime/manifest.json names each such program under
`kernels`: the emitter, its arguments, the driver (runtime/c/kernels/<main>) and the system libraries
it links. The gate and the CMake build both write the translation unit with this tool, so the two
compile the same text (BUILD-2h, docs/BCIR_BUILD_ROADMAP.md):

    emit_kernel.py kernel_ols                          # the unit on stdout
    emit_kernel.py kernel_ols -o k.c --depfile k.c.d   # the unit in k.c, and what it was made from

The unit is the emitter's text, a newline, then the driver's text: the driver calls the function the
kernel defines above it, so it is no unit of its own. The depfile is a Make rule naming every file the
unit was made from -- the manifest, the driver, this tool and each oracle module the emitter imported --
so a build that reads it re-emits the kernel when any of them changes, and only then.

Exit 0 with the unit written, 2 when the kernel cannot be written (no such kernel, an entry
tools/build/manifest.py's M15 refuses, an unreadable manifest or driver, an unwritable output), 1 when
the emitter raises: every exit is a verdict (docs/security/laws.md L1).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "runtime" / "manifest.json"


def _manifest_tool():
    spec = importlib.util.spec_from_file_location(
        "bcir_build_manifest", ROOT / "tools" / "build" / "manifest.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def unit_text(manifest: dict, name: str) -> str:
    """The kernel's translation unit. The caller has asked `kernel_problems` first; the emitter is
    imported from this checkout's oracle and may raise."""
    entry = manifest["kernels"][name]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    emitters = importlib.import_module("bcir.lower.c_kernel")
    driver = (ROOT / "runtime" / "c" / "kernels" / entry["main"]).read_text(encoding="utf-8")
    return getattr(emitters, entry["emit"])(*entry["args"]) + "\n" + driver


def _make_path(path: Path) -> str:
    """A path as a Make rule spells it: a space escaped, a `$` doubled."""
    return str(path).replace("$", "$$").replace(" ", "\\ ")


def depfile_text(out: Path, driver: Path) -> str:
    """`out: <every file the unit was made from>`: the manifest, the driver, this tool and every module
    under the checkout that is loaded now -- the emitter and what it imported."""
    made_from = {MANIFEST.resolve(), driver.resolve(), Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if not file:
            continue
        path = Path(file).resolve()
        if path.suffix == ".py" and ROOT in path.parents:
            made_from.add(path)
    deps = " \\\n  ".join(_make_path(p) for p in sorted(made_from))
    return f"{_make_path(out)}: \\\n  {deps}\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kernel", help="a name under the manifest's `kernels`")
    parser.add_argument("-o", "--out", type=Path, help="write the unit here (default: stdout)")
    parser.add_argument(
        "--depfile", type=Path, help="also write a Make rule of what it was made from"
    )
    args = parser.parse_args(argv)
    if args.depfile is not None and args.out is None:
        print("emit_kernel: UNUSABLE: --depfile names what -o writes; pass -o", file=sys.stderr)
        return 2
    tool = _manifest_tool()
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        print(f"emit_kernel: UNUSABLE: {MANIFEST}: {exc}", file=sys.stderr)
        return 2
    problems = tool.kernel_problems(manifest, args.kernel, tool.KernelFacts(ROOT))
    if problems:
        for problem in problems:
            print(f"emit_kernel: UNUSABLE: {problem}", file=sys.stderr)
        return 2
    try:
        text = unit_text(manifest, args.kernel)
    except OSError as exc:
        print(f"emit_kernel: UNUSABLE: {args.kernel}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 -- the emitter's own failure is this tool's FAIL verdict
        emitter = manifest["kernels"][args.kernel]["emit"]
        print(f"emit_kernel: FAIL: {emitter} raised {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if not isinstance(text, str):
        print(f"emit_kernel: FAIL: {args.kernel}'s emitter returned no text", file=sys.stderr)
        return 1
    if args.out is None:
        sys.stdout.write(text)
        return 0
    driver = ROOT / "runtime" / "c" / "kernels" / manifest["kernels"][args.kernel]["main"]
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8", newline="\n")
        if args.depfile is not None:
            args.depfile.parent.mkdir(parents=True, exist_ok=True)
            args.depfile.write_text(depfile_text(args.out, driver), encoding="utf-8", newline="\n")
    except OSError as exc:
        print(f"emit_kernel: UNUSABLE: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
