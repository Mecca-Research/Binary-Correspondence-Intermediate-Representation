#!/usr/bin/env python3
"""The pinning manager's gate (BUILD-8, docs/BCIR_BUILD_ROADMAP.md §8): the tools a configured tree
builds with are the tools it was configured with.

    pins.py --deps-index build/cmake-gcc/bcir-deps.json

The configure pins every tool the build and the rails' BCIRfile run -- cc, cxx, ar, python -- by
resolved path and by the sha256 of its bytes (`tools` in bcir-deps.json; R1, registry-first: a
compiler is named by what it is). This gate:

1. reads the index, held to its schema (D1, bcir.toolchain.deps_index_problems);
2. hashes every pinned file again: one whose bytes moved since the configure is a finding -- the
   tree would now build with a compiler it was not configured with (an upgrade under a configured
   tree, a wrapper rewritten), and the fix is to reconfigure, never to let it pass;
3. writes the rails' BCIRfile from the pins (tools/build/bcirfile.py --pins: every tool declared
   by its pin, not re-hashed) and judges it with the tools checked (MK5): the plan names exactly the
   pinned tools, and they still are what their pins say.

Exit 0 when the host still holds every pin, 1 with a finding, 2 when it cannot judge (no index, an
index that breaks D1): every exit is a verdict (docs/security/laws.md L1).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bcir.make import check, parse  # noqa: E402
from bcir.toolchain import DepsIndexError, deps_pins, load_deps_index  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"bcir_build_{name}", ROOT / "tools" / "build" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def drift(pins: dict[str, tuple[str, str]]) -> list[str]:
    """Each pinned tool whose bytes are no longer its pin, or that is gone."""
    found: list[str] = []
    for name, (path, pinned) in sorted(pins.items()):
        try:
            h = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    h.update(chunk)
        except OSError as exc:
            found.append(f"{name} {path} cannot be read ({exc.strerror}): reconfigure")
            continue
        if f"sha256:{h.hexdigest()}" != pinned:
            found.append(
                f"{name} {path} is sha256:{h.hexdigest()} now, pinned {pinned} at configure: "
                "the tree would build with a tool it was not configured with; reconfigure"
            )
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--deps-index", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        pins = deps_pins(load_deps_index(args.deps_index))
    except DepsIndexError as exc:
        print(f"pins: UNUSABLE: {exc}")
        return 2
    findings = drift(pins)
    if not findings:
        generator = _load("bcirfile")
        sanitizer = _load("sanitizer")
        cc, cxx, ar, python = (pins[n][0] for n in ("cc", "cxx", "ar", "python"))
        tsan, _why = sanitizer.probe(cc, "thread")
        text, _planned = generator.generate(
            cc, cxx, ar, python, tsan, identities={n: pins[n][1] for n in pins}
        )
        for finding in check(parse(text.encode("ascii")), ROOT, check_tools=True):
            findings.append(f"the pinned BCIRfile: {finding}")
    for finding in findings:
        print(f"  FAIL {finding}")
    if findings:
        print(f"pins: FAIL ({len(findings)} finding(s)) in {args.deps_index}")
        return 1
    held = ", ".join(f"{n}={pins[n][1][7:19]}" for n in sorted(pins))
    print(f"pins: PASS every pinned tool is its pin ({held}); the pinned BCIRfile passes MK0-MK5")
    return 0


if __name__ == "__main__":
    sys.exit(main())
