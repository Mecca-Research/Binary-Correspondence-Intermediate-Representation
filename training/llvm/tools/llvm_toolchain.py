"""One rule for finding an LLVM tool, shared by every Python gate that needs one.

The shell gates have always resolved `${base}${LLVM_SUFFIX}` before a bare name --
`verify-examples.sh`'s `find_tool` does exactly that, and it is how CI points a whole job
at one release. The Python gates did not: they asked for `mlir-opt` first and fell back to
`mlir-opt-23`, which is the same list in the wrong order.

That ordering is a real defect on the LLVM 23 job, not a style question. The job sets
`LLVM_SUFFIX=-23` and asserts the `-23` binaries exist, but a Python gate preferring the
bare name would still run whatever unversioned `mlir-opt` happened to be on PATH -- and
then report a passing "ceiling" result that was measured against a different release. The
job would be green and its evidence would be about the wrong compiler.

So the search order here matches the shell's, and it lives in one place because two gates
had the same defect and a third would have acquired it:

  1. ``${base}${LLVM_SUFFIX}`` when LLVM_SUFFIX is set  -- the release the job selected
  2. ``base``                                           -- whatever the host calls default
  3. ``base-NN`` for the highest NN on PATH             -- a versioned install, newest first
"""

from __future__ import annotations

import os
import re
import shutil

__all__ = ["find_llvm_tool", "suffixed_names"]

_SUFFIXED = re.compile(r"-(\d+)$")


def suffixed_names(base: str) -> list[str]:
    """Every `base-NN` on PATH, newest major first."""
    found: dict[int, str] = {}
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory:
            continue
        try:
            entries = os.listdir(directory)
        except OSError:
            # An unreadable or missing PATH entry is normal, not a finding.
            continue
        for entry in entries:
            if not entry.startswith(f"{base}-"):
                continue
            match = _SUFFIXED.search(entry)
            if match and shutil.which(entry):
                found.setdefault(int(match.group(1)), entry)
    return [found[major] for major in sorted(found, reverse=True)]


def find_llvm_tool(base: str) -> str | None:
    """Resolve one LLVM tool, honouring LLVM_SUFFIX first. None when absent."""
    suffix = os.environ.get("LLVM_SUFFIX", "")
    if suffix:
        selected = shutil.which(f"{base}{suffix}")
        if selected:
            return selected
        # LLVM_SUFFIX is a deliberate choice by whoever set it. If the tool it names is
        # not here, falling through to a different release would answer a question nobody
        # asked, so the caller is told the tool is absent and decides what that means.
        return None
    direct = shutil.which(base)
    if direct:
        return direct
    for name in suffixed_names(base):
        resolved = shutil.which(name)
        if resolved:
            return resolved
    return None
