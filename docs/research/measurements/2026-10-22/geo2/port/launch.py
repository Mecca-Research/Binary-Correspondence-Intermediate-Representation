"""Run one of the PR818 bundle's scripts unchanged on a cgroup-v1 host.

The study's protocol builds a host block that reads two cgroup-v2 files (cpu.max, memory.max); this
host mounts cgroup v1, so those two reads are answered from their v1 equivalents. Nothing else is
touched: the scripts, their sources and their hashes stay as uploaded, and the block is discarded
whenever the protocol is already frozen (the saved protocol is returned).
usage: python launch.py SCRIPT [ARGS...]
"""

import pathlib
import runpy
import sys

script = pathlib.Path(sys.argv[1]).resolve()
_read_text = pathlib.Path.read_text
V1 = {
    "/sys/fs/cgroup/cpu.max": (
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
    ),
    "/sys/fs/cgroup/memory.max": ("/sys/fs/cgroup/memory/memory.limit_in_bytes",),
}


def read_text(self, *args, **kwargs):
    key = str(self)
    if key in V1 and not self.exists():
        parts = []
        for other in V1[key]:
            try:
                parts.append(_read_text(pathlib.Path(other)).strip())
            except OSError:
                parts.append("unavailable")
        return "cgroup-v1 " + " ".join(parts)
    return _read_text(self, *args, **kwargs)


pathlib.Path.read_text = read_text
sys.argv = [str(script)] + sys.argv[2:]
sys.path.insert(0, str(script.parent))
runpy.run_path(str(script), run_name="__main__")
