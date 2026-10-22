"""GEO-2b level 2: the three edits Codex's drivers need to drive an ABI 5 library.

ABI 5 (REMAT) appends one field, `uint64_t recompute`, to bcir_decoder_state; the research heads
recompute through their own tiers, so it is 0 in every call. The drivers assert ABI 4; level 2 admits
5 where they assert. After every native call the library's refusal count (build_main.py) must still
be 0: main's products return whether they ran, and a research step never skips one silently.
Every other line of every driver is the bundle's. Writes the unified diff beside the work copy.
usage: python3 -I host_abi5.py WORK_DIR
"""

import difflib, sys
from pathlib import Path

work = Path(sys.argv[1])
EDITS = {
    "native_abi.py": [
        (
            '            ("workspace",ct.c_void_p),\n            ("workspace_bytes",_SZ),\n        ]',
            '            ("workspace",ct.c_void_p),\n            ("workspace_bytes",_SZ),\n'
            '            ("recompute",ct.c_uint64),  # ABI 5 (REMAT); 0: the research heads recompute through their own tiers\n'
            "        ]",
        ),
        ("    _fields_=_State._fields_[:-2]", "    _fields_=_State._fields_[:-3]"),
    ],
    "experiment.py": [
        (
            "abi=int(lib.bcir_decoder_abi_version());assert abi in (3,4)",
            "abi=int(lib.bcir_decoder_abi_version());assert abi in (3,5)",
        ),
        ("state_type=_State if abi==4 else _State3", "state_type=_State if abi==5 else _State3"),
        ("        if abi==4:\n", "        if abi==5:\n"),
    ],
    "policy.py": [
        (
            "assert lib.bcir_decoder_abi_version()==4,'budget controller requires updated ABI4'",
            "assert lib.bcir_decoder_abi_version()==5,'budget controller requires main\\'s ABI5'\n"
            "        lib.bcir_geo_refused.restype=ct.c_ulong",
        ),
        (
            "        native_seconds=time.perf_counter()-native_start;self.check(rc);assert done.value==1\n",
            "        native_seconds=time.perf_counter()-native_start;self.check(rc);assert done.value==1\n"
            "        assert self.lib.bcir_geo_refused()==0,'a product refused its workspace'\n",
        ),
        (
            "        for start in range(0,b,m):out.append(self.forward(np.ascontiguousarray(x[start:start+m])).copy())\n",
            "        for start in range(0,b,m):out.append(self.forward(np.ascontiguousarray(x[start:start+m])).copy())\n"
            "        assert self.lib.bcir_geo_refused()==0,'a product refused its workspace'\n",
        ),
    ],
}
patch = []
for name, edits in EDITS.items():
    p = work / name
    before = p.read_text()
    after = before
    for a, b in edits:
        assert after.count(a) == 1, (name, a[:60], after.count(a))
        after = after.replace(a, b)
    p.write_text(after)
    patch += difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True), f"a/{name}", f"b/{name}"
    )
(work.parent / (work.name + "-host-abi5.diff")).write_text("".join(patch))
print("".join(patch))
