"""GEO-2b level 1: the PR818 study's research overlays, as Codex's build_upgrade.py generates them
(byte-identical to the bundle's build/*.c), compiled against current main's tensor unit.

The overlay keeps its own trainer (the research head, its compact layouts, reconstruction tiers,
masked loss and the PR818 grafts); only bcir_tensor.c/bcir_tensor.h -- every kernel the trainer
calls -- come from origin/main. Flags are build_upgrade.py's compile_one, unchanged.
usage: python3 -I build_level1.py WORK_DIR INCLUDE_DIR [NAME ...]
"""

import hashlib, json, subprocess, sys
from pathlib import Path

work, inc = Path(sys.argv[1]), Path(sys.argv[2])
names = sys.argv[3:] or [
    "guard_compact",
    "guard_frame_ff_compact",
    "guard_layer_frame_ff",
    "query_guard_compact",
    "query_guard_frame_ff_compact",
    "query_guard_layer_frame_ff",
    "guard_hybrid_compact",
    "attention",
]
tensor = inc / "bcir_tensor.c"
for name in names:
    extra = []
    base = name
    if name.endswith("_probe"):
        base, extra = (
            name[: -len("_probe")],
            ["-DBCIR_DT_WORKSPACE_PROBE=1", "-DBCIR_DT_PANELS_PROBE=1"],
        )
    elif name.endswith("_nopanels"):
        base, extra = name[: -len("_nopanels")], ["-DBCIR_DT_NO_PANELS=1"]
    src = work / "build" / f"{base}.c"
    out = work / "build" / f"{name}.so"
    cmd = [
        "gcc",
        "-std=c11",
        "-O3",
        "-fPIC",
        "-shared",
        "-ffp-contract=off",
        "-fno-math-errno",
        "-pthread",
        "-I",
        str(inc),
        "-DGP_RANK=22",
        "-DGP_PROJECTOR=0",
        "-DGP_SYMMETRIC=0",
        "-DGP_HEADS=1",
        *extra,
        str(tensor),
        str(src),
        "-lm",
        "-o",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode:
        sys.exit(f"{name}: build failed\n{r.stderr}")
    (work / "build" / f"{name}-build-level1.json").write_text(
        json.dumps(
            {
                "command": cmd,
                "overlay_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                "tensor_sha256": hashlib.sha256(tensor.read_bytes()).hexdigest(),
                "tensor_header_sha256": hashlib.sha256(
                    (inc / "bcir_tensor.h").read_bytes()
                ).hexdigest(),
                "trainer_header_sha256": hashlib.sha256(
                    (inc / "bcir_decoder_train.h").read_bytes()
                ).hexdigest(),
                "stderr": r.stderr,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(name, "built", len(r.stderr.splitlines()), "warning line(s)", flush=True)
