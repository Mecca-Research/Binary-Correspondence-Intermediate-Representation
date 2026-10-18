"""The attention alone, one tree against another: attention_time.c built against each tree's
bcir_tensor.c (GCC, the trainer's flags), then the two run in turn round by round -- the order
reversed every other round -- at one and four threads, one JSON line per tree, round and count.

usage: attention_ab.py ROUNDS OUT.jsonl BASE_TREE NEW_TREE   (a tree: a runtime/c directory)
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    rounds, out = int(sys.argv[1]), Path(sys.argv[2])
    trees = {"base": sys.argv[3], "attn7": sys.argv[4]}
    with tempfile.TemporaryDirectory() as temporary:
        exe = {}
        for name, tree in trees.items():
            exe[name] = Path(temporary) / f"attention-{name}"
            subprocess.run(
                ["gcc", "-O3", "-ffp-contract=off", "-fno-math-errno", "-pthread", "-I", tree,
                 str(HERE / "attention_time.c"), f"{tree}/bcir_tensor.c", "-o", str(exe[name]),
                 "-lm"],
                check=True,
            )  # fmt: skip
        names = list(trees)
        with out.open("w", encoding="utf-8") as f:
            for r in range(rounds):
                for name in names if r % 2 == 0 else names[::-1]:
                    for threads in (1, 4):
                        run = subprocess.run([str(exe[name]), "30", str(threads)], check=True,
                                             capture_output=True, text=True)  # fmt: skip
                        rec = json.loads(run.stdout)
                        rec.update(arm=name, round=r + 1)
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
