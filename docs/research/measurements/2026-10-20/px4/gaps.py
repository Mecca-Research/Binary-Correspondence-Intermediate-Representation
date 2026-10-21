"""PX-4 (2026-10-20): the trace corpus's placements against their bound -- every placed pack run of
`train.plan.place.mismatch` (`native_decoder_fixtures.place_mismatch`: the 356 pack runs, each
placed by `native_pack.place_by_lifetime` and run on a trainer lending exactly the placement), its
extent's gap to G26's alignment-aware concurrent-live bound over the values it places, and whether
the exact search proved the layout optimal within its budget.

The gaps are read beside the row, never counted in it: a placement above its bound is a gap the
search's budget leaves (TMSAO-2 at best), not a divergence; one below it would be a defect (two
live values in one byte), and the row counts that.

  python3 gaps.py OUT.json
"""

import argparse
import collections
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", type=Path)
    a = p.parse_args()
    from bcir.tests import native_decoder_fixtures as nf

    with tempfile.TemporaryDirectory() as tmp:
        path = nf.pack_library(Path(tmp) / "pack")
        if path is None:
            raise SystemExit("no C compiler: the placed runs need the pack library")
        bad, runs, notes, gaps = nf.place_mismatch(path)
    hist = collections.Counter(gap for gap, _ in gaps)
    out = {"runs": runs, "bad": bad, "notes": notes[:5],
           "at_bound": hist.get(0, 0),
           "at_bound_proved": sum(1 for gap, proved in gaps if gap == 0 and proved),
           "above_bound": sum(n for gap, n in hist.items() if gap > 0),
           "max_gap_bytes": max(hist) if hist else 0,
           "proved": sum(1 for _, proved in gaps if proved),
           "gap_histogram_bytes": {str(gap): n for gap, n in sorted(hist.items())}}  # fmt: skip
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "notes"}))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
