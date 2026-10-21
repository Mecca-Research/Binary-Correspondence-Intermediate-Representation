"""PX-3b-2 (2026-10-21): this host's measured dispatch carried into K_BCIR's own candidates -- a
dispatch.py record's table (PX-3b-1's measurement) priced by `native_choose.measured_prices`, the
comparison decoder's K-step training pack planned on the AVX-512 profile carrying those prices,
hydrated by GEM, placed by its ExecutionPlanV1 and lowered by the C (`bcir_decoder_lower_pack`):
the plan's measured steps, the dispatch the plan states (`native_pack.plan_dispatch`), whether the
lowered words are the program `step_program` states under the carried picks, and whether the
lowered pack trains the state the chosen dispatch's program trains, bit for bit, from one seed.

  python3 carry.py MEASUREMENT.json OUT.json [--cc gcc]
"""

import argparse
import ctypes as ct
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))


def digest(nd) -> str:
    h = hashlib.sha256()
    for buf in ("weights", "moment1", "moment2"):
        h.update(ct.string_at(nd.buffers[buf], nd.state.plan.parameters * 4))
    h.update(repr((nd.state.step, nd.state.beta1_power, nd.state.beta2_power)).encode())
    return h.hexdigest()[:16]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("measurement", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--cc", default="gcc")
    a = p.parse_args()
    from bcir.frontends.models.decode import DecoderSpec
    from bcir.hosted.models.native import NativeAdamW, NativeDecoder
    from bcir.hosted.models.native_choose import (ChoiceTable, choose_dispatch, measured_prices,
                                                  trainer_shape)  # fmt: skip
    from bcir.hosted.models.native_launch import program_words, step_program
    from bcir.hosted.models.native_pack import pack_artifacts, plan_dispatch, training_pack
    from bcir.kbcir.cost import TargetProfile

    record = json.loads(a.measurement.read_text())
    # the record holds the table as an object; its one spelling is the canonical JSON
    table = ChoiceTable.from_json(json.dumps(record["table"], sort_keys=True, separators=(",", ":"),
                                             ensure_ascii=True))  # fmt: skip
    spec = DecoderSpec(260, 384, 8, 6, 1024, activation="silu_gate", n_kv_heads=4,
                       tied_embeddings=True)  # fmt: skip
    batch, context, k = 8, 128, record["k"]
    common = dict(batch_size=batch, context_length=context, threads=record["threads"],
                  contract=record["contract"])  # fmt: skip
    lib = NativeDecoder.build(a.out.parent / (a.out.stem + "-build"), cc=a.cc, pack=True)
    lowered = NativeDecoder(lib, spec, **common)
    shape = trainer_shape(lowered, steps=k, instances=True)
    if shape != table.shape:
        raise SystemExit(f"the table was measured at {table.shape}, this trainer is {shape}")
    decisions = lowered.program_decisions(steps=k)
    chosen = choose_dispatch(table, spec, shape)
    prices, left = measured_prices(table, spec, shape, **decisions)
    carried = {kind: alt for kind, alt in chosen.items() if kind not in left}
    t0 = time.perf_counter()
    tp = training_pack(spec, batch, context, **decisions)
    target = TargetProfile.x86_avx512()
    pack, plan_bytes, _, plan = pack_artifacts(
        tp, target=target.with_measured(prices) if prices else target)  # fmt: skip
    t1 = time.perf_counter()
    words = lowered.lower_pack(pack, plan_bytes)
    t2 = time.perf_counter()
    want = program_words(step_program(spec, batch, context, **decisions, dispatch=carried or None))
    # one seed: the lowered pack, and the chosen dispatch's program on a second trainer
    n = batch * context
    tokens = [(7 * i + 3) % 260 for i in range(n * k)]
    targets = [(7 * i + 4) % 260 for i in range(n * k)]
    opt = NativeAdamW(lr=1e-4)
    stated = NativeDecoder(lib, spec, **common)
    chosen_words = program_words(step_program(spec, batch, context, **decisions,
                                              dispatch=chosen or None))  # fmt: skip
    lowered.run_program(words, tokens, targets, steps=k, optimizer=opt)
    stated.run_program(chosen_words, tokens, targets, steps=k, optimizer=opt)
    states = {"lowered pack": digest(lowered), "chosen program": digest(stated)}
    lowered.close()
    stated.close()
    out = {
        "measurement": a.measurement.name,
        "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip(),  # fmt: skip
        "worktree_diff_sha256_16": hashlib.sha256(subprocess.run(
            ["git", "-C", str(REPO), "diff", "HEAD", "--", "runtime/c", "bcir"],
            capture_output=True).stdout).hexdigest()[:16],  # fmt: skip
        # the code that ran, by content: with an empty diff above, any commit whose trees are these
        "trees": {path: subprocess.run(["git", "-C", str(REPO), "rev-parse", f"HEAD:{path}"],
                                       capture_output=True, text=True).stdout.strip()
                  for path in ("bcir", "runtime/c")},  # fmt: skip
        "contract": record["contract"], "threads": record["threads"], "k": k,
        "chosen": chosen, "uncarried": list(left), "carried": carried,
        "measured_prices": None if prices is None else {
            "generation": prices.generation,
            "entries": [[e.op, e.count, e.alternative, e.q8] for e in prices.entries]},
        "claims": len(plan.steps), "lifetimes": len(plan.lifetimes),
        "measured_steps": sorted({s.candidate for s in plan.steps if "@" in s.candidate}),
        "measured_step_count": sum(1 for s in plan.steps if "@" in s.candidate),
        "plan_dispatch": sorted({f"instance {i} workers {w}" for i, w in plan_dispatch(plan).values()}),
        "words_are_the_carried_program": list(words) == list(want),
        "state_sha256_16": states,
        "one_state": len(set(states.values())) == 1,
        "seconds": {"plan_hydrate_place": round(t1 - t0, 3), "lower": round(t2 - t1, 4)},
    }  # fmt: skip
    print(json.dumps(out, indent=1))
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
