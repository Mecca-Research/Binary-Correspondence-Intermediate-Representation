"""Matched PR816 before/after training, using the repository's pinned LM experiment.

Reuses train_comparison.py without editing production or the earlier harness.
The earlier train stage forces Torch to one thread even when --threads is given;
this adapter makes the requested compute thread count apply to both backends.
It also loads each revision's own Python binding, so ABI 1 and ABI 3 never mix.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("bcir", "bcir-openblas", "torch-eager", "torch-compiled"),
        required=True,
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--contract", default="v1")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=128)
    parser.add_argument(
        "--evaluation-windows",
        type=int,
        help="Bounded timing runs only; full quality runs omit this",
    )
    parser.add_argument("--task", default="lm", choices=("lm", "copy"))
    parser.add_argument("--lr-scale", type=float, default=0.25)
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        OMP_NUM_THREADS=str(args.threads),
        MKL_NUM_THREADS=str(args.threads),
        OPENBLAS_NUM_THREADS=str(args.threads),
        OPENBLAS_THREAD_TIMEOUT="4",
        TORCHINDUCTOR_COMPILE_THREADS="2",
    )
    sys.path.insert(0, str(repo))
    here = Path(__file__).resolve()
    # An explicit repository owns both the measurement source and the binding.
    source = repo / "docs/research/measurements/2026-10-11/train_comparison.py"
    spec = importlib.util.spec_from_file_location("pr816_training_source", source)
    tc = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = tc
    spec.loader.exec_module(tc)
    tc.REPO = repo
    sys.path.insert(0, str(repo))

    from bcir.hosted.models.native import NativeDecoder, parameter_layout

    build = tc.build_library(args.output / "library", "gcc")
    binding_parameters = inspect.signature(NativeDecoder).parameters

    def make_native(data, library, *, openblas=False, threads=1, contract="v1"):
        settings = {
            "batch_size": tc.BATCH,
            "context_length": tc.CONTEXT,
            "max_memory_bytes": 4 << 30,
        }
        if "threads" in binding_parameters:
            settings.update(threads=threads, contract=contract)
        elif contract != "v1" or (threads != 1 and not openblas):
            raise ValueError("this revision has no pool or alternate arithmetic contract")
        decoder = NativeDecoder(library, tc._spec(), **settings)
        flat = tc._load(data, "weights.f32", "f")
        decoder.load_state_dict(
            {name: flat[off : off + n] for name, _shape, off, n in parameter_layout(tc._spec())}
        )
        info = {"provider": "bcir-native", "contract": contract, "threads": threads}
        if openblas:
            path, config = tc._openblas(threads)
            decoder.use_cblas(path, symbol="scipy_cblas_sgemm")
            info.update(
                provider="scipy_cblas_sgemm",
                openblas=config,
                admission=getattr(decoder, "provider_admission", None),
            )
        return decoder, info

    tc._native = make_native
    # Both pinned versions compile sources afresh inside train(); reuse this checked build.
    tc.build_library = lambda *_args, **_kwargs: build
    train_source = inspect.getsource(tc.train)
    old_env = (
        'os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")'
    )
    assert train_source.count(old_env) == 1
    assert train_source.count("torch.set_num_threads(1)") == 1
    train_source = train_source.replace(
        old_env,
        "os.environ.update(OMP_NUM_THREADS=str(_COMPUTE_THREADS), MKL_NUM_THREADS=str(_COMPUTE_THREADS), OPENBLAS_NUM_THREADS=str(_COMPUTE_THREADS))",
    )
    train_source = train_source.replace(
        "torch.set_num_threads(1)", "torch.set_num_threads(_COMPUTE_THREADS)"
    )
    marker = "    rss_ready = _rss_mb()"
    assert train_source.count(marker) == 1
    train_source = train_source.replace(
        marker,
        "    _CAPTURED_STATE.update(nd=nd) if backend.startswith('bcir') else _CAPTURED_STATE.update(model=model, optimizer=opt)\n"
        + marker,
    )
    tc.__dict__["_CAPTURED_STATE"] = {}
    tc.__dict__["_COMPUTE_THREADS"] = args.threads
    exec(compile(train_source, str(here) + ":matched_threads", "exec"), tc.__dict__)

    # Keep one complete run directory per observation; preparation bytes are shared read-only.
    # train() writes only runs/*.json and lib/*, both redirected below.
    original_load = tc._load
    tc._load = lambda _data, name, code: original_load(args.data, name, code)
    original_eval = tc._eval_sets

    def evaluate_sets(_data, task):
        sets = original_eval(args.data, task)
        if args.evaluation_windows:
            n = args.evaluation_windows
            if n < tc.BATCH or n % tc.BATCH:
                raise ValueError("evaluation windows must be a positive multiple of batch")
            sets = {
                name: (x[:n], y[:n], mask[:n], [row for row in spans if row[0] < n])
                for name, (x, y, mask, spans) in sets.items()
            }
        return sets

    tc._eval_sets = evaluate_sets
    (args.output / "prepare.json").write_bytes((args.data / "prepare.json").read_bytes())
    for file in args.data.glob("*-test.json"):
        (args.output / file.name).write_bytes(file.read_bytes())

    signature = inspect.signature(tc.train).parameters
    kwargs = dict(lr_scale=args.lr_scale)
    if "threads" in signature:
        kwargs.update(threads=args.threads, contract=args.contract)
    elif args.backend.startswith("bcir"):
        # Pre-pool train() does not pass thread/contract arguments to _native().
        tc._native = lambda data, library, openblas=False: make_native(
            data, library, openblas=openblas, threads=args.threads, contract=args.contract
        )
    tc.train(args.output, args.task, args.backend, args.steps, args.eval_every, "gcc", **kwargs)

    records = list((args.output / "runs").glob("*.json"))
    if len(records) != 1:
        raise RuntimeError("one observation must have exactly one run record")
    record = json.loads(records[0].read_text())
    record.update(
        requested_compute_threads=args.threads,
        binding_supports_pool="threads" in binding_parameters,
        harness_adapter_sha256=hashlib.sha256(here.read_bytes()).hexdigest(),
        source_harness_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        evaluation_windows=args.evaluation_windows,
        native_pool_threads=args.threads
        if "threads" in binding_parameters and args.backend.startswith("bcir")
        else (1 if args.backend.startswith("bcir") else None),
    )
    if args.backend.startswith("torch"):
        import torch

        record["actual_torch_threads"] = torch.get_num_threads()
        assert torch.get_num_threads() == args.threads
    if args.save_checkpoint:
        checkpoint = args.output / "checkpoint.bin"
        state = tc.__dict__["_CAPTURED_STATE"]
        if "nd" in state:
            state["nd"].save(checkpoint)
        else:
            torch.save(
                {
                    "model": state["model"].state_dict(),
                    "optimizer": state["optimizer"].state_dict(),
                    "steps": args.steps,
                },
                checkpoint,
            )
        record["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        record["checkpoint_bytes"] = checkpoint.stat().st_size
    records[0].write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "record": str(records[0]),
                "median_step_ms": record["median_step_ms"],
                "threads": args.threads,
                "contract": args.contract,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
