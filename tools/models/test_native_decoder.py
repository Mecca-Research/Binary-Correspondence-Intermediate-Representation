#!/usr/bin/env python3
"""Required native C vs independent PyTorch decoder training differential.

No downloads or accelerators. Every named gradient, tied/GQA, untied/MHA,
context-one, accumulation and exact resume are checked. Timings are observations,
never CI thresholds. --cblas explicitly selects an optional LP64 native provider.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from bcir.frontends.models.decode import DecoderSpec, next_token_logits, decode_with_kv_cache
from bcir.frontends.models.weights_io import write_q8_decoder, read_q8_decoder
from bcir.hosted.models.model import HostedLlama
from bcir.hosted.models.native import NativeAdamW, NativeDecoder
from tools.models.run_hosted_model_gate import _build_c_cli


def _copy_reference(native, model):
    native.load_state_dict(
        {name: value.detach().flatten().tolist() for name, value in model.state_dict().items()}
    )


def _parameter_error(native, model, *, gradients=False):
    values, errors = native.state_dict(gradients=gradients), {}
    for name, param in model.named_parameters():
        actual = torch.tensor(values[name], dtype=torch.float32).reshape(param.shape)
        expected = param.grad if gradients else param.detach()
        assert expected is not None, name
        errors[name] = float((actual - expected).abs().max())
        torch.testing.assert_close(
            actual,
            expected,
            atol=4e-6 if gradients else 5e-5,
            rtol=2e-3 if gradients else 3e-3,
            msg=name,
        )
    assert set(values) == set(errors), "every native parameter needs a reference gradient"
    return max(errors.values()), errors


def _snapshot(native):
    return tuple(
        ctypes.string_at(native.buffers[n], native.state.plan.parameters * 4)
        for n in ("weights", "moment1", "moment2")
    )


def _reference_optimizer(model, options):
    return torch.optim.AdamW(
        model.parameters(),
        lr=options.lr,
        betas=(options.beta1, options.beta2),
        eps=options.epsilon,
        weight_decay=options.weight_decay,
        foreach=False,
        fused=False,
    )


def _reference_step(model, optimizer, x, y, options, lr):
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    if options.grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip)
    optimizer.param_groups[0]["lr"] = lr
    optimizer.step()
    return float(loss.detach())


def _native(library, spec, batch, length, cblas, symbol):
    native = NativeDecoder(library, spec, batch_size=batch, context_length=length)
    if cblas:
        native.use_cblas(cblas, symbol=symbol)
    return native


def _deployment(native, directory, prompt):
    weights = native.decoder_weights()
    logits = torch.tensor(native.forward(prompt + prompt)).reshape(2, len(prompt), -1)[0, -1]
    oracle = torch.tensor(next_token_logits(prompt, native.spec, weights))
    torch.testing.assert_close(logits, oracle, atol=4e-5, rtol=3e-4)
    path = directory / "trained.bcirq8"
    hashes = {
        "model": hashlib.sha256(bytes(native.buffers["weights"])).hexdigest(),
        "config": hashlib.sha256(
            json.dumps(asdict(native.spec), sort_keys=True).encode()
        ).hexdigest(),
        "tokenizer": hashlib.sha256(b"synthetic-integer-token-ids-v1").hexdigest(),
    }
    write_q8_decoder(
        path,
        native.spec,
        weights,
        source_hashes=hashes,
        tokenizer_ids={"bos": 1, "eos": 2, "pad": 0},
        context_length=native.context_length,
    )
    spec, quantized, _ = read_q8_decoder(path)
    executable = directory / ("bcir-llama.exe" if os.name == "nt" else "bcir-llama")
    _build_c_cli(executable)
    result = subprocess.run(
        [
            str(executable),
            "--model",
            str(path),
            "--prompt-ids",
            ",".join(map(str, prompt)),
            "--max-new",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    expected = decode_with_kv_cache(prompt, spec, quantized, max_new=1)
    assert json.loads(result.stdout)["generated_ids"] == expected
    return {"generated_ids": expected, "q8_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _case(library, directory, *, tied, kvheads, layers, length, cblas, symbol):
    directory.mkdir(parents=True, exist_ok=True)
    spec = DecoderSpec(
        32, 16, 4, layers, 24, activation="silu_gate", n_kv_heads=kvheads, tied_embeddings=tied
    )
    native = _native(library, spec, 2, length, cblas, symbol)
    torch.manual_seed(712 + layers + length)
    model = HostedLlama(spec, context_length=length)
    _copy_reference(native, model)
    x = torch.tensor([[1 + i % 4 for i in range(length)], [3 + i % 4 for i in range(length)]])
    y = (x + 1) % spec.vocab_size
    xf, yf = x.flatten().tolist(), y.flatten().tolist()
    logits, loss = model(x, y)
    actual = torch.tensor(native.forward(xf)).reshape(logits.shape)
    torch.testing.assert_close(actual, logits.detach(), atol=3e-6, rtol=1e-4)
    initial = native.backward(xf, yf)
    assert abs(initial - float(loss.detach())) < 1e-6
    loss.backward()
    grad_error, family_errors = _parameter_error(native, model, gradients=True)
    saved = native.state_dict(gradients=True)
    native.backward(xf, yf, accumulate=True)
    for name, value in native.state_dict(gradients=True).items():
        torch.testing.assert_close(
            torch.tensor(value), 2 * torch.tensor(saved[name]), atol=1e-6, rtol=1e-5
        )
    native.update(NativeAdamW(lr=0.0), gradient_scale=0.5)
    assert native.state.step == 1
    native = _native(library, spec, 2, length, cblas, symbol)
    _copy_reference(native, model)
    options = NativeAdamW(lr=0.01, weight_decay=0.03, grad_clip=0.2)
    optimizer = _reference_optimizer(model, options)
    rates = [options.lr * (1 - 0.01 * i) for i in range(20)]
    events = native.run(xf * 20, yf * 20, steps=20, optimizer=options, learning_rates=rates)
    reference = [_reference_step(model, optimizer, x, y, options, rate) for rate in rates]
    assert max(abs(e["loss"] - r) for e, r in zip(events, reference)) < 3e-5
    assert events[-1]["loss"] < initial * 0.6
    weight_error, _ = _parameter_error(native, model)
    path = directory / "state.bcirdt"
    native.save(path)
    resumed = NativeDecoder.resume(library, path)
    if cblas:
        resumed.use_cblas(cblas, symbol=symbol)
    assert native.run(xf * 3, yf * 3, steps=3, optimizer=options) == resumed.run(
        xf * 3, yf * 3, steps=3, optimizer=options
    )
    assert _snapshot(native) == _snapshot(resumed)
    assert (native.state.beta1_power, native.state.beta2_power) == (
        resumed.state.beta1_power,
        resumed.state.beta2_power,
    )
    # Changing a future token cannot affect the first-position logits.
    changed = list(xf)
    changed[length - 1] = (changed[length - 1] + 1) % spec.vocab_size
    if length > 1:
        assert native.forward(xf)[: spec.vocab_size] == native.forward(changed)[: spec.vocab_size]
    deployment = _deployment(native, directory, xf[:length])
    return {
        "spec": asdict(spec),
        "context": length,
        "parameters": native.state.plan.parameters,
        "initial_loss": initial,
        "last_loss": events[-1]["loss"],
        "max_gradient_error": grad_error,
        "gradient_errors": family_errors,
        "max_weight_error": weight_error,
        "losses": [e["loss"] for e in events],
        "exact_resume": True,
        "deployment": deployment,
    }


def _accumulation(library, cblas, symbol):
    spec = DecoderSpec(32, 16, 4, 2, 24, activation="silu_gate", n_kv_heads=1)
    whole = _native(library, spec, 2, 4, cblas, symbol)
    micro = _native(library, spec, 1, 4, cblas, symbol)
    micro.load_state_dict(whole.state_dict())
    x, y = [1, 2, 3, 4, 3, 4, 5, 6], [2, 3, 4, 5, 4, 5, 6, 7]
    options = NativeAdamW(lr=0.003, weight_decay=0.03, grad_clip=0.2)
    one = whole.run(x * 5, y * 5, steps=5, optimizer=options)
    two = micro.run(x * 5, y * 5, steps=5, optimizer=options, gradient_accumulation=2)
    assert max(abs(a["loss"] - b["loss"]) for a, b in zip(one, two)) < 2e-6
    error = max(abs(a - b) for a, b in zip(whole.buffers["weights"], micro.buffers["weights"]))
    assert error < 5e-6, error
    return {"micro_batches": 2, "updates": 5, "max_weight_error": error}


def _benchmark(library, cblas, symbol):
    spec = DecoderSpec(
        128, 64, 4, 2, 128, activation="silu_gate", n_kv_heads=2, tied_embeddings=False
    )
    native = _native(library, spec, 2, 16, cblas, symbol)
    torch.manual_seed(211)
    model = HostedLlama(spec, context_length=16)
    _copy_reference(native, model)
    x = torch.tensor([[1 + i % 16 for i in range(16)], [9 + i % 16 for i in range(16)]])
    y = (x + 1) % spec.vocab_size
    xf, yf = x.flatten().tolist(), y.flatten().tolist()
    options = NativeAdamW(lr=0.001, weight_decay=0.03, grad_clip=1.0)
    optimizer = _reference_optimizer(model, options)
    native.run(xf, yf, steps=1, optimizer=options)
    _reference_step(model, optimizer, x, y, options, options.lr)
    started = time.perf_counter()
    native.run(xf * 8, yf * 8, steps=8, optimizer=options)
    native_time = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(8):
        _reference_step(model, optimizer, x, y, options, options.lr)
    reference_time = time.perf_counter() - started
    # Default fused SDPA can amplify rounding noise in near-zero Adam coordinates;
    # correctness is established separately with the math SDPA trajectory above.
    error = max(
        float((torch.tensor(native.state_dict()[name]).reshape(p.shape) - p.detach()).abs().max())
        for name, p in model.named_parameters()
    )
    return {
        "spec": asdict(spec),
        "parameters": native.state.plan.parameters,
        "batch": 2,
        "context": 16,
        "updates": 8,
        "provider": native.backend,
        "native_seconds": native_time,
        "pytorch_seconds": reference_time,
        "eager_over_native": reference_time / native_time,
        "max_weight_error": error,
        "reference": "PyTorch eager, default SDPA, torch.optim.AdamW foreach=False fused=False",
        "scope": "CPU FP32 full forward/backward/global clip/AdamW; native includes input marshalling and events",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cblas")
    parser.add_argument("--cblas-symbol", default="cblas_sgemm")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    library = NativeDecoder.build(args.output_dir / "native")
    with sdpa_kernel(SDPBackend.MATH):
        cases = [
            _case(
                library,
                args.output_dir / f"case-{i}",
                cblas=args.cblas,
                symbol=args.cblas_symbol,
                **settings,
            )
            for i, settings in enumerate(
                (
                    {"tied": True, "kvheads": 1, "layers": 2, "length": 4},
                    {"tied": False, "kvheads": 4, "layers": 1, "length": 4},
                    {"tied": False, "kvheads": 2, "layers": 2, "length": 1},
                )
            )
        ]
        accumulation = _accumulation(library, args.cblas, args.cblas_symbol)
    report = {
        "schema": "bcir.native_decoder_gate.v1",
        "torch": torch.__version__,
        "cases": cases,
        "accumulation": accumulation,
        "benchmark": _benchmark(library, args.cblas, args.cblas_symbol),
        "hardware": "CPU only; no accelerator/backward/distributed validation",
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print("PASS native decoder every-gradient/AdamW/accumulation/resume/train-to-Q8-C differential")
    print(json.dumps(report["benchmark"], sort_keys=True))


if __name__ == "__main__":
    main()
