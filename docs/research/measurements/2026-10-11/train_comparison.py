"""Train one ~10M-parameter decoder on BCIR's native C trainer and on PyTorch, and compare.

Run with an environment that has NumPy and CPU PyTorch (experiment dependencies, never BCIR's);
the BCIR side needs only a C compiler. Stages run serially, and every `train` is its own
process so its peak resident memory is its own:

  prepare   the corpus (the repository's tracked Markdown at a pinned commit, bytes), the
            copy, associative-recall and addition tasks with held-out problems, identical
            initial weights and one batch schedule per task, as long as its step budget -> DIR;
            --seed draws another replicate's initial weights and schedules (the corpus, its
            splits and the held-out problems stay), recorded in prepare.json when it is not
            the default
  train     one backend on one task -> DIR/runs/<task>-<backend>.json
  identity  BCIR's training state, bit for bit, across ISA variants, compilers, a repeat,
            thread counts (--threads) and contracts (--contract, --variants)
  baseline  the pre-update kernels against these, same steps: identical state, time per step
  kernels   the same, kernel by kernel: every GEMM shape of a step on every ISA variant,
            attention forward and backward, AdamW; identical outputs, time per call
  breakdown where one step's time goes: BCIR's step by kernel family (per compiler, and with
            OpenBLAS), eager PyTorch's by phase and by operator family
  bcir-breakdown
            one BCIR step by kernel family through the entry points the trainer calls under
            the run's pool and contracts (--threads, --contract); no PyTorch needed
  context   time and memory against context at 1,024 tokens per step: one attention layer
            forward and backward on BCIR and on PyTorch, and the decoder's step at T=64..1024
            on BCIR, BCIR with OpenBLAS and eager PyTorch, each in its own process; and the
            step at T=128 on two and four threads where the backend has them
  summarize every run -> DIR/summary.json and a Markdown table

  python train_comparison.py prepare --output DIR
  python train_comparison.py train --data DIR --task lm --backend bcir
  python train_comparison.py summarize --data DIR

Tasks: lm (next-byte prediction on the corpus, scored by cross-entropy and held-out word
completion), copy (eight letters, then the same eight: held-out exact match), recall
(in-context key/value retrieval: six pairs, then a query) and add (three-digit addition, the sum
least-significant digit first). The report trains lm and copy on every backend; recall and add
are kept for the configurations it records. The structured tasks mask unpredictable bytes out
of the loss through their targets.

Backends: bcir (BCIR's own C kernels: the widest bit-identical ISA variant the CPU runs, no
FMA), bcir-openblas (the same trainer with OpenBLAS sgemm behind its CBLAS seam), torch-eager
(HostedLlama, torch.optim.AdamW defaults), torch-compiled (the same modules through
torch.compile, fused AdamW), and torch-eager-bf16 and torch-compiled-bf16 (the same two under
PyTorch's CPU mixed precision, torch.autocast to bfloat16 around the forward and the loss, the
weights, gradients and moments in FP32: the matched baseline of BCIR's bf16 contract, NDT-BF16).
All run on CPU from the same weights, batches and learning rates, in FP32 unless named bf16 (or
under a BCIR contract that says otherwise). This is a measurement harness, not a production
runtime or a CI timing gate.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import math
import os
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
from array import array
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "bcir").is_dir())
sys.path.insert(0, str(REPO))

CORPUS_COMMIT = "75d50d4f6e280e35a25032024edffd92f25c52f2"  # main when the report was measured
# bytes, a document separator, evaluation padding (never trained), the addition task's
# don't-care target (an operand digit is unpredictable, so its target is this constant)
SEP, PAD, IGN, VOCAB = 256, 257, 258, 260
# The structured tasks. Every problem has a fixed length in bytes; `masked` holds the offsets of
# the bytes nothing before them predicts (their training targets are IGN), `answer` the offsets
# the evaluation scores.
TASKS = {
    # eight random lowercase letters, '=', the same eight letters
    "copy": {"length": 18, "masked": frozenset(range(8)), "answer": tuple(range(9, 18))},
    # six key/value pairs (distinct lowercase keys, digit values), ':', a query key, its value
    "recall": {"length": 16, "masked": frozenset((*range(12), 13)), "answer": (14, 15)},
    # aaa+bbb= then the sum's four digits least significant first
    "add": {"length": 13, "masked": frozenset((0, 1, 2, 4, 5, 6)), "answer": (8, 9, 10, 11, 12)},
}
SPEC = dict(vocab=VOCAB, width=384, heads=8, kvheads=4, layers=6, ff=1024)
BATCH, CONTEXT = 8, 128
OPT = dict(peak=2e-3, floor=2e-4, warmup=50, beta1=0.9, beta2=0.95, eps=1e-8, decay=0.1, clip=1.0)
SEED = 20261011
BACKENDS = ("bcir", "bcir-openblas", "torch-eager", "torch-compiled", "torch-eager-bf16",
            "torch-compiled-bf16")  # fmt: skip


def _spec():
    from bcir.frontends.models.decode import DecoderSpec

    return DecoderSpec(
        SPEC["vocab"], SPEC["width"], SPEC["heads"], SPEC["layers"], SPEC["ff"],
        activation="silu_gate", n_kv_heads=SPEC["kvheads"], tied_embeddings=True,
    )  # fmt: skip


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(*args) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True
    ).stdout


def _rate(step: int, steps: int) -> float:
    if step < OPT["warmup"]:
        return OPT["peak"] * (step + 1) / OPT["warmup"]
    span = max(1, steps - OPT["warmup"])
    phase = min(1.0, (step - OPT["warmup"]) / span)
    return OPT["floor"] + 0.5 * (OPT["peak"] - OPT["floor"]) * (1 + math.cos(math.pi * phase))


# --- prepare ---------------------------------------------------------------------------------
def _corpus():
    """Every tracked Markdown file under docs/ and training/ at CORPUS_COMMIT (not this
    experiment's own measurements), split by a hash of its path: 90% train, 5% validation,
    5% test. Documents are bytes joined by SEP."""
    paths = [
        p
        for p in _git("ls-tree", "-r", "--name-only", CORPUS_COMMIT).splitlines()
        if p.endswith(".md")
        and (p.startswith("docs/") or p.startswith("training/"))
        and not p.startswith("docs/research/measurements/")
    ]
    splits = {"train": [], "validation": [], "test": []}
    for path in sorted(paths):
        raw = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{CORPUS_COMMIT}:{path}"],
            check=True,
            capture_output=True,
        ).stdout.replace(b"\r\n", b"\n")
        bucket = int(_sha(path.encode())[:8], 16) % 20
        splits["test" if bucket == 0 else "validation" if bucket == 1 else "train"].append(raw)
    out = {}
    for name, docs in splits.items():
        stream = array("H")
        for doc in docs:
            stream.append(SEP)
            stream.extend(doc)
        out[name] = (stream, len(docs))
    return out, len(paths)


def _problem(a: int, b: int) -> bytes:
    """`aaa+bbb=` and the sum's four digits least significant first, then a newline: every
    problem is 13 bytes, so the answer is a fixed-width field."""
    return f"{a:03d}+{b:03d}={f'{a + b:04d}'[::-1]}\n".encode()


def _copy_text(letters: str) -> bytes:
    """Eight letters, '=', the same eight letters, then a newline: 18 bytes."""
    return f"{letters}={letters}\n".encode()


def _recall_text(pairs: str, query: str) -> bytes:
    """`k1v1...k6v6:` then the query key and its value, then a newline: 16 bytes."""
    return f"{pairs}:{query}{pairs[pairs.index(query) + 1]}\n".encode()


def _target(task: str, stream, position: int) -> int:
    """The training target after `position`: the next token, except that in a structured task's
    stream (whole problems from offset zero) a byte nothing before it predicts -- an operand
    digit, a key or a value of the pairs, the query key -- is replaced by IGN. That is
    prompt-loss masking through the targets, so every backend trains on the same streams; the
    model learns to answer IGN there and the loss concentrates on the answer and separators."""
    shape = TASKS.get(task)
    if shape and (position + 1) % shape["length"] in shape["masked"]:
        return IGN
    return stream[position + 1]


def _structured(draw, text):
    """2,000 held-out problems from `draw`, then a stream of at least two million bytes of
    problems `draw` makes that are not among them."""
    test = set()
    while len(test) < 2000:
        test.add(draw())
    stream = array("H")
    while len(stream) < 2_000_000:
        problem = draw()
        if problem not in test:
            stream.extend(text(*problem))
    return stream, sorted(test)


def _copy():
    rng = random.Random(SEED + 4)
    return _structured(
        lambda: ("".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=8)),), _copy_text
    )


def _recall():
    rng = random.Random(SEED + 3)

    def draw():
        keys = rng.sample("abcdefghijklmnopqrstuvwxyz", 6)
        pairs = "".join(f"{k}{rng.randrange(10)}" for k in keys)
        return pairs, rng.choice(keys)

    return _structured(draw, _recall_text)


def _addition():
    rng = random.Random(SEED + 1)
    return _structured(lambda: (rng.randrange(1000), rng.randrange(1000)), _problem)


def _windows(stream, count, rng):
    """`count` training windows of CONTEXT+1 tokens at random offsets."""
    return [rng.randrange(len(stream) - CONTEXT - 1) for _ in range(count)]


def _initial_weights(spec, seed=SEED):
    """The native trainer's own initialization: norms one, every matrix N(0, 0.02)."""
    from bcir.hosted.models.native import parameter_layout

    rng = random.Random(seed)
    flat = array("f")
    for name, _shape, _offset, count in parameter_layout(spec):
        if "layernorm" in name or name == "model.norm.weight":
            flat.extend([1.0] * count)
        else:
            flat.extend(rng.gauss(0.0, 0.02) for _ in range(count))
    return flat


def prepare(out: Path, budgets: dict, seed: int = SEED):
    out.mkdir(parents=True, exist_ok=True)
    spec = _spec()
    corpus, documents = _corpus()
    structured = {"add": _addition(), "recall": _recall(), "copy": _copy()}
    weights = _initial_weights(spec, seed)
    (out / "weights.f32").write_bytes(weights.tobytes())
    rng = random.Random(seed + 2)
    record = {
        "corpus_commit": CORPUS_COMMIT,
        "spec": SPEC,
        "batch": BATCH,
        "context": CONTEXT,
        "optimizer": OPT,
        "parameters": len(weights),
        "weights_sha256": _sha(weights.tobytes()),
        "documents": documents,
        "tasks": {},
    }
    if seed != SEED:  # the default seed's prepare.json stays the 10-11 record's
        record["seed"] = seed
    for name, (stream, docs) in corpus.items():
        (out / f"lm-{name}.u16").write_bytes(stream.tobytes())
        record["tasks"].setdefault("lm", {})[name] = {
            "tokens": len(stream), "documents": docs, "sha256": _sha(stream.tobytes()),
        }  # fmt: skip
    formats = {
        "add": "aaa+bbb= then the sum's four digits least-significant first, newline",
        "recall": "six key/value pairs (distinct lowercase keys, digit values), ':', a query "
        "key, its value, newline",
        "copy": "eight random lowercase letters, '=', the same eight letters, newline",
    }
    for task, (stream, test) in structured.items():
        (out / f"{task}-train.u16").write_bytes(stream.tobytes())
        record["tasks"][task] = {
            "train": {"tokens": len(stream), "sha256": _sha(stream.tobytes())},
            "test_problems": len(test),
            "format": formats[task],
            "targets": "the next byte; IGN where nothing before it predicts it (prompt-loss "
            "masking): the copied letters, an operand digit, a pair's key or value, the query",
            "metrics": "cross-entropy and accuracy over the answer bytes; teacher-forced "
            "exact match (every answer byte right, which is greedy decoding's answer)",
        }
        (out / f"{task}-test.json").write_text(json.dumps(test))
    for task, stream in (
        ("lm", corpus["train"][0]), ("add", structured["add"][0]),
        ("recall", structured["recall"][0]), ("copy", structured["copy"][0]),
    ):  # fmt: skip
        offsets = array("I", _windows(stream, budgets[task] * BATCH, rng))
        (out / f"{task}-schedule.u32").write_bytes(offsets.tobytes())
        record["tasks"][task]["schedule_sha256"] = _sha(offsets.tobytes())
        record["tasks"][task]["steps"] = budgets[task]
    (out / "prepare.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"prepared": str(out), "parameters": len(weights)}), flush=True)


# --- shared training pieces ------------------------------------------------------------------
def _load(data: Path, name: str, code: str):
    values = array(code)
    values.frombytes((data / name).read_bytes())
    return values


def _batch(stream, offsets, step, task):
    xs, ys = [], []
    for offset in offsets[step * BATCH : (step + 1) * BATCH]:
        xs.extend(stream[offset : offset + CONTEXT])
        if task in TASKS:
            ys.extend(_target(task, stream, offset + i) for i in range(CONTEXT))
        else:
            ys.extend(stream[offset + 1 : offset + CONTEXT + 1])
    return xs, ys


def _word_spans(x, y):
    """Word completion on held-out text: for every word of four or more ASCII letters that
    starts and ends inside a window, the target positions of its letters after the first, so
    exact match is the share of such words whose every remaining letter the model predicts
    (teacher-forced, which is greedy completion from the word's first letter)."""

    def letter(v):
        return 65 <= v <= 90 or 97 <= v <= 122

    spans = []
    for w in range(len(y)):
        row, i = y[w].tolist(), 0
        while i < len(row):
            if not letter(row[i]):
                i += 1
                continue
            j = i
            while j < len(row) and letter(row[j]):
                j += 1
            # starts after a non-letter (the input byte at i is the target before it) and ends
            # before the window does
            if j - i >= 4 and not letter(int(x[w][i])) and j < len(row):
                spans.append((w, i + 1, j - 1))
            i = j
    return spans


def _eval_sets(data: Path, task: str):
    """Fixed evaluation batches: (inputs, targets, mask, problems) with problems the answer
    spans of a structured task's test set ((window, first, last) target positions)."""
    import numpy as np

    sets = {}
    if task == "lm":
        for split, count in (("validation", 64), ("test", 128)):
            stream = _load(data, f"lm-{split}.u16", "H")
            stride = (len(stream) - CONTEXT - 1) // count
            xs, ys = [], []
            for i in range(count):
                window = stream[i * stride : i * stride + CONTEXT + 1]
                xs.append(window[:-1])
                ys.append(window[1:])
            x, y = np.array(xs, np.int64), np.array(ys, np.int64)
            sets[split] = (x, y, np.ones_like(x, bool), _word_spans(x, y))
        return sets
    problems = [tuple(p) for p in json.loads((data / f"{task}-test.json").read_text())]
    text_of = {"add": _problem, "recall": _recall_text, "copy": _copy_text}[task]
    answer = TASKS[task]["answer"]
    windows, spans, current, marks = [], [], [], []
    for problem in problems:
        text = list(text_of(*problem))
        if len(current) + len(text) > CONTEXT + 1:
            windows.append(current + [PAD] * (CONTEXT + 1 - len(current)))
            spans.extend((len(windows) - 1, f, l) for f, l in marks)
            current, marks = [], []
        start = len(current)
        current.extend(text)
        # the target positions that predict the answer bytes (a target is the next byte)
        marks.append((start + answer[0] - 1, start + answer[-1] - 1))
    if current:
        windows.append(current + [PAD] * (CONTEXT + 1 - len(current)))
        spans.extend((len(windows) - 1, f, l) for f, l in marks)
    while len(windows) % BATCH:
        windows.append([PAD] * (CONTEXT + 1))
    w = np.array(windows, np.int64)
    x, y = w[:, :-1], w[:, 1:]
    mask = np.zeros_like(y, bool)  # cross-entropy and accuracy over the answer tokens
    for window, first, last in spans:
        mask[window, first : last + 1] = True
    sets["test"] = (x, y, mask, spans)
    return sets


def _metrics(logits, y, mask, spans):
    import numpy as np

    logits = logits.astype(np.float64)
    top = logits.max(-1, keepdims=True)
    lse = top[..., 0] + np.log(np.exp(logits - top).sum(-1))
    picked = np.take_along_axis(logits, y[..., None], -1)[..., 0]
    ce = (lse - picked)[mask]
    pred = logits.argmax(-1)
    hit = (pred == y)[mask]
    out = {"cross_entropy": float(ce.mean()), "bits_per_byte": float(ce.mean() / math.log(2)),
           "accuracy": float(hit.mean()), "tokens": int(mask.sum())}  # fmt: skip
    if spans:
        # teacher-forced: every answer position right is exactly greedy decoding's answer
        hits = [pred[w, f : l + 1] == y[w, f : l + 1] for w, f, l in spans]
        out["exact_match"] = sum(bool(h.all()) for h in hits) / len(hits)
        out["answer_accuracy"] = float(np.concatenate(hits).mean())
        out["problems"] = len(hits)
    return out


def _rss_mb(field: str = "VmRSS") -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1]) / 1024
    return float("nan")


# NativeDecoder.build's optimization flags; before this update it built without -fno-math-errno
FLAGS = ("-O3", "-ffp-contract=off", "-fno-math-errno")
BASE_FLAGS = ("-O3", "-ffp-contract=off")
WARNINGS = ("-Wall", "-Wextra", "-Wpedantic", "-Werror")


def build_library(out: Path, cc: str, defines=(), sources: Path | None = None, flags=FLAGS) -> Path:
    """The native trainer as NativeDecoder.build compiles it, with optional defines and an
    optional source directory and flags (the baseline stage builds the pre-update kernels as
    they shipped)."""
    src = sources or (REPO / "runtime" / "c")
    out.mkdir(parents=True, exist_ok=True)
    target = out / "bcir_decoder_train.so"
    command = [cc, "-std=c11", *flags, *WARNINGS, "-DBCIR_DT_BUILD_SHARED", "-shared", "-fPIC",
               *defines, "-I", str(src), str(src / "bcir_tensor.c"),
               str(src / "bcir_decoder_train.c"), "-o", str(target), "-lm", "-pthread"]  # fmt: skip
    subprocess.run(command, check=True, capture_output=True, timeout=300)
    return target


def _openblas(threads: int = 1):
    import scipy_openblas32

    # OpenBLAS's worker threads spin after each call for 2^OPENBLAS_THREAD_TIMEOUT cycles before
    # they sleep (2^28 by default, about a tenth of a second): beside the trainer's pool they
    # burn the cores the attention, the update and the norms run on between two GEMMs. Read at
    # load, so it is set before the library loads; a value in the environment already wins.
    os.environ.setdefault("OPENBLAS_THREAD_TIMEOUT", "4")
    path = Path(scipy_openblas32.get_lib_dir()) / "libscipy_openblas.so"
    lib = ctypes.CDLL(str(path))
    lib.scipy_openblas_set_num_threads(threads)
    lib.scipy_openblas_get_config.restype = ctypes.c_char_p
    config = lib.scipy_openblas_get_config().decode()
    return path, f"{config} THREAD_TIMEOUT={os.environ['OPENBLAS_THREAD_TIMEOUT']}"


def _native(data: Path, library: Path, *, openblas=False, threads=1, contract="v1"):
    from bcir.hosted.models.native import NativeDecoder, parameter_layout

    spec = _spec()
    nd = NativeDecoder(
        library, spec, batch_size=BATCH, context_length=CONTEXT, max_memory_bytes=4 << 30,
        threads=threads, contract=contract,
    )  # fmt: skip
    flat = _load(data, "weights.f32", "f")
    nd.load_state_dict({name: flat[off : off + n] for name, _s, off, n in parameter_layout(spec)})
    info = {"provider": "portable-c", "contract": contract, "threads": threads}
    if openblas:
        path, config = _openblas(threads)
        nd.use_cblas(path, symbol="scipy_cblas_sgemm")
        info = {"provider": "scipy_cblas_sgemm", "openblas": config, "contract": contract,
                "threads": threads, "admission": nd.provider_admission}  # fmt: skip
    return nd, info


def _state_digest_native(nd) -> str:
    digest = hashlib.sha256()
    for name in ("weights", "moment1", "moment2"):
        digest.update(ctypes.string_at(nd.buffers[name], nd.state.plan.parameters * 4))
    return digest.hexdigest()


def _torch_model(data: Path):
    import torch

    from bcir.hosted.models.model import HostedLlama
    from bcir.hosted.models.native import parameter_layout

    spec = _spec()
    model = HostedLlama(spec)
    flat = _load(data, "weights.f32", "f")
    state = {}
    for name, shape, off, n in parameter_layout(spec):
        state[name] = torch.tensor(flat[off : off + n], dtype=torch.float32).reshape(shape)
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    model.load_state_dict(state, strict=True)
    return model


def _state_digest_torch(model, optimizer) -> str:
    digest = hashlib.sha256()
    for _name, param in model.state_dict().items():
        digest.update(param.detach().contiguous().numpy().tobytes())
    for group in optimizer.param_groups:
        for param in group["params"]:
            state = optimizer.state.get(param, {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    digest.update(state[key].contiguous().numpy().tobytes())
    return digest.hexdigest()


# --- train -----------------------------------------------------------------------------------
def train(
    data: Path, task: str, backend: str, steps: int | None, eval_every: int, cc: str,
    lr_scale: float = 1.0, threads: int = 1, contract: str = "v1",
):  # fmt: skip
    import numpy as np

    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    record = json.loads((data / "prepare.json").read_text())
    steps = steps or record["tasks"][task]["steps"]
    stream = _load(data, f"{task}-train.u16", "H")
    offsets = _load(data, f"{task}-schedule.u32", "I")
    assert len(offsets) >= steps * BATCH, "the schedule is shorter than the run"
    sets = _eval_sets(data, task)
    rates = [lr_scale * _rate(s, steps) for s in range(steps)]
    started = time.perf_counter()
    rss_start = _rss_mb()
    info: dict = {"backend": backend, "task": task, "steps": steps, "threads": threads,
                  "contract": contract, "lr_scale": lr_scale,
                  "seed": record.get("seed", SEED)}  # fmt: skip
    compile_s = None
    if backend.startswith("bcir"):
        from bcir.hosted.models.native import NativeAdamW

        library = build_library(data / "lib" / cc.replace("/", "_"), cc)
        nd, provider = _native(data, library, openblas=backend == "bcir-openblas",
                               threads=threads, contract=contract)  # fmt: skip
        info.update(provider, compiler=_compiler(cc), library_sha256=_sha(library.read_bytes()))
        opt = NativeAdamW(
            lr=OPT["peak"], beta1=OPT["beta1"], beta2=OPT["beta2"], epsilon=OPT["eps"],
            weight_decay=OPT["decay"], grad_clip=OPT["clip"],
        )  # fmt: skip

        def step_fn(s):
            x, y = _batch(stream, offsets, s, task)
            event = nd.run(x, y, steps=1, optimizer=opt, learning_rates=[rates[s]])[0]
            return event["loss"], event["grad_norm"]

        def logits_fn(x):
            out = []
            for i in range(0, len(x), BATCH):
                raw = nd.forward([int(v) for v in x[i : i + BATCH].reshape(-1)])
                out.append(np.frombuffer(raw, np.float32).reshape(BATCH, CONTEXT, VOCAB))
            return np.concatenate(out)

        digest_fn = lambda: _state_digest_native(nd)  # noqa: E731
        plan = nd.state.plan
        # the trainer's arenas, planned before the first step: weights, gradients and both
        # moments persist; activations, scratch and the bf16 GEMMs' packing workspace (NDT-BF16,
        # 0 under the other contracts) are the step's transient arena
        memory = {"persistent_bytes": 16 * plan.parameters,
                  "transient_bytes": 4 * (plan.activations + plan.scratch) + nd.workspace_bytes,
                  "workspace_bytes": nd.workspace_bytes,
                  "arena_bytes": nd.memory_bytes}  # fmt: skip
        memory_fn = lambda: memory  # noqa: E731
    else:
        import torch
        import torch.nn.functional as F

        torch.set_num_threads(threads)  # the 10-11 runs: 1
        torch.set_num_interop_threads(1)
        model = _torch_model(data)
        compiled = backend.startswith("torch-compiled")
        bf16 = backend.endswith("-bf16")

        def amp():
            # PyTorch's CPU mixed precision: autocast runs the linears and the attention in
            # bfloat16 (its lower-precision op list), the products of two of them stay bfloat16
            # (the SwiGLU), and the residual stream, the norms, the loss and the optimizer stay
            # FP32 by type promotion and its FP32 list
            return torch.autocast("cpu", dtype=torch.bfloat16) if bf16 else contextlib.nullcontext()

        opt = torch.optim.AdamW(
            model.parameters(), lr=OPT["peak"], betas=(OPT["beta1"], OPT["beta2"]),
            eps=OPT["eps"], weight_decay=OPT["decay"], **({"fused": True} if compiled else {}),
        )  # fmt: skip
        info.update(torch=torch.__version__, cpu_capability=torch.backends.cpu.get_cpu_capability(),
                    torch_threads=torch.get_num_threads(),
                    autocast="bfloat16" if bf16 else None)  # fmt: skip
        if compiled:
            from bcir.hosted.models.model import _rope_cache

            spec = _spec()

            def loss_of(x, y):
                values = model.model.embed_tokens(x)
                cos, sin = _rope_cache(
                    CONTEXT, spec.d_k, spec.rope_base, values.device, values.dtype
                )
                for layer in model.model.layers:
                    values = layer(values, cos, sin)
                logits = F.linear(model.model.norm(values), model.lm_head.weight)
                return F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))

            if bf16:
                fp32_loss = loss_of

                def loss_of(x, y):
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        return fp32_loss(x, y)

            loss_of = torch.compile(loss_of, fullgraph=True)
        else:

            def loss_of(x, y):
                with amp():
                    return model(x, y)[1]

        optimizer_step = opt.step

        def step_fn(s):
            x, y = _batch(stream, offsets, s, task)
            xt = torch.tensor(x, dtype=torch.long).view(BATCH, CONTEXT)
            yt = torch.tensor(y, dtype=torch.long).view(BATCH, CONTEXT)
            for group in opt.param_groups:
                group["lr"] = rates[s]
            opt.zero_grad(set_to_none=True)
            loss = loss_of(xt, yt)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), OPT["clip"])
            optimizer_step()
            return float(loss.detach()), float(norm)

        def logits_fn(x):
            # in training-sized batches, as the native trainer evaluates, so an evaluation's
            # activations do not set the peak memory the training is measured by
            model.eval()
            out = []
            with torch.no_grad(), amp():  # a bf16 run's model is evaluated as it trained
                for i in range(0, len(x), BATCH):
                    batch = torch.from_numpy(np.ascontiguousarray(x[i : i + BATCH]))
                    out.append(model(batch)[0].float().numpy())
            model.train()
            return np.concatenate(out)

        digest_fn = lambda: _state_digest_torch(model, opt)  # noqa: E731

        def memory_fn():
            # the persistent tensors after training, exactly: parameters, their gradients and
            # AdamW's moments; the transient activations have no peak counter on CPU here
            tensors = [*model.parameters(), *(p.grad for p in model.parameters())]
            tensors += [
                v for s in opt.state.values() for v in s.values() if torch.is_tensor(v) and v.dim()
            ]
            persistent = sum(v.numel() * v.element_size() for v in tensors if v is not None)
            return {"persistent_bytes": persistent, "transient_bytes": None,
                    "arena_bytes": None}  # fmt: skip

    rss_ready = _rss_mb()

    def evaluate(split):
        x, y, mask, spans = sets[split]
        t0 = time.perf_counter()
        out = _metrics(logits_fn(x), y, mask, spans)
        out["seconds"] = time.perf_counter() - t0
        return out

    eval_split = "validation" if task == "lm" else "test"
    losses, norms, step_s, evals = [], [], [], []
    trained = 0.0
    evals.append({"step": 0, "train_seconds": 0.0, **evaluate(eval_split)})
    for s in range(steps):
        t0 = time.perf_counter()
        loss, norm = step_fn(s)
        dt = time.perf_counter() - t0
        if s < 2 and backend.startswith("torch-compiled"):
            compile_s = (compile_s or 0.0) + dt
        trained += dt
        losses.append(loss)
        norms.append(norm)
        step_s.append(dt)
        if not math.isfinite(loss):
            raise RuntimeError(f"{backend}: non-finite loss at step {s}")
        if (s + 1) % eval_every == 0 or s + 1 == steps:
            evals.append({"step": s + 1, "train_seconds": trained, **evaluate(eval_split)})
            print(json.dumps({"backend": backend, "task": task, **evals[-1]}), flush=True)
    final = {split: evaluate(split) for split in sets}
    warm = step_s[2:] if len(step_s) > 4 else step_s
    info.update(
        losses=losses, grad_norms=norms, step_seconds=step_s, evals=evals, final=final,
        median_step_ms=1000 * statistics.median(warm),
        tokens_per_second=BATCH * CONTEXT / statistics.median(warm),
        train_seconds=trained, wall_seconds=time.perf_counter() - started,
        compile_and_first_steps_seconds=compile_s, state_sha256=digest_fn(), memory=memory_fn(),
        rss_mb={"start": rss_start, "ready": rss_ready, "end": _rss_mb(),
                "peak": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024},
        environment=_environment(),
    )  # fmt: skip
    out = data / "runs"
    out.mkdir(exist_ok=True)
    name = (
        f"{task}-{backend}"
        + ("" if contract == "v1" else f"-{contract}")
        + ("" if threads == 1 else f"-t{threads}")
    )
    (out / f"{name}.json").write_text(json.dumps(info, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"saved": name, "median_step_ms": info["median_step_ms"],
                      "final": final}), flush=True)  # fmt: skip


def _compiler(cc: str) -> str:
    return subprocess.run([cc, "--version"], capture_output=True, text=True).stdout.splitlines()[0]


def _environment():
    cpu = "?"
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    return {
        "python": sys.version.split()[0], "platform": platform.platform(), "cpu": cpu,
        "cpus": os.cpu_count(), "commit": _git("rev-parse", "HEAD").strip(),
    }  # fmt: skip


def _libraries():
    """The experiment environment's numerical libraries, as the summary stage sees them (every
    run used the same environment)."""
    import numpy
    import scipy_openblas32
    import torch

    return {
        "torch": torch.__version__,
        "torch_config": torch.__config__.show(),
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "numpy": numpy.__version__,
        "scipy_openblas32": scipy_openblas32.__version__,
        "openblas_config": _openblas()[1],
    }


# --- identity and baseline -------------------------------------------------------------------
def _native_steps(data, library, steps, *, openblas=False, threads=1, contract="v1"):
    from bcir.hosted.models.native import NativeAdamW

    stream = _load(data, "lm-train.u16", "H")
    offsets = _load(data, "lm-schedule.u32", "I")
    nd, _ = _native(data, library, openblas=openblas, threads=threads, contract=contract)
    opt = NativeAdamW(
        lr=OPT["peak"], beta1=OPT["beta1"], beta2=OPT["beta2"], epsilon=OPT["eps"],
        weight_decay=OPT["decay"], grad_clip=OPT["clip"],
    )  # fmt: skip
    times, losses = [], []
    for s in range(steps):
        x, y = _batch(stream, offsets, s, "lm")
        t0 = time.perf_counter()
        losses.append(
            nd.run(x, y, steps=1, optimizer=opt, learning_rates=[_rate(s, 1000)])[0]["loss"]
        )
        times.append(time.perf_counter() - t0)
    digest = _state_digest_native(nd)
    nd.close()
    return digest, times, losses


def identity(
    data: Path, steps: int, compilers, threads=(1,), contracts=("v1",),
    variants=("widest", "avx2", "portable"),
):  # fmt: skip
    """The same 20 BCIR steps through every ISA variant, every compiler, every thread count of
    the deterministic pool (NDT-THREADS), every contract (NDT-FMA, NDT-EXP) and a repeat: one
    state per contract. `avx512` caps the tensor unit below AMX (NDT-BF16: the bf16 contract's
    AVX-512 instance in place of its tile instance); under it and `avx2` and `portable` the bf16
    GEMMs run the instance the cap leaves (AVX-512, portable, portable)."""
    builds = []
    for cc in compilers:
        for label, defines in (("widest", ()), ("avx512", ("-DBCIR_TENSOR_MAX_VARIANT=4",)),
                               ("avx2", ("-DBCIR_TENSOR_MAX_VARIANT=2",)),
                               ("portable", ("-DBCIR_TENSOR_NO_DISPATCH",))):  # fmt: skip
            if label not in variants:
                continue  # the portable fused instance calls libm fmaf per term: correct, hours
            lib = build_library(data / "lib" / f"id-{Path(cc).name}-{label}", cc, defines)
            builds.append((f"{_compiler(cc)} / {label}", lib))
    rows = []
    for name, lib in builds + builds[:1]:
        for contract in contracts:
            for n in threads:
                digest, times, losses = _native_steps(
                    data, lib, steps, threads=n, contract=contract
                )
                rows.append({"build": name, "contract": contract, "threads": n,
                             "state_sha256": digest, "last_loss": losses[-1],
                             "median_step_ms": 1000 * statistics.median(times)})  # fmt: skip
                print(json.dumps(rows[-1]), flush=True)
    by_contract = {c: len({r["state_sha256"] for r in rows if r["contract"] == c}) == 1
                   for c in contracts}  # fmt: skip
    result = {"steps": steps, "rows": rows, "identical_by_contract": by_contract,
              "identical": all(by_contract.values()),
              "contracts_distinct": len({r["state_sha256"] for r in rows}) == len(contracts)}  # fmt: skip
    (data / "identity.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"identical": result["identical"]}), flush=True)


def _base_sources(data: Path, base: str) -> Path:
    """The trainer's four sources as they were at `base`."""
    old_src = data / "baseline-src"
    old_src.mkdir(parents=True, exist_ok=True)
    for name in ("bcir_tensor.c", "bcir_tensor.h", "bcir_decoder_train.c", "bcir_decoder_train.h"):
        (old_src / name).write_bytes(
            subprocess.run(
                ["git", "-C", str(REPO), "show", f"{base}:runtime/c/{name}"],
                check=True,
                capture_output=True,
            ).stdout
        )
    return old_src


def baseline(data: Path, steps: int, cc: str, base: str, old_flags=BASE_FLAGS):
    """The kernels before this update (the sources at `base`, built with `old_flags`) against
    these on the same steps: the portable trainer's state must not move a bit, and each pair is
    timed interleaved."""
    old_src = _base_sources(data, base)
    old = build_library(data / "lib" / "baseline-old", cc, sources=old_src, flags=tuple(old_flags))
    new = build_library(data / "lib" / "baseline-new", cc)
    rows = {}
    for provider in ("portable", "openblas"):
        runs = {"old": [], "new": []}
        digests = {}
        for rep in range(2):
            for label, lib in (
                (("old", old), ("new", new)) if rep % 2 == 0 else (("new", new), ("old", old))
            ):
                digest, times, _ = _native_steps(data, lib, steps, openblas=provider == "openblas")
                runs[label].extend(times[1:])
                digests.setdefault(label, set()).add(digest)
        rows[provider] = {
            "old_median_step_ms": 1000 * statistics.median(runs["old"]),
            "new_median_step_ms": 1000 * statistics.median(runs["new"]),
            "speedup": statistics.median(runs["old"]) / statistics.median(runs["new"]),
            "state_identical": digests["old"] == digests["new"] and len(digests["new"]) == 1,
            "samples": {k: v for k, v in runs.items()},
        }
        print(
            json.dumps({provider: {k: v for k, v in rows[provider].items() if k != "samples"}}),
            flush=True,
        )
    result = {"base": base, "compiler": _compiler(cc), "steps": steps, "rows": rows,
              "flags": {"old": list(old_flags), "new": list(FLAGS)}}  # fmt: skip
    (data / "baseline.json").write_text(json.dumps(result, indent=1) + "\n")


# --- kernels ---------------------------------------------------------------------------------
KERNEL_C = r"""
#define _POSIX_C_SOURCE 200809L
/* The trainer's kernels at the comparison's shape (B=8, T=128, the 9.84M decoder): every GEMM
 * shape one training step issues, attention forward and backward, and the AdamW update. Built
 * once against the sources before the update and once against these; each build prints its
 * median times and a hash of every output, and the stage requires the hashes to agree. */
#include "bcir_decoder_train.h"
#include "bcir_tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define REPS 7
static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}
static unsigned long long rs = 20261011ULL;
static float rnd(void) {
  rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
  return (float)((double)(rs >> 11) / 9007199254740992.0 - 0.5);
}
static float *fill(size_t n) {
  float *p = malloc(n * sizeof *p); size_t i;
  if (!p) exit(3);
  for (i = 0; i < n; i++) p[i] = rnd();
  return p;
}
static unsigned long long fnv(const void *p, size_t n, unsigned long long h) {
  const unsigned char *b = p; size_t i;
  for (i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
  return h;
}
static int order(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}
static double median(double *v) { qsort(v, REPS, sizeof *v, order); return v[REPS / 2]; }
/* One step's GEMMs: per linear, y = x W^T forward, dx = dy W (beta 1 where it accumulates into
 * the shared input adjoint) and dW += dy^T x; calls per step over six layers and the head. */
static const struct { const char *name; int ta, tb; size_t m, n, k; float beta; int calls; } G[] = {
  {"q,o forward", 0, 1, 1024, 384, 384, 0.0f, 12}, {"k,v forward", 0, 1, 1024, 192, 384, 0.0f, 12},
  {"gate,up forward", 0, 1, 1024, 1024, 384, 0.0f, 12},
  {"down forward", 0, 1, 1024, 384, 1024, 0.0f, 6}, {"head forward", 0, 1, 1024, 260, 384, 0.0f, 1},
  {"q,o dx", 0, 0, 1024, 384, 384, 0.0f, 12}, {"k,v dx", 0, 0, 1024, 384, 192, 1.0f, 12},
  {"gate dx", 0, 0, 1024, 384, 1024, 0.0f, 6}, {"up dx", 0, 0, 1024, 384, 1024, 1.0f, 6},
  {"down dx", 0, 0, 1024, 1024, 384, 0.0f, 6}, {"head dx", 0, 0, 1024, 384, 260, 0.0f, 1},
  {"q,o dW", 1, 0, 384, 384, 1024, 1.0f, 12}, {"k,v dW", 1, 0, 192, 384, 1024, 1.0f, 12},
  {"gate,up dW", 1, 0, 1024, 384, 1024, 1.0f, 12}, {"down dW", 1, 0, 384, 1024, 1024, 1.0f, 6},
  {"head dW", 1, 0, 260, 384, 1024, 1.0f, 1}};
static void gemm(size_t g) {
  size_t m = G[g].m, n = G[g].n, k = G[g].k; int r, v, agree = 1; double t[REPS];
  float *a = fill(m * k), *b = fill(k * n), *c0 = fill(m * n), *c = malloc(m * n * sizeof *c);
  unsigned long long h = 0;
  if (!c) exit(3);
  printf("%s{\"name\": \"%s\", \"ta\": %d, \"tb\": %d, \"m\": %zu, \"n\": %zu, \"k\": %zu, "
         "\"calls\": %d, \"ms\": {", g ? ", " : "", G[g].name, G[g].ta, G[g].tb, m, n, k, G[g].calls);
#ifdef BCIR_KB_NEW
  for (v = 1; v <= 4; v *= 2) {
    unsigned long long hv;
    if (!(bcir_tensor_mm_variants() & v)) continue;
    for (r = 0; r < REPS; r++) {
      double t0;
      memcpy(c, c0, m * n * sizeof *c);
      t0 = now();
      if (bcir_tensor_mm_variant(v, G[g].ta, G[g].tb, m, n, k, 1.0f, a, b, G[g].beta, c) != v) exit(4);
      t[r] = now() - t0;
    }
    hv = fnv(c, m * n * sizeof *c, 1469598103934665603ULL);
    if (h && hv != h) agree = 0;
    h = hv;
    printf("%s\"%d\": %.4f", v > 1 ? ", " : "", v, median(t));
  }
#else
  for (r = 0; r < REPS; r++) {
    double t0;
    memcpy(c, c0, m * n * sizeof *c);
    t0 = now();
    bcir_tensor_mm(NULL, G[g].ta, G[g].tb, m, n, k, 1.0f, a, b, G[g].beta, c);
    t[r] = now() - t0;
  }
  h = fnv(c, m * n * sizeof *c, 1469598103934665603ULL);
  (void)v;
  printf("\"mm\": %.4f", median(t));
#endif
  printf("}, \"hash\": \"%016llx\", \"variants_agree\": %s}", h, agree ? "true" : "false");
  free(a); free(b); free(c0); free(c);
}
static void attention(void) {
  size_t B = 8, T = 128, H = 8, KV = 4, D = 48, nq = B * T * H * D, nkv = B * T * KV * D;
  size_t np = B * H * T * T; int r; double tf[REPS], tb[REPS];
  float *q = fill(nq), *k = fill(nkv), *v = fill(nkv), *dy = fill(nq), *p = fill(np), *y = fill(nq);
  float *pc = fill(np), *dq = fill(nq), *dk = fill(nkv), *dv = fill(nkv);
  unsigned long long hf, hb;
  for (r = 0; r < REPS; r++) {
    double t0 = now();
    bcir_tensor_attention(B, T, H, KV, D, q, k, v, p, y);
    tf[r] = now() - t0;
  }
  hf = fnv(y, nq * sizeof *y, fnv(p, np * sizeof *p, 1469598103934665603ULL));
  for (r = 0; r < REPS; r++) {
    double t0;
    memcpy(pc, p, np * sizeof *p);
    t0 = now();
#ifdef BCIR_KB_NEW
    bcir_tensor_attention_backward_inplace(B, T, H, KV, D, q, k, v, pc, dy, dq, dk, dv);
#else
    bcir_tensor_attention_backward(B, T, H, KV, D, q, k, v, pc, dy, dq, dk, dv);
#endif
    tb[r] = now() - t0;
  }
  hb = fnv(dv, nkv * sizeof *dv, fnv(dk, nkv * sizeof *dk, fnv(dq, nq * sizeof *dq,
      1469598103934665603ULL)));
  printf("\"attention\": {\"forward_ms\": %.4f, \"backward_ms\": %.4f, \"forward_hash\": "
         "\"%016llx\", \"backward_hash\": \"%016llx\"}", median(tf), median(tb), hf, hb);
  free(q); free(k); free(v); free(dy); free(p); free(y); free(pc); free(dq); free(dk); free(dv);
}
static void adamw(void) {
  bcir_decoder_spec sp = {260, 384, 8, 4, 6, 1024, 8, 128, 1, 10000.0, 1e-6};
  bcir_decoder_adamw o = {2e-3, 0.9, 0.95, 1e-8, 0.1, 1.0};
  bcir_decoder_plan plan; bcir_decoder_state s; size_t i, n; int r; double t[REPS], norm = 0.0;
  unsigned long long h;
  if (bcir_decoder_make_plan(&sp, &plan)) exit(5);
  n = plan.parameters;
  memset(&s, 0, sizeof s);
  s.plan = plan;
  s.weights = fill(n); s.gradients = fill(n); s.moment1 = calloc(n, 4); s.moment2 = calloc(n, 4);
  s.activations = calloc(plan.activations, 4); s.scratch = calloc(plan.scratch, 4);
  if (!s.moment1 || !s.moment2 || !s.activations || !s.scratch) exit(3);
  s.weight_capacity = s.gradient_capacity = s.moment1_capacity = s.moment2_capacity = n;
  s.activation_capacity = plan.activations; s.scratch_capacity = plan.scratch;
  s.beta1_power = s.beta2_power = 1.0;
  for (i = 0; i < n; i++) s.weights[i] *= 0.04f;
  for (r = -2; r < REPS; r++) {   /* two untimed updates first: the moments are then nonzero */
    double t0;
    for (i = 0; i < n; i++) s.gradients[i] = rnd() * (r % 3 ? 1e-3f : 4.0f);   /* some clip */
    t0 = now();
    if (bcir_decoder_update(&s, &o, 1.0, &norm)) exit(6);
    if (r >= 0) t[r] = now() - t0;
  }
  h = fnv(s.moment2, n * 4, fnv(s.moment1, n * 4, fnv(s.weights, n * 4, 1469598103934665603ULL)));
  h = fnv(&norm, sizeof norm, fnv(&s.beta2_power, 8, fnv(&s.beta1_power, 8, h)));
  printf("\"adamw\": {\"parameters\": %zu, \"ms\": %.4f, \"hash\": \"%016llx\"}", n, median(t), h);
  free(s.weights); free(s.gradients); free(s.moment1); free(s.moment2); free(s.activations);
  free(s.scratch);
}
int main(void) {
  size_t g;
  printf("{\"gemm\": [");
  for (g = 0; g < sizeof G / sizeof G[0]; g++) gemm(g);
  printf("], ");
  attention();
  printf(", ");
  adamw();
#ifdef BCIR_KB_NEW
  printf(", \"variants\": %d}\n", bcir_tensor_mm_variants());
#else
  printf(", \"variants\": 1}\n");
#endif
  return 0;
}
"""


def kernels(data: Path, cc: str, base: str, rounds: int, old_flags=BASE_FLAGS):
    """The trainer's kernels before this update (the sources at `base`, built as they shipped:
    `old_flags`) and after, at the comparison's shape: every GEMM shape of one step on every ISA
    variant, attention forward and backward, AdamW. Rounds alternate the builds; every output must
    hash the same before and after, and across the variants."""
    out = data / "kernels"
    out.mkdir(parents=True, exist_ok=True)
    (out / "kernel_bench.c").write_text(KERNEL_C)
    builds = {}
    old_src = _base_sources(data, base)
    # a base that already has the instruction-set variants is measured variant by variant too
    old_new_api = "bcir_tensor_mm_variant" in (old_src / "bcir_tensor.h").read_text()
    for label, src, flags, defines in (
        ("old", old_src, tuple(old_flags), ("-DBCIR_KB_NEW",) if old_new_api else ()),
        ("new", REPO / "runtime" / "c", FLAGS, ("-DBCIR_KB_NEW",)),
    ):
        exe = out / f"kernels-{label}"
        command = [cc, "-std=c11", *flags, *WARNINGS, *defines, "-I", str(src),
                   str(out / "kernel_bench.c"), str(src / "bcir_tensor.c"),
                   str(src / "bcir_decoder_train.c"), "-o", str(exe), "-lm"]  # fmt: skip
        subprocess.run(command, check=True, capture_output=True, timeout=300)
        builds[label] = exe
    samples = {"old": [], "new": []}
    for r in range(rounds):
        for label in ("old", "new") if r % 2 == 0 else ("new", "old"):
            run = subprocess.run(
                [str(builds[label])], check=True, capture_output=True, timeout=3600
            )
            samples[label].append(json.loads(run.stdout))
            print(json.dumps({"round": r, "build": label}), flush=True)
    first = {label: runs[0] for label, runs in samples.items()}
    hashes = {
        label: [
            ([g["hash"] for g in s["gemm"]], s["attention"]["forward_hash"],
             s["attention"]["backward_hash"], s["adamw"]["hash"])
            for s in runs
        ]
        for label, runs in samples.items()
    }  # fmt: skip
    checked = ("old", "new") if old_new_api else ("new",)
    identical = (
        all(h == hashes["old"][0] for h in hashes["old"] + hashes["new"])
        and all(g["variants_agree"] for label in checked for s in samples[label] for g in s["gemm"])
    )  # fmt: skip

    def med(label, pick):
        return statistics.median(pick(s) for s in samples[label])

    rows, totals = [], {}
    variants = sorted(first["new"]["gemm"][0]["ms"], key=int)
    old_variants = variants if old_new_api else []
    for i, g in enumerate(first["old"]["gemm"]):
        row = {key: g[key] for key in ("name", "ta", "tb", "m", "n", "k", "calls")}
        # the base's own entry point: its single kernel, or its widest variant
        widest = "mm" if not old_new_api else variants[-1]
        row["old_ms"] = med("old", lambda s, i=i, w=widest: s["gemm"][i]["ms"][w])
        for v in old_variants:
            row[f"old_ms_variant_{v}"] = med("old", lambda s, i=i, v=v: s["gemm"][i]["ms"][v])
        for v in variants:
            row[f"new_ms_variant_{v}"] = med("new", lambda s, i=i, v=v: s["gemm"][i]["ms"][v])
        rows.append(row)
    keys = ["old_ms"] + [f"old_ms_variant_{v}" for v in old_variants]
    for key in keys + [f"new_ms_variant_{v}" for v in variants]:
        totals[key] = sum(row["calls"] * row[key] for row in rows)
    result = {
        "base": base,
        "compiler": _compiler(cc),
        "flags": {"old": list(old_flags), "new": list(FLAGS)},
        "rounds": rounds,
        "variants": first["new"]["variants"],
        "outputs_identical": identical,
        "gemm": rows,
        "gemm_ms_per_step": totals,
        "attention_per_layer_ms": {
            label: {
                "forward": med(label, lambda s: s["attention"]["forward_ms"]),
                "backward": med(label, lambda s: s["attention"]["backward_ms"]),
            }
            for label in samples
        },
        "adamw_ms": {label: med(label, lambda s: s["adamw"]["ms"]) for label in samples},
        "samples": samples,
    }
    (data / "kernels.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("samples", "gemm")}), flush=True)


# --- breakdown -------------------------------------------------------------------------------
def _median_ms(fn, rounds: int) -> float:
    """The median wall time of fn() in milliseconds over `rounds` calls, after one untimed call."""
    fn()
    samples = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        samples.append(1000 * (time.perf_counter() - t0))
    return statistics.median(samples)


def _bcir_breakdown(
    data: Path, library: Path, rounds: int, openblas: bool, threads: int = 1, contract: str = "v1"
) -> dict:
    """One BCIR step at the comparison's shape, whole and by part: the binding's stages, then
    every kernel family the step calls, each timed alone through the entry points the trainer
    calls under the run's pool and contracts (the pooled kernels, the fused or blocked GEMM and
    the v2 exponential kernels when the contract selects them, and the provider the run uses for
    the linears), with the calls one step makes."""
    import numpy as np
    from bcir.hosted.models.native import NativeAdamW

    spec = _spec()
    nd, provider = _native(data, library, openblas=openblas, threads=threads, contract=contract)
    lib = nd.lib
    n, d, f, v = BATCH * CONTEXT, spec.d_model, spec.d_ff, spec.vocab_size
    h, kh, hd, layers = spec.n_heads, spec.kv_heads, spec.d_k, spec.n_layers
    kv = kh * hd
    P, Z, F32 = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_float
    MM = [P, ctypes.c_int, ctypes.c_int, Z, Z, Z, F32, P, P, F32, P]
    signatures = {
        "rms_pool": [P, Z, Z, F32, P, P, P],
        "rms_backward_pool": [P, Z, Z, F32, P, P, P, P, P, P],  # pooled since NDT-REDUCE
        "rope_pool": [P, Z, Z, Z, Z, ctypes.c_double, P, ctypes.c_int],
        "silu_pool": [P, Z, P, P],
        "silu_v2_pool": [P, Z, P, P],
        "swiglu_backward_pool": [P, Z, P, P, P, P, P],
        "swiglu_backward_v2_pool": [P, Z, P, P, P, P, P],
        "attention_pool": [P, Z, Z, Z, Z, Z, P, P, P, P, P],
        "attention_v2_pool": [P, Z, Z, Z, Z, Z, P, P, P, P, P],
        "attention_backward_inplace_pool": [P, Z, Z, Z, Z, Z, P, P, P, P, P, P, P, P],
        "mm_pool": MM,
        "mm_fused_pool": MM,
        "mm_kblock_pool": MM,
        "mm_pair_pool": [P, ctypes.c_int, P, P],  # a linear's two VJPs as one job (NDT-SCHED-0)
        "linear": [P, Z, Z, Z, P, P, P],
        "linear_backward": [P, Z, Z, Z, P, P, P, P, P, F32],
    }
    fn = {}
    for name, args in signatures.items():
        fn[name] = getattr(lib, "bcir_tensor_" + name)
        fn[name].argtypes, fn[name].restype = args, None
    # the trainer's routing (bcir_decoder_train.c: gemm, silu, swiglu_backward, attention): the
    # state's pool (NULL when serial), the fused GEMM under "fused", the v2 kernels under "exp"
    pool = nd.state.pool
    parts_of = set(contract.split("+"))
    mm = fn[
        "mm_kblock_pool"
        if "kblock" in parts_of
        else "mm_fused_pool"
        if "fused" in parts_of
        else "mm_pool"
    ]
    v2 = "_v2_pool" if "exp" in parts_of else "_pool"
    silu_fn, swiglu_fn, att_fn = fn["silu" + v2], fn["swiglu_backward" + v2], fn["attention" + v2]
    rng = np.random.default_rng(SEED)

    def buf(*shape, scale=1.0):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def zero(*shape):
        return np.zeros(shape, np.float32)

    p = lambda a: a.ctypes.data  # noqa: E731
    prov = ctypes.addressof(nd.state.provider)
    eps, base = float(spec.rms_norm_eps), float(spec.rope_base)
    x, g, y, dy, dx, dg = buf(n, d), buf(d), zero(n, d), buf(n, d), zero(n, d), zero(d)
    inv = zero(n)  # the gamma VJP's per-row 1/rms between its two pooled phases
    q, k, val, ctx, dq, dk, dv = buf(n, d), buf(n, kv), buf(n, kv), zero(n, d), zero(n, d), \
        zero(n, kv), zero(n, kv)  # fmt: skip
    probs = zero(BATCH, h, CONTEXT, CONTEXT)
    gate, up, hidden, dh, dgate, dup = buf(n, f), buf(n, f), zero(n, f), buf(n, f), zero(n, f), \
        zero(n, f)  # fmt: skip
    weights = {"q": (d, d), "k": (kv, d), "v": (kv, d), "o": (d, d), "gate": (f, d),
               "up": (f, d), "down": (d, f), "head": (v, d)}  # fmt: skip
    w = {name: buf(*shape, scale=0.02) for name, shape in weights.items()}
    dw = {name: zero(*shape) for name, shape in weights.items()}
    outs = {"q": zero(n, d), "k": zero(n, kv), "v": zero(n, kv), "o": zero(n, d),
            "gate": zero(n, f), "up": zero(n, f), "down": zero(n, d), "head": zero(n, v)}  # fmt: skip
    ins = {"q": x, "k": x, "v": x, "o": ctx, "gate": x, "up": x, "down": hidden, "head": x}
    # dx beta per linear in the backward's order (0 overwrites, 1 accumulates into the shared dx)
    beta = {"down": 0.0, "gate": 0.0, "up": 1.0, "o": 0.0, "q": 0.0, "k": 1.0, "v": 1.0,
            "head": 0.0}  # fmt: skip
    dys = {"q": buf(n, d), "k": buf(n, kv), "v": buf(n, kv), "o": buf(n, d), "gate": buf(n, f),
           "up": buf(n, f), "down": buf(n, d), "head": buf(n, v)}  # fmt: skip
    dxs = {"q": dx, "k": dx, "v": dx, "o": zero(n, d), "gate": dx, "up": dx, "down": zero(n, f),
           "head": dx}  # fmt: skip
    per_layer = ("q", "k", "v", "o", "gate", "up", "down")

    def linear(name):
        i, o = weights[name][1], weights[name][0]
        if openblas:  # the provider's GEMM, threaded as the provider threads it
            fn["linear"](prov, n, i, o, p(ins[name]), p(w[name]), p(outs[name]))
        else:
            mm(pool, 0, 1, n, o, i, 1.0, p(ins[name]), p(w[name]), 0.0, p(outs[name]))

    class Call(ctypes.Structure):  # bcir_tensor_mm_call
        _fields_ = [("ta", ctypes.c_int), ("tb", ctypes.c_int), ("m", Z), ("n", Z), ("k", Z),
                    ("alpha", F32), ("a", P), ("b", P), ("beta", F32), ("c", P)]  # fmt: skip

    pair_contract = 2 if "kblock" in parts_of else 1 if "fused" in parts_of else 0

    def linear_backward(name):
        i, o = weights[name][1], weights[name][0]
        if openblas:
            fn["linear_backward"](prov, n, i, o, p(ins[name]), p(w[name]), p(dys[name]),
                                  p(dxs[name]), p(dw[name]), beta[name])  # fmt: skip
        else:  # the trainer's own dispatch: the two VJPs as one pool job
            cx = Call(0, 0, n, i, o, 1.0, p(dys[name]), p(w[name]), beta[name], p(dxs[name]))
            cw = Call(1, 0, o, i, n, 1.0, p(dys[name]), p(ins[name]), 1.0, p(dw[name]))
            fn["mm_pair_pool"](pool, pair_contract, ctypes.byref(cx), ctypes.byref(cw))

    def gemm_forward():
        for _ in range(layers):
            for name in per_layer:
                linear(name)
        linear("head")

    def gemm_backward():
        linear_backward("head")
        for _ in range(layers):
            for name in ("down", "gate", "up", "o", "q", "k", "v"):
                linear_backward(name)

    att = {"forward": [], "backward": []}

    def attention():
        # the backward consumes the probabilities the forward leaves, so each layer runs both
        tf = tb = 0.0
        for _ in range(layers):
            t0 = time.perf_counter()
            att_fn(pool, BATCH, CONTEXT, h, kh, hd, p(q), p(k), p(val), p(probs), p(ctx))
            t1 = time.perf_counter()
            fn["attention_backward_inplace_pool"](pool, BATCH, CONTEXT, h, kh, hd, p(q), p(k),
                                                  p(val), p(probs), p(dy), p(dq), p(dk), p(dv))  # fmt: skip
            t2 = time.perf_counter()
            tf, tb = tf + t1 - t0, tb + t2 - t1
        att["forward"].append(1000 * tf)
        att["backward"].append(1000 * tb)

    def norms():
        for _ in range(2 * layers + 1):
            fn["rms_pool"](pool, n, d, eps, p(x), p(g), p(y))

    def norms_backward():
        for _ in range(2 * layers + 1):
            fn["rms_backward_pool"](pool, n, d, eps, p(x), p(g), p(dy), p(dx), p(dg), p(inv))

    def rope(inverse):
        for _ in range(layers):
            fn["rope_pool"](pool, BATCH, CONTEXT, h, hd, base, p(dq if inverse else q), inverse)
            fn["rope_pool"](pool, BATCH, CONTEXT, kh, hd, base, p(dk if inverse else k), inverse)

    def silu():
        for _ in range(layers):
            silu_fn(pool, n * f, p(gate), p(hidden))

    def swiglu_backward():
        for _ in range(layers):
            swiglu_fn(pool, n * f, p(gate), p(up), p(dh), p(dgate), p(dup))

    # the decoder's own stages through the binding, on the comparison's batches
    stream = _load(data, "lm-train.u16", "H")
    offsets = _load(data, "lm-schedule.u32", "I")
    opt = NativeAdamW(
        lr=OPT["peak"] * 0.25, beta1=OPT["beta1"], beta2=OPT["beta2"], epsilon=OPT["eps"],
        weight_decay=OPT["decay"], grad_clip=OPT["clip"],
    )  # fmt: skip
    xs, ys = _batch(stream, offsets, 0, "lm")
    # paired rounds: every round times the stages and then each kernel family alone, so "the
    # rest" (the step less its kernels) is a difference taken under one drift of a shared host,
    # not between medians taken a minute apart; the first round warms up and is dropped
    timed = {
        "batch": lambda: _batch(stream, offsets, 1, "lm"),
        "binding": lambda: (nd._tokens(xs), nd._tokens(ys)),
        "step": lambda: nd.run(xs, ys, steps=1, optimizer=opt),
        "forward": lambda: nd.forward(xs),
        "loss_backward": lambda: nd.backward(xs, ys),
        "update": lambda: nd.update(opt),
        "gemm_forward": gemm_forward, "gemm_backward": gemm_backward, "attention": attention,
        "rms": norms, "rms_backward": norms_backward,
        "rope": lambda: rope(0), "rope_inverse": lambda: rope(1),
        "silu": silu, "swiglu_backward": swiglu_backward,
    }  # fmt: skip
    samples = {name: [] for name in timed}
    for _ in range(rounds + 1):
        for name, call in timed.items():
            t0 = time.perf_counter()
            call()
            samples[name].append(1000 * (time.perf_counter() - t0))

    def median(name):
        return statistics.median(samples[name][1:])

    stages = {
        k: median(k) for k in ("batch", "binding", "step", "forward", "loss_backward", "update")
    }
    parts = {"gemm_forward": median("gemm_forward"), "gemm_backward": median("gemm_backward")}
    parts["attention_forward"] = statistics.median(att["forward"][1:])
    parts["attention_backward"] = statistics.median(att["backward"][1:])
    parts.update({k: median(k) for k in ("rms", "rms_backward", "rope", "rope_inverse", "silu",
                                         "swiglu_backward")})  # fmt: skip
    kernels = sum(parts.values()) + stages["update"]
    return {
        **provider, "parts_ms": parts, "stages_ms": stages,
        "rest_ms": stages["step"] - stages["binding"] - kernels,
        "training_step_ms": stages["batch"] + stages["step"],
    }  # fmt: skip


def _torch_breakdown(data: Path, rounds: int) -> dict:
    """One eager PyTorch step at the comparison's shape: its phases by wall time, then the
    forward and backward's operators under torch.profiler, self CPU time by family."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = _torch_model(data)
    opt = torch.optim.AdamW(
        model.parameters(), lr=OPT["peak"] * 0.25, betas=(OPT["beta1"], OPT["beta2"]),
        eps=OPT["eps"], weight_decay=OPT["decay"],
    )  # fmt: skip
    stream = _load(data, "lm-train.u16", "H")
    offsets = _load(data, "lm-schedule.u32", "I")
    names = ("batch", "zero_grad", "forward", "backward", "clip", "optimizer")
    phases = {name: [] for name in names}
    ops: dict = {}

    def step(s, record=False, profiled=False):
        t = [time.perf_counter()]
        x, y = _batch(stream, offsets, s, "lm")
        xt = torch.tensor(x, dtype=torch.long).view(BATCH, CONTEXT)
        yt = torch.tensor(y, dtype=torch.long).view(BATCH, CONTEXT)
        t.append(time.perf_counter())
        opt.zero_grad(set_to_none=True)
        t.append(time.perf_counter())
        if profiled:
            with profile(activities=[ProfilerActivity.CPU]) as prof:
                loss = model(xt, yt)[1]
                loss.backward()
            for event in prof.key_averages():
                ops[event.key] = ops.get(event.key, 0.0) + event.self_cpu_time_total / 1000
            t += [t[-1], time.perf_counter()]
        else:
            loss = model(xt, yt)[1]
            t.append(time.perf_counter())
            loss.backward()
            t.append(time.perf_counter())
        torch.nn.utils.clip_grad_norm_(model.parameters(), OPT["clip"])
        t.append(time.perf_counter())
        opt.step()
        t.append(time.perf_counter())
        if record:
            for name, a, b in zip(names, t, t[1:]):
                phases[name].append(1000 * (b - a))

    for s in range(2):
        step(s)
    for s in range(2, 2 + rounds):
        step(s, record=True)
    for s in range(2 + rounds, 2 + 2 * rounds):
        step(s, profiled=True)
    medians = {name: statistics.median(values) for name, values in phases.items()}
    per_step = {key: ms / rounds for key, ms in ops.items()}

    def family(key):
        if "scaled_dot_product" in key or "flash_attention" in key:
            return "attention"
        if key in ("aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm", "aten::matmul",
                   "aten::linear"):  # fmt: skip
            return "gemm"
        return "other"

    families: dict = {}
    for key, ms in per_step.items():
        families[family(key)] = families.get(family(key), 0.0) + ms
    top = sorted(per_step.items(), key=lambda kv: -kv[1])[:15]
    return {
        "phases_ms": medians,
        "training_step_ms": sum(medians.values()),
        "profile_forward_backward_self_cpu_ms": families,
        "profile_total_ms": sum(per_step.values()),
        "top_operators_ms": dict(top),
        "torch": torch.__version__,
    }


def bcir_breakdown_stage(
    data: Path, cc: str, rounds: int, threads: int, contract: str, openblas: bool = False
):
    """The BCIR side of `breakdown` for one (contract, thread count): where a step spends its
    time on the pool under each contract (NDT-MEASURE), with OpenBLAS behind the linears when
    asked (admitted under the blocked contract, its own arithmetic under the others); no
    PyTorch is needed."""
    library = build_library(data / "lib" / cc.replace("/", "_"), cc)
    result = {"compiler": _compiler(cc), "threads": threads, "contract": contract, "rounds": rounds,
              **_bcir_breakdown(data, library, rounds, openblas=openblas, threads=threads,
                                contract=contract)}  # fmt: skip
    name = f"breakdown-bcir{'-openblas' if openblas else ''}-{contract}-t{threads}"
    (data / f"{name}.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"saved": name, **{k: v for k, v in result.items() if isinstance(v, (int, float))}}),
          flush=True)  # fmt: skip


def breakdown(data: Path, compilers, rounds: int):
    """Where one training step's time goes at the comparison's shape: on BCIR's own kernels as
    each compiler builds them, on BCIR with OpenBLAS behind the CBLAS seam (the first compiler's
    build), and on eager PyTorch. One thread each."""
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    result = {
        "rounds": rounds,
        "shape": {"batch": BATCH, "context": CONTEXT, **SPEC},
        "bcir": {},
        "environment": _environment(),
    }
    for i, cc in enumerate(compilers):
        library = build_library(data / "lib" / f"breakdown-{i}", cc)
        result["bcir"][_compiler(cc)] = _bcir_breakdown(data, library, rounds, openblas=False)
        print(json.dumps({"compiler": _compiler(cc), "done": "bcir"}), flush=True)
        if i == 0:
            result["bcir-openblas"] = {
                "compiler": _compiler(cc),
                **_bcir_breakdown(data, library, rounds, openblas=True),
            }
    result["torch-eager"] = _torch_breakdown(data, rounds)
    (data / "breakdown.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result, indent=1, sort_keys=True), flush=True)


# --- context ---------------------------------------------------------------------------------
# The 2026-10-10 plan's acceptance evidence for attention: time and memory against context, and
# the decoder trained at 64, 128, 256 tokens and beyond. Every row keeps the comparison's 1,024
# tokens per call or step (B*T), so the linears do the same work at every context and only the
# attention's work grows, with T. The same runs at the comparison's context on more threads
# answer the plan's other open question: BCIR's own kernels run one thread, while OpenBLAS and
# PyTorch can use every core.
CONTEXTS = (64, 128, 256, 512, 1024)
TOKENS = BATCH * CONTEXT
CONTEXT_BACKENDS = ("bcir", "bcir-openblas", "torch-eager")
THREADS = (2, 4)


def _attention_at(t: int, rounds: int, forward, backward, rng) -> dict:
    """One attention layer at context `t` and batch TOKENS // t, forward then backward: BCIR's
    kernels, and PyTorch's `HostedAttention` path (KV heads repeated, causal SDPA) through
    autograd, on the same values. Medians of `rounds` alternating calls after one untimed call,
    and the largest difference between the two frameworks' results."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.profiler import ProfilerActivity, profile

    spec = _spec()
    h, kh, hd = spec.n_heads, spec.kv_heads, spec.d_k
    b = TOKENS // t
    p = lambda a: a.ctypes.data  # noqa: E731
    q, dy = (rng.standard_normal((b * t, h * hd)).astype(np.float32) for _ in range(2))
    k, v = (rng.standard_normal((b * t, kh * hd)).astype(np.float32) for _ in range(2))
    probs = np.zeros((b, h, t, t), np.float32)
    ctx, dq, dk, dv = np.zeros_like(q), np.zeros_like(q), np.zeros_like(k), np.zeros_like(v)

    def bcir_layer():
        dq[:], dk[:], dv[:] = 0.0, 0.0, 0.0
        t0 = time.perf_counter()
        forward(b, t, h, kh, hd, p(q), p(k), p(v), p(probs), p(ctx))
        t1 = time.perf_counter()
        backward(b, t, h, kh, hd, p(q), p(k), p(v), p(probs), p(dy), p(dq), p(dk), p(dv))
        return t1 - t0, time.perf_counter() - t1

    # PyTorch's operands are the same values in HostedAttention's [batch, head, time, dim]
    def heads(a, n):
        return torch.from_numpy(a).view(b, t, n, hd).transpose(1, 2).contiguous()

    tq, tk, tv, tdy = heads(q, h), heads(k, kh), heads(v, kh), heads(dy, h)
    for tensor in (tq, tk, tv):
        tensor.requires_grad_()

    def torch_layer():
        tq.grad = tk.grad = tv.grad = None
        t0 = time.perf_counter()
        out = F.scaled_dot_product_attention(
            tq, tk.repeat_interleave(h // kh, dim=1), tv.repeat_interleave(h // kh, dim=1),
            dropout_p=0.0, is_causal=True,
        )  # fmt: skip
        t1 = time.perf_counter()
        out.backward(tdy)
        return t1 - t0, time.perf_counter() - t1, out

    times = {"bcir": ([], []), "torch": ([], [])}
    for i in range(rounds + 1):  # alternating, the first round untimed
        fb, bb = bcir_layer()
        ft, bt, out = torch_layer()
        if i:
            for name, (f_s, b_s) in (("bcir", (fb, bb)), ("torch", (ft, bt))):
                times[name][0].append(1000 * f_s)
                times[name][1].append(1000 * b_s)

    def flat(a, n):
        return a.detach().transpose(1, 2).reshape(b * t, n * hd).numpy()

    row = {
        "context": t, "batch": b,
        "bcir_forward_ms": statistics.median(times["bcir"][0]),
        "bcir_backward_ms": statistics.median(times["bcir"][1]),
        "torch_forward_ms": statistics.median(times["torch"][0]),
        "torch_backward_ms": statistics.median(times["torch"][1]),
        "bcir_probabilities_bytes": probs.nbytes,
        "max_abs_diff_vs_torch": {
            "y": float(np.abs(ctx - flat(out, h)).max()),
            "dq": float(np.abs(dq - flat(tq.grad, h)).max()),
            "dk": float(np.abs(dk - flat(tk.grad, kh)).max()),
            "dv": float(np.abs(dv - flat(tv.grad, kh)).max()),
        },
    }  # fmt: skip
    if t == CONTEXT:  # which of PyTorch's kernels ran: one profiled layer
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            torch_layer()
        row["torch_attention_operators"] = sorted(
            e.key for e in prof.key_averages() if "scaled_dot_product" in e.key or "flash" in e.key
        )
    return row


def _attention_curve(library: Path, rounds: int) -> list:
    """One attention layer at the decoder's head shape at every context (`_attention_at`)."""
    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    lib = ctypes.CDLL(str(library))
    P, Z = ctypes.c_void_p, ctypes.c_size_t
    forward = lib.bcir_tensor_attention
    forward.argtypes, forward.restype = [Z, Z, Z, Z, Z, P, P, P, P, P], None
    backward = lib.bcir_tensor_attention_backward_inplace
    backward.argtypes, backward.restype = [Z, Z, Z, Z, Z, P, P, P, P, P, P, P, P], None
    rng = np.random.default_rng(SEED)
    rows = []
    for t in CONTEXTS:
        rows.append(_attention_at(t, rounds, forward, backward, rng))
        print(json.dumps(rows[-1]), flush=True)
    return rows


def _context_run(
    data: Path, backend: str, t: int, steps: int, library: Path | None, threads: int = 1
) -> dict:
    """One process: `steps` training steps of the 9.84M decoder at context `t` and batch
    TOKENS // t on one backend and `threads` compute threads (BCIR's own kernels have one), from
    the comparison's initial weights and at its per-step learning rates (`--lr-scale 0.25`), on
    language-model windows drawn for this context. Its peak resident memory is its own."""
    n = str(threads)
    os.environ.update(OMP_NUM_THREADS=n, MKL_NUM_THREADS=n, OPENBLAS_NUM_THREADS=n)
    b = TOKENS // t
    stream = _load(data, "lm-train.u16", "H")
    rng = random.Random(SEED + t)
    offsets = [rng.randrange(len(stream) - t - 1) for _ in range(steps * b)]

    def batch(s):
        xs, ys = [], []
        for offset in offsets[s * b : (s + 1) * b]:
            xs.extend(stream[offset : offset + t])
            ys.extend(stream[offset + 1 : offset + t + 1])
        return xs, ys

    rates = [0.25 * _rate(s, 1000) for s in range(steps)]
    out: dict = {"backend": backend, "context": t, "batch": b, "steps": steps, "threads": threads}
    if backend.startswith("bcir"):
        from bcir.hosted.models.native import NativeAdamW, NativeDecoder, parameter_layout

        spec = _spec()
        nd = NativeDecoder(library, spec, batch_size=b, context_length=t, max_memory_bytes=8 << 30)
        flat = _load(data, "weights.f32", "f")
        nd.load_state_dict({name: flat[o : o + n] for name, _s, o, n in parameter_layout(spec)})
        if backend == "bcir-openblas":
            nd.use_cblas(_openblas(threads)[0], symbol="scipy_cblas_sgemm")
        opt = NativeAdamW(
            lr=OPT["peak"], beta1=OPT["beta1"], beta2=OPT["beta2"], epsilon=OPT["eps"],
            weight_decay=OPT["decay"], grad_clip=OPT["clip"],
        )  # fmt: skip
        out["arena_bytes"] = nd.memory_bytes

        def step(s):
            x, y = batch(s)
            return nd.run(x, y, steps=1, optimizer=opt, learning_rates=[rates[s]])[0]["loss"]
    else:
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        model = _torch_model(data)
        opt = torch.optim.AdamW(
            model.parameters(), lr=OPT["peak"], betas=(OPT["beta1"], OPT["beta2"]),
            eps=OPT["eps"], weight_decay=OPT["decay"],
        )  # fmt: skip
        out["torch"] = torch.__version__

        def step(s):
            x, y = batch(s)
            xt = torch.tensor(x, dtype=torch.long).view(b, t)
            yt = torch.tensor(y, dtype=torch.long).view(b, t)
            for group in opt.param_groups:
                group["lr"] = rates[s]
            opt.zero_grad(set_to_none=True)
            loss = model(xt, yt)[1]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), OPT["clip"])
            opt.step()
            return float(loss.detach())

    losses, seconds = [], []
    for s in range(steps):
        t0 = time.perf_counter()
        losses.append(step(s))
        seconds.append(time.perf_counter() - t0)
    # VmHWM is this address space's own peak: ru_maxrss would also carry the parent's resident
    # set across fork and exec, and the parent has PyTorch loaded
    out.update(
        losses=losses, step_seconds=seconds, median_step_ms=1000 * statistics.median(seconds[2:]),
        peak_rss_mb=_rss_mb("VmHWM"),
    )  # fmt: skip
    return out


def context(data: Path, cc: str, rounds: int, steps: int, repeats: int):
    """Time and memory against context, at the comparison's 1,024 tokens per call or step: one
    attention layer forward and backward on BCIR and on PyTorch, then the decoder's training
    step at every context on BCIR's own kernels, BCIR with OpenBLAS and eager PyTorch, each run
    in its own process, and at the comparison's context on more threads where a backend has
    them. The runs repeat `repeats` times, in alternating order, and a configuration's step time
    is the median of every timed step it ran. The decoder's losses on the same batches are
    compared with eager PyTorch's."""
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    library = build_library(data / "lib" / "context", cc)
    result = {
        "tokens": TOKENS, "contexts": list(CONTEXTS), "rounds": rounds, "steps": steps,
        "compiler": _compiler(cc), "attention": _attention_curve(library, rounds), "decoder": [],
        "environment": _environment(),
    }  # fmt: skip
    runs = [(t, backend, 1) for t in CONTEXTS for backend in CONTEXT_BACKENDS]
    runs += [(CONTEXT, backend, n) for n in THREADS for backend in CONTEXT_BACKENDS[1:]]
    for repeat in range(repeats):
        for t, backend, threads in runs if repeat % 2 == 0 else runs[::-1]:
            command = [sys.executable, str(Path(__file__).resolve()), "context-run",
                       "--data", str(data), "--backend", backend, "--context", str(t),
                       "--steps", str(steps), "--library", str(library),
                       "--threads", str(threads)]  # fmt: skip
            proc = subprocess.run(command, check=True, capture_output=True, text=True,
                                  timeout=3600)  # fmt: skip
            row = {"repeat": repeat, **json.loads(proc.stdout.strip().splitlines()[-1])}
            result["decoder"].append(row)
            print(json.dumps({key: row.get(key) for key in ("repeat", "backend", "context",
                  "threads", "median_step_ms", "peak_rss_mb", "arena_bytes")}), flush=True)  # fmt: skip
    summary = []
    for t, backend, threads in runs:
        mine = [r for r in result["decoder"]
                if (r["context"], r["backend"], r["threads"]) == (t, backend, threads)]  # fmt: skip
        timed = [s for r in mine for s in r["step_seconds"][2:]]
        row = {"context": t, "batch": TOKENS // t, "backend": backend, "threads": threads,
               "median_step_ms": 1000 * statistics.median(timed), "timed_steps": len(timed),
               "peak_rss_mb": max(r["peak_rss_mb"] for r in mine),
               "arena_bytes": mine[0].get("arena_bytes")}  # fmt: skip
        if backend != "torch-eager":  # the same batches, so the same losses up to rounding
            reference = {r["repeat"]: r["losses"] for r in result["decoder"]
                         if (r["context"], r["backend"], r["threads"]) ==
                         (t, "torch-eager", threads)}  # fmt: skip
            row["max_abs_loss_diff_vs_torch_eager"] = max(
                abs(a - b) for r in mine for a, b in zip(r["losses"], reference[r["repeat"]])
            )
        summary.append(row)
    result["decoder_summary"] = summary
    (data / "context.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"saved": str(data / "context.json")}), flush=True)


# --- summarize -------------------------------------------------------------------------------
# Quality levels for time to quality: the language model's validation cross-entropy (nats per
# byte), and the structured tasks' held-out exact match.
LM_LEVELS = (2.0, 1.75, 1.6)
EXACT_LEVELS = (0.5, 0.9, 0.99)


def summarize(data: Path):
    runs = {}
    for path in sorted((data / "runs").glob("*.json")):
        record = json.loads(path.read_text())
        runs[(record["task"], record["backend"])] = record
    table, out = [], {"runs": {}}
    for (task, backend), r in sorted(runs.items()):
        ref = runs.get((task, "torch-eager"))
        agree = None
        if ref is not None and backend != "torch-eager":
            pairs = list(zip(r["losses"], ref["losses"]))
            agree = {
                "max_abs_loss_diff": max(abs(a - b) for a, b in pairs),
                "max_abs_loss_diff_first_10": max(abs(a - b) for a, b in pairs[:10]),
                "mean_abs_loss_diff_last_100": statistics.mean(abs(a - b) for a, b in pairs[-100:]),
            }
        fin = r["final"]["test"]
        row = {
            "task": task, "backend": backend, "median_step_ms": r["median_step_ms"],
            "tokens_per_second": r["tokens_per_second"], "train_seconds": r["train_seconds"],
            "wall_seconds": r["wall_seconds"], "peak_rss_mb": r["rss_mb"]["peak"],
            "final_test": fin, "final_validation": r["final"].get("validation"),
            "loss_agreement_vs_torch_eager": agree, "state_sha256": r["state_sha256"],
            "compile_and_first_steps_seconds": r.get("compile_and_first_steps_seconds"),
        }  # fmt: skip
        out["runs"][f"{task}/{backend}"] = row
        table.append(
            f"| {task} | {backend} | {r['median_step_ms']:.0f} | {r['tokens_per_second']:.0f} | "
            f"{r['train_seconds'] / 60:.1f} | {fin['cross_entropy']:.4f} | {fin['accuracy']:.4f} | "
            f"{fin['exact_match']:.4f} | {r['rss_mb']['peak']:.0f} |"
        )
    header = ("| task | backend | ms/step | tokens/s | train min | test CE (nats) | test accuracy | "
              "exact match (lm: words) | peak RSS MiB |\n|---|---|---|---|---|---|---|---|---|")  # fmt: skip
    out["table"] = header + "\n" + "\n".join(table)
    # Time to quality: each backend's step and training seconds at the first evaluation that
    # reached a level -- for the language model fixed validation cross-entropies and the worst
    # final one among the backends (a quality every backend reached), for the structured tasks
    # fixed held-out exact-match levels.
    out["time_to_target"] = {}
    for task in sorted({t for t, _ in runs}):
        group = {b: r for (t, b), r in runs.items() if t == task}
        if task == "lm":
            worst = max(r["evals"][-1]["cross_entropy"] for r in group.values())
            goals = {f"validation cross-entropy <= {level:.4f} nats/byte":
                     lambda e, level=level: e["cross_entropy"] <= level
                     for level in (*LM_LEVELS, worst)}  # fmt: skip
        else:
            goals = {f"held-out exact match >= {g}": lambda e, g=g: e["exact_match"] >= g
                     for g in EXACT_LEVELS}  # fmt: skip
        out["time_to_target"][task] = {
            what: {b: next(({"step": e["step"], "seconds": e["train_seconds"]}
                            for e in r["evals"] if hit(e)), None) for b, r in group.items()}
            for what, hit in goals.items()
        }  # fmt: skip
    out["libraries"] = _libraries()
    (data / "summary.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(out["table"])
    print(json.dumps(out["time_to_target"], indent=1))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lm-steps", type=int, default=1000)
    p.add_argument("--copy-steps", type=int, default=1000)
    p.add_argument("--recall-steps", type=int, default=1000)
    p.add_argument("--add-steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=SEED, help="another replicate's weights/schedules")
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--task", choices=("lm", *TASKS), required=True)
    t.add_argument("--backend", choices=BACKENDS, required=True)
    t.add_argument("--steps", type=int)
    t.add_argument("--eval-every", type=int, default=100)
    t.add_argument("--cc", default="gcc")
    t.add_argument("--lr-scale", type=float, default=1.0, help="scale every learning rate")
    t.add_argument("--threads", type=int, default=1,
                   help="the BCIR pool's size, or PyTorch's intra-op threads (default 1)")  # fmt: skip
    t.add_argument("--contract", default="v1",
                   help="a NativeDecoder contract (bcir backends): v1, kblock+exp+reduce, bf16, ...")  # fmt: skip
    i = sub.add_parser("identity")
    i.add_argument("--data", type=Path, required=True)
    i.add_argument("--steps", type=int, default=20)
    i.add_argument("--cc", action="append")
    i.add_argument("--threads", default="1", help="comma-separated pool sizes, each a row")
    i.add_argument("--contract", default="v1", help="comma-separated contracts, each a row")
    i.add_argument("--variants", default="widest,avx2,portable",
                   help="comma-separated builds: widest, avx512, avx2, portable")  # fmt: skip
    bb = sub.add_parser(
        "bcir-breakdown", help="the BCIR side of breakdown alone: no PyTorch needed"
    )
    bb.add_argument("--data", type=Path, required=True)
    bb.add_argument("--cc", default="gcc")
    bb.add_argument("--rounds", type=int, default=5)
    bb.add_argument("--threads", type=int, default=1)
    bb.add_argument("--contract", default="v1")
    bb.add_argument("--openblas", action="store_true")
    b = sub.add_parser("baseline")
    b.add_argument("--data", type=Path, required=True)
    b.add_argument("--steps", type=int, default=6)
    b.add_argument("--cc", default="gcc")
    b.add_argument("--base", default=CORPUS_COMMIT)
    b.add_argument(
        "--old-flags", default=" ".join(BASE_FLAGS), help="the base build's optimization flags"
    )
    k = sub.add_parser("kernels")
    k.add_argument("--data", type=Path, required=True)
    k.add_argument("--rounds", type=int, default=3)
    k.add_argument("--cc", default="gcc")
    k.add_argument("--base", default=CORPUS_COMMIT)
    k.add_argument(
        "--old-flags", default=" ".join(BASE_FLAGS), help="the base build's optimization flags"
    )
    r = sub.add_parser("breakdown")
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--rounds", type=int, default=5)
    r.add_argument("--cc", action="append")
    c = sub.add_parser("context")
    c.add_argument("--data", type=Path, required=True)
    c.add_argument("--cc", default="gcc")
    c.add_argument("--rounds", type=int, default=5)
    c.add_argument("--steps", type=int, default=12)
    c.add_argument("--repeats", type=int, default=2)
    cr = sub.add_parser("context-run", help="one process of the context stage (internal)")
    cr.add_argument("--data", type=Path, required=True)
    cr.add_argument("--backend", choices=CONTEXT_BACKENDS, required=True)
    cr.add_argument("--context", type=int, choices=CONTEXTS, required=True)
    cr.add_argument("--steps", type=int, default=8)
    cr.add_argument("--library", type=Path)
    cr.add_argument("--threads", type=int, default=1)
    s = sub.add_parser("summarize")
    s.add_argument("--data", type=Path, required=True)
    a = parser.parse_args()
    if a.stage == "prepare":
        budgets = {"lm": a.lm_steps, "copy": a.copy_steps, "recall": a.recall_steps,
                   "add": a.add_steps}  # fmt: skip
        prepare(a.output, budgets, a.seed)
    elif a.stage == "train":
        train(a.data, a.task, a.backend, a.steps, a.eval_every, a.cc, a.lr_scale, a.threads,
              a.contract)  # fmt: skip
    elif a.stage == "identity":
        identity(a.data, a.steps, a.cc or ["gcc"], tuple(int(n) for n in a.threads.split(",")),
                 tuple(a.contract.split(",")), tuple(a.variants.split(",")))  # fmt: skip
    elif a.stage == "bcir-breakdown":
        bcir_breakdown_stage(a.data, a.cc, a.rounds, a.threads, a.contract, a.openblas)
    elif a.stage == "baseline":
        baseline(a.data, a.steps, a.cc, a.base, a.old_flags.split())
    elif a.stage == "kernels":
        kernels(a.data, a.cc, a.base, a.rounds, a.old_flags.split())
    elif a.stage == "breakdown":
        breakdown(a.data, a.cc or ["gcc"], a.rounds)
    elif a.stage == "context":
        context(a.data, a.cc, a.rounds, a.steps, a.repeats)
    elif a.stage == "context-run":
        row = _context_run(a.data, a.backend, a.context, a.steps, a.library, a.threads)
        print(json.dumps(row), flush=True)
    else:
        summarize(a.data)


if __name__ == "__main__":
    main()
