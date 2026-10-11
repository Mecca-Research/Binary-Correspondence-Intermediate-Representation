"""Train one ~10M-parameter decoder on BCIR's native C trainer and on PyTorch, and compare.

Run with an environment that has NumPy and CPU PyTorch (experiment dependencies, never BCIR's);
the BCIR side needs only a C compiler. Stages run serially, and every `train` is its own
process so its peak resident memory is its own:

  prepare   the corpus (the repository's tracked Markdown at a pinned commit, bytes), the
            associative-recall and addition tasks with held-out problems, identical initial
            weights and one batch schedule per task, as long as its step budget -> DIR
  train     one backend on one task -> DIR/runs/<task>-<backend>.json
  identity  BCIR's training state, bit for bit, across ISA variants, compilers and a repeat
  baseline  the pre-update kernels against these, same steps: identical state, time per step
  kernels   the same, kernel by kernel: every GEMM shape of a step on every ISA variant,
            attention forward and backward, AdamW; identical outputs, time per call
  summarize every run -> DIR/summary.json and a Markdown table

  python train_comparison.py prepare --output DIR
  python train_comparison.py train --data DIR --task lm --backend bcir
  python train_comparison.py summarize --data DIR

Tasks: lm (next-byte prediction on the corpus), recall (in-context key/value retrieval: six
pairs, then a query, scored by held-out exact match) and add (three-digit addition, the sum
least-significant digit first; kept because no run left its plateau in budget, see the
report). The structured tasks mask unpredictable bytes out of the loss through their targets.

Backends: bcir (BCIR's own C kernels: the widest bit-identical ISA variant the CPU runs, no
FMA), bcir-openblas (the same trainer with OpenBLAS sgemm behind its CBLAS seam), torch-eager
(HostedLlama, torch.optim.AdamW defaults), torch-compiled (the same modules through
torch.compile, fused AdamW). All run one compute thread on CPU in FP32 from the same weights,
batches and learning rates. This is a measurement harness, not a production runtime or a CI
timing gate.
"""

from __future__ import annotations

import argparse
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
BACKENDS = ("bcir", "bcir-openblas", "torch-eager", "torch-compiled")


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


def _initial_weights(spec):
    """The native trainer's own initialization: norms one, every matrix N(0, 0.02)."""
    from bcir.hosted.models.native import parameter_layout

    rng = random.Random(SEED)
    flat = array("f")
    for name, _shape, _offset, count in parameter_layout(spec):
        if "layernorm" in name or name == "model.norm.weight":
            flat.extend([1.0] * count)
        else:
            flat.extend(rng.gauss(0.0, 0.02) for _ in range(count))
    return flat


def prepare(out: Path, budgets: dict):
    out.mkdir(parents=True, exist_ok=True)
    spec = _spec()
    corpus, documents = _corpus()
    structured = {"add": _addition(), "recall": _recall(), "copy": _copy()}
    weights = _initial_weights(spec)
    (out / "weights.f32").write_bytes(weights.tobytes())
    rng = random.Random(SEED + 2)
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
    out = {"cross_entropy": float(ce.mean()), "bits_per_token": float(ce.mean() / math.log(2)),
           "accuracy": float(hit.mean()), "tokens": int(mask.sum())}  # fmt: skip
    if spans:
        # teacher-forced: every answer position right is exactly greedy decoding's answer
        hits = [pred[w, f : l + 1] == y[w, f : l + 1] for w, f, l in spans]
        out["exact_match"] = sum(bool(h.all()) for h in hits) / len(hits)
        out["answer_accuracy"] = float(np.concatenate(hits).mean())
        out["problems"] = len(hits)
    return out


def _rss_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
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
               str(src / "bcir_decoder_train.c"), "-o", str(target), "-lm"]  # fmt: skip
    subprocess.run(command, check=True, capture_output=True, timeout=300)
    return target


def _openblas():
    import scipy_openblas32

    path = Path(scipy_openblas32.get_lib_dir()) / "libscipy_openblas.so"
    lib = ctypes.CDLL(str(path))
    lib.scipy_openblas_set_num_threads(1)
    lib.scipy_openblas_get_config.restype = ctypes.c_char_p
    return path, lib.scipy_openblas_get_config().decode()


def _native(data: Path, library: Path, *, openblas=False):
    from bcir.hosted.models.native import NativeDecoder, parameter_layout

    spec = _spec()
    nd = NativeDecoder(
        library, spec, batch_size=BATCH, context_length=CONTEXT, max_memory_bytes=4 << 30
    )
    flat = _load(data, "weights.f32", "f")
    nd.load_state_dict({name: flat[off : off + n] for name, _s, off, n in parameter_layout(spec)})
    info = {"provider": "portable-c"}
    if openblas:
        path, config = _openblas()
        nd.use_cblas(path, symbol="scipy_cblas_sgemm")
        info = {"provider": "scipy_cblas_sgemm", "openblas": config}
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
    lr_scale: float = 1.0,
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
    info: dict = {"backend": backend, "task": task, "steps": steps, "threads": 1,
                  "lr_scale": lr_scale}  # fmt: skip
    compile_s = None
    if backend.startswith("bcir"):
        from bcir.hosted.models.native import NativeAdamW

        library = build_library(data / "lib" / cc.replace("/", "_"), cc)
        nd, provider = _native(data, library, openblas=backend == "bcir-openblas")
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
    else:
        import torch
        import torch.nn.functional as F

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        model = _torch_model(data)
        compiled = backend == "torch-compiled"
        opt = torch.optim.AdamW(
            model.parameters(), lr=OPT["peak"], betas=(OPT["beta1"], OPT["beta2"]),
            eps=OPT["eps"], weight_decay=OPT["decay"], **({"fused": True} if compiled else {}),
        )  # fmt: skip
        info.update(torch=torch.__version__, cpu_capability=torch.backends.cpu.get_cpu_capability())
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

            loss_of = torch.compile(loss_of, fullgraph=True)
        else:
            loss_of = lambda x, y: model(x, y)[1]  # noqa: E731
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
            with torch.no_grad():
                for i in range(0, len(x), BATCH):
                    batch = torch.from_numpy(np.ascontiguousarray(x[i : i + BATCH]))
                    out.append(model(batch)[0].numpy())
            model.train()
            return np.concatenate(out)

        digest_fn = lambda: _state_digest_torch(model, opt)  # noqa: E731

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
        if s < 2 and backend == "torch-compiled":
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
        compile_and_first_steps_seconds=compile_s, state_sha256=digest_fn(),
        rss_mb={"start": rss_start, "ready": rss_ready, "end": _rss_mb(),
                "peak": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024},
        environment=_environment(),
    )  # fmt: skip
    out = data / "runs"
    out.mkdir(exist_ok=True)
    (out / f"{task}-{backend}.json").write_text(json.dumps(info, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"saved": f"{task}-{backend}", "median_step_ms": info["median_step_ms"],
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
def _native_steps(data, library, steps, *, openblas=False):
    from bcir.hosted.models.native import NativeAdamW

    stream = _load(data, "lm-train.u16", "H")
    offsets = _load(data, "lm-schedule.u32", "I")
    nd, _ = _native(data, library, openblas=openblas)
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
    return _state_digest_native(nd), times, losses


def identity(data: Path, steps: int, compilers):
    """The same 20 BCIR steps through every ISA variant, every compiler and a repeat: one state."""
    builds = []
    for cc in compilers:
        for label, defines in (("widest", ()), ("avx2", ("-DBCIR_TENSOR_MAX_VARIANT=2",)),
                               ("portable", ("-DBCIR_TENSOR_NO_DISPATCH",))):  # fmt: skip
            lib = build_library(data / "lib" / f"id-{Path(cc).name}-{label}", cc, defines)
            builds.append((f"{_compiler(cc)} / {label}", lib))
    rows = []
    for name, lib in builds + builds[:1]:
        digest, times, losses = _native_steps(data, lib, steps)
        rows.append({"build": name, "state_sha256": digest, "last_loss": losses[-1],
                     "median_step_ms": 1000 * statistics.median(times)})  # fmt: skip
        print(json.dumps(rows[-1]), flush=True)
    result = {"steps": steps, "rows": rows,
              "identical": len({r["state_sha256"] for r in rows}) == 1}  # fmt: skip
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


def baseline(data: Path, steps: int, cc: str, base: str):
    """The kernels before this update (the sources at `base`) against these on the same steps:
    the portable trainer's state must not move a bit, and each pair is timed interleaved."""
    old_src = _base_sources(data, base)
    old = build_library(data / "lib" / "baseline-old", cc, sources=old_src, flags=BASE_FLAGS)
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
    result = {"base": base, "compiler": _compiler(cc), "steps": steps, "rows": rows}
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


def kernels(data: Path, cc: str, base: str, rounds: int):
    """The trainer's kernels before this update (the sources at `base`, built as they shipped)
    and after, at the comparison's shape: every GEMM shape of one step on every ISA variant,
    attention forward and backward, AdamW. Rounds alternate the builds; every output must hash
    the same before and after, and across the variants."""
    out = data / "kernels"
    out.mkdir(parents=True, exist_ok=True)
    (out / "kernel_bench.c").write_text(KERNEL_C)
    builds = {}
    for label, src, flags, defines in (
        ("old", _base_sources(data, base), BASE_FLAGS, ()),
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
    identical = (
        all(h == hashes["old"][0] for h in hashes["old"] + hashes["new"])
        and all(g["variants_agree"] for s in samples["new"] for g in s["gemm"])
    )  # fmt: skip

    def med(label, pick):
        return statistics.median(pick(s) for s in samples[label])

    rows, totals = [], {}
    variants = sorted(first["new"]["gemm"][0]["ms"], key=int)
    for i, g in enumerate(first["old"]["gemm"]):
        row = {key: g[key] for key in ("name", "ta", "tb", "m", "n", "k", "calls")}
        row["old_ms"] = med("old", lambda s, i=i: s["gemm"][i]["ms"]["mm"])
        for v in variants:
            row[f"new_ms_variant_{v}"] = med("new", lambda s, i=i, v=v: s["gemm"][i]["ms"][v])
        rows.append(row)
    for key in ["old_ms"] + [f"new_ms_variant_{v}" for v in variants]:
        totals[key] = sum(row["calls"] * row[key] for row in rows)
    result = {
        "base": base,
        "compiler": _compiler(cc),
        "flags": {"old": list(BASE_FLAGS), "new": list(FLAGS)},
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


# --- summarize -------------------------------------------------------------------------------
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
                "mean_abs_loss_diff_last_100": statistics.mean(abs(a - b) for a, b in pairs[-100:]),
            }
        key = "test"
        fin = r["final"][key]
        row = {
            "task": task, "backend": backend, "median_step_ms": r["median_step_ms"],
            "tokens_per_second": r["tokens_per_second"], "train_seconds": r["train_seconds"],
            "peak_rss_mb": r["rss_mb"]["peak"], "final_test": fin,
            "final_validation": r["final"].get("validation"), "loss_agreement_vs_torch_eager": agree,
            "compile_and_first_steps_seconds": r.get("compile_and_first_steps_seconds"),
        }  # fmt: skip
        out["runs"][f"{task}/{backend}"] = row
        metric = fin.get("exact_match", fin["bits_per_token"])
        table.append(
            f"| {task} | {backend} | {r['median_step_ms']:.0f} | {r['tokens_per_second']:.0f} | "
            f"{r['train_seconds'] / 60:.1f} | {fin['cross_entropy']:.4f} | {fin['accuracy']:.4f} | "
            f"{metric:.4f} | {r['rss_mb']['peak']:.0f} |"
        )
    header = ("| task | backend | ms/step | tokens/s | train min | test CE (nats) | test accuracy | "
              "bits/byte or exact match | peak RSS MB |\n|---|---|---|---|---|---|---|---|---|")  # fmt: skip
    out["table"] = header + "\n" + "\n".join(table)
    # Time to quality: for the language model, the worst final validation cross-entropy among
    # the backends (a quality every backend reached); for addition, fixed held-out exact-match
    # levels. Each backend's step and training seconds at the first evaluation that met it.
    out["time_to_target"] = {}
    for task in sorted({t for t, _ in runs}):
        group = {b: r for (t, b), r in runs.items() if t == task}
        if task == "lm":
            worst = max(r["evals"][-1]["cross_entropy"] for r in group.values())
            goals = {f"validation cross-entropy <= {worst:.4f} nats/byte":
                     lambda e, worst=worst: e["cross_entropy"] <= worst}  # fmt: skip
        else:
            goals = {f"held-out exact match >= {g}": lambda e, g=g: e["exact_match"] >= g
                     for g in (0.5, 0.9, 0.99)}  # fmt: skip
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
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--task", choices=("lm", *TASKS), required=True)
    t.add_argument("--backend", choices=BACKENDS, required=True)
    t.add_argument("--steps", type=int)
    t.add_argument("--eval-every", type=int, default=100)
    t.add_argument("--cc", default="gcc")
    t.add_argument("--lr-scale", type=float, default=1.0, help="scale every learning rate")
    i = sub.add_parser("identity")
    i.add_argument("--data", type=Path, required=True)
    i.add_argument("--steps", type=int, default=20)
    i.add_argument("--cc", action="append")
    b = sub.add_parser("baseline")
    b.add_argument("--data", type=Path, required=True)
    b.add_argument("--steps", type=int, default=6)
    b.add_argument("--cc", default="gcc")
    b.add_argument("--base", default=CORPUS_COMMIT)
    k = sub.add_parser("kernels")
    k.add_argument("--data", type=Path, required=True)
    k.add_argument("--rounds", type=int, default=3)
    k.add_argument("--cc", default="gcc")
    k.add_argument("--base", default=CORPUS_COMMIT)
    s = sub.add_parser("summarize")
    s.add_argument("--data", type=Path, required=True)
    a = parser.parse_args()
    if a.stage == "prepare":
        budgets = {"lm": a.lm_steps, "copy": a.copy_steps, "recall": a.recall_steps,
                   "add": a.add_steps}  # fmt: skip
        prepare(a.output, budgets)
    elif a.stage == "train":
        train(a.data, a.task, a.backend, a.steps, a.eval_every, a.cc, a.lr_scale)
    elif a.stage == "identity":
        identity(a.data, a.steps, a.cc or ["gcc"])
    elif a.stage == "baseline":
        baseline(a.data, a.steps, a.cc, a.base)
    elif a.stage == "kernels":
        kernels(a.data, a.cc, a.base, a.rounds)
    else:
        summarize(a.data)


if __name__ == "__main__":
    main()
