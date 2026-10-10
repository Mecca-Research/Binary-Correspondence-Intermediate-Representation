#!/usr/bin/env python3
"""Build a BCIR-native language model from the ground up: corpus, tokenizer, training, BCIRQ8.

One recipe (`tools/models/configs/*.json`, schema `bcir.lab_model.v1`) states the whole model:
which tracked files form the corpus and how they split, the byte-fallback BPE it trains, the
`bcir.hosted_train.v1` decoder and optimizer, and the evaluation prompt. Every stage is BCIR's
own and nothing is downloaded:

* the corpus goes through `bcir.hosted.training.data` (license filter, exact dedup, a salted
  deterministic train/validation split, digests);
* the tokenizer is `bcir.hosted.training.bpe.BytePairTokenizer`, trained on a fixed prefix of
  the train split;
* the decoder trains in C (`bcir.hosted.models.native`, forward/backward/AdamW), from a seed,
  under a warmup-then-cosine learning rate;
* the weights leave as a checkpoint (`.bcirdt`) and as BCIRQ8 (`write_q8_decoder`), the artifact
  every BCIR inference rail reads.

The output directory receives `corpus.json`, `tokenizer.json`, `train_log.json`, `model.bcirdt`,
`model.bcirq8` and `model_card.json`, which ties them together by SHA-256. The corpus is the
tracked files at the checked-out commit, so a model is reproducible from its commit and recipe.

    python3 tools/models/build_lab_model.py tools/models/configs/bcir-docs-860k.json \\
        --output-dir build/lab-model
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bcir.frontends.models.decode import decode_with_kv_cache  # noqa: E402
from bcir.frontends.models.weights_io import read_q8_decoder, write_q8_decoder  # noqa: E402
from bcir.hosted.models.native import NativeAdamW, NativeDecoder, parameter_layout  # noqa: E402
from bcir.hosted.models.spec import HostedTrainSpec  # noqa: E402
from bcir.hosted.training.bpe import BytePairTokenizer, token_source_from_corpus  # noqa: E402
from bcir.hosted.training.data import (  # noqa: E402
    DataPreparationSpec,
    RawDocument,
    prepare_corpus,
)

RECIPE_SCHEMA = "bcir.lab_model.v1"
_RECIPE_FIELDS = {"schema", "name", "corpus", "tokenizer", "train", "evaluation"}
_CORPUS_FIELDS = {"include", "license", "section_marker", "split_salt", "validation_permyriad"}
_TOKENIZER_FIELDS = {"kind", "vocab_size", "min_frequency", "sample_bytes"}
_EVALUATION_FIELDS = {"prompt", "max_new_tokens", "validation_batches"}
#: Steps per native `run` call: one progress line and one learning-rate slice each.
_CHUNK = 50


def _unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def _exact(obj, fields: set, what: str) -> dict:
    if not isinstance(obj, dict) or set(obj) != fields:
        raise ValueError(f"{what} must have exactly the fields {sorted(fields)}")
    return obj


def _int(value, what: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{what} must be an integer >= {minimum}")
    return value


def load_recipe(path) -> tuple[dict, HostedTrainSpec, str]:
    """The recipe, its training spec and its SHA-256; every field checked before any work."""
    raw = Path(path).read_bytes()
    if len(raw) > 1 << 20:
        raise ValueError("a recipe is at most 1 MiB")
    try:
        recipe = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"malformed recipe JSON: {exc}") from exc
    _exact(recipe, _RECIPE_FIELDS, "the recipe")
    if recipe["schema"] != RECIPE_SCHEMA:
        raise ValueError(f"unsupported recipe schema {recipe['schema']!r}")
    if not isinstance(recipe["name"], str) or not recipe["name"]:
        raise ValueError("the recipe needs a name")
    corpus = _exact(recipe["corpus"], _CORPUS_FIELDS, "corpus")
    if (
        not isinstance(corpus["include"], list)
        or not corpus["include"]
        or any(
            not isinstance(p, str) or not p or p.startswith(("/", ":")) for p in corpus["include"]
        )
    ):
        raise ValueError("corpus.include must list relative pathspecs")
    for key in ("license", "section_marker", "split_salt"):
        if not isinstance(corpus[key], str) or not corpus[key]:
            raise ValueError(f"corpus.{key} must be a nonempty string")
    _int(corpus["validation_permyriad"], "corpus.validation_permyriad", 1)
    tokenizer = _exact(recipe["tokenizer"], _TOKENIZER_FIELDS, "tokenizer")
    if tokenizer["kind"] != "bpe":
        raise ValueError("tokenizer.kind must be 'bpe'")
    _int(tokenizer["vocab_size"], "tokenizer.vocab_size", 261)
    _int(tokenizer["min_frequency"], "tokenizer.min_frequency", 1)
    _int(tokenizer["sample_bytes"], "tokenizer.sample_bytes", 1)
    evaluation = _exact(recipe["evaluation"], _EVALUATION_FIELDS, "evaluation")
    if not isinstance(evaluation["prompt"], str) or not evaluation["prompt"]:
        raise ValueError("evaluation.prompt must be a nonempty string")
    _int(evaluation["max_new_tokens"], "evaluation.max_new_tokens", 1)
    _int(evaluation["validation_batches"], "evaluation.validation_batches", 1)
    spec = HostedTrainSpec.from_json(json.dumps(recipe["train"]))
    if spec.decoder.vocab_size != tokenizer["vocab_size"]:
        raise ValueError("the decoder's vocabulary must be the tokenizer's")
    return recipe, spec, hashlib.sha256(raw).hexdigest()


def sections(text: str, marker: str) -> list[str]:
    """`text` cut before every line that starts with `marker`; blank pieces dropped."""
    parts = re.split(rf"(?m)^(?={re.escape(marker)})", text)
    return [part for part in parts if part.strip()]


def corpus_documents(root: Path, corpus: dict) -> tuple[list[RawDocument], list[dict]]:
    """The tracked files the recipe selects, one document per section, and a file manifest."""
    listed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", *corpus["include"]],
        capture_output=True,
        check=True,
        timeout=120,
    ).stdout
    paths = sorted({p for p in listed.decode("utf-8").split("\0") if p})
    if not paths:
        raise ValueError("the corpus selects no tracked file")
    documents, files = [], []
    for rel in paths:
        raw = (root / rel).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        pieces = sections(raw.decode("utf-8"), corpus["section_marker"])
        files.append({"path": rel, "bytes": len(raw), "sha256": sha, "sections": len(pieces)})
        for index, piece in enumerate(pieces):
            documents.append(
                RawDocument(f"{rel}#{index}", piece, f"repo:{rel}", corpus["license"], sha)
            )
    return documents, files


def learning_rates(spec: HostedTrainSpec, steps: int) -> list[float]:
    """Linear warmup to the peak, then a half cosine to the floor, one rate per update."""
    out = []
    warm = min(spec.warmup_steps, steps)
    span = max(1, steps - warm)
    for step in range(steps):
        if step < warm:
            out.append(spec.learning_rate * (step + 1) / warm)
        else:
            progress = (step - warm) / span
            out.append(
                spec.min_learning_rate
                + 0.5
                * (spec.learning_rate - spec.min_learning_rate)
                * (1.0 + math.cos(math.pi * progress))
            )
    return out


def _flat(rows: list[list[int]]) -> tuple[list[int], list[int]]:
    xs, ys = [], []
    for row in rows:
        xs.extend(row[:-1])
        ys.extend(row[1:])
    return xs, ys


def validation_loss(trainer: NativeDecoder, source, batches: int) -> float:
    """Mean token cross-entropy over `batches` consecutive validation batches (no update)."""
    total = 0.0
    for index in range(batches):
        xs, ys = _flat(source.batch(index, trainer.batch_size, trainer.context_length))
        total += trainer.backward(xs, ys)
    return total / batches


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=60
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def build(recipe_path, output_dir, *, steps: int | None = None, log=print) -> dict:
    """Run the recipe end to end into `output_dir`; the model card."""
    recipe, spec, recipe_sha = load_recipe(recipe_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    steps = spec.steps if steps is None else _int(steps, "steps")
    started = time.monotonic()

    documents, files = corpus_documents(ROOT, recipe["corpus"])
    corpus_cfg = recipe["corpus"]
    corpus = prepare_corpus(
        documents,
        DataPreparationSpec(
            allowed_licenses=(corpus_cfg["license"],),
            validation_permyriad=corpus_cfg["validation_permyriad"],
            split_salt=corpus_cfg["split_salt"],
        ),
    )
    _write_json(out / "corpus.json", {"files": files, "report": corpus.report.to_dict()})
    log(
        f"corpus: {len(files)} files, {corpus.report.train_documents} train / "
        f"{corpus.report.validation_documents} validation sections, {corpus.digest[:16]}"
    )

    tok_cfg = recipe["tokenizer"]
    sample, size = [], 0
    for document in corpus.split("train"):
        if size >= tok_cfg["sample_bytes"]:
            break
        sample.append(document.text)
        size += len(document.text.encode("utf-8"))
    tokenizer = BytePairTokenizer.train(
        sample, vocab_size=tok_cfg["vocab_size"], min_frequency=tok_cfg["min_frequency"]
    )
    if tokenizer.vocab_size != spec.decoder.vocab_size:
        raise ValueError(
            f"the tokenizer reached {tokenizer.vocab_size} tokens, not the recipe's "
            f"{spec.decoder.vocab_size}: the sample has too few repeated pairs"
        )
    # The file is the tokenizer's canonical JSON exactly: from_json refuses any other spelling.
    with open(out / "tokenizer.json", "w", encoding="utf-8", newline="\n") as stream:
        stream.write(tokenizer.to_json())
    tokenizer_sha = _sha256(out / "tokenizer.json")
    train = token_source_from_corpus(corpus, tokenizer, split="train")
    valid = token_source_from_corpus(corpus, tokenizer, split="validation")
    log(
        f"tokenizer: {tokenizer.vocab_size} tokens from {size} sample bytes; train "
        f"{train.manifest.token_count} tokens, validation {valid.manifest.token_count}"
    )

    library = NativeDecoder.build(out / "native")
    trainer = NativeDecoder(
        library,
        spec.decoder,
        batch_size=spec.batch_size,
        context_length=spec.context_length,
        seed=spec.seed,
    )
    optimizer = NativeAdamW(
        lr=spec.learning_rate,
        beta1=spec.beta1,
        beta2=spec.beta2,
        epsilon=spec.epsilon,
        weight_decay=spec.weight_decay,
        grad_clip=spec.grad_clip,
    )
    rates = learning_rates(spec, steps)
    initial_valid = validation_loss(trainer, valid, recipe["evaluation"]["validation_batches"])
    events = []
    train_started = time.monotonic()
    for start in range(0, steps, _CHUNK):
        count = min(_CHUNK, steps - start)
        xs, ys = [], []
        for index in range(start, start + count):
            bx, by = _flat(train.batch(index, spec.batch_size, spec.context_length))
            xs += bx
            ys += by
        chunk = trainer.run(
            xs, ys, steps=count, optimizer=optimizer, learning_rates=rates[start : start + count]
        )
        for offset, event in enumerate(chunk):
            events.append(
                {
                    "step": start + offset,
                    "loss": event["loss"],
                    "grad_norm": event["grad_norm"],
                    "lr": rates[start + offset],
                }
            )
        mean = sum(e["loss"] for e in chunk) / len(chunk)
        log(f"steps {start + count}/{steps}: mean loss {mean:.4f}")
        if not math.isfinite(mean):
            raise ValueError(f"training diverged at step {start + count}")
    train_seconds = time.monotonic() - train_started
    final_valid = validation_loss(trainer, valid, recipe["evaluation"]["validation_batches"])
    _write_json(out / "train_log.json", {"events": events})

    trainer.save(out / "model.bcirdt")
    float_weights = trainer.decoder_weights()
    checkpoint_sha = _sha256(out / "model.bcirdt")
    write_q8_decoder(
        out / "model.bcirq8",
        spec.decoder,
        float_weights,
        source_hashes={"model": checkpoint_sha, "config": recipe_sha, "tokenizer": tokenizer_sha},
        tokenizer_ids={
            "bos": tokenizer.bos_token_id,
            "eos": tokenizer.eos_token_id,
            "pad": tokenizer.pad_token_id,
        },
        context_length=spec.context_length,
    )
    q8_spec, q8_weights, _meta = read_q8_decoder(out / "model.bcirq8")
    prompt = recipe["evaluation"]["prompt"]
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    generated = decode_with_kv_cache(
        prompt_ids, q8_spec, q8_weights, recipe["evaluation"]["max_new_tokens"]
    )
    layout_params = sum(n for _name, _shape, _offset, n in parameter_layout(spec.decoder))
    card = {
        "schema": "bcir.lab_model_card.v1",
        "name": recipe["name"],
        "recipe_sha256": recipe_sha,
        "train_spec_sha256": spec.digest,
        "decoder": recipe["train"]["decoder"],
        "parameters": layout_params,
        "corpus": {
            "files": len(files),
            "corpus_sha256": corpus.digest,
            "input_sha256": corpus.report.input_sha256,
            "train_documents": corpus.report.train_documents,
            "validation_documents": corpus.report.validation_documents,
            "train_tokens": train.manifest.token_count,
            "validation_tokens": valid.manifest.token_count,
            "license": corpus_cfg["license"],
        },
        "tokenizer": {
            "sha256": tokenizer_sha,
            "vocab_size": tokenizer.vocab_size,
            "sample_bytes": size,
        },
        "training": {
            "steps": steps,
            "tokens_seen": steps * spec.batch_size * spec.context_length,
            "first_loss": events[0]["loss"],
            "last_loss": events[-1]["loss"],
            "validation_loss_initial": initial_valid,
            "validation_loss_final": final_valid,
            "validation_perplexity_final": math.exp(final_valid),
            "seconds": train_seconds,
            "tokens_per_second": steps * spec.batch_size * spec.context_length / train_seconds,
        },
        "artifacts": {
            "checkpoint_sha256": checkpoint_sha,
            "bcirq8_sha256": _sha256(out / "model.bcirq8"),
            "bcirq8_bytes": (out / "model.bcirq8").stat().st_size,
        },
        "sample": {
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "generated_ids": generated,
            "text": tokenizer.decode(prompt_ids + generated),
        },
        "provenance": {
            "commit": _git(ROOT, "rev-parse", "HEAD"),
            "corpus_tree_dirty": bool(
                _git(ROOT, "status", "--porcelain", "--", *recipe["corpus"]["include"])
            ),
            "host": f"{platform.system()} {platform.machine()}",
            "python": platform.python_version(),
            "seconds": time.monotonic() - started,
        },
    }
    _write_json(out / "model_card.json", card)
    log(
        f"model: {layout_params} parameters, validation loss {initial_valid:.3f} -> "
        f"{final_valid:.3f} (perplexity {math.exp(final_valid):.1f}), "
        f"{card['training']['tokens_per_second']:.0f} tokens/s"
    )
    log(f"sample: {card['sample']['text']!r}")
    return card


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("recipe")
    parser.add_argument("--output-dir", default=str(ROOT / "build" / "lab-model"))
    parser.add_argument("--steps", type=int, help="override the recipe's step count")
    args = parser.parse_args(argv)
    try:
        build(args.recipe, args.output_dir, steps=args.steps)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"build_lab_model: FAIL {exc}", file=sys.stderr)
        return 1
    print("build_lab_model: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
