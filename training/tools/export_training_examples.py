#!/usr/bin/env python3
"""Feed the corpus into BCIR's own training stack, not into a JSON dump.

Tier 3 emits chat-format records, and until now nothing in this repository
consumed them: the corpus was a database that happened to sit next to a training
system. BCIR already owns every piece needed to close that gap --

    bcir.hosted.training.data       provenance-preserving corpus preparation
    bcir.hosted.training.bpe        a deterministic byte-fallback tokenizer
    bcir.hosted.training.contracts  SFTExample, PreferenceExample, and friends
    bcir.hosted.training.pipeline   an append-only content-addressed ledger

-- so this builds the edges rather than a second stack. The corpus becomes the
*corpus provider* for those components: chunks become `RawDocument`s that BCIR
prepares and splits, BCIR's tokenizer is trained on the result, distillation
records become `SFTExample`s under BCIR's contract, and every stage is recorded
in BCIR's ledger. Nothing here reimplements a tokenizer, a split policy, a
provenance digest, or an example schema.

**Preferences decided by a verifier, not by a rater.** Preference data normally
needs human labels, which this corpus has no way to produce honestly. It has
something better: artifacts whose status a *gate* already decided. A checked-in
exercise solution is IR the assembler accepts; an `*.invalid.ll.txt` fixture is
IR the assembler is proven to reject. "Accepted is preferred to rejected" is
then a legality verdict, not a taste judgment -- the same order this repository
applies everywhere else, where legality precedes cost.

The pairing is between *validity classes*, so the prompt says exactly that and
claims no topical relationship between the two responses. A pair whose rejected
side does not attempt the prompt's task would be a misleading label; asking for
"a module the verifier accepts" is a question both sides genuinely answer.

**The direction of the dependency is unchanged.** `training/` is never a build
dependency of BCIR. The import is lazy, through the single door in
`bcir_native.py`, and its absence is an honest skip rather than a local
reimplementation of a contract this corpus would then have invented for itself.

    python3 training/tools/export_training_examples.py --out build/training/export
    python3 training/tools/export_training_examples.py --require-bcir
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
CORPUS_ROOT = TOOLS_DIR.parent
REPO_ROOT = CORPUS_ROOT.parent

LICENSE = "LicenseRef-BCIR-NC-1.0"
BUILDER = "training/tools/export_training_examples.py"
SCHEMA = "bcir-training/export/v1"

DEFAULT_CHUNKS = Path("build/training/chunks")
DEFAULT_DISTILL = Path("build/training/distill")
DEFAULT_OUT = Path("build/training/export")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BCIR_UNAVAILABLE = 3

# Small enough that byte-fallback still carries most rare identifiers, large
# enough that common IR keywords earn their own token. Declared, not tuned: the
# gate checks properties of the tokenizer, never a vocabulary size.
DEFAULT_VOCAB = 1024
DEFAULT_MIN_FREQUENCY = 3
# 0 means every prepared training document. BPE training is superlinear in
# (corpus x merges) -- measured here at 79s for the full corpus at vocab 512 and
# over three minutes at vocab 1024 -- so the gate runs a bounded configuration
# and the manifest records how many documents the tokenizer actually saw. A
# tokenizer trained on a sample is a different artifact from one trained on all
# of it, and the set says which it is rather than leaving that to be assumed.
DEFAULT_TOKENIZER_DOCUMENTS = 0

LEGALITY_PROMPT = "Produce an LLVM IR module that the assembler and verifier accept."


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS_DIR / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def raw_documents(chunks: list[dict], data_module):
    """Corpus chunks, in the shape BCIR's corpus preparation already accepts.

    Every field it wants, the chunk already carries -- identity, text, source,
    licence, and the digest of the file the text came from. That correspondence
    is why this is an edge and not an adapter.
    """
    return [
        data_module.RawDocument(
            document_id=chunk["chunk_id"].split(":", 1)[1],
            text=chunk["text"],
            source=chunk["source_path"],
            license=LICENSE,
            source_sha256=chunk["source_sha256"],
        )
        for chunk in chunks
    ]


def sft_examples(records: list[dict], tokenizer, contracts):
    """Tier-3 records under BCIR's own supervised-example contract.

    The contract requires a `provenance_sha256`, which is exactly the discipline
    Tier 3 already keeps: a record exists only because a gate checks its answer,
    and the digest binds the example to the record that carries that gate.
    """
    examples = []
    for record in records:
        system, user, assistant = (message["content"] for message in record["messages"])
        prompt_ids = tokenizer.encode(f"{system}\n\n{user}", add_bos=True)
        response_ids = tokenizer.encode(assistant, add_eos=True)
        if not prompt_ids or not response_ids:
            continue
        examples.append(
            contracts.SFTExample(
                prompt_ids=tuple(prompt_ids),
                response_ids=tuple(response_ids),
                provenance_sha256=record["record_id"].split(":", 1)[1],
                weight=1.0,
            )
        )
    return examples


# A fixture is only a "rejected" sample if the ASSEMBLER refuses it. Two classes
# in this corpus are declared invalid and are nevertheless accepted:
#
#   assemble-valid-semantically-risky   assembles; the risk is in the semantics
#   semantic-only                       assembles; the defect is meaning, not form
#
# Labelling either as rejected would put valid IR on the losing side of a
# preference pair -- a false label, and worse than having fewer pairs. They are
# excluded by name rather than by a regex that happens to miss them, and the
# counts are reported so the exclusion is visible.
ACCEPTED_DESPITE_DECLARED_INVALID = ("assemble-valid", "semantic-only")
DECLARED_INVALID = re.compile(
    r";\s*(?:Intentionally invalid|Invalid)\b|;\s*adversarial-class:\s*intentionally-invalid",
    re.IGNORECASE,
)


def _fixture_is_rejected(text: str) -> bool:
    header = "\n".join(text.splitlines()[:6])
    if any(marker in header for marker in ACCEPTED_DESPITE_DECLARED_INVALID):
        return False
    return bool(DECLARED_INVALID.search(header))


def legality_preferences(tokenizer, contracts, subject_root: Path):
    """Preference pairs whose winner a gate already decided.

    `chosen` is IR the exercise gate assembles; `rejected` is IR the
    invalid-fixture gate proves the assembler refuses. No human ranked these.
    """
    accepted: list[tuple[str, str]] = []
    for solution in sorted(subject_root.rglob("*.solution.ll")):
        text = solution.read_text(encoding="utf-8").strip()
        if text:
            accepted.append((solution.relative_to(REPO_ROOT).as_posix(), text))

    rejected: list[tuple[str, str]] = []
    skipped_accepted = 0
    for fixture in sorted(subject_root.rglob("*.invalid.ll.txt")):
        text = fixture.read_text(encoding="utf-8")
        if not _fixture_is_rejected(text):
            skipped_accepted += 1
            continue
        body = "\n".join(
            line for line in text.splitlines() if not line.strip().startswith(";")
        ).strip()
        if body:
            rejected.append((fixture.relative_to(REPO_ROOT).as_posix(), body))

    counts = {
        "accepted_modules": len(accepted),
        "rejected_fixtures": len(rejected),
        "declared_invalid_but_assembles": skipped_accepted,
    }
    if not accepted or not rejected:
        return [], counts

    prompt_ids = tuple(tokenizer.encode(LEGALITY_PROMPT, add_bos=True))
    pairs = []
    # Deterministic round-robin: every rejected fixture is used, and each is
    # paired with a different accepted module, so no single solution dominates.
    for index, (bad_path, bad_text) in enumerate(rejected):
        good_path, good_text = accepted[index % len(accepted)]
        chosen = tuple(tokenizer.encode(good_text, add_eos=True))
        rejected_ids = tuple(tokenizer.encode(bad_text, add_eos=True))
        if not chosen or not rejected_ids or chosen == rejected_ids:
            continue
        identity = canonical_json(
            {"prompt": LEGALITY_PROMPT, "chosen": good_path, "rejected": bad_path}
        )
        pairs.append(
            (
                contracts.PreferenceExample(
                    prompt_ids=prompt_ids,
                    chosen_ids=chosen,
                    rejected_ids=rejected_ids,
                    provenance_sha256=contracts.sha256_text(identity),
                    weight=1.0,
                ),
                {"chosen": good_path, "rejected": bad_path},
            )
        )
    return pairs, counts


def export(
    *,
    chunk_dir: Path,
    distill_dir: Path,
    out: Path,
    vocab: int,
    min_frequency: int,
    tokenizer_documents: int = DEFAULT_TOKENIZER_DOCUMENTS,
) -> dict:
    native = load_tool("bcir_native")
    hosted = native.load_hosted_training()
    contracts, data, bpe, pipeline = (
        hosted["contracts"],
        hosted["data"],
        hosted["bpe"],
        hosted["pipeline"],
    )

    chunks = []
    for path in sorted(chunk_dir.glob("*.chunks.jsonl")):
        chunks.extend(read_jsonl(path))
    if not chunks:
        raise SystemExit(
            f"export_training_examples: no chunks at {chunk_dir}; an export of an "
            "empty corpus produces a tokenizer trained on nothing"
        )

    # 1. BCIR prepares and splits the corpus, and reports on what it dropped.
    spec = data.DataPreparationSpec(allowed_licenses=(LICENSE,), min_characters=32)
    corpus = data.prepare_corpus(raw_documents(chunks, data), spec)

    # 2. BCIR's tokenizer, trained on what BCIR prepared.
    train_docs = corpus.split("train")
    tokenizer_docs = train_docs if tokenizer_documents <= 0 else train_docs[:tokenizer_documents]
    tokenizer = bpe.BytePairTokenizer.train(
        [document.text for document in tokenizer_docs],
        vocab_size=vocab,
        min_frequency=min_frequency,
    )

    # 3. BCIR's example contracts.
    records = []
    for path in sorted(distill_dir.glob("*.distill.jsonl")):
        records.extend(read_jsonl(path))
    sft = sft_examples(records, tokenizer, contracts)
    preferences, legality_counts = legality_preferences(tokenizer, contracts, CORPUS_ROOT / "llvm")

    # 4. BCIR's ledger, recording only what the corpus actually produced.
    #
    # The ledger enforces a real training DAG, and in it `sft` means a MODEL was
    # trained on SFT examples -- not that examples exist. A first cut of this
    # appended `sft` and `preference` stages and the ledger rejected them, which
    # was the ledger being right: the corpus produces training *inputs* and runs
    # no training stage. So it records `data` and `tokenizer`, which are the two
    # artifacts it genuinely owns, and claims nothing downstream of them.
    ledger = pipeline.TrainingPipelineLedger()
    corpus_digest = corpus.digest
    tokenizer_digest = tokenizer.digest
    sft_digest = contracts.sha256_text(
        canonical_json([dataclasses.asdict(example) for example in sft])
    )
    preference_digest = contracts.sha256_text(
        canonical_json([dataclasses.asdict(example) for example, _ in preferences])
    )
    for stage, parents, output, kind in (
        ("data", (), corpus_digest, "prepared-corpus"),
        ("tokenizer", (corpus_digest,), tokenizer_digest, "byte-fallback-bpe"),
    ):
        ledger.append(
            pipeline.PipelineStageRecord(
                stage=stage,
                parent_sha256=parents,
                output_sha256=output,
                report_sha256=contracts.sha256_text(canonical_json({"stage": stage})),
                artifact_kind=kind,
            )
        )

    out.mkdir(parents=True, exist_ok=True)
    # BCIR's writer refuses to replace an existing prepared corpus, which is the
    # right default for a training artifact. This is a build directory being
    # regenerated, so the previous one is cleared explicitly rather than by
    # weakening that refusal.
    shutil.rmtree(out / "prepared", ignore_errors=True)
    data.write_prepared_corpus(corpus, out / "prepared")
    # Written with no trailing newline on purpose: BCIR's `from_json` rejects
    # anything that is not exactly its canonical serialization, so a courtesy
    # newline here would make the file unreadable by the very reader that
    # defines the format.
    (out / "tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")
    (out / "sft.jsonl").write_text(
        "".join(canonical_json(dataclasses.asdict(e)) + "\n" for e in sft), encoding="utf-8"
    )
    (out / "preferences.jsonl").write_text(
        "".join(
            canonical_json({**dataclasses.asdict(e), "pair": origin}) + "\n"
            for e, origin in preferences
        ),
        encoding="utf-8",
    )
    pipeline.write_pipeline_ledger(out / "ledger.json", ledger)

    manifest = {
        "schema": SCHEMA,
        "corpus": "training",
        "builder": BUILDER,
        "consumes": sorted(f"bcir.hosted.training.{name}" for name in hosted),
        "corpus_sha256": corpus_digest,
        "tokenizer": {
            "sha256": tokenizer_digest,
            "vocab_size": tokenizer.vocab_size,
            "min_frequency": min_frequency,
            "trained_on_documents": len(tokenizer_docs),
            "bounded": len(tokenizer_docs) < len(train_docs),
        },
        "documents": {
            "prepared": len(corpus.documents),
            "train": len(train_docs),
            "validation": len(corpus.split("validation")),
        },
        "examples": {
            "sft": len(sft),
            "preference": len(preferences),
            "preference_sources": legality_counts,
        },
        "preference_basis": (
            "A gate decided each side: chosen assembles, rejected is proven to be "
            "refused. No human ranked these, and the prompt asks for validity "
            "rather than implying the two responses attempt the same task. "
            "Fixtures declared invalid that the assembler nevertheless ACCEPTS "
            "(semantic-only, assemble-valid-semantically-risky) are excluded: "
            "putting valid IR on the losing side would be a false label."
        ),
        "example_digests": {"sft": sft_digest, "preference": preference_digest},
        "downstream": (
            "These are training INPUTS. The ledger records `data` and `tokenizer` "
            "because those are what this corpus produced; it records no `sft` or "
            "`dpo` stage, since no model was trained here and BCIR's ledger uses "
            "those names for trained artifacts."
        ),
        "ledger_sha256": ledger.digest,
        "license": LICENSE,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--distill", type=Path, default=DEFAULT_DISTILL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
    parser.add_argument("--min-frequency", type=int, default=DEFAULT_MIN_FREQUENCY)
    parser.add_argument(
        "--tokenizer-documents",
        type=int,
        default=DEFAULT_TOKENIZER_DOCUMENTS,
        help="bound the documents the tokenizer trains on (0 = all); recorded in the manifest",
    )
    parser.add_argument(
        "--require-bcir",
        action="store_true",
        help="fail instead of skipping when BCIR's hosted training stack is absent",
    )
    args = parser.parse_args(argv)

    native = load_tool("bcir_native")
    try:
        manifest = export(
            chunk_dir=args.chunks,
            distill_dir=args.distill,
            out=args.out,
            vocab=args.vocab,
            min_frequency=args.min_frequency,
            tokenizer_documents=args.tokenizer_documents,
        )
    except native.BackendUnavailable as exc:
        if args.require_bcir:
            print(f"export_training_examples: BCIR required: {exc}", file=sys.stderr)
            return EXIT_BCIR_UNAVAILABLE
        print(f"[skip]    {exc}")
        print("[skip]    nothing written -- a skip is never a pass")
        return EXIT_OK

    print(f"[consume] {', '.join(manifest['consumes'])}")
    print(
        f"[corpus]  {manifest['documents']['prepared']} prepared document(s) "
        f"({manifest['documents']['train']} train / "
        f"{manifest['documents']['validation']} validation)"
    )
    bounded = " (bounded sample)" if manifest["tokenizer"]["bounded"] else ""
    print(
        f"[token]   vocab {manifest['tokenizer']['vocab_size']} trained on "
        f"{manifest['tokenizer']['trained_on_documents']} document(s){bounded}"
    )
    print(
        f"[examples] {manifest['examples']['sft']} SFT, "
        f"{manifest['examples']['preference']} verifier-decided preference pair(s)"
    )
    print(f"[write]   {args.out}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
