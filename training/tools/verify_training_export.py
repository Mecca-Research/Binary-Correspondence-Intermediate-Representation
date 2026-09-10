#!/usr/bin/env python3
"""Gate the corpus-to-BCIR training edge.

The export exists so `training/` stops being a passive database, so what this
checks is that the edge is real and honest rather than that a file was written:

  1. **Anti-vacuity.** An export with no documents or no examples FAILS.
  2. **BCIR's contracts, not lookalikes.** Every emitted example is fed back
     through the real `SFTExample` / `PreferenceExample` constructors. Those
     validate token tuples, digests and weights themselves, so a record this
     corpus shaped wrongly cannot be reconstructed -- which is the point of
     using BCIR's contracts instead of writing a schema here.
  3. **No false preference label.** A preference pair claims the assembler
     refuses its rejected side. Fixtures that are declared invalid and
     nevertheless ASSEMBLE (`semantic-only`, `assemble-valid-semantically-risky`)
     must never appear there: putting valid IR on the losing side teaches the
     opposite of the intended lesson, and no schema would catch it.
  4. **The tokenizer round-trips.** BCIR's byte-fallback contract means any
     corpus text survives encode/decode; if it does not, the vocabulary is
     broken and every token id downstream is wrong.
  5. **The ledger is BCIR's and it parses.** Re-read through BCIR's own reader,
     with its DAG rules enforced -- and it records only stages the corpus
     actually produced.
  6. **Determinism.** Two exports of the same tree agree.

    python3 training/tools/verify_training_export.py
    python3 training/tools/verify_training_export.py --require-bcir
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
CORPUS_ROOT = TOOLS_DIR.parent
REPO_ROOT = CORPUS_ROOT.parent

# Bounded so the gate stays well under a minute; the manifest records that the
# tokenizer saw a sample, so the artifact never overstates what it was built on.
GATE_VOCAB = 384
GATE_TOKENIZER_DOCUMENTS = 200
LEDGER_STAGES = ("data", "tokenizer")


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def require(self, condition: bool, message: str) -> bool:
        self.checks += 1
        if not condition:
            self.failures.append(message)
        return bool(condition)


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS_DIR / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def check_examples(out: Path, contracts, report: Report) -> None:
    sft = read_jsonl(out / "sft.jsonl")
    preferences = read_jsonl(out / "preferences.jsonl")
    report.require(bool(sft), "the export produced no supervised examples")
    report.require(bool(preferences), "the export produced no preference pairs")

    for index, row in enumerate(sft):
        try:
            contracts.SFTExample(
                prompt_ids=tuple(row["prompt_ids"]),
                response_ids=tuple(row["response_ids"]),
                provenance_sha256=row["provenance_sha256"],
                weight=row["weight"],
            )
        except (ValueError, KeyError, TypeError) as exc:
            report.require(False, f"sft[{index}] is not a valid SFTExample: {exc}")
            break

    for index, row in enumerate(preferences):
        try:
            contracts.PreferenceExample(
                prompt_ids=tuple(row["prompt_ids"]),
                chosen_ids=tuple(row["chosen_ids"]),
                rejected_ids=tuple(row["rejected_ids"]),
                provenance_sha256=row["provenance_sha256"],
                weight=row["weight"],
            )
        except (ValueError, KeyError, TypeError) as exc:
            report.require(False, f"preferences[{index}] is not valid: {exc}")
            break


def check_preference_truth(out: Path, exporter, report: Report) -> None:
    """The rejected side must be IR the assembler actually refuses."""
    preferences = read_jsonl(out / "preferences.jsonl")
    mislabelled: list[str] = []
    for row in preferences:
        rejected = REPO_ROOT / row["pair"]["rejected"]
        chosen = REPO_ROOT / row["pair"]["chosen"]
        if not rejected.is_file() or not chosen.is_file():
            report.require(False, f"preference pair names a missing file: {row['pair']}")
            continue
        if not exporter._fixture_is_rejected(rejected.read_text(encoding="utf-8")):
            mislabelled.append(row["pair"]["rejected"])
    report.require(
        not mislabelled,
        f"{len(mislabelled)} preference pair(s) put IR the assembler ACCEPTS on the "
        f"rejected side: {', '.join(mislabelled[:3])}",
    )
    print(f"[prefer]  {len(preferences)} pair(s), rejected side verified as refused")


def check_tokenizer(out: Path, bpe, report: Report) -> None:
    tokenizer = bpe.BytePairTokenizer.from_json((out / "tokenizer.json").read_text())
    report.require(tokenizer.vocab_size >= 256, "vocabulary smaller than the byte fallback")
    probes = [
        "define i32 @main() { ret i32 0 }",
        "getelementptr inbounds",
        "éè unicode and punctuation: |>+-*/",
    ]
    for probe in probes:
        report.require(
            tokenizer.decode(tokenizer.encode(probe)) == probe,
            f"tokenizer does not round-trip {probe[:32]!r}; byte fallback is broken",
        )
    print(f"[token]   vocab {tokenizer.vocab_size}, {len(probes)} round-trip probe(s)")


def check_ledger(out: Path, pipeline, report: Report) -> None:
    ledger = pipeline.read_pipeline_ledger(out / "ledger.json")
    stages = tuple(record.stage for record in ledger.records)
    report.require(
        stages == LEDGER_STAGES,
        f"ledger records {stages}, expected {LEDGER_STAGES}; a stage name like 'sft' "
        "would claim a model was trained here, and none was",
    )
    print(f"[ledger]  {' -> '.join(stages)} (BCIR's DAG rules enforced on read)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-bcir", action="store_true")
    args = parser.parse_args(argv)

    native = load_tool("bcir_native")
    exporter = load_tool("export_training_examples")
    chunks_module = load_tool("build_chunks")
    distill_module = load_tool("build_distillation")

    try:
        hosted = native.load_hosted_training()
    except native.BackendUnavailable as exc:
        if args.require_bcir:
            print(f"verify_training_export: BCIR required: {exc}", file=sys.stderr)
            return 1
        print(f"[skip]    {exc}")
        print("[skip]    the corpus-to-BCIR edge is unverified here, not verified-absent")
        return 0

    report = Report()
    workspace = Path(tempfile.mkdtemp(prefix="bcir-training-export-"))
    try:
        chunk_dir, distill_dir = workspace / "chunks", workspace / "distill"
        chunks_module.main(["--out", str(chunk_dir)])
        distill_module.main(["--out", str(distill_dir)])

        def run(target: Path) -> dict:
            return exporter.export(
                chunk_dir=chunk_dir,
                distill_dir=distill_dir,
                out=target,
                vocab=GATE_VOCAB,
                min_frequency=3,
                tokenizer_documents=GATE_TOKENIZER_DOCUMENTS,
            )

        first = run(workspace / "a")
        report.require(first["documents"]["prepared"] > 0, "the export prepared no documents")
        report.require(
            first["documents"]["validation"] > 0,
            "no validation split; an export with nothing held out cannot be evaluated",
        )
        report.require(
            sorted(first["consumes"]) == sorted(f"bcir.hosted.training.{n}" for n in hosted),
            "the manifest does not name the BCIR modules the export actually used",
        )
        print(f"[consume] {', '.join(first['consumes'])}")
        print(
            f"[corpus]  {first['documents']['prepared']} prepared "
            f"({first['documents']['train']} train / {first['documents']['validation']} validation)"
        )

        check_examples(workspace / "a", hosted["contracts"], report)
        check_preference_truth(workspace / "a", exporter, report)
        check_tokenizer(workspace / "a", hosted["bpe"], report)
        check_ledger(workspace / "a", hosted["pipeline"], report)

        second = run(workspace / "b")
        for key in ("corpus_sha256", "ledger_sha256"):
            report.require(
                first[key] == second[key],
                f"two exports disagree on {key}; the edge is not deterministic",
            )
        report.require(
            first["example_digests"] == second["example_digests"],
            "two exports produce different examples from the same corpus",
        )
        print("[det]     two exports agree exactly")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    if report.failures:
        print("training export gate: FAILED", file=sys.stderr)
        for failure in report.failures[:40]:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"training export gate: PASSED ({report.checks} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
