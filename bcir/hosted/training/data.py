"""Deterministic, provenance-preserving corpus preparation for hosted training."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .contracts import canonical_json, require_int, require_sha256, sha256_text


@dataclass(frozen=True)
class DataPreparationSpec:
    allowed_licenses: tuple[str, ...]
    min_characters: int = 1
    max_document_bytes: int = 4 * 1024 * 1024
    validation_permyriad: int = 500
    split_salt: str = "bcir-data-v1"
    reject_control_characters: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.allowed_licenses, (tuple, list)) or not self.allowed_licenses \
                or any(not isinstance(item, str) or not item for item in self.allowed_licenses):
            raise ValueError("allowed_licenses must contain nonempty license identifiers")
        if len(set(self.allowed_licenses)) != len(self.allowed_licenses):
            raise ValueError("allowed_licenses contains duplicates")
        require_int(self.min_characters, "min_characters", minimum=1)
        require_int(self.max_document_bytes, "max_document_bytes", minimum=1,
                    maximum=1 << 30)
        require_int(self.validation_permyriad, "validation_permyriad", minimum=0,
                    maximum=9999)
        if not isinstance(self.split_salt, str) or not self.split_salt:
            raise ValueError("split_salt must be a nonempty string")
        if type(self.reject_control_characters) is not bool:
            raise ValueError("reject_control_characters must be bool")

    @property
    def digest(self) -> str:
        return sha256_text(canonical_json(asdict(self)))


@dataclass(frozen=True)
class RawDocument:
    document_id: str
    text: str
    source: str
    license: str
    source_sha256: str

    def __post_init__(self) -> None:
        for field in ("document_id", "text", "source", "license"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise ValueError(f"document {field} must be a nonempty string")
        require_sha256(self.source_sha256, "source_sha256")


@dataclass(frozen=True)
class PreparedDocument:
    document_id: str
    text: str
    source: str
    license: str
    source_sha256: str
    content_sha256: str
    split: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DataPreparationReport:
    accepted: int
    rejected_license: int
    rejected_size: int
    rejected_short: int
    rejected_control: int
    exact_duplicates: int
    train_documents: int
    validation_documents: int
    input_sha256: str
    corpus_sha256: str
    policy_sha256: str

    def __post_init__(self) -> None:
        for field in ("accepted", "rejected_license", "rejected_size", "rejected_short",
                      "rejected_control", "exact_duplicates", "train_documents",
                      "validation_documents"):
            require_int(getattr(self, field), field)
        for field in ("input_sha256", "corpus_sha256", "policy_sha256"):
            require_sha256(getattr(self, field), field)
        if self.accepted != self.train_documents + self.validation_documents:
            raise ValueError("accepted count does not match split counts")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PreparedCorpus:
    documents: tuple[PreparedDocument, ...]
    report: DataPreparationReport

    def split(self, name: str) -> tuple[PreparedDocument, ...]:
        if name not in ("train", "validation"):
            raise ValueError("split must be train or validation")
        return tuple(document for document in self.documents if document.split == name)

    @property
    def digest(self) -> str:
        return self.report.corpus_sha256


def _clean_text(text: str) -> tuple[str, bool]:
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    found_control = False
    output = []
    for character in text:
        category = unicodedata.category(character)
        if category == "Cc" and character not in "\n\t":
            found_control = True
            output.append(" ")
        else:
            output.append(character)
    # Trailing horizontal whitespace is not semantic training data.  Preserve line
    # boundaries and interior spaces; canonicalize only the unstable edge.
    cleaned = "\n".join(line.rstrip(" \t") for line in "".join(output).split("\n")).strip()
    return cleaned, found_control


def prepare_corpus(documents: Iterable[RawDocument],
                   spec: DataPreparationSpec) -> PreparedCorpus:
    if not isinstance(spec, DataPreparationSpec):
        raise ValueError("spec must be a DataPreparationSpec")
    rows = []
    input_digest = hashlib.sha256()
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    counts = {"rejected_license": 0, "rejected_size": 0, "rejected_short": 0,
              "rejected_control": 0, "exact_duplicates": 0}
    for index, document in enumerate(documents):
        if not isinstance(document, RawDocument):
            raise ValueError(f"document {index} is not a RawDocument")
        if document.document_id in seen_ids:
            raise ValueError(f"duplicate document_id {document.document_id!r}")
        seen_ids.add(document.document_id)
        input_digest.update(canonical_json(asdict(document)).encode("utf-8")); input_digest.update(b"\n")
        if document.license not in spec.allowed_licenses:
            counts["rejected_license"] += 1
            continue
        cleaned, found_control = _clean_text(document.text)
        if found_control and spec.reject_control_characters:
            counts["rejected_control"] += 1
            continue
        encoded = cleaned.encode("utf-8")
        if len(encoded) > spec.max_document_bytes:
            counts["rejected_size"] += 1
            continue
        if len(cleaned) < spec.min_characters:
            counts["rejected_short"] += 1
            continue
        content_sha = hashlib.sha256(encoded).hexdigest()
        if content_sha in seen_content:
            counts["exact_duplicates"] += 1
            continue
        seen_content.add(content_sha)
        split_value = int(hashlib.sha256(
            f"{spec.split_salt}\0{document.document_id}\0{content_sha}".encode()).hexdigest()[:16], 16)
        split = "validation" if split_value % 10000 < spec.validation_permyriad else "train"
        rows.append(PreparedDocument(
            document.document_id, cleaned, document.source, document.license,
            document.source_sha256, content_sha, split))
    rows.sort(key=lambda row: (row.split, row.content_sha256, row.document_id))
    corpus_payload = [row.to_dict() for row in rows]
    corpus_sha = sha256_text(canonical_json(corpus_payload))
    train = sum(row.split == "train" for row in rows)
    validation = len(rows) - train
    report = DataPreparationReport(
        accepted=len(rows), train_documents=train, validation_documents=validation,
        input_sha256=input_digest.hexdigest(), corpus_sha256=corpus_sha,
        policy_sha256=spec.digest, **counts)
    return PreparedCorpus(tuple(rows), report)


def write_prepared_corpus(corpus: PreparedCorpus, output_dir: os.PathLike | str) -> Path:
    """Atomically publish canonical JSONL train/validation shards and a manifest."""
    if not isinstance(corpus, PreparedCorpus):
        raise ValueError("corpus must be a PreparedCorpus")
    target = Path(output_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to replace prepared corpus {target}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        files = {}
        for split in ("train", "validation"):
            path = temporary / f"{split}.jsonl"
            digest = hashlib.sha256(); byte_count = 0
            with path.open("xb") as stream:
                for document in corpus.split(split):
                    payload = canonical_json(document.to_dict()).encode("utf-8") + b"\n"
                    stream.write(payload); digest.update(payload); byte_count += len(payload)
                stream.flush(); os.fsync(stream.fileno())
            files[path.name] = {"bytes": byte_count, "sha256": digest.hexdigest()}
        manifest = {"schema": "bcir.prepared_corpus.v1", "files": files,
                    "report": corpus.report.to_dict()}
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("xb") as stream:
            stream.write(canonical_json(manifest).encode("utf-8") + b"\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, target)
        return target
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
