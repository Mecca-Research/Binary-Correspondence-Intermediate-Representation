#!/usr/bin/env python3
"""Dependency-free, checksum-first fetcher for BCIR's pinned model gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

_PINS = Path(__file__).with_name("model_pins.json")


def load_pins() -> dict:
    with _PINS.open(encoding="utf-8") as fh:
        pins = json.load(fh)
    if pins.get("schema") != "bcir.model_pins.v1":
        raise ValueError(f"{_PINS}: unsupported model pin schema")
    return pins


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_file(path: Path, info: dict) -> bool:
    return (path.is_file() and path.stat().st_size == int(info["bytes"])
            and _digest(path) == str(info["sha256"]).lower())


def _publish(part: Path, target: Path, tries: int = 8) -> None:
    """Move a verified staging file into place, tolerating a lost Windows rename race.

    ``os.replace`` is atomic on every platform, but atomic is not the same as always
    permitted: on Windows ``MoveFileEx`` fails with ERROR_ACCESS_DENIED (WinError 5, which
    Python raises as ``PermissionError``) when the *destination* is momentarily open —
    another gate publishing the same pin, a concurrent reader, or a virus scanner that
    opened the file the instant it appeared.  POSIX has no such rule and never takes this
    branch, so the retry is gated on the platform rather than applied blindly, where it
    would paper over a genuine permissions fault.

    Only the publish is retried, never the download.  By the time this is called the bytes
    are written, fsynced and checksum-verified, so a lost race is a scheduling accident and
    re-fetching a file we already hold and have proven correct would be the wrong repair.
    The wait is short and bounded (~1.8 s in total) and the last failure is re-raised, so a
    destination that is genuinely unwritable still fails rather than hanging.
    """
    for attempt in range(1, tries + 1):
        try:
            os.replace(part, target)
            return
        except PermissionError:
            if os.name != "nt" or attempt == tries:
                raise
            time.sleep(0.05 * attempt)


def _download(url: str, target: Path, info: dict, attempts: int = 3) -> None:
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        part: Path | None = None
        try:
            # Every attempt owns a private 0600 staging file.  A fixed
            # ``target.part`` lets concurrent gates unlink or replace each
            # other's in-flight download and is also a symlink-following trap
            # when a caller deliberately selects a shared cache directory.
            fd, part_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".part", dir=target.parent
            )
            part = Path(part_name)
            request = urllib.request.Request(url, headers={"User-Agent": "BCIR-model-gate/1"})
            with os.fdopen(fd, "wb") as out:
                with urllib.request.urlopen(request, timeout=60) as response:
                    written = 0
                    while chunk := response.read(1024 * 1024):
                        written += len(chunk)
                        if written > int(info["bytes"]):
                            raise ValueError(
                                f"downloaded {target.name} exceeds its pinned byte length"
                            )
                        out.write(chunk)
                    out.flush()
                    os.fsync(out.fileno())
            if not verify_file(part, info):
                raise ValueError(f"downloaded {target.name} failed size/SHA-256 verification")
            _publish(part, target)
            return
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
            if part is not None:
                part.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(attempt)
    raise RuntimeError(f"failed to fetch {target.name} after {attempts} attempts: {last_error}")


def fetch_model(pin_name: str = "tinyllama-v0", *, cache_dir: str | os.PathLike | None = None,
                offline: bool = False) -> tuple[Path, dict]:
    pins = load_pins()
    try:
        pin = pins["models"][pin_name]
    except KeyError as exc:
        raise ValueError(f"unknown model pin {pin_name!r}") from exc
    root = Path(cache_dir or os.environ.get("BCIR_MODEL_CACHE", "~/.cache/bcir/models"))
    model_dir = root.expanduser() / pin_name / str(pin["revision"])
    model_dir.mkdir(parents=True, exist_ok=True)
    base = (f"https://huggingface.co/{pin['repository']}/resolve/"
            f"{pin['revision']}/")
    for name, info in pin["files"].items():
        target = model_dir / name
        if verify_file(target, info):
            continue
        if offline:
            raise RuntimeError(f"offline model gate: missing or corrupt cached file {target}")
        _download(base + name, target, info)
    return model_dir, pin


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", default="tinyllama-v0")
    parser.add_argument("--cache-dir")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    model_dir, _ = fetch_model(args.pin, cache_dir=args.cache_dir, offline=args.offline)
    print(model_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
