from __future__ import annotations

import hashlib
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.models import fetch_pinned_model as fetcher


def test_pinned_download_uses_private_staging_files() -> None:
    payload = b"BCIR pinned model asset\n"
    info = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    rendezvous = threading.Barrier(2)

    class Response:
        def __init__(self) -> None:
            self.done = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self, _size: int) -> bytes:
            if self.done:
                return b""
            self.done = True
            rendezvous.wait(timeout=5)
            return payload

    original = fetcher.urllib.request.urlopen
    fetcher.urllib.request.urlopen = lambda _request, timeout=0: Response()
    try:
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "model.safetensors"
            # The old downloader unlinked this predictable name.  It must now
            # be outside the operation's ownership, whether it is a regular
            # file or (on platforms that permit it) an attacker-made symlink.
            legacy_part = target.with_name(target.name + ".part")
            legacy_part.write_bytes(b"not ours")
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(fetcher._download, "https://invalid.example/model", target, info, 1)
                    for _ in range(2)
                ]
                for future in futures:
                    future.result(timeout=10)
            assert target.read_bytes() == payload
            assert legacy_part.read_bytes() == b"not ours"
            assert list(Path(td).glob(f".{target.name}.*.part")) == []
    finally:
        fetcher.urllib.request.urlopen = original


def test_publish_retries_a_lost_windows_rename_race_but_never_a_posix_one() -> None:
    """The publish step tolerates ERROR_ACCESS_DENIED on Windows, and only there.

    `os.replace` is atomic everywhere, which is not the same as always permitted: on
    Windows `MoveFileEx` refuses when the *destination* is momentarily open, so two gates
    publishing the same pin can have one of them lose the race even though its bytes are
    already written, fsynced and checksum-verified. Discarding a verified download over a
    scheduling accident is the wrong repair, so the publish retries.

    The second half is the more important one. POSIX has no such rule, so a
    `PermissionError` there is a genuine permissions fault and must surface on the first
    try -- a retry that fired on every platform would turn a real misconfiguration into a
    slow one.
    """
    real_replace = fetcher.os.replace
    real_name = fetcher.os.name
    calls: list[int] = []

    def flaky(src, dst, *, refusals: int):
        calls.append(1)
        if len(calls) <= refusals:
            raise PermissionError(5, "Access is denied")
        return real_replace(src, dst)

    try:
        with tempfile.TemporaryDirectory() as td:
            part = Path(td) / "staged"
            target = Path(td) / "published"

            fetcher.os.name = "nt"
            calls.clear()
            part.write_bytes(b"payload")
            fetcher.os.replace = lambda s, d: flaky(s, d, refusals=2)
            fetcher._publish(part, target)
            assert target.read_bytes() == b"payload"
            assert len(calls) == 3, "the first two refusals must be retried, not fatal"

            # And the bound holds: refusals that never stop are re-raised, not looped on.
            calls.clear()
            part.write_bytes(b"payload")
            fetcher.os.replace = lambda s, d: flaky(s, d, refusals=99)
            try:
                fetcher._publish(part, target, tries=3)
            except PermissionError:
                assert len(calls) == 3, "the retry budget must be exactly `tries`"
            else:
                raise AssertionError("an unyielding refusal must be re-raised")

            fetcher.os.name = "posix"
            calls.clear()
            part.write_bytes(b"payload")
            fetcher.os.replace = lambda s, d: flaky(s, d, refusals=1)
            try:
                fetcher._publish(part, target)
            except PermissionError:
                assert len(calls) == 1, "POSIX must not retry a genuine permissions fault"
            else:
                raise AssertionError("a POSIX PermissionError must not be retried")
    finally:
        fetcher.os.replace = real_replace
        fetcher.os.name = real_name
