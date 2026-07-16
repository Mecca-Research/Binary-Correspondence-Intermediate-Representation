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
                    pool.submit(fetcher._download, "https://invalid.example/model", target,
                                info, 1)
                    for _ in range(2)
                ]
                for future in futures:
                    future.result(timeout=10)
            assert target.read_bytes() == payload
            assert legacy_part.read_bytes() == b"not ours"
            assert list(Path(td).glob(f".{target.name}.*.part")) == []
    finally:
        fetcher.urllib.request.urlopen = original
