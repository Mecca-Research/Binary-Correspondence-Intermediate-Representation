"""G21 PLAN-SIGN on the C rail: `runtime/c/bcir_ed25519.c` and `bcir_plan_sign.c` held to the
RFC 8032 vectors, to the oracle's signatures over seeded keys and messages, and to the oracle's
verdict on every case of `plan_sign_fixtures.cases()`. Builds `runtime/c/test_plan_sign.c`;
without a C compiler (the quick tier hides one on purpose) every test returns early.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile

from bcir.abi import ed25519
from bcir.tests import plan_sign_fixtures as fx

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_C = os.path.join(_ROOT, "runtime", "c")
_UNITS = ("test_plan_sign.c", "bcir_plan_sign.c", "bcir_ed25519.c", "bcir_sha256.c")


def _build(tmp: str) -> str | None:
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        return None
    exe = os.path.join(tmp, "test_plan_sign")
    cmd = [cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-I", _C]
    subprocess.run(cmd + [os.path.join(_C, u) for u in _UNITS] + ["-o", exe], check=True)
    return exe


def _write(tmp: str, name: str, data: bytes) -> str:
    path = os.path.join(tmp, name)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def test_the_c_twin_reproduces_the_rfc_8032_vectors():
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return
        done = subprocess.run([exe, "vectors"], capture_output=True, text=True)
        assert done.returncode == 0 and done.stdout.strip() == "vectors=ok", done.stdout


def test_both_rails_sign_alike():
    """Ed25519 is deterministic: over seeded keys and messages of every length class the twin's
    public key and signature are the oracle's, byte for byte."""
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return
        for i in range(24):
            sk = hashlib.sha256(b"parity %d" % i).digest()
            msg = (hashlib.sha512(b"m %d" % i).digest() * 3)[: i * 9]
            out = subprocess.run(
                [exe, "sign", sk.hex(), _write(tmp, "m", msg)], capture_output=True, text=True
            ).stdout.split()
            assert out == [ed25519.public_key(sk).hex(), ed25519.sign(sk, msg).hex()], i


def test_both_rails_give_every_case_the_same_verdict():
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return
        checked = 0
        for name, statement, store, plan, pack, now, want in fx.cases():
            args = [
                exe,
                "verify",
                _write(tmp, "st", statement),
                _write(tmp, "store", store),
                _write(tmp, "plan", plan),
                _write(tmp, "pack", pack) if pack is not None else "-",
                str(now),
            ]
            done = subprocess.run(args, capture_output=True, text=True)
            assert done.stdout.strip() == f"verdict={want}", (name, done.stdout, done.stderr)
            assert (done.returncode == 0) == (want == "ok"), name
            checked += 1
        assert checked == len(fx.cases()) >= 30
