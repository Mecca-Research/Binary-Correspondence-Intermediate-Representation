#!/usr/bin/env python3
"""Bounded subprocess execution shared by the assurance rails.

One predicate for a recurring contract: the child runs in its own session,
both pipes drain under a per-stream byte budget, and a timeout or overflow
puts the whole process group down and comes back as a structured outcome —
never an unbounded ``communicate()``, never an escaping exception.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from typing import Any


TREE_KILL_TIMEOUT = 15  # the Windows tree terminator is itself bounded


def put_down_group(proc: Any) -> None:
    """Kill the child's whole process TREE — a descendant holding the pipes
    must not outlive the bound. POSIX kills the session; Windows has no
    session to kill, so the documented tree terminator stands in. The direct
    kill is the last resort on both, and this never raises: it runs inside
    timeout and overflow paths that must still return a structured verdict."""
    if os.name != "nt" and hasattr(os, "killpg"):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
    elif os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                check=False,
                timeout=TREE_KILL_TIMEOUT,
            )
        except Exception:
            # Deliberately total: this runs inside timeout and overflow
            # paths that MUST still return a structured verdict, so no
            # failure of the terminator — absent, wedged, or a wrapped
            # subprocess implementation raising something unforeseen — may
            # escape and turn a bounded FAIL into a traceback. The direct
            # kill below is the guaranteed floor.
            pass
    try:
        proc.kill()
    except Exception:
        pass


def _drain(stream: Any, chunks: list[bytes], cap: int, state: dict[str, Any]) -> None:
    """Read one pipe under the byte budget; on overflow the process group is
    put down and the pipe kept draining unretained so no writer can block."""
    received = 0
    while True:
        chunk = stream.read(65536)
        if not chunk:
            return
        if state["overflow"]:
            continue
        received += len(chunk)
        if received > cap:
            state["overflow"] = True
            put_down_group(state["proc"])
            continue
        chunks.append(chunk)


def _feed(proc: Any, data: bytes) -> None:
    try:
        proc.stdin.write(data)
        proc.stdin.close()
    except (OSError, ValueError):
        # The child exited (or was put down) before reading its stdin; the
        # outcome fields already tell that story.
        pass


def run_bounded(
    cmd: list[str],
    *,
    timeout: float,
    cap: int,
    stdin_data: bytes | None = None,
    cwd: Any = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run ``cmd`` bounded in wall time and retained output.

    Returns ``{"launched", "timed_out", "overflow", "pipes_held",
    "returncode", "stdout", "stderr", "error"}`` where stdout/stderr are the
    retained (cap-bounded) bytes. Every failure shape is a field, never an
    exception: the caller turns the outcome into its own structured verdict.
    """
    outcome: dict[str, Any] = {
        "launched": False,
        "timed_out": False,
        "overflow": False,
        "pipes_held": False,
        "returncode": None,
        "stdout": b"",
        "stderr": b"",
        "error": "",
    }
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
            start_new_session=os.name != "nt",
        )
    except OSError as exc:
        outcome["error"] = f"launch failed: {exc}"
        return outcome
    outcome["launched"] = True
    out_chunks: list[bytes] = []
    err_chunks: list[bytes] = []
    state: dict[str, Any] = {"overflow": False, "proc": proc}
    workers = [
        threading.Thread(
            target=_drain,
            args=(proc.stdout, out_chunks, cap, state),
            daemon=True,
        ),
        threading.Thread(
            target=_drain,
            args=(proc.stderr, err_chunks, cap, state),
            daemon=True,
        ),
    ]
    if stdin_data is not None:
        workers.append(threading.Thread(target=_feed, args=(proc, stdin_data), daemon=True))
    for worker in workers:
        worker.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        outcome["timed_out"] = True
        put_down_group(proc)
        proc.wait()
    for worker in workers:
        worker.join(5.0)
    outcome["overflow"] = state["overflow"]
    if any(worker.is_alive() for worker in workers):
        # A pipe still open after the group was reaped means an escaped
        # descendant holds it; the caller cannot vouch for output it never
        # saw the end of.
        outcome["pipes_held"] = True
        put_down_group(proc)
    outcome["returncode"] = proc.returncode
    outcome["stdout"] = b"".join(out_chunks)
    outcome["stderr"] = b"".join(err_chunks)
    return outcome
