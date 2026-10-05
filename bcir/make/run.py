"""The BCIR Make runner (BUILD-7): execute a judged plan -- content-addressed reuse, the two-worker
scheduler, the observed footprint, and one telemetry record per task through the ring.

Per target, in the IR's canonical order, at most ``workers`` at once:

- **Up to date** when the recorded generation tag is its tag and every output is there: nothing runs.
- **From the cache** when the artifact cache holds its tag (``<cache>/<tag>/``): the outputs are
  restored, each checked against the digest recorded beside it, and nothing runs. The cache is
  content-addressed by the generation tag, so a hit is the same claim on the same inputs.
- **Run** otherwise: its outputs are removed, its commands run in order (argv, no shell, the tool
  resolved by its declared path, never by PATH), and then its *observed* footprint is held to its
  claims -- every declared write exists, and no file appeared or changed in the directories it
  writes that no target claims (MK3, observed). Two targets that write into one directory never run
  at once, so whatever appears there while a target runs is that target's own: a compiler's
  temporary beside a concurrent neighbour's object is not mistaken for this target's stray (found by
  the first BCIR Make build of the rails with Clang, which stages an object as `<name>.o.tmp`). The
  outputs then go to the cache and the tag to the recorded state, written after every target so an
  interrupted run keeps what it finished.

A failed target fails its readers without running them; nothing new starts after a failure unless
``keep_going``. Each task that ran, or was restored, becomes one TelemetryEnvelopeV0 ``datadna``
record bound to its generation (the claim id, the duration in ns, the bytes it wrote), published
through a live ring (bcir.gem.ring, the G15 SPSC ring) and drained into a log of decoded records.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

from ..abi.telemetry_envelope import TelemetryEnvelope, decode_envelope, encode_envelope
from ..gem.ring import RingConsumer, RingGeometry, RingProducer, format_ring
from .grammar import BcirFile, Target
from .laws import Lowered, file_digest
from .plan import generation_tags

TELEMETRY_SOURCE = 0x4D414B45  # "MAKE": the bcir-make producer
ENVELOPE_SLOT = 192  # a datadna envelope (124 bytes) and the slot header, in 64-byte units


@dataclass
class TaskResult:
    target: str
    tag: str
    outcome: str  # "up-to-date" | "cache" | "ran" | "failed" | "skipped"
    seconds: float = 0.0
    detail: str = ""
    bytes_written: int = 0


@dataclass
class RunReport:
    results: dict[str, TaskResult] = field(default_factory=dict)
    telemetry: list[dict] = field(default_factory=list)

    def count(self, outcome: str) -> int:
        return sum(1 for r in self.results.values() if r.outcome == outcome)

    @property
    def ok(self) -> bool:
        return not any(r.outcome in ("failed", "skipped") for r in self.results.values())

    def executed(self) -> list[str]:
        return [name for name, r in self.results.items() if r.outcome in ("ran", "failed")]


class Cache:
    """The artifact cache: ``<root>/<tag[:2]>/<tag>/files/<path>`` with ``index.json`` mapping each
    path to its sha256. Written into a scratch directory and renamed into place, so a reader never
    sees half an entry; read only after every file's digest matches its index (a cache is input)."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _entry(self, tag: str) -> Path:
        return self.root / tag[:2] / tag

    def restore(self, tag: str, writes: list[str], into: Path) -> bool:
        entry = self._entry(tag)
        try:
            index = json.loads((entry / "index.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not isinstance(index, dict) or sorted(index) != sorted(writes):
            return False
        for rel in writes:
            cached = entry / "files" / rel
            try:
                if file_digest(cached) != index[rel]:
                    return False
            except OSError:
                return False
        for rel in writes:
            dest = into / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".bcir-make-tmp")
            shutil.copy2(entry / "files" / rel, tmp)
            os.replace(tmp, dest)
        return True

    def store(self, tag: str, writes: list[str], root: Path) -> None:
        entry = self._entry(tag)
        if entry.exists():
            return
        entry.parent.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix=f".{tag[:8]}-", dir=entry.parent))
        try:
            index = {}
            for rel in writes:
                dest = scratch / "files" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / rel, dest)
                index[rel] = file_digest(dest)
            (scratch / "index.json").write_text(
                json.dumps(index, sort_keys=True), encoding="utf-8", newline="\n"
            )
            try:
                os.rename(scratch, entry)
            except OSError:
                pass  # another run stored the same generation first: the same bytes
        finally:
            shutil.rmtree(scratch, ignore_errors=True)


class TaskTelemetry:
    """One envelope per task, published through a live ring and drained as it goes."""

    def __init__(self, session: int, tasks: int) -> None:
        slots = 2
        while slots < max(2, min(tasks, 1 << 20)):
            slots <<= 1
        geometry = RingGeometry(
            policy="backpressure",
            payload="telemetry",
            slot_size=ENVELOPE_SLOT,
            slot_count=slots,
            ring_id=TELEMETRY_SOURCE,
        )
        self.region = bytearray(geometry.region_size)
        format_ring(self.region, geometry)
        self.producer = RingProducer(self.region)
        self.consumer = RingConsumer(self.region)
        self.producer.attach()
        self.consumer.attach()
        self.session = session
        self.seq = 0
        self.records: list[dict] = []
        self.lock = threading.Lock()

    def record(self, claim_id: int, tag: str, start_ns: int, seconds: float, written: int) -> None:
        with self.lock:
            self.seq += 1
            env = TelemetryEnvelope(
                kind="datadna",
                source=TELEMETRY_SOURCE,
                session=self.session,
                seq=self.seq % (1 << 32),
                generation=(int(tag[:8], 16) or 1),
                clock="monotonic",
                unit="ns",
                timestamp=start_ns,
                record=(claim_id, int(seconds * 1e9), written, 0, 0, 0, 0),
            )
            outcome = self.producer.publish(encode_envelope(env))
            if outcome.verdict != "ok":
                raise RuntimeError(f"the task telemetry ring refused a record ({outcome.status})")
            got = self.consumer.consume()
            if got.verdict != "delivered":
                raise RuntimeError(f"the task telemetry ring delivered nothing ({got.status})")
            back = decode_envelope(got.payload)
            self.records.append(
                {
                    "seq": back.seq,
                    "claim_id": back.record[0],
                    "generation": back.generation,
                    "duration_ns": back.record[1],
                    "bytes": back.record[2],
                }
            )


def _snapshot(dirs: set[Path]) -> dict[str, tuple[int, int]]:
    seen: dict[str, tuple[int, int]] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for entry in d.iterdir():
            if entry.is_file():
                st = entry.stat()
                seen[str(entry)] = (st.st_size, st.st_mtime_ns)
    return seen


def tool_env(name: str) -> str:
    """The variable a command finds a ``uses`` tool's path in: BCIR_TOOL_ and the name, upper-case,
    every character that is not a letter or digit an underscore."""
    return "BCIR_TOOL_" + "".join(c if c.isalnum() else "_" for c in name.upper())


def _run_target(
    t: Target, root: Path, tools: dict[str, str], claimed: set[str], log_dir: Path
) -> tuple[bool, str, int]:
    """Remove the outputs, run the commands, hold the observed footprint to the claims."""
    for rel in t.writes:
        out = root / rel
        if out.exists():
            out.unlink()
        out.parent.mkdir(parents=True, exist_ok=True)
    dirs = {(root / rel).parent for rel in t.writes}
    before = _snapshot(dirs)
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{t.name}.log"
    env = dict(os.environ, **{tool_env(name): tools[name] for name in t.uses})
    with open(log, "wb") as sink:
        for argv in t.runs:
            cmd = [tools[argv[0]], *argv[1:]]
            sink.write(("$ " + " ".join(argv) + "\n").encode("utf-8", "replace"))
            sink.flush()
            try:
                proc = subprocess.run(cmd, cwd=root, env=env, stdout=sink, stderr=subprocess.STDOUT)
            except OSError as exc:
                return False, f"{argv[0]} could not be run ({exc.strerror})", 0
            if proc.returncode != 0:
                return False, f"`{' '.join(argv[:3])} ...` exited {proc.returncode} (log {log})", 0
    missing = [rel for rel in t.writes if not (root / rel).is_file()]
    if missing:
        return False, f"MK3 observed: it did not write {missing[0]}", 0
    after = _snapshot(dirs)
    stray = sorted(
        p
        for p, sig in after.items()
        if before.get(p) != sig and Path(p).relative_to(root).as_posix() not in claimed
    )
    if stray:
        rel = Path(stray[0]).relative_to(root).as_posix()
        return False, f"MK3 observed: it wrote {rel}, which no target claims", 0
    return True, "", sum((root / rel).stat().st_size for rel in t.writes)


def execute(
    bf: BcirFile,
    lowered: Lowered,
    root: Path,
    *,
    state_path: Path,
    cache_dir: Path,
    log_dir: Path,
    workers: int = 2,
    keep_going: bool = False,
    session: int | None = None,
) -> RunReport:
    """Run the plan: every target up to date, restored from the cache, or run, readers after
    their producers, at most ``workers`` at once."""
    tags = generation_tags(bf, lowered, root)
    try:
        recorded = (
            json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
        )
    except (OSError, ValueError):
        recorded = {}
    if not isinstance(recorded, dict):
        recorded = {}
    state = {k: v for k, v in recorded.items() if isinstance(k, str) and isinstance(v, str)}
    tools = {tool.name: tool.path for tool in bf.tools}
    claimed = {p for t in bf.targets for p in t.writes}
    by_name = {t.name: t for t in bf.targets}
    deps = {
        t.name: [
            n
            for n, pid in lowered.phase.items()
            if pid in lowered.module.phase_map()[lowered.phase[t.name]].deps
        ]
        for t in bf.targets
    }
    cache = Cache(cache_dir)
    telemetry = TaskTelemetry(
        session or (os.getpid() << 16 | (time.monotonic_ns() & 0xFFFF)) or 1, len(bf.targets)
    )
    report = RunReport()
    order = [t.name for t in bf.targets]
    rank = {name: i for i, name in enumerate(order)}
    pending = set(order)
    running: dict[Future, str] = {}
    stopped = False
    lock = threading.Lock()

    def save_state() -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_path.with_name(state_path.name + ".tmp")
        tmp.write_text(json.dumps(state, sort_keys=True, indent=0), encoding="utf-8", newline="\n")
        os.replace(tmp, state_path)

    def work(name: str) -> TaskResult:
        t = by_name[name]
        tag = tags[name]
        if state.get(name) == tag and all((root / rel).is_file() for rel in t.writes):
            return TaskResult(name, tag, "up-to-date")
        start = time.monotonic_ns()
        t0 = time.monotonic()
        if cache.restore(tag, t.writes, root):
            seconds = time.monotonic() - t0
            written = sum((root / rel).stat().st_size for rel in t.writes)
            telemetry.record(lowered.phase[name], tag, start, seconds, written)
            return TaskResult(name, tag, "cache", seconds, "", written)
        ok, why, written = _run_target(t, root, tools, claimed, log_dir)
        seconds = time.monotonic() - t0
        if not ok:
            return TaskResult(name, tag, "failed", seconds, why)
        cache.store(tag, t.writes, root)
        telemetry.record(lowered.phase[name], tag, start, seconds, written)
        return TaskResult(name, tag, "ran", seconds, "", written)

    dirs = {name: {(root / rel).parent for rel in by_name[name].writes} for name in order}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending or running:
            ready = sorted(
                (n for n in pending if all(d in report.results for d in deps[n])), key=rank.get
            )
            busy = set().union(*(dirs[n] for n in running.values()))
            for name in ready:
                failed_dep = next(
                    (d for d in deps[name] if report.results[d].outcome in ("failed", "skipped")),
                    None,
                )
                if failed_dep is not None:
                    pending.discard(name)
                    report.results[name] = TaskResult(
                        name, tags[name], "skipped", 0.0, f"needs {failed_dep}"
                    )
                    continue
                if stopped or len(running) >= workers or dirs[name] & busy:
                    continue  # a target writing into a running one's directory waits for it
                pending.discard(name)
                running[pool.submit(work, name)] = name
                busy |= dirs[name]
            if not running:
                if pending and stopped:
                    for name in sorted(pending, key=rank.get):
                        report.results[name] = TaskResult(
                            name, tags[name], "skipped", 0.0, "the run stopped at a failure"
                        )
                    pending.clear()
                if not pending:
                    break
                continue
            done, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for future in done:
                name = running.pop(future)
                result = future.result()
                with lock:
                    report.results[name] = result
                    if result.outcome in ("ran", "cache", "up-to-date"):
                        state[name] = result.tag
                        save_state()
                    elif result.outcome == "failed" and not keep_going:
                        stopped = True
    report.results = {name: report.results[name] for name in order if name in report.results}
    report.telemetry = telemetry.records
    return report
