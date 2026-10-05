"""The BCIR Make laws (BUILD-6): what a BCIRfile must satisfy before anything is planned or run.

A BCIRfile is lowered onto the IR the rest of BCIR is judged in: every file is a Resource, every
target a Phase holding one claim that reads and writes those resources, and every edge -- a target
reading what another writes, or an explicit ``after`` -- a phase dependency. The anti-cycle law is
then the verifier's own predicate over that module (``phase_graph_has_cycle``), and the plan's order
is its own canonical order (``topological_phase_ids``), applied unchanged (§8: "the verifier's
anti-cycle and ownership laws apply unchanged").

    MK0  grammar     the text is a version-1 BCIRfile (bcir.make.grammar)
    MK1  claims      every target writes at least one file and runs at least one command; a file has
                     one writer; a target never reads what it writes; every ``after`` names a target
    MK2  anti-cycle  the phase graph has no cycle (the verifier's predicate)
    MK3  footprint   every argument of a command that names a file of the tree, or a file a target
                     writes, is a declared read or write of its target, and every declared write is
                     named by one of its commands: the command's static footprint agrees with its
                     claims (BUILD-7's runner checks the observed footprint)
    MK4  tags        every read exists -- a file in the tree or one a target writes -- and every
                     ``reads-tree`` is a directory, so each target's generation tag is computable
                     from what it declares and nothing else
    MK5  tools       every command starts with a declared tool, every ``uses`` names one, every
                     declared tool is used, and, when the tools are checked, each one's bytes have
                     the identity it declares

Every law is checked over every target -- none stops at the first finding of another -- and the
findings come out in file order, so two runs over one BCIRfile report one text.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from ..model import Claim, Domain, Module, Opcode, Phase, Resource
from ..model.graph import phase_graph_has_cycle
from .grammar import BcirFile, is_repo_path


@dataclass(frozen=True)
class Finding:
    code: str
    target: str
    message: str

    def __str__(self) -> str:
        return (
            f"{self.code} {self.target}: {self.message}"
            if self.target
            else f"{self.code}: {self.message}"
        )


@dataclass(frozen=True)
class Lowered:
    """A BCIRfile on the IR: the module, each path's resource id and each target's phase id."""

    module: Module
    rid: dict[str, int]
    phase: dict[str, int]
    writer: dict[str, str]


def lower(bf: BcirFile) -> Lowered:
    """Files become resources (ids in sorted path order), targets phases (ids in file order),
    each with one dispatch claim over its reads and writes; a phase depends on the writers of
    what it reads and on what it is declared `after`."""
    paths = sorted({p for t in bf.targets for p in (*t.reads, *t.writes)})
    rid = {p: i + 1 for i, p in enumerate(paths)}
    phase = {t.name: i + 1 for i, t in enumerate(bf.targets)}
    writer: dict[str, str] = {}
    for t in bf.targets:
        for p in t.writes:
            writer.setdefault(p, t.name)
    resources = {rid[p]: Resource(rid=rid[p], name=p) for p in paths}
    phases: list[Phase] = []
    for i, t in enumerate(bf.targets):
        deps: list[int] = []
        for p in t.reads:
            producer = writer.get(p)
            if producer is not None and producer != t.name and phase[producer] not in deps:
                deps.append(phase[producer])
        for name in t.after:
            if name in phase and phase[name] not in deps:
                deps.append(phase[name])
        claim = Claim(
            id=i + 1,
            opcode=Opcode.GEM_DISPATCH,
            rd=tuple(rid[p] for p in t.reads),
            wr=tuple(rid[p] for p in t.writes),
            domain=Domain.RAM,
            op="make.run",
        )
        phases.append(Phase(phase_id=phase[t.name], deps=tuple(deps), claims=[claim]))
    module = Module(name="bcirfile", resources=resources, phases=phases)
    return Lowered(module, rid, phase, writer)


def file_digest(path: Path) -> str:
    """The sha256 of a file's bytes, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cycle_witness(bf: BcirFile, lowered: Lowered) -> list[str]:
    """One cycle's targets, in order, found by a deterministic depth-first walk."""
    names = {v: k for k, v in lowered.phase.items()}
    deps = {p.phase_id: p.deps for p in lowered.module.phases}
    color: dict[int, int] = {}
    for start in (lowered.phase[t.name] for t in bf.targets):
        if start in color:
            continue
        path: list[int] = []
        stack = [(start, iter(deps[start]))]
        color[start] = 1
        path.append(start)
        while stack:
            node, it = stack[-1]
            for dep in it:
                if color.get(dep) == 1:
                    return [names[n] for n in path[path.index(dep) :]]
                if dep not in color:
                    color[dep] = 1
                    path.append(dep)
                    stack.append((dep, iter(deps[dep])))
                    break
            else:
                stack.pop()
                path.pop()
                color[node] = 2
    return []


def check(bf: BcirFile, root: Path, *, check_tools: bool = False) -> list[Finding]:
    """Every MK1-MK5 finding over ``bf``, judged against the tree at ``root``."""
    findings: list[Finding] = []
    lowered = lower(bf)
    names = {t.name for t in bf.targets}
    tools = {t.name: t for t in bf.tools}
    writers: dict[str, list[str]] = {}
    for t in bf.targets:
        for p in t.writes:
            writers.setdefault(p, []).append(t.name)
    # MK1 claims
    for t in bf.targets:
        if not t.writes:
            findings.append(
                Finding("MK1", t.name, "writes nothing: a target is a claim on its outputs")
            )
        if not t.runs:
            findings.append(Finding("MK1", t.name, "runs no command"))
        for p in t.writes:
            if len(writers[p]) > 1 and writers[p][0] == t.name:
                findings.append(
                    Finding(
                        "MK1",
                        t.name,
                        f"{p} has {len(writers[p])} writers ({', '.join(writers[p])})",
                    )
                )
        for p in sorted(set(t.reads) & set(t.writes)):
            findings.append(Finding("MK1", t.name, f"reads {p}, which it writes"))
        for name in t.after:
            if name not in names:
                findings.append(Finding("MK1", t.name, f"after {name}, which is no target"))
            elif name == t.name:
                findings.append(Finding("MK1", t.name, "is after itself"))
    # MK2 anti-cycle, decided by the IR's own predicate
    if phase_graph_has_cycle(lowered.module):
        witness = _cycle_witness(bf, lowered)
        findings.append(
            Finding(
                "MK2",
                witness[0] if witness else "",
                f"the targets form a cycle: {' -> '.join(witness + witness[:1])}",
            )
        )
    # MK3 footprint (static)
    written = set(writers)
    for t in bf.targets:
        claimed = set(t.reads) | set(t.writes)
        named: set[str] = set()
        for argv in t.runs:
            for token in argv[1:]:
                for candidate in _paths_in(token):
                    if candidate in written or (root / candidate).is_file():
                        named.add(candidate)
                        if candidate not in claimed and not _under_tree(candidate, t.reads_tree):
                            findings.append(
                                Finding(
                                    "MK3",
                                    t.name,
                                    f"a command names {candidate}, which it neither reads nor writes",
                                )
                            )
        for p in t.writes:
            if p not in named:
                findings.append(
                    Finding("MK3", t.name, f"writes {p}, which none of its commands names")
                )
    # MK4 tags: every input is something
    for t in bf.targets:
        for p in t.reads:
            if p not in written and not (root / p).is_file():
                findings.append(
                    Finding("MK4", t.name, f"reads {p}: no file and no target's output")
                )
        for d in t.reads_tree:
            if not (root / d).is_dir():
                findings.append(Finding("MK4", t.name, f"reads-tree {d}: no such directory"))
    # MK5 tools
    used: set[str] = set()
    for t in bf.targets:
        for argv in t.runs:
            if argv[0] not in tools:
                findings.append(
                    Finding("MK5", t.name, f"runs {argv[0]}, which is no declared tool")
                )
            used.add(argv[0])
        for name in t.uses:
            if name not in tools:
                findings.append(Finding("MK5", t.name, f"uses {name}, which is no declared tool"))
            used.add(name)
    for tool in bf.tools:
        if tool.name not in used:
            findings.append(Finding("MK5", tool.name, "is declared and never run"))
        elif check_tools:
            path = Path(tool.path)
            try:
                digest = file_digest(path)
            except OSError as exc:
                findings.append(
                    Finding("MK5", tool.name, f"{tool.path} cannot be read ({exc.strerror})")
                )
                continue
            if f"sha256:{digest}" != tool.identity:
                findings.append(
                    Finding(
                        "MK5",
                        tool.name,
                        f"{tool.path} is sha256:{digest}, not its declared {tool.identity}",
                    )
                )
    return findings


def _paths_in(token: str) -> list[str]:
    """The repo paths an argument may name: the argument itself; a flag's attached value
    (`-Iruntime/c`, `-Lbuild/lib`); and what follows `=` in a flag (`--out=build/x`). A candidate
    counts only when it is a file of the tree or a target's output (``check`` decides that)."""
    out: list[str] = []
    if is_repo_path(token):
        out.append(token)
    if token.startswith("-"):
        if len(token) > 2 and token[1].isalpha() and is_repo_path(token[2:]):
            out.append(token[2:])
        if "=" in token:
            value = token.split("=", 1)[1]
            if is_repo_path(value):
                out.append(value)
    return out


def _under_tree(path: str, trees: list[str]) -> bool:
    return any(path == d or path.startswith(d.rstrip("/") + "/") for d in trees)


def walk_tree(root: Path, directory: str) -> list[str]:
    """The files under ``directory`` a ``reads-tree`` claims: every regular file, by repo path, in
    sorted order, skipping a path any of whose components starts with `.` or is `__pycache__`."""
    base = root / directory
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "__pycache__")
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            full = Path(dirpath) / name
            if full.is_file() and not full.is_symlink():
                out.append(full.relative_to(root).as_posix())
    return sorted(out)
