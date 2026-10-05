"""The BCIR Make planner (BUILD-6): generation tags, staleness and the schedule, as a dry run.

Staleness is a generation tag, never a timestamp. A target's tag is the sha256 of a canonical text
naming everything it is made from:

    bcir-make generation 1
    target <name>
    tool <name> <identity>            every tool its commands run or it `uses`, by name
    run <argv...>                     its commands, in order
    read <path> file:<sha256>         a file of the tree, by its bytes
    read <path> tag:<sha256>          a file a target writes, by that target's tag
    tree <dir> <sha256>               a reads-tree, by the paths and bytes of every file under it
    write <path>                      what it claims to write

So a change to any input, command or tool changes the tag of the target and of every target that
reads what it writes, and no other change does (`after` orders, it does not feed). A target runs
when its recorded tag differs or an output is missing, and is reused otherwise.

The schedule is the IR's own canonical order over the lowered module (`topological_phase_ids`),
cut into waves of at most `workers` targets: a target goes in the first wave after every target it
depends on, and later when that wave is full. Two workers by default, the cap AGENTS.md sets.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ..model.graph import topological_phase_ids
from .grammar import BcirFile
from .laws import Lowered, file_digest, walk_tree

TAG_HEADER = "bcir-make generation 1"


def _tree_digest(root: Path, directory: str) -> str:
    h = hashlib.sha256()
    for rel in walk_tree(root, directory).files:
        h.update(f"{rel}\0{file_digest(root / rel)}\n".encode("ascii"))
    return h.hexdigest()


def tag_texts(bf: BcirFile, lowered: Lowered, root: Path) -> dict[str, str]:
    """Each target's canonical tag text, built in dependency order (a read another target writes is
    named by that target's tag)."""
    tools = {t.name: t for t in bf.tools}
    by_phase = {lowered.phase[t.name]: t for t in bf.targets}
    texts: dict[str, str] = {}
    tags: dict[str, str] = {}
    files: dict[str, str] = {}
    trees: dict[str, str] = {}  # one walk per tree per pass, however many targets read it

    def digest(path: str) -> str:
        if path not in files:
            files[path] = file_digest(root / path)
        return files[path]

    for pid in topological_phase_ids(lowered.module):
        t = by_phase[pid]
        lines = [TAG_HEADER, f"target {t.name}"]
        for name in sorted({argv[0] for argv in t.runs} | set(t.uses)):
            tool = tools.get(name)
            lines.append(f"tool {name} {tool.identity if tool else 'undeclared'}")
        lines += [f"run {' '.join(argv)}" for argv in t.runs]
        for p in sorted(t.reads):
            producer = lowered.writer.get(p)
            if producer is not None and producer != t.name and producer in tags:
                lines.append(f"read {p} tag:{tags[producer]}")
            else:
                lines.append(f"read {p} file:{digest(p)}")
        for d in sorted(t.reads_tree):
            if d not in trees:
                trees[d] = _tree_digest(root, d)
            lines.append(f"tree {d} {trees[d]}")
        lines += [f"write {p}" for p in sorted(t.writes)]
        text = "\n".join(lines) + "\n"
        texts[t.name] = text
        tags[t.name] = hashlib.sha256(text.encode("ascii")).hexdigest()
    return texts


def generation_tags(bf: BcirFile, lowered: Lowered, root: Path) -> dict[str, str]:
    """Each target's generation tag (the sha256 of its tag text)."""
    return {
        name: hashlib.sha256(text.encode("ascii")).hexdigest()
        for name, text in tag_texts(bf, lowered, root).items()
    }


def schedule(bf: BcirFile, lowered: Lowered, workers: int = 2) -> list[list[str]]:
    """Waves of at most ``workers`` targets in the IR's canonical order; a target follows every
    target it depends on."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    by_phase = {lowered.phase[t.name]: t.name for t in bf.targets}
    deps = {p.phase_id: p.deps for p in lowered.module.phases}
    wave_of: dict[int, int] = {}
    waves: list[list[str]] = []
    for pid in topological_phase_ids(lowered.module):
        w = max((wave_of[d] + 1 for d in deps[pid] if d in wave_of), default=0)
        while w < len(waves) and len(waves[w]) >= workers:
            w += 1
        while len(waves) <= w:
            waves.append([])
        waves[w].append(by_phase[pid])
        wave_of[pid] = w
    return waves


@dataclass(frozen=True)
class Decision:
    target: str
    tag: str
    action: str  # "run" | "reuse"
    why: str


def decide(
    bf: BcirFile, tags: dict[str, str], root: Path, state: dict[str, str] | None
) -> dict[str, Decision]:
    """Run or reuse, per target: reuse only when the recorded tag is the tag and every output is
    there; a stale producer makes its readers stale through their tags, not through this rule."""
    out: dict[str, Decision] = {}
    recorded = state or {}
    for t in bf.targets:
        tag = tags[t.name]
        missing = [p for p in t.writes if not (root / p).is_file()]
        if recorded.get(t.name) != tag:
            why = (
                "no recorded generation" if t.name not in recorded else "its generation tag changed"
            )
            out[t.name] = Decision(t.name, tag, "run", why)
        elif missing:
            out[t.name] = Decision(t.name, tag, "run", f"{missing[0]} is missing")
        else:
            out[t.name] = Decision(t.name, tag, "reuse", "same generation, outputs present")
    return out


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("a key is recorded twice")
    return dict(pairs)


def load_state(path: Path) -> dict[str, str]:
    """A recorded state: a JSON object of target name -> 64-hex tag, each name once, as the runner
    writes it. Anything else is refused -- a key recorded twice too, which would otherwise mean
    whichever came last (a second spelling of one state: docs/security/laws.md, class A)."""
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
    if not isinstance(data, dict) or not all(
        isinstance(k, str)
        and isinstance(v, str)
        and len(v) == 64
        and all(c in "0123456789abcdef" for c in v)
        for k, v in data.items()
    ):
        raise ValueError(f"{path} is not a JSON object of target -> sha256 tag")
    return data
