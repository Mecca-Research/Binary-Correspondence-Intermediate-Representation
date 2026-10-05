"""The BCIRfile grammar, version 1 (BUILD-6, docs/BCIR_BUILD_ROADMAP.md §8).

A BCIRfile declares tools by recorded identity and targets as claims over files. It is line-oriented
ASCII with one spelling for everything, so the oracle and its C twin (BUILD-8) read it alike:

    bcirfile 1
    # a comment line
    tool cc /usr/bin/gcc sha256:<64 lowercase hex>
    target bcir_runtime.o
      reads runtime/c/bcir_runtime.c runtime/c/bcir_runtime.h
      writes build/obj/bcir_runtime.o
      run cc -std=c2x -O2 -c runtime/c/bcir_runtime.c -o build/obj/bcir_runtime.o

- The first line that is neither blank nor a comment is ``bcirfile 1``.
- ``tool NAME PATH IDENTITY`` and ``target NAME`` start at column 0; a target's attributes follow it,
  indented by exactly two spaces: ``reads``, ``reads-tree``, ``writes``, ``after``, ``uses`` and
  ``run``, each with at least one token and each repeatable (tokens accumulate; ``run`` lines are
  commands, in order). ``uses`` names tools a command reaches without starting with them -- a
  wrapper's compiler -- which the runner hands it by path (``BCIR_TOOL_<NAME>``) and the tag counts.
- Tokens are separated by exactly one space; a token is printable ASCII without a space. No tabs, no
  carriage returns, no trailing space, no non-ASCII byte -- a second spelling of the same file is a
  second file to agree on, and there is none.
- A comment is a whole line whose first non-space character is ``#``. There are no inline comments.
- Names (tools, targets) match ``[A-Za-z0-9][A-Za-z0-9_.+-]*``. A file path is repo-relative: segments of
  ``[A-Za-z0-9_.+-]`` joined by ``/``, never ``.`` or ``..``, never empty, never absolute. A tool's PATH
  may be absolute (where it lives on the host); its IDENTITY is ``sha256:`` and 64 lowercase hex digits,
  the digest of the executable's bytes (R1, registry-first: a tool is named by what it is).

The parser enforces the grammar before it converts anything (docs/security/laws.md: never parse a
wire format with a host-language parser), bounds what it reads, and stops at the first error with its
line number: a malformed BCIRfile is a verdict, never a partial plan.

BCIR Make is a POSIX build, as the rails it builds are: an absolute tool PATH is a POSIX path (a
drive letter or a backslash has no spelling here) and the runner executes POSIX programs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_BYTES = 16 * 1024 * 1024
MAX_LINE = 65536
MAX_TARGETS = 100_000
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", re.ASCII)
SEGMENT = re.compile(r"[A-Za-z0-9_.+-]+", re.ASCII)
IDENTITY = re.compile(r"sha256:[0-9a-f]{64}", re.ASCII)
ATTRIBUTES = ("reads", "reads-tree", "writes", "after", "uses", "run")


class GrammarError(Exception):
    """The BCIRfile is not version-1 text; ``line`` is 1-based (0 for the file as a whole)."""

    def __init__(self, line: int, message: str) -> None:
        super().__init__(f"line {line}: {message}" if line else message)
        self.line = line
        self.message = message


@dataclass(frozen=True)
class Tool:
    name: str
    path: str
    identity: str
    line: int


@dataclass
class Target:
    name: str
    line: int
    reads: list[str] = field(default_factory=list)
    reads_tree: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)
    after: list[str] = field(default_factory=list)
    uses: list[str] = field(default_factory=list)
    runs: list[tuple[str, ...]] = field(default_factory=list)


@dataclass
class BcirFile:
    tools: list[Tool] = field(default_factory=list)
    targets: list[Target] = field(default_factory=list)


def is_repo_path(token: str) -> bool:
    """A repo-relative path: non-empty segments of the safe set, never `.` or `..`."""
    segments = token.split("/")
    return all(SEGMENT.fullmatch(s) and s not in (".", "..") for s in segments)


def _is_tool_path(token: str) -> bool:
    if token.startswith("/"):
        return len(token) > 1 and is_repo_path(token[1:])
    return is_repo_path(token)


def parse(data: bytes) -> BcirFile:
    """The BcirFile ``data`` spells, or GrammarError at the first departure from version 1."""
    if len(data) > MAX_BYTES:
        raise GrammarError(0, f"the file is {len(data)} bytes; the bound is {MAX_BYTES}")
    if any(b > 0x7E or (b < 0x20 and b != 0x0A) for b in data):
        bad = next(i for i, b in enumerate(data) if b > 0x7E or (b < 0x20 and b != 0x0A))
        line = data.count(b"\n", 0, bad) + 1
        raise GrammarError(line, f"byte 0x{data[bad]:02x} is not printable ASCII or a line feed")
    text = data.decode("ascii")
    if text and not text.endswith("\n"):
        raise GrammarError(text.count("\n") + 1, "the last line has no line feed")
    lines = text.split("\n")[:-1] if text else []
    result = BcirFile()
    header_seen = False
    current: Target | None = None
    tool_names: set[str] = set()
    target_names: set[str] = set()
    for number, line in enumerate(lines, 1):
        if len(line) > MAX_LINE:
            raise GrammarError(number, f"the line is {len(line)} bytes; the bound is {MAX_LINE}")
        if line == "":
            continue
        if line.endswith(" "):
            raise GrammarError(number, "trailing space")
        if line.lstrip(" ").startswith("#"):
            continue
        if not header_seen:
            if line != "bcirfile 1":
                raise GrammarError(number, "the first line is not `bcirfile 1`")
            header_seen = True
            continue
        indented = line.startswith("  ")
        body = line[2:] if indented else line
        if body.startswith(" "):
            raise GrammarError(number, "indentation is exactly two spaces, on a target's attribute")
        tokens = body.split(" ")
        if any(t == "" for t in tokens):
            raise GrammarError(number, "tokens are separated by exactly one space")
        key, args = tokens[0], tokens[1:]
        if not indented:
            if key == "tool":
                if len(args) != 3:
                    raise GrammarError(number, "`tool NAME PATH IDENTITY`")
                name, path, identity = args
                if not NAME.fullmatch(name):
                    raise GrammarError(
                        number, f"tool name {name!r} is not [A-Za-z0-9][A-Za-z0-9_.+-]*"
                    )
                if not _is_tool_path(path):
                    raise GrammarError(number, f"tool path {path!r} is not a path")
                if not IDENTITY.fullmatch(identity):
                    raise GrammarError(number, f"tool identity {identity!r} is not sha256:<64 hex>")
                if name in tool_names or name in target_names:
                    raise GrammarError(number, f"{name} is declared twice")
                tool_names.add(name)
                result.tools.append(Tool(name, path, identity, number))
                current = None
            elif key == "target":
                if len(args) != 1:
                    raise GrammarError(number, "`target NAME`")
                name = args[0]
                if not NAME.fullmatch(name):
                    raise GrammarError(
                        number, f"target name {name!r} is not [A-Za-z0-9][A-Za-z0-9_.+-]*"
                    )
                if name in target_names or name in tool_names:
                    raise GrammarError(number, f"{name} is declared twice")
                if len(result.targets) >= MAX_TARGETS:
                    raise GrammarError(number, f"more than {MAX_TARGETS} targets")
                target_names.add(name)
                current = Target(name, number)
                result.targets.append(current)
            else:
                raise GrammarError(number, f"{key!r} is not `tool` or `target`")
            continue
        if current is None:
            raise GrammarError(number, "an attribute outside a target")
        if key not in ATTRIBUTES:
            raise GrammarError(number, f"{key!r} is not one of {', '.join(ATTRIBUTES)}")
        if not args:
            raise GrammarError(number, f"`{key}` names nothing")
        if key == "run":
            if not NAME.fullmatch(args[0]):
                raise GrammarError(number, f"a command starts with a tool name, not {args[0]!r}")
            current.runs.append(tuple(args))
            continue
        bucket = {
            "reads": current.reads,
            "reads-tree": current.reads_tree,
            "writes": current.writes,
            "after": current.after,
            "uses": current.uses,
        }[key]
        for token in args:
            if key in ("after", "uses"):
                if not NAME.fullmatch(token):
                    what = "a target" if key == "after" else "a tool"
                    raise GrammarError(number, f"`{key}` names {what}, not {token!r}")
            elif not is_repo_path(token):
                raise GrammarError(number, f"{token!r} is not a repo-relative path")
            if token in bucket:
                raise GrammarError(number, f"{current.name} {key} {token} twice")
            bucket.append(token)
    if not header_seen:
        raise GrammarError(0, "the file has no `bcirfile 1` line")
    return result
