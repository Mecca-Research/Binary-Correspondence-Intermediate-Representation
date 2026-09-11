"""A generated artifact has the same bytes on every host, newlines included.

Python's text mode translates ``"\\n"`` to the platform's line ending on write. So
``open(path, "w", encoding="utf-8")`` and ``Path.write_text(text, encoding="utf-8")``
with no ``newline=`` argument produce LF on Linux and CRLF on Windows, from identical
source and identical logic.

This is not hypothetical here. ``training/tools/build_chunks.py`` wrote the retrieval
corpus that way, so on Windows every chunk file was CRLF. The contents were logically
the same; everything derived from their *bytes* was not. The corpus fingerprint moved,
every per-file ``sha256`` the catalog manifest records moved, every part's content
digest moved, and ``locator.bin`` moved -- its byte offsets shift with the length of
every line before them. Two hosts disagreed about the content address of an identical
corpus, so a generation published on one could never verify on the other. Nothing saw
it, because nothing had ever compared a corpus built on one host against the same
corpus built on another; it surfaced only when a fingerprint was pinned into the
repository and the Windows host-portability job read it back.

Six generators write *tracked* files the same way, which is the same defect with a
quieter consequence -- a working tree that differs from the index on one host only:

    bcir/abi/q8_tables.py            -> runtime/c/bcir_q8_tables.h
    bcir/kbcir/differential.py       -> three MLIR corpora under mlir/test/passes/
    bcir/verify/structural_corpus.py -> mlir/test/passes/structural_corpus.mlir
    .claude/.../build_digest.py      -> .claude/context/BCIR_DIGEST.md

Their drift gates all compare in *text* mode, where universal newlines translate CRLF
back to ``"\\n"`` on read -- so the gates stay green while the file on disk differs
(`docs/security/laws.md` L11: a witness must hit the law it exists to test).

**The repository already declares this rule** for everything it tracks:
``.gitattributes`` opens with ``* text=auto eol=lf``. What it cannot reach is a file a
tool writes at run time, under ``build/`` or anywhere else. This test extends the same
declaration to the tools.

**Declared scope.** A static read for two shapes: ``Path.write_text`` and ``open``
whose mode is a *literal* containing ``w`` or ``a`` without ``b``. A write assembled
at run time, routed through a helper this cannot see, or opened with a computed mode
is out of scope and belongs to a linter rather than to a BCIR rail. Inside the scope
this is exact; outside it, it does not grow -- the answer to the next soundness
question is to point here, not to add an interpreter.

**And the rule is total inside that scope: there is no allowlist.** Pinning the line
ending costs nothing on a throwaway log and is never wrong -- no tree here generates a
``.bat``, ``.cmd`` or ``.ps1``, the only artifacts that would want CRLF. An
exemption list of "the sites we judged safe" would be a second thing to maintain and
would be wrong the first time somebody digested one of them (L15). What *is* excluded
is structural and decidable: a test or gate module, whose fixtures are written and
read in one process on one host and never compared as bytes against another's.
"""

import ast
import pathlib
import subprocess

_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Every tree that carries tools which generate files. ``.claude`` is included because
#: it holds tracked developer scripts, one of which regenerates a tracked digest --
#: a policy that skipped it would audit less than it claims.
_TREES = ("bcir", "tools", "training", ".claude")

#: Directory names that mean the same thing at any depth.
_SKIP_PARTS = {".git", "__pycache__", "node_modules"}

#: Generated trees, matched as a path PREFIX rather than as a name anywhere in the
#: path, so a checkout that itself sits under a directory called "build" is not
#: silently skipped entire.
_SKIP_ROOTS = (("build",), ("bcir", "dataset"), ("training", "llvm", "dataset"))

#: Floors. A matcher that silently stops matching reports a clean tree, which is the
#: failure mode this whole file exists to prevent (L2).
_MIN_MODULES = 200
_MIN_PINNED = 40
_MIN_TRACKED_TEXT = 1500


def _is_gate(rel: str) -> bool:
    """Whether a module is a test or a gate rather than a producer of artifacts.

    Structural and decidable, not a list of names: a gate builds its inputs and reads
    them back in the same process on the same host, so a translated newline cannot
    make two hosts disagree about anything. Both spellings of the prefix appear in
    this tree (`verify_database.py`, `verify-langref-delta.py`).
    """
    name = rel.rsplit("/", 1)[-1]
    return "/tests/" in f"/{rel}" or name.startswith(("test_", "verify_", "verify-"))


def _skipped(parts: tuple) -> bool:
    if any(part in _SKIP_PARTS for part in parts):
        return True
    return any(parts[: len(prefix)] == prefix for prefix in _SKIP_ROOTS)


def _tracked() -> set | None:
    """Every path git tracks, or None outside a checkout.

    The reconciliation below needs this: a tracked module the walk never yielded is
    absent from the worktree, and reporting clean around it would claim more than was
    inspected (`docs/security/laws.md` L15).
    """
    if not (_ROOT / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(_ROOT), "ls-files", "-z"], capture_output=True, check=False
    )
    if result.returncode != 0:
        return None
    return {item.decode("utf-8", "surrogateescape") for item in result.stdout.split(b"\0") if item}


def _text_writes(source: str) -> tuple[list[int], list[int]]:
    """(unpinned, pinned) line numbers of the text writes in one parsed module."""
    unpinned: list[int] = []
    pinned: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", getattr(node.func, "id", ""))
        if name == "write_text":
            matched = True
        elif name == "open":
            mode = None
            # Path.open takes the mode first; the builtin takes it second.
            index = 0 if getattr(node.func, "attr", "") == "open" else 1
            if len(node.args) > index and isinstance(node.args[index], ast.Constant):
                mode = node.args[index].value
            for keyword in node.keywords:
                if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                    mode = keyword.value.value
            matched = isinstance(mode, str) and "b" not in mode and bool({"w", "a"} & set(mode))
        else:
            matched = False
        if not matched:
            continue
        if "newline" in {keyword.arg for keyword in node.keywords}:
            pinned.append(node.lineno)
        else:
            unpinned.append(node.lineno)
    return unpinned, pinned


def _modules() -> tuple[list[tuple[str, pathlib.Path]], list[str]]:
    """Artifact-producing modules to read, and tracked ones the walk never yielded."""
    tracked = _tracked()
    found: list[tuple[str, pathlib.Path]] = []
    seen: set[str] = set()
    for tree in _TREES:
        base = _ROOT / tree
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            # Suffix folded: rglob("*.py") is literal on Linux, so a tracked
            # generator named `emit.PY` would dodge the audit there and be read on
            # the case-folding hosts -- one condition, two answers (L12).
            if path.suffix.lower() != ".py":
                continue
            parts = path.relative_to(_ROOT).parts
            if _skipped(parts):
                continue
            rel = "/".join(parts)
            if _is_gate(rel):
                continue
            if tracked is not None and rel not in tracked:
                continue
            seen.add(rel)
            found.append((rel, path))
    missing = []
    if tracked is not None:
        for rel in sorted(tracked):
            if not rel.lower().endswith(".py") or rel in seen:
                continue
            parts = tuple(rel.split("/"))
            if parts[0] not in _TREES or _skipped(parts) or _is_gate(rel):
                continue
            missing.append(rel)
    return found, missing


def test_every_generator_pins_its_line_ending():
    """No artifact-producing module lets the host choose how a line ends."""
    modules, missing = _modules()
    assert not missing, (
        "tracked modules the walk never yielded, so the audit covers less than it "
        f"claims: {missing[:8]}"
    )
    findings = []
    pinned_total = 0
    for rel, path in modules:
        unpinned, pinned = _text_writes(path.read_text(encoding="utf-8"))
        pinned_total += len(pinned)
        findings += [f"{rel}:{line}" for line in unpinned]
    assert not findings, (
        "these writes let the host choose the line ending, so the file they produce "
        'is not the same bytes everywhere -- pass newline="\\n": ' + ", ".join(findings)
    )
    # Anti-vacuity, both halves. A walk that yields nothing, or a matcher that stops
    # matching, reports a clean tree either way.
    assert len(modules) >= _MIN_MODULES, (
        f"only {len(modules)} artifact-producing module(s) were read, below the "
        f"{_MIN_MODULES} this tree holds; the walk is not covering what it claims"
    )
    assert pinned_total >= _MIN_PINNED, (
        f"the matcher recognised only {pinned_total} pinned write(s), below the "
        f"{_MIN_PINNED} this tree holds; it has stopped matching the shape it audits"
    )


def test_the_matcher_sees_both_spellings_and_neither_false_positive():
    """The predicate itself, against writes that must and must not be flagged.

    The check above can only fail when the matcher works. This is the witness for the
    matcher, so a silent regression in it shows up as a failure here rather than as a
    clean run over an unaudited tree (L11).
    """
    flagged, accepted = _text_writes(
        "\n".join(
            (
                'open(p, "w")',  # builtin, mode positional
                'open(p, mode="w", encoding="utf-8")',  # builtin, mode by keyword
                'p.open("w", encoding="utf-8")',  # Path.open, mode first
                "p.write_text(t)",  # Path.write_text
                'open(p, "a")',  # append translates too
                'open(p, "wt")',  # explicit text mode
            )
        )
    )
    assert len(flagged) == 6 and not accepted, f"missed a write shape: {flagged}"

    flagged, accepted = _text_writes(
        "\n".join(
            (
                'open(p, "w", newline="\\n")',
                'p.write_text(t, encoding="utf-8", newline="\\n")',
                'p.open("w", newline="")',
            )
        )
    )
    assert not flagged and len(accepted) == 3, f"a pinned write was flagged: {flagged}"

    flagged, accepted = _text_writes(
        "\n".join(
            (
                'open(p, "wb")',  # bytes: text mode never touches these
                'open(p, "rb")',
                'open(p, "r", encoding="utf-8")',  # reading is not writing
                "p.read_text()",
                'p.write_bytes(b"x")',
            )
        )
    )
    assert not flagged and not accepted, f"a non-text-write was matched: {flagged + accepted}"


def test_no_tracked_text_file_carries_a_carriage_return():
    """The artifacts themselves, on disk: `.gitattributes` cashed rather than declared.

    The rule above is about source. This is about what that source produced, and it is
    the half a tidy source sweep cannot satisfy on its own.

    Derived, not curated. Six tracked files in this tree are written by a generator --
    `runtime/c/bcir_q8_tables.h`, three MLIR corpora from `bcir/kbcir/differential.py`,
    `mlir/test/passes/structural_corpus.mlir`, and `.claude/context/BCIR_DIGEST.md` --
    and listing those six here would be a list to maintain and to get wrong the next
    time somebody adds a seventh (`docs/security/laws.md` L15). Asking the whole index
    instead costs one walk and cannot go stale: `.gitattributes` opens with
    `* text=auto eol=lf`, so this is that declaration, checked rather than trusted.

    Binary files are excluded by git's own heuristic -- a NUL byte in the first 8 KiB --
    rather than by extension, because `text=auto` uses the same rule and a check that
    disagreed with git about what is text would be a second opinion nobody asked for.
    """
    tracked = _tracked()
    if tracked is None:
        # Outside a checkout (an installed wheel, a fixture tree) there is no index to
        # ask. That is an absence of the subject, not a passing result.
        raise AssertionError("not a git checkout: there is no tracked tree to inspect")
    text_files = 0
    carriage = []
    for rel in sorted(tracked):
        path = _ROOT / rel
        try:
            raw = path.read_bytes()
        except OSError:
            # A tracked path absent from the worktree (sparse checkout, unstaged
            # deletion) is reported, never skipped into a clean verdict.
            carriage.append(f"{rel} (unreadable)")
            continue
        if b"\0" in raw[:8000]:
            continue
        text_files += 1
        returns = raw.count(b"\r")
        if returns:
            carriage.append(f"{rel} ({returns} CR)")
    assert not carriage, (
        "a tracked text file carries carriage returns, so it was written on a host "
        f"that chose them: {carriage[:8]}"
    )
    assert text_files >= _MIN_TRACKED_TEXT, (
        f"only {text_files} tracked text file(s) were inspected, below the "
        f"{_MIN_TRACKED_TEXT} this repository holds; the walk is not covering it"
    )
