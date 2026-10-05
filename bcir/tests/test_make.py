"""BCIR Make, the oracle (BUILD-6): the BCIRfile grammar, the MK0-MK5 laws on the lowered IR, and
the dry-run planner's generation tags and waves.

Every law has negative fixtures that each must fire with that law's code, over a BCIRfile that is
otherwise clean (docs/security/laws.md L2, L11); the tags are asked the question that catches a
silent drop -- do two inputs that differ only in one thing differ in the tag? -- for every kind of
input, and for the things that must not matter.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from bcir.make import GrammarError, check, decide, generation_tags, lower, parse, schedule
from bcir.make.__main__ import dry_run, main
from bcir.make.laws import Finding
from bcir.model.graph import phase_graph_has_cycle

TOOL_BYTES = b"#!/bin/sh\nexit 0\n"
TOOL_ID = "sha256:" + hashlib.sha256(TOOL_BYTES).hexdigest()


def _posix() -> bool:
    """BCIR Make names a tool by its POSIX path -- the grammar has no other spelling of an absolute
    path -- and runs POSIX programs, so on another host (a Windows runner) there is nothing here to
    judge, and each test that builds a BCIRfile over a host path says so by returning."""
    return os.name == "posix"


def _tree(tmp: str, files: dict[str, str] | None = None) -> Path:
    root = Path(tmp)
    (root / "bin").mkdir(exist_ok=True)
    (root / "bin" / "cc").write_bytes(TOOL_BYTES)
    for rel, text in (files or {"src/a.c": "a\n", "src/b.c": "b\n", "src/h.h": "h\n"}).items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    return root


def _good(root: Path) -> str:
    return f"""bcirfile 1
# objects, then an archive over them
tool cc {root}/bin/cc {TOOL_ID}
target a.o
  reads src/a.c src/h.h
  writes out/a.o
  run cc -c src/a.c -o out/a.o
target b.o
  reads src/b.c src/h.h
  writes out/b.o
  run cc -c src/b.c -o out/b.o
target lib
  reads out/a.o out/b.o
  writes out/lib.a
  run cc -ar out/lib.a out/a.o out/b.o
"""


def _findings(text: str, root: Path, **kw) -> list[Finding]:
    return check(parse(text.encode("ascii")), root, **kw)


def test_a_clean_bcirfile_plans_in_the_irs_order():
    if not _posix():
        return
    with tempfile.TemporaryDirectory() as tmp:
        root = _tree(tmp)
        bf = parse(_good(root).encode())
        assert check(bf, root, check_tools=True) == []
        lowered = lower(bf)
        assert len(lowered.module.phases) == 3 and len(lowered.module.resources) == 6
        assert not phase_graph_has_cycle(lowered.module)
        assert schedule(bf, lowered, 2) == [["a.o", "b.o"], ["lib"]]
        assert schedule(bf, lowered, 1) == [["a.o"], ["b.o"], ["lib"]]
        rc, out = dry_run(_good(root).encode(), root, check_tools=True)
        assert rc == 0 and "laws: PASS (MK0-MK5)" in out and "plan: 3 to run" in out, out


def test_every_departure_from_the_grammar_is_mk0():
    """One spelling: each of these is refused, with its line, and the dry run reports MK0."""
    base = "bcirfile 1\ntool cc /bin/cc " + TOOL_ID + "\ntarget t\n  writes o\n  run cc o\n"
    parse(base.encode())  # the base itself is version-1 text
    faults = {
        "no header": base.replace("bcirfile 1\n", ""),
        "another version": base.replace("bcirfile 1", "bcirfile 2"),
        "a tab": base.replace("  writes o", "\twrites o"),
        "three-space indent": base.replace("  writes o", "   writes o"),
        "two spaces between tokens": base.replace("run cc o", "run cc  o"),
        "trailing space": base.replace("run cc o", "run cc o "),
        "a carriage return": base.replace("run cc o\n", "run cc o\r\n"),
        "non-ASCII": base.replace("writes o", "writes ö"),
        "no final line feed": base.rstrip("\n"),
        "a dotdot path": base.replace("writes o", "writes ../o"),
        "an absolute read": base.replace("writes o", "writes /o"),
        "an empty segment": base.replace("writes o", "writes a//o"),
        "an unknown attribute": base.replace("  writes o", "  makes o"),
        "an attribute before any target": "bcirfile 1\n  writes o\n",
        "a short identity": base.replace(TOOL_ID, TOOL_ID[:-1]),
        "an upper-case identity": base.replace(
            TOOL_ID, TOOL_ID.upper().replace("SHA256", "sha256")
        ),
        "a bad tool name": base.replace("tool cc", "tool -cc"),
        "a duplicate target": base + "target t\n  writes p\n  run cc p\n",
        "a target named as a tool": base + "target cc\n  writes p\n  run cc p\n",
        "a duplicate read": base.replace("  writes o\n", "  reads a a\n  writes o\n"),
        "an empty attribute": base.replace("  writes o", "  writes"),
        "an inline comment is a token, and `#` is no tool": base.replace("run cc o", "run # cc o"),
    }
    for what, text in faults.items():
        try:
            parse(text.encode("utf-8"))
        except GrammarError as exc:
            assert exc.line >= 0, what
        else:
            raise AssertionError(f"{what}: parsed")
        rc, out = dry_run(text.encode("utf-8"), Path("."))
        assert rc == 1 and "  MK0: " in out, (what, out)


def test_each_law_fires_on_its_own_violation():
    """Each fault breaks one law of a clean BCIRfile; the finding carries that law's code."""
    if not _posix():
        return
    with tempfile.TemporaryDirectory() as tmp:
        root = _tree(tmp)
        good = _good(root)
        assert _findings(good, root, check_tools=True) == []
        faults = {
            "MK1": [
                good.replace("  writes out/a.o\n", "").replace(" -o out/a.o", ""),  # writes nothing
                good.replace("  run cc -c src/b.c -o out/b.o\n", ""),  # runs nothing
                good.replace("writes out/b.o", "writes out/a.o"),  # two writers
                good.replace("reads src/a.c src/h.h", "reads src/a.c src/h.h out/a.o"),  # reads own
                good.replace("  writes out/lib.a\n", "  after nowhere\n  writes out/lib.a\n"),
                good.replace("  writes out/lib.a\n", "  after lib\n  writes out/lib.a\n"),
            ],
            "MK2": [
                good.replace("  reads src/a.c src/h.h\n", "  reads src/a.c src/h.h\n  after lib\n"),
            ],
            "MK3": [
                good.replace("reads out/a.o out/b.o", "reads out/a.o"),  # names out/b.o unread
                good.replace(
                    "-c src/a.c -o out/a.o", "-c src/a.c -o out/elsewhere.o"
                ),  # write unnamed
                good.replace(
                    "-c src/b.c -o", "-c src/b.c -Isrc/a.c -o"
                ),  # a flag's value is a path
            ],
            "MK4": [
                good.replace("reads src/a.c src/h.h", "reads src/a.c src/h.h src/gone.h"),
                good.replace("  writes out/a.o\n", "  reads-tree nodir\n  writes out/a.o\n"),
            ],
            "MK5": [
                good.replace("run cc -c src/a.c", "run gcc -c src/a.c"),  # undeclared tool
                good + "",  # placeholder replaced below
            ],
        }
        faults["MK5"][1] = good.replace(
            f"tool cc {root}/bin/cc {TOOL_ID}\n",
            f"tool cc {root}/bin/cc {TOOL_ID}\ntool unused {root}/bin/cc {TOOL_ID}\n",
        )
        for code, texts in faults.items():
            for n, text in enumerate(texts):
                assert text != good, (code, n, "the fault did not apply")
                findings = _findings(text, root)
                assert any(f.code == code for f in findings), (code, n, findings)
        # a tool whose bytes are not its identity, judged only when tools are checked
        (root / "bin" / "cc").write_bytes(TOOL_BYTES + b"# changed\n")
        assert _findings(good, root) == []
        changed = _findings(good, root, check_tools=True)
        assert [f.code for f in changed] == ["MK5"] and "not its declared" in changed[0].message


def test_a_cycle_is_named_by_its_targets():
    if not _posix():
        return
    with tempfile.TemporaryDirectory() as tmp:
        root = _tree(tmp)
        text = _good(root).replace(
            "  reads src/a.c src/h.h\n", "  reads src/a.c src/h.h\n  after lib\n"
        )
        findings = [f for f in _findings(text, root) if f.code == "MK2"]
        assert len(findings) == 1 and "a.o -> lib -> a.o" in findings[0].message, findings


def test_the_tag_covers_each_input_and_nothing_else():
    """Two trees that differ in one thing differ in the tags of exactly the targets that read it,
    and of their readers; a file nobody reads, and an `after` edge, change no tag."""
    if not _posix():
        return
    with tempfile.TemporaryDirectory() as tmp:
        root = _tree(
            tmp, {"src/a.c": "a\n", "src/b.c": "b\n", "src/h.h": "h\n", "src/other.c": "x\n"}
        )
        good = _good(root)

        def tags(text: str) -> dict[str, str]:
            bf = parse(text.encode())
            return generation_tags(bf, lower(bf), root)

        base = tags(good)
        assert len(set(base.values())) == 3
        (root / "src/a.c").write_text("a changed\n")
        after_a = tags(good)
        assert after_a["a.o"] != base["a.o"] and after_a["lib"] != base["lib"]
        assert after_a["b.o"] == base["b.o"]
        (root / "src/a.c").write_text("a\n")
        assert tags(good) == base
        (root / "src/h.h").write_text("h changed\n")
        after_h = tags(good)
        assert all(after_h[t] != base[t] for t in base), "a shared header reaches every reader"
        (root / "src/h.h").write_text("h\n")
        (root / "src/other.c").write_text("x changed\n")
        assert tags(good) == base, "a file nobody reads changed a tag"
        changed_cmd = tags(good.replace("-c src/b.c -o out/b.o", "-O3 -c src/b.c -o out/b.o"))
        assert changed_cmd["b.o"] != base["b.o"] and changed_cmd["a.o"] == base["a.o"]
        other_tool = "sha256:" + "0" * 64
        assert tags(good.replace(TOOL_ID, other_tool)) != base, "the tool's identity is no input"
        ordered = tags(good.replace("  writes out/lib.a\n", "  after a.o\n  writes out/lib.a\n"))
        assert ordered == base, "an `after` edge fed the tag"


def test_reuse_needs_the_same_generation_and_the_outputs():
    if not _posix():
        return
    with tempfile.TemporaryDirectory() as tmp:
        root = _tree(tmp)
        bf = parse(_good(root).encode())
        tags = generation_tags(bf, lower(bf), root)
        assert all(d.action == "run" for d in decide(bf, tags, root, None).values())
        for rel in ("out/a.o", "out/b.o", "out/lib.a"):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(b"built\n")
        same = decide(bf, tags, root, dict(tags))
        assert all(d.action == "reuse" for d in same.values()), same
        (root / "out/b.o").unlink()
        gone = decide(bf, tags, root, dict(tags))
        assert gone["b.o"].action == "run" and gone["a.o"].action == "reuse"
        stale = decide(bf, tags, root, {**tags, "lib": "0" * 64})
        assert stale["lib"].action == "run" and "changed" in stale["lib"].why
        state = Path(tmp) / "state.json"
        state.write_text(json.dumps(tags), encoding="utf-8")
        (root / "out/b.o").write_bytes(b"built\n")
        bcirfile = Path(tmp) / "BCIRfile"
        bcirfile.write_text(_good(root), encoding="ascii")
        rc = main(["--dry-run", "-f", str(bcirfile), "--root", str(root), "--state", str(state)])
        assert rc == 0


def test_the_cli_says_unusable_rather_than_guessing():
    if not _posix():
        return
    with tempfile.TemporaryDirectory() as tmp:
        root = _tree(tmp)
        bcirfile = Path(tmp) / "BCIRfile"
        bcirfile.write_text(_good(root), encoding="ascii")
        bad_state = Path(tmp) / "bad.json"
        bad_state.write_text('{"a.o": "short"}', encoding="utf-8")
        assert main(["--dry-run", "-f", str(Path(tmp) / "absent"), "--root", str(root)]) == 2
        assert (
            main(["--dry-run", "-f", str(bcirfile), "--root", str(root), "--state", str(bad_state)])
            == 2
        )
        assert main(["--dry-run", "-f", str(bcirfile), "--root", str(root), "--workers", "0"]) == 2
        assert main(["--dry-run", "-f", str(bcirfile), "--root", str(Path(tmp) / "nodir")]) == 2
        assert main(["--dry-run", "-f", str(bcirfile), "--root", str(root)]) == 0
