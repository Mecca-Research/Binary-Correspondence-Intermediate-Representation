#!/usr/bin/env python3
"""BCIR Make's twin parity gate (BUILD-8, docs/BCIR_BUILD_ROADMAP.md §9): the C twin's
`bcir-make --dry-run` (runtime/c/bcir_make.c) prints what the oracle's does (bcir/make/), byte for
byte, over a generated corpus of BCIRfiles.

    make_parity.py --twin build/cmake-gcc/runtime/c/bcir-make [--cases 400] [--seed 808]

Each case is a scratch tree -- tools by identity, sources, a `reads-tree` directory holding the
things a tree walk must skip (dot names, `__pycache__`, symlinks) -- and a BCIRfile over it,
generated as a lawful DAG and then, for most cases, broken one way: a grammar fault (each spelling
the grammar refuses), a law fault (each of MK1-MK5's findings), or a state the planner reads
(none, current, stale, partial) or must refuse (a key twice, a value that is no tag, data after the
object, bytes that are not UTF-8). Outputs are sometimes present, so a plan reuses as well as runs;
the workers and the tool check vary. The rails' own BCIRfile (tools/build/bcirfile.py, the sections
planned) is one more case. Both rails get the same argv; their exit statuses must agree, and so
must their stdout -- except on exit 2 (nothing could be judged), where only the status is the
verdict and the reason on stderr is each rail's own words.

A corpus that never made a law fire would prove nothing about it (docs/security/laws.md L2): the
gate requires every MK code among the verdicts, plans that run and plans that reuse, and refused
states, and fails as vacuous otherwise.

Exit 0 when every case agrees, 1 with a difference (the first few are printed with both outputs),
2 when the gate cannot run: every exit is a verdict (laws.md L1).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bcir.make import lower, parse  # noqa: E402
from bcir.make.__main__ import main as oracle_main  # noqa: E402
from bcir.make.plan import generation_tags  # noqa: E402

CODES = ("MK0", "MK1", "MK2", "MK3", "MK4", "MK5")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"bcir_build_{name}", ROOT / "tools" / "build" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# --- the corpus ------------------------------------------------------------------------------


class Case:
    def __init__(
        self, name: str, root: Path, text: bytes, argv_extra: list[str], state: bytes | None
    ):
        self.name = name
        self.root = root
        self.text = text
        self.argv_extra = argv_extra
        self.state = state


def _tree(rng: random.Random, root: Path) -> dict:
    """A scratch tree: tools, sources, headers, and a directory a tree claim walks."""
    (root / "bin").mkdir(parents=True)
    tools = {}
    for name in rng.sample(["cc", "ar", "py", "sh", "gen"], rng.randint(1, 3)):
        body = f"#!/bin/sh\n# {name} {rng.random()}\nexit 0\n".encode()
        (root / "bin" / name).write_bytes(body)
        tools[name] = "sha256:" + hashlib.sha256(body).hexdigest()
    sources = []
    for d in ("src", "inc", "src/sub"):
        (root / d).mkdir(parents=True, exist_ok=True)
        for i in range(rng.randint(1, 4)):
            rel = f"{d}/{'f' if d != 'inc' else 'h'}{i}{rng.choice(['.c', '.h', '.txt', '-x.c'])}"
            (root / rel).write_text(f"{rel} {rng.random()}\n", encoding="ascii", newline="\n")
            sources.append(rel)
    lib = root / "lib"
    (lib / "deep" / "er").mkdir(parents=True)
    for rel in ("a.txt", "deep/b.txt", "deep/er/c.dat", "deep/__pycache__x.txt"):
        (lib / rel).write_text(f"{rel}\n", encoding="ascii", newline="\n")
    (lib / ".hidden").write_text("skipped\n", encoding="ascii", newline="\n")
    (lib / ".dotdir").mkdir()
    (lib / ".dotdir" / "in.txt").write_text("skipped\n", encoding="ascii", newline="\n")
    (lib / "__pycache__").mkdir()
    (lib / "__pycache__" / "x y.pyc").write_bytes(b"\0skipped")
    (lib / "link.txt").symlink_to(lib / "a.txt")
    (lib / "linkdir").symlink_to(lib / "deep")
    (lib / "dangling").symlink_to(lib / "absent")
    return {"tools": tools, "sources": sources}


def _bcirfile(rng: random.Random, root: Path, made: dict) -> tuple[list[str], dict]:
    """A lawful BCIRfile's lines over the tree, and what it declared."""
    tools = made["tools"]
    names = sorted(tools)
    lines = ["bcirfile 1", f"# generated case over {len(made['sources'])} sources"]
    abs_tool = rng.random() < 0.7
    for name in names:
        path = f"{root}/bin/{name}" if abs_tool else f"bin/{name}"
        lines.append(f"tool {name} {path} {tools[name]}")
    targets = []
    outputs: list[str] = []
    used: set[str] = set()
    n = rng.randint(1, 18)
    for i in range(n):
        name = rng.choice(["t", "obj", "lib.a", "x+y", "T-", "z_"]) + str(i)
        writes = [f"out/{i}/{k}{rng.choice(['.o', '.a', ''])}" for k in range(rng.randint(1, 3))]
        reads = rng.sample(made["sources"], rng.randint(0, min(3, len(made["sources"]))))
        if outputs and rng.random() < 0.7:
            reads += rng.sample(outputs, rng.randint(1, min(3, len(outputs))))
        after = []
        if targets and rng.random() < 0.3:
            after = [rng.choice(targets)["name"]]
        trees = ["lib"] if rng.random() < 0.25 else []
        tool = rng.choice(names)
        uses = [u for u in names if u != tool and rng.random() < 0.3]
        used.add(tool)
        used.update(uses)
        argv = [tool]
        for r in reads:
            argv.append(
                rng.choice([r, f"-I{r}", f"--in={r}"]) if r.startswith(("src", "inc")) else r
            )
        if trees and rng.random() < 0.5:
            argv.append("-Llib/deep/b.txt")
        for w in writes:
            argv += rng.choice([["-o", w], [f"--out={w}"], [w]])
        runs = [argv]
        if rng.random() < 0.2:
            runs.append([tool, "--touch", writes[0]])
        lines.append(f"target {name}")
        if reads:
            for chunk in (reads[: len(reads) // 2], reads[len(reads) // 2 :]):
                if chunk:
                    lines.append("  reads " + " ".join(chunk))
        if trees:
            lines.append("  reads-tree " + " ".join(trees))
        lines.append("  writes " + " ".join(writes))
        if after:
            lines.append("  after " + " ".join(after))
        if uses:
            lines.append("  uses " + " ".join(uses))
        for r in runs:
            lines.append("  run " + " ".join(r))
        if rng.random() < 0.15:
            lines.append("  # a comment inside a target")
        if rng.random() < 0.1:
            lines.append("")
        targets.append({"name": name, "writes": writes})
        outputs += writes
    unused = [t for t in names if t not in used]
    if unused:  # keep the clean case lawful: every declared tool is used
        lines = [ln for ln in lines if not any(ln.startswith(f"tool {u} ") for u in unused)]
    return lines, {
        "targets": targets,
        "outputs": outputs,
        "tools": [t for t in names if t in used],
        "sources": made["sources"],
    }


GRAMMAR_FAULTS = (
    "tab",
    "trailing",
    "double",
    "nolf",
    "header",
    "noheader",
    "indent1",
    "indent3",
    "badname",
    "quotes",
    "dotdot",
    "slashes",
    "absread",
    "duptarget",
    "duptoken",
    "outside",
    "unknownattr",
    "unknownkey",
    "noargs",
    "toolargs",
    "badid",
    "badtoolpath",
    "longline",
    "nonascii",
    "runname",
    "afterbad",
    "usesbad",
    "cr",
)
LAW_FAULTS = (
    "nowrites",
    "noruns",
    "twowriters",
    "readswrite",
    "afterunknown",
    "afterself",
    "cycle",
    "unclaimed",
    "unnamed",
    "missingread",
    "missingtree",
    "unspellable",
    "undeclared",
    "usesundeclared",
    "unused",
    "identity",
)


def _grammar_fault(rng: random.Random, lines: list[str], kind: str) -> list[str]:
    body = [i for i, ln in enumerate(lines) if ln.startswith("  ") and not ln.startswith("  #")]
    tops = [i for i, ln in enumerate(lines) if ln.startswith("target ")]
    out = list(lines)
    i = rng.choice(body) if body else len(out) - 1
    if kind == "tab":
        out[i] = out[i].replace(" ", "\t", 1)
    elif kind == "trailing":
        out[i] += " "
    elif kind == "double":
        out[i] = out[i].replace(" ", "  ", 2) if out[i].count(" ") > 2 else out[i] + "  x"
    elif kind == "header":
        out[0] = rng.choice(["bcirfile 2", "bcirfile", "BCIRfile 1", " bcirfile 1"])
    elif kind == "noheader":
        out = [ln for ln in out if ln != "bcirfile 1"]
    elif kind == "indent1":
        out[i] = out[i][1:]
    elif kind == "indent3":
        out[i] = " " + out[i]
    elif kind == "badname":
        t = rng.choice(tops)
        out[t] = "target " + rng.choice(["-bad", ".x", "a/b", "a=b", "a,b", "_x"])
    elif kind == "quotes":
        t = rng.choice(tops)
        out[t] = "target " + rng.choice(["it's", 'say"hi"', "both'\"", "back\\slash"])
    elif kind == "dotdot":
        out[i] = "  reads ../etc/passwd"
    elif kind == "slashes":
        out[i] = "  writes out//x"
    elif kind == "absread":
        out[i] = "  reads /etc/hosts"
    elif kind == "duptarget":
        t = rng.choice(tops)
        out.insert(t, out[t])
    elif kind == "duptoken":
        claims = [k for k in body if not out[k].startswith("  run ")]
        if claims:
            k = rng.choice(claims)
            out[k] = out[k] + " " + out[k].split(" ")[-1]
        else:
            out.insert(tops[0] + 1, "  writes w/1 w/1")
    elif kind == "outside":
        out.insert(2, "  reads src/f0.c")
    elif kind == "unknownattr":
        out[i] = "  produces x/y"
    elif kind == "unknownkey":
        out.insert(len(out), rng.choice(["rule x", "Target y", "tools a b c"]))
    elif kind == "noargs":
        out[i] = "  " + rng.choice(["reads", "writes", "run", "after", "uses", "reads-tree"])
    elif kind == "toolargs":
        out.insert(1, "tool onlytwo /bin/x")
    elif kind == "badid":
        out.insert(1, "tool q /bin/q sha256:" + rng.choice(["ABC", "0" * 63, "g" * 64, "0" * 65]))
    elif kind == "badtoolpath":
        out.insert(
            1, "tool q " + rng.choice(["/", "a//b", "/x/../y", "C:x"]) + " sha256:" + "0" * 64
        )
    elif kind == "longline":
        out.insert(len(out), "# " + "x" * 65536)
    elif kind == "nonascii":
        out[i] = out[i] + " café"
    elif kind == "runname":
        out[i] = "  run ./tool x"
    elif kind == "afterbad":
        out[i] = "  after a/b"
    elif kind == "usesbad":
        out[i] = "  uses -x"
    elif kind == "cr":
        out[i] = out[i] + "\r"
    return out


def _law_fault(
    rng: random.Random, lines: list[str], decl: dict, kind: str
) -> tuple[list[str], list[str]]:
    """The lines with one law broken, and extra argv (the tool check)."""
    out = list(lines)
    tops = [i for i, ln in enumerate(out) if ln.startswith("target ")]
    t = rng.choice(tops)
    nxt = next((i for i in tops if i > t), len(out))
    name = out[t].split(" ", 1)[1]
    extra: list[str] = []
    if kind == "nowrites":
        out = [ln for k, ln in enumerate(out) if not (t < k < nxt and ln.startswith("  writes "))]
    elif kind == "noruns":
        out = [ln for k, ln in enumerate(out) if not (t < k < nxt and ln.startswith("  run "))]
    elif kind == "twowriters":
        mine = set(_writes(out, t, nxt))
        others = [o for o in decl["outputs"] if o not in mine]
        out.insert(
            t + 1, f"  writes {rng.choice(others)}" if others else "  writes out/never/named.o"
        )
    elif kind == "readswrite":
        w = _writes(out, t, nxt)
        if w:
            out.insert(t + 1, f"  reads {w[0]}")
    elif kind == "afterunknown":
        out.insert(t + 1, "  after nosuchtarget")
    elif kind == "afterself":
        out.insert(t + 1, f"  after {name}")
    elif kind == "cycle":
        first = out[tops[0]].split(" ", 1)[1]
        last = out[tops[-1]].split(" ", 1)[1]
        if first != last:  # the last after the first and the first after the last: a cycle
            out.insert(tops[-1] + 1, f"  after {first}")
        out.insert(tops[0] + 1, f"  after {last}")  # one target: after itself, a cycle of one
    elif kind == "unclaimed":
        runs = [k for k in range(t + 1, nxt) if out[k].startswith("  run ")]
        mine = {
            r for ln in out[t + 1 : nxt] if ln.startswith("  reads ") for r in ln.split(" ")[3:]
        }
        foreign = [s for s in decl["sources"] if s not in mine]
        if runs and foreign:
            other = rng.choice(foreign)
            out[runs[0]] += " " + rng.choice([other, f"-I{other}", f"--x={other}"])
    elif kind == "unnamed":
        out.insert(t + 1, "  writes out/never/named.o")
    elif kind == "missingread":
        out.insert(t + 1, "  reads src/nope.c")
    elif kind == "missingtree":
        out.insert(t + 1, "  reads-tree nodir")
    elif kind == "unspellable":
        if "  reads-tree lib" not in out[t + 1 : nxt]:
            out.insert(t + 1, "  reads-tree lib")
        extra.append("UNSPELLABLE")
    elif kind == "undeclared":
        runs = [k for k in range(t + 1, nxt) if out[k].startswith("  run ")]
        if runs:
            parts = out[runs[0]].split(" ")
            parts[3] = "ghost"
            out[runs[0]] = " ".join(parts)
    elif kind == "usesundeclared":
        out.insert(t + 1, "  uses phantom")
    elif kind == "unused":
        out.insert(1, f"tool spare /bin/sh sha256:{'0' * 64}")
    elif kind == "identity":
        tool_lines = [k for k, ln in enumerate(out) if ln.startswith("tool ")]
        k = rng.choice(tool_lines)
        parts = out[k].split(" ")
        parts[3] = "sha256:" + hashlib.sha256(parts[3].encode()).hexdigest()
        out[k] = " ".join(parts)
        extra.append("--check-tools")
    return out, extra


def _writes(lines: list[str], t: int, nxt: int) -> list[str]:
    return [w for ln in lines[t + 1 : nxt] if ln.startswith("  writes ") for w in ln.split(" ")[3:]]


STATE_FAULTS = (
    "dupkey",
    "short",
    "upper",
    "nested",
    "trailing",
    "bom",
    "notutf8",
    "array",
    "number",
)


def _state(rng: random.Random, root: Path, text: bytes, kind: str) -> bytes | None:
    try:
        bf = parse(text)
        tags = generation_tags(bf, lower(bf), root)
    except Exception:  # noqa: BLE001 -- a broken case: any state will do
        tags = {}
    names = sorted(tags)
    if kind == "current":
        state = dict(tags)
    elif kind == "stale":
        state = {
            n: hashlib.sha256(n.encode()).hexdigest() if rng.random() < 0.4 else tags[n]
            for n in names
        }
    elif kind == "partial":
        state = {n: tags[n] for n in names if rng.random() < 0.5}
        state["not-a-target"] = "0" * 64
    elif kind == "escaped":
        body = ",".join(
            f'"{n}": "'
            + "".join(f"\\u{ord(c):04x}" if rng.random() < 0.2 else c for c in tags[n])
            + '"'
            for n in names
        )
        return ("{" + body + "}").encode()
    elif kind == "dupkey":
        n = names[0] if names else "x"
        return f'{{"{n}": "{"0" * 64}", "{n}": "{"1" * 64}"}}'.encode()
    elif kind == "short":
        return b'{"a": "abc"}'
    elif kind == "upper":
        return ('{"a": "' + "A" * 64 + '"}').encode()
    elif kind == "nested":
        return b'{"a": {"b": "c"}}'
    elif kind == "trailing":
        return b'{"a": "' + b"0" * 64 + b'"} x'
    elif kind == "bom":
        return b"\xef\xbb\xbf{}"
    elif kind == "notutf8":
        return b'{"\xff": "' + b"0" * 64 + b'"}'
    elif kind == "array":
        return b"[]"
    elif kind == "number":
        return b'{"a": 1}'
    else:
        return None
    return json.dumps(state, sort_keys=True, indent=rng.choice([0, None])).encode()


FAULTS = (
    [("grammar", k) for k in GRAMMAR_FAULTS]
    + [("law", k) for k in LAW_FAULTS]
    + [("state", k) for k in STATE_FAULTS]
)
CLEAN_STATES = ("current", "none", "stale", "partial", "escaped")
# every fault once, each third case clean between them: this many cases cover the whole list
MIN_CASES = len(FAULTS) * 3 // 2 + 3


def build_corpus(rng: random.Random, base: Path, count: int) -> list[Case]:
    """``count`` cases: every third one a lawful BCIRfile, the others one fault each -- every fault
    of FAULTS once, in a seeded order, before any is repeated -- so a corpus of MIN_CASES or more
    makes every law, every grammar refusal and every state refusal fire (L2)."""
    cases: list[Case] = []
    plan = rng.sample(FAULTS, len(FAULTS))
    faulted = 0
    for i in range(count):
        root = base / f"c{i:04d}"
        made = _tree(rng, root)
        lines, decl = _bcirfile(rng, root, made)
        extra: list[str] = []
        if i % 3 == 0:
            family, kind = "clean", "clean"
            skind = CLEAN_STATES[(i // 3) % len(CLEAN_STATES)]
        else:
            family, kind = plan[faulted] if faulted < len(plan) else rng.choice(FAULTS)
            faulted += 1
            skind = kind if family == "state" else rng.choice(CLEAN_STATES)
        if family == "grammar":
            lines = _grammar_fault(rng, lines, kind)
        elif family == "law":
            lines, extra = _law_fault(rng, lines, decl, kind)
        if "UNSPELLABLE" in extra:
            extra.remove("UNSPELLABLE")
            (root / "lib" / rng.choice(["two words.txt", "caf\u00e9.txt", "x@y"])).write_text(
                "u\n", encoding="utf-8", newline="\n"
            )
        text = ("\n".join(lines) + "\n").encode("utf-8")
        if kind == "nolf":
            text = text.rstrip(b"\n")
        for out in decl["outputs"]:  # some outputs are there, so a plan can reuse
            if rng.random() < 0.6:
                (root / out).parent.mkdir(parents=True, exist_ok=True)
                (root / out).write_text("built\n", encoding="ascii", newline="\n")
        if rng.random() < 0.3:
            extra += ["--workers", rng.choice(["1", "3", "007", "+2", "100000000000000000000"])]
        if "--check-tools" not in extra and rng.random() < 0.3:
            extra.append("--check-tools")
        state = _state(rng, root, text, skind)
        cases.append(Case(f"c{i:04d}:{kind}:{skind}", root, text, extra, state))
    return cases


# --- running both rails -----------------------------------------------------------------------


def run_oracle(argv: list[str]) -> tuple[int, bytes]:
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = oracle_main(argv)
    except Exception as exc:  # noqa: BLE001 -- a traceback is a finding (L1), reported as a difference
        return -2, f"<the oracle raised {type(exc).__name__}: {exc}>".encode()
    return rc, out.getvalue().encode("utf-8")


def run_twin(twin: str, argv: list[str]) -> tuple[int, bytes]:
    try:
        done = subprocess.run([twin, *argv], capture_output=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return -1, f"<twin did not run: {exc}>".encode()
    return done.returncode, done.stdout


def compare(twin: str, case: Case, work: Path) -> tuple[bool, tuple, tuple, list[str]]:
    bcirfile = work / f"{case.name.split(':')[0]}.BCIRfile"
    bcirfile.write_bytes(case.text)
    argv = ["--dry-run", "-f", str(bcirfile), "--root", str(case.root), *case.argv_extra]
    if case.state is not None:
        state = work / f"{case.name.split(':')[0]}.state.json"
        state.write_bytes(case.state)
        argv += ["--state", str(state)]
    a, b = run_oracle(argv), run_twin(twin, argv)
    same = a[0] == b[0] and (a[0] == 2 or a[1] == b[1])
    return same, a, b, argv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--twin", required=True, help="the C twin (runtime/c/bcir_make.c built)")
    parser.add_argument("--cases", type=int, default=400)
    parser.add_argument("--seed", type=int, default=808)
    parser.add_argument("--no-rails", action="store_true", help="leave the rails' BCIRfile out")
    parser.add_argument("--keep", type=Path, help="keep the corpus here")
    args = parser.parse_args(argv)
    if os.name != "posix":
        print("make-parity: UNUSABLE: BCIR Make is POSIX's (the grammar spells POSIX paths)")
        return 2
    twin = shutil.which(args.twin) or args.twin
    if not (os.path.isfile(twin) and os.access(twin, os.X_OK)):
        print(f"make-parity: UNUSABLE: no twin at {args.twin}")
        return 2
    if args.cases < MIN_CASES:
        print(f"make-parity: UNUSABLE: --cases below {MIN_CASES} cannot cover every fault (L2)")
        return 2
    rng = random.Random(args.seed)
    with tempfile.TemporaryDirectory(prefix="bcir-make-parity-") as tmp:
        base = args.keep.resolve() if args.keep else Path(tmp)
        if args.keep:
            shutil.rmtree(base, ignore_errors=True)
        base.mkdir(parents=True, exist_ok=True)
        work = base / "work"
        work.mkdir()
        cases = build_corpus(rng, base, args.cases)
        if not args.no_rails:
            generator = _load("bcirfile")
            cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
            ar, bash = shutil.which("ar"), shutil.which("bash")
            if cc and ar and bash:
                text, _planned = generator.generate(
                    str(Path(cc).resolve()),
                    None,
                    str(Path(ar).resolve()),
                    str(Path(sys.executable).resolve()),
                    False,
                    out="build/bcir-make-parity",
                    bash=str(Path(bash).resolve()),
                    path_compilers=generator.path_compilers(),
                )
                cases.append(Case("rails", ROOT, text.encode("ascii"), ["--check-tools"], None))
        differ: list[tuple] = []
        seen_codes: set[str] = set()
        statuses = {0: 0, 1: 0, 2: 0}
        reused = ran = 0
        for case in cases:
            same, a, b, argv = compare(twin, case, work)
            statuses[a[0]] = statuses.get(a[0], 0) + 1
            text = a[1].decode("utf-8", "replace")
            for code in CODES:
                if f"  {code} " in text or f"  {code}: " in text:
                    seen_codes.add(code)
            reused += text.count("\nreuse ")
            ran += text.count("\nrun ")
            if not same:
                differ.append((case.name, argv, a, b))
        refused_states = sum(
            1 for c in cases if c.name.endswith(STATE_FAULTS) and c.state is not None
        )
        print(
            f"make-parity: {len(cases)} cases: {statuses.get(0, 0)} plans, {statuses.get(1, 0)} "
            f"law or grammar verdicts, {statuses.get(2, 0)} unusable; {ran} runs and {reused} "
            f"reuses planned; codes seen {', '.join(sorted(seen_codes))}"
        )
        vacuous = []
        if seen_codes != set(CODES):
            vacuous.append(f"no case made {', '.join(sorted(set(CODES) - seen_codes))} fire")
        if not ran or not reused:
            vacuous.append("no plan both ran and reused")
        if not refused_states or not statuses.get(2):
            vacuous.append("no state was refused")
        for name, argv, a, b in differ[:5]:
            print(f"  DIFFER {name}: oracle exit {a[0]}, twin exit {b[0]}")
            print(f"    argv: {' '.join(argv)}")
            for label, (rc, out) in (("oracle", a), ("twin", b)):
                print(f"    {label}:")
                for line in out.decode("utf-8", "replace").splitlines()[:12]:
                    print(f"      {line}")
        if differ:
            print(f"make-parity: FAIL ({len(differ)} of {len(cases)} cases differ)")
            return 1
        if vacuous:
            print(f"make-parity: FAIL (vacuous corpus: {'; '.join(vacuous)})")
            return 1
        print(f"make-parity: PASS ({len(cases)} cases, 0 differ)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
