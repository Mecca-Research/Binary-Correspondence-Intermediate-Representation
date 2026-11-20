"""PORT-4: the event transducer in C (`runtime/c/bcir_etl.c`, behind `bcir-etl`) against the oracle (`bcir/etl`).

N4 of the native port audit: driver-packet ingestion had no executable transducer or grammar-to-claims parser in C.
`bcir_etl.c` ports the three oracle modules, freestanding -- the finite-state transducer (`etl.fsm`), the event
descriptors (`etl.events`) and the mini-ROP grammar (`etl.parse`) -- and `bcir-etl` spells each verdict in the
oracle's words. `check_tool` holds the two rails to one text over hand-written programs and machines, generated ones
and their mutations: every claim, resource, run, capture and trace step, and every refusal with the offset or element
it names. `runtime/c/test_etl.c` holds the C library's own contracts (short buffers, NULLs) and `fuzz_etl.c` its
invariants under libFuzzer; the gate's `etl` section runs both harnesses and this comparison.
"""

import os
import random
import subprocess
import tempfile

from bcir.etl import Event, EventKind, EventStream, State, Transducer, Transition, parse_rop
from bcir.etl.parse import LexError, ParseError
from bcir.toolchain import host_c_compiler

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_C = os.path.join(_ROOT, "runtime", "c")
_CC = host_c_compiler()  # None in the quick tier, which hides the host's compilers


def expected_rop(text: str) -> str:
    """The oracle's verdict on `text`, as `bcir-etl rop` spells it."""
    try:
        module = parse_rop(text)
    except LexError as e:
        message = str(e)
        at = int(message.rsplit(" ", 1)[1])
        if ord(text[at]) > 0x7F:  # the tool spells no Unicode repr: it names the offset
            return f"ERR unexpected non-ASCII character at offset {at}\n"
        return f"ERR {message}\n"
    except ParseError as e:
        return f"ERR {e}\n"
    lines = [
        f"resource {r.rid} {r.name}" for r in sorted(module.resources.values(), key=lambda r: r.rid)
    ]
    for c in module.phases[0].claims:
        lines.append(
            f"claim {c.id} {c.op} rd={','.join(map(str, c.rd))} wr={','.join(map(str, c.wr))} count={c.count} "
            f"lane={c.lane.name} stride={c.stride_class.name}"
        )
    return "".join(line + "\n" for line in lines)


def _word(s: str) -> str:
    return s if s else '""'


def fsm_file(states, transitions, start, runs) -> str:
    """A machine and its runs in `bcir-etl fsm`'s text: `states` (name, accepting, error), `transitions` (src, dst,
    on, action-or-None)."""
    out = []
    for name, acc, err in states:
        out.append(
            " ".join(
                ["state", _word(name)] + (["accepting"] if acc else []) + (["error"] if err else [])
            )
        )
    for src, dst, on, action in transitions:
        out.append(
            " ".join(
                ["tx", _word(src), _word(dst), _word(on)]
                + ([_word(action)] if action is not None else [])
            )
        )
    out.append(f"start {_word(start)}")
    out += [" ".join(["run"] + [_word(s) for s in run]) for run in runs]
    return "\n".join(out) + "\n"


def expected_fsm(states, transitions, start, runs) -> str:
    """The oracle's verdict, as `bcir-etl fsm` spells it: the machine refused, or each run."""
    try:
        ss = [State(name, accepting=acc, error=err) for name, acc, err in states]
        ts = [Transition(src, dst, on=on, action=action) for src, dst, on, action in transitions]
        machine = Transducer(ss, ts, start=start)
    except ValueError as e:
        return f"ERR {e}\n"
    out = []
    for run in runs:
        r = machine.run(run)
        captures = ",".join(f"{a}:{s}" for a, s in r.captures)
        trace = ",".join(f"{a}/{s}/{b}" for a, s, b in r.trace)
        out.append(
            f"run accepted={int(r.accepted)} final={r.final_state} captures={captures} trace={trace}\n"
        )
    return "".join(out)


def expected_event(line: list) -> str:
    """The oracle's verdict on one descriptor line, as `bcir-etl event` spells it."""
    try:
        if line[0] == "stream":
            EventStream(
                line[1], kind=line[2], encoding=line[3], element_bits=line[4], max_window=line[5]
            )
        elif line[0] == "kind":
            EventKind(line[1])
        else:
            Event(line[1], span=(line[2], line[3]))
    except ValueError as e:
        return f"ERR {e}\n"
    return "ok\n"


# Hand-written programs: each statement, each separator form, case in a lane or stride, and each refusal.
_ROP = (
    "claim add { reads A, B; writes C; count 1024; lane u; stride unit }",
    "claim m{reads C;writes A}\nclaim n { lane GGG ; stride Random ; count 0 ; }",
    "",
    "   \t\r\n",
    "claim a {}",
    "claim a { reads A, A, B }",
    "claim add {} claim add {}",
    "claim a { count 9223372036854775807 }",
    "claim a { count 00042 }",
    "claim a { # }",
    "claim a { reads A } @",
    "claim a { '",
    "claim a { \\ }",
    "claim a { \x00 }",
    "claim a { \x0b }",
    "claim a { \x7f }",
    "claim é",
    "x claim",
    "clam a {}",
    "claim {",
    "claim 3 {}",
    "claim a",
    "claim a { count x }",
    "claim a { reads A B }",
    "claim a { reads A,, B }",
    "claim a { reads }",
    "claim a { foo X }",
    "claim a { lane q }",
    "claim a { stride zz }",
    "claim a { count 9223372036854775808 }",
    "claim a { count 99999999999999999999999 }",
    "claim a { reads A; reads B }",
    "claim a { lane u; lane h }",
    "claim a { reads",
    "claim a { count 1 ;; }",
    "claim a { count 1 } }",
    "claim a { writes A",
)


# Hand-written machines: the oracle's test machine, a catch-all into an error state, and each refusal in its order.
_GOOD = [("s0", False, False), ("s1", False, False), ("acc", True, False), ("bad", False, True)]
_GOOD_TX = [
    ("s0", "s1", "claim", "begin"),
    ("s1", "acc", "ident", "name"),
    ("s1", "bad", "*", None),
    ("bad", "s0", "x", None),
]
_FSM = (
    (
        _GOOD,
        _GOOD_TX,
        "s0",
        [["claim", "ident"], ["claim"], ["ident"], ["claim", "zz", "x"], [], ["claim", "*"]],
    ),
    ([("a", True, False)], [("a", "a", "*", "any")], "a", [["p", "q", "*"], []]),
    ([("", False, False)], [], "", [[]]),
    ([("a", True, True)], [], "a", [[]]),
    ([("a", False, False)], [("", "a", "x", None)], "a", [[]]),
    ([("a", False, False)], [("a", "", "x", None)], "a", [[]]),
    ([("a", False, False)], [("a", "a", "", None)], "a", [[]]),
    ([("a", False, False), ("a", True, False)], [], "a", [[]]),
    ([("a", False, False)], [("b", "a", "x", None)], "a", [[]]),
    ([("a", False, False)], [("a", "b", "x", None)], "a", [[]]),
    ([("a", False, False)], [("a", "a", "x", None), ("a", "a", "x", "y")], "a", [[]]),
    ([("a", False, False)], [], "b", [[]]),
)


def _random_rop(rng: random.Random) -> str:
    ws = lambda: rng.choice([" ", "", "  ", "\t", "\n", "\r\n"])  # noqa: E731
    names = ["A", "B", "C", "x1", "_t", "Q9"]
    claims = []
    for _ in range(rng.randint(0, 3)):
        stmts = []
        for kw in rng.sample(["reads", "writes", "count", "lane", "stride"], rng.randint(0, 5)):
            if kw in ("reads", "writes"):
                arg = (ws() + "," + ws()).join(rng.choice(names) for _ in range(rng.randint(1, 3)))
            elif kw == "count":
                arg = str(rng.choice([0, 1, 7, 1024, 2**31, 2**63 - 1, 2**63, 10**20]))
            elif kw == "lane":
                arg = rng.choice(["u", "U", "ux", "T", "ggg", "a", "H", "w"])
            else:
                arg = rng.choice(["scalar", "UNIT", "strided", "CacheLine", "tile", "random", "zz"])
            stmts.append(f"{kw} {ws()}{arg}")
        body = (ws() + ";" + ws()).join(stmts) + (rng.choice([";", ""]) if stmts else "")
        claims.append(f"claim {rng.choice(['add', 'mul', 'k_2'])}{ws()}{{{ws()}{body}{ws()}}}")
    return ws().join(claims)


def _mutants(text: str, rng: random.Random, count: int) -> list:
    out = []
    pool = "{};, \t\nclaimreadswritescountlanestride0123456789AB_#@'\\\x00\x0b"
    for _ in range(count):
        t = list(text)
        for _ in range(rng.randint(1, 3)):
            op = rng.randrange(3)
            at = rng.randrange(len(t) + 1)
            if op == 0:
                t.insert(at, rng.choice(pool))
            elif t and at < len(t):
                if op == 1:
                    del t[at]
                else:
                    t[at] = rng.choice(pool)
        out.append("".join(t))
    return out


def _machines(rng: random.Random, count: int) -> list:
    pool = ["s0", "s1", "s2", "acc", "dead"]
    syms = ["a", "b", "c", "*"]
    out = []
    for _ in range(count):
        names = rng.sample(pool, rng.randint(1, 5))
        if rng.random() < 0.08:
            names.append(rng.choice(names))  # a repeated name
        if rng.random() < 0.05:
            names.insert(rng.randrange(len(names) + 1), "")  # an empty one
        states = [(n, rng.random() < 0.3, rng.random() < 0.15) for n in names]
        targets = names + (["nowhere"] if rng.random() < 0.08 else [])
        transitions = []
        for _ in range(rng.randint(0, 8)):
            transitions.append(
                (
                    rng.choice(targets) if rng.random() > 0.03 else "",
                    rng.choice(targets),
                    rng.choice(syms) if rng.random() > 0.03 else "",
                    rng.choice([None, None, "x", "y", ""]),
                )
            )
        start = rng.choice(names + (["nostart"] if rng.random() < 0.05 else []))
        runs = [
            [rng.choice(["a", "b", "c", "d", "*"]) for _ in range(rng.randint(0, 8))]
            for _ in range(3)
        ]
        out.append((states, transitions, start, runs))
    return out


def _events(rng: random.Random, count: int) -> list:
    out = []
    for _ in range(count):
        k = rng.randrange(3)
        name = rng.choice(["uart0", "rx", "", "pkt.1"])
        if k == 0:
            out.append(
                [
                    "stream",
                    name,
                    rng.choice(
                        ["text", "binary", "telemetry", "packet", "driver", "token", "video"]
                    ),
                    rng.choice(["utf8", "bytes", "le", "be", "records", "custom", "utf16"]),
                    rng.choice([8, 1, 0, -3, 64]),
                    rng.choice([4096, 1, 0, -1]),
                ]
            )
        elif k == 1:
            out.append(["kind", name])
        else:
            out.append(["event", name, rng.choice([0, 3, -1]), rng.choice([0, 2, 9])])
    return out


def check_tool(tool: str) -> dict:
    """`bcir-etl`'s text == the oracle's over every corpus; the counts compared. Raises AssertionError on the first
    difference. The gate's `etl` section runs this against the tool it built."""
    rng = random.Random(4)
    programs = list(_ROP) + [_random_rop(rng) for _ in range(300)]
    programs += [
        m
        for p in _ROP[:4] + tuple(programs[len(_ROP) : len(_ROP) + 40])
        for m in _mutants(p, rng, 6)
    ]
    machines = list(_FSM) + _machines(rng, 300)
    events = _events(rng, 300)
    with tempfile.TemporaryDirectory() as d:

        def run(mode: str, data: bytes) -> str:
            path = os.path.join(d, "in")
            with open(path, "wb") as fh:
                fh.write(data)
            r = subprocess.run([tool, mode, path], capture_output=True)
            assert r.returncode == 0, (mode, data[:200], r.stdout[-300:], r.stderr[-300:])
            return r.stdout.decode("utf-8")

        for text in programs:
            assert run("rop", text.encode("utf-8")) == expected_rop(text), text
        for states, transitions, start, runs in machines:
            want = expected_fsm(states, transitions, start, runs)
            assert run("fsm", fsm_file(states, transitions, start, runs).encode()) == want, (
                states,
                transitions,
                start,
            )
        body = "".join(" ".join(_word(str(w)) for w in line) + "\n" for line in events)
        assert run("event", body.encode()) == "".join(expected_event(line) for line in events)
    return {"programs": len(programs), "machines": len(machines), "events": len(events)}


def _build(d: str):
    """`bcir-etl` and `test_etl`, built from the checkout's sources; (None, None) where there are none to build (an
    installed package ships no runtime/c) or no compiler (the quick tier hides it)."""
    from bcir.tests.run_all import _is_source_checkout

    if _CC is None or not _is_source_checkout():
        return None, None
    ext = ".exe" if os.name == "nt" else ""
    tool, harness = os.path.join(d, "bcir-etl" + ext), os.path.join(d, "test_etl" + ext)
    for out, main in ((tool, "bcir_etl_tool.c"), (harness, "test_etl.c")):
        b = subprocess.run(
            [_CC, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-I", _C, "-o", out,
             os.path.join(_C, main), os.path.join(_C, "bcir_etl.c")],
            capture_output=True, text=True,
        )  # fmt: skip
        assert b.returncode == 0, b.stderr[-2000:]
    return tool, harness


def test_the_c_event_transducer_gives_the_oracle_s_verdict_on_every_corpus():
    """PORT-4. `bcir-etl` == the oracle over the hand-written, generated and mutated mini-ROP programs, machines with
    their runs, and event descriptors: every claim, resource, capture and trace step, and every refusal in the
    oracle's words at the offset or element it names."""
    with tempfile.TemporaryDirectory() as d:
        tool, _ = _build(d)
        if tool is None:
            return
        counts = check_tool(tool)
    assert (
        counts["programs"] > 500
        and counts["machines"] == 300 + len(_FSM)
        and counts["events"] == 300
    ), counts


def test_the_c_event_transducer_holds_its_own_contracts():
    """PORT-4. `test_etl.c`: a golden parse and run, every output buffer one short (a capacity, never a write past it),
    every NULL, and each refusal's element, field and offset."""
    with tempfile.TemporaryDirectory() as d:
        _, harness = _build(d)
        if harness is None:
            return
        r = subprocess.run([harness], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "OK", r.stdout[-2000:]


def test_the_corpora_reach_every_verdict():
    """The comparison is worth what its corpora reach (anti-vacuity): every refusal the oracle makes, an accepted
    program, a machine refused for each reason and one run to acceptance, and every descriptor refusal."""
    rng = random.Random(4)
    programs = list(_ROP) + [_random_rop(rng) for _ in range(300)]
    seen = {
        expected_rop(p).split(" ", 2)[1] if expected_rop(p).startswith("ERR") else "ok"
        for p in programs
    }
    assert {"ok", "unexpected", "expected", "unknown", "repeated", "count"} <= seen, seen
    machines = list(_FSM) + _machines(random.Random(4), 300)
    texts = [expected_fsm(*m) for m in machines]
    for words in (
        "fsm state: name",
        "accepting or an error",
        "fsm transition: src",
        "fsm transition: dst",
        "fsm transition: on",
        "duplicate fsm state",
        "names unknown state",
        "is declared twice",
        "unknown start state",
        "accepted=1",
        "accepted=0",
        "captures=begin:claim",
        "/*/",
    ):
        assert any(words in t for t in texts), words
    assert (
        sum("accepted" in t for t in texts[len(_FSM) :]) > 100
    )  # the generated machines mostly run
    events = {expected_event(e).split(":")[0] for e in _events(random.Random(4), 300)}
    assert len(events) >= 5, events
