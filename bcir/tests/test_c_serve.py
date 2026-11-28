"""N10's serving control plane in C: the paged KV page table and the token schema, at parity with the oracle.

`runtime/c/bcir_serve.c` is the C twin of `bcir.frontends.models.paged_kv.PagedKV`'s registry acts (construction
as allocation, each page's view admitted by D-R4; the capacity law; the data_gen a row bumps; eviction and its
map_gen) and of `bcir.frontends.models.serve`'s `check_token_dfa` and constrained step (`_masked_argmax`, the edge,
the deadlock). The numerics stay the decoder's. Held here, over generated decoders, manifests and schemas:

- **The page table** is PagedKV's refusal for refusal at construction (a refused page named with its admission's
  bits), and state for state after every row, eviction and lookup of a session driven to its capacity and past it.
- **The schema** is check_token_dfa's verdict message for message (each kind one bit of `bcir_dfa_check`'s), and
  each constrained step the oracle's token and successor, or its deadlock, over logits with ties, infinities and
  NaNs.
- **The contracts** the oracle cannot state run in `test_serve` itself: too few page records, page arithmetic past
  64 bits, the schema's layout and order, the transition bound, every NULL.
"""

from __future__ import annotations

import math
import os
import random
import subprocess
import tempfile

from bcir.frontends.models import serve
from bcir.frontends.models.decode import DecoderSpec
from bcir.frontends.models.paged_kv import PagedKV
from bcir.kbcir.device_manifest import DeviceManifest, MemoryBank, StridedView
from bcir.model import Domain
from bcir.tests.test_c_dma import refusal_bits
from bcir.tests.test_model_decode import _toy_weights
from bcir.toolchain import host_c_compiler

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_C = os.path.join(_ROOT, "runtime", "c")
_CC = host_c_compiler()  # None in the quick tier, which hides the host's compilers

# bcir_serve.h's status codes and check bits
KV_CAPACITY, KV_PAGE_SIZE, KV_RID, KV_BANK, KV_PAGES, KV_RID_RANGE = 2, 3, 4, 5, 6, 7
KV_VIEW, KV_FULL, KV_FREED, KV_POS, KV_NO_PAGE, KV_LIVE, KV_EVICTED = 9, 10, 11, 12, 13, 14, 15
DFA_DEADLOCK = 2
_DFA_BITS = (
    ("vocab_size must be", 0x001),
    ("needs at least one state", 0x002),
    ("states; limit is", 0x004),
    ("start state", 0x008),
    ("transitions", 0x010),
    ("outside the vocab", 0x020),
    ("duplicate allowed token", 0x040),
    ("which does not exist", 0x080),
)


def _manifest(r: random.Random) -> DeviceManifest:
    banks = tuple(
        MemoryBank(
            f"b{k}",
            Domain.RAM,
            r.choice((1 << 12, 1 << 13, 1 << 16)),
            native_tile=r.choice((1, 1, 2, 4, 8)),
            align=r.choice((1, 4, 64)),
        )
        for k in range(3)
    )
    distance = ((0, 256, 256), (256, 0, 256), (256, 256, 0))
    return DeviceManifest(device="serve", banks=banks, distance=distance)


def _spec(r: random.Random) -> DecoderSpec:
    n_heads = r.choice((1, 2, 2, 4))
    d_k = r.choice((2, 4))
    n_kv = r.choice([0] + [k for k in range(1, n_heads + 1) if n_heads % k == 0])
    return DecoderSpec(
        vocab_size=16,
        d_model=n_heads * d_k,
        n_heads=n_heads,
        n_layers=r.choice((1, 2, 3)),
        d_ff=8,
        n_kv_heads=n_kv,
    )


def _page_view(kv_bank: str, p: int, page_size: int, row_elems: int) -> StridedView:
    page_bytes = page_size * row_elems * 4
    return StridedView(kv_bank, p * page_bytes, (page_size, row_elems), (row_elems, 1), 4)


def _expected_init(spec, capacity, page_size, man, bank, base_rid):
    """(PagedKV or None, the line test_serve answers)."""
    try:
        kv = PagedKV(spec, capacity, page_size, man, bank, base_rid=base_rid)
    except KeyError:
        return None, f"ERR {KV_BANK} 0 0"
    except ValueError as exc:
        text = str(exc)
        for needle, code in (
            ("capacity must be", KV_CAPACITY),
            ("page_size must be", KV_PAGE_SIZE),
            ("base_rid must be", KV_RID),
            ("pages; limit is", KV_PAGES),
            ("page RID range overflows", KV_RID_RANGE),
        ):
            if needle in text:
                return None, f"ERR {code} 0 0"
        assert text.startswith("PagedKV page "), text
        p = int(text.split()[2])
        row_elems = 2 * max(1, spec.n_layers * spec.kv_dim)
        bits = refusal_bits(_page_view(bank, p, page_size, row_elems), man)
        assert bits, text
        return None, f"ERR {KV_VIEW} {p} {bits}"
    return kv, f"ok {kv.n_pages} {kv.row_elems} {kv.page_size * kv.row_elems * 4}"


def _state(kv: PagedKV) -> str:
    pages = " ".join(
        f"{pg.rid}:{pg.map_gen}:{pg.data_gen}:{int(kv.views[k] is not None)}"
        for k, pg in enumerate(kv.pages)
    )
    return f"pos {kv.pos} {pages}"


def _act(fn, refusals) -> str:
    try:
        return f"ok {fn()}"
    except ValueError as exc:
        for needle, code in refusals:
            if needle in str(exc):
                return f"ERR {code}"
        raise


def _session(r: random.Random, kv: PagedKV, spec, w, lines, want) -> None:
    """Drive a session past its capacity: rows, lookups and evictions, the state after each."""
    row = w.embedding.row(1)
    ops = ["append"] * kv.capacity + ["pageof", "evict"] * r.randrange(1, 4)
    r.shuffle(ops)
    ops += ["append"] + ["evict"] * (kv.n_pages + 2) + ["append", "pageof"]
    for op in ops:
        if op == "append":
            lines.append("append")
            want.append(
                _act(
                    lambda: (kv.step_row(row, spec, w), kv.pos)[1],
                    (("exceeds capacity", KV_FULL), ("was evicted", KV_FREED)),
                )
            )
        elif op == "pageof":
            pos = r.randrange(0, kv.capacity + 2)
            lines.append(f"pageof {pos}")
            want.append(_act(lambda pos=pos: kv.page_of(pos), (("position must be", KV_POS),)))
        else:
            p = r.randrange(0, kv.n_pages + 1)
            lines.append(f"evict {p}")
            want.append(
                _act(
                    lambda p=p: kv.evict(p).offset_bytes,
                    (
                        ("no page", KV_NO_PAGE),
                        ("refused at pos", KV_LIVE),
                        ("already evicted", KV_EVICTED),
                    ),
                )
            )
        lines.append("state")
        want.append(_state(kv))


def _logit_text(x: float) -> str:
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return float.hex(x)


def _dfa_bits(dfa, vocab: int) -> int:
    msgs = serve.check_token_dfa(dfa, vocab)
    bits = 0
    for m in msgs:
        hits = [bit for needle, bit in _DFA_BITS if needle in m]
        assert len(hits) == 1, m  # every message one kind
        bits |= hits[0]
    return bits


def _random_dfa(r: random.Random):
    """(the request line, the oracle's TokenDFA, vocab): sorted runs, so duplicates are adjacent."""
    vocab = r.choice((1, 5, 12, 12)) if r.random() > 0.03 else 0
    n = r.randrange(1, 7) if r.random() > 0.03 else 0
    start = r.randrange(0, n) if n and r.random() > 0.05 else n
    allowed, edges, text = [], [], []
    for _ in range(n):
        k = r.choice((1, 2, 3, 3, 4, 5)) if r.random() > 0.06 else 0
        span = vocab + (2 if r.random() < 0.15 else 0)
        toks = sorted(r.randrange(0, max(span, 1)) for _ in range(k))
        if r.random() > 0.15:
            toks = sorted(set(toks))
        nxt = [r.randrange(0, n + (1 if r.random() < 0.06 else 0)) for _ in toks]
        edge: dict = {}
        for t, s in zip(toks, nxt):
            edge.setdefault(t, s)  # the edge map keeps the first entry, as the C reader does
        allowed.append(tuple(toks))
        edges.append(edge)
        text.append(" ".join([str(len(toks))] + [f"{t}:{s}" for t, s in zip(toks, nxt)]))
    dfa = serve.TokenDFA(allowed=tuple(allowed), edges=tuple(edges), start=start)
    return " ".join([f"dfa {vocab} {start} {n}"] + text), dfa, vocab


_LOGITS = (-2.0, -0.5, 0.0, 0.0, 0.25, 1.0, 1.0, 3.5, math.inf, -math.inf, math.nan)


def requests(seed: int = 43, sessions: int = 160, schemas: int = 400):
    """(request lines, the oracle's answers) over generated sessions and schemas."""
    r = random.Random(seed)
    lines, want = [], []
    weights: dict = {}
    for k in range(sessions):
        man = _manifest(r)
        names = [b.name for b in man.banks]
        lines.append("reset")
        lines += [f"bank {b.capacity_bytes} {b.native_tile} {b.align}" for b in man.banks]
        spec = _spec(r)
        capacity = r.choice((1, 2, 7, 16, 24, 31, 40))
        page_size = r.randrange(1, capacity + 1)
        bank = r.choice(names) if r.random() > 0.05 else "nowhere"
        base_rid = 7000
        edge = r.random()
        if edge < 0.04:
            capacity = r.choice((0, (1 << 20) + 1))
        elif edge < 0.08:
            page_size = r.choice((0, capacity + 1))
        elif edge < 0.11:
            base_rid = r.randrange(0, 5)
        elif edge < 0.14:  # the last RIDs: one page fits, more do not
            base_rid, page_size = (1 << 32) - 1 - r.randrange(0, 2), 1
        elif edge < 0.16:
            capacity, page_size = (1 << 16) + 1 + r.randrange(0, 3), 1
        bank_index = names.index(bank) if bank in names else len(names)
        lines.append(
            f"kv {spec.n_layers} {spec.kv_dim} {capacity} {page_size} {bank_index} {base_rid}"
        )
        kv, line = _expected_init(spec, capacity, page_size, man, bank, base_rid)
        want.append(line)
        if kv is not None:
            if spec not in weights:
                weights[spec] = _toy_weights(spec)
            _session(r, kv, spec, weights[spec], lines, want)
    for _ in range(schemas):
        line, dfa, vocab = _random_dfa(r)
        lines.append(line)
        bits = _dfa_bits(dfa, vocab)
        want.append(f"check {bits}")
        if bits:
            continue
        state = dfa.start
        for _ in range(r.randrange(1, 12)):
            if r.random() < 0.1:
                state = r.randrange(0, len(dfa.allowed))
            logits = [r.choice(_LOGITS) for _ in range(vocab)]
            lines.append(f"step {state} " + " ".join(_logit_text(x) for x in logits))
            allowed = dfa.allowed[state]
            if not allowed:
                want.append(f"ERR {DFA_DEADLOCK}")
                state = dfa.start
                continue
            token = serve._masked_argmax(logits, allowed)
            nxt = dfa.edges[state][token]
            want.append(f"ok {token} {nxt}")
            state = nxt
    return lines, want


def check_driver(exe: str) -> dict:
    """`test_serve FILE` == the oracle over every request; what the corpus reached. The one comparison the test and
    the gate's `serve` section run."""
    lines, want = requests()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "requests")
        with open(path, "w", encoding="ascii") as f:
            f.write("".join(line + "\n" for line in lines))
        try:
            ran = subprocess.run([exe, path], capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AssertionError(f"test_serve did not run: {exc}") from exc
    assert ran.returncode == 0, ran.stdout[-2000:] + ran.stderr[-2000:]
    got = ran.stdout.splitlines()
    asked = [line for line in lines if not line.startswith(("bank", "reset"))]
    assert len(got) == len(want) == len(asked), (len(got), len(want), len(asked))
    for line, w, g in zip(asked, want, got):
        assert g == w, (line[:200], w[:200], g[:200])
    seen: dict = {"tables": 0, "rows": 0, "evicted": 0, "schemas": 0, "steps": 0}
    for line, w in zip(asked, want):
        verb = line.split()[0]
        if w.startswith("ERR"):
            key = f"{verb}.err{w.split()[1]}"
            seen[key] = seen.get(key, 0) + 1
        elif verb == "kv":
            seen["tables"] += 1
        elif verb == "append":
            seen["rows"] += 1
        elif verb == "evict":
            seen["evicted"] += 1
        elif verb == "dfa":
            bits = int(w.split()[1])
            key = "schemas" if bits == 0 else "refused"
            seen[key] = seen.get(key, 0) + 1
            for _, bit in _DFA_BITS:
                if bits & bit:
                    seen[f"bit{bit}"] = seen.get(f"bit{bit}", 0) + 1
        elif verb == "step":
            seen["steps"] += 1
    return seen


def _build(tmp: str) -> str | None:
    from bcir.tests.run_all import _is_source_checkout

    if _CC is None or not _is_source_checkout():
        return None
    exe = os.path.join(tmp, "test_serve" + (".exe" if os.name == "nt" else ""))
    argv = [_CC, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-I", _C]
    sources = ("test_serve.c", "bcir_serve.c", "bcir_dma.c")
    argv += [os.path.join(_C, s) for s in sources] + ["-o", exe]
    try:
        built = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AssertionError(f"test_serve did not build: {exc}") from exc
    assert built.returncode == 0, built.stderr[-3000:]
    return exe


def test_the_c_twin_pages_and_constrains_as_the_oracle_does():
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return
        seen = check_driver(exe)
        # not vacuous: tables built and refused for every reason, sessions run past their capacity, schemas
        # admitted and refused for every kind, steps taken and deadlocked
        assert seen["tables"] > 25 and seen["rows"] > 500 and seen["evicted"] > 20, seen
        for code in (KV_CAPACITY, KV_PAGE_SIZE, KV_RID, KV_BANK, KV_PAGES, KV_RID_RANGE, KV_VIEW):
            assert seen.get(f"kv.err{code}", 0) > 0, (code, seen)
        assert seen.get(f"append.err{KV_FULL}", 0) > 25, seen
        for code in (KV_NO_PAGE, KV_LIVE, KV_EVICTED):
            assert seen.get(f"evict.err{code}", 0) > 3, (code, seen)
        assert seen.get(f"pageof.err{KV_POS}", 0) > 3, seen
        assert seen["schemas"] > 100 and seen["steps"] > 400, seen
        for _, bit in _DFA_BITS[:2] + _DFA_BITS[3:4] + _DFA_BITS[5:]:
            assert seen.get(f"bit{bit}", 0) > 2, (bit, seen)
        assert seen.get(f"step.err{DFA_DEADLOCK}", 0) > 3, seen


def test_the_c_twin_holds_its_contracts():
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return
        ran = subprocess.run([exe], capture_output=True, text=True, timeout=300)
        assert ran.returncode == 0 and ran.stdout.strip() == "test_serve: OK", (
            ran.stdout + ran.stderr
        )
