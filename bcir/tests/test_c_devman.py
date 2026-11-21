"""N11 in C: device manifests bound at instantiation, at parity with the oracle.

`runtime/c/bcir_devman.c` is the C twin of `bcir.kbcir.device_manifest`'s `load_device_manifest`,
`check_device_manifest` and `probe_agree`: an instance binds its device's static schema (D-R1) from the persisted
envelope when it starts. Held here, over generated manifests -- names with quotes, backslashes, control characters,
non-ASCII and astral code points, every domain, capacities to 10^17 -- written as the oracle writes them and as a
human might (other key orders and spacing), and over mutations reaching every refusal:

- **The load** is `load_device_manifest`'s verdict for every envelope: the manifest (its banks and its digest,
  re-derived over the oracle's canonical JSON octet for octet) or the oracle's refusal, in the oracle's order,
  with the bank or row it names and, for an inconsistent manifest, each kind of message `check_device_manifest`
  writes.
- **The binding** is the oracle's admission: each generated view against the bank table the loaded manifest binds
  gets `check_strided_view`'s verdict bit for bit (N12's admission, now over a manifest read natively).
- **The probe** is `probe_agree`'s veto: its kinds and its message count for every observation.
- **The contracts** the oracle cannot state run in `test_devman` itself: a short workspace or bank table, a tile
  past 32 bits, the digest of a known envelope, every NULL.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import tempfile

from bcir.kbcir.device_manifest import (
    DeviceManifest,
    MemoryBank,
    StridedView,
    check_device_manifest,
    load_device_manifest,
    probe_agree,
    save_device_manifest,
)
from bcir.model import Domain
from bcir.tests.test_c_dma import refusal_bits
from bcir.toolchain import host_c_compiler

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_C = os.path.join(_ROOT, "runtime", "c")
_CC = host_c_compiler()  # None in the quick tier, which hides the host's compilers

# bcir_devman.h's status codes and bits
SIZE, READ, ROOT, KIND, SCHEMA_TYPE, SCHEMA_NEWER, SCHEMA = 2, 3, 4, 5, 6, 7, 8
FIELDS, TARGET, STALE, TYPES, BANK_SHAPE, BANK_TYPES, DOMAIN = 9, 10, 11, 12, 13, 14, 15
DISTANCE, DIGEST_FORM, TAMPERED, INVALID = 16, 17, 18, 19
_REFUSALS = (  # the oracle's messages, most specific first
    ("exceeds the 16 MiB", SIZE),
    ("malformed device manifest", READ),
    ("root must be an object", ROOT),
    ("document (kind=", KIND),
    ("schema must be an integer", SCHEMA_TYPE),
    ("is newer than", SCHEMA_NEWER),
    ("unsupported device-manifest schema", SCHEMA),
    ("missing or unknown fields", FIELDS),
    ("was authored for target", TARGET),
    ("is STALE", STALE),
    ("must have five fields", BANK_SHAPE),
    ("has unknown domain", DOMAIN),
    ("distance row", DISTANCE),
    ("device manifest bank", BANK_TYPES),  # "device manifest bank N has invalid field types"
    ("invalid field types", TYPES),
    ("digest must be lowercase", DIGEST_FORM),
    ("does not re-derive", TAMPERED),
    ("invalid device manifest:", INVALID),
)
_CHECK_BITS = (
    ("device must be a nonempty string", 0x001),
    ("no banks", 0x002),
    ("bank names must be nonempty", 0x004),
    ("duplicate bank names", 0x008),
    ("capacity must be >= 1", 0x010),
    ("native_tile must be >= 1", 0x020),
    ("align must be >= 1", 0x040),
    ("align must be a power of two", 0x080),
    ("distance matrix is not", 0x100),
    ("(staying put is free)", 0x200),
    ("must be >= 1 Q8", 0x400),
    ("(asymmetric)", 0x800),
)
PROBE_UNKNOWN, PROBE_CAPACITY, PROBE_TILE = 1, 2, 4

_NAMES = (
    "hbm0",
    "sram",
    "dram_far",
    "x",
    'q"uote',
    "back\\slash",
    "tab\tline\n",
    "é",
    "日本",
    "\U0001f600",
    "\x7f",
)


def _check_bits(man: DeviceManifest) -> int:
    bits = 0
    for message in check_device_manifest(man):
        hits = [bit for needle, bit in _CHECK_BITS if needle in message]
        assert len(hits) == 1, message
        bits |= hits[0]
    return bits


def _manifest(r: random.Random) -> DeviceManifest:
    n = r.randrange(1, 6)
    names = r.sample(_NAMES, n)
    banks = tuple(
        MemoryBank(
            names[k],
            r.choice(list(Domain)),
            r.choice((1, 4096, 1 << 20, 10**12, 10**17)),
            r.choice((1, 1, 2, 4, 8, 16)),
            r.choice((1, 4, 64, 4096)),
        )
        for k in range(n)
    )
    d = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d[i][j] = d[j][i] = r.choice((1, 128, 256, 512, 10**6))
    return DeviceManifest(
        device=r.choice(("cpu", "npu-7", 'dé"vice', "x" * 300)),
        banks=banks,
        distance=tuple(tuple(row) for row in d),
        target=r.choice(("", "x86_avx512", "aarch64_sve", "t\\1")),
        cal_gen=r.choice((0, 3, 2**40)),
    )


def _doc(man: DeviceManifest) -> dict:
    return {
        "kind": "bcir.device_manifest",
        "schema": 1,
        "device": man.device,
        "target": man.target,
        "cal_gen": man.cal_gen,
        "banks": [
            [b.name, int(b.domain), b.capacity_bytes, b.native_tile, b.align] for b in man.banks
        ],
        "distance": [list(row) for row in man.distance],
        "digest": man.digest,
    }


def _resealed(man: DeviceManifest, doc: dict, **bank_or_field) -> dict:
    """A manifest made inconsistent and its digest re-derived, so it reaches check_device_manifest."""
    doc = json.loads(json.dumps(doc))
    doc.update(bank_or_field)
    bad = DeviceManifest(
        device=doc["device"],
        banks=tuple(MemoryBank(b[0], Domain(b[1]), b[2], b[3], b[4]) for b in doc["banks"]),
        distance=tuple(tuple(row) for row in doc["distance"]),
        target=doc["target"],
        cal_gen=doc["cal_gen"],
    )
    doc["digest"] = bad.digest
    return doc


def _mutations(r: random.Random, man: DeviceManifest):
    """(the envelope's octets, a label) for every refusal, from one manifest."""
    doc = _doc(man)
    n = len(man.banks)

    def edit(**kv):
        d = json.loads(json.dumps(doc))
        d.update(kv)
        return d

    def bank(i, k, value):
        d = json.loads(json.dumps(doc))
        d["banks"][i][k] = value
        return d

    i = r.randrange(n)
    text = json.dumps(doc)
    out = [
        (text[: r.randrange(1, len(text) - 1)].encode(), "truncated"),
        ((text + " x").encode(), "trailing"),
        (text.replace('"kind"', '"kind": "x", "kind"', 1).encode(), "duplicate name"),
        (text.replace('"schema": 1', '"schema": NaN', 1).encode(), "NaN"),
        (text.replace('"device": ', '"device": "\\ud800" ', 1).encode(), "lone surrogate"),
        (text.encode().replace(b'"cpu"', b'"c\xffu"', 1), "not UTF-8"),
        (text.replace('"cal_gen": ', '"cal_gen": 1234567890123456789', 1).encode(), "19 digits"),
        (json.dumps(edit(distance=[[0]] * 257)).encode(), "257 elements"),
        (json.dumps(edit(banks=[[[[[[[["deep"]]]]]]]])).encode(), "too deep"),
        (json.dumps([doc]).encode(), "array root"),
        (b'"bcir.device_manifest"', "string root"),
        (json.dumps(edit(kind="other")).encode(), "kind"),
        (json.dumps({k: v for k, v in doc.items() if k != "kind"}).encode(), "no kind"),
        (json.dumps(edit(schema="1")).encode(), "schema string"),
        (json.dumps(edit(schema=1.0)).encode(), "schema float"),
        (json.dumps(edit(schema=True)).encode(), "schema bool"),
        (json.dumps(edit(schema=2)).encode(), "schema newer"),
        (json.dumps(edit(schema=0)).encode(), "schema older"),
        (json.dumps(edit(extra=1)).encode(), "extra field"),
        (json.dumps({k: v for k, v in doc.items() if k != "target"}).encode(), "no target"),
        (json.dumps(edit(device=7)).encode(), "device int"),
        (json.dumps(edit(target=None)).encode(), "target null"),
        (json.dumps(edit(cal_gen="3")).encode(), "cal_gen string"),
        (json.dumps(edit(cal_gen=-1)).encode(), "cal_gen negative"),
        (json.dumps(edit(cal_gen=1.5)).encode(), "cal_gen float"),
        (json.dumps(edit(cal_gen=True)).encode(), "cal_gen bool"),
        (json.dumps(edit(banks={"a": 1})).encode(), "banks object"),
        (json.dumps(edit(distance="near")).encode(), "distance string"),
        (
            json.dumps(
                edit(banks=doc["banks"][:i] + [doc["banks"][i][:4]] + doc["banks"][i + 1 :])
            ).encode(),
            "bank 4",
        ),
        (
            json.dumps(edit(banks=doc["banks"][:i] + ["bank"] + doc["banks"][i + 1 :])).encode(),
            "bank string",
        ),
        (json.dumps(bank(i, 0, 5)).encode(), "bank name int"),
        (json.dumps(bank(i, 1, True)).encode(), "domain bool"),
        (json.dumps(bank(i, 2, 4096.0)).encode(), "capacity float"),
        (json.dumps(bank(i, 4, "64")).encode(), "align string"),
        (json.dumps(bank(i, 1, 6)).encode(), "domain 6"),
        (json.dumps(bank(i, 1, -1)).encode(), "domain -1"),
        (
            json.dumps(
                edit(distance=doc["distance"][:i] + ["row"] + doc["distance"][i + 1 :])
            ).encode(),
            "row string",
        ),
        (
            json.dumps(
                edit(distance=doc["distance"][:i] + [[0.5] * n] + doc["distance"][i + 1 :])
            ).encode(),
            "row float",
        ),
        (
            json.dumps(
                edit(distance=doc["distance"][:i] + [[False] * n] + doc["distance"][i + 1 :])
            ).encode(),
            "row bool",
        ),
        (json.dumps(edit(digest=doc["digest"].upper())).encode(), "digest upper"),
        (json.dumps(edit(digest=doc["digest"][:63])).encode(), "digest short"),
        (json.dumps(edit(digest="g" * 64)).encode(), "digest not hex"),
        (json.dumps(edit(digest=7)).encode(), "digest int"),
        (json.dumps(bank(i, 2, man.banks[i].capacity_bytes + 1)).encode(), "tampered capacity"),
        (json.dumps(edit(device=man.device + "!")).encode(), "tampered device"),
        (json.dumps(_resealed(man, doc, device="")).encode(), "empty device"),
        (json.dumps(_resealed(man, doc, banks=[], distance=[])).encode(), "no banks"),
    ]
    for k, value in ((0, ""), (2, 0), (3, 0), (4, 0), (4, 3), (4, -8)):
        out.append(
            (json.dumps(_resealed(man, bank(i, k, value))).encode(), f"bank field {k}={value!r}")
        )
    if n > 1:
        dup = json.loads(json.dumps(doc))
        dup["banks"][1][0] = dup["banks"][0][0]
        out.append((json.dumps(_resealed(man, dup)).encode(), "duplicate bank names"))
        rows = json.loads(json.dumps(doc))
        rows["distance"][0] = rows["distance"][0][:-1]
        out.append((json.dumps(_resealed(man, rows)).encode(), "ragged distance"))
        asym = json.loads(json.dumps(doc))
        asym["distance"][0][1] += 1
        out.append((json.dumps(_resealed(man, asym)).encode(), "asymmetric"))
        zero = json.loads(json.dumps(doc))
        zero["distance"][0][1] = zero["distance"][1][0] = 0
        out.append((json.dumps(_resealed(man, zero)).encode(), "zero distance"))
    diag = json.loads(json.dumps(doc))
    diag["distance"][0][0] = 5
    out.append((json.dumps(_resealed(man, diag)).encode(), "diagonal"))
    out.append(
        (json.dumps(_resealed(man, doc, distance=doc["distance"][:-1])).encode(), "short matrix")
    )
    return out


def _expected_load(path: str, target: str, cal_gen) -> str:
    try:
        man = load_device_manifest(path, expect_target=target, expect_cal_gen=cal_gen)
    except ValueError as exc:
        text = str(exc)
        code = next(c for needle, c in _REFUSALS if needle in text)
        where, bits = 0, 0
        if code in (BANK_SHAPE, BANK_TYPES, DOMAIN, DISTANCE):
            where = int(text.split("bank " if code != DISTANCE else "row ")[1].split()[0])
        if code == INVALID:
            for message in text.split("invalid device manifest: ", 1)[1].split("; "):
                hits = [bit for needle, bit in _CHECK_BITS if needle in message]
                bits |= hits[0] if hits else 0
        return f"ERR {code} {where} {bits}"
    banks = " ".join(
        f"{b.name.encode().hex()}:{int(b.domain)}:{b.capacity_bytes}:{b.native_tile}:{b.align}"
        for b in man.banks
    )
    return f"ok {man.digest} {len(man.banks)} {banks} rows {len(man.distance)}"


def _view_text(view: StridedView, man: DeviceManifest) -> str:
    bank = [b.name for b in man.banks].index(view.bank)
    shape, strides = list(view.shape), list(view.strides)
    return " ".join(
        map(
            str,
            [bank, view.offset_bytes, view.elem_bytes, len(shape), *shape, len(strides), *strides],
        )
    )


def requests(tmp: str, seed: int = 47, manifests: int = 40):
    """(request lines, the oracle's answers) over generated envelopes, probes and views."""
    r = random.Random(seed)
    lines, want = [], []
    serial = 0

    def envelope(octets: bytes) -> str:
        nonlocal serial
        serial += 1
        path = os.path.join(tmp, f"m{serial}.json")
        with open(path, "wb") as f:
            f.write(octets)
        return path

    for _ in range(manifests):
        man = _manifest(r)
        path = os.path.join(tmp, f"saved{serial}.json")
        serial += 1
        save_device_manifest(path, man)
        doc = _doc(man)
        keys = list(doc)
        r.shuffle(keys)
        respelled = json.dumps(
            {k: doc[k] for k in keys}, ensure_ascii=r.random() < 0.5, separators=(",", ":")
        )
        for p in (path, envelope(respelled.encode())):
            for target, cal in (
                ("", None),
                (man.target, man.cal_gen),
                ("other", None),
                ("", man.cal_gen + 1),
            ):
                lines.append(
                    f"load {p} {target.encode().hex() if target else '-'} {'-' if cal is None else cal}"
                )
                want.append(_expected_load(p, target, cal))
        lines.append(f"load {path} - -")
        want.append(_expected_load(path, "", None))
        loaded = load_device_manifest(path)
        for _ in range(4):  # the probe's veto, against the manifest just loaded
            observed = {}
            for b in r.sample(loaded.banks, r.randrange(0, len(loaded.banks) + 1)):
                facts = {}
                if r.random() < 0.7:
                    facts["capacity_bytes"] = b.capacity_bytes + (1 if r.random() < 0.3 else 0)
                if r.random() < 0.7:
                    facts["native_tile"] = b.native_tile * (2 if r.random() < 0.3 else 1)
                observed[b.name] = facts
            if r.random() < 0.3:
                observed["ghost"] = {"capacity_bytes": 1}
            msgs = probe_agree(loaded, observed)
            bits = 0
            for m in msgs:
                bits |= (
                    PROBE_UNKNOWN
                    if "unknown bank" in m
                    else PROBE_CAPACITY
                    if "capacity_bytes" in m
                    else PROBE_TILE
                )
            parts = [str(len(observed))]
            for name, facts in observed.items():
                parts += [
                    name.encode().hex(),
                    str(facts.get("capacity_bytes", "-")),
                    str(facts.get("native_tile", "-")),
                ]
            lines.append("probe " + " ".join(parts))
            want.append(f"probe {bits} {len(msgs)}")
        for _ in range(6):  # views against the bound table: N12's admission over the native load
            b = r.choice(loaded.banks)
            rows = r.choice((1, 2, 4, 15, 16))
            cols = r.choice((1, 4, 16, 64))
            offset = r.choice(
                (0, b.align, 3, b.capacity_bytes - 64 if b.capacity_bytes > 64 else 0)
            )
            view = StridedView(
                b.name, offset, (rows, cols), (cols * r.choice((1, 2)), 1), r.choice((1, 4))
            )
            lines.append("admit " + _view_text(view, loaded))
            want.append(str(refusal_bits(view, loaded)))
        for octets, _label in _mutations(r, man):
            p = envelope(octets)
            target, cal = (man.target, man.cal_gen) if r.random() < 0.3 else ("", None)
            lines.append(
                f"load {p} {target.encode().hex() if target else '-'} {'-' if cal is None else cal}"
            )
            want.append(_expected_load(p, target, cal))
    many = dict(
        _doc(man), banks=[[[0] * 255] * 256] * 5
    )  # past the scan's 2**18 nodes, under 1 MiB
    nodes = envelope(json.dumps(many, separators=(",", ":")).encode())
    lines.append(f"load {nodes} - -")
    want.append(_expected_load(nodes, "", None))
    big = envelope(b" " * ((16 << 20) + 1))
    lines.append(f"load {big} - -")
    want.append(_expected_load(big, "", None))
    return lines, want


def check_driver(exe: str) -> dict:
    """`test_devman FILE` == the oracle over every request; what the corpus reached. The one comparison the test and
    the gate's `devman` section run."""
    with tempfile.TemporaryDirectory() as tmp:
        lines, want = requests(tmp)
        path = os.path.join(tmp, "requests")
        with open(path, "w", encoding="ascii") as f:
            f.write("".join(line + "\n" for line in lines))
        try:
            ran = subprocess.run([exe, path], capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AssertionError(f"test_devman did not run: {exc}") from exc
    assert ran.returncode == 0, ran.stdout[-2000:] + ran.stderr[-2000:]
    got = ran.stdout.splitlines()
    assert len(got) == len(want) == len(lines), (len(got), len(want), len(lines))
    for line, w, g in zip(lines, want, got):
        assert g == w, (line[:200], w[:300], g[:300])
    seen: dict = {"loaded": 0, "probes": 0, "vetoed": 0, "views": 0, "admitted": 0}
    for line, w in zip(lines, want):
        verb = line.split()[0]
        if verb == "load":
            if w.startswith("ok"):
                seen["loaded"] += 1
            else:
                code, bits = int(w.split()[1]), int(w.split()[3])
                seen[f"err{code}"] = seen.get(f"err{code}", 0) + 1
                for _, bit in _CHECK_BITS:
                    if bits & bit:
                        seen[f"bit{bit}"] = seen.get(f"bit{bit}", 0) + 1
        elif verb == "probe":
            seen["probes"] += 1
            seen["vetoed"] += int(w.split()[1]) != 0
        else:
            seen["views"] += 1
            seen["admitted"] += w == "0"
    return seen


def _build(tmp: str) -> str | None:
    from bcir.tests.run_all import _is_source_checkout

    if _CC is None or not _is_source_checkout():
        return None
    exe = os.path.join(tmp, "test_devman" + (".exe" if os.name == "nt" else ""))
    argv = [_CC, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-I", _C]
    sources = (
        "test_devman.c",
        "bcir_devman.c",
        "bcir_jer.c",
        "bcir_sha256.c",
        "bcir_dma.c",
        "bcir_runtime.c",
    )
    argv += [os.path.join(_C, s) for s in sources] + ["-o", exe]
    try:
        built = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AssertionError(f"test_devman did not build: {exc}") from exc
    assert built.returncode == 0, built.stderr[-3000:]
    return exe


def test_the_c_twin_binds_manifests_as_the_oracle_loads_them():
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return
        seen = check_driver(exe)
        # not vacuous: manifests loaded in both spellings, every refusal reached, every consistency kind, probes
        # vetoed and agreeing, views admitted and refused over the bound table
        assert seen["loaded"] > 100, seen
        for code in range(SIZE, INVALID + 1):
            assert seen.get(f"err{code}", 0) > 0, (code, seen)
        for _, bit in _CHECK_BITS:
            assert seen.get(f"bit{bit}", 0) > 0, (bit, seen)
        assert 20 < seen["vetoed"] < seen["probes"], seen
        assert 20 < seen["admitted"] < seen["views"], seen


def test_the_c_twin_holds_its_contracts():
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return
        ran = subprocess.run([exe], capture_output=True, text=True, timeout=300)
        assert ran.returncode == 0 and ran.stdout.strip() == "test_devman: OK", (
            ran.stdout + ran.stderr
        )


def test_the_oracle_reads_through_the_bounded_reader():
    """One reader on both rails: the oracle refuses what the bounded scan refuses (an escaped lone surrogate, a
    19-digit integer, a 257-element array) and compares a calibration generation only when it is an integer."""
    man = DeviceManifest(
        device="d",
        banks=(MemoryBank("a", Domain.RAM, 4096, 1, 64),),
        distance=((0,),),
        target="t",
        cal_gen=3,
    )
    doc = _doc(man)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.json")
        for text, expect, needle in (
            (json.dumps(doc).replace('"device": "d"', '"device": "\\ud800"'), None, "malformed"),
            (
                json.dumps(doc).replace('"cal_gen": 3', '"cal_gen": 1234567890123456789'),
                None,
                "malformed",
            ),
            (json.dumps(dict(doc, distance=[[0]] * 257)), None, "malformed"),
            (
                json.dumps(dict(doc, cal_gen=True)),
                0,
                "invalid field types",
            ),  # not STALE: not an integer
            (json.dumps(dict(doc, cal_gen=3.0)), 0, "invalid field types"),
        ):
            with open(path, "w", encoding="ascii") as f:
                f.write(text)
            try:
                load_device_manifest(path, expect_cal_gen=expect)
                raise AssertionError("expected a refusal")
            except ValueError as exc:
                assert needle in str(exc), (needle, str(exc))
        save_device_manifest(path, man)
        assert load_device_manifest(path, expect_cal_gen=3) == man
