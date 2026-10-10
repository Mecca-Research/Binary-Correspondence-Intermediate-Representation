"""QUAL-3: the lab-model builder and the qualification harness, held to their contracts.

`tools/models/build_lab_model.py` builds a BCIR-native decoder from a recipe (the tracked corpus,
BCIR's BPE, the native C trainer, BCIRQ8), and `tools/models/run_qualification.py` runs the whole
vertical stack on it: the program, its plan and pack, every projection (JSON <-> ASN.1 <-> GEM),
GEM against the monolithic runner on both rails, latency and memory traffic. These tests drive
both on a tiny recipe -- a corpus of four tracked documents, a 280-token BPE, a one-layer decoder,
two training steps -- so every stage runs in seconds, and assert the failures as well as the
pass: a recipe the builder must refuse before any work, a model card that names another
artifact, a tokenizer that is not the one the weights were trained with.

The simulated memory traffic is read from callgrind's own profile parts, one per rail, which
bcir-qualify's callgrind build dumps around one run of each rail; the reader is held to that
format's grammar here without valgrind, and, where valgrind is installed, the bracketed build
is held to measuring one run of each rail and the plain build to measuring nothing.

The CI job `qualification` runs the real recipe (`configs/bcir-docs-860k.json`) under
`--require-callgrind` on x86-64 and AArch64; this module is its unit half. It needs the
checkout's tools/ tree, git and a C compiler, so it is repository-only (`run_all`).
"""

from __future__ import annotations

import copy
import functools
import importlib.util
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_RECIPE = ROOT / "tools" / "models" / "configs" / "bcir-docs-860k.json"
_CC = __import__("bcir.toolchain", fromlist=["host_c_compiler"]).host_c_compiler()
#: The tiny model is built (C trainer, BCIRQ8) only in the thorough tier.
_THOROUGH = os.environ.get("BCIR_THOROUGH") == "1"
#: valgrind's callgrind, for the bracketed build's own test; CI's qualification job requires it.
_VALGRIND = shutil.which("valgrind")


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "models" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_recipe() -> dict:
    """The shipped recipe at the smallest size every stage still exercises."""
    recipe = json.loads(_RECIPE.read_text(encoding="utf-8"))
    recipe["name"] = "BCIR-Docs-Tiny"
    recipe["corpus"]["include"] = [
        "docs/PERFORMANCE_AUDIT.md",
        "docs/BCIR_TARGET_ACCESS.md",
        "docs/BCIR_NATIVE_OBJECT_GATE.md",
        "docs/ONBOARDING_DEEP_DIVE.md",
    ]
    recipe["corpus"]["validation_permyriad"] = 2000
    recipe["tokenizer"].update(vocab_size=280, sample_bytes=32768)
    recipe["evaluation"].update(max_new_tokens=4, validation_batches=1)
    train = recipe["train"]
    train.update(batch_size=2, context_length=16, steps=2, warmup_steps=1, checkpoint_every=2)
    train["decoder"].update(d_model=16, n_heads=2, n_kv_heads=1, n_layers=1, d_ff=32)
    train["decoder"]["vocab_size"] = 280
    return recipe


def _write(path: Path, recipe: dict) -> Path:
    path.write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _refusal(builder, path: Path) -> str:
    try:
        builder.load_recipe(path)
    except ValueError as error:
        return str(error)
    raise AssertionError(f"{path.name} was accepted")


def test_a_recipe_is_checked_whole_before_any_work():
    """Every field is checked before a byte of corpus is read: a duplicate key (which JSON
    would silently resolve to the last), a field the schema does not have, another schema, a
    pathspec outside the checkout, a tokenizer smaller than its own byte alphabet, a decoder
    whose vocabulary is not the tokenizer's. Both halves: the shipped recipe loads."""
    builder = _tool("build_lab_model")
    recipe, spec, digest = builder.load_recipe(_RECIPE)
    assert recipe["schema"] == builder.RECIPE_SCHEMA and len(digest) == 64
    assert spec.decoder.vocab_size == recipe["tokenizer"]["vocab_size"]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "recipe.json"
        text = _RECIPE.read_text(encoding="utf-8")
        path.write_text(text.replace('"name":', '"name": "first",\n  "name":', 1), "utf-8")
        assert "duplicate key 'name'" in _refusal(builder, path)
        cases = {
            "exactly the fields": lambda r: r.update(extra=1),
            "unsupported recipe schema": lambda r: r.update(schema="bcir.lab_model.v0"),
            "relative pathspecs": lambda r: r["corpus"].update(include=["/etc/passwd"]),
            "tokenizer.vocab_size": lambda r: r["tokenizer"].update(vocab_size=260),
            "the tokenizer's": lambda r: r["train"]["decoder"].update(vocab_size=2048),
        }
        for message, edit in cases.items():
            bad = copy.deepcopy(json.loads(text))
            edit(bad)
            assert message in _refusal(builder, _write(path, bad)), message


@functools.lru_cache(maxsize=1)
def _tiny_model() -> tuple[Path, dict] | None:
    """The tiny recipe built once per process: (output directory, model card)."""
    if _CC is None or not _THOROUGH:
        return None
    builder = _tool("build_lab_model")
    root = Path(tempfile.mkdtemp(prefix="bcir-qual-"))
    recipe = _write(root / "recipe.json", _tiny_recipe())
    card = builder.build(recipe, root / "model", log=lambda *_: None)
    return root / "model", card


def test_the_builder_ties_every_artifact_to_the_card():
    """The card names each artifact by SHA-256 and the BCIRQ8 file names the tokenizer it was
    trained with; the tokenizer file is its canonical JSON exactly (the reader refuses any
    other spelling, so a trailing newline made the shipped model unloadable); the sample the
    card records is what the BCIRQ8 weights generate."""
    built = _tiny_model()
    if built is None:
        return  # a C compiler and the thorough tier: the native trainer is compiled
    from bcir.frontends.models.decode import decode_with_kv_cache
    from bcir.frontends.models.weights_io import read_q8_decoder
    from bcir.hosted.training.bpe import BytePairTokenizer

    model_dir, card = built
    builder = _tool("build_lab_model")
    sha = builder._sha256
    tokenizer_text = (model_dir / "tokenizer.json").read_text(encoding="utf-8")
    tokenizer = BytePairTokenizer.from_json(tokenizer_text)
    assert tokenizer.to_json() == tokenizer_text
    spec, weights, meta = read_q8_decoder(model_dir / "model.bcirq8")
    assert card["tokenizer"]["sha256"] == sha(model_dir / "tokenizer.json") == meta.tokenizer_sha256
    assert card["artifacts"]["bcirq8_sha256"] == sha(model_dir / "model.bcirq8")
    assert card["artifacts"]["checkpoint_sha256"] == sha(model_dir / "model.bcirdt")
    assert meta.source_model_sha256 == card["artifacts"]["checkpoint_sha256"]
    assert card["parameters"] > 0 and card["training"]["steps"] == 2
    sample = card["sample"]
    assert sample["prompt_ids"] == tokenizer.encode(sample["prompt"], add_bos=True)
    assert sample["generated_ids"] == decode_with_kv_cache(sample["prompt_ids"], spec, weights, 4)


def _qualify(model_dir: Path):
    """The harness on the tiny model. Its wall-clock ratio row is not judged here: a request of
    a one-layer, 16-wide decoder is microseconds of kernel work, so GEM's per-run trust boundary
    (the pack's CRC and R10 walk, re-verified on every run) dominates it -- about 4x here, where
    the real model runs at 1.06. The band (`--max-gem-ratio`, 1.5) is for a real model, and CI's
    `qualification` job applies it to one."""
    harness = _tool("run_qualification")
    return harness.qualify(
        model_dir,
        max_new=4,
        json_max_new=2,
        reps=1,
        max_gem_ratio=math.inf,
        callgrind="off",
        log=lambda *_: None,
    )


def test_the_qualification_passes_end_to_end_on_a_built_model():
    """Every gated check of the harness on a real (tiny) BCIR-native model: the artifacts are
    the card's, the program verifies, every projection round-trips, every Q8 rail generates the
    same tokens, GEM and the monolithic runner agree to the bit on both rails and from the DER
    converted in C, the JSON-carried request reproduces the headline's prefix, and counted
    traffic equals planned traffic. A check that never ran cannot pass: each is named."""
    built = _tiny_model()
    if built is None:
        return
    model_dir, card = built
    report, ok = _qualify(model_dir)
    names = [row["check"] for row in report["checks"] if row["ok"]]
    failed = [row for row in report["checks"] if not row["ok"]]
    assert ok and report["status"] == "PASS" and not failed, failed
    for needle in (
        "model artifact is the card's",
        "the tokenizer is the one",
        "the program verifies (R1-R25)",
        "round trip: pack der",
        "every Q8 rail generates the same tokens",
        "C GEM == monolithic C, bit for bit (native pack)",
        "C GEM == monolithic C, bit for bit (der pack)",
        "oracle GEM == oracle monolithic, bit for bit",
        "JSON -> ASN.1 -> GEM",
        "counted traffic == planned traffic, op for op",
    ):
        assert any(name.startswith(needle) for name in names), needle
    assert report["schema"] == "bcir.qualification_report.v1"
    assert report["model"]["bcirq8_sha256"] == card["artifacts"]["bcirq8_sha256"]
    assert report["correctness"]["model_card_sample_matches"] is True
    assert report["correctness"]["logits"]["max_abs_c_vs_python"] <= 1e-9
    assert report["correctness"]["float_reference"]["tokens"]  # the checkpoint was read
    assert report["traffic"]["simulated"] == {"skipped": "not requested"}


def test_the_qualification_refuses_a_model_its_card_does_not_name():
    """A qualification is of one model: a card naming another BCIRQ8 file fails its check (the
    rest still runs, so the report says what else held), and a tokenizer that is not the one the
    weights name -- here its canonical JSON with a newline appended -- is refused by the reader,
    and the command exits 1 without a report of PASS."""
    built = _tiny_model()
    if built is None:
        return
    model_dir, _card = built
    harness = _tool("run_qualification")
    with tempfile.TemporaryDirectory() as tmp:
        other = Path(tmp) / "model"
        shutil.copytree(model_dir, other, ignore=shutil.ignore_patterns("native"))
        card_path = other / "model_card.json"
        card = json.loads(card_path.read_text(encoding="utf-8"))
        card["artifacts"]["bcirq8_sha256"] = "00" * 32
        card_path.write_text(json.dumps(card), encoding="utf-8")
        report, ok = _qualify(other)
        failed = [row["check"] for row in report["checks"] if not row["ok"]]
        assert not ok and report["status"] == "FAIL"
        assert failed == ["model artifact is the card's"], failed
        card["artifacts"]["bcirq8_sha256"] = _card["artifacts"]["bcirq8_sha256"]
        card_path.write_text(json.dumps(card), encoding="utf-8")
        tokenizer = other / "tokenizer.json"
        tokenizer.write_text(tokenizer.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        out = Path(tmp) / "report.json"
        argv = ["--model-dir", str(other), "--output", str(out), "--max-new", "4"]
        argv += ["--json-max-new", "2", "--reps", "1", "--no-callgrind"]
        assert harness.main(argv) == 1
        assert not out.exists()


def test_the_harness_refuses_arguments_that_would_measure_nothing():
    """A run of no tokens, no repetitions or a non-positive ratio band would report a vacuous
    PASS; each is a usage error (exit 2) before any model is read."""
    harness = _tool("run_qualification")
    base = ["--model-dir", "/nonexistent", "--output", "/nonexistent/report.json"]
    for extra in (
        ["--max-new", "0"],
        ["--json-max-new", "0"],
        ["--reps", "0"],
        ["--max-gem-ratio", "0"],
        ["--max-gem-ratio", "nan"],
    ):
        assert harness.main(base + extra) == 2, extra


_EVENTS = "Ir Dr Dw I1mr D1mr D1mw ILmr DLmr DLmw"


def _part(trigger: str, totals: str, *, events: str = _EVENTS) -> str:
    """A callgrind profile part laid out as callgrind 3.22 writes one -- its header, a cost entry,
    a call, the totals line -- with the given trigger, events and totals."""
    return (
        "# callgrind format\nversion: 1\ncreator: callgrind-3.22.0\npid: 1019\n"
        "cmd:  ./bcir-qualify model.bcirq8 pack.bin --native --prompt 1,36,271 --reps 1\n"
        "part: 2\n\n\n"
        "desc: I1 cache: 32768 B, 64 B, 8-way associative\n"
        "desc: D1 cache: 32768 B, 64 B, 8-way associative\n"
        "desc: LL cache: 8388608 B, 64 B, 16-way associative\n\n"
        "desc: Timerange: Basic block 69696 - 109699\n"
        f"desc: Trigger: {trigger}\n\n"
        f"positions: line\nevents: {events}\nsummary: {totals}\n\n\n"
        "ob=(5) ./bcir-qualify\nfl=(168) runtime/c/bcir_qualify_cli.c\nfn=(552) main\n"
        "10 3 0 1 1 0 0 1\n-5 1 0 1\ncfn=(522)\ncalls=0 0 \n0 1040017 0 8 3 0 0 3\n\n"
        f"totals: {totals}\n"
    )


def _part_refusal(harness, text: str) -> str:
    try:
        harness.callgrind_part(text)
    except ValueError as error:
        return str(error)
    raise AssertionError(f"a malformed callgrind part was read: {text[-120:]!r}")


def test_a_callgrind_part_is_read_by_its_own_grammar():
    """The simulated rows come from callgrind's machine format, read by its grammar: the trigger,
    the events and the totals, a short totals line padded with the zero counts the format leaves
    out. QUAL-3's first reader scraped callgrind_annotate's human-readable table instead, and on
    AArch64 it parsed a line that was no function's row (`int('')`). A part outside the grammar
    is refused, never read as a number: a header line missing or twice, an event twice or not
    ASCII, more counts than events, and each count a host-language parser would take but the
    format never writes -- the annotate table's thousands separators, an underscore, a sign, a
    non-ASCII digit -- or a doubled, trailing or missing separator."""
    harness = _tool("run_qualification")
    good = _part("Client Request: gem", "1040019 0 9 4 0 0 4")
    trigger, totals = harness.callgrind_part(good)
    assert trigger == "Client Request: gem"
    assert totals == {
        "Ir": 1040019, "Dr": 0, "Dw": 9, "I1mr": 4, "D1mr": 0,
        "D1mw": 0, "ILmr": 4, "DLmr": 0, "DLmw": 0,
    }  # fmt: skip
    refusals = {
        "one 'totals:' line, not 0": [good.replace("totals: ", "total: ")],
        "one 'totals:' line, not 2": [good + "totals: 1040019\n"],
        "one 'events:' line, not 0": [good.replace("events: ", "event: ")],
        "one 'desc: Trigger:' line, not 2": [
            good.replace("desc: Timerange", "desc: Trigger: Client Request: monolithic\ndesc: T")
        ],
        "events outside the grammar": [
            _part("Client Request: gem", "5", events=events)
            for events in ("Ir Ir Dw", "Ir Dr İr", "Ir  Dr", "Ir Dr ", "")
        ],
        "totals outside the grammar": [
            _part("Client Request: gem", totals)
            for totals in (
                "1,040,019 0 9",
                "1_040_019 0 9",
                "+1040019 0 9",
                "١٠٤٠٠١٩ 0 9",
                "1040019  0 9",
                "1040019 0 9 ",
                "1 2 3 4 5 6 7 8 9 10",
                "",
            )
        ],
    }
    for message, texts in refusals.items():
        for text in texts:
            refusal = _part_refusal(harness, text)
            assert message in refusal, (message, refusal)


def test_each_rail_is_one_part_and_an_absent_part_is_not_a_zero():
    """A profile yields exactly one part per rail, the one its client request named, and the
    row derived from it sums its counts (data references, last-level data misses and the octets
    those move, one modelled line each). A profile of a binary that brackets nothing (only the
    termination dump), a rail dumped twice, a request no rail makes, a rail's part counted
    without the cache simulation or counting no instruction, and a malformed part anywhere in
    the profile are each refused: a measurement that did not happen is not a zero."""
    harness = _tool("run_qualification")
    gem = _part("Client Request: gem", "1040019 7 9 4 5 6 4 2 3")
    mono = _part("Client Request: monolithic", "520017 0 8 4 0 0 4")
    end = _part("Program termination", "4580 1001 849 231 49 56 215 23 48")
    rails = harness.callgrind_rails([mono, end, gem])
    assert sorted(rails) == ["gem", "monolithic"]
    assert rails["gem"]["DLmw"] == 3 and rails["monolithic"]["DLmw"] == 0
    row = harness.traffic_row(rails["gem"])
    assert row["data_refs"] == 7 + 9 and row["ll_data_misses"] == 2 + 3
    assert row["ll_miss_octets"] == 5 * harness.CACHE_MODEL["LL"][2] == 320
    assert {k: row[k] for k in rails["gem"]} == rails["gem"]
    refusals = {
        "no callgrind part for the gem and monolithic rail": [end],
        "no callgrind part for the monolithic rail": [gem, end],
        "two callgrind parts for the gem rail": [gem, mono, gem],
        "a client request no rail makes": [gem, mono, _part("Client Request: gem2", "5")],
        "the gem part does not count Dr, Dw, I1mr, D1mr, D1mw, ILmr, DLmr, DLmw": [
            _part("Client Request: gem", "5", events="Ir"),
            mono,
        ],
        "the gem part counted no instructions": [_part("Client Request: gem", "0 7 9"), mono],
        "one 'totals:' line, not 0": [gem, mono, end.replace("totals: ", "total: ")],
    }
    for message, parts in refusals.items():
        try:
            harness.callgrind_rails(parts)
        except ValueError as error:
            assert message in str(error), (message, str(error))
        else:
            raise AssertionError(f"accepted: {message}")


def test_the_callgrind_build_measures_one_run_of_each_rail():
    """Under valgrind, bcir-qualify's callgrind build yields one part per rail: profiling two
    repetitions counts the same instructions and data references per rail as profiling one (a
    bracket around every repetition would dump each rail twice, and be refused), the rows are
    the counts' sums, a directory holding an earlier profile is refused before anything runs,
    and the plain build, which brackets nothing, measures nothing rather than zero. The real
    model runs the same path in CI (`--require-callgrind`, x86-64 and AArch64)."""
    built = _tiny_model()
    if built is None or not _VALGRIND:
        return  # the thorough tier with valgrind's callgrind
    from bcir.frontends.models.weights_io import read_q8_decoder

    model_dir, card = built
    harness = _tool("run_qualification")
    q8 = model_dir / "model.bcirq8"
    spec, _weights, _meta = read_q8_decoder(q8)
    ids = card["sample"]["prompt_ids"]
    program = harness.decoder_program(spec, len(ids), 4)
    result = harness.optimize(
        program.module, harness.TargetProfile.for_host(), harness.Theta.cool()
    )
    pack = harness.hydrate_pipelined(program.module, result, depth=2)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        native = work / "pack.bin"
        native.write_bytes(harness.encode(pack))
        sim, reason = harness.callgrind_traffic(q8, native, ids, work)
        assert sim is not None, reason
        exe = work / "callgrind-build" / "bcir-qualify"
        twice = harness.profile_rails(exe, q8, native, ids, work / "twice", reps=2)
        for rail in harness.CALLGRIND_RAILS:
            row = sim[rail]
            assert row["Ir"] > 0 and row["Dr"] > 0 and row["Dw"] > 0, (rail, row)
            assert row["data_refs"] == row["Dr"] + row["Dw"], (rail, row)
            assert row["ll_miss_octets"] == 64 * row["ll_data_misses"], (rail, row)
            once = {event: row[event] for event in ("Ir", "Dr", "Dw")}
            assert once == {event: twice[rail][event] for event in once}, (rail, once, twice)
        try:  # parts an earlier profile left would be read as this one's
            harness.profile_rails(exe, q8, native, ids, work / "twice")
        except harness.QualificationError as error:
            assert "already holds a callgrind profile" in str(error), error
        else:
            raise AssertionError("a profile was read from a directory an earlier one wrote")
        plain = harness.build_qualify(work / "plain")
        try:
            harness.profile_rails(plain, q8, native, ids, work / "plain-profile")
        except ValueError as error:
            assert "no callgrind part for the gem and monolithic rail" in str(error), error
        else:
            raise AssertionError("the plain build's profile was read as a measurement")
