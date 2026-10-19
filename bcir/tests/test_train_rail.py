"""The train rail (TRAIN-RAIL): the training corpus is the IR's data, one way (AGENTS.md section 2,
docs/research/BCIR_TRAINING_ACCELERATION_ROADMAP.md section 5).

tools/models/train_rail.sh trains the BCIR-native decoder on the corpus recipe
(tools/models/configs/bcir-corpus-860k.json: docs/ and training/) through
tools/models/build_lab_model.py, then reads the corpus witness, tools/models/train_rail.py: every
tracked Markdown file of both trees is in what the model trained on -- `train.corpus.unselected`
0, the rail's exact GEM+ row. CTest runs the script as `train-corpus` (label train), BCIR Make as
the task `train.corpus`, CI's qualification job before qualifying the model. The training itself is
CI's and `ctest -L train`'s (a compiler and minutes); what is held here needs neither: the recipe's
selection against the tree, the witness firing on a narrower recipe and refusing what it cannot
read, the script refusing what it cannot run, and every place the rail lives naming the same
script, recipe, roots and step count (L14).
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from bcir.tests.test_qualification import _tool

_ROOT = Path(__file__).resolve().parents[2]
_RECIPE = _ROOT / "tools" / "models" / "configs" / "bcir-corpus-860k.json"
_DOCS_RECIPE = _ROOT / "tools" / "models" / "configs" / "bcir-docs-860k.json"
_SCRIPT = _ROOT / "tools" / "models" / "train_rail.sh"
_WITNESS = _ROOT / "tools" / "models" / "train_rail.py"


def _posix_bash() -> str | None:
    from bcir.tests.run_all import posix_bash  # one probe, bounded, for every script-running test

    return posix_bash()


def test_the_corpus_recipe_selects_every_tracked_markdown_file_of_both_trees():
    """The law's row read from the recipe alone: the selection reaches every tracked `*.md` under
    docs/ and training/ and nothing outside them; the recipe is the docs recipe's model, tokenizer
    and licence, so the two differ only in what they read."""
    rail = _tool("train_rail")
    selected = rail.selected_by_recipe(_ROOT, _RECIPE)
    tracked = rail.tracked_markdown(_ROOT)
    assert rail.unselected(_ROOT, selected) == []
    assert set(selected) == set(tracked), sorted(set(selected) ^ set(tracked))[:10]
    for root in rail.ROOTS:
        assert any(path.startswith(root + "/") for path in tracked), f"no {root}/ file is tracked"
    assert tuple(rail.ROOTS) == ("docs", "training")
    corpus = json.loads(_RECIPE.read_text(encoding="utf-8"))
    docs = json.loads(_DOCS_RECIPE.read_text(encoding="utf-8"))
    assert corpus["train"] == docs["train"] and corpus["tokenizer"] == docs["tokenizer"]
    assert corpus["corpus"]["license"] == docs["corpus"]["license"] == "LicenseRef-BCIR-NC-1.0"
    assert corpus["corpus"]["include"] == [
        "docs/*.md",
        "docs/**/*.md",
        "training/*.md",
        "training/**/*.md",
    ]
    assert corpus["corpus"]["split_salt"] != docs["corpus"]["split_salt"], "two corpora, two splits"


def test_the_witness_fires_on_a_narrower_recipe_and_refuses_what_it_cannot_read():
    """RED before GREEN: the docs recipe's selection leaves every training/ file unselected and the
    CLI says so with exit 1; a selection of nothing is refused before any count; a corpus.json that
    is not a file manifest, or is absent, is UNUSABLE (exit 2), never zero (L1, L2); the recipe's
    own selection and a built model's manifest of the same files read the same row."""
    rail = _tool("train_rail")
    tracked = rail.tracked_markdown(_ROOT)
    left = rail.unselected(_ROOT, rail.selected_by_recipe(_ROOT, _DOCS_RECIPE))
    assert left and all(path.startswith("training/") for path in left), left[:5]
    assert len(left) == sum(1 for path in tracked if path.startswith("training/"))
    with tempfile.TemporaryDirectory(prefix="bcir-train-rail-") as tmp:
        recipe = json.loads(_RECIPE.read_text(encoding="utf-8"))
        recipe["corpus"]["include"] = ["docs/no-such-tree/*.md"]
        nothing = Path(tmp) / "nothing.json"
        nothing.write_text(json.dumps(recipe), encoding="utf-8")
        try:
            rail.selected_by_recipe(_ROOT, nothing)
        except ValueError as error:
            assert "selects no tracked file" in str(error)
        else:
            raise AssertionError("a recipe that selects nothing was accepted")
        card = Path(tmp) / "corpus.json"
        card.write_text(json.dumps({"files": [{"path": p} for p in tracked]}), encoding="utf-8")
        assert rail.unselected(_ROOT, rail.selected_by_card(card)) == []
        for bad in ('{"files": "docs/README.md"}', '{"files": [{"bytes": 3}]}', "[]"):
            card.write_text(bad, encoding="utf-8")
            try:
                rail.selected_by_card(card)
            except ValueError as error:
                assert "not a corpus manifest" in str(error), bad
            else:
                raise AssertionError(f"accepted {bad!r} as a corpus manifest")

        def cli(*args: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                [sys.executable, str(_WITNESS), "check", *args],
                capture_output=True,
                cwd=_ROOT,
                timeout=300,
            )

        clean = cli("--recipe", str(_RECIPE))
        assert clean.returncode == 0 and b"unselected=0" in clean.stdout, clean
        narrow = cli("--recipe", str(_DOCS_RECIPE))
        assert narrow.returncode == 1, narrow
        assert f"unselected={len(left)}".encode() in narrow.stdout, narrow.stdout
        assert b"unselected training/" in narrow.stdout
        absent = cli("--corpus-json", str(Path(tmp) / "absent.json"))
        assert absent.returncode == 2 and b"UNUSABLE" in absent.stderr, absent
        card.write_text("[]", encoding="utf-8")
        malformed = cli("--corpus-json", str(card))
        assert malformed.returncode == 2 and b"UNUSABLE" in malformed.stderr, malformed


def test_the_rail_script_refuses_what_it_cannot_run():
    """Exit 2 before any work: no output directory, a step count that is no count, an interpreter
    that is not there, a recipe that is not there -- each a verdict, none a training run."""
    from bcir.tests.run_all import _child_timeout_seconds

    bash = _posix_bash()
    if bash is None:
        return
    bound = _child_timeout_seconds()  # the harness's child bound, not a timing (run_all.posix_bash)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    with tempfile.TemporaryDirectory(prefix="bcir-train-rail-") as tmp:
        for args, extra in (
            ([], {}),
            ([tmp, "ten"], {}),
            ([tmp, "40", "more"], {}),
            ([tmp, "40"], {"PYTHON": str(Path(tmp) / "no-such-python")}),
            ([tmp, "40"], {"BCIR_TRAIN_RECIPE": "tools/models/configs/no-such-recipe.json"}),
        ):
            run = subprocess.run(
                [bash, str(_SCRIPT), *args],
                capture_output=True,
                cwd=_ROOT,
                env={**env, **extra},
                timeout=bound,
            )
            assert run.returncode == 2, (args, extra, run.stdout, run.stderr)
            assert not (Path(tmp) / "corpus.json").exists(), "it trained before refusing"


def test_the_rail_is_one_script_in_every_place_it_runs():
    """One script, one recipe, two roots, one step count, read out of each place (L14): the CTest
    entry and its cache variable, the preset, the BCIR Make task (its reads are what decides the
    model; the gate's scratch copy carries the trees it reads), the two CI jobs, the GEM+ row, the
    law's statement in AGENTS.md and the ladder's row."""
    script = _SCRIPT.read_text(encoding="utf-8")
    assert "tools/models/configs/bcir-corpus-860k.json" in script
    assert "train_rail.py check" in script and "build_lab_model.py" in script
    cmake = (_ROOT / "cmake" / "BCIRTests.cmake").read_text(encoding="utf-8")
    entry = re.search(
        r"add_test\(NAME train-corpus\n(.*?)\n(?=set_tests_properties\(train-corpus)", cmake, re.S
    )
    assert entry, "no train-corpus CTest entry"
    assert "tools/models/train_rail.sh" in entry.group(1) and "${BCIR_TRAIN_STEPS}" in entry.group(
        1
    )
    assert 'set_tests_properties(train-corpus PROPERTIES LABELS "train" PROCESSORS 2' in cmake
    lists = (_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    steps = re.search(r"^set\(BCIR_TRAIN_STEPS (\d+) CACHE STRING", lists, re.M)
    assert steps and 'NOT BCIR_TRAIN_STEPS MATCHES "^[1-9][0-9]*$"' in lists, (
        "no bounded step count"
    )
    presets = json.loads((_ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
    train = [p for p in presets["testPresets"] if p["name"] == "train"]
    assert train and train[0]["filter"] == {"include": {"label": "^train$"}}
    assert train[0]["execution"] == {"jobs": 2}

    spec = importlib.util.spec_from_file_location(
        "bcir_tools_bcirfile", _ROOT / "tools" / "build" / "bcirfile.py"
    )
    bcirfile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bcirfile)
    assert bcirfile.TRAIN_RECIPE == "tools/models/configs/bcir-corpus-860k.json"
    assert bcirfile.TRAIN_SCRIPT == "tools/models/train_rail.sh"
    assert bcirfile.TRAIN_STEPS == int(steps.group(1)), "CTest and BCIR Make train the same smoke"
    assert set(bcirfile.TRAIN_TREES) >= {"docs", "training"}
    assert {"runtime/c/bcir_tensor.c", "runtime/c/bcir_decoder_train.c"} <= set(
        bcirfile.TRAIN_READS
    )
    spec = importlib.util.spec_from_file_location(
        "bcir_tools_make_gate", _ROOT / "tools" / "build" / "make_gate.py"
    )
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    assert set(bcirfile.TRAIN_TREES) <= set(gate.COPIED), (
        "the scratch copy lacks a tree the task reads"
    )
    cc = shutil.which("gcc") or shutil.which("clang") or shutil.which("cc")
    ar = shutil.which("ar")
    bash = _posix_bash()
    if cc and ar and bash:
        from bcir.make import check, parse

        text, planned = bcirfile.generate(
            cc, None, ar, sys.executable, False, bash=bash, path_compilers=bcirfile.path_compilers()
        )
        assert planned["train"] == ["corpus"]
        bf = parse(text.encode("ascii"))
        assert check(bf, _ROOT, check_tools=True) == []
        task = next(t for t in bf.targets if t.name == "train.corpus")
        assert bcirfile.TRAIN_RECIPE in task.reads and set(bcirfile.TRAIN_READS) <= set(task.reads)
        assert set(task.reads_tree) == set(bcirfile.TRAIN_TREES) and set(task.uses) == {
            "cc",
            "python",
        }
        (run,) = task.runs
        assert run[0] == "bash" and run[1] == "tools/build/run_section.sh"
        assert bcirfile.TRAIN_SCRIPT in run and str(bcirfile.TRAIN_STEPS) == run[-1]
        text_without, planned_without = bcirfile.generate(cc, None, ar, sys.executable, False)
        assert planned_without["train"] == [] and "train.corpus" not in text_without

    workflow = (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "bash tools/models/train_rail.sh build/lab-model 200" in workflow, (
        "the qualification job"
    )
    assert "-L train -j 2" in workflow, "the cmake-build job"
    assert "bcir-docs-860k.json" not in workflow, "CI trains the corpus recipe, not the docs one"
    agents = (_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "train_rail.sh" in agents and "bcir-corpus-860k.json" in agents
    roadmap = (_ROOT / "docs" / "research" / "BCIR_TRAINING_ACCELERATION_ROADMAP.md").read_text(
        encoding="utf-8"
    )
    row = re.search(r"^\| \*\*TRAIN-RAIL\*\*.*\| \*\*landed\*\* \|$", roadmap, re.M)
    assert row, "the ladder's TRAIN-RAIL row is not landed"
    recipe = json.loads(_RECIPE.read_text(encoding="utf-8"))
    assert recipe["native"] == {"contract": "kblock+exp+reduce", "threads": "host"}, (
        "the rail trains under the fast contract at the host's threads (TRAIN-POLICY)"
    )
    # the host's threads, except where a CTest entry declares fewer processors than the host has:
    # there the entry's trainer takes what the entry declares (two; the Make gate's task one, its
    # two workers being its two processors)
    assert '"BCIR_TRAIN_THREADS=2"' in entry.group(1), "train-corpus declares PROCESSORS 2"
    make = re.search(
        r"add_test\(NAME build-make\n(.*?)\n\s*set_tests_properties\(build-make ", cmake, re.S
    )
    assert make and '"BCIR_TRAIN_THREADS=1"' in make.group(1), "build-make's train task"
    assert 'set_tests_properties(build-make PROPERTIES LABELS "build;c" PROCESSORS 2' in cmake
    sys.path.insert(0, str(_ROOT / "tools" / "perf"))
    try:
        import gemplus_baseline as harness
    finally:
        sys.path.pop(0)
    metric = next(m for m in harness.METRICS if m.key == "train.corpus.unselected")
    assert metric.kind == "exact" and metric.bound == 0 and metric.slice_owner == "TRAIN-RAIL"
