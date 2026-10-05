"""bcir-opt's lit suite (BUILD-5, docs/BCIR_BUILD_ROADMAP.md §9): `mlir/lit.cfg.py` runs each
fixture's RUN lines as written, against the bcir-opt a build made and the FileCheck and mlir-opt of
its LLVM. These tests hold the suite to the repository and to its own refusals, without lit or a
compiled bcir-opt:

- it runs every .mlir under mlir/ that carries a RUN line, and only those: a RUN line in a file the
  suite excludes is a contract nothing executes, and a file it selects with no RUN line is one lit
  reports UNRESOLVED (laws.md L15); each name `NOT_LIT` excludes names exactly one file, so an
  exclusion by name never hides a second (lit's `excludes` match at any depth);
- every command a RUN line starts is a tool the suite pins by path; anything else would come from
  the host's PATH, where another LLVM major can sit;
- the substitution replaces exactly those command words, never an option (`-bcir-optimize`) or a
  path segment;
- a missing exec_root, a missing or non-executable tool, a tool with no LLVM version, and a mixed
  major are each a lit fatal, never a smaller suite or a mixed toolset (L1, L2);
- the config's test format survives the pickling lit's worker pool puts every test through: a
  format class defined in the config itself does not (it is not importable by name in a worker),
  and the suite died before its first test on the first run that used two workers (L11: the
  witness pickles the format, and fails on a config that defines its own);
- the CI job that installs the tools requires the suite and runs it, as the local check does (L2).
"""

from __future__ import annotations

import ast
import json
import os
import pickle
import re
import sys
import tempfile
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_MLIR = _ROOT / "mlir"
_CFG = _MLIR / "lit.cfg.py"


def _literals() -> dict:
    """NOT_LIT and TOOLS as the config spells them."""
    found = {}
    for node in ast.parse(_CFG.read_text(encoding="utf-8")).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ("NOT_LIT", "TOOLS")
        ):
            found[node.targets[0].id] = ast.literal_eval(node.value)
    assert set(found) == {"NOT_LIT", "TOOLS"}, sorted(found)
    return found


def run_lines(text: str) -> list[str]:
    """What follows `RUN:` on each line that has it -- lit's own reading -- with a trailing `\\`
    joining the next RUN line."""
    lines: list[str] = []
    pending = ""
    for raw in text.splitlines():
        at = raw.find("RUN:")
        if at < 0:
            continue
        body = raw[at + 4 :].strip()
        if body.endswith("\\"):
            pending += body[:-1] + " "
            continue
        lines.append(pending + body)
        pending = ""
    return lines


def commands(line: str) -> list[str]:
    """The command word of each stage of a RUN line's pipeline."""
    words = []
    for stage in re.split(r"\|\||&&|\||;", line):
        tokens = stage.split()
        while tokens and tokens[0] == "not":
            tokens = tokens[1:]
        if tokens:
            words.append(tokens[0])
    return words


def _mlir_files() -> dict[tuple[str, ...], Path]:
    return {
        path.relative_to(_MLIR).parts: path
        for path in sorted(_MLIR.rglob("*.mlir"))
        if path.is_file()
    }


def _selected(files, not_lit) -> set:
    """What lit selects: every file of the suffix whose path has no excluded name (`excludes`
    matches a file or directory name at any depth)."""
    return {rel for rel in files if not set(rel) & set(not_lit)}


def test_the_suite_runs_every_run_line_under_mlir_and_only_those():
    names = _literals()
    files = _mlir_files()
    selected = _selected(files, names["NOT_LIT"])
    with_run = {rel for rel, path in files.items() if run_lines(path.read_text(encoding="utf-8"))}
    assert len(selected) > 100, "the fixture directories moved"
    outside = sorted("/".join(rel) for rel in with_run - selected)
    silent = sorted("/".join(rel) for rel in selected - with_run)
    assert outside == [], f"RUN lines the suite does not run: {outside}"
    assert silent == [], f"suite files with no RUN line (lit: UNRESOLVED): {silent}"
    directories = {d.name for d in _MLIR.rglob("*") if d.is_dir()}
    for name in names["NOT_LIT"]:
        hits = [rel for rel in files if rel[-1] == name]
        assert len(hits) == 1, f"NOT_LIT names {name}, which names {len(hits)} files under mlir/"
        assert name not in directories, f"NOT_LIT's {name} also names a directory under mlir/"
        assert not run_lines(files[hits[0]].read_text(encoding="utf-8")), name


def test_every_command_a_run_line_starts_is_a_pinned_tool():
    pinned = set(_literals()["TOOLS"].values())
    seen: set[str] = set()
    for rel, path in _mlir_files().items():
        for line in run_lines(path.read_text(encoding="utf-8")):
            for word in commands(line):
                assert word in pinned, (
                    f"{'/'.join(rel)}: RUN line calls {word!r}, not a pinned tool"
                )
                seen.add(word)
    assert seen == pinned, f"pinned tools no RUN line calls: {sorted(pinned - seen)}"


# --- the config itself, executed over stubs of lit's API -----------------------------------------


class _Fatal(Exception):
    pass


class _LitConfig:
    def __init__(self, params: dict[str, str]) -> None:
        self.params = params
        self.notes: list[str] = []

    def fatal(self, message: str):
        raise _Fatal(message)

    def note(self, message: str) -> None:
        self.notes.append(message)


class ShTest:
    """lit.formats.ShTest's discovery (lit/formats/base.py FileBasedTest): every file of a configured
    suffix that is not a dot file and not in `excludes`. Registered as lit.formats.ShTest, so it
    pickles by that name as the real one does."""

    def __init__(self, execute_external=False):
        self.execute_external = execute_external

    def getTestsInDirectory(self, testSuite, path_in_suite, litConfig, localConfig):
        source = testSuite.getSourcePath(path_in_suite)
        for name in sorted(os.listdir(source)):
            if name.startswith(".") or name in localConfig.excludes:
                continue
            if not os.path.isdir(os.path.join(source, name)) and (
                os.path.splitext(name)[1] in localConfig.suffixes
            ):
                yield types.SimpleNamespace(path_in_suite=tuple(path_in_suite) + (name,))


ShTest.__module__ = "lit.formats"
ShTest.__qualname__ = "ShTest"


def _stub_lit() -> dict[str, types.ModuleType]:
    """lit, lit.formats (ShTest above) and lit.TestingConfig, the module lit executes a config in."""
    lit = types.ModuleType("lit")
    formats = types.ModuleType("lit.formats")
    formats.ShTest = ShTest
    lit.formats = formats
    testing_config = types.ModuleType("lit.TestingConfig")
    lit.TestingConfig = testing_config
    return {"lit": lit, "lit.formats": formats, "lit.TestingConfig": testing_config}


def _discover(config, lit_config) -> set:
    """lit's discovery (lit/discovery.py): every directory below the suite root but `Output`,
    `.svn`, `.git` and the excluded names, each listed by the config's test format."""
    suite = types.SimpleNamespace(getSourcePath=lambda parts: str(_MLIR.joinpath(*parts)))
    found: set = set()
    todo = [()]
    while todo:
        rel = todo.pop()
        found |= {
            test.path_in_suite
            for test in config.test_format.getTestsInDirectory(suite, rel, lit_config, config)
        }
        here = _MLIR.joinpath(*rel)
        for name in sorted(os.listdir(here)):
            if name in ("Output", ".svn", ".git") or name in config.excludes:
                continue
            if (here / name).is_dir():
                todo.append(rel + (name,))
    return found


def _load(params: dict[str, str], text: str | None = None):
    """Execute mlir/lit.cfg.py (or ``text``) as lit would, over the stubs: in the namespace of
    lit.TestingConfig, so a class the config defines belongs to that module, as under lit. The
    config, the lit_config, and the pickled test format -- or the exception pickling raised, which
    is what lit's worker pool would raise before the first test."""
    saved = {name: sys.modules.get(name) for name in ("lit", "lit.formats", "lit.TestingConfig")}
    sys.modules.update(_stub_lit())
    try:
        config = types.SimpleNamespace(substitutions=[], environment={}, excludes=set())
        lit_config = _LitConfig(params)
        source = _CFG.read_text(encoding="utf-8") if text is None else text
        code = compile(source, str(_CFG), "exec")
        exec(code, {"__file__": str(_CFG), "__name__": "lit.TestingConfig", "config": config,
                    "lit_config": lit_config})  # fmt: skip
        try:
            pickled = pickle.dumps(config)
        except (pickle.PicklingError, AttributeError, TypeError) as exc:
            pickled = exc
        return config, lit_config, pickled
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _tool(directory: Path, name: str, banner: str) -> str:
    path = directory / name
    path.write_text(f"#!/bin/sh\necho 'LLVM (http://llvm.org/):'\necho '  {banner}'\n")
    path.chmod(0o755)
    return str(path)


def _tools(directory: Path, majors=(23, 23, 23)) -> dict[str, str]:
    names = _literals()["TOOLS"]
    return {
        param: _tool(directory, name, f"LLVM version {major}.1.2")
        for (param, name), major in zip(names.items(), majors)
    }


def _posix() -> bool:
    # The fakes are POSIX shell scripts; the suite itself runs where bcir-opt builds, never on a
    # Windows runner, which is where this returns False.
    return os.name == "posix" and Path("/bin/sh").exists()


def test_the_config_selects_exactly_the_suite():
    if not _posix():
        return
    names = _literals()
    with tempfile.TemporaryDirectory() as tmp:
        params = {**_tools(Path(tmp)), "exec_root": str(Path(tmp) / "exec")}
        config, lit_config, pickled = _load(params)
        assert config.test_source_root == str(_MLIR)
        assert config.test_exec_root == str(Path(tmp) / "exec")
        # lit's own shell: the RUN lines mean the same on every host, whatever its /bin/sh is
        assert type(config.test_format) is ShTest and config.test_format.execute_external is False
        assert config.suffixes == [".mlir"] and config.excludes == set(names["NOT_LIT"])
        found = _discover(config, lit_config)
        expected = _selected(_mlir_files(), names["NOT_LIT"])
        assert found == expected, sorted(found ^ expected)
        assert any("LLVM 23" in note for note in lit_config.notes), lit_config.notes
        assert isinstance(pickled, bytes), f"lit cannot hand this config to a worker: {pickled!r}"


def test_a_format_defined_in_the_config_cannot_reach_a_worker():
    """The witness for the defect the first MLIR 23 run found: a test format class defined in the
    config is not importable by name where lit's worker unpickles it, so the pickling fails."""
    if not _posix():
        return
    with tempfile.TemporaryDirectory() as tmp:
        params = {**_tools(Path(tmp)), "exec_root": str(Path(tmp) / "exec")}
        own = _CFG.read_text(encoding="utf-8") + (
            "\n\nclass _Fixtures(lit.formats.ShTest):\n    pass\n\n\n"
            "config.test_format = _Fixtures(execute_external=False)\n"
        )
        _config, _lit_config, pickled = _load(params, own)
        assert not isinstance(pickled, bytes), (
            "a config-defined format pickled: the witness is blind"
        )


def test_the_substitution_replaces_only_command_words():
    if not _posix():
        return
    with tempfile.TemporaryDirectory() as tmp:
        tools = _tools(Path(tmp))
        config, _, _ = _load({**tools, "exec_root": tmp})
        by_name = {name: tools[param] for param, name in _literals()["TOOLS"].items()}

        def substitute(line: str) -> str:
            for pattern, replacement in config.substitutions:
                line = re.sub(pattern, replacement.replace("\\", "\\\\"), line)
            return line

        lines = 0
        for path in _mlir_files().values():
            for line in run_lines(path.read_text(encoding="utf-8")):
                done = substitute(line)
                words = commands(done)
                assert words and all(w in by_name.values() for w in words), (line, done)
                lines += 1
        assert lines > 100
        bo, fc, mo = by_name["bcir-opt"], by_name["FileCheck"], by_name["mlir-opt"]
        cases = {
            "bcir-opt -bcir-optimize %s | FileCheck %s --check-prefix=CHECK-FileCheck": (
                f"{bo} -bcir-optimize %s | {fc} %s --check-prefix=CHECK-FileCheck"
            ),
            "mlir-opt --irdl-file=%S/../../irdl/bcir.irdl.mlir %s": (
                f"{mo} --irdl-file=%S/../../irdl/bcir.irdl.mlir %s"
            ),
            "bcir-opt %s | bcir-opt | FileCheck %s": f"{bo} %s | {bo} | {fc} %s",
            "bcir-opt /opt/bcir-opt/x.mlir FileCheck-23 bcir-opt.sh my-mlir-opt": (
                f"{bo} /opt/bcir-opt/x.mlir FileCheck-23 bcir-opt.sh my-mlir-opt"
            ),
        }
        for line, want in cases.items():
            assert substitute(line) == want, (line, substitute(line))


def test_the_config_refuses_what_it_cannot_run():
    if not _posix():
        return
    with tempfile.TemporaryDirectory() as tmp:
        here = Path(tmp)
        good = {**_tools(here), "exec_root": str(here / "exec")}
        _load(good)  # the honest set loads

        def refused(params: dict[str, str], why: str) -> None:
            try:
                _load(params)
            except _Fatal as fatal:
                assert why in str(fatal), (why, str(fatal))
            else:
                raise AssertionError(f"accepted: {why}")

        refused({k: v for k, v in good.items() if k != "exec_root"}, "exec_root")
        for param in _literals()["TOOLS"]:
            refused({k: v for k, v in good.items() if k != param}, f"--param {param}=PATH")
            refused({**good, param: str(here / "absent")}, "is not an executable file")
        plain = here / "plain"
        plain.mkdir()
        unexecutable = _tool(plain, "FileCheck", "LLVM version 23.1.2")
        os.chmod(unexecutable, 0o644)
        refused({**good, "filecheck": unexecutable}, "is not an executable file")
        mute = here / "mute"
        mute.mkdir()
        refused({**good, "mlir_opt": _tool(mute, "mlir-opt", "no version here")}, "no LLVM version")
        mixed = here / "mixed"
        mixed.mkdir()
        refused({**_tools(mixed, (23, 22, 23)), "exec_root": tmp}, "not one LLVM major")


def test_ci_and_the_local_check_require_and_run_the_suite():
    tests = (_ROOT / "cmake" / "BCIRTests.cmake").read_text(encoding="utf-8")
    assert "add_test(NAME mlir-lit" in tests and "exec_root=${CMAKE_BINARY_DIR}/mlir-lit" in tests
    for param in _literals()["TOOLS"]:
        assert f'--param "{param}=' in tests, param
    assert 'message(FATAL_ERROR "BCIR_REQUIRE_LIT=ON' in tests
    workflow = (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    job = re.search(r"^  mlir-rail-validate:\n(.*?)(?=^  [a-z0-9-]+:\n|\Z)", workflow, re.S | re.M)
    assert job is not None, "no mlir-rail-validate job"
    body = job.group(1)
    assert "cmake --preset mlir" in body and "-DBCIR_REQUIRE_LIT=ON" in body
    assert "ctest --preset mlir" in body, "the job runs no `ctest --preset mlir`"
    presets = json.loads((_ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
    tests = {p["name"]: p for p in presets["testPresets"]}
    assert tests["mlir"]["configurePreset"] == "mlir"
    assert tests["mlir"]["filter"] == {"include": {"label": "^mlir$"}}
    local = (_ROOT / "tools" / "local" / "check_latest.sh").read_text(encoding="utf-8")
    assert "-DBCIR_REQUIRE_LIT=ON" in local
