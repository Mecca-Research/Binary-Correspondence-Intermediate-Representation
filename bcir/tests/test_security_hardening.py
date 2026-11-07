"""Deterministic regressions for local/CI privilege and staging boundaries."""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from bcir.channel_plugin import load_manifest


_ROOT = Path(__file__).resolve().parents[2]


def test_checkout_tools_package_wins_over_an_installed_namesake() -> None:
    """A generic installed ``tools`` package must not shadow this checkout's tools tree."""
    with tempfile.TemporaryDirectory() as td:
        hostile = Path(td) / "tools"
        hostile.mkdir()
        (hostile / "__init__.py").write_text("ORIGIN = 'hostile'\n", encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = td
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "from tools.models import fetch_pinned_model; print(fetch_pinned_model.__file__)",
            ],
            cwd=_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert probe.returncode == 0, probe.stderr
        assert str(_ROOT / "tools/models/fetch_pinned_model.py") in probe.stdout


def test_mlir_bootstrap_never_executes_an_unverified_existing_file() -> None:
    if os.name == "nt" or shutil.which("bash") is None:
        return
    # Keep the probe on the native POSIX filesystem.  WSL may inherit a Windows
    # TEMP path whose translated permission bits are intentionally non-private.
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        private = Path(td) / "private"
        private.mkdir(mode=0o700)
        marker = private / "executed"
        micromamba = private / "micromamba"
        micromamba.write_text(
            '#!/bin/sh\nprintf reached > "$BCIR_BOOTSTRAP_MARKER"\n',
            encoding="utf-8",
        )
        micromamba.chmod(0o700)
        env = dict(os.environ)
        env.update(
            {
                "MICROMAMBA": str(micromamba),
                "MAMBA_ROOT_PREFIX": str(private / "mamba"),
                "BCIR_BOOTSTRAP_MARKER": str(marker),
            }
        )
        result = subprocess.run(
            ["bash", str(_ROOT / "tools/local/setup_mlir.sh")],
            cwd=_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0
        assert "unexpected SHA-256" in result.stderr
        assert not marker.exists(), "an unverified prepositioned bootstrap was executed"


def test_validation_scripts_use_operation_private_tempdirs() -> None:
    scripts = (
        "tools/wsl/check_passes.sh",
        "tools/wsl/check_bytecode.sh",
        "tools/wsl/check_ods_examples.sh",
        "tools/irdl/check_corpus.sh",
        "tools/local/check_rail.sh",
    )
    fixed_names = re.compile(r"/tmp/(?:pe|bce|bc\.mlirbc|ods_err|irdl_err|rail22_|rail_)")
    for relative in scripts:
        text = (_ROOT / relative).read_text(encoding="utf-8")
        assert "mktemp -d" in text, relative
        assert not fixed_names.search(text), relative


def test_no_gate_pipes_a_writer_into_an_early_exiting_reader_under_pipefail() -> None:
    """A gate that reports a MATCH as a failure is worse than no gate.

    `printf '%s' "$blob" | grep -q PATTERN` looks exact and is a trap. `grep -q` exits at the
    first match, the writer then takes SIGPIPE, and under `set -o pipefail` the pipeline's
    status becomes 141 — so *finding* the pattern fails the check. `head` does the same to a
    writer with more to say, and inside `VAR="$(writer | head -1)"` under `set -e` the 141
    ends the script.

    It fired twice. On `#jerindex`, the moment per-tier instantiation made a disassembly exceed
    the 64 KiB pipe buffer, it reported "the tier fell back to scalar" about an object that
    contained four NEON instructions. And the buffer was never the whole condition: bash
    line-buffers its builtins' output, so `printf '%s' "$multi_line"` writes once per line, and
    a reader that matches an early line can exit before the next write -- a race no buffer size
    closes. A quiet host almost never shows its window; when the section-parity gate began
    running sections side by side, the `recover` section's CMake run failed with both of its
    patterns in the output it printed (2026-11-07).

    So no script under `tools/` that sets `pipefail` ends a pipeline in a reader that stops
    early -- `grep -q`, `-m`, `-l` or their long spellings, or `head`. Reading every line
    (`grep PATTERN >/dev/null`, `sed -n '1,10p'`) keeps the status (a match is 0, none is 1, a
    failing writer still fails the pipeline) and closes the race whatever the writer. The same
    two-write pipeline, with a pause between its writes, shows both halves here wherever a
    POSIX bash is available. A script without `pipefail` is out of scope: there the pipeline's
    status is its last command's, and a writer's SIGPIPE cannot reach it.
    """
    reader = re.compile(
        r"(?<!\|)\|(?!\|)\s*(?:grep\s+(?:-[a-zA-Z]*[qml][a-zA-Z0-9]*\b|--(?:quiet|silent|max-count|"
        r"files-with-matches)\b)|head\b)"
    )
    scanned: list[str] = []
    full_readers = 0
    for script in sorted((_ROOT / "tools").rglob("*.sh")):
        text = script.read_text(encoding="utf-8")
        if "pipefail" not in text:
            continue
        scanned.append(script.relative_to(_ROOT).as_posix())
        for number, line in enumerate(text.splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            assert not reader.search(line), (
                f"{script.relative_to(_ROOT)}:{number} ends a pipeline in a reader that exits "
                f"early under `pipefail`; a match can SIGPIPE the writer and be reported as a "
                f"failure. Read every line (grep ... >/dev/null, sed -n '1,Np').\n  {line.strip()}"
            )
            full_readers += len(re.findall(r"\|\s*grep [^|]*>/dev/null", line))
    sections = [s for s in scanned if s.startswith("tools/c/sections/")]
    assert len(sections) >= 100 and len(scanned) - len(sections) >= 15, (
        "the scan no longer reaches the gate scripts",
        len(sections),
        len(scanned),
    )
    simd = [s for s in scanned if "objdump" in (_ROOT / s).read_text(encoding="utf-8")]
    assert len(simd) >= 2, f"expected the two SIMD gates to disassemble; found {simd}"
    assert full_readers >= 50, (
        "the gates' pattern checks are not where this test looks",
        full_readers,
    )
    bash = shutil.which("bash")
    if os.name != "posix" or bash is None:
        return
    writer = '{ printf "21\\n"; sleep 0.2; printf "more\\n"; }'
    early = subprocess.run(
        [bash, "-c", f'set -o pipefail; {writer} | grep -q "^21$"'], capture_output=True, timeout=30
    )
    full = subprocess.run(
        [bash, "-c", f'set -o pipefail; {writer} | grep "^21$" >/dev/null'],
        capture_output=True,
        timeout=30,
    )
    assert early.returncode == 141, ("the early reader no longer fails a found match", early)
    assert full.returncode == 0, ("reading every line does not pass a found match", full)


def test_no_workflow_step_pipes_a_network_fetch_into_another_command() -> None:
    """A pipeline's status is its last command's, so a piped fetch cannot fail.

    Both LLVM jobs installed apt.llvm.org's signing key with

        wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key \
          | sudo tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc >/dev/null

    and `tee` succeeds on an empty stream. A failed fetch therefore wrote an empty
    keyring, the step passed, and `apt-get update` twenty lines later reported

        NO_PUBKEY 15CF4D18AF4F7421
        E: The repository '...llvm-toolchain-noble-22 InRelease' is not signed.

    which reads as an upstream signing problem and was this workflow's own swallowed
    error -- the same step had passed fifteen minutes earlier on the previous commit,
    with the same key. `set -o pipefail` is not available here either: a GitHub `run:`
    block is `bash -e`, not `bash -eo pipefail`.

    So the shape is refused. Fetch to a file, check the file, then use it -- which is
    what `tools/ci/add_llvm_apt_repo.sh` does for both jobs (L14: the copies had
    already begun to matter, because the defect was in both and surfaced in one).

    **Continuations are joined first, and that is the whole point.** The defect this
    exists to refuse spans two lines -- `wget ... \\` and then `| sudo tee ...` -- so
    a per-line regex would let the exact original code straight back in. Scanning
    line by line was this witness's own version of the mistake it was written to
    catch: a check that passes on the thing it names (L11). Shell continuations are
    folded into one logical line before matching, and the reported number is where
    that logical line begins.

    Declared scope: a literal `curl` or `wget` followed by a `|` in the same logical
    shell line of a workflow file. A fetch assembled at run time, or routed through
    a variable, is out of scope and belongs to a shell linter; inside the scope this
    is exact and does not grow.
    """
    offender = re.compile(r"\b(?:curl|wget)\b[^|#\n]*\|")
    workflows = sorted((_ROOT / ".github" / "workflows").glob("*.yml"))
    assert len(workflows) >= 1, "no workflow files found; this check would pass vacuously"
    scanned = joined = 0
    for workflow in workflows:
        logical: list[tuple[int, str]] = []
        pending, start = "", 0
        for number, line in enumerate(workflow.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            scanned += 1
            if not pending:
                start = number
            if stripped.endswith("\\"):
                pending += stripped[:-1] + " "
                continue
            logical.append((start, pending + stripped))
            if pending:
                joined += 1
            pending = ""
        if pending:
            logical.append((start, pending))
        for number, line in logical:
            assert not offender.search(line), (
                f"{workflow.relative_to(_ROOT)}:{number} pipes a network fetch into "
                "another command, so the fetch's exit status is discarded and a "
                "failed download becomes a confusing error further down. Fetch to a "
                "file, check it, then use it.\n  " + line
            )
    assert scanned >= 200, f"only {scanned} workflow line(s) scanned; the walk is not covering them"
    assert joined >= 1, (
        "no shell continuation was folded, so the two-line shape this check exists "
        "for is not being formed and the scan is per-line after all"
    )


def test_no_source_hand_declares_a_libc_function() -> None:
    """The #699 bug, refused by shape rather than caught on a phone.

    A native bench needed a prototype for `syscall`, which glibc hides behind a feature macro,
    so it declared `syscall` and `ioctl` itself. bionic types ioctl's request parameter as
    `int` where glibc uses `unsigned long`, so the local prototype matched one libc and the
    compiler on the other was right to reject it — and the whole file failed to build under
    Termux, aborting both aarch64 calibration runs.

    Only a libc knows its own signatures. A source that needs a prototype takes it from the
    system header. This is enforced here as well as in `#targetabi` because the suite runs in
    places the shell gate may not, and it is the one part of that failure a machine with no
    bionic sysroot can detect at all.
    """
    declaration = re.compile(r"^\s*extern\s+[a-zA-Z_][\w \t*]*\([^;]*\)\s*;")
    offenders = []
    for pattern in ("runtime/c/*.c", "runtime/c/*.h", "runtime/cpp/*.cpp", "runtime/cpp/*.h"):
        for path in sorted(_ROOT.glob(pattern)):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if 'extern "C"' in line or line.strip().startswith(("*", "//")):
                    continue
                if declaration.match(line):
                    offenders.append(f"{path.relative_to(_ROOT)}:{number}: {line.strip()}")
    assert not offenders, (
        "a source declares a function the system header should declare; only that libc knows "
        "its own signatures (see #699):\n  " + "\n  ".join(offenders)
    )


def test_workflow_dependencies_are_sha_pinned_and_tokens_are_read_only() -> None:
    sha = re.compile(r"^[0-9a-f]{40}$")
    for workflow in sorted((_ROOT / ".github/workflows").glob("*.yml")):
        text = workflow.read_text(encoding="utf-8")
        assert "pull_request_target:" not in text
        assert re.search(r"(?m)^permissions:\n  contents: read$", text), workflow.name
        dependencies = re.findall(r"(?m)^\s*uses:\s*([^@\s]+)@([^\s#]+)", text)
        assert dependencies, workflow.name
        for name, revision in dependencies:
            assert sha.fullmatch(revision), f"{workflow.name}: {name}@{revision} is mutable"


def test_ci_runs_once_per_change() -> None:
    """The main workflow runs once per change: every pull request (its merge with main), every
    commit main lands, and a branch without a pull request by hand. A push to every branch ran the
    whole workflow twice for a branch with a pull request -- push and pull_request, 36 jobs each --
    and the second copy queued behind the first (2026-11-07). Holding the triggers here keeps both
    halves: dropping `pull_request` would leave pull requests ungated, and widening `push` again
    would bring the doubled runs back."""
    text = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    block = re.search(r"(?m)^on:\n((?:[ #].*\n|\n)*)", text)
    assert block is not None, "ci.yml names its triggers in a block under `on:`"
    lines = [
        line
        for line in block.group(1).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert lines == [
        "  push:",
        "    branches: [main]",
        "  pull_request:",
        "  workflow_dispatch:",
    ], lines


def test_every_linux_runner_names_its_ubuntu_release() -> None:
    """CI's default toolchain -- Clang 18, GCC 13, Python 3.12, the baseline AGENTS.md judges every
    change against beside the newest releases -- is the one Ubuntu 24.04 ships. `ubuntu-latest`
    names whatever release GitHub promotes next: on 2026-11-01 it became 26.04, whose GCC reports
    the C23 `__STDC_VERSION__` Clang does (the preprocessor differential's impl-defined pin stopped
    diverging) and whose archive has no LLVM 18 (the corpus baseline's `--require-baseline`), and
    every job went red with no change to the tree. A runner's release changes only by an edit here,
    with its baseline moved alongside. Windows stays on `windows-latest`."""
    linux = re.compile(r"ubuntu-[0-9][0-9.]*[0-9](?:-arm)?")
    for workflow in sorted((_ROOT / ".github/workflows").glob("*.yml")):
        text = workflow.read_text(encoding="utf-8")
        assert "ubuntu-latest" not in text, f"{workflow.name} runs on an unnamed Ubuntu release"
        labels = re.findall(r"(?m)^\s*runs-on:\s*(\S+)", text)
        labels += [
            label.strip()
            for row in re.findall(r"(?m)^\s*os:\s*\[([^\]]*)\]", text)
            for label in row.split(",")
        ]
        assert labels, workflow.name
        for label in labels:
            if label.startswith("${{") or label.startswith("windows-"):
                continue
            assert linux.fullmatch(label), f"{workflow.name}: runner {label!r}"
            assert label.split("-")[1] == "24.04", (
                f"{workflow.name}: {label} is not the baseline's 24.04"
            )


def test_python_style_gate_and_hooks_run_one_pinned_ruff() -> None:
    """The formatter's output is version-specific, so the pre-commit hooks, the CI style job
    and the `dev` extra must name ONE ruff. A drifting trio is how a tree stops being
    format-clean without anyone changing a policy, and then the hook rewrites whole files
    into unrelated changes (the 2026-09-04 finding behind the one-time reformat)."""
    import tomllib

    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pins = [
        entry
        for entry in pyproject["project"]["optional-dependencies"]["dev"]
        if entry.startswith("ruff")
    ]
    assert len(pins) == 1 and re.fullmatch(r"ruff==\d+\.\d+\.\d+", pins[0]), pins
    # One scope for both: the hooks pass python/pyi files only, so CI's tree walk must be
    # pinned to the same types -- ruff 0.16 otherwise formats Python fences inside Markdown,
    # which the hooks never see (the first run of the CI job failed on four .md files).
    assert pyproject["tool"]["ruff"]["include"] == ["*.py", "*.pyi"]

    hooks = (_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "entry: ruff check --force-exclude --fix" in hooks
    assert "entry: ruff format --force-exclude" in hooks

    ci = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    job = re.search(r"(?ms)^  python-style:\n(.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)", ci)
    assert job, "ci.yml has no python-style job"
    body = job.group(1)
    assert "python -m ruff format --check --force-exclude ." in body
    assert "python -m ruff check --force-exclude ." in body
    # The job installs the pin it READS from pyproject.toml; a second spelling of the pin in
    # the workflow is the drift this test exists to refuse.
    assert 'startswith("ruff==")' in body
    assert "ruff==" not in body.replace('startswith("ruff==")', "")


def test_the_style_gate_and_the_hooks_spare_the_same_frozen_bundles() -> None:
    """The research bundles under docs/research are frozen evidence: each one's protocol and
    manifest hash its files' bytes, so the style gate's ruff and the byte-rewriting hooks
    (trailing whitespace, end of file, line endings) must leave them as uploaded -- restyling one
    breaks the reproduction it exists for (GEO-2 reran the PR818 study from those bytes; three of
    its files had already gained a trailing newline from a web rename). All four spare the same
    roots, each named on its own (no glob: a new bundle is a decision), and where the tree holds
    one it is a bundle: its manifest -- MANIFEST_SHA256.json, SOURCE_MANIFEST.json or
    manifest.json, the three spellings the bundles use -- is there to hold its bytes to."""
    import tomllib

    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    excluded = sorted(
        p for p in pyproject["tool"]["ruff"]["extend-exclude"] if p.startswith("docs/research/")
    )
    assert excluded and all(re.fullmatch(r"docs/research/[a-z0-9-]+", p) for p in excluded), (
        excluded
    )
    names = sorted(p.rsplit("/", 1)[1] for p in excluded)
    hooks = (_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    for hook in ("trailing-whitespace", "end-of-file-fixer", "mixed-line-ending"):
        start = hooks.index(f"- id: {hook}\n")
        end = hooks.find("- id:", start + 1)
        block = hooks[start : end if end != -1 else len(hooks)]
        spared = re.search(r"\^docs/research/\(([a-z0-9|-]+)\)/", block)
        assert spared, f"the {hook} hook does not spare the frozen bundles"
        assert sorted(spared.group(1).split("|")) == names, (hook, spared.group(1), names)
    manifests = ("MANIFEST_SHA256.json", "SOURCE_MANIFEST.json", "manifest.json")
    for p in excluded:
        if (_ROOT / p).exists():
            assert any((_ROOT / p / m).is_file() for m in manifests), f"{p} is not a frozen bundle"


def test_build_system_excludes_known_vulnerable_setuptools_releases() -> None:
    """The isolated PEP 517 backend must exclude GHSA-h35f-9h28-mq5c and GHSA-xmj3-9gf7-h52w."""
    text = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = re.search(r"(?ms)^\[build-system\]\s*$\n(?P<body>.*?)(?=^\[|\Z)", text)
    assert section is not None, "pyproject.toml has no [build-system] table"
    declaration = re.search(r"(?m)^requires\s*=\s*(\[[^\n]+\])\s*$", section.group("body"))
    assert declaration is not None, "[build-system] has no one-line requires declaration"
    requirements = ast.literal_eval(declaration.group(1))
    setuptools = [item for item in requirements if item.startswith("setuptools")]
    assert len(setuptools) == 1, f"expected one setuptools requirement, found {setuptools!r}"
    floor_match = re.fullmatch(r"setuptools>=(\d+)\.(\d+)\.(\d+)", setuptools[0])
    assert floor_match is not None, f"unsupported setuptools requirement: {setuptools[0]!r}"
    floor = tuple(int(part) for part in floor_match.groups())
    assert floor >= (84, 1, 0), (
        f"setuptools floor {floor!r} permits releases affected by "
        "GHSA-h35f-9h28-mq5c / CVE-2026-59890 (fixed in 83.0.0) or GHSA-xmj3-9gf7-h52w "
        "(fixed in 84.1.0); require >=84.1.0"
    )


def test_deeply_nested_plugin_json_fails_as_a_boundary_value_error() -> None:
    """Host recursion limits must not escape an artifact parser as an internal error."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "deep.channel.json"
        path.write_text("[" * 2000 + "]" * 2000, encoding="utf-8")
        try:
            load_manifest(str(path))
            raise AssertionError("deeply nested plugin JSON was accepted")
        except ValueError:
            pass
