"""The newest-toolchain policy: every change is judged on the newest LLVM/Clang/MLIR 23 and Node 24
as well as on the CI-default Clang 18 / GCC 13 (.claude/skills/bcir-latest-toolchain).

`tools/local/latest_toolchain.py` decides what "newest" and "installed" mean and fetches Node 24;
`tools/local/check_latest.sh` runs the gates on them; the `clang23` preset and the cmake-build
`clang23` cell carry the same check into CI. These tests hold the tool's decisions offline (every
upstream source is mocked: an optional network must not change a unit test's verdict, laws.md L19),
prove its refusals fire (L2), and hold the skill, the agent entry points, the presets and the
workflow to one statement of the rule (L14).
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SKILL = _ROOT / ".claude" / "skills" / "bcir-latest-toolchain" / "SKILL.md"
_SCRIPT = _ROOT / "tools" / "local" / "check_latest.sh"
_WORKFLOW = _ROOT / ".github" / "workflows" / "ci.yml"


def _job(workflow: str, name: str) -> str:
    """One job's text out of ci.yml: from its key to the next top-level job key."""
    found = re.search(rf"^  {name}:\n(.*?)(?=^  [a-z0-9-]+:\n|\Z)", workflow, re.S | re.M)
    assert found, f"no {name} job"
    return found.group(1)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tool = _load(_ROOT / "tools" / "local" / "latest_toolchain.py", "bcir_tools_latest_toolchain")


def test_the_newest_llvm_23_is_read_from_the_release_tags():
    """Release tags only, compared as numbers: `-rc`, `-init`, peeled `^{}` lines and other majors
    (`llvmorg-230.*` included) are not releases of 23, and 23.1.10 is newer than 23.1.2."""
    listing = "\n".join(
        f"0123abcd\t{ref}"
        for ref in (
            "refs/tags/llvmorg-23-init",
            "refs/tags/llvmorg-23.1.0-rc3",
            "refs/tags/llvmorg-23.1.0",
            "refs/tags/llvmorg-23.1.2",
            "refs/tags/llvmorg-23.1.2^{}",
            "refs/tags/llvmorg-23.1.10",
            "refs/tags/llvmorg-22.1.8",
            "refs/tags/llvmorg-230.1.0",
        )
    )
    assert tool.newest_llvm_from_tags(listing) == (23, 1, 10)
    assert tool.newest_llvm_from_tags("0123abcd\trefs/tags/llvmorg-23.1.0-rc1\n") is None
    assert tool.newest_llvm_from_tags("") is None


def test_the_newest_python_3_15_is_read_from_the_release_tags():
    """CPython's release tags are `v3.15.<patch>`; `a`, `b` and `rc` suffixes, other series
    (3.14, 3.150) and peeled lines are not releases of 3.15, and 3.15.10 is newer than 3.15.2."""
    listing = "\n".join(
        f"0123abcd\t{ref}"
        for ref in (
            "refs/tags/v3.15.0a1",
            "refs/tags/v3.15.0b4",
            "refs/tags/v3.15.0rc3",
            "refs/tags/v3.15.0",
            "refs/tags/v3.15.2",
            "refs/tags/v3.15.2^{}",
            "refs/tags/v3.15.10",
            "refs/tags/v3.14.7",
            "refs/tags/v3.150.0",
            "refs/tags/v3.16.0",
        )
    )
    assert tool.newest_python_from_tags(listing) == (3, 15, 10)
    assert tool.newest_python_from_tags("0123abcd\trefs/tags/v3.15.0rc3\n") is None
    assert tool.newest_python_from_tags("") is None
    # The installed side applies the same rule: `--version` of a release candidate is not a release.
    assert tool.PYTHON_VERSION.search("Python 3.15.0\n") is not None
    assert tool.PYTHON_VERSION.search("Python 3.15.0rc3\n") is None


def test_the_newest_node_24_is_read_from_the_index():
    entries = [
        {"version": "v25.0.0"},
        {"version": "v24.9.0"},
        {"version": "v24.21.0", "lts": "Krypton"},
        {"version": "v24.21.0-rc.1"},
        {"version": "v22.22.2"},
        {"version": 24},
        "v24.30.0",
    ]
    assert tool.newest_node_from_index(entries) == (24, 21, 0)
    assert tool.newest_node_from_index([{"version": "v22.1.0"}]) is None
    assert tool.newest_node_from_index({"version": "v24.1.0"}) is None


def test_a_verdict_is_never_current_on_an_unread_source():
    """L1/L2: an upstream that could not be read is UNKNOWN (exit 2), never CURRENT; a missing,
    older or incoherent tool is BEHIND (exit 1); only everything current exits 0."""
    old, new = (23, 1, 1), (23, 1, 2)
    assert tool.verdict(None, new) == "MISSING"
    assert tool.verdict(None, None) == "MISSING"
    assert tool.verdict(new, None) == "UNKNOWN"
    assert tool.verdict(old, new) == "OUTDATED"
    assert tool.verdict(new, new) == "CURRENT"
    assert tool.verdict((23, 2, 0), new) == "CURRENT"  # a newer local build is not behind
    assert tool.exit_code(["CURRENT", "CURRENT"]) == 0
    assert tool.exit_code(["CURRENT", "UNKNOWN"]) == 2
    assert tool.exit_code(["UNKNOWN", "OUTDATED"]) == 1
    assert tool.exit_code(["MISSING"]) == 1
    assert tool.exit_code(["INCOHERENT", "CURRENT"]) == 1
    assert tool.exit_code([]) == 2


_PYTHON = {"exe": Path("/opt/py315/bin/python3.15"), "python": (3, 15, 0)}
_PYTHON_NEWEST = ((3, 15, 0), "v3.15.0 in cpython")


def _status(
    llvm, node, llvm_newest, node_newest, python=_PYTHON, python_newest=_PYTHON_NEWEST
) -> tuple[int, str]:
    saved = {
        name: getattr(tool, name)
        for name in (
            "installed_llvm",
            "installed_node",
            "installed_python",
            "query_llvm",
            "query_node",
            "query_python",
        )
    }
    try:
        tool.installed_llvm = lambda: llvm
        tool.installed_node = lambda: node
        tool.installed_python = lambda: python
        tool.query_llvm = lambda: llvm_newest
        tool.query_node = lambda: node_newest
        tool.query_python = lambda: python_newest
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = tool.status()
        return code, out.getvalue()
    finally:
        for name, value in saved.items():
            setattr(tool, name, value)


def test_status_reports_each_tool_and_fails_closed():
    here = Path("/opt/llvm-23/bin")
    llvm = {"bin": here, "clang": (23, 1, 2), "llvm": (23, 1, 2), "mlir": (23, 1, 2)}
    node = {"bin": Path("/opt/node/bin"), "node": (24, 21, 0)}
    current = ((23, 1, 2), "llvmorg-23.1.2"), ((24, 21, 0), "v24.21.0")
    code, text = _status(llvm, node, *current)
    assert code == 0 and text.rstrip().endswith("latest-toolchain: CURRENT"), text
    unread = (None, "github.com: 403")
    code, text = _status(llvm, node, unread, (None, "nodejs.org: 403"), python_newest=unread)
    assert code == 2 and "latest-toolchain: UNKNOWN" in text and "CURRENT\n" not in text, text
    code, text = _status(llvm, node, ((23, 1, 3), "llvmorg-23.1.3"), current[1])
    assert code == 1 and "OUTDATED" in text and "setup_mlir.sh" in text, text
    incoherent = {**llvm, "llvm": (23, 1, 1)}
    code, text = _status(incoherent, node, *current)
    assert code == 1 and "INCOHERENT" in text, text
    code, text = _status(llvm, {"bin": None, "node": None}, *current)
    assert code == 1 and "MISSING" in text and "install-node" in text, text
    # The Python row fails closed the same three ways, and its hint names the installers.
    code, text = _status(llvm, node, *current, python={"exe": None, "python": None})
    assert code == 1 and "python 3.15 installed -" in text and "MISSING" in text, text
    assert "uv python install 3.15" in text and "conda-forge" in text, text
    code, text = _status(llvm, node, *current, python_newest=((3, 15, 1), "v3.15.1 in cpython"))
    assert code == 1 and "OUTDATED" in text and "uv python install 3.15" in text, text
    code, text = _status(llvm, node, *current, python_newest=(None, "github.com: 403"))
    assert code == 2 and text.rstrip().endswith("latest-toolchain: UNKNOWN"), text
    assert re.search(r"^latest-toolchain: python 3\.15 .* UNKNOWN$", text, re.M), text


def _tarball(name: str, members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:xz") as archive:
        for path, data in members.items():
            info = tarfile.TarInfo(path if path.startswith("..") else f"{name}/{path}")
            info.size = len(data)
            info.mode = 0o755
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _install(blobs: dict[str, bytes], *, allow_unsigned: bool, signed=None, **platform):
    saved_fetch, saved_signed = tool.fetch, tool.signed_shasums
    try:
        tool.fetch = lambda url, limit: blobs[url.rsplit("/", 1)[-1]]
        if signed is not None:
            tool.signed_shasums = signed
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "node"
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = tool.install_node(
                    "24.99.0",
                    dest,
                    None,
                    allow_unsigned,
                    machine=platform.get("machine", "x86_64"),
                    system=platform.get("system", "linux"),
                )
            installed = sorted(p.relative_to(tmp).as_posix() for p in Path(tmp).rglob("*"))
        return code, out.getvalue(), installed
    finally:
        tool.fetch, tool.signed_shasums = saved_fetch, saved_signed


def test_install_node_checks_before_it_extracts():
    """Nothing is extracted unless the tarball matches a checksum list whose signature verified
    (or --allow-unsigned said otherwise, and the output says so); a tarball that writes outside
    its directory is refused; an unsupported platform or another major is unusable."""
    try:
        import lzma  # noqa: F401 - the tarballs are .tar.xz
    except ImportError:
        return
    name = "node-v24.99.0-linux-x64"
    good = _tarball(name, {"bin/node": b"#!/bin/sh\necho v24.99.0\n"})
    sums = f"{hashlib.sha256(good).hexdigest()}  {name}.tar.xz\n".encode()
    blobs = {"SHASUMS256.txt": sums, "SHASUMS256.txt.asc": b"signed", f"{name}.tar.xz": good}

    def unsigned(asc, keys_dir):
        raise tool.Refused("no good signature on SHASUMS256.txt.asc")

    code, text, installed = _install(blobs, allow_unsigned=False, signed=unsigned)
    assert code == 1 and "REFUSED" in text and installed == [], (text, installed)
    code, text, installed = _install(
        blobs, allow_unsigned=False, signed=lambda asc, keys: sums.decode()
    )
    assert code == 0 and f"node/{name}/bin/node" in installed and "signature-verified" in text, text
    code, text, installed = _install(blobs, allow_unsigned=True)
    assert code == 0 and "NOT signature-verified" in text, text
    tampered = {**blobs, f"{name}.tar.xz": good + b"\0"}
    code, text, installed = _install(tampered, allow_unsigned=True)
    assert code == 1 and "sha256" in text and not any(name in p for p in installed), (
        text,
        installed,
    )
    escaping = _tarball(name, {"bin/node": b"x", "../outside": b"x"})
    hostile = {
        **blobs,
        "SHASUMS256.txt": f"{hashlib.sha256(escaping).hexdigest()}  {name}.tar.xz\n".encode(),
        f"{name}.tar.xz": escaping,
    }
    code, text, installed = _install(hostile, allow_unsigned=True)
    assert code == 1 and "outside" in text and "outside" not in " ".join(installed), (
        text,
        installed,
    )
    assert _install(blobs, allow_unsigned=True, machine="sparc64")[0] == 2
    assert _install(blobs, allow_unsigned=True, system="darwin")[0] == 2
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert tool.install_node("22.1.0", Path("unused"), None, True, "x86_64", "linux") == 2


def test_the_rule_is_stated_where_agents_and_ci_read_it():
    """One rule, one set of majors, read out of each place it lives (L14): the skill, AGENTS.md,
    CONTRIBUTING.md, the digest, the tool, the check's legs, the preset and the workflow."""
    skill = _SKILL.read_text(encoding="utf-8")
    front = re.match(r"^---\n(.*?)\n---\n", skill, re.S)
    assert front and "name: bcir-latest-toolchain" in front.group(1), "the skill's frontmatter"
    for needle in ("Clang 18", "23", "Node 24", "Python 3.15"):
        assert needle in front.group(1), f"the skill's description does not name {needle}"
    for path in ("tools/local/check_latest.sh", "tools/local/latest_toolchain.py"):
        assert path in skill and (_ROOT / path).is_file(), path
    for doc in ("AGENTS.md", "CONTRIBUTING.md", ".claude/context/BCIR_DIGEST.md"):
        text = (_ROOT / doc).read_text(encoding="utf-8")
        assert "tools/local/check_latest.sh" in text, f"{doc} does not send changes to the check"
    agents = (_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert ".claude/skills/bcir-latest-toolchain" in agents

    # The check's legs are the skill's table, in order.
    script = _SCRIPT.read_text(encoding="utf-8")
    legs = re.search(r'^LEGS_ALL="([^"]+)"$', script, re.M)
    assert legs, "check_latest.sh names no legs"
    table = re.findall(r"^\| `([a-z]+)` \|", skill, re.M)
    assert table == legs.group(1).split(), (table, legs.group(1))
    # Two workers everywhere (AGENTS.md): every ctest/run_all/cmake --build names -j 2 or a preset.
    for line in script.splitlines():
        if re.search(r"\b(ctest|run_all|cmake --build)\b", line) and not line.lstrip().startswith(
            "#"
        ):
            assert "-j 2" in line or "--preset" in line, line
    assert not re.search(r"-j ?0\b|--jobs=auto", script)
    assert "BCIR_REQUIRE_LLVM=1" in script and "BCIR_REQUIRE_TSAN" in script
    # The Python legs run on the newest interpreter the tool found, never on whatever `python3` is.
    assert 'PY_LATEST="${BCIR_PYTHON_LATEST}"' in script
    for leg in ("quick", "thorough"):
        assert f'"${{PY_LATEST}}" -m bcir.tests.run_all --tier {leg} -j 2' in script, leg
    assert not re.search(r'"\$\{PY\}" -m bcir\.tests\.run_all', script)

    # The majors agree across the tool, the preset and the workflow.
    assert (tool.LLVM_MAJOR, tool.NODE_MAJOR, tool.PYTHON_SERIES) == (23, 24, (3, 15))
    presets = json.loads((_ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
    configure = {p["name"]: p for p in presets["configurePresets"]}
    assert configure["clang23"]["cacheVariables"] == {
        "CMAKE_C_COMPILER": f"clang-{tool.LLVM_MAJOR}",
        "CMAKE_CXX_COMPILER": f"clang++-{tool.LLVM_MAJOR}",
    }
    assert {"name": "clang23", "configurePreset": "clang23", "jobs": 2} in presets["buildPresets"]
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    oracle = _job(workflow, "oracle-llvm-latest")
    assert f"node-version: {tool.NODE_MAJOR}" in oracle and "BCIR_REQUIRE_LLVM=1" in oracle
    assert f"clang-{tool.LLVM_MAJOR}" in oracle
    assert f'python-version: "{tool.PYTHON_LABEL}"' in oracle, "the oracle on 23 runs on 3.15"
    assert f'"Python {tool.PYTHON_LABEL}."*)' in oracle, "and confirms the interpreter it got"
    assert f"clang-{tool.LLVM_MAJOR}" in _job(workflow, "c-rails-llvm-latest")
    cmake = _job(workflow, "cmake-build")
    assert re.search(r"compiler: \[[^\]]*\bclang23\b", cmake), "the CMake job has no clang23 cell"
    assert f"add_llvm_apt_repo.sh {tool.LLVM_MAJOR}" in cmake and "BCIR_REQUIRE_TSAN=ON" in cmake


def _python_matrix(job_text: str) -> list[tuple[int, int]]:
    """The `python: [...]` matrix of a job, as (major, minor) pairs."""
    found = re.search(r"^\s+python: \[([^\]]*)\]", job_text, re.M)
    assert found, "the job has no python matrix"
    return sorted(
        tuple(int(part) for part in cell.strip().strip('"').split("."))
        for cell in found.group(1).split(",")
    )


def test_every_declared_python_version_is_a_ci_cell():
    """pyproject.toml's classifiers, its `requires-python`, the tool's PYTHON_SERIES and the three
    interpreter matrices of ci.yml are one list, read out of each place it lives (L14): the floor
    job runs the floor, host-portability and python-newer run the rest, the newest is the series
    the local check judges on, and the series has no gap -- a classifier no cell runs, or a cell
    no classifier names, fails here rather than shipping as a claim nothing exercises."""
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = sorted(
        (3, int(found.group(1)))
        for entry in project["classifiers"]
        if (found := re.fullmatch(r"Programming Language :: Python :: 3\.(\d+)", entry, re.ASCII))
    )
    floor = tuple(int(part) for part in project["requires-python"].lstrip(">=~^ ").split(".")[:2])
    assert declared, "pyproject.toml names no interpreter"
    assert declared[0] == floor, (declared, floor)
    assert declared[-1] == tool.PYTHON_SERIES, (declared, tool.PYTHON_SERIES)
    assert declared == [(3, minor) for minor in range(floor[1], tool.PYTHON_SERIES[1] + 1)], (
        declared
    )
    assert "Programming Language :: Python :: 3 :: Only" in project["classifiers"]

    workflow = _WORKFLOW.read_text(encoding="utf-8")
    floor_cells = _python_matrix(_job(workflow, "python-floor"))
    host_cells = _python_matrix(_job(workflow, "host-portability"))
    newer_cells = _python_matrix(_job(workflow, "python-newer"))
    assert floor_cells == [floor], floor_cells
    assert tool.PYTHON_SERIES in host_cells, "the host matrix does not run the newest interpreter"
    assert tool.PYTHON_SERIES in newer_cells, "python-newer does not run the newest interpreter"
    exercised = sorted(set(floor_cells) | set(host_cells) | set(newer_cells))
    assert exercised == declared, (exercised, declared)
    # The host matrix also keeps the CI default (ubuntu-latest's interpreter) every other job runs.
    assert (3, 12) in host_cells, host_cells
    # The job that runs the newest beside LLVM 23 and Node 24 is the one the skill names.
    assert f'python-version: "{tool.PYTHON_LABEL}"' in _job(workflow, "oracle-llvm-latest")
    skill = _SKILL.read_text(encoding="utf-8")
    for job_name in ("python-newer", "host-portability", "oracle-llvm-latest", "python-floor"):
        assert f"`{job_name}`" in skill, f"the skill does not name {job_name}"


def test_the_analyzer_leg_fails_on_any_diagnostic():
    """tools/c/analyze_hosted.sh holds the one list of hosted units that CI's two analyzer steps and
    the `analyze` leg run. `--analyze` reports its findings on stderr and still exits 0, so a unit
    with any diagnostic fails the script, as does one whose analysis exits nonzero or never ran; a
    clean run passes, and a missing clang or a worker count that is no count is UNUSABLE (2),
    never a pass (L1, L2). The clang here is a stand-in, so nothing is analysed for real."""
    from bcir.tests.run_all import _child_timeout_seconds, posix_bash

    bash = posix_bash()
    if bash is None:
        return  # no POSIX shell here (a Windows runner's bash can be the WSL launcher)
    bound = _child_timeout_seconds()  # the harness's child bound, not a timing (run_all.posix_bash)
    script = _ROOT / "tools" / "c" / "analyze_hosted.sh"
    workflow = (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert workflow.count("bash tools/c/analyze_hosted.sh") == 2, "both CI analyzer steps run it"
    inline = [ln for ln in workflow.splitlines() if "--analyze" in ln and "#" not in ln]
    assert not inline, f"ci.yml runs the analyzer over a list of its own: {inline}"
    with tempfile.TemporaryDirectory() as tmp:
        clang = Path(tmp) / "clang"

        def run(body: str, **env: str) -> subprocess.CompletedProcess:
            clang.write_text(
                '#!/bin/sh\ncase "$1" in --version) echo "stand-in clang 1"; exit 0 ;; esac\n'
                + body,
                encoding="utf-8",
                newline="\n",
            )
            clang.chmod(0o755)
            return subprocess.run(
                [bash, str(script)],
                env={"PATH": "/usr/bin:/bin", "CLANG": str(clang), **env},
                capture_output=True,
                timeout=bound,
            )

        clean = run("exit 0\n")
        assert clean.returncode == 0, clean
        assert re.search(rb"analyze-hosted: PASS \(\d+ units", clean.stdout), clean.stdout
        noisy = run('case "$*" in *bcir_make.c*) echo "warning: a finding" >&2 ;; esac\nexit 0\n')
        assert noisy.returncode == 1 and b"runtime/c/bcir_make.c" in noisy.stderr, noisy
        assert b"a finding" in noisy.stderr and b"bcir_cfront.c" not in noisy.stderr, noisy
        assert run("exit 3\n").returncode == 1
        assert run("exit 0\n", ANALYZE_JOBS="0").returncode == 2
        assert run("exit 0\n", ANALYZE_JOBS="two").returncode == 2
        absent = subprocess.run(
            [bash, str(script)],
            env={"PATH": "/usr/bin:/bin", "CLANG": str(Path(tmp) / "absent")},
            capture_output=True,
            timeout=bound,
        )
        assert absent.returncode == 2 and b"UNUSABLE" in absent.stderr, absent


def test_the_check_refuses_what_it_cannot_run():
    """A usage error is exit 2 before any leg runs: an unknown leg is not silently dropped (the
    run would otherwise pass while judging less than asked). Each launch is bounded by the harness's
    child bound, not a minute: refusing `--bogus` took Git Bash over 60 s on a loaded Windows runner."""
    from bcir.tests.run_all import _child_timeout_seconds, posix_bash

    bash = posix_bash()
    if bash is None:
        return  # no POSIX shell here (a Windows runner's bash can be the WSL launcher)
    bound = _child_timeout_seconds()
    for args in (["--legs", "cmake,nope"], ["--bogus"], ["--legs", ","]):
        run = subprocess.run([bash, str(_SCRIPT), *args], capture_output=True, timeout=bound)
        assert run.returncode == 2, (args, run.returncode, run.stdout, run.stderr)
    shown = subprocess.run([bash, str(_SCRIPT), "--help"], capture_output=True, timeout=bound)
    assert shown.returncode == 0 and b"Legs" in shown.stdout and b"set -uo" not in shown.stdout
