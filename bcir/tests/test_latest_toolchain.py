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
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SKILL = _ROOT / ".claude" / "skills" / "bcir-latest-toolchain" / "SKILL.md"
_SCRIPT = _ROOT / "tools" / "local" / "check_latest.sh"


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


def _status(llvm, node, llvm_newest, node_newest) -> tuple[int, str]:
    saved = {
        name: getattr(tool, name)
        for name in ("installed_llvm", "installed_node", "query_llvm", "query_node")
    }
    try:
        tool.installed_llvm = lambda: llvm
        tool.installed_node = lambda: node
        tool.query_llvm = lambda: llvm_newest
        tool.query_node = lambda: node_newest
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
    code, text = _status(llvm, node, (None, "github.com: 403"), (None, "nodejs.org: 403"))
    assert code == 2 and "latest-toolchain: UNKNOWN" in text and "CURRENT\n" not in text, text
    code, text = _status(llvm, node, ((23, 1, 3), "llvmorg-23.1.3"), current[1])
    assert code == 1 and "OUTDATED" in text and "setup_mlir.sh" in text, text
    incoherent = {**llvm, "llvm": (23, 1, 1)}
    code, text = _status(incoherent, node, *current)
    assert code == 1 and "INCOHERENT" in text, text
    code, text = _status(llvm, {"bin": None, "node": None}, *current)
    assert code == 1 and "MISSING" in text and "install-node" in text, text


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
    for needle in ("Clang 18", "23", "Node 24"):
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

    # The majors agree across the tool, the preset and the workflow.
    assert (tool.LLVM_MAJOR, tool.NODE_MAJOR) == (23, 24)
    presets = json.loads((_ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
    configure = {p["name"]: p for p in presets["configurePresets"]}
    assert configure["clang23"]["cacheVariables"] == {
        "CMAKE_C_COMPILER": f"clang-{tool.LLVM_MAJOR}",
        "CMAKE_CXX_COMPILER": f"clang++-{tool.LLVM_MAJOR}",
    }
    assert {"name": "clang23", "configurePreset": "clang23", "jobs": 2} in presets["buildPresets"]
    workflow = (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    def job(name: str) -> str:
        found = re.search(rf"^  {name}:\n(.*?)(?=^  [a-z0-9-]+:\n|\Z)", workflow, re.S | re.M)
        assert found, f"no {name} job"
        return found.group(1)

    oracle = job("oracle-llvm-latest")
    assert f"node-version: {tool.NODE_MAJOR}" in oracle and "BCIR_REQUIRE_LLVM=1" in oracle
    assert f"clang-{tool.LLVM_MAJOR}" in oracle
    assert f"clang-{tool.LLVM_MAJOR}" in job("c-rails-llvm-latest")
    cmake = job("cmake-build")
    assert re.search(r"compiler: \[[^\]]*\bclang23\b", cmake), "the CMake job has no clang23 cell"
    assert f"add_llvm_apt_repo.sh {tool.LLVM_MAJOR}" in cmake and "BCIR_REQUIRE_TSAN=ON" in cmake


def test_the_check_refuses_what_it_cannot_run():
    """A usage error is exit 2 before any leg runs: an unknown leg is not silently dropped (the
    run would otherwise pass while judging less than asked)."""
    bash = shutil.which("bash")
    if (
        bash is None
        or subprocess.run([bash, "-c", "echo ok"], capture_output=True).stdout != b"ok\n"
    ):
        return  # no POSIX shell here (a Windows runner's bash is the WSL launcher)
    for args in (["--legs", "cmake,nope"], ["--bogus"], ["--legs", ","]):
        run = subprocess.run([bash, str(_SCRIPT), *args], capture_output=True, timeout=60)
        assert run.returncode == 2, (args, run.returncode, run.stdout, run.stderr)
    shown = subprocess.run([bash, str(_SCRIPT), "--help"], capture_output=True, timeout=60)
    assert shown.returncode == 0 and b"Legs" in shown.stdout and b"set -uo" not in shown.stdout
