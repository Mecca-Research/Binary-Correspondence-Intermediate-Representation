"""The build manifest (runtime/manifest.json) reconciles with the tree, and its checker fires on
every violation it exists to catch.

`tools/build/manifest.py` is the BUILD-1 gate (docs/BCIR_BUILD_ROADMAP.md): CMake reads the
manifest directly, and the shell gates, the Python harnesses, the memory-class inventory, the
C++ seam's lists and the presets are held to it. Every rule gets its injected violation here
(docs/security/laws.md L2, L11), the scanners are shown to have examined the real gates, and
the parity gate is shown to refuse before it compiles. The tools are not in the wheel, so this
module is registered repository-only in run_all.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = _ROOT / "tools" / "build"


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(f"bcir_tools_build_{name}", _TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


manifest_tool = _load_tool("manifest")
parity_tool = _load_tool("build_parity")

_TREE = None


def _tree():
    global _TREE
    if _TREE is None:
        _TREE = manifest_tool.Tree()
    return _TREE


def _manifest() -> dict:
    return manifest_tool.load()


def _mutated(mutate) -> dict:
    manifest = copy.deepcopy(_manifest())
    mutate(manifest)
    return manifest


def test_the_manifest_reconciles_with_the_tree():
    findings = manifest_tool.check(_manifest(), _tree())
    assert findings == [], "\n".join(findings)


# One injected violation per rule (and per distinct clause where a rule has several): the
# checker must name the rule, or the rule is a comment (L2).
_FAULTS = (
    ("M1", "the schema tag", lambda m: m.__setitem__("schema", "bcir-build-manifest.v0")),
    ("M1", "a kind missing", lambda m: m.pop("fuzzers")),
    ("M1", "an empty freestanding list", lambda m: m.__setitem__("freestanding_checks", [])),
    (
        "M2",
        "an unknown unit key",
        lambda m: m["libraries"]["bcir_base"].__setitem__("flags", ["-O3"]),
    ),
    (
        "M2",
        "a class on a harness",
        lambda m: m["harnesses"]["test_runtime"].__setitem__("class", "hosted_tool"),
    ),
    ("M2", "a zero max_len", lambda m: m["fuzzers"]["fuzz_asn1"].__setitem__("max_len", "0")),
    (
        "M2",
        "a unit without sources",
        lambda m: m["tools"]["bcir_microbench"].__setitem__("sources", []),
    ),
    (
        "M3",
        "a source that is not a file",
        lambda m: m["libraries"]["bcir_base"]["sources"].append("bcir_invented.c"),
    ),
    (
        "M4",
        "a unit in no library",
        lambda m: m["libraries"]["bcir_base"]["sources"].remove("bcir_sha256.c"),
    ),
    (
        "M4",
        "a unit in two libraries",
        lambda m: m["libraries"]["bcir_artifact"]["sources"].append("bcir_sha256.c"),
    ),
    (
        "M4",
        "a fixture as a unit",
        lambda m: m["harnesses"]["test_runtime"]["sources"].append("cfront_abi.c"),
    ),
    (
        "M5",
        "an unknown library",
        lambda m: m["tools"]["bcir-cc"]["libraries"].append("bcir_nowhere"),
    ),
    (
        "M5",
        "a dependency cycle",
        lambda m: m["libraries"]["bcir_base"].__setitem__("libraries", ["bcir_streampack"]),
    ),
    (
        "M6",
        "a foreign link name",
        lambda m: m["harnesses"]["test_ring"].__setitem__("link", ["dl"]),
    ),
    (
        "M6",
        "an option without its dash",
        lambda m: m["libraries"]["bcir_models"].__setitem__("options", ["O2"]),
    ),
    (
        "M7",
        "a library class its units do not carry",
        lambda m: m["libraries"]["bcir_base"].__setitem__("class", "hosted_tool"),
    ),
    (
        "M8",
        "a gate-compiled unit dropped",
        lambda m: m["freestanding_checks"].remove("bcir_runtime.c"),
    ),
    (
        "M8",
        "a hosted unit as freestanding",
        lambda m: m["freestanding_checks"].append("bcir_cfront.c"),
    ),
    (
        "M9",
        "a gate link outside the closure",
        lambda m: m["harnesses"]["test_control_plane"].__setitem__("libraries", ["bcir_base"]),
    ),
    (
        "M9",
        "a fuzz target's -max_len",
        lambda m: m["fuzzers"]["fuzz_asn1"].__setitem__("max_len", "4096"),
    ),
    (
        "M10",
        "a Python harness link outside the closure",
        lambda m: m["harnesses"]["test_decode"].__setitem__("libraries", []),
    ),
    (
        "M11",
        "the seam's C++ units",
        lambda m: m["seam_libraries"]["bcir_seam"].__setitem__("sources", ["bcir_handoff.cpp"]),
    ),
)


def test_every_injected_violation_is_a_finding():
    tree = _tree()
    for code, what, mutate in _FAULTS:
        findings = manifest_tool.check(_mutated(mutate), tree)
        assert any(f.startswith(f"{code}:") for f in findings), (
            f"{code} ({what}) did not fire: {findings}"
        )


def test_the_presets_hold_the_two_worker_law():
    errors: list[str] = []
    manifest_tool.check_presets(_tree().presets, errors)
    assert errors == [], errors
    presets = copy.deepcopy(_tree().presets)
    presets["buildPresets"][0]["jobs"] = 4
    presets["testPresets"][0]["execution"]["jobs"] = 8
    presets["testPresets"].append(
        {"name": "stray", "configurePreset": "nowhere", "execution": {"jobs": 2}}
    )
    errors = []
    manifest_tool.check_presets(presets, errors)
    assert [e for e in errors if "runs 4 jobs" in e], errors
    assert [e for e in errors if "runs 8 jobs" in e], errors
    assert [e for e in errors if "unknown configure preset" in e], errors
    errors = []
    manifest_tool.check_presets("CMakePresets.json: not JSON", errors)
    assert errors and errors[0].startswith("M12:"), errors


def test_the_scanners_examined_the_real_gates():
    """A scanner that matched nothing would make M8-M11 vacuous (L2)."""
    tree = _tree()
    assert tree.missing_gates == [], tree.missing_gates
    runtime_groups = tree.gates["tools/c/check_runtime.sh"]
    with_main = [g for g in runtime_groups if any(t.startswith("test_") for t in g[1])]

    assert len(runtime_groups) >= 150, len(runtime_groups)
    assert len(with_main) >= 25, len(with_main)
    assert tree.fuzz is not None and len(tree.fuzz) == len(_manifest()["fuzzers"]), tree.fuzz
    assert {"bcir_runtime.c", "bcir_kplan.c", "bcir_jer.c", "bcir_per.c"} <= tree.freestanding, (
        tree.freestanding
    )
    assert len(tree.python) >= 40, len(tree.python)
    assert all(lists is not None for lists in tree.seam.values()), tree.seam


def test_python_groups_follow_the_literal_containers():
    source = (
        'A = ("bcir_runtime.c", "test_runtime.c")\n'
        'B = [os.path.join(C, "bcir_exec.c"), ["bcir_plan.c"], "not_a_source.c", f"{x}.c"]\n'
        'single = ROOT / "bcir_sha256.c"\n'
        'call(str(_C / "fuzz_asn1.c"), "-lm")\n'
    )
    groups = {tuple(g) for g in manifest_tool.python_groups(source)}
    assert groups == {
        ("bcir_runtime.c", "test_runtime.c"),
        ("bcir_exec.c",),
        ("bcir_plan.c",),
        ("bcir_sha256.c",),
        ("fuzz_asn1.c",),
    }, groups
    broken = manifest_tool.python_groups('x = ("bcir_runtime.c",\n')
    assert broken == [["bcir_runtime.c"]], broken


def test_the_closure_is_dependents_first_and_once():
    data = manifest_tool.closure(_manifest(), "tools", "bcir-cc")
    assert data["sources"][0] == "bcir_cc.c"
    assert data["sources"].count("bcir_runtime.c") == 1
    assert data["libraries"] == ["bcir_cfront", "bcir_streampack", "bcir_base"], data["libraries"]
    assert data["link"] == ["m"], data["link"]
    assert all(opt.startswith(("-", "gcc:", "clang:")) for opt in data["options"]), data["options"]


def test_the_cmake_reader_and_the_checker_agree():
    """CMake reads the same manifest; the two readers must agree on its kinds and keys (L12)."""
    cmake = (_ROOT / "cmake" / "BCIRManifest.cmake").read_text(encoding="utf-8")
    kinds = re.search(r"set\(BCIR_MANIFEST_KINDS ([^)]*)\)", cmake)
    assert kinds is not None
    assert tuple(kinds.group(1).split()) == manifest_tool.KINDS, kinds.group(1)
    for key in manifest_tool.UNIT_KEYS:
        assert re.search(rf'"\$\{{_unit\}}" {key} ', cmake), (
            f"the CMake reader does not read {key!r}"
        )
    assert manifest_tool.SCHEMA in cmake
    root = (_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "runtime/manifest.json" in root


def test_an_unusable_manifest_is_reported_as_such():
    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / "missing.json"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = manifest_tool.main(["--manifest", str(missing), "--check"])
        assert rc == 2 and "unusable" in out.getvalue(), (rc, out.getvalue())
        bad = Path(tmp) / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = manifest_tool.main(["--manifest", str(bad), "--check"])
        assert rc == 2 and "unusable" in out.getvalue(), (rc, out.getvalue())
        shape = Path(tmp) / "shape.json"
        shape.write_text(json.dumps({"schema": "other"}), encoding="utf-8")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = manifest_tool.main(["--manifest", str(shape), "--check"])
        assert rc == 1 and "FAIL: M1:" in out.getvalue(), (rc, out.getvalue())


def test_the_parity_gate_refuses_before_it_compiles():
    """No compiler, no bcir-cc or no fixtures is exit 2, never a pass, and nothing is built."""
    with tempfile.TemporaryDirectory() as tmp:
        empty = Path(tmp)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = parity_tool.main(["--bcir-cc", str(empty / "absent"), "--cc", sys.executable])
        assert rc == 2 and "UNUSABLE" in out.getvalue(), (rc, out.getvalue())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = parity_tool.main(["--bcir-cc", sys.executable, "--cc", str(empty / "no-such-cc")])
        assert rc == 2 and "no compiler" in out.getvalue(), (rc, out.getvalue())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = parity_tool.main(
                ["--bcir-cc", sys.executable, "--cc", sys.executable, "--fixtures", str(empty)]
            )
        assert rc == 2 and "INVALID" in out.getvalue(), (rc, out.getvalue())
        assert not list(empty.iterdir()), "the gate wrote into the fixture directory"


def test_the_build_tools_are_tracked_and_not_ignored():
    """The checker and the parity gate must reach every checkout: `.gitignore`'s `build/` once
    matched tools/build/ too, so the tools passed here untracked and were absent on CI (L21: a
    skip is where a shipping defect hides). Without git in reach there is nothing to judge."""
    if shutil.which("git") is None or not (_ROOT / ".git").exists():
        return
    for name in ("manifest.py", "build_parity.py"):
        rel = f"tools/build/{name}"
        ignored = subprocess.run(["git", "check-ignore", "-q", rel], cwd=_ROOT, capture_output=True)
        assert ignored.returncode == 1, (
            f"{rel} is ignored by .gitignore (exit {ignored.returncode})"
        )
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", rel], cwd=_ROOT, capture_output=True
        )
        assert tracked.returncode == 0, f"{rel} is not tracked by git"


def test_the_ci_owns_the_build_gate():
    """The gate's CI owner (L2): a job configures with both compilers and runs the build label."""
    workflow = (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    job = re.search(r"^  cmake-build:\n(.*?)(?=^  [a-z0-9-]+:\n|\Z)", workflow, re.S | re.M)
    assert job is not None, "no cmake-build job"
    body = job.group(1)
    assert "gcc" in body and "clang" in body, "the job does not cover both compilers"
    assert re.search(r"ctest .*-L build", body), "the job does not run the build label"
    assert "BCIR_BUILD_MLIR=OFF" in body, (
        "the MLIR law has its own job; this one must not depend on a found package"
    )
