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

import ast
import contextlib
import copy
import importlib.util
import io
import json
import os
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
section_tool = _load_tool("section_parity")
mutate_tool = _load_tool("mutate")
sanitizer_tool = _load_tool("sanitizer")

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
        # the runtime gate links nothing since BUILD-8; check_memory_discipline.sh still does
        "M9",
        "a gate link outside the closure",
        lambda m: m["harnesses"]["test_memory_discipline"].__setitem__("libraries", []),
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
    (
        "M13",
        "a section's unknown harness",
        lambda m: m["sections"]["runtime"].__setitem__("harnesses", ["test_nowhere"]),
    ),
    (
        "M13",
        "a section taking a fuzzer (harnesses, variants and tools are the section binaries)",
        lambda m: m["sections"]["bcir_cc"].__setitem__("harnesses", ["fuzz_asn1"]),
    ),
    (
        "M13",
        "a section's missing script",
        lambda m: m["sections"]["runtime"].__setitem__("script", "tools/c/sections/absent.sh"),
    ),
    ("M13", "a script no section registers", lambda m: m["sections"].pop("encoder")),
    (
        "M13",
        "a section without harnesses",
        lambda m: m["sections"]["executor"].__setitem__("harnesses", []),
    ),
    (
        "M13",
        "a varies pattern without its one capture group",
        lambda m: m["sections"]["ring"].__setitem__("varies", ["concurrent.violations=[0-9]+"]),
    ),
    (
        "M14",
        "a variant of no harness",
        lambda m: m["variants"]["test_kplan_O0"].__setitem__("of", "test_nowhere"),
    ),
    (
        "M14",
        "a mutation whose anchor is gone",
        lambda m: m["variants"]["test_kplan_mutant"]["mutation"].__setitem__("find", "no such law"),
    ),
    (
        "M14",
        "a mutation whose anchor repeats",
        lambda m: m["variants"]["test_kplan_mutant"]["mutation"].__setitem__("find", "return"),
    ),
    (
        "M14",
        "a mutation that changes nothing",
        lambda m: m["variants"]["test_kplan_mutant"]["mutation"].__setitem__(
            "replace", m["variants"]["test_kplan_mutant"]["mutation"]["find"]
        ),
    ),
    (
        "M14",
        "a mutation outside the harness's closure",
        lambda m: m["variants"]["test_kplan_mutant"]["mutation"].__setitem__(
            "file", "bcir_cfront.c"
        ),
    ),
    (
        "M14",
        "a variant no section runs",
        lambda m: m["variants"].__setitem__(
            "test_extra", {"of": "test_runtime", "options": ["-O1"]}
        ),
    ),
    (
        "M14",
        "a variant that changes nothing",
        lambda m: m["variants"].__setitem__("test_kplan_O0", {"of": "test_kplan"}),
    ),
    (
        "M14",
        "a variant that shadows a unit",
        lambda m: m["variants"].__setitem__(
            "test_runtime", {"of": "test_exec", "options": ["-O1"]}
        ),
    ),
    (
        "M14",
        "a variant in an unknown C standard",
        lambda m: m["variants"]["test_x86_interrupt_c11"].__setitem__("standard", 99),
    ),
    (
        "M14",
        "a variant in its harness's own standard (no change)",
        lambda m: m["variants"]["test_x86_interrupt_c11"].__setitem__("standard", 23),
    ),
    (
        "M14",
        "a standard spelled as a boolean",
        lambda m: m["variants"]["test_q8_tables_c11"].__setitem__("standard", True),
    ),
    (
        "M14",
        "a variant with a sanitizer the probe does not know",
        lambda m: m["variants"]["test_ring_tsan"].__setitem__("sanitizer", "address"),
    ),
    (
        "M13",
        "a section mixing a sanitizer build with a plain one",
        lambda m: m["sections"]["ring_tsan"]["harnesses"].__setitem__(1, "test_ring_O0"),
    ),
    (
        "M13",
        "a section taking a kernel the manifest lacks",
        lambda m: m["sections"]["ols"].__setitem__("harnesses", ["kernel_nope"]),
    ),
    ("M15", "kernels that are no object", lambda m: m.__setitem__("kernels", [])),
    ("M15", "a kernel without its driver", lambda m: m["kernels"]["kernel_ols"].pop("main")),
    (
        "M15",
        "a kernel with a key no kernel has",
        lambda m: m["kernels"]["kernel_ols"].__setitem__("options", ["-O3"]),
    ),
    (
        "M15",
        "an emitter the oracle lacks",
        lambda m: m["kernels"]["kernel_ols"].__setitem__("emit", "emit_nothing_c"),
    ),
    (
        "M15",
        "an emitter that is no emit_* name",
        lambda m: m["kernels"]["kernel_ols"].__setitem__("emit", "os.system"),
    ),
    (
        "M15",
        "a floating kernel argument",
        lambda m: m["kernels"]["kernel_ols"].__setitem__("args", [8, 2.5]),
    ),
    (
        "M15",
        "a kernel argument spelled as a boolean",
        lambda m: m["kernels"]["kernel_ols"].__setitem__("args", [True]),
    ),
    (
        "M15",
        "a driver that does not exist",
        lambda m: m["kernels"]["kernel_ols"].__setitem__("main", "absent.c"),
    ),
    (
        "M15",
        "a driver named by a path",
        lambda m: m["kernels"]["kernel_ols"].__setitem__("main", "../ols.c"),
    ),
    (
        "M15",
        "a kernel linking a library outside m and pthread",
        lambda m: m["kernels"]["kernel_ols"].__setitem__("link", ["lapack"]),
    ),
    ("M15", "an empty link list", lambda m: m["kernels"]["kernel_tree"].__setitem__("link", [])),
    (
        "M15",
        "a kernel no section runs",
        lambda m: m["kernels"].__setitem__(
            "kernel_extra", {"emit": "emit_tree_predict_c", "args": [5, 2, "t"], "main": "tree.c"}
        ),
    ),
    (
        "M15",
        "a kernel that shadows a unit",
        lambda m: m["kernels"].__setitem__("test_runtime", m["kernels"].pop("kernel_tree")),
    ),
    ("M15", "a driver no kernel names", lambda m: m["kernels"].pop("kernel_kmeans")),
    (
        "M13",
        "a compiler-only section that names a binary",
        lambda m: m["sections"]["inlineasm"].__setitem__("harnesses", ["test_runtime"]),
    ),
    (
        "M13",
        "a compiler_only flag that is not true",
        lambda m: m["sections"]["portio"].__setitem__("compiler_only", "yes"),
    ),
    (
        "M13",
        "a compiler_only flag of false on a section without binaries",
        lambda m: m["sections"]["barrier"].__setitem__("compiler_only", False),
    ),
    ("M16", "delegated gates that are no object", lambda m: m.__setitem__("delegated", [])),
    (
        "M16",
        "a delegated entry with an unknown key",
        lambda m: m["delegated"]["cpp-sycl"].__setitem__("timeout", 60),
    ),
    (
        "M16",
        "a delegated entry without labels",
        lambda m: m["delegated"]["cpp-sycl"].pop("labels"),
    ),
    (
        "M16",
        "a delegated name that is no CTest name",
        lambda m: m["delegated"].__setitem__("SYCL", m["delegated"].pop("cpp-sycl")),
    ),
    (
        "M16",
        "a delegated label outside c and cpp",
        lambda m: m["delegated"]["cpp-sycl"].__setitem__("labels", ["fuzz"]),
    ),
    (
        "M16",
        "a delegated script M9 does not read",
        lambda m: m["delegated"]["cpp-sycl"].__setitem__("script", "tools/cpp/check_other.sh"),
    ),
    (
        "M16",
        "delegating to the runtime gate itself",
        lambda m: m["delegated"]["cpp-sycl"].__setitem__("script", "tools/c/check_runtime.sh"),
    ),
    (
        "M16",
        "delegating to the cfront sanitizer, which keeps its own switch",
        lambda m: m["delegated"]["cpp-sycl"].__setitem__("script", "tools/c/sanitize_cfront.sh"),
    ),
    (
        "M16",
        "two entries delegating to one script",
        lambda m: m["delegated"].__setitem__(
            "c-memory-classes", dict(m["delegated"]["c-memory-discipline"])
        ),
    ),
    (
        "M16",
        "a gate the runtime gate calls with no entry (CTest would run it only inside c-runtime)",
        lambda m: m["delegated"].pop("cpp-sycl"),
    ),
    (
        "M16",
        "an entry renamed away from the name the gate's SKIP line gives",
        lambda m: m["delegated"].__setitem__("cpp-sycl-oracle", m["delegated"].pop("cpp-sycl")),
    ),
)


def test_every_injected_violation_is_a_finding():
    tree = _tree()
    for code, what, mutate in _FAULTS:
        findings = manifest_tool.check(_mutated(mutate), tree)
        assert any(f.startswith(f"{code}:") for f in findings), (
            f"{code} ({what}) did not fire: {findings}"
        )


def _writer_with(manifest: dict, tmp: str):
    """The kernel writer reading `manifest` instead of the checkout's."""
    writer = _load_tool("emit_kernel")
    path = Path(tmp) / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    writer.MANIFEST = path
    return writer


def test_the_kernel_writer_appends_the_driver_to_the_emitters_text():
    """emit_kernel.py writes a kernel's unit -- the emitter's text, a newline, the driver -- and a
    depfile naming the emitter's modules and the driver; it refuses what M15 refuses (2), an
    emitter that raises is its FAIL (1), and --depfile without -o is a usage error."""
    from bcir.lower.c_kernel import emit_tree_predict_c

    writer = _load_tool("emit_kernel")
    driver = _ROOT / "runtime" / "c" / "kernels" / "tree.c"
    with tempfile.TemporaryDirectory() as tmp:
        unit, dep = Path(tmp) / "k" / "kernel_tree.c", Path(tmp) / "k" / "kernel_tree.c.d"
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = writer.main(["kernel_tree", "-o", str(unit), "--depfile", str(dep)])
        assert rc == 0, err.getvalue()
        want = emit_tree_predict_c(5, 2, "tree_p") + "\n" + driver.read_text(encoding="utf-8")
        assert unit.read_text(encoding="utf-8") == want
        rule = dep.read_text(encoding="utf-8")
        assert rule.startswith(f"{unit}:"), rule[:200]
        for made_from in (
            driver,
            _ROOT / "bcir" / "lower" / "c_kernel.py",
            _ROOT / "runtime" / "manifest.json",
        ):
            assert str(made_from) in rule, (made_from, rule)
        for argv, code, said in (
            (["kernel_nope"], 2, "no manifest kernel named 'kernel_nope'"),
            (["kernel_tree", "--depfile", str(dep)], 2, "pass -o"),
        ):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = writer.main(argv)
            assert rc == code and said in err.getvalue(), (argv, rc, err.getvalue())
        broken = copy.deepcopy(_manifest())
        broken["kernels"]["kernel_tree"]["args"] = []  # the emitter is called with none and raises
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = _writer_with(broken, tmp).main(["kernel_tree", "-o", str(unit)])
        assert rc == 1 and "emit_tree_predict_c raised TypeError" in err.getvalue(), (
            rc,
            err.getvalue(),
        )
        refused = copy.deepcopy(_manifest())
        refused["kernels"]["kernel_tree"]["emit"] = "emit_nothing_c"
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = _writer_with(refused, tmp).main(["kernel_tree"])
        assert rc == 2 and "M15:" in err.getvalue(), (rc, err.getvalue())


def test_the_gate_must_probe_before_it_shows_a_sanitizer_variants_section():
    """A gate that decides on the TSan section by a probe of its own can disagree with BCIR Make and
    the CMake build about the host (L12); asking tools/build/sanitizer.py is part of M14."""
    tree = manifest_tool.Tree()
    tree.runtime_gate_text = tree.runtime_gate_text.replace(
        manifest_tool.SANITIZER_PROBE, "tools/c/own_probe.sh"
    )
    findings = manifest_tool.check(_manifest(), tree)
    assert any(
        f.startswith("M14:") and "test_ring_tsan" in f and "sanitizer.py" in f for f in findings
    ), findings


def test_the_sanitizer_probe_answers_one_question_for_every_build():
    """sanitizer.probe: available only when a trivial -fsanitize program builds AND runs (the
    gate's predicate). A compiler that cannot run, one that refuses the flag, and a binary that
    builds but does not start are each unavailable with a reason (L1); the checker's sanitizer
    list is the probe's (one list, L14)."""
    assert manifest_tool.SANITIZERS == sanitizer_tool.NAMES
    ok, why = sanitizer_tool.probe(str(_ROOT / "no-such-compiler"), "thread")
    assert not ok and "could not be run" in why, why
    ok, why = sanitizer_tool.probe(sys.executable, "thread")
    assert not ok and "does not build a trivial program" in why, why
    try:
        sanitizer_tool.probe(sys.executable, "address")
    except ValueError as exc:
        assert "unknown sanitizer" in str(exc)
    else:
        raise AssertionError("an unknown sanitizer was probed")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert sanitizer_tool.main(["--cc", sys.executable, "address"]) == 2
        assert sanitizer_tool.main(["--cc", sys.executable, "thread"]) == 1
    assert "UNUSABLE" in out.getvalue() and "UNAVAILABLE" in out.getvalue(), out.getvalue()
    if os.name != "posix":
        return  # the fake compiler below is an executable script
    with tempfile.TemporaryDirectory() as tmp:
        for status, expected in ((3, False), (0, True)):
            fake = Path(tmp) / f"fakecc{status}"
            fake.write_text(
                f"#!{sys.executable}\n"
                "import os, sys\n"
                "out = sys.argv[sys.argv.index('-o') + 1]\n"
                f"open(out, 'w').write('#!/bin/sh\\nexit {status}\\n')\n"
                "os.chmod(out, 0o755)\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            ok, why = sanitizer_tool.probe(str(fake), "thread")
            assert ok is expected, (status, why)
            if not expected:
                assert "built but did not run" in why and "status 3" in why, why


def test_the_applier_edits_once_or_refuses():
    """mutate.py writes the mutant only when its anchor occurs exactly once and the edit changes the
    source; an anchor that is gone or that repeats, or a no-op edit, is exit 1 with nothing
    written; an unknown variant or a variant without a mutation is exit 2."""
    manifest = _manifest()
    mutation = manifest["variants"]["test_kplan_mutant"]["mutation"]
    original = (_ROOT / "runtime" / "c" / mutation["file"]).read_bytes()
    mutant = mutate_tool.mutant_bytes(manifest, "test_kplan_mutant")
    assert mutant == original.replace(mutation["find"].encode(), mutation["replace"].encode(), 1)
    assert mutant != original and mutation["find"].encode() not in mutant
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        out = work / "mutant.c"
        assert mutate_tool.main(["--variant", "test_kplan_mutant", "--out", str(out)]) == 0
        assert out.read_bytes() == mutant
        cases = {
            "gone": ("no such law", 1),
            "repeats": ("return", 1),
            "no-op": (None, 1),
        }
        for label, (find, expected) in cases.items():
            broken = copy.deepcopy(manifest)
            edit = broken["variants"]["test_kplan_mutant"]["mutation"]
            if find is None:
                edit["replace"] = edit["find"]
            else:
                edit["find"] = find
            path = work / f"{label}.json"
            path.write_text(json.dumps(broken), encoding="utf-8")
            target = work / f"{label}.c"
            with contextlib.redirect_stderr(io.StringIO()):
                rc = mutate_tool.main(
                    [
                        "--variant",
                        "test_kplan_mutant",
                        "--out",
                        str(target),
                        "--manifest",
                        str(path),
                    ]
                )
            assert rc == expected and not target.exists(), (label, rc)
        for variant in ("test_nowhere", "test_kplan_O0"):
            with contextlib.redirect_stderr(io.StringIO()):
                rc = mutate_tool.main(["--variant", variant, "--out", str(work / "x.c")])
            assert rc == 2 and not (work / "x.c").exists(), (variant, rc)


def test_a_varying_value_is_masked_only_where_declared():
    """mask(): only the capture group of a declared pattern is replaced, on the lines it matches; a
    pattern that matches nothing is reported stale."""
    output = b"  PASS rows (violations=0)\n  PASS fires (fault: violations=3)\n"
    masked, stale = section_tool.mask(output, [r"fault: violations=([0-9]+)", r"never=([0-9]+)"])
    assert masked == b"  PASS rows (violations=0)\n  PASS fires (fault: violations=<varies>)\n", (
        masked
    )
    assert stale == [r"never=([0-9]+)"], stale


def test_a_compiler_only_section_must_run_the_compiler():
    """A section that takes no binary judges what the oracle emits under CC; a script that never
    runs ${CC} would judge nothing at all, and pass (L2)."""
    tree = manifest_tool.Tree()
    assert '"${CC}"' in tree.section_texts["barrier.sh"]
    tree.section_texts["barrier.sh"] = tree.section_texts["barrier.sh"].replace('"${CC}"', '"cc"')
    findings = manifest_tool.check(_manifest(), tree)
    assert any(
        f.startswith("M13:") and "sections/barrier" in f and "never runs" in f for f in findings
    ), findings


def test_every_delegated_gate_runs_once_under_ctest():
    """The runtime gate skips the gates it delegates to when CTest's c-runtime entry sets
    BCIR_SKIP_DELEGATED_GATES=1, and CMake registers each as its own entry: a call outside the
    guard runs twice, a guard naming another entry sends the log's reader to the wrong place, a
    second call or a missing switch on the CMake side breaks the once, and a missing registration
    loses the gate from ctest altogether (L12). Each is an M16 finding."""
    gate_rel = "tools/cpp/check_sycl.sh"
    faults = {
        "the guard removed": lambda g: g.replace(
            'if [ "${BCIR_SKIP_DELEGATED_GATES:-0}" = "1" ]; then\n'
            '  echo "  SKIP SYCL backend differential oracle (BCIR_SKIP_DELEGATED_GATES=1; '
            'CTest runs it as cpp-sycl)"\nelif bash',
            "if bash",
        ),
        "the SKIP line naming another entry": lambda g: g.replace(
            "CTest runs it as cpp-sycl)", "CTest runs it as cpp-handoff)"
        ),
        "a second call": lambda g: g + f'\nbash "${{ROOT}}/{gate_rel}"\n',
        "an unregistered script": lambda g: g + '\nbash "${ROOT}/tools/c/check_other.sh"\n',
        "the sanitizer's own guard removed": lambda g: g.replace(
            'if [ "${BCIR_SKIP_CFRONT_SANITIZE:-0}" = "1" ]; then', "if false; then"
        ),
    }
    clean = _tree()
    assert manifest_tool.check(_manifest(), clean) == []
    for what, fault in faults.items():
        tree = copy.copy(clean)  # a Tree is read once; each fault edits its own copy's text
        tree.runtime_gate_text = fault(clean.runtime_gate_text)
        assert tree.runtime_gate_text != clean.runtime_gate_text, f"{what}: the fault did not apply"
        findings = manifest_tool.check(_manifest(), tree)
        assert any(f.startswith("M16:") for f in findings), f"{what} is no finding: {findings}"
    cmake_faults = {
        "c-runtime without the switch": lambda c: c.replace('"BCIR_SKIP_DELEGATED_GATES=1"', ""),
        "the registration loop removed": lambda c: c.replace(
            "IN LISTS BCIR_MANIFEST_delegated", "IN LISTS BCIR_MANIFEST_other"
        ),
    }
    for what, fault in cmake_faults.items():
        tree = copy.copy(clean)
        tree.tests_cmake_text = fault(clean.tests_cmake_text)
        assert tree.tests_cmake_text != clean.tests_cmake_text, f"{what}: the fault did not apply"
        findings = manifest_tool.check(_manifest(), tree)
        assert any(f.startswith("M16:") for f in findings), f"{what} is no finding: {findings}"


def test_the_gate_shows_every_section_once_and_builds_through_bcir_make():
    """BUILD-8: BCIR Make runs every section as a task, and the runtime gate shows each recorded
    verdict once. A section the gate stopped showing would still pass as a CTest entry while the
    gate judged less (L12); one shown twice, an unknown name, a section script the gate runs itself
    (a second run beside the task's), a gate that no longer builds through BCIR Make with the
    sections planned, and one without a guard that keeps an earlier run's verdict (on disk after
    a law break or a failed build) from being shown as this run's are each an M13 finding."""
    clean = _tree()
    assert manifest_tool.check(_manifest(), clean) == []
    faults = {
        "a section not shown": (
            lambda g: g.replace("show_section runtime || exit 1", ""),
            "shows section runtime 0 time(s)",
        ),
        "a section shown twice": (
            lambda g: g + "\nshow_section runtime || exit 1\n",
            "shows section runtime 2 time(s)",
        ),
        "an unknown name shown": (
            lambda g: g + "\nshow_section nowhere || exit 1\n",
            "shows 'nowhere', which is no manifest section",
        ),
        "a section script run by the gate": (
            lambda g: g + '\nbash "${ROOT}/tools/c/sections/runtime.sh" x || exit 1\n',
            "runs tools/c/sections/runtime.sh itself",
        ),
        "the sections not planned": (
            lambda g: g.replace('--cc "${CC}" --sections', '--cc "${CC}"'),
            "does not build through BCIR Make",
        ),
        "no BCIR Make run": (
            lambda g: g.replace("python3 -m bcir.make -f", "python3 -m bcir.other -f"),
            "does not build through BCIR Make",
        ),
        # an earlier run's verdicts lie on disk; each guard keeps the gate from showing them
        "the laws not asked": (
            lambda g: g.replace("grep -cx 'laws: PASS (MK0-MK5)'", "grep -cx 'laws: PASS'"),
            "could show an earlier run's verdict",
        ),
        "a failed build not stopping the gate": (
            lambda g: g.replace("grep -E '^(failed|skipped) '", "grep -E '^(failed) '"),
            "could show an earlier run's verdict",
        ),
        "an unplanned section's verdict shown": (
            lambda g: g.replace('grep -Fxq "target section.$1"', "true"),
            "could show an earlier run's verdict",
        ),
    }
    for what, (fault, needle) in faults.items():
        tree = copy.copy(clean)
        tree.runtime_gate_text = fault(clean.runtime_gate_text)
        assert tree.runtime_gate_text != clean.runtime_gate_text, f"{what}: the fault did not apply"
        findings = manifest_tool.check(_manifest(), tree)
        assert any(f.startswith("M13:") and needle in f for f in findings), (what, findings)


def test_the_section_parity_gate_sees_a_differing_output():
    """compare_section: two builds whose section outputs differ are a finding; identical passing
    output is parity; identical output without a PASS line is not a pass (L2). Without a probed
    POSIX shell (a Windows runner's `bash` is the WSL launcher) the gate refuses rather than
    judges, and that refusal is what this host is held to."""
    shell = section_tool.posix_shell()
    if shell is None:
        try:
            section_tool.compare_section(Path("none.sh"), [], [], python=sys.executable, timeout=30)
        except RuntimeError as exc:
            assert "POSIX shell" in str(exc)
        else:
            raise AssertionError("compare_section ran without a shell")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = section_tool.main(
                ["--harness-dir", str(_ROOT / "runtime" / "c"), "--cc", sys.executable]
            )
        assert rc == 2 and "no POSIX shell" in out.getvalue(), (rc, out.getvalue())
        return
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        script = work / "section.sh"
        script.write_text(
            '#!/usr/bin/env bash\nset -uo pipefail\ncat "$1"\n[ -z "${2:-}" ] || cat "$2"\n',
            encoding="utf-8",
        )
        a = work / "a"
        b = work / "b"
        a.write_text("  PASS one\n", encoding="utf-8")
        b.write_text("  PASS one\n", encoding="utf-8")
        same = section_tool.compare_section(
            script, [a], [b], python=sys.executable, timeout=30, shell=shell
        )
        assert same["differ"] == [] and same["passed"], same
        b.write_text("  PASS two\n", encoding="utf-8")
        diff = section_tool.compare_section(
            script, [a], [b], python=sys.executable, timeout=30, shell=shell
        )
        assert diff["differ"] == ["stdout"] and diff["passed"], diff
        # a section that compiles what a tool emits gets the one compiler, on both runs
        cc_script = work / "cc.sh"
        cc_script.write_text(
            '#!/usr/bin/env bash\necho "  PASS compiled with ${CC:-unset}"\n', encoding="utf-8"
        )
        named = section_tool.compare_section(
            cc_script, [a], [b], python=sys.executable, timeout=30, shell=shell, cc="cc-under-test"
        )
        assert named["differ"] == [] and named["passed"], named
        for run in (named["make"], named["cmake"]):
            assert run[1] == b"  PASS compiled with cc-under-test\n", run
        a.write_text("nothing judged\n", encoding="utf-8")
        b.write_text("nothing judged\n", encoding="utf-8")
        vacuous = section_tool.compare_section(
            script, [a], [b], python=sys.executable, timeout=30, shell=shell
        )
        assert vacuous["differ"] == [] and not vacuous["passed"], vacuous
        failing = work / "failing.sh"
        failing.write_text('#!/usr/bin/env bash\necho "  FAIL: x"; exit 1\n', encoding="utf-8")
        varying = work / "varying.sh"
        varying.write_text(
            "#!/usr/bin/env bash\n"
            'n=$(cat "$(dirname "$0")/count" 2>/dev/null || echo 0); n=$((n + 1))\n'
            'echo "$n" > "$(dirname "$0")/count"\n'
            'echo "  PASS fires (fault: violations=$n)"\n',
            encoding="utf-8",
        )
        undeclared = section_tool.compare_section(
            varying, [a], [b], python=sys.executable, timeout=30, shell=shell
        )
        assert undeclared["differ"] == ["stdout"] and undeclared["unstable"], undeclared
        declared = section_tool.compare_section(
            varying,
            [a],
            [b],
            python=sys.executable,
            timeout=30,
            shell=shell,
            varies=[r"fault: violations=([0-9]+)"],
        )
        assert declared["differ"] == [] and declared["stale"] == [] and declared["passed"], declared
        stale = section_tool.compare_section(
            script,
            [a],
            [b],
            python=sys.executable,
            timeout=30,
            shell=shell,
            varies=[r"never=([0-9]+)"],
        )
        assert stale["stale"] == [r"never=([0-9]+)"], stale
        failed = section_tool.compare_section(
            failing, [a], [b], python=sys.executable, timeout=30, shell=shell
        )
        assert failed["differ"] == [] and not failed["passed"] and failed["cmake"][0] == 1, failed
        # a compiler-only section takes no binary: its two runs must still agree and pass, so a
        # verdict that is not a function of its inputs is a finding there too
        alone = section_tool.compare_section(
            cc_script, [], [], python=sys.executable, timeout=30, shell=shell, cc="cc-under-test"
        )
        assert alone["differ"] == [] and alone["passed"], alone
        drifting = section_tool.compare_section(
            varying, [], [], python=sys.executable, timeout=30, shell=shell
        )
        assert drifting["differ"] == ["stdout"] and drifting["unstable"], drifting


def test_a_bash_that_is_not_a_shell_is_not_a_shell():
    """posix_shell() probes: a `bash` that prints a notice and exits 1 (the Windows WSL launcher)
    or exits 0 without saying ok is no shell; a real one is returned as given."""
    with tempfile.TemporaryDirectory() as tmp:
        stub = Path(tmp) / "bash"
        stub.write_text(
            "#!/bin/sh\necho 'Windows Subsystem for Linux has no installed distributions.'\nexit 1\n",
            encoding="utf-8",
        )
        stub.chmod(0o755)
        saved = os.environ.get("BCIR_SHELL")
        os.environ["BCIR_SHELL"] = str(stub)
        try:
            assert section_tool.posix_shell() is None
            silent = Path(tmp) / "quiet"
            silent.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            silent.chmod(0o755)
            os.environ["BCIR_SHELL"] = str(silent)
            assert section_tool.posix_shell() is None
            real = shutil.which("sh")
            if real is not None:
                os.environ["BCIR_SHELL"] = real
                assert section_tool.posix_shell() == real
        finally:
            if saved is None:
                os.environ.pop("BCIR_SHELL", None)
            else:
                os.environ["BCIR_SHELL"] = saved


def test_the_section_parity_gate_refuses_before_it_compiles():
    """No harness binaries, an unknown section name: exit 2, never a pass, nothing compiled."""
    with tempfile.TemporaryDirectory() as tmp:
        empty = Path(tmp)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = section_tool.main(["--harness-dir", str(empty), "--cc", sys.executable])
        assert rc == 2 and "UNUSABLE" in out.getvalue(), (rc, out.getvalue())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = section_tool.main(
                ["--harness-dir", str(empty), "--cc", sys.executable, "--section", "nowhere"]
            )
        assert rc == 2 and "no manifest section" in out.getvalue(), (rc, out.getvalue())
        if section_tool.posix_shell() is not None:
            # a section that takes a tool, and no tool directory to find the CMake build's copy in
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = section_tool.main(
                    ["--harness-dir", str(empty), "--cc", sys.executable, "--section", "bcir_cc"]
                )
            assert rc == 2 and "pass --tool-dir" in out.getvalue(), (rc, out.getvalue())
        assert not list(empty.iterdir())


def test_an_unbuilt_variant_is_reported_by_name_never_compared_or_hidden():
    """The CMake build passes the variants it could not build (`--unbuilt VARIANT=REASON`): their
    sections are reported as not compared, with the reason; a malformed or unknown entry is
    unusable, and a run left with nothing to compare is vacuous, not a pass (L1, L2)."""
    base = ["--harness-dir", str(_ROOT / "runtime" / "c"), "--cc", sys.executable]
    for extra, code, needle in (
        (["--unbuilt", "test_nowhere=no reason"], 2, "UNUSABLE"),
        (["--unbuilt", "test_ring_tsan="], 2, "UNUSABLE"),
        (["--section", "ring_tsan", "--unbuilt", "test_ring_plain=no runtime"], 2, "INVALID"),
    ):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = section_tool.main(base + extra)
        assert rc == code and needle in out.getvalue(), (extra, rc, out.getvalue())


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
    """A scanner that matched nothing would make M8-M11 vacuous (L2). BUILD-2 moved the runtime
    gate's section text into the section scripts, which are scanned as gate text, and BUILD-8
    retired the gate's compile lines (BCIR Make builds from the manifest), so the runtime gate
    itself names no source -- a compile line put back would show here -- and what moved must still
    have been examined."""
    tree = _tree()
    assert tree.missing_gates == [], tree.missing_gates
    runtime_groups = tree.gates["tools/c/check_runtime.sh"]
    section_groups = [
        group
        for rel, groups in tree.gates.items()
        if rel.startswith(manifest_tool.SECTIONS_DIR + "/")
        for group in groups
    ]
    every_group = [group for groups in tree.gates.values() for group in groups]
    with_main = [g for g in every_group if any(t.startswith("test_") for t in g[1])]

    assert runtime_groups == [], runtime_groups[:3]
    assert len(section_groups) >= 150, len(section_groups)
    assert len(every_group) >= 230, len(every_group)
    assert len(with_main) >= 15, len(with_main)
    assert tree.fuzz is not None and len(tree.fuzz) == len(_manifest()["fuzzers"]), tree.fuzz
    assert {"bcir_runtime.c", "bcir_kplan.c", "bcir_jer.c", "bcir_per.c"} <= tree.freestanding, (
        tree.freestanding
    )
    assert len(tree.python) >= 40, len(tree.python)
    assert all(lists is not None for lists in tree.seam.values()), tree.seam


def test_a_section_script_is_scanned_as_gate_text():
    """A section script is gate text moved out of tools/c/check_runtime.sh, so M9 reads it as it read
    the gate: a unit a section compiles is held to the manifest as the gate's own lines were (L15).
    The bcir-cc sections compile what the tool emits against runtime units, and the scan must have
    found those lines, or M9 is vacuous over them (L2)."""
    tree = manifest_tool.Tree()  # its own: the fault below must not reach the shared tree
    rel = "tools/c/sections/recover.sh"
    assert rel in tree.gates and rel not in tree.missing_gates, sorted(tree.gates)
    named = {token for _tag, tokens in tree.gates[rel] for token in tokens}
    assert {"bcir_quarantine.c", "bcir_quarantine_recover.c"} <= named, named
    # a section's compile line that links a harness main with a unit outside its closure (two
    # names, never one literal: M10 holds this file's literals to the manifest too)
    main = "test_runtime.c"
    outside = "bcir_quarantine.c"
    tree.gates[rel] = [*tree.gates[rel], ("line 1", [main, outside])]
    findings = manifest_tool.check(_manifest(), tree)
    assert any(
        f.startswith("M9:") and rel in f and "bcir_quarantine.c" in f and "outside" in f
        for f in findings
    ), findings


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


def test_a_fixture_is_held_to_what_the_gate_links():
    """bcir/tests/gate_links.py, which the #719 tests read the gate's link list through now that
    the gate has none of its own (BUILD-8), refuses both drifts it exists for (L11): a fixture
    compiling a unit the closure lacks -- the `bcir_oer.c` case -- and a fixture that builds
    without the unit's main. Its own list passes."""
    from bcir.tests import planner_fixtures as pf
    from bcir.tests.gate_links import assert_fixture_links, manifest_links

    own, linked = manifest_links("harnesses", "test_kplan")
    assert own == {"test_kplan.c"} and {"bcir_kplan.c", "bcir_runtime.c"} <= linked, linked
    assert_fixture_links("harnesses", "test_kplan", (*pf.C_UNITS, pf.HARNESS))
    for fixture, refusal in (
        ((*pf.C_UNITS, pf.HARNESS, "bcir_oer.c"), "compiles ['bcir_oer.c']"),
        (pf.C_UNITS, "without its main ['test_kplan.c']"),
    ):
        try:
            assert_fixture_links("harnesses", "test_kplan", fixture)
        except AssertionError as exc:
            assert refusal in str(exc), exc
        else:
            raise AssertionError(f"{fixture} passed")


def test_the_cmake_reader_and_the_checker_agree():
    """CMake reads the same manifest; the two readers must agree on its kinds and keys (L12)."""
    cmake = (_ROOT / "cmake" / "BCIRManifest.cmake").read_text(encoding="utf-8")
    assert "BCIR_MANIFEST_sections" in cmake and '"${_unit}" script' in cmake, (
        "the CMake reader does not read sections"
    )
    assert "BCIR_MANIFEST_variants" in cmake and "mutation file" in cmake, (
        "the CMake reader does not read variants"
    )
    assert "CMAKE_CONFIGURE_DEPENDS" in cmake, (
        "an edit to the manifest would not re-run the configure"
    )
    assert '"${_unit}" standard' in cmake and "sanitizer _san" in cmake, (
        "the CMake reader does not read a variant's standard and sanitizer"
    )
    deps = (_ROOT / "cmake" / "BCIRDependencies.cmake").read_text(encoding="utf-8")
    assert manifest_tool.SANITIZER_PROBE in deps and "BCIR_REQUIRE_TSAN" in deps, (
        "the configure must ask the gate's sanitizer predicate, and own its absence when required"
    )
    tests = (_ROOT / "cmake" / "BCIRTests.cmake").read_text(encoding="utf-8")
    assert "BCIR_UNBUILT_VARIANTS" in tests and "--unbuilt" in tests, (
        "the section-parity gate is not told which variants this tree did not build"
    )
    assert '"CC=${CMAKE_C_COMPILER}"' in tests and "--tool-dir" in tests, (
        "the section entries need the configured compiler, the parity gate the tools' directory"
    )
    assert (
        "BCIR_MANIFEST_kernels" in cmake
        and '"${_unit}" main' in cmake
        and '"${_unit}" link' in cmake
    ), "the CMake reader does not read the kernels' drivers and libraries"
    assert '"${_unit}" compiler_only' in cmake, (
        "the CMake reader does not read a section's compiler_only, so it would refuse one or accept "
        "an empty section"
    )
    assert (
        "BCIR_MANIFEST_delegated" in cmake
        and '"${_unit}" script _dscript' in cmake
        and '"${_unit}" labels' in cmake
    ), "the CMake reader does not read the delegated gates"
    assert "IN LISTS BCIR_MANIFEST_delegated" in tests and "BCIR_SKIP_DELEGATED_GATES=1" in tests, (
        "CMake must register every delegated gate and let c-runtime skip them"
    )
    runtime = (_ROOT / "runtime" / "c" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert manifest_tool.KERNEL_WRITER in runtime and "DEPFILE" in runtime, (
        "the CMake build must write each kernel with the gate's writer, and rebuild it from its depfile"
    )
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


def test_the_dependency_index_is_held_to_its_schema():
    """The configure's bcir-deps.json: well-formed passes; the defect that shipped first (CMake's ON
    in place of JSON's true), a missing schema and an empty row list are each a finding."""
    good = {
        "schema": "bcir-deps.v1",
        "c_compiler": "GNU 13.3.0",
        "cxx_compiler": "GNU 13.3.0",
        "system": "Linux x86_64",
        "dependencies": [
            {"name": "PYTHON3", "found": True, "detail": " (3.11 at /usr/bin/python3)"},
            {"name": "THREADS", "found": True, "detail": ""},
            {"name": "MLIR", "found": False, "detail": " (set MLIR_DIR)"},
            {"name": "FFTW3F", "found": True, "detail": " (links)", "link": ["-lfftw3f"]},
            {"name": "LAPACKE", "found": False, "detail": " (absent)", "link": []},
            {"name": "GSL", "found": False, "detail": " (absent)", "link": []},
            {"name": "SLEEF", "found": False, "detail": " (absent)", "link": []},
            {"name": "CERF", "found": False, "detail": " (absent)", "link": []},
        ],
        "tools": [
            {"name": n, "path": f"/usr/bin/{n}", "identity": "sha256:" + "0" * 64}
            for n in ("cc", "cxx", "ar", "python")
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bcir-deps.json"
        path.write_text(json.dumps(good), encoding="utf-8")
        assert manifest_tool.check_deps_index(path) == []
        # an optional library's row is the one a harness reads in place of its own probe (BUILD-4):
        # missing, flagless while found, or flagged while absent, it is a finding
        for fault, what in (
            (
                [r for r in good["dependencies"] if r["name"] != "LAPACKE"],
                "LAPACKE is not recorded",
            ),
            (
                [{**r, "link": []} if r["name"] == "FFTW3F" else r for r in good["dependencies"]],
                "FFTW3F is found=True",
            ),
            (
                [
                    {**r, "link": ["-lgsl"]} if r["name"] == "GSL" else r
                    for r in good["dependencies"]
                ],
                "GSL is found=False",
            ),
        ):
            path.write_text(json.dumps({**good, "dependencies": fault}), encoding="utf-8")
            findings = manifest_tool.check_deps_index(path)
            assert any(what in f for f in findings), (what, findings)
        path.write_text(json.dumps(good), encoding="utf-8")
        path.write_text(json.dumps(good).replace("true", "ON"), encoding="utf-8")
        findings = manifest_tool.check_deps_index(path)
        assert findings and all(f.startswith("D1:") for f in findings), findings
        path.write_text(json.dumps({**good, "schema": "other"}), encoding="utf-8")
        assert any("schema" in f for f in manifest_tool.check_deps_index(path))
        path.write_text(json.dumps({**good, "dependencies": []}), encoding="utf-8")
        assert any("empty" in f for f in manifest_tool.check_deps_index(path))
        dup = {**good, "dependencies": good["dependencies"] + [good["dependencies"][0]]}
        path.write_text(json.dumps(dup), encoding="utf-8")
        assert any("twice" in f for f in manifest_tool.check_deps_index(path))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = manifest_tool.main(["--deps-index", str(Path(tmp) / "absent.json")])
        assert rc == 1 and "D1:" in out.getvalue(), (rc, out.getvalue())
    cmake = (_ROOT / "cmake" / "BCIRTests.cmake").read_text(encoding="utf-8")
    assert "--deps-index" in cmake, "ctest does not validate the index"


def test_one_predicate_answers_what_the_host_has():
    """The optional libraries are probed in one place, bcir.toolchain's (BUILD-4): the configure
    asks it, and so does every harness. A module that writes its own probe program -- a C string
    carrying one of the libraries' headers and a main -- is a second answer to the same question,
    the shape that let the FFTW probe pass on a library that lacks the symbol (laws.md L14). And
    the configure runs the predicate rather than a find_library of its own."""
    headers = ("<fftw3.h>", "<lapacke.h>", "<gsl/", "<sleef.h>", "<cerf.h>")
    owner = _ROOT / "bcir" / "toolchain.py"
    found: list[str] = []
    for tree in ("bcir", "tools"):
        for path in sorted((_ROOT / tree).rglob("*.py")):
            if path == owner or "__pycache__" in path.parts:
                continue
            try:
                module = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, SyntaxError):
                continue
            for node in ast.walk(module):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and "main(" in node.value
                    and any(f"#include {h}" in node.value for h in headers)
                ):
                    found.append(f"{path.relative_to(_ROOT)}:{node.lineno}")
    assert found == [], f"library probes outside bcir.toolchain: {found}"
    deps = (_ROOT / "cmake" / "BCIRDependencies.cmake").read_text(encoding="utf-8")
    assert "-m bcir.toolchain probe-libraries" in deps, "the configure does not ask the predicate"
    assert "find_library" not in deps and "pkg_check_modules" not in deps, (
        "the configure answers the library question its own way"
    )
    tests = (_ROOT / "cmake" / "BCIRTests.cmake").read_text(encoding="utf-8")
    assert "BCIR_DEPS_INDEX=${CMAKE_BINARY_DIR}/bcir-deps.json" in tests, (
        "the Python entries do not read the tree's index"
    )


def test_the_ci_owns_the_build_gate():
    """The gate's CI owner (L2): a job configures with both compilers and runs the build label."""
    workflow = (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    job = re.search(r"^  cmake-build:\n(.*?)(?=^  [a-z0-9-]+:\n|\Z)", workflow, re.S | re.M)
    assert job is not None, "no cmake-build job"
    body = job.group(1)
    assert "gcc" in body and "clang" in body, "the job does not cover both compilers"
    assert re.search(r"ctest .*-L build", body), "the job does not run the build label"
    assert re.search(r"ctest .*-L section", body), "the job does not run the migrated sections"
    assert "BCIR_BUILD_MLIR=OFF" in body, (
        "the MLIR law has its own job; this one must not depend on a found package"
    )
    assert "BCIR_REQUIRE_TSAN=ON" in body and "libclang-rt" in body, (
        "the job installs the ThreadSanitizer runtime, so it must own its absence (L2)"
    )


def test_a_pinned_tool_whose_bytes_moved_is_a_finding():
    """BUILD-8's pins gate: a tool whose bytes are still its pin passes; one rewritten since the
    configure -- a compiler upgraded under a configured tree -- and one that is gone each fail it,
    naming the tool and saying reconfigure."""
    pins_tool = _load_tool("pins")
    with tempfile.TemporaryDirectory() as tmp:
        tool = Path(tmp) / "cc"
        tool.write_bytes(b"#!/bin/sh\nexit 0\n")
        pinned = "sha256:" + __import__("hashlib").sha256(tool.read_bytes()).hexdigest()
        assert pins_tool.drift({"cc": (str(tool), pinned)}) == []
        tool.write_bytes(b"#!/bin/sh\nexit 1\n")
        moved = pins_tool.drift({"cc": (str(tool), pinned)})
        assert len(moved) == 1 and moved[0].startswith("cc ") and "reconfigure" in moved[0], moved
        gone = pins_tool.drift({"cc": (str(Path(tmp) / "absent"), pinned)})
        assert len(gone) == 1 and "cannot be read" in gone[0], gone
