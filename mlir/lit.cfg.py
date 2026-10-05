# -*- Python -*-
"""bcir-opt's lit suite (BUILD-5, docs/BCIR_BUILD_ROADMAP.md §9): every .mlir file under mlir/ --
the fixtures under `mlir/test` and `mlir/examples` -- runs its own RUN lines, the contract each one
states, against the bcir-opt this build made and the FileCheck and mlir-opt of the LLVM it was
built against. The two .mlir files that are another gate's input are excluded by name (NOT_LIT).

    python3 lit.py -sv -j 2 --param bcir_opt=PATH --param filecheck=PATH --param mlir_opt=PATH \\
        --param exec_root=DIR mlir

The `mlir-lit` CTest entry (cmake/BCIRTests.cmake) passes the four. A missing one, a path that is
not an executable, or three tools that are not one LLVM major is a configuration failure, never a
smaller suite or a mixed toolset (docs/security/laws.md L1, L2): the rail targets one coherent
major. The suite writes only under `exec_root`, never into the source tree.

`tools/wsl/check_passes.sh` runs the same fixtures with invocations of its own, and adds what a RUN
line cannot say (the mutated fixtures, the plan scores it greps for); this suite holds each fixture
to the RUN lines it carries, which until BUILD-5 nothing executed as written.

The test format is lit's own ShTest, never a class defined here: lit pickles every test, its config
and so its format, to hand it to a worker, and a class defined in this file is not importable by
name there -- the suite then dies before its first test (found by the first run on MLIR 23).
"""

import os
import re
import subprocess

import lit.formats

# The .mlir files under mlir/ that are not lit tests, by name (lit's `excludes`, which match a file
# or directory name at any depth): each is another gate's input and states no RUN line, and each
# name occurs once under mlir/ (bcir/tests/test_mlir_lit_suite.py), so it excludes nothing else.
NOT_LIT = {
    # the IRDL projection (mlir/irdl/) the irdl fixtures load and tools/irdl/check_corpus.sh checks
    "bcir.irdl.mlir",
    # tools/wsl/check_asm_lowering.sh pipes the lowered module through the real backend: FileCheck
    # alone is what let a lowering that does not assemble pass, so the file states no RUN line.
    "asm_lowering_smoke.mlir",
}
# param -> the name RUN lines call it by
TOOLS = {"bcir_opt": "bcir-opt", "filecheck": "FileCheck", "mlir_opt": "mlir-opt"}
_LLVM_VERSION = re.compile(rb"LLVM version (\d+)\.\d+")


def _major(path):
    """The LLVM major `path --version` reports, or None."""
    try:
        done = subprocess.run([path, "--version"], capture_output=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    found = _LLVM_VERSION.search(done.stdout + done.stderr)
    return int(found.group(1)) if found else None


config.name = "BCIR"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir"]
config.excludes = set(NOT_LIT)
config.test_source_root = os.path.dirname(os.path.abspath(__file__))
exec_root = lit_config.params.get("exec_root", "")
if not exec_root:
    lit_config.fatal(
        "--param exec_root=DIR is required: the suite writes nothing into the source tree"
    )
config.test_exec_root = os.path.abspath(exec_root)

majors = {}
for param, name in TOOLS.items():
    path = lit_config.params.get(param, "")
    if not path:
        lit_config.fatal(f"--param {param}=PATH is required (the {name} the RUN lines call)")
    if not (os.path.isfile(path) and os.access(path, os.X_OK)):
        lit_config.fatal(f"{param}: {path!r} is not an executable file")
    majors[name] = _major(path)
    if majors[name] is None:
        lit_config.fatal(f"{param}: {path!r} --version reports no LLVM version")
    # A RUN line names the tool as a command word; an option (`-bcir-optimize`) or a path segment
    # that contains the name is not one.
    config.substitutions.append(
        (rf"(?<![\w./-]){re.escape(name)}(?![\w.-])", os.path.abspath(path))
    )
if len(set(majors.values())) != 1:
    found = ", ".join(f"{name} {major}" for name, major in majors.items())
    lit_config.fatal(
        f"the tools are not one LLVM major ({found}): the rail targets one coherent major"
    )
lit_config.note(f"LLVM {majors['bcir-opt']}: " + ", ".join(lit_config.params[p] for p in TOOLS))
