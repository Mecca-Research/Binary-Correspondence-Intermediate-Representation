#!/usr/bin/env python3
"""Keep the training corpus's production-MLIR chapters honest about `mlir/`.

Chapters 08 and 09 of `18-mlir-lowering-to-llvm/` teach production dialect and
conversion-pass implementation by citing this repository's own law rail: exact
files, exact CMake idioms, exact MLIR APIs. Those citations are the chapter's
evidence, and evidence that nobody re-checks becomes prose.

This gate re-checks them. It needs no MLIR toolchain at all -- it reads the
rail's sources -- so it runs everywhere the corpus runs, including hosts where
the MLIR rail itself is an honest skip.

    python3 llvm-training/tools/verify-mlir-rail-references.py

It also enforces the negative half: idioms the rail deliberately no longer uses
(the pre-LLVM-23 spellings) must stay absent, so a chapter that teaches the
current API cannot quietly start describing a tree that regressed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = TRAINING_ROOT.parent
MLIR_ROOT = REPO_ROOT / "mlir"

CHAPTERS = (
    TRAINING_ROOT / "18-mlir-lowering-to-llvm" / "08-production-dialect-and-build-integration.md",
    TRAINING_ROOT / "18-mlir-lowering-to-llvm" / "09-production-conversion-pass.md",
)

# Files the chapters cite by path. A citation to a file that moved is a broken
# lesson, not just a broken link.
CITED_PATHS = (
    "mlir/CMakeLists.txt",
    "mlir/README.md",
    "mlir/include/BCIR/BCIRCoreOps.td",
    "mlir/include/BCIR/BCIROps.td",
    "mlir/include/BCIR/BCIRAttrs.td",
    "mlir/include/BCIR/BCIRTypes.td",
    "mlir/include/BCIR/BCIRInterfaces.td",
    "mlir/include/BCIR/BCIRDialect.td",
    "mlir/lib/BCIRDialect.cpp",
    "mlir/lib/BCIRPasses.cpp",
    "mlir/lib/passes/BCIRConvertToLLVM.cpp",
    "mlir/lib/passes/BCIRVerifyPass.cpp",
    "mlir/lib/passes/BCIRPromotePass.cpp",
    "mlir/tools/bcir-opt.cpp",
    "mlir/test/passes",
)

# (file, needle, why the chapter depends on it)
REQUIRED_IDIOMS = (
    ("mlir/CMakeLists.txt", "mlir_tablegen(", "chapter 08 teaches per-generator tablegen calls"),
    (
        "mlir/CMakeLists.txt",
        "add_public_tablegen_target(",
        "chapter 08 teaches grouping generators into one target",
    ),
    (
        "mlir/CMakeLists.txt",
        "add_mlir_dialect_library(",
        "chapter 08 teaches the dialect library target",
    ),
    (
        "mlir/CMakeLists.txt",
        "LLVM_TARGET_DEFINITIONS",
        "chapter 08 teaches that this is positional state",
    ),
    (
        "mlir/CMakeLists.txt",
        "${CMAKE_CURRENT_BINARY_DIR}",
        "chapter 08 teaches that generated .inc files live in the build tree",
    ),
    ("mlir/CMakeLists.txt", "-gen-dialect-decls", "chapter 08's generator-flag table"),
    ("mlir/CMakeLists.txt", "-gen-op-decls", "chapter 08's generator-flag table"),
    ("mlir/CMakeLists.txt", "-gen-typedef-decls", "chapter 08's generator-flag table"),
    ("mlir/CMakeLists.txt", "-gen-attrdef-decls", "chapter 08's generator-flag table"),
    ("mlir/CMakeLists.txt", "-gen-enum-decls", "chapter 08's generator-flag table"),
    ("mlir/CMakeLists.txt", "-gen-op-interface-decls", "chapter 08's generator-flag table"),
    (
        "mlir/include/BCIR/BCIRCoreOps.td",
        "defvar BCIR_ClaimAttrs",
        "chapter 08 quotes the shared attribute bundle",
    ),
    (
        "mlir/include/BCIR/BCIRCoreOps.td",
        "OptionalAttr<",
        "chapter 08 teaches non-disturbing attribute addition",
    ),
    (
        "mlir/include/BCIR/BCIRCoreOps.td",
        "DefaultValuedAttr<",
        "chapter 08 teaches non-disturbing attribute addition",
    ),
    ("mlir/tools/bcir-opt.cpp", "DialectRegistry", "chapter 08 quotes dialect registration"),
    ("mlir/tools/bcir-opt.cpp", "MlirOptMain", "chapter 08 quotes the tool entry point"),
    ("mlir/lib/BCIRPasses.cpp", "registerPass", "chapter 08 quotes pass registration"),
    (
        "mlir/lib/passes/BCIRConvertToLLVM.cpp",
        "OpConversionPattern",
        "chapter 09's pattern anatomy",
    ),
    (
        "mlir/lib/passes/BCIRConvertToLLVM.cpp",
        "LLVMTypeConverter",
        "chapter 09's TypeConverter section",
    ),
    ("mlir/lib/passes/BCIRConvertToLLVM.cpp", "ConversionTarget", "chapter 09's legality section"),
    ("mlir/lib/passes/BCIRConvertToLLVM.cpp", "RewritePatternSet", "chapter 09's skeleton"),
    (
        "mlir/lib/passes/BCIRConvertToLLVM.cpp",
        "applyPatternsGreedily",
        "chapter 09's greedy-rewriter section",
    ),
    (
        "mlir/lib/passes/BCIRPromotePass.cpp",
        "applyPatternsGreedily",
        "chapter 09's API-migration table",
    ),
)

# Spellings the rail migrated away from for LLVM 23. Chapter 09 states that the
# tree uses the new API; if an old spelling comes back, the chapter is wrong.
FORBIDDEN_IDIOMS = (
    ("applyPatternsAndFoldGreedily", "chapter 09 states the rail moved to applyPatternsGreedily"),
    ("builder.create<", "chapter 09 states the rail moved to OpTy::create(builder, ...)"),
    ("rewriter.create<", "chapter 09 states the rail moved to OpTy::create(rewriter, ...)"),
)

FORBIDDEN_SCAN_DIRS = ("mlir/lib", "mlir/include", "mlir/tools")


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def iter_sources(directory: Path):
    for suffix in (".cpp", ".h", ".td"):
        yield from directory.rglob(f"*{suffix}")


def main() -> int:
    failures: list[str] = []
    checks = 0

    if not MLIR_ROOT.is_dir():
        # The MLIR rail is part of this repository; its absence is a broken
        # checkout, not an optional-tool skip.
        print(
            f"MLIR rail reference gate: FAILED (missing {relpath(MLIR_ROOT)})",
            file=sys.stderr,
        )
        return 1

    for chapter in CHAPTERS:
        checks += 1
        if not chapter.is_file():
            failures.append(f"missing chapter: {relpath(chapter)}")

    for cited in CITED_PATHS:
        checks += 1
        target = REPO_ROOT / cited
        if not target.exists():
            failures.append(f"chapter cites '{cited}', which does not exist in this tree")

    for cited, needle, why in REQUIRED_IDIOMS:
        checks += 1
        target = REPO_ROOT / cited
        if not target.is_file():
            failures.append(f"cannot check '{needle}': {cited} is missing")
            continue
        if needle not in target.read_text(encoding="utf-8", errors="replace"):
            failures.append(f"{cited} no longer contains '{needle}' ({why})")

    for needle, why in FORBIDDEN_IDIOMS:
        checks += 1
        offenders: list[str] = []
        for directory in FORBIDDEN_SCAN_DIRS:
            root = REPO_ROOT / directory
            if not root.is_dir():
                continue
            for source in iter_sources(root):
                if needle in source.read_text(encoding="utf-8", errors="replace"):
                    offenders.append(relpath(source))
        if offenders:
            shown = ", ".join(sorted(offenders)[:5])
            failures.append(f"pre-LLVM-23 spelling '{needle}' reappeared in {shown} ({why})")

    # Every chapter link into ../../ must resolve, so a moved rail file is caught
    # even when it is not in the citation list above.
    link_pattern = re.compile(r"\]\((\.\./\.\./[^)#]+)\)")
    for chapter in CHAPTERS:
        if not chapter.is_file():
            continue
        for match in link_pattern.finditer(chapter.read_text(encoding="utf-8")):
            checks += 1
            target = (chapter.parent / match.group(1)).resolve()
            if not target.exists():
                failures.append(f"{relpath(chapter)}: link '{match.group(1)}' does not resolve")

    if failures:
        print("MLIR rail reference gate: FAILED", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"MLIR rail reference gate: PASSED ({checks} reference check(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
