"""Opening the engine: one module loader, one way to open a catalog.

Every tool under `training/tools/` is run as a script from the repository root, so a
plain `import catalog` would resolve differently depending on how the tool was
invoked -- or collide with an unrelated module that happens to share the name. Each
tool used to answer that for itself, and the answers differed in ways that matter.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent.parent
CORPUS_ROOT = TOOLS_DIR.parent
REPO_ROOT = CORPUS_ROOT.parent


def load(name: str):
    """Load a sibling tool once, reusing the module object.

    The cache is keyed on the file the module came from, not on its name alone.
    Returning a same-named module from somewhere else would hand back different class
    objects, so an `except module.SomeError` would compare an instance of one module's
    class against another's, match nothing, and let the exception escape -- and
    executing a fresh module only to discard it for whatever was already cached does
    the same thing with extra work.
    """
    path = TOOLS_DIR / f"{name}.py"
    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "__file__", None) == str(path):
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"db.engine: no tool named {name!r} at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def catalog():
    """The catalog module -- the artifacts, their readers and their refusals."""
    return load("catalog")


def planner():
    """The plan module -- predicates, legality, and the twelve-axis cost model."""
    return load("plan")


def open_catalog(catalog_dir: Path, chunk_dir: Path, *, verify_content: bool = False):
    """Load a catalog, or raise the catalog module's own refusal.

    A thin pass-through, and deliberately not a fallback: a caller that cannot tell
    "absent" from "stale" will eventually treat one of them as "fine", which is the
    reason `Catalog.load` refuses in the first place.
    """
    return catalog().Catalog.load(catalog_dir, chunk_dir, verify_content=verify_content)
