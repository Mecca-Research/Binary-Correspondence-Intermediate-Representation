"""What the runtime gate links, read where it is decided (BUILD-8).

tools/c/check_runtime.sh spells no link line of its own any more: BCIR Make builds every harness
and tool a section runs from its runtime/manifest.json closure (tools/build/bcirfile.py) -- the
unit's own sources, then its libraries' archives. A Python fixture that builds the same binary from
a list of its own is held to that closure here, which is the #719 trap read from its new source:
`bcir_oer.c` joined a fixture's list and not the gate's, the fixture built, and the gate failed with
an undefined reference minutes in. Two halves, each a way the lists have drifted: the fixture
compiles the unit's own main, and every source it compiles is one the gate links.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterable

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def manifest_links(kind: str, name: str) -> tuple[set[str], set[str]] | None:
    """(the unit's own sources, every source its closure links), or None outside a source
    checkout -- the wheel ships neither tools/ nor runtime/."""
    from bcir.tests.run_all import _is_source_checkout

    if not _is_source_checkout():
        return None
    path = os.path.join(_ROOT, "tools", "build", "manifest.py")
    spec = importlib.util.spec_from_file_location("bcir_tools_build_manifest_links", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    manifest = module.load()
    own = set(manifest[kind][name]["sources"])
    return own, set(module.closure(manifest, kind, name)["sources"])


def assert_fixture_links(kind: str, name: str, fixture: Iterable[str]) -> None:
    """A fixture's source list builds what the gate builds as `kind/name`, and links nothing the
    gate does not."""
    found = manifest_links(kind, name)
    if found is None:
        return
    own, linked = found
    fixture = set(fixture)
    assert own <= fixture, f"the fixture builds {kind}/{name} without its main {sorted(own)}"
    outside = sorted(fixture - linked)
    assert not outside, (
        f"the fixture compiles {outside}, which {kind}/{name}'s manifest closure -- what the "
        "runtime gate links -- lacks"
    )
