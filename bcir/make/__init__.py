"""BCIR Make: the build as a BCIR program (BUILD-6..8, docs/BCIR_BUILD_ROADMAP.md §8).

A BCIRfile declares tools by recorded identity and targets as claims over files
(`bcir.make.grammar`). The laws (`bcir.make.laws`, MK0-MK5) lower it onto the IR -- files are
resources, targets are phases -- and judge it there with the verifier's own predicates; the planner
(`bcir.make.plan`) gives each target a generation tag from what it declares and schedules the
stale ones in the IR's canonical order under the two-worker cap. `python -m bcir.make --dry-run`
prints that plan; the runner that executes it is BUILD-7.
"""

from __future__ import annotations

from .grammar import BcirFile, GrammarError, Target, Tool, parse
from .laws import Finding, check, lower
from .plan import decide, generation_tags, schedule

__all__ = [
    "BcirFile",
    "Finding",
    "GrammarError",
    "Target",
    "Tool",
    "check",
    "decide",
    "generation_tags",
    "lower",
    "parse",
    "schedule",
]
