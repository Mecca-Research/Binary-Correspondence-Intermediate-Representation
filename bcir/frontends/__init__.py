"""CT3: BCIR front-ends -- the assembly-perf <-> high-level bridge paradigms.

Both parse source text and emit BCIR claims, reusing `bcir.verify` + the GEM/K_BCIR
backend (a front-end produces *IR*, not host-language code):

- `rop`  -- a registry-first *declarative* form (module/resource/phase/claim).
- `map`  -- a terse *macro-assembly* form (one operation per line).
"""

from .map import parse_map
from .rop import parse_rop_program

__all__ = ["parse_map", "parse_rop_program"]
