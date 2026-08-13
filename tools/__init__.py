"""BCIR repository tooling.

The explicit package marker makes checkout-local ``tools.*`` imports win over
unrelated installed packages that use the generic ``tools`` name.  Repository
tooling is not part of the distributed ``bcir`` wheel; ``pyproject.toml``
packages only ``bcir*``.
"""
