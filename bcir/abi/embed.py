"""The one C preprocessor probe for a C23 ``#embed`` of a generated blob (docs/security/laws.md L14).

``#embed`` is C23 (ISO/IEC 9899:2023 6.10.3). A compiler that implements it also offers it to earlier
language modes as an extension, with a C23-extension diagnostic that a strict C11 consumer's
``-Werror`` turns into an error: Clang 19 and later do, which the C11 CMake build of
``test_q8_tables`` found under Clang 23. So the probe admits ``#embed`` only in C23 mode and only when
the toolchain can find the blob; every other build takes the byte-identical fallback array, which is
also what "the fallback self-checks under C11" has to mean on every compiler.

``__has_embed`` is probed in a NESTED ``#if``: on a pre-``#embed`` toolchain it is not a defined macro,
and ``defined(__has_embed) && __has_embed(...)`` would still try to expand the right operand. The
nested form leaves it unparsed when unsupported. Both generators that bake a blob -- the frozen Q8
table (``bcir.abi.q8_tables``) and the large inference weight tables (``bcir.lower.inference``) --
spell the probe through this function, so the condition cannot drift between them.
"""

from __future__ import annotations

# __STDC_VERSION__ of ISO C23; an earlier standard (or C++, where it is undefined) takes the fallback.
C23_STDC_VERSION = "202311L"


def embed_probe(blob: str, macro: str) -> str:
    """Preprocessor lines defining ``macro`` to 1 when ``#embed "blob"`` may be used in this build:
    C23 mode, and a toolchain that finds the file. The caller's fallback defines it to 0."""
    if not blob or '"' in blob or "\n" in blob or not macro.isidentifier():
        raise ValueError(f"not a blob name and a C macro: {blob!r}, {macro!r}")
    return (
        f"#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= {C23_STDC_VERSION} && defined(__has_embed)\n"
        f'#  if __has_embed("{blob}")\n'
        f"#    define {macro} 1\n"
        f"#  endif\n"
        f"#endif\n"
    )
