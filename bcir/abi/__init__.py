"""The frozen BCIR StreamPack binary ABI (the portable artifact).

The StreamPack is BCIR's WASM-analog: a self-contained, portable, hot executable
artifact. `streampack_abi` defines a **versioned, frozen wire format** (v1) plus a
reference encoder/decoder. The same bytes are consumed by the Python oracle, the
(forthcoming) C runtime (`runtime/c/bcir_streampack.h`), and any embedder.
"""

from .streampack_abi import (
    ABI_MAGIC,
    ABI_VERSION,
    AbiError,
    StreamPackInspection,
    WireSpan,
    decode,
    encode,
    inspect_stream_pack,
)

__all__ = [
    "ABI_MAGIC", "ABI_VERSION", "AbiError", "StreamPackInspection", "WireSpan",
    "decode", "encode", "inspect_stream_pack",
]
