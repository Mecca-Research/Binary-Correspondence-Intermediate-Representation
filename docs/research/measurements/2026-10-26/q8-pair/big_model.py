"""Q8-PAIR's synthetic large decoder: a BCIRQ8 v1 artifact (docs/BCIR_LANGREF.md section 16) at TinyLlama-1.1B's widths
-- vocabulary 32,000, width 2,048, 32 query and 4 KV heads, FFN 5,632, group 32, tied embeddings -- with LAYERS layers
of deterministic pseudo-random codes (SHAKE-256 of the tensor's name; the code -128 taken as 0) and one exponent a
tensor kind, so that a decode's weights pass this host's L3 and the decoder reads them from memory a token. The file
is laid out as bcir.frontends.models.weights_io.write_q8_decoder lays one out; the C loader checks every CRC, span and
shape before a decode runs. Dependency-free.
usage: big_model.py OUT LAYERS  -> one JSON line: the file's size, tensors and SHA-256"""

import hashlib
import json
import struct
import sys
import zlib

if len(sys.argv) != 3:
    sys.exit(__doc__.rsplit("usage: ", 1)[1])
out, layers = sys.argv[1], int(sys.argv[2])
VOCAB, D, HEADS, KV_HEADS, FF, GROUP = 32000, 2048, 32, 4, 5632, 32
KVD = D // HEADS * KV_HEADS
HEADER = struct.Struct("<8sHHIIHBB6I Iiii dd II QQQ II")
ENTRY = struct.Struct("<HhBBHIIIIIIQQ")
NO_128 = bytes(range(128)) + b"\x00" + bytes(range(129, 256))


def align8(n):
    return (n + 7) // 8 * 8


def tensor(tid, layer, dims, exponent, name):
    """(id, layer, rank, dims, exponent bytes, code bytes): a matrix's codes pseudo-random, a norm's all 64 (1.0)"""
    count = dims[0] * dims[1]
    groups = (count + GROUP - 1) // GROUP
    if dims[1] == 1:
        codes = b"\x40" * count
    else:
        codes = (
            hashlib.shake_256(f"q8-pair big model {name}".encode()).digest(count).translate(NO_128)
        )
    return tid, layer, 1 if dims[1] == 1 else 2, dims, struct.pack("<h", exponent) * groups, codes


tensors = [tensor(1, -1, (VOCAB, D), -10, "embedding")]
for k in range(layers):
    tensors += [
        tensor(10, k, (D, 1), -6, f"{k}.g_attn"),
        tensor(11, k, (D, D), -12, f"{k}.w_q"),
        tensor(12, k, (D, KVD), -12, f"{k}.w_k"),
        tensor(13, k, (D, KVD), -12, f"{k}.w_v"),
        tensor(14, k, (D, D), -12, f"{k}.w_o"),
        tensor(15, k, (D, 1), -6, f"{k}.g_ff"),
        tensor(16, k, (D, FF), -12, f"{k}.w_gate"),
        tensor(17, k, (D, FF), -12, f"{k}.w_up"),
        tensor(18, k, (FF, D), -12, f"{k}.w_down"),
    ]
tensors.append(tensor(100, -1, (D, 1), -6, "g_final"))

directory_offset = 224
payload_offset = align8(directory_offset + len(tensors) * 48)
payload = bytearray()
entries = bytearray()
for tid, layer, rank, dims, exps, codes in tensors:
    payload += b"\0" * (align8(payload_offset + len(payload)) - payload_offset - len(payload))
    e_off = payload_offset + len(payload)
    payload += exps
    payload += b"\0" * (align8(payload_offset + len(payload)) - payload_offset - len(payload))
    c_off = payload_offset + len(payload)
    payload += codes
    crc = zlib.crc32(codes, zlib.crc32(exps)) & 0xFFFFFFFF
    entries += ENTRY.pack(
        tid,
        layer,
        rank,
        0,
        0,
        dims[0],
        dims[1],
        dims[0] * dims[1],
        len(exps) // 2,
        crc,
        0,
        e_off,
        c_off,
    )
entries += b"\0" * (payload_offset - directory_offset - len(entries))
body = bytes(entries) + bytes(payload)
size = payload_offset + len(payload)
digests = b"".join(
    hashlib.sha256(f"q8-pair big model {what}".encode()).digest()
    for what in ("model", "config", "tok")
)
header = bytearray(
    HEADER.pack(
        b"BCIRQ8\0\0",
        1,
        224,
        0x01020304,
        1,
        GROUP,
        8,
        0,
        VOCAB,
        D,
        HEADS,
        KV_HEADS,
        layers,
        FF,
        2048,
        1,
        -1,
        -1,
        10000.0,
        1e-5,
        len(tensors),
        48,
        directory_offset,
        payload_offset,
        size,
        zlib.crc32(body) & 0xFFFFFFFF,
        0,
    )
    + digests
    + b"\0" * 8
)
assert len(header) == 224
struct.pack_into("<I", header, 116, zlib.crc32(header) & 0xFFFFFFFF)
data = bytes(header) + body
assert len(data) == size
with open(out, "wb") as fh:
    fh.write(data)
print(
    json.dumps(
        {
            "layers": layers,
            "tensors": len(tensors),
            "bytes": size,
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    )
)
