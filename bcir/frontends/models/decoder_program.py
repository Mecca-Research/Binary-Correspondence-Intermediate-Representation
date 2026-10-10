"""QUAL-1: greedy generation by a dense decoder, as a BCIR program of its operations.

`decoder_program(spec, prompt_len, max_new)` states a generation as a `Module` with ONE CLAIM PER
DECODER OPERATION PER TOKEN: the embedding gather, both RMSNorms, the q/k/v, o, gate, up and
down projections, RoPE, the KV-cache append, causal attention, SwiGLU, the final norm, the
vocabulary head and the greedy argmax that writes the next token to the tape. Dependent
operations sit in successive phases; independent ones share a phase (q, k and v read the same
normed row; so do gate and up), which is the parallelism the plan may realize. The hazards are
the true data flow: every claim reads and writes exactly the resources its operation touches,
so the R-laws judge the program, K_BCIR prices it, and GEM hydrates it like any other program.

The program is EXECUTABLE. Its claims carry what an interpreter needs and nothing else: the
op string, the read and write RIDs, and the block each claim hydrates to (`offset`, `count`):

    dec.embed        rd (TOK, EMBED)            wr (X)            offset = tape position read
    dec.rmsnorm      rd (src, gamma)            wr (dst)
    dec.matvec       rd (src, W)                wr (dst)
    dec.rope         rd (Q, K)                  wr (QR, KR)       offset = position
    dec.kv_append    rd (KR, V)                 wr (KC_l, VC_l)   offset = position
    dec.attention    rd (QR, KC_l, VC_l)        wr (CTX)          offset = position
    dec.matvec_add   rd (src, W, X)             wr (X)            (projection + residual)
    dec.swiglu       rd (GATE, UP)              wr (FF)
    dec.head         rd (FINAL, EMBED|HEAD)     wr (LOGITS)
    dec.argmax       rd (LOGITS)                wr (TOK)          offset = tape slot written

`count` is the operation's work (multiply-adds, elements or rows), which is what K_BCIR prices.
The RID scheme below is an ABI shared with the C interpreter (`runtime/c/bcir_decoder_gem.c`):
it binds every RID it meets by these rules and refuses any other.

The claims declare `bounds="masked"` with a `bounds` verify contract: R7's single affine extent
cannot describe a claim that reads a row of one resource and a matrix of another, and every
position and row index is checked at run time by the interpreter instead (`verify="bounds"`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

from ...model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from .decode import DecoderSpec

# --- the RID scheme (ABI: runtime/c/bcir_decoder_gem.h mirrors every value) ------------------

#: Activations, one buffer each, reused by every layer and token.
RID_X, RID_H, RID_Q, RID_K, RID_V, RID_QR, RID_KR, RID_CTX = 1, 2, 3, 4, 5, 6, 7, 8
RID_H2, RID_GATE, RID_UP, RID_FF, RID_FINAL, RID_LOGITS = 9, 10, 11, 12, 13, 14
#: The token tape: the prompt, then each generated id, one int32 per position.
RID_TOK = 16
#: Global weights: the embedding table, the final norm's gain, an untied head.
RID_EMBED, RID_G_FINAL, RID_HEAD = 32, 33, 34
#: Per-layer resources: `LAYER_BASE + LAYER_STRIDE * layer + slot`.
LAYER_BASE, LAYER_STRIDE = 256, 16
SLOT_G_ATTN, SLOT_WQ, SLOT_WK, SLOT_WV, SLOT_WO = 0, 1, 2, 3, 4
SLOT_G_FF, SLOT_W_GATE, SLOT_W_UP, SLOT_W_DOWN = 5, 6, 7, 8
SLOT_K_CACHE, SLOT_V_CACHE = 9, 10
#: A bound the RID arithmetic and the interpreter's tables share.
MAX_LAYERS = 1024

OP_EMBED = "dec.embed"
OP_RMSNORM = "dec.rmsnorm"
OP_MATVEC = "dec.matvec"
OP_ROPE = "dec.rope"
OP_KV_APPEND = "dec.kv_append"
OP_ATTENTION = "dec.attention"
OP_MATVEC_ADD = "dec.matvec_add"
OP_SWIGLU = "dec.swiglu"
OP_HEAD = "dec.head"
OP_ARGMAX = "dec.argmax"
OPS = (
    OP_EMBED,
    OP_RMSNORM,
    OP_MATVEC,
    OP_ROPE,
    OP_KV_APPEND,
    OP_ATTENTION,
    OP_MATVEC_ADD,
    OP_SWIGLU,
    OP_HEAD,
    OP_ARGMAX,
)

#: Activations are held in double precision (the BCIRQ8 rails accumulate in double); weights
#: are BCIRQ8: one int8 code per element and one little-endian int16 exponent per group.
ACTIVATION_BYTES = 8
TOKEN_BYTES = 4
Q8_GROUP = 32


def layer_rid(layer: int, slot: int) -> int:
    if not 0 <= layer < MAX_LAYERS or not 0 <= slot < LAYER_STRIDE:
        raise ValueError(f"no RID for layer {layer} slot {slot}")
    return LAYER_BASE + LAYER_STRIDE * layer + slot


def q8_bytes(elements: int) -> int:
    """The octets a BCIRQ8 tensor of `elements` occupies: codes plus a 2-octet exponent a group."""
    return elements + 2 * ((elements + Q8_GROUP - 1) // Q8_GROUP)


@dataclass(frozen=True)
class DecoderProgram:
    """A generation as a program: the module, its shape, and its tape layout."""

    spec: DecoderSpec
    prompt_len: int
    max_new: int
    module: Module

    @property
    def capacity(self) -> int:
        """Tape positions: the prompt and every generated id."""
        return self.prompt_len + self.max_new

    @property
    def positions(self) -> int:
        """Positions the decoder steps through: every token but the last generated one."""
        return self.capacity - 1

    @property
    def claims(self) -> int:
        return sum(len(phase.claims) for phase in self.module.phases)


def _resources(spec: DecoderSpec, capacity: int) -> list[Resource]:
    d, kvd, ff, vocab = spec.d_model, spec.kv_heads * spec.d_k, spec.d_ff, spec.vocab_size

    def act(rid: int, n: int, name: str) -> Resource:
        return Resource(rid=rid, elem_bytes=ACTIVATION_BYTES, shape=(n,), name=name)

    def weight(rid: int, shape: tuple, name: str) -> Resource:
        return Resource(rid=rid, elem_bytes=1, shape=shape, name=name)

    out = [
        act(RID_X, d, "x"),
        act(RID_H, d, "h"),
        act(RID_Q, d, "q"),
        act(RID_K, kvd, "k"),
        act(RID_V, kvd, "v"),
        act(RID_QR, d, "q_rope"),
        act(RID_KR, kvd, "k_rope"),
        act(RID_CTX, d, "context"),
        act(RID_H2, d, "h2"),
        act(RID_GATE, ff, "gate"),
        act(RID_UP, ff, "up"),
        act(RID_FF, ff, "ff"),
        act(RID_FINAL, d, "final"),
        act(RID_LOGITS, vocab, "logits"),
        Resource(rid=RID_TOK, elem_bytes=TOKEN_BYTES, shape=(capacity,), name="tokens"),
        weight(RID_EMBED, (vocab, d), "embed"),
        weight(RID_G_FINAL, (d,), "g_final"),
    ]
    if not spec.tied_embeddings:
        out.append(weight(RID_HEAD, (vocab, d), "lm_head"))
    for layer in range(spec.n_layers):
        rows = (
            (SLOT_G_ATTN, (d,), "g_attn"),
            (SLOT_WQ, (d, d), "wq"),
            (SLOT_WK, (d, kvd), "wk"),
            (SLOT_WV, (d, kvd), "wv"),
            (SLOT_WO, (d, d), "wo"),
            (SLOT_G_FF, (d,), "g_ff"),
            (SLOT_W_GATE, (d, ff), "w_gate"),
            (SLOT_W_UP, (d, ff), "w_up"),
            (SLOT_W_DOWN, (ff, d), "w_down"),
        )
        for slot, shape, name in rows:
            out.append(weight(layer_rid(layer, slot), shape, f"l{layer}.{name}"))
        for slot, name in ((SLOT_K_CACHE, "k_cache"), (SLOT_V_CACHE, "v_cache")):
            out.append(act(layer_rid(layer, slot), capacity * kvd, f"l{layer}.{name}"))
    return out


class _Builder:
    def __init__(self, module: Module) -> None:
        self.module = module
        self.claim_id = 0
        self.phase_id = 0

    def phase(self, *claims: dict) -> None:
        built = []
        for fields in claims:
            built.append(Claim(id=self.claim_id, domain=Domain.RAM, bounds="masked", **fields))
            self.claim_id += 1
        deps = (self.phase_id - 1,) if self.phase_id else ()
        self.module.add_phase(Phase(phase_id=self.phase_id, deps=deps, claims=built))
        self.phase_id += 1


def _vector(op: str, rd: tuple, wr: tuple, count: int, *, offset: int = 0) -> dict:
    return dict(
        opcode=Opcode.MUL,
        lane=Lane.U,
        stride_class=StrideClass.UNIT,
        count=count,
        rd=rd,
        wr=wr,
        op=op,
        offset=offset,
        verify="bounds",
    )


def _tile(op: str, rd: tuple, wr: tuple, count: int, *, offset: int = 0) -> dict:
    return dict(
        opcode=Opcode.T_MACC,
        lane=Lane.T,
        stride_class=StrideClass.TILE,
        count=count,
        rd=rd,
        wr=wr,
        op=op,
        offset=offset,
        verify="bounds",
        cost_class="compute",
    )


def decoder_program(spec: DecoderSpec, prompt_len: int, max_new: int) -> DecoderProgram:
    """Greedy generation of `max_new` tokens after a `prompt_len`-token prompt, as a program.

    Position p embeds tape[p] and runs every layer; from the prompt's last position on, the
    final norm, the head and the argmax write tape[p + 1]. The last generated id is never
    embedded, so `prompt_len + max_new - 1` positions run -- the token-by-token prefill and
    the decode steps of `bcir_llama_generate_greedy`, operation for operation."""
    if not isinstance(spec, DecoderSpec):
        raise ValueError("decoder_program needs a DecoderSpec")
    if spec.activation != "silu_gate":
        raise ValueError("decoder_program lowers the Llama/SwiGLU decoder (activation='silu_gate')")
    if spec.n_layers > MAX_LAYERS:
        raise ValueError(f"at most {MAX_LAYERS} layers")
    if type(prompt_len) is not int or prompt_len < 1:
        raise ValueError("prompt_len must be an integer >= 1")
    if type(max_new) is not int or max_new < 1:
        raise ValueError("max_new must be an integer >= 1")
    capacity = prompt_len + max_new
    if capacity > 1 << 20:
        raise ValueError("a program holds at most 2^20 positions")
    d, dk = spec.d_model, spec.d_k
    kvd, ff, vocab = spec.kv_heads * dk, spec.d_ff, spec.vocab_size
    head_rid = RID_EMBED if spec.tied_embeddings else RID_HEAD
    module = Module(name=f"decoder_{spec.n_layers}l_{d}d_{prompt_len}p_{max_new}n")
    for resource in _resources(spec, capacity):
        module.add_resource(resource)
    b = _Builder(module)
    for pos in range(capacity - 1):
        b.phase(
            dict(
                opcode=Opcode.GGG_LOAD,
                lane=Lane.GGG,
                stride_class=StrideClass.RANDOM,
                count=d,
                rd=(RID_TOK, RID_EMBED),
                wr=(RID_X,),
                op=OP_EMBED,
                offset=pos,
                verify="bounds",
            )
        )
        for layer in range(spec.n_layers):
            lr = partial(layer_rid, layer)  # the layer's RID, by slot
            kc, vc = lr(SLOT_K_CACHE), lr(SLOT_V_CACHE)
            b.phase(_vector(OP_RMSNORM, (RID_X, lr(SLOT_G_ATTN)), (RID_H,), d))
            b.phase(
                _tile(OP_MATVEC, (RID_H, lr(SLOT_WQ)), (RID_Q,), d * d),
                _tile(OP_MATVEC, (RID_H, lr(SLOT_WK)), (RID_K,), d * kvd),
                _tile(OP_MATVEC, (RID_H, lr(SLOT_WV)), (RID_V,), d * kvd),
            )
            b.phase(_vector(OP_ROPE, (RID_Q, RID_K), (RID_QR, RID_KR), d + kvd, offset=pos))
            b.phase(_vector(OP_KV_APPEND, (RID_KR, RID_V), (kc, vc), 2 * kvd, offset=pos))
            b.phase(
                _tile(OP_ATTENTION, (RID_QR, kc, vc), (RID_CTX,), 2 * d * (pos + 1), offset=pos)
            )
            b.phase(_tile(OP_MATVEC_ADD, (RID_CTX, lr(SLOT_WO), RID_X), (RID_X,), d * d))
            b.phase(_vector(OP_RMSNORM, (RID_X, lr(SLOT_G_FF)), (RID_H2,), d))
            b.phase(
                _tile(OP_MATVEC, (RID_H2, lr(SLOT_W_GATE)), (RID_GATE,), d * ff),
                _tile(OP_MATVEC, (RID_H2, lr(SLOT_W_UP)), (RID_UP,), d * ff),
            )
            b.phase(_vector(OP_SWIGLU, (RID_GATE, RID_UP), (RID_FF,), ff))
            b.phase(_tile(OP_MATVEC_ADD, (RID_FF, lr(SLOT_W_DOWN), RID_X), (RID_X,), ff * d))
        if pos >= prompt_len - 1:
            b.phase(_vector(OP_RMSNORM, (RID_X, RID_G_FINAL), (RID_FINAL,), d))
            b.phase(_tile(OP_HEAD, (RID_FINAL, head_rid), (RID_LOGITS,), vocab * d))
            b.phase(
                dict(
                    opcode=Opcode.LOAD,
                    lane=Lane.U,
                    stride_class=StrideClass.UNIT,
                    count=vocab,
                    rd=(RID_LOGITS,),
                    wr=(RID_TOK,),
                    op=OP_ARGMAX,
                    offset=pos + 1,
                    verify="bounds",
                )
            )
    return DecoderProgram(spec, prompt_len, max_new, module)


def claim_traffic(spec: DecoderSpec, claim: Claim) -> tuple[int, int]:
    """(octets read, octets written) by one claim's operation, exactly as the C interpreter
    counts them: BCIRQ8 weights at a code an element and two exponent octets a group, activations and
    cache rows in double, the tape in int32. A projection reads its whole matrix; attention
    reads the `offset + 1` cached rows of both caches. The embedding row's exponents are its
    `d_model / 32` groups: exact when `d_model` is a multiple of the group (every BCIR model
    so far); otherwise the row's groups depend on the token, and the interpreter counts the
    ones it touches."""
    d, kvd, ff, vocab = spec.d_model, spec.kv_heads * spec.d_k, spec.d_ff, spec.vocab_size
    a, op = ACTIVATION_BYTES, claim.op
    if op == OP_EMBED:
        return TOKEN_BYTES + q8_bytes(d), d * a
    if op == OP_RMSNORM:
        return d * a + q8_bytes(d), d * a
    if op in (OP_MATVEC, OP_MATVEC_ADD):
        n_in = ff if claim.rd[0] == RID_FF else d
        n_out = claim.count // n_in
        residual = n_out * a if op == OP_MATVEC_ADD else 0
        return n_in * a + q8_bytes(n_in * n_out) + residual, n_out * a
    if op == OP_ROPE:
        return (d + kvd) * a, (d + kvd) * a
    if op == OP_KV_APPEND:
        return 2 * kvd * a, 2 * kvd * a
    if op == OP_ATTENTION:
        rows = claim.offset + 1
        return d * a + 2 * rows * kvd * a, d * a
    if op == OP_SWIGLU:
        return 2 * ff * a, ff * a
    if op == OP_HEAD:
        return d * a + q8_bytes(vocab * d), vocab * a
    if op == OP_ARGMAX:
        return vocab * a, TOKEN_BYTES
    raise ValueError(f"no traffic model for {op!r}")


def program_traffic(program: DecoderProgram) -> dict[str, tuple[int, int]]:
    """Planned traffic per op kind and in total: {op: (read, written)} over every claim."""
    totals: dict[str, list[int]] = {}
    for phase in program.module.phases:
        for claim in phase.claims:
            r, w = claim_traffic(program.spec, claim)
            row = totals.setdefault(claim.op, [0, 0])
            row[0] += r
            row[1] += w
    out = {op: (r, w) for op, (r, w) in sorted(totals.items())}
    out["total"] = (sum(r for r, _ in out.values()), sum(w for _, w in out.values()))
    return out
