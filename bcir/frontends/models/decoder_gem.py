"""QUAL-2: a decoder program's StreamPack, executed by the oracle -- the twin of
`runtime/c/bcir_decoder_gem.c`.

`run_decoder_pack(pack, spec, weights, prompt)` takes the StreamPack a decoder program
(`decoder_program.py`) hydrated to -- decoded from any of its projections: the native wire, DER,
OER or JER -- and executes it the way the C interpreter does. Claims dispatch in GEM's order
over the pack (phases by first appearance, claims ascending by id within a phase: the order
`bcir_exec.c` dispatches), and each claim runs the operation its op string names on the operands
its RIDs name. The kernels are the oracle's own -- `rmsnorm_reference`, `matmul_reference`,
`rope_reference`, `softmax_reference`, `sigmoid`, the head and the argmax of `decode.py` -- called
exactly as `KVCache._step_row` calls them, operation for operation, so the run reproduces
`decode_with_kv_cache` bit for bit: the oracle's half of "GEM computes what the monolithic runner
computes". The C half holds its interpreter to `bcir_llama_generate_greedy` the same way.

The pack is checked as the C interpreter checks it, refusal for refusal: every claim against
the RID scheme and the model's shapes before anything runs, and every position, cache row and
tape slot against the data flow so far while it runs. A refusal raises `DecoderProgramError`,
whose `kind` names the C refusal it mirrors (`program`, `order`, `arg`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ...kbcir.activation import softmax_reference
from ...kbcir.matmul import matmul_reference
from ...kbcir.recurrent import sigmoid
from ...kbcir.transformer_grads import rmsnorm_reference, rope_reference
from .decode import DecoderSpec, DecoderWeights, _argmax, _split_head, head_logits
from .decoder_program import (
    LAYER_BASE,
    LAYER_STRIDE,
    OP_ARGMAX,
    OP_ATTENTION,
    OP_EMBED,
    OP_HEAD,
    OP_KV_APPEND,
    OP_MATVEC,
    OP_MATVEC_ADD,
    OP_RMSNORM,
    OP_ROPE,
    OP_SWIGLU,
    OPS,
    RID_CTX,
    RID_EMBED,
    RID_FF,
    RID_FINAL,
    RID_G_FINAL,
    RID_GATE,
    RID_H,
    RID_H2,
    RID_HEAD,
    RID_K,
    RID_KR,
    RID_LOGITS,
    RID_Q,
    RID_QR,
    RID_TOK,
    RID_UP,
    RID_V,
    RID_X,
    SLOT_G_ATTN,
    SLOT_G_FF,
    SLOT_K_CACHE,
    SLOT_V_CACHE,
    SLOT_W_DOWN,
    SLOT_W_GATE,
    SLOT_W_UP,
    SLOT_WK,
    SLOT_WO,
    SLOT_WQ,
    SLOT_WV,
)

#: The claim table's bound, as the C interpreter's (DGEM_MAX_CLAIMS).
MAX_CLAIMS = 1 << 20


class DecoderProgramError(ValueError):
    """A pack the interpreter refuses. `kind` is the C refusal it mirrors: `program` (a claim
    that is not a decoder operation of this model), `order` (a claim outside its data-flow
    order) or `arg` (a prompt the program does not take)."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(f"{kind}: {message}")
        self.kind = kind


@dataclass(frozen=True)
class _Claim:
    op: str
    rd: tuple[int, ...]
    wr: tuple[int, ...]
    offset: int
    count: int
    layer: int = -1


@dataclass
class DecoderRun:
    """What one execution of a decoder pack produced: the generated ids, the scores that chose
    the last one, the claims in dispatch order, and how many claims of each op ran."""

    tokens: list[int]
    logits: list[float]
    order: list[int]
    claims: dict[str, int] = field(default_factory=dict)


def pack_dispatch_order(pack) -> list[int]:
    """GEM's order over a pack: phases by first appearance, then ascending claim id."""
    rank: dict[int, int] = {}
    for seg in pack.segments:
        rank.setdefault(seg.phase_id, len(rank))
    return [
        seg.claim_id for seg in sorted(pack.segments, key=lambda s: (rank[s.phase_id], s.claim_id))
    ]


def _layer_slot(rid: int, n_layers: int) -> tuple[int, int] | None:
    if rid < LAYER_BASE:
        return None
    layer, slot = divmod(rid - LAYER_BASE, LAYER_STRIDE)
    return (layer, slot) if layer < n_layers else None


def _check_claim(spec: DecoderSpec, c: _Claim) -> _Claim:
    """`c` against the RID scheme and the model's shapes (bcir_decoder_gem.c check_claim)."""
    d, kvd, ff, vocab = spec.d_model, spec.kv_dim, spec.d_ff, spec.vocab_size
    nl = spec.n_layers

    def refuse() -> DecoderProgramError:
        return DecoderProgramError("program", f"{c.op} {c.rd}->{c.wr} count {c.count}")

    def ls(rid: int, *slots: int) -> int:
        hit = _layer_slot(rid, nl)
        if hit is None or hit[1] not in slots:
            raise refuse()
        return hit[0]

    op, rd, wr = c.op, c.rd, c.wr
    if op == OP_EMBED:
        ok = rd == (RID_TOK, RID_EMBED) and wr == (RID_X,) and c.count == d
    elif op == OP_RMSNORM:
        if len(rd) != 2 or len(wr) != 1 or rd[0] != RID_X or c.count != d:
            raise refuse()
        if rd[1] == RID_G_FINAL and wr[0] == RID_FINAL:
            return c
        hit = _layer_slot(rd[1], nl)
        ok = hit is not None and (
            (hit[1] == SLOT_G_ATTN and wr[0] == RID_H) or (hit[1] == SLOT_G_FF and wr[0] == RID_H2)
        )
        if ok:
            return _Claim(op, rd, wr, c.offset, c.count, hit[0])
    elif op in (OP_MATVEC, OP_MATVEC_ADD):
        add = op == OP_MATVEC_ADD
        if len(rd) != (3 if add else 2) or len(wr) != 1:
            raise refuse()
        hit = _layer_slot(rd[1], nl)
        if hit is None:
            raise refuse()
        layer, slot = hit
        shapes = {
            (RID_H, SLOT_WQ, RID_Q): (d, d),
            (RID_H, SLOT_WK, RID_K): (d, kvd),
            (RID_H, SLOT_WV, RID_V): (d, kvd),
            (RID_H2, SLOT_W_GATE, RID_GATE): (d, ff),
            (RID_H2, SLOT_W_UP, RID_UP): (d, ff),
        }
        if add:
            shapes = {(RID_CTX, SLOT_WO, RID_X): (d, d), (RID_FF, SLOT_W_DOWN, RID_X): (ff, d)}
            if rd[2] != RID_X:
                raise refuse()
        shape = shapes.get((rd[0], slot, wr[0]))
        ok = shape is not None and c.count == shape[0] * shape[1]
        if ok:
            return _Claim(op, rd, wr, c.offset, c.count, layer)
    elif op == OP_ROPE:
        ok = rd == (RID_Q, RID_K) and wr == (RID_QR, RID_KR) and c.count == d + kvd
    elif op == OP_KV_APPEND:
        if rd != (RID_KR, RID_V) or len(wr) != 2 or c.count != 2 * kvd:
            raise refuse()
        layer = ls(wr[0], SLOT_K_CACHE)
        if ls(wr[1], SLOT_V_CACHE) == layer:
            return _Claim(op, rd, wr, c.offset, c.count, layer)
        ok = False
    elif op == OP_ATTENTION:
        if len(rd) != 3 or rd[0] != RID_QR or wr != (RID_CTX,) or c.offset >= 1 << 32:
            raise refuse()
        layer = ls(rd[1], SLOT_K_CACHE)
        if ls(rd[2], SLOT_V_CACHE) == layer and c.count == 2 * d * (c.offset + 1):
            return _Claim(op, rd, wr, c.offset, c.count, layer)
        ok = False
    elif op == OP_SWIGLU:
        ok = rd == (RID_GATE, RID_UP) and wr == (RID_FF,) and c.count == ff
    elif op == OP_HEAD:
        head = RID_EMBED if spec.tied_embeddings else RID_HEAD
        ok = rd == (RID_FINAL, head) and wr == (RID_LOGITS,) and c.count == vocab * d
    elif op == OP_ARGMAX:
        ok = rd == (RID_LOGITS,) and wr == (RID_TOK,) and c.count == vocab
    else:
        ok = False
    if not ok:
        raise refuse()
    return c


def load_decoder_pack(pack, spec: DecoderSpec) -> tuple[dict[int, _Claim], int, int]:
    """The pack's claim table, checked whole, and its tape: (claims, capacity, prompt_len).
    Segments and blocks pair by index; claim ids are the table's dense indices."""
    if spec.activation != "silu_gate":
        raise DecoderProgramError("program", "the interpreter runs the Llama/SwiGLU decoder")
    n = len(pack.segments)
    if not n or n > MAX_CLAIMS or len(pack.blocks) != n:
        raise DecoderProgramError("program", f"{n} segments and {len(pack.blocks)} blocks")
    claims: dict[int, _Claim] = {}
    for seg, block in zip(pack.segments, pack.blocks):
        if not 0 <= seg.claim_id < n or seg.claim_id in claims or seg.opcode not in OPS:
            raise DecoderProgramError("program", f"claim {seg.claim_id} op {seg.opcode!r}")
        if len(seg.reads) > 3 or len(seg.writes) > 2:
            raise DecoderProgramError("program", f"claim {seg.claim_id}: too many operands")
        claim = _Claim(seg.opcode, tuple(seg.reads), tuple(seg.writes), block.base, block.count)
        claims[seg.claim_id] = _check_claim(spec, claim)
    argmax = [c.offset for c in claims.values() if c.op == OP_ARGMAX]
    n_embed = sum(1 for c in claims.values() if c.op == OP_EMBED)
    if not argmax:
        raise DecoderProgramError("program", "no argmax claim: the program writes no token")
    first, last = min(argmax), max(argmax)
    if first < 1 or last >= 1 << 31 or len(argmax) != last + 1 - first or n_embed != last:
        raise DecoderProgramError("program", "the argmax and embed claims do not span one tape")
    capacity = last + 1
    for c in claims.values():
        if c.op != OP_ARGMAX and c.offset >= capacity - 1:
            raise DecoderProgramError("program", f"{c.op} at position {c.offset} is off the tape")
    return claims, capacity, first


def run_decoder_pack(
    pack, spec: DecoderSpec, weights: DecoderWeights, prompt: list[int]
) -> DecoderRun:
    """Execute the decoder program `pack` on `prompt` (exactly the program's prompt length)."""
    claims, capacity, prompt_len = load_decoder_pack(pack, spec)
    if len(prompt) != prompt_len or any(
        type(t) is not int or not 0 <= t < spec.vocab_size for t in prompt
    ):
        raise DecoderProgramError("arg", f"the program takes {prompt_len} ids in the vocabulary")
    d, nh, dk, kvh, kvd = spec.d_model, spec.n_heads, spec.d_k, spec.kv_heads, spec.kv_dim
    group, eps, ff = nh // kvh, spec.rms_norm_eps, spec.d_ff
    # Every activation a program can name starts each run at zero, as in the C interpreter: a
    # claim reads what the run wrote, or zeros -- never what an earlier run left.
    sizes = {RID_X: d, RID_H: d, RID_Q: d, RID_K: kvd, RID_V: kvd, RID_QR: d, RID_KR: kvd}
    sizes |= {RID_CTX: d, RID_H2: d, RID_GATE: ff, RID_UP: ff, RID_FF: ff, RID_FINAL: d}
    acts = {rid: [0.0] * n for rid, n in sizes.items()}
    acts[RID_LOGITS] = [0.0] * spec.vocab_size
    k_cache = [[[] for _ in range(kvh)] for _ in range(spec.n_layers)]
    v_cache = [[[] for _ in range(kvh)] for _ in range(spec.n_layers)]
    tape = list(prompt)
    next_position, current, final_at, head_at = 0, None, None, None
    order = pack_dispatch_order(pack)
    counts = dict.fromkeys(OPS, 0)

    def order_error(c: _Claim) -> DecoderProgramError:
        return DecoderProgramError("order", f"{c.op} at {c.offset} (position {current})")

    def gamma(c: _Claim) -> list[float]:
        if c.layer < 0:
            return list(weights.g_final)
        lw = weights.layers[c.layer]
        return list(lw.g_attn if (c.rd[1] - LAYER_BASE) % LAYER_STRIDE == SLOT_G_ATTN else lw.g_ff)

    def matrix(c: _Claim) -> tuple:
        lw = weights.layers[c.layer]
        slot = (c.rd[1] - LAYER_BASE) % LAYER_STRIDE
        return {
            SLOT_WQ: lw.w_q,
            SLOT_WK: lw.w_k,
            SLOT_WV: lw.w_v,
            SLOT_WO: lw.w_o,
            SLOT_W_GATE: lw.w_gate,
            SLOT_W_UP: lw.w1,
            SLOT_W_DOWN: lw.w2,
        }[slot]

    for claim_id in order:
        c = claims[claim_id]
        if c.op != OP_EMBED and current is None:
            raise order_error(c)
        if c.op == OP_EMBED:
            if c.offset != next_position or c.offset >= len(tape):
                raise order_error(c)
            current, next_position = c.offset, c.offset + 1
            acts[RID_X] = list(weights.embedding.row(tape[current]))
        elif c.op == OP_RMSNORM:
            acts[c.wr[0]] = rmsnorm_reference(acts[RID_X], 1, d, gamma(c), eps=eps)
            if c.wr[0] == RID_FINAL:
                final_at = current
        elif c.op in (OP_MATVEC, OP_MATVEC_ADD):
            n_in = ff if c.rd[0] == RID_FF else d
            n_out = c.count // n_in
            y = matmul_reference(acts[c.rd[0]], matrix(c), 1, n_out, n_in)
            if c.op == OP_MATVEC_ADD:  # the projection, then the residual
                x = acts[RID_X]
                acts[RID_X] = [x[i] + y[i] for i in range(d)]
            else:
                acts[c.wr[0]] = y
        elif c.op == OP_ROPE:
            if c.offset != current:
                raise order_error(c)
            q, k, base = acts[RID_Q], acts[RID_K], spec.rope_base
            acts[RID_QR] = [
                v
                for hd in range(nh)
                for v in rope_reference(_split_head(q, 1, hd, nh, dk), 1, dk, base, current)
            ]
            acts[RID_KR] = [
                v
                for g in range(kvh)
                for v in rope_reference(_split_head(k, 1, g, kvh, dk), 1, dk, base, current)
            ]
        elif c.op == OP_KV_APPEND:
            if c.offset != current or len(k_cache[c.layer][0]) != current:
                raise order_error(c)
            for g in range(kvh):
                k_cache[c.layer][g].append(acts[RID_KR][g * dk : (g + 1) * dk])
                v_cache[c.layer][g].append(_split_head(acts[RID_V], 1, g, kvh, dk))
        elif c.op == OP_ATTENTION:
            if c.offset != current or len(k_cache[c.layer][0]) != current + 1:
                raise order_error(c)
            sc = 1.0 / math.sqrt(dk)  # AttentionSpec(t, d_k).scale, as _step_row takes it
            concat = [0.0] * d
            for hd in range(nh):
                qh = acts[RID_QR][hd * dk : (hd + 1) * dk]
                ks, vs = k_cache[c.layer][hd // group], v_cache[c.layer][hd // group]
                t_len = len(ks)
                s = [0.0] * t_len
                for j in range(t_len):
                    acc = 0.0
                    kr = ks[j]
                    for ch in range(dk):
                        acc += qh[ch] * kr[ch]
                    s[j] = acc * sc
                a = softmax_reference(s, axis_len=t_len)
                ctx = [0.0] * dk
                for ch in range(dk):
                    acc = 0.0
                    for j in range(t_len):
                        acc += a[j] * vs[j][ch]
                    ctx[ch] = acc
                concat[hd * dk : (hd + 1) * dk] = ctx
            acts[RID_CTX] = concat
        elif c.op == OP_SWIGLU:
            g, u = acts[RID_GATE], acts[RID_UP]
            acts[RID_FF] = [g[i] * sigmoid(g[i]) * u[i] for i in range(ff)]
        elif c.op == OP_HEAD:
            if final_at != current:
                raise order_error(c)
            acts[RID_LOGITS] = head_logits(acts[RID_FINAL], weights)
            head_at = current
        elif c.op == OP_ARGMAX:
            if head_at != current or c.offset != current + 1 or c.offset != len(tape):
                raise order_error(c)
            tape.append(_argmax(acts[RID_LOGITS]))
        counts[c.op] += 1
    if len(tape) != capacity:
        raise DecoderProgramError("order", f"the run filled {len(tape)} of {capacity} positions")
    return DecoderRun(tape[prompt_len:], list(acts[RID_LOGITS]), order, counts)
