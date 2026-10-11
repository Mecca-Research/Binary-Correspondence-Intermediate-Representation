"""The native decoder's training step as a GEM+ program (NDT-GEM).

`runtime/c/bcir_decoder_train.c` runs one training step -- forward, mean cross-entropy, every
parameter gradient, the clipped AdamW update -- as a fixed sequence of kernel calls over six
caller-owned arenas. Until this module the step stood outside GEM+: no lower bound priced it, and
nothing but the parameter count held the C plan to an independent statement of what it allocates.

`training_step_program` states the step in the vocabulary GEM+'s bounds read, derived from the
spec alone (no C, no ctypes):

* **buffers** -- every span of every arena, in the C plan's own order (`params`, `activations`
  and the backward workspace of `bcir_decoder_train.c`), so the arena sums are a second
  derivation of `bcir_decoder_make_plan`'s three counts, held to it by `train.plan.mismatch`;
* **values** -- a buffer's versions: an overwrite (`beta == 0`, a fresh cache entry) starts a
  new value, a read-modify-write (an accumulated gradient, a residual add, in-place RoPE) keeps
  the value it reads, so a workspace buffer reused by every layer is one value per layer;
* **steps** -- one per kernel call, in the C rail's order, each with the values it reads and
  writes and its counted work.

**Counted work** is the multiply-adds of the contractions every implementation of this step
performs, at two units each: every linear's forward product and its two VJPs, and the causal
attention's four contractions (scores and context forward; the probability adjoint, the query,
key and value adjoints backward -- once each: the C kernel's second pass over the probability
adjoint is above the count, not in it). Normalization, softmax, RoPE, SwiGLU, cross-entropy and
the optimizer are not counted, so the count is a lower bound on the step's work, never an
estimate of it.

Over that program:

* `roofline` is G26's hierarchical roofline (`kbcir.io_bounds.roofline_bound`) for the C order:
  no execution of it is faster than the counted work at the peak rate, nor than the red-blue
  pebble traffic across each memory boundary at its bandwidth. The bound is as sound as the
  peak and bandwidths handed to it: declared hardware peaks give a bound on the silicon, a rate
  measured on this host a bound on that host's kernels (TMSAO-3), never a certificate.
* `transient_lower_bound` is G26's alignment-aware concurrent-live bound
  (`static_memory.layout_lower_bound`) over the activation and workspace values, their lifetimes
  the step ticks between their first and last use. The C plan retains every cache and reserves
  one workspace span per buffer; the bound is what any layout of the same order without
  recomputation needs, and `transient_incumbent` is GEM+'s incumbent layout portfolio over the
  same values -- the headroom is stated, not claimed realized: the C ABI's fixed spans are
  unchanged.

Declared, not modeled: recomputation (it changes the order, and so the bound's premise), the
tokens' and the logits' host copies, the provider's internal blocking, and the second pass of
the optimizer's preflight (it reads what the commit reads, so it moves no first-touch traffic).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ...frontends.models.decode import DecoderSpec
from ...kbcir.io_bounds import roofline_bound
from ...kbcir.static_memory import LayoutItem, incumbent_layout, layout_lower_bound

#: The C plan's six arenas, in `bcir_decoder_state`'s order. The four parameter arenas share
#: one count (`plan.parameters`), so the plan states three sums.
ARENAS = ("weights", "gradients", "moment1", "moment2", "activations", "scratch")
#: The arenas a step allocates and releases: what a layout of the step may fold.
TRANSIENT = ("activations", "scratch")
_FLOAT = 4

#: The per-layer parameter tensors, in `params()` order, with their HF names.
_LAYER = (
    ("norm1", "input_layernorm.weight"),
    ("q", "self_attn.q_proj.weight"),
    ("k", "self_attn.k_proj.weight"),
    ("v", "self_attn.v_proj.weight"),
    ("o", "self_attn.o_proj.weight"),
    ("norm2", "post_attention_layernorm.weight"),
    ("gate", "mlp.gate_proj.weight"),
    ("up", "mlp.up_proj.weight"),
    ("down", "mlp.down_proj.weight"),
)
#: The per-layer forward cache, in `activations()` order.
_CACHE = ("n1", "q", "k", "v", "p", "ctx", "x2", "n2", "gate", "up", "hidden")
#: The backward workspace, in `bcir_decoder_loss_backward`'s order.
_WORKSPACE = ("cur", "tmp", "dn", "dq", "dk", "dv", "dc", "dg", "du", "dh", "dl")


@dataclass(frozen=True)
class Buffer:
    """One span of one arena: its name, its arena and its float count."""

    name: str
    arena: str
    elements: int


@dataclass(frozen=True)
class Step:
    """One kernel call: the values it reads and writes (value ids), its counted work, and the
    GEMMs it issues through the provider, as `bcir_tensor_mm`'s (ta, tb, m, n, k)."""

    name: str
    reads: tuple[int, ...]
    writes: tuple[int, ...]
    work: int
    gemms: tuple[tuple[int, int, int, int, int], ...] = ()


@dataclass(frozen=True)
class TrainingStepProgram:
    """The step (module docstring): buffers in arena order, values (id -> buffer name), steps
    in the C order, and the values live after the step (the state the caller keeps)."""

    spec: DecoderSpec
    batch: int
    time: int
    buffers: tuple[Buffer, ...]
    values: tuple[str, ...]  # value id -> the buffer it is a version of
    steps: tuple[Step, ...]
    live_out: frozenset[int]

    def arena_elements(self) -> dict[str, int]:
        """Each arena's float count, summed from its buffers: the plan's own counts."""
        out = dict.fromkeys(ARENAS, 0)
        for buf in self.buffers:
            out[buf.arena] += buf.elements
        return out

    def work(self) -> int:
        """The counted work (module docstring): two units per counted multiply-add."""
        return sum(step.work for step in self.steps)

    def sizes(self) -> dict[int, int]:
        """Each value's size in bytes: its buffer's."""
        by_name = {buf.name: buf.elements * _FLOAT for buf in self.buffers}
        by_name.update({name: n * 4 for name, n in self._external()})
        return {vid: by_name[name] for vid, name in enumerate(self.values)}

    def _external(self):
        n = self.batch * self.time
        return (("tokens", n), ("targets", n))

    def gemm_shapes(self) -> list[tuple[int, int, int, int, int]]:
        """The distinct GEMMs the step issues, (ta, tb, m, n, k), in first-issue order."""
        return list(dict.fromkeys(g for step in self.steps for g in step.gemms))

    def order(self) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """The steps as `io_bounds` reads them: (reads, writes), in order."""
        return [(step.reads, step.writes) for step in self.steps]

    def roofline(self, peak, levels=()):
        """G26's hierarchical roofline for the C order: `peak` work units per second, `levels`
        (capacity bytes, bandwidth bytes per second) fastest first, exclusive. Returns (seconds,
        terms) as exact rationals (`roofline_bound`)."""
        return roofline_bound(
            self.order(), self.sizes(), self.work(), Fraction(peak), levels, self.live_out
        )

    def transient_items(self) -> list[LayoutItem]:
        """The activation and workspace values as layout items: a value is live from the step
        that writes it to the step after its last use (half-open ticks)."""
        arena = {buf.name: buf.arena for buf in self.buffers}
        first: dict[int, int] = {}
        last: dict[int, int] = {}
        for tick, step in enumerate(self.steps):
            for vid in (*step.reads, *step.writes):
                first.setdefault(vid, tick)
                last[vid] = tick
        sizes = self.sizes()
        return [
            LayoutItem(vid, sizes[vid], _FLOAT, first[vid], last[vid] + 1)
            for vid in sorted(first)
            if arena.get(self.values[vid]) in TRANSIENT
        ]

    def transient_lower_bound(self) -> int:
        """G26's alignment-aware concurrent-live bound over the transient values, in bytes."""
        return layout_lower_bound(self.transient_items())

    def transient_incumbent(self):
        """GEM+'s incumbent layout portfolio over the same values (`incumbent_layout`)."""
        return incumbent_layout(self.transient_items())

    def transient_bytes(self) -> int:
        """What the C plan reserves for the same values: its two transient arenas."""
        counts = self.arena_elements()
        return sum(counts[a] for a in TRANSIENT) * _FLOAT


class _Builder:
    """Values as buffer versions: `new` starts one (an overwrite), `cur` names the live one."""

    def __init__(self):
        self.values: list[str] = []
        self.live: dict[str, int] = {}
        self.steps: list[Step] = []

    def new(self, name: str) -> int:
        self.values.append(name)
        self.live[name] = len(self.values) - 1
        return self.live[name]

    def cur(self, name: str) -> int:
        if name not in self.live:  # an input the step reads before writing: its initial value
            return self.new(name)
        return self.live[name]

    def step(self, name, reads, writes=(), new=(), work=0, gemms=()):
        """A call reading `reads`, read-modify-writing `writes` and overwriting `new`; a
        GEMM's work is its own (two units per multiply-add)."""
        r = tuple(self.cur(b) for b in (*reads, *writes))
        w = tuple(self.cur(b) for b in writes) + tuple(self.new(b) for b in new)
        work += sum(2 * m * n * k for _ta, _tb, m, n, k in gemms)
        self.steps.append(Step(name, r, w, work, tuple(gemms)))

    def linear(self, name, rows, inp, out, x, weight, y):
        """`bcir_tensor_linear`: y = x W^T, one GEMM (0, 1, rows, out, in)."""
        self.step(name, (x, weight), new=(y,), gemms=((0, 1, rows, out, inp),))

    def linear_vjp(self, name, rows, inp, out, x, weight, dy, dw, dx, accumulate_dx):
        """`bcir_tensor_linear_backward`: dx = dy W (overwritten or accumulated) and
        dW += dy^T x -- the GEMMs (0, 0, rows, in, out) and (1, 0, out, in, rows)."""
        gemms = ((0, 0, rows, inp, out), (1, 0, out, inp, rows))
        if accumulate_dx:
            self.step(name, (x, weight, dy), (dw, dx), gemms=gemms)
        else:
            self.step(name, (x, weight, dy), (dw,), new=(dx,), gemms=gemms)


def _geometry(spec: DecoderSpec, batch: int, time: int):
    if spec.activation != "silu_gate":
        raise ValueError("the native step is the Llama/SwiGLU decoder")
    for value in (batch, time):
        if type(value) is not int or value < 1:
            raise ValueError("batch and time must be positive integers")
    hd = spec.d_model // spec.n_heads
    return dict(
        n=batch * time, d=spec.d_model, k=spec.kv_heads * hd, f=spec.d_ff, v=spec.vocab_size,
        h=spec.n_heads, kh=spec.kv_heads, hd=hd, b=batch, t=time, L=spec.n_layers,
    )  # fmt: skip


def training_step_program(spec: DecoderSpec, batch: int, time: int) -> TrainingStepProgram:
    """One `bcir_decoder_run` step of `spec` at `batch` x `time` (module docstring)."""
    g = _geometry(spec, batch, time)
    n, d, k, f, v, L = g["n"], g["d"], g["k"], g["f"], g["v"], g["L"]
    b, t, h, hd = g["b"], g["t"], g["h"], g["hd"]
    nd, nk, nf, nv, na = n * d, n * k, n * f, n * v, b * h * t * t
    causal = b * h * hd * (t * (t + 1) // 2)  # one causal contraction's multiply-adds

    # --- buffers, in the C plan's arena order --------------------------------------------
    shapes = {"q": d * d, "k": k * d, "v": k * d, "o": d * d, "gate": f * d, "up": f * d,
              "down": d * f, "norm1": d, "norm2": d}  # fmt: skip
    params = [("embed", v * d)]
    for layer in range(L):
        params += [(f"L{layer}.{name}", shapes[name]) for name, _hf in _LAYER]
    params.append(("final_norm", d))
    if not spec.tied_embeddings:
        params.append(("head", v * d))
    buffers = [Buffer(f"{arena}:{name}", arena, size)
               for arena in ARENAS[:4] for name, size in params]  # fmt: skip
    cache = {"n1": nd, "q": nd, "k": nk, "v": nk, "p": na, "ctx": nd, "x2": nd, "n2": nd,
             "gate": nf, "up": nf, "hidden": nf}  # fmt: skip
    buffers += [Buffer(f"x{layer}", "activations", nd) for layer in range(L + 1)]
    for layer in range(L):
        buffers += [Buffer(f"L{layer}.{name}", "activations", cache[name]) for name in _CACHE]
    buffers += [Buffer("norm", "activations", nd), Buffer("logits", "activations", nv)]
    work_sizes = {"cur": nd, "tmp": nd, "dn": nd, "dq": nd, "dk": nk, "dv": nk, "dc": nd,
                  "dg": nf, "du": nf, "dh": nf, "dl": nv}  # fmt: skip
    buffers += [Buffer(name, "scratch", work_sizes[name]) for name in _WORKSPACE]

    w = lambda name: f"weights:{name}"  # noqa: E731
    gr = lambda name: f"gradients:{name}"  # noqa: E731
    head = "embed" if spec.tied_embeddings else "head"
    s = _Builder()

    # --- forward --------------------------------------------------------------------------
    s.step("embed", ("tokens", w("embed")), new=("x0",))
    for layer in range(L):
        c = lambda name, layer=layer: f"L{layer}.{name}"  # noqa: E731
        p = lambda name, layer=layer: w(f"L{layer}.{name}")  # noqa: E731
        x, y = f"x{layer}", f"x{layer + 1}"
        s.step(c("rms1"), (x, p("norm1")), new=(c("n1"),))
        for proj, out in (("q", d), ("k", k), ("v", k)):
            s.linear(c(proj), n, d, out, c("n1"), p(proj), c(proj))
        s.step(c("rope.q"), (), (c("q"),))
        s.step(c("rope.k"), (), (c("k"),))
        s.step(c("attention"), (c("q"), c("k"), c("v")), new=(c("p"), c("ctx")), work=4 * causal)
        s.linear(c("o"), n, d, d, c("ctx"), p("o"), c("x2"))
        s.step(c("residual1"), (x,), (c("x2"),))
        s.step(c("rms2"), (c("x2"), p("norm2")), new=(c("n2"),))
        s.linear(c("gate"), n, d, f, c("n2"), p("gate"), c("gate"))
        s.linear(c("up"), n, d, f, c("n2"), p("up"), c("up"))
        s.step(c("silu"), (c("gate"),), new=(c("hidden"),))
        s.step(c("swiglu"), (c("up"),), (c("hidden"),))
        s.linear(c("down"), n, f, d, c("hidden"), p("down"), y)
        s.step(c("residual2"), (c("x2"),), (y,))
    s.step("final_norm", (f"x{L}", w("final_norm")), new=("norm",))
    s.linear("head", n, d, v, "norm", w(head), "logits")

    # --- the loss and every gradient --------------------------------------------------------
    for name, _size in params:  # accumulate == 0 zeroes the gradient arena
        s.step(f"zero.{name}", (), new=(gr(name),))
    s.step("cross_entropy", ("logits", "targets"), new=("dl",))
    s.linear_vjp("head.vjp", n, d, v, "norm", w(head), "dl", gr(head), "dn", False)
    s.step("final_norm.vjp", (f"x{L}", w("final_norm"), "dn"), (gr("final_norm"),), new=("cur",))
    for layer in reversed(range(L)):
        c = lambda name, layer=layer: f"L{layer}.{name}"  # noqa: E731
        p = lambda name, layer=layer: w(f"L{layer}.{name}")  # noqa: E731
        q = lambda name, layer=layer: gr(f"L{layer}.{name}")  # noqa: E731
        s.linear_vjp(c("down.vjp"), n, f, d, c("hidden"), p("down"), "cur", q("down"), "dh", False)
        s.step(c("swiglu.vjp"), (c("gate"), c("up"), "dh"), new=("dg", "du"))
        s.linear_vjp(c("gate.vjp"), n, d, f, c("n2"), p("gate"), "dg", q("gate"), "dn", False)
        s.linear_vjp(c("up.vjp"), n, d, f, c("n2"), p("up"), "du", q("up"), "dn", True)
        s.step(c("rms2.vjp"), (c("x2"), p("norm2"), "dn"), (q("norm2"),), new=("tmp",))
        s.step(c("residual2.vjp"), ("tmp",), ("cur",))
        s.linear_vjp(c("o.vjp"), n, d, d, c("ctx"), p("o"), "cur", q("o"), "dc", False)
        # the in-place backward rewrites the layer's probabilities with their score adjoint:
        # p is read-modify-written, so its lifetime ends here exactly as when it was only read
        s.step(c("attention.vjp"), (c("q"), c("k"), c("v"), "dc"), (c("p"),),
               new=("dq", "dk", "dv"), work=8 * causal)  # fmt: skip
        s.step(c("rope.q.vjp"), (), ("dq",))
        s.step(c("rope.k.vjp"), (), ("dk",))
        s.linear_vjp(c("q.vjp"), n, d, d, c("n1"), p("q"), "dq", q("q"), "dn", False)
        s.linear_vjp(c("k.vjp"), n, d, k, c("n1"), p("k"), "dk", q("k"), "dn", True)
        s.linear_vjp(c("v.vjp"), n, d, k, c("n1"), p("v"), "dv", q("v"), "dn", True)
        s.step(c("rms1.vjp"), (f"x{layer}", p("norm1"), "dn"), (q("norm1"),), new=("tmp",))
        s.step(c("residual1.vjp"), ("tmp",), ("cur",))
    s.step("embed.vjp", ("tokens", "cur"), (gr("embed"),))

    # --- the clipped AdamW update -----------------------------------------------------------
    for name, _size in params:
        s.step(f"norm.{name}", (gr(name),))
    for name, _size in params:
        s.step(f"adamw.{name}", (gr(name),), (w(name), f"moment1:{name}", f"moment2:{name}"))

    keep = {f"{arena}:{name}" for arena in ARENAS[:4] for name, _size in params}
    live_out = frozenset(s.live[name] for name in keep)
    return TrainingStepProgram(
        spec, batch, time, tuple(buffers), tuple(s.values), tuple(s.steps), live_out
    )


__all__ = [
    "ARENAS",
    "Buffer",
    "Step",
    "TRANSIENT",
    "TrainingStepProgram",
    "training_step_program",
]
