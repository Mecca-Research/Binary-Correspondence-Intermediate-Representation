"""PX-5 (LAYOUT): GF(2) linear layouts beside BCIR's integer-affine addressing.

BCIR addresses a resource by integer-affine maps (`offset`, `stride_k`, `stride_class`; G3's SoA/AoS
pivot over them). The kernels the native trainer runs keep layouts those maps do not state: the AMX
tiles and the bf16 GEMM panels (16 rows by 32 depths of A; 16 depth pairs by 16 columns of B, the
pair interleaved), the AVX-512 register blocks (16 lanes a vector, one A element broadcast across
them), the 16 x 16 register transposes. Each is a permutation, a duplication or a projection of the
BITS of its indices -- a linear map over GF(2) -- the representation Triton's Linear Layouts give
every GPU layout (the comparison in docs/research/BCIR_TRITON_COMPARATIVE_ANALYSIS.md, §3a, item 1).

`LinearLayout` is such a map: named input dimensions and named output dimensions, each a power of
two in size, and for every input bit its image -- one value per output dimension; an index's image
is the XOR of the images of its set bits. Composition is composition, a bijection's inverse is its
inverse (Gauss-Jordan over GF(2)), and a layout conversion -- the bytes one layout holds rewritten
as another holds them -- is `B^-1 . A`. A map that is not injective broadcasts: its KERNEL is the set
of input directions that land on the same element (the 16 lanes of a broadcast register), and the
preimage of an element is one index plus that kernel's span (`preimage`).

GF(2) states only powers of two. What is not -- a panel of 96 depths is three 32-depth tiles, the
AVX-512 B tile three 16-lane vectors -- is the integer-affine half: `TiledLayout` is a GF(2) tile
(every in-tile offset bit to the low bits of the coordinates, a bijection) under an integer grid of
tiles (mixed radix, tile-major in a stated order), with the shape's tail masked (a coordinate past
the shape holds the padding the packer writes, zero). The two halves meet at the tile's edge and
nowhere else: the algebra inherits the power-of-two limit Triton's does (the "Hexcute" critique),
and states it where it applies rather than padding past it.

The layouts the trainer's kernels keep are stated in `bcir.hosted.models.native_layouts` and held
there to the C that keeps them, bit for bit (PX-5's witness). Nothing here generates a layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product as _cartesian

__all__ = [
    "LinearLayout",
    "TiledLayout",
    "bits_of",
]


def bits_of(size: int) -> int:
    """The bits of a power-of-two size (ValueError for any other size: GF(2) states only those)."""
    if type(size) is not int or size < 1 or size & (size - 1):
        raise ValueError(f"a GF(2) dimension is a power of two, not {size!r}")
    return size.bit_length() - 1


def _dims(spec) -> tuple[tuple[str, int], ...]:
    """(name, size) pairs, in order: each name a string, named once; each size a power of two."""
    out = tuple((name, size) for name, size in spec)
    names = [name for name, _ in out]
    if any(type(name) is not str for name in names) or len(set(names)) != len(names):
        raise ValueError(f"the dimensions are not distinct names: {names}")
    for _, size in out:
        bits_of(size)
    return out


def _pack(dims, values: dict) -> int:
    """`values` -- one per dimension of `dims`, each inside its size -- as one bit vector, the first
    dimension lowest (ValueError for a value missing, extra or outside its dimension)."""
    if set(values) != {name for name, _ in dims}:
        raise ValueError(f"values for {sorted(values)}, not for {[name for name, _ in dims]}")
    flat, shift = 0, 0
    for name, size in dims:
        v = values[name]
        if type(v) is not int or not 0 <= v < size:
            raise ValueError(f"{name}={v!r} outside [0, {size})")
        flat |= v << shift
        shift += bits_of(size)
    return flat


def _unpack(dims, flat: int) -> dict[str, int]:
    """A bit vector as one value per dimension of `dims` (`_pack`'s inverse)."""
    out, shift = {}, 0
    for name, size in dims:
        out[name] = (flat >> shift) & (size - 1)
        shift += bits_of(size)
    return out


@dataclass(frozen=True)
class LinearLayout:
    """A GF(2) linear map from the input dimensions' bits to the output dimensions' bits (module
    docstring). `ins` and `outs` are (name, size) pairs, sizes powers of two; `images` holds, for
    every input bit -- the input dimensions in order, each from its low bit -- the image as one
    value per output dimension (in `outs` order)."""

    ins: tuple[tuple[str, int], ...]
    outs: tuple[tuple[str, int], ...]
    images: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        ins, outs = _dims(self.ins), _dims(self.outs)
        object.__setattr__(self, "ins", ins)
        object.__setattr__(self, "outs", outs)
        images = tuple(tuple(image) for image in self.images)
        object.__setattr__(self, "images", images)
        if len(images) != sum(bits_of(size) for _, size in ins):
            raise ValueError("one image per input bit")
        for image in images:
            if len(image) != len(outs) or any(
                type(v) is not int or not 0 <= v < size for v, (_, size) in zip(image, outs)
            ):
                raise ValueError(f"an image outside the outputs: {image}")
        object.__setattr__(self, "_cols", tuple(self._flat_image(k) for k in range(len(images))))

    # --- construction ---------------------------------------------------------------------

    @staticmethod
    def identity(dims) -> LinearLayout:
        """Each dimension to itself."""
        dims = _dims(dims)
        images = []
        for k, (_, size) in enumerate(dims):
            for b in range(bits_of(size)):
                images.append(tuple((1 << b) if j == k else 0 for j in range(len(dims))))
        return LinearLayout(dims, dims, tuple(images))

    @staticmethod
    def from_bits(ins, outs, wiring) -> LinearLayout:
        """The map whose input bit (name, b) has the image `wiring[(name, b)]`: a list of output
        bits (name, b) it sets (XOR-ed), [] for a broadcast bit; every input bit named once."""
        ins, outs = _dims(ins), _dims(outs)
        at = {name: k for k, (name, _) in enumerate(outs)}
        images, seen = [], set()
        for name, size in ins:
            for b in range(bits_of(size)):
                image = [0] * len(outs)
                for out_name, out_bit in wiring[(name, b)]:
                    image[at[out_name]] ^= 1 << out_bit
                images.append(tuple(image))
                seen.add((name, b))
        if set(wiring) != seen:
            raise ValueError(f"wiring names bits the inputs lack: {sorted(set(wiring) - seen)}")
        return LinearLayout(ins, outs, tuple(images))

    @staticmethod
    def from_function(ins, outs, fn) -> LinearLayout:
        """The map `fn` (a dict of input values -> a dict of output values) is, read from its
        images of the unit inputs -- and checked linear on EVERY input (ValueError otherwise), so
        a function that is not a GF(2) map is never stated as one. For the small spaces a tile or
        a register spans (at most 2^16 inputs)."""
        ins, outs = _dims(ins), _dims(outs)
        total = sum(bits_of(size) for _, size in ins)
        if total > 16:
            raise ValueError("from_function checks every input: at most 16 input bits")
        zero = {name: 0 for name, _ in ins}
        images = []
        for name, size in ins:
            for b in range(bits_of(size)):
                got = fn(dict(zero, **{name: 1 << b}))
                images.append(tuple(got[out] for out, _ in outs))
        layout = LinearLayout(ins, outs, tuple(images))
        for values in _cartesian(*(range(size) for _, size in ins)):
            x = dict(zip((name for name, _ in ins), values))
            want = fn(dict(x))
            if layout.apply(**x) != {out: want[out] for out, _ in outs}:
                raise ValueError(f"not linear over GF(2): at {x}")
        return layout

    # --- the map ------------------------------------------------------------------------------

    @property
    def in_bits(self) -> int:
        return len(self.images)

    @property
    def out_bits(self) -> int:
        return sum(bits_of(size) for _, size in self.outs)

    def _flat_image(self, k: int) -> int:
        flat, shift = 0, 0
        for v, (_, size) in zip(self.images[k], self.outs):
            flat |= v << shift
            shift += bits_of(size)
        return flat

    def _columns(self) -> tuple[int, ...]:
        """Each input bit's image as one bit vector (the output dimensions concatenated)."""
        return self._cols

    def _apply_flat(self, flat: int) -> int:
        """The image of a flat input: the XOR of its set bits' images."""
        acc = 0
        for k, column in enumerate(self._columns()):
            if flat >> k & 1:
                acc ^= column
        return acc

    def apply(self, **x) -> dict[str, int]:
        """The image of the input `x` (one value per input dimension): the XOR of its set bits'
        images."""
        return _unpack(self.outs, self._apply_flat(_pack(self.ins, x)))

    def compose(self, inner: LinearLayout) -> LinearLayout:
        """self . inner: `inner`'s outputs are this map's inputs (the same names and sizes)."""
        if inner.outs != self.ins:
            raise ValueError(f"compose: {inner.outs} feed {self.ins}")
        images = []
        for k in range(inner.in_bits):
            y = _unpack(self.outs, self._apply_flat(inner._flat_image(k)))
            images.append(tuple(y[name] for name, _ in self.outs))
        return LinearLayout(inner.ins, self.outs, tuple(images))

    def _eliminate(self) -> tuple[dict[int, tuple[int, int]], list[int]]:
        """Gaussian elimination over GF(2) on the images: the pivots (an output bit -> an image
        reduced to that top bit, and the input combination whose image it is) and a basis of the
        kernel (flat inputs the map sends to zero)."""
        pivots: dict[int, tuple[int, int]] = {}
        kernel = []
        for k, column in enumerate(self._columns()):
            v, combo = column, 1 << k
            while v:
                top = v.bit_length() - 1
                if top not in pivots:
                    pivots[top] = (v, combo)
                    break
                pv, pc = pivots[top]
                v ^= pv
                combo ^= pc
            else:
                kernel.append(combo)
        return pivots, kernel

    @property
    def rank(self) -> int:
        return len(self._eliminate()[0])

    @property
    def injective(self) -> bool:
        return self.rank == self.in_bits

    @property
    def surjective(self) -> bool:
        return self.rank == self.out_bits

    @property
    def bijective(self) -> bool:
        return self.injective and self.surjective

    def kernel(self) -> list[dict[str, int]]:
        """A basis of the inputs the map sends to zero: the directions a broadcast repeats along
        (empty for an injective map)."""
        return [_unpack(self.ins, v) for v in self._eliminate()[1]]

    def preimage(self, **y) -> tuple[dict[str, int], list[dict[str, int]]] | None:
        """Every input the map sends to `y`: one of them and the kernel basis whose span, added to
        it, is all of them -- or None where `y` is not an image."""
        v, combo = _pack(self.outs, y), 0
        pivots, kernel = self._eliminate()
        while v:
            top = v.bit_length() - 1
            if top not in pivots:
                return None
            pv, pc = pivots[top]
            v ^= pv
            combo ^= pc
        return _unpack(self.ins, combo), [_unpack(self.ins, k) for k in kernel]

    def inverse(self) -> LinearLayout:
        """The inverse of a bijection (ValueError otherwise): outputs to inputs."""
        if not self.bijective:
            raise ValueError("only a bijection has an inverse")
        images = []
        for name, size in self.outs:
            for b in range(bits_of(size)):
                found = self.preimage(**{n: ((1 << b) if n == name else 0) for n, _ in self.outs})
                assert found is not None  # surjective
                x, _ = found
                images.append(tuple(x[n] for n, _ in self.ins))
        return LinearLayout(self.outs, self.ins, tuple(images))

    def convert_to(self, other: LinearLayout) -> LinearLayout:
        """The conversion from this layout's storage to `other`'s: other^-1 . self -- an index of
        this layout to the index of `other` that holds the same element (both onto the same
        outputs; `other` a bijection)."""
        return other.inverse().compose(self)

    def product(self, other: LinearLayout) -> LinearLayout:
        """The two maps side by side, each acting on its own dimensions -- disjoint names, which the
        constructor holds (a dimension named twice is refused)."""
        zeros_b = (0,) * len(other.outs)
        zeros_a = (0,) * len(self.outs)
        images = tuple(image + zeros_b for image in self.images) + tuple(
            zeros_a + image for image in other.images
        )
        return LinearLayout(self.ins + other.ins, self.outs + other.outs, images)

    def rename(self, ins: dict | None = None, outs: dict | None = None) -> LinearLayout:
        """The same map with dimensions renamed (`ins`/`outs`: old name -> new name)."""
        ins, outs = ins or {}, outs or {}
        return LinearLayout(
            tuple((ins.get(n, n), s) for n, s in self.ins),
            tuple((outs.get(n, n), s) for n, s in self.outs),
            self.images,
        )

    def reorder_outs(self, names) -> LinearLayout:
        """The same map with its output dimensions listed in the order `names`."""
        names = list(names)
        at = {n: k for k, (n, _) in enumerate(self.outs)}
        if sorted(names) != sorted(at):
            raise ValueError(f"reorder_outs: {names} is not {sorted(at)}")
        return LinearLayout(
            self.ins,
            tuple(self.outs[at[n]] for n in names),
            tuple(tuple(image[at[n]] for n in names) for image in self.images),
        )

    def table(self) -> list[tuple[dict, dict]]:
        """Every (input, image) pair, inputs in order (for the small spaces of a tile)."""
        if self.in_bits > 16:
            raise ValueError("table: at most 16 input bits")
        return [(_unpack(self.ins, v), _unpack(self.outs, self._apply_flat(v)))
                for v in range(1 << self.in_bits)]  # fmt: skip


@dataclass(frozen=True)
class TiledLayout:
    """A storage layout of a 2-D (or n-D) array: a GF(2) TILE -- a bijection from the in-tile
    offset to the low bits of the coordinates -- under an integer GRID of tiles, `grid` the tile
    counts per dimension and `order` the dimensions from the slowest-varying tile coordinate to the
    fastest; `shape` the array's own extent, every coordinate at or past it padding (the tail the
    packer writes as zero). The offset of a coordinate is its tile's rank in the grid times the
    tile's size, plus the tile map's preimage of its low bits (module docstring)."""

    tile: LinearLayout  # ins: (("offset", tile size),); outs: one per dimension, the tile's extent
    grid: tuple[tuple[str, int], ...]  # (dimension, tiles), any positive counts (mixed radix)
    order: tuple[str, ...]  # the grid's dimensions, slowest first
    shape: tuple[tuple[str, int], ...]  # (dimension, extent)

    def __post_init__(self):
        if len(self.tile.ins) != 1 or self.tile.ins[0][0] != "offset" or not self.tile.bijective:
            raise ValueError("a tile is a bijection from one 'offset' dimension")
        dims = [n for n, _ in self.tile.outs]
        if sorted(n for n, _ in self.grid) != sorted(dims) or sorted(self.order) != sorted(dims):
            raise ValueError("the grid and its order name the tile's dimensions")
        if sorted(n for n, _ in self.shape) != sorted(dims):
            raise ValueError("the shape names the tile's dimensions")
        extent = dict(self.tile.outs)
        for name, count in self.grid:
            if type(count) is not int:
                raise ValueError(f"tiles along {name}: {count!r}")
            if not 0 < dict(self.shape)[name] <= count * extent[name]:  # so a tile at least
                raise ValueError(f"{name}: the shape outside the grid")

    @property
    def tile_size(self) -> int:
        return self.tile.ins[0][1]

    @property
    def size(self) -> int:
        tiles = 1
        for _, count in self.grid:
            tiles *= count
        return tiles * self.tile_size

    def coords(self, offset: int) -> dict[str, int]:
        """The coordinates the element at `offset` holds (padding past the shape included)."""
        if type(offset) is not int or not 0 <= offset < self.size:
            raise ValueError(f"offset {offset!r} outside [0, {self.size})")
        rank, inside = divmod(offset, self.tile_size)
        counts, extent = dict(self.grid), dict(self.tile.outs)
        tile = {}
        for name in reversed(self.order):  # the fastest coordinate first
            rank, tile[name] = divmod(rank, counts[name])
        low = self.tile.apply(offset=inside)
        return {name: tile[name] * extent[name] + low[name] for name, _ in self.tile.outs}

    def offset(self, **x) -> int:
        """The offset that holds the coordinates `x` (padding coordinates included)."""
        counts, extent = dict(self.grid), dict(self.tile.outs)
        if set(x) != set(extent) or any(type(v) is not int for v in x.values()):
            raise ValueError(f"coordinates {x} are not one integer for each of {sorted(extent)}")
        rank = 0
        for name in self.order:
            t, _ = divmod(x[name], extent[name])
            if not 0 <= t < counts[name]:
                raise ValueError(f"{name}={x[name]} outside the grid")
            rank = rank * counts[name] + t
        found = self.tile.preimage(**{n: x[n] % extent[n] for n, _ in self.tile.outs})
        assert found is not None  # the tile is a bijection
        return rank * self.tile_size + found[0]["offset"]

    def padding(self, offset: int) -> bool:
        """Whether `offset` holds a coordinate past the shape (a tail the packer zeroes)."""
        x = self.coords(offset)
        return any(x[name] >= extent for name, extent in self.shape)
