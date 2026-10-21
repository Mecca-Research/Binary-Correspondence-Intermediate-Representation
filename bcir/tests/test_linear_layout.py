"""PX-5 (LAYOUT): the GF(2) layout algebra's own laws (`bcir.kbcir.linear_layout`).

A layout is a linear map over GF(2) on the bits of its indices: each law here is checked on every
input of small spaces or on seeded random maps, never on a hand-picked example alone -- the inverse
inverts, composition composes, a conversion takes one layout's storage to another's element for
element, a broadcast's preimage is one index plus its kernel's span, a map that is not linear is
refused rather than stated, and a size that is not a power of two is refused at the GF(2) half and
taken by the tiled layout's integer grid.
"""

import random

from bcir.kbcir.linear_layout import LinearLayout, TiledLayout, bits_of


def _random_layout(rng, ins, outs, *, density=0.4):
    """A random GF(2) map: each input bit's image a random subset of the output bits."""
    wiring = {}
    for name, size in ins:
        for b in range(bits_of(size)):
            wiring[(name, b)] = [(o, ob) for o, osize in outs for ob in range(bits_of(osize))
                                 if rng.random() < density]  # fmt: skip
    return LinearLayout.from_bits(ins, outs, wiring)


def _random_bijection(rng, dims):
    """A random invertible map on `dims`: a random bit permutation then random XOR swizzles."""
    n = sum(bits_of(size) for _, size in dims)
    flat = [(name, b) for name, size in dims for b in range(bits_of(size))]
    perm = list(range(n))
    rng.shuffle(perm)
    wiring = {flat[k]: [flat[perm[k]]] for k in range(n)}
    layout = LinearLayout.from_bits(dims, dims, wiring)
    for _ in range(n):  # elementary row operations keep it invertible
        i, j = rng.sample(range(n), 2)
        swizzle = LinearLayout.from_bits(dims, dims, {
            flat[k]: [flat[k]] + ([flat[j]] if k == i else []) for k in range(n)
        })  # fmt: skip
        layout = swizzle.compose(layout)
    return layout


def test_a_size_must_be_a_power_of_two():
    """GF(2) states powers of two only; anything else is refused where it is named."""
    assert [bits_of(s) for s in (1, 2, 16, 512)] == [0, 1, 4, 9]
    for bad in (0, 3, 48, 96, -4, 2.0, True):
        try:
            bits_of(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"size {bad!r} admitted")
    try:
        LinearLayout.identity((("x", 48),))
    except ValueError:
        pass
    else:
        raise AssertionError("a 48-wide GF(2) dimension stated")


def test_inverse_and_compose_on_random_bijections():
    """For seeded random bijections (bit permutations and XOR swizzles): the inverse composed with
    the map is the identity both ways, on every input; composition is associative; and a map's
    table is a permutation of its space."""
    rng = random.Random(1729)
    dims = (("a", 16), ("b", 8))
    ident = LinearLayout.identity(dims)
    for _ in range(40):
        f, g, h = (_random_bijection(rng, dims) for _ in range(3))
        assert f.bijective and f.inverse().compose(f) == ident and f.compose(f.inverse()) == ident
        assert f.compose(g).compose(h) == f.compose(g.compose(h))
        images = [tuple(sorted(y.items())) for _, y in f.table()]
        assert len(set(images)) == 16 * 8


def test_apply_is_the_xor_of_the_set_bits():
    """A random map's image of every input is the XOR of its set bits' images -- the definition,
    held on all 2^8 inputs of each of 30 random maps against a plain loop."""
    rng = random.Random(7)
    ins, outs = (("x", 16), ("y", 16)), (("p", 32), ("q", 8))
    for _ in range(30):
        f = _random_layout(rng, ins, outs)
        for x in range(16):
            for y in range(16):
                p = q = 0
                for k, bit in enumerate([(x >> b) & 1 for b in range(4)] +
                                        [(y >> b) & 1 for b in range(4)]):  # fmt: skip
                    if bit:
                        p ^= f.images[k][0]
                        q ^= f.images[k][1]
                assert f.apply(x=x, y=y) == {"p": p, "q": q}


def test_a_broadcast_preimage_is_one_index_and_its_kernel():
    """A broadcast -- 16 lanes holding one element -- maps every lane to the element: its kernel is
    the four lane bits, and the preimage of the element is lane 0 plus their span (all 16 lanes); a
    map half broadcast keeps the other half's bits; an element no input reaches has no preimage."""
    bc = LinearLayout.from_bits((("lane", 16),), (("row", 1),), {("lane", b): [] for b in range(4)})
    one, kernel = bc.preimage(row=0)
    assert one == {"lane": 0} and len(kernel) == 4 and not bc.injective and bc.surjective
    span = {0}
    for v in kernel:
        span |= {s ^ v["lane"] for s in span}
    assert span == set(range(16))
    half = LinearLayout.from_bits((("lane", 16),), (("col", 4),),
                                  {("lane", 0): [("col", 0)], ("lane", 1): [("col", 1)],
                                   ("lane", 2): [], ("lane", 3): []})  # fmt: skip
    for col in range(4):
        x, kern = half.preimage(col=col)
        assert half.apply(**x) == {"col": col} and len(kern) == 2
    narrow = LinearLayout.from_bits((("x", 2),), (("y", 4),), {("x", 0): [("y", 0)]})
    assert narrow.preimage(y=2) is None and narrow.injective and not narrow.surjective


def test_the_kernel_and_the_preimages_are_the_maps():
    """For seeded random maps of six input bits onto three output bits, read against every input:
    the kernel basis spans exactly the inputs sent to zero (independent: its span has 2^len
    members, and len + rank is the input bits); each image's preimage is one input sent there plus
    that span -- every input sent there, no other -- and an element nothing reaches has none; and
    the map is injective, surjective, exactly as its table says."""
    rng = random.Random(29)
    ins, outs = (("x", 16), ("y", 4)), (("p", 8),)
    inputs = [{"x": x, "y": y} for y in range(4) for x in range(16)]
    for _ in range(40):
        f = _random_layout(rng, ins, outs, density=rng.choice((0.2, 0.4, 0.6)))
        image = {(x["x"], x["y"]): f.apply(**x)["p"] for x in inputs}
        kernel = f.kernel()
        span = {(0, 0)}
        for v in kernel:
            span |= {(a ^ v["x"], b ^ v["y"]) for a, b in span}
        assert span == {key for key, p in image.items() if p == 0}
        assert len(span) == 1 << len(kernel) and len(kernel) + f.rank == 6
        for p in range(8):
            reached = {key for key, q in image.items() if q == p}
            found = f.preimage(p=p)
            if not reached:
                assert found is None, p
                continue
            one, basis = found
            assert image[one["x"], one["y"]] == p and basis == kernel
            assert {(one["x"] ^ a, one["y"] ^ b) for a, b in span} == reached
        values = set(image.values())
        assert f.injective == (len(values) == 64) and f.surjective == (len(values) == 8)


def _refused(label, fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"{label} admitted")


def test_a_malformed_map_or_argument_is_refused():
    """Each rule the algebra keeps refuses what breaks it, with ValueError: a dimension named twice,
    an image count other than the input bits, an image outside the outputs, wiring that names a bit
    the inputs lack; an input or output value missing, extra, outside its dimension or not an int;
    a composition whose dimensions do not meet; the inverse of a map that is not a bijection; an
    output order that drops a dimension; and a tiled layout over a tile that is not a bijection
    from one offset, or a grid that is not the tile's."""
    two = (("x", 4),)
    f = LinearLayout.identity(two)
    bc = LinearLayout.from_bits((("lane", 4),), (("row", 1),), {("lane", b): [] for b in range(2)})
    narrow = LinearLayout.from_bits((("x", 2),), (("y", 4),), {("x", 0): [("y", 0)]})
    tile = LinearLayout.identity((("offset", 4),)).rename(outs={"offset": "c"})
    cases = [
        ("a dimension named twice", lambda: LinearLayout.identity((("x", 2), ("x", 4)))),
        ("a dimension named by a number", lambda: LinearLayout.identity(((1, 2),))),
        ("a size given as a float", lambda: LinearLayout.identity((("x", 2.0),))),
        ("an image value given as a float", lambda: LinearLayout(two, two, ((1.0,), (2,)))),
        ("one image for two bits", lambda: LinearLayout(two, two, ((1,),))),
        ("an image outside the outputs", lambda: LinearLayout(two, two, ((1,), (4,)))),
        ("an image of two values for one output", lambda: LinearLayout(two, two, ((1, 0), (2, 0)))),
        (
            "wiring past the inputs",
            lambda: LinearLayout.from_bits(
                two, two, {("x", 0): [("x", 0)], ("x", 1): [("x", 1)], ("x", 2): []}
            ),
        ),
        ("an input past its dimension", lambda: f.apply(x=4)),
        ("a negative input", lambda: f.apply(x=-1)),
        ("a bool input", lambda: f.apply(x=True)),
        ("an extra input", lambda: f.apply(x=1, z=0)),
        ("a missing input", lambda: f.apply()),
        ("an output past its dimension", lambda: f.preimage(x=4)),
        ("an extra output", lambda: f.preimage(x=1, z=0)),
        ("a composition that does not meet", lambda: f.compose(narrow)),
        ("the inverse of a broadcast", bc.inverse),
        ("the inverse of a map onto part of its outputs", narrow.inverse),
        (
            "an output order that drops one",
            lambda: f.product(LinearLayout.identity((("z", 2),))).reorder_outs(["x"]),
        ),
        (
            "a tile that is not a bijection",
            lambda: TiledLayout(
                bc.rename(ins={"lane": "offset"}), (("row", 2),), ("row",), (("row", 2),)
            ),
        ),
        (
            "a tile from another input",
            lambda: TiledLayout(tile.rename(ins={"offset": "o"}), (("c", 2),), ("c",), (("c", 8),)),
        ),
        (
            "a grid of another dimension",
            lambda: TiledLayout(tile, (("d", 2),), ("d",), (("c", 8),)),
        ),
        ("a grid of no tiles", lambda: TiledLayout(tile, (("c", 0),), ("c",), (("c", 1),))),
        ("an empty shape", lambda: TiledLayout(tile, (("c", 2),), ("c",), (("c", 0),))),
        (
            "a tile count not an integer",
            lambda: TiledLayout(tile, (("c", 2.5),), ("c",), (("c", 9),)),
        ),
        (
            "an order of another dimension",
            lambda: TiledLayout(tile, (("c", 2),), ("d",), (("c", 8),)),
        ),
        (
            "a shape of another dimension",
            lambda: TiledLayout(tile, (("c", 2),), ("c",), (("d", 8),)),
        ),
    ]
    for label, fn in cases:
        _refused(label, fn)
    t = TiledLayout(tile, (("c", 2),), ("c",), (("c", 7),))
    for label, fn in (("an offset past the layout", lambda: t.coords(8)),
                      ("a coordinate past the grid", lambda: t.offset(c=8)),
                      ("a coordinate missing", lambda: t.offset()),
                      ("an extra coordinate", lambda: t.offset(c=1, d=0)),
                      ("a coordinate not an integer", lambda: t.offset(c=2.0))):  # fmt: skip
        _refused(label, fn)


def test_a_conversion_takes_one_storage_to_the_other():
    """For two random storage layouts of one 2-D space (bijections from an offset to the
    coordinates), the conversion B^-1 . A takes every offset of A to the offset of B holding the
    same coordinates."""
    rng = random.Random(11)
    coords = (("row", 8), ("col", 16))
    for _ in range(20):
        a = _random_bijection(rng, coords).rename(ins={"row": "o_hi", "col": "o_lo"})
        b = _random_bijection(rng, coords).rename(ins={"row": "o_hi", "col": "o_lo"})
        conv = a.convert_to(b)
        for x, y in a.table():
            assert b.apply(**conv.apply(**x)) == y


def test_a_map_that_is_not_linear_is_refused():
    """`from_function` reads a map from its images of the unit inputs and checks it on every input:
    an affine map (a constant XOR), a map adding with carries, and a map dropping a bit after
    another are each refused, and a bit permutation is stated."""
    ins, outs = (("x", 16),), (("y", 16),)
    for name, fn in (
        ("affine", lambda v: {"y": v["x"] ^ 1}),
        ("carry", lambda v: {"y": (v["x"] + v["x"]) % 16 if v["x"] < 8 else (v["x"] + 3) % 16}),
        ("plus", lambda v: {"y": (v["x"] + 1) % 16}),
    ):
        try:
            LinearLayout.from_function(ins, outs, fn)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{name} stated as linear")
    rev = LinearLayout.from_function(ins, outs, lambda v: {"y": int(f"{v['x']:04b}"[::-1], 2)})
    # the bit reversal is its own inverse, its dimensions named the other way
    assert rev.bijective and rev.inverse() == rev.rename(ins={"x": "y"}, outs={"y": "x"})


def test_product_and_reorder_keep_each_part():
    """Two maps side by side act each on its own dimensions; a reordered map is the same map."""
    rng = random.Random(3)
    f = _random_layout(rng, (("a", 4),), (("p", 8),))
    g = _random_layout(rng, (("b", 8),), (("q", 4),))
    fg = f.product(g)
    for a in range(4):
        for b in range(8):
            assert fg.apply(a=a, b=b) == f.apply(a=a) | g.apply(b=b)
    swapped = fg.reorder_outs(["q", "p"])
    assert swapped.apply(a=3, b=5) == fg.apply(a=3, b=5) and swapped.outs != fg.outs
    try:
        f.product(f)
    except ValueError:
        pass
    else:
        raise AssertionError("a product over shared dimensions")


def test_a_tiled_layout_round_trips_and_masks_its_tail():
    """A GF(2) tile under a mixed-radix grid (three tiles one way): every offset's coordinates map
    back to it, every coordinate of the padded grid is held exactly once, and the coordinates past
    the shape -- the tail -- are exactly the padding offsets."""
    tile = LinearLayout.from_function((("offset", 32),), (("r", 4), ("c", 8)),
                                      lambda v: {"r": v["offset"] >> 3, "c": v["offset"] & 7})  # fmt: skip
    t = TiledLayout(tile, (("r", 2), ("c", 3)), ("r", "c"), (("r", 7), ("c", 20)))
    seen = set()
    for o in range(t.size):
        x = t.coords(o)
        assert t.offset(**x) == o
        seen.add((x["r"], x["c"]))
        assert t.padding(o) == (x["r"] >= 7 or x["c"] >= 20)
    assert seen == {(r, c) for r in range(8) for c in range(24)}
    try:
        TiledLayout(tile, (("r", 2), ("c", 3)), ("r", "c"), (("r", 9), ("c", 20)))
    except ValueError:
        pass
    else:
        raise AssertionError("a shape past its grid")
