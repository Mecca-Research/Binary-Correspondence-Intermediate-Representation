"""cgen — an ASN.1 module compiled to straight-line C codecs.

BCIR's C rail decodes ASN.1 through *plans*: `bcir_oer.c` and `bcir_per_plan.c` walk a field
table a caller fills in from the write plan, and `bcir_emit.c` interprets the plan to encode.
That is the right design for a decoder that must serve every schema from one freestanding
binary, and it is the design every table-driven ASN.1 toolkit shares -- including FFASN1, the
codec behind the one published measurement this module exists to re-run (Bittl et al. 2015,
docs/research/BCIR_ITS_ASN1_BINARY_STUDY.md). An interpreter pays for its generality on every
field: a dispatch on the field kind, a load of the bounds, a loop over a table. A hand-written
binary codec pays none of it, and "binary encoding is faster than ASN.1" is, in large part, a
measurement of that difference.

**This compiles the dispatch away.** The schema is known when the codec is built, so every
decision a plan interpreter makes per field -- which width, which bounds, which alternative
numbering, whether a preamble exists -- is made once, here, and the C that comes out is the
sequence of loads and stores the encoding needs and nothing else: one function per constructed
type, fixed widths as constants, presence bits as literal masks. What remains is the work the
ENCODING asks for, which is what a comparison between encodings should measure.

**One value model, every rule.** `CModel` assigns every type reachable from the roots one C
type, and every rule's codec reads and writes that same struct. So a COER encoder, a UPER
encoder and a V1.1.1-binary encoder of the transcription module start from identical bytes in
memory, and a difference in their cost is a difference of encoding rule -- never of data model.

* INTEGER: the narrowest `uintN_t`/`intN_t` holding the type's value set (`uint64_t` for a
  semi-constrained one, `int64_t` for an unconstrained or extensible one).
* ENUMERATED: `uint32_t`, the enumerator's NUMBER (BCIR's value convention on every rail).
* OCTET STRING: `prefix_octN` (an `N`-octet array in a struct) when the SIZE is fixed, else
  `prefix_octets`, a `(pointer, length)` view. A byte-aligned decoder points the view INTO the
  input -- zero copy -- and UPER, whose strings start at any bit, copies into the arena.
* SEQUENCE: a struct, `has_<name>` before each OPTIONAL component.
* SEQUENCE OF: `{ size_t n; T *v; }`, the elements contiguous.
* CHOICE: `{ uint32_t tag; union { ... } u; }`, `tag` the alternative's index in definition
  order, named by `PREFIX_<Type>_<alternative>` constants.
* A recursive reference (`schema.Reference`, the front end's back-edge) is a POINTER, so every
  struct has a finite size.

**No heap.** A decoder takes a caller's arena (`prefix_arena`, a bump allocator over caller
memory, which may start at any address) for the three things that need storage beyond the
struct -- SEQUENCE OF elements, recursion targets, and UPER's unaligned strings -- and never
calls an allocator. Running out is a status, not a crash. Recursion is bounded by
`PREFIX_MAX_DEPTH`.

**Total and canonical.** Every decoder is a trust-boundary decoder: every read is bounds-
checked, every count is checked against what the input can hold before anything is allocated,
and only the CANONICAL spelling of a value is accepted (X.696 §31, X.691's canonical PER, the
binary rule's minimal IntX) -- a decoder that accepted two spellings of one value would let a
peer choose the octets a digest is taken over (laws.md, Class A). The parity tests hold every
codec to the oracle: the same octets out, the same values back, and a refusal wherever the
oracle refuses.

**Refusal is at generation time.** A construct this compiler does not translate -- DEFAULT,
extension additions, a SET, a character string -- raises `Asn1Error` naming it, when the C is
generated. A codec that silently skipped a construct would disagree with the oracle on exactly
the values that use it, and the parity test would report an unexplained byte rather than the
missing feature it is.

The rules:

* `COER`: X.696 CANONICAL-OER -- the rule IEEE 1609.2 and the published ASN.1 editions of
  TS 103 097 adopted for the security envelope.
* `UPER`: X.691 CANONICAL-PER, UNALIGNED.
* `ByteRule(...)`: a TLS-style presentation-language rule of the kind TS 103 097 V1.1.1
  defines -- a one-octet type code before every CHOICE, an IntX length before every
  variable-length vector, everything else fixed-width big-endian. `its_security.V111_RULE`
  is V1.1.1's.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .tags import Asn1Error, Universal

#: How many recursive references one value may pass through in a generated decoder. The
#: Python rails allow `schema.MAX_RECURSION`; a C decoder recursing on attacker-chosen depth
#: needs a bound on its own stack, and a certificate signed by a certificate signed by a
#: certificate is already further than any security envelope goes.
MAX_DEPTH = 8

#: The largest count or length the generated PER codec carries without X.691 §11.9.3.8's
#: fragmentation (16K units). A longer one is refused by name in both directions.
PER_FRAGMENT_UNIT = 16384


def _uper_bare_size(k) -> None:
    """X.691 §17.5-§17.7 write a SIZE-fixed OCTET STRING bare only below 64K octets; at 64K or
    more §17.8 gives it a length determinant like any other, and at that size the determinant
    is fragmented (§11.9.3.8). The oracle writes SIZE(65536) as 65538 octets starting `c4`; a
    codec writing the 65536 bare octets is a different wire format, so the generator refuses
    the type for UPER -- in both directions, at generation -- as it refuses every other
    fragmented length. COER has no such bound (X.696 §14.1: a fixed size is never prefixed)."""
    from .per import _64K

    if k.size >= _64K:
        raise Asn1Error(
            f"cgen: a SIZE ({k.size}) OCTET STRING takes a fragmented length determinant under "
            f"UPER (X.691 17.8, 11.9.3.8), which this compiler does not build"
        )


@dataclass(frozen=True)
class ByteRule:
    """A schema-directed byte rule in the style of TS 103 097 V1.1.1's presentation language.

    `type_codes` maps a CHOICE type's name to its alternatives' one-octet type codes; every
    CHOICE reachable from a root must be listed, and an alternative without a code cannot be
    encoded. Vectors (SEQUENCE OF, and an OCTET STRING whose SIZE is not fixed) carry their
    length in OCTETS as an IntX; `(0..MAX)` integers are IntX values; every other INTEGER is
    a fixed-width big-endian word the width of its range (`intx_max` bounds the IntX code).
    """

    name: str
    type_codes: dict
    intx_max: int = (1 << 56) - 1


COER = "coer"
UPER = "uper"


def _rule_name(rule) -> str:
    if rule == COER or rule == UPER:
        return rule
    if isinstance(rule, ByteRule):
        return rule.name
    raise Asn1Error(f"cgen: unknown rule {rule!r}")


# --- the value model ------------------------------------------------------------------------


@dataclass
class CMember:
    """One component (SEQUENCE) or alternative (CHOICE) as C sees it."""

    name: str
    kind: "CType"
    optional: bool = False
    pointer: bool = False  # a recursive reference: a pointer to an arena node
    #: A SEQUENCE OF element whose type is a recursive reference. The element array already
    #: breaks the by-value cycle, so it is no pointer -- but decoding one is a level of
    #: recursion, and the depth bound must count it.
    recursive: bool = False
    comp: object = None  # the schema Component, for the CHOICE tag

    @property
    def field(self) -> str:
        return _c_ident(self.name)


@dataclass
class CType:
    """One C representation. `kind` is one of: seq, choice, seqof, octfix, octvar, int, enum,
    bool, null. `asn1` is the (resolved) schema type it represents."""

    kind: str
    asn1: object
    cname: str = ""
    members: list = field(default_factory=list)
    element: "CMember | None" = None  # seqof
    size: int = 0  # octfix
    ctype: str = ""  # int: the C scalar type
    signed: bool = False
    extensible: bool = False

    @property
    def is_struct(self) -> bool:
        return self.kind in ("seq", "choice", "seqof", "octfix")


#: C11's keywords, which an ASN.1 identifier may spell (`Payload ::= CHOICE { signed ... }`).
_C_KEYWORDS = frozenset(
    "auto break case char const continue default do double else enum extern float for goto if "
    "inline int long register restrict return short signed sizeof static struct switch typedef "
    "union unsigned void volatile while _Alignas _Alignof _Atomic _Bool _Complex _Generic "
    "_Imaginary _Noreturn _Static_assert _Thread_local".split()
)


def _c_ident(name: str) -> str:
    """An ASN.1 identifier as a C one: `-` (legal in X.680 names) becomes `_`, and a C keyword
    gains a trailing `_`."""
    out = "".join(ch if ch.isalnum() else "_" for ch in name)
    out = out if out and not out[0].isdigit() else "_" + out
    return out + "_" if out in _C_KEYWORDS else out


class CModel:
    """The C types of everything reachable from `roots`, in a definable order."""

    def __init__(self, types: dict, roots, prefix: str):
        self.prefix = prefix
        self.types = types
        self.roots = list(roots)
        self._by_id: dict[int, CType] = {}
        self._names: set[str] = set()
        self.order: list[CType] = []  # struct types, each after everything it holds by value
        self._fixed: dict[int, CType] = {}
        for name in self.roots:
            if name not in types:
                raise Asn1Error(f"cgen: no type {name!r} in the module")
            self.ctype_of(types[name], name)
        self._order_structs()

    # -- naming ------------------------------------------------------------------------------

    def _named(self, kind) -> str | None:
        name = getattr(kind, "name", None)
        if name and name in self.types and self.types[name] is kind:
            return name
        return None

    def _fresh(self, hint: str) -> str:
        base = _c_ident(hint)
        name, n = base, 2
        while name in self._names:
            name, n = f"{base}_{n}", n + 1
        self._names.add(name)
        return name

    # -- the mapping ---------------------------------------------------------------------------

    def ctype_of(self, kind, hint: str) -> CType:
        from .schema import Choice, Primitive, Reference, Sequence, SequenceOf, Set, SetOf, resolve

        kind = resolve(kind) if isinstance(kind, Reference) else kind
        if id(kind) in self._by_id:
            return self._by_id[id(kind)]
        hint = self._named(kind) or hint
        if isinstance(kind, (Set, SetOf)):
            raise Asn1Error(
                f"cgen: {hint} is a SET/SET OF; X.691/X.696 order SETs by tag and "
                f"this compiler does not build that ordering"
            )
        if isinstance(kind, Sequence):
            c = CType("seq", kind, cname=self._fresh(hint), extensible=bool(kind.extensible))
            self._by_id[id(kind)] = c
            for comp in kind.components:
                if comp.extension:
                    raise Asn1Error(
                        f"cgen: {hint}.{comp.name} is an extension addition, which "
                        f"this compiler does not build (X.691 19.7, X.696 16.4)"
                    )
                if comp.has_default:
                    raise Asn1Error(
                        f"cgen: {hint}.{comp.name} has a DEFAULT, which this "
                        f"compiler does not build (X.691 19.5, X.696 31.9)"
                    )
                c.members.append(self._member(comp, f"{c.cname}_{comp.name}"))
            return c
        if isinstance(kind, Choice):
            c = CType("choice", kind, cname=self._fresh(hint), extensible=bool(kind.extensible))
            self._by_id[id(kind)] = c
            for alt in kind.alternatives:
                if alt.extension:
                    raise Asn1Error(
                        f"cgen: {hint}.{alt.name} is an extension alternative, "
                        f"which this compiler does not build (X.691 23.8, X.696 20.2)"
                    )
                c.members.append(self._member(alt, f"{c.cname}_{alt.name}"))
            return c
        if isinstance(kind, SequenceOf):
            c = CType("seqof", kind, cname=self._fresh(hint))
            self._by_id[id(kind)] = c
            element = self._member_of_type(kind.element, f"{c.cname}_item", "item")
            element.recursive, element.pointer = element.pointer, False
            if element.kind.kind == "null":
                raise Asn1Error(
                    f"cgen: {c.cname} is a SEQUENCE OF NULL, whose count no input "
                    f"bounds; this compiler refuses it"
                )
            c.element = element
            return c
        if isinstance(kind, Primitive):
            c = self._primitive(kind, hint)
            self._by_id[id(kind)] = c
            return c
        raise Asn1Error(
            f"cgen: {hint} is a {type(kind).__name__}, which this compiler does not translate"
        )

    def _member(self, comp, hint: str) -> CMember:
        m = self._member_of_type(comp.type, hint, comp.name)
        m.optional = bool(comp.optional)
        m.comp = comp
        return m

    def _member_of_type(self, kind, hint: str, name: str) -> CMember:
        from .schema import Reference

        pointer = isinstance(kind, Reference)
        return CMember(name=name, kind=self.ctype_of(kind, hint), pointer=pointer)

    def _primitive(self, kind, hint: str) -> CType:
        from .constraints import Extensible
        from .oer import _fixed_size

        u = kind.universal
        if u == Universal.NULL:
            return CType("null", kind)
        if u == Universal.BOOLEAN:
            return CType("bool", kind, ctype="uint8_t")
        if u == Universal.ENUMERATED:
            numbers = [n for _name, n in kind.enumeration]
            if any(n < 0 or n > 127 for n in numbers):
                raise Asn1Error(
                    f"cgen: {hint} has an enumerator outside 0..127; X.696 11.4's "
                    f"long form is not built here"
                )
            return CType("enum", kind, ctype="uint32_t", extensible=bool(kind.enum_extensible))
        if u == Universal.INTEGER:
            constraint = getattr(kind, "constraint", None)
            if isinstance(constraint, Extensible) or constraint is None:
                return CType("int", kind, ctype="int64_t", signed=True)
            low, high = constraint.value_bounds()
            return CType("int", kind, ctype=_int_ctype(low, high, hint), signed=_signed(low))
        if u == Universal.OCTET_STRING:
            size = _fixed_size(kind)
            if size is not None:
                if size not in self._fixed:
                    self._fixed[size] = CType("octfix", None, cname=f"oct{size}", size=size)
                return self._fixed[size]
            return CType("octvar", kind)
        raise Asn1Error(
            f"cgen: {hint} is a {getattr(kind, 'name', '') or Universal(u).name}; "
            f"this compiler translates INTEGER, ENUMERATED, BOOLEAN, NULL and OCTET "
            f"STRING primitives"
        )

    # -- ordering -------------------------------------------------------------------------------

    def structs(self) -> list[CType]:
        return self.order

    def _order_structs(self) -> None:
        """Every struct after everything it holds BY VALUE. Pointers and SEQUENCE OF arrays do
        not constrain the order (a forward typedef is enough), which is what makes a recursive
        module definable at all."""
        seen: set[int] = set()
        ordered: list[CType] = []
        for c in self._fixed.values():
            seen.add(id(c))
            ordered.append(c)

        later: list[CType] = []

        def visit(c: CType, stack: tuple) -> None:
            if not c.is_struct or id(c) in seen:
                return
            if id(c) in stack:  # pragma: no cover - a by-value cycle is a front-end bug
                raise Asn1Error(f"cgen: {c.cname} contains itself by value")
            for m in c.members:
                if m.pointer:
                    later.append(m.kind)  # a pointer constrains nothing: a typedef suffices
                else:
                    visit(m.kind, stack + (id(c),))
            if c.element is not None:
                later.append(c.element.kind)  # nor does an element array
            if id(c) not in seen:  # a by-value member may have reached it through `later`
                seen.add(id(c))
                ordered.append(c)

        for name in self.roots:
            later.append(self.ctype_of(self.types[name], name))
        while later:
            visit(later.pop(0), ())
        self.order = ordered


def _signed(low) -> bool:
    return low is None or low < 0


def _int_ctype(low, high, hint: str) -> str:
    if low is not None and low >= 0:
        if high is None:
            return "uint64_t"
        for bits in (8, 16, 32, 64):
            if high < (1 << bits):
                return f"uint{bits}_t"
        raise Asn1Error(f"cgen: {hint} needs more than 64 bits")
    if low is None or high is None:
        return "int64_t"
    for bits in (8, 16, 32, 64):
        if -(1 << (bits - 1)) <= low and high < (1 << (bits - 1)):
            return f"int{bits}_t"
    raise Asn1Error(f"cgen: {hint} needs more than 64 bits")


# --- C emission: shared prelude --------------------------------------------------------------

_PRELUDE = r"""
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* The I/O primitives run once per field and are a few instructions each, so they are forced
 * inline where the compiler takes the hint. A large generated file otherwise exhausts GCC's
 * unit-growth budget, the primitive stays out of line, and the cursor every codec function
 * keeps in a local goes through memory on every field. */
#if defined(__GNUC__)
#define P_HOT __attribute__((always_inline))
#else
#define P_HOT
#endif

#ifndef P_MAX_DEPTH
#define P_MAX_DEPTH @MAXDEPTH@
#endif

static void *P_alloc(P_arena *a, size_t count, size_t size, size_t align) {
  size_t at, bytes;
  if (count != 0 && size > (SIZE_MAX / count)) return NULL;
  bytes = count * size;
  /* Aligned as an ADDRESS: the arena is caller memory, which may start at any octet. */
  at = a->used + (size_t)(((uintptr_t)0 - ((uintptr_t)a->base + a->used)) & (align - 1u));
  if (at < a->used || at > a->cap || bytes > a->cap - at) return NULL;
  a->used = at + bytes;
  return a->base + at;
}
"""

_BYTE_IO = r"""
/* --- octet-aligned I/O (COER and the byte rule) --------------------------------------------- */
typedef struct { uint8_t *p, *end; } P_bw;
typedef struct { const uint8_t *p, *end; } P_br;
#define P_NEED(w, k) do { if ((size_t)((w)->end - (w)->p) < (size_t)(k)) return P_E_SPACE; } while (0)
#define P_HAVE(r, k) do { if ((size_t)((r)->end - (r)->p) < (size_t)(k)) return P_E_TRUNC; } while (0)

/* Big-endian words of 1, 2, 4 and 8 octets as single loads and stores (a byte swap on a
 * little-endian host) -- the widths every fixed-size INTEGER takes -- and a loop otherwise.
 * `n` is a constant at almost every call site, so the tests fold away. */
#if defined(__GNUC__) && defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define P_BSWAP16(x) __builtin_bswap16(x)
#define P_BSWAP32(x) __builtin_bswap32(x)
#define P_BSWAP64(x) __builtin_bswap64(x)
#define P_NATIVE_WORDS 1
#elif defined(__GNUC__) && defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define P_BSWAP16(x) (x)
#define P_BSWAP32(x) (x)
#define P_BSWAP64(x) (x)
#define P_NATIVE_WORDS 1
#else
#define P_NATIVE_WORDS 0
#endif
static inline P_HOT void P_store_be(uint8_t *p, uint64_t v, unsigned n) {
#if P_NATIVE_WORDS
  if (n == 8u) { uint64_t x = P_BSWAP64(v); memcpy(p, &x, 8); return; }
  if (n == 4u) { uint32_t x = P_BSWAP32((uint32_t)v); memcpy(p, &x, 4); return; }
  if (n == 2u) { uint16_t x = P_BSWAP16((uint16_t)v); memcpy(p, &x, 2); return; }
#endif
  while (n-- > 0u) { p[n] = (uint8_t)v; v >>= 8; }
}
static inline P_HOT uint64_t P_load_be(const uint8_t *p, unsigned n) {
  uint64_t v = 0;
  unsigned i;
#if P_NATIVE_WORDS
  if (n == 8u) { uint64_t x; memcpy(&x, p, 8); return P_BSWAP64(x); }
  if (n == 4u) { uint32_t x; memcpy(&x, p, 4); return P_BSWAP32(x); }
  if (n == 2u) { uint16_t x; memcpy(&x, p, 2); return P_BSWAP16(x); }
#endif
  for (i = 0; i < n; i++) v = (v << 8) | p[i];
  return v;
}
static inline P_HOT unsigned P_uoctets(uint64_t v) { /* minimal octets of an unsigned value, >= 1 */
  unsigned n = 1;
  while (n < 8u && (v >> (8u * n)) != 0u) n++;
  return n;
}
static inline P_HOT unsigned P_soctets(int64_t v) { /* minimal two's complement octets, >= 1 */
  unsigned n = 1;
  while (n < 8u) {
    int64_t lo = -((int64_t)1 << (8u * n - 1u)), hi = ((int64_t)1 << (8u * n - 1u)) - 1;
    if (v >= lo && v <= hi) break;
    n++;
  }
  return n;
}
"""

_COER_IO = r"""
/* --- X.696 CANONICAL-OER primitives ---------------------------------------------------------- */
static inline P_HOT int P_oer_put_len(P_bw *w, size_t n) { /* 8.6: short form below 128, else minimal long */
  if (n < 128u) { P_NEED(w, 1); *w->p++ = (uint8_t)n; return P_OK; }
  {
    unsigned k = P_uoctets((uint64_t)n);
    P_NEED(w, 1u + k);
    *w->p++ = (uint8_t)(0x80u | k);
    P_store_be(w->p, (uint64_t)n, k);
    w->p += k;
  }
  return P_OK;
}
static inline P_HOT int P_oer_get_len(P_br *r, size_t *n) { /* 31.2: only the canonical spelling */
  unsigned k;
  uint64_t v;
  P_HAVE(r, 1);
  if ((r->p[0] & 0x80u) == 0u) { *n = r->p[0]; r->p++; return P_OK; }
  k = r->p[0] & 0x7Fu;
  if (k == 0u || k > 8u) return P_E_MALFORMED;
  P_HAVE(r, 1u + k);
  if (r->p[1] == 0u) return P_E_MALFORMED;          /* a redundant leading zero octet */
  v = P_load_be(r->p + 1, k);
  if (v < 128u) return P_E_MALFORMED;               /* the short form was available */
  if (v > (uint64_t)SIZE_MAX) return P_E_LIMIT;
  r->p += 1u + k;
  *n = (size_t)v;
  return P_OK;
}
static inline P_HOT int P_oer_put_uvar(P_bw *w, uint64_t v) { /* 10.3 e) / 19.1's quantity */
  unsigned k = P_uoctets(v);
  P_NEED(w, 1u + k);
  *w->p++ = (uint8_t)k;
  P_store_be(w->p, v, k);
  w->p += k;
  return P_OK;
}
static inline P_HOT int P_oer_get_uvar(P_br *r, uint64_t *v) {
  size_t k;
  int st = P_oer_get_len(r, &k);
  if (st != P_OK) return st;
  if (k == 0u) return P_E_MALFORMED;
  if (k > 8u) return P_E_LIMIT;
  P_HAVE(r, k);
  if (k > 1u && r->p[0] == 0u) return P_E_MALFORMED; /* 31.4: the fewest octets */
  *v = P_load_be(r->p, (unsigned)k);
  r->p += k;
  return P_OK;
}
static inline P_HOT int P_oer_put_svar(P_bw *w, int64_t v) { /* 10.4 e) */
  unsigned k = P_soctets(v);
  P_NEED(w, 1u + k);
  *w->p++ = (uint8_t)k;
  P_store_be(w->p, (uint64_t)v, k);
  w->p += k;
  return P_OK;
}
static inline P_HOT int P_oer_get_svar(P_br *r, int64_t *v) {
  size_t k;
  uint64_t u;
  int st = P_oer_get_len(r, &k);
  if (st != P_OK) return st;
  if (k == 0u) return P_E_MALFORMED;
  if (k > 8u) return P_E_LIMIT;
  P_HAVE(r, k);
  if (k > 1u && ((r->p[0] == 0x00u && (r->p[1] & 0x80u) == 0u) ||
                 (r->p[0] == 0xFFu && (r->p[1] & 0x80u) != 0u)))
    return P_E_MALFORMED;                           /* 31.4: the fewest octets */
  u = P_load_be(r->p, (unsigned)k);
  if (k < 8u && (r->p[0] & 0x80u)) u |= ~(uint64_t)0 << (8u * k);  /* sign-extend */
  *v = (int64_t)u;
  r->p += k;
  return P_OK;
}
"""

_UPER_IO = r"""
/* --- X.691 UNALIGNED PER bit I/O -------------------------------------------------------------
 * Word-at-a-time. The writer keeps its pending bits MSB-aligned in a 64-bit accumulator and
 * flushes whole octets with one big-endian 8-octet store, so a field costs a shift and an OR
 * and a flush happens once per 64 bits; a run of octets at a bit offset `s` moves eight octets
 * per iteration as `hi | (x >> s)`. The reader loads a big-endian 64-bit window at the bit
 * cursor, so a field of up to 57 bits is one load and two shifts, and an unaligned octet run
 * is reassembled eight octets at a time.
 *
 * The 8-octet store can write SCRATCH past the octets it accounts for (the accumulator's
 * not-yet-final bits), always inside the caller's buffer; the encoding is `out[0..len)`.
 * A codec function keeps the writer in a local and hands back only what a field can change
 * (`p`, `acc`, `n`); `start` and `end` are fixed for the whole encode. */
typedef struct { uint8_t *p, *end, *start; uint64_t acc; unsigned n; } P_pw;
typedef struct { const uint8_t *base; size_t pos, len, nbytes; } P_pr; /* pos/len in bits */

static inline P_HOT uint64_t P_be64(const uint8_t *p) {
#if defined(__GNUC__) && defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  uint64_t v;
  memcpy(&v, p, 8);
  return __builtin_bswap64(v);
#elif defined(__GNUC__) && defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  uint64_t v;
  memcpy(&v, p, 8);
  return v;
#else
  uint64_t v = 0;
  unsigned i;
  for (i = 0; i < 8u; i++) v = (v << 8) | p[i];
  return v;
#endif
}
static inline P_HOT void P_put_be64(uint8_t *p, uint64_t v) {
#if defined(__GNUC__) && defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  v = __builtin_bswap64(v);
  memcpy(p, &v, 8);
#elif defined(__GNUC__) && defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  memcpy(p, &v, 8);
#else
  unsigned i;
  for (i = 0; i < 8u; i++) p[i] = (uint8_t)(v >> (56u - 8u * i));
#endif
}
/* Write the accumulator's whole octets; afterwards fewer than eight bits are pending. */
static inline P_HOT int P_flush(P_pw *w) {
  unsigned k = w->n >> 3, i;
  if (k == 0u) return P_OK;
  if ((size_t)(w->end - w->p) >= 8u) {
    P_put_be64(w->p, w->acc);
  } else {
    if ((size_t)(w->end - w->p) < k) return P_E_SPACE;
    for (i = 0; i < k; i++) w->p[i] = (uint8_t)(w->acc >> (56u - 8u * i));
  }
  w->p += k;
  w->acc = k == 8u ? 0u : w->acc << (8u * k);
  w->n -= 8u * k;
  return P_OK;
}
static inline P_HOT int P_put(P_pw *w, uint64_t v, unsigned bits) { /* bits <= 57, v < 2^bits */
  if (bits == 0u) return P_OK;
  if (w->n + bits > 64u) {
    int st = P_flush(w);
    if (st != P_OK) return st;
  }
  w->acc |= v << (64u - w->n - bits);
  w->n += bits;
  return P_OK;
}
static inline P_HOT int P_put64(P_pw *w, uint64_t v, unsigned bits) { /* bits <= 64 */
  int st;
  if (bits > 32u) {
    st = P_put(w, v >> 32, bits - 32u);
    if (st != P_OK) return st;
    return P_put(w, v & 0xFFFFFFFFu, 32u);
  }
  return P_put(w, v, bits);
}
static inline P_HOT int P_put_octets(P_pw *w, const uint8_t *src, size_t n) {
  size_t i = 0;
  unsigned s;
  uint64_t hi;
  int st = P_flush(w);
  if (st != P_OK) return st;
  if ((size_t)(w->end - w->p) < n) return P_E_SPACE;
  s = w->n;
  if (s == 0u) {
    if (n) memcpy(w->p, src, n);
    w->p += n;
    return P_OK;
  }
  hi = w->acc; /* s pending bits, MSB-aligned */
  /* Each store below writes the next eight octets of the run, all inside the n checked above:
   * the output advances with `i`, so the loop needs no space test of its own. */
  for (; n - i >= 8u; i += 8u) {
    uint64_t x = P_be64(src + i);
    P_put_be64(w->p, hi | (x >> s));
    hi = x << (64u - s);
    w->p += 8;
  }
  for (; i < n; i++) {
    uint64_t x = src[i];
    *w->p++ = (uint8_t)((hi >> 56) | (x >> s));
    hi = x << (64u - s);
  }
  w->acc = hi;
  return P_OK;
}
static inline int P_pw_finish(P_pw *w) { /* 11.1: pad to an octet; an empty encoding is one octet */
  int st = P_flush(w);
  if (st != P_OK) return st;
  if (w->n > 0u) {
    if (w->p == w->end) return P_E_SPACE;
    *w->p++ = (uint8_t)(w->acc >> 56);
    w->acc = 0u;
    w->n = 0u;
  } else if (w->p == w->start) {
    if (w->p == w->end) return P_E_SPACE;
    *w->p++ = 0u;
  }
  return P_OK;
}

/* The last few octets of the input, as a window zero-padded past its end: the slow path. It
 * takes values, not the reader, so the reader's address never escapes into a call and the
 * cursor a decoder keeps in a local can stay in registers. */
static uint64_t P_window(const uint8_t *base, size_t nbytes, size_t byte) {
  uint64_t v = 0;
  unsigned i;
  for (i = 0; i < 8u; i++) v = (v << 8) | (byte + i < nbytes ? base[byte + i] : 0u);
  return v;
}
/* Eight readable octets at the cursor hold at least 57 bits past it, so the one test that
 * picks the single-load path also answers truncation for every field it serves. */
static inline P_HOT int P_get(P_pr *r, unsigned bits, uint64_t *out) { /* bits <= 57 */
  size_t byte = r->pos >> 3;
  uint64_t w;
  if (bits == 0u) { *out = 0; return P_OK; }
  if (byte + 8u <= r->nbytes) {
    w = P_be64(r->base + byte);
  } else {
    if (r->len - r->pos < bits) return P_E_TRUNC;
    w = P_window(r->base, r->nbytes, byte);
  }
  *out = (w << (r->pos & 7u)) >> (64u - bits);
  r->pos += bits;
  return P_OK;
}
static inline P_HOT int P_get64(P_pr *r, unsigned bits, uint64_t *out) { /* bits <= 64 */
  uint64_t hi, lo;
  int st;
  if (bits <= 32u) return P_get(r, bits, out);
  st = P_get(r, bits - 32u, &hi);
  if (st != P_OK) return st;
  st = P_get(r, 32u, &lo);
  if (st != P_OK) return st;
  *out = (hi << 32) | lo;
  return P_OK;
}
static inline P_HOT int P_get_octets(P_pr *r, uint8_t *dst, size_t n) {
  size_t i = 0;
  unsigned s = (unsigned)(r->pos & 7u);
  const uint8_t *src;
  if ((r->len - r->pos) / 8u < n) return P_E_TRUNC;
  src = r->base + (r->pos >> 3);
  r->pos += 8u * n;
  if (s == 0u) {
    if (n) memcpy(dst, src, n);
    return P_OK;
  }
  /* n octets at bit offset s touch n + 1 input octets, so src[i + 8] exists below. */
  for (; n - i >= 8u; i += 8u) P_put_be64(dst + i, (P_be64(src + i) << s) | (src[i + 8u] >> (8u - s)));
  for (; i < n; i++) dst[i] = (uint8_t)((src[i] << s) | (src[i + 1u] >> (8u - s)));
  return P_OK;
}
static inline P_HOT unsigned P_bits_for(uint64_t range_minus_one) { /* ceil(log2(range)) */
  unsigned b = 0;
  while (b < 64u && (range_minus_one >> b) != 0u) b++;
  return b;
}
/* 11.9.3.6/11.9.3.7: an unconstrained length below 16K; longer is fragmentation (refused). */
static inline P_HOT int P_put_ulen(P_pw *w, size_t n) {
  if (n < 128u) return P_put(w, n, 8u);
  if (n < 16384u) return P_put(w, 0x8000u | n, 16u);
  return P_E_LIMIT;
}
static inline P_HOT int P_get_ulen(P_pr *r, size_t *n) {
  uint64_t first, second;
  int st = P_get(r, 8u, &first);
  if (st != P_OK) return st;
  if ((first & 0x80u) == 0u) { *n = (size_t)first; return P_OK; }
  if ((first & 0x40u) != 0u) return P_E_LIMIT;      /* a fragment: not carried here */
  st = P_get(r, 8u, &second);
  if (st != P_OK) return st;
  *n = (size_t)(((first & 0x3Fu) << 8) | second);
  if (*n < 128u) return P_E_MALFORMED;              /* the one-octet form was available */
  return P_OK;
}
/* 11.7/11.8: semi-constrained and unconstrained whole numbers, as a length then octets. */
static inline P_HOT int P_put_uvar(P_pw *w, uint64_t v) {
  unsigned k = P_uoctets(v);
  int st = P_put_ulen(w, k);
  if (st != P_OK) return st;
  return P_put64(w, v, 8u * k);
}
static inline P_HOT int P_get_uvar(P_pr *r, uint64_t *v) {
  size_t k;
  int st = P_get_ulen(r, &k);
  if (st != P_OK) return st;
  if (k == 0u) return P_E_MALFORMED;
  if (k > 8u) return P_E_LIMIT;
  st = P_get64(r, (unsigned)(8u * k), v);
  if (st != P_OK) return st;
  if (k > 1u && (*v >> (8u * (k - 1u))) == 0u) return P_E_MALFORMED;  /* fewest octets */
  return P_OK;
}
static inline P_HOT int P_put_svar(P_pw *w, int64_t v) {
  unsigned k = P_soctets(v);
  uint64_t u = (uint64_t)v;
  int st = P_put_ulen(w, k);
  if (st != P_OK) return st;
  if (k < 8u) u &= ((uint64_t)1 << (8u * k)) - 1u;
  return P_put64(w, u, 8u * k);
}
static inline P_HOT int P_get_svar(P_pr *r, int64_t *v) {
  size_t k;
  uint64_t u;
  int st = P_get_ulen(r, &k);
  if (st != P_OK) return st;
  if (k == 0u) return P_E_MALFORMED;
  if (k > 8u) return P_E_LIMIT;
  st = P_get64(r, (unsigned)(8u * k), &u);
  if (st != P_OK) return st;
  if (k < 8u && ((u >> (8u * k - 1u)) & 1u)) u |= ~(uint64_t)0 << (8u * k);
  if (P_soctets((int64_t)u) != k) return P_E_MALFORMED;               /* fewest octets */
  *v = (int64_t)u;
  return P_OK;
}
"""

_INTX_IO = r"""
/* --- the byte rule's IntX: n leading one bits, a zero, then 7 + 7n value bits --------------- */
static inline P_HOT int P_put_intx(P_bw *w, uint64_t v) {
  unsigned n = 0;
  if (v > P_INTX_MAX) return P_E_VALUE;
  while (n < 7u && v >= ((uint64_t)1 << (7u + 7u * n))) n++;
  P_NEED(w, n + 1u);
  P_store_be(w->p, v, n + 1u);
  if (n) w->p[0] |= (uint8_t)(0xFFu << (8u - n));
  w->p += n + 1u;
  return P_OK;
}
static inline P_HOT int P_get_intx(P_br *r, uint64_t *v) {
  unsigned n = 0;
  uint8_t first;
  P_HAVE(r, 1);
  first = r->p[0];
  while (n < 8u && (first & (0x80u >> n))) n++;
  if (n > 7u) return P_E_MALFORMED;
  P_HAVE(r, 1u + n);
  *v = P_load_be(r->p, 1u + n) & ((((uint64_t)1) << (7u + 7u * n)) - 1u);
  if (n && *v < ((uint64_t)1 << (7u + 7u * (n - 1u)))) return P_E_MALFORMED; /* shortest only */
  r->p += 1u + n;
  return P_OK;
}
static inline P_HOT unsigned P_intx_len(uint64_t v) {
  unsigned n = 0;
  while (n < 7u && v >= ((uint64_t)1 << (7u + 7u * n))) n++;
  return n + 1u;
}
"""


# --- per-rule code emission ------------------------------------------------------------------


class _Emitter:
    """Writes one rule's encoder and decoder for every struct in a model."""

    def __init__(self, model: CModel, rule):
        self.m = model
        self.rule = rule
        self.rname = _rule_name(rule)
        self.P = model.prefix
        self.byte = rule == COER or isinstance(rule, ByteRule)
        # The cursor types in their library spelling, so the helper selection sees the code
        # name them (a codec of fixed octets alone calls no helper that would) and `generate`
        # gives them the file's prefix with the rest.
        self.W = "P_bw" if self.byte else "P_pw"
        self.R = "P_br" if self.byte else "P_pr"
        self._tmp = 0

    def fn(self, verb: str, c: CType) -> str:
        return f"{self.P}{self.rname}_{verb}_{c.cname}"

    def t(self) -> str:
        self._tmp += 1
        return f"t{self._tmp}"

    # -- prototypes and bodies -----------------------------------------------------------------

    def prototypes(self) -> list[str]:
        out = []
        for c in self.m.structs():
            if c.kind == "octfix":
                continue
            out.append(
                f"static int {self.fn('enc', c)}({self.W} *w, const {self.P}{c.cname} *v, "
                f"unsigned depth);"
            )
            out.append(
                f"static int {self.fn('dec', c)}({self.R} *r, {self.P}{c.cname} *v, "
                f"{self.P}arena *a, unsigned depth);"
            )
        return out

    def bodies(self) -> list[str]:
        out = []
        for c in self.m.structs():
            if c.kind == "octfix":
                continue
            out.append(self._enc_fn(c))
            out.append(self._dec_fn(c))
        return out

    def entry_points(self) -> list[str]:
        out = []
        for name in self.m.roots:
            c = self.m.ctype_of(self.m.types[name], name)
            enc, dec = self.fn("enc", c), self.fn("dec", c)
            pub_e = f"{self.P}{self.rname}_encode_{c.cname}"
            pub_d = f"{self.P}{self.rname}_decode_{c.cname}"
            T = f"{self.P}{c.cname}"
            if self.byte:
                out.append(
                    f"int {pub_e}(const {T} *v, uint8_t *out, size_t cap, size_t *len) {{\n"
                    f"  {self.W} w;\n  int st;\n  w.p = out; w.end = out + cap;\n"
                    f"  st = {enc}(&w, v, 0u);\n  if (st != P_OK) return st;\n"
                    f"  *len = (size_t)(w.p - out);\n  return P_OK;\n}}\n"
                )
                out.append(
                    f"int {pub_d}(const uint8_t *in, size_t len, {T} *v, {self.P}arena *a) {{\n"
                    f"  {self.R} r;\n  int st;\n  r.p = in; r.end = in + len;\n"
                    f"  st = {dec}(&r, v, a, 0u);\n  if (st != P_OK) return st;\n"
                    f"  return r.p == r.end ? P_OK : P_E_MALFORMED; /* nothing may follow */\n}}\n"
                )
            else:
                out.append(
                    f"int {pub_e}(const {T} *v, uint8_t *out, size_t cap, size_t *len) {{\n"
                    f"  {self.W} w;\n  int st;\n"
                    f"  w.p = out; w.end = out + cap; w.start = out; w.acc = 0; w.n = 0;\n"
                    f"  st = {enc}(&w, v, 0u);\n  if (st == P_OK) st = P_pw_finish(&w);\n"
                    f"  if (st != P_OK) return st;\n"
                    f"  *len = (size_t)(w.p - out);\n  return P_OK;\n}}\n"
                )
                out.append(
                    f"int {pub_d}(const uint8_t *in, size_t len, {T} *v, {self.P}arena *a) {{\n"
                    f"  {self.R} r;\n  int st;\n  size_t used;\n"
                    f"  if (len > SIZE_MAX / 8u) return P_E_LIMIT;\n"
                    f"  r.base = in; r.pos = 0; r.len = 8u * len; r.nbytes = len;\n"
                    f"  st = {dec}(&r, v, a, 0u);\n  if (st != P_OK) return st;\n"
                    f"  /* 11.1: the encoding is the value's bits padded with ZERO bits to an\n"
                    f"   * octet -- one octet when the value has no bits -- and nothing more. */\n"
                    f"  used = r.pos == 0u ? 8u : (r.pos + 7u) & ~(size_t)7u;\n"
                    f"  if (used != r.len) return P_E_MALFORMED;\n"
                    f"  if (r.pos & 7u) {{\n"
                    f"    uint8_t last = in[len - 1u];\n"
                    f"    if ((uint8_t)(last << (r.pos & 7u)) != 0u) return P_E_MALFORMED;\n"
                    f"  }} else if (r.pos == 0u && in[0] != 0u) return P_E_MALFORMED;\n"
                    f"  return P_OK;\n}}\n"
                )
        return out

    # -- functions per constructed type ----------------------------------------------------------

    # The cursor lives in a LOCAL copy (`ws`/`rs`) for the whole function, and `w`/`r` point at
    # it. Only inlined helpers ever see that address, so scalar replacement keeps the cursor in
    # registers; a call to another generated function writes it back first and reloads it
    # after (`_sync_call`). Without this every field's cursor update was a store the decoder's
    # own `uint8_t` field stores might alias, so it was reloaded around each of them -- about
    # a third of a UPER decode's instructions, by callgrind.

    def _enc_fn(self, c: CType) -> str:
        body = getattr(self, f"_enc_{c.kind}")(c)
        return (
            f"static int {self.fn('enc', c)}({self.W} *win, const {self.P}{c.cname} *v, "
            f"unsigned depth) {{\n"
            f"  int st = P_OK;\n  {self.W} ws = *win;\n  {self.W} *w = &ws;\n  (void)depth;\n"
            + _unused("w", body)
            + body
            + "".join(f"  win->{f} = ws.{f};\n" for f in self._mutable("enc"))
            + "  return st;\n}\n"
        )

    def _dec_fn(self, c: CType) -> str:
        body = getattr(self, f"_dec_{c.kind}")(c)
        return (
            f"static int {self.fn('dec', c)}({self.R} *rin, {self.P}{c.cname} *v, "
            f"{self.P}arena *a, unsigned depth) {{\n"
            f"  int st = P_OK;\n  {self.R} rs = *rin;\n  {self.R} *r = &rs;\n"
            f"  (void)a; (void)depth;\n"
            + _unused("r", body)
            + body
            + "".join(f"  rin->{f} = rs.{f};\n" for f in self._mutable("dec"))
            + "  return st;\n}\n"
        )

    def _mutable(self, direction: str) -> tuple[str, ...]:
        """The cursor fields a call can change: a reader only moves, a bit writer also carries
        its pending bits. Everything else in the cursor is invariant for the whole decode."""
        if direction == "dec":
            return ("p",) if self.byte else ("pos",)
        return ("p",) if self.byte else ("p", "acc", "n")

    def _sync_call(self, call: str, direction: str) -> str:
        """`call` on the CALLER's cursor: its mutable fields written back before and reloaded
        after. Inside a byte rule's vector the reader is `sub`, a local already in memory, and
        is passed as is."""
        if direction == "dec" and getattr(self, "_in_sub", False):
            return f"  st = {call.format(cur='r')};\n  if (st != P_OK) return st;\n"
        outer, local = ("win", "ws") if direction == "enc" else ("rin", "rs")
        fields = self._mutable(direction)
        out = "".join(f"  {outer}->{f} = {local}.{f};\n" for f in fields)
        out += f"  st = {call.format(cur=outer)};\n"
        out += "".join(f"  {local}.{f} = {outer}->{f};\n" for f in fields)
        return out + "  if (st != P_OK) return st;\n"

    # SEQUENCE ---------------------------------------------------------------------------------

    def _preamble_bits(self, c: CType) -> list[str]:
        """The C expressions of the preamble's bits, in order (extension bit first)."""
        bits = ["0"] if c.extensible else []
        for m in c.members:
            if m.optional:
                bits.append(f"(v->has_{m.field} ? 1u : 0u)")
        return bits

    def _enc_seq(self, c: CType) -> str:
        out = []
        bits = self._preamble_bits(c)
        if bits and isinstance(self.rule, ByteRule):
            raise Asn1Error(
                f"cgen: {c.cname} has OPTIONAL components or an extension marker, "
                f"which {self.rname} has no way to spell"
            )
        if bits:
            if self.rule == COER:  # X.696 16.2: the bits, MSB first, zero-padded to octets
                n = (len(bits) + 7) // 8
                out.append(f"  P_NEED(w, {n});\n")
                for k in range(n):
                    parts = [f"({b} << {7 - (i % 8)})" for i, b in enumerate(bits) if i // 8 == k]
                    out.append(f"  w->p[{k}] = (uint8_t)({' | '.join(parts)});\n")
                out.append(f"  w->p += {n};\n")
            else:  # X.691 19.1-19.2: the extension bit, then one bit per OPTIONAL
                # In pieces P_put can carry: a type with more OPTIONAL components than that has
                # a preamble wider than one call's 57 bits.
                for chunk in _bit_chunks(bits):
                    expr = " | ".join(
                        f"((uint64_t){b} << {len(chunk) - 1 - i})" for i, b in enumerate(chunk)
                    )
                    out.append(
                        f"  st = P_put(w, {expr}, {len(chunk)}u);\n  if (st != P_OK) return st;\n"
                    )
        for m in c.members:
            code = self._enc_member(m, f"v->{m.field}")
            if m.optional:
                out.append(f"  if (v->has_{m.field}) {{\n{_indent(code)}  }}\n")
            else:
                out.append(code)
        return "".join(out)

    def _dec_seq(self, c: CType) -> str:
        out = []
        bits = []
        if c.extensible:
            bits.append(None)
        for m in c.members:
            if m.optional:
                bits.append(m)
        if bits and isinstance(self.rule, ByteRule):
            raise Asn1Error(
                f"cgen: {c.cname} has OPTIONAL components or an extension marker, "
                f"which {self.rname} has no way to spell"
            )
        if bits:
            if self.rule == COER:
                n = (len(bits) + 7) // 8
                out.append(f"  P_HAVE(r, {n});\n")
                for i, m in enumerate(bits):
                    test = f"(r->p[{i // 8}] >> {7 - (i % 8)}) & 1u"
                    if m is None:
                        out.append(f"  if ({test}) return P_E_LIMIT; /* additions: not built */\n")
                    else:
                        out.append(f"  v->has_{m.field} = (uint8_t)({test});\n")
                pad = n * 8 - len(bits)
                if pad:  # 16.2.4 (31): the padding bits are zero
                    out.append(f"  if (r->p[{n - 1}] & {(1 << pad) - 1}u) return P_E_MALFORMED;\n")
                out.append(f"  r->p += {n};\n")
            else:
                for chunk in _bit_chunks(bits):  # as the encoder writes it: P_get's 57 at a time
                    t = self.t()
                    out.append(
                        f"  {{\n    uint64_t {t};\n    st = P_get(r, {len(chunk)}u, &{t});\n"
                        f"    if (st != P_OK) return st;\n"
                    )
                    for i, m in enumerate(chunk):
                        test = f"({t} >> {len(chunk) - 1 - i}) & 1u"
                        if m is None:
                            out.append(
                                f"    if ({test}) return P_E_LIMIT; /* additions: not built */\n"
                            )
                        else:
                            out.append(f"    v->has_{m.field} = (uint8_t)({test});\n")
                    out.append("  }\n")
        for m in c.members:
            code = self._dec_member(m, f"v->{m.field}")
            if m.optional:
                out.append(f"  if (v->has_{m.field}) {{\n{_indent(code)}  }}\n")
            else:
                out.append(code)
        return "".join(out)

    # CHOICE -----------------------------------------------------------------------------------

    def _codes(self, c: CType) -> dict:
        codes = self.rule.type_codes.get(c.cname)
        if codes is None:
            raise Asn1Error(f"cgen: {self.rname} has no type codes for the CHOICE {c.cname}")
        return codes

    def _oer_tag(self, c: CType, m: CMember) -> int:
        """X.696 8.7/20.1: the alternative's outermost tag, as its one-octet spelling."""
        tag = m.comp.outer_tag() if m.comp is not None else None
        if tag is None:
            raise Asn1Error(
                f"cgen: {c.cname}.{m.name} is an untagged CHOICE alternative; OER "
                f"needs its outermost tag (X.696 20.1 NOTE 3)"
            )
        if tag.number >= 63:
            raise Asn1Error(f"cgen: {c.cname}.{m.name}'s tag needs X.696 8.7.2.3's long form")
        return (int(tag.cls) << 6) | tag.number

    def _per_index(self, c: CType) -> dict:
        """X.691 23.2: an alternative's index is its place in the CANONICAL tag order."""
        from .per import _ordered_alternatives

        order = [comp.name for comp in _ordered_alternatives(c.asn1)]
        return {m.name: order.index(m.name) for m in c.members}

    def _enc_choice(self, c: CType) -> str:
        out = ["  switch (v->tag) {\n"]
        n = len(c.members)
        index = self._per_index(c) if self.rule == UPER else {}
        for i, m in enumerate(c.members):
            out.append(f"  case {i}u:\n")
            if self.rule == COER:  # 20.1: the alternative's tag
                out.append(f"    P_NEED(w, 1);\n    *w->p++ = 0x{self._oer_tag(c, m):02x}u;\n")
            elif self.rule == UPER:  # 23.5-23.7: the extension bit and the index
                width = (n - 1).bit_length() if n > 1 else 0
                total = width + (1 if c.extensible else 0)
                if total:
                    out.append(
                        f"    st = P_put(w, {index[m.name]}u, {total}u);\n"
                        f"    if (st != P_OK) return st;\n"
                    )
            else:
                codes = self._codes(c)
                if m.name not in codes:
                    out.append("    return P_E_VALUE; /* no type code */\n")
                    continue
                out.append(f"    P_NEED(w, 1);\n    *w->p++ = {codes[m.name]}u;\n")
            out.append(
                _indent(self._enc_member(m, f"v->u.{m.field}")) if m.kind.kind != "null" else ""
            )
            out.append("    break;\n")
        out.append("  default:\n    return P_E_VALUE;\n  }\n")
        return "".join(out)

    def _dec_choice(self, c: CType) -> str:
        out = []
        n = len(c.members)
        t = self.t()
        if self.rule == COER:
            out.append(
                f"  {{\n    unsigned {t};\n    P_HAVE(r, 1);\n    {t} = *r->p++;\n"
                f"    switch ({t}) {{\n"
            )
            for i, m in enumerate(c.members):
                out.append(f"    case 0x{self._oer_tag(c, m):02x}u: v->tag = {i}u; break;\n")
            # An unknown tag is malformed for a non-extensible CHOICE; for an extensible one it
            # is an alternative a newer version added, which this codec does not carry.
            unknown = "P_E_LIMIT" if c.extensible else "P_E_MALFORMED"
            out.append(f"    default: return {unknown};\n    }}\n  }}\n")
        elif self.rule == UPER:
            width = (n - 1).bit_length() if n > 1 else 0
            total = width + (1 if c.extensible else 0)
            index = self._per_index(c)
            if total:
                out.append(
                    f"  {{\n    uint64_t {t};\n    st = P_get(r, {total}u, &{t});\n"
                    f"    if (st != P_OK) return st;\n"
                )
                if c.extensible:
                    out.append(
                        f"    if (({t} >> {width}) & 1u) return P_E_LIMIT; "
                        f"/* an extension alternative: not built */\n"
                    )
                mask = (1 << width) - 1 if width else 0
                out.append(f"    switch ({t} & {mask}u) {{\n")
                for i, m in enumerate(c.members):
                    out.append(f"    case {index[m.name]}u: v->tag = {i}u; break;\n")
                out.append("    default: return P_E_MALFORMED;\n    }\n  }\n")
            else:
                out.append("  v->tag = 0u;\n")
        else:
            codes = self._codes(c)
            out.append(
                f"  {{\n    unsigned {t};\n    P_HAVE(r, 1);\n    {t} = *r->p++;\n"
                f"    switch ({t}) {{\n"
            )
            for i, m in enumerate(c.members):
                if m.name in codes:
                    out.append(f"    case {codes[m.name]}u: v->tag = {i}u; break;\n")
            out.append("    default: return P_E_MALFORMED;\n    }\n  }\n")
        out.append("  switch (v->tag) {\n")
        for i, m in enumerate(c.members):
            out.append(f"  case {i}u:\n")
            if m.kind.kind != "null":
                out.append(_indent(self._dec_member(m, f"v->u.{m.field}")))
            out.append("    break;\n")
        out.append("  default:\n    return P_E_MALFORMED;\n  }\n")
        return "".join(out)

    # SEQUENCE OF ------------------------------------------------------------------------------

    def _count_bounds(self, c: CType):
        from .per import _size_bounds

        low, high, ext = _size_bounds(c.asn1)
        return low, high, ext

    def _enc_seqof(self, c: CType) -> str:
        low, high, ext = self._count_bounds(c)
        if ext:
            raise Asn1Error(
                f"cgen: {c.cname} has an extensible SIZE, which this compiler does not build"
            )
        out = ["  size_t i;\n  if (v->n != 0u && v->v == NULL) return P_E_VALUE;\n"]
        if low:
            out.append(f"  if (v->n < {low}u) return P_E_VALUE;\n")
        if high is not None:
            out.append(f"  if (v->n > {high}u) return P_E_VALUE;\n")
        if self.rule == COER:  # 19.1: the quantity
            out.append("  st = P_oer_put_uvar(w, (uint64_t)v->n);\n  if (st != P_OK) return st;\n")
        elif self.rule == UPER:  # 20.6 via 11.9
            out.append(self._per_put_length("v->n", low, high))
        else:  # the byte rule: the body's length in octets, as an IntX, BEFORE the body
            out.append(
                "  {\n    uint8_t *len_at, *body;\n    size_t body_len;\n    unsigned k;\n"
                "    P_NEED(w, 1);\n    len_at = w->p++;\n    body = w->p;\n"
                "    for (i = 0; i < v->n; i++) {\n"
                + _indent(_indent(self._enc_member(c.element, "v->v[i]")))
                + "    }\n"
                "    body_len = (size_t)(w->p - body);\n"
                "    k = P_intx_len((uint64_t)body_len);\n"
                "    if ((uint64_t)body_len > P_INTX_MAX) return P_E_VALUE;\n"
                "    if (k > 1u) { /* the length outgrew its reserved octet: move the body */\n"
                "      P_NEED(w, k - 1u);\n"
                "      memmove(body + (k - 1u), body, body_len);\n"
                "      w->p += k - 1u;\n"
                "    }\n"
                "    {\n      P_bw lw;\n      lw.p = len_at; lw.end = len_at + k;\n"
                "      st = P_put_intx(&lw, (uint64_t)body_len);\n"
                "      if (st != P_OK) return st;\n    }\n  }\n"
            )
            return "".join(out)
        out.append("  for (i = 0; i < v->n; i++) {\n")
        out.append(_indent(self._enc_member(c.element, "v->v[i]")))
        out.append("  }\n")
        return "".join(out)

    def _dec_seqof(self, c: CType) -> str:
        low, high, _ext = self._count_bounds(c)
        elem = self._ctype_decl(c.element)
        out = ["  size_t i;\n"]
        if isinstance(self.rule, ByteRule):
            # The count is not on the wire: a vector gives its length in OCTETS. When every
            # element has one size the count is a division and the array is allocated once,
            # exactly. Otherwise the array starts at four slots and doubles as the elements
            # decode -- IN PLACE while it is still the arena's last allocation (its elements
            # allocated nothing), moved to a fresh one only when they did. It used to move at
            # every doubling, abandoning each outgrown copy, so a long vector took about four
            # times its own size and ran out of arena well before the arena was full.
            #
            # This is the time-optimal spelling, and the arena it reserves is more than the
            # structs need: the capacity a short vector leaves unused stays reserved, since
            # giving it back costs a compare and a store per vector -- 4-10% of the whole
            # decode's instructions on these envelopes, by callgrind -- and counting the
            # elements first (one pass over each vector before decoding it) sizes every array
            # exactly but measured 1.5-1.8x the decode's time. V1.1.1 has no count to size
            # an array by; the study reports both the arena this decoder reserves and what the
            # same structs need (`docs/research/BCIR_ITS_ASN1_BINARY_STUDY.md` §5). The
            # elements read through `sub`, a reader over exactly the vector's octets, so no
            # element can run past the vector.
            if self._min_bits(c.element) == 0:
                raise Asn1Error(
                    f"cgen: {c.cname}'s element can encode in zero octets, so {self.rname} "
                    f"cannot count its vector; this compiler refuses it"
                )
            size = self._byte_size(c.element)
            out.append(
                "  {\n    uint64_t body_len;\n    P_br sub;\n"
                "    st = P_get_intx(r, &body_len);\n    if (st != P_OK) return st;\n"
                "    if (body_len > (uint64_t)(r->end - r->p)) return P_E_TRUNC;\n"
                "    sub.p = r->p; sub.end = r->p + (size_t)body_len;\n"
            )
            if size is not None:
                out.append(
                    f"    size_t n;\n    if (body_len % {size}u != 0u) return P_E_MALFORMED;\n"
                    f"    n = (size_t)(body_len / {size}u);\n"
                )
                if low:  # the SIZE bounds, on the count, before anything is allocated for it
                    out.append(f"    if (n < {low}u) return P_E_MALFORMED;\n")
                if high is not None:
                    out.append(f"    if (n > {high}u) return P_E_MALFORMED;\n")
                out.append(
                    "    v->n = n;\n    v->v = NULL;\n    if (n) {\n"
                    f"      v->v = ({elem} *)P_alloc(a, n, sizeof({elem}), _Alignof({elem}));\n"
                    "      if (v->v == NULL) return P_E_LIMIT;\n    }\n"
                    "    for (i = 0; i < n; i++) {\n"
                    "      P_br *r = &sub; /* the element reads the vector, not past it */\n"
                    + _indent(_indent(self._in_sub_member(c.element, "v->v[i]")))
                    + "    }\n    r->p = sub.end;\n  }\n"
                )
                return "".join(out)
            # The array is the arena's last allocation exactly when its end is the arena's fill;
            # the test reads only what the loop already holds, so the loop is unchanged by it.
            out.append(
                "    {\n      size_t cap = 4u;\n"
                f"      {elem} *items = ({elem} *)P_alloc(a, cap, sizeof({elem}), "
                f"_Alignof({elem}));\n"
                "      if (items == NULL) return P_E_LIMIT;\n"
                "      v->n = 0u;\n"
                "      while (sub.p != sub.end) {\n"
                "        if (v->n == cap) {\n"
                "          if ((uint8_t *)(items + cap) == a->base + a->used &&\n"
                f"              (a->cap - a->used) / sizeof({elem}) >= cap) {{\n"
                f"            a->used += cap * sizeof({elem}); /* in place */\n"
                "          } else {\n"
                f"            {elem} *grown = ({elem} *)P_alloc(a, 2u * cap, sizeof({elem}), "
                f"_Alignof({elem}));\n"
                "            if (grown == NULL) return P_E_LIMIT;\n"
                f"            memcpy(grown, items, cap * sizeof({elem}));\n"
                "            items = grown;\n          }\n"
                "          cap *= 2u;\n        }\n"
                "        {\n          P_br *r = &sub; /* the element reads the vector, not past it */\n"
                + _indent(_indent(_indent(_indent(self._in_sub_member(c.element, "items[v->n]")))))
                + "        }\n        v->n++;\n      }\n"
                "      v->v = items;\n    }\n"
                "    r->p = sub.end;\n  }\n"
            )
            if low:
                out.append(f"  if (v->n < {low}u) return P_E_MALFORMED;\n")
            if high is not None:
                out.append(f"  if (v->n > {high}u) return P_E_MALFORMED;\n")
            out.append("  (void)i;\n")
            return "".join(out)
        t = self.t()
        if self.rule == COER:
            out.append(
                f"  {{\n    uint64_t {t};\n    st = P_oer_get_uvar(r, &{t});\n"
                f"    if (st != P_OK) return st;\n"
                f"    if ({t} > (uint64_t)SIZE_MAX) return P_E_LIMIT;\n"
                f"    v->n = (size_t){t};\n  }}\n"
            )
        else:
            out.append(self._per_get_length("v->n", low, high))
        if low:
            out.append(f"  if (v->n < {low}u) return P_E_MALFORMED;\n")
        if high is not None:
            out.append(f"  if (v->n > {high}u) return P_E_MALFORMED;\n")
        # A hostile count must not reserve arena memory the input cannot fill. An element of at
        # least one bit (UPER) or one octet (COER) bounds the count by the input left. One that
        # can encode in no bits bounds nothing -- ten empty elements are a valid encoding of a
        # count of ten -- so only the SIZE upper bound checked above can, and without one the
        # type is refused at generation time.
        if self._min_bits(c.element) == 0:
            if high is None:
                raise Asn1Error(
                    f"cgen: {c.cname}'s element can encode in zero bits, so a count "
                    f"is not bounded by the input; this compiler refuses it"
                )
        elif self.rule == COER:
            out.append("  if (v->n > (size_t)(r->end - r->p)) return P_E_TRUNC;\n")
        else:
            out.append("  if (v->n > r->len - r->pos) return P_E_TRUNC;\n")
        out.append(
            f"  v->v = NULL;\n  if (v->n) {{\n"
            f"    v->v = ({elem} *)P_alloc(a, v->n, sizeof({elem}), _Alignof({elem}));\n"
            f"    if (v->v == NULL) return P_E_LIMIT;\n  }}\n"
            f"  for (i = 0; i < v->n; i++) {{\n"
        )
        out.append(_indent(self._dec_member(c.element, "v->v[i]")))
        out.append("  }\n")
        return "".join(out)

    def _in_sub_member(self, m: CMember, lv: str) -> str:
        self._in_sub = True
        try:
            return self._dec_member(m, lv)
        finally:
            self._in_sub = False

    def _byte_size(self, m: CMember):
        """The octets every encoding of `m` takes under the byte rule, or None when it varies --
        a vector of such elements is counted by division, with no pass at all."""
        k = m.kind
        if m.pointer:
            return None  # a recursive reference: no fixed size
        if k.kind == "octfix":
            return k.size
        if k.kind == "enum":
            return 1
        if k.kind == "null":
            return 0
        if k.kind == "int":
            return _byte_rule_width(k)
        if k.kind == "seq":
            sizes = [self._byte_size(x) for x in k.members]
            return None if any(s is None for s in sizes) else sum(sizes)
        return None  # a CHOICE's alternatives and a vector's length vary

    def _min_bits(self, m: CMember) -> int:
        """A lower bound on the bits of any encoding of `m` under this rule, and zero EXACTLY
        when some value of it encodes in no bits at all -- the one fact a count's bound turns
        on. UPER spells several such values: a single-value INTEGER, a one-enumerator
        ENUMERATED, a one-alternative CHOICE of one, a SIZE-fixed count of them (13.2.1, 14.2,
        23.4, 20.5). This used to answer 1 for every one of those, so a SEQUENCE OF them
        refused its own valid encodings as truncated."""
        from .per import _size_bounds, _value_bounds

        k = m.kind
        if m.pointer:
            return self._min_bits(CMember(m.name, k))
        if k.kind == "null":
            return 0
        if k.kind == "octfix":
            return 8 * k.size
        if k.kind == "seq":
            if k.extensible or any(x.optional for x in k.members):
                return 1 if self.rule == UPER else 8  # the preamble
            return sum(self._min_bits(x) for x in k.members if not x.optional)
        if self.rule != UPER:
            return 8  # octet-aligned: a tag, a type code, a length or the value's own octet
        if k.kind == "int":
            (low, high), ext = _value_bounds(k.asn1)
            return 0 if not ext and low is not None and low == high else 1
        if k.kind == "enum":
            return 0 if len(k.asn1.enumeration) == 1 and not k.extensible else 1
        if k.kind == "choice":
            if len(k.members) == 1 and not k.extensible:
                return self._min_bits(k.members[0])
            return 1
        if k.kind in ("octvar", "seqof"):
            low, high, ext = _size_bounds(k.asn1)
            if not ext and high is not None and low == high:  # no length on the wire
                return 8 * low if k.kind == "octvar" else low * self._min_bits(k.element)
            return 1
        return 1  # BOOLEAN

    # PER length helpers -----------------------------------------------------------------------

    def _per_put_length(self, expr: str, low: int, high) -> str:
        if high is not None and high < 65536:  # 11.9.3.3: a constrained whole number
            if high == low:
                return ""
            bits = (high - low).bit_length()
            return (
                f"  st = P_put(w, (uint64_t)({expr} - {low}u), {bits}u);\n"
                f"  if (st != P_OK) return st;\n"
            )
        return f"  st = P_put_ulen(w, {expr});\n  if (st != P_OK) return st;\n"

    def _per_get_length(self, lvalue: str, low: int, high) -> str:
        t = self.t()
        if high is not None and high < 65536:
            if high == low:
                return f"  {lvalue} = {low}u;\n"
            bits = (high - low).bit_length()
            return (
                f"  {{\n    uint64_t {t};\n    st = P_get(r, {bits}u, &{t});\n"
                f"    if (st != P_OK) return st;\n"
                f"    if ({t} > {high - low}u) return P_E_MALFORMED;\n"
                f"    {lvalue} = (size_t)({t} + {low}u);\n  }}\n"
            )
        return f"  st = P_get_ulen(r, &{lvalue});\n  if (st != P_OK) return st;\n"

    # members ----------------------------------------------------------------------------------

    def _ctype_decl(self, m: CMember) -> str:
        k = m.kind
        if k.kind in ("seq", "choice", "seqof", "octfix"):
            return f"{self.P}{k.cname}"
        if k.kind == "octvar":
            return f"{self.P}octets"
        if k.kind == "null":
            return "uint8_t"
        return k.ctype

    def _enc_member(self, m: CMember, lv: str) -> str:
        k = m.kind
        if m.pointer:
            # A caller's value can be cyclic (a certificate whose signer points back at it), so
            # the encoder bounds the recursion exactly as the decoder does.
            return (
                f"  if ({lv} == NULL) return P_E_VALUE;\n"
                f"  if (depth >= P_MAX_DEPTH) return P_E_LIMIT;\n"
                + self._sync_call(f"{self.fn('enc', k)}({{cur}}, {lv}, depth + 1u)", "enc")
            )
        if k.kind in ("seq", "choice", "seqof"):
            if m.recursive:
                return "  if (depth >= P_MAX_DEPTH) return P_E_LIMIT;\n" + self._sync_call(
                    f"{self.fn('enc', k)}({{cur}}, &{lv}, depth + 1u)", "enc"
                )
            return self._sync_call(f"{self.fn('enc', k)}({{cur}}, &{lv}, depth)", "enc")
        return getattr(self, f"_enc_prim_{self._family()}")(k, lv)

    def _dec_member(self, m: CMember, lv: str) -> str:
        k = m.kind
        if m.pointer:
            T = f"{self.P}{k.cname}"
            return (
                f"  if (depth >= P_MAX_DEPTH) return P_E_LIMIT;\n"
                f"  {lv} = ({T} *)P_alloc(a, 1u, sizeof({T}), _Alignof({T}));\n"
                f"  if ({lv} == NULL) return P_E_LIMIT;\n"
                + self._sync_call(f"{self.fn('dec', k)}({{cur}}, {lv}, a, depth + 1u)", "dec")
            )
        if k.kind in ("seq", "choice", "seqof"):
            if m.recursive:
                return "  if (depth >= P_MAX_DEPTH) return P_E_LIMIT;\n" + self._sync_call(
                    f"{self.fn('dec', k)}({{cur}}, &{lv}, a, depth + 1u)", "dec"
                )
            return self._sync_call(f"{self.fn('dec', k)}({{cur}}, &{lv}, a, depth)", "dec")
        return getattr(self, f"_dec_prim_{self._family()}")(k, lv)

    def _family(self) -> str:
        if self.rule == COER:
            return "coer"
        if self.rule == UPER:
            return "uper"
        return "byte"

    # primitives: COER -------------------------------------------------------------------------

    def _enc_prim_coer(self, k: CType, lv: str) -> str:
        from .oer import _integer_form

        if k.kind == "null":
            return ""
        if k.kind == "bool":
            return f"  P_NEED(w, 1);\n  *w->p++ = {lv} ? 0xFFu : 0x00u;\n"
        if k.kind == "enum":
            # 11.3: an enumerator's NUMBER, one octet for 0..127. An extensible type also
            # carries a number its root does not name -- one a newer version added -- since the
            # number itself is on the wire (unlike PER, whose extension index needs the list).
            check = (
                f"  if ({lv} > 127u) return P_E_VALUE;\n"
                if k.extensible
                else self._enum_check(k, lv)
            )
            return check + f"  P_NEED(w, 1);\n  *w->p++ = (uint8_t){lv};\n"
        if k.kind == "octfix":
            return (
                f"  P_NEED(w, {k.size});\n  memcpy(w->p, {lv}.b, {k.size});\n  w->p += {k.size};\n"
            )
        if k.kind == "octvar":
            return (
                self._size_check(k, lv, "P_E_VALUE")
                + f"  if ({lv}.n != 0u && {lv}.p == NULL) return P_E_VALUE;\n"
                + f"  st = P_oer_put_len(w, {lv}.n);\n  if (st != P_OK) return st;\n"
                + f"  P_NEED(w, {lv}.n);\n  if ({lv}.n) memcpy(w->p, {lv}.p, {lv}.n);\n"
                + f"  w->p += {lv}.n;\n"
            )
        if k.kind == "int":
            width, signed = _integer_form(k.asn1)
            check = self._int_check(k, lv)
            if width is not None:
                return (
                    check + f"  P_NEED(w, {width});\n"
                    f"  P_store_be(w->p, (uint64_t){lv}, {width});\n  w->p += {width};\n"
                )
            if signed:
                return (
                    check
                    + f"  st = P_oer_put_svar(w, (int64_t){lv});\n  if (st != P_OK) return st;\n"
                )
            return (
                check + f"  st = P_oer_put_uvar(w, (uint64_t){lv});\n  if (st != P_OK) return st;\n"
            )
        raise Asn1Error(f"cgen: no COER encoding for {k.kind}")  # pragma: no cover

    def _dec_prim_coer(self, k: CType, lv: str) -> str:
        from .oer import _integer_form

        t = self.t()
        if k.kind == "null":
            return ""
        if k.kind == "bool":
            return (
                f"  P_HAVE(r, 1);\n  if (r->p[0] != 0x00u && r->p[0] != 0xFFu) "
                f"return P_E_MALFORMED;\n  {lv} = r->p[0] ? 1u : 0u;\n  r->p++;\n"
            )
        if k.kind == "enum":
            # The long form (11.4) is refused: every enumerator here is 0..127 (checked at
            # generation), and a value a newer version numbered above 127 is not carried.
            check = "" if k.extensible else self._enum_check(k, lv, "P_E_MALFORMED")
            return (
                f"  P_HAVE(r, 1);\n  if (r->p[0] & 0x80u) return P_E_LIMIT;\n"
                f"  {lv} = r->p[0];\n  r->p++;\n" + check
            )
        if k.kind == "octfix":
            return (
                f"  P_HAVE(r, {k.size});\n  memcpy({lv}.b, r->p, {k.size});\n  r->p += {k.size};\n"
            )
        if k.kind == "octvar":  # zero copy: the view points into the input
            return (
                f"  st = P_oer_get_len(r, &{lv}.n);\n  if (st != P_OK) return st;\n"
                f"  P_HAVE(r, {lv}.n);\n  {lv}.p = r->p;\n  r->p += {lv}.n;\n"
                + self._size_check(k, lv, "P_E_MALFORMED")
            )
        if k.kind == "int":
            width, signed = _integer_form(k.asn1)
            if width is not None:
                load = f"P_load_be(r->p, {width})"
                if signed:
                    bits = 8 * width
                    load = (
                        (
                            f"(int64_t)(({load} ^ ((uint64_t)1 << {bits - 1})) - "
                            f"((uint64_t)1 << {bits - 1}))"
                        )
                        if bits < 64
                        else f"(int64_t){load}"
                    )
                return (
                    f"  {{\n    P_HAVE(r, {width});\n    {k.ctype} {t} = ({k.ctype}){load};\n"
                    f"    r->p += {width};\n"
                    + _indent(self._int_check(k, t, "P_E_MALFORMED"))
                    + f"    {lv} = {t};\n  }}\n"
                )
            # The value constraint is checked on the way in exactly as on the way out: a decoder
            # that accepted what its own encoder refuses would hand the caller a value no
            # canonical encoding of the type carries (`_int_check`'s one spelling, both ways).
            if signed:
                return (
                    f"  {{\n    int64_t {t};\n    st = P_oer_get_svar(r, &{t});\n"
                    f"    if (st != P_OK) return st;\n"
                    + _indent(self._int_range_from(k, t, signed=True))
                    + f"    {lv} = ({k.ctype}){t};\n  }}\n"
                    + self._int_check(k, lv, "P_E_MALFORMED")
                )
            return (
                f"  {{\n    uint64_t {t};\n    st = P_oer_get_uvar(r, &{t});\n"
                f"    if (st != P_OK) return st;\n"
                + _indent(self._int_range_from(k, t, signed=False))
                + f"    {lv} = ({k.ctype}){t};\n  }}\n"
                + self._int_check(k, lv, "P_E_MALFORMED")
            )
        raise Asn1Error(f"cgen: no COER decoding for {k.kind}")  # pragma: no cover

    # primitives: UPER -------------------------------------------------------------------------

    def _enc_prim_uper(self, k: CType, lv: str) -> str:
        from .per import _size_bounds, _value_bounds

        if k.kind == "null":
            return ""
        if k.kind == "bool":
            return f"  st = P_put(w, {lv} ? 1u : 0u, 1u);\n  if (st != P_OK) return st;\n"
        if k.kind == "enum":  # 14: the index among the root, sorted by number
            numbers = sorted(n for _name, n in k.asn1.enumeration)
            width = (len(numbers) - 1).bit_length() if len(numbers) > 1 else 0
            out = [self._enum_check(k, lv)]
            index = self._enum_index_expr(numbers, lv)
            total = width + (1 if k.extensible else 0)
            if total:
                out.append(
                    f"  st = P_put(w, (uint64_t)({index}), {total}u);\n"
                    f"  if (st != P_OK) return st;\n"
                )
            return "".join(out)
        if k.kind == "octfix":  # 17.6/17.7: no length; UNALIGNED never aligns
            _uper_bare_size(k)
            return f"  st = P_put_octets(w, {lv}.b, {k.size});\n  if (st != P_OK) return st;\n"
        if k.kind == "octvar":
            low, high, ext = _size_bounds(k.asn1)
            if ext:
                raise Asn1Error("cgen: an extensible SIZE is not built for UPER")
            return (
                self._size_check(k, lv, "P_E_VALUE")
                + f"  if ({lv}.n != 0u && {lv}.p == NULL) return P_E_VALUE;\n"
                + self._per_put_length(f"{lv}.n", low, high)
                + f"  st = P_put_octets(w, {lv}.p, {lv}.n);\n  if (st != P_OK) return st;\n"
            )
        if k.kind == "int":
            (low, high), ext = _value_bounds(k.asn1)
            out = []
            if ext:  # 13.1: one bit, then the root form inside the root, unconstrained outside
                out.append(
                    f"  if ({_root_test(f'(int64_t){lv}', low, high, k)}) {{\n"
                    f"    st = P_put(w, 0u, 1u);\n    if (st != P_OK) return st;\n"
                    + _indent(self._per_int_root(lv, low, high))
                    + f"  }} else {{\n    st = P_put(w, 1u, 1u);\n"
                    f"    if (st != P_OK) return st;\n"
                    f"    st = P_put_svar(w, (int64_t){lv});\n"
                    f"    if (st != P_OK) return st;\n  }}\n"
                )
                return "".join(out)
            return self._int_check(k, lv) + self._per_int_root(lv, low, high)
        raise Asn1Error(f"cgen: no UPER encoding for {k.kind}")  # pragma: no cover

    def _per_int_root(self, lv: str, low, high) -> str:
        if low is not None and high is not None:  # 13.2.6: a constrained whole number
            span = high - low
            if span == 0:
                return ""
            bits = span.bit_length()
            off = f"((uint64_t){lv} - (uint64_t)({_c_int(low)}))"
            put = "P_put64" if bits > 56 else "P_put"
            return f"  st = {put}(w, {off}, {bits}u);\n  if (st != P_OK) return st;\n"
        if low is not None:  # 13.2.5 / 11.7: semi-constrained
            return (
                f"  st = P_put_uvar(w, (uint64_t){lv} - (uint64_t)({_c_int(low)}));\n"
                f"  if (st != P_OK) return st;\n"
            )
        return f"  st = P_put_svar(w, (int64_t){lv});\n  if (st != P_OK) return st;\n"

    def _dec_prim_uper(self, k: CType, lv: str) -> str:
        from .per import _size_bounds, _value_bounds

        t = self.t()
        if k.kind == "null":
            return ""
        if k.kind == "bool":
            return (
                f"  {{\n    uint64_t {t};\n    st = P_get(r, 1u, &{t});\n"
                f"    if (st != P_OK) return st;\n    {lv} = (uint8_t){t};\n  }}\n"
            )
        if k.kind == "enum":
            numbers = sorted(n for _name, n in k.asn1.enumeration)
            width = (len(numbers) - 1).bit_length() if len(numbers) > 1 else 0
            total = width + (1 if k.extensible else 0)
            out = [f"  {{\n    uint64_t {t} = 0;\n"]
            if total:
                out.append(f"    st = P_get(r, {total}u, &{t});\n    if (st != P_OK) return st;\n")
            if k.extensible:
                out.append(
                    f"    if (({t} >> {width}) & 1u) return P_E_LIMIT; "
                    f"/* an extension enumerator: not built */\n"
                )
            mask = (1 << width) - 1 if width else 0
            out.append(f"    {t} &= {mask}u;\n    switch ({t}) {{\n")
            for i, n in enumerate(numbers):
                out.append(f"    case {i}u: {lv} = {n}u; break;\n")
            out.append("    default: return P_E_MALFORMED;\n    }\n  }\n")
            return "".join(out)
        if k.kind == "octfix":
            _uper_bare_size(k)
            return f"  st = P_get_octets(r, {lv}.b, {k.size});\n  if (st != P_OK) return st;\n"
        if k.kind == "octvar":  # copied: an unaligned string has no octet slice to point at
            low, high, _ext = _size_bounds(k.asn1)
            return (
                self._per_get_length(f"{lv}.n", low, high)
                + f"  if ({lv}.n > (r->len - r->pos) / 8u) return P_E_TRUNC;\n"
                + f"  {{\n    uint8_t *{t} = NULL;\n"
                + f"    if ({lv}.n) {{\n"
                + f"      {t} = (uint8_t *)P_alloc(a, {lv}.n, 1u, 1u);\n"
                + f"      if ({t} == NULL) return P_E_LIMIT;\n    }}\n"
                + f"    st = P_get_octets(r, {t}, {lv}.n);\n    if (st != P_OK) return st;\n"
                + f"    {lv}.p = {t};\n  }}\n"
                + self._size_check(k, lv, "P_E_MALFORMED")
            )
        if k.kind == "int":
            (low, high), ext = _value_bounds(k.asn1)
            if ext:
                u = self.t()
                return (
                    f"  {{\n    uint64_t {u};\n    st = P_get(r, 1u, &{u});\n"
                    f"    if (st != P_OK) return st;\n    if ({u} == 0u) {{\n"
                    + _indent(_indent(self._per_int_root_dec(k, lv, low, high)))
                    + f"    }} else {{\n      int64_t {t};\n"
                    f"      st = P_get_svar(r, &{t});\n      if (st != P_OK) return st;\n"
                    f"      /* inside the root a value MUST take the root form */\n"
                    f"      if ({_root_test(t, low, high, k)}) return P_E_MALFORMED;\n"
                    f"      {lv} = ({k.ctype}){t};\n    }}\n  }}\n"
                )
            return self._per_int_root_dec(k, lv, low, high)
        raise Asn1Error(f"cgen: no UPER decoding for {k.kind}")  # pragma: no cover

    def _per_int_root_dec(self, k: CType, lv: str, low, high) -> str:
        t = self.t()
        if low is not None and high is not None:
            span = high - low
            if span == 0:
                return f"  {lv} = ({k.ctype})({_c_int(low)});\n"
            bits = span.bit_length()
            get = "P_get64" if bits > 57 else "P_get"
            check = (
                ""
                if span == (1 << bits) - 1
                else (f"    if ({t} > {span}u) return P_E_MALFORMED;\n")
            )
            return (
                f"  {{\n    uint64_t {t};\n    st = {get}(r, {bits}u, &{t});\n"
                f"    if (st != P_OK) return st;\n{check}"
                f"    {lv} = ({k.ctype})({t} + (uint64_t)({_c_int(low)}));\n  }}\n"
            )
        # A semi-constrained or unconstrained root form carries no upper bound on the wire (13.2.5,
        # 13.2.4: `(MIN..10)` is unconstrained in PER), so the type's own bounds are checked
        # after the read, as the encoder checks them before the write.
        if low is not None:
            return (
                f"  {{\n    uint64_t {t};\n    st = P_get_uvar(r, &{t});\n"
                f"    if (st != P_OK) return st;\n"
                + _indent(self._int_range_from(k, t, signed=False, offset=low))
                + f"    {lv} = ({k.ctype})({t} + (uint64_t)({_c_int(low)}));\n  }}\n"
                + self._int_check(k, lv, "P_E_MALFORMED")
            )
        return (
            f"  {{\n    int64_t {t};\n    st = P_get_svar(r, &{t});\n"
            f"    if (st != P_OK) return st;\n    {lv} = ({k.ctype}){t};\n  }}\n"
            + self._int_check(k, lv, "P_E_MALFORMED")
        )

    # primitives: the byte rule ----------------------------------------------------------------

    def _enc_prim_byte(self, k: CType, lv: str) -> str:
        if k.kind == "null":
            return ""
        if k.kind == "enum":
            return self._enum_check(k, lv) + f"  P_NEED(w, 1);\n  *w->p++ = (uint8_t){lv};\n"
        if k.kind == "octfix":
            return (
                f"  P_NEED(w, {k.size});\n  memcpy(w->p, {lv}.b, {k.size});\n  w->p += {k.size};\n"
            )
        if k.kind == "octvar":
            return (
                self._size_check(k, lv, "P_E_VALUE")
                + f"  if ({lv}.n != 0u && {lv}.p == NULL) return P_E_VALUE;\n"
                f"  st = P_put_intx(w, (uint64_t){lv}.n);\n  if (st != P_OK) return st;\n"
                f"  P_NEED(w, {lv}.n);\n  if ({lv}.n) memcpy(w->p, {lv}.p, {lv}.n);\n"
                f"  w->p += {lv}.n;\n"
            )
        if k.kind == "int":
            width = _byte_rule_width(k)
            if width is None:
                return (
                    self._int_check(k, lv)
                    + f"  st = P_put_intx(w, (uint64_t){lv});\n  if (st != P_OK) return st;\n"
                )
            return (
                self._int_check(k, lv) + f"  P_NEED(w, {width});\n"
                f"  P_store_be(w->p, (uint64_t){lv}, {width});\n  w->p += {width};\n"
            )
        raise Asn1Error(f"cgen: {self.rname} has no encoding for {k.kind}")

    def _dec_prim_byte(self, k: CType, lv: str) -> str:
        t = self.t()
        if k.kind == "null":
            return ""
        if k.kind == "enum":
            return f"  P_HAVE(r, 1);\n  {lv} = r->p[0];\n  r->p++;\n" + self._enum_check(
                k, lv, "P_E_MALFORMED"
            )
        if k.kind == "octfix":
            return (
                f"  P_HAVE(r, {k.size});\n  memcpy({lv}.b, r->p, {k.size});\n  r->p += {k.size};\n"
            )
        if k.kind == "octvar":  # zero copy
            return (
                f"  {{\n    uint64_t {t};\n    st = P_get_intx(r, &{t});\n"
                f"    if (st != P_OK) return st;\n"
                f"    if ({t} > (uint64_t)(r->end - r->p)) return P_E_TRUNC;\n"
                f"    {lv}.p = r->p;\n    {lv}.n = (size_t){t};\n    r->p += {lv}.n;\n  }}\n"
                + self._size_check(k, lv, "P_E_MALFORMED")
            )
        if k.kind == "int":
            width = _byte_rule_width(k)
            if width is None:
                return (
                    f"  {{\n    uint64_t {t};\n    st = P_get_intx(r, &{t});\n"
                    f"    if (st != P_OK) return st;\n    {lv} = ({k.ctype}){t};\n  }}\n"
                    + self._int_check(k, lv, "P_E_MALFORMED")
                )
            load = f"P_load_be(r->p, {width})"
            if k.signed:
                bits = 8 * width
                load = (
                    f"(int64_t)(({load} ^ ((uint64_t)1 << {bits - 1})) - "
                    f"((uint64_t)1 << {bits - 1}))"
                )
            return (
                f"  P_HAVE(r, {width});\n  {lv} = ({k.ctype}){load};\n  r->p += {width};\n"
                + self._int_check(k, lv, "P_E_MALFORMED")
            )
        raise Asn1Error(f"cgen: {self.rname} has no decoding for {k.kind}")

    # checks -----------------------------------------------------------------------------------

    def _enum_check(self, k: CType, lv: str, status: str = "P_E_VALUE") -> str:
        numbers = sorted(n for _name, n in k.asn1.enumeration)
        if numbers == list(range(len(numbers))):
            return f"  if ({lv} >= {len(numbers)}u) return {status};\n"
        cases = " && ".join(f"{lv} != {n}u" for n in numbers)
        return f"  if ({cases}) return {status};\n"

    def _enum_index_expr(self, numbers: list[int], lv: str) -> str:
        if numbers == list(range(len(numbers))):
            return lv
        expr = "0u"
        for i, n in reversed(list(enumerate(numbers))):
            expr = f"({lv} == {n}u ? {i}u : {expr})"
        return expr

    def _int_check(self, k: CType, lv: str, status: str = "P_E_VALUE") -> str:
        """The value set of a non-extensible constrained INTEGER, where the C type is wider."""
        from .constraints import Extensible

        constraint = getattr(k.asn1, "constraint", None)
        if constraint is None or isinstance(constraint, Extensible):
            return ""
        low, high = constraint.value_bounds()
        tests = []
        ctype_low, ctype_high = _ctype_range(k.ctype)
        if low is not None and low > ctype_low:
            tests.append(
                f"(int64_t){lv} < {_c_int(low)}" if k.signed else f"(uint64_t){lv} < {low}u"
            )
        if high is not None and high < ctype_high:
            tests.append(
                f"(int64_t){lv} > {_c_int(high)}" if k.signed else f"(uint64_t){lv} > {high}u"
            )
        if not tests:
            return ""
        return f"  if ({' || '.join(tests)}) return {status};\n"

    def _int_range_from(self, k: CType, t: str, *, signed: bool, offset: int = 0) -> str:
        """After a variable-size read: the value must fit the C type (else LIMIT)."""
        lo, hi = _ctype_range(k.ctype)
        if signed:
            tests = []
            if lo > -(1 << 63):
                tests.append(f"{t} < {_c_int(lo)}")
            if hi < (1 << 63) - 1:
                tests.append(f"{t} > {hi}")
            return f"  if ({' || '.join(tests)}) return P_E_LIMIT;\n" if tests else ""
        limit = hi - offset
        if limit < (1 << 64) - 1:
            return f"  if ({t} > {limit}u) return P_E_LIMIT;\n"
        return ""

    def _size_check(self, k: CType, lv: str, status: str) -> str:
        from .per import _size_bounds

        low, high, ext = _size_bounds(k.asn1)
        if ext:
            return ""
        tests = []
        if low:
            tests.append(f"{lv}.n < {low}u")
        if high is not None:
            tests.append(f"{lv}.n > {high}u")
        return f"  if ({' || '.join(tests)}) return {status};\n" if tests else ""


def _byte_rule_width(k: CType):
    """The byte rule's fixed width for an INTEGER, or None for an IntX `(0..MAX)`."""
    constraint = getattr(k.asn1, "constraint", None)
    if constraint is None:
        raise Asn1Error("cgen: the byte rule has no encoding for an unconstrained INTEGER")
    low, high = constraint.value_bounds()
    if low == 0 and high is None:
        return None
    for width, (lo, hi) in (
        (1, (0, 255)),
        (2, (0, 65535)),
        (4, (0, 4294967295)),
        (8, (0, 18446744073709551615)),
        (4, (-2147483648, 2147483647)),
    ):
        if (low, high) == (lo, hi):
            return width
    raise Asn1Error(f"cgen: the byte rule has no width for INTEGER ({low}..{high})")


def _ctype_range(ctype: str) -> tuple[int, int]:
    bits = int("".join(ch for ch in ctype if ch.isdigit()))
    if ctype.startswith("u"):
        return 0, (1 << bits) - 1
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def _c_int(value: int) -> str:
    """A C integer literal of `value` that is never an out-of-range constant."""
    if value == -(1 << 63):
        return "(-9223372036854775807 - 1)"
    if value < 0:
        return f"({value}LL)"
    return f"{value}ULL" if value > (1 << 63) - 1 else f"{value}LL"


def _unused(name: str, body: str) -> str:
    """`(void)name;` when `body` never names the cursor pointer `name` -- a SEQUENCE whose every
    member is constructed reaches its cursor only through the calls' write-back, and an unused
    local is an error under the -Wall -Wextra -Werror these files are held to."""
    return "" if re.search(rf"\b{name}\b", body) else f"  (void){name};\n"


#: The most bits one P_put or P_get carries: a 64-bit word less the 7 a cursor can already sit
#: into its octet.
_BITS_PER_CALL = 57


def _bit_chunks(bits: list) -> list[list]:
    """`bits`, in order, cut into the runs one P_put or P_get carries."""
    return [bits[i : i + _BITS_PER_CALL] for i in range(0, len(bits), _BITS_PER_CALL)]


def _root_test(var: str, low, high, k: "CType") -> str:
    """The C test that `var`, an `int64_t`, lies in an extensible INTEGER's root (X.691 13.1).

    Either bound may be absent -- `(0..MAX, ...)` has no upper one -- and is then no test at
    all; a root with neither is every value. The C type of an extensible INTEGER is `int64_t`
    (its extension can reach past any root), so a root bound outside it has no C spelling the
    comparison could use, and the type is refused rather than compared through a conversion."""
    for bound in (low, high):
        if bound is not None and not -(1 << 63) <= bound <= (1 << 63) - 1:
            raise Asn1Error(
                f"cgen: {k.cname or 'an extensible INTEGER'}'s root bound {bound} is outside "
                f"int64_t, the C type an extensible INTEGER is carried in"
            )
    tests = []
    if low is not None:
        tests.append(f"{var} >= {low if low > -(1 << 63) else _c_int(low)}")
    if high is not None:
        tests.append(f"{var} <= {high}")
    return " && ".join(tests) if tests else "1"


def _indent(code: str) -> str:
    return "".join(("  " + line) if line.strip() else line for line in code.splitlines(True))


# --- the header: types ----------------------------------------------------------------------


def _type_decls(model: CModel) -> str:
    P = model.prefix
    out = [
        f"typedef struct {{ const uint8_t *p; size_t n; }} {P}octets;\n",
        f"typedef struct {{ uint8_t *base; size_t cap; size_t used; }} {P}arena;\n",
    ]
    for c in model.structs():
        out.append(f"typedef struct {P}{c.cname} {P}{c.cname};\n")
    for c in model.structs():
        out.append(_struct_def(model, c))
    return "".join(out)


def _member_decl(model: CModel, m: CMember) -> str:
    P = model.prefix
    k = m.kind
    if k.kind == "null":
        return ""
    if k.kind in ("seq", "choice", "seqof", "octfix"):
        star = "*" if m.pointer else ""
        return f"{P}{k.cname} {star}{m.field};"
    if k.kind == "octvar":
        return f"{P}octets {m.field};"
    return f"{k.ctype} {m.field};"


def _struct_def(model: CModel, c: CType) -> str:
    P = model.prefix
    if c.kind == "octfix":
        return f"struct {P}{c.cname} {{ uint8_t b[{c.size}]; }};\n"
    if c.kind == "seqof":
        elem = c.element
        k = elem.kind
        if k.kind in ("seq", "choice", "seqof", "octfix"):
            et = f"{P}{k.cname}"
        elif k.kind == "octvar":
            et = f"{P}octets"
        elif k.kind == "null":
            raise Asn1Error(f"cgen: {c.cname} is a SEQUENCE OF NULL")
        else:
            et = k.ctype
        return f"struct {P}{c.cname} {{ size_t n; {et} *v; }};\n"
    if c.kind == "seq":
        lines = []
        for m in c.members:
            if m.optional:
                lines.append(f"  uint8_t has_{m.field};")
            decl = _member_decl(model, m)
            if decl:
                lines.append(f"  {decl}")
        if not lines:
            lines.append("  uint8_t empty_;")
        return f"struct {P}{c.cname} {{\n" + "\n".join(lines) + "\n};\n"
    if c.kind == "choice":
        consts = "".join(
            f"#define {P.upper()}{c.cname}_{m.field} {i}u\n" for i, m in enumerate(c.members)
        )
        lines = [f"    {d}" for d in (_member_decl(model, m) for m in c.members) if d]
        if not lines:
            lines.append("    uint8_t empty_;")
        return (
            consts
            + f"struct {P}{c.cname} {{\n  uint32_t tag;\n  union {{\n"
            + "\n".join(lines)
            + "\n  } u;\n};\n"
        )
    raise Asn1Error(f"cgen: {c.kind} is not a struct")  # pragma: no cover


# --- equality (for differential tests) --------------------------------------------------------


def _eq_functions(model: CModel) -> str:
    P = model.prefix
    out = []
    for c in model.structs():
        if c.kind == "octfix":
            continue
        out.append(f"int {P}eq_{c.cname}(const {P}{c.cname} *x, const {P}{c.cname} *y);\n")
    for c in model.structs():
        if c.kind == "octfix":
            continue
        body = []
        if c.kind == "seq":
            for m in c.members:
                cond = _eq_member(model, m, f"x->{m.field}", f"y->{m.field}")
                if m.optional:
                    body.append(f"  if (x->has_{m.field} != y->has_{m.field}) return 0;\n")
                    if cond:
                        body.append(f"  if (x->has_{m.field} && !({cond})) return 0;\n")
                elif cond:
                    body.append(f"  if (!({cond})) return 0;\n")
        elif c.kind == "choice":
            body.append("  if (x->tag != y->tag) return 0;\n  switch (x->tag) {\n")
            for i, m in enumerate(c.members):
                cond = _eq_member(model, m, f"x->u.{m.field}", f"y->u.{m.field}")
                body.append(f"  case {i}u: return {cond or '1'};\n")
            body.append("  default: return 0;\n  }\n")
        else:
            cond = _eq_member(model, c.element, "x->v[i]", "y->v[i]")
            body.append(
                "  size_t i;\n  if (x->n != y->n) return 0;\n"
                f"  for (i = 0; i < x->n; i++) if (!({cond or '1'})) return 0;\n"
            )
        out.append(
            f"int {P}eq_{c.cname}(const {P}{c.cname} *x, const {P}{c.cname} *y) "
            f"{{\n" + "".join(body) + "  return 1;\n}\n"
        )
    return "".join(out)


def _eq_member(model: CModel, m: CMember, a: str, b: str) -> str:
    P = model.prefix
    k = m.kind
    if k.kind == "null":
        return ""
    if k.kind == "octfix":
        return f"memcmp({a}.b, {b}.b, {k.size}) == 0"
    if k.kind == "octvar":
        return f"{a}.n == {b}.n && ({a}.n == 0u || memcmp({a}.p, {b}.p, {a}.n) == 0)"
    if k.kind in ("seq", "choice", "seqof"):
        if m.pointer:
            return f"{a} != NULL && {b} != NULL && {P}eq_{k.cname}({a}, {b})"
        return f"{P}eq_{k.cname}(&{a}, &{b})"
    return f"{a} == {b}"


# --- static values (Python value -> C initializer) --------------------------------------------


class ValueWriter:
    """Writes Python values of the model's types as static C data: `define(name, root, value)`
    returns the declarations that make `name` a `const` value of that root type."""

    def __init__(self, model: CModel):
        self.m = model
        self.decls: list[str] = []
        self._n = 0

    def _sym(self) -> str:
        self._n += 1
        return f"{self.m.prefix}d{self._n}"

    def define(self, name: str, root: str, value) -> str:
        c = self.m.ctype_of(self.m.types[root], root)
        init = self._init(CMember("root", c), value)
        self.decls.append(f"static const {self.m.prefix}{c.cname} {name} = {init};\n")
        return "".join(self.decls)

    def _init(self, m: CMember, value) -> str:
        P = self.m.prefix
        k = m.kind
        if m.pointer:
            sym = self._sym()
            inner = self._init(CMember(m.name, k), value)
            self.decls.append(f"static {P}{k.cname} {sym} = {inner};\n")
            return f"&{sym}"
        if k.kind == "seq":
            parts = []
            for mm in k.members:
                if mm.optional:
                    parts.append(f".has_{mm.field} = {1 if mm.name in value else 0}")
                if mm.name in value and mm.kind.kind != "null":
                    parts.append(f".{mm.field} = {self._init(mm, value[mm.name])}")
            return "{" + ", ".join(parts) + "}" if parts else "{0}"
        if k.kind == "choice":
            name, inner = value
            index = next(i for i, mm in enumerate(k.members) if mm.name == name)
            mm = k.members[index]
            if mm.kind.kind == "null":
                return f"{{.tag = {index}u}}"
            return f"{{.tag = {index}u, .u.{mm.field} = {self._init(mm, inner)}}}"
        if k.kind == "seqof":
            items = list(value)
            if not items:
                return "{0, NULL}"
            sym = self._sym()
            et = _struct_def(self.m, k).split("{ size_t n; ")[1].split(" *v;")[0]
            body = ", ".join(self._init(k.element, item) for item in items)
            self.decls.append(f"static {et} {sym}[] = {{{body}}};\n")
            return f"{{{len(items)}u, {sym}}}"
        if k.kind == "octfix":
            data = bytes(value)
            return "{{" + ", ".join(f"0x{b:02x}" for b in data) + "}}"
        if k.kind == "octvar":
            data = bytes(value)
            if not data:
                return "{NULL, 0u}"
            sym = self._sym()
            self.decls.append(
                f"static const uint8_t {sym}[] = {{{', '.join(f'0x{b:02x}' for b in data)}}};\n"
            )
            return f"{{{sym}, {len(data)}u}}"
        if k.kind == "bool":
            return "1" if value else "0"
        if k.kind in ("int", "enum"):
            return _c_int(int(value)) if k.kind == "int" else f"{int(value)}u"
        raise Asn1Error(f"cgen: no initializer for {k.kind}")  # pragma: no cover


# --- the whole file ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CSource:
    """A generated translation unit: `header` declares the types and entry points, `source`
    defines them (it includes nothing of its own beyond the C library headers)."""

    header: str
    source: str
    entry_points: tuple[str, ...]
    header_name: str = ""


def generate(types: dict, roots, *, prefix: str, rules, equality: bool = False) -> CSource:
    """Compile `roots` (type names in `types`) to C for every rule in `rules`."""
    if not prefix or not prefix.replace("_", "").isalnum():
        raise Asn1Error(f"cgen: prefix {prefix!r} is not a C identifier prefix")
    model = CModel(types, roots, prefix)
    emitters = [_Emitter(model, rule) for rule in rules]
    names = [e.rname for e in emitters]
    if len(set(names)) != len(names):
        raise Asn1Error(f"cgen: two rules share a name: {names}")

    entry = []
    proto = []
    for e in emitters:
        for name in model.roots:
            c = model.ctype_of(types[name], name)
            T = f"{prefix}{c.cname}"
            entry += [f"{prefix}{e.rname}_encode_{c.cname}", f"{prefix}{e.rname}_decode_{c.cname}"]
            proto.append(
                f"int {prefix}{e.rname}_encode_{c.cname}(const {T} *v, uint8_t *out, "
                f"size_t cap, size_t *len);\n"
            )
            proto.append(
                f"int {prefix}{e.rname}_decode_{c.cname}(const uint8_t *in, size_t len, "
                f"{T} *v, {prefix}arena *a);\n"
            )
    guard = f"{prefix.upper()}H"
    up = prefix.upper()
    statuses = (
        "/* Statuses. Every generated entry point returns one; nothing else leaves a decoder. */\n"
        f"#define {up}OK 0\n"
        f"#define {up}E_SPACE 1     /* the output buffer is too small */\n"
        f"#define {up}E_TRUNC 2     /* the input ends inside a value */\n"
        f"#define {up}E_MALFORMED 3 /* octets no canonical encoding of this type contains */\n"
        f"#define {up}E_VALUE 4     /* a value outside its type, handed to an encoder */\n"
        f"#define {up}E_LIMIT 5     /* the arena, the depth, or a length this codec carries */\n"
    )
    if equality:
        proto += [
            f"int {prefix}eq_{c.cname}(const {prefix}{c.cname} *x, const {prefix}{c.cname} *y);\n"
            for c in model.structs()
            if c.kind != "octfix"
        ]
    header = (
        f"/* Generated by bcir.asn1.cgen -- do not edit. */\n#ifndef {guard}\n#define {guard}\n"
        "#include <stddef.h>\n#include <stdint.h>\n"
        + statuses
        + _type_decls(model)
        + "".join(proto)
        + f"#endif /* {guard} */\n"
    )
    library = _PRELUDE.replace("P_arena", f"{prefix}arena").replace("@MAXDEPTH@", str(MAX_DEPTH))
    library += _BYTE_IO + _COER_IO + _UPER_IO
    byte_rules = [r for r in rules if isinstance(r, ByteRule)]
    if byte_rules:
        if len({r.intx_max for r in byte_rules}) != 1:
            raise Asn1Error("cgen: byte rules in one file must share their IntX bound")
        library += f"#define P_INTX_MAX {byte_rules[0].intx_max}ULL\n" + _INTX_IO
    body = []
    for e in emitters:
        body += e.prototypes()
    for e in emitters:
        body += e.bodies()
        body += e.entry_points()
    code = "\n".join(body)
    if equality:
        code += _eq_functions(model)
    text = _used_helpers(library, code) + code
    # The runtime helpers are spelled with a `P_` stem; give them this file's prefix so two
    # generated files link together.
    text = (
        text.replace("P_bw", f"{prefix}bw")
        .replace("P_br", f"{prefix}br")
        .replace("P_pw", f"{prefix}pw")
        .replace("P_pr", f"{prefix}pr")
    )
    text = _rename_helpers(text, prefix, up)
    header_name = f"{prefix}codec.h"
    source = f'/* Generated by bcir.asn1.cgen -- do not edit. */\n#include "{header_name}"\n' + text
    return CSource(header=header, source=source, entry_points=tuple(entry), header_name=header_name)


def _used_helpers(library: str, code: str) -> str:
    """The parts of the runtime library `code` reaches, in library order.

    The library is split at top-level boundaries into chunks -- a function, a typedef, a
    macro, a whole preprocessor conditional -- and a chunk is kept when something kept names
    what it defines (a conditional defines every macro inside it). Macros and typedefs are kept
    whenever any kept chunk names them, so a file that never decodes UPER carries no bit
    reader at all, and `-Wunused-function` has nothing to say about what remains. A chunk that
    defines nothing (an include, a banner comment) is always kept."""

    chunks: list[str] = []
    current: list[str] = []
    depth = 0  # nesting of the top-level conditional being collected: `#if` to its `#endif`
    leading = False  # `current` is a comment so far, which belongs to the definition below it
    for line in library.splitlines(True):
        top = not current_is_open(current)
        starts = (
            depth == 0
            and top
            and line.startswith(("static ", "typedef ", "#define ", "#include", "#if", "/*"))
        )
        if starts and current and not (leading and not line.startswith("/*")):
            chunks.append("".join(current))
            current = []
        if starts:
            # A banner stands alone and is always kept; any other comment leads what follows.
            leading = line.startswith("/*") and not line.startswith("/* ---")
        current.append(line)
        if top and line.startswith("#if"):
            depth += 1
        elif top and line.startswith("#endif") and depth:
            depth -= 1
    if current:
        chunks.append("".join(current))

    def defines(text: str) -> set[str]:
        body = re.sub(r"\A(?:\s*/\*.*?\*/)*\s*", "", text, flags=re.S)  # past a leading comment
        if body.startswith("#if"):
            return set(re.findall(r"^#define (P_[A-Za-z0-9_]+)", body, re.M))
        m = (
            re.match(r"static[^(]*?\b(P_[A-Za-z0-9_]+)\s*\(", body)
            or re.match(r"#define (P_[A-Za-z0-9_]+)", body)
            or re.search(
                r"\}\s*(P_[A-Za-z0-9_]+)\s*;",
                body.split("\n")[0] if body.startswith("typedef") else "",
            )
        )
        return {m.group(1)} if m else set()

    names = [defines(text) for text in chunks]
    defined = set().union(*names)
    wanted = set(re.findall(r"\bP_[A-Za-z0-9_]+\b", code)) & defined
    changed = True
    while changed:
        changed = False
        for text, own in zip(chunks, names):
            if own & wanted:
                new = (set(re.findall(r"\bP_[A-Za-z0-9_]+\b", text)) & defined) - wanted
                if new:
                    wanted |= new
                    changed = True
    return "".join(text for text, own in zip(chunks, names) if not own or own & wanted)


def current_is_open(lines: list[str]) -> bool:
    """Whether the chunk being collected is inside a brace (a function body in progress)."""
    depth = 0
    for line in lines:
        depth += line.count("{") - line.count("}")
    return depth > 0


def _rename_helpers(text: str, prefix: str, up: str) -> str:

    # Every macro the library spells `P_UPPER` (statuses, bounds, P_HOT, the byte-swap words)
    # takes the upper-case prefix, every function and type the lower-case one: nothing the
    # generated file defines leaks an unprefixed name into the program that includes it.
    text = re.sub(r"\bP_([A-Z][A-Z0-9_]*)\b", lambda m: f"{up}{m.group(1)}", text)
    return re.sub(r"\bP_([a-z][A-Za-z0-9_]*)\b", lambda m: f"{prefix}{m.group(1)}", text)


__all__ = ["COER", "UPER", "ByteRule", "CModel", "CSource", "MAX_DEPTH", "ValueWriter", "generate"]
